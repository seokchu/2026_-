"""P1-4. 선택된 고정 정책의 FP/FN 이 **어느 공정조건에 집중되는가** (심사표 3번 대응).

규칙:
  - 정책은 `28_sequential_operating_points.csv` 의 SEL 선택 운용점을 그대로 쓴다(재탐색 금지).
  - 조건은 **결정시점(T-60) 정보**로만 정의한다. 작업자가 그 시점에 알 수 있는 것만 쓴다.
  - 사전 선언 상호작용 2개: (A) 전환형 x 고부하, (B) 급변 x 비조업.
    사후에 눈에 띈 조건은 여기서 결론내지 않고 다음 차수 LOCK 에서만 확인한다.
  - 불확실성은 일(day) 블록 bootstrap 으로 제시한다(시계열 상관 때문에 단순 이항 CI 금지).
목표시각 테이블 구성은 sequential_policy.py 와 동일 규칙(미러 구현, 주석으로 명시).
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, yaml
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / "analysis"))
from common import save_table, fig_path, mpl, TAB                        # noqa: E402
plt = mpl()

cfg = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
PRE = cfg["output"]["table_prefix"]
STAGES = {"early_h60": (4, 60), "confirmA_h45": (3, 45), "confirmB_h15": (1, 15)}
LOCK_DAYS, TEST_DAYS = 14, 14
rng = np.random.default_rng(cfg["seed"])

o = pd.read_csv(TAB / f"{PRE}_oof_rows.csv", parse_dates=["ts15"], encoding="utf-8-sig")
o = o.sort_values("ts15").reset_index(drop=True)
thr = pd.read_csv(TAB / f"{PRE}_cal_alert_thresholds.csv", encoding="utf-8-sig")
TM = {(int(r.fold), int(r.horizon_min), float(r.q)): float(r.threshold_kw) for r in thr.itertuples()}

base = o[["ts15", "fold", "clean", "kw", "thr_adaptive", "peak_now"]].copy()
base = base.rename(columns={"ts15": "T", "peak_now": "label"})
for name, (hh, back) in STAGES.items():
    dec = o[["ts15", "fold", f"pred_h{hh}"]].copy()
    dec["T"] = dec.ts15 + pd.Timedelta(minutes=back)
    dec = dec.rename(columns={"ts15": f"t_{name}", "fold": f"fold_{name}",
                              f"pred_h{hh}": f"score_{name}"})
    base = base.merge(dec, on="T", how="left")
# 결정시점(T-60) 의 공정 문맥 — 작업자가 그 시점에 아는 정보만
ctx = o[["ts15", "regime", "is_operating", "kw_ramp4", "kw", "band_h4", "int_width_h4",
         "ood_flag", "생산량"]].rename(columns={"ts15": "t_early_h60", "kw": "kw_at_decision",
                                               "band_h4": "band_early"})
base = base.merge(ctx, on="t_early_h60", how="left")
tt = base.dropna(subset=["score_early_h60", "score_confirmA_h45", "score_confirmB_h15",
                         "regime"]).reset_index(drop=True)
tt["date"] = tt["T"].dt.normalize()
tt["hour"] = tt["T"].dt.hour

cl = tt[tt.clean].reset_index(drop=True)
end = cl["T"].max()
lock_start = end - pd.Timedelta(days=LOCK_DAYS)
test_start = lock_start - pd.Timedelta(days=TEST_DAYS)
SEL = cl[cl["T"] < test_start].reset_index(drop=True)
TEST = cl[(cl["T"] >= test_start) & (cl["T"] < lock_start)].reset_index(drop=True)
LOCK = cl[cl["T"] >= lock_start].reset_index(drop=True)

# 조건 컷은 SEL(과거)에서만 산출한다
RAMP_HI = float(np.quantile(SEL.kw_ramp4.abs(), .8))
LOAD_HI = float(np.quantile(SEL.kw_at_decision, .8))
WID_HI = float(np.quantile(SEL.int_width_h4, .8))

pol = pd.read_csv(TAB / "28_sequential_operating_points.csv", encoding="utf-8-sig")
P = pol[(pol.cost_ratio_FN_FP == cfg["policy"]["default_cost_ratio"]) &
        (pol.scope == "all_policies") & (pol.applied_to == "TEST")].iloc[0]
print("고정 정책:", P.policy, P.early_q, P.confirm_stage, P.confirm_q, P.gate)


def mask_stage(sub, stage, qv):
    hh, _ = STAGES[stage]
    t = np.array([TM[(int(f), hh * 15, float(qv))] for f in sub[f"fold_{stage}"].values])
    return sub[f"score_{stage}"].values >= t


def alert_of(sub):
    if str(P.policy).startswith("single_"):
        return mask_stage(sub, str(P.policy).replace("single_", ""), P.early_q)
    early = mask_stage(sub, "early_h60", P.early_q)
    conf = mask_stage(sub, str(P.confirm_stage), P.confirm_q)
    unsure = sub.band_early.isin(cfg["diagnosis"]["uncertain_bands"]).values
    if P.gate == "confirm_only_if_unsure":
        return early & (conf | ~unsure)
    return early & conf


def conditions(sub):
    r = sub.regime.values
    return {
        "ALL": np.ones(len(sub), bool),
        "regime_LOW_LOAD": r == "LOW_LOAD",
        "regime_STABLE_OPERATION": r == "STABLE_OPERATION",
        "regime_TRANSITION_LIKE": r == "TRANSITION_LIKE",
        "non_operating": sub.is_operating.values == 0,
        "operating": sub.is_operating.values == 1,
        "high_ramp_top20": sub.kw_ramp4.abs().values >= RAMP_HI,
        "high_load_top20": sub.kw_at_decision.values >= LOAD_HI,
        "wide_interval_top20": sub.int_width_h4.values >= WID_HI,
        "band_LOW_or_OOD": sub.band_early.isin(cfg["diagnosis"]["uncertain_bands"]).values,
        "ood_flag": sub.ood_flag.values == 1,
        "hour_00_06": (sub.hour.values >= 0) & (sub.hour.values < 6),
        "hour_06_12": (sub.hour.values >= 6) & (sub.hour.values < 12),
        "hour_12_18": (sub.hour.values >= 12) & (sub.hour.values < 18),
        "hour_18_24": sub.hour.values >= 18,
        # 사전 선언 상호작용
        "INTERACT_A_transition_x_high_load": (r == "TRANSITION_LIKE") &
                                             (sub.kw_at_decision.values >= LOAD_HI),
        "INTERACT_B_high_ramp_x_non_operating": (sub.kw_ramp4.abs().values >= RAMP_HI) &
                                                (sub.is_operating.values == 0)}


def confusion(sub, alert, m):
    y = (sub.label.values == 1)[m]; a = alert[m]
    tp = int((a & y).sum()); fp = int((a & ~y).sum())
    fn = int((~a & y).sum()); tn = int((~a & ~y).sum())
    return dict(n=int(m.sum()), share_of_window=float(m.mean()),
                peak_prevalence=float(y.mean()) if m.sum() else np.nan,
                tp=tp, fp=fp, fn=fn, tn=tn,
                precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1),
                f1=2 * tp / max(2 * tp + fp + fn, 1),
                fn_rate_given_peak=fn / max(tp + fn, 1),
                fp_rate_given_nonpeak=fp / max(fp + tn, 1))


rows = []
for wtag, sub in [("TEST", TEST), ("LOCK_완전잠금", LOCK)]:
    alert = alert_of(sub)
    tot_fn = int(((~alert) & (sub.label.values == 1)).sum())
    tot_fp = int((alert & (sub.label.values != 1)).sum())
    for cname, m in conditions(sub).items():
        c = confusion(sub, alert, m)
        rows.append(dict(window=wtag, condition=cname,
                         share_of_all_FN=c["fn"] / max(tot_fn, 1),
                         share_of_all_FP=c["fp"] / max(tot_fp, 1), **c))
res = pd.DataFrame(rows)
save_table(res, "28_error_conditions")

# ---- 상호작용 2개: 일 블록 bootstrap CI (FN율 / FP율)
boot = []
for wtag, sub in [("TEST", TEST), ("LOCK_완전잠금", LOCK)]:
    alert = alert_of(sub)
    days = sub.date.unique()
    conds = conditions(sub)
    for cname in ["INTERACT_A_transition_x_high_load", "INTERACT_B_high_ramp_x_non_operating",
                  "regime_TRANSITION_LIKE", "band_LOW_or_OOD", "ALL"]:
        fnr, fpr = [], []
        for _ in range(1000):
            pick = rng.choice(days, size=len(days), replace=True)
            idx = np.concatenate([np.flatnonzero(sub.date.values == dd) for dd in pick])
            s = sub.iloc[idx]; a = alert[idx]; m = conds[cname][idx]
            y = (s.label.values == 1)
            tp = int((a & y & m).sum()); fn = int((~a & y & m).sum())
            fp = int((a & ~y & m).sum()); tn = int((~a & ~y & m).sum())
            if tp + fn > 0:
                fnr.append(fn / (tp + fn))
            if fp + tn > 0:
                fpr.append(fp / (fp + tn))
        boot.append(dict(window=wtag, condition=cname, n=int(conds[cname].sum()),
                         fn_rate_mean=float(np.mean(fnr)) if fnr else np.nan,
                         fn_ci_lo=float(np.quantile(fnr, .025)) if fnr else np.nan,
                         fn_ci_hi=float(np.quantile(fnr, .975)) if fnr else np.nan,
                         fp_rate_mean=float(np.mean(fpr)) if fpr else np.nan,
                         fp_ci_lo=float(np.quantile(fpr, .025)) if fpr else np.nan,
                         fp_ci_hi=float(np.quantile(fpr, .975)) if fpr else np.nan,
                         n_boot=1000, method="일 블록 bootstrap"))
bt = pd.DataFrame(boot)
save_table(bt, "28_error_conditions_bootstrap")
save_table(pd.DataFrame([dict(ramp_cut_top20_SEL=RAMP_HI, load_cut_top20_SEL=LOAD_HI,
                              interval_width_cut_top20_SEL=WID_HI,
                              policy=P.policy, early_q=P.early_q, confirm_stage=P.confirm_stage,
                              confirm_q=P.confirm_q, gate=P.gate,
                              note="조건 컷은 SEL(과거)에서만 산출. TEST/LOCK 재탐색 없음")]),
           "28_error_conditions_setup")
t = res[res.window == "TEST"]
print(t[["condition", "n", "peak_prevalence", "tp", "fp", "fn", "precision", "recall", "f1",
         "fn_rate_given_peak", "share_of_all_FN", "share_of_all_FP"]].round(3).to_string(index=False))
print(bt.round(3).to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
s = t[~t.condition.isin(["ALL"])].sort_values("share_of_all_FN", ascending=False).head(8)
axes[0].barh(s.condition, s.share_of_all_FN, color="#c33")
axes[0].set_xlabel("전체 FN 중 비중"); axes[0].set_title("미탐이 몰리는 조건 (TEST)")
axes[0].tick_params(labelsize=6)
s2 = t[~t.condition.isin(["ALL"])].sort_values("share_of_all_FP", ascending=False).head(8)
axes[1].barh(s2.condition, s2.share_of_all_FP, color="#27b")
axes[1].set_xlabel("전체 FP 중 비중"); axes[1].set_title("오경보가 몰리는 조건 (TEST)")
axes[1].tick_params(labelsize=6)
fig.suptitle("FIGURE 9. 고정 정책의 FP/FN 공정조건 분포 (결정시점 정보 기준)", fontsize=9)
fig.tight_layout(); fig.savefig(fig_path("28_error_conditions")); plt.close(fig)
