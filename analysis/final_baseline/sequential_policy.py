"""P0-1 + P0-2 + P1-5. 동일 목표시각 순차 경보 + 정책선택/평가 시간분리 + 운영단위 분리.

이전 구현의 결함(모의평가 지적):
  (1) 같은 행에서 early=h60(t), conf=h15(t) 를 묶었다 -> 두 예측의 **목표시각이 다르다**
      (t+60 vs t+15). '같은 피크를 15분 뒤 재확인' 이 구현되지 않았다.
  (2) 경보 임계값은 CAL 에서 만들었지만 **정책 선택**을 평가창에서 했다 -> 사후 최적화.
  (3) false_alerts_per_day 가 15분 스텝/일인데 '건/일' 로 서술했다.

이 스크립트의 설계:
  - **목표시각 T 중심**. T 의 라벨은 '실측 수요(T) >= thr(T)' (직전 30일 p95, 정의 불변).
    1단계 조기경보: 결정시점 T-60 에서 pred_h4 (목표 T)
    2단계 확인     : 변형 A = T-45 에서 pred_h3 (잔여 대응 45분)
                     변형 B = T-15 에서 pred_h1 (잔여 대응 15분)
    상태: watch -> confirmed / rejected / expired
  - 임계값은 각 결정시점 폴드의 CAL 예측분포에서(기존 산출물 재사용).
  - **선택 구간 SEL / 평가 구간 TEST / 완전잠금 LOCK** 을 시간순으로 분리한다.
    정책(조기·확인 분위, 게이트, 비용비)은 SEL 에서만 고른다. TEST·LOCK 에는 선택된 하나만 적용.
  - 운영단위를 3종으로 분리 보고: 경보 스텝/일, 경보 사건/일, 작업자 확인분/일.
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
QS = cfg["policy"]["threshold_grid"]
RATIOS = cfg["policy"]["cost_ratios"]
GATES = cfg["policy"]["gates"]
CHECK_MIN = 5.0          # [가설] 경보 1건당 작업자 확인 소요 5분 — 비용 아님, 부담 표기용
LOCK_DAYS = 14           # 완전잠금 holdout
TEST_DAYS = 14           # 평가 구간
UNSURE = cfg["diagnosis"]["uncertain_bands"]

o = pd.read_csv(TAB / f"{PRE}_oof_rows.csv", parse_dates=["ts15"], encoding="utf-8-sig")
o = o.sort_values("ts15").reset_index(drop=True)
thr = pd.read_csv(TAB / f"{PRE}_cal_alert_thresholds.csv", encoding="utf-8-sig")
TM = {(int(r.fold), int(r.horizon_min), float(r.q)): float(r.threshold_kw) for r in thr.itertuples()}

# ---- 목표시각 T 테이블: 각 단계의 결정시점 행을 T 에 맞춰 붙인다
base = o[["ts15", "fold", "clean", "kw", "thr_adaptive", "peak_now", "regime",
          "is_operating", "kw_ramp4", "band_h4", "band_h1"]].copy()
base = base.rename(columns={"ts15": "T", "peak_now": "label"})
STAGES = {"early_h60": (4, 60), "confirmA_h45": (3, 45), "confirmB_h15": (1, 15)}
for name, (hh, back) in STAGES.items():
    dec = o[["ts15", "fold", f"pred_h{hh}", "band_h4"]].copy()
    dec["T"] = dec.ts15 + pd.Timedelta(minutes=back)
    dec = dec.rename(columns={"ts15": f"t_{name}", "fold": f"fold_{name}",
                              f"pred_h{hh}": f"score_{name}", "band_h4": f"band_{name}"})
    base = base.merge(dec, on="T", how="left")
need = ["score_early_h60", "score_confirmA_h45", "score_confirmB_h15"]
tt = base.dropna(subset=need).reset_index(drop=True)
tt["date"] = tt["T"].dt.normalize()
print("목표시각 행", len(tt), "| 피크 라벨", int(tt.label.sum()), "| clean", int(tt.clean.sum()))


def mask(sub, stage, qv):
    """해당 단계 결정시점의 폴드별 CAL 임계값을 적용."""
    hh, _ = STAGES[stage]
    t = np.array([TM[(int(f), hh * 15, float(qv))] for f in sub[f"fold_{stage}"].values])
    return sub[f"score_{stage}"].values >= t


def runs(m, gap=2):
    idx = np.flatnonzero(m)
    if not len(idx):
        return []
    out, s, p = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - p <= gap:
            p = i; continue
        out.append((s, p)); s, p = i, i
    out.append((s, p))
    return out


def evaluate(sub, alert, advisory, stage_confirm):
    """목표시각 단위 혼동행렬 + 이벤트 단위 포착 + 운영단위 3종."""
    lab = sub.label.values == 1
    days = sub.date.nunique()
    tp = int((alert & lab).sum()); fp = int((alert & ~lab).sum())
    fn = int((~alert & lab).sum()); tn = int((~alert & ~lab).sum())
    ev = runs(lab)
    resid = 60 if stage_confirm is None else int(STAGES[stage_confirm][1])
    det, leads = 0, []
    Tv = sub["T"].values
    for s, e in ev:
        a = alert[s:e + 1]
        if a.any():
            det += 1
            first = s + int(np.flatnonzero(a)[0])
            # 확인 시점 -> 이벤트 시작까지 남은 시간
            dec_t = sub[f"t_{stage_confirm}"].values[first] if stage_confirm else \
                sub["t_early_h60"].values[first]
            leads.append(max(0.0, (Tv[s] - dec_t) / np.timedelta64(1, "m")))
    al = runs(alert)
    return dict(n_targets=len(sub), n_days=days, prevalence=float(lab.mean()),
                tp=tp, fp=fp, fn=fn, tn=tn,
                precision=tp / max(tp + fp, 1), recall_target=tp / max(tp + fn, 1),
                f1=2 * tp / max(2 * tp + fp + fn, 1),
                n_events=len(ev), events_detected=det, missed_events=len(ev) - det,
                event_recall=det / max(len(ev), 1),
                residual_lead_min=resid,
                mean_lead_to_onset_min=float(np.mean(leads)) if leads else np.nan,
                lead_ge_30min_share=float(np.mean([l >= 30 for l in leads])) if leads else np.nan,
                false_alert_steps=fp, false_alert_steps_per_day=fp / max(days, 1),
                alert_steps_per_day=float(alert.sum()) / max(days, 1),
                alert_events_per_day=len(al) / max(days, 1),
                operator_check_min_per_day=len(al) * CHECK_MIN / max(days, 1),
                advisories_per_day=float(advisory.sum()) / max(days, 1))


def sweep(sub):
    rows = []
    unsure = sub.band_early_h60.isin(UNSURE).values
    zero = np.zeros(len(sub), bool)
    for stage in ["early_h60", "confirmA_h45", "confirmB_h15"]:
        for qv in QS:
            a = mask(sub, stage, qv)
            rows.append(dict(policy=f"single_{stage}", early_q=qv, confirm_stage="",
                             confirm_q=np.nan, gate="none", confirm_rejection_rate=0.0,
                             **evaluate(sub, a, zero, None if stage == "early_h60" else stage)))
    for cstage in ["confirmA_h45", "confirmB_h15"]:
        for qe in QS:
            early = mask(sub, "early_h60", qe)
            for qc in QS:
                conf = mask(sub, cstage, qc)
                rej = float((early & ~conf).sum() / max(early.sum(), 1))
                for gate in GATES:
                    if gate == "none":
                        alert, adv = early & conf, zero
                    elif gate == "advisory_if_unsure":
                        alert, adv = early & conf, early & ~conf & unsure
                    else:
                        alert, adv = early & (conf | ~unsure), early & ~conf & unsure
                    rows.append(dict(policy=f"seq_h60_to_{cstage}", early_q=qe,
                                     confirm_stage=cstage, confirm_q=qc, gate=gate,
                                     confirm_rejection_rate=rej,
                                     **evaluate(sub, alert, adv, cstage)))
    cur = pd.DataFrame(rows)
    for r in RATIOS:
        cur[f"cost_r{r}"] = (cur.false_alert_steps + r * cur.missed_events) / (r * cur.n_events.clip(lower=1))
    return cur


def apply_one(sub, row):
    unsure = sub.band_early_h60.isin(UNSURE).values
    zero = np.zeros(len(sub), bool)
    if row.policy.startswith("single_"):
        stage = row.policy.replace("single_", "")
        a = mask(sub, stage, row.early_q)
        return evaluate(sub, a, zero, None if stage == "early_h60" else stage)
    early = mask(sub, "early_h60", row.early_q)
    conf = mask(sub, row.confirm_stage, row.confirm_q)
    if row.gate == "none":
        alert, adv = early & conf, zero
    elif row.gate == "advisory_if_unsure":
        alert, adv = early & conf, early & ~conf & unsure
    else:
        alert, adv = early & (conf | ~unsure), early & ~conf & unsure
    return evaluate(sub, alert, adv, row.confirm_stage)


# ---- 시간 분리: SEL(선택) / TEST(평가) / LOCK(완전잠금)
cl = tt[tt.clean].reset_index(drop=True)
end = cl["T"].max()
lock_start = end - pd.Timedelta(days=LOCK_DAYS)
test_start = lock_start - pd.Timedelta(days=TEST_DAYS)
SEL = cl[cl["T"] < test_start].reset_index(drop=True)
TEST = cl[(cl["T"] >= test_start) & (cl["T"] < lock_start)].reset_index(drop=True)
LOCK = cl[cl["T"] >= lock_start].reset_index(drop=True)
split = pd.DataFrame([dict(split=k, start=str(s["T"].min()), end=str(s["T"].max()),
                           n_targets=len(s), n_days=s.date.nunique(),
                           n_events=len(runs(s.label.values == 1)), peak_targets=int(s.label.sum()))
                      for k, s in [("SEL_정책선택", SEL), ("TEST_평가", TEST), ("LOCK_완전잠금", LOCK)]])
save_table(split, "28_sequential_split")
print(split.to_string(index=False))

sel_cur = sweep(SEL); save_table(sel_cur, "28_sequential_policy_curve_SEL")
test_cur = sweep(TEST)         # 참고용(사후최적화 상한 비교)
rows, boot = [], []
rng = np.random.default_rng(cfg["seed"])


def block_boot(sub, row_a, row_b, r, n=2000):
    """일(day) 블록 bootstrap — 비용차(a-b) CI."""
    days = sub.date.unique()
    out = []
    for _ in range(n):
        pick = rng.choice(days, size=len(days), replace=True)
        s = pd.concat([sub[sub.date == dd] for dd in pick], ignore_index=True)
        if (s.label == 1).sum() == 0:
            continue
        ma, mb = apply_one(s, row_a), apply_one(s, row_b)
        ca = (ma["false_alert_steps"] + r * ma["missed_events"]) / (r * max(ma["n_events"], 1))
        cb = (mb["false_alert_steps"] + r * mb["missed_events"]) / (r * max(mb["n_events"], 1))
        out.append(ca - cb)
    out = np.asarray(out)
    return dict(delta_mean=float(out.mean()), ci_lo=float(np.quantile(out, .025)),
                ci_hi=float(np.quantile(out, .975)),
                seq_better_prob=float((out < 0).mean()))


for r in RATIOS:
    c = f"cost_r{r}"
    # 선행시간 제약 범위: SEL 에서 '평균 선행 >= minimum_lead_minutes' 인 후보만 두고 비용 최소화
    ml0 = cfg["policy"]["minimum_lead_minutes"]
    lead_ok = sel_cur[sel_cur.mean_lead_to_onset_min >= ml0]
    scopes = [("all_policies", sel_cur),
              ("single_only", sel_cur[sel_cur.policy.str.startswith("single")]),
              ("sequential_only", sel_cur[sel_cur.policy.str.startswith("seq")])]
    if len(lead_ok):
        scopes += [(f"lead_constrained_ge{ml0}min", lead_ok)]
    for scope, g in scopes:
        if not len(g):
            continue
        pick = g.loc[g[c].idxmin()]
        for wtag, sub in [("TEST", TEST), ("LOCK", LOCK)]:
            m = apply_one(sub, pick)
            cost = (m["false_alert_steps"] + r * m["missed_events"]) / (r * max(m["n_events"], 1))
            rows.append(dict(cost_ratio_FN_FP=r, scope=scope, selected_on="SEL", applied_to=wtag,
                             policy=pick.policy, early_q=pick.early_q,
                             confirm_stage=pick.confirm_stage, confirm_q=pick.confirm_q,
                             gate=pick.gate, normalized_expected_cost=cost,
                             cost_on_SEL=float(pick[c]), **m))
    # 정직한 선택 vs 사후최적화 상한
    post = test_cur.loc[test_cur[c].idxmin()]
    rows.append(dict(cost_ratio_FN_FP=r, scope="post_hoc_on_TEST(참고)", selected_on="TEST",
                     applied_to="TEST", policy=post.policy, early_q=post.early_q,
                     confirm_stage=post.confirm_stage, confirm_q=post.confirm_q, gate=post.gate,
                     normalized_expected_cost=float(post[c]), cost_on_SEL=np.nan,
                     **{k: post[k] for k in evaluate(TEST, np.zeros(len(TEST), bool),
                                                     np.zeros(len(TEST), bool), None)}))
    # 블록 bootstrap: 순차 최적 vs 단일 최적 (SEL 선택, TEST/LOCK 적용)
    a = sel_cur[sel_cur.policy.str.startswith("seq")].loc[
        sel_cur[sel_cur.policy.str.startswith("seq")][c].idxmin()]
    b = sel_cur[sel_cur.policy.str.startswith("single")].loc[
        sel_cur[sel_cur.policy.str.startswith("single")][c].idxmin()]
    for wtag, sub in [("TEST", TEST), ("LOCK", LOCK)]:
        boot.append(dict(cost_ratio_FN_FP=r, applied_to=wtag, seq_policy=a.policy,
                         seq_early_q=a.early_q, seq_confirm_q=a.confirm_q, seq_gate=a.gate,
                         single_policy=b.policy, single_q=b.early_q,
                         **block_boot(sub, a, b, r, n=500)))
res = pd.DataFrame(rows); save_table(res, "28_sequential_operating_points")
bt = pd.DataFrame(boot); save_table(bt, "28_sequential_bootstrap")

# ---- 판정: 사전 선언 규칙
r0 = cfg["policy"]["default_cost_ratio"]
t_seq = res[(res.cost_ratio_FN_FP == r0) & (res.scope == "sequential_only") & (res.applied_to == "TEST")].iloc[0]
t_sgl = res[(res.cost_ratio_FN_FP == r0) & (res.scope == "single_only") & (res.applied_to == "TEST")].iloc[0]
wins = int(sum(1 for r in RATIOS
               if res[(res.cost_ratio_FN_FP == r) & (res.scope == "sequential_only") &
                      (res.applied_to == "TEST")].iloc[0].normalized_expected_cost <
               res[(res.cost_ratio_FN_FP == r) & (res.scope == "single_only") &
                   (res.applied_to == "TEST")].iloc[0].normalized_expected_cost))
sig = bt[(bt.cost_ratio_FN_FP == r0) & (bt.applied_to == "TEST")].iloc[0]
verdict = ("순차 우월(유의)" if (wins >= 4 and sig.ci_hi < 0) else
           "순차 우월 비유의 — 주장 철회" if wins >= 4 else "단일 경보 우월 — 순차 주장 철회")
save_table(pd.DataFrame([dict(
    verdict=verdict, ratios_sequential_wins_on_TEST=f"{wins}/{len(RATIOS)}",
    cost_seq_TEST=float(t_seq.normalized_expected_cost), cost_single_TEST=float(t_sgl.normalized_expected_cost),
    recall_seq_TEST=float(t_seq.event_recall), recall_single_TEST=float(t_sgl.event_recall),
    lead_seq_TEST=float(t_seq.mean_lead_to_onset_min), lead_single_TEST=float(t_sgl.mean_lead_to_onset_min),
    boot_delta=float(sig.delta_mean), boot_ci_lo=float(sig.ci_lo), boot_ci_hi=float(sig.ci_hi),
    rule="사전 선언: 7개 비용비 중 4개 이상에서 TEST 비용 우월 AND 기본 비용비의 일블록 bootstrap 95% CI 상한<0 이면 '순차 우월'",
    note="정책은 SEL 에서만 선택. TEST/LOCK 은 적용만. 목표시각 T 기준 평가")]),
    "28_sequential_verdict")
print(res[(res.cost_ratio_FN_FP == r0)][["scope", "applied_to", "policy", "early_q", "confirm_stage",
      "confirm_q", "gate", "normalized_expected_cost", "event_recall", "mean_lead_to_onset_min",
      "false_alert_steps_per_day", "alert_events_per_day", "operator_check_min_per_day"]].round(3).to_string(index=False))
print(bt.round(4).to_string(index=False))
print("판정:", verdict)

# ---- 시간축 예제 3건 (포착 / 기각 / 미탐)
pick = res[(res.cost_ratio_FN_FP == r0) & (res.scope == "all_policies") & (res.applied_to == "TEST")].iloc[0]
sub = TEST.copy()
early = mask(sub, "early_h60", pick.early_q)
cstage = pick.confirm_stage if pick.confirm_stage else "confirmA_h45"
conf = mask(sub, cstage, pick.confirm_q) if pick.confirm_stage else np.ones(len(sub), bool)
alert = early & conf
lab = sub.label.values == 1
ex = []
for tag, m in [("confirmed_catch", alert & lab), ("rejected_at_confirm", early & ~conf & ~lab),
               ("missed", ~alert & lab)]:
    idx = np.flatnonzero(m)
    if not len(idx):
        continue
    i = int(idx[len(idx) // 2])
    ex.append(dict(case=tag, target_time=str(sub["T"].iloc[i]),
                   t_early=str(sub.t_early_h60.iloc[i]), score_early=round(float(sub.score_early_h60.iloc[i]), 2),
                   t_confirm=str(sub[f"t_{cstage}"].iloc[i]),
                   score_confirm=round(float(sub[f"score_{cstage}"].iloc[i]), 2),
                   threshold_early=round(TM[(int(sub.fold_early_h60.iloc[i]), 60, float(pick.early_q))], 2),
                   threshold_confirm=round(TM[(int(sub[f"fold_{cstage}"].iloc[i]),
                                               STAGES[cstage][0] * 15, float(pick.confirm_q))], 2)
                   if pick.confirm_stage else np.nan,
                   actual_kw=round(float(sub.kw.iloc[i]), 2), peak_threshold=round(float(sub.thr_adaptive.iloc[i]), 1),
                   label=int(sub.label.iloc[i]), state="confirmed" if alert[i] else ("rejected" if early[i] else "no_watch"),
                   residual_lead_min=STAGES[cstage][1] if pick.confirm_stage else 60,
                   reliability_band_at_early=sub.band_early_h60.iloc[i]))
save_table(pd.DataFrame(ex), "28_sequential_timeline_examples")
print(pd.DataFrame(ex).to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
for scope, mk in [("sequential_only", "o-"), ("single_only", "s--")]:
    s = res[(res.scope == scope) & (res.applied_to == "TEST")].sort_values("cost_ratio_FN_FP")
    axes[0].plot(s.cost_ratio_FN_FP, s.normalized_expected_cost, mk, label=scope)
    axes[1].plot(s.cost_ratio_FN_FP, s.event_recall, mk, label=f"{scope} 포착률")
axes[0].set_xscale("log"); axes[0].set_xlabel("C_FN/C_FP"); axes[0].set_ylabel("정규화 기대비용(TEST)")
axes[0].legend(fontsize=7); axes[0].set_title("SEL 선택 -> TEST 적용 비용")
axes[1].set_xscale("log"); axes[1].set_xlabel("C_FN/C_FP"); axes[1].legend(fontsize=7)
axes[1].set_title("이벤트 포착률(TEST)")
fig.suptitle("FIGURE 8. 목표시각 기준 순차 경보 — 정책선택(SEL)과 평가(TEST) 분리", fontsize=9)
fig.tight_layout(); fig.savefig(fig_path("28_sequential_policy")); plt.close(fig)
