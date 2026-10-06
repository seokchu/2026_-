"""30-B. 놓친 사건(FN)·헛경고(FP)를 **원인 설명이 아니라 실패 위치**로 분해한다.

사전 규칙(결과 보기 전 고정):
  ① forecast_miss      : 수치 예측 자체가 임계를 크게 밑돌았다 (pred_margin < -2 x 해당 창 MAE)
  ② rule_blocked       : 예측은 위험을 가리켰으나(pred_margin >= -1 x MAE) 경고 규칙이 막았다
                         (조기 임계 미달 또는 확인 단계 기각)
  ③ undetermined       : 위 둘로 갈리지 않는다 — **'본질적 예측 불가' 나 '센서 부족' 으로 부르지 않는다**
복구 가능성 점검(③·② 대상): 더 낮은 임계(q0.50) / 보정 확률 >= 0.5 / 다른 확인 단계(T-45) 로
  잡혔는지를 같은 사건에 대해 확인한다. 이것이 '다른 모델·기준으로 개선 가능한가' 의 1차 답이다.
평가 기간은 TEST(15일)·LOCK(15일)을 분리 보고한다. 학습 구간은 포함하지 않는다.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, yaml
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / "analysis"))
from common import save_table, TAB                                       # noqa: E402

cfg = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
PRE = cfg["output"]["table_prefix"]
STAGES = {"early_h60": (4, 60), "confirmA_h45": (3, 45), "confirmB_h15": (1, 15)}
LOCK_DAYS = TEST_DAYS = 14

o = pd.read_csv(TAB / f"{PRE}_oof_rows.csv", parse_dates=["ts15"], encoding="utf-8-sig").sort_values("ts15")
thr = pd.read_csv(TAB / f"{PRE}_cal_alert_thresholds.csv", encoding="utf-8-sig")
TM = {(int(r.fold), int(r.horizon_min), float(r.q)): float(r.threshold_kw) for r in thr.itertuples()}
prob = pd.read_csv(TAB / "28_peak_probability_rows.csv", parse_dates=["ts15"], encoding="utf-8-sig")
p60 = prob[prob.horizon_min == 60][["ts15", "p_cal"]].rename(columns={"p_cal": "p_cal_h60"})

base = o[["ts15", "fold", "clean", "kw", "thr_adaptive", "peak_now", "regime", "is_operating",
          "kw_ramp4", "band_h4", "abs_err_h4"]].rename(columns={"ts15": "T", "peak_now": "label"})
for name, (hh, back) in STAGES.items():
    dec = o[["ts15", "fold", f"pred_h{hh}", f"y_h{hh}"]].copy()
    dec["T"] = dec.ts15 + pd.Timedelta(minutes=back)
    dec = dec.rename(columns={"ts15": f"t_{name}", "fold": f"fold_{name}",
                              f"pred_h{hh}": f"score_{name}", f"y_h{hh}": f"target_{name}"})
    base = base.merge(dec, on="T", how="left")
base = base.merge(p60.rename(columns={"ts15": "t_early_h60"}), on="t_early_h60", how="left")
tt = base.dropna(subset=["score_early_h60", "score_confirmA_h45", "score_confirmB_h15"]).reset_index(drop=True)
tt["date"] = tt["T"].dt.normalize()
cl = tt[tt.clean].reset_index(drop=True)
end = cl["T"].max(); lock_start = end - pd.Timedelta(days=LOCK_DAYS)
test_start = lock_start - pd.Timedelta(days=TEST_DAYS)
WINDOWS = {"TEST": cl[(cl["T"] >= test_start) & (cl["T"] < lock_start)].reset_index(drop=True),
           "LOCK_완전잠금": cl[cl["T"] >= lock_start].reset_index(drop=True)}

pol = pd.read_csv(TAB / "28_sequential_operating_points.csv", encoding="utf-8-sig")
r0 = cfg["policy"]["default_cost_ratio"]
POLS = {}
for scope in ["all_policies", f"lead_constrained_ge{cfg['policy']['minimum_lead_minutes']}min"]:
    r = pol[(pol.cost_ratio_FN_FP == r0) & (pol.scope == scope) & (pol.applied_to == "TEST")]
    if len(r):
        POLS[scope] = r.iloc[0]


def mask_stage(sub, stage, qv):
    hh, _ = STAGES[stage]
    t = np.array([TM[(int(f), hh * 15, float(qv))] for f in sub[f"fold_{stage}"].values])
    return sub[f"score_{stage}"].values >= t, t


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


rows, fp_rows = [], []
for pname, P in POLS.items():
    for wtag, sub in WINDOWS.items():
        if not len(sub):
            continue
        mae_w = float(sub.abs_err_h4.mean())
        early, thr_e = mask_stage(sub, "early_h60", P.early_q)
        if isinstance(P.confirm_stage, str) and P.confirm_stage:
            conf, thr_c = mask_stage(sub, str(P.confirm_stage), P.confirm_q)
            cstage = str(P.confirm_stage)
        else:
            conf, thr_c, cstage = np.ones(len(sub), bool), np.full(len(sub), np.nan), ""
        unsure = sub.band_h4.isin(cfg["diagnosis"]["uncertain_bands"]).values
        alert = early & (conf | ~unsure) if P.gate == "confirm_only_if_unsure" else early & conf
        lab = sub.label.values == 1
        # 더 낮은 임계 / 확률 / 다른 확인단계로 복구 가능한가
        low_early, _ = mask_stage(sub, "early_h60", 0.50)
        # 다른 확인 단계(T-45)로 복구 가능했는지: 확인 분위가 없는 단일 정책이면 0.82 를 대입한다
        alt_q = float(P.confirm_q) if np.isfinite(pd.to_numeric(P.confirm_q, errors="coerce")) else 0.82
        altA, _ = mask_stage(sub, "confirmA_h45", alt_q) if cstage != "confirmA_h45" else (conf, None)
        prob_hit = sub.p_cal_h60.values >= 0.5
        pred_margin = sub.score_early_h60.values - sub.thr_adaptive.values
        for s, e in runs(lab):
            caught = bool(alert[s:e + 1].any())
            i = s + int(np.argmax(pred_margin[s:e + 1]))     # 사건 내 가장 유리한 시점
            pm = float(pred_margin[i])
            if caught:
                cause = "caught"
            elif pm < -2 * mae_w:
                cause = "①forecast_miss"
            elif pm >= -1 * mae_w:
                cause = "②rule_blocked"
            else:
                cause = "③undetermined"
            rows.append(dict(policy_scope=pname, window=wtag, onset=str(sub["T"].iloc[s]),
                             n_steps=e - s + 1, cause=cause, caught=caught,
                             pred_margin_best_kw=pm, mae_window=mae_w,
                             actual_peak_kw=float(sub.kw.iloc[s:e + 1].max()),
                             peak_threshold_kw=float(sub.thr_adaptive.iloc[s]),
                             early_threshold_kw=float(thr_e[i]),
                             early_passed=bool(early[s:e + 1].any()),
                             confirm_passed=bool(conf[s:e + 1].any()) if cstage else np.nan,
                             regime_at_decision=sub.regime.iloc[i],
                             band_at_decision=sub.band_h4.iloc[i],
                             p_cal_h60_best=float(np.nanmax(sub.p_cal_h60.values[s:e + 1])),
                             recover_low_threshold_q050=bool(low_early[s:e + 1].any()),
                             recover_prob_ge_050=bool(np.nan_to_num(prob_hit[s:e + 1]).any()),
                             recover_other_confirm_stage=bool(altA[s:e + 1].any()) if altA is not None else np.nan))
        # 헛경고: 경보 사건 중 라벨 0 만 포함한 구간
        for s, e in runs(alert):
            if lab[s:e + 1].any():
                continue
            i = s + int(np.argmax(pred_margin[s:e + 1]))
            fp_rows.append(dict(policy_scope=pname, window=wtag, start=str(sub["T"].iloc[s]),
                                n_steps=e - s + 1,
                                max_pred_margin_kw=float(pred_margin[s:e + 1].max()),
                                max_actual_margin_kw=float((sub.kw.values - sub.thr_adaptive.values)[s:e + 1].max()),
                                regime_at_decision=sub.regime.iloc[i],
                                band_at_decision=sub.band_h4.iloc[i],
                                p_cal_h60_best=float(np.nanmax(sub.p_cal_h60.values[s:e + 1])),
                                near_miss=bool(((sub.kw.values - sub.thr_adaptive.values)[s:e + 1] >= -5).any())))
ev = pd.DataFrame(rows); save_table(ev, "30_missed_event_taxonomy")
fp = pd.DataFrame(fp_rows); save_table(fp, "30_false_alarm_cases")

summ = (ev.groupby(["policy_scope", "window", "cause"])
        .agg(n_events=("onset", "size"),
             mean_pred_margin_kw=("pred_margin_best_kw", "mean"),
             mean_p_cal=("p_cal_h60_best", "mean"),
             recover_low_threshold=("recover_low_threshold_q050", "mean"),
             recover_prob=("recover_prob_ge_050", "mean"),
             recover_other_confirm=("recover_other_confirm_stage", "mean")).reset_index())
tot = ev.groupby(["policy_scope", "window"]).onset.size().rename("n_all_events").reset_index()
summ = summ.merge(tot, on=["policy_scope", "window"])
summ["share_of_events"] = summ.n_events / summ.n_all_events
save_table(summ, "30_missed_event_summary")
fps = (fp.groupby(["policy_scope", "window"])
       .agg(n_false_alarm_events=("start", "size"), mean_steps=("n_steps", "mean"),
            near_miss_share=("near_miss", "mean"),
            mean_pred_margin_kw=("max_pred_margin_kw", "mean")).reset_index()) if len(fp) else pd.DataFrame()
if len(fps):
    save_table(fps, "30_false_alarm_summary")
print(summ.round(3).to_string(index=False))
if len(fps):
    print(fps.round(3).to_string(index=False))
print(ev[ev.cause != "caught"].head(12).to_string(index=False))
