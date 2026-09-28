"""07-Q. 작업자 진단 계층 (Task Q) — 물리적 조치 지시가 아니라 '무엇을 모르는지' 를 알린다.

출력 필드: forecast_horizon / predicted_power / prediction_interval / reliability_band /
           peak_risk / peak_threshold / operating_regime / transition_flag / ood_flag /
           advisory_level / diagnostic_reason / information_gap_flag
금지: 설비 지정·정지·생산 조정 등 물리적 조치 문장. 본 스크립트는 그런 문장을 생성하지 않는다.

정책 파라미터는 07O 권고 운용점(비용비 10, 원본구간)에서 가져온다.
정보부족(information_gap) 판정의 유효성은 별도로 검증한다(H20):
  플래그가 실제 큰 오차/구간실패를 선별하는가 — AP / MAE / coverage 로 측정.
"""
import numpy as np, pandas as pd
from _fe import save_table, fig_path, mpl, TAB
from sklearn.metrics import average_precision_score, roc_auc_score
plt = mpl()

o = pd.read_csv(TAB / "07O_policy_rows.csv", parse_dates=["ts15"], encoding="utf-8-sig")
pol = pd.read_csv(TAB / "07O_recommended_operating_point.csv", encoding="utf-8-sig")
P = pol[(pol.window == "oof_clean_Jul_Sep") & (pol.scope == "all_policies") &
        (pol.cost_ratio_FN_FP == 10)].iloc[0]
he = int(P.early_horizon_min // 15) if np.isfinite(P.early_horizon_min) else 4
hc = int(P.confirm_horizon_min // 15) if np.isfinite(P.confirm_horizon_min) else 1
print("적용 정책:", P.policy, "gate", P.gate, "early q", P.early_q, "confirm q", P.confirm_q)

s = o[o.clean].reset_index(drop=True).copy()
qe = float(np.quantile(s[f"pred_h{he}"], P.early_q))
qc = float(np.quantile(s[f"pred_h{hc}"], P.confirm_q)) if np.isfinite(P.confirm_q) else np.inf
early = (s[f"pred_h{he}"] >= qe).values
conf = (s[f"pred_h{hc}"] >= qc).values if np.isfinite(qc) else np.ones(len(s), bool)
unsure = s.band.isin(["LOW", "OOD"]).values
trans = (s.regime == "transition").values

# ---- 등급: 조치 지시가 아니라 '주의 수준'
alert = early & (conf | ~unsure) if P.gate == "confirm_only_if_unsure" else early & conf
advis = early & ~conf & unsure
level = np.where(s.ood_flag.values == 1, "HUMAN_REVIEW",
        np.where(alert & ~unsure, "WARN_CONFIRMED",
        np.where(alert & unsure, "WARN_LOW_CONFIDENCE",
        np.where(advis, "ADVISORY_UNCERTAIN",
        np.where(early, "EARLY_WATCH",
        np.where(unsure, "MONITOR_LOW_CONFIDENCE", "NORMAL_MONITORING"))))))

# ---- 진단 사유 (사실 기반 문장만)
w_hi = float(np.quantile(s.int_width, .8))
ramp_hi = float(np.quantile(s.kw_ramp4.abs(), .9))
reasons = []
gap = np.zeros(len(s), bool)
for i, r in s.iterrows():
    rs = []
    if r.ood_flag == 1:
        rs.append("학습분포 밖의 입력(OOD 거리 상위 1%)")
    if r.int_width >= w_hi:
        rs.append("예측구간이 넓어 신뢰도가 낮음")
    if abs(r.kw_ramp4) >= ramp_hi:
        rs.append("최근 1시간 부하 변화가 학습구간 상위 10% 수준")
    if r.regime == "transition":
        rs.append("전환형 운전 패턴 감지")
    if early[i] and not conf[i]:
        rs.append("장기 시점은 피크 위험, 단기 확인에서는 미확인 — 시점 불일치")
    if r.band == "LOW" and r.is_operating == 0:
        rs.append("비조업 구간인데 불확실성이 큼")
    if not rs:
        rs.append("예측 신뢰 구간 정상 — 통상 감시")
    # 정보부족: 불확실성이 크고, 그 원인을 현재 보유 정보로 지목할 수 없는 경우
    ctx_known = (abs(r.kw_ramp4) >= ramp_hi) or (r.regime == "transition") or (r.is_operating == 0)
    if (r.ood_flag == 1 or r.int_width >= w_hi) and not ctx_known:
        rs.append("추가 공정/설비 상태정보가 있어야 원인 구분 가능")
        gap[i] = True
    reasons.append(" / ".join(rs))
s["advisory_level"] = level
s["diagnostic_reason"] = reasons
s["information_gap_flag"] = gap.astype(int)
s["transition_flag"] = trans.astype(int)
s["peak_risk_early"] = early.astype(int)
s["peak_risk_confirmed"] = alert.astype(int)
s["prediction_interval"] = ("[" + s.int_lo.round(1).astype(str) + ", " + s.int_hi.round(1).astype(str) + "]")

OUT = ["ts15", "forecast_horizon", "predicted_power", "prediction_interval", "reliability_band",
       "peak_risk_early", "peak_risk_confirmed", "peak_threshold", "operating_regime",
       "transition_flag", "ood_flag", "advisory_level", "diagnostic_reason", "information_gap_flag",
       "actual_power", "abs_error"]
s["forecast_horizon"] = f"t+{he*15}min"
s["predicted_power"] = s[f"pred_h{he}"].round(2)
s["reliability_band"] = s.band
s["peak_threshold"] = s.thr_adaptive.round(1)
s["operating_regime"] = s.regime
s["actual_power"] = s[f"y_h{he}"].round(2)
s["abs_error"] = (s[f"y_h{he}"] - s[f"pred_h{he}"]).abs().round(2)

# ---- 등급/사유 분포
dist = s.groupby("advisory_level").agg(
    n=("ts15", "size"), share=("ts15", lambda x: len(x) / len(s)),
    per_day=("ts15", lambda x: len(x) / (len(s) / 96)),
    mae=("abs_error", "mean"), mean_interval_width=("int_width", "mean"),
    peak_prevalence=("peak_h4", "mean"), transition_share=("transition_flag", "mean"),
    info_gap_share=("information_gap_flag", "mean")).reset_index()
save_table(dist, "07Q_advisory_level_distribution")

# ---- 대표 사례 표: 등급 x 레짐 x 밴드 층화 추출, 최소 50건
rng = np.random.default_rng(20260926)
parts = []
for _, g in s.groupby(["advisory_level", "operating_regime", "reliability_band"]):
    k = min(len(g), 2)
    parts.append(g.sample(k, random_state=20260926))
samp = pd.concat(parts)
if len(samp) < 50:
    samp = pd.concat([samp, s.drop(samp.index).sample(50 - len(samp), random_state=20260926)])
samp = samp.sort_values("ts15")
save_table(samp[OUT], "07Q_operator_advisory_sample")

# ---- H20 검증: 정보부족 플래그가 실제 어려운 사례를 선별하는가
hi_err = (s.abs_error >= np.quantile(s.abs_error, .8)).astype(int)
fail = ((s[f"y_h{he}"] < s.int_lo) | (s[f"y_h{he}"] > s.int_hi)).astype(int)
vrows = []
for fname, fl in [("information_gap_flag", s.information_gap_flag.values),
                  ("ood_flag", s.ood_flag.values),
                  ("band_LOW_or_OOD", s.band.isin(["LOW", "OOD"]).astype(int).values),
                  ("transition_flag", s.transition_flag.values),
                  ("int_width_top20", (s.int_width >= np.quantile(s.int_width, .8)).astype(int).values)]:
    if fl.sum() == 0:
        vrows.append(dict(flag=fname, n_flagged=0, share=0.0)); continue
    vrows.append(dict(flag=fname, n_flagged=int(fl.sum()), share=float(fl.mean()),
                      mae_flagged=float(s.abs_error[fl == 1].mean()),
                      mae_unflagged=float(s.abs_error[fl == 0].mean()),
                      mae_ratio=float(s.abs_error[fl == 1].mean() / s.abs_error[fl == 0].mean()),
                      coverage_flagged=float(1 - fail[fl == 1].mean()),
                      coverage_unflagged=float(1 - fail[fl == 0].mean()),
                      peak_prevalence_flagged=float(s.peak_h4[fl == 1].mean()),
                      auc_for_high_error=float(roc_auc_score(hi_err, fl)),
                      ap_for_high_error=float(average_precision_score(hi_err, fl)),
                      lift_over_base=float(average_precision_score(hi_err, fl) / hi_err.mean())))
val = pd.DataFrame(vrows)
save_table(val, "07Q_information_gap_validation")

# 정보부족 플래그의 고유 질문: '불확실한데 이유를 현재 정보로 못 짚는' 경우가
# '이유를 짚을 수 있는' 경우와 실제로 다른가.
uns = s.band.isin(["LOW", "OOD"]).values
srows = []
for tag, m in [("unsure_and_info_gap", uns & (s.information_gap_flag == 1).values),
               ("unsure_but_context_explained", uns & (s.information_gap_flag == 0).values),
               ("confident", ~uns)]:
    if m.sum() == 0:
        continue
    srows.append(dict(group=tag, n=int(m.sum()), share=float(m.mean()),
                      mae=float(s.abs_error[m].mean()),
                      coverage=float(1 - fail[m].mean()),
                      mean_interval_width=float(s.int_width[m].mean()),
                      peak_prevalence=float(s.peak_h4[m].mean()),
                      transition_share=float(s.transition_flag[m].mean()),
                      high_error_share=float(hi_err[m].mean())))
save_table(pd.DataFrame(srows), "07Q_information_gap_split")
print(pd.DataFrame(srows).to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
d = dist.sort_values("n")
axes[0].barh(d.advisory_level, d.per_day, color="#27b")
axes[0].set_xlabel("일 평균 발생 건수"); axes[0].set_title("작업자 등급별 부담")
v = val.dropna(subset=["mae_ratio"])
axes[1].bar(v.flag, v.mae_ratio, color="#c33")
axes[1].axhline(1, color="k", lw=.8); axes[1].set_ylabel("MAE 비(플래그/비플래그)")
axes[1].set_xticklabels(v.flag, rotation=20, ha="right", fontsize=6)
axes[1].set_title("진단 플래그의 난이도 선별력")
fig.tight_layout(); fig.savefig(fig_path("07Q_operator_diagnostic")); plt.close(fig)
print(dist.to_string(index=False)); print(val.to_string(index=False))
print("표본 건수:", len(samp))
