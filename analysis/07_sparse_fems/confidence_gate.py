"""07-G. 신뢰도 게이트 의사결정 정책 — 자동화 등급을 신뢰도로 나눌 수 있는가.

07F_row_scores.csv (밴드 / 예측오차 / 구간 / 피크위험) 를 쓴다.
임계를 임의로 고르지 않고 훑어서 운용곡선을 만든다.
정책 등급:
  AUTO        : 신뢰도 HIGH + 피크위험 높음   → 자동 개입 후보
  AUTO_NORMAL : 신뢰도 HIGH + 피크위험 낮음   → 정상 자동운전
  ADVISE      : 신뢰도 MEDIUM + 피크위험 높음 → 작업자 권고
  WARN_ONLY   : 신뢰도 LOW                    → 경고만
  HUMAN_REVIEW: OOD                            → 자동제어 금지
안전 인증 주장이 아니라 정책 실험이다.
"""
import numpy as np, pandas as pd
from _fe import save_table, fig_path, mpl, TAB
plt = mpl()

s = pd.read_csv(TAB / "07F_row_scores.csv", parse_dates=["ts15"], encoding="utf-8-sig")
thr_peak = float(np.quantile(s.y_true, .95))     # 보고용 표시값 (임계 자체는 분위 훑기로 대체)
rows, tier_rows = [], []
for rq in np.round(np.arange(0.80, 0.995, 0.01), 3):
    t = float(np.quantile(s.pred, rq))
    risk = s.pred.values >= t
    tier = np.where(s.band.values == "OOD", "HUMAN_REVIEW",
           np.where(s.band.values == "LOW", "WARN_ONLY",
           np.where((s.band.values == "MEDIUM") & risk, "ADVISE",
           np.where((s.band.values == "MEDIUM") & ~risk, "MONITOR",
           np.where(risk, "AUTO", "AUTO_NORMAL")))))
    auto = np.isin(tier, ["AUTO", "AUTO_NORMAL"])
    act = tier == "AUTO"
    peak = s.is_peak.values == 1
    rows.append(dict(risk_quantile=rq, risk_threshold_kw=t,
                     auto_eligible_share=float(auto.mean()),
                     auto_action_share=float(act.mean()),
                     human_review_share=float((tier == "HUMAN_REVIEW").mean()),
                     warn_only_share=float((tier == "WARN_ONLY").mean()),
                     advise_share=float((tier == "ADVISE").mean()),
                     peak_recall_overall=float((act & peak).sum() / max(peak.sum(), 1)),
                     # 실무상 '조치가 발생하는 집합' = AUTO + ADVISE (작업자 권고 포함)
                     actionable_share=float(np.isin(tier, ["AUTO", "ADVISE"]).mean()),
                     peak_recall_actionable=float((np.isin(tier, ["AUTO", "ADVISE"]) & peak).sum() / max(peak.sum(), 1)),
                     precision_actionable=float((np.isin(tier, ["AUTO", "ADVISE"]) & peak).sum()
                                                / max(np.isin(tier, ["AUTO", "ADVISE"]).sum(), 1)),
                     peak_recall_within_eligible=float((act & peak).sum() / max((auto & peak).sum(), 1)),
                     false_action_rate=float((act & ~peak).sum() / max(act.sum(), 1)),
                     mae_auto=float(s.abs_err[auto].mean()) if auto.any() else np.nan,
                     mae_non_auto=float(s.abs_err[~auto].mean()) if (~auto).any() else np.nan,
                     unresolved_share=float(np.isin(tier, ["ADVISE", "WARN_ONLY", "HUMAN_REVIEW"]).mean())))
    if abs(rq - 0.90) < 1e-9:
        for tg in np.unique(tier):
            m = tier == tg
            tier_rows.append(dict(risk_quantile=rq, tier=tg, n=int(m.sum()), share=float(m.mean()),
                                  mae=float(s.abs_err[m].mean()),
                                  peak_prevalence=float(peak[m].mean()),
                                  interval_coverage=float(1 - s.interval_fail[m].mean()),
                                  mean_interval_width=float(s.int_width[m].mean())))
cur = pd.DataFrame(rows)
save_table(cur, "07G_confidence_gate_curve")
save_table(pd.DataFrame(tier_rows), "07G_tier_profile_at_q90")

fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
axes[0].plot(cur.risk_quantile, cur.peak_recall_overall, "o-", label="피크 포착률")
axes[0].plot(cur.risk_quantile, cur.false_action_rate, "s-", label="오작동률")
axes[0].plot(cur.risk_quantile, cur.auto_action_share, "^-", label="자동개입 비율")
axes[0].set_xlabel("피크위험 임계 분위"); axes[0].legend(fontsize=7); axes[0].set_title("게이트 운용곡선")
tp = pd.DataFrame(tier_rows).sort_values("mae")
axes[1].barh(range(len(tp)), tp.mae, color="#27b")
axes[1].set_yticks(range(len(tp))); axes[1].set_yticklabels(tp.tier, fontsize=7)
axes[1].set_xlabel("등급별 실측 MAE (kw)"); axes[1].set_title("자동화 등급별 실제 오차 (q=0.90)")
fig.tight_layout(); fig.savefig(fig_path("07G_confidence_gate")); plt.close(fig)
print(cur.to_string(index=False)); print(pd.DataFrame(tier_rows).to_string(index=False))
