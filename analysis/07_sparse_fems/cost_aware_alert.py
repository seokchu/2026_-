"""07-H. 비용 인식 피크 경보 — FP/FN 비용비에 따라 최적 임계가 어떻게 움직이는가.

금액은 공식 문서로 확정되지 않았으므로(09 보고서) 절대 KRW 를 쓰지 않는다.
비용비 C_FN/C_FP = 1,2,5,10,20,50,100 에 대해 정규화 기대비용을 최소화하는 임계를 찾는다.
점수 = OOF 회귀 예측값(kw). 임계는 예측값 분위를 훑는다.
"""
import numpy as np, pandas as pd
from _fe import save_table, fig_path, mpl, TAB
plt = mpl()

o = pd.read_csv(TAB / "07C_oof_predictions.csv", parse_dates=["ts15"], encoding="utf-8-sig")
RATIOS = [1, 2, 5, 10, 20, 50, 100]
QS = np.round(np.arange(0.50, 0.999, 0.01), 3)
rows, best = [], []
for wtag, sub in [("oof_all_2021", o), ("oof_clean_Jul_Sep", o[o.clean])]:
    y, p = sub.is_peak.values, sub.pred.values
    n_peak, n_norm = int(y.sum()), int((y == 0).sum())
    if n_peak == 0:
        continue
    for qv in QS:
        t = float(np.quantile(p, qv))
        a = p >= t
        tp, fp = int((a & (y == 1)).sum()), int((a & (y == 0)).sum())
        fn, tn = n_peak - tp, n_norm - fp
        for r in RATIOS:
            cost = (fp + r * fn) / (r * n_peak)          # 전부 놓쳤을 때 비용 = 1 로 정규화
            rows.append(dict(window=wtag, quantile=qv, threshold_kw=t, cost_ratio_FN_FP=r,
                             expected_normalized_cost=cost, precision=tp / max(tp + fp, 1),
                             recall=tp / n_peak, alerts_per_day=float(a.mean() * 96),
                             alert_frequency=float(a.mean()), false_alarms=fp,
                             false_alarms_per_day=fp / (len(sub) / 96), missed_peaks=fn,
                             tp=tp, tn=tn, n=len(sub)))
cur = pd.DataFrame(rows)
save_table(cur, "07H_cost_alert_curve")
opt = cur.loc[cur.groupby(["window", "cost_ratio_FN_FP"]).expected_normalized_cost.idxmin()]
save_table(opt[["window", "cost_ratio_FN_FP", "quantile", "threshold_kw", "expected_normalized_cost",
                "precision", "recall", "alerts_per_day", "false_alarms_per_day", "missed_peaks"]],
           "07H_cost_optimal_policy")

fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
d = opt[opt.window == "oof_clean_Jul_Sep"]
axes[0].plot(d.cost_ratio_FN_FP, d["quantile"], "o-"); axes[0].set_xscale("log")
axes[0].set_xlabel("C_FN / C_FP"); axes[0].set_ylabel("최적 임계 분위"); axes[0].set_title("비용비 → 최적 임계")
axes[1].plot(d.cost_ratio_FN_FP, d.recall, "o-", label="recall")
axes[1].plot(d.cost_ratio_FN_FP, d.precision, "s-", label="precision")
axes[1].set_xscale("log"); axes[1].legend(); axes[1].set_xlabel("C_FN / C_FP")
axes[1].set_title("비용비 → 포착률/정밀도")
axes[2].plot(d.cost_ratio_FN_FP, d.alerts_per_day, "o-")
axes[2].set_xscale("log"); axes[2].set_xlabel("C_FN / C_FP"); axes[2].set_ylabel("일 경보 수")
axes[2].set_title("비용비 → 경보 부담 (원본 구간)")
fig.tight_layout(); fig.savefig(fig_path("07H_cost_aware_alert")); plt.close(fig)
print(opt.to_string(index=False))
