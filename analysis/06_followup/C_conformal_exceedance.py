"""06C. Quantile / Conformal 기반 초과수요 점수가 실제 extreme demand 를 선별하는가.

타깃: 1시간 뒤 수요 kw(t+4). 시간순 3분할 train / calibration / test.
  - QR   : HistGradientBoostingRegressor(loss="quantile") 분위 예측 (보정 없음)
  - CQR  : Conformalized Quantile Regression (Romano et al. 2019) — 분위 예측을
           calibration 잔차로 보정해 marginal coverage 를 보장한다.
경보 규칙: 상한 예측이 피크 임계(thr) 이상이면 경보.
측정: coverage / false alarm / missed peak / alert frequency / precision / recall.
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from common import load_power, power_long, save_table, fig_path, mpl, SEED
plt = mpl()

q = power_long(load_power()).sort_values("ts15").reset_index(drop=True)
for k in (1, 2, 4, 96):
    q[f"kw_lag{k}"] = q.kw.shift(k)
q["kw_roll4"] = q.kw.shift(1).rolling(4).mean()
q["kw_roll96"] = q.kw.shift(1).rolling(96).mean()
q["is_operating"] = (q["생산량"] > 0).astype(int)
q["y_next1h"] = q.kw.shift(-4)
q["clean"] = q.날짜 >= 20210701
FE = ["시간", "day", "m", "기온", "습도", "풍속", "강수량", "생산량", "is_operating",
      "kw_lag1", "kw_lag2", "kw_lag4", "kw_lag96", "kw_roll4", "kw_roll96", "kw"]

def qr(a):
    return HistGradientBoostingRegressor(loss="quantile", quantile=a, random_state=SEED)

cov_rows, alert_rows, curve_rows = [], [], []
for wtag, sub in [("all_2021", q), ("clean_Jul_Sep", q[q.clean])]:
    sub = sub.dropna(subset=["kw_roll96", "y_next1h"]).reset_index(drop=True)
    n = len(sub)
    i1, i2 = int(n * .6), int(n * .8)
    tr, ca, te = sub.iloc[:i1], sub.iloc[i1:i2], sub.iloc[i2:]
    Xtr, Xca, Xte = (d[FE].fillna(-1).values for d in (tr, ca, te))
    ytr, yca, yte = tr.y_next1h.values, ca.y_next1h.values, te.y_next1h.values
    thr = float(np.quantile(sub.kw, .95))          # 피크 임계: 해당 구간 p95
    is_peak = (yte >= thr).astype(int)

    # ---- CQR: 중앙 90% 구간 + 상한 전용 분위들
    for alpha, lo_a, hi_a in [(0.10, 0.05, 0.95), (0.20, 0.10, 0.90)]:
        m_lo, m_hi = qr(lo_a).fit(Xtr, ytr), qr(hi_a).fit(Xtr, ytr)
        c_lo, c_hi = m_lo.predict(Xca), m_hi.predict(Xca)
        E = np.maximum(c_lo - yca, yca - c_hi)
        Q = float(np.quantile(E, min(1.0, (1 - alpha) * (1 + 1 / len(E)))))
        t_lo, t_hi = m_lo.predict(Xte), m_hi.predict(Xte)
        for tag, lo, hi in [("QR_uncalibrated", t_lo, t_hi),
                            ("CQR_calibrated", t_lo - Q, t_hi + Q)]:
            inside = (yte >= lo) & (yte <= hi)
            cov_rows.append(dict(window=wtag, method=tag, target_coverage=1 - alpha,
                                 empirical_coverage=float(inside.mean()),
                                 coverage_gap=float(inside.mean() - (1 - alpha)),
                                 mean_interval_width=float((hi - lo).mean()),
                                 conformal_Q=Q if tag.startswith("CQR") else np.nan,
                                 cov_operating=float(inside[te.is_operating == 1].mean()),
                                 cov_idle=float(inside[te.is_operating == 0].mean()),
                                 cov_when_peak=float(inside[is_peak == 1].mean()),
                                 n_test=len(te)))

    # ---- 초과수요 점수: 상한 분위 예측 - 임계 (연속 점수) 와 분류기 비교
    m95 = qr(0.95).fit(Xtr, ytr)
    c95 = m95.predict(Xca)
    E95 = yca - c95                                   # 단측 conformal
    Q95 = float(np.quantile(E95, min(1.0, 0.95 * (1 + 1 / len(E95)))))
    up95 = m95.predict(Xte) + Q95
    clf = HistGradientBoostingClassifier(random_state=SEED).fit(
        np.vstack([Xtr, Xca]), (np.r_[ytr, yca] >= thr).astype(int))
    p_clf = clf.predict_proba(Xte)[:, 1]
    for tag, score in [("CQR_upper95_margin", up95 - thr),
                       ("QR_upper95_margin(uncal)", m95.predict(Xte) - thr),
                       ("direct_classifier_prob", p_clf)]:
        alert_rows.append(dict(window=wtag, score=tag, n_test=len(te), threshold_kw=thr,
                               peak_prevalence=float(is_peak.mean()),
                               auc=float(roc_auc_score(is_peak, score)),
                               ap=float(average_precision_score(is_peak, score))))

    # ---- 운용 곡선: 상한 분위 수준을 훑으며 경보 지표
    for a in [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.975, 0.99]:
        m = qr(a).fit(Xtr, ytr)
        cal = m.predict(Xca)
        Qa = float(np.quantile(yca - cal, min(1.0, a * (1 + 1 / len(cal)))))
        for tag, up in [("QR", m.predict(Xte)), ("CQR", m.predict(Xte) + Qa)]:
            alert = (up >= thr).astype(int)
            tp = int(((alert == 1) & (is_peak == 1)).sum())
            fp = int(((alert == 1) & (is_peak == 0)).sum())
            fn = int(((alert == 0) & (is_peak == 1)).sum())
            tn = int(((alert == 0) & (is_peak == 0)).sum())
            curve_rows.append(dict(window=wtag, method=tag, quantile=a,
                                   alert_frequency=float(alert.mean()),
                                   alerts_per_day=float(alert.mean() * 96),
                                   recall_caught_peaks=tp / max(tp + fn, 1),
                                   missed_peaks=fn, missed_peak_rate=fn / max(tp + fn, 1),
                                   false_alarms=fp, false_alarm_rate=fp / max(fp + tn, 1),
                                   precision=tp / max(tp + fp, 1), tp=tp, tn=tn,
                                   conformal_Q=Qa))
save_table(pd.DataFrame(cov_rows), "06C_power_conformal_coverage")
save_table(pd.DataFrame(alert_rows), "06C_power_exceedance_score_ranking")
curve = pd.DataFrame(curve_rows)
save_table(curve, "06C_power_alert_operating_curve")

fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
for ax, wtag in zip(axes, ["all_2021", "clean_Jul_Sep"]):
    for meth, mk in [("QR", "o--"), ("CQR", "s-")]:
        d = curve[(curve.window == wtag) & (curve.method == meth)].sort_values("alert_frequency")
        ax.plot(d.alert_frequency, d.recall_caught_peaks, mk, label=meth)
    ax.set_xlabel("경보 발령 빈도 (전체 15분 스텝 대비)"); ax.set_ylabel("피크 포착률 (recall)")
    ax.set_title(f"{wtag} — 경보 운용 곡선"); ax.legend()
fig.tight_layout(); fig.savefig(fig_path("06C_power_alert_curve")); plt.close(fig)
print("done")
