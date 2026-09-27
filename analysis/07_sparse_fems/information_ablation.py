"""07-B. 정보 ablation — 정보 수준(L0~L4) × horizon 별 예측/피크/신뢰도 성능.

목표는 변수 중요도가 아니라 "특정 운영 품질을 달성하는 데 필요한 최소 정보 수준".
L4(잠재문맥)는 07D 가 캐시(07D_latent_state_features.csv)를 만들어 둔 경우에만 포함한다.
"""
import numpy as np, pandas as pd
from pathlib import Path
import _fe
from _fe import (build, window, split, peak_threshold, metrics, paired_boot,
                 save_table, fig_path, mpl, SEED, TAB)
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score, roc_auc_score
plt = mpl()

LAT_CACHE = TAB / "07D_latent_state_features.csv"
LAT_COLS = ["lat_state", "lat_dwell", "lat_p_high"]
levels = dict(_fe.LEVELS)
latent_cols = None
if LAT_CACHE.exists():
    latent_cols = LAT_COLS
    levels["L4_+latent_context"] = _fe.LEVELS["L3_+production"] + LAT_COLS
q_all = build(latent_cols=latent_cols)

def qr(a):
    return HistGradientBoostingRegressor(loss="quantile", quantile=a, random_state=SEED)

BUDGETS = [0.02, 0.05, 0.10, 0.20]
rows, err_store, budget_rows = [], {}, []
for wtag in ["all_2021", "clean_Jul_Sep"]:
    sub = window(q_all, wtag)
    ALLF = sorted(set().union(*levels.values()))
    for h in _fe.HORIZONS:
        # 정보수준 간 공정 비교: 모든 수준의 특성이 결측 없는 공통 행집합만 사용한다.
        sub_h = sub.dropna(subset=ALLF + [f"y_h{h}"]).reset_index(drop=True)
        for lname, F in levels.items():
            tr, ca, te = split(sub_h, F, h)
            fit = pd.concat([tr, ca])
            y, yv = te[f"y_h{h}"].values, f"y_h{h}"
            thr = peak_threshold(fit)
            rq = float(np.quantile(np.abs(fit.kw_ramp4), .9))
            is_peak = (y >= thr).astype(int)
            p = HistGradientBoostingRegressor(random_state=SEED).fit(fit[F].values, fit[yv].values).predict(te[F].values)
            m = metrics(te, y, p, thr, rq)
            # 예측구간: CQR(90%) — 보정은 calibration 구간에서만
            m_lo, m_hi = qr(.05).fit(tr[F].values, tr[yv].values), qr(.95).fit(tr[F].values, tr[yv].values)
            E = np.maximum(m_lo.predict(ca[F].values) - ca[yv].values,
                           ca[yv].values - m_hi.predict(ca[F].values))
            Q = float(np.quantile(E, min(1.0, .9 * (1 + 1 / len(E)))))
            lo, hi = m_lo.predict(te[F].values) - Q, m_hi.predict(te[F].values) + Q
            inside = (y >= lo) & (y <= hi)
            # 피크 점수는 회귀 예측값 자체(운영상 추가 모델 불필요)
            rows.append(dict(window=wtag, horizon_min=h * 15, level=lname, n_features=len(F),
                             **m, peak_auc=float(roc_auc_score(is_peak, p)) if is_peak.sum() else np.nan,
                             peak_ap=float(average_precision_score(is_peak, p)) if is_peak.sum() else np.nan,
                             cqr_coverage=float(inside.mean()),
                             cqr_coverage_peak=float(inside[is_peak == 1].mean()) if is_peak.sum() else np.nan,
                             cqr_width=float((hi - lo).mean()), conformal_Q=Q,
                             peak_threshold_kw=thr))
            err_store[(wtag, h, lname)] = np.abs(y - p)
            for b in BUDGETS:                      # 고정 경보 예산에서의 포착률
                k = max(1, int(round(b * len(p))))
                sel = np.zeros(len(p), bool); sel[np.argsort(-p)[:k]] = True
                tp = int((sel & (is_peak == 1)).sum()); fp = int((sel & (is_peak == 0)).sum())
                budget_rows.append(dict(window=wtag, horizon_min=h * 15, level=lname,
                                        alert_budget=b, alerts_per_day=b * 96,
                                        recall=tp / max(is_peak.sum(), 1),
                                        precision=tp / max(tp + fp, 1),
                                        false_alarms=fp,
                                        false_alarm_rate=fp / max((is_peak == 0).sum(), 1),
                                        missed_peaks=int(is_peak.sum() - tp)))
res = pd.DataFrame(rows)
best = res.groupby(["window", "horizon_min"]).mae.min()
res["mae_ratio_vs_best_level"] = [r.mae / best[(r.window, r.horizon_min)] for r in res.itertuples()]
save_table(res, "07B_information_ablation")
save_table(pd.DataFrame(budget_rows), "07B_ablation_alert_budget")

ref = "L3_+production"
boot = [dict(window=w, horizon_min=h * 15, level=l, vs=ref,
             **paired_boot(e, err_store[(w, h, ref)]))
        for (w, h, l), e in err_store.items() if l != ref]
save_table(pd.DataFrame(boot), "07B_ablation_bootstrap")

fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
d = res[res.window == "clean_Jul_Sep"]
for lname in levels:
    s = d[d.level == lname]
    axes[0].plot(s.horizon_min, s.mae, "o-", label=lname)
    axes[1].plot(s.horizon_min, s.peak_ap, "o-", label=lname)
    axes[2].plot(s.horizon_min, s.mae_peak, "o-", label=lname)
for ax, t in zip(axes, ["MAE", "피크 AP", "피크 조건 MAE"]):
    ax.set_xlabel("horizon (분)"); ax.set_title(f"clean_Jul_Sep — 정보수준별 {t}")
axes[0].legend(fontsize=6)
fig.tight_layout(); fig.savefig(fig_path("07B_information_ablation")); plt.close(fig)
print(res[["window", "horizon_min", "level", "mae", "mae_operating", "mae_peak",
           "peak_ap", "cqr_coverage", "cqr_coverage_peak"]].to_string(index=False))
