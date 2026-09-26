"""05b. Re-run the peak findings on the duplicate-free original window only (2021-07-01~09-14)."""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score, average_precision_score, mean_absolute_error
from common import load_power, power_long, save_table, SEED
rng = np.random.default_rng(SEED)

p = load_power(); q = power_long(p).sort_values("ts15").reset_index(drop=True)
rows = []
for tag, sub in [("all_2021", q), ("clean_Jul_Sep", q[q.날짜 >= 20210701].reset_index(drop=True))]:
    s = sub.kw.values
    # precursor
    hi = np.quantile(s, .99)
    on = np.where((s >= hi) & (np.r_[False, s[:-1] < hi]))[0]; on = on[on >= 8]
    bgi = rng.choice(np.setdiff1d(np.arange(8, len(s)), on), size=min(2000, len(s) - 9), replace=False)
    sl_on = (s[on - 1] - s[on - 4]) / 3
    sl_bg = (s[bgi - 1] - s[bgi - 4]) / 3
    rows.append(dict(window=tag, n_points=len(sub), metric="ramp_slope_1h_before_p99",
                     n_onsets=len(on), median_onset=float(np.median(sl_on)),
                     median_background=float(np.median(sl_bg)),
                     mwu_p=float(stats.mannwhitneyu(sl_on, sl_bg).pvalue)))
save_table(pd.DataFrame(rows), "05b_power_precursor_clean_vs_all")

# peak classification + forecast, temporal split, clean window only
out = []
for tag, sub in [("all_2021", q), ("clean_Jul_Sep", q[q.날짜 >= 20210701].reset_index(drop=True))]:
    sub = sub.copy()
    for k in (1, 2, 4, 96):
        sub[f"kw_lag{k}"] = sub.kw.shift(k)
    sub["kw_roll4"] = sub.kw.shift(1).rolling(4).mean()
    sub["kw_roll96"] = sub.kw.shift(1).rolling(96).mean()
    FE = ["시간", "day", "m", "기온", "습도", "풍속", "강수량", "생산량", "공장인원",
          "kw_lag1", "kw_lag2", "kw_lag4", "kw_lag96", "kw_roll4", "kw_roll96"]
    thr = np.quantile(sub.kw, .95)
    sub["peak_t4"] = (sub.kw.shift(-4) >= thr).astype(float)
    cl = sub.dropna(subset=["kw_roll96", "peak_t4"])
    X = cl[FE].fillna(-1).values; y = cl.peak_t4.values.astype(int); cut = int(len(cl) * .8)
    pr = RandomForestClassifier(n_estimators=400, min_samples_leaf=5, class_weight="balanced_subsample",
                                random_state=SEED, n_jobs=-1).fit(X[:cut], y[:cut]).predict_proba(X[cut:])[:, 1]
    out.append(dict(window=tag, task="t+1h p95 peak", n_train=cut, n_test=len(cl) - cut,
                    prevalence_test=float(y[cut:].mean()),
                    auc=float(roc_auc_score(y[cut:], pr)), ap=float(average_precision_score(y[cut:], pr))))
    yr = cl.kw.values
    pf = HistGradientBoostingRegressor(random_state=SEED).fit(X[:cut], yr[:cut]).predict(X[cut:])
    out.append(dict(window=tag, task="kw(t) regression", n_train=cut, n_test=len(cl) - cut,
                    prevalence_test=np.nan, auc=np.nan,
                    ap=np.nan, mae=float(mean_absolute_error(yr[cut:], pf)),
                    mae_persistence=float(mean_absolute_error(yr[cut:], cl.kw_lag1.values[cut:])),
                    mae_over_std=float(mean_absolute_error(yr[cut:], pf) / yr[cut:].std())))
save_table(pd.DataFrame(out), "05b_power_baselines_clean_vs_all")

# equal-output peak spread, clean window only
dayq = q[q.날짜 >= 20210701].groupby("날짜").agg(kw_peak=("kw", "max"), kwh=("kw", "sum"),
                                              prod=("생산량", "sum"), temp=("기온", "mean"),
                                              active=("생산량", lambda z: int((z > 0).sum() / 4))).reset_index()
dayq = dayq[dayq["prod"] > 0]
dayq["bin"] = pd.qcut(dayq["prod"], 3, duplicates="drop")
eq = []
for b, g_ in dayq.groupby("bin", observed=True):
    lo = g_[g_.kw_peak <= g_.kw_peak.median()]; hiq = g_[g_.kw_peak > g_.kw_peak.median()]
    for c in ["temp", "active", "kwh", "prod"]:
        eq.append(dict(prod_bin=str(b), n_lo=len(lo), n_hi=len(hiq), feature=c,
                       median_low_peak=float(lo[c].median()), median_high_peak=float(hiq[c].median()),
                       mwu_p=float(stats.mannwhitneyu(lo[c], hiq[c]).pvalue)))
save_table(pd.DataFrame(eq), "05b_power_equal_output_clean_window")
print("done")
