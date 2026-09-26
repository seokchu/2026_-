"""04. Press hydraulic pump: burst structure, acquisition leakage, coupling, aliasing, separability."""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from scipy import signal, stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict, GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from common import load_press, segment, save_table, fig_path, mpl, SEED
plt = mpl()
CH = ["AI0_Vibration", "AI1_Vibration", "AI2_Current"]
Q = 1.1920929   # 24-bit ADC LSB pattern

n, o = load_press()
n["seg"] = "N" + segment(n).astype(str)
o["seg"] = "O" + segment(o).astype(str)
d = pd.concat([n, o], ignore_index=True)

# ---- 1. sampling / burst geometry
geo = []
for nm, x in [("normal", n), ("outlier", o)]:
    dt = x.TimeStamp.diff().dt.total_seconds()
    sl = x.groupby("seg").size()
    geo.append(dict(file=nm, rows=len(x), t_start=str(x.TimeStamp.min()), t_end=str(x.TimeStamp.max()),
                    wallclock_span_s=float((x.TimeStamp.max() - x.TimeStamp.min()).total_seconds()),
                    median_dt_s=float(dt.median()), fs_hz=float(1 / dt.median()),
                    nyquist_hz=float(.5 / dt.median()),
                    n_bursts=int(sl.size), burst_len_max=int(sl.max()), burst_len_median=float(sl.median()),
                    dead_time_s=float(dt[dt > .15].sum()), duty_cycle=float(len(x) * dt.median() / (x.TimeStamp.max() - x.TimeStamp.min()).total_seconds())))
save_table(pd.DataFrame(geo), "04_press_sampling_geometry")

# ---- 2. acquisition-grid leakage
lk = []
for nm, x in [("normal", n), ("outlier", o)]:
    r = x.AI2_Current.values / Q
    dev = np.abs(r - np.round(r))
    dec = pd.concat([x[c].astype(str).str.split(".").str[-1].str.len() for c in CH[:2]])
    lk.append(dict(file=nm, current_on_ADC_grid=bool(dev.max() < 1e-3), grid_dev_mean=float(dev.mean()),
                   grid_dev_max=float(dev.max()), vib_decimal_places_mode=int(dec.mode().iloc[0]),
                   current_unique_ratio=float(x.AI2_Current.nunique() / len(x)),
                   current_min_step=float(np.diff(np.sort(x.AI2_Current.unique())).min())))
save_table(pd.DataFrame(lk), "04_press_acquisition_leakage")

# a single leakage feature alone
d["grid_resid"] = np.abs(d.AI2_Current.values / Q - np.round(d.AI2_Current.values / Q))
d["decimals"] = d.AI0_Vibration.astype(str).str.split(".").str[-1].str.len()
row_leak = pd.DataFrame([
    dict(feature="grid_resid (|current/1.1920929 - round|)", auc=roc_auc_score(d.Equipment_state, -d.grid_resid)),
    dict(feature="decimals of AI0_Vibration string", auc=roc_auc_score(d.Equipment_state, d.decimals)),
])
save_table(row_leak, "04_press_single_leak_feature_auc")

# ---- 3. window features (physical only)
def feats(g):
    out = {}
    for c in CH:
        x = g[c].values
        ax = np.abs(x)
        rms = np.sqrt((x ** 2).mean())
        out |= {f"{c}_mean": x.mean(), f"{c}_std": x.std(), f"{c}_rms": rms,
                f"{c}_skew": stats.skew(x), f"{c}_kurt": stats.kurtosis(x),
                f"{c}_p2p": x.max() - x.min(), f"{c}_peak": ax.max(),
                f"{c}_crest": ax.max() / rms if rms else np.nan,
                f"{c}_impulse": ax.max() / ax.mean() if ax.mean() else np.nan,
                f"{c}_shape": rms / ax.mean() if ax.mean() else np.nan,
                f"{c}_clearance": ax.max() / (np.sqrt(ax).mean() ** 2) if ax.mean() else np.nan,
                f"{c}_zcr": float(np.mean(np.diff(np.sign(x)) != 0)),
                f"{c}_energy": float((x ** 2).sum()),
                f"{c}_acf1": float(np.corrcoef(x[:-1], x[1:])[0, 1]) if len(x) > 3 else np.nan,
                f"{c}_acf2": float(np.corrcoef(x[:-2], x[2:])[0, 1]) if len(x) > 4 else np.nan}
        if len(x) >= 16:
            f, P = signal.welch(x, fs=10, nperseg=min(len(x), 32))
            Pn = P / P.sum()
            out |= {f"{c}_domfreq": float(f[np.argmax(P)]),
                    f"{c}_centroid": float((f * Pn).sum()),
                    f"{c}_spec_entropy": float(-(Pn * np.log(Pn + 1e-12)).sum()),
                    f"{c}_bandlow": float(P[f <= 1].sum() / P.sum()),
                    f"{c}_bandhigh": float(P[f > 3].sum() / P.sum())}
    out["corr_AI0_AI1"] = float(np.corrcoef(g.AI0_Vibration, g.AI1_Vibration)[0, 1])
    out["corr_AI0_AI2"] = float(np.corrcoef(g.AI0_Vibration, g.AI2_Current)[0, 1])
    out["corr_AI1_AI2"] = float(np.corrcoef(g.AI1_Vibration, g.AI2_Current)[0, 1])
    x0, x1 = g.AI0_Vibration.values, g.AI1_Vibration.values
    cc = signal.correlate(x0 - x0.mean(), x1 - x1.mean(), mode="full")
    out["xcorr_peak_lag_AI0_AI1"] = int(np.argmax(np.abs(cc)) - (len(x0) - 1))
    out["n"] = len(g)
    out["y"] = int(g.Equipment_state.iloc[0])
    out["t0"] = g.TimeStamp.iloc[0]
    return out

W = pd.DataFrame([{"seg": k, **feats(g)} for k, g in d.groupby("seg", sort=False)])
W = W[W.n >= 16].reset_index(drop=True)
save_table(W, "04_press_window_features")

# ---- 4. window-level class comparison
cmp = []
FV = [c for c in W.columns if c not in ("seg", "y", "t0", "n")]
for c in FV:
    a, b = W.loc[W.y == 1, c].dropna(), W.loc[W.y == 0, c].dropna()
    if len(a) < 3 or len(b) < 3 or W[c].nunique() < 3: continue
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    cmp.append(dict(feature=c, auc=u / (len(a) * len(b)), n_out=len(a), n_norm=len(b),
                    median_normal=float(b.median()), median_outlier=float(a.median()), mwu_p=p))
cmp = pd.DataFrame(cmp)
cmp["auc_abs"] = (cmp.auc - .5).abs()
save_table(cmp.sort_values("auc_abs", ascending=False), "04_press_window_feature_separation")

# ---- 5. baseline: leak-free feature sets, burst-grouped CV
def run(cols, tag):
    X = W[cols].replace([np.inf, -np.inf], np.nan).fillna(0).values
    y = W.y.values
    grp = W.seg.values
    out = {}
    for nm, mk in [("logreg", make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))),
                   ("rf", RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
                                                 class_weight="balanced_subsample", random_state=SEED, n_jobs=-1))]:
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
        p = cross_val_predict(mk, X, y, cv=cv, groups=grp, method="predict_proba")[:, 1]
        out[nm] = dict(feature_set=tag, model=nm, n_features=len(cols),
                       auc=roc_auc_score(y, p), ap=average_precision_score(y, p))
    return list(out.values())

amp = [c for c in FV if c.endswith(("_std", "_rms", "_p2p", "_peak", "_energy"))]
shp = [c for c in FV if c.endswith(("_skew", "_kurt", "_crest", "_impulse", "_shape", "_clearance", "_zcr"))]
spc = [c for c in FV if c.endswith(("_domfreq", "_centroid", "_spec_entropy", "_bandlow", "_bandhigh"))]
cpl = [c for c in FV if c.startswith("corr_") or c.startswith("xcorr_")]
sets = {"amplitude_only": amp, "shape_only": shp, "spectral_only": spc, "coupling_only": cpl,
        "vib_only_all": [c for c in FV if c.startswith(("AI0", "AI1"))],
        "current_only_all": [c for c in FV if c.startswith("AI2")],
        "all_physical": FV}
rows = []
for tag, cols in sets.items():
    rows += run(cols, tag)
save_table(pd.DataFrame(rows), "04_press_baseline_by_feature_family")

# ---- 6. coupling breakdown + aliasing evidence
fig, ax = plt.subplots(2, 3, figsize=(12, 6))
ax[0, 0].plot(n.TimeStamp.values[:200], n.AI2_Current.values[:200], ".-", ms=2)
ax[0, 0].set_title("normal AI2_Current: 주기 ≈1.8 s 정현파 (10 Hz 샘플링 → 에일리어싱 의심)")
ax[0, 1].plot(o.TimeStamp.values[:200], o.AI2_Current.values[:200], ".-", ms=2, color="crimson")
ax[0, 1].set_title("outlier AI2_Current: 정현파 소실")
for lbl, x, c in [("normal", n, "steelblue"), ("outlier", o, "crimson")]:
    f, P = signal.welch(x.AI2_Current.values, fs=10, nperseg=256)
    ax[0, 2].semilogy(f, P, label=lbl, color=c)
ax[0, 2].legend(); ax[0, 2].set_xlabel("Hz"); ax[0, 2].set_title("AI2_Current PSD (fs=10 Hz, Nyquist 5 Hz)")
ax[1, 0].hist(W.loc[W.y == 0, "corr_AI0_AI1"], bins=40, alpha=.7, label="normal", density=True)
ax[1, 0].hist(W.loc[W.y == 1, "corr_AI0_AI1"], bins=15, alpha=.7, label="outlier", density=True, color="crimson")
ax[1, 0].legend(); ax[1, 0].set_title("윈도 단위 corr(AI0,AI1): 부호 반전")
ax[1, 1].scatter(W.AI0_Vibration_std, W.AI1_Vibration_std, c=W.y.map({0: "steelblue", 1: "crimson"}), s=10)
ax[1, 1].set_xlabel("AI0 std"); ax[1, 1].set_ylabel("AI1 std"); ax[1, 1].set_title("진폭 평면: 두 클래스 완전 분리")
ax[1, 2].scatter(d.AI2_Current, d.grid_resid, c=d.Equipment_state.map({0: "steelblue", 1: "crimson"}), s=1)
ax[1, 2].set_title("ADC 격자 잔차: outlier만 0 (취득경로 누출)")
fig.tight_layout(); fig.savefig(fig_path("04_press_diagnostics")); plt.close(fig)
print("done")

# ---- 7. false-alarm forensics: one-class model trained on NORMAL only
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EmpiricalCovariance
FVp = [c for c in FV if not c.startswith("AI2_Current_")]      # drop the leaky channel
Xn = W.loc[W.y == 0, FVp].replace([np.inf, -np.inf], np.nan).fillna(0)
Xa = W[FVp].replace([np.inf, -np.inf], np.nan).fillna(0)
iso = IsolationForest(n_estimators=400, random_state=SEED).fit(Xn)
W["iso_score"] = -iso.score_samples(Xa)
mu, sd = Xn.mean(), Xn.std().replace(0, 1)
cov = EmpiricalCovariance().fit(((Xn - mu) / sd).values)
W["maha"] = cov.mahalanobis(((Xa - mu) / sd).values)
fa = []
for c in ["iso_score", "maha"]:
    thr = W.loc[W.y == 0, c].quantile(.99)      # 1% false-alarm budget on normal
    flag = W[c] >= thr
    fa.append(dict(score=c, threshold_at_1pct_FA_on_normal=float(thr),
                   recall_on_outlier=float(flag[W.y == 1].mean()),
                   false_alarm_rate_on_normal=float(flag[W.y == 0].mean()),
                   n_false_alarm_windows=int((flag & (W.y == 0)).sum()),
                   auc=float(roc_auc_score(W.y, W[c]))))
save_table(pd.DataFrame(fa), "04_press_false_alarm_budget")
FAW = W[(W.y == 0) & (W.maha >= W.loc[W.y == 0, "maha"].quantile(.99))]
save_table(FAW[["seg", "t0", "maha", "iso_score", "AI0_Vibration_std", "AI1_Vibration_std",
                "corr_AI0_AI1", "AI2_Current_acf1"]], "04_press_false_alarm_windows")

# ---- 8. does subsequence-shape mining (matrix-profile style discord) add anything over the features?
def discord(x, m=20, stride=2):
    idx = np.arange(0, len(x) - m + 1, stride)
    S = np.lib.stride_tricks.sliding_window_view(x, m)[idx]
    S = (S - S.mean(1, keepdims=True)) / (S.std(1, keepdims=True) + 1e-9)
    nn = np.full(len(S), np.inf)
    for a in range(0, len(S), 512):
        blk = S[a:a + 512]
        D = np.sqrt(np.maximum(((blk ** 2).sum(1)[:, None] + (S ** 2).sum(1)[None, :] - 2 * blk @ S.T), 0))
        for r in range(len(blk)):
            g = a + r
            lo, hi = max(0, g - m // stride), min(len(S), g + m // stride + 1)
            D[r, lo:hi] = np.inf
            nn[g] = D[r].min()
    return idx, nn

mp_rows = []
for c in CH:
    for nm, x in [("normal", n[c].values), ("outlier", o[c].values)]:
        i, nn = discord(x)
        mp_rows.append(dict(channel=c, file=nm, n_subseq=len(nn), mp_median=float(np.median(nn)),
                            mp_p95=float(np.quantile(nn, .95)), mp_max=float(nn.max())))
mp = pd.DataFrame(mp_rows)
save_table(mp, "04_press_matrix_profile_summary")
piv = mp.pivot(index="channel", columns="file", values="mp_median")
piv["ratio_outlier_over_normal"] = piv["outlier"] / piv["normal"]
save_table(piv.reset_index(), "04_press_matrix_profile_ratio", index=False)
print("appendix done")
