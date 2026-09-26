"""04b. Fair re-measurement after destroying the acquisition fingerprint.

Both files are re-quantized onto ONE common, coarse grid that is coarser than
either file's own LSB (normal current step 0.00011 / outlier 1.19209;
vibration 6 vs 8 decimal places). If the separation survives this, it is
physical; if it collapses, the earlier AUC was acquisition-path leakage.
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from scipy import signal, stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from common import load_press, segment, save_table, SEED
CH = ["AI0_Vibration", "AI1_Vibration", "AI2_Current"]
Q = 1.1920929
VIB_STEP, CUR_STEP = 1e-3, 2.0     # coarser than every per-file LSB

n, o = load_press()
n["seg"] = "N" + segment(n).astype(str); o["seg"] = "O" + segment(o).astype(str)
d = pd.concat([n, o], ignore_index=True)
for c in CH[:2]:
    d[c] = np.round(d[c] / VIB_STEP) * VIB_STEP
d["AI2_Current"] = np.round(d.AI2_Current / CUR_STEP) * CUR_STEP

# fingerprint must now be gone
chk = []
for nm, g in d.groupby(d.seg.str[0]):
    r = g.AI2_Current.values / Q
    chk.append(dict(file="normal" if nm == "N" else "outlier",
                    grid_dev_mean=float(np.abs(r - np.round(r)).mean()),
                    current_min_step=float(np.diff(np.sort(g.AI2_Current.unique())).min()),
                    vib_min_step=float(np.diff(np.sort(g.AI0_Vibration.unique())).min())))
save_table(pd.DataFrame(chk), "04b_press_fingerprint_after_requantization")

def feats(g):
    out = {}
    for c in CH:
        x = g[c].values; ax = np.abs(x); rms = np.sqrt((x ** 2).mean())
        out |= {f"{c}_std": x.std(), f"{c}_rms": rms, f"{c}_p2p": x.max() - x.min(),
                f"{c}_peak": ax.max(), f"{c}_kurt": stats.kurtosis(x), f"{c}_skew": stats.skew(x),
                f"{c}_crest": ax.max() / rms if rms else 0.0,
                f"{c}_zcr": float(np.mean(np.diff(np.sign(x)) != 0)),
                f"{c}_acf1": float(np.corrcoef(x[:-1], x[1:])[0, 1])}
        f, P = signal.welch(x, fs=10, nperseg=min(len(x), 32))
        Pn = P / P.sum()
        out |= {f"{c}_centroid": float((f * Pn).sum()),
                f"{c}_spec_entropy": float(-(Pn * np.log(Pn + 1e-12)).sum()),
                f"{c}_bandhigh": float(P[f > 3].sum() / P.sum())}
    out["corr_AI0_AI1"] = float(np.corrcoef(g.AI0_Vibration, g.AI1_Vibration)[0, 1])
    out["n"] = len(g); out["y"] = int(g.Equipment_state.iloc[0])
    return out

W = pd.DataFrame([{"seg": k, **feats(g)} for k, g in d.groupby("seg", sort=False)])
W = W[W.n >= 16].reset_index(drop=True)
FV = [c for c in W.columns if c not in ("seg", "y", "n")]
save_table(W, "04b_press_window_features_requantized")

rows = []
sets = {"all_physical": FV,
        "vib_only": [c for c in FV if c.startswith(("AI0", "AI1"))],
        "current_only": [c for c in FV if c.startswith("AI2")],
        "amplitude_only": [c for c in FV if c.endswith(("_std", "_rms", "_p2p", "_peak"))],
        "spectral_only": [c for c in FV if c.endswith(("_centroid", "_spec_entropy", "_bandhigh"))]}
for tag, cols in sets.items():
    X = W[cols].replace([np.inf, -np.inf], np.nan).fillna(0).values
    y, g = W.y.values, W.seg.values
    for nm, mk in [("logreg", make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))),
                   ("rf", RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
                                                 class_weight="balanced_subsample", random_state=SEED, n_jobs=-1))]:
        pr = cross_val_predict(mk, X, y, cv=StratifiedGroupKFold(5, shuffle=True, random_state=SEED),
                               groups=g, method="predict_proba")[:, 1]
        rows.append(dict(feature_set=tag, model=nm, n_features=len(cols),
                         auc=roc_auc_score(y, pr), ap=average_precision_score(y, pr)))
save_table(pd.DataFrame(rows), "04b_press_baseline_after_leak_removal")

cmp = []
for c in FV:
    a, b = W.loc[W.y == 1, c].dropna(), W.loc[W.y == 0, c].dropna()
    if len(a) < 3 or W[c].nunique() < 3: continue
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    cmp.append(dict(feature=c, auc=u / (len(a) * len(b)), mwu_p=p,
                    median_normal=float(b.median()), median_outlier=float(a.median())))
cmp = pd.DataFrame(cmp); cmp["auc_abs"] = (cmp.auc - .5).abs()
save_table(cmp.sort_values("auc_abs", ascending=False), "04b_press_feature_separation_after_leak_removal")
print("done")
