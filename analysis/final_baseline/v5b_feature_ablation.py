"""35-B. 경보 특성집합 ablation — 어떤 특성이 clean 구간에서 실제로 돕는가.

35 에서 margin 특성이 폴드 평균 PR-AUC 는 올렸으나(0.48->0.54) 주 보고 구간(clean)에서는
오히려 F1 을 떨어뜨렸다. 원인 후보는 하계휴가(2021-07-31~08-09) 구간에서 값이 폭주하는
누적형 특성(steps_since_peak, peak_cnt_672, prod_today_cum)이다. 집합을 쪼개 확인한다.

지표는 clean 구간 pooled 기준(주 보고 구간). 임계는 CAL isotonic 확률에서만 고른다.
"""
import sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd, yaml
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "analysis" / "07_sparse_fems"))
from common import save_table, SEED                                       # noqa: E402
import features as FT                                                     # noqa: E402
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,  # noqa: E402
                              HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor)
from sklearn.isotonic import IsotonicRegression                           # noqa: E402
from sklearn.model_selection import TimeSeriesSplit                       # noqa: E402
from sklearn.metrics import matthews_corrcoef, average_precision_score, roc_auc_score  # noqa: E402

cfg = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
NS, CALF = cfg["data"]["n_splits"], cfg["data"]["cal_fraction"]
H, HM = 4, 60
t0 = time.time()
d, _ = FT.build(use_external=False)
d = FT.common_rows(d, "CORE").reset_index(drop=True)
BASE = list(FT.CORE)

d["margin"] = d.kw - d.thr_adaptive
d["margin_ratio"] = d.kw / d.thr_adaptive
for k in (1, 2, 4, 8, 96):
    d[f"margin_lag{k}"] = d.margin.shift(k)
d["margin_ramp4"] = d.margin - d.margin_lag4
d["margin_roll4_max"] = d.margin.rolling(4).max()
d["margin_roll96_max"] = d.margin.rolling(96).max()
d["margin_roll96_mean"] = d.margin.rolling(96).mean()
d["peak_now"] = (d.kw >= d.thr_adaptive).astype(float)
d["peak_cnt_96"] = d.peak_now.shift(1).rolling(96).sum()
d["peak_cnt_672"] = d.peak_now.shift(1).rolling(672).sum()
_l = np.full(len(d), np.nan); c = np.nan
for i, v in enumerate(d.peak_now.values):
    _l[i] = c; c = 0.0 if v == 1 else (c + 1 if np.isfinite(c) else np.nan)
d["steps_since_peak"] = np.clip(_l, 0, 400)          # 휴무 구간 폭주를 막는 상한
d["day_i"] = d.ts15.dt.normalize()
d["today_max_margin"] = d.groupby("day_i").kw.cummax() - d.thr_adaptive

MCORE = ["margin", "margin_ratio", "margin_lag1", "margin_lag2", "margin_lag4", "margin_lag8",
         "margin_lag96", "margin_ramp4", "margin_roll4_max", "margin_roll96_max",
         "margin_roll96_mean", "today_max_margin"]
MHIST = ["peak_cnt_96", "peak_cnt_672", "steps_since_peak"]
MFC = ["pred_margin", "q90_margin", "q75_margin"]
MTOD = ["tod_peak_rate"]
VARIANTS = {
    "V0 base(43)": BASE,
    "V1 +margin": BASE + MCORE,
    "V2 +margin+피크이력": BASE + MCORE + MHIST,
    "V3 +margin+예측마진": BASE + MCORE + MFC,
    "V4 +margin+시간대피크율": BASE + MCORE + MTOD,
    "V5 전부": BASE + MCORE + MHIST + MFC + MTOD,
}
MODELS = {
    "RandomForest": lambda: RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
                                                   random_state=SEED, n_jobs=-1),
    "ExtraTrees": lambda: ExtraTreesClassifier(n_estimators=400, min_samples_leaf=2,
                                               random_state=SEED, n_jobs=-1),
    "HistGBM": lambda: HistGradientBoostingClassifier(random_state=SEED),
}
FOLDS = list(TimeSeriesSplit(n_splits=NS).split(d))


def conf(y, pr):
    tp = int((pr & y).sum()); fp = int((pr & ~y).sum())
    fn = int((~pr & y).sum()); tn = int((~pr & ~y).sum())
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    return tp, fp, fn, tn, p, r, 2 * p * r / max(p + r, 1e-9)


def pick_cut(pc, yc):
    grid = np.unique(np.r_[np.linspace(0.02, 0.95, 160), np.quantile(pc, np.linspace(.7, .999, 60))])
    best = (-1, .5)
    for t in grid:
        f1 = conf(yc, pc >= t)[6]
        if f1 > best[0]: best = (f1, float(t))
    return best[1]


store = []
for fi, (itr, ite) in enumerate(FOLDS):
    cut = int(len(itr) * (1 - CALF))
    TR, CAs, TE = d.iloc[itr[:cut]].copy(), d.iloc[itr[cut:]].copy(), d.iloc[ite].copy()
    g = TR.groupby("tod").peak_now.agg(["sum", "count"]); p0 = float(TR.peak_now.mean())
    rate = ((g["sum"] + 50 * p0) / (g["count"] + 50)).to_dict()
    Xb = TR[BASE + MCORE].fillna(-1).values
    yr = TR[f"y_h{H}"].values - TR.kw.values
    reg = HistGradientBoostingRegressor(random_state=SEED).fit(Xb, yr)
    q90 = HistGradientBoostingRegressor(loss="quantile", quantile=.90, random_state=SEED).fit(Xb, yr)
    q75 = HistGradientBoostingRegressor(loss="quantile", quantile=.75, random_state=SEED).fit(Xb, yr)
    for z in (TR, CAs, TE):
        z["tod_peak_rate"] = z.tod.map(rate).fillna(p0)
        B = z[BASE + MCORE].fillna(-1).values
        base_m = z.kw.values - z.thr_adaptive.values
        z["pred_margin"] = reg.predict(B) + base_m
        z["q90_margin"] = q90.predict(B) + base_m
        z["q75_margin"] = q75.predict(B) + base_m
    ytr = (TR[f"y_h{H}"].values >= TR.thr_adaptive.values)
    yca = (CAs[f"y_h{H}"].values >= CAs.thr_adaptive.values)
    yte = (TE[f"y_h{H}"].values >= TE.thr_adaptive.values)
    for vname, F in VARIANTS.items():
        Xtr, Xca, Xte = (z[F].fillna(-1).values for z in (TR, CAs, TE))
        raw = {}
        for mname, mk in MODELS.items():
            m = mk().fit(Xtr, ytr)
            raw[mname] = (m.predict_proba(Xca)[:, 1], m.predict_proba(Xte)[:, 1])
        raw["SoftVote"] = (np.mean([raw[k][0] for k in MODELS], 0),
                           np.mean([raw[k][1] for k in MODELS], 0))
        for mname, (pc, pt) in raw.items():
            iso = IsotonicRegression(out_of_bounds="clip").fit(pc, yca)
            pcc, ptc = iso.predict(pc), iso.predict(pt)
            t = pick_cut(pcc, yca)
            store.append(pd.DataFrame(dict(ts15=TE.ts15.values, fold=fi, variant=vname,
                                           model=mname, score=ptc, alert=ptc >= t,
                                           label=yte, clean=TE.clean.values)))
    print(f"  fold{fi} ({time.time()-t0:.0f}s)")

S = pd.concat(store, ignore_index=True)
out = []
for wtag, g0 in [("all_2021", S), ("clean_Jul_Sep", S[S.clean])]:
    for (v, m), g in g0.groupby(["variant", "model"]):
        y = g.label.values.astype(bool); a = g.alert.values.astype(bool)
        tp, fp, fn, tn, p, r, f1 = conf(y, a)
        sc = g.score.values
        orc = max(conf(y, sc >= np.quantile(sc, q))[6] for q in np.linspace(.80, .999, 80))
        out.append(dict(window=wtag, variant=v, model=m, n=len(g), prevalence=float(y.mean()),
                        precision=p, recall=r, f1=f1,
                        mcc=float(matthews_corrcoef(y, a)),
                        pr_auc=float(average_precision_score(y, sc)),
                        roc_auc=float(roc_auc_score(y, sc)),
                        f1_best_threshold=float(orc),
                        alerts_per_day=float(a.mean() * 96), tp=tp, fp=fp, fn=fn, tn=tn))
R = pd.DataFrame(out)
save_table(R, "35_feature_ablation")
S.to_csv(ROOT / "analysis/tables/35_feature_ablation_scores.csv", index=False, encoding="utf-8-sig")
c = R[R.window == "clean_Jul_Sep"].sort_values("f1", ascending=False)
print(c[["variant", "model", "precision", "recall", "f1", "mcc", "pr_auc",
         "f1_best_threshold", "alerts_per_day"]].round(4).to_string(index=False))
print(f"[done] {time.time()-t0:.0f}s")
