"""35-D. 운영점 선택 규칙 비교 — 같은 모델에서 임계 선택 방식만 바꿔 본다.

35-B/C 에서 드러난 문제: margin 특성은 순위능력(PR-AUC)을 확실히 올리는데,
운영점(임계) 선택 방식에 따라 최종 F1 이 0.39~0.54 로 요동친다.
선택 규칙 자체를 비교 대상에 올린다. 전부 CAL 라벨만 사용한다.

  R1 폴드별 확률 임계      — CAL F1 최대 확률값을 그 폴드 TEST 에 적용
  R2 폴드별 경보 비율      — CAL F1 최대 '발령 비율' 을 TEST 점수분포 분위로 적용
  R3 전체 CAL 통합 확률 임계 — 모든 폴드 CAL 을 합쳐 임계 1개를 정해 전 폴드에 적용
  R4 전체 CAL 통합 경보 비율
진단용으로 R0(TEST 사후 최적)도 함께 기록한다. 선택에는 쓰지 않는다.
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
                              HistGradientBoostingClassifier)
from sklearn.isotonic import IsotonicRegression                           # noqa: E402
from sklearn.model_selection import TimeSeriesSplit                       # noqa: E402
from sklearn.metrics import matthews_corrcoef, average_precision_score, roc_auc_score  # noqa: E402

cfg = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
NS, CALF = cfg["data"]["n_splits"], cfg["data"]["cal_fraction"]
HSET = {1: 15, 2: 30, 3: 45, 4: 60}
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
d["steps_since_peak"] = np.clip(_l, 0, 400)
d["day_i"] = d.ts15.dt.normalize()
d["today_max_margin"] = d.groupby("day_i").kw.cummax() - d.thr_adaptive
MCORE = ["margin", "margin_ratio", "margin_lag1", "margin_lag2", "margin_lag4", "margin_lag8",
         "margin_lag96", "margin_ramp4", "margin_roll4_max", "margin_roll96_max",
         "margin_roll96_mean", "today_max_margin"]
MHIST = ["peak_cnt_96", "peak_cnt_672", "steps_since_peak"]
SETS = {"V0_base": BASE, "V2_margin_hist": BASE + MCORE + MHIST}
MODELS = {"RandomForest": lambda: RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
                                                         random_state=SEED, n_jobs=-1),
          "ExtraTrees": lambda: ExtraTreesClassifier(n_estimators=400, min_samples_leaf=2,
                                                     random_state=SEED, n_jobs=-1),
          "HistGBM": lambda: HistGradientBoostingClassifier(random_state=SEED)}
FOLDS = list(TimeSeriesSplit(n_splits=NS).split(d))

rows = []
for h, hm in HSET.items():
    for fi, (itr, ite) in enumerate(FOLDS):
        cut = int(len(itr) * (1 - CALF))
        TR, CAs, TE = d.iloc[itr[:cut]], d.iloc[itr[cut:]], d.iloc[ite]
        ytr = (TR[f"y_h{h}"].values >= TR.thr_adaptive.values)
        yca = (CAs[f"y_h{h}"].values >= CAs.thr_adaptive.values)
        yte = (TE[f"y_h{h}"].values >= TE.thr_adaptive.values)
        for sname, F in SETS.items():
            Xtr, Xca, Xte = (z[F].fillna(-1).values for z in (TR, CAs, TE))
            raw = {}
            for mname, mk in MODELS.items():
                m = mk().fit(Xtr, ytr)
                raw[mname] = (m.predict_proba(Xca)[:, 1], m.predict_proba(Xte)[:, 1])
            raw["SoftVote"] = (np.mean([raw[k][0] for k in MODELS], 0),
                               np.mean([raw[k][1] for k in MODELS], 0))
            for mname, (pc, pt) in raw.items():
                iso = IsotonicRegression(out_of_bounds="clip").fit(pc, yca)
                rows.append(pd.DataFrame(dict(horizon_min=hm, fold=fi, feature_set=sname,
                                              model=mname, part="CAL",
                                              score=iso.predict(pc), label=yca,
                                              clean=CAs.clean.values, ts15=CAs.ts15.values)))
                rows.append(pd.DataFrame(dict(horizon_min=hm, fold=fi, feature_set=sname,
                                              model=mname, part="TEST",
                                              score=iso.predict(pt), label=yte,
                                              clean=TE.clean.values, ts15=TE.ts15.values)))
    print(f"  h{hm} fits ({time.time()-t0:.0f}s)")
D = pd.concat(rows, ignore_index=True)
D.to_csv(ROOT / "analysis/tables/35_opselect_scores.csv", index=False, encoding="utf-8-sig")


def conf(y, pr):
    tp = int((pr & y).sum()); fp = int((pr & ~y).sum())
    fn = int((~pr & y).sum()); tn = int((~pr & ~y).sum())
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    return tp, fp, fn, tn, p, r, 2 * p * r / max(p + r, 1e-9)


TGRID = np.unique(np.round(np.linspace(0.01, 0.95, 190), 4))
RGRID = np.round(np.arange(0.02, 0.25, 0.005), 4)


def best_t(sc, y):
    return max(((conf(y, sc >= t)[6], t) for t in TGRID))[1]


def best_r(sc, y):
    return max(((conf(y, sc >= np.quantile(sc, 1 - r))[6], r) for r in RGRID))[1]


out = []
for (hm, fs, mn), g in D.groupby(["horizon_min", "feature_set", "model"]):
    cal, te = g[g.part == "CAL"], g[g.part == "TEST"]
    gt = best_t(cal.score.values, cal.label.values.astype(bool))
    gr = best_r(cal.score.values, cal.label.values.astype(bool))
    preds = {k: np.zeros(len(te), bool) for k in ("R1", "R2", "R3", "R4")}
    te = te.reset_index(drop=True)
    for fi in sorted(te.fold.unique()):
        mte = (te.fold == fi).values
        c = cal[cal.fold == fi]
        t1 = best_t(c.score.values, c.label.values.astype(bool))
        r1 = best_r(c.score.values, c.label.values.astype(bool))
        s = te.score.values[mte]
        preds["R1"][mte] = s >= t1
        preds["R2"][mte] = s >= np.quantile(s, 1 - r1)
        preds["R3"][mte] = s >= gt
        preds["R4"][mte] = s >= np.quantile(s, 1 - gr)
    for wtag, m in [("all_2021", np.ones(len(te), bool)), ("clean_Jul_Sep", te.clean.values)]:
        y = te.label.values.astype(bool)[m]
        sc = te.score.values[m]
        orc = max(conf(y, sc >= np.quantile(sc, 1 - r))[6] for r in RGRID)
        for rule, pr in preds.items():
            tp, fp, fn, tn, p, r_, f1 = conf(y, pr[m])
            out.append(dict(window=wtag, horizon_min=hm, feature_set=fs, model=mn, rule=rule,
                            precision=p, recall=r_, f1=f1,
                            mcc=float(matthews_corrcoef(y, pr[m])) if pr[m].any() else 0.0,
                            pr_auc=float(average_precision_score(y, sc)),
                            roc_auc=float(roc_auc_score(y, sc)),
                            f1_oracle=float(orc),
                            alerts_per_day=float(pr[m].mean() * 96),
                            tp=tp, fp=fp, fn=fn, tn=tn, n=int(m.sum())))
R = pd.DataFrame(out); save_table(R, "35_opselect_metrics")
c = R[(R.window == "clean_Jul_Sep") & (R.horizon_min == 60)].sort_values("f1", ascending=False)
print(c[["feature_set", "model", "rule", "precision", "recall", "f1", "mcc", "pr_auc",
         "f1_oracle", "alerts_per_day"]].round(4).head(16).to_string(index=False))
print("\n규칙별 clean 평균 F1")
print(R[R.window == "clean_Jul_Sep"].groupby(["rule", "feature_set"]).f1.mean().round(4).to_string())
print(f"[done] {time.time()-t0:.0f}s")
