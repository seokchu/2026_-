"""36. 고사용량 경보 최종 구성 — 생산 적용본.

35-A~D 결과로 확정한 구성:
  특성  V2 = CORE 43개 + 임계거리 margin 12개 + 피크이력 3개
  모델  RandomForest(400, min_samples_leaf=2). 클래스 가중 미사용(정밀도 악화).
  보정  폴드별 CAL isotonic
  운영점 R3 — 전 폴드 CAL 을 통합해 F1 최대가 되는 확률 임계 1개를 정하고 전 구간에 적용
비교군: 이전 구성(CORE 43특성 RF), HistGBM, persistence 규칙, 회귀 마진, 사전확률.
4개 horizon 전부 산출한다.
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
from sklearn.ensemble import (RandomForestClassifier, HistGradientBoostingClassifier,  # noqa: E402
                              HistGradientBoostingRegressor)
from sklearn.isotonic import IsotonicRegression                           # noqa: E402
from sklearn.model_selection import TimeSeriesSplit                       # noqa: E402
from sklearn.metrics import (matthews_corrcoef, average_precision_score,  # noqa: E402
                             roc_auc_score, brier_score_loss)

cfg = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
NS, CALF = cfg["data"]["n_splits"], cfg["data"]["cal_fraction"]
HSET = {1: 15, 2: 30, 3: 45, 4: 60}
t0 = time.time()


def build_alert_frame():
    d, _ = FT.build(use_external=False)
    d = FT.common_rows(d, "CORE").reset_index(drop=True)
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
    l = np.full(len(d), np.nan); c = np.nan
    for i, v in enumerate(d.peak_now.values):
        l[i] = c; c = 0.0 if v == 1 else (c + 1 if np.isfinite(c) else np.nan)
    d["steps_since_peak"] = np.clip(l, 0, 400)
    d["day_i"] = d.ts15.dt.normalize()
    d["today_max_margin"] = d.groupby("day_i").kw.cummax() - d.thr_adaptive
    return d


MCORE = ["margin", "margin_ratio", "margin_lag1", "margin_lag2", "margin_lag4", "margin_lag8",
         "margin_lag96", "margin_ramp4", "margin_roll4_max", "margin_roll96_max",
         "margin_roll96_mean", "today_max_margin"]
MHIST = ["peak_cnt_96", "peak_cnt_672", "steps_since_peak"]


def conf(y, pr):
    tp = int((pr & y).sum()); fp = int((pr & ~y).sum())
    fn = int((~pr & y).sum()); tn = int((~pr & ~y).sum())
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    return tp, fp, fn, tn, p, r, 2 * p * r / max(p + r, 1e-9)


def event_recall(lab, pred, tol=4):
    lab = np.asarray(lab).astype(bool); pred = np.asarray(pred).astype(bool)
    g = (lab.astype(int) != np.r_[0, lab.astype(int)[:-1]]).cumsum()
    ev = [np.where((g == k) & lab)[0] for k in np.unique(g[lab])]
    if not ev: return np.nan, 0
    hit = sum(1 for i in ev if pred[max(i[0] - tol, 0):i[-1] + 1].any())
    return hit / len(ev), len(ev)


if __name__ == "__main__":
    d = build_alert_frame()
    BASE = list(FT.CORE)
    V2 = BASE + MCORE + MHIST
    FOLDS = list(TimeSeriesSplit(n_splits=NS).split(d))
    TGRID = np.unique(np.round(np.linspace(0.01, 0.95, 190), 4))
    parts = []
    for h, hm in HSET.items():
        for fi, (itr, ite) in enumerate(FOLDS):
            cut = int(len(itr) * (1 - CALF))
            TR, CAs, TE = d.iloc[itr[:cut]], d.iloc[itr[cut:]], d.iloc[ite]
            ytr = (TR[f"y_h{h}"].values >= TR.thr_adaptive.values)
            yca = (CAs[f"y_h{h}"].values >= CAs.thr_adaptive.values)
            yte = (TE[f"y_h{h}"].values >= TE.thr_adaptive.values)
            cands = {}
            for nm, F, mk in [
                ("제안 (margin 특성 + RF)", V2,
                 lambda: RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
                                                random_state=SEED, n_jobs=-1)),
                ("이전 구성 (기본특성 + RF)", BASE,
                 lambda: RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
                                                random_state=SEED, n_jobs=-1)),
                ("HistGBM (기본특성)", BASE,
                 lambda: HistGradientBoostingClassifier(random_state=SEED))]:
                m = mk().fit(TR[F].fillna(-1).values, ytr)
                pc = m.predict_proba(CAs[F].fillna(-1).values)[:, 1]
                pt = m.predict_proba(TE[F].fillna(-1).values)[:, 1]
                iso = IsotonicRegression(out_of_bounds="clip").fit(pc, yca)
                cands[nm] = (iso.predict(pc), iso.predict(pt))
            rg = HistGradientBoostingRegressor(random_state=SEED).fit(
                TR[BASE].fillna(-1).values, TR[f"y_h{h}"].values - TR.kw.values)
            mc = rg.predict(CAs[BASE].fillna(-1).values) + CAs.kw.values - CAs.thr_adaptive.values
            mt = rg.predict(TE[BASE].fillna(-1).values) + TE.kw.values - TE.thr_adaptive.values
            rk = lambda a, b: (pd.Series(a).rank(pct=True).values, pd.Series(b).rank(pct=True).values)
            cands["회귀 마진 (베이스라인)"] = rk(mc, mt)
            cands["persistence 규칙 (베이스라인)"] = rk(
                CAs.kw.values - CAs.thr_adaptive.values, TE.kw.values - TE.thr_adaptive.values)
            cands["사전확률 (베이스라인)"] = (np.full(len(yca), yca.mean()),
                                        np.full(len(yte), ytr.mean()))
            for nm, (pc, pt) in cands.items():
                parts.append(pd.DataFrame(dict(horizon_min=hm, fold=fi, model=nm, part="CAL",
                                               score=pc, label=yca, clean=CAs.clean.values,
                                               ts15=CAs.ts15.values)))
                parts.append(pd.DataFrame(dict(horizon_min=hm, fold=fi, model=nm, part="TEST",
                                               score=pt, label=yte, clean=TE.clean.values,
                                               ts15=TE.ts15.values)))
        print(f"  h{hm} ({time.time()-t0:.0f}s)")
    D = pd.concat(parts, ignore_index=True)
    rows, store = [], []
    for (hm, nm), g in D.groupby(["horizon_min", "model"]):
        cal = g[g.part == "CAL"]; te = g[g.part == "TEST"].sort_values(["fold", "ts15"])
        yc = cal.label.values.astype(bool)
        t = max(((conf(yc, cal.score.values >= q)[6], q) for q in TGRID))[1]
        pr = te.score.values >= t
        te = te.assign(alert=pr)
        store.append(te.assign(model=nm, horizon_min=hm))
        for wtag, m in [("all_2021", np.ones(len(te), bool)), ("clean_Jul_Sep", te.clean.values)]:
            y = te.label.values.astype(bool)[m]; a = pr[m]; sc = te.score.values[m]
            tp, fp, fn, tn, p, r, f1 = conf(y, a)
            er, nev = event_recall(y, a)
            rows.append(dict(window=wtag, horizon_min=hm, model=nm, n=int(m.sum()),
                             prevalence=float(y.mean()), precision=p, recall=r, f1=f1,
                             mcc=float(matthews_corrcoef(y, a)) if a.any() and (~a).any() else 0.0,
                             pr_auc=float(average_precision_score(y, sc)),
                             roc_auc=float(roc_auc_score(y, sc)) if sc.std() > 0 else np.nan,
                             brier=float(brier_score_loss(y, np.clip(sc, 0, 1))),
                             balanced_accuracy=(r + tn / max(tn + fp, 1)) / 2,
                             specificity=tn / max(tn + fp, 1),
                             accuracy=(tp + tn) / max(len(y), 1),
                             event_recall=er, n_events=nev,
                             alerts_per_day=float(a.mean() * 96), prob_cut=t,
                             tp=tp, fp=fp, fn=fn, tn=tn))
    R = pd.DataFrame(rows); save_table(R, "36_alert_final_metrics")
    pd.concat(store, ignore_index=True).to_csv(
        ROOT / "analysis/tables/36_alert_final_scores.csv", index=False, encoding="utf-8-sig")
    c = R[R.window == "clean_Jul_Sep"]
    print(c.pivot_table(index="model", columns="horizon_min",
                        values=["f1", "precision", "alerts_per_day"]).round(3).to_string())
    print("\n4 horizon 평균 (clean)")
    print(c.groupby("model")[["precision", "recall", "f1", "mcc", "pr_auc", "event_recall",
                              "alerts_per_day"]].mean().round(4)
          .sort_values("f1", ascending=False).to_string())
    print(f"[done] {time.time()-t0:.0f}s")
