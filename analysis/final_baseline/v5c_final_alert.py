"""35-C. 경보 최종 구성 — 운영점을 '확률값' 이 아니라 '경보 예산(비율)' 으로 고른다.

35-B 에서 확인된 것:
  - margin 계열 특성은 순위능력(PR-AUC)과 정밀도를 올린다.
  - 그런데 CAL 에서 고른 '확률 임계값' 이 TEST 로 전이되지 않아 F1 이 깎인다.
    (폴드마다 양성 비율이 0.031~0.091 로 3배 차이난다)
해결: CAL 에서 최적 '경보 발령 비율' 을 고르고, TEST 에는 같은 비율을 적용한다.
      TEST 라벨은 쓰지 않는다. 산업 경보 운영에서 쓰는 alarm budget 방식이다.
후처리로 delay-timer(n-out-of-m) 도 CAL 에서 함께 고른다.
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
FEAT_V2 = BASE + MCORE + MHIST
SETS = {"V0_base": BASE, "V2_margin_hist": FEAT_V2}
MODELS = {"RandomForest": lambda: RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
                                                         random_state=SEED, n_jobs=-1),
          "ExtraTrees": lambda: ExtraTreesClassifier(n_estimators=400, min_samples_leaf=2,
                                                     random_state=SEED, n_jobs=-1),
          "HistGBM": lambda: HistGradientBoostingClassifier(random_state=SEED)}
FOLDS = list(TimeSeriesSplit(n_splits=NS).split(d))
RATES = np.round(np.arange(0.02, 0.22, 0.005), 4)
POST = [("none", 1, 1), ("2of2", 2, 2), ("2of3", 2, 3), ("3of3", 3, 3), ("2of4", 2, 4),
        ("3of4", 3, 4)]


def conf(y, pr):
    tp = int((pr & y).sum()); fp = int((pr & ~y).sum())
    fn = int((~pr & y).sum()); tn = int((~pr & ~y).sum())
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    return tp, fp, fn, tn, p, r, 2 * p * r / max(p + r, 1e-9)


def k_of_m(x, k, m):
    if m == 1: return x.astype(bool)
    return (pd.Series(x.astype(float)).rolling(m, min_periods=1).sum().values >= k)


store, sel = [], []
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
                best = None
                for rate in RATES:
                    tc = np.quantile(pc, 1 - rate)
                    for pname, k, mm in POST:
                        pr = k_of_m(pc >= tc, k, mm)
                        f1 = conf(yca, pr)[6]
                        if best is None or f1 > best[0]:
                            best = (f1, rate, pname, k, mm)
                _, rate, pname, k, mm = best
                tt = np.quantile(pt, 1 - rate)           # 같은 '비율' 을 TEST 점수분포에 적용
                pr = k_of_m(pt >= tt, k, mm)
                tp, fp, fn, tn, p, r, f1 = conf(yte, pr)
                sel.append(dict(horizon_min=hm, fold=fi, feature_set=sname, model=mname,
                                alert_rate=rate, post=pname, delay_steps=mm - 1,
                                residual_lead_min=hm - (mm - 1) * 15, cal_f1=best[0],
                                test_f1=f1, precision=p, recall=r))
                store.append(pd.DataFrame(dict(ts15=TE.ts15.values, fold=fi, horizon_min=hm,
                                               feature_set=sname, model=mname, score=pt,
                                               alert=pr, label=yte, clean=TE.clean.values)))
    print(f"  h{hm} ({time.time()-t0:.0f}s)")
SEL = pd.DataFrame(sel); save_table(SEL, "35_final_alert_selection")
S = pd.concat(store, ignore_index=True)
out = []
for wtag, g0 in [("all_2021", S), ("clean_Jul_Sep", S[S.clean])]:
    for (hm, fs, mn), g in g0.groupby(["horizon_min", "feature_set", "model"]):
        y = g.label.values.astype(bool); a = g.alert.values.astype(bool)
        tp, fp, fn, tn, p, r, f1 = conf(y, a)
        sub = SEL[(SEL.horizon_min == hm) & (SEL.feature_set == fs) & (SEL.model == mn)]
        out.append(dict(window=wtag, horizon_min=hm, feature_set=fs, model=mn, n=len(g),
                        prevalence=float(y.mean()), precision=p, recall=r, f1=f1,
                        mcc=float(matthews_corrcoef(y, a)),
                        pr_auc=float(average_precision_score(y, g.score)),
                        roc_auc=float(roc_auc_score(y, g.score)),
                        alerts_per_day=float(a.mean() * 96),
                        mean_residual_lead_min=float(sub.residual_lead_min.mean()),
                        tp=tp, fp=fp, fn=fn, tn=tn))
R = pd.DataFrame(out); save_table(R, "35_final_alert_metrics")
S.to_csv(ROOT / "analysis/tables/35_final_alert_scores.csv", index=False, encoding="utf-8-sig")
c = R[(R.window == "clean_Jul_Sep")].sort_values(["horizon_min", "f1"], ascending=[True, False])
print(c[["horizon_min", "feature_set", "model", "precision", "recall", "f1", "mcc",
         "pr_auc", "alerts_per_day"]].round(4).to_string(index=False))
print(f"[done] {time.time()-t0:.0f}s")
