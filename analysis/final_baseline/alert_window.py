"""38. 경보 라벨 정의 교정 — '1시간 안에 발생하는가'.

종전 라벨 y(t+60)>=임계 는 '정확히 그 15분 슬롯'을 요구해 15분만 어긋나도 오답이 된다.
시스템 출력("앞으로 한 시간 안에 고사용량")과 평가 라벨이 달랐다. 세 정의를 같은 조건에서 비교한다.
"""
import sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd, yaml
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "analysis" / "07_sparse_fems"))
from common import save_table, SEED                                       # noqa: E402
import features as FT                                                     # noqa: E402
from alert_final import build_alert_frame, MCORE, MHIST, conf, event_recall  # noqa: E402
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,  # noqa: E402
                              HistGradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression                       # noqa: E402
from sklearn.pipeline import make_pipeline                                # noqa: E402
from sklearn.preprocessing import StandardScaler                          # noqa: E402
from sklearn.isotonic import IsotonicRegression                           # noqa: E402
from sklearn.model_selection import TimeSeriesSplit                       # noqa: E402
from sklearn.metrics import (matthews_corrcoef, average_precision_score,  # noqa: E402
                             roc_auc_score, brier_score_loss)
cfg = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
NS, CALF = cfg["data"]["n_splits"], cfg["data"]["cal_fraction"]
TG = np.unique(np.round(np.linspace(0.01, 0.95, 190), 4)); t0 = time.time()
d = build_alert_frame(); FULL = list(FT.CORE) + MCORE + MHIST
thr = d.thr_adaptive.values
d["L_step60"] = (d.y_h4.values >= thr)
d["L_win30"] = (d.y_h1.values >= thr) | (d.y_h2.values >= thr)
d["L_win60"] = d.L_win30 | (d.y_h3.values >= thr) | (d.y_h4.values >= thr)
NAME = {"L_step60": "정확히 60분 뒤 그 슬롯 (종전)", "L_win30": "앞으로 30분 안에 발생",
        "L_win60": "앞으로 1시간 안에 발생 (운영 정의)"}
MODELS = {"Random Forest": lambda: RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
                                                          random_state=SEED, n_jobs=-1),
          "Extra Trees": lambda: ExtraTreesClassifier(n_estimators=400, min_samples_leaf=2,
                                                      random_state=SEED, n_jobs=-1),
          "HistGradientBoosting": lambda: HistGradientBoostingClassifier(random_state=SEED),
          "Logistic Regression": lambda: make_pipeline(
              StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED))}
FOLDS = list(TimeSeriesSplit(n_splits=NS).split(d))


def rec(wtag, lab, mname, y, pr, sc):
    tp, fp, fn, tn, p, r, f1 = conf(y, pr)
    er, nev = event_recall(y, pr)
    return dict(window=wtag, label=lab, label_desc=NAME[lab], model=mname, n=len(y),
                prevalence=float(y.mean()), accuracy=(tp + tn) / max(len(y), 1),
                precision=p, recall=r, specificity=tn / max(tn + fp, 1),
                balanced_accuracy=(r + tn / max(tn + fp, 1)) / 2, f1=f1,
                mcc=float(matthews_corrcoef(y, pr)) if pr.any() and (~pr).any() else 0.0,
                roc_auc=float(roc_auc_score(y, sc)) if np.std(sc) > 0 else np.nan,
                pr_auc=float(average_precision_score(y, sc)),
                brier=float(brier_score_loss(y, np.clip(sc, 0, 1))),
                event_recall=er, n_events=nev, alerts_per_day=float(pr.mean() * 96),
                tp=tp, fp=fp, fn=fn, tn=tn)


rows, keep = [], []
for lab in NAME:
    cands = {k: ([], [], [], [], [], []) for k in list(MODELS) + ["persistence 규칙", "사전확률"]}
    for itr, ite in FOLDS:
        cut = int(len(itr) * (1 - CALF))
        TR, CA, TE = d.iloc[itr[:cut]], d.iloc[itr[cut:]], d.iloc[ite]
        for mname, mk in MODELS.items():
            m = mk().fit(TR[FULL].fillna(-1).values, TR[lab].values)
            pc = m.predict_proba(CA[FULL].fillna(-1).values)[:, 1]
            pt = m.predict_proba(TE[FULL].fillna(-1).values)[:, 1]
            iso = IsotonicRegression(out_of_bounds="clip").fit(pc, CA[lab].values)
            pc, pt = iso.predict(pc), iso.predict(pt)
            for a, b in zip(cands[mname], (pc, CA[lab].values, pt, TE[lab].values,
                                           TE.clean.values, TE.ts15.values)):
                a.append(b)
        a_ = CA.kw.values - CA.thr_adaptive.values; b_ = TE.kw.values - TE.thr_adaptive.values
        lo, hi = a_.min(), a_.max()
        for a, b in zip(cands["persistence 규칙"],
                        ((a_ - lo) / (hi - lo + 1e-9), CA[lab].values,
                         np.clip((b_ - lo) / (hi - lo + 1e-9), 0, 1), TE[lab].values,
                         TE.clean.values, TE.ts15.values)):
            a.append(b)
        pv = float(CA[lab].mean())
        for a, b in zip(cands["사전확률"], (np.full(len(CA), pv), CA[lab].values,
                                        np.full(len(TE), pv), TE[lab].values,
                                        TE.clean.values, TE.ts15.values)):
            a.append(b)
    for mname, v in cands.items():
        cs, cy, ts, ty, tc, tt = [np.concatenate(x) for x in v]
        t = max(((conf(cy, cs >= q)[6], q) for q in TG))[1]
        pr = ts >= t
        keep.append(pd.DataFrame(dict(label=lab, model=mname, score=ts, alert=pr, y=ty,
                                      clean=tc, ts15=tt)))
        for wtag, m_ in [("all_2021", np.ones(len(ts), bool)), ("clean_Jul_Sep", tc)]:
            rows.append(rec(wtag, lab, mname, ty[m_], pr[m_], ts[m_]))
    print(f"  {lab} ({time.time()-t0:.0f}s)")
R = pd.DataFrame(rows); save_table(R, "38_window_label_metrics")
K = pd.concat(keep, ignore_index=True)
K[K.clean].to_csv(ROOT / "analysis/tables/38_window_label_scores.csv", index=False,
                  encoding="utf-8-sig")
c = R[R.window == "clean_Jul_Sep"]
for lab in NAME:
    print(f"\n[{NAME[lab]}] 양성 {c[c.label==lab].prevalence.iloc[0]:.4f}")
    print(c[c.label == lab].sort_values("f1", ascending=False)[
        ["model", "precision", "recall", "f1", "mcc", "pr_auc", "balanced_accuracy",
         "alerts_per_day"]].round(3).to_string(index=False))
print(f"[done] {time.time()-t0:.0f}s")
