"""34. 표준 벤치마크 — 교과서/실무에서 통용되는 모델·지표로 먼저 비교한다.

취지: 우리만의 지표(무편향 오라클 F1, 사건 recall, OOD 패널티 등)를 앞세우지 않는다.
      널리 쓰이는 모델과 표준 지표로 먼저 위치를 보이고, 고유 분석은 그 뒤에 덧붙인다.

TASK-R 수치 예측 (h60) — 표준 지표: MAE / RMSE / MAPE / sMAPE / MASE / R²
TASK-C 이진 분류 (h60) — 표준 지표: Accuracy / Precision / Recall / F1 / Specificity /
                          Balanced Accuracy / MCC / ROC-AUC / PR-AUC / Brier / LogLoss

프로토콜은 기존과 동일(rolling-origin 5 fold, TRAIN/CAL/TEST, 적합은 TRAIN(+CAL)만).
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
from sklearn.linear_model import (LinearRegression, Ridge, Lasso, ElasticNet,  # noqa: E402
                                  LogisticRegression)
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier   # noqa: E402
from sklearn.svm import SVR, SVC                                          # noqa: E402
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier    # noqa: E402
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,  # noqa: E402
                              GradientBoostingRegressor, HistGradientBoostingRegressor,
                              RandomForestClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier, HistGradientBoostingClassifier)
from sklearn.naive_bayes import GaussianNB                                # noqa: E402
from sklearn.neural_network import MLPRegressor, MLPClassifier            # noqa: E402
from sklearn.pipeline import make_pipeline                                # noqa: E402
from sklearn.preprocessing import StandardScaler                          # noqa: E402
from sklearn.model_selection import TimeSeriesSplit                       # noqa: E402
from sklearn.metrics import (r2_score, matthews_corrcoef, roc_auc_score,  # noqa: E402
                             average_precision_score, brier_score_loss, log_loss)

cfg = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
NS, CALF = cfg["data"]["n_splits"], cfg["data"]["cal_fraction"]
H, HM = 4, 60
SUB_SVM = 5000                       # SVR/SVC 는 O(n^2) 이라 학습표본을 제한한다(명시)
t0 = time.time()
d, _ = FT.build(use_external=False); d = FT.common_rows(d, "CORE")
F = FT.CORE
FOLDS = list(TimeSeriesSplit(n_splits=NS).split(d))
print(f"[data] rows={len(d)} feats={len(F)}")


def sc(m):
    return make_pipeline(StandardScaler(), m)


REG = {
    "Naive (직전값)":            "persistence",
    "Seasonal naive (24h 전)":   "seasonal",
    "Linear Regression":         lambda: sc(LinearRegression()),
    "Ridge":                     lambda: sc(Ridge(alpha=1.0)),
    "Lasso":                     lambda: sc(Lasso(alpha=0.01, max_iter=5000)),
    "ElasticNet":                lambda: sc(ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000)),
    "k-NN (k=10)":               lambda: sc(KNeighborsRegressor(n_neighbors=10, n_jobs=-1)),
    "SVR (RBF)":                 lambda: sc(SVR(C=10.0, epsilon=1.0)),
    "Decision Tree":             lambda: DecisionTreeRegressor(min_samples_leaf=5, random_state=SEED),
    "Random Forest":             lambda: RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                                               random_state=SEED, n_jobs=-1),
    "Extra Trees":               lambda: ExtraTreesRegressor(n_estimators=300, min_samples_leaf=2,
                                                             random_state=SEED, n_jobs=-1),
    "Gradient Boosting":         lambda: GradientBoostingRegressor(random_state=SEED),
    "HistGradientBoosting":      lambda: HistGradientBoostingRegressor(random_state=SEED),
    "MLP (64,32)":               lambda: sc(MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=400,
                                                         early_stopping=True, random_state=SEED)),
    "HistGBM + 극단가중 (제출 모델)": lambda: HistGradientBoostingRegressor(random_state=SEED),
}
WEIGHTED = {"HistGBM + 극단가중 (제출 모델)"}      # 이 모델만 표본가중을 쓴다(나머지는 기본 적합)


def relevance_w(yf, lam=4.0):
    lo, hi = np.quantile(yf, [.90, .995])
    return 1.0 + lam * np.clip((yf - lo) / max(hi - lo, 1e-6), 0, 1)


rows, keep = [], {}
ycol = f"y_h{H}"
for fi, (itr, ite) in enumerate(FOLDS):
    FIT, TE = d.iloc[itr], d.iloc[ite]
    yte = TE[ycol].values
    Xf, Xt = FIT[F].fillna(-1).values, TE[F].fillna(-1).values
    base_f, base_t = FIT.kw.values, TE.kw.values
    w = relevance_w(FIT[ycol].values)
    # MASE 분모: 학습구간 seasonal naive(96스텝=24시간) 평균절대오차
    sn = np.abs(FIT.kw.values[96:] - FIT.kw.values[:-96]).mean()
    for nm, mk in REG.items():
        if mk == "persistence":
            p = base_t
        elif mk == "seasonal":
            p = TE.kw_lag96.fillna(TE.kw).values
        else:
            m = mk()
            if nm.startswith("SVR"):
                idx = np.linspace(0, len(Xf) - 1, min(SUB_SVM, len(Xf))).astype(int)
                m.fit(Xf[idx], FIT[ycol].values[idx] - base_f[idx])
            elif nm in WEIGHTED:
                m.fit(Xf, FIT[ycol].values - base_f, sample_weight=w)
            else:
                m.fit(Xf, FIT[ycol].values - base_f)
            p = m.predict(Xt) + base_t
        e = yte - p; ae = np.abs(e)
        nz = yte > 1e-6
        rows.append(dict(fold=fi, model=nm, n=len(yte),
                         mae=float(ae.mean()), rmse=float(np.sqrt((e ** 2).mean())),
                         mape=float((ae[nz] / yte[nz]).mean() * 100),
                         smape=float((2 * ae / (np.abs(yte) + np.abs(p) + 1e-9)).mean() * 100),
                         mase=float(ae.mean() / sn), r2=float(r2_score(yte, p)),
                         mae_peak=float(ae[yte >= TE.thr_adaptive.values].mean())))
        keep.setdefault(nm, []).append(pd.DataFrame(dict(ts15=TE.ts15.values, fold=fi,
                                                         y=yte, pred=p, clean=TE.clean.values,
                                                         thr=TE.thr_adaptive.values)))
    print(f"  [R] fold{fi} ({time.time()-t0:.0f}s)")
R = pd.DataFrame(rows); save_table(R, "34_standard_benchmark_regression_rows")
RA = R.groupby("model").agg(mae=("mae", "mean"), rmse=("rmse", "mean"), mape=("mape", "mean"),
                            smape=("smape", "mean"), mase=("mase", "mean"), r2=("r2", "mean"),
                            mae_peak=("mae_peak", "mean")).reset_index().sort_values("mae")
save_table(RA, "34_standard_benchmark_regression")
print(RA.round(3).to_string(index=False))

cl_rows = []
for nm, lst in keep.items():
    g = pd.concat(lst); g = g[g.clean]
    e = np.abs(g.y.values - g.pred.values); nz = g.y.values > 1e-6
    sn = np.abs(d[d.clean].kw.values[96:] - d[d.clean].kw.values[:-96]).mean()
    cl_rows.append(dict(window="clean_Jul_Sep", model=nm, n=len(g),
                        mae=float(e.mean()), rmse=float(np.sqrt(((g.y - g.pred) ** 2).mean())),
                        mape=float((e[nz] / g.y.values[nz]).mean() * 100),
                        smape=float((2 * e / (np.abs(g.y.values) + np.abs(g.pred.values) + 1e-9)).mean() * 100),
                        mase=float(e.mean() / sn), r2=float(r2_score(g.y, g.pred)),
                        mae_peak=float(e[g.y.values >= g.thr.values].mean())))
RC = pd.DataFrame(cl_rows).sort_values("mae")
save_table(RC, "34_standard_benchmark_regression_clean")
print("\n[clean]\n" + RC.round(3).to_string(index=False))


# ───────────────────────────── TASK-C 표준 분류 모델
CLF = {
    "Majority class (전부 정상)": "majority",
    "Naive rule (현재값>임계)":    "rule",
    "Logistic Regression":        lambda: sc(LogisticRegression(max_iter=3000, random_state=SEED)),
    "Gaussian Naive Bayes":       lambda: sc(GaussianNB()),
    "k-NN (k=10)":                lambda: sc(KNeighborsClassifier(n_neighbors=10, n_jobs=-1)),
    "SVM (RBF)":                  lambda: sc(SVC(C=10.0, probability=True, random_state=SEED)),
    "Decision Tree":              lambda: DecisionTreeClassifier(min_samples_leaf=5, random_state=SEED),
    "Random Forest":              lambda: RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
                                                                 random_state=SEED, n_jobs=-1),
    "Extra Trees":                lambda: ExtraTreesClassifier(n_estimators=400, min_samples_leaf=2,
                                                               random_state=SEED, n_jobs=-1),
    "Extra Trees + 클래스가중 (제출 모델)": lambda: ExtraTreesClassifier(
        n_estimators=400, min_samples_leaf=2, class_weight="balanced",
        random_state=SEED, n_jobs=-1),
    "Gradient Boosting":          lambda: GradientBoostingClassifier(random_state=SEED),
    "HistGradientBoosting":       lambda: HistGradientBoostingClassifier(random_state=SEED),
    "MLP (64,32)":                lambda: sc(MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=400,
                                                           early_stopping=True, random_state=SEED)),
}
crows, cstore = [], []
for fi, (itr, ite) in enumerate(FOLDS):
    cut = int(len(itr) * (1 - CALF))
    TR, CAs, TE = d.iloc[itr[:cut]], d.iloc[itr[cut:]], d.iloc[ite]
    ytr = (TR[ycol].values >= TR.thr_adaptive.values)
    yca = (CAs[ycol].values >= CAs.thr_adaptive.values)
    yte = (TE[ycol].values >= TE.thr_adaptive.values)
    Xtr, Xca, Xte = (z[F].fillna(-1).values for z in (TR, CAs, TE))
    for nm, mk in CLF.items():
        if mk == "majority":
            pc, pt = np.zeros(len(yca)), np.zeros(len(yte))
        elif mk == "rule":
            pc = (CAs.kw.values - CAs.thr_adaptive.values)
            pt = (TE.kw.values - TE.thr_adaptive.values)
            lo, hi = pc.min(), pc.max()
            pc, pt = (pc - lo) / (hi - lo + 1e-9), np.clip((pt - lo) / (hi - lo + 1e-9), 0, 1)
        else:
            m = mk()
            if nm.startswith("SVM"):
                idx = np.linspace(0, len(Xtr) - 1, min(SUB_SVM, len(Xtr))).astype(int)
                m.fit(Xtr[idx], ytr[idx])
            else:
                m.fit(Xtr, ytr)
            pc, pt = m.predict_proba(Xca)[:, 1], m.predict_proba(Xte)[:, 1]
        # 운영 임계는 CAL 에서 F1 최대 (TEST 미사용)
        cand = np.unique(np.quantile(pc, np.linspace(0.50, 0.9995, 200)))
        best, cut_t = -1, 1.1
        for t in cand:
            pr = pc >= t
            tp = (pr & yca).sum(); fp = (pr & ~yca).sum(); fn = (~pr & yca).sum()
            f1 = 2 * tp / max(2 * tp + fp + fn, 1)
            if f1 > best: best, cut_t = f1, t
        pr = pt >= cut_t
        tp = int((pr & yte).sum()); fp = int((pr & ~yte).sum())
        fn = int((~pr & yte).sum()); tn = int((~pr & ~yte).sum())
        prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        crows.append(dict(fold=fi, model=nm, n=len(yte), prevalence=float(yte.mean()),
                          accuracy=(tp + tn) / len(yte), precision=prec, recall=rec,
                          specificity=spec, balanced_accuracy=(rec + spec) / 2,
                          f1=2 * prec * rec / max(prec + rec, 1e-9),
                          mcc=float(matthews_corrcoef(yte, pr)) if pr.any() and (~pr).any() else 0.0,
                          roc_auc=float(roc_auc_score(yte, pt)) if pt.std() > 0 else np.nan,
                          pr_auc=float(average_precision_score(yte, pt)),
                          brier=float(brier_score_loss(yte, np.clip(pt, 0, 1))),
                          log_loss=float(log_loss(yte, np.clip(pt, 1e-6, 1 - 1e-6))),
                          tp=tp, fp=fp, fn=fn, tn=tn))
        cstore.append(pd.DataFrame(dict(ts15=TE.ts15.values, fold=fi, model=nm, score=pt,
                                        alert=pr, label=yte, clean=TE.clean.values)))
    print(f"  [C] fold{fi} ({time.time()-t0:.0f}s)")
C = pd.DataFrame(crows); save_table(C, "34_standard_benchmark_alert_rows")
CAg = C.groupby("model").agg(accuracy=("accuracy", "mean"), precision=("precision", "mean"),
                             recall=("recall", "mean"), specificity=("specificity", "mean"),
                             balanced_accuracy=("balanced_accuracy", "mean"), f1=("f1", "mean"),
                             mcc=("mcc", "mean"), roc_auc=("roc_auc", "mean"),
                             pr_auc=("pr_auc", "mean"), brier=("brier", "mean"),
                             log_loss=("log_loss", "mean"), tp=("tp", "sum"), fp=("fp", "sum"),
                             fn=("fn", "sum")).reset_index().sort_values("f1", ascending=False)
save_table(CAg, "34_standard_benchmark_alert")
print(CAg.round(4).to_string(index=False))

CS = pd.concat(cstore, ignore_index=True); cc = CS[CS.clean]
out = []
for nm, g in cc.groupby("model"):
    tp = int((g.alert & g.label).sum()); fp = int((g.alert & ~g.label).sum())
    fn = int((~g.alert & g.label).sum()); tn = int((~g.alert & ~g.label).sum())
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1)
    out.append(dict(window="clean_Jul_Sep", model=nm, n=len(g), prevalence=float(g.label.mean()),
                    accuracy=(tp + tn) / len(g), precision=prec, recall=rec, specificity=spec,
                    balanced_accuracy=(rec + spec) / 2,
                    f1=2 * prec * rec / max(prec + rec, 1e-9),
                    mcc=float(matthews_corrcoef(g.label, g.alert)),
                    roc_auc=float(roc_auc_score(g.label, g.score)) if g.score.std() > 0 else np.nan,
                    pr_auc=float(average_precision_score(g.label, g.score)),
                    brier=float(brier_score_loss(g.label, np.clip(g.score, 0, 1))),
                    tp=tp, fp=fp, fn=fn, tn=tn))
CCl = pd.DataFrame(out).sort_values("f1", ascending=False)
save_table(CCl, "34_standard_benchmark_alert_clean")
print("\n[clean]\n" + CCl.round(4).to_string(index=False))
print(f"[done] {time.time()-t0:.0f}s")
