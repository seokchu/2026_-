"""33. 모델 선정 — 베이스라인 포함 다중 모델 비교 (회귀 + 고사용량 경보 분류).

대회 요건: "베이스라인을 포함하여 2개 이상의 모델을 비교하고, 동일한 평가조건에서
어떤 차이가 나타났는지와 최종모델을 선택한 이유를 제시한다."

두 과제를 분리해 각각 비교한다.
  TASK-R 수치 예측   : t+15/30/45/60분 kw. 지표 MAE / 고사용량 구간 MAE / RMSE
  TASK-C 고사용량 경보: y(t+h) >= thr(t) 이진 판정. 지표 F1 / PR-AUC / precision / recall / 사건 recall

진단도 함께 낸다(D1~D3): F1 이 낮은 원인이 '본질적 한계'인지 '모델 편향'인지.

평가 규약은 v1/v2 와 동일. 임계·보정·스태커는 TRAIN(+CAL)에서만 적합. TEST 는 적용만.
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd, yaml
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "analysis" / "07_sparse_fems"))
from common import save_table, SEED                                      # noqa: E402
import features as FT                                                    # noqa: E402
import models as MD                                                      # noqa: E402
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,   # noqa: E402
                              HistGradientBoostingRegressor,
                              RandomForestClassifier, ExtraTreesClassifier,
                              HistGradientBoostingClassifier)
from sklearn.linear_model import Ridge, LogisticRegression                # noqa: E402
from sklearn.preprocessing import StandardScaler                          # noqa: E402
from sklearn.pipeline import make_pipeline                                # noqa: E402
from sklearn.isotonic import IsotonicRegression                           # noqa: E402
from sklearn.model_selection import TimeSeriesSplit                       # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss  # noqa: E402

cfg = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
NS, CALF = cfg["data"]["n_splits"], cfg["data"]["cal_fraction"]
HOR = {1: 15, 2: 30, 3: 45, 4: 60}
t0 = time.time()
d, mode = FT.build(use_external=False)
d = FT.common_rows(d, mode)
F = FT.CORE
FOLDS = list(TimeSeriesSplit(n_splits=NS).split(d))
print(f"[data] rows={len(d)} feats={len(F)} folds={NS}")


def splits():
    for fi, (itr, ite) in enumerate(FOLDS):
        cut = int(len(itr) * (1 - CALF))
        yield fi, d.iloc[itr[:cut]], d.iloc[itr[cut:]], d.iloc[ite]


# ─────────────────────────────────────────── TASK-R : 회귀 모델 비교
REG = {
    "R0_persistence":   None,                                   # 베이스라인(직전값)
    "R1_ridge":         lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    "R2_random_forest": lambda: RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                                      random_state=SEED, n_jobs=-1),
    "R3_extra_trees":   lambda: ExtraTreesRegressor(n_estimators=300, min_samples_leaf=2,
                                                    random_state=SEED, n_jobs=-1),
    "R4_hist_gbm":      lambda: HistGradientBoostingRegressor(random_state=SEED),
    "R5_hist_gbm_deep": lambda: HistGradientBoostingRegressor(random_state=SEED, max_iter=400,
                                                              learning_rate=0.06,
                                                              max_leaf_nodes=63,
                                                              l2_regularization=1.0),
}
rrows, rpred = [], []
for h, hm in HOR.items():
    y = f"y_h{h}"
    for fi, TR, CA, TE in splits():
        FIT = pd.concat([TR, CA]); yte = TE[y].values
        thr = TE.thr_adaptive.values; pk = yte >= thr
        w = MD.relevance_weight(FIT[y].values)
        for name, mk in REG.items():
            if mk is None:
                p = TE.kw.values
            else:                                   # v2 레시피(잔차타깃+극단가중) 공통 적용
                m = mk()
                try:
                    m.fit(FIT[F].fillna(-1).values, FIT[y].values - FIT.kw.values,
                          **({"sample_weight": w} if "ridge" not in name else {}))
                except TypeError:
                    m.fit(FIT[F].fillna(-1).values, FIT[y].values - FIT.kw.values)
                p = m.predict(TE[F].fillna(-1).values) + TE.kw.values
            e = np.abs(yte - p)
            rrows.append(dict(task="R", horizon_min=hm, fold=fi, model=name, n=len(yte),
                              mae=float(e.mean()), rmse=float(np.sqrt((e**2).mean())),
                              mae_peak=float(e[pk].mean()) if pk.any() else np.nan,
                              bias_peak=float((yte - p)[pk].mean()) if pk.any() else np.nan,
                              err_p95=float(np.quantile(e, .95))))
            if hm == 60:
                rpred.append(pd.DataFrame(dict(ts15=TE.ts15.values, fold=fi, model=name,
                                               y=yte, pred=p, thr=thr,
                                               clean=TE.clean.values)))
    print(f"  [R] h{hm} done ({time.time()-t0:.0f}s)")
R = pd.DataFrame(rrows); save_table(R, "33_model_compare_regression_rows")
RA = (R.groupby(["horizon_min", "model"])
        .agg(mae=("mae", "mean"), rmse=("rmse", "mean"), mae_peak=("mae_peak", "mean"),
             bias_peak=("bias_peak", "mean"), err_p95=("err_p95", "mean")).reset_index())
save_table(RA, "33_model_compare_regression")
print(RA[RA.horizon_min == 60].round(3).to_string(index=False))


# ─────────────────────────────────────── TASK-C : 경보 분류 모델 비교
def ece(y, p, bins=10):
    e, n = 0.0, len(y)
    for i in range(bins):
        m = (p >= i / bins) & (p < (i + 1) / bins if i < bins - 1 else p <= 1.0)
        if m.sum():
            e += m.sum() / n * abs(y[m].mean() - p[m].mean())
    return float(e)


def event_recall(ts, lab, pred, tol_steps=4):
    """연속 피크 구간을 1사건으로 묶고, 사건 내 또는 tol 이내에 경보가 있으면 포착."""
    lab = np.asarray(lab); pred = np.asarray(pred)
    grp = (lab.astype(int) != np.r_[0, lab.astype(int)[:-1]]).cumsum()
    ev = [np.where((grp == g) & lab)[0] for g in np.unique(grp[lab])]
    if not ev:
        return np.nan, 0
    hit = 0
    for idx in ev:
        a, b = max(idx[0] - tol_steps, 0), min(idx[-1] + 1, len(pred))
        if pred[a:b].any():
            hit += 1
    return hit / len(ev), len(ev)


def pick_threshold(pc, yc):
    """CAL 에서 F1 최대가 되는 점수 컷. TEST 를 쓰지 않는다."""
    cands = np.unique(np.quantile(pc, np.linspace(0.50, 0.999, 200)))
    best, bt = -1, cands[-1]
    for t in cands:
        pr = pc >= t
        tp = (pr & yc).sum(); fp = (pr & ~yc).sum(); fn = (~pr & yc).sum()
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        if f1 > best:
            best, bt = f1, t
    return bt


H = 4; HM = 60
crows, cpred, drows = [], [], []
for fi, TR, CA, TE in splits():
    y = f"y_h{H}"
    ytr = (TR[y].values >= TR.thr_adaptive.values)
    yca = (CA[y].values >= CA.thr_adaptive.values)
    yte = (TE[y].values >= TE.thr_adaptive.values)
    Xtr, Xca, Xte = (z[F].fillna(-1).values for z in (TR, CA, TE))
    FIT = pd.concat([TR, CA]); Xfit = FIT[F].fillna(-1).values
    yfit = (FIT[y].values >= FIT.thr_adaptive.values)

    # --- 보조 점수원: 회귀 마진 / 분위 마진 (TRAIN 에서만 적합)
    wreg = MD.relevance_weight(TR[y].values)
    reg = HistGradientBoostingRegressor(random_state=SEED).fit(
        Xtr, TR[y].values - TR.kw.values, sample_weight=wreg)
    def rmargin(Z, base, thr):
        return reg.predict(Z) + base - thr
    q90 = HistGradientBoostingRegressor(loss="quantile", quantile=0.90,
                                        random_state=SEED).fit(Xtr, TR[y].values - TR.kw.values)
    q75 = HistGradientBoostingRegressor(loss="quantile", quantile=0.75,
                                        random_state=SEED).fit(Xtr, TR[y].values - TR.kw.values)
    def qmargin(m, Z, base, thr):
        return m.predict(Z) + base - thr

    scores = {}
    # C0 베이스라인: 직전값이 임계를 넘는가 (규칙)
    scores["C0_persistence_rule"] = (CA.kw.values - CA.thr_adaptive.values,
                                     TE.kw.values - TE.thr_adaptive.values)
    # C1 베이스라인2: 평균 회귀 마진 (v2 점예측을 그대로 경보에 쓰는 경우)
    scores["C1_reg_margin"] = (rmargin(Xca, CA.kw.values, CA.thr_adaptive.values),
                               rmargin(Xte, TE.kw.values, TE.thr_adaptive.values))
    # C2 분위회귀 q0.90 초과마진 — 평균회귀의 피크 과소예측을 직접 겨냥
    scores["C2_quantile90_margin"] = (qmargin(q90, Xca, CA.kw.values, CA.thr_adaptive.values),
                                      qmargin(q90, Xte, TE.kw.values, TE.thr_adaptive.values))
    # C3~C6 직접 분류기 (TRAIN 적합 -> CAL 보정/임계 -> TEST 적용)
    CLF = {
        "C3_logistic":     make_pipeline(StandardScaler(),
                                         LogisticRegression(max_iter=2000, class_weight="balanced",
                                                            random_state=SEED)),
        "C4_random_forest": RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
                                                   class_weight="balanced_subsample",
                                                   random_state=SEED, n_jobs=-1),
        "C5_extra_trees":   ExtraTreesClassifier(n_estimators=400, min_samples_leaf=2,
                                                 class_weight="balanced",
                                                 random_state=SEED, n_jobs=-1),
        "C6_hist_gbm":      HistGradientBoostingClassifier(random_state=SEED),
    }
    for nm, m in CLF.items():
        m.fit(Xtr, ytr)
        scores[nm] = (m.predict_proba(Xca)[:, 1], m.predict_proba(Xte)[:, 1])
    # C7 스태킹: 기본특성 + 회귀마진 + 분위마진 + 구간정보 -> HGB 분류기
    #    기저모델은 TRAIN 에서, 스태커는 CAL 에서 적합, 임계도 CAL. TEST 미사용.
    def stack_X(Z, base, thr):
        return np.c_[Z,
                     rmargin(Z, base, thr),
                     qmargin(q90, Z, base, thr),
                     qmargin(q75, Z, base, thr),
                     base - thr]
    Sca, Ste = stack_X(Xca, CA.kw.values, CA.thr_adaptive.values), \
               stack_X(Xte, TE.kw.values, TE.thr_adaptive.values)
    Str = stack_X(Xtr, TR.kw.values, TR.thr_adaptive.values)
    stk = HistGradientBoostingClassifier(random_state=SEED).fit(Str, ytr)
    scores["C7_stacked"] = (stk.predict_proba(Sca)[:, 1], stk.predict_proba(Ste)[:, 1])
    # C8 스태킹 + CAL isotonic 보정
    iso = IsotonicRegression(out_of_bounds="clip").fit(scores["C7_stacked"][0], yca)
    scores["C8_stacked_isotonic"] = (iso.predict(scores["C7_stacked"][0]),
                                     iso.predict(scores["C7_stacked"][1]))

    for nm, (sc, st) in scores.items():
        cut = pick_threshold(sc, yca)
        pr = st >= cut
        tp = int((pr & yte).sum()); fp = int((pr & ~yte).sum())
        fn = int((~pr & yte).sum()); tn = int((~pr & ~yte).sum())
        prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
        er, nev = event_recall(TE.ts15.values, yte, pr)
        isprob = st.min() >= 0 and st.max() <= 1
        crows.append(dict(task="C", horizon_min=HM, fold=fi, model=nm, n=len(yte),
                          prevalence=float(yte.mean()),
                          f1=2 * prec * rec / max(prec + rec, 1e-9),
                          precision=prec, recall=rec,
                          pr_auc=float(average_precision_score(yte, st)),
                          roc_auc=float(roc_auc_score(yte, st)),
                          brier=float(brier_score_loss(yte, st)) if isprob else np.nan,
                          ece=ece(yte, st) if isprob else np.nan,
                          event_recall=er, n_events=nev,
                          alerts_per_day=float(pr.mean() * 96),
                          tp=tp, fp=fp, fn=fn, tn=tn))
        cpred.append(pd.DataFrame(dict(ts15=TE.ts15.values, fold=fi, model=nm,
                                       score=st, alert=pr, label=yte,
                                       clean=TE.clean.values)))
    print(f"  [C] fold{fi} done ({time.time()-t0:.0f}s)")

C = pd.DataFrame(crows); save_table(C, "33_model_compare_alert_rows")
CA_ = (C.groupby("model").agg(f1=("f1", "mean"), precision=("precision", "mean"),
                              recall=("recall", "mean"), pr_auc=("pr_auc", "mean"),
                              roc_auc=("roc_auc", "mean"), brier=("brier", "mean"),
                              ece=("ece", "mean"), event_recall=("event_recall", "mean"),
                              alerts_per_day=("alerts_per_day", "mean"),
                              tp=("tp", "sum"), fp=("fp", "sum"), fn=("fn", "sum"))
         .reset_index().sort_values("f1", ascending=False))
save_table(CA_, "33_model_compare_alert")
print(CA_.round(4).to_string(index=False))

CP = pd.concat(cpred, ignore_index=True)
cl = CP[CP.clean]
cc = []
for nm, g in cl.groupby("model"):
    tp = int((g.alert & g.label).sum()); fp = int((g.alert & ~g.label).sum())
    fn = int((~g.alert & g.label).sum()); tn = int((~g.alert & ~g.label).sum())
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    er, nev = event_recall(g.ts15.values, g.label.values, g.alert.values)
    cc.append(dict(window="clean_Jul_Sep", model=nm, n=len(g),
                   prevalence=float(g.label.mean()),
                   f1=2 * prec * rec / max(prec + rec, 1e-9), precision=prec, recall=rec,
                   pr_auc=float(average_precision_score(g.label, g.score)),
                   event_recall=er, n_events=nev,
                   alerts_per_day=float(g.alert.mean() * 96),
                   tp=tp, fp=fp, fn=fn, tn=tn))
CC = pd.DataFrame(cc).sort_values("f1", ascending=False)
save_table(CC, "33_model_compare_alert_clean")
print("\n[clean_Jul_Sep]\n" + CC.round(4).to_string(index=False))


# ─────────────────────────────────────── 진단 D1~D3 : F1 이 낮은 원인
rp = pd.concat(rpred, ignore_index=True)
rp = rp[rp.clean]
dg = []
rng = np.random.default_rng(SEED)
for nm, g in rp.groupby("model"):
    y, p, thr = g.y.values, g.pred.values, g.thr.values
    e = y - p; lab = y >= thr
    # D1 피크 구간 편향
    # D2 잔차분포 주입 오라클 F1 (완벽한 예측 + 같은 크기의 무편향 오차)
    f1o = []
    for _ in range(10):
        yn = y + rng.permutation(e)
        best = max(2 * ((yn >= np.quantile(yn, q)) & lab).sum() /
                   max(2 * ((yn >= np.quantile(yn, q)) & lab).sum() +
                       ((yn >= np.quantile(yn, q)) & ~lab).sum() +
                       ((yn < np.quantile(yn, q)) & lab).sum(), 1)
                   for q in np.linspace(0.80, 0.995, 60))
        f1o.append(best)
    dg.append(dict(model=nm, window="clean_Jul_Sep", n=len(g), mae=float(np.abs(e).mean()),
                   bias_all=float(e.mean()), bias_peak=float(e[lab].mean()),
                   peak_pred_above_thr=float((p[lab] >= thr[lab]).mean()),
                   peak_within_mae_of_thr=float((np.abs(y - thr)[lab] <=
                                                 np.abs(e).mean()).mean()),
                   oracle_f1_same_error=float(np.mean(f1o))))
D = pd.DataFrame(dg).sort_values("bias_peak")
save_table(D, "33_f1_ceiling_diagnosis")
print("\n[진단] 왜 F1 이 낮은가\n" + D.round(4).to_string(index=False))
print(f"\n[done] {time.time()-t0:.0f}s")
