"""07-F. 신뢰도 메타모델 — 타깃 관측 이전에 '이번 예측이 어려운가'를 맞힐 수 있는가.

타깃 3종: abs_err(회귀) / high_err(상위20% 분류) / interval_fail(CQR 90% 구간 실패)
예측자: 결정시점 문맥 + 모델 불일치 + OOD 거리 + CQR 구간폭 + 잠재상태
비교: Ridge·Logistic / depth-3 tree / HGB  (SR 은 07E 결과로 별도 판정)
운영 평가: HIGH/MEDIUM/LOW/OOD 밴드별 실측 결과가 단조인가.
산출 07F_row_scores.csv 는 07G 가 재사용한다.
"""
import numpy as np, pandas as pd
import _fe
from _fe import build, save_table, fig_path, mpl, SEED, TAB
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, mean_absolute_error, brier_score_loss
from scipy.stats import spearmanr
plt = mpl()

H = 4
LAT = ["lat_state", "lat_dwell", "lat_p_high"]
o = pd.read_csv(TAB / "07C_oof_predictions.csv", parse_dates=["ts15"], encoding="utf-8-sig")
q = build(latent_cols=LAT)
F3 = _fe.LEVELS["L3_+production"]
o = o.merge(q[["ts15"] + [c for c in F3 if c not in o.columns]], on="ts15", how="left")

# ---- 폴드별 CQR 구간 (보정은 각 폴드 학습구간의 뒤쪽 20% 로만)
def qr(a):
    return HistGradientBoostingRegressor(loss="quantile", quantile=a, random_state=SEED)
o["int_lo"] = np.nan; o["int_hi"] = np.nan
for f in sorted(o.fold.unique()):
    tr, te = o[o.fold < f], o[o.fold == f]
    if len(tr) < 500:
        continue
    cut = int(len(tr) * .8)
    a, c = tr.iloc[:cut], tr.iloc[cut:]
    m_lo, m_hi = qr(.05).fit(a[F3].values, a.y_true.values), qr(.95).fit(a[F3].values, a.y_true.values)
    E = np.maximum(m_lo.predict(c[F3].values) - c.y_true.values,
                   c.y_true.values - m_hi.predict(c[F3].values))
    Q = float(np.quantile(E, min(1.0, .9 * (1 + 1 / len(E)))))
    o.loc[te.index, "int_lo"] = m_lo.predict(te[F3].values) - Q
    o.loc[te.index, "int_hi"] = m_hi.predict(te[F3].values) + Q
o = o.dropna(subset=["int_lo"]).reset_index(drop=True)
o["int_width"] = o.int_hi - o.int_lo
o["interval_fail"] = ((o.y_true < o.int_lo) | (o.y_true > o.int_hi)).astype(int)

PRED = ["kw", "kw_ramp4", "abs_ramp", "kw_std4", "kw_std96", "생산량", "prod_diff", "is_operating",
        "시간", "tod", "day", "기온", "습도", "disagreement", "ood_maha", "int_width"] + LAT
folds = sorted(o.fold.unique())
tr, te = o[o.fold < folds[-1]], o[o.fold == folds[-1]]
Xtr, Xte = tr[PRED].values, te[PRED].values
hi_thr = float(np.quantile(tr.abs_err, .8))
ytr_hi, yte_hi = (tr.abs_err >= hi_thr).astype(int), (te.abs_err >= hi_thr).astype(int)

rows, err_pred = [], {}
REG = {"ridge": lambda: make_pipeline(StandardScaler(), Ridge(1.0)),
       "tree_depth3": lambda: DecisionTreeRegressor(max_depth=3, random_state=SEED),
       "hgb": lambda: HistGradientBoostingRegressor(random_state=SEED)}
for n, mk in REG.items():
    p = mk().fit(Xtr, tr.abs_err.values).predict(Xte)
    err_pred[n] = p
    rows.append(dict(target="abs_err", model=n, mae=float(mean_absolute_error(te.abs_err, p)),
                     rmse=float(np.sqrt(((te.abs_err - p) ** 2).mean())),
                     spearman=float(spearmanr(p, te.abs_err).statistic),
                     auc=float(roc_auc_score(yte_hi, p)), ap=float(average_precision_score(yte_hi, p)),
                     brier=np.nan, n_test=len(te)))
CLF = {"logistic": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
       "tree_depth3": lambda: DecisionTreeClassifier(max_depth=3, random_state=SEED),
       "hgb": lambda: HistGradientBoostingClassifier(random_state=SEED)}
clf_prob = {}
for tgt, ytr_c, yte_c in [("high_err", ytr_hi, yte_hi),
                          ("interval_fail", tr.interval_fail.values, te.interval_fail.values)]:
    for n, mk in CLF.items():
        if len(np.unique(ytr_c)) < 2:
            continue
        p = mk().fit(Xtr, ytr_c).predict_proba(Xte)[:, 1]
        clf_prob[(tgt, n)] = p
        rows.append(dict(target=tgt, model=n, mae=np.nan, rmse=np.nan,
                         spearman=float(spearmanr(p, te.abs_err).statistic),
                         auc=float(roc_auc_score(yte_c, p)) if len(np.unique(yte_c)) > 1 else np.nan,
                         ap=float(average_precision_score(yte_c, p)) if len(np.unique(yte_c)) > 1 else np.nan,
                         brier=float(brier_score_loss(yte_c, p)), n_test=len(te)))
# 단순 대리 기준선
for n, s in [("baseline_int_width", te.int_width.values), ("baseline_disagreement", te.disagreement.values),
             ("baseline_abs_ramp", te.abs_ramp.values), ("baseline_ood_maha", te.ood_maha.values)]:
    rows.append(dict(target="high_err", model=n, mae=np.nan, rmse=np.nan,
                     spearman=float(spearmanr(s, te.abs_err).statistic),
                     auc=float(roc_auc_score(yte_hi, s)), ap=float(average_precision_score(yte_hi, s)),
                     brier=np.nan, n_test=len(te)))
res = pd.DataFrame(rows)
save_table(res, "07F_reliability_meta_models")

# ---- 밴드: 난이도 순위력(spearman)이 가장 높은 신호로 만든다.
#      학습 메타모델이 단순 신호(CQR 구간폭)보다 못할 수 있으므로 후보에 함께 넣는다.
cand = {**err_pred, "int_width": te.int_width.values}
cand_tr = {**{n: REG[n]().fit(Xtr, tr.abs_err.values).predict(Xtr) for n in REG},
           "int_width": tr.int_width.values}
sp_all = {n: float(spearmanr(v, te.abs_err).statistic) for n, v in cand.items()}
best = max(sp_all, key=sp_all.get)
pe = cand[best]
cuts = np.quantile(cand_tr[best], [.5, .8])
ood_cut = float(np.quantile(tr.ood_maha, .99))
band = np.where(te.ood_maha.values > ood_cut, "OOD",
                np.where(pe <= cuts[0], "HIGH", np.where(pe <= cuts[1], "MEDIUM", "LOW")))
thr_peak = float(np.quantile(tr.y_true, .95))
alert = te.pred.values >= thr_peak
brows = []
for b in ["HIGH", "MEDIUM", "LOW", "OOD"]:
    m = band == b
    if m.sum() == 0:
        continue
    s = te[m]
    brows.append(dict(band=b, n=int(m.sum()), share=float(m.mean()),
                      mae=float(s.abs_err.mean()), err_p90=float(np.quantile(s.abs_err, .9)),
                      interval_coverage=float(1 - s.interval_fail.mean()),
                      mean_interval_width=float(s.int_width.mean()),
                      peak_prevalence=float(s.is_peak.mean()),
                      peak_miss_rate=float(((s.is_peak == 1) & (~alert[m])).sum() / max((s.is_peak == 1).sum(), 1)),
                      false_alarm_rate=float(((s.is_peak == 0) & (alert[m])).sum() / max((s.is_peak == 0).sum(), 1))))
bands = pd.DataFrame(brows)
order = [b for b in ["HIGH", "MEDIUM", "LOW"] if b in set(bands.band)]
mono = bands.set_index("band").loc[order].mae
save_table(bands, "07F_reliability_bands")
save_table(pd.DataFrame([dict(best_error_model=best, spearman_by_signal=str({k: round(v,3) for k,v in sp_all.items()}), monotonic_mae=bool(mono.is_monotonic_increasing),
                              mae_low_over_high=float(mono.iloc[-1] / mono.iloc[0]),
                              ood_cut_maha=ood_cut, high_err_threshold_kw=hi_thr,
                              band_cuts=str(list(np.round(cuts, 3))))]),
           "07F_reliability_band_summary")

out = te[["ts15", "fold", "clean", "y_true", "pred", "abs_err", "is_peak", "is_operating",
          "int_lo", "int_hi", "int_width", "interval_fail", "disagreement", "ood_maha"]].copy()
out["pred_abs_err"] = pe
out["band"] = band
out["p_high_err"] = clf_prob.get(("high_err", "hgb"), np.full(len(te), np.nan))
save_table(out, "07F_row_scores")
print(res.to_string(index=False)); print(bands.to_string(index=False))
