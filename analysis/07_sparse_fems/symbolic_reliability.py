"""07-E. Symbolic Regression 의 실용 역할 3가지 검증 (주 예측기로는 쓰지 않는다).

ROLE1 잔차/난이도 식        : 타깃 = OOF 절대오차
ROLE2 고오차 위험 신뢰도 대리 : 예측 절대오차를 신뢰도 밴드로 변환
ROLE3 모델 불일치 신호       : SR 예측기와 HGB 예측기의 차이가 실제 실패를 예측하는가
비교군: Ridge / depth-3 tree / HGB 메타모델. SR 은 스스로 자리를 벌어야 한다.
"""
import numpy as np, pandas as pd
from _fe import save_table, fig_path, mpl, SEED, TAB
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score, mean_absolute_error
from scipy.stats import spearmanr
from gplearn.genetic import SymbolicRegressor
plt = mpl()

o = pd.read_csv(TAB / "07C_oof_predictions.csv", parse_dates=["ts15"], encoding="utf-8-sig")
CTX = ["kw", "kw_ramp4", "abs_ramp", "kw_std4", "kw_std96", "생산량", "prod_diff",
       "is_operating", "시간", "기온", "습도", "ood_maha"]
tr, te = o[o.fold <= 3].reset_index(drop=True), o[o.fold == 4].reset_index(drop=True)
Xtr, Xte = tr[CTX].values, te[CTX].values
etr, ete = tr.abs_err.values, te.abs_err.values
hi_thr = float(np.quantile(etr, .8))                  # 고오차 기준은 train 에서만
hi_te = (ete >= hi_thr).astype(int)

SR_KW = dict(population_size=600, generations=12, tournament_size=20, parsimony_coefficient=0.01,
             function_set=("add", "sub", "mul", "div", "sqrt", "abs", "min", "max"),
             max_samples=0.6, random_state=SEED, n_jobs=-1, verbose=0)

rows, score_store = [], {}
def ev(name, pred):
    rows.append(dict(role="ROLE1_error_equation", model=name,
                     mae_of_error_pred=float(mean_absolute_error(ete, pred)),
                     rmse_of_error_pred=float(np.sqrt(((ete - pred) ** 2).mean())),
                     spearman_vs_true_error=float(spearmanr(pred, ete).statistic),
                     auc_high_error=float(roc_auc_score(hi_te, pred)),
                     n_test=len(ete)))
    score_store[name] = pred

sr1 = SymbolicRegressor(**SR_KW).fit(Xtr, etr)
ev("SR_gplearn", sr1.predict(Xte))
ev("ridge", make_pipeline(StandardScaler(), Ridge(1.0)).fit(Xtr, etr).predict(Xte))
ev("tree_depth3", DecisionTreeRegressor(max_depth=3, random_state=SEED).fit(Xtr, etr).predict(Xte))
ev("hgb", HistGradientBoostingRegressor(random_state=SEED).fit(Xtr, etr).predict(Xte))
ev("baseline_abs_ramp", te.abs_ramp.values)            # 단순 대리 신호
ev("baseline_kw_std4", te.kw_std4.values)

# ---- ROLE3: SR 예측기 vs HGB 예측기 불일치
sr_f = SymbolicRegressor(**SR_KW).fit(tr[CTX].values, tr.y_true.values)
sr_pred_te = sr_f.predict(Xte)
dis_sr = np.abs(sr_pred_te - te.pred.values)
for nm, sc in [("disagreement_SR_vs_HGB", dis_sr),
               ("disagreement_ridge_vs_HGB", te.disagreement.values)]:
    rows.append(dict(role="ROLE3_disagreement", model=nm, mae_of_error_pred=np.nan,
                     rmse_of_error_pred=np.nan,
                     spearman_vs_true_error=float(spearmanr(sc, ete).statistic),
                     auc_high_error=float(roc_auc_score(hi_te, sc)), n_test=len(ete)))
    rows.append(dict(role="ROLE3_disagreement_peakmiss", model=nm, mae_of_error_pred=np.nan,
                     rmse_of_error_pred=np.nan,
                     spearman_vs_true_error=np.nan,
                     auc_high_error=float(roc_auc_score(te.is_peak.values, sc))
                     if te.is_peak.nunique() > 1 else np.nan, n_test=len(ete)))
res = pd.DataFrame(rows)
save_table(res, "07E_sr_reliability_comparison")

# ---- 식 안정성: 폴드별 재적합 후 식/성능 변동
stab = []
for f in (2, 3, 4):
    t2 = o[o.fold < f]
    s = SymbolicRegressor(**SR_KW).fit(t2[CTX].values, t2.abs_err.values)
    stab.append(dict(train_folds=f"<{f}", program=str(s._program), length=s._program.length_,
                     spearman_on_fold4=float(spearmanr(s.predict(Xte), ete).statistic)))
stab.append(dict(train_folds="<=3(main)", program=str(sr1._program), length=sr1._program.length_,
                 spearman_on_fold4=float(spearmanr(score_store["SR_gplearn"], ete).statistic)))
save_table(pd.DataFrame(stab), "07E_sr_equation_stability")

# ---- 채택 게이트
best_other = res[(res.role == "ROLE1_error_equation") & (res.model != "SR_gplearn")].auc_high_error.max()
sr_auc = float(res[(res.model == "SR_gplearn")].auc_high_error.iloc[0])
dis_auc = float(res[res.model == "disagreement_SR_vs_HGB"].auc_high_error.iloc[0])
sp = pd.DataFrame(stab).spearman_on_fold4
# 무료 대안(CQR 구간폭)을 이기지 못하면 SR 은 자리를 벌지 못한다 → 07F 표를 기준선으로 쓴다.
free_auc = np.nan
fp = TAB / "07F_reliability_meta_models.csv"
if fp.exists():
    f = pd.read_csv(fp, encoding="utf-8-sig")
    m = f[f.model == "baseline_int_width"]
    free_auc = float(m.auc.iloc[0]) if len(m) else np.nan
gate = dict(sr_auc_high_error=sr_auc, best_non_sr_auc=float(best_other),
            free_baseline_int_width_auc=free_auc,
            sr_within_0_02_of_best=bool(sr_auc >= best_other - 0.02),
            disagreement_auc_high_error=dis_auc,
            disagreement_useful=bool(dis_auc >= 0.60),
            disagreement_beats_free_baseline=bool(dis_auc > (free_auc if free_auc == free_auc else 0.60)),
            equation_stable_across_folds=bool(sp.min() > 0.30 and sp.std() < 0.10),
            n_distinct_programs=int(pd.DataFrame(stab).program.nunique()))
stable = gate["equation_stable_across_folds"]
beats_free = gate["disagreement_beats_free_baseline"]
if stable and (gate["sr_within_0_02_of_best"] or beats_free):
    gate["decision"] = "ADOPT"          # 무료 대안까지 이기고 식도 안정
elif stable and gate["disagreement_useful"]:
    gate["decision"] = "OPTIONAL"       # 쓸모는 있으나 무료 대안이 같거나 더 낫다
else:
    gate["decision"] = "REJECT"
save_table(pd.DataFrame([gate]), "07E_sr_adoption_gate")
print(res.to_string(index=False)); print(gate)
print("main equation:", sr1._program)
