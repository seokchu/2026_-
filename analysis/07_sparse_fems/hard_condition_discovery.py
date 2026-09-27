"""07-C. 난조건 발견 — OOF 예측 오차에서 출발한다(수작업 정의 금지).

난조건 = 표본외 예측손실/보정실패가 전체 기준선보다 실질적으로 나쁜 문맥.
절차:
  1) TimeSeriesSplit(5) 로 h4(60분) OOF 예측을 만든다. 각 fold 모델은 과거만 학습한다.
  2) 정보수준(L0~L4)별로 같은 폴드 구조에서 OOF 를 만들어 조건별 MAE 를 비교한다.
     → 조건이 '정보 제한형'인지 '내재적 난이도'인지 판별한다(H3).
  3) depth-3 회귀트리(타깃=OOF 절대오차) + 분위구간 단일/쌍 서브그룹으로 규칙 후보를 만든다.
  4) 효과크기·최소지원·폴드 간 재현성 3중 필터를 통과한 것만 난조건으로 승격한다.
산출 07C_oof_predictions.csv 는 07F/07G 가 재사용한다.
"""
import numpy as np, pandas as pd
import _fe
from _fe import build, window, peak_threshold, save_table, fig_path, mpl, SEED, TAB
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.covariance import EmpiricalCovariance
from sklearn.tree import DecisionTreeRegressor
plt = mpl()

H = 4
LAT = ["lat_state", "lat_dwell", "lat_p_high"]
levels = dict(_fe.LEVELS)
if (TAB / "07D_latent_state_features.csv").exists():
    levels["L4_+latent_context"] = _fe.LEVELS["L3_+production"] + LAT
q = build(latent_cols=LAT if (TAB / "07D_latent_state_features.csv").exists() else None)
ALLF = sorted(set().union(*levels.values()))
d = q.dropna(subset=ALLF + [f"y_h{H}"]).reset_index(drop=True)
y = d[f"y_h{H}"].values
tss = TimeSeriesSplit(n_splits=5)
folds = list(tss.split(d))

oof = {l: np.full(len(d), np.nan) for l in levels}
oof["ridge_L3"] = np.full(len(d), np.nan)
fold_id = np.full(len(d), -1)
ood = np.full(len(d), np.nan)
REF = "L3_+production"
for fi, (itr, ite) in enumerate(folds):
    fold_id[ite] = fi
    for l, F in levels.items():
        m = HistGradientBoostingRegressor(random_state=SEED).fit(d.iloc[itr][F].values, y[itr])
        oof[l][ite] = m.predict(d.iloc[ite][F].values)
    r = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(d.iloc[itr][levels[REF]].values, y[itr])
    oof["ridge_L3"][ite] = r.predict(d.iloc[ite][levels[REF]].values)
    cv = EmpiricalCovariance().fit(d.iloc[itr][_fe.L0].values)   # OOD 거리도 과거만으로 적합
    ood[ite] = cv.mahalanobis(d.iloc[ite][_fe.L0].values)

msk = fold_id >= 0
o = d[msk].copy()
o["fold"] = fold_id[msk]
o["y_true"] = y[msk]
for l in oof:
    o[f"pred_{l}"] = oof[l][msk]
o["pred"] = o[f"pred_{REF}"]
o["abs_err"] = np.abs(o.y_true - o.pred)
o["disagreement"] = np.abs(o.pred - o.pred_ridge_L3)
o["ood_maha"] = ood[msk]
thr_peak = peak_threshold(d.iloc[folds[0][0]])         # 첫 fold 학습구간에서만 산출
o["is_peak"] = (o.y_true >= thr_peak).astype(int)
o["clean"] = o["clean"].astype(bool)
o["abs_ramp"] = o.kw_ramp4.abs()
save_table(o[["ts15", "fold", "clean", "y_true", "pred", "abs_err", "disagreement", "ood_maha",
              "is_peak", "is_operating", "생산량", "prod_diff", "시간", "tod", "day", "kw",
              "kw_ramp4", "abs_ramp", "kw_std4", "kw_std96", "기온", "습도"] +
             [f"pred_{l}" for l in levels] + ([*LAT] if "lat_state" in o else [])],
           "07C_oof_predictions")

overall = o.abs_err.mean()
CTX = ["is_operating", "생산량", "prod_diff", "시간", "tod", "day", "kw", "kw_ramp4", "abs_ramp",
       "kw_std4", "kw_std96", "기온", "습도", "disagreement", "ood_maha"] + \
      ([*LAT] if "lat_state" in o else [])

# ---- 규칙 후보 1: depth-3 회귀트리 리프
tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=max(200, int(len(o) * .02)),
                             random_state=SEED).fit(o[CTX].values, o.abs_err.values)
T = tree.tree_
def leaf_rules():
    out, stack = {}, [(0, [])]
    while stack:
        n, path = stack.pop()
        if T.children_left[n] == -1:
            out[n] = " AND ".join(path) if path else "ALL"
            continue
        f, t = CTX[T.feature[n]], T.threshold[n]
        stack.append((T.children_left[n], path + [f"{f}<={t:.3g}"]))
        stack.append((T.children_right[n], path + [f"{f}>{t:.3g}"]))
    return out
rules = leaf_rules()
leaf = tree.apply(o[CTX].values)
cands = [(f"tree_leaf_{n}", rules[n], leaf == n) for n in np.unique(leaf)]

# ---- 규칙 후보 2: 단일/쌍 분위 서브그룹 (상위 분위만, 조합 폭발 방지)
bin_specs = {}
for c in ["생산량", "abs_ramp", "kw", "kw_std4", "disagreement", "ood_maha", "기온"]:
    hi = float(np.quantile(o[c], .8))
    bin_specs[f"{c}_top20"] = (f"{c}>{hi:.3g}", o[c] > hi)
bin_specs["idle"] = ("is_operating==0", o.is_operating == 0)
bin_specs["operating"] = ("is_operating==1", o.is_operating == 1)
for k, (r, m) in bin_specs.items():
    cands.append((f"single_{k}", r, m.values if hasattr(m, "values") else m))
keys = list(bin_specs)
for i in range(len(keys)):
    for j in range(i + 1, len(keys)):
        (r1, m1), (r2, m2) = bin_specs[keys[i]], bin_specs[keys[j]]
        m = np.asarray(m1) & np.asarray(m2)
        if m.sum() >= max(150, int(len(o) * .01)):
            cands.append((f"pair_{keys[i]}&{keys[j]}", f"{r1} AND {r2}", m))

rows = []
for cid, rule, m in cands:
    if m.sum() < max(150, int(len(o) * .01)):
        continue
    s = o[m]
    fr = s.groupby("fold").abs_err.mean() / o.groupby("fold").abs_err.mean()
    rows.append(dict(
        hard_condition_id=cid, rule=rule, n_samples=int(m.sum()), share_of_data=float(m.mean()),
        mae=float(s.abs_err.mean()), mae_ratio_vs_overall=float(s.abs_err.mean() / overall),
        peak_prevalence=float(s.is_peak.mean()),
        mae_clean=float(s[s.clean].abs_err.mean()) if s.clean.any() else np.nan,
        mae_ratio_clean=float(s[s.clean].abs_err.mean() / o[o.clean].abs_err.mean()) if s.clean.any() else np.nan,
        folds_with_ratio_gt_1_3=int((fr > 1.3).sum()), n_folds_present=int(fr.notna().sum()),
        fold_ratio_min=float(fr.min()), fold_ratio_max=float(fr.max()),
        mae_L0=float(np.abs(s.y_true - s["pred_L0_power_only"]).mean()),
        mae_L3=float(np.abs(s.y_true - s[f"pred_{REF}"]).mean()),
        info_gain_L0_to_L3=float(np.abs(s.y_true - s["pred_L0_power_only"]).mean() -
                                np.abs(s.y_true - s[f"pred_{REF}"]).mean()),
    ))
hc = pd.DataFrame(rows)
hc["info_gain_pct"] = hc.info_gain_L0_to_L3 / hc.mae_L0
g_all = (np.abs(o.y_true - o["pred_L0_power_only"]).mean() - overall) / np.abs(o.y_true - o["pred_L0_power_only"]).mean()
hc["info_limited"] = hc.info_gain_pct > max(0.05, g_all * 1.5)   # 전체 평균보다 정보이득이 크면 정보제한형
hc["promoted"] = (hc.mae_ratio_vs_overall >= 1.3) & (hc.share_of_data >= .01) & \
                 (hc.folds_with_ratio_gt_1_3 >= 4) & (hc.mae_ratio_clean >= 1.2)
hc = hc.sort_values("mae_ratio_vs_overall", ascending=False)
save_table(hc, "07C_hard_conditions")

# ---- 정보수준 × 승격 난조건 MAE 표 (H2/H3)
lv_rows = []
for r in hc[hc.promoted].itertuples():
    m = next(mm for cid, _, mm in cands if cid == r.hard_condition_id)
    s = o[m]
    for l in levels:
        lv_rows.append(dict(hard_condition_id=r.hard_condition_id, level=l,
                            mae=float(np.abs(s.y_true - s[f"pred_{l}"]).mean()),
                            mae_overall_same_level=float(np.abs(o.y_true - o[f"pred_{l}"]).mean())))
lv = pd.DataFrame(lv_rows)
if len(lv):
    lv["ratio"] = lv.mae / lv.mae_overall_same_level
save_table(lv, "07C_hard_condition_by_level")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
top = hc.sort_values("mae_ratio_vs_overall", ascending=False).head(12)[::-1]
axes[0].barh(range(len(top)), top.mae_ratio_vs_overall,
             color=["#c33" if p else "#aaa" for p in top.promoted])
axes[0].set_yticks(range(len(top))); axes[0].set_yticklabels(top.hard_condition_id, fontsize=6)
axes[0].axvline(1, color="k", lw=.8); axes[0].set_xlabel("MAE ratio vs overall")
axes[0].set_title("난조건 후보 (빨강=승격)")
axes[1].scatter(hc.mae_ratio_vs_overall, hc.info_gain_pct, c=["#c33" if p else "#aaa" for p in hc.promoted], s=18)
axes[1].axhline(0, color="k", lw=.8)
axes[1].set_xlabel("MAE ratio vs overall"); axes[1].set_ylabel("L0→L3 정보이득 비율")
axes[1].set_title("난이도 vs 정보민감도")
fig.tight_layout(); fig.savefig(fig_path("07C_hard_conditions")); plt.close(fig)
print("overall OOF MAE", round(overall, 3), "| promoted", int(hc.promoted.sum()), "/", len(hc))
print(hc.head(14)[["hard_condition_id", "rule", "n_samples", "mae", "mae_ratio_vs_overall",
                   "mae_ratio_clean", "folds_with_ratio_gt_1_3", "info_gain_pct", "info_limited",
                   "promoted"]].to_string(index=False))
