"""06A. 조업/비조업 상태 분리가 회귀 성능을 실제로 개선하는가.

Global / State-feature / State-specific 세 모델을 동일한 시간순 holdout 에서 비교한다.
설계 결정:
  - 상태는 사전에 알 수 있는 정보여야 한다. is_operating = (생산량 > 0) 은 생산계획에서
    사전 확정되므로 예측 시점에 알 수 있다고 간주한다.
  - 기본 특성집합(FE_base)에서 생산량·공장인원을 뺀다. 생산량을 넣으면 상태가 그 함수라
    "상태 추가 효과"가 정의상 0 에 가까워지기 때문이다(공장인원은 생산량과 spearman 0.994).
  - 참고선으로 생산량을 포함한 global(FE_full) 도 같이 보고한다.
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from common import load_power, power_long, save_table, fig_path, mpl, SEED
plt = mpl()

q = power_long(load_power()).sort_values("ts15").reset_index(drop=True)
for k in (1, 2, 4, 96):
    q[f"kw_lag{k}"] = q.kw.shift(k)
q["kw_roll4"] = q.kw.shift(1).rolling(4).mean()
q["kw_roll96"] = q.kw.shift(1).rolling(96).mean()
q["is_operating"] = (q["생산량"] > 0).astype(int)
q["clean"] = q.날짜 >= 20210701

FE_BASE = ["시간", "day", "m", "기온", "습도", "풍속", "강수량",
           "kw_lag1", "kw_lag2", "kw_lag4", "kw_lag96", "kw_roll4", "kw_roll96"]
FE_FULL = FE_BASE + ["생산량", "공장인원"]

def fit_pred(mk, Xtr, ytr, Xte):
    return mk().fit(Xtr, ytr).predict(Xte)

MODELS = {"ridge": lambda: Ridge(alpha=1.0),
          "hgb": lambda: HistGradientBoostingRegressor(random_state=SEED)}

rows, resid_store = [], {}
for wtag, sub in [("all_2021", q), ("clean_Jul_Sep", q[q.clean])]:
    sub = sub.dropna(subset=["kw_roll96"]).reset_index(drop=True)
    cut = int(len(sub) * .8)
    tr, te = sub.iloc[:cut], sub.iloc[cut:]
    y_te = te.kw.values
    st_te = te.is_operating.values
    for mname, mk in MODELS.items():
        variants = {}
        # 1) global — 상태 정보 없음
        variants["global"] = fit_pred(mk, tr[FE_BASE].fillna(-1).values, tr.kw.values,
                                      te[FE_BASE].fillna(-1).values)
        # 2) state feature — 상태를 특성 한 개로 투입
        F2 = FE_BASE + ["is_operating"]
        variants["state_feature"] = fit_pred(mk, tr[F2].fillna(-1).values, tr.kw.values,
                                             te[F2].fillna(-1).values)
        # 3) state specific — 상태별 독립 모델
        pred = np.full(len(te), np.nan)
        for s in (0, 1):
            m_tr, m_te = tr.is_operating == s, te.is_operating == s
            if m_tr.sum() < 50 or m_te.sum() == 0:
                continue
            pred[m_te.values] = fit_pred(mk, tr.loc[m_tr, FE_BASE].fillna(-1).values,
                                         tr.loc[m_tr, "kw"].values,
                                         te.loc[m_te, FE_BASE].fillna(-1).values)
        pred = np.where(np.isnan(pred), variants["global"], pred)
        variants["state_specific"] = pred
        # 참고선: 생산량 포함 global
        variants["global_with_production(ref)"] = fit_pred(mk, tr[FE_FULL].fillna(-1).values,
                                                           tr.kw.values, te[FE_FULL].fillna(-1).values)
        base_mae = mean_absolute_error(y_te, variants["global"])
        for vtag, pr in variants.items():
            rows.append(dict(window=wtag, model=mname, variant=vtag,
                             n_train=cut, n_test=len(te),
                             mae=mean_absolute_error(y_te, pr),
                             rmse=float(np.sqrt(((y_te - pr) ** 2).mean())),
                             mae_operating=mean_absolute_error(y_te[st_te == 1], pr[st_te == 1]),
                             mae_idle=mean_absolute_error(y_te[st_te == 0], pr[st_te == 0]),
                             mae_delta_vs_global=mean_absolute_error(y_te, pr) - base_mae,
                             mae_pct_vs_global=(mean_absolute_error(y_te, pr) - base_mae) / base_mae))
            resid_store[(wtag, mname, vtag)] = (y_te, pr, st_te)
res = pd.DataFrame(rows)
save_table(res, "06A_power_state_split_regression")

# 유의성: global vs 각 variant 의 절대오차 차이에 대한 paired bootstrap
boot = []
rng = np.random.default_rng(SEED)
for (wtag, mname, vtag), (y_te, pr, _) in resid_store.items():
    if vtag == "global":
        continue
    y0, p0, _ = resid_store[(wtag, mname, "global")]
    d = np.abs(y_te - pr) - np.abs(y0 - p0)
    idx = rng.integers(0, len(d), size=(2000, len(d)))
    bs = d[idx].mean(1)
    boot.append(dict(window=wtag, model=mname, variant=vtag,
                     mean_abs_err_diff=float(d.mean()),
                     ci_lo=float(np.quantile(bs, .025)), ci_hi=float(np.quantile(bs, .975)),
                     improves_significantly=bool(np.quantile(bs, .975) < 0)))
save_table(pd.DataFrame(boot), "06A_power_state_split_bootstrap")

fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
for ax, wtag in zip(axes, ["all_2021", "clean_Jul_Sep"]):
    d = res[(res.window == wtag) & (res.model == "hgb")]
    ax.bar(range(len(d)), d.mae, color=["#888", "#2b7", "#27b", "#bbb"])
    ax.set_xticks(range(len(d))); ax.set_xticklabels(d.variant, rotation=20, ha="right", fontsize=7)
    ax.set_ylabel("MAE"); ax.set_title(f"{wtag} (HGB) — 상태 분리 효과")
fig.tight_layout(); fig.savefig(fig_path("06A_power_state_split")); plt.close(fig)
print("done")
