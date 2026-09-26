"""06B. 복제(증강) 처리 방식이 실제 성능을 바꾸는가 — Data Reliability Layer 검증.

핵심 설계: 학습 방식만 4가지로 바꾸고 **평가 집합은 하나로 고정**한다.
평가 집합 = 원본 구간(2021-07-01 이후) 중 마지막 20%. 원본이므로 복제 누출이 없다.

학습 변형:
  raw_all      : 테스트 이전 전체 일자 (복제 포함)
  dedup        : profile_id 당 가장 이른 1일만 남김
  inv_dup_wgt  : 전체를 쓰되 sample_weight = 1 / profile_n
  clean_only   : 테스트 이전 원본 구간(7-9월)만
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, roc_auc_score, average_precision_score
from common import load_power, power_long, save_table, fig_path, mpl, SEED
plt = mpl()

p = load_power()
HC = ["15분", "30분", "45분", "60분"]
sig = p.groupby("날짜")[HC].apply(lambda d: tuple(d.values.ravel()))
pid = sig.groupby(sig).ngroup().rename("profile_id")
pmap = pid.to_frame().join(pid.to_frame().groupby("profile_id").size().rename("profile_n"), on="profile_id")
pmap["first_date_of_profile"] = pmap.groupby("profile_id").apply(lambda g: g.index.min()).reindex(pmap.profile_id).values

q = power_long(p).sort_values("ts15").reset_index(drop=True)
q = q.join(pmap, on="날짜")
for k in (1, 2, 4, 96):
    q[f"kw_lag{k}"] = q.kw.shift(k)
q["kw_roll4"] = q.kw.shift(1).rolling(4).mean()
q["kw_roll96"] = q.kw.shift(1).rolling(96).mean()
q["is_operating"] = (q["생산량"] > 0).astype(int)
q = q.dropna(subset=["kw_roll96"]).reset_index(drop=True)
FE = ["시간", "day", "m", "기온", "습도", "풍속", "강수량", "생산량", "is_operating",
      "kw_lag1", "kw_lag2", "kw_lag4", "kw_lag96", "kw_roll4", "kw_roll96"]

clean_days = sorted(q.loc[q.날짜 >= 20210701, "날짜"].unique())
test_days = set(clean_days[int(len(clean_days) * .8):])          # 원본 구간 마지막 20%
te = q[q.날짜.isin(test_days)]
pool = q[~q.날짜.isin(test_days) & (q.ts15 < te.ts15.min())]
thr = np.quantile(q.kw, .95)
te_y_reg = te.kw.values
te_y_clf = (te.kw >= thr).astype(int).values

variants = {
    "raw_all":     (pool, None),
    "dedup":       (pool[pool.날짜 == pool.first_date_of_profile], None),
    "inv_dup_wgt": (pool, 1.0 / pool.profile_n.values),
    "clean_only":  (pool[pool.날짜 >= 20210701], None),
}
rows = []
for vtag, (tr, w) in variants.items():
    Xtr, Xte = tr[FE].fillna(-1).values, te[FE].fillna(-1).values
    base = dict(variant=vtag, n_train_rows=len(tr), n_train_days=int(tr.날짜.nunique()),
                n_test_rows=len(te), n_test_days=len(test_days))
    for mname, mk in [("ridge", lambda: Ridge(alpha=1.0)),
                      ("hgb", lambda: HistGradientBoostingRegressor(random_state=SEED))]:
        m = mk()
        m.fit(Xtr, tr.kw.values, sample_weight=w) if w is not None else m.fit(Xtr, tr.kw.values)
        pr = m.predict(Xte)
        rows.append({**base, "model": mname, "task": "kw(t) regression",
                     "mae": mean_absolute_error(te_y_reg, pr),
                     "rmse": float(np.sqrt(((te_y_reg - pr) ** 2).mean())),
                     "mae_over_std": mean_absolute_error(te_y_reg, pr) / te_y_reg.std(),
                     "auc": np.nan, "ap": np.nan})
    c = HistGradientBoostingClassifier(random_state=SEED)
    ytr = (tr.kw >= thr).astype(int).values
    c.fit(Xtr, ytr, sample_weight=w) if w is not None else c.fit(Xtr, ytr)
    pc = c.predict_proba(Xte)[:, 1]
    rows.append({**base, "model": "hgb_clf", "task": f"kw>=p95({thr:.0f}) classification",
                 "mae": np.nan, "rmse": np.nan, "mae_over_std": np.nan,
                 "auc": roc_auc_score(te_y_clf, pc), "ap": average_precision_score(te_y_clf, pc)})
res = pd.DataFrame(rows)
save_table(res, "06B_power_duplication_handling")

# paired bootstrap: raw_all 대비
rng = np.random.default_rng(SEED)
preds = {}
for vtag, (tr, w) in variants.items():
    m = HistGradientBoostingRegressor(random_state=SEED)
    Xtr = tr[FE].fillna(-1).values
    m.fit(Xtr, tr.kw.values, sample_weight=w) if w is not None else m.fit(Xtr, tr.kw.values)
    preds[vtag] = m.predict(te[FE].fillna(-1).values)
boot = []
a0 = np.abs(te_y_reg - preds["raw_all"])
for vtag, pr in preds.items():
    if vtag == "raw_all":
        continue
    d = np.abs(te_y_reg - pr) - a0
    idx = rng.integers(0, len(d), size=(2000, len(d)))
    bs = d[idx].mean(1)
    boot.append(dict(variant=vtag, model="hgb", mean_abs_err_diff_vs_raw_all=float(d.mean()),
                     ci_lo=float(np.quantile(bs, .025)), ci_hi=float(np.quantile(bs, .975)),
                     better_than_raw_all=bool(np.quantile(bs, .975) < 0),
                     worse_than_raw_all=bool(np.quantile(bs, .025) > 0)))
save_table(pd.DataFrame(boot), "06B_power_duplication_bootstrap")

fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.6))
d = res[res.model == "hgb"]
ax[0].bar(d.variant, d.mae, color="#27b"); ax[0].set_ylabel("MAE"); ax[0].set_title("복제 처리 방식별 회귀 MAE (동일 원본 테스트셋)")
ax[0].tick_params(axis="x", rotation=20)
c = res[res.model == "hgb_clf"]
ax[1].bar(c.variant, c.ap, color="#b72"); ax[1].set_ylabel("AP"); ax[1].set_title("p95 초과 분류 AP")
ax[1].tick_params(axis="x", rotation=20)
fig.tight_layout(); fig.savefig(fig_path("06B_power_duplication_handling")); plt.close(fig)
print("done")
