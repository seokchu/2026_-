"""31-D. v2 ablation 을 v1 보고 구간(clean_Jul_Sep)으로 한정해 집계 + 중복일 fold 교차 감사."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "analysis" / "07_sparse_fems"))
from common import save_table, SEED                                      # noqa: E402
import features as FT, _fe                                               # noqa: E402
from sklearn.model_selection import TimeSeriesSplit                      # noqa: E402

# --- 중복일이 fold 의 TRAIN/TEST 양쪽에 걸치는가 (= 평가 누수)
d, _ = FT.build(use_external=False); d = FT.common_rows(d, "CORE")
F = list(TimeSeriesSplit(n_splits=5).split(d))
day = d.ts15.dt.normalize()
q = _fe.build()
W = q.pivot_table(index="날짜", columns=q.ts15.dt.hour * 4 + q.ts15.dt.minute // 15,
                  values="kw").dropna()
k = W.round(6).apply(lambda r: tuple(r.values), axis=1); dup = k.duplicated(keep=False)
groups = [set(v.index) for _, v in W[dup].groupby(k[dup])]
rows = []
for fi, (itr, ite) in enumerate(F):
    tr = set(pd.to_datetime(day.iloc[itr]).dt.strftime("%Y%m%d").astype(int))
    te = set(pd.to_datetime(day.iloc[ite]).dt.strftime("%Y%m%d").astype(int))
    rows.append(dict(fold=fi, n_train_days=len(tr), n_test_days=len(te),
                     n_dup_groups_crossing=sum(1 for g in groups if g & tr and g & te),
                     test_start=str(d.ts15.iloc[ite[0]].date()),
                     test_end=str(d.ts15.iloc[ite[-1]].date()),
                     test_all_in_clean=bool(d.ts15.iloc[ite[0]] >= pd.Timestamp("2021-07-01"))))
X = pd.DataFrame(rows); save_table(X, "31_duplicate_fold_crossing")
print(X.to_string(index=False))

# --- clean 구간 한정 집계
o = pd.read_csv(ROOT / "analysis/tables/31_v2_oof_rows.csv", parse_dates=["ts15"],
                encoding="utf-8-sig")
cl = o[o.ts15 >= pd.Timestamp("2021-07-01")]
res = []
for v in sorted(cl.variant.unique()):
    for hm in (15, 30, 45, 60):
        s = cl[(cl.variant == v) & (cl.horizon_min == hm)]
        e = np.abs(s.y.values - s.pred.values); pk = s.y.values >= s.thr.values
        res.append(dict(window="clean_Jul_Sep", variant=v, horizon_min=hm, n=len(s),
                        mae=float(e.mean()), rmse=float(np.sqrt((e**2).mean())),
                        mae_peak=float(e[pk].mean()), mae_nonpeak=float(e[~pk].mean()),
                        n_peak=int(pk.sum()), err_p95=float(np.quantile(e, .95))))
R = pd.DataFrame(res)
b = R[R.variant == "A0_baseline"].set_index("horizon_min")
R["mae_vs_A0_pct"] = R.apply(lambda r: 100*(r.mae-b.loc[r.horizon_min, "mae"])/b.loc[r.horizon_min, "mae"], axis=1)
R["maepeak_vs_A0_pct"] = R.apply(lambda r: 100*(r.mae_peak-b.loc[r.horizon_min, "mae_peak"])/b.loc[r.horizon_min, "mae_peak"], axis=1)
save_table(R, "31_v2_clean_summary")
print(R.round(3).to_string(index=False))


def block_boot(a, b_, days, n=1000, seed=SEED):
    dd = a - b_; codes, uniq = pd.factorize(days)
    idx = [np.where(codes == j)[0] for j in range(len(uniq))]
    rng = np.random.default_rng(seed)
    out = np.array([dd[np.concatenate([idx[j] for j in rng.integers(0, len(uniq), len(uniq))])].mean()
                    for _ in range(n)])
    return float(dd.mean()), float(np.quantile(out, .025)), float(np.quantile(out, .975))


bt = []
for hm in (15, 30, 45, 60):
    a = cl[(cl.variant == "A5_all") & (cl.horizon_min == hm)].sort_values(["fold", "ts15"])
    z = cl[(cl.variant == "A0_baseline") & (cl.horizon_min == hm)].sort_values(["fold", "ts15"])
    ea, ez = np.abs(a.y.values-a.pred.values), np.abs(z.y.values-z.pred.values)
    pk = a.y.values >= a.thr.values
    dys = pd.to_datetime(a.ts15.values).normalize()
    for scope, m in (("ALL", np.ones(len(a), bool)), ("peak_only", pk)):
        dl, lo, hi = block_boot(ea[m], ez[m], dys[m])
        bt.append(dict(window="clean_Jul_Sep", horizon_min=hm, scope=scope, n=int(m.sum()),
                       comparison="A5_all - A0_baseline", delta_mae=dl, ci_lo=lo, ci_hi=hi,
                       improves=bool(hi < 0)))
B = pd.DataFrame(bt); save_table(B, "31_v2_clean_bootstrap")
print(B.round(4).to_string(index=False))
