"""31-C. OOD 의 역할 재정의 — '포기 신호' 가 아니라 '취약조건 지도 + 보강 대상'.

세 가지를 구분해서 답한다.
  C1 OOD 점수가 실제로 큰 오차를 가리키는가 (지시력)        — 상관·십분위 곡선
  C2 OOD 고점 구간을 '가중 학습' 으로 보강하면 나아지는가   — A6 변형
  C3 학습분포에서 멀어질수록 성능이 어떻게 무너지는가        — fold 순서별 열화
전부 h60 에 한정한다(계산비용). 적합은 TRAIN(+CAL) 에서만.
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
import _fe as _FE                                                        # noqa: E402

# ablation 기준선은 v1 의 특성집합이어야 한다. features.CORE 는 v2 채택 후 확장되었으므로
# 여기서는 확장 전 집합을 직접 쓴다(그러지 않으면 A0 가 이미 A2+A4 를 포함해 비교가 무의미해진다).
BASE_FEATS = _FE.LEVELS["L3_+production"]
from sklearn.ensemble import HistGradientBoostingRegressor               # noqa: E402
from sklearn.covariance import EmpiricalCovariance                       # noqa: E402
from sklearn.preprocessing import StandardScaler                         # noqa: E402
from sklearn.model_selection import TimeSeriesSplit                      # noqa: E402
from scipy.stats import spearmanr                                        # noqa: E402

cfg = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
CALF = cfg["data"]["cal_fraction"]; NS = cfg["data"]["n_splits"]
OODQ = cfg["reliability"]["ood_quantile"]
H, HM = 4, 60
t0 = time.time()


# v2_forecast_improve.py 와 동일 정의 (모듈 임포트 시 그 스크립트가 통째로 실행되므로 복제)
def add_cyclic(x):
    o = {}
    for k in (1, 2, 3):
        o[f"tod_sin{k}"] = np.sin(2 * np.pi * k * x.tod / 96)
        o[f"tod_cos{k}"] = np.cos(2 * np.pi * k * x.tod / 96)
    o["dow_sin"] = np.sin(2 * np.pi * x.day / 7); o["dow_cos"] = np.cos(2 * np.pi * x.day / 7)
    return pd.DataFrame(o, index=x.index)


def add_shutdown(x):
    op = x.is_operating.values.astype(int)
    run_off = np.zeros(len(op)); run_on = np.zeros(len(op))
    for i in range(len(op)):
        if op[i] == 0:
            run_off[i] = (run_off[i - 1] + 1) if i else 1; run_on[i] = 0
        else:
            run_on[i] = (run_on[i - 1] + 1) if i else 1; run_off[i] = 0
    return pd.DataFrame({"off_run_len": run_off, "on_run_len": run_on,
                         "restart_recent": (run_on > 0) & (run_on <= 8),
                         "kw_vs_roll96": x.kw.values - x.kw_roll96.values,
                         "kw_ratio_roll96": x.kw.values / np.maximum(x.kw_roll96.values, 1.0)},
                        index=x.index).astype(float)


def relevance_w(ytr, lam=4.0):
    lo, hi = np.quantile(ytr, [.90, .995])
    return 1.0 + lam * np.clip((ytr - lo) / max(hi - lo, 1e-6), 0, 1)


VARIANTS = {"A0_baseline": dict(resid=False, cyc=False, wgt=False, shut=False),
            "A5_all": dict(resid=True, cyc=True, wgt=True, shut=True)}


d, mode = FT.build(use_external=False)
d = FT.common_rows(d, mode)
FOLDS = list(TimeSeriesSplit(n_splits=NS).split(d))
OODF = ["kw", "kw_ramp4", "kw_std4", "kw_roll96", "생산량", "기온"]


def design(x, o):
    X = [x[BASE_FEATS]]
    if o["cyc"]: X.append(add_cyclic(x))
    if o["shut"]: X.append(add_shutdown(x))
    return pd.concat(X, axis=1).fillna(-1).values


rows, deciles, oof = [], [], []
for fi, (itr, ite) in enumerate(FOLDS):
    cut = int(len(itr) * (1 - CALF))
    TR, CA, TE = d.iloc[itr[:cut]], d.iloc[itr[cut:]], d.iloc[ite]
    FIT = pd.concat([TR, CA])
    y = f"y_h{H}"; yte = TE[y].values

    sc = StandardScaler().fit(TR[OODF].fillna(-1).values)
    cov = EmpiricalCovariance().fit(sc.transform(TR[OODF].fillna(-1).values))
    m_tr = cov.mahalanobis(sc.transform(TR[OODF].fillna(-1).values))
    cut_ood = float(np.quantile(m_tr, OODQ))
    m_fit = cov.mahalanobis(sc.transform(FIT[OODF].fillna(-1).values))
    m_te = cov.mahalanobis(sc.transform(TE[OODF].fillna(-1).values))

    preds = {}
    for vt in ("A0_baseline", "A5_all"):
        o = VARIANTS[vt]
        Xf, Xt = design(FIT, o), design(TE, o)
        yf = FIT[y].values - (FIT.kw.values if o["resid"] else 0)
        sw = relevance_w(FIT[y].values) if o["wgt"] else None
        preds[vt] = (HistGradientBoostingRegressor(random_state=SEED)
                     .fit(Xf, yf, sample_weight=sw).predict(Xt)
                     + (TE.kw.values if o["resid"] else 0))
    # C2: A5 + OOD 가중 (분포 가장자리 표본을 더 보게 한다)
    o = VARIANTS["A5_all"]
    Xf, Xt = design(FIT, o), design(TE, o)
    yf = FIT[y].values - FIT.kw.values
    w_rel = relevance_w(FIT[y].values)
    w_ood = 1.0 + 3.0 * np.clip((m_fit - np.quantile(m_tr, .80)) /
                                max(cut_ood - np.quantile(m_tr, .80), 1e-6), 0, 1)
    preds["A6_ood_weight"] = (HistGradientBoostingRegressor(random_state=SEED)
                              .fit(Xf, yf, sample_weight=w_rel * w_ood).predict(Xt)
                              + TE.kw.values)

    for vt, p in preds.items():
        e = np.abs(yte - p)
        is_ood = m_te >= cut_ood
        r, _ = spearmanr(m_te, e)
        rows.append(dict(fold=fi, variant=vt, n_test=len(TE), horizon_min=HM,
                         mae=float(e.mean()),
                         mae_ood=float(e[is_ood].mean()) if is_ood.any() else np.nan,
                         mae_in=float(e[~is_ood].mean()), n_ood=int(is_ood.sum()),
                         ood_share=float(is_ood.mean()), ood_cut=cut_ood,
                         spearman_ood_vs_abserr=float(r),
                         mean_maha_test=float(m_te.mean()),
                         maha_shift_vs_train=float(m_te.mean() / max(m_tr.mean(), 1e-9))))
        q = pd.qcut(m_te, 10, labels=False, duplicates="drop")
        for k in np.unique(q):
            mk = q == k
            deciles.append(dict(fold=fi, variant=vt, maha_decile=int(k), n=int(mk.sum()),
                                mean_maha=float(m_te[mk].mean()), mae=float(e[mk].mean())))
        oof.append(pd.DataFrame(dict(ts15=TE.ts15.values, fold=fi, variant=vt, y=yte, pred=p,
                                     maha=m_te, is_ood=is_ood, thr=TE.thr_adaptive.values)))
    print(f"  fold{fi} done ({time.time()-t0:.0f}s)")

R = pd.DataFrame(rows); save_table(R, "31_ood_role_rows")
agg = R.groupby("variant").agg(mae=("mae", "mean"), mae_ood=("mae_ood", "mean"),
                               mae_in=("mae_in", "mean"),
                               spearman=("spearman_ood_vs_abserr", "mean"),
                               ood_share=("ood_share", "mean")).reset_index()
agg["ood_penalty_ratio"] = agg.mae_ood / agg.mae_in
save_table(agg, "31_ood_role_summary")
save_table(pd.DataFrame(deciles), "31_ood_decile_curve")
O = pd.concat(oof, ignore_index=True)

# C3: fold 가 뒤로 갈수록(학습분포에서 멀어질수록) 열화하는가
c3 = (R.groupby(["variant", "fold"]).agg(mae=("mae", "mean"),
                                         shift=("maha_shift_vs_train", "mean")).reset_index())
save_table(c3, "31_ood_fold_drift")
print("[C1/C2]\n" + agg.round(4).to_string(index=False))
print("[C3]\n" + c3.round(3).to_string(index=False))


def block_boot(a, b, days, n=1000, seed=SEED):
    dd = a - b; codes, uniq = pd.factorize(days)
    idx = [np.where(codes == k)[0] for k in range(len(uniq))]
    rng = np.random.default_rng(seed)
    o = np.array([dd[np.concatenate([idx[k] for k in rng.integers(0, len(uniq), len(uniq))])].mean()
                  for _ in range(n)])
    return float(dd.mean()), float(np.quantile(o, .025)), float(np.quantile(o, .975))


bt = []
base = O[O.variant == "A5_all"].sort_values(["fold", "ts15"])
for vt in ("A6_ood_weight",):
    a = O[O.variant == vt].sort_values(["fold", "ts15"])
    for scope, m in [("ALL", np.ones(len(a), bool)), ("OOD_only", a.is_ood.values),
                     ("peak_only", a.y.values >= a.thr.values)]:
        dlt, lo, hi = block_boot(np.abs(a.y.values - a.pred.values)[m],
                                 np.abs(base.y.values - base.pred.values)[m],
                                 pd.to_datetime(a.ts15.values).normalize()[m])
        bt.append(dict(comparison=f"{vt} - A5_all", scope=scope, n=int(m.sum()),
                       delta_mae=dlt, ci_lo=lo, ci_hi=hi, improves=bool(hi < 0)))
BT = pd.DataFrame(bt); save_table(BT, "31_ood_weight_bootstrap")
print("[C2-boot]\n" + BT.round(4).to_string(index=False))
print(f"[done] {time.time()-t0:.0f}s")
