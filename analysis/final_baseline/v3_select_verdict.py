"""33-B. 최종모델 선정 — 후보 2종(ExtraTrees vs HistGBM)의 호라이즌별 유의성 + 채택 판정.

사전 선언한 선정 규칙(실행 전 고정):
  TASK-R 수치 예측
    (a) clean 구간 4개 horizon 전부에서 MAE 가 현행(HistGBM) 대비 악화되지 않는다
    (b) h60 의 일 블록 bootstrap 95% CI 상한 < 0
    (c) 고사용량 구간 MAE 가 악화되지 않는다
  TASK-C 고사용량 경보
    config.yaml 의 operator_advisory_capacity(=12건/일) 를 넘지 않는 후보 중 clean F1 최대,
    동점 시 사건 recall 로 정한다. (용량 제약은 본 실험 이전부터 config 에 있던 값이다.)
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd, yaml
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "analysis" / "07_sparse_fems"))
from common import save_table, SEED                                      # noqa: E402
import features as FT, models as MD                                      # noqa: E402
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor  # noqa: E402
from sklearn.model_selection import TimeSeriesSplit                      # noqa: E402

cfg = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
NS, CALF = cfg["data"]["n_splits"], cfg["data"]["cal_fraction"]
CAP = cfg["policy"]["operator_advisory_capacity"]
HOR = {1: 15, 2: 30, 3: 45, 4: 60}
t0 = time.time()
d, _ = FT.build(use_external=False); d = FT.common_rows(d, "CORE")
F = FT.CORE
FOLDS = list(TimeSeriesSplit(n_splits=NS).split(d))

CAND = {"R4_hist_gbm": lambda: HistGradientBoostingRegressor(random_state=SEED),
        "R3_extra_trees": lambda: ExtraTreesRegressor(n_estimators=300, min_samples_leaf=2,
                                                      random_state=SEED, n_jobs=-1)}
rows = []
for h, hm in HOR.items():
    y = f"y_h{h}"
    for fi, (itr, ite) in enumerate(FOLDS):
        cut = int(len(itr) * (1 - CALF))
        FIT, TE = d.iloc[itr], d.iloc[ite]
        w = MD.relevance_weight(FIT[y].values)
        for nm, mk in CAND.items():
            p = mk().fit(FIT[F].fillna(-1).values, FIT[y].values - FIT.kw.values,
                         sample_weight=w).predict(TE[F].fillna(-1).values) + TE.kw.values
            rows.append(pd.DataFrame(dict(ts15=TE.ts15.values, fold=fi, horizon_min=hm,
                                          model=nm, y=TE[y].values, pred=p,
                                          thr=TE.thr_adaptive.values, clean=TE.clean.values)))
    print(f"  h{hm} ({time.time()-t0:.0f}s)")
P = pd.concat(rows, ignore_index=True)
cl = P[P.clean]


def block_boot(a, b, days, n=1000, seed=SEED):
    dd = a - b; codes, uniq = pd.factorize(days)
    idx = [np.where(codes == k)[0] for k in range(len(uniq))]
    rng = np.random.default_rng(seed)
    o = np.array([dd[np.concatenate([idx[k] for k in rng.integers(0, len(uniq), len(uniq))])].mean()
                  for _ in range(n)])
    return float(dd.mean()), float(np.quantile(o, .025)), float(np.quantile(o, .975))


summ, boot = [], []
for hm in HOR.values():
    for nm in CAND:
        g = cl[(cl.model == nm) & (cl.horizon_min == hm)]
        e = np.abs(g.y.values - g.pred.values); pk = g.y.values >= g.thr.values
        summ.append(dict(window="clean_Jul_Sep", horizon_min=hm, model=nm, n=len(g),
                         mae=float(e.mean()), rmse=float(np.sqrt((e**2).mean())),
                         mae_peak=float(e[pk].mean()),
                         bias_peak=float((g.y.values - g.pred.values)[pk].mean()),
                         err_p95=float(np.quantile(e, .95))))
    a = cl[(cl.model == "R3_extra_trees") & (cl.horizon_min == hm)].sort_values(["fold", "ts15"])
    b = cl[(cl.model == "R4_hist_gbm") & (cl.horizon_min == hm)].sort_values(["fold", "ts15"])
    ea, eb = np.abs(a.y.values - a.pred.values), np.abs(b.y.values - b.pred.values)
    pk = a.y.values >= a.thr.values
    dys = pd.to_datetime(a.ts15.values).normalize()
    for scope, m in (("ALL", np.ones(len(a), bool)), ("peak_only", pk)):
        dl, lo, hi = block_boot(ea[m], eb[m], dys[m])
        boot.append(dict(window="clean_Jul_Sep", horizon_min=hm, scope=scope, n=int(m.sum()),
                         comparison="R3_extra_trees - R4_hist_gbm", delta_mae=dl,
                         ci_lo=lo, ci_hi=hi, improves=bool(hi < 0)))
S = pd.DataFrame(summ); B = pd.DataFrame(boot)
save_table(S, "33_select_regression_clean"); save_table(B, "33_select_regression_bootstrap")
print(S.round(3).to_string(index=False)); print(B.round(3).to_string(index=False))

piv = S.pivot(index="horizon_min", columns="model", values=["mae", "mae_peak"])
cond_a = bool((piv[("mae", "R3_extra_trees")] <= piv[("mae", "R4_hist_gbm")]).all())
cond_b = bool(B[(B.horizon_min == 60) & (B.scope == "ALL")].improves.iloc[0])
cond_c = bool((piv[("mae_peak", "R3_extra_trees")] <= piv[("mae_peak", "R4_hist_gbm")]).all())
V = pd.DataFrame([dict(task="R_point_forecast", candidate="R3_extra_trees",
                       incumbent="R4_hist_gbm",
                       cond_a_all_horizons_not_worse=cond_a,
                       cond_b_h60_bootstrap_sig=cond_b,
                       cond_c_peak_mae_not_worse=cond_c,
                       adopt=bool(cond_a and cond_b and cond_c),
                       mae_h60_candidate=float(piv.loc[60, ("mae", "R3_extra_trees")]),
                       mae_h60_incumbent=float(piv.loc[60, ("mae", "R4_hist_gbm")]),
                       rule="사전 선언: (a) 4개 horizon 전부 MAE 악화 없음 AND "
                            "(b) h60 일블록 bootstrap 95% CI 상한<0 AND (c) 고사용량 MAE 악화 없음")])

CC = pd.read_csv(ROOT / "analysis/tables/33_model_compare_alert_clean.csv", encoding="utf-8-sig")
ok = CC[CC.alerts_per_day <= CAP].sort_values(["f1", "event_recall"], ascending=False)
win = ok.iloc[0]
V2 = pd.DataFrame([dict(task="C_peak_alert", candidate=win.model,
                        incumbent="C1_reg_margin(현행 점예측 마진)",
                        cond_a_all_horizons_not_worse=np.nan,
                        cond_b_h60_bootstrap_sig=np.nan,
                        cond_c_peak_mae_not_worse=np.nan,
                        adopt=True,
                        mae_h60_candidate=np.nan, mae_h60_incumbent=np.nan,
                        rule=f"사전 선언: 경보 {CAP}건/일 이하 후보 중 clean F1 최대, "
                             f"동점 시 사건 recall. 선택={win.model} "
                             f"(F1 {win.f1:.3f}, 사건recall {win.event_recall:.3f}, "
                             f"{win.alerts_per_day:.1f}건/일)")])
save_table(pd.concat([V, V2], ignore_index=True), "33_model_selection_verdict")
print("\n[채택 판정]")
print(f"  TASK-R: ExtraTrees 채택={cond_a and cond_b and cond_c} (a={cond_a} b={cond_b} c={cond_c})")
print(f"  TASK-C: {win.model} 선택 — F1 {win.f1:.3f} / 사건recall {win.event_recall:.3f} / "
      f"{win.alerts_per_day:.1f}건·일 (용량 {CAP} 이하)")
print(f"[done] {time.time()-t0:.0f}s")
