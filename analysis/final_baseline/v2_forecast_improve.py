"""v2-A. 예측 성능 개선 실험 — 설계상 방해요인 진단 + 해결안 ablation.

평가 규약은 Baseline v1 과 동일하다(바꾸면 비교가 깨진다):
  - clean_Jul_Sep 구간, TimeSeriesSplit 5 fold, fold 내부 TRAIN/CAL/TEST (CAL=TRAIN 뒤 20%)
  - 모든 적합(가중치·인코딩·임계·스케일)은 TRAIN(+CAL) 에서만. TEST 는 적용만.
  - 경고 성능(F1/FP/FN)과 수치 예측 성능(MAE)은 끝까지 분리해 보고한다.

실험 축
  P1 중복일 감사            — 합성 증강으로 생긴 완전 동일 프로파일 날짜가 학습/평가에 섞였는가
  P2 피크 임계 기준 민감도  — '직전 30일 p95' 가 임의 설정인지 데이터로 확인
  P3 예측 성능 ablation     — 잔차타깃 / 주기인코딩 / 극단가중 / 휴무재가동 특성
  P4 스트레스 구간 성능     — OOD 를 '포기 신호' 가 아니라 '취약 조건 지도' 로 사용
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
import _fe                                                               # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor  # noqa: E402
from sklearn.linear_model import Ridge                                   # noqa: E402
from sklearn.model_selection import TimeSeriesSplit                      # noqa: E402
from sklearn.covariance import EmpiricalCovariance                       # noqa: E402

cfg = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
HOR = {1: 15, 2: 30, 3: 45, 4: 60}
NS = cfg["data"]["n_splits"]; CALF = cfg["data"]["cal_fraction"]
t0 = time.time()

# ─────────────────────────────────────────────────────────────── P1 중복일 감사
q0 = _fe.build()
W = q0.pivot_table(index="날짜", columns=q0.ts15.dt.hour * 4 + q0.ts15.dt.minute // 15,
                   values="kw").dropna()
key = W.round(6).apply(lambda r: tuple(r.values), axis=1)
dup = key.duplicated(keep=False)
grp = key[dup]
rows = [dict(scope="all_2021", n_days=len(W), n_dup_days=int(dup.sum()),
             n_dup_groups=int(grp.nunique()), n_redundant_days=int(dup.sum() - grp.nunique()),
             dup_day_share=float(dup.sum() / len(W)))]
Wc = W[W.index >= 20210701]
kc = Wc.round(6).apply(lambda r: tuple(r.values), axis=1); dc = kc.duplicated(keep=False)
rows.append(dict(scope="clean_Jul_Sep", n_days=len(Wc), n_dup_days=int(dc.sum()),
                 n_dup_groups=int(kc[dc].nunique()),
                 n_redundant_days=int(dc.sum() - kc[dc].nunique()),
                 dup_day_share=float(dc.sum() / len(Wc))))
cross = sum(1 for _, v in W[dup].groupby(grp) if min(v.index) < 20210701 <= max(v.index))
rows.append(dict(scope="cross_boundary_groups", n_days=np.nan, n_dup_days=np.nan,
                 n_dup_groups=cross, n_redundant_days=np.nan, dup_day_share=np.nan))
save_table(pd.DataFrame(rows), "31_duplicate_day_audit")
dup_days = sorted(Wc[dc].index.tolist())
print(f"[P1] all_2021 중복일 {int(dup.sum())}/{len(W)} ({grp.nunique()}군), "
      f"clean 중복일 {int(dc.sum())}/{len(Wc)} {dup_days}, 경계교차군 {cross}")

# ─────────────────────────────────────────────── 공통 데이터 (CORE 모드 고정)
d, mode = FT.build(use_external=False)
d = FT.common_rows(d, mode)
FOLDS = list(TimeSeriesSplit(n_splits=NS).split(d))
print(f"[data] mode={mode} rows={len(d)} base_feats={len(BASE_FEATS)} folds={NS}")


def fold_slices(d, folds):
    for fi, (itr, ite) in enumerate(folds):
        cut = int(len(itr) * (1 - CALF))
        yield fi, d.iloc[itr[:cut]], d.iloc[itr[cut:]], d.iloc[ite]


# ───────────────────────────────────────── P2 피크 임계 기준 민감도
def thr_series(kw, win_days, qtl, min_days=7):
    return kw.shift(1).rolling(96 * win_days, min_periods=96 * min_days).quantile(qtl)


p2 = []
base_pred = {}
for win in (14, 30, 60, 90):
    for qtl in (.90, .95, .97):
        t = thr_series(d.kw, win, qtl)
        ok = t.notna()
        lab = (d.kw > t) & ok
        ev = int(((lab.astype(int).diff() == 1) | (lab & ~lab.shift(1).fillna(False))).sum())
        dayl = t[ok].groupby(d.ts15.dt.normalize()[ok]).last()
        p2.append(dict(rule=f"trailing{win}d_p{int(qtl*100)}", window_days=win, quantile=qtl,
                       n_usable=int(ok.sum()), n_peak_steps=int(lab.sum()),
                       prevalence=float(lab[ok].mean()), n_events=ev,
                       thr_mean=float(t[ok].mean()), thr_std=float(t[ok].std()),
                       thr_min=float(t[ok].min()), thr_max=float(t[ok].max()),
                       thr_daily_absdiff_mean=float(dayl.diff().abs().mean()),
                       warmup_lost_steps=int((~ok).sum())))
for fx in (180., 187., 190.):                        # 고정 임계 비교군 (타 팀 관행 θ=187)
    lab = d.kw > fx
    ev = int((lab & ~lab.shift(1).fillna(False)).sum())
    p2.append(dict(rule=f"fixed_{int(fx)}kw", window_days=np.nan, quantile=np.nan,
                   n_usable=len(d), n_peak_steps=int(lab.sum()), prevalence=float(lab.mean()),
                   n_events=ev, thr_mean=fx, thr_std=0.0, thr_min=fx, thr_max=fx,
                   thr_daily_absdiff_mean=0.0, warmup_lost_steps=0))
P2 = pd.DataFrame(p2)
save_table(P2, "31_peak_threshold_sensitivity")
print("[P2]\n" + P2[["rule", "n_peak_steps", "prevalence", "n_events", "thr_std",
                     "thr_daily_absdiff_mean", "warmup_lost_steps"]].round(4).to_string(index=False))


# ───────────────────────────────────────── P3 특성/타깃 변형 정의
def add_cyclic(x):
    """tod(0..95)·요일의 푸리에 인코딩. 결정론적 — 누수 없음."""
    o = {}
    for k in (1, 2, 3):
        o[f"tod_sin{k}"] = np.sin(2 * np.pi * k * x.tod / 96)
        o[f"tod_cos{k}"] = np.cos(2 * np.pi * k * x.tod / 96)
    o["dow_sin"] = np.sin(2 * np.pi * x.day / 7); o["dow_cos"] = np.cos(2 * np.pi * x.day / 7)
    return pd.DataFrame(o, index=x.index)


def add_shutdown(x):
    """휴무 지속/재가동 경과 — 모두 t 시점까지의 인과 정보."""
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
    """SERA 계열 relevance 가중. TRAIN 분위에서만 산출한다."""
    lo, hi = np.quantile(ytr, [.90, .995])
    phi = np.clip((ytr - lo) / max(hi - lo, 1e-6), 0, 1)
    return 1.0 + lam * phi


VARIANTS = {
    "A0_baseline":        dict(resid=False, cyc=False, wgt=False, shut=False),
    "A1_resid":           dict(resid=True,  cyc=False, wgt=False, shut=False),
    "A2_cyclic":          dict(resid=False, cyc=True,  wgt=False, shut=False),
    "A3_extreme_weight":  dict(resid=False, cyc=False, wgt=True,  shut=False),
    "A4_shutdown_feat":   dict(resid=False, cyc=False, wgt=False, shut=True),
    "A5_all":             dict(resid=True,  cyc=True,  wgt=True,  shut=True),
}


def design(x, o):
    X = [x[BASE_FEATS]]
    if o["cyc"]: X.append(add_cyclic(x))
    if o["shut"]: X.append(add_shutdown(x))
    return pd.concat(X, axis=1).fillna(-1).values


def hgb(seed=SEED):
    return HistGradientBoostingRegressor(random_state=seed)


res, oof = [], []
for h, hm in HOR.items():
    y = f"y_h{h}"
    for fi, TR, CA, TE in fold_slices(d, FOLDS):
        FIT = pd.concat([TR, CA])
        yte = TE[y].values
        # 베이스라인 3종
        cand = {"B0_persistence": TE.kw.values,
                "B1_ridge": Ridge(alpha=1.0).fit(FIT[BASE_FEATS].fillna(-1).values, FIT[y].values)
                                  .predict(TE[BASE_FEATS].fillna(-1).values),
                "B2_rf": RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=-1)
                         .fit(FIT[BASE_FEATS].fillna(-1).values, FIT[y].values)
                         .predict(TE[BASE_FEATS].fillna(-1).values)}
        for vt, o in VARIANTS.items():
            Xf, Xt = design(FIT, o), design(TE, o)
            yf = FIT[y].values - (FIT.kw.values if o["resid"] else 0)
            sw = relevance_w(TR[y].values) if o["wgt"] else None
            if o["wgt"]:
                swf = relevance_w(np.r_[TR[y].values, CA[y].values])
                p = hgb().fit(Xf, yf, sample_weight=swf).predict(Xt)
            else:
                p = hgb().fit(Xf, yf).predict(Xt)
            cand[vt] = p + (TE.kw.values if o["resid"] else 0)
        thr = TE.thr_adaptive.values
        pk = yte >= thr
        for name, p in cand.items():
            e = np.abs(yte - p)
            res.append(dict(horizon_min=hm, fold=fi, variant=name, n_test=len(yte),
                            mae=float(e.mean()), rmse=float(np.sqrt((e ** 2).mean())),
                            mae_peak=float(e[pk].mean()) if pk.any() else np.nan,
                            mae_nonpeak=float(e[~pk].mean()),
                            n_peak=int(pk.sum()), err_p95=float(np.quantile(e, .95))))
            if name in ("A0_baseline", "A5_all"):
                oof.append(pd.DataFrame(dict(ts15=TE.ts15.values, fold=fi, horizon_min=hm,
                                             variant=name, y=yte, pred=p, thr=thr,
                                             kw=TE.kw.values, regime_ramp=TE.kw_ramp4.abs().values,
                                             is_operating=TE.is_operating.values)))
        print(f"  h{hm} fold{fi} done  ({time.time()-t0:.0f}s)")
R = pd.DataFrame(res)
save_table(R, "31_v2_ablation_rows")
agg = (R.groupby(["horizon_min", "variant"])
         .agg(mae=("mae", "mean"), rmse=("rmse", "mean"), mae_peak=("mae_peak", "mean"),
              mae_nonpeak=("mae_nonpeak", "mean"), err_p95=("err_p95", "mean")).reset_index())
b = agg[agg.variant == "A0_baseline"].set_index("horizon_min")
agg["mae_vs_A0_pct"] = agg.apply(lambda r: 100 * (r.mae - b.loc[r.horizon_min, "mae"]) /
                                 b.loc[r.horizon_min, "mae"], axis=1)
agg["maepeak_vs_A0_pct"] = agg.apply(lambda r: 100 * (r.mae_peak - b.loc[r.horizon_min, "mae_peak"]) /
                                     b.loc[r.horizon_min, "mae_peak"], axis=1)
save_table(agg, "31_v2_ablation_summary")
print("[P3]\n" + agg.round(3).to_string(index=False))

O = pd.concat(oof, ignore_index=True)
save_table(O, "31_v2_oof_rows")

# paired day-block bootstrap: A5 vs A0 (h60)
def block_boot(a, b, days, n=1000, seed=SEED):
    dd = a - b
    codes, uniq = pd.factorize(days)
    rng = np.random.default_rng(seed); out = []
    idx = [np.where(codes == k)[0] for k in range(len(uniq))]
    for _ in range(n):
        s = rng.integers(0, len(uniq), len(uniq))
        out.append(dd[np.concatenate([idx[k] for k in s])].mean())
    out = np.array(out)
    return float(dd.mean()), float(np.quantile(out, .025)), float(np.quantile(out, .975))


bt = []
for hm in HOR.values():
    a = O[(O.variant == "A5_all") & (O.horizon_min == hm)].sort_values(["fold", "ts15"])
    z = O[(O.variant == "A0_baseline") & (O.horizon_min == hm)].sort_values(["fold", "ts15"])
    dlt, lo, hi = block_boot(np.abs(a.y.values - a.pred.values),
                             np.abs(z.y.values - z.pred.values),
                             pd.to_datetime(a.ts15.values).normalize())
    bt.append(dict(horizon_min=hm, comparison="A5_all - A0_baseline", delta_mae=dlt,
                   ci_lo=lo, ci_hi=hi, improves=bool(hi < 0), significant=bool(hi < 0 or lo > 0)))
BT = pd.DataFrame(bt); save_table(BT, "31_v2_bootstrap")
print("[P3-boot]\n" + BT.round(4).to_string(index=False))

# ───────────────────────────────────────── P4 스트레스 구간 성능 지도
st = []
for hm in HOR.values():
    for vt in ("A0_baseline", "A5_all"):
        s = O[(O.variant == vt) & (O.horizon_min == hm)]
        e = np.abs(s.y.values - s.pred.values)
        ts = pd.to_datetime(s.ts15.values)
        conds = {
            "ALL": np.ones(len(s), bool),
            "peak_only": s.y.values >= s.thr.values,
            "high_ramp_p90": s.regime_ramp.values >= np.quantile(s.regime_ramp.values, .90),
            "non_operating": s.is_operating.values == 0,
            "restart_1h": (s.is_operating.values == 1) &
                          (pd.Series(s.is_operating.values).rolling(4, min_periods=1)
                           .min().values == 0),
            "midday_12_18": np.isin(ts.hour, range(12, 18)),
            "weekend": ts.dayofweek >= 5,
        }
        for cn, m in conds.items():
            if m.sum() == 0: continue
            st.append(dict(horizon_min=hm, variant=vt, condition=cn, n=int(m.sum()),
                           share=float(m.mean()), mae=float(e[m].mean()),
                           mae_ratio_vs_all=float(e[m].mean() / e.mean()),
                           err_p95=float(np.quantile(e[m], .95))))
S = pd.DataFrame(st); save_table(S, "31_v2_stress_map")
print("[P4] h60 취약조건\n" + S[(S.horizon_min == 60)].round(3).to_string(index=False))
print(f"\n[done] {time.time()-t0:.0f}s")
