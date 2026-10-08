"""35. 경보 성능 업그레이드 — 정밀도 중심.

문제: 모든 모델의 정밀도가 0.18~0.38 로 낮다. 경보 1건 중 2/3 이 헛경고다.

문헌이 제시하는 세 갈래를 전부 구현해 동일 조건에서 비교한다.
  (1) 임계 거리 기반 특성 추가 — margin(kw-thr), 비율, 당일 누적 최대, 최근 피크 빈도,
      마지막 피크 이후 경과, 시간대별 과거 피크율(TRAIN 전용 target encoding + smoothing),
      회귀/분위회귀 예측 마진.
  (2) 산업 경보 표준 후처리 — delay-timer(n-out-of-m) 와 확률 이동평균.
      CAL 에서 임계와 함께 고른다. 선행시간 손실을 함께 기록한다.
  (3) 모델 — RandomForest / ExtraTrees / HistGBM / soft-voting 앙상블.
      클래스 가중은 쓰지 않는다(문헌·자체 실험 모두 정밀도 악화).

평가 규약 동일: rolling-origin 5fold, TRAIN 적합 / CAL 선택 / TEST 적용.
"""
import sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd, yaml
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "analysis" / "07_sparse_fems"))
from common import save_table, SEED                                       # noqa: E402
import features as FT                                                     # noqa: E402
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,  # noqa: E402
                              HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor)
from sklearn.model_selection import TimeSeriesSplit                       # noqa: E402
from sklearn.metrics import (matthews_corrcoef, average_precision_score,  # noqa: E402
                             roc_auc_score, brier_score_loss)
from sklearn.isotonic import IsotonicRegression                           # noqa: E402

cfg = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
NS, CALF = cfg["data"]["n_splits"], cfg["data"]["cal_fraction"]
H, HM = 4, 60
t0 = time.time()

d, _ = FT.build(use_external=False)
d = FT.common_rows(d, "CORE").reset_index(drop=True)
BASE = list(FT.CORE)

# ── (1) 임계 거리 기반 특성. 전부 결정시점 t 까지의 관측으로만 만든다.
d["margin"] = d.kw - d.thr_adaptive
d["margin_ratio"] = d.kw / d.thr_adaptive
for k in (1, 2, 4, 8, 96):
    d[f"margin_lag{k}"] = d.margin.shift(k)
d["margin_ramp4"] = d.margin - d.margin_lag4
d["margin_roll4_max"] = d.margin.rolling(4).max()
d["margin_roll96_max"] = d.margin.rolling(96).max()
d["margin_roll96_mean"] = d.margin.rolling(96).mean()
d["peak_now"] = (d.kw >= d.thr_adaptive).astype(float)
d["peak_cnt_96"] = d.peak_now.shift(1).rolling(96).sum()
d["peak_cnt_672"] = d.peak_now.shift(1).rolling(672).sum()
_last = np.full(len(d), np.nan); c = np.nan
pk = d.peak_now.values
for i in range(len(d)):
    _last[i] = c
    c = 0.0 if pk[i] == 1 else (c + 1 if np.isfinite(c) else np.nan)
d["steps_since_peak"] = _last
d["day_i"] = d.ts15.dt.normalize()
d["kw_today_max"] = d.groupby("day_i").kw.cummax()
d["today_max_margin"] = d.kw_today_max - d.thr_adaptive
d["prod_today_cum"] = d.groupby("day_i")["생산량"].cumsum()
EXTRA = ["margin", "margin_ratio", "margin_lag1", "margin_lag2", "margin_lag4", "margin_lag8",
         "margin_lag96", "margin_ramp4", "margin_roll4_max", "margin_roll96_max",
         "margin_roll96_mean", "peak_cnt_96", "peak_cnt_672", "steps_since_peak",
         "today_max_margin", "prod_today_cum"]
FEAT = BASE + EXTRA + ["tod_peak_rate", "pred_margin", "q90_margin", "q75_margin"]
FOLDS = list(TimeSeriesSplit(n_splits=NS).split(d))
print(f"[data] rows={len(d)} base={len(BASE)} extra={len(EXTRA)}+4")


def tod_rate(train_df, prior_w=50.0):
    """시간대(0~95)별 과거 피크율. TRAIN 라벨만 쓰고 전역평균으로 smoothing 한다."""
    g = train_df.groupby("tod").peak_now.agg(["sum", "count"])
    p0 = float(train_df.peak_now.mean())
    return ((g["sum"] + prior_w * p0) / (g["count"] + prior_w)).to_dict(), p0


def confusion(y, pr):
    tp = int((pr & y).sum()); fp = int((pr & ~y).sum())
    fn = int((~pr & y).sum()); tn = int((~pr & ~y).sum())
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return tp, fp, fn, tn, prec, rec, f1


def k_of_m(x, k, m):
    """delay-timer: 최근 m 스텝 중 k 개 이상이 양성이면 경보(현재 스텝 포함)."""
    if m == 1:
        return x.astype(bool)
    s = pd.Series(x.astype(float)).rolling(m, min_periods=1).sum().values
    return (s >= k) & x.astype(bool) if k == m else (s >= k)


def smooth(p, m):
    return pd.Series(p).rolling(m, min_periods=1).mean().values


POST = [("none", 1, 1)] + [(f"{k}of{m}", k, m) for k, m in
                           [(2, 2), (2, 3), (3, 3), (2, 4), (3, 4), (4, 4), (3, 5), (4, 5)]]
SMOOTH = [1, 2, 3, 4]

rows, store, selrows = [], [], []
for fi, (itr, ite) in enumerate(FOLDS):
    cut = int(len(itr) * (1 - CALF))
    TR, CAs, TE = d.iloc[itr[:cut]].copy(), d.iloc[itr[cut:]].copy(), d.iloc[ite].copy()
    rate, p0 = tod_rate(TR)
    for z in (TR, CAs, TE):
        z["tod_peak_rate"] = z.tod.map(rate).fillna(p0)
    # 회귀/분위 마진 (TRAIN 적합)
    Xb_tr = TR[BASE + EXTRA].fillna(-1).values
    yreg = TR[f"y_h{H}"].values - TR.kw.values
    reg = HistGradientBoostingRegressor(random_state=SEED).fit(Xb_tr, yreg)
    q90 = HistGradientBoostingRegressor(loss="quantile", quantile=.90,
                                        random_state=SEED).fit(Xb_tr, yreg)
    q75 = HistGradientBoostingRegressor(loss="quantile", quantile=.75,
                                        random_state=SEED).fit(Xb_tr, yreg)
    for z in (TR, CAs, TE):
        Xb = z[BASE + EXTRA].fillna(-1).values
        z["pred_margin"] = reg.predict(Xb) + z.kw.values - z.thr_adaptive.values
        z["q90_margin"] = q90.predict(Xb) + z.kw.values - z.thr_adaptive.values
        z["q75_margin"] = q75.predict(Xb) + z.kw.values - z.thr_adaptive.values

    ytr = (TR[f"y_h{H}"].values >= TR.thr_adaptive.values)
    yca = (CAs[f"y_h{H}"].values >= CAs.thr_adaptive.values)
    yte = (TE[f"y_h{H}"].values >= TE.thr_adaptive.values)

    SETS = {"base(기존 43특성)": BASE, "base+margin(신규)": FEAT}
    MODELS = {
        "RandomForest": lambda: RandomForestClassifier(n_estimators=500, min_samples_leaf=2,
                                                       random_state=SEED, n_jobs=-1),
        "ExtraTrees": lambda: ExtraTreesClassifier(n_estimators=500, min_samples_leaf=2,
                                                   random_state=SEED, n_jobs=-1),
        "HistGBM": lambda: HistGradientBoostingClassifier(random_state=SEED),
    }
    for sname, F in SETS.items():
        Xtr, Xca, Xte = (z[F].fillna(-1).values for z in (TR, CAs, TE))
        probs = {}
        for mname, mk in MODELS.items():
            m = mk().fit(Xtr, ytr)
            probs[mname] = (m.predict_proba(Xca)[:, 1], m.predict_proba(Xte)[:, 1])
        probs["SoftVote(RF+ET+HGB)"] = (np.mean([probs[k][0] for k in MODELS], 0),
                                        np.mean([probs[k][1] for k in MODELS], 0))
        # CAL isotonic 보정: 폴드마다 점수 분포가 달라 절대 임계가 전이되지 않는 문제를 없앤다.
        # 보정은 CAL 라벨만 쓴다(TEST 미사용).
        probs = {k: (lambda iso, a, b: (iso.predict(a), iso.predict(b)))(
                    IsotonicRegression(out_of_bounds="clip").fit(v[0], yca), v[0], v[1])
                 for k, v in probs.items()}
        for mname, (pc, pt) in probs.items():
            # CAL 에서 (평활 m, delay-timer k/m, 확률 임계) 를 함께 고른다
            best = None
            for sm in SMOOTH:
                pcs, pts = smooth(pc, sm), smooth(pt, sm)
                grid = np.unique(np.quantile(pcs, np.linspace(0.70, 0.9995, 150)))
                for pname, k, m in POST:
                    for t in grid:
                        raw = pcs >= t
                        pr = k_of_m(raw, k, m) if m > 1 else raw
                        _, _, _, _, prec, rec, f1 = confusion(yca, pr)
                        if best is None or f1 > best[0]:
                            best = (f1, sm, pname, k, m, float(t))
            _, sm, pname, k, m, t = best
            pts = smooth(pt, sm)
            raw = pts >= t
            pr = k_of_m(raw, k, m) if m > 1 else raw
            tp, fp, fn, tn, prec, rec, f1 = confusion(yte, pr)
            # 진단용 상한: TEST 라벨로 임계를 사후 최적화했을 때의 F1 (선택에는 쓰지 않는다)
            f1_oracle = max(confusion(yte, pts >= tt)[6]
                            for tt in np.quantile(pts, np.linspace(0.70, 0.999, 80)))
            rows.append(dict(fold=fi, feature_set=sname, model=mname, n=len(yte),
                             smooth_m=sm, post=pname, prob_cut=t,
                             precision=prec, recall=rec, f1=f1,
                             mcc=float(matthews_corrcoef(yte, pr)) if pr.any() and (~pr).any() else 0.0,
                             pr_auc=float(average_precision_score(yte, pt)),
                             roc_auc=float(roc_auc_score(yte, pt)),
                             brier=float(brier_score_loss(yte, np.clip(pt, 0, 1))),
                             alerts_per_day=float(pr.mean() * 96),
                             f1_oracle_threshold=float(f1_oracle),
                             delay_steps=(m - 1), residual_lead_min=HM - (m - 1) * 15,
                             tp=tp, fp=fp, fn=fn, tn=tn))
            store.append(pd.DataFrame(dict(ts15=TE.ts15.values, fold=fi, feature_set=sname,
                                           model=mname, score=pt, alert=pr, label=yte,
                                           clean=TE.clean.values)))
    print(f"  fold{fi} done ({time.time()-t0:.0f}s)")

R = pd.DataFrame(rows); save_table(R, "35_alert_upgrade_rows")
S = pd.concat(store, ignore_index=True)
cl = S[S.clean]
out = []
for (fs, mn), g in cl.groupby(["feature_set", "model"]):
    tp, fp, fn, tn, prec, rec, f1 = confusion(g.label.values, g.alert.values)
    sub = R[(R.feature_set == fs) & (R.model == mn)]
    out.append(dict(window="clean_Jul_Sep", feature_set=fs, model=mn, n=len(g),
                    prevalence=float(g.label.mean()), precision=prec, recall=rec, f1=f1,
                    mcc=float(matthews_corrcoef(g.label.values, g.alert.values)),
                    pr_auc=float(average_precision_score(g.label, g.score)),
                    roc_auc=float(roc_auc_score(g.label, g.score)),
                    alerts_per_day=float(g.alert.mean() * 96),
                    mean_residual_lead_min=float(sub.residual_lead_min.mean()),
                    post_mode=sub.post.mode().iloc[0], smooth_mode=int(sub.smooth_m.mode().iloc[0]),
                    tp=tp, fp=fp, fn=fn, tn=tn))
C = pd.DataFrame(out).sort_values("f1", ascending=False)
save_table(C, "35_alert_upgrade_clean")
print(C.round(4).to_string(index=False))
S.to_csv(ROOT / "analysis/tables/35_alert_upgrade_rows_detail.csv", index=False,
         encoding="utf-8-sig")
print(f"[done] {time.time()-t0:.0f}s")
