"""30-A. 운전 상태별 개별 예측모델이 전체 모델보다 유리한가 — 재검증(h15/h30/h45/h60).

배경: 07M(1단계)은 h60 단독·구간 p95 피크정의에서 '3상태 전용 모델은 global 대비 +0.023 kw,
      비유의' 로 보고했다. 그 결론이 현재 최종 모델·분할·4개 horizon 에도 유지되는지 확인한다.

비교 (동일 입력·동일 타깃·동일 시간순 분할):
  V1 global                : 전체 데이터 1개 모델
  V2 global_plus_state     : 전체 데이터 + 운전상태를 입력 특성으로 명시(서수 0/1/2)
  V3 state2_specific       : 조업/비조업 2상태 전용 모델
  V4 state3_specific       : LOW_LOAD / STABLE_OPERATION / TRANSITION_LIKE 3상태 전용 모델

규칙:
  - 상태 라벨은 **결정시점 t 정보만** 사용(미래 조업여부·실측 사용량 금지). ramp 컷은 폴드 TRAIN p90.
  - 상태별 모델은 해당 상태 **행만 골라 학습**한다. 시계열을 이어 붙이지 않는다 —
    특성(lag/rolling)은 원본 연속 시계열에서 미리 계산된 값이며, 부분집합 선택이 그 값을 바꾸지 않는다.
  - 라우팅은 TEST 행의 결정시점 상태로만 결정한다.
  - 사전 규칙: 상태의 학습표본 < MIN_TRAIN(500) 이거나 TEST 에 그 상태가 없으면 **global 예측으로 대체**.
  - 가정 민감도: assumption='plan_known'(생산량(t)·is_operating(t) 사용) vs
                 'strict_lag'(생산 정보는 1시간 지연만, 상태도 is_operating_lag4 로 정의).
채택 규칙(사전 선언, 결과 보기 전 고정) — ADOPT_RULE 문자열에 기록.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, yaml
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "analysis" / "07_sparse_fems"))
from common import save_table                                            # noqa: E402
import features as FT, models as MD, _fe                                 # noqa: E402
from sklearn.model_selection import TimeSeriesSplit                      # noqa: E402
from sklearn.cluster import KMeans                                       # noqa: E402
from sklearn.preprocessing import StandardScaler                         # noqa: E402
from sklearn.metrics import silhouette_score, adjusted_rand_score        # noqa: E402

cfg = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
SEED, H = cfg["seed"], [1, 2, 3, 4]
MIN_TRAIN = 500
LOCK_DAYS = 14
ADOPT_RULE = ("사전 선언: 상태전용 모델은 (a) 원본구간 4개 horizon 전부에서 전체 MAE 개선 AND "
              "(b) h15·h60 일블록 bootstrap 95% CI 상한 < 0 AND (c) 어떤 상태의 MAE 도 5% 초과 악화 없음 "
              "AND (d) 고부하·급변 구간 MAE 악화 없음 — 4개 모두 충족할 때만 채택")

ASSUMP = {"plan_known": (_fe.LEVELS["L3_+production"], "is_operating"),
          "strict_lag": (_fe.LEVELS["L3s_+production_strict"], "is_operating_lag4")}

q, mode = FT.build(cfg["external"]["use_external_data"], cfg["peak"]["window_days"],
                   cfg["peak"]["quantile"], cfg["peak"]["min_days"])
d = FT.common_rows(q, mode)
d = d.dropna(subset=["is_operating_lag4"]).reset_index(drop=True)
folds = list(TimeSeriesSplit(n_splits=cfg["data"]["n_splits"]).split(d))
clean_end = d[d.clean]["ts15"].max()
lock_start = clean_end - pd.Timedelta(days=LOCK_DAYS)
print("행", len(d), "| clean", int(d.clean.sum()), "| LOCK 시작", lock_start)


def state3(df, ramp_cut, opcol):
    """결정시점 정보만. 조업여부 1시간내 변화 or |ramp| >= TRAIN p90 -> TRANSITION_LIKE."""
    sw = df[opcol].values != df.is_operating_lag4.values
    tr_ = sw | (df.kw_ramp4.abs().values >= ramp_cut)
    return np.where(tr_, "TRANSITION_LIKE",
                    np.where(df[opcol].values == 0, "LOW_LOAD", "STABLE_OPERATION"))


rows, errs, fb_rows, row_store = [], {}, [], []
for aname, (F, opcol) in ASSUMP.items():
    for h in H:
        ycol = f"y_h{h}"
        pred = {v: np.full(len(d), np.nan) for v in
                ["V1_global", "V2_global_plus_state", "V3_state2_specific", "V4_state3_specific"]}
        st3 = np.array([""] * len(d), dtype=object)
        st2 = np.where(d[opcol].values == 1, "operating", "idle")
        for fi, (itr, ite) in enumerate(folds):
            cut = int(len(itr) * (1 - cfg["data"]["cal_fraction"]))
            TR = d.iloc[itr[:cut]]
            ramp_cut = float(np.quantile(TR.kw_ramp4.abs(), cfg["regime"]["ramp_quantile"]))
            s3_all = state3(d, ramp_cut, opcol)
            st3[ite] = s3_all[ite]
            fit_idx, te = np.r_[itr], ite                 # TRAIN+CAL 로 적합(TEST 미사용)
            Xf, yf = d.iloc[fit_idx][F].values, d.iloc[fit_idx][ycol].values
            Xt = d.iloc[te][F].values
            g = MD.point_model("hgb", SEED).fit(Xf, yf)
            pred["V1_global"][te] = g.predict(Xt)
            # V2: 상태를 입력 특성으로
            code = {"LOW_LOAD": 0, "STABLE_OPERATION": 1, "TRANSITION_LIKE": 2}
            Xf2 = np.c_[Xf, [code[s] for s in s3_all[fit_idx]]]
            Xt2 = np.c_[Xt, [code[s] for s in s3_all[te]]]
            pred["V2_global_plus_state"][te] = MD.point_model("hgb", SEED).fit(Xf2, yf).predict(Xt2)
            # V3 / V4: 상태 전용 모델 + 사전 규칙 fallback
            for vname, labels in [("V3_state2_specific", st2), ("V4_state3_specific", s3_all)]:
                out = pred["V1_global"][te].copy()         # 기본값 = global (fallback)
                for st in np.unique(labels[fit_idx]):
                    m_fit = labels[fit_idx] == st
                    m_te = labels[te] == st
                    if m_fit.sum() < MIN_TRAIN or m_te.sum() == 0:
                        fb_rows.append(dict(assumption=aname, horizon_min=h * 15, fold=fi,
                                            variant=vname, state=st, n_train=int(m_fit.sum()),
                                            n_test=int(m_te.sum()), action="fallback_to_global"))
                        continue
                    mdl = MD.point_model("hgb", SEED).fit(Xf[m_fit], yf[m_fit])
                    out[m_te] = mdl.predict(Xt[m_te])
                pred[vname][te] = out
        m = ~np.isnan(pred["V1_global"])
        o = d[m].copy()
        o["state3"], o["state2"] = st3[m], st2[m]
        for v, p in pred.items():
            o[v] = p[m]
            o[f"err_{v}"] = (o[ycol] - o[v]).abs()
        o["date"] = o.ts15.dt.normalize()
        # 고부하·급변 컷은 첫 폴드 학습구간에서만 산출
        pre = d.iloc[folds[0][0]]
        LOAD_HI = float(np.quantile(pre.kw, .8)); RAMP_HI = float(np.quantile(pre.kw_ramp4.abs(), .8))
        wins = [("all_2021", np.ones(len(o), bool)), ("clean_Jul_Sep", o.clean.values),
                ("clean_LOCK_last14d", (o.clean & (o.ts15 >= lock_start)).values)]
        keep = ["ts15", "clean", "date", "state3", "state2", "kw", "kw_ramp4", "thr_adaptive", ycol] + \
               [c for c in o.columns if c.startswith(("V", "err_"))]
        row_store.append(o[keep].assign(assumption=aname, horizon_min=h * 15))
        for wtag, wm in wins:
            s = o[wm]
            if not len(s):
                continue
            errs[(aname, h, wtag)] = s
            for v in pred:
                e = s[f"err_{v}"].values
                hi_load = s.kw.values >= LOAD_HI
                hi_ramp = s.kw_ramp4.abs().values >= RAMP_HI
                peak = (s[ycol].values >= s.thr_adaptive.values)
                rec = dict(assumption=aname, window=wtag, horizon_min=h * 15, variant=v, n=len(s),
                           mae=float(e.mean()), rmse=float(np.sqrt((e ** 2).mean())),
                           err_p90=float(np.quantile(e, .9)), err_p99=float(np.quantile(e, .99)),
                           err_max=float(e.max()),
                           mae_high_load_top20=float(e[hi_load].mean()),
                           mae_high_ramp_top20=float(e[hi_ramp].mean()),
                           mae_peak_condition=float(e[peak].mean()) if peak.any() else np.nan,
                           load_cut=LOAD_HI, ramp_cut=RAMP_HI)
                for st in ["LOW_LOAD", "STABLE_OPERATION", "TRANSITION_LIKE"]:
                    ms = s.state3.values == st
                    rec[f"mae_{st}"] = float(e[ms].mean()) if ms.any() else np.nan
                    rec[f"n_{st}"] = int(ms.sum())
                rows.append(rec)
save_table(pd.concat(row_store, ignore_index=True), "30_state_model_rows")
res = pd.DataFrame(rows)
b = res[res.variant == "V1_global"].set_index(["assumption", "window", "horizon_min"])
for c in ["mae", "mae_high_load_top20", "mae_high_ramp_top20", "mae_TRANSITION_LIKE"]:
    res[f"d_{c}_vs_global"] = [r[c] - b.loc[(r.assumption, r.window, r.horizon_min), c]
                              for _, r in res.iterrows()]
save_table(res, "30_state_model_comparison")
save_table(pd.DataFrame(fb_rows) if fb_rows else pd.DataFrame([dict(note="fallback 발생 없음")]),
           "30_state_model_fallbacks")

# ---- 일(day) 블록 paired bootstrap: 같은 시점끼리 절대오차 차이
rng = np.random.default_rng(SEED)
boot = []
for (aname, h, wtag), s in errs.items():
    if wtag == "all_2021":
        continue
    codes, uniq = pd.factorize(s.date.values)
    groups = [np.flatnonzero(codes == i) for i in range(len(uniq))]
    for v in ["V2_global_plus_state", "V3_state2_specific", "V4_state3_specific"]:
        diff = (s[f"err_{v}"].values - s["err_V1_global"].values)
        draws = []
        for _ in range(2000):
            pick = rng.integers(0, len(groups), size=len(groups))
            ii = np.concatenate([groups[j] for j in pick])
            draws.append(diff[ii].mean())
        draws = np.asarray(draws)
        boot.append(dict(assumption=aname, window=wtag, horizon_min=h * 15, variant=v,
                         vs="V1_global", delta_mae=float(diff.mean()),
                         ci_lo=float(np.quantile(draws, .025)), ci_hi=float(np.quantile(draws, .975)),
                         significant=bool(np.quantile(draws, .975) < 0 or np.quantile(draws, .025) > 0),
                         improves=bool(np.quantile(draws, .975) < 0),
                         n_days=len(uniq), n_rows=len(s), method="일 블록 paired bootstrap 2000회"))
bt = pd.DataFrame(boot)
save_table(bt, "30_state_model_bootstrap")

# ---- '3개 군집이 자연적으로 존재하는가' — 규칙 라벨과 분리해 확인
pre = d.iloc[folds[0][0]]
FC = ["kw", "kw_ramp4", "kw_std4", "kw_roll4"]
sc = StandardScaler().fit(pre[FC].values)
Z = sc.transform(pre[FC].values)
sub = Z[rng.choice(len(Z), size=min(4000, len(Z)), replace=False)]
ramp_cut0 = float(np.quantile(pre.kw_ramp4.abs(), cfg["regime"]["ramp_quantile"]))
rule_pre = state3(pre, ramp_cut0, "is_operating")
krows = []
for k in range(2, 7):
    km = KMeans(k, n_init=10, random_state=SEED).fit(Z)
    krows.append(dict(k=k, inertia=float(km.inertia_),
                      silhouette=float(silhouette_score(sub, km.predict(sub))),
                      ari_vs_rule3=float(adjusted_rand_score(rule_pre, km.labels_)),
                      largest_cluster_share=float(pd.Series(km.labels_).value_counts(normalize=True).max())))
kk = pd.DataFrame(krows)
kk["is_best_silhouette"] = kk.silhouette == kk.silhouette.max()
save_table(kk, "30_cluster_count_check")
print(kk.round(4).to_string(index=False))

# ---- 채택 판정
cl = res[(res.window == "clean_Jul_Sep") & (res.assumption == "plan_known")]
g = cl[cl.variant == "V1_global"].set_index("horizon_min")
ver = []
for v in ["V2_global_plus_state", "V3_state2_specific", "V4_state3_specific"]:
    s = cl[cl.variant == v].set_index("horizon_min")
    a = bool((s.mae < g.mae).all())
    bb = bt[(bt.window == "clean_Jul_Sep") & (bt.assumption == "plan_known") & (bt.variant == v)]
    b_ok = bool(bb[bb.horizon_min.isin([15, 60])].improves.all() and
                bb[bb.horizon_min.isin([15, 60])].significant.all())
    c_ok = bool(all((s[f"mae_{st}"] <= g[f"mae_{st}"] * 1.05).all()
                    for st in ["LOW_LOAD", "STABLE_OPERATION", "TRANSITION_LIKE"]))
    d_ok = bool((s.mae_high_load_top20 <= g.mae_high_load_top20).all() and
                (s.mae_high_ramp_top20 <= g.mae_high_ramp_top20).all())
    ver.append(dict(variant=v, cond_a_all_horizons_better=a, cond_b_bootstrap_sig_h15_h60=b_ok,
                    cond_c_no_state_worse_5pct=c_ok, cond_d_subgroups_not_worse=d_ok,
                    adopt=bool(a and b_ok and c_ok and d_ok),
                    mae_h60=float(s.loc[60, "mae"]), mae_h60_global=float(g.loc[60, "mae"]),
                    delta_h60=float(s.loc[60, "mae"] - g.loc[60, "mae"]),
                    rule=ADOPT_RULE))
vv = pd.DataFrame(ver)
save_table(vv, "30_state_model_verdict")
print(cl[["horizon_min", "variant", "mae", "mae_high_load_top20", "mae_high_ramp_top20",
          "mae_TRANSITION_LIKE", "d_mae_vs_global"]].round(4).to_string(index=False))
print(bt[(bt.window == "clean_Jul_Sep") & (bt.assumption == "plan_known")][
    ["horizon_min", "variant", "delta_mae", "ci_lo", "ci_hi", "significant", "improves"]].round(4).to_string(index=False))
print(vv.drop(columns=["rule"]).to_string(index=False))
