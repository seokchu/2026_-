"""07-N. 외부 공공데이터 스트레스 테스트 (Task N).

질문: "현실적으로 조달 가능한 외부 공공데이터를 넣으면 LOW/OOD/전환/난조건 문제가 실제로 사라지는가?"
  E0 = KAMP 현재 최선 정보집합 (L3)
  E1 = E0 + 공휴일/근무일 달력 (holidays 패키지)
  E2 = E1 + 외부 기상 관측 (NOAA ISD, 추정 최근접 관측소)
  E3 = E2 + 태양기하 + 요금 시간대 구조
  E4 = 타 공장 공개데이터 전이표현 — 미수행(이유는 보고서 18에 기록)

규칙: 정보수준 간 공정 비교를 위해 모든 수준의 특성이 결측 없는 공통 행집합만 쓴다.
      피크 라벨은 프로젝트 표준(직전 30일 p95, 적응형·leak-free).
      모든 임계·보정분위·밴드컷·OOD 컷은 각 폴드의 과거 구간에서만 적합한다.
"""
import numpy as np, pandas as pd
import _fe
from _fe import build, save_table, fig_path, mpl, SEED, TAB
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.covariance import EmpiricalCovariance
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import average_precision_score, roc_auc_score
plt = mpl()

CALX = ["is_holiday", "is_workday", "is_bridge_day", "days_to_next_nonworkday",
        "days_since_prev_nonworkday", "nonwork_run_len"]
WEAX = ["ext_temp", "ext_dewpoint", "ext_pressure", "ext_wind", "ext_visibility",
        "ext_temp_diff1h", "ext_temp_roll24h", "ext_dew_spread", "ext_heat_index_proxy",
        "ext_pressure_diff3h", "ext_prev_day_tmax", "ext_prev_day_tmin"]
TARX = ["solar_elevation", "is_daylight", "day_length_h", "tariff_season_code", "tariff_load_zone_code"]
E0 = _fe.LEVELS["L3_+production"]
LEVELS = {"E0_kamp_only": E0, "E1_+holiday_calendar": E0 + CALX,
          "E2_+external_weather": E0 + CALX + WEAX, "E3_+solar_tariff": E0 + CALX + WEAX + TARX}


def build_ext():
    q = build()
    cal = pd.read_csv(TAB / "07N_external_calendar_2021.csv", parse_dates=["date"], encoding="utf-8-sig")
    q["date_only"] = q.ts15.dt.normalize()
    q = q.merge(cal[["date"] + CALX], left_on="date_only", right_on="date", how="left")
    w = pd.read_csv(TAB / "07N_external_weather_hourly_filled.csv", parse_dates=["ts"], encoding="utf-8-sig")
    # 15분 시점 t 에는 t 이하의 마지막 정시 관측만 쓸 수 있다.
    q["ts_hour"] = q.ts15.dt.floor("h")
    q = q.merge(w[["ts"] + WEAX], left_on="ts_hour", right_on="ts", how="left")
    s = pd.read_csv(TAB / "07N_external_solar_tariff_2021.csv", parse_dates=["ts15"], encoding="utf-8-sig")
    s["tariff_season_code"] = s.tariff_season.map({"winter": 0, "spring_fall": 1, "summer": 2})
    s["tariff_load_zone_code"] = s.tariff_load_zone.map({"light": 0, "mid": 1, "peak": 2})
    q = q.merge(s[["ts15"] + TARX], on="ts15", how="left")
    # 표준 피크 라벨: 직전 30일 p95 (현재값 제외)
    q["thr_adaptive"] = q.kw.shift(1).rolling(96 * 30, min_periods=96 * 7).quantile(.95)
    return q


def qr(a):
    return HistGradientBoostingRegressor(loss="quantile", quantile=a, random_state=SEED)


def label_regime(df, ramp_cut):
    sw = df.is_operating.values != df.is_operating_lag4.values
    tr_ = sw | (df.kw_ramp4.abs().values >= ramp_cut)
    return np.where(tr_, "transition", np.where(df.is_operating.values == 0, "stop", "normal_run"))


def event_metrics(peak_now, alert):
    idx = np.flatnonzero(peak_now == 1)
    if not len(idx):
        return np.nan, np.nan, 0
    ev, start, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - prev <= 2:
            prev = i; continue
        ev.append(start); start, prev = i, i
    ev.append(start)
    det, leads = 0, []
    for s in ev:
        w = alert[max(0, s - 4):s + 1]
        if w.any():
            det += 1
            leads.append((s - (max(0, s - 4) + int(np.flatnonzero(w)[0]))) * 15)
    return det / len(ev), (float(np.mean(leads)) if leads else np.nan), len(ev)


q = build_ext()
ALLF = sorted(set().union(*LEVELS.values()))
need = ALLF + ["thr_adaptive", "is_operating_lag4"] + [f"y_h{h}" for h in _fe.HORIZONS]
d = q.dropna(subset=need).reset_index(drop=True)
print("공통 행집합", len(d), "/ 원본", len(q))
folds = list(TimeSeriesSplit(n_splits=5).split(d))
RAMP_CUT = float(np.quantile(d.iloc[folds[0][0]].kw_ramp4.abs(), .9))

rows, oof_store, err_store = [], {}, {}
for lname, F in LEVELS.items():
    for h in _fe.HORIZONS:
        y = d[f"y_h{h}"].values
        p = np.full(len(d), np.nan); pr = np.full(len(d), np.nan)
        fid = np.full(len(d), -1); ood = np.full(len(d), np.nan)
        for fi, (itr, ite) in enumerate(folds):
            fid[ite] = fi
            p[ite] = HistGradientBoostingRegressor(random_state=SEED).fit(
                d.iloc[itr][F].values, y[itr]).predict(d.iloc[ite][F].values)
            pr[ite] = make_pipeline(StandardScaler(), Ridge(1.0)).fit(
                d.iloc[itr][F].values, y[itr]).predict(d.iloc[ite][F].values)
            cv = EmpiricalCovariance().fit(d.iloc[itr][F].values)
            ood[ite] = cv.mahalanobis(d.iloc[ite][F].values)
        m = fid >= 0
        o = d[m].copy()
        o["y_true"], o["pred"], o["pred_ridge"], o["fold"], o["ood_maha"] = y[m], p[m], pr[m], fid[m], ood[m]
        o["abs_err"] = np.abs(o.y_true - o.pred)
        o["disagreement"] = np.abs(o.pred - o.pred_ridge)
        o["is_peak"] = (o.y_true >= o.thr_adaptive).astype(int)
        o["peak_now"] = (o.kw >= o.thr_adaptive).astype(int)
        o["regime"] = label_regime(o, RAMP_CUT)
        oof_store[(lname, h)] = o
        for wtag, s in [("oof_all_2021", o), ("oof_clean_Jul_Sep", o[o.clean])]:
            s = s.reset_index(drop=True)
            err_store[(wtag, h, lname)] = s.abs_err.values
            thrq = float(np.quantile(s.pred, 1 - 10 / 96))        # 일 10건 예산
            er, lead, nev = event_metrics(s.peak_now.values, (s.pred >= thrq).values)
            rows.append(dict(window=wtag, level=lname, horizon_min=h * 15, n=len(s),
                             mae=float(s.abs_err.mean()),
                             rmse=float(np.sqrt((s.abs_err ** 2).mean())),
                             peak_prevalence=float(s.is_peak.mean()),
                             peak_ap=float(average_precision_score(s.is_peak, s.pred)),
                             peak_auc=float(roc_auc_score(s.is_peak, s.pred)),
                             event_recall_at_10_per_day=er, mean_lead_min=lead, n_events=nev,
                             mae_transition=float(s[s.regime == "transition"].abs_err.mean()),
                             mae_normal_run=float(s[s.regime == "normal_run"].abs_err.mean()),
                             mae_stop=float(s[s.regime == "stop"].abs_err.mean()),
                             transition_peak_prevalence=float(s[s.regime == "transition"].is_peak.mean())))
res = pd.DataFrame(rows)
b = res[res.level == "E0_kamp_only"].set_index(["window", "horizon_min"])
for c in ["mae", "peak_ap", "event_recall_at_10_per_day", "mae_transition"]:
    res[f"d_{c}_vs_E0"] = [r[c] - b.loc[(r.window, r.horizon_min), c] for _, r in res.iterrows()]
save_table(res, "07N_external_ablation_forecast")

# E0 대비 paired bootstrap (공통 행집합이므로 행 대응이 보장된다)
boot = [dict(window=w, horizon_min=h * 15, level=l, vs="E0_kamp_only",
             **_fe.paired_boot(e, err_store[(w, h, "E0_kamp_only")]))
        for (w, h, l), e in err_store.items() if l != "E0_kamp_only"]
save_table(pd.DataFrame(boot), "07N_external_ablation_bootstrap")
print(pd.DataFrame(boot)[lambda x: x.window == "oof_clean_Jul_Sep"].to_string(index=False))

# ---- 신뢰도 / OOD / 밴드 (h=60분)
brows, drows, hrows = [], [], []
for lname, F in LEVELS.items():
    o = oof_store[(lname, 4)].copy()
    o["int_lo"] = np.nan; o["int_hi"] = np.nan; o["band"] = ""
    for f in sorted(o.fold.unique()):
        tr, te = o[o.fold < f], o[o.fold == f]
        if len(tr) < 800:
            continue
        cut = int(len(tr) * .8)
        a, c = tr.iloc[:cut], tr.iloc[cut:]
        m_lo, m_hi = qr(.05).fit(a[F].values, a.y_true.values), qr(.95).fit(a[F].values, a.y_true.values)
        E = np.maximum(m_lo.predict(c[F].values) - c.y_true.values,
                       c.y_true.values - m_hi.predict(c[F].values))
        Q = float(np.quantile(E, min(1.0, .9 * (1 + 1 / len(E)))))
        lo, hi = m_lo.predict(te[F].values) - Q, m_hi.predict(te[F].values) + Q
        o.loc[te.index, "int_lo"], o.loc[te.index, "int_hi"] = lo, hi
        wtr = (m_hi.predict(c[F].values) + Q) - (m_lo.predict(c[F].values) - Q)
        cuts = np.quantile(wtr, [.5, .8])
        ood_cut = float(np.quantile(tr.ood_maha, .99))
        w_te = hi - lo
        bd = np.where(te.ood_maha.values > ood_cut, "OOD",
                      np.where(w_te <= cuts[0], "HIGH", np.where(w_te <= cuts[1], "MEDIUM", "LOW")))
        o.loc[te.index, "band"] = bd
    o = o[o.band != ""].reset_index(drop=True)
    o["int_width"] = o.int_hi - o.int_lo
    o["covered"] = ((o.y_true >= o.int_lo) & (o.y_true <= o.int_hi)).astype(int)
    oof_store[("banded", lname)] = o
    for wtag, s in [("oof_all_2021", o), ("oof_clean_Jul_Sep", o[o.clean])]:
        for bnd in ["HIGH", "MEDIUM", "LOW", "OOD"]:
            t = s[s.band == bnd]
            if not len(t):
                continue
            brows.append(dict(window=wtag, level=lname, band=bnd, n=len(t), share=len(t) / len(s),
                              mae=float(t.abs_err.mean()), coverage=float(t.covered.mean()),
                              mean_interval_width=float(t.int_width.mean()),
                              peak_prevalence=float(t.is_peak.mean()),
                              transition_share=float((t.regime == "transition").mean())))
        pk = s[s.is_peak == 1]
        drows.append(dict(window=wtag, level=lname, n=len(s),
                          low_share=float((s.band == "LOW").mean()),
                          ood_share=float((s.band == "OOD").mean()),
                          low_or_ood_share=float(s.band.isin(["LOW", "OOD"]).mean()),
                          coverage_all=float(s.covered.mean()),
                          coverage_peak_only=float(pk.covered.mean()) if len(pk) else np.nan,
                          mean_width=float(s.int_width.mean()),
                          transition_low_or_ood=float(s[s.regime == "transition"].band
                                                      .isin(["LOW", "OOD"]).mean()),
                          transition_mae=float(s[s.regime == "transition"].abs_err.mean())))
    # ---- 난조건: 같은 기준으로 재발견 (tree) + 고정 참조조건
    s = o.copy()
    CTX = ["is_operating", "생산량", "prod_diff", "시간", "tod", "day", "kw", "kw_ramp4",
           "kw_std4", "kw_std96", "기온", "습도", "disagreement", "ood_maha"]
    s["abs_ramp"] = s.kw_ramp4.abs()
    ov = s.abs_err.mean()
    tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=max(200, int(len(s) * .02)),
                                 random_state=SEED).fit(s[CTX].values, s.abs_err.values)
    leaf = tree.apply(s[CTX].values)
    npro = 0
    for lf in np.unique(leaf):
        m = leaf == lf
        if m.mean() < .01:
            continue
        fr = s[m].groupby("fold").abs_err.mean() / s.groupby("fold").abs_err.mean()
        cl = s[m & s.clean.values]
        if (s[m].abs_err.mean() / ov >= 1.3) and (fr > 1.3).sum() >= 4 and len(cl) and \
           (cl.abs_err.mean() / s[s.clean].abs_err.mean() >= 1.2):
            npro += 1
    refs = {"disagreement_top20": s.disagreement >= np.quantile(s.disagreement, .8),
            "disagreement_top20_and_idle": (s.disagreement >= np.quantile(s.disagreement, .8)) & (s.is_operating == 0),
            "abs_ramp_top20": s.abs_ramp >= np.quantile(s.abs_ramp, .8),
            "regime_transition": s.regime == "transition"}
    for rname, m in refs.items():
        m = np.asarray(m)
        hrows.append(dict(level=lname, condition=rname, n=int(m.sum()), share=float(m.mean()),
                          mae=float(s[m].abs_err.mean()), mae_ratio_vs_overall=float(s[m].abs_err.mean() / ov),
                          low_or_ood_share=float(s[m].band.isin(["LOW", "OOD"]).mean()),
                          peak_prevalence=float(s[m].is_peak.mean()),
                          n_tree_promoted_conditions=npro))
save_table(pd.DataFrame(brows), "07N_external_reliability_bands")
save_table(pd.DataFrame(drows), "07N_external_reliability_summary")
save_table(pd.DataFrame(hrows), "07N_external_hard_conditions")

# ---- 판정 규칙 (사전 명시)
dd = pd.DataFrame(drows)
cl = dd[dd.window == "oof_clean_Jul_Sep"].set_index("level")
f = res[(res.window == "oof_clean_Jul_Sep") & (res.horizon_min == 60)].set_index("level")
e0, e3 = "E0_kamp_only", "E3_+solar_tariff"
mae_gain = (f.loc[e0, "mae"] - f.loc[e3, "mae"]) / f.loc[e0, "mae"]
low_drop = (cl.loc[e0, "low_or_ood_share"] - cl.loc[e3, "low_or_ood_share"]) / cl.loc[e0, "low_or_ood_share"]
pk_gain = cl.loc[e3, "coverage_peak_only"] - cl.loc[e0, "coverage_peak_only"]
tr_gain = (cl.loc[e0, "transition_mae"] - cl.loc[e3, "transition_mae"]) / cl.loc[e0, "transition_mae"]
hc = pd.DataFrame(hrows).set_index(["level", "condition"])
hc_gone = int(hc.loc[(e3, "disagreement_top20"), "n_tree_promoted_conditions"] == 0)
case = ("CASE1_외부데이터가 문제를 해결" if (low_drop > .5 and pk_gain > .05 and tr_gain > .2)
        else "CASE2_부분 개선, 신뢰도/난조건 문제 잔존" if (mae_gain > .02 or low_drop > .1 or tr_gain > .05)
        else "CASE3_기여 거의 없음 — 경량 FEMS 방향 강화")
save_table(pd.DataFrame([dict(
    decision=case, mae_gain_pct_h60=float(mae_gain), low_or_ood_reduction_pct=float(low_drop),
    peak_coverage_gain=float(pk_gain), transition_mae_reduction_pct=float(tr_gain),
    tree_promoted_conditions_E0=int(hc.loc[(e0, "disagreement_top20"), "n_tree_promoted_conditions"]),
    tree_promoted_conditions_E3=int(hc.loc[(e3, "disagreement_top20"), "n_tree_promoted_conditions"]),
    hard_conditions_gone=hc_gone,
    rule="CASE1: LOW/OOD -50% 이상 & 피크 coverage +0.05 이상 & 전환 MAE -20% 이상 / "
         "CASE2: MAE -2% 또는 LOW/OOD -10% 또는 전환 MAE -5% / CASE3: 그 미만")]),
    "07N_external_decision")
print(res[(res.window == "oof_clean_Jul_Sep")][["level", "horizon_min", "mae", "peak_ap",
                                               "event_recall_at_10_per_day", "mae_transition"]].to_string(index=False))
print(dd.to_string(index=False))
print(pd.DataFrame(hrows).to_string(index=False))
print("판정:", case)

fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
c = res[(res.window == "oof_clean_Jul_Sep")]
for lv in LEVELS:
    s = c[c.level == lv].sort_values("horizon_min")
    axes[0].plot(s.horizon_min, s.mae, "o-", label=lv)
    axes[1].plot(s.horizon_min, s.peak_ap, "o-", label=lv)
axes[0].set_xlabel("horizon(분)"); axes[0].set_ylabel("MAE (kw)"); axes[0].set_title("외부데이터 수준별 MAE")
axes[1].set_xlabel("horizon(분)"); axes[1].set_ylabel("피크 AP"); axes[1].set_title("피크 선별력")
x = np.arange(len(LEVELS))
axes[2].bar(x - .2, [cl.loc[l, "low_or_ood_share"] for l in LEVELS], .4, label="LOW+OOD 비중")
axes[2].bar(x + .2, [cl.loc[l, "coverage_peak_only"] for l in LEVELS], .4, label="피크 coverage")
axes[2].set_xticks(x); axes[2].set_xticklabels(list(LEVELS), rotation=20, ha="right", fontsize=6)
axes[2].set_title("신뢰도 지표 (원본구간)")
for ax in axes:
    ax.legend(fontsize=6)
fig.tight_layout(); fig.savefig(fig_path("07N_external_data_stress_test")); plt.close(fig)
