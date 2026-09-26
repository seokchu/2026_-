"""05. Power / resource optimization: augmentation forensics, peak anatomy, archetypes, baselines."""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from scipy import stats, signal
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, roc_auc_score, average_precision_score
from common import load_power, power_long, save_table, fig_path, mpl, SEED
plt = mpl()
rng = np.random.default_rng(SEED)

p = load_power()
HC = ["15분", "30분", "45분", "60분"]

# ---------------- 1. augmentation forensics
sig = p.groupby("날짜")[HC].apply(lambda d: tuple(d.values.ravel()))
grp = sig.groupby(sig).ngroup().rename("profile_id")
day = p.groupby("날짜").agg(prod=("생산량", "sum"), dom=("d", "first"), month=("m", "first"),
                          dow=("day", "first"), kw_max=("60분", "max"), temp=("기온", "mean")).join(grp)
day = day.join(day.groupby("profile_id").size().rename("profile_n"), on="profile_id")
day["is_duplicated_profile"] = day.profile_n > 1
save_table(day.reset_index(), "05_power_day_profile_map")

dupsum = day.groupby("month").agg(n_days=("prod", "size"), dup_rate=("is_duplicated_profile", "mean"),
                                  unique_profiles=("profile_id", "nunique"))
save_table(dupsum.reset_index(), "05_power_duplication_by_month")

mult = day[day.profile_n > 1]
dom1 = mult.groupby("profile_id").dom.nunique()
aug = dict(total_days=len(day), unique_power_profiles=int(day.profile_id.nunique()),
           duplicated_days=int(day.is_duplicated_profile.sum()),
           multi_day_groups=int(len(dom1)),
           groups_sharing_same_day_of_month=int((dom1 == 1).sum()),
           largest_group_size=int(day.profile_n.max()),
           clean_window_start="2021-07-01", clean_window_days=int((day.index >= 20210701).sum()),
           clean_window_dup_rate=float(day[day.index >= 20210701].is_duplicated_profile.mean()))
r = mult[mult["prod"] > 0].groupby("profile_id")["prod"].agg(["size", "min", "max", "mean", "std"])
r = r[r["size"] > 1]
aug["within_identical_power_group_production_CV_median"] = float((r["std"] / r["mean"]).median())
aug["within_identical_power_group_production_CV_max"] = float((r["std"] / r["mean"]).max())
save_table(pd.DataFrame([aug]), "05_power_augmentation_forensics")

# relationship strength: augmented vs clean window
q = power_long(p)
q["clean"] = q.날짜 >= 20210701
rel = []
for nm, sub in [("augmented_Jan_Jun", q[q.날짜 < 20210701]), ("clean_Jul_Sep", q[q.clean])]:
    for c in ["생산량", "기온", "공장인원", "습도", "풍속"]:
        m = sub[[c, "kw"]].dropna()
        rel.append(dict(window=nm, feature=c, n=len(m),
                        spearman_with_kw=float(stats.spearmanr(m[c], m.kw).statistic)))
save_table(pd.DataFrame(rel), "05_power_relationship_clean_vs_augmented")

# ---------------- 2. data-quality flags
dq = [dict(issue="시간 컬럼 손상", detail="2021-07-13 / 2021-07-15 의 24행에서 시간값이 전력값으로 대체됨. 행순서로 복원",
           n_rows=int(p.hour_corrupt.sum())),
      dict(issue="전력 0 기록", detail="2021-08-28/29/09-08 의 15분 계측치 0 = 계측 결측 추정(생산량 0 아님 확인 필요)",
           n_rows=int((q.kw == 0).sum())),
      dict(issue="공factory 인원 = 생산량의 결정함수", detail=f"spearman(공장인원,생산량)={p[['공장인원','생산량']].corr(method='spearman').iloc[0,1]:.3f} → 독립 정보 아님",
           n_rows=len(p)),
      dict(issue="전기요금(계절) = 월의 결정함수", detail="3단(109.8/167.2/191.6), 월로 완전 결정 → 예측 특성으로 쓰면 월 대리변수",
           n_rows=len(p)),
      dict(issue="결측", detail=f"풍속 {int(p.풍속.isna().sum())}, 강수량 {int(p.강수량.isna().sum())}, 공장인원 {int(p.공장인원.isna().sum())}",
           n_rows=int(p[["풍속", "강수량", "공장인원"]].isna().any(axis=1).sum())),
      dict(issue="상한 포화 의심", detail=f"kw=222 가 4일에서 최댓값으로 반복 (222 초과 없음)", n_rows=int((q.kw == 222).sum()))]
save_table(pd.DataFrame(dq), "05_power_quality_flags")

# ---------------- 3. peak event definitions
q = q.sort_values("ts15").reset_index(drop=True)
s = q.kw.values
defs = {}
defs["p95_global"] = s >= np.quantile(s, .95)
defs["p99_global"] = s >= np.quantile(s, .99)
roll = pd.Series(s).rolling(96 * 7, min_periods=96).quantile(.95).bfill().values
defs["rolling7d_p95"] = s >= roll
pk, _ = signal.find_peaks(s, prominence=30)
lm = np.zeros(len(s), bool); lm[pk] = True
defs["local_max_prom30"] = lm
defs["daily_max"] = q.groupby("날짜").kw.transform("max").values == s

cmpd = []
for nm, m in defs.items():
    cmpd.append(dict(definition=nm, n_flagged=int(m.sum()), share=float(m.mean()),
                     kw_min_flagged=float(s[m].min()), kw_median_flagged=float(np.median(s[m])),
                     share_in_clean_window=float(q.clean[m].mean()),
                     share_with_production=float((q["생산량"][m] > 0).mean()),
                     n_distinct_days=int(q.날짜[m].nunique())))
save_table(pd.DataFrame(cmpd), "05_power_peak_definitions")
jac = pd.DataFrame({a: {b: float((defs[a] & defs[b]).sum() / (defs[a] | defs[b]).sum()) for b in defs} for a in defs})
save_table(jac.reset_index(), "05_power_peak_definition_jaccard", index=False)

# ---------------- 4. peak event anatomy (contiguous runs above rolling p95)
m = defs["rolling7d_p95"]
ev_id = (pd.Series(m).astype(int).diff().fillna(0) == 1).cumsum().where(pd.Series(m)).values
ev = []
base = pd.Series(s).rolling(96, min_periods=8).median().bfill().values
for e in pd.unique(ev_id[~pd.isna(ev_id)]):
    ix = np.where(ev_id == e)[0]
    a, b = ix[0], ix[-1]
    pre = s[max(0, a - 8):a]
    post = s[b + 1:b + 9]
    peak_rel = int(np.argmax(s[a:b + 1]))
    ev.append(dict(event=int(e), start=str(q.ts15.iloc[a]), duration_15min=len(ix),
                   peak_kw=float(s[a:b + 1].max()), baseline_kw=float(base[a]),
                   area_above_baseline=float((s[a:b + 1] - base[a]).clip(0).sum()),
                   rise_slope_kw_per_15min=float((s[a:b + 1].max() - (pre[-1] if len(pre) else s[a])) / max(peak_rel, 1)),
                   rise_steps=peak_rel, recovery_steps=int(len(ix) - peak_rel - 1),
                   pre_mean=float(pre.mean()) if len(pre) else np.nan,
                   post_mean=float(post.mean()) if len(post) else np.nan,
                   rebound=float(post.max() - base[a]) if len(post) else np.nan,
                   hour=int(q.시간.iloc[a + peak_rel]), dow=int(q.day.iloc[a]), month=int(q.m.iloc[a]),
                   production_hour=float(q["생산량"].iloc[a + peak_rel]),
                   temp=float(q.기온.iloc[a + peak_rel]), clean_window=bool(q.clean.iloc[a]),
                   date=int(q.날짜.iloc[a])))
ev = pd.DataFrame(ev)
save_table(ev, "05_power_peak_events")

# ---------------- 5. peak archetypes: Euclidean vs DTW on aligned peak windows
L = 12
Wm = []
meta = []
for _, e in ev.iterrows():
    i = int(q.index[q.ts15 == pd.Timestamp(e.start)][0])
    a, b = i - 4, i - 4 + L
    if a < 0 or b > len(s): continue
    w = s[a:b].astype(float)
    Wm.append((w - w.min()) / max(w.max() - w.min(), 1e-9))
    meta.append(e)
Wm = np.array(Wm); meta = pd.DataFrame(meta).reset_index(drop=True)

def dtw(a, b, band=3):
    n, m_ = len(a), len(b)
    D = np.full((n + 1, m_ + 1), np.inf); D[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(max(1, i - band), min(m_, i + band) + 1):
            D[i, j] = abs(a[i - 1] - b[j - 1]) + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return D[n, m_]

DM = np.zeros((len(Wm), len(Wm)))
for i in range(len(Wm)):
    for j in range(i + 1, len(Wm)):
        DM[i, j] = DM[j, i] = dtw(Wm[i], Wm[j])

arch = []
for tag, model, X in [("euclidean_ward", AgglomerativeClustering(n_clusters=4), Wm),
                      ("dtw_average", AgglomerativeClustering(n_clusters=4, metric="precomputed", linkage="average"), DM),
                      ("kmeans_euclid", KMeans(4, n_init=10, random_state=SEED), Wm)]:
    lab = model.fit_predict(X)
    meta[f"cl_{tag}"] = lab
    for c in np.unique(lab):
        mm = lab == c
        arch.append(dict(method=tag, cluster=int(c), n=int(mm.sum()),
                         mean_peak_kw=float(meta.peak_kw[mm].mean()),
                         mean_duration=float(meta.duration_15min[mm].mean()),
                         mean_rise_steps=float(meta.rise_steps[mm].mean()),
                         mean_rise_slope=float(meta.rise_slope_kw_per_15min[mm].mean()),
                         median_hour=float(meta.hour[mm].median()),
                         mean_production=float(meta.production_hour[mm].mean()),
                         mean_temp=float(meta.temp[mm].mean()),
                         clean_window_share=float(meta.clean_window[mm].mean())))
save_table(pd.DataFrame(arch), "05_power_peak_archetypes")
save_table(meta, "05_power_peak_event_clusters")
# silhouette-ish: agreement between metrics
from sklearn.metrics import adjusted_rand_score, silhouette_score
agree = dict(ARI_euclid_vs_dtw=float(adjusted_rand_score(meta.cl_euclidean_ward, meta.cl_dtw_average)),
             ARI_kmeans_vs_dtw=float(adjusted_rand_score(meta.cl_kmeans_euclid, meta.cl_dtw_average)),
             silhouette_euclid=float(silhouette_score(Wm, meta.cl_euclidean_ward)),
             silhouette_dtw=float(silhouette_score(DM, meta.cl_dtw_average, metric="precomputed")),
             n_events=len(meta))
save_table(pd.DataFrame([agree]), "05_power_archetype_validity")

# ---------------- 6. baselines: next-15min demand + peak classification
q["kw_lag1"] = q.kw.shift(1); q["kw_lag2"] = q.kw.shift(2); q["kw_lag4"] = q.kw.shift(4)
q["kw_lag96"] = q.kw.shift(96); q["kw_roll4"] = q.kw.shift(1).rolling(4).mean()
q["kw_roll96"] = q.kw.shift(1).rolling(96).mean()
q["prod_lag1"] = q["생산량"].shift(4)
FE = ["시간", "day", "m", "기온", "습도", "풍속", "강수량", "생산량", "공장인원",
      "kw_lag1", "kw_lag2", "kw_lag4", "kw_lag96", "kw_roll4", "kw_roll96", "prod_lag1"]
qq = q.dropna(subset=["kw_lag96", "kw_roll96"]).copy()
res = []
for wtag, sub in [("all_2021", qq), ("clean_Jul_Sep_only", qq[qq.clean])]:
    X = sub[FE].fillna(-1).values; y = sub.kw.values
    cut = int(len(sub) * .8)
    for nm, mk in [("ridge", Ridge(alpha=1.0)), ("hgb", HistGradientBoostingRegressor(random_state=SEED)),
                   ("persistence_lag1", None)]:
        if nm == "persistence_lag1":
            pred = sub.kw_lag1.values[cut:]
        else:
            pred = mk.fit(X[:cut], y[:cut]).predict(X[cut:])
        res.append(dict(window=wtag, target="kw(t)", model=nm, split="temporal 80/20",
                        mae=float(mean_absolute_error(y[cut:], pred)),
                        rmse=float(np.sqrt(((y[cut:] - pred) ** 2).mean())),
                        mae_over_std=float(mean_absolute_error(y[cut:], pred) / y[cut:].std())))
    # random split (leak demo)
    ix = rng.permutation(len(sub)); tr, te = ix[:cut], ix[cut:]
    pr = HistGradientBoostingRegressor(random_state=SEED).fit(X[tr], y[tr]).predict(X[te])
    res.append(dict(window=wtag, target="kw(t)", model="hgb_RANDOM_split(LEAK)", split="random 80/20",
                    mae=float(mean_absolute_error(y[te], pr)),
                    rmse=float(np.sqrt(((y[te] - pr) ** 2).mean())),
                    mae_over_std=float(mean_absolute_error(y[te], pr) / y[te].std())))
save_table(pd.DataFrame(res), "05_power_forecast_baseline")

# peak classification 1 hour ahead (4 steps)
qq["peak_t_plus4"] = (qq.kw.shift(-4) >= np.quantile(s, .95)).astype(float)
cl = qq.dropna(subset=["peak_t_plus4"])
X = cl[FE].fillna(-1).values; y = cl.peak_t_plus4.values.astype(int)
cut = int(len(cl) * .8)
mdl = RandomForestClassifier(n_estimators=400, min_samples_leaf=5, class_weight="balanced_subsample",
                             random_state=SEED, n_jobs=-1).fit(X[:cut], y[:cut])
pp = mdl.predict_proba(X[cut:])[:, 1]
pc = [dict(task="1시간 뒤 p95 피크 여부", split="temporal 80/20", auc=float(roc_auc_score(y[cut:], pp)),
           ap=float(average_precision_score(y[cut:], pp)), prevalence=float(y[cut:].mean()))]
save_table(pd.DataFrame(pc), "05_power_peak_classification")
imp = pd.DataFrame({"feature": FE, "importance": mdl.feature_importances_}).sort_values("importance", ascending=False)
save_table(imp, "05_power_peak_clf_importance")

# residual forensics
te = cl.iloc[cut:].copy(); te["score"] = pp
te["err"] = te.peak_t_plus4 - te.score
save_table(te.groupby("시간").agg(n=("score", "size"), prevalence=("peak_t_plus4", "mean"),
                                mean_score=("score", "mean"), mean_err=("err", "mean")).reset_index(),
           "05_power_peak_error_by_hour")

# ---------------- 7. controllability table
ctrl = [("15분/30분/45분/60분", "목표(결과)", "-", "예측 대상"),
        ("생산량", "간접 통제 가능", "생산계획으로 조정 가능하나 수요에 종속", "시간당 생산량"),
        ("공장인원", "통제 가능(단, 생산량과 spearman 0.994 → 독립 조작 불가)", "실질적으로 생산량의 함수", ""),
        ("시간", "통제 가능(스케줄링)", "작업 시간대 이동 = 피크 회피 수단", ""),
        ("day", "통제 가능(요일 배치)", "주말 가동 여부", ""),
        ("인건비", "통제 가능(교대/수당 정책)", "{1.0,1.5} 2수준만 존재", ""),
        ("전기요금(계절)", "통제 불가(요금제)", "월의 결정함수", ""),
        ("기온/습도/풍속/강수량", "통제 불가(외생)", "예보로 사전 입력 가능", ""),
        ("m, d", "통제 불가(달력)", "", "")]
save_table(pd.DataFrame(ctrl, columns=["variable", "controllability", "note", "detail"]), "05_power_controllability")

# ---------------- figures
fig, ax = plt.subplots(3, 2, figsize=(12, 9))
ax[0, 0].plot(q.ts15, q.kw, lw=.3)
ax[0, 0].axvline(pd.Timestamp("2021-07-01"), color="crimson")
ax[0, 0].set_title("15분 수요 전체 (빨간선 이후만 중복 없는 원본 구간)")
hm = q.pivot_table(index="시간", columns="day", values="kw", aggfunc="mean")
im = ax[0, 1].imshow(hm.values, aspect="auto", cmap="magma"); plt.colorbar(im, ax=ax[0, 1])
ax[0, 1].set_yticks(range(24)); ax[0, 1].set_yticklabels(hm.index, fontsize=6)
ax[0, 1].set_xticks(range(len(hm.columns))); ax[0, 1].set_xticklabels(hm.columns)
ax[0, 1].set_title("시간×요일 평균 수요")
ax[1, 0].bar(dupsum.index, dupsum.dup_rate)
ax[1, 0].set_title("월별 '다른 날과 전력프로파일이 완전 동일'한 날 비율")
for c in sorted(meta.cl_dtw_average.unique()):
    ax[1, 1].plot(Wm[meta.cl_dtw_average == c].mean(0), label=f"DTW cl{c} n={(meta.cl_dtw_average==c).sum()}")
ax[1, 1].legend(fontsize=7); ax[1, 1].set_title("피크 궤적 원형 (DTW 군집 평균, 정규화)")
ax[2, 0].scatter(q["생산량"], q.kw, s=2, alpha=.2, c=q.clean.map({True: "crimson", False: "steelblue"}))
ax[2, 0].set_xlabel("생산량(시간)"); ax[2, 0].set_ylabel("kw"); ax[2, 0].set_title("생산량-수요 (빨강=7~9월 원본)")
ax[2, 1].scatter(ev.rise_steps, ev.peak_kw, c=ev.hour, cmap="twilight", s=18)
ax[2, 1].set_xlabel("상승 스텝 수(15분)"); ax[2, 1].set_ylabel("peak kw"); ax[2, 1].set_title("급상승형 vs 누적형 피크 (색=시각)")
fig.tight_layout(); fig.savefig(fig_path("05_power_overview")); plt.close(fig)
print("done", len(ev), "events")

# ---------------- 8. efficiency frontier: same output, lower peak?
dayq = q.groupby("날짜").agg(kw_peak=("kw", "max"), kw_mean=("kw", "mean"), kwh_proxy=("kw", "sum"),
                            prod=("생산량", "sum"), temp=("기온", "mean"), dow=("day", "first"),
                            month=("m", "first"), active_hours=("생산량", lambda s: int((s > 0).sum() / 4)),
                            clean=("clean", "first")).reset_index()
dayq = dayq[dayq["prod"] > 0].copy()
dayq["kwh_per_unit"] = dayq.kwh_proxy / dayq["prod"]
dayq["peak_per_unit"] = dayq.kw_peak / dayq["prod"]
dayq["bin"] = pd.qcut(dayq["prod"], 6, duplicates="drop")
eff = dayq.groupby("bin", observed=True).agg(n=("날짜", "size"), prod_median=("prod", "median"),
                                             peak_min=("kw_peak", "min"), peak_median=("kw_peak", "median"),
                                             peak_max=("kw_peak", "max"),
                                             kwh_min=("kwh_proxy", "min"), kwh_median=("kwh_proxy", "median"),
                                             kwh_max=("kwh_proxy", "max"),
                                             active_hours_of_min_peak=("active_hours", "first")).reset_index()
eff["peak_spread_pct_of_median"] = (eff.peak_max - eff.peak_min) / eff.peak_median
eff["kwh_spread_pct_of_median"] = (eff.kwh_max - eff.kwh_min) / eff.kwh_median
save_table(eff.astype({"bin": str}), "05_power_efficiency_frontier")
save_table(dayq.astype({"bin": str}), "05_power_daily_efficiency")

# what distinguishes low-peak from high-peak days at equal output?
sp = []
for b, g_ in dayq.groupby("bin", observed=True):
    if len(g_) < 10: continue
    lo = g_[g_.kw_peak <= g_.kw_peak.quantile(.25)]
    hi = g_[g_.kw_peak >= g_.kw_peak.quantile(.75)]
    for c in ["active_hours", "temp", "kwh_proxy", "kw_mean", "prod"]:
        sp.append(dict(prod_bin=str(b), n_lo=len(lo), n_hi=len(hi), feature=c,
                       median_low_peak_days=float(lo[c].median()), median_high_peak_days=float(hi[c].median()),
                       mwu_p=float(stats.mannwhitneyu(lo[c], hi[c]).pvalue)))
save_table(pd.DataFrame(sp), "05_power_low_vs_high_peak_at_equal_output")

# ---------------- 9. peak precursor: state 1h before a p99 peak
hi = np.quantile(s, .99)
onset = np.where((s >= hi) & (np.r_[False, s[:-1] < hi]))[0]
pre = []
for i in onset:
    if i < 8: continue
    pre.append(dict(i=int(i), t=str(q.ts15.iloc[i]), hour=int(q.시간.iloc[i]),
                    kw_t=float(s[i]), kw_m1=float(s[i - 1]), kw_m4=float(s[i - 4]), kw_m8=float(s[i - 8]),
                    slope_1h=float((s[i - 1] - s[i - 4]) / 3), slope_2h=float((s[i - 1] - s[i - 8]) / 7),
                    prod_t=float(q["생산량"].iloc[i]), prod_m4=float(q["생산량"].iloc[i - 4]),
                    clean=bool(q.clean.iloc[i])))
pre = pd.DataFrame(pre)
rand = rng.choice(np.setdiff1d(np.arange(8, len(s)), onset), size=min(2000, len(s) - 8), replace=False)
bg = pd.DataFrame(dict(slope_1h=(s[rand - 1] - s[rand - 4]) / 3, kw_m4=s[rand - 4], kw_m1=s[rand - 1]))
prec = [dict(metric=c, median_before_p99_peak=float(pre[c].median()), median_background=float(bg[c].median()),
             mwu_p=float(stats.mannwhitneyu(pre[c], bg[c]).pvalue), n_onsets=len(pre))
        for c in ["slope_1h", "kw_m4", "kw_m1"]]
save_table(pd.DataFrame(prec), "05_power_peak_precursor")
save_table(pre, "05_power_peak_onsets")
print("appendix done")
