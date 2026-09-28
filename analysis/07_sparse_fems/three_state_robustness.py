"""07-P. 3상태 구조의 견고성 (Task P) — '데이터 기반 거시 운전 레짐' 으로서 방어 가능한가.

용어: 공식 산업 표준 상태가 아니다. stop/transition/normal_run 은 데이터에서 유도한
      거시 운전 레짐(macro operating regime)이며 '셋업'으로 명명하지 않는다.

정의 4종을 비교한다(모두 결정시점 정보만, 컷/모델은 과거 구간에서만 적합):
  A rule_ramp    : 조업여부 1시간내 변화 or |kw_ramp4| >= train p90   (현행)
  B changepoint  : 인과적 변화점 통계(최근 1시간 평균 - 그 앞 2시간 평균)가 train p90 이상
                   + 수준(train 1D KMeans 2군)으로 저부하/정상 구분. 생산량·ramp컷 미사용
  C gmm3         : GaussianMixture(3) on [kw, kw_ramp4, kw_std4] (train 적합)
  D hmm3_latent  : 07D 의 forward-filtering HMM 상태 (기존 산출물 재사용)
질문: "저부하 / 안정조업 / 전환형 3구조가 방법과 무관하게 반복 등장하는가?"
"""
import numpy as np, pandas as pd
import _fe
from _fe import build, save_table, fig_path, mpl, SEED, TAB
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score
plt = mpl()

o = pd.read_csv(TAB / "07O_policy_rows.csv", parse_dates=["ts15"], encoding="utf-8-sig")
q = build(latent_cols=["lat_state"] if (TAB / "07D_latent_state_features.csv").exists() else None)
cols = ["ts15", "is_operating_lag4", "kw_lag4", "kw_lag8", "kw_roll4"] + \
       (["lat_state"] if "lat_state" in q else [])
o = o.merge(q[cols], on="ts15", how="left").dropna(subset=["kw_std4", "kw_roll4"]).reset_index(drop=True)
TRAIN = o.fold == 0                                  # 첫 폴드 구간을 '과거' 로 삼아 컷/모델 적합

# ---- A: 현행 규칙
ramp_cut = float(np.quantile(o.loc[TRAIN, "kw_ramp4"].abs(), .9))
sw = o.is_operating.values != o.is_operating_lag4.values
A = np.where(sw | (o.kw_ramp4.abs().values >= ramp_cut), "transition",
             np.where(o.is_operating.values == 0, "stop", "normal_run"))

# ---- B: 인과적 변화점 (생산량·ramp컷 미사용)
recent = o.kw_roll4.values                                   # 최근 1시간 평균
prev = o.kw_lag4.rolling(8, min_periods=4).mean().values if hasattr(o.kw_lag4, "rolling") else None
prev = pd.Series(o.kw_lag4.values).rolling(8, min_periods=4).mean().values
cp_stat = np.abs(recent - prev)
cp_cut = float(np.nanquantile(cp_stat[TRAIN.values], .9))
km = KMeans(n_clusters=2, n_init=10, random_state=SEED).fit(o.loc[TRAIN, ["kw"]].values)
lo_cluster = int(np.argmin(km.cluster_centers_.ravel()))
lvl = km.predict(o[["kw"]].values)
B = np.where(cp_stat >= cp_cut, "transition",
             np.where(lvl == lo_cluster, "stop", "normal_run"))

# ---- C: GMM 3성분
FC = ["kw", "kw_ramp4", "kw_std4"]
sc = StandardScaler().fit(o.loc[TRAIN, FC].values)
gm = GaussianMixture(3, covariance_type="full", random_state=SEED).fit(sc.transform(o.loc[TRAIN, FC].values))
cl = gm.predict(sc.transform(o[FC].values))
# 라벨을 해석 가능한 이름으로 사상: 평균 kw 최저 = stop, |ramp| 최대 = transition, 나머지 = normal_run
prof = pd.DataFrame({"c": cl, "kw": o.kw.values, "ar": o.kw_ramp4.abs().values}).groupby("c").mean()
st_c = int(prof.kw.idxmin()); tr_c = int(prof.drop(index=st_c).ar.idxmax())
C = np.array(["normal_run"] * len(o), dtype=object)
C[cl == st_c] = "stop"; C[cl == tr_c] = "transition"

DEFS = {"A_rule_ramp": A, "B_changepoint": B, "C_gmm3": C}
if "lat_state" in o:
    D = o.lat_state.fillna(-1).astype(int).astype(str).radd("hmm_").values
    DEFS["D_hmm3_latent"] = D

# ---- 레짐 프로파일
rows = []
for dname, lab in DEFS.items():
    for st in pd.unique(lab):
        m = np.asarray(lab) == st
        s = o[m]
        rows.append(dict(definition=dname, state=str(st), n=int(m.sum()), occupancy=float(m.mean()),
                         mean_kw=float(s.kw.mean()), mean_abs_ramp=float(s.kw_ramp4.abs().mean()),
                         mean_kw_std4=float(s.kw_std4.mean()),
                         operating_share=float(s.is_operating.mean()),
                         mean_production=float(s["생산량"].mean()),
                         peak_prevalence=float(s.peak_now.mean()),
                         mae_h60=float(s.abs_err_h4.mean()),
                         low_or_ood_share=float(s.band.isin(["LOW", "OOD"]).mean()),
                         high_band_share=float((s.band == "HIGH").mean()),
                         mean_interval_width=float(s.int_width.mean())))
prof_t = pd.DataFrame(rows)
save_table(prof_t, "07P_regime_profiles")

# ---- 정의 간 일치도
names = list(DEFS)
ari = pd.DataFrame(index=names, columns=names, dtype=float)
for a in names:
    for b in names:
        ari.loc[a, b] = float(adjusted_rand_score(DEFS[a], DEFS[b]))
save_table(ari.reset_index().rename(columns={"index": "definition"}), "07P_definition_agreement_ari")

# 3상태로 붕괴시킨 교차표 (D 는 hmm 라벨이므로 이름 사상 없이 비교만)
ct = []
for a in names:
    for b in names:
        if a >= b:
            continue
        t = pd.crosstab(pd.Series(DEFS[a], name=a), pd.Series(DEFS[b], name=b), normalize="index")
        for i in t.index:
            ct.append(dict(def_a=a, def_b=b, state_a=str(i),
                           **{f"b_{c}": float(t.loc[i, c]) for c in t.columns}))
save_table(pd.DataFrame(ct), "07P_definition_crosstab")

# ---- 구조 반복성 판정 (사전 기준)
#  각 정의에서 (1) 저부하·저ramp·저피크 레짐 (2) 고부하·저ramp 레짐 (3) 고ramp·고오차 레짐이 모두 존재하는가
vr = []
for dname in names:
    p = prof_t[prof_t.definition == dname].set_index("state")
    kw_med, ar_med = o.kw.median(), o.kw_ramp4.abs().median()
    has_low = bool(((p.mean_kw < kw_med) & (p.mean_abs_ramp < p.mean_abs_ramp.max()) &
                    (p.peak_prevalence < 0.05)).any())
    has_stable = bool(((p.mean_kw >= kw_med) & (p.mean_abs_ramp <= p.mean_abs_ramp.median())).any())
    hi = p.mean_abs_ramp.idxmax()
    has_trans = bool(p.loc[hi, "mae_h60"] > p.mae_h60.median() and p.loc[hi, "mean_abs_ramp"] > ar_med)
    vr.append(dict(definition=dname, n_states=len(p), has_low_load_regime=has_low,
                   has_stable_run_regime=has_stable, has_transition_like_regime=has_trans,
                   three_structure_present=bool(has_low and has_stable and has_trans),
                   max_ramp_state=str(hi), max_ramp_state_mae=float(p.loc[hi, "mae_h60"]),
                   overall_mae=float(o.abs_err_h4.mean()),
                   max_ramp_state_mae_ratio=float(p.loc[hi, "mae_h60"] / o.abs_err_h4.mean()),
                   max_ramp_state_peak_prevalence=float(p.loc[hi, "peak_prevalence"]),
                   max_ramp_state_low_or_ood=float(p.loc[hi, "low_or_ood_share"])))
ver = pd.DataFrame(vr)
n_ok = int(ver.three_structure_present.sum())
mean_ari = float(ari.values[np.triu_indices(len(names), 1)].mean())
verdict = ("YES_데이터기반 3레짐 구조 반복" if n_ok >= len(names) - 1 and mean_ari >= .3 else
           "WEAK_구조는 반복되나 정의 간 일치도 낮음" if n_ok >= len(names) - 1 else
           "NO_3상태 표현 강하게 쓰지 말 것")
ver["verdict"] = verdict
ver["mean_pairwise_ari"] = mean_ari
save_table(ver, "07P_three_state_verdict")

fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
for i, k in enumerate(["mean_kw", "mean_abs_ramp", "mae_h60"]):
    for dname in names:
        p = prof_t[prof_t.definition == dname]
        axes[i].scatter(p.occupancy, p[k], label=dname, s=40)
        for _, r in p.iterrows():
            axes[i].annotate(r.state[:12], (r.occupancy, r[k]), fontsize=5)
    axes[i].set_xlabel("레짐 점유율"); axes[i].set_ylabel(k); axes[i].legend(fontsize=5)
axes[0].set_title("정의별 레짐 구조 — 부하 수준"); axes[1].set_title("변동 크기"); axes[2].set_title("예측오차")
fig.tight_layout(); fig.savefig(fig_path("07P_three_state_robustness")); plt.close(fig)
print(prof_t.to_string(index=False)); print(ari.round(3).to_string())
print(ver.to_string(index=False)); print("판정:", verdict)
