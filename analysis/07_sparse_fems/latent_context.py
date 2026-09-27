"""07-D. NILM 유사 잠재 문맥 타당성 — 장비 단위 진실이 없으므로 '분해'라 부르지 않는다.

질문: 집계 수요의 구조만으로 안정적인 잠재 운전문맥 표현을 얻을 수 있는가.
후보: GaussianHMM(3/4/5, forward-only filtering) / GaussianMixture / NMF(지연창)
누수 방지:
  - 모든 잠재 모델은 all_2021 앞 60% 구간에서만 적합하고 전 구간에 적용한다(미래 미사용).
  - HMM 은 forward-backward(평활) 대신 forward filtering 만 사용한다. 평활 posterior 는
    미래 관측을 쓰므로 실시간 배치에서 불가능하다.
  - 잠재 특성 입력은 집계 수요 계열(L0)뿐이다. 생산량/날씨는 넣지 않는다.
"""
import numpy as np, pandas as pd
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import NMF
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score
from hmmlearn.hmm import GaussianHMM
import _fe
from _fe import build, save_table, fig_path, mpl, SEED
plt = mpl()

OBS = ["kw", "kw_diff1", "kw_roll4", "kw_std4"]
LAGW = ["kw", "kw_lag1", "kw_lag2", "kw_lag3", "kw_lag4", "kw_lag8"]
q = build().dropna(subset=OBS + LAGW).reset_index(drop=True)
X = q[OBS].values
FIT_END = int(len(q) * .6)                      # 적합 구간 = all_2021 앞 60%
sc = StandardScaler().fit(X[:FIT_END])
Z = sc.transform(X)


def hmm_filter(m, Z):
    """forward filtering 만으로 얻은 상태 posterior (미래 관측 미사용)."""
    logB = np.column_stack([multivariate_normal(m.means_[k], m.covars_[k], allow_singular=True).logpdf(Z)
                            for k in range(m.n_components)])
    logA = np.log(m.transmat_ + 1e-300)
    a = np.log(m.startprob_ + 1e-300) + logB[0]
    out = np.empty((len(Z), m.n_components))
    for t in range(len(Z)):
        if t:
            a = logB[t] + np.logaddexp.reduce(a[:, None] + logA, axis=0)
        a -= a.max()
        out[t] = np.exp(a) / np.exp(a).sum()
        a = np.log(out[t] + 1e-300)
    return out


def fit_hmm(k, seed, end=FIT_END):
    m = GaussianHMM(n_components=k, covariance_type="diag", n_iter=60, random_state=seed)
    m.fit(Z[:end])
    return m


cands, post_store = {}, {}
for k in (3, 4, 5):
    m = fit_hmm(k, SEED)
    post = hmm_filter(m, Z)
    cands[f"hmm{k}"] = post.argmax(1); post_store[f"hmm{k}"] = post
for k in (3, 4):
    g = GaussianMixture(k, random_state=SEED, covariance_type="full").fit(Z[:FIT_END])
    post = g.predict_proba(Z)
    cands[f"gmm{k}"] = post.argmax(1); post_store[f"gmm{k}"] = post
W = q[LAGW].values
nmf = NMF(n_components=3, random_state=SEED, max_iter=400, init="nndsvda").fit(W[:FIT_END])
H = nmf.transform(W)
cands["nmf3"] = H.argmax(1); post_store["nmf3"] = H / (H.sum(1, keepdims=True) + 1e-9)

# ---- A. 안정성: 시드 민감도 + 시간 폴드 간 일치도(ARI)
stab = []
for name in cands:
    if name.startswith("hmm"):
        k = int(name[-1])
        seeds = [fit_hmm(k, s) for s in (SEED, SEED + 1, SEED + 2)]
        labs = [hmm_filter(m, Z).argmax(1) for m in seeds]
        half = fit_hmm(k, SEED, end=int(FIT_END * .5))
        lab_half = hmm_filter(half, Z).argmax(1)
    elif name.startswith("gmm"):
        k = int(name[-1])
        labs = [GaussianMixture(k, random_state=s, covariance_type="full").fit(Z[:FIT_END]).predict(Z)
                for s in (SEED, SEED + 1, SEED + 2)]
        lab_half = GaussianMixture(k, random_state=SEED, covariance_type="full").fit(
            Z[:int(FIT_END * .5)]).predict(Z)
    else:
        labs = [NMF(3, random_state=s, max_iter=400, init="nndsvda").fit_transform(W).argmax(1)
                for s in (SEED, SEED + 1, SEED + 2)]
        lab_half = NMF(3, random_state=SEED, max_iter=400, init="nndsvda").fit(
            W[:int(FIT_END * .5)]).transform(W).argmax(1)
    seed_ari = float(np.mean([adjusted_rand_score(labs[0], l) for l in labs[1:]]))
    lab = cands[name]
    tail = slice(FIT_END, None)
    stab.append(dict(method=name, n_states=len(np.unique(lab)),
                     seed_ari=seed_ari,
                     halfdata_ari=float(adjusted_rand_score(lab, lab_half)),
                     states_seen_in_train=int(len(np.unique(lab[:FIT_END]))),
                     states_seen_in_tail=int(len(np.unique(lab[tail]))),
                     min_state_share=float(pd.Series(lab).value_counts(normalize=True).min()),
                     stable=bool(seed_ari > .7 and adjusted_rand_score(lab, lab_half) > .6)))
stab = pd.DataFrame(stab)
save_table(stab, "07D_latent_stability")

# ---- B. 해석: 상태별 프로파일 (이름은 붙이지 않는다)
thr = float(np.quantile(q.kw[:FIT_END], .95))
interp = []
for name, lab in cands.items():
    g = q.assign(state=lab).groupby("state")
    for s, d in g:
        interp.append(dict(method=name, state=int(s), share=len(d) / len(q),
                           mean_kw=float(d.kw.mean()), mean_ramp=float(d.kw_diff1.mean()),
                           std_kw=float(d.kw.std()), mean_production=float(d["생산량"].mean()),
                           operating_share=float(d.is_operating.mean()),
                           modal_hour=int(d["시간"].mode().iloc[0]),
                           peak_rate=float((d.kw >= thr).mean()),
                           mean_temp=float(d["기온"].mean())))
save_table(pd.DataFrame(interp), "07D_latent_state_profiles")

# ---- 캐시: 가장 안정적인 방법을 고른다(안정성 우선, 동률이면 상태수 적은 것)
pick = stab.sort_values(["stable", "seed_ari", "halfdata_ari"], ascending=False).iloc[0].method
post = post_store[pick]
lab = cands[pick]
hi_state = int(np.argmax([q.kw[lab == s].mean() for s in range(post.shape[1])]))
chg = np.r_[True, lab[1:] != lab[:-1]]
dwell = np.zeros(len(lab), int)
c = 0
for i, ch in enumerate(chg):
    c = 0 if ch else c + 1
    dwell[i] = c
out = pd.DataFrame(dict(ts15=q.ts15, lat_state=lab, lat_dwell=dwell, lat_p_high=post[:, hi_state]))
save_table(out, "07D_latent_state_features")
print("selected latent method:", pick, "high-load state:", hi_state)
print(stab.to_string(index=False))
