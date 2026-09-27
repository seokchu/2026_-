"""07-M. 3상태(정지 / 셋업·전환 / 정상조업) 분리가 2상태보다 오차 구조를 더 설명하는가.

07C 난조건 결과가 "비조업 전체는 쉽고 전환점이 어렵다" 를 암시했다. 이를 직접 검증한다.
상태 정의는 결정시점 t 까지의 정보만 쓴다 (미래 조업 여부 사용 금지):
  transition : 최근 1시간 안에 조업여부가 바뀌었거나 |kw_ramp4| 가 train p90 이상
  stop       : transition 아님 & 생산량 == 0
  normal_run : transition 아님 & 생산량 > 0
컷(ramp p90)은 train 구간에서만 산출한다.
평가: 상태별 OOF 오차/피크유병률, paired bootstrap, 상태더미 설명력(R^2),
      global vs 2상태전용 vs 3상태전용 모델, hmm3 잠재상태와의 일치도.
"""
import numpy as np, pandas as pd
import _fe
from _fe import build, split, save_table, fig_path, mpl, SEED, TAB, paired_boot
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import adjusted_rand_score, r2_score
plt = mpl()

H = 4
F = _fe.LEVELS["L3_+production"]
o = pd.read_csv(TAB / "07C_oof_predictions.csv", parse_dates=["ts15"], encoding="utf-8-sig")
q = build()
o = o.merge(q[["ts15", "is_operating_lag4"]], on="ts15", how="left")
# ramp 컷은 첫 폴드 이전 구간(= OOF 시작 전 학습구간)에서만
pre = q[q.ts15 < o.ts15.min()]
RAMP_CUT = float(np.quantile(pre.kw_ramp4.abs().dropna(), .9))


def label_states(df, ramp_cut=RAMP_CUT):
    sw = df.is_operating.values != df.is_operating_lag4.values
    tr_ = sw | (df.kw_ramp4.abs().values >= ramp_cut)
    return np.where(tr_, "transition", np.where(df.is_operating.values == 0, "stop", "normal_run"))


o["state3"] = label_states(o)
o["state2"] = np.where(o.is_operating.values == 1, "operating", "idle")
save_table(pd.DataFrame([dict(ramp_cut_kw=RAMP_CUT, n_pre_rows=len(pre),
                             definition="transition = 조업여부 1시간내 변화 or |kw_ramp4|>=train p90")]),
           "07M_state_definition")

rows = []
for wtag, sub in [("oof_all_2021", o), ("oof_clean_Jul_Sep", o[o.clean])]:
    base = sub.abs_err.mean()
    for scheme, col in [("2state", "state2"), ("3state", "state3")]:
        for st, s in sub.groupby(col):
            rows.append(dict(window=wtag, scheme=scheme, state=st, n=len(s), share=float(len(s) / len(sub)),
                             mae=float(s.abs_err.mean()), mae_ratio_vs_overall=float(s.abs_err.mean() / base),
                             err_p90=float(np.quantile(s.abs_err, .9)),
                             rmse=float(np.sqrt((s.abs_err ** 2).mean())),
                             peak_prevalence=float(s.is_peak.mean()),
                             mean_abs_ramp=float(s.abs_ramp.mean()),
                             mean_disagreement=float(s.disagreement.mean()),
                             mean_ood_maha=float(s.ood_maha.mean())))
prof = pd.DataFrame(rows)
save_table(prof, "07M_state_error_profile")

# ---- 상태별 MAE 차이 부트스트랩 (정상조업 대비)
brows = []
rng = np.random.default_rng(SEED)
for wtag, sub in [("oof_all_2021", o), ("oof_clean_Jul_Sep", o[o.clean])]:
    ref = sub[sub.state3 == "normal_run"].abs_err.values
    for st in ["transition", "stop"]:
        a = sub[sub.state3 == st].abs_err.values
        if len(a) < 30:
            continue
        bs = np.array([rng.choice(a, len(a), True).mean() - rng.choice(ref, len(ref), True).mean()
                       for _ in range(2000)])
        brows.append(dict(window=wtag, state=st, vs="normal_run", n=len(a),
                          mae_diff=float(a.mean() - ref.mean()),
                          ci_lo=float(np.quantile(bs, .025)), ci_hi=float(np.quantile(bs, .975)),
                          significant=bool(np.quantile(bs, .025) > 0 or np.quantile(bs, .975) < 0)))
save_table(pd.DataFrame(brows), "07M_state_mae_bootstrap")

# ---- 설명력: 상태 더미만으로 |오차| 를 얼마나 설명하는가
exp_rows = []
for wtag, sub in [("oof_all_2021", o), ("oof_clean_Jul_Sep", o[o.clean])]:
    for scheme, col in [("2state", "state2"), ("3state", "state3")]:
        pred = sub.groupby(col).abs_err.transform("mean")
        exp_rows.append(dict(window=wtag, scheme=scheme, n=len(sub),
                             r2_abs_err_from_state=float(r2_score(sub.abs_err, pred)),
                             mae_spread=float(sub.groupby(col).abs_err.mean().max() -
                                              sub.groupby(col).abs_err.mean().min())))
save_table(pd.DataFrame(exp_rows), "07M_state_explanatory_power")

# ---- 상태전용 모델이 global 을 이기는가 (시간순 holdout, 06A 와 같은 설계)
mrows, boot2 = [], []
for wtag in ["all_2021", "clean_Jul_Sep"]:
    sub = (q if wtag == "all_2021" else q[q.clean]).copy()
    tr, ca, te = split(sub, F, H)
    fit = pd.concat([tr, ca])
    fit["state3"], te = label_states(fit), te.copy()
    te["state3"] = label_states(te)
    fit["state2"] = np.where(fit.is_operating == 1, "operating", "idle")
    te["state2"] = np.where(te.is_operating == 1, "operating", "idle")
    y = te[f"y_h{H}"].values
    g = HistGradientBoostingRegressor(random_state=SEED).fit(fit[F].values, fit[f"y_h{H}"].values)
    preds = {"global": g.predict(te[F].values)}
    for scheme in ["state2", "state3"]:
        p = np.full(len(te), np.nan)
        for st in fit[scheme].unique():
            mf, mt = fit[scheme] == st, (te[scheme] == st).values
            if mf.sum() < 200 or mt.sum() == 0:
                continue
            p[mt] = HistGradientBoostingRegressor(random_state=SEED).fit(
                fit.loc[mf, F].values, fit.loc[mf, f"y_h{H}"].values).predict(te.loc[mt, F].values)
        preds[f"{scheme}_specific"] = np.where(np.isnan(p), preds["global"], p)
    for k, p in preds.items():
        e = np.abs(y - p)
        mrows.append(dict(window=wtag, variant=k, n_test=len(te), mae=float(e.mean()),
                          rmse=float(np.sqrt((e ** 2).mean())),
                          mae_transition=float(e[(te.state3 == "transition").values].mean()),
                          mae_stop=float(e[(te.state3 == "stop").values].mean()),
                          mae_normal=float(e[(te.state3 == "normal_run").values].mean())))
        if k != "global":
            boot2.append(dict(window=wtag, variant=k, vs="global",
                              **paired_boot(e, np.abs(y - preds["global"]))))
save_table(pd.DataFrame(mrows), "07M_state_specific_models")
save_table(pd.DataFrame(boot2), "07M_state_specific_bootstrap")

# ---- hmm3 잠재상태와의 일치도 (같은 것을 보고 있는가)
if "lat_state" in o:
    m = o.lat_state.notna()
    ari = float(adjusted_rand_score(o.state3[m], o.lat_state[m].astype(int)))
    ct = pd.crosstab(o.state3[m], o.lat_state[m].astype(int), normalize="index").reset_index()
    ct.insert(0, "adjusted_rand_vs_latent", ari)
    save_table(ct, "07M_state_vs_latent_crosstab")

fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
d = prof[(prof.window == "oof_clean_Jul_Sep") & (prof.scheme == "3state")].sort_values("mae")
axes[0].bar(d.state, d.mae, color="#27b"); axes[0].set_ylabel("OOF MAE (kw)")
axes[0].set_title("3상태별 예측오차 (clean)")
axes[1].bar(d.state, d.peak_prevalence, color="#c33"); axes[1].set_ylabel("피크 유병률")
axes[1].set_title("3상태별 피크 유병률")
e = prof[(prof.window == "oof_clean_Jul_Sep")]
axes[2].bar(range(len(e)), e.mae_ratio_vs_overall,
            color=["#888" if s == "2state" else "#2b7" for s in e.scheme])
axes[2].set_xticks(range(len(e))); axes[2].set_xticklabels(e.state + "\n" + e.scheme, fontsize=6)
axes[2].axhline(1, color="k", lw=.8); axes[2].set_title("2상태 vs 3상태 MAE 비")
fig.tight_layout(); fig.savefig(fig_path("07M_three_state")); plt.close(fig)
print(prof.to_string(index=False)); print(pd.DataFrame(brows).to_string(index=False))
print(pd.DataFrame(exp_rows).to_string(index=False)); print(pd.DataFrame(mrows).to_string(index=False))
print(pd.DataFrame(boot2).to_string(index=False))
