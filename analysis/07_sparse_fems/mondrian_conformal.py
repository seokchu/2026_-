"""07-J. Mondrian(조건부) conformal 이 global CQR 보다 피크 coverage 를 개선하는가.

배경: 08 보고서에서 global CQR 의 coverage 가 조업 0.719 / 비조업 0.998 로 갈라졌다.
marginal coverage 는 맞아도 조건부 coverage 는 틀린 전형적 형태다.
설계:
  - 분위모델 q05/q95 는 train 에서만 적합. 보정잔차 분위 Q 는 calibration 구간에서만 산출.
  - Mondrian = 결정시점에 알 수 있는 그룹별로 Q 를 따로 잡는다(Vovk 2003 taxonomy).
  - 그룹 정의도 train 에서만 컷을 정한다. calibration 표본 50개 미만 그룹은 global Q 로 후퇴.
  - 평가: 전체 coverage 가 아니라 조업/비조업/실피크/상위예측 구간의 조건부 coverage,
    그리고 같은 상한을 경보로 썼을 때의 precision/recall 을 같이 본다.
"""
import numpy as np, pandas as pd
import _fe
from _fe import build, window, split, peak_threshold, save_table, fig_path, mpl, SEED, TAB
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
plt = mpl()

ALPHA = 0.10
LAT = ["lat_state", "lat_dwell", "lat_p_high"]
has_lat = (TAB / "07D_latent_state_features.csv").exists()
F = _fe.LEVELS["L3_+production"]
MIN_CAL = 50


def qr(a):
    return HistGradientBoostingRegressor(loss="quantile", quantile=a, random_state=SEED)


def conf_q(E, alpha=ALPHA):
    return float(np.quantile(E, min(1.0, (1 - alpha) * (1 + 1 / len(E)))))


def groupings(tr, ca, te, pred_tr, pred_ca, pred_te):
    """이름 -> (ca 그룹라벨, te 그룹라벨). 컷은 tr 에서만."""
    g = {"global": (np.zeros(len(ca), int), np.zeros(len(te), int))}
    g["by_operating"] = (ca.is_operating.values.astype(int), te.is_operating.values.astype(int))
    c = np.quantile(pred_tr, [.5, .8])
    g["by_pred_level"] = (np.digitize(pred_ca, c), np.digitize(pred_te, c))
    c2 = np.quantile(np.abs(tr.kw_ramp4), [.5, .9])
    g["by_abs_ramp"] = (np.digitize(np.abs(ca.kw_ramp4), c2), np.digitize(np.abs(te.kw_ramp4), c2))
    g["by_operating_x_pred_level"] = (ca.is_operating.values.astype(int) * 3 + np.digitize(pred_ca, c),
                                      te.is_operating.values.astype(int) * 3 + np.digitize(pred_te, c))
    if has_lat and "lat_state" in ca:
        g["by_latent_state"] = (ca.lat_state.fillna(-1).astype(int).values,
                                te.lat_state.fillna(-1).astype(int).values)
    return g


def eval_block(tr, ca, te, wtag, h, emode, fold=-1):
    ytr, yca, yte = (d[f"y_h{h}"].values for d in (tr, ca, te))
    m_lo, m_hi = qr(ALPHA / 2).fit(tr[F].values, ytr), qr(1 - ALPHA / 2).fit(tr[F].values, ytr)
    lo_ca, hi_ca = m_lo.predict(ca[F].values), m_hi.predict(ca[F].values)
    lo_te, hi_te = m_lo.predict(te[F].values), m_hi.predict(te[F].values)
    E = np.maximum(lo_ca - yca, yca - hi_ca)                     # CQR 비적합도
    pm = HistGradientBoostingRegressor(random_state=SEED).fit(tr[F].values, ytr)
    p_tr, p_ca, p_te = pm.predict(tr[F].values), pm.predict(ca[F].values), pm.predict(te[F].values)
    thr = peak_threshold(tr)
    is_peak = yte >= thr
    top_dec = p_te >= np.quantile(p_tr, .9)
    Qg = conf_q(E)
    fixed_part = te.is_operating.values.astype(int) * 3 + np.digitize(p_te, np.quantile(p_tr, [.5, .8]))
    for gname, (gca, gte) in groupings(tr, ca, te, p_tr, p_ca, p_te).items():
        Q = np.full(len(te), Qg)
        n_fb = 0
        for lab in np.unique(gte):
            mca, mte = gca == lab, gte == lab
            if mca.sum() >= MIN_CAL:
                Q[mte] = conf_q(E[mca])
            else:
                n_fb += int(mte.sum())
            if mte.sum():
                qq = conf_q(E[mca]) if mca.sum() >= MIN_CAL else Qg
                ins = (yte[mte] >= lo_te[mte] - qq) & (yte[mte] <= hi_te[mte] + qq)
                grows.append(dict(window=wtag, eval_mode=emode, fold=fold, horizon_min=h * 15,
                                  grouping=gname, group=int(lab),
                                  n_cal=int(mca.sum()), n_test=int(mte.sum()), conformal_Q=qq,
                                  coverage=float(ins.mean()), target=1 - ALPHA,
                                  coverage_gap=float(ins.mean() - (1 - ALPHA)),
                                  mean_width=float((hi_te[mte] - lo_te[mte] + 2 * qq).mean()),
                                  fallback_global_Q=bool(mca.sum() < MIN_CAL)))
        lo, hi = lo_te - Q, hi_te + Q
        ins = (yte >= lo) & (yte <= hi)
        alert = hi >= thr
        tp = int((alert & is_peak).sum()); fp = int((alert & ~is_peak).sum())
        fn = int((~alert & is_peak).sum())
        gcov = pd.Series(ins).groupby(gte).mean()
        # 공정 비교용: 모든 grouping 을 같은 고정 분할(조업 x 예측수준)에서 평가한다.
        fx = pd.Series(ins).groupby(fixed_part).agg(["mean", "size"])
        fx = fx[fx["size"] >= 30]["mean"]
        rows.append(dict(
            window=wtag, eval_mode=emode, fold=fold, horizon_min=h * 15, grouping=gname, n_groups=int(len(np.unique(gte))),
            n_test=len(te), n_fallback_rows=n_fb, target_coverage=1 - ALPHA,
            coverage=float(ins.mean()), coverage_gap=float(ins.mean() - (1 - ALPHA)),
            cov_operating=float(ins[te.is_operating.values == 1].mean()),
            cov_idle=float(ins[te.is_operating.values == 0].mean()),
            cov_when_peak=float(ins[is_peak].mean()) if is_peak.any() else np.nan,
            cov_top_pred_decile=float(ins[top_dec].mean()) if top_dec.any() else np.nan,
            worst_group_coverage=float(gcov.min()), worst_group_gap=float((gcov - (1 - ALPHA)).abs().max()),
            fixed_partition_worst_gap=float((fx - (1 - ALPHA)).abs().max()),
            fixed_partition_worst_coverage=float(fx.min()), n_fixed_groups=int(len(fx)),
            mean_interval_width=float((hi - lo).mean()),
            width_when_peak=float((hi - lo)[is_peak].mean()) if is_peak.any() else np.nan,
            alert_precision=tp / max(tp + fp, 1), alert_recall=tp / max(tp + fn, 1),
            alerts_per_day=float(alert.mean() * 96), peak_threshold_kw=thr,
            peak_prevalence=float(is_peak.mean())))

rows, grows = [], []
q_all = build(latent_cols=LAT if has_lat else None)
for wtag in ["all_2021", "clean_Jul_Sep"]:
    sub = window(q_all, wtag)
    for h in _fe.HORIZONS:
        # (1) 단일 시간순 holdout (60/20/20)
        tr, ca, te = split(sub, F, h)
        eval_block(tr, ca, te, wtag, h, "holdout_60_20_20")
        # (2) OOF 폴드 — 원본구간의 피크 표본이 holdout 에서 20개뿐이라 표본을 늘린다.
        #     각 폴드: 과거의 앞 80% = train, 뒤 20% = calibration, 폴드구간 = test.
        need = list(dict.fromkeys(F + [f"y_h{h}"]))
        dd = sub.dropna(subset=need).reset_index(drop=True)
        for fi, (itr, ite) in enumerate(TimeSeriesSplit(n_splits=5).split(dd)):
            cut = int(len(itr) * .8)
            if cut < 500 or len(itr) - cut < MIN_CAL * 2:
                continue
            eval_block(dd.iloc[itr[:cut]], dd.iloc[itr[cut:]], dd.iloc[ite], wtag, h, "oof_folds", fi)

res = pd.DataFrame(rows)
base = res[res.grouping == "global"].set_index(["window", "eval_mode", "fold", "horizon_min"])
for c in ["cov_when_peak", "cov_operating", "fixed_partition_worst_gap", "mean_interval_width",
          "alert_precision", "alert_recall"]:
    res[f"d_{c}_vs_global"] = [r[c] - base.loc[(r.window, r.eval_mode, r.fold, r.horizon_min), c]
                              for _, r in res.iterrows()]
save_table(res, "07J_mondrian_conformal")
save_table(pd.DataFrame(grows), "07J_mondrian_group_detail")

# OOF 폴드 평균 요약 (표본 많은 쪽이 주장 근거)
agg = (res[res.eval_mode == "oof_folds"]
       .groupby(["window", "grouping", "horizon_min"])
       [["coverage", "cov_operating", "cov_idle", "cov_when_peak", "fixed_partition_worst_gap",
         "mean_interval_width", "alert_precision", "alert_recall", "n_test"]].mean().reset_index())
save_table(agg, "07J_mondrian_oof_summary")

fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
d = agg[agg.window == "clean_Jul_Sep"]
for gname in d.grouping.unique():
    s = d[d.grouping == gname].sort_values("horizon_min")
    axes[0].plot(s.horizon_min, s.cov_when_peak, "o-", label=gname)
    axes[1].plot(s.horizon_min, s.fixed_partition_worst_gap, "o-", label=gname)
    axes[2].plot(s.horizon_min, s.mean_interval_width, "o-", label=gname)
axes[0].axhline(1 - ALPHA, color="k", lw=.8, ls="--")
axes[0].set_title("실피크 구간 coverage"); axes[1].set_title("고정분할 최악그룹 이탈")
axes[2].set_title("평균 구간폭 (kw)")
for ax in axes:
    ax.set_xlabel("horizon (분)"); ax.legend(fontsize=6)
fig.suptitle("07J Mondrian conformal — clean_Jul_Sep", fontsize=9)
fig.tight_layout(); fig.savefig(fig_path("07J_mondrian_conformal")); plt.close(fig)
print(agg.to_string(index=False))
print(res[["window", "eval_mode", "horizon_min", "grouping", "coverage", "cov_operating", "cov_idle",
           "cov_when_peak", "fixed_partition_worst_gap", "mean_interval_width", "alert_precision",
           "alert_recall"]].to_string(index=False))
