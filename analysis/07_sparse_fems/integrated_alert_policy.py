"""07-O. 비용 x 다중 horizon 통합 경보 정책 (Task O).

구조: 1단계 조기경보(긴 horizon) -> 2단계 확인(짧은 horizon) -> 작업자 권고.
07H(비용 최적 임계)와 07L(다중 horizon)을 하나의 정책 공간으로 합쳐 훑는다.

정책 파라미터: 조기 horizon(h45/h60) x 조기 임계분위 x 확인 horizon(h15/h30) x 확인 임계분위 x 게이트 3종
게이트:
  none              : 신뢰도/전환 무관하게 확인 단계만 적용
  advisory_if_unsure: 확인 단계가 부정해도 LOW/OOD 이면 '작업자 권고'로 남긴다(경보 아님)
  confirm_only_if_unsure: HIGH 신뢰도면 확인 없이 바로 경보, MEDIUM 이하만 확인 요구

평가는 이벤트 단위. 경보도 연속 구간을 하나의 경보 이벤트로 묶어 오경보를 과대계상하지 않는다.
피크 정의: 직전 30일 p95(적응형). 비용비 C_FN/C_FP = 1,2,5,10,20,50,100.
산출 07O_policy_rows.csv 는 07Q(작업자 진단)가 재사용한다.
"""
import numpy as np, pandas as pd
import _fe
from _fe import build, save_table, fig_path, mpl, SEED
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.covariance import EmpiricalCovariance
plt = mpl()

F = _fe.LEVELS["L3_+production"]
RATIOS = [1, 2, 5, 10, 20, 50, 100]
QS = np.round(np.arange(.50, .995, .02), 3)


def qr(a):
    return HistGradientBoostingRegressor(loss="quantile", quantile=a, random_state=SEED)


q = build()
q["thr_adaptive"] = q.kw.shift(1).rolling(96 * 30, min_periods=96 * 7).quantile(.95)
need = F + ["thr_adaptive", "is_operating_lag4"] + [f"y_h{h}" for h in _fe.HORIZONS]
d = q.dropna(subset=need).reset_index(drop=True)
folds = list(TimeSeriesSplit(n_splits=5).split(d))
RAMP_CUT = float(np.quantile(d.iloc[folds[0][0]].kw_ramp4.abs(), .9))

pred = {h: np.full(len(d), np.nan) for h in _fe.HORIZONS}
lo4, hi4, ood = (np.full(len(d), np.nan) for _ in range(3))
band = np.full(len(d), "", dtype=object)
fid = np.full(len(d), -1)
for fi, (itr, ite) in enumerate(folds):
    fid[ite] = fi
    for h in _fe.HORIZONS:
        pred[h][ite] = HistGradientBoostingRegressor(random_state=SEED).fit(
            d.iloc[itr][F].values, d.iloc[itr][f"y_h{h}"].values).predict(d.iloc[ite][F].values)
    cv = EmpiricalCovariance().fit(d.iloc[itr][F].values)
    ood[ite] = cv.mahalanobis(d.iloc[ite][F].values)
    # CQR(90%) — 보정은 학습구간 뒤쪽 20% 로만
    cut = int(len(itr) * .8)
    a, c = d.iloc[itr[:cut]], d.iloc[itr[cut:]]
    m_lo, m_hi = qr(.05).fit(a[F].values, a.y_h4.values), qr(.95).fit(a[F].values, a.y_h4.values)
    E = np.maximum(m_lo.predict(c[F].values) - c.y_h4.values, c.y_h4.values - m_hi.predict(c[F].values))
    Q = float(np.quantile(E, min(1.0, .9 * (1 + 1 / len(E)))))
    lo4[ite], hi4[ite] = m_lo.predict(d.iloc[ite][F].values) - Q, m_hi.predict(d.iloc[ite][F].values) + Q
    wc = (m_hi.predict(c[F].values) + Q) - (m_lo.predict(c[F].values) - Q)
    cuts, ood_cut = np.quantile(wc, [.5, .8]), float(np.quantile(ood[itr[itr < len(d)]][~np.isnan(ood[itr])], .99)) \
        if np.isfinite(ood[itr]).any() else np.inf
    w = hi4[ite] - lo4[ite]
    band[ite] = np.where(ood[ite] > ood_cut, "OOD",
                         np.where(w <= cuts[0], "HIGH", np.where(w <= cuts[1], "MEDIUM", "LOW")))

m = fid >= 0
o = d[m].copy().reset_index(drop=True)
for h in _fe.HORIZONS:
    o[f"pred_h{h}"] = pred[h][m]
o["fold"], o["ood_maha"], o["band"] = fid[m], ood[m], band[m]
o["int_lo"], o["int_hi"] = lo4[m], hi4[m]
o["int_width"] = o.int_hi - o.int_lo
o["abs_err_h4"] = np.abs(o.y_h4 - o.pred_h4)
o["peak_now"] = (o.kw >= o.thr_adaptive).astype(int)
for h in _fe.HORIZONS:
    o[f"peak_h{h}"] = (o[f"y_h{h}"] >= o.thr_adaptive).astype(int)
sw = o.is_operating.values != o.is_operating_lag4.values
o["regime"] = np.where(sw | (o.kw_ramp4.abs().values >= RAMP_CUT), "transition",
                       np.where(o.is_operating.values == 0, "stop", "normal_run"))
o["ood_flag"] = (o.band == "OOD").astype(int)
save_table(o[["ts15", "fold", "clean", "kw", "thr_adaptive", "peak_now"] +
             [f"y_h{h}" for h in _fe.HORIZONS] + [f"pred_h{h}" for h in _fe.HORIZONS] +
             [f"peak_h{h}" for h in _fe.HORIZONS] +
             ["int_lo", "int_hi", "int_width", "band", "ood_maha", "ood_flag", "regime",
              "is_operating", "생산량", "kw_ramp4", "kw_std4", "abs_err_h4"]], "07O_policy_rows")


def runs(mask):
    """연속 True 구간의 (시작,끝) 목록."""
    idx = np.flatnonzero(mask)
    if not len(idx):
        return []
    out, s, p = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - p <= 2:
            p = i; continue
        out.append((s, p)); s, p = i, i
    out.append((s, p))
    return out


def evaluate(sub, alert, advisory, hmax):
    """이벤트 단위 포착 + 스텝 단위 오경보.

    경보 구간을 묶어서 오경보를 세면 임계를 낮춰 전구간을 덮는 편법이 이득을 본다.
    따라서 오경보는 스텝 단위로 센다: 시각 t 의 경보는 (t, t+hmax] 안에 피크가 있으면 적중,
    없으면 오경보. 선행시간도 경보 horizon 을 넘을 수 없으므로 hmax 로 제한한다.
    """
    ev = runs(sub.peak_now.values == 1)
    al = runs(alert)
    days = len(sub) / 96
    k = int(hmax // 15)
    peak = sub.peak_now.values == 1
    # 스텝 t 의 향후 k 스텝 내 피크 존재 여부
    fut = np.zeros(len(sub), bool)
    for j in range(1, k + 1):
        fut[:len(sub) - j] |= peak[j:]
    tp_steps = int((alert & fut).sum()); fp_steps = int((alert & ~fut).sum())
    det, leads = 0, []
    for s, e in ev:
        w = alert[max(0, s - k):s]
        if w.any():
            det += 1
            first = max(0, s - k) + int(np.flatnonzero(w)[0])
            leads.append(min(hmax, (s - first) * 15))
    return dict(n_events=len(ev), events_detected=det, missed_events=len(ev) - det,
                event_recall=det / max(len(ev), 1),
                mean_lead_min=float(np.mean(leads)) if leads else np.nan,
                median_lead_min=float(np.median(leads)) if leads else np.nan,
                lead_ge_30min_share=float(np.mean([l >= 30 for l in leads])) if leads else np.nan,
                n_alert_events=len(al), alert_events_per_day=len(al) / days,
                false_alert_steps=fp_steps, false_alerts_per_day=fp_steps / days,
                step_precision=tp_steps / max(tp_steps + fp_steps, 1),
                precision_alert_events=np.nan,
                advisories_per_day=float(advisory.sum()) / days,
                alert_steps_per_day=float(alert.sum()) / days)


rows = []
for wtag, sub in [("oof_all_2021", o), ("oof_clean_Jul_Sep", o[o.clean].reset_index(drop=True))]:
    sub = sub.reset_index(drop=True)
    unsure = sub.band.isin(["LOW", "OOD"]).values
    base_ev = len(runs(sub.peak_now.values == 1))
    # 기준선: 단일 horizon
    for h in _fe.HORIZONS:
        for qe in QS:
            a = (sub[f"pred_h{h}"] >= np.quantile(sub[f"pred_h{h}"], qe)).values
            rows.append(dict(window=wtag, policy=f"single_h{h*15}", early_horizon_min=h * 15,
                             early_q=qe, confirm_horizon_min=np.nan, confirm_q=np.nan, gate="none",
                             confirm_rejection_rate=0.0,
                             **evaluate(sub, a, np.zeros(len(sub), bool), h * 15)))
    # 통합: 조기경보 + 확인
    for he in [3, 4]:
        for hc in [1, 2]:
            for qe in QS:
                early = (sub[f"pred_h{he}"] >= np.quantile(sub[f"pred_h{he}"], qe)).values
                for qc in QS:
                    conf = (sub[f"pred_h{hc}"] >= np.quantile(sub[f"pred_h{hc}"], qc)).values
                    for gate in ["none", "advisory_if_unsure", "confirm_only_if_unsure"]:
                        if gate == "none":
                            alert = early & conf
                            adv = np.zeros(len(sub), bool)
                        elif gate == "advisory_if_unsure":
                            alert = early & conf
                            adv = early & ~conf & unsure
                        else:
                            alert = early & (conf | ~unsure)
                            adv = early & ~conf & unsure
                        rej = float((early & ~conf).sum() / max(early.sum(), 1))
                        rows.append(dict(window=wtag, policy=f"escalate_h{he*15}_to_h{hc*15}",
                                         early_horizon_min=he * 15, early_q=qe,
                                         confirm_horizon_min=hc * 15, confirm_q=qc, gate=gate,
                                         confirm_rejection_rate=rej,
                                         **evaluate(sub, alert, adv, he * 15)))
cur = pd.DataFrame(rows)
for r in RATIOS:
    # 정규화 기대비용: 오경보는 스텝 단위, 미탐은 이벤트 단위. 전부 놓친 경우 = 1
    cur[f"cost_r{r}"] = (cur.false_alert_steps + r * cur.missed_events) / (r * cur.n_events)
save_table(cur, "07O_integrated_policy_curve")

best = []
for (wtag), g in cur.groupby("window"):
    for r in RATIOS:
        for scope, gg in [("all_policies", g), ("single_horizon_only", g[g.policy.str.startswith("single")]),
                          ("escalation_only", g[g.policy.str.startswith("escalate")])]:
            i = gg[f"cost_r{r}"].idxmin()
            row = gg.loc[i]
            best.append(dict(window=wtag, cost_ratio_FN_FP=r, scope=scope, policy=row.policy,
                             early_horizon_min=row.early_horizon_min, early_q=row.early_q,
                             confirm_horizon_min=row.confirm_horizon_min, confirm_q=row.confirm_q,
                             gate=row.gate, normalized_expected_cost=row[f"cost_r{r}"],
                             event_recall=row.event_recall, missed_events=row.missed_events,
                             mean_lead_min=row.mean_lead_min, lead_ge_30min_share=row.lead_ge_30min_share,
                             false_alerts_per_day=row.false_alerts_per_day,
                             alert_events_per_day=row.alert_events_per_day,
                             step_precision=row.step_precision,
                             advisories_per_day=row.advisories_per_day,
                             confirm_rejection_rate=row.confirm_rejection_rate))
bp = pd.DataFrame(best)
save_table(bp, "07O_recommended_operating_point")

# 에스컬레이션이 단일 horizon 을 이기는가 (비용 기준)
cmp_rows = []
for wtag in cur.window.unique():
    for r in RATIOS:
        s = bp[(bp.window == wtag) & (bp.cost_ratio_FN_FP == r)].set_index("scope")
        cmp_rows.append(dict(window=wtag, cost_ratio_FN_FP=r,
                             cost_single=float(s.loc["single_horizon_only", "normalized_expected_cost"]),
                             cost_escalation=float(s.loc["escalation_only", "normalized_expected_cost"]),
                             escalation_wins=bool(s.loc["escalation_only", "normalized_expected_cost"] <
                                                  s.loc["single_horizon_only", "normalized_expected_cost"]),
                             best_policy=s.loc["all_policies", "policy"],
                             best_gate=s.loc["all_policies", "gate"],
                             recall_single=float(s.loc["single_horizon_only", "event_recall"]),
                             recall_escalation=float(s.loc["escalation_only", "event_recall"]),
                             lead_single=float(s.loc["single_horizon_only", "mean_lead_min"]),
                             lead_escalation=float(s.loc["escalation_only", "mean_lead_min"])))
save_table(pd.DataFrame(cmp_rows), "07O_escalation_vs_single")

fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
b = bp[(bp.window == "oof_clean_Jul_Sep") & (bp.scope == "all_policies")]
axes[0].plot(b.cost_ratio_FN_FP, b.normalized_expected_cost, "o-")
axes[0].set_xscale("log"); axes[0].set_xlabel("C_FN / C_FP"); axes[0].set_ylabel("정규화 기대비용")
axes[0].set_title("비용비별 최적 정책 비용")
axes[1].plot(b.cost_ratio_FN_FP, b.event_recall, "o-", label="이벤트 포착률")
axes[1].plot(b.cost_ratio_FN_FP, b.lead_ge_30min_share, "s-", label="선행 30분 이상")
axes[1].set_xscale("log"); axes[1].legend(fontsize=7); axes[1].set_title("운영지표")
axes[2].plot(b.cost_ratio_FN_FP, b.false_alerts_per_day, "o-", label="오경보/일")
axes[2].plot(b.cost_ratio_FN_FP, b.advisories_per_day, "s-", label="작업자 권고/일")
axes[2].set_xscale("log"); axes[2].legend(fontsize=7); axes[2].set_title("운영 부담")
fig.tight_layout(); fig.savefig(fig_path("07O_integrated_alert_policy")); plt.close(fig)
print(bp[(bp.window == "oof_clean_Jul_Sep") & (bp.scope == "all_policies")].to_string(index=False))
print(pd.DataFrame(cmp_rows).to_string(index=False))
