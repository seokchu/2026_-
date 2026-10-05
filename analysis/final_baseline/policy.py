"""Baseline v1 — Module F/G. 비용 인식 h60->h15 에스컬레이션 + 비용x선행시간 파레토.

오경보는 스텝 단위로 센다(경보 구간을 묶으면 임계를 낮춰 전구간을 덮는 편법이 이득을 본다).
시각 t 의 경보는 (t, t+hmax] 안에 피크가 있으면 적중, 없으면 오경보.
경보 임계는 분위를 각 폴드 CAL 예측분포에서 값으로 변환해 적용한다(TEST 분포 미사용).
"""
import numpy as np, pandas as pd


def runs(mask, gap=2):
    idx = np.flatnonzero(mask)
    if not len(idx):
        return []
    out, s, p = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - p <= gap:
            p = i; continue
        out.append((s, p)); s, p = i, i
    out.append((s, p))
    return out


def evaluate(sub, alert, advisory, hmax, min_lead):
    ev, al = runs(sub.peak_now.values == 1), runs(alert)
    days = len(sub) / 96
    k = int(hmax // 15)
    peak = sub.peak_now.values == 1
    fut = np.zeros(len(sub), bool)
    for j in range(1, k + 1):
        fut[:len(sub) - j] |= peak[j:]
    tp = int((alert & fut).sum()); fp = int((alert & ~fut).sum())
    det, leads = 0, []
    for s, _ in ev:
        w = alert[max(0, s - k):s]
        if w.any():
            det += 1
            first = max(0, s - k) + int(np.flatnonzero(w)[0])
            leads.append(min(hmax, (s - first) * 15))
    return dict(n_events=len(ev), events_detected=det, missed_events=len(ev) - det,
                event_recall=det / max(len(ev), 1),
                mean_lead_min=float(np.mean(leads)) if leads else np.nan,
                median_lead_min=float(np.median(leads)) if leads else np.nan,
                lead_ge_min_share=float(np.mean([l >= min_lead for l in leads])) if leads else np.nan,
                n_alert_events=len(al), alert_events_per_day=len(al) / days,
                false_alert_steps=fp, false_alerts_per_day=fp / days,
                step_precision=tp / max(tp + fp, 1),
                confirmed_alerts_per_day=float(alert.sum()) / days,
                advisories_per_day=float(advisory.sum()) / days)


def thr_map(thr):
    return {(r.fold, int(r.horizon_min), float(r.q)): r.threshold_kw for r in thr.itertuples()}


def mask_at(sub, tm, hmin, qv):
    """폴드별 CAL 임계값을 각 행에 매핑해 경보 마스크를 만든다."""
    t = np.array([tm[(f, hmin, qv)] for f in sub.fold.values])
    return (sub[f"pred_h{hmin // 15}"].values >= t)


def sweep(o, thr, cfg):
    pc = cfg["policy"]
    tm = thr_map(thr)
    he, hc = pc["early_horizon_min"], pc["confirm_horizon_min"]
    ml = pc["minimum_lead_minutes"]
    rows = []
    for wtag, sub in [("all_2021", o), ("clean_Jul_Sep", o[o.clean == True])]:
        sub = sub.sort_values("ts15").reset_index(drop=True)
        unsure = sub[f"band_h{he // 15}"].isin(cfg["diagnosis"]["uncertain_bands"]).values
        zero = np.zeros(len(sub), bool)
        for hmin in [15, 30, 45, 60]:
            for qv in pc["threshold_grid"]:
                a = mask_at(sub, tm, hmin, qv)
                rows.append(dict(window=wtag, policy=f"single_h{hmin}", early_horizon_min=hmin,
                                 early_q=qv, confirm_horizon_min=np.nan, confirm_q=np.nan,
                                 gate="none", confirm_rejection_rate=0.0,
                                 **evaluate(sub, a, zero, hmin, ml)))
        for qe in pc["threshold_grid"]:
            early = mask_at(sub, tm, he, qe)
            for qc in pc["threshold_grid"]:
                conf = mask_at(sub, tm, hc, qc)
                rej = float((early & ~conf).sum() / max(early.sum(), 1))
                for gate in pc["gates"]:
                    if gate == "none":
                        alert, adv = early & conf, zero
                    elif gate == "advisory_if_unsure":
                        alert, adv = early & conf, early & ~conf & unsure
                    else:
                        alert, adv = early & (conf | ~unsure), early & ~conf & unsure
                    rows.append(dict(window=wtag, policy=f"escalate_h{he}_to_h{hc}",
                                     early_horizon_min=he, early_q=qe, confirm_horizon_min=hc,
                                     confirm_q=qc, gate=gate, confirm_rejection_rate=rej,
                                     **evaluate(sub, alert, adv, he, ml)))
    cur = pd.DataFrame(rows)
    for r in pc["cost_ratios"]:
        cur[f"cost_r{r}"] = (cur.false_alert_steps + r * cur.missed_events) / (r * cur.n_events)
    cur["within_advisory_capacity"] = cur.advisories_per_day <= pc["operator_advisory_capacity"]
    return cur


def best_points(cur, cfg):
    rows = []
    for wtag, g in cur.groupby("window"):
        for r in cfg["policy"]["cost_ratios"]:
            for scope, gg in [("all_policies", g),
                              ("single_horizon_only", g[g.policy.str.startswith("single")]),
                              ("escalation_only", g[g.policy.str.startswith("escalate")])]:
                row = gg.loc[gg[f"cost_r{r}"].idxmin()]
                rows.append(dict(window=wtag, cost_ratio_FN_FP=r, scope=scope,
                                 normalized_expected_cost=float(row[f"cost_r{r}"]),
                                 **{c: row[c] for c in
                                    ["policy", "early_horizon_min", "early_q", "confirm_horizon_min",
                                     "confirm_q", "gate", "event_recall", "missed_events",
                                     "mean_lead_min", "lead_ge_min_share", "false_alerts_per_day",
                                     "confirmed_alerts_per_day", "advisories_per_day",
                                     "step_precision", "confirm_rejection_rate", "n_events"]}))
    return pd.DataFrame(rows)


def escalation_vs_single(bp, cfg):
    rows = []
    for wtag in bp.window.unique():
        for r in cfg["policy"]["cost_ratios"]:
            s = bp[(bp.window == wtag) & (bp.cost_ratio_FN_FP == r)].set_index("scope")
            e, g = s.loc["escalation_only"], s.loc["single_horizon_only"]
            rows.append(dict(window=wtag, cost_ratio_FN_FP=r,
                             cost_single=float(g.normalized_expected_cost),
                             cost_escalation=float(e.normalized_expected_cost),
                             escalation_wins=bool(e.normalized_expected_cost < g.normalized_expected_cost),
                             recall_single=float(g.event_recall), recall_escalation=float(e.event_recall),
                             lead_single=float(g.mean_lead_min), lead_escalation=float(e.mean_lead_min),
                             fa_single=float(g.false_alerts_per_day),
                             fa_escalation=float(e.false_alerts_per_day),
                             best_policy=s.loc["all_policies", "policy"],
                             best_gate=s.loc["all_policies", "gate"]))
    return pd.DataFrame(rows)


def pareto(cur, cfg, window="clean_Jul_Sep"):
    """2목적: 정규화 기대비용(최소) x 평균 선행시간(최대). 가중합으로 접지 않는다."""
    r = cfg["policy"]["default_cost_ratio"]
    s = cur[(cur.window == window) & cur.mean_lead_min.notna() &
            (cur.event_recall > 0)].copy()
    s["cost"] = s[f"cost_r{r}"]
    s = s.sort_values(["cost", "mean_lead_min"], ascending=[True, False]).reset_index(drop=True)
    keep, best_lead = [], -np.inf
    for i, row in s.iterrows():
        if row.mean_lead_min > best_lead:
            keep.append(i); best_lead = row.mean_lead_min
    out = s.loc[keep, ["policy", "early_horizon_min", "early_q", "confirm_horizon_min",
                       "confirm_q", "gate", "cost", "mean_lead_min", "event_recall",
                       "false_alerts_per_day", "lead_ge_min_share", "advisories_per_day",
                       "confirm_rejection_rate", "within_advisory_capacity"]].copy()
    out.insert(0, "cost_ratio_FN_FP", r)
    out.insert(0, "window", window)
    return out.sort_values("cost").reset_index(drop=True), s
