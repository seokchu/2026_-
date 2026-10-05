"""Baseline v1 — Module H/J. 2축 진단(신뢰도 x 설명가능성) + 작업자 출력.

금지: 설비 지정·정지·생산 조정 등 물리적 조치 문장. 본 모듈은 그런 문장을 만들지 않는다.
허용 정보부족 유형(후보 '정보 종류' 이며 특정 센서 지목이 아니다):
  설비 가동상태 / 공정단계 / 셋업·전환 로그 / 제품·로트 문맥 / 정비·설비상태 이력
"""
import numpy as np, pandas as pd
import policy as PL

FORBIDDEN = ["끄", "중지", "미루", "줄이", "정지하", "가동 중단", "설비 A", "열처리"]
OUT_COLS = (["timestamp"] + [f"forecast_{h*15}m" for h in (1, 2, 3, 4)] +
            [f"interval_low_{h*15}m" for h in (1, 2, 3, 4)] +
            [f"interval_high_{h*15}m" for h in (1, 2, 3, 4)] +
            [f"reliability_{h*15}m" for h in (1, 2, 3, 4)] +
            ["current_peak_threshold", "peak_risk_score_60m", "early_peak_risk",
             "confirmed_peak_risk", "operating_regime", "transition_like_flag", "ood_flag",
             "uncertainty_explained", "information_gap_flag", "advisory_level",
             "diagnostic_reason", "recommended_information_type", "actual_power_60m",
             "abs_error_60m"])


def apply_policy(o, thr, cfg, bp):
    """기본 비용비의 비용최적 정책을 적용해 조기/확인 경보 마스크를 만든다."""
    r = cfg["policy"]["default_cost_ratio"]
    P = bp[(bp.window == "clean_Jul_Sep") & (bp.scope == "all_policies") &
           (bp.cost_ratio_FN_FP == r)].iloc[0]
    s = o[o.clean == True].sort_values("ts15").reset_index(drop=True)
    tm = PL.thr_map(thr)
    he = int(P.early_horizon_min)
    early = PL.mask_at(s, tm, he, float(P.early_q))
    if np.isfinite(P.confirm_horizon_min):
        hc = int(P.confirm_horizon_min)
        conf = PL.mask_at(s, tm, hc, float(P.confirm_q))
    else:
        hc, conf = he, np.ones(len(s), bool)
    unsure = s[f"band_h{he // 15}"].isin(cfg["diagnosis"]["uncertain_bands"]).values
    if P.gate == "confirm_only_if_unsure":
        alert = early & (conf | ~unsure)
    else:
        alert = early & conf
    return s, P, he, hc, early, conf, alert, unsure


def build_output(o, thr, cfg, bp):
    s, P, he, hc, early, conf, alert, unsure = apply_policy(o, thr, cfg, bp)
    ramp_hi = float(np.quantile(s.ramp_cut_train, .5))        # 폴드 TRAIN p90 컷의 대표값
    trans = (s.regime == "TRANSITION_LIKE").values
    # 축2: 불확실성을 현재 관측으로 설명할 수 있는가
    explained = (trans | (s.kw_ramp4.abs().values >= ramp_hi) |
                 (s.is_operating.values == 0) | (early & ~conf))
    gap = unsure & ~explained
    lv = np.where(s.ood_flag.values == 1, "HUMAN_REVIEW",
         np.where(alert & ~unsure, "CONFIRMED_WARNING",
         np.where(alert & unsure, "LOW_CONFIDENCE_WARNING",
         np.where(gap, "INFORMATION_GAP",
         np.where(early, "EARLY_WATCH", "NORMAL")))))
    reasons, infotype = [], []
    for i, r in s.iterrows():
        rs = []
        if r.ood_flag == 1:
            rs.append("학습분포 밖의 입력(OOD 거리 상위 1%)")
        if unsure[i]:
            rs.append("예측구간이 넓어 신뢰도 낮음")
        if abs(r.kw_ramp4) >= ramp_hi:
            rs.append("최근 1시간 부하 변화가 학습 상위 10% 수준")
        if trans[i]:
            rs.append("전환형 고변동 레짐 감지")
        if early[i] and not conf[i]:
            rs.append("60분 시점은 피크 위험, 15분 확인에서는 미확인 — 시점 불일치")
        if early[i] and conf[i]:
            rs.append("조기·확인 시점 모두 피크 위험")
        if r.is_operating == 0 and unsure[i]:
            rs.append("비조업 구간인데 불확실성 큼")
        if gap[i]:
            rs.append("현재 변수로는 불확실성의 원인을 구분할 수 없음")
        if not rs:
            rs.append("예측구간 정상 — 통상 감시")
        reasons.append(" / ".join(rs))
        if gap[i] or r.ood_flag == 1:
            infotype.append("설비 가동상태 또는 공정단계 정보가 있으면 원인 구분 가능")
        elif trans[i]:
            infotype.append("셋업·전환 로그가 있으면 전환 원인 확인 가능")
        elif unsure[i] and r.is_operating == 0:
            infotype.append("설비 가동상태 정보 확인 필요")
        elif unsure[i]:
            infotype.append("제품·로트 문맥 정보가 있으면 변동 설명 가능")
        else:
            infotype.append("추가 정보 불필요")
    out = pd.DataFrame({"timestamp": s.ts15, "current_peak_threshold": s.thr_adaptive.round(1),
                        "peak_risk_score_60m": s.peak_risk_score_h4.round(3),
                        "early_peak_risk": early.astype(int),
                        "confirmed_peak_risk": alert.astype(int),
                        "operating_regime": s.regime, "transition_like_flag": trans.astype(int),
                        "ood_flag": s.ood_flag, "uncertainty_explained": explained.astype(int),
                        "information_gap_flag": gap.astype(int), "advisory_level": lv,
                        "diagnostic_reason": reasons, "recommended_information_type": infotype,
                        "actual_power_60m": s.y_h4.round(2),
                        "abs_error_60m": s.abs_err_h4.round(2)})
    for h in (1, 2, 3, 4):
        out[f"forecast_{h*15}m"] = s[f"pred_h{h}"].round(2)
        out[f"interval_low_{h*15}m"] = s[f"int_lo_h{h}"].round(2)
        out[f"interval_high_{h*15}m"] = s[f"int_hi_h{h}"].round(2)
        out[f"reliability_{h*15}m"] = s[f"band_h{h}"]
    out = out[OUT_COLS]
    meta = dict(policy=P.policy, gate=P.gate, early_horizon_min=he, confirm_horizon_min=hc,
                early_q=float(P.early_q), confirm_q=float(P.confirm_q),
                cost_ratio=cfg["policy"]["default_cost_ratio"])
    return out, s, meta, dict(early=early, conf=conf, alert=alert, unsure=unsure,
                              explained=explained, gap=gap)


def routing_2x2(s, fl):
    """2x2 라우팅: 신뢰도(신뢰/불확실) x 설명가능성(설명/미설명)."""
    rows = []
    fail = 1 - s.covered_h4.values
    for tag, m in [("A_reliable_explained", ~fl["unsure"] & fl["explained"]),
                   ("B_uncertain_explained", fl["unsure"] & fl["explained"]),
                   ("C_uncertain_unexplained(information_gap)", fl["unsure"] & ~fl["explained"]),
                   ("D_reliable_unexplained", ~fl["unsure"] & ~fl["explained"])]:
        if m.sum() == 0:
            continue
        rows.append(dict(cell=tag, n=int(m.sum()), share=float(m.mean()),
                         mae_h60=float(s.abs_err_h4[m].mean()),
                         coverage_h60=float(1 - fail[m].mean()),
                         mean_interval_width=float(s.int_width_h4[m].mean()),
                         peak_prevalence=float(s.peak_h4[m].mean()),
                         transition_share=float((s.regime[m] == "TRANSITION_LIKE").mean()),
                         ood_share=float(s.ood_flag[m].mean()),
                         per_day=float(m.sum() / (len(s) / 96))))
    return pd.DataFrame(rows)


def gap_validation(s, fl):
    from sklearn.metrics import average_precision_score, roc_auc_score
    hi = (s.abs_err_h4 >= np.quantile(s.abs_err_h4, .8)).astype(int).values
    fail = 1 - s.covered_h4.values
    rows = []
    cands = [("information_gap_flag", fl["gap"].astype(int)),
             ("ood_flag", s.ood_flag.values),
             ("band_LOW_or_OOD", fl["unsure"].astype(int)),
             ("transition_like_flag", (s.regime == "TRANSITION_LIKE").astype(int).values),
             ("int_width_top20", (s.int_width_h4 >= np.quantile(s.int_width_h4, .8)).astype(int).values)]
    for name, f in cands:
        if f.sum() == 0:
            rows.append(dict(flag=name, n_flagged=0, share=0.0)); continue
        rows.append(dict(flag=name, n_flagged=int(f.sum()), share=float(f.mean()),
                         mae_flagged=float(s.abs_err_h4[f == 1].mean()),
                         mae_unflagged=float(s.abs_err_h4[f == 0].mean()),
                         mae_ratio=float(s.abs_err_h4[f == 1].mean() / s.abs_err_h4[f == 0].mean()),
                         coverage_flagged=float(1 - fail[f == 1].mean()),
                         coverage_unflagged=float(1 - fail[f == 0].mean()),
                         peak_prevalence_flagged=float(s.peak_h4[f == 1].mean()),
                         auc_for_high_error=float(roc_auc_score(hi, f)),
                         ap_for_high_error=float(average_precision_score(hi, f)),
                         lift_over_base=float(average_precision_score(hi, f) / hi.mean())))
    return pd.DataFrame(rows)


def advisory_distribution(out, s):
    d = out.copy(); d["abs_err"] = s.abs_err_h4.values; d["width"] = s.int_width_h4.values
    d["peak"] = s.peak_h4.values
    g = d.groupby("advisory_level").agg(
        n=("timestamp", "size"), share=("timestamp", lambda x: len(x) / len(d)),
        per_day=("timestamp", lambda x: len(x) / (len(d) / 96)),
        mae_h60=("abs_err", "mean"), mean_interval_width=("width", "mean"),
        peak_prevalence=("peak", "mean"),
        information_gap_share=("information_gap_flag", "mean"),
        transition_share=("transition_like_flag", "mean")).reset_index()
    return g


def sample_cases(out, n_min=50, seed=20260926):
    parts = [g.sample(min(len(g), 2), random_state=seed) for _, g in
             out.groupby(["advisory_level", "operating_regime", "reliability_60m"])]
    samp = pd.concat(parts)
    if len(samp) < n_min:
        samp = pd.concat([samp, out.drop(samp.index).sample(n_min - len(samp), random_state=seed)])
    return samp.sort_values("timestamp")


def forbidden_hits(out):
    txt = (out.diagnostic_reason.astype(str) + " " + out.recommended_information_type.astype(str))
    return int(sum(txt.str.contains(w).sum() for w in FORBIDDEN))
