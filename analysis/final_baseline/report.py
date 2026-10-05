"""Baseline v1 — 통합 지표표 / 그림 / 기계판독 요약."""
import json
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score
from common import fig_path, mpl

H = [1, 2, 3, 4]


def consolidated(o, rel, bp, pf, gap, dist, cfg, window="clean_Jul_Sep"):
    """horizon 별 예측/신뢰도/피크/레짐 + 정책·진단 요약을 한 표로."""
    s = o[o.clean == True] if window == "clean_Jul_Sep" else o
    s = s.reset_index(drop=True)
    r = cfg["policy"]["default_cost_ratio"]
    P = bp[(bp.window == window) & (bp.scope == "all_policies") & (bp.cost_ratio_FN_FP == r)].iloc[0]
    gaprow = gap[gap.flag == "information_gap_flag"].iloc[0]
    rows = []
    for h in H:
        e = s[f"abs_err_h{h}"].values
        pers = np.abs(s[f"y_h{h}"] - s.kw).mean()
        rb = rel[(rel.window == window) & (rel.horizon_min == h * 15)].set_index("band")
        rows.append(dict(
            window=window, horizon_min=h * 15, feature_mode=s[f"mode_h{h}"].mode()[0],
            mae=float(e.mean()), rmse=float(np.sqrt((e ** 2).mean())),
            nmae=float(e.mean() / s[f"y_h{h}"].abs().mean()),
            persistence_improvement=float(1 - e.mean() / pers),
            coverage=float(rb.loc["ALL", "coverage"]),
            coverage_peak_only=float(rb.loc["ALL", "coverage_peak_only"]),
            mean_interval_width=float(rb.loc["ALL", "mean_interval_width"]),
            mae_band_HIGH=float(rb.loc["HIGH", "mae"]) if "HIGH" in rb.index else np.nan,
            mae_band_MEDIUM=float(rb.loc["MEDIUM", "mae"]) if "MEDIUM" in rb.index else np.nan,
            mae_band_LOW=float(rb.loc["LOW", "mae"]) if "LOW" in rb.index else np.nan,
            mae_band_OOD=float(rb.loc["OOD", "mae"]) if "OOD" in rb.index else np.nan,
            peak_prevalence=float(s[f"peak_h{h}"].mean()),
            peak_ap=float(average_precision_score(s[f"peak_h{h}"], s[f"pred_h{h}"])),
            mae_low_load=float(e[s.regime == "LOW_LOAD"].mean()),
            mae_stable=float(e[s.regime == "STABLE_OPERATION"].mean()),
            mae_transition=float(e[s.regime == "TRANSITION_LIKE"].mean()),
            peak_prevalence_transition=float(s[s.regime == "TRANSITION_LIKE"][f"peak_h{h}"].mean()),
            policy_cost_ratio=r, policy=P.policy, policy_gate=P.gate,
            policy_event_recall=float(P.event_recall), policy_missed_events=float(P.missed_events),
            policy_false_alerts_per_day=float(P.false_alerts_per_day),
            policy_mean_lead_min=float(P.mean_lead_min),
            policy_lead_ge_30min_share=float(P.lead_ge_min_share),
            policy_normalized_cost=float(P.normalized_expected_cost),
            information_gap_rate=float(gaprow.share),
            operator_review_rate=float((dist[dist.advisory_level.isin(
                ["HUMAN_REVIEW", "INFORMATION_GAP"])].n.sum()) / dist.n.sum()),
            information_gap_mae=float(gaprow.mae_flagged),
            information_gap_coverage=float(gaprow.coverage_flagged)))
    return pd.DataFrame(rows)


def core_vs_public(pf, rel, o, window="clean_Jul_Sep"):
    """CORE vs PUBLIC: 평균성능 개선 대비 남는 신뢰도 문제."""
    s = (o[o.clean == True] if window == "clean_Jul_Sep" else o).reset_index(drop=True)
    rows = []
    for h in H:
        rec = dict(window=window, horizon_min=h * 15)
        for m in ("CORE", "PUBLIC"):
            c = f"pred_{m}_h{h}"
            if c not in s:
                continue
            e = (s[f"y_h{h}"] - s[c]).abs()
            rec[f"mae_{m.lower()}"] = float(e.mean())
            rec[f"peak_ap_{m.lower()}"] = float(average_precision_score(s[f"peak_h{h}"], s[c]))
        if "mae_public" in rec:
            rec["rel_mae_gain_public"] = (rec["mae_core"] - rec["mae_public"]) / rec["mae_core"]
        rb = rel[(rel.window == window) & (rel.horizon_min == h * 15)].set_index("band")
        rec["low_or_ood_share_selected"] = float(
            sum(rb.loc[b, "share"] for b in ("LOW", "OOD") if b in rb.index))
        rec["coverage_peak_only_selected"] = float(rb.loc["ALL", "coverage_peak_only"])
        rec["mae_transition_selected"] = float(s[s.regime == "TRANSITION_LIKE"][f"abs_err_h{h}"].mean())
        rec["selected_mode"] = s[f"mode_h{h}"].mode()[0]
        rows.append(rec)
    return pd.DataFrame(rows)


def figures(o, rel, par, cvp, dist, rout, meta, cfg):
    plt = mpl()
    pre = cfg["output"]["figure_prefix"]
    s = o[o.clean == True].sort_values("ts15").reset_index(drop=True)

    # FIG 1 — 시스템 구조
    fig, ax = plt.subplots(figsize=(12, 2.8)); ax.axis("off"); ax.grid(False)
    boxes = ["저계측 입력\n(집계전력·달력·생산)", "다중시점 예측\nh15~h60", "신뢰도\nCQR 구간+밴드",
             "적응형 피크위험\n직전30일 p95", "비용인식 경보\nh60→h15 확인", "정보부족 진단\n2축 라우팅",
             "작업자 전달\n권고등급·사유"]
    for i, b in enumerate(boxes):
        x = i / len(boxes)
        ax.add_patch(plt.Rectangle((x + .005, .25), 1 / len(boxes) - .02, .5,
                                   fc="#eaf2fb", ec="#27b"))
        ax.text(x + (1 / len(boxes)) / 2 - .005, .5, b, ha="center", va="center", fontsize=7.5)
        if i < len(boxes) - 1:
            ax.annotate("", xy=(x + 1 / len(boxes) - .013, .5), xytext=(x + 1 / len(boxes) - .02, .5),
                        arrowprops=dict(arrowstyle="->", color="#333"))
    ax.set_title("FIGURE 1. Baseline v1 시스템 구조 (설비 제어 없음 — 작업자 전달에서 종료)")
    fig.tight_layout(); fig.savefig(fig_path(f"{pre}_fig1_architecture")); plt.close(fig)

    # FIG 2 — 실측 vs 예측 + 구간 + 적응형 임계
    seg = s.iloc[int(len(s) * .55):int(len(s) * .55) + 96 * 3]
    fig, ax = plt.subplots(figsize=(11, 3.4))
    t = seg.ts15 + pd.Timedelta(minutes=60)
    ax.plot(t, seg.y_h4, lw=1.2, color="#222", label="실측 수요(t+60분)")
    ax.plot(t, seg.pred_h4, lw=1.2, color="#27b", label="예측(h60)")
    ax.fill_between(t, seg.int_lo_h4, seg.int_hi_h4, color="#27b", alpha=.18, label="CQR 90% 구간")
    ax.plot(t, seg.thr_adaptive, "--", color="#c33", lw=1, label="적응형 피크 임계(직전30일 p95)")
    ax.set_ylabel("수요 (kw)"); ax.legend(fontsize=7)
    ax.set_title("FIGURE 2. 원본구간 3일 — 예측·구간·적응형 피크 임계")
    fig.tight_layout(); fig.savefig(fig_path(f"{pre}_fig2_forecast_interval")); plt.close(fig)

    # FIG 3 — 밴드별 MAE / coverage
    rb = rel[(rel.window == "clean_Jul_Sep") & (rel.horizon_min == 60) &
             (rel.band != "ALL")].set_index("band").reindex(["HIGH", "MEDIUM", "LOW", "OOD"]).dropna(how="all")
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.bar(rb.index, rb.mae, color="#27b"); ax.set_ylabel("MAE (kw)")
    a2 = ax.twinx(); a2.plot(rb.index, rb.coverage, "o-", color="#c33"); a2.grid(False)
    a2.set_ylabel("구간 coverage"); a2.axhline(.9, ls=":", color="#c33", lw=.8)
    ax.set_title("FIGURE 3. 신뢰도 밴드 vs 실제 오차·coverage (h60, 원본구간)")
    fig.tight_layout(); fig.savefig(fig_path(f"{pre}_fig3_reliability_bands")); plt.close(fig)

    # FIG 4 — 비용 x 선행시간 파레토
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.scatter(par["all"].cost, par["all"].mean_lead_min, s=4, color="#bbb", label="전체 정책")
    ax.plot(par["front"].cost, par["front"].mean_lead_min, "o-", color="#c33", ms=4,
            label="파레토 전선")
    ax.set_xlabel(f"정규화 기대비용 (C_FN/C_FP={cfg['policy']['default_cost_ratio']})")
    ax.set_ylabel("평균 선행시간(분)"); ax.legend(fontsize=7)
    ax.set_title("FIGURE 4. 비용 x 선행시간 파레토 (원본구간)")
    fig.tight_layout(); fig.savefig(fig_path(f"{pre}_fig4_cost_lead_pareto")); plt.close(fig)

    # FIG 5 — CORE vs PUBLIC
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    axes[0].plot(cvp.horizon_min, cvp.mae_core, "o-", label="CORE")
    if "mae_public" in cvp:
        axes[0].plot(cvp.horizon_min, cvp.mae_public, "s-", label="PUBLIC-enhanced")
    axes[0].set_xlabel("horizon(분)"); axes[0].set_ylabel("MAE (kw)"); axes[0].legend(fontsize=7)
    axes[0].set_title("평균 예측오차")
    x = np.arange(len(cvp))
    axes[1].bar(x - .2, cvp.low_or_ood_share_selected, .4, label="LOW+OOD 비중")
    axes[1].bar(x + .2, cvp.coverage_peak_only_selected, .4, label="피크구간 coverage")
    axes[1].set_xticks(x); axes[1].set_xticklabels(cvp.horizon_min)
    axes[1].set_xlabel("horizon(분)"); axes[1].legend(fontsize=7)
    axes[1].set_title("남는 신뢰도 문제 (선택 모드)")
    fig.suptitle("FIGURE 5. 공공 외부데이터 전/후 — 평균 개선 vs 잔존 신뢰도 문제", fontsize=9)
    fig.tight_layout(); fig.savefig(fig_path(f"{pre}_fig5_core_vs_public")); plt.close(fig)

    # FIG 6 — 작업자 라우팅 분포
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    d = dist.sort_values("per_day")
    axes[0].barh(d.advisory_level, d.per_day, color="#27b")
    axes[0].set_xlabel("일 평균 건수"); axes[0].set_title("권고 등급별 운영 부담")
    axes[1].bar(range(len(rout)), rout.mae_h60, color="#c33")
    axes[1].set_xticks(range(len(rout)))
    axes[1].set_xticklabels([c.split("(")[0] for c in rout.cell], rotation=20, ha="right", fontsize=6)
    axes[1].set_ylabel("MAE (kw)"); axes[1].set_title("2x2 라우팅 셀별 실제 오차")
    fig.suptitle("FIGURE 6. 작업자 진단 분포 / 라우팅", fontsize=9)
    fig.tight_layout(); fig.savefig(fig_path(f"{pre}_fig6_operator_routing")); plt.close(fig)


def summary_json(path, cfg, o, cons, cvp, gap, rout, esc, par, meta, env, limits, nexts):
    c = cons.set_index("horizon_min")
    d = dict(
        topic="저계측 제조환경을 위한 신뢰도 기반 다중시점 최대수요전력 사전경보 및 정보부족 진단 시스템",
        git_commit=env["git_commit"], generated_at=env["timestamp"],
        python_version=env["python_version"], package_versions=env["packages"],
        evaluation_window=env["eval_window"], clean_window=env["clean_window"],
        selected_point_model=cfg["forecast"]["point_model"],
        selected_feature_mode_by_horizon={int(h): c.loc[h, "feature_mode"] for h in c.index},
        peak_definition=f"trailing_{cfg['peak']['window_days']}d_p{int(cfg['peak']['quantile']*100)}",
        reliability_method=("quantile HGB + split conformal(CQR, groups=%s) + interval-width bands"
                            " + Mahalanobis OOD" % cfg["reliability"].get("conformal_groups", "none")),
        regime_method=cfg["regime"]["method"],
        alert_policy=meta, default_cost_ratio=cfg["policy"]["default_cost_ratio"],
        main_metrics={int(h): {k: float(c.loc[h, k]) for k in
                               ["mae", "rmse", "nmae", "persistence_improvement", "coverage",
                                "coverage_peak_only", "peak_ap", "mae_transition"]} for h in c.index},
        policy_result={k: float(c.loc[60, k]) for k in
                       ["policy_event_recall", "policy_false_alerts_per_day", "policy_mean_lead_min",
                        "policy_lead_ge_30min_share", "policy_normalized_cost"]},
        escalation_wins_by_ratio={int(r.cost_ratio_FN_FP): bool(r.escalation_wins)
                                  for r in esc[esc.window == "clean_Jul_Sep"].itertuples()},
        pareto_front_size=int(len(par["front"])),
        external_data_effect=cvp.to_dict(orient="records"),
        information_gap_result=gap.to_dict(orient="records"),
        routing_2x2=rout.to_dict(orient="records"),
        known_limitations=limits, next_steps=nexts)
    path.write_text(json.dumps(d, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return d
