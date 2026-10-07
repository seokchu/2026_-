"""Baseline v1 — 단일 진입점.

실행:
  python3 analysis/final_baseline/run_baseline.py --config analysis/final_baseline/config.yaml
옵션:
  --no-external   공공 외부데이터 없이 CORE 모드만 (캐시 없는 환경 재현용)
  --quick         폴드 3개 / 임계격자 축소 (스모크 테스트용, 보고용 수치 아님)
"""
import argparse, json, platform, subprocess, sys, time
from pathlib import Path
import numpy as np, pandas as pd, yaml, joblib

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "analysis" / "07_sparse_fems"))

from common import save_table, TAB                     # noqa: E402
import features as FT, pipeline as PP, policy as PL, diagnosis as DG, report as RP  # noqa: E402
import models as MD                                    # noqa: E402

H = [1, 2, 3, 4]


def env_record(cfg, o):
    import sklearn, scipy, matplotlib
    try:
        gc = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                            text=True).stdout.strip()
    except Exception:
        gc = "unknown"
    cl = o[o.clean == True]
    return dict(git_commit=gc, timestamp=pd.Timestamp.now().isoformat(),
                python_version=platform.python_version(), platform=platform.platform(),
                packages={m.__name__: m.__version__ for m in
                          (np, pd, sklearn, scipy, matplotlib, joblib, yaml)},
                eval_window=f"{o.ts15.min()} ~ {o.ts15.max()} (n={len(o)})",
                clean_window=f"{cl.ts15.min()} ~ {cl.ts15.max()} (n={len(cl)})")


def deployment_fit(d, cons, cfg, art_dir, bp):
    """평가 종료 후 전체 허용 과거 데이터로 재적합 — 배포용 산출물.

    평가와 배포의 신뢰도 보정 방식/기본 경보정책이 반드시 같아야 한다.
    """
    seed, rcfg = cfg["seed"], cfg["reliability"]
    art_dir.mkdir(parents=True, exist_ok=True)
    mode_by_h = {int(r.horizon_min): r.feature_mode for r in cons.itertuples()}
    n = len(d); cut = int(n * (1 - cfg["data"]["cal_fraction"]))
    TR, CA = d.iloc[:cut], d.iloc[cut:]

    # 평가에서 선택된 기본 운용점을 배포 번들에도 고정한다.
    primary = cfg["data"]["primary_window"]
    r0 = cfg["policy"]["default_cost_ratio"]
    prow = bp[(bp.window == primary) & (bp.cost_ratio_FN_FP == r0) &
              (bp.scope == "all_policies")]
    if len(prow) != 1:
        raise RuntimeError(f"default policy row not unique: window={primary}, r={r0}, n={len(prow)}")
    prow = prow.iloc[0]
    default_policy = dict(
        policy=str(prow.policy), gate=str(prow.gate),
        early_horizon_min=int(prow.early_horizon_min),
        early_q=float(prow.early_q),
        confirm_horizon_min=(int(prow.confirm_horizon_min)
                             if pd.notna(prow.confirm_horizon_min) else None),
        confirm_q=(float(prow.confirm_q) if pd.notna(prow.confirm_q) else None),
        cost_ratio=float(r0))

    bundle = dict(seed=seed, config=cfg, mode_by_horizon=mode_by_h,
                  ramp_cut=float(np.quantile(TR.kw_ramp4.abs(), cfg["regime"]["ramp_quantile"])),
                  horizons=H, default_policy=default_policy, models={})
    for h in H:
        F = FT.CORE if mode_by_h[h * 15] == "CORE" else FT.PUBLIC
        ycol = f"y_h{h}"
        recipe = cfg["forecast"].get("point_recipe", "v2_a5")
        if recipe == "v1":
            pt = MD.point_model(cfg["forecast"]["point_model"], seed).fit(d[F].values,
                                                                         d[ycol].values)
        else:  # 잔차타깃 + 극단가중: 예측값에 결정시점 kw 를 다시 더해야 한다
            w = MD.relevance_weight(d[ycol].values)
            pt = MD.point_model(cfg["forecast"]["point_model"], seed).fit(
                d[F].values, d[ycol].values - d.kw.values, sample_weight=w)
        rb_tr = TR.kw.values if recipe != "v1" else 0.0
        rb_ca = CA.kw.values if recipe != "v1" else 0.0
        ytr_r, yca_r = TR[ycol].values - rb_tr, CA[ycol].values - rb_ca
        conformal_mode = rcfg.get("conformal_groups", "none")
        if conformal_mode == "level3":
            level_cuts = np.quantile(TR.kw.values, [1 / 3, 2 / 3])
            gcal = np.digitize(CA.kw.values, level_cuts)
            gprobe = np.digitize(CA.kw.values[:1], level_cuts)
            _, _, cal_width, (m_lo, m_hi, Qs, Qg) = MD.fit_cqr_grouped(
                TR[F].values, ytr_r, CA[F].values, yca_r,
                CA[F].values[:1], gcal, gprobe,
                *rcfg["quantile_levels"], rcfg["conformal_alpha"], seed)
            conformal_spec = dict(
                mode="level3",
                level_cuts=[float(x) for x in level_cuts],
                group_Q={int(k): float(v) for k, v in Qs.items()},
                global_Q=float(Qg))
            Q_compat = float(Qg)
        else:
            _, _, cal_width, (m_lo, m_hi, Q) = MD.fit_cqr(
                TR[F].values, ytr_r, CA[F].values, yca_r,
                CA[F].values[:1], *rcfg["quantile_levels"],
                rcfg["conformal_alpha"], seed)
            conformal_spec = dict(mode="global", global_Q=float(Q))
            Q_compat = float(Q)

        cv, ood_cut = MD.ood_scorer(TR[F].values, rcfg["ood_quantile"])
        cal_pred = pt.predict(CA[F].values) + rb_ca   # 잔차 -> 수준 복원
        bundle["models"][h] = dict(
            features=F, point=pt, point_recipe=recipe, q_lo=m_lo, q_hi=m_hi,
            conformal_Q=Q_compat, conformal_spec=conformal_spec,
            band_cuts=[float(x) for x in np.quantile(cal_width, rcfg["band_width_quantiles"])],
            ood=cv, ood_cut=ood_cut,
            alert_thresholds={float(q): float(np.quantile(cal_pred, q))
                              for q in cfg["policy"]["threshold_grid"]})
    joblib.dump(bundle, art_dir / "baseline_v1.joblib")
    return bundle, mode_by_h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    ap.add_argument("--no-external", action="store_true")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    if a.no_external:
        cfg["external"]["use_external_data"] = False
    if a.quick:
        cfg["data"]["n_splits"] = 3
        cfg["policy"]["threshold_grid"] = [0.6, 0.8, 0.9]
        cfg["forecast"]["eval_baselines"] = ["persistence", "hgb"]
    np.random.seed(cfg["seed"])
    pre, t0 = cfg["output"]["table_prefix"], time.time()

    # 1) 데이터 + 특성
    q, mode = FT.build(cfg["external"]["use_external_data"], cfg["peak"]["window_days"],
                       cfg["peak"]["quantile"], cfg["peak"]["min_days"])
    d = FT.common_rows(q, mode)
    print(f"[1] feature mode={mode}  rows={len(d)}  clean={int(d.clean.sum())}")
    save_table(FT.metadata(mode), f"{pre}_feature_metadata")

    # 2) 시간순 평가
    o, sel, thr, reg = PP.run_evaluation(d, mode, cfg)
    print(f"[2] OOF rows={len(o)} clean={int(o.clean.sum())}  {time.time()-t0:.0f}s")
    save_table(o, f"{pre}_oof_rows")
    save_table(sel, f"{pre}_feature_mode_selection")
    save_table(thr, f"{pre}_cal_alert_thresholds")
    if len(reg):
        save_table(reg, f"{pre}_regime_agreement")
    pf = PP.forecast_metrics(o, cfg); save_table(pf, f"{pre}_forecast_metrics")
    rel = PP.reliability_metrics(o); save_table(rel, f"{pre}_reliability_metrics")

    # 3) 통합 경보 정책 + 파레토
    cur = PL.sweep(o, thr, cfg); save_table(cur, f"{pre}_policy_curve")
    bp = PL.best_points(cur, cfg); save_table(bp, f"{pre}_recommended_operating_point")
    esc = PL.escalation_vs_single(bp, cfg); save_table(esc, f"{pre}_escalation_vs_single")
    front, allp = PL.pareto(cur, cfg); save_table(front, f"{pre}_cost_lead_pareto")
    print(f"[3] 정책 {len(cur)}개 평가, 파레토 {len(front)}점  {time.time()-t0:.0f}s")

    # 4) 정보부족 진단 + 작업자 출력
    out, s, meta, fl = DG.build_output(o, thr, cfg, bp)
    rout = DG.routing_2x2(s, fl); save_table(rout, f"{pre}_routing_2x2")
    gap = DG.gap_validation(s, fl); save_table(gap, f"{pre}_information_gap_validation")
    dist = DG.advisory_distribution(out, s); save_table(dist, f"{pre}_advisory_distribution")
    save_table(out, f"{pre}_operator_output")
    save_table(DG.sample_cases(out), f"{pre}_operator_output_sample")
    nf = DG.forbidden_hits(out)
    print(f"[4] 적용 정책 {meta}  금지문구 {nf}건")
    assert nf == 0, "작업자 출력에 물리적 조치 문구가 포함되었다"

    # 5) 통합 지표 / CORE vs PUBLIC / 그림
    cons = RP.consolidated(o, rel, bp, pf, gap, dist, cfg); save_table(cons, f"{pre}_final_metrics")
    cvp = RP.core_vs_public(pf, rel, o); save_table(cvp, f"{pre}_core_vs_public")
    RP.figures(o, rel, dict(front=front, all=allp), cvp, dist, rout, meta, cfg)

    # 6) 배포 적합 + 산출물
    art = ROOT / cfg["output"]["artifacts_dir"]
    bundle, mode_by_h = deployment_fit(d, cons, cfg, art, bp)
    env = env_record(cfg, o)
    (art / "model_metadata.json").write_text(json.dumps(dict(
        **{k: v for k, v in env.items()}, feature_mode_by_horizon=mode_by_h,
        point_model=cfg["forecast"]["point_model"],
        peak_definition=f"trailing_{cfg['peak']['window_days']}d_p95",
        cost_ratio=cfg["policy"]["default_cost_ratio"],
        reliability_thresholds={h: bundle["models"][h]["band_cuts"] for h in H},
        ood_cuts={h: bundle["models"][h]["ood_cut"] for h in H},
        regime_method=cfg["regime"]["method"], ramp_cut=bundle["ramp_cut"]),
        ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    limits = ["단일 KAMP 대상 공장·단일 연도(2021) 데이터",
              "1~6월 증강 구간 포함. 보고 기준은 원본 7~9월",
              "피크 정의는 운영적 정의(직전 30일 p95). KEPCO 계약전력/요금 최대수요와 동일하지 않다",
              "생산량(t)은 생산계획 사전확정 가정에 의존(민감도는 보고서 10 L3s 참조)",
              "외부 기상은 추정 최근접 관측소(울산) 기반. 공식 위치 미공개",
              "피크위험은 보정된 확률이 아니라 정규화 점수",
              "Category B 타 공장 전이는 공개 10개 공장에 대한 단일 leave-one-factory-out 설계만 검증; 전이학습 일반화 금지"]
    nexts = ["계약전력·요금 단가 확인 후 비용비를 실제 단위로 치환",
             "설비 가동상태 1~2채널 실측 확보 시 정보부족/신뢰도 개선량 재검증",
             "타 공장 소량 미세조정·도메인 적응은 별도 후속 검증",
             "경보 임계의 온라인 재보정(드리프트 대응)"]
    RP.summary_json(HERE / "baseline_summary.json", cfg, o, cons, cvp, gap, rout, esc,
                    dict(front=front, all=allp), meta, env, limits, nexts)
    print(f"[6] artifacts {art}  총 {time.time()-t0:.0f}s")
    print(cons[["horizon_min", "feature_mode", "mae", "rmse", "persistence_improvement",
                "coverage", "coverage_peak_only", "peak_ap", "mae_transition"]].to_string(index=False))
    print(esc[esc.window == "clean_Jul_Sep"].to_string(index=False))
    print(front.head(12).to_string(index=False))
    print(rout.to_string(index=False)); print(gap.to_string(index=False))
    print(dist.to_string(index=False))


if __name__ == "__main__":
    main()
