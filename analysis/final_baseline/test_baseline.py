"""Baseline v1 수용 테스트 (run_baseline.py 실행 후 돌린다).

python3 analysis/final_baseline/test_baseline.py
프레임워크 없이 assert 만 쓴다. 실패하면 비정상 종료한다.
"""
import inspect, json, sys
from pathlib import Path
import numpy as np, pandas as pd, yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "analysis" / "07_sparse_fems"))
from common import TAB                                  # noqa: E402
import features as FT, models as MD, policy as PL, diagnosis as DG, inference as IN  # noqa: E402

CFG = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
PRE = CFG["output"]["table_prefix"]
T = lambda n: pd.read_csv(TAB / f"{PRE}_{n}.csv", encoding="utf-8-sig")
ok = lambda m: print(f"  PASS  {m}")


def t_targets_and_leakage():
    q, mode = FT.build()
    for h in (1, 2, 3, 4):
        s = q.dropna(subset=[f"y_h{h}"])
        assert np.allclose(s[f"y_h{h}"].values, q.kw.shift(-h).dropna().values[:len(s)]), h
    ok("h15/h30/h45/h60 타깃 정렬 = kw.shift(-h)")
    feats = set(FT.CORE) | set(FT.PUBLIC_EXTRA)
    assert "평균" not in feats and "공장인원" not in feats and "인건비" not in feats
    md = T("feature_metadata")
    assert (md[md.feature_name == "평균"].information_group == "EXCLUDED").all()
    ok("'평균' 등 타깃 누수/중복 변수 특성 제외")
    # 적응형 임계가 과거만 보는가 — 임의 시점 직접 재계산
    rng = np.random.default_rng(0)
    for i in rng.integers(96 * 40, len(q) - 1, 20):
        want = q.kw.iloc[max(0, i - 96 * 30):i].quantile(.95)
        got = q.thr_adaptive.iloc[i]
        assert np.isnan(got) or abs(want - got) < 1e-6, (i, want, got)
    ok("직전 30일 p95 임계는 과거 관측만 사용")
    # 외부 특성은 결정시점 가용 (정시 관측 join, 예보 없음)
    assert (md[md.requires_external_data].available_at_decision_time).all()
    ok("외부 특성 전부 결정시점 가용")


def t_temporal_protocol():
    o = T("oof_rows")
    o["ts15"] = pd.to_datetime(o.ts15)
    g = o.groupby("fold").ts15.agg(["min", "max"]).sort_index()
    assert (g["min"].values[1:] > g["max"].values[:-1]).all(), "폴드가 시간순이 아니다"
    ok("rolling-origin 폴드 시간순 (랜덤 분할 없음)")
    thr = T("cal_alert_thresholds")
    assert set(thr.fold.unique()) == set(o.fold.unique()) and thr.threshold_kw.notna().all()
    piv = thr.pivot_table(index="fold", columns=["horizon_min", "q"], values="threshold_kw")
    assert piv.nunique().max() > 1, "폴드별 CAL 임계가 상수 — CAL 산출이 아닐 수 있다"
    ok("경보 임계는 폴드별 CAL 예측분포에서 산출")
    sig = inspect.signature(MD.fit_cqr).parameters
    assert "yte" not in sig and "y_te" not in sig
    ok("CQR 보정에 TEST 라벨 인자 없음")
    sel = T("feature_mode_selection")
    assert sel.rule.nunique() == 1 and sel.cal_mae_core.notna().all()
    ok("특성모드 게이트는 CAL 지표 + 사전 고정 규칙")


def t_reliability_and_peak():
    rel = T("reliability_metrics")
    r = rel[(rel.window == "clean_Jul_Sep") & (rel.horizon_min == 60)].set_index("band")
    assert r.loc["HIGH", "mae"] < r.loc["MEDIUM", "mae"] < r.loc["LOW", "mae"]
    ok(f"밴드 단조성 HIGH<MEDIUM<LOW ({r.loc['HIGH','mae']:.2f}/"
       f"{r.loc['MEDIUM','mae']:.2f}/{r.loc['LOW','mae']:.2f})")
    o = T("oof_rows")
    assert ((o.kw >= o.thr_adaptive).astype(int) == o.peak_now).all()
    for h in (1, 2, 3, 4):
        assert ((o[f"y_h{h}"] >= o.thr_adaptive).astype(int) == o[f"peak_h{h}"]).all()
    ok("피크 라벨 = 예측 대상값 >= 직전 30일 p95")


def t_operator_output():
    out = T("operator_output")
    assert DG.forbidden_hits(out) == 0
    ok("작업자 출력에 물리적 조치 지시 0건")
    allowed = {"NORMAL", "EARLY_WATCH", "CONFIRMED_WARNING", "LOW_CONFIDENCE_WARNING",
               "INFORMATION_GAP", "HUMAN_REVIEW"}
    assert set(out.advisory_level) <= allowed
    assert len(T("operator_output_sample")) >= 50
    for c in ["timestamp", "forecast_60m", "interval_low_60m", "reliability_60m",
              "current_peak_threshold", "early_peak_risk", "confirmed_peak_risk",
              "operating_regime", "ood_flag", "uncertainty_explained",
              "information_gap_flag", "advisory_level", "diagnostic_reason",
              "recommended_information_type"]:
        assert c in out.columns, c
    ok("필수 출력 필드 + 등급 어휘 준수, 대표사례 50건 이상")


def t_artifacts_and_determinism():
    b = IN.load()
    q, mode = FT.build()
    d = FT.common_rows(q, mode)
    p = IN.predict(d.tail(20), b)
    assert len(p) == 20 and p.filter(like="forecast_").notna().all().all()
    ok("배포 산출물 재로딩 + 추론 동작")
    F = FT.CORE
    sl = d.iloc[:3000]
    a = MD.point_model("hgb", CFG["seed"]).fit(sl[F].values, sl.y_h4.values).predict(sl[F].values[:50])
    c = MD.point_model("hgb", CFG["seed"]).fit(sl[F].values, sl.y_h4.values).predict(sl[F].values[:50])
    assert np.allclose(a, c)
    ok("동일 seed 재적합 수치 일치")
    js = json.loads((HERE / "baseline_summary.json").read_text(encoding="utf-8"))
    assert js["peak_definition"] == "trailing_30d_p95" and js["git_commit"]
    ok("요약 JSON 메타데이터 기록")


def t_core_fallback():
    q, mode = FT.build(use_external=False)
    assert mode == "CORE" and all(c in q for c in FT.CORE)
    d = FT.common_rows(q, mode)
    assert len(d) > 0 and list(FT.feature_sets(mode)) == ["CORE"]
    ok(f"외부 캐시 없이 CORE 모드 동작 (rows={len(d)})")


if __name__ == "__main__":
    for f in [t_targets_and_leakage, t_temporal_protocol, t_reliability_and_peak,
              t_operator_output, t_artifacts_and_determinism, t_core_fallback]:
        print(f.__name__)
        f()
    print("\nALL ACCEPTANCE TESTS PASSED")
