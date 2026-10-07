"""Baseline v1 — Module A. 단일 특성 파이프라인 (CORE / PUBLIC 두 모드).

규칙:
  - 결정시점 t 의 정보만 사용한다. '평균' 열은 타깃 4개의 평균이므로 영구 제외.
  - 외부 공공데이터는 t 이하의 마지막 정시 관측/사전 확정 달력만 join 한다(예보 미사용).
  - 캐시 테이블이 없으면 PUBLIC 모드를 자동 포기하고 CORE 로 강등한다(graceful fallback).
특성 집합 정의는 07 단계의 _fe 모듈을 재사용한다(중복 구현 금지).
"""
from pathlib import Path
import sys
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "analysis" / "07_sparse_fems"))
import _fe                                              # noqa: E402
from common import TAB                                  # noqa: E402

# v2: 주기 인코딩 + 휴무/재가동 특성을 CORE 에 추가한다(31_v2_ablation 근거, A2+A4).
#     전부 결정시점 t 까지의 정보 또는 달력으로부터 결정론적으로 계산된다 → 누수 없음.
V2CYC = [f"tod_{f}{k}" for k in (1, 2, 3) for f in ("sin", "cos")] + ["dow_sin", "dow_cos"]
V2SHUT = ["off_run_len", "on_run_len", "restart_recent", "kw_vs_roll96", "kw_ratio_roll96"]
V2EXTRA = V2CYC + V2SHUT
CORE = _fe.LEVELS["L3_+production"] + V2EXTRA   # power+calendar+weather(관측)+production+v2
CALX = ["is_holiday", "is_workday", "is_bridge_day", "days_to_next_nonworkday",
        "days_since_prev_nonworkday", "nonwork_run_len"]
WEAX = ["ext_temp", "ext_dewpoint", "ext_pressure", "ext_wind", "ext_visibility",
        "ext_temp_diff1h", "ext_temp_roll24h", "ext_dew_spread", "ext_heat_index_proxy",
        "ext_pressure_diff3h", "ext_prev_day_tmax", "ext_prev_day_tmin"]
TARX = ["solar_elevation", "is_daylight", "day_length_h", "tariff_season_code",
        "tariff_load_zone_code"]
PUBLIC_EXTRA = CALX + WEAX + TARX
PUBLIC = CORE + PUBLIC_EXTRA

EXT_TABLES = ["07N_external_calendar_2021", "07N_external_weather_hourly_filled",
              "07N_external_solar_tariff_2021"]

GROUP = ({c: "power_history" for c in _fe.L0} | {c: "calendar" for c in _fe.CAL} |
         {c: "weather_observed" for c in _fe.WEA} | {c: "production" for c in _fe.PRD} |
         {c: "public_calendar" for c in CALX} | {c: "public_weather" for c in WEAX} |
         {c: "solar_tariff" for c in TARX} | {c: "cyclic_time" for c in V2CYC} |
         {c: "operation_state_history" for c in V2SHUT})


def add_v2_features(q):
    """주기 인코딩(결정론적) + 휴무/재가동 경과(인과적)."""
    for k in (1, 2, 3):
        q[f"tod_sin{k}"] = np.sin(2 * np.pi * k * q.tod / 96)
        q[f"tod_cos{k}"] = np.cos(2 * np.pi * k * q.tod / 96)
    q["dow_sin"] = np.sin(2 * np.pi * q.day / 7)
    q["dow_cos"] = np.cos(2 * np.pi * q.day / 7)
    op = q.is_operating.values.astype(int)
    off = np.zeros(len(op)); on = np.zeros(len(op))
    for i in range(len(op)):
        if op[i] == 0:
            off[i] = (off[i - 1] + 1) if i else 1; on[i] = 0
        else:
            on[i] = (on[i - 1] + 1) if i else 1; off[i] = 0
    q["off_run_len"], q["on_run_len"] = off, on
    q["restart_recent"] = ((on > 0) & (on <= 8)).astype(float)
    q["kw_vs_roll96"] = q.kw.values - q.kw_roll96.values
    q["kw_ratio_roll96"] = q.kw.values / np.maximum(q.kw_roll96.values, 1.0)
    return q


def external_cache_ready():
    return all((TAB / f"{t}.csv").exists() for t in EXT_TABLES)


def build(use_external=True, peak_window_days=30, peak_quantile=.95, peak_min_days=7):
    """15분 수요 시계열 + 결정시점 특성 + h1~h4 타깃 + 적응형 피크 임계.

    반환 (df, mode) — mode 는 'PUBLIC' 또는 'CORE'(강등 포함).
    """
    q = _fe.build()
    q = add_v2_features(q)
    mode = "CORE"
    if use_external and external_cache_ready():
        cal = pd.read_csv(TAB / "07N_external_calendar_2021.csv", parse_dates=["date"],
                          encoding="utf-8-sig")
        q["date_only"] = q.ts15.dt.normalize()
        q = q.merge(cal[["date"] + CALX], left_on="date_only", right_on="date", how="left")
        w = pd.read_csv(TAB / "07N_external_weather_hourly_filled.csv", parse_dates=["ts"],
                        encoding="utf-8-sig")
        q["ts_hour"] = q.ts15.dt.floor("h")             # t 이하 마지막 정시 관측만
        q = q.merge(w[["ts"] + WEAX], left_on="ts_hour", right_on="ts", how="left")
        s = pd.read_csv(TAB / "07N_external_solar_tariff_2021.csv", parse_dates=["ts15"],
                        encoding="utf-8-sig")
        s["tariff_season_code"] = s.tariff_season.map({"winter": 0, "spring_fall": 1, "summer": 2})
        s["tariff_load_zone_code"] = s.tariff_load_zone.map({"light": 0, "mid": 1, "peak": 2})
        q = q.merge(s[["ts15"] + TARX], on="ts15", how="left")
        mode = "PUBLIC"
    # 표준 피크 라벨: 직전 30일 p95, 현재/미래값 제외
    win = 96 * peak_window_days
    q["thr_adaptive"] = (q.kw.shift(1).rolling(win, min_periods=96 * peak_min_days)
                         .quantile(peak_quantile))
    return q, mode


def feature_sets(mode):
    """mode 별 사용 가능 특성집합."""
    return {"CORE": CORE} if mode == "CORE" else {"CORE": CORE, "PUBLIC": PUBLIC}


def common_rows(q, mode):
    """정보수준 공정 비교 규칙: 모든 모드 특성이 결측 없는 공통 행집합."""
    feats = sorted(set().union(*feature_sets(mode).values()))
    need = feats + ["thr_adaptive", "is_operating_lag4"] + [f"y_h{h}" for h in _fe.HORIZONS]
    return q.dropna(subset=need).reset_index(drop=True)


def metadata(mode):
    """기계판독 특성 메타데이터 (Module A 요구 필드)."""
    rows = []
    for c in sorted(set(feature_sets(mode)["CORE"]) | set(PUBLIC_EXTRA)):
        g = GROUP.get(c, "unknown")
        ext = g in ("public_calendar", "public_weather", "solar_tariff")
        future = g in ("calendar", "public_calendar", "solar_tariff")
        assumed = c in ("생산량", "prod_diff", "is_operating")
        note = {"power_history": "t 시점까지의 집계 수요",
                "calendar": "달력/계절요금 — 사전 확정",
                "weather_observed": "KAMP 동시각 관측값(예보 미사용)",
                "production": "ERP 시간집계. 생산계획 사전확정 가정 시 사용",
                "public_calendar": "holidays 0.83(KR) 파생 — 사전 확정",
                "public_weather": "NOAA ISD, t 이하 마지막 정시 관측",
                "solar_tariff": "결정론적 태양기하 + KEPCO 시간대 구조"}.get(g, "")
        rows.append(dict(feature_name=c, source="KAMP" if not ext else "public_external",
                         information_group=g, available_at_decision_time=True,
                         future_known=future, requires_external_data=ext,
                         safe_for_h15=not assumed or True, safe_for_h30=True,
                         safe_for_h45=True, safe_for_h60=True,
                         assumption_dependent=assumed, notes=note))
    for c, why in _fe.EXCLUDED.items():
        rows.append(dict(feature_name=c, source="KAMP", information_group="EXCLUDED",
                         available_at_decision_time=True, future_known=False,
                         requires_external_data=False, safe_for_h15=False, safe_for_h30=False,
                         safe_for_h45=False, safe_for_h60=False, assumption_dependent=False,
                         notes=why))
    return pd.DataFrame(rows)
