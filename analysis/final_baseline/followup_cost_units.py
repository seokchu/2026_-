"""후속 S. 무차원 비용비 C_FN/C_FP 를 실제 운영 단위로 환산하는 구조.

전제: KAMP 데이터에는 계약전력·요금 단가·사업장 식별 정보가 없다. kamp-ai.kr 명세는 로그인
필요로 미확인(보고서 09). 따라서 **단가를 발명하지 않는다.** 대신 비용비를 파라미터 함수로 쓴다.

  C_FN(미탐 1건)  = 기본요금단가[원/kW·월] x 피크 초과 ΔkW x 요금적용 개월수
  C_FP(오경보 1스텝) = 확인 소요시간[시간] x 확인 인건비[원/시간]
  r = C_FN / C_FP

ΔkW 는 데이터에서 측정한다(피크 이벤트의 실제 초과량). 나머지 3개는 **입력 파라미터**이며
값은 [가설] 로 표기한다. 각 시나리오의 r 을 정책 곡선에 대입해 최적 운용점을 찾는다.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, yaml
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "analysis" / "07_sparse_fems"))
from common import save_table, TAB                            # noqa: E402
import features as FT                                         # noqa: E402

cfg = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
PRE = cfg["output"]["table_prefix"]

# ---- 1) 데이터에서 측정 가능한 부분: 피크 이벤트 초과량 ΔkW
o = pd.read_csv(TAB / f"{PRE}_oof_rows.csv", parse_dates=["ts15"], encoding="utf-8-sig")
s = o[o.clean == True].sort_values("ts15").reset_index(drop=True)
peak = s.kw.values >= s.thr_adaptive.values
exc = (s.kw.values - s.thr_adaptive.values)[peak]
idx = np.flatnonzero(peak)
ev, start, prev = [], idx[0], idx[0]
for i in idx[1:]:
    if i - prev <= 2:
        prev = i; continue
    ev.append((start, prev)); start, prev = i, i
ev.append((start, prev))
ev_exc = [float((s.kw.values - s.thr_adaptive.values)[a:b + 1].max()) for a, b in ev]
meas = dict(n_events=len(ev), delta_kw_mean=float(np.mean(ev_exc)),
            delta_kw_median=float(np.median(ev_exc)), delta_kw_p90=float(np.quantile(ev_exc, .9)),
            delta_kw_max=float(np.max(ev_exc)), exceed_step_mean=float(exc.mean()),
            source="23_baseline_oof_rows.csv (clean_Jul_Sep, 직전 30일 p95 기준 초과량)")
save_table(pd.DataFrame([meas]), "24_cost_unit_measured_exceedance")

# ---- 2) 파라미터 시나리오 (값은 모두 [가설] — 공식 단가 확인 시 치환)
SCEN = [
    dict(scenario="S1_낮은단가_짧은적용", basic_charge_krw_per_kw_month=5000, applied_months=1,
         check_minutes=10, labor_krw_per_hour=20000),
    dict(scenario="S2_낮은단가_12개월적용", basic_charge_krw_per_kw_month=5000, applied_months=12,
         check_minutes=10, labor_krw_per_hour=20000),
    dict(scenario="S3_중간단가_12개월적용", basic_charge_krw_per_kw_month=7000, applied_months=12,
         check_minutes=5, labor_krw_per_hour=25000),
    dict(scenario="S4_높은단가_12개월_짧은확인", basic_charge_krw_per_kw_month=9000, applied_months=12,
         check_minutes=2, labor_krw_per_hour=30000),
]
cur = pd.read_csv(TAB / f"{PRE}_policy_curve.csv", encoding="utf-8-sig")
cur = cur[cur.window == "clean_Jul_Sep"].reset_index(drop=True)
RATIOS = cfg["policy"]["cost_ratios"]
rows = []
for sc in SCEN:
    for dk_tag, dk in [("median", meas["delta_kw_median"]), ("mean", meas["delta_kw_mean"]),
                       ("p90", meas["delta_kw_p90"])]:
        c_fn = sc["basic_charge_krw_per_kw_month"] * dk * sc["applied_months"]
        c_fp = sc["check_minutes"] / 60 * sc["labor_krw_per_hour"]
        r = c_fn / c_fp
        near = min(RATIOS, key=lambda x: abs(np.log(x) - np.log(r)))
        best = cur.loc[cur[f"cost_r{near}"].idxmin()]
        rows.append(dict(**sc, delta_kw_basis=dk_tag, delta_kw=dk, C_FN_krw=c_fn, C_FP_krw=c_fp,
                         implied_cost_ratio=r, nearest_evaluated_ratio=near,
                         optimal_policy=best.policy, early_q=best.early_q,
                         confirm_q=best.confirm_q, gate=best.gate,
                         event_recall=best.event_recall, false_alerts_per_day=best.false_alerts_per_day,
                         mean_lead_min=best.mean_lead_min,
                         normalized_cost=best[f"cost_r{near}"],
                         note="단가·확인시간·인건비는 [가설] 입력. ΔkW 는 데이터 측정값"))
res = pd.DataFrame(rows)
save_table(res, "24_cost_unit_scenarios")
print(pd.DataFrame([meas]).to_string(index=False))
print(res[["scenario", "delta_kw_basis", "implied_cost_ratio", "nearest_evaluated_ratio",
           "optimal_policy", "early_q", "confirm_q", "event_recall",
           "false_alerts_per_day", "mean_lead_min"]].round(3).to_string(index=False))
print("\n[해석] 모든 시나리오의 함의 비용비가 평가 구간(1~100) 어디에 떨어지는지만 말할 수 있다.")
