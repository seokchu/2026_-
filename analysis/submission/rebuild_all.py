"""전체 산출물 재생성 — 코드 변경 시 표·그림·보고서·발표자료를 한 번에 동기화한다.

실행: python3 analysis/submission/rebuild_all.py [--skip-baseline] [--skip-slow]
순서가 중요하다. 뒤 단계는 앞 단계의 표를 읽는다.
"""
import subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FB = "analysis/final_baseline"
SUB = "analysis/submission"
args = set(sys.argv[1:])

STEPS = [
    ("baseline",  f"{FB}/run_baseline.py",       "표·그림 23_*, FIG 1~6, 배포 아티팩트"),
    ("v2_ablate", f"{FB}/v2_forecast_improve.py", "표 31_v2_* (모델 비교·조건별 성능)"),
    ("v2_clean",  f"{FB}/v2_clean_summary.py",   "표 31_v2_clean_*, 31_duplicate_fold_crossing"),
    ("v2_ood",    f"{FB}/v2_stress_robust.py",   "표 31_ood_*"),
    ("leadtime",  f"{FB}/leadtime_basis.py",     "표 31_industry/leadtime/tariff 근거"),
    ("peakprob",  f"{FB}/peak_probability.py",   "표 28_peak_probability_*, FIG 7"),
    ("seqpolicy", f"{FB}/sequential_policy.py",  "표 28_sequential_*, FIG 8"),
    ("errcond",   f"{FB}/error_conditions.py",   "표 28_error_conditions_*, FIG 9"),
    ("missed",    f"{FB}/missed_event_taxonomy.py", "표 30_missed_event_*, 30_false_alarm_*"),
    ("visuals",   f"{FB}/state_visuals.py",      "FIG 10~13, 표 30_fnfp_by_state"),
    ("mondrian",  f"{FB}/followup_mondrian.py",  "표·그림 24_mondrian_*"),
    ("costunits", f"{FB}/followup_cost_units.py", "표 24_cost_units_*"),
    ("addedchan", f"{FB}/followup_added_channel.py", "표·그림 24_added_channel_*"),
    ("statemodel", f"{FB}/state_model_test.py",  "표 30_state_model_* (느림)"),
    ("tests",     f"{FB}/test_baseline.py",      "수용테스트"),
    ("report",    f"{SUB}/build_report.py",      "결과보고서 .hwpx"),
    ("pptx",      f"{SUB}/build_pptx.py",        "발표자료 .pptx"),
]
SLOW = {"statemodel", "v2_ablate"}

t0 = time.time()
fail = []
for tag, script, what in STEPS:
    if tag == "baseline" and "--skip-baseline" in args:
        print(f"[skip] {tag}"); continue
    if tag in SLOW and "--skip-slow" in args:
        print(f"[skip] {tag}"); continue
    s = time.time()
    r = subprocess.run([sys.executable, "-u", script], cwd=ROOT,
                       capture_output=True, text=True)
    ok = "ok " if r.returncode == 0 else "FAIL"
    print(f"[{ok}] {tag:11s} {time.time()-s:7.1f}s  {what}")
    if r.returncode:
        fail.append(tag)
        print("    " + (r.stderr.strip().splitlines() or ["(no stderr)"])[-1][:300])
print(f"\n총 {time.time()-t0:.0f}s, 실패 {len(fail)}개 {fail}")
sys.exit(1 if fail else 0)
