"""07 전체 재현 실행기. 의존 순서대로 돌린다. 각 단계 런타임을 기록한다."""
import subprocess, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
STEPS = ["feature_availability.py", "multi_horizon_baseline.py", "latent_context.py",
         "information_ablation.py", "hard_condition_discovery.py", "symbolic_reliability.py",
         "reliability_meta.py", "confidence_gate.py", "cost_aware_alert.py",
         "value_of_information.py",
         # 2단계 (미수행 1~4 검증)
         "mondrian_conformal.py", "peak_definition.py", "multihorizon_alert.py", "three_state.py",
         # 3단계 (외부데이터·통합경보·3레짐 견고성·작업자 진단)
         "fetch_external.py", "external_data_stress_test.py", "integrated_alert_policy.py",
         "three_state_robustness.py", "operator_diagnostic.py"]
if __name__ == "__main__":
    for s in STEPS:
        t = time.time()
        r = subprocess.run([sys.executable, str(HERE / s)], cwd=HERE)
        print(f"[{s}] rc={r.returncode} {time.time()-t:.1f}s", flush=True)
        if r.returncode:
            sys.exit(r.returncode)
