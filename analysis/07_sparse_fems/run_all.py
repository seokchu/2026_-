"""07 전체 재현 실행기. 의존 순서대로 돌린다. 각 단계 런타임을 기록한다."""
import subprocess, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
STEPS = ["feature_availability.py", "multi_horizon_baseline.py", "latent_context.py",
         "information_ablation.py", "hard_condition_discovery.py", "symbolic_reliability.py",
         "reliability_meta.py", "confidence_gate.py", "cost_aware_alert.py",
         "value_of_information.py"]
if __name__ == "__main__":
    for s in STEPS:
        t = time.time()
        r = subprocess.run([sys.executable, str(HERE / s)], cwd=HERE)
        print(f"[{s}] rc={r.returncode} {time.time()-t:.1f}s", flush=True)
        if r.returncode:
            sys.exit(r.returncode)
