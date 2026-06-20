"""Run the full pipeline end-to-end on the full dataset: spatial/frequency
parameter sweeps, ROI enhancement, classification across all scenarios, and
final evaluation -- streaming each script's output to the terminal live
(identical to running them one by one).
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

PIPELINE_SCRIPTS = [
    ROOT / "filtering_spatial" / "optimize_spatial.py",
    ROOT / "filtering_frequency" / "optimize_frequency.py",
    ROOT / "enhancement_roi" / "run_local.py",
    ROOT / "classification" / "run_local.py",
    ROOT / "evaluation" / "run_local.py",
]


def run_script(script: Path) -> None:
    print(f"\n{'=' * 80}\nRUNNING: {script.relative_to(ROOT)}\n{'=' * 80}\n", flush=True)
    result = subprocess.run([sys.executable, str(script)], cwd=ROOT)
    if result.returncode != 0:
        print(f"\n{script.relative_to(ROOT)} failed with exit code {result.returncode}", flush=True)
        sys.exit(result.returncode)


def main() -> None:
    for script in PIPELINE_SCRIPTS:
        run_script(script)
    print(f"\n{'=' * 80}\nFULL PIPELINE COMPLETE\n{'=' * 80}", flush=True)


if __name__ == "__main__":
    main()
