"""Run every analysis of Assignment 1: plots go to plot/, LaTeX numbers to report/generated/."""
import runpy
import sys
from pathlib import Path

SRC = Path(__file__).parent / "src"
sys.path.insert(0, str(SRC))
for script in ["turning_analysis.py", "drilling_analysis.py"]:
    print(f"\n######## {script}")
    sys.argv = [script]
    runpy.run_path(str(SRC / script), run_name="__main__")
