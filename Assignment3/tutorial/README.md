# Assignment – Tutorial "Stability lobes"

Finish milling of a thin 6061-T6 wall (flexible part, 1 DOF), D = 20 mm, Z = 3, η = 45°, Ae = 1 mm,
ω1 = 1068.1 rad/s, m1 = 2.63 kg, ξ = 4 %, Kt = 550 MPa, Kr = 150 MPa. Down milling in both models.

| Folder | Content |
| --- | --- |
| `src/params.py` | all tutorial data (SI units) + derived k, f1, kr |
| `src/analytical_lobes.py` | Q3.2–3.3: analytical lobes (same formulas as the Excel file), preferred speeds, unconditional limit |
| `src/numerical_lobes.py` | Q4.1–4.3: corrected semi-discretization script (`--helix 0` / `--helix 45`) |
| `src/compare_lobes.py` | Q4.2–4.3: analytical vs numerical, key numbers, helix pitch |
| `excel/analytical_lobes_tutorial.xlsx` | course Excel file filled with the tutorial data (recalculates when opened in Excel) |
| `plots/` | figures (PNG + PDF for LaTeX) |
| `results/` | numbers (JSON) and raw grids (NPZ) |
| `tutorial_answers.md` | draft answers to all questions |

Run (from `src/`, with the Assignment 1 venv or any Python with numpy, scipy, matplotlib):

```
python analytical_lobes.py
python numerical_lobes.py --helix 0
python numerical_lobes.py --helix 45
python compare_lobes.py
```

Fixes applied to the course file `2.2 numerical_lobes.py`:
1. `MC = inv(C) @ C` → `c/m` (the damping term was replaced by 1).
2. 1×1 arrays assigned to scalar cells → scalars (ValueError with numpy ≥ 2).
3. tooth angle taken modulo 2π, as in slide 26 (needed with the helix).
4. `up_or_down` labels: 0 → arccos(1−2Ae/D) is **up** milling, arccos(2Ae/D−1) → π is **down** milling.
5. inner loops over z, teeth and sub-steps vectorised (same result, ~10 s instead of ~2 min).
