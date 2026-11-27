# Assignment 3 – Project "Slot milling"

Solid carbide end mill D = 12 mm, Z = 3, η = 45°, Lc = 30 mm, L = 76 mm, ξ = 1 %, Al 6061-T6 (Kt = 550 MPa, Kr = 150 MPa).
Tool = cantilever cylinder (E = 600 GPa, ρ = 14 500 kg/m³), rigid holder, modal mass = mass of the free cylinder.
Analytical lobes (slot milling, φ = 0 → π), target Ap = Lc/2 = 15 mm, N ≤ 6000 rpm.

Run from `src/`:  `python slot_milling.py`
→ `results/iterations.json` (dynamic parameters and check for l = 46, 42, 38, 34, 30 mm),
  `results/final.json` (selected speeds), `plots/lobes_free_length.png`, `plots/lobes_final.png`.
