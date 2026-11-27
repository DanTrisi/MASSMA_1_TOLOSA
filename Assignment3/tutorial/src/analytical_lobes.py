"""
Tutorial part 3 - analytical stability lobes (Altintas & Budak, 1 DOF), same formulas as analytical_lobes.xlsx.

    alpha_x = 1/2 [-cos 2t - 2 kr t - kr sin 2t]_{phi_st}^{phi_ex}
    Re[G]   = (1/k) (1 - r^2) / ((1 - r^2)^2 + 4 xi^2 r^2),   r = fc / f1
    Ap      = 2 pi / (Z alpha_x Kt Re[G])
    N       = 60 * 2 pi fc / (Z (2 pi m + 2 pi - 2 atan((r^2 - 1) / (2 xi r))))
"""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from params import *

OUT = Path(__file__).resolve().parents[1]
M_MAX = 8
N_MIN, N_MAX = 1000, 10000      # same window as the numerical script
AP_MAX = 30.0                   # [mm]


def angles(mode):
    a = np.arccos((R - AE) / R)
    return (np.pi - a, np.pi) if mode == "down" else (0.0, a)


def alpha_x(st, ex, kr):
    F = lambda t: -np.cos(2 * t) - 2 * kr * t - kr * np.sin(2 * t)
    return 0.5 * (F(ex) - F(st))


phi_st, phi_ex = angles(MODE)
ax_coef = alpha_x(phi_st, phi_ex, KR_KT)

k_mm  = K1 / 1e3                 # [N/mm]
Kt_mm = KT / 1e6                 # [N/mm^2]
r   = 1 + np.logspace(-6, np.log10(2.0), 20000)
fc  = r * F1
ReG = (1 / k_mm) * (1 - r**2) / ((1 - r**2)**2 + 4 * XI**2 * r**2)     # [mm/N]
Ap  = 2 * np.pi / (Z * ax_coef * Kt_mm * ReG)                          # [mm]
eps = 2 * np.pi - 2 * np.arctan((r**2 - 1) / (2 * XI * r))

N_grid = np.linspace(N_MIN, N_MAX, 20000)
env = np.full_like(N_grid, np.inf)
curves = []
for m in range(M_MAX + 1):
    N = 60 * 2 * np.pi * fc / (Z * (2 * np.pi * m + eps))
    curves.append(N)
    i = np.argsort(N)
    env = np.minimum(env, np.interp(N_grid, N[i], Ap[i], left=np.inf, right=np.inf))

# --- results ---
Ap_min = Ap.min()
fc_at_min = fc[np.argmax(ReG == ReG.min())]
pref = [60 * F1 / (Z * (m + 1)) for m in range(4)]          # Q3.2: tooth pass freq = f1/(m+1)
# lobe peaks = local maxima of the envelope inside the window
from scipy.signal import find_peaks
pk, _ = find_peaks(np.nan_to_num(env, posinf=0))
peaks = [(float(N_grid[p]), float(env[p])) for p in pk]
# stable windows above a given depth (e.g. 2 x unconditional limit)
Ap_target = 2 * Ap_min
stable = env > Ap_target
edges = np.flatnonzero(np.diff(stable.astype(int)))
windows = []
s = 0 if stable[0] else None
for e in edges:
    if stable[e + 1]:
        s = e + 1
    elif s is not None:
        windows.append((float(N_grid[s]), float(N_grid[e]))); s = None
if s is not None:
    windows.append((float(N_grid[s]), float(N_grid[-1])))

res = dict(mode=MODE, f1_Hz=F1, k_N_per_m=K1, kr=KR_KT,
           phi_st_deg=float(np.degrees(phi_st)), phi_ex_deg=float(np.degrees(phi_ex)),
           alpha_x=float(ax_coef), Ap_unconditional_mm=float(Ap_min), fc_at_min_Hz=float(fc_at_min),
           preferred_speeds_rpm=pref, lobe_peaks_rpm_mm=peaks,
           Ap_target_mm=float(Ap_target), stable_windows_at_target_rpm=windows)
(OUT / "results" / "analytical.json").write_text(json.dumps(res, indent=2))
for k_, v in res.items():
    print(f"{k_:30s} {v}")

# --- plot ---
fig, ax = plt.subplots(figsize=(9, 5))
for m, N in enumerate(curves):
    ax.plot(N, Ap, lw=0.8, alpha=0.6, label=f"m = {m}" if m < 5 else None)
env_p = env.copy(); env_p[~np.isfinite(env_p)] = np.nan
ax.plot(N_grid, env_p, "k", lw=2, label="stability limit")
ax.fill_between(N_grid, 0, np.fmin(env_p, AP_MAX), color="tab:green", alpha=0.12)
ax.axhline(Ap_min, ls="--", c="gray", lw=1)
ax.text(N_MAX - 150, Ap_min - 0.4, f"unconditional limit {Ap_min:.2f} mm", fontsize=9, ha="right", va="top")
for p in pref:
    ax.axvline(p, ls=":", c="tab:red", lw=1)
ax.set_xlim(N_MIN, N_MAX); ax.set_ylim(0, AP_MAX)
ax.set_xlabel("Spindle speed N [rpm]"); ax.set_ylabel("Axial depth of cut Ap [mm]")
ax.set_title(f"Analytical stability lobes ({MODE} milling, Ae = {AE*1e3:g} mm, Z = {Z})")
ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "plots" / "analytical_lobes.png", dpi=200)
fig.savefig(OUT / "plots" / "analytical_lobes.pdf")
np.savez(OUT / "results" / "analytical_envelope.npz", N=N_grid, Ap=env_p)
