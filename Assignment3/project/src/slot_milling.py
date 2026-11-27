"""
Project - slot milling: choice of the free tool length and of the spindle speed (analytical lobes).

Tool: solid carbide end mill D = 12 mm, Z = 3, Lc = 30 mm, L = 76 mm, xi = 1 %.
Tool model (slide 7 + teacher's indications): cantilever cylinder of diameter D and free length l,
tool holder and spindle rigid, modal mass = mass of the free cylinder.
    I = pi D^4 / 64,  k = 3 E I / l^3,  m = rho pi D^2 / 4 * l,  w = sqrt(k/m),  c = 2 xi sqrt(k m)
Lobes: same analytical formulas as the course Excel file (Altintas & Budak), slot milling phi = 0 -> pi,
computed for all the lobe orders that fall below the maximum spindle speed.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[1]

# ---- data ----
D, Z, ETA, LC, L_TOT, XI = 12e-3, 3, np.radians(45), 30e-3, 76e-3, 0.01
E, RHO = 600e9, 14500.0               # carbide (confirmed by the teacher)
KT, KR = 550.0, 150.0                 # [N/mm^2], same 6061-T6 alloy as the tutorial
N_MAX = 6000.0                        # maximum spindle speed [rpm]
N_MIN = 1000.0                        # lower bound of the study (Vc = 38 m/min)
AP_TARGET = LC / 2 * 1e3              # target axial depth of cut [mm] = Lc / 2
FREE_LENGTHS = [46, 42, 38, 34, 30]   # [mm] from a long overhang down to the minimum l = Lc

kr = KR / KT
phi_st, phi_ex = 0.0, np.pi           # slot milling: full immersion
F = lambda t: -np.cos(2 * t) - 2 * kr * t - kr * np.sin(2 * t)
alpha_x = 0.5 * (F(phi_ex) - F(phi_st))
I = np.pi * D**4 / 64
A = np.pi * D**2 / 4


def dynamics(l):
    k = 3 * E * I / l**3
    m = RHO * A * l
    w = np.sqrt(k / m)
    return dict(l_mm=l * 1e3, k_N_per_mm=k / 1e3, m_kg=m, f1_Hz=w / (2 * np.pi),
                c_Ns_per_m=2 * XI * np.sqrt(k * m), w=w, k=k)


def lobes(dyn, N_grid):
    k_mm, w = dyn["k"] / 1e3, dyn["w"]
    r = 1 + np.logspace(-7, 0, 200000)
    ReG = (1 / k_mm) * (1 - r**2) / ((1 - r**2)**2 + 4 * XI**2 * r**2)
    Ap = 2 * np.pi / (Z * alpha_x * KT * ReG)
    eps = 2 * np.pi - 2 * np.arctan((r**2 - 1) / (2 * XI * r))
    env = np.full_like(N_grid, np.inf)
    m = 0
    while True:
        N = 60 * w * r / (Z * (2 * np.pi * m + eps))
        if N.max() < N_grid[0]:
            break
        if N.min() < N_grid[-1]:
            i = np.argsort(N)
            env = np.minimum(env, np.interp(N_grid, N[i], Ap[i], left=np.inf, right=np.inf))
        m += 1
    return env, Ap.min(), m


N_grid = np.linspace(N_MIN, N_MAX, 25000)
rows, envs = [], {}
for l_mm in FREE_LENGTHS:
    dyn = dynamics(l_mm / 1e3)
    env, ap_lim, m_max = lobes(dyn, N_grid)
    envs[l_mm] = env
    stable = env > AP_TARGET
    ok = bool(stable.any() and (~stable).any())
    rows.append(dict(l_mm=l_mm, k_N_per_mm=dyn["k_N_per_mm"], m_kg=dyn["m_kg"], f1_Hz=dyn["f1_Hz"],
                     c_Ns_per_m=dyn["c_Ns_per_m"], Ap_lim_mm=ap_lim, Ap_max_mm=float(env.max()),
                     N_at_Ap_max=float(N_grid[env.argmax()]), lobe_orders_up_to=m_max,
                     stable_fraction=float(stable.mean()), check=ok))
    print(f"l = {l_mm} mm: k = {dyn['k_N_per_mm']:.0f} N/mm, m = {dyn['m_kg']*1e3:.1f} g, "
          f"f1 = {dyn['f1_Hz']:.0f} Hz, Ap_lim = {ap_lim:.2f} mm, max Ap in range = {env.max():.2f} mm, "
          f"stable at {AP_TARGET:.0f} mm: {100*stable.mean():.0f} % of the range -> {'OK' if ok else 'NO'}")

# ---- final configuration: shortest free length (stiffest tool) ----
l_fin = FREE_LENGTHS[-1]
env = envs[l_fin]
Vc = lambda N: np.pi * D * 1e3 * N / 1000
# stable windows at the target depth; take the one closest to the maximum speed (highest Vc)
s = np.flatnonzero(env > AP_TARGET)
blocks = np.split(s, np.flatnonzero(np.diff(s) > 1) + 1)
win_st = blocks[-1]
i_st = int(win_st[len(win_st) // 2])                      # middle of the window -> margin on both sides
N_st = float(N_grid[i_st])
win = (N_grid > N_st - 400) & (N_grid < N_grid[win_st[0]])
i_un = int(np.flatnonzero(win)[env[win].argmin()])        # lobe minimum just below the stable window
N_un = float(N_grid[i_un])
final = dict(l_mm=l_fin, Ap_target_mm=AP_TARGET, alpha_x=alpha_x,
             N_stable_rpm=N_st, Vc_stable=float(Vc(N_st)), Ap_lim_at_N_stable=float(env[i_st]), Ap_peak_mm=float(env[win_st].max()), N_peak_rpm=float(N_grid[win_st[env[win_st].argmax()]]),
             stable_window_rpm=[float(N_grid[win_st[0]]), float(N_grid[win_st[-1]])],
             N_unstable_rpm=N_un, Vc_unstable=float(Vc(N_un)), Ap_lim_at_N_unstable=float(env[i_un]),
             tooth_pass_stable_Hz=Z * N_st / 60, Vc_max=float(Vc(N_MAX)))
(OUT / "results" / "iterations.json").write_text(json.dumps(rows, indent=2))
(OUT / "results" / "final.json").write_text(json.dumps(final, indent=2))
print(json.dumps(final, indent=2))

# ---- LaTeX table of the iterations ----
with open(OUT / "results" / "iterations_rows.tex", "w") as f:
    for r in rows:
        f.write(f"{r['l_mm']:.0f} & {r['k_N_per_mm']:.0f} & {r['m_kg']*1e3:.1f} & {r['f1_Hz']:.0f} & "
                f"{r['Ap_lim_mm']:.2f} & {r['Ap_max_mm']:.2f} & {'yes' if r['check'] else 'no'} \\\\\n")

# ---- plots ----
fig, ax = plt.subplots(figsize=(9, 5))
for l_mm in FREE_LENGTHS:
    ax.plot(N_grid, envs[l_mm], lw=1.3, label=f"l = {l_mm} mm")
ax.axhline(AP_TARGET, c="k", ls="--", lw=1.2, label=f"target Ap = {AP_TARGET:.0f} mm")
ax.set_xlim(N_MIN, N_MAX); ax.set_ylim(0, 30); ax.grid(alpha=0.3)
ax.set_xlabel("Spindle speed N [rpm]"); ax.set_ylabel("Axial depth of cut Ap [mm]")
ax.set_title("Slot milling - analytical stability limit for different free tool lengths")
ax.legend(fontsize=8, loc="upper left", ncol=2)
fig.tight_layout(); fig.savefig(OUT / "plots" / "lobes_free_length.png", dpi=200); fig.savefig(OUT / "plots" / "lobes_free_length.pdf")

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(N_grid, env, "k", lw=1.8, label=f"stability limit, l = {l_fin} mm")
ax.fill_between(N_grid, 0, np.fmin(env, 40), color="tab:green", alpha=0.12)
ax.axhline(AP_TARGET, c="gray", ls="--", lw=1.2, label=f"target Ap = {AP_TARGET:.0f} mm")
ax.plot(N_st, AP_TARGET, "o", c="tab:green", ms=8, label=f"stable: N = {N_st:.0f} rpm")
ax.plot(N_un, AP_TARGET, "X", c="tab:red", ms=9, label=f"unstable: N = {N_un:.0f} rpm")
ax.set_xlim(4000, N_MAX); ax.set_ylim(0, 30); ax.grid(alpha=0.3)
ax.set_xlabel("Spindle speed N [rpm]"); ax.set_ylabel("Axial depth of cut Ap [mm]")
ax.set_title(f"Slot milling - analytical stability lobes, free length l = {l_fin} mm")
ax.legend(fontsize=8, loc="upper left")
fig.tight_layout(); fig.savefig(OUT / "plots" / "lobes_final.png", dpi=200); fig.savefig(OUT / "plots" / "lobes_final.pdf")

# ---- stable windows at the target depth, per free length ----
wins = {}
for l_mm in FREE_LENGTHS:
    s = np.flatnonzero(envs[l_mm] > AP_TARGET)
    bl = np.split(s, np.flatnonzero(np.diff(s) > 1) + 1) if s.size else []
    wins[l_mm] = [[round(float(N_grid[b[0]])), round(float(N_grid[b[-1]]))] for b in bl]
(OUT / "results" / "stable_windows.json").write_text(json.dumps(wins, indent=1))
print(wins)
