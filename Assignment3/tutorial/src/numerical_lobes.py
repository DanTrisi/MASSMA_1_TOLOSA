"""
Tutorial part 4 - numerical stability lobes by semi-discretization (Insperger & Stepan 2011).
Corrected version of the course file "2.2 numerical_lobes.py":

  1. MC = inv(C) @ C  ->  inv(M) @ C          (damping term c/m was replaced by 1)
  2. 1x1 arrays assigned to scalar cells      (ValueError with numpy >= 2.x) -> scalars
  3. tooth angle reduced modulo 2*pi           (slide 26: g_j uses phi_j mod 2pi; needed with helix)
  4. up/down labels: fist = 0 .. arccos(1-2aD) is UP milling, arccos(2aD-1) .. pi is DOWN milling

Usage:  python numerical_lobes.py --helix 0      (question 4.1/4.2)
        python numerical_lobes.py --helix 45     (question 4.3)
"""
import argparse, time, json
from pathlib import Path
import numpy as np
from scipy.linalg import expm, eigvals
import matplotlib.pyplot as plt
import params as P

ap = argparse.ArgumentParser()
ap.add_argument("--helix", type=float, default=0.0, help="helix angle [deg]")
ap.add_argument("--mode", default=P.MODE, choices=["up", "down"])
ap.add_argument("--stx", type=int, default=120)
ap.add_argument("--sty", type=int, default=120)
args = ap.parse_args()
OUT = Path(__file__).resolve().parents[1]
t0 = time.time()

# ---- model parameters (SI) ----
N   = P.Z                 # number of teeth
D   = P.D                 # tool diameter [m]
eta = np.radians(args.helix)
Kt  = P.KT                # [N/m^2]
kr  = P.KR_KT             # Kr/Kt [-]
f1  = P.F1                # natural frequency [Hz]
rd1 = P.XI                # damping ratio [-]
s1  = P.K1                # stiffness [N/m]
aD  = P.AE_D              # Ae/D [-]

if args.mode == "up":
    fist, fiex = 0.0, np.arccos(1 - 2 * aD)
else:
    fist, fiex = np.arccos(2 * aD - 1), np.pi

# ---- grid ----
stx, sty = args.stx, args.sty
wst, wfi = 0.0, 30.0          # depth of cut [mm]
nost, ofi = 1000.0, 10000.0   # spindle speed [rpm]

# ---- semi-discretization parameters ----
m, n, intn = 30, 30, 30
wa = wb = 0.5

MM = np.zeros((n + 2, n + 2))
d = np.ones(n + 1); d[:2] = 0
MM += np.diag(d, -1)
MM[2, 0] = 1

wn = 2 * np.pi * f1
mass = s1 / wn**2                 # [kg]
c = 2 * rd1 * wn * mass           # [Ns/m]
k_m = s1 / mass                   # k/m
c_m = c / mass                    # c/m   (fix 1)

ei = np.zeros((stx + 1, sty + 1)); ss = np.zeros_like(ei); dc = np.zeros_like(ei)

for y in range(sty + 1):
    w = (wst + y * (wfi - wst) / sty) / 1000          # depth of cut [m]
    # directional coefficient h_i over one tooth period (vectorised over z, teeth, sub-steps)
    dtr = 2 * np.pi / N / n
    dz = w / m
    z = (np.arange(1, m + 1) - 0.5) * dz                      # (m,)
    kk = np.arange(N)                                         # (N,)
    hh = np.arange(1, intn + 1) * dtr / intn                  # (intn,)
    h_i = np.zeros(n)
    for i in range(1, n + 1):
        fi = (i * dtr - np.tan(eta) / (D / 2) * z[:, None, None]
              + kk[None, :, None] * 2 * np.pi / N + hh[None, None, :])
        fi = np.mod(fi, 2 * np.pi)                            # fix 3
        g = (fi >= fist) & (fi <= fiex)
        h_i[i - 1] = np.sum(Kt * g * (-np.sin(fi) + kr * np.cos(fi)) * np.cos(fi) * dz) / intn
    for x in range(stx + 1):
        o = nost + x * (ofi - nost) / stx                     # spindle speed [rpm]
        tau = 60 / o / N
        dt = tau / n
        Fi = np.eye(n + 2)
        for i in range(n):
            A = np.array([[0.0, 1.0], [-k_m - h_i[i] / mass, -c_m]])   # fix 2
            B = np.array([[0.0, 0.0], [h_i[i] / mass, 0.0]])
            Pm = expm(A * dt)
            R = (Pm - np.eye(2)) @ np.linalg.solve(A, B)
            MM[:2, :2] = Pm
            MM[:2, n] = wa * R[:, 0]
            MM[:2, n + 1] = wb * R[:, 0]
            Fi = MM @ Fi
        ei[x, y] = np.max(np.abs(eigvals(Fi)))
        ss[x, y] = o
        dc[x, y] = w * 1000

tag = f"helix{int(args.helix)}_{args.mode}"
np.savez(OUT / "results" / f"numerical_{tag}.npz", ss=ss, dc=dc, ei=ei)
print(f"[{tag}] calculation time: {time.time() - t0:.1f} s")

fig, ax = plt.subplots(figsize=(9, 5))
ax.contourf(ss, dc, ei, levels=[0, 1], colors=["tab:green"], alpha=0.12)
ax.contour(ss, dc, ei, levels=[1], colors="k", linewidths=2)
ax.set_xlabel("Spindle speed N [rpm]"); ax.set_ylabel("Axial depth of cut Ap [mm]")
ax.set_xlim(nost, ofi); ax.set_ylim(wst, wfi); ax.grid(alpha=0.3)
ax.set_title(f"Numerical stability lobes (semi-discretization), helix {args.helix:g} deg, {args.mode} milling")
fig.tight_layout()
fig.savefig(OUT / "plots" / f"numerical_{tag}.png", dpi=200)
fig.savefig(OUT / "plots" / f"numerical_{tag}.pdf")
