"""Question 4.2 / 4.3 - analytical vs numerical lobes, and key numbers."""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import params as P

OUT = Path(__file__).resolve().parents[1]
an = np.load(OUT / "results" / "analytical_envelope.npz")
res = {}


def critical_depth(d):
    """First depth (per spindle speed) where the eigenvalue norm reaches 1."""
    ss, dc, ei = d["ss"][:, 0], d["dc"][0, :], d["ei"]
    ap = np.full(len(ss), np.nan)
    for x in range(len(ss)):
        idx = np.flatnonzero(ei[x] >= 1)
        if idx.size and idx[0] > 0:            # linear interpolation between grid points
            j = idx[0]
            e0, e1 = ei[x, j - 1], ei[x, j]
            ap[x] = dc[j - 1] + (1 - e0) / (e1 - e0) * (dc[j] - dc[j - 1])
    return ss, ap


fig, axs = plt.subplots(2, 1, figsize=(9, 9), sharex=True)
for ax, tag, title in zip(axs, ["helix0_down", "helix45_down"], ["no helix (4.1-4.2)", "helix 45 deg (4.3)"]):
    d = np.load(OUT / "results" / f"numerical_{tag}.npz")
    ss, ap = critical_depth(d)
    ax.contour(d["ss"], d["dc"], d["ei"], levels=[1], colors="k", linewidths=2)
    ax.plot([], [], "k", lw=2, label="numerical (semi-discretization)")
    ax.plot(an["N"], an["Ap"], c="tab:blue", lw=1.5, ls="--", label="analytical (Altintas)")
    hopf = (ss < 6000); flip = (ss > 6300) & (ss < 8000)
    r = dict(numerical_min_Ap_mm=float(np.nanmin(ap)),
             numerical_min_Ap_at_rpm=float(ss[np.nanargmin(ap)]),
             hopf_min_Ap_mm=float(np.nanmin(ap[hopf])), hopf_min_at_rpm=float(ss[hopf][np.nanargmin(ap[hopf])]),
             flip_min_Ap_mm=float(np.nanmin(ap[flip])), flip_min_at_rpm=float(ss[flip][np.nanargmin(ap[flip])]))
    res[tag] = r
    ax.axvline(2 * P.F1 * 60 / P.Z, c="tab:red", ls=":", lw=1.2, label="tooth pass freq. = 2 f1")
    ax.set_ylim(0, 30); ax.set_xlim(1000, 10000); ax.grid(alpha=0.3)
    ax.set_ylabel("Axial depth of cut Ap [mm]"); ax.set_title(title); ax.legend(fontsize=8, loc="upper left")
axs[1].set_xlabel("Spindle speed N [rpm]")
p = np.pi * P.D / (P.Z * np.tan(P.ETA)) * 1e3
axs[1].annotate("", xy=(9600, 2.5 + p), xytext=(9600, 2.5), arrowprops=dict(arrowstyle="<->"))
axs[1].text(9550, 2.5 + p / 2, f"p = {p:.1f} mm", va="center", ha="right", fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "plots" / "analytical_vs_numerical.png", dpi=200)
fig.savefig(OUT / "plots" / "analytical_vs_numerical.pdf")

res["N_ratio2_rpm"] = 2 * P.F1 * 60 / P.Z
res["helix_pitch_mm"] = p
(OUT / "results" / "numerical_summary.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
