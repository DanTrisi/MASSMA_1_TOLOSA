"""
Drilling part of Assignment 1 (point 4).

With the specific cutting pressure identified in turning on the smallest diameter
at constant spindle speed (X = 24 mm, N = 1200 rpm) and the corresponding cutting
speed, predict feed force, cutting power and torque along time for:
  - pre-drilling  D = 5 mm,  point angle 120 deg, machining length 10 mm, f = 0.2 mm/rev
  - drilling      D = 13 mm, point angle 120 deg, machining length 20 mm, f = 0.2 mm/rev,
    in the same position (inside the 5 mm pilot hole).
Kf = 0.4 Kc (indication of the course). The machining length is taken as the
depth of the drill tip below the part surface at the end of the feed.

Run (after turning_analysis.py, which writes the reference Kc):
    uv run python src/drilling_analysis.py
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from machining import TABLE_DIR, DrillingOperation
from plotstyle import C, apply_style, mark_times, save

KF_OVER_KC = 0.4
POINT_ANGLE = 120.0
F = 0.2

QUANTITIES = [("Ff", r"Feed force $F_f$ (N)", C["ff"]),
              ("Pc", r"Cutting power $P_c$ (W)", C["pc"]),
              ("Mc", r"Torque $M_c$ (N$\cdot$m)", C["cut"])]


def operations(kc: float, vc: float) -> list[DrillingOperation]:
    pre = DrillingOperation("d5", d=5.0, point_angle=POINT_ANGLE, vc=vc, f=F, kc=kc,
                            kf_over_kc=KF_OVER_KC, length=10.0)
    drill = DrillingOperation("d13", d=13.0, point_angle=POINT_ANGLE, vc=vc, f=F, kc=kc,
                              kf_over_kc=KF_OVER_KC, length=20.0,
                              d_pilot=pre.d, pilot_length=pre.length)
    return [pre, drill]


def title(op: DrillingOperation) -> str:
    extra = f", inside the D{op.d_pilot:.0f} pilot hole" if op.d_pilot else ""
    return (f"D = {op.d:.0f} mm{extra}: N = {op.n:.0f} rpm, "
            f"$V_f$ = {op.vf:.0f} mm/min, L = {op.length:.0f} mm")


def plot_operation(op: DrillingOperation, pred: pd.DataFrame) -> None:
    ev = op.events()
    fig, axes = plt.subplots(3, 1, figsize=(7.5, 6.6), sharex=True)
    for ax, (col, lab, color) in zip(axes, QUANTITIES):
        ax.plot(pred.t, pred[col], color=color, lw=1.8)
        ax.set_ylabel(lab)
        ax.set_ylim(0, pred[col].max() * 1.25)
        mark_times(ax, {f"${k}$": v[1] for k, v in ev.items()})
    axes[0].set_title(title(op))
    axes[-1].set_xlabel("Time from the start of the feed (s)")
    save(fig, f"drilling_prediction_{op.name}")


def plot_sequence(ops, preds) -> None:
    """Both operations on the same vertical scale (one column per operation)."""
    fig, axes = plt.subplots(3, 2, figsize=(10, 6.4), sharey="row",
                             gridspec_kw={"width_ratios": [1, 2.4]})
    for j, (op, pred) in enumerate(zip(ops, preds)):
        for i, (col, lab, color) in enumerate(QUANTITIES):
            ax = axes[i, j]
            ax.plot(pred.t, pred[col], color=color, lw=1.8)
            mark_times(ax, {f"${k}$": v[1] for k, v in op.events().items()})
            if j == 0:
                ax.set_ylabel(lab)
        axes[0, j].set_title(f"{'Pre-drilling' if j == 0 else 'Drilling'} D{op.d:.0f}")
        axes[-1, j].set_xlabel("Time (s)")
    for i, (col, _, _) in enumerate(QUANTITIES):
        axes[i, 0].set_ylim(0, max(p[col].max() for p in preds) * 1.25)
    save(fig, "drilling_sequence")


def plot_edge_speed(ops) -> None:
    pre, drill = ops
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    for op, color in [(pre, C["p1"]), (drill, C["p2"])]:
        r = np.linspace(0, op.d / 2, 100)
        ax.plot(r, 2 * np.pi * r * op.n / 1000, color=color,
                label=f"D{op.d:.0f}, N = {op.n:.0f} rpm")
    ax.axvspan(drill.d_pilot / 2, drill.d / 2, color=C["band"], lw=0, zorder=0)
    ax.annotate("lip length cutting while\nenlarging the pilot hole",
                (drill.d_pilot / 2 + 0.15, 10), fontsize=8, color="#3a3a37")
    ax.set_xlabel("Radius on the lip r (mm)")
    ax.set_ylabel(r"$V_c(r)$ (m/min)")
    ax.set_title("Cutting speed along the lips")
    ax.legend(loc="upper left")
    save(fig, "drilling_edge_speed")


def main() -> None:
    apply_style()
    ref = pd.read_csv(TABLE_DIR / "kc_reference.csv").iloc[0]
    kc, vc = float(ref.Kc), float(ref.Vc)
    print(f"Reference from turning (X = {ref.X:.0f} mm, N = {ref.N:.0f} rpm): "
          f"Kc = {kc:.1f} MPa, Vc = {vc:.2f} m/min")
    ops = operations(kc, vc)
    preds = [op.predict() for op in ops]
    macros = {"DrKc": f"{kc:.0f}", "DrKf": f"{KF_OVER_KC * kc:.0f}", "DrVc": f"{vc:.1f}"}
    rows = []
    for op, pred, tag in zip(ops, preds, ["Five", "Thirteen"]):
        ev = op.events()
        print(f"\n{op.name}: N = {op.n:.1f} rpm, Vf = {op.vf:.1f} mm/min")
        for k, (z, t, txt) in ev.items():
            print(f"  {k}: z = {z:6.3f} mm, t = {t:6.3f} s  ({txt})")
        energy = np.trapezoid(pred.Pc, pred.t)
        macros.update({f"Dr{tag}N": f"{op.n:.0f}", f"Dr{tag}Vf": f"{op.vf:.0f}",
                       f"Dr{tag}E": f"{energy:.0f}"})
        words = {"t_0": "Zero", "t_1": "One", "t_2": "Two", "t_3": "Three"}
        for k, (z, t, txt) in ev.items():       # LaTeX macro names cannot hold digits
            macros[f"Dr{tag}T{words[k]}"] = f"{t:.3f}"
            macros[f"Dr{tag}Z{words[k]}"] = f"{z:.2f}"
        # steady values of each phase (sampled in the middle of the phase)
        keys = list(ev)
        phases = [(keys[i], keys[i + 1]) for i in range(len(keys) - 1)]
        for a, b in phases[1:]:
            tm = 0.5 * (ev[a][1] + ev[b][1])
            s = pred.iloc[(pred.t - tm).abs().idxmin()]
            phase = f"{a}-{b}"
            print(f"  steady {phase}: Ff = {s.Ff:.1f} N, Pc = {s.Pc:.1f} W, "
                  f"Mc = {s.Mc:.3f} N.m, Pf = {s.Pf:.2f} W, lip r = {s.r_in:.2f}-{s.r_out:.2f}")
            la, lb = (k.replace("_", "_{") + "}" for k in (a, b))
            rows.append(f"D{op.d:.0f} & ${la}$--${lb}$ & "
                        f"{s.r_in:.1f}--{s.r_out:.1f} & {s.Ff:.0f} & {s.Pc:.0f} & "
                        f"{s.Mc:.2f} & {s.Pf:.1f} \\\\")
        plot_operation(op, pred)
        print(f"  cutting energy = {energy:.0f} J")
    plot_sequence(ops, preds)
    plot_edge_speed(ops)

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    (TABLE_DIR / "drilling_rows.tex").write_text("\n".join(rows) + "\n")
    (TABLE_DIR / "drilling_macros.tex").write_text(
        "\n".join(rf"\newcommand{{\{k}}}{{{v}}}" for k, v in macros.items()) + "\n")


if __name__ == "__main__":
    main()
