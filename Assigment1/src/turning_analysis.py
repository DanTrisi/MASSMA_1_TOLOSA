"""
Turning part of Assignment 1.

1) Predict the cutting power of the two programs of the chapter with the specific
   cutting pressure given by the chapter for the smallest diameter (X = 24 mm,
   N = 1200 rpm): Kc = 3230.3 MPa.
2) Compare the predicted timings / power profile with the total spindle power
   recorded by the CNC (GitHub data: cutting run and air-cutting run).
3) Identify the steady-state cutting power and the spindle power without cutting,
   and compute the specific cutting pressure Kc of every pass. The value of the
   smallest diameter of Program II is exported for the drilling prediction.

Run:  python src/turning_analysis.py   (or python run_all.py)
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from machining import (DATA_DIR, PLOT_DIR, TABLE_DIR, TurningProgram, check_data,
                       find_feed_passes, kc_turning, load_trace, moving_average,
                       steady_state_mean, steady_state_range, time_at_z)
from plotstyle import C, apply_style, mark_times, save

# --------------------------------------------------------------------------
# Data of the chapter "Integrating Environmental Impact into Cutting Speed
# Decisions in CNC Turning" (A. C. Araujo, "Inspiration for your assignment"):
# AISI 4135 bar, D 48 -> 24 mm over 40 mm, DNMG NC5330 coated carbide insert,
# ap = 1 mm, f = 0.2 mm/rev.
# --------------------------------------------------------------------------
AP, F = 1.0, 0.2
DIAMETERS = np.arange(46.0, 23.0, -2.0)
NOSE_RADIUS = 0.8       # [mm] DNMG 1506 08 - assumption, only for the t0 -> t1 ramp
KC_CHAPTER = 3230.3     # [MPa] chapter Table 2, Program 2 (N = 1200 rpm), X = 24 mm
# chapter Table 2: Kc (MPa) of every pass, X = 46 ... 24 mm
KC_CHAPTER_TABLE = {
    "vc150": [3012.4, 2967.6, 2984.8, 3021.4, 2997.9, 3024.8,
              3034.9, 2997.6, 2993.7, 3003.1, 2996.3, 3025.6],
    "n1200": [3019.6, 2999.6, 3027.6, 3046.9, 3068.8, 3088.4,
              3117.8, 3120.7, 3150.4, 3178.8, 3190.5, 3230.3],
}

PROGRAMS = {
    "vc150": dict(prog=TurningProgram("vc150", r"Program I ($V_c$ = 150 m/min)", AP, F,
                                      DIAMETERS, vc=150.0),
                  cut="ANNACHPTVC150.csv", air="ANNACHPTVC150VIDE.csv"),
    "n1200": dict(prog=TurningProgram("n1200", r"Program II ($N$ = 1200 rpm)", AP, F,
                                      DIAMETERS, n=1200.0),
                  cut="ANNACHPT1200RPM.csv", air="ANNACHPT1200RPMVIDE.csv"),
}
# steady-state window on the programmed Z (tool fully engaged, far from entry/exit)
Z_HI, Z_LO = -3.0, -37.0


def analyse_program(key: str) -> dict:
    cfg = PROGRAMS[key]
    prog: TurningProgram = cfg["prog"]
    check_data(cfg["cut"], cfg["air"])
    cut, air = load_trace(DATA_DIR / cfg["cut"]), load_trace(DATA_DIR / cfg["air"])
    pc_pass, pa_pass = find_feed_passes(cut), find_feed_passes(air)
    assert len(pc_pass) == len(pa_pass) == len(prog.diameters), "pass detection failed"

    pred = prog.predict(KC_CHAPTER, NOSE_RADIUS)
    rows = []
    dt = cut["time"].diff().median()
    p_smooth = moving_average(cut["power"].to_numpy(), int(round(0.1 / dt)))
    for (_, pc), (_, pa), (_, pr) in zip(pc_pass.iterrows(), pa_pass.iterrows(),
                                         pred.iterrows()):
        p_tot, s_tot = steady_state_mean(cut, pc.i_start, pc.i_end, Z_HI, Z_LO)
        p_air, s_air = steady_state_mean(air, pa.i_start, pa.i_end, Z_HI, Z_LO)
        p_cut = p_tot - p_air
        # minimum / maximum of the noisy signal (1 % and 99 % percentiles)
        tot_min, tot_max = steady_state_range(cut, pc.i_start, pc.i_end, Z_HI, Z_LO)
        air_min, air_max = steady_state_range(air, pa.i_start, pa.i_end, Z_HI, Z_LO)
        # program timings, read on the measured position signal
        t_feed = pc.t_start
        t0 = time_at_z(cut, pc.i_start, pc.i_end, prog.z_contact)
        t2 = time_at_z(cut, pc.i_start, pc.i_end, prog.z_end)
        # timings seen on the power signal: 50 % of the cutting power above idle
        seg = slice(int(pc.i_start), int(pc.i_end) + int(round(0.5 / dt)))
        ts, ps = cut["time"].to_numpy()[seg], p_smooth[seg]
        above = ps > p_air + 0.5 * p_cut
        i_on = np.argmax(above)
        i_off = i_on + np.argmin(above[i_on:]) - 1   # end of the first run above threshold
        t_on, t_off = ts[i_on], ts[i_off]
        # power ramp: 10 % -> 90 % of the cutting power
        rise = ts[np.flatnonzero(ps > p_air + 0.9 * p_cut)[0]] - \
            ts[np.flatnonzero(ps > p_air + 0.1 * p_cut)[0]]
        rows.append({
            "pass": pc["pass"], "X": pc.X, "N": pr.N, "Vc": pr.Vc,
            "Vf_pred": pr.Vf, "Vf_meas": pc.Vf_meas,
            "t_feed": t_feed, "t0_prog": t0, "t2_prog": t2,
            "t0_pred": t_feed + pr.dt_t0, "t1_pred": t_feed + pr.dt_t1,
            "t2_pred": t_feed + pr.dt_t2,
            "t_on": t_on, "t_off": t_off, "rise": rise,
            "Tcut_pred": pr.dt_t2 - pr.dt_t0, "Tcut_meas": t_off - t_on,
            "P_total": p_tot, "P_total_std": s_tot, "P_air": p_air, "P_air_std": s_air,
            "Pc_meas": p_cut, "Pc_pred": pr.Pc_pred,
            "Kc": float(kc_turning(p_cut, F, AP, pr.Vc)),
            "P_total_min": tot_min, "P_total_max": tot_max,
            "P_air_min": air_min, "P_air_max": air_max,
            # minimum with minimum, maximum with maximum (the ripple is common to both runs)
            "Pc_min": tot_min - air_min, "Pc_max": tot_max - air_max,
            "Kc_min": float(kc_turning(tot_min - air_min, F, AP, pr.Vc)),
            "Kc_max": float(kc_turning(tot_max - air_max, F, AP, pr.Vc)),
            # crossed pairing (worst case), for comparison only
            "Kc_cross_lo": float(kc_turning(tot_min - air_max, F, AP, pr.Vc)),
            "Kc_cross_hi": float(kc_turning(tot_max - air_min, F, AP, pr.Vc)),
            "Fc": p_cut / (pr.Vc / 60.0),
            "MRR": pr.MRR,
        })
    res = pd.DataFrame(rows)
    res["Kc_chapter"] = KC_CHAPTER_TABLE[key]
    res["Pc_err"] = 100 * (res.Pc_pred - res.Pc_meas) / res.Pc_meas
    # energy of each pass (cutting run, between t_on and t_off) and SEC
    vol = np.pi / 4 * ((res.X + 2 * AP) ** 2 - res.X ** 2) * (prog.z_contact - prog.z_end)
    res["E_cut_kJ"] = res.Pc_meas * res.Tcut_meas / 1000
    res["E_tot_kJ"] = res.P_total * res.Tcut_meas / 1000
    res["SEC_cut"] = res.E_cut_kJ * 1000 / vol   # J/mm^3
    res["SEC_tot"] = res.E_tot_kJ * 1000 / vol
    return dict(prog=prog, cut=cut, air=air, passes_cut=pc_pass, passes_air=pa_pass,
                res=res)


def predicted_trace(out: dict) -> tuple[np.ndarray, np.ndarray]:
    """Predicted total power along the cutting-run time axis:
    spindle power without cutting (identified) + mechanistic Pc during cutting,
    with a linear ramp between t0 and t1."""
    t = out["cut"]["time"].to_numpy()
    p = np.full_like(t, np.nan)
    for _, r in out["res"].iterrows():
        m = (t >= r.t_feed) & (t <= r.t2_pred)
        ramp = np.clip((t[m] - r.t0_pred) / (r.t1_pred - r.t0_pred), 0, 1)
        p[m] = r.P_air + r.Pc_pred * ramp
    return t, p


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
def plot_program_path(outs: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, (key, out) in zip(axes, outs.items()):
        cut = out["cut"]
        ax.plot(cut["time"], cut["prog_pos_z"], color=C["cut"], label="Z (axial)")
        ax.plot(cut["time"], cut["prog_pos_x"], color=C["air"], label="X (diameter)")
        ax.set_ylim(-45, 60)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Programmed position (mm)")
        ax.set_title(out["prog"].label)
        ax.legend(loc="upper right")
    save(fig, "turning_programmed_path")


def plot_raw_power(out: dict) -> None:
    cut, air, prog = out["cut"], out["air"], out["prog"]
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(air["time"], air["power"], color=C["air"], lw=0.6, label="Air cutting (no chip)")
    ax.plot(cut["time"], cut["power"], color=C["cut"], lw=0.6, label="Cutting")
    for _, r in out["res"].iterrows():
        ax.axvspan(r.t0_prog, r.t2_prog, color=C["band"], lw=0, zorder=0)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Spindle power (W)")
    ax.set_title(f"{prog.label}: recorded spindle power (shaded: tool in the bar)")
    ax.set_ylim(-1500, 11000)
    ax.legend(loc="upper left", ncol=2)
    save(fig, f"turning_power_raw_{prog.name}")


def plot_prediction_vs_measure(out: dict) -> None:
    cut, prog, res = out["cut"], out["prog"], out["res"]
    t_pred, p_pred = predicted_trace(out)
    dt = cut["time"].diff().median()
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(cut["time"], cut["power"], color=C["cut_light"], lw=0.5, label="Measured (raw)")
    ax.plot(cut["time"], moving_average(cut["power"].to_numpy(), int(0.2 / dt)),
            color=C["cut"], lw=1.2, label="Measured (0.2 s moving average)")
    ax.plot(t_pred, p_pred, color=C["pred"], lw=1.6, ls="--",
            label=rf"Predicted: $P_0$ + $K_c f a_p V_c$ ($K_c$ = {KC_CHAPTER:.0f} MPa)")
    ax.set_ylim(0, 3200)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Spindle power (W)")
    ax.set_title(f"{prog.label}: predicted vs measured total power")
    ax.legend(loc="lower right", ncol=3, fontsize=7.5)
    save(fig, f"turning_prediction_vs_measured_{prog.name}")


def plot_zoom_pass(outs: dict, k: int = 3) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, out in zip(axes, outs.values()):
        cut, air, res = out["cut"], out["air"], out["res"]
        r = res.iloc[k - 1]
        pa = out["passes_air"].iloc[k - 1]
        shift = r.t_feed - pa.t_start            # align the air run on the same pass
        lo, hi = r.t_feed - 1.0, r.t2_prog + 1.5
        m = (cut["time"] > lo) & (cut["time"] < hi)
        ma = (air["time"] + shift > lo) & (air["time"] + shift < hi)
        t_pred, p_pred = predicted_trace(out)
        mp = (t_pred > lo) & (t_pred < hi)
        ax.plot(air["time"][ma] + shift, air["power"][ma], color=C["air"], lw=0.6,
                label="Air cutting")
        ax.plot(cut["time"][m], cut["power"][m], color=C["cut"], lw=0.6, label="Cutting")
        ax.plot(t_pred[mp], p_pred[mp], color=C["pred"], lw=1.6, ls="--", label="Predicted")
        mark_times(ax, {"$t_0$": r.t0_pred, "$t_1$": r.t1_pred, "$t_2$": r.t2_pred})
        ax.set_xlim(lo, hi)
        ax.set_ylim(-1200, 5000)
        ax.set_xlabel("Time (s)")
        ax.set_title(f"{out['prog'].label}, pass {k} (X = {r.X:.0f} mm)")
    axes[0].set_ylabel("Spindle power (W)")
    axes[1].legend(loc="upper center", ncol=3)
    save(fig, "turning_zoom_pass")


def plot_entry_exit(outs: dict, k: int = 3) -> None:
    """Close-up of tool entry and exit: timing of the power step vs program."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 5.6), sharey=True)
    for row, out in zip(axes, outs.values()):
        cut, res = out["cut"], out["res"]
        r = res.iloc[k - 1]
        t_pred, p_pred = predicted_trace(out)
        for ax, (centre, name) in zip(row, [(r.t0_pred, "entry"), (r.t2_pred, "exit")]):
            lo, hi = centre - 0.6, centre + 0.6
            m = (cut["time"] > lo) & (cut["time"] < hi)
            mp = (t_pred > lo) & (t_pred < hi)
            ax.plot(cut["time"][m], cut["power"][m], color=C["cut"], lw=0.8,
                    marker="o", ms=2, label="Measured")
            ax.plot(t_pred[mp], p_pred[mp], color=C["pred"], lw=1.6, ls="--", label="Predicted")
            marks = {"$t_0$": r.t0_pred, "$t_1$": r.t1_pred} if name == "entry" \
                else {"$t_2$": r.t2_pred}
            mark_times(ax, marks)
            ax.set_xlim(lo, hi)
            ax.set_title(f"{out['prog'].label}: tool {name}, pass {k}", fontsize=9)
            ax.set_xlabel("Time (s)")
        row[0].set_ylabel("Spindle power (W)")
    axes[0, 0].legend(loc="upper left")
    save(fig, "turning_entry_exit")


def plot_steady_state(outs: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, out in zip(axes, outs.values()):
        res = out["res"]
        ax.plot(res.X, res.P_total, color=C["cut"], marker="o", label=r"$P_{total}$ (cutting run)")
        ax.plot(res.X, res.P_air, color=C["air"], marker="s", label=r"$P_0$ (air cutting)")
        ax.plot(res.X, res.Pc_meas, color=C["pc"], marker="D",
                label=r"$P_c = P_{total} - P_0$")
        ax.plot(res.X, res.Pc_pred, color=C["pred"], ls="--",
                label=rf"$P_c$ predicted ($K_c$ = {KC_CHAPTER:.0f} MPa)")
        ax.set_xlabel("Programmed diameter X (mm)")
        ax.set_title(out["prog"].label)
        ax.invert_xaxis()
    axes[0].set_ylabel("Steady-state power (W)")
    axes[0].set_ylim(0, 2600)
    axes[1].legend(loc="upper right", fontsize=8)
    save(fig, "turning_steady_state_power")


def plot_kc(outs: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for (key, out), col, mk in zip(outs.items(), [C["p1"], C["p2"]], ["o", "s"]):
        res = out["res"]
        axes[0].plot(res.X, res.Kc, color=col, marker=mk,
                     label=f"{out['prog'].label}: this work")
        axes[0].plot(res.X, res.Kc_chapter, color=col, marker=mk, ls="none", mfc="none",
                     ms=9, label=f"{out['prog'].label}: chapter Table 2")
        axes[1].plot(res.Vc, res.Kc, color=col, marker=mk, ls="none",
                     label=out["prog"].label)
    ref = outs["n1200"]["res"].iloc[-1]
    for ax, x, dx, ha in zip(axes, [ref.X, ref.Vc], [-14, 14], ["right", "left"]):
        ax.annotate(f"reference for drilling\nX = 24 mm, N = 1200 rpm\n$K_c$ = {ref.Kc:.0f} MPa",
                    (x, ref.Kc), xytext=(dx, 6), textcoords="offset points",
                    ha=ha, va="center", fontsize=8)
        ax.set_ylim(2800, 3350)
        ax.set_ylabel(r"$K_c$ (MPa)")
    axes[0].set_xlabel("Programmed diameter X (mm)")
    axes[0].invert_xaxis()
    axes[0].legend(loc="lower left", fontsize=7)
    axes[1].set_xlabel(r"Cutting speed $V_c$ on X (m/min)")
    axes[0].set_title("Specific cutting pressure per pass")
    axes[1].set_title(r"$K_c$ versus cutting speed")
    axes[1].legend(loc="upper right", fontsize=8)
    save(fig, "turning_kc")


def plot_kc_range(outs: dict) -> None:
    """Kc of every pass from the mean powers, with the band obtained from the
    minimum and maximum of the noisy power signals."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, out, col, mk in zip(axes, outs.values(), [C["p1"], C["p2"]], ["o", "s"]):
        res = out["res"]
        lo = np.minimum(res.Kc_min, res.Kc_max)
        hi = np.maximum(res.Kc_min, res.Kc_max)
        ax.fill_between(res.X, lo, hi, color=col, alpha=0.18, lw=0,
                        label="band between the two")
        ax.plot(res.X, res.Kc_min, color=col, lw=1.0, ls="--", marker="v", ms=4,
                label=r"$K_c$ from minimum powers")
        ax.plot(res.X, res.Kc_max, color=col, lw=1.0, ls=":", marker="^", ms=4,
                label=r"$K_c$ from maximum powers")
        ax.plot(res.X, res.Kc, color=col, marker=mk, label=r"$K_c$ from mean powers")
        ax.set_ylim(2700, 3450)
        ax.set_xlabel("Programmed diameter X (mm)")
        ax.set_title(out["prog"].label)
        ax.invert_xaxis()
        ax.legend(loc="upper left")
    axes[0].set_ylabel(r"$K_c$ (MPa)")
    save(fig, "turning_kc_range")


def plot_idle_power(outs: dict) -> tuple[np.ndarray, float]:
    r1, r2 = outs["vc150"]["res"], outs["n1200"]["res"]
    coef = np.polyfit(r1.N, r1.P_air, 2)
    nn = np.linspace(1000, 2050, 200)
    r2_fit = np.polyval(coef, 1200.0)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    axes[0].plot(r1.N, r1.P_air, color=C["p1"], marker="o", ls="none",
                 label="Program I passes")
    axes[0].plot(r2.N, r2.P_air, color=C["p2"], marker="s", ls="none",
                 label="Program II passes")
    axes[0].plot(nn, np.polyval(coef, nn), color=C["pred"], ls="--",
                 label="Quadratic fit on Program I")
    axes[0].set_xlabel("Spindle speed N (rpm)")
    axes[0].set_ylabel(r"$P_0$ without cutting (W)")
    axes[0].set_title("Spindle power without cutting")
    axes[0].legend(loc="upper left", fontsize=8)
    axes[1].plot(r2["pass"], r2.P_air, color=C["p2"], marker="s")
    axes[1].axhline(r2_fit, color=C["pred"], ls="--", lw=1.2,
                    label=f"Fit of Program I at 1200 rpm ({r2_fit:.0f} W)")
    axes[1].set_xlabel("Pass number (N = 1200 rpm)")
    axes[1].set_ylabel(r"$P_0$ (W)")
    axes[1].set_title("Program II: drift at constant speed")
    axes[1].legend(loc="upper right", fontsize=8)
    save(fig, "turning_idle_power")
    return coef, r2_fit


def plot_timing(outs: dict) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    for out, col, mk in zip(outs.values(), [C["p1"], C["p2"]], ["o", "s"]):
        res = out["res"]
        ax.plot(res.X, res.Tcut_pred, color=col, ls="--", lw=1.2)
        ax.plot(res.X, res.Tcut_meas, color=col, marker=mk, ls="none",
                label=f"{out['prog'].label}")
    ax.plot([], [], color="0.4", ls="--", label="Predicted (40 mm / $V_f$)")
    ax.set_xlabel("Programmed diameter X (mm)")
    ax.set_ylabel("Cutting time per pass (s)")
    ax.invert_xaxis()
    ax.set_title("Cutting time: prediction vs power signal")
    ax.legend(loc="lower left", fontsize=8)
    save(fig, "turning_cutting_time")


def plot_ripple(outs: dict, k: int = 3) -> dict[str, float]:
    """Spectrum of the power during the steady state of pass k: the ripple of the
    raw signal is at the spindle rotation frequency, with and without cutting."""
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    out = outs["n1200"]
    freqs = {}
    for run, color, lab in [("air", C["air"], "Air cutting"), ("cut", C["cut"], "Cutting")]:
        df = out[run]
        ps = out["passes_cut" if run == "cut" else "passes_air"].iloc[k - 1]
        seg = df.iloc[int(ps.i_start):int(ps.i_end)]
        seg = seg[(seg.prog_pos_z <= Z_HI) & (seg.prog_pos_z >= Z_LO)]
        p = seg["power"].to_numpy() - seg["power"].mean()
        amp = 2 * np.abs(np.fft.rfft(p * np.hanning(p.size))) / (p.size / 2)
        fr = np.fft.rfftfreq(p.size, seg["time"].diff().median())
        ax.plot(fr, amp, color=color, lw=1.0, label=lab)
        freqs[run] = fr[np.argmax(amp)]
    ax.axvline(1200 / 60, color=C["mark"], ls=":", lw=1)
    ax.annotate("N/60 = 20 Hz", (20, 1.0), xycoords=("data", "axes fraction"),
                xytext=(3, -12), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude (W)")
    ax.set_title(f"Spectrum of the spindle power, Program II pass {k}")
    ax.legend(loc="upper right")
    save(fig, "turning_power_spectrum")
    return freqs


# --------------------------------------------------------------------------
# LaTeX tables
# --------------------------------------------------------------------------
def write_tables(outs: dict, idle_coef: np.ndarray, idle_fit_1200: float) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    for key, out in outs.items():
        res = out["res"]
        lines = []
        for _, r in res.iterrows():
            lines.append(
                f"{r['pass']:.0f} & {r.X:.0f} & {r.N:.0f} & {r.Vc:.1f} & "
                f"{r.P_total:.0f} & {r.P_air:.0f} & {r.Pc_meas:.0f} & {r.Fc:.0f} & "
                f"{r.Kc:.0f} & {r.Kc_chapter:.0f} \\\\")
        (TABLE_DIR / f"turning_{key}_rows.tex").write_text("\n".join(lines) + "\n")
        lines = []
        for _, r in res.iterrows():
            lines.append(
                f"{r['pass']:.0f} & {r.X:.0f} & {r.Vf_pred:.1f} & {r.Vf_meas:.1f} & "
                f"{r.t0_pred - r.t_feed:.2f} & {r.t_on - r.t_feed:.2f} & "
                f"{r.Tcut_pred:.2f} & {r.Tcut_meas:.2f} & {r.rise:.2f} & "
                f"{r.E_tot_kJ:.1f} & {r.SEC_tot:.2f} \\\\")
        (TABLE_DIR / f"timing_{key}_rows.tex").write_text("\n".join(lines) + "\n")
        lines = [f"{r['pass']:.0f} & {r.X:.0f} & {r.N:.0f} & {r.Vc:.1f} & {r.Vf_pred:.1f} & "
                 f"{r.Pc_pred:.0f} & {r.t0_pred - r.t_feed:.2f} & {r.t1_pred - r.t_feed:.2f} & "
                 f"{r.t2_pred - r.t_feed:.2f} \\\\"
                 for _, r in res.iterrows()]
        (TABLE_DIR / f"prediction_{key}_rows.tex").write_text("\n".join(lines) + "\n")
        lines = [f"{r['pass']:.0f} & {r.X:.0f} & {r.P_total_min:.0f} & {r.P_total_max:.0f} & "
                 f"{r.P_air_min:.0f} & {r.P_air_max:.0f} & {r.Pc_min:.0f} & {r.Pc_max:.0f} & "
                 f"{r.Kc_min:.0f} & {r.Kc:.0f} & {r.Kc_max:.0f} \\\\"
                 for _, r in res.iterrows()]
        (TABLE_DIR / f"range_{key}_rows.tex").write_text("\n".join(lines) + "\n")
        res.to_csv(TABLE_DIR / f"turning_{key}_results.csv", index=False)

    # reference for the drilling prediction: smallest diameter, constant spindle speed
    ref = outs["n1200"]["res"].iloc[-1]
    pd.DataFrame([{"X": ref.X, "N": ref.N, "Vc": ref.Vc, "Kc": ref.Kc}]).to_csv(
        TABLE_DIR / "kc_reference.csv", index=False)

    r1, r2 = outs["vc150"]["res"], outs["n1200"]["res"]
    allkc = pd.concat([r1.Kc, r2.Kc])
    ref = r2.iloc[-1]
    dev = pd.concat([100 * (r1.Kc - r1.Kc_chapter) / r1.Kc_chapter,
                     100 * (r2.Kc - r2.Kc_chapter) / r2.Kc_chapter]).abs()
    macros = {
        "KcChapter": f"{KC_CHAPTER:.1f}",
        "KcRef": f"{ref.Kc:.0f}", "VcRef": f"{ref.Vc:.1f}", "PcRef": f"{ref.Pc_meas:.0f}",
        "FcRef": f"{ref.Fc:.0f}", "KcDevMax": f"{dev.max():.1f}",
        "PcErrOneMin": f"{r1.Pc_err.min():.1f}", "PcErrOneMax": f"{r1.Pc_err.max():.1f}",
        "PcErrTwoMin": f"{r2.Pc_err.min():.1f}", "PcErrTwoMax": f"{r2.Pc_err.max():.1f}",
        "PcTwoPredFirst": f"{r2.Pc_pred.iloc[0]:.0f}",
        "PcTwoPredLast": f"{r2.Pc_pred.iloc[-1]:.0f}",
        "KcOneMean": f"{r1.Kc.mean():.0f}", "KcOneStd": f"{r1.Kc.std():.0f}",
        "KcTwoMean": f"{r2.Kc.mean():.0f}", "KcTwoStd": f"{r2.Kc.std():.0f}",
        "KcTwoMin": f"{r2.Kc.min():.0f}", "KcTwoMax": f"{r2.Kc.max():.0f}",
        "KcAllMean": f"{allkc.mean():.0f}",
        "KcRefMin": f"{ref.Kc_min:.0f}", "KcRefMax": f"{ref.Kc_max:.0f}",
        "KcRangeMin": f"{pd.concat([r1.Kc_min, r1.Kc_max, r2.Kc_min, r2.Kc_max]).min():.0f}",
        "KcRangeMax": f"{pd.concat([r1.Kc_min, r1.Kc_max, r2.Kc_min, r2.Kc_max]).max():.0f}",
        "BandOne": f"{(r1.Kc_max - r1.Kc_min).abs().max():.0f}",
        "BandTwo": f"{(r2.Kc_max - r2.Kc_min).abs().max():.0f}",
        "DevMinMax": f"{100 * pd.concat([(r.Kc_min - r.Kc).abs() / r.Kc for r in (r1, r2)] + [(r.Kc_max - r.Kc).abs() / r.Kc for r in (r1, r2)]).max():.1f}",
        "KcCrossLo": f"{ref.Kc_cross_lo:.0f}", "KcCrossHi": f"{ref.Kc_cross_hi:.0f}",
        "SwingCut": f"{pd.concat([r1.P_total_max - r1.P_total_min, r2.P_total_max - r2.P_total_min]).mean():.0f}",
        "SwingAir": f"{pd.concat([r1.P_air_max - r1.P_air_min, r2.P_air_max - r2.P_air_min]).mean():.0f}",
        "PcOneMean": f"{r1.Pc_meas.mean():.0f}", "PcOnePred": f"{r1.Pc_pred.mean():.0f}",
        "PcTwoFirst": f"{r2.Pc_meas.iloc[0]:.0f}", "PcTwoLast": f"{r2.Pc_meas.iloc[-1]:.0f}",
        "PzeroOneFirst": f"{r1.P_air.iloc[0]:.0f}", "PzeroOneLast": f"{r1.P_air.iloc[-1]:.0f}",
        "PzeroTwoFirst": f"{r2.P_air.iloc[0]:.0f}", "PzeroTwoLast": f"{r2.P_air.iloc[-1]:.0f}",
        "PzeroFitTwelve": f"{idle_fit_1200:.0f}",
        "IdleA": f"{idle_coef[2]:.0f}", "IdleB": f"{idle_coef[1]:.3f}",
        "IdleC": f"{idle_coef[0] * 1e4:.2f}",
        "TcutOne": f"{r1.Tcut_meas.sum():.1f}", "TcutOnePred": f"{r1.Tcut_pred.sum():.1f}",
        "TcutTwo": f"{r2.Tcut_meas.sum():.1f}", "TcutTwoPred": f"{r2.Tcut_pred.sum():.1f}",
        "ToneTot": f"{outs['vc150']['cut']['time'].iloc[-1]:.1f}",
        "TtwoTot": f"{outs['n1200']['cut']['time'].iloc[-1]:.1f}",
        "EOne": f"{r1.E_tot_kJ.sum():.0f}", "ETwo": f"{r2.E_tot_kJ.sum():.0f}",
        "ECutOne": f"{r1.E_cut_kJ.sum():.0f}", "ECutTwo": f"{r2.E_cut_kJ.sum():.0f}",
        "RiseMean": f"{pd.concat([r1.rise, r2.rise]).mean():.2f}",
        "TonDelayOne": f"{(r1.t_on - r1.t0_pred).mean():.2f}",
        "TonDelayTwo": f"{(r2.t_on - r2.t0_pred).mean():.2f}",
        "SECcutOne": f"{r1.SEC_cut.mean():.2f}", "SECcutTwo": f"{r2.SEC_cut.mean():.2f}",
    }
    tex = "\n".join(rf"\newcommand{{\{k}}}{{{v}}}" for k, v in macros.items())
    (TABLE_DIR / "turning_macros.tex").write_text(tex + "\n")


def main() -> None:
    apply_style()
    PLOT_DIR.mkdir(exist_ok=True)
    outs = {key: analyse_program(key) for key in PROGRAMS}
    print(f"Kc used for the prediction (chapter) = {KC_CHAPTER} MPa")
    for key, out in outs.items():
        print(f"\n=== {out['prog'].label}")
        cols = ["pass", "X", "N", "Vc", "P_total", "P_air", "Pc_meas", "Pc_pred", "Pc_err",
                "Kc", "Kc_chapter", "Tcut_pred", "Tcut_meas", "rise"]
        print(out["res"][cols].round(2).to_string(index=False))
        plot_raw_power(out)
        plot_prediction_vs_measure(out)
    plot_program_path(outs)
    plot_zoom_pass(outs)
    plot_entry_exit(outs)
    plot_steady_state(outs)
    plot_kc(outs)
    plot_kc_range(outs)
    coef, fit1200 = plot_idle_power(outs)
    plot_timing(outs)
    print("ripple frequency (Hz):", plot_ripple(outs))
    write_tables(outs, coef, fit1200)


if __name__ == "__main__":
    main()
