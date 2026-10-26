"""
Shared models and data-processing tools for Assignment 1
(Multi-Agent System for Smart Machining - INSA Toulouse / ECIU).

Units used everywhere (unless stated otherwise):
    lengths  [mm], diameters [mm], feed f [mm/rev], depth of cut ap [mm]
    cutting speed Vc [m/min], spindle speed N [rev/min], feed velocity Vf [mm/min]
    specific cutting pressure Kc [MPa = N/mm^2 = J/cm^3]  (1 J/mm^3 = 1000 MPa)
    force [N], power [W], torque [N.m], time [s]

With these units the mechanistic power model of the course reads
    Pc [W] = Kc [N/mm^2] * f [mm] * ap [mm] * Vc [m/min] / 60
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
PLOT_DIR = ROOT / "plot"
TABLE_DIR = ROOT / "report" / "generated"


# ---------------------------------------------------------------------------
# Kinematics
# ---------------------------------------------------------------------------
def spindle_speed(vc: float | np.ndarray, d: float | np.ndarray) -> np.ndarray:
    """N [rpm] that gives cutting speed vc [m/min] on diameter d [mm]."""
    return 1000.0 * np.asarray(vc) / (np.pi * np.asarray(d))


def cutting_speed(n: float | np.ndarray, d: float | np.ndarray) -> np.ndarray:
    """Vc [m/min] on diameter d [mm] at spindle speed n [rpm]."""
    return np.pi * np.asarray(d) * np.asarray(n) / 1000.0


# ---------------------------------------------------------------------------
# Turning model
# ---------------------------------------------------------------------------
@dataclass
class TurningProgram:
    """Longitudinal turning program of the chapter (one pass per diameter).
    As in the chapter, the cutting speed is taken on the programmed diameter X
    (tool point P): Vc = pi X N / 1000."""
    name: str
    label: str
    ap: float                  # radial depth of cut per pass [mm]
    f: float                   # feed per revolution [mm/rev]
    diameters: np.ndarray      # programmed (finished) diameter X of each pass [mm]
    vc: float | None = None    # constant cutting speed (G96) [m/min]
    n: float | None = None     # constant spindle speed (G97) [rpm]
    z_start: float = 1.0       # programmed start of the feed motion [mm]
    z_contact: float = 0.0     # face of the workpiece [mm]
    z_end: float = -40.0       # programmed end of the feed motion [mm]

    def spindle_speed(self) -> np.ndarray:
        if self.vc is not None:   # G96: N computed on the tool-tip diameter X
            return spindle_speed(self.vc, self.diameters)
        return np.full_like(self.diameters, self.n, dtype=float)

    def predict(self, kc: float, nose_radius: float = 0.8) -> pd.DataFrame:
        """Mechanistic prediction (course model Pc = Kc f ap Vc) for each pass."""
        n = self.spindle_speed()
        vf = self.f * n                                  # mm/min
        vc = cutting_speed(n, self.diameters)            # m/min at X
        fc = kc * self.f * self.ap                       # N
        pc = fc * vc / 60.0                              # W
        t_air = (self.z_start - self.z_contact) / vf * 60.0
        t_eng = nose_radius / vf * 60.0                  # nose radius enters the bar
        t_cut = (self.z_contact - self.z_end) / vf * 60.0
        return pd.DataFrame({
            "pass": np.arange(1, len(n) + 1),
            "X": self.diameters,
            "N": n,
            "Vf": vf,
            "Vc": vc,
            "Fc_pred": fc,
            "Pc_pred": pc,
            "MRR": self.f * self.ap * vc,                # cm^3/min (chapter Eq. 7)
            "dt_t0": t_air,           # feed start -> t0 (tool touches the bar)
            "dt_t1": t_air + t_eng,   # feed start -> t1 (edge fully engaged)
            "dt_t2": t_air + t_cut,   # feed start -> t2 (end of the machined length)
        })


# ---------------------------------------------------------------------------
# Drilling model (two-flute twist drill, chisel edge neglected)
# ---------------------------------------------------------------------------
@dataclass
class DrillingOperation:
    """Axial drilling, optionally inside a pre-drilled (pilot) hole.

    Each lip is split into radial elements. An element at radius r cuts when its
    axial position z_tip - r cot(point/2) is inside the material, i.e. below the
    part surface (z > 0) and not in the pilot hole (r < d_pilot/2 and above the
    conical bottom of the pilot hole). Per element and per lip (course slides):
        dFc = Kc fz dr,  dFf = Kf fz dr,  dPc = dFc Vc(r),  Vc(r) = 2 pi r N / 1000
    z is the depth of the drill tip below the part surface [mm]."""
    name: str
    d: float                    # drill diameter [mm]
    point_angle: float          # [deg]
    vc: float                   # peripheral cutting speed [m/min]
    f: float                    # feed per revolution [mm/rev]
    kc: float                   # specific cutting pressure [MPa]
    kf_over_kc: float           # feed (thrust) to cutting pressure ratio
    length: float               # machining length = final tip depth [mm]
    d_pilot: float = 0.0        # diameter of the pre-drilled hole [mm]
    pilot_length: float = 0.0   # tip depth of the pre-drilled hole [mm]
    approach: float = 1.0       # air feed before the tip reaches the surface [mm]
    n_elements: int = 2000

    @property
    def n(self) -> float:
        return float(spindle_speed(self.vc, self.d))

    @property
    def vf(self) -> float:
        return self.f * self.n                                  # mm/min

    @property
    def cot(self) -> float:
        return 1.0 / np.tan(np.radians(self.point_angle / 2.0))

    def time_at(self, z_tip: float) -> float:
        """Time [s] (origin: start of the feed, tip `approach` mm above the part)."""
        return (z_tip + self.approach) / (self.vf / 60.0)

    def events(self) -> dict[str, tuple[float, float, str]]:
        """Characteristic instants: label -> (tip depth, time, description)."""
        r_in, r_out = self.d_pilot / 2.0, self.d / 2.0
        ev = {"t_0": (r_in * self.cot, "lips touch the part"),
              "t_1": (r_out * self.cot, "whole lip length engaged")}
        if self.d_pilot > 0:
            ev["t_1"] = (r_out * self.cot, "annulus fully engaged (enlarging)")
            ev["t_2"] = (self.pilot_length, "tip reaches the pilot-hole bottom")
            ev["t_3"] = (self.length, "end of machining length")
        else:
            ev["t_0"] = (0.0, "tip touches the part")
            ev["t_2"] = (self.length, "end of machining length")
        return {k: (z, self.time_at(z), txt) for k, (z, txt) in ev.items()}

    def engaged(self, z_tip: float, r: np.ndarray) -> np.ndarray:
        z_el = z_tip - r * self.cot
        in_part = z_el > 1e-9
        in_pilot = (r < self.d_pilot / 2.0) & \
            (z_el <= self.pilot_length - r * self.cot + 1e-9)
        return in_part & ~in_pilot

    def predict(self, dt: float = 5e-4, retract: float = 0.15) -> pd.DataFrame:
        dr = self.d / 2.0 / self.n_elements
        r = (np.arange(self.n_elements) + 0.5) * dr
        t_end = self.time_at(self.length)
        t = np.arange(0.0, t_end + retract, dt)
        z_tip = -self.approach + self.vf / 60.0 * t
        fz = self.f / 2.0
        kf = self.kf_over_kc * self.kc
        vc_r = 2 * np.pi * r * self.n / 1000.0                 # m/min
        pc, ff, r_lo, r_hi = (np.zeros_like(t) for _ in range(4))
        for i, z in enumerate(z_tip):
            if t[i] > t_end:          # drill retracted from the hole bottom
                continue
            m = self.engaged(z, r)
            if not m.any():
                continue
            pc[i] = 2 * self.kc * fz * dr * np.sum(vc_r[m]) / 60.0   # W
            ff[i] = 2 * kf * fz * dr * m.sum()                        # N
            r_lo[i], r_hi[i] = r[m].min() - dr / 2, r[m].max() + dr / 2
        omega = 2 * np.pi * self.n / 60.0
        return pd.DataFrame({"t": t, "z_tip": z_tip, "Pc": pc, "Ff": ff,
                             "Mc": pc / omega, "r_in": r_lo, "r_out": r_hi,
                             "Pf": ff * self.vf / 60.0 / 1000.0})     # W


def kc_turning(pc: float | np.ndarray, f: float, ap: float,
               vc: float | np.ndarray) -> np.ndarray:
    """Specific cutting pressure in turning: Kc = Pc / (f ap Vc)."""
    return np.asarray(pc) / (f * ap * np.asarray(vc) / 60.0)


# ---------------------------------------------------------------------------
# Siemens Sinumerik trace files (CNC "Trace" function)
# ---------------------------------------------------------------------------
TRACE_COLUMNS = ["time", "power", "prog_pos_z", "prog_pos_x",
                 "meas_pos_z", "meas_pos_x", "spindle1", "spindle2"]


def load_trace(path: Path | str) -> pd.DataFrame:
    """Read a trace CSV (10 header lines) whatever the number of channels."""
    df = pd.read_csv(path, skiprows=10, header=None)
    df = df.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all").dropna()
    df.columns = TRACE_COLUMNS[: df.shape[1]]
    return df.reset_index(drop=True)


def moving_average(y: np.ndarray, window: int) -> np.ndarray:
    """Centred moving average (window in samples)."""
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="same")


def find_feed_passes(df: pd.DataFrame, min_duration: float = 3.0) -> pd.DataFrame:
    """Locate the longitudinal feed motions (Z decreasing at feed rate) from the
    programmed position signal. Returns one row per pass with start/end index."""
    t = df["time"].to_numpy()
    z = df["prog_pos_z"].to_numpy()
    vz = np.gradient(z, t) * 60.0                         # mm/min
    feeding = (vz < -30.0) & (vz > -1000.0)                # feed, not rapid
    edges = np.flatnonzero(np.diff(feeding.astype(int))) + 1
    bounds = np.r_[0, edges, len(feeding)]
    rows = []
    for s, e in zip(bounds[:-1], bounds[1:]):
        if feeding[s] and t[e - 1] - t[s] >= min_duration * 0.1:
            rows.append((s, e))
    # merge pieces of the same pass (e.g. feed change while the spindle accelerates)
    merged: list[list[int]] = []
    for s, e in rows:
        if merged and t[s] - t[merged[-1][1] - 1] < 0.2:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    out = []
    for s, e in merged:
        if t[e - 1] - t[s] < min_duration:
            continue
        seg = df.iloc[s:e]
        # feed velocity from a linear fit of z(t) on the steady part
        steady = seg[seg["prog_pos_z"] < seg["prog_pos_z"].iloc[0] - 5]
        slope = np.polyfit(steady["time"], steady["prog_pos_z"], 1)[0]
        out.append({"i_start": s, "i_end": e,
                    "t_start": t[s], "t_end": t[e - 1],
                    "z_start": z[s], "z_end": z[e - 1],
                    "X": float(np.round(seg["prog_pos_x"].median(), 3)),
                    "Vf_meas": -slope * 60.0})
    passes = pd.DataFrame(out)
    passes.insert(0, "pass", np.arange(1, len(passes) + 1))
    return passes


def time_at_z(df: pd.DataFrame, i_start: int, i_end: int, z_target: float) -> float:
    """Time at which the programmed Z crosses z_target inside a pass."""
    seg = df.iloc[int(i_start):int(i_end)]
    z = seg["prog_pos_z"].to_numpy()[::-1]
    t = seg["time"].to_numpy()[::-1]
    return float(np.interp(z_target, z, t))


def steady_state_mean(df: pd.DataFrame, i_start: int, i_end: int,
                      z_hi: float, z_lo: float) -> tuple[float, float]:
    """Mean and standard deviation of the power while z_lo <= Z <= z_hi."""
    seg = df.iloc[int(i_start):int(i_end)]
    m = (seg["prog_pos_z"] <= z_hi) & (seg["prog_pos_z"] >= z_lo)
    p = seg.loc[m, "power"]
    return float(p.mean()), float(p.std())


def steady_state_range(df: pd.DataFrame, i_start: int, i_end: int,
                       z_hi: float, z_lo: float, q: float = 1.0) -> tuple[float, float]:
    """Minimum and maximum of the power while z_lo <= Z <= z_hi, taken as the
    q-th and (100-q)-th percentiles so that a single outlier sample is ignored."""
    seg = df.iloc[int(i_start):int(i_end)]
    m = (seg["prog_pos_z"] <= z_hi) & (seg["prog_pos_z"] >= z_lo)
    p = seg.loc[m, "power"].to_numpy()
    return float(np.percentile(p, q)), float(np.percentile(p, 100 - q))
