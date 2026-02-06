#!/usr/bin/env python3
"""
String design calculator (first principles or D'Addario UW) — plain text output
- Modes:
    * uw   : D'Addario unit-weight method (official tension formula)
    * geom : geometric first-principles for round-wound strings
- Computes: linear density (μ), tension (T), frequency/length conversions, wave speed, inharmonicity B,
            and stiff partials for the drone string.
- Output: CSV (default) or TSV (--sep tsv). No formatting besides plain text.

References implemented:
- UW tension formula (imperial): T_lbf = (UW * (2 * L_in * F)^2) / 386.4  → convert to SI.
- Ideal string: f = (1/(2L)) * sqrt(T/μ),  T = 4 L^2 f^2 μ
- Round-wound μ: μ = ρ_core * A_core + ρ_wrap * A_wrap * λ, with λ = sqrt(1 + (2π r_center / p)^2)
- Inharmonicity (pinned ends): B = (π^3 * E * r_core^4) / (4 * T * L^2), f_n ≈ n f1 * sqrt(1 + B n^2)
"""

from dataclasses import dataclass
from math import pi, sqrt
import argparse
import sys

# -----------------------------
# Note utilities (A4 = 440 Hz)
# -----------------------------

NOTE_ORDER = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def note_to_frequency(note: str, A4: float = 440.0) -> float:
    s = note.strip().upper()
    # flats -> sharps (simple mapping)
    s = (s.replace('B#', 'C')
           .replace('E#', 'F')
           .replace('DB', 'C#')
           .replace('EB', 'D#')
           .replace('GB', 'F#')
           .replace('AB', 'G#')
           .replace('BB', 'A#'))
    if len(s) < 2:
        raise ValueError("Bad note: {}".format(note))
    name, octave_str = s[:-1], s[-1]
    if name not in NOTE_ORDER:
        raise ValueError("Unrecognized note: {}".format(note))
    octave = int(octave_str)
    semitone_index = NOTE_ORDER.index(name)
    midi = (octave + 1) * 12 + semitone_index
    return A4 * (2 ** ((midi - 69) / 12.0))


# -----------------------------
# Unit helpers
# -----------------------------

def mm_to_m(x_mm: float) -> float: return x_mm / 1000.0
def m_to_mm(x_m: float) -> float:  return x_m * 1000.0
def cm_to_m(x_cm: float) -> float: return x_cm / 100.0
def m_to_cm(x_m: float) -> float:  return x_m * 100.0
def in_to_m(x_in: float) -> float: return x_in * 0.0254
def m_to_in(x_m: float) -> float:  return x_m / 0.0254

# -----------------------------
# Materials and specs
# -----------------------------

@dataclass
class Material:
    name: str
    density: float     # kg/m^3
    youngs_E: float    # Pa

# Defaults
STEEL_MUSIC_WIRE = Material("High-carbon steel (music wire)", density=7850.0, youngs_E=2.00e11)
BRONZE_8020     = Material("80/20 bronze wrap", density=8400.0, youngs_E=1.10e11)

@dataclass
class RoundWoundString:
    overall_d: float   # meters
    core_d: float      # meters
    wrap_d: float      # meters
    core_mat: Material
    wrap_mat: Material
    pitch: float       # axial advance per wrap (m)

    def center_radius(self) -> float:
        return self.core_d / 2.0 + self.wrap_d / 2.0

# -----------------------------
# Physics
# -----------------------------

def linear_density_roundwound(spec: RoundWoundString) -> float:
    """
    μ = ρ_core * A_core + ρ_wrap * A_wrap * λ,
    λ = sqrt(1 + (2π r_center / p)^2)
    """
    r_core = spec.core_d / 2.0
    A_core = pi * r_core * r_core
    r_wrap = spec.wrap_d / 2.0
    A_wrap = pi * r_wrap * r_wrap
    r_center = spec.center_radius()
    p = spec.pitch if spec.pitch > 0.0 else 1e-9
    helix_factor = sqrt(1.0 + (2.0 * pi * r_center / p) ** 2)
    mu = spec.core_mat.density * A_core + spec.wrap_mat.density * A_wrap * helix_factor
    return mu  # kg/m

def tension_for(f: float, L: float, mu: float) -> float:
    return 4.0 * (L ** 2) * (f ** 2) * mu  # N

def frequency_for(T: float, L: float, mu: float) -> float:
    return (1.0 / (2.0 * L)) * sqrt(T / mu)  # Hz

def length_for(f: float, T: float, mu: float) -> float:
    return (1.0 / (2.0 * f)) * sqrt(T / mu)  # m

def wave_speed(T: float, mu: float) -> float:
    return sqrt(T / mu)  # m/s

def inharmonicity_B_pinned(core_radius: float, core_E: float, T: float, L: float) -> float:
    """
    Russell's pinned-end form:
    B = (π^3 * E * r_core^4) / (4 * T * L^2)
    """
    return (pi ** 3) * core_E * (core_radius ** 4) / (4.0 * T * (L ** 2))

def stiff_partial(n: int, f1: float, B: float) -> float:
    return n * f1 * sqrt(1.0 + B * (n ** 2))

# -----------------------------
# D'Addario UW method (imperial base)
# -----------------------------

def tension_from_UW(UW_lbf_per_in: float, L_m: float, f_hz: float) -> float:
    """
    Tension from D'Addario's official formula:
      T_lbf = (UW * (2 * L_in * F)^2) / 386.4
    Convert to Newtons.
    """
    L_in = m_to_in(L_m)
    T_lbf = (UW_lbf_per_in * (2.0 * L_in * f_hz) ** 2) / 386.4
    T_N = T_lbf * 4.44822
    return T_N

def effective_mu_from_TLF(T_N: float, L_m: float, f_hz: float) -> float:
    """
    From ideal relation T = 4 L^2 f^2 μ → μ = T / (4 L^2 f^2)
    This 'μ' is the effective linear density consistent with the UW-based T.
    """
    denom = 4.0 * (L_m ** 2) * (f_hz ** 2)
    return T_N / denom

# -----------------------------
# Output helpers (plain text)
# -----------------------------

def fmt(x: float, places: int = 3) -> str:
    # No f-strings: build a dynamic format string safely
    return ("{0:." + str(places) + "f}").format(x)

def print_rows_plain(rows: list[dict[str, str]], sep: str) -> None:
    headers = [
        "Label","Note","Freq_Hz","Length_cm",
        "Mode","Overall_d_mm","Core_d_mm","Wrap_d_mm","Pitch_mm",
        "mu_kg_per_m","Tension_N","Tension_kgf","WaveSpeed_m_per_s","B"
    ]
    print(sep.join(headers))
    for r in rows:
        print(sep.join(r.get(h, "") for h in headers))

# -----------------------------
# Build your instrument rows
# -----------------------------

def build_rows(mode: str,
               a4: float,
               # melody (PB060)
               melody_overall_mm: float,
               melody_core_ratio: float,
               melody_pitch_ratio: float,
               # drone (PB075)
               drone_overall_mm: float,
               drone_core_ratio: float,
               drone_pitch_ratio: float,
               # melody notes & lengths (m), drone note & length (m)
               melody_spec: list[tuple[str, float]],
               drone_note: str,
               drone_L_m: float,
               # UW (if mode=uw)
               uw_melody: float,
               uw_drone: float,
               partials: int,
               sep: str) -> None:

    rows: list[dict[str, str]] = []

    # Build geometric wound specs (used for dimensions and for GEOM mode)
    def wound_from_overall(overall_d_m: float, core_ratio: float, pitch_ratio: float) -> RoundWoundString:
        core_d = core_ratio * overall_d_m
        wrap_d = (overall_d_m - core_d) / 2.0
        if wrap_d <= 0.0:
            wrap_d = 1e-6
        pitch = wrap_d * pitch_ratio
        if pitch <= 0.0:
            pitch = 1e-9
        return RoundWoundString(
            overall_d=overall_d_m,
            core_d=core_d,
            wrap_d=wrap_d,
            core_mat=STEEL_MUSIC_WIRE,
            wrap_mat=BRONZE_8020,
            pitch=pitch
        )

    overall_melody_m = mm_to_m(melody_overall_mm)
    overall_drone_m  = mm_to_m(drone_overall_mm)

    geom_melody = wound_from_overall(overall_melody_m, melody_core_ratio, melody_pitch_ratio)
    geom_drone  = wound_from_overall(overall_drone_m,  drone_core_ratio,  drone_pitch_ratio)

    # Helper to compute one line
    def compute_line(label: str, note: str, L_m: float, role: str) -> dict[str, str]:
        f = note_to_frequency(note, A4=a4)

        if role == "drone":
            UW = uw_drone
            spec = geom_drone
        else:
            UW = uw_melody
            spec = geom_melody

        if mode == "uw":
            T = tension_from_UW(UW, L_m, f)
            mu_eff = effective_mu_from_TLF(T, L_m, f)
            c = wave_speed(T, mu_eff)
            core_r = spec.core_d / 2.0
            B = inharmonicity_B_pinned(core_r, spec.core_mat.youngs_E, T, L_m)
            return {
                "Label": label,
                "Note": note,
                "Freq_Hz": fmt(f, 3),
                "Length_cm": fmt(m_to_cm(L_m), 3),
                "Mode": "UW",
                "Overall_d_mm": fmt(m_to_mm(spec.overall_d), 3),
                "Core_d_mm": fmt(m_to_mm(spec.core_d), 3),
                "Wrap_d_mm": fmt(m_to_mm(spec.wrap_d), 3),
                "Pitch_mm": fmt(m_to_mm(spec.pitch), 3),
                "mu_kg_per_m": fmt(mu_eff, 6),
                "Tension_N": fmt(T, 3),
                "Tension_kgf": fmt(T / 9.80665, 3),
                "WaveSpeed_m_per_s": fmt(c, 3),
                "B": fmt(B, 6)
            }

        elif mode == "geom":
            mu = linear_density_roundwound(spec)
            T = tension_for(f, L_m, mu)
            c = wave_speed(T, mu)
            core_r = spec.core_d / 2.0
            B = inharmonicity_B_pinned(core_r, spec.core_mat.youngs_E, T, L_m)
            return {
                "Label": label,
                "Note": note,
                "Freq_Hz": fmt(f, 3),
                "Length_cm": fmt(m_to_cm(L_m), 3),
                "Mode": "GEOM",
                "Overall_d_mm": fmt(m_to_mm(spec.overall_d), 3),
                "Core_d_mm": fmt(m_to_mm(spec.core_d), 3),
                "Wrap_d_mm": fmt(m_to_mm(spec.wrap_d), 3),
                "Pitch_mm": fmt(m_to_mm(spec.pitch), 3),
                "mu_kg_per_m": fmt(mu, 6),
                "Tension_N": fmt(T, 3),
                "Tension_kgf": fmt(T / 9.80665, 3),
                "WaveSpeed_m_per_s": fmt(c, 3),
                "B": fmt(B, 6)
            }

        else:
            raise ValueError("mode must be 'uw' or 'geom'")

    # Drone first
    rows.append(compute_line("Drone", drone_note, drone_L_m, "drone"))
    # Melody strings
    for i, (note, Lm) in enumerate(melody_spec, start=1):
        rows.append(compute_line(str(i), note, Lm, "melody"))

    # Print table
    print_rows_plain(rows, sep=sep)

    # Optional: drone partials
    if partials > 0:
        f1 = note_to_frequency(drone_note, A4=a4)
        if mode == "uw":
            T = tension_from_UW(uw_drone, drone_L_m, f1)
            mu_eff = effective_mu_from_TLF(T, drone_L_m, f1)
            core_r = geom_drone.core_d / 2.0
            B = inharmonicity_B_pinned(core_r, geom_drone.core_mat.youngs_E, T, drone_L_m)
        else:
            mu = linear_density_roundwound(geom_drone)
            T = tension_for(f1, drone_L_m, mu)
            core_r = geom_drone.core_d / 2.0
            B = inharmonicity_B_pinned(core_r, geom_drone.core_mat.youngs_E, T, drone_L_m)

        # Header
        print(sep.join(["Partials_Label", "n", "f_n_Hz"]))
        for n in range(1, partials + 1):
            fn = stiff_partial(n, f1, B)
            print(sep.join(["DronePartials", str(n), fmt(fn, 3)]))


# -----------------------------
# CLI
# -----------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Vibrating string calculator — UW or GEOM mode — plain text output"
    )
    p.add_argument("--mode", choices=["uw", "geom"], default="uw",
                   help="Computation mode: 'uw' (D'Addario unit weight) or 'geom' (first-principles geometry). Default: uw")
    p.add_argument("--sep", choices=["csv", "tsv"], default="csv",
                   help="Plain text separator: csv (comma) or tsv (tab). Default: csv")
    p.add_argument("--a4", type=float, default=440.0, help="A4 reference frequency (Hz). Default 440.0")

    # Default instrument (your setup)
    p.add_argument("--drone-note", default="D1", help="Drone note (e.g., D1).")
    p.add_argument("--drone-length", type=float, default=1.000, help="Drone speaking length in meters. Default 1.000")
    p.add_argument("--melody-lengths", default="D2:0.651,E2:0.580,F2:0.548,G2:0.488,A2:0.435",
                   help="Comma-separated note:length_m list for melody. Example D2:0.651,E2:0.580,...")

    # Geometry params (used in both modes for dimensions, B, etc.)
    p.add_argument("--melody-overall-mm", type=float, default=1.524, help="Melody overall diameter (mm), default PB060=1.524")
    p.add_argument("--melody-core-ratio", type=float, default=0.55, help="Melody core_d / overall_d, default 0.55")
    p.add_argument("--melody-pitch-ratio", type=float, default=1.0, help="Melody pitch / wrap_d, default 1.0 (adjacent turns)")

    p.add_argument("--drone-overall-mm", type=float, default=1.905, help="Drone overall diameter (mm), default PB075=1.905")
    p.add_argument("--drone-core-ratio", type=float, default=0.55, help="Drone core_d / overall_d, default 0.55")
    p.add_argument("--drone-pitch-ratio", type=float, default=1.0, help="Drone pitch / wrap_d, default 1.0 (adjacent turns)")

    # UW values (only used when mode=uw)
    p.add_argument("--uw-melody", type=float, default=0.000545, help="Melody UW (lb/in) for PB060. Default 0.000545")
    p.add_argument("--uw-drone",  type=float, default=0.000850, help="Drone  UW (lb/in) for PB075. Default 0.000850")

    # Drone partials
    p.add_argument("--partials", type=int, default=0, help="If >0, also print that many drone partials with inharmonicity.")

    args = p.parse_args(argv)
    sep = "," if args.sep == "csv" else "\t"

    # Parse melody spec
    melody_pairs: list[tuple[str, float]] = []
    for chunk in args.melody_lengths.split(","):
        if not chunk.strip():
            continue
        parts = chunk.split(":")
        if len(parts) != 2:
            raise ValueError("Bad melody-lengths chunk: '{}'".format(chunk))
        name = parts[0].strip()
        Lval = float(parts[1].strip())
        melody_pairs.append((name, Lval))

    build_rows(
        mode=args.mode,
        a4=args.a4,
        melody_overall_mm=args.melody_overall_mm,
        melody_core_ratio=args.melody_core_ratio,
        melody_pitch_ratio=args.melody_pitch_ratio,
        drone_overall_mm=args.drone_overall_mm,
        drone_core_ratio=args.drone_core_ratio,
        drone_pitch_ratio=args.drone_pitch_ratio,
        melody_spec=melody_pairs,
        drone_note=args.drone_note,
        drone_L_m=args.drone_length,
        uw_melody=args.uw_melody,
        uw_drone=args.uw_drone,
        partials=args.partials,
        sep=sep
    )
    return 0

if __name__ == "__main__":
    sys.exit(main())