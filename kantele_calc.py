#!/usr/bin/env python3
"""
Kantele / vibrating-string design calculator — layout-first (length generation)

Key change vs earlier versions:
- Gauge MUST be explicit on the command line.
  - In --mode uw: you must pass --uw-melody and --uw-drone.
  - In --mode geom: you must pass the geometry parameters that define μ.
- Optional: you can tag gauges with human-readable names (e.g. PB060 / PB075),
  and those tags are included in every output row to make iterative runs auditable.

Design intent:
- Melody strings: SAME gauge/material -> compute ALL melody speaking lengths from ONE anchor (note:length).
- Drone string: can be DIFFERENT gauge -> either:
    (a) compute its tension from a chosen physical length, or
    (b) compute its required length from a chosen target tension, or
    (c) match drone tension to the melody tension implied by the melody anchor.

End correction:
- Physics uses L_eff; CAD uses L_phys.
  L_eff = L_phys - end_correction

Output:
- CSV (default) or TSV (--sep tsv). Plain text.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi, sqrt
import argparse
import re
import sys

# -----------------------------
# Note utilities (A4 = 440 Hz)
# -----------------------------

NOTE_ORDER = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

_FLAT_TO_SHARP = {
    'B#': 'C',
    'E#': 'F',
    'DB': 'C#',
    'EB': 'D#',
    'GB': 'F#',
    'AB': 'G#',
    'BB': 'A#',
    'CB': 'B',
    'FB': 'E',
}

_NOTE_RE = re.compile(r'^([A-G])([#B]?)(-?\d+)$')

def note_to_frequency(note: str, A4: float = 440.0) -> float:
    s = note.strip().upper().replace('♭', 'B').replace('♯', '#')
    m = _NOTE_RE.match(s)
    if not m:
        raise ValueError(f"Bad note format: '{note}' (expected like D1, C#4, Db3, etc.)")
    letter, accidental, octave_str = m.group(1), m.group(2), m.group(3)
    name = f"{letter}{accidental}"
    name = _FLAT_TO_SHARP.get(name, name)
    if name not in NOTE_ORDER:
        raise ValueError(f"Unrecognized note name: '{note}' -> '{name}'")
    octave = int(octave_str)
    semitone_index = NOTE_ORDER.index(name)
    midi = (octave + 1) * 12 + semitone_index
    return A4 * (2.0 ** ((midi - 69) / 12.0))


# -----------------------------
# Unit helpers
# -----------------------------

def mm_to_m(x_mm: float) -> float: return x_mm / 1000.0
def m_to_mm(x_m: float) -> float:  return x_m * 1000.0
def m_to_cm(x_m: float) -> float:  return x_m * 100.0
def m_to_in(x_m: float) -> float:  return x_m / 0.0254

# -----------------------------
# Materials and specs
# -----------------------------

@dataclass
class Material:
    name: str
    density: float     # kg/m^3
    youngs_E: float    # Pa

STEEL_MUSIC_WIRE = Material("High-carbon steel (music wire)", density=7850.0, youngs_E=2.00e11)
BRONZE_8020     = Material("80/20 bronze wrap", density=8400.0, youngs_E=1.10e11)

@dataclass
class RoundWoundString:
    overall_d: float   # meters
    core_d: float      # meters
    wrap_d: float      # meters
    core_mat: Material
    wrap_mat: Material
    pitch: float       # m

    def center_radius(self) -> float:
        return self.core_d / 2.0 + self.wrap_d / 2.0

# -----------------------------
# Physics
# -----------------------------

def linear_density_roundwound(spec: RoundWoundString) -> float:
    """
    μ = ρ_core * A_core + ρ_wrap * A_wrap * λ
    λ = sqrt(1 + (2π r_center / p)^2)
    """
    r_core = spec.core_d / 2.0
    A_core = pi * r_core * r_core
    r_wrap = spec.wrap_d / 2.0
    A_wrap = pi * r_wrap * r_wrap
    r_center = spec.center_radius()
    p = spec.pitch if spec.pitch > 0.0 else 1e-9
    helix_factor = sqrt(1.0 + (2.0 * pi * r_center / p) ** 2)
    return spec.core_mat.density * A_core + spec.wrap_mat.density * A_wrap * helix_factor

def tension_for(f: float, L_eff: float, mu: float) -> float:
    return 4.0 * (L_eff ** 2) * (f ** 2) * mu  # N

def length_for(f: float, T: float, mu: float) -> float:
    return (1.0 / (2.0 * f)) * sqrt(T / mu)  # m (effective)

def wave_speed(T: float, mu: float) -> float:
    return sqrt(T / mu)  # m/s

def inharmonicity_B_pinned(core_radius: float, core_E: float, T: float, L_eff: float) -> float:
    if T <= 0.0 or L_eff <= 0.0:
        return 0.0
    return (pi ** 3) * core_E * (core_radius ** 4) / (4.0 * T * (L_eff ** 2))

def stiff_partial(n: int, f1: float, B: float) -> float:
    return n * f1 * sqrt(1.0 + B * (n ** 2))

# -----------------------------
# UW helpers
# -----------------------------

def mu_from_UW(UW_lbf_per_in: float) -> float:
    """
    UW is weight per length (lbf/in). Convert to μ (kg/m).
    """
    g = 9.80665
    UW_N_per_m = UW_lbf_per_in * 4.4482216152605 / 0.0254
    return UW_N_per_m / g

# -----------------------------
# Output helpers
# -----------------------------

def fmt(x: float, places: int = 3) -> str:
    return ("{0:." + str(places) + "f}").format(x)

def print_rows_plain(rows: list[dict[str, str]], sep: str) -> None:
    headers = [
        "Label","Role","GaugeName",
        "Note","Freq_Hz",
        "LengthEff_cm","LengthPhys_cm","EndCorr_mm",
        "Mode",
        "Overall_d_mm","Core_d_mm","Wrap_d_mm","Pitch_mm",
        "mu_kg_per_m","Tension_N","Tension_kgf","WaveSpeed_m_per_s","B"
    ]
    print(sep.join(headers))
    for r in rows:
        print(sep.join(r.get(h, "") for h in headers))

# -----------------------------
# Spec building
# -----------------------------

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

def compute_mu(mode: str, spec: RoundWoundString, UW: float | None) -> float:
    if mode == "uw":
        if UW is None:
            raise ValueError("In --mode uw you must provide UW explicitly.")
        return mu_from_UW(UW)
    if mode == "geom":
        return linear_density_roundwound(spec)
    raise ValueError("mode must be 'uw' or 'geom'")

# -----------------------------
# Core computation
# -----------------------------

def build_rows_design(
    *,
    mode: str,
    a4: float,
    sep: str,
    melody_notes: list[str],
    melody_anchor_note: str,
    melody_anchor_Lphys_m: float,
    drone_note: str,
    drone_Lphys_m: float | None,
    drone_target_tension_N: float | None,
    drone_match_melody_tension: bool,
    # gauge tags
    melody_gauge_name: str,
    drone_gauge_name: str,
    # specs
    geom_melody: RoundWoundString,
    geom_drone: RoundWoundString,
    uw_melody: float | None,
    uw_drone: float | None,
    # practical
    end_correction_mm: float,
    partials: int,
) -> None:
    rows: list[dict[str, str]] = []
    end_corr_m = mm_to_m(end_correction_mm)

    mu_mel = compute_mu(mode, geom_melody, uw_melody)
    mu_drn = compute_mu(mode, geom_drone,  uw_drone)

    # Melody implied tension from anchor
    f_anchor = note_to_frequency(melody_anchor_note, A4=a4)
    L_anchor_eff = melody_anchor_Lphys_m - end_corr_m
    if L_anchor_eff <= 0.0:
        raise ValueError("Melody anchor effective length <= 0. Increase anchor length or reduce end correction.")
    T_mel = tension_for(f_anchor, L_anchor_eff, mu_mel)

    def make_row(label: str, role: str, gauge_name: str, note: str, L_eff: float, L_phys: float,
                 spec: RoundWoundString, mu: float) -> dict[str, str]:
        f = note_to_frequency(note, A4=a4)
        T = tension_for(f, L_eff, mu)
        c = wave_speed(T, mu) if mu > 0.0 else 0.0
        core_r = spec.core_d / 2.0
        B = inharmonicity_B_pinned(core_r, spec.core_mat.youngs_E, T, L_eff)
        return {
            "Label": label,
            "Role": role,
            "GaugeName": gauge_name,
            "Note": note,
            "Freq_Hz": fmt(f, 3),
            "LengthEff_cm": fmt(m_to_cm(L_eff), 3),
            "LengthPhys_cm": fmt(m_to_cm(L_phys), 3),
            "EndCorr_mm": fmt(end_correction_mm, 3),
            "Mode": mode.upper(),
            "Overall_d_mm": fmt(m_to_mm(spec.overall_d), 3),
            "Core_d_mm": fmt(m_to_mm(spec.core_d), 3),
            "Wrap_d_mm": fmt(m_to_mm(spec.wrap_d), 3),
            "Pitch_mm": fmt(m_to_mm(spec.pitch), 3),
            "mu_kg_per_m": fmt(mu, 6),
            "Tension_N": fmt(T, 3),
            "Tension_kgf": fmt(T / 9.80665, 3),
            "WaveSpeed_m_per_s": fmt(c, 3),
            "B": fmt(B, 6),
        }

    # Melody: length ratios
    for i, note in enumerate(melody_notes, start=1):
        f = note_to_frequency(note, A4=a4)
        L_eff = L_anchor_eff * (f_anchor / f)
        L_phys = L_eff + end_corr_m
        rows.append(make_row(str(i), "melody", melody_gauge_name, note, L_eff, L_phys, geom_melody, mu_mel))

    # Drone: length<->tension logic
    f_drn = note_to_frequency(drone_note, A4=a4)

    T_target = drone_target_tension_N
    if drone_match_melody_tension:
        T_target = T_mel

    if T_target is not None:
        L_eff_drn = length_for(f_drn, T_target, mu_drn)
        L_phys_drn = L_eff_drn + end_corr_m
    else:
        if drone_Lphys_m is None:
            raise ValueError("Drone needs either --drone-length OR (--drone-target-tension-n / --drone-match-melody-tension).")
        L_eff_drn = drone_Lphys_m - end_corr_m
        if L_eff_drn <= 0.0:
            raise ValueError("Drone effective length <= 0. Increase drone length or reduce end correction.")
        L_phys_drn = drone_Lphys_m

    rows.insert(0, make_row("Drone", "drone", drone_gauge_name, drone_note, L_eff_drn, L_phys_drn, geom_drone, mu_drn))

    print_rows_plain(rows, sep=sep)

    if partials > 0:
        T_drn = tension_for(f_drn, L_eff_drn, mu_drn)
        core_r = geom_drone.core_d / 2.0
        B = inharmonicity_B_pinned(core_r, geom_drone.core_mat.youngs_E, T_drn, L_eff_drn)
        print(sep.join(["Partials_Label", "n", "f_n_Hz"]))
        for n in range(1, partials + 1):
            fn = stiff_partial(n, f_drn, B)
            print(sep.join(["DronePartials", str(n), fmt(fn, 3)]))

# -----------------------------
# CLI parsing
# -----------------------------

def parse_note_list(s: str) -> list[str]:
    out: list[str] = []
    for chunk in s.split(","):
        t = chunk.strip()
        if t:
            out.append(t)
    if not out:
        raise ValueError("Empty note list.")
    return out

def parse_anchor(s: str) -> tuple[str, float]:
    parts = s.split(":")
    if len(parts) != 2:
        raise ValueError(f"Bad anchor '{s}', expected NOTE:LENGTH_M (e.g. D2:0.651)")
    note = parts[0].strip()
    L = float(parts[1].strip())
    return note, L

def require_explicit(args, name: str) -> None:
    if getattr(args, name) is None:
        raise ValueError(f"Missing required argument: --{name.replace('_','-')} (gauge must be explicit).")

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Kantele design calculator — strict gauge-explicit CLI; generates lengths from melody anchor."
    )
    p.add_argument("--mode", choices=["uw", "geom"], default="uw",
                   help="μ model: 'uw' (Unit Weight) or 'geom' (round-wound geometry). Default: uw")
    p.add_argument("--sep", choices=["csv", "tsv"], default="csv",
                   help="Output separator: csv or tsv. Default: csv")
    p.add_argument("--a4", type=float, default=440.0, help="A4 reference frequency (Hz). Default 440.0")

    # Melody layout
    p.add_argument("--melody-notes", default="D2,E2,F2,G2,A2",
                   help="Comma-separated melody notes. Example: D2,E2,F2,G2,A2")
    p.add_argument("--melody-anchor", default="D2:0.651",
                   help="Anchor melody note and PHYSICAL length (m): NOTE:L_m. Example: D2:0.651")

    # Drone
    p.add_argument("--drone-note", default="D1", help="Drone note (e.g., D1).")
    p.add_argument("--drone-length", type=float, default=1.000,
                   help="Drone PHYSICAL length (m) used if no drone target tension is set. Default 1.000")
    p.add_argument("--drone-target-tension-n", type=float, default=None,
                   help="If set, compute drone length required to hit this tension (N).")
    p.add_argument("--drone-match-melody-tension", action="store_true",
                   help="If set, drone target tension = melody implied tension from the anchor.")

    # End correction
    p.add_argument("--end-correction-mm", type=float, default=0.0,
                   help="End correction (mm): L_eff = L_phys - correction. Default 0.0")

    # Gauge tags (recommended)
    p.add_argument("--melody-gauge-name", default="",
                   help="Human-readable gauge tag for melody (e.g. PB060). Included in output.")
    p.add_argument("--drone-gauge-name", default="",
                   help="Human-readable gauge tag for drone (e.g. PB075). Included in output.")

    # --- Gauge-defining parameters: STRICT (no defaults) ---
    # UW (required in uw mode)
    p.add_argument("--uw-melody", type=float, default=None, help="(uw mode) Melody UW (lb/in). REQUIRED in uw mode.")
    p.add_argument("--uw-drone",  type=float, default=None, help="(uw mode) Drone  UW (lb/in). REQUIRED in uw mode.")

    # Geometry (required in geom mode)
    p.add_argument("--melody-overall-mm", type=float, default=None, help="(geom mode) Melody overall diameter (mm). REQUIRED in geom mode.")
    p.add_argument("--melody-core-ratio", type=float, default=None, help="(geom mode) Melody core_d / overall_d. REQUIRED in geom mode.")
    p.add_argument("--melody-pitch-ratio", type=float, default=None, help="(geom mode) Melody pitch / wrap_d. REQUIRED in geom mode.")

    p.add_argument("--drone-overall-mm", type=float, default=None, help="(geom mode) Drone overall diameter (mm). REQUIRED in geom mode.")
    p.add_argument("--drone-core-ratio", type=float, default=None, help="(geom mode) Drone core_d / overall_d. REQUIRED in geom mode.")
    p.add_argument("--drone-pitch-ratio", type=float, default=None, help="(geom mode) Drone pitch / wrap_d. REQUIRED in geom mode.")

    # Partials
    p.add_argument("--partials", type=int, default=0, help="If >0, print that many drone partials with inharmonicity.")

    args = p.parse_args(argv)
    sep = "," if args.sep == "csv" else "\t"

    melody_notes = parse_note_list(args.melody_notes)
    anchor_note, anchor_Lphys = parse_anchor(args.melody_anchor)

    # Validate notes early
    _ = note_to_frequency(anchor_note, A4=args.a4)
    for n in melody_notes:
        _ = note_to_frequency(n, A4=args.a4)
    _ = note_to_frequency(args.drone_note, A4=args.a4)

    # Enforce explicit gauge parameters
    if args.mode == "uw":
        require_explicit(args, "uw_melody")
        require_explicit(args, "uw_drone")
        # Specs still needed for dimensions/B output (can be placeholders); but in uw mode they do not set μ.
        # We still need SOME geometry numbers to output diameters/B. Force explicit geometry too? Not necessary.
        # Instead, infer a minimal spec from optional geom args if provided; otherwise use safe placeholders.
        # For UW-driven design, you likely still want diameters correct: provide them anyway.
        # We'll set placeholder diameters if user doesn't care.
        if args.melody_overall_mm is None or args.melody_core_ratio is None or args.melody_pitch_ratio is None:
            geom_melody = wound_from_overall(mm_to_m(1.0), 0.6, 1.0)  # placeholder
        else:
            geom_melody = wound_from_overall(mm_to_m(args.melody_overall_mm), args.melody_core_ratio, args.melody_pitch_ratio)

        if args.drone_overall_mm is None or args.drone_core_ratio is None or args.drone_pitch_ratio is None:
            geom_drone = wound_from_overall(mm_to_m(1.0), 0.6, 1.0)  # placeholder
        else:
            geom_drone = wound_from_overall(mm_to_m(args.drone_overall_mm), args.drone_core_ratio, args.drone_pitch_ratio)

    else:  # geom
        require_explicit(args, "melody_overall_mm")
        require_explicit(args, "melody_core_ratio")
        require_explicit(args, "melody_pitch_ratio")
        require_explicit(args, "drone_overall_mm")
        require_explicit(args, "drone_core_ratio")
        require_explicit(args, "drone_pitch_ratio")
        geom_melody = wound_from_overall(mm_to_m(args.melody_overall_mm), args.melody_core_ratio, args.melody_pitch_ratio)
        geom_drone  = wound_from_overall(mm_to_m(args.drone_overall_mm),  args.drone_core_ratio,  args.drone_pitch_ratio)

    # Drone physical length usage
    drone_Lphys = None
    if args.drone_target_tension_n is None and (not args.drone_match_melody_tension):
        drone_Lphys = float(args.drone_length)

    build_rows_design(
        mode=args.mode,
        a4=args.a4,
        sep=sep,
        melody_notes=melody_notes,
        melody_anchor_note=anchor_note,
        melody_anchor_Lphys_m=anchor_Lphys,
        drone_note=args.drone_note,
        drone_Lphys_m=drone_Lphys,
        drone_target_tension_N=args.drone_target_tension_n,
        drone_match_melody_tension=bool(args.drone_match_melody_tension),
        melody_gauge_name=args.melody_gauge_name,
        drone_gauge_name=args.drone_gauge_name,
        geom_melody=geom_melody,
        geom_drone=geom_drone,
        uw_melody=args.uw_melody,
        uw_drone=args.uw_drone,
        end_correction_mm=args.end_correction_mm,
        partials=args.partials,
    )
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)
