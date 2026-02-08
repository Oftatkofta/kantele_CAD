#!/usr/bin/env python3
"""Kantele string length calculator 

Purpose
- Generate physical speaking lengths for a 5-note melody scale from one anchor string.
- Handle a separate drone string that may have a different linear density.

Model
Ideal string fundamental:
    f = (1/(2 L_eff)) * sqrt(T / mu)

End correction
We separate the effective vibrating length used in physics (L_eff) from the
physical CAD length (L_phys):
    L_eff = L_phys - delta_end

Inputs for string density
For each of (melody, drone) you must specify exactly one of:
- mu in kg/m  (--mu-*-kg-per-m)
- UW in lbf/in (--uw-*-lbf-per-in)  [D'Addario convention: weight per unit length]

Output is plain CSV/TSV.
"""

from __future__ import annotations

import argparse
import re
import sys
from math import sqrt

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
    """Parse a note like D2, C#4, Db3 into Hz using equal temperament."""
    s = note.strip().upper().replace('♭', 'B').replace('♯', '#')
    m = _NOTE_RE.match(s)
    if not m:
        raise ValueError(f"Bad note format: '{note}' (expected like D1, C#4, Db3)")
    letter, accidental, octave_str = m.group(1), m.group(2), m.group(3)
    name = f"{letter}{accidental}"
    name = _FLAT_TO_SHARP.get(name, name)
    if name not in NOTE_ORDER:
        raise ValueError(f"Unrecognized note name: '{note}' -> '{name}'")
    octave = int(octave_str)
    midi = (octave + 1) * 12 + NOTE_ORDER.index(name)
    return A4 * (2.0 ** ((midi - 69) / 12.0))


def mm_to_m(x_mm: float) -> float:
    return x_mm / 1000.0


def m_to_cm(x_m: float) -> float:
    return x_m * 100.0


def m_to_in(x_m: float) -> float:
    return x_m / 0.0254


def mu_from_uw_lbf_per_in(UW_lbf_per_in: float) -> float:
    """Convert Unit Weight (lbf/in) to linear density mu (kg/m)."""
    g = 9.80665
    # UW is weight/length. Convert to N/m then divide by g -> kg/m.
    UW_N_per_m = UW_lbf_per_in * 4.4482216152605 / 0.0254
    return UW_N_per_m / g


def tension_for(f_hz: float, L_eff_m: float, mu_kg_per_m: float) -> float:
    """T = 4 L^2 f^2 mu"""
    return 4.0 * (L_eff_m ** 2) * (f_hz ** 2) * mu_kg_per_m


def length_for(f_hz: float, T_N: float, mu_kg_per_m: float) -> float:
    """Solve for effective length: L = (1/(2f)) * sqrt(T/mu)."""
    return (1.0 / (2.0 * f_hz)) * sqrt(T_N / mu_kg_per_m)


def wave_speed(T_N: float, mu_kg_per_m: float) -> float:
    return sqrt(T_N / mu_kg_per_m)


def fmt(x: float, places: int = 3) -> str:
    return ("{0:." + str(places) + "f}").format(x)


def parse_anchor(s: str) -> tuple[str, float]:
    """Parse NOTE:LENGTH_M, e.g. D2:0.651"""
    parts = s.split(':')
    if len(parts) != 2:
        raise ValueError("Anchor must be NOTE:LENGTH_M (e.g. D2:0.651)")
    return parts[0].strip(), float(parts[1].strip())


def parse_scale_5(s: str) -> list[str]:
    notes = [x.strip() for x in s.split(',') if x.strip()]
    if len(notes) != 5:
        raise ValueError("Scale must contain exactly 5 notes, comma-separated.")
    return notes



def default_scale_from_anchor(anchor_note: str) -> list[str]:
    """Generate a 5-note diatonic letter sequence starting at a natural anchor note.

    Examples:
      D2 -> D2,E2,F2,G2,A2
      A2 -> A2,B2,C3,D3,E3
    """
    s = anchor_note.strip().upper().replace('♭', 'B').replace('♯', '#')
    m = _NOTE_RE.match(s)
    if not m:
        raise ValueError(f"Bad note format: '{anchor_note}'")
    letter, accidental, octave_str = m.group(1), m.group(2), m.group(3)
    if accidental:
        raise ValueError("Default scale generation requires a natural anchor note (no #/b). Provide --scale explicitly.")
    octave0 = int(octave_str)
    letters = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
    i0 = letters.index(letter)

    out: list[str] = []
    for k in range(5):
        idx = i0 + k
        out_letter = letters[idx % 7]
        out_oct = octave0 + (idx // 7)
        out.append(f"{out_letter}{out_oct}")
    return out


def require_one_of(name_a: str, val_a, name_b: str, val_b) -> None:
    if (val_a is None) == (val_b is None):
        raise ValueError(f"Specify exactly one of --{name_a} or --{name_b}.")


def print_rows(rows: list[dict[str, str]], sep: str) -> None:
    headers = [
        "Label", "Role", "Note", "Freq_Hz",
        "LengthEff_cm", "LengthPhys_cm", "EndCorr_mm",
        "mu_kg_per_m", "Tension_N", "Tension_kgf", #"WaveSpeed_m_per_s",
    ]
    print(sep.join(headers))
    for r in rows:
        print(sep.join(r.get(h, "") for h in headers))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Kantele layout calculator (5-note scale + drone)")
    p.add_argument("--sep", choices=["csv", "tsv"], default="csv", help="csv or tsv output")
    p.add_argument("--a4", type=float, default=440.0, help="A4 reference in Hz")

    # Scale and anchor
    p.add_argument(
        "--scale",
        default=None,
        help=("Optional: exactly 5 notes (comma-separated). "
              "If omitted, a default 5-note diatonic sequence is generated from the anchor note "
              "(e.g. D2 -> D2,E2,F2,G2,A2; A2 -> A2,B2,C3,D3,E3)."),
    )
    p.add_argument(
        "--anchor",
        default="D2:0.650",
        help="Anchor melody string as NOTE:LENGTH_M (physical). Default D2:0.650",
    )

    # Drone definition
    p.add_argument("--drone-note", default="D1", help="Drone note, e.g. D1")
    p.add_argument("--drone-length-m", type=float, default=1.000, help="Drone physical length in meters")
    p.add_argument("--drone-target-tension-n", type=float, default=None, help="If set, solve drone length for this tension (N)")
    p.add_argument("--drone-match-melody-tension", action="store_true", help="Set drone tension equal to melody tension")

    # End correction
    p.add_argument("--end-correction-mm", type=float, default=0.0, help="delta_end in mm (L_eff = L_phys - delta_end)")

    # Linear density inputs (explicit units)
    p.add_argument("--mu-melody-kg-per-m", type=float, default=None, help="Melody linear density mu in kg/m")
    p.add_argument("--uw-melody-lbf-per-in", type=float, default=None, help="Melody Unit Weight UW in lbf/in")

    p.add_argument("--mu-drone-kg-per-m", type=float, default=None, help="Drone linear density mu in kg/m")
    p.add_argument("--uw-drone-lbf-per-in", type=float, default=None, help="Drone Unit Weight UW in lbf/in")

    args = p.parse_args(argv)
    sep = "," if args.sep == "csv" else "\t"

    anchor_note, anchor_Lphys = parse_anchor(args.anchor)

    if args.scale is None:
        scale_notes = default_scale_from_anchor(anchor_note)
    else:
        scale_notes = parse_scale_5(args.scale)
        # Guard: scale should start on the anchor pitch (same octave)
        if abs(note_to_frequency(scale_notes[0], A4=args.a4) - note_to_frequency(anchor_note, A4=args.a4)) > 1e-9:
            raise ValueError("Scale first note must match the anchor note (same pitch and octave).")

    # Validate notes early
    _ = note_to_frequency(anchor_note, A4=args.a4)
    for n in scale_notes:
        _ = note_to_frequency(n, A4=args.a4)
    _ = note_to_frequency(args.drone_note, A4=args.a4)

    require_one_of("mu-melody-kg-per-m", args.mu_melody_kg_per_m, "uw-melody-lbf-per-in", args.uw_melody_lbf_per_in)
    require_one_of("mu-drone-kg-per-m", args.mu_drone_kg_per_m, "uw-drone-lbf-per-in", args.uw_drone_lbf_per_in)

    mu_mel = args.mu_melody_kg_per_m if args.mu_melody_kg_per_m is not None else mu_from_uw_lbf_per_in(args.uw_melody_lbf_per_in)
    mu_drn = args.mu_drone_kg_per_m if args.mu_drone_kg_per_m is not None else mu_from_uw_lbf_per_in(args.uw_drone_lbf_per_in)

    if mu_mel <= 0.0 or mu_drn <= 0.0:
        raise ValueError("mu must be positive.")

    delta_end_m = mm_to_m(args.end_correction_mm)

    # Melody tension is implied by the anchor and melody mu.
    f_anchor = note_to_frequency(anchor_note, A4=args.a4)
    L_anchor_eff = anchor_Lphys - delta_end_m
    if L_anchor_eff <= 0.0:
        raise ValueError("Anchor effective length <= 0. Increase anchor length or reduce end correction.")

    T_mel = tension_for(f_anchor, L_anchor_eff, mu_mel)

    def row(label: str, role: str, note: str, L_eff: float, L_phys: float, mu: float) -> dict[str, str]:
        f = note_to_frequency(note, A4=args.a4)
        T = tension_for(f, L_eff, mu)
        c = wave_speed(T, mu)
        return {
            "Label": label,
            #"Role": role,
            "Note": note,
            "Freq_Hz": fmt(f, 3),
            "LengthEff_cm": fmt(m_to_cm(L_eff), 3),
            "LengthPhys_cm": fmt(m_to_cm(L_phys), 3),
            "EndCorr_mm": fmt(args.end_correction_mm, 3),
            "mu_kg_per_m": fmt(mu, 6),
            "Tension_N": fmt(T, 3),
            "Tension_kgf": fmt(T / 9.80665, 3),
            #"WaveSpeed_m_per_s": fmt(c, 3),
        }

    rows: list[dict[str, str]] = []

    # Drone: either fixed length -> tension, or fixed tension -> length.
    f_drn = note_to_frequency(args.drone_note, A4=args.a4)
    T_target = args.drone_target_tension_n
    if args.drone_match_melody_tension:
        T_target = T_mel

    if T_target is not None:
        if T_target <= 0.0:
            raise ValueError("Target tension must be positive.")
        L_eff_drn = length_for(f_drn, T_target, mu_drn)
        L_phys_drn = L_eff_drn + delta_end_m
    else:
        L_phys_drn = float(args.drone_length_m)
        L_eff_drn = L_phys_drn - delta_end_m
        if L_eff_drn <= 0.0:
            raise ValueError("Drone effective length <= 0. Increase drone length or reduce end correction.")

    rows.append(row("Drone", "drone", args.drone_note, L_eff_drn, L_phys_drn, mu_drn))

    # Melody scale: lengths from anchor ratios (same mu and same tension by construction).
    for i, note in enumerate(scale_notes, start=1):
        f = note_to_frequency(note, A4=args.a4)
        L_eff = L_anchor_eff * (f_anchor / f)
        L_phys = L_eff + delta_end_m
        rows.append(row(str(i), "melody", note, L_eff, L_phys, mu_mel))

    print_rows(rows, sep=sep)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)
