# Kantele String & Layout Calculator

This repository contains a **string-length calculator for kantele design**.

The script is intended for **instrument layout and iteration**, not as a general physics simulator.  
Its primary output is **string speaking lengths**, suitable for use as **parametric constraints in CAD** (e.g. FreeCAD).

The calculator answers the practical question:

> Given a tuning and string gauges, what string lengths are required?

---

## Design Assumptions

The tool is based on the following assumptions, which match common kantele construction practice.

---

### 1. Melody strings have the same gauge

Melody strings are assumed to:
- be made of the same material
- have the same diameter (gauge)
- therefore have the same linear density μ (mass per unit length)

For an ideal string, the fundamental frequency is

<p align="left">
  <img alt="f = 1/(2L) * sqrt(T/mu)" src="https://latex.codecogs.com/svg.image?f%20%3D%20%5Cfrac%7B1%7D%7B2L%7D%5Csqrt%7B%5Cfrac%7BT%7D%7B%5Cmu%7D%7D" />
</p>

If μ and tension T are the same for all melody strings, this reduces to

<p align="left">
  <img alt="L proportional to 1 over f" src="https://latex.codecogs.com/svg.image?L%20%5Cpropto%20%5Cfrac%7B1%7D%7Bf%7D" />
</p>

This means:
- once one melody string length is chosen,
- all other melody lengths follow directly from frequency ratios.

---

### 2. The drone may have a different gauge

The drone string often:
- uses a thicker gauge
- has a different μ

Because of this, the drone cannot be handled using ratios alone.  
The script allows the drone to be treated separately by:

1. fixing its physical length and computing the resulting tension
2. fixing a target tension and computing the required length
3. matching the tension implied by the melody strings

---

### 3. Physical length is not the same as acoustic length

Real strings do not vibrate exactly from pin center to bridge center.  
The effective speaking length is usually slightly shorter than the physical distance.

The calculator uses the relation

<p align="left">
  <img alt="L_eff = L_physical - Delta_end" src="https://latex.codecogs.com/svg.image?L_%7B%5Ctext%7Beff%7D%7D%20%3D%20L_%7B%5Ctext%7Bphysical%7D%7D%20-%20%5CDelta_%7B%5Ctext%7Bend%7D%7D" />
</p>

where:
- `L_eff` is used in the string equations
- `L_physical` is used in CAD
- `Δ_end` is an empirical correction (typically a few millimeters)

This allows geometry adjustments without changing the tuning model.

---

## How the Calculator Works

### Melody strings

You provide:
- a list of melody notes (for example: `D4,E4,F4,G4,A4`)
- one **anchor note with a physical length**

All other melody string lengths are computed using

<p align="left">
  <img alt="L_i = L_anchor * f_anchor / f_i" src="https://latex.codecogs.com/svg.image?L_i%20%3D%20L_%7B%5Ctext%7Banchor%7D%7D%20%5Ccdot%20%5Cfrac%7Bf_%7B%5Ctext%7Banchor%7D%7D%7D%7Bf_i%7D" />
</p>

This keeps melody string tension uniform and makes the layout scale predictably.

---

### Drone string

The drone string is handled separately. Depending on the command-line options, the script will:

- compute drone tension from a fixed length
- compute drone length from a target tension
- compute drone length so that its tension matches the melody tension

---

## String Gauge and Linear Density

The string gauge determines the linear density μ.

The calculator requires the gauge to be specified explicitly.

### `uw` mode
- Uses manufacturer Unit Weight (UW) data
- μ is derived directly from UW
- Recommended when using commercial strings

### `geom` mode
- Computes μ from a round-wound geometric model
- Requires core diameter, wrap diameter, and pitch
- Intended for exploratory work

The script will not run unless gauge parameters are provided.

---

## Typical Usage Examples

### Example 1: Melody layout with fixed drone length

```bash
./kantele_calc.py --mode uw \
  --uw-melody 0.000545 \
  --uw-drone 0.000850 \
  --melody-gauge-name PB060 \
  --drone-gauge-name PB075 \
  --melody-notes "D2,E2,F2,G2,A2" \
  --melody-anchor "D2:0.651" \
  --drone-note D1 \
  --drone-length 1.000
