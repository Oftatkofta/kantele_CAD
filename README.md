# Kantele String & Layout Calculator

This repository contains a **string length calculator for kantele design**.

The purpose of the script is to support **instrument layout**.  
It computes **string speaking lengths** that can be used directly as **parametric constraints in CAD** (for example in FreeCAD).

The script does **not** attempt to replace empirical tuning or final voicing.  
It answers a narrower, earlier design question:

> Given a tuning and a set of string gauges, where must the strings be placed?

---

## Physical Model Used

The calculator is based on the standard ideal-string relation for the fundamental frequency:

<p align="left">
  <img alt="f = 1/(2L) * sqrt(T/mu)" src="https://latex.codecogs.com/svg.image?f%20%3D%20%5Cfrac%7B1%7D%7B2L%7D%5Csqrt%7B%5Cfrac%7BT%7D%7B%5Cmu%7D%7D" />
</p>

where:
- `f` is frequency
- `L` is effective speaking length
- `T` is string tension
- `μ` is linear density (mass per unit length)

This equation is rearranged as needed to compute lengths or tensions.

---

## Assumptions About the Instrument

### 1. Melody strings have the same gauge

All melody strings are assumed to:
- be made from the same material
- have the same diameter (gauge)
- therefore have the same linear density μ

If μ and tension are the same for all melody strings, the frequency equation reduces to a simple proportionality:

<p align="left">
  <img alt="L proportional to 1 over f" src="https://latex.codecogs.com/svg.image?L%20%5Cpropto%20%5Cfrac%7B1%7D%7Bf%7D" />
</p>

This is the key observation used in the design workflow:
- once one melody string length is fixed,
- all other melody lengths follow from frequency ratios.

---

### 2. The drone may use a different gauge

The drone string often uses a thicker gauge and therefore a different μ.  
Because of this, the drone string is treated separately.

Depending on design goals, the drone can be:
1. fixed in length (compute resulting tension)
2. fixed in tension (compute required length)
3. matched in tension to the melody strings

---

### 3. Physical length vs effective length

In a real instrument, the string does not vibrate exactly from pin center to bridge center.  
The effective speaking length is slightly shorter.

The calculator uses the relation:

<p align="left">
  <img alt="L_eff = L_physical - Delta_end" src="https://latex.codecogs.com/svg.image?L_%7B%5Ctext%7Beff%7D%7D%20%3D%20L_%7B%5Ctext%7Bphysical%7D%7D%20-%20%5CDelta_%7B%5Ctext%7Bend%7D%7D" />
</p>

where:
- `L_eff` is used in the physics
- `L_physical` is what you place in CAD
- `Δ_end` is an empirical correction (typically a few millimeters)

---

## How the Calculator Is Used

### Step 1: Choose a melody anchor

You must choose:
- one melody note
- a **physical length** for that note

Example:
