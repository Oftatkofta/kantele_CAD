# Kantele String Length Calculator

This repository contains a **string length calculator for kantele design**.

The purpose of the script is to support **instrument layout**.  
It computes **string speaking lengths** that can be used directly as **parametric constraints in CAD** (for example in FreeCAD).

It answers the design question:

> Given a tuning and a string gauge, how long will the strings be?

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

This is used in the design workflow:
- once one melody string length is fixed,
- all other melody lengths follow from frequency ratios.

---

### 2. An optional drone string may use a different gauge

A traditional kantelde does not have a drone string, but a custom built hypnotic doom base kantele might.
The drone string will sit one octave below the D string and can use a thicker gauge and different tension, and therefore a different μ and T.  


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

`Δ_end` can be roughly approximated to be: Δ_end​≈δrod​+δpin, where δpin is the radius of the tuning pin and δrod​​​ the radius of the string bar.


---

## How the Calculator Is Used

### Step 1: Choose a melody anchor

You must choose:
- one melody note (assumes minor tuning with the notes DEFGA) 
- a **physical length** for that note

Example:

D2 at 650 mm using string PB060 (Phosphor Bronze Acoustic String, .060 gauge)


This fixes the **absolute scale** of the instrument for the chosen string gauge.

---

### Step 2: Generate the remaining melody lengths

For melody note `i`, the length is computed as:

<p align="left">
  <img alt="L_i = L_anchor * f_anchor / f_i" src="https://latex.codecogs.com/svg.image?L_i%20%3D%20L_%7B%5Ctext%7Banchor%7D%7D%20%5Ccdot%20%5Cfrac%7Bf_%7B%5Ctext%7Banchor%7D%7D%7D%7Bf_i%7D" />
</p>

This ensures:
- uniform tension across melody strings
- predictable scaling when the anchor changes

---

## String Gauge Specification

The string gauge determines the linear density `μ`.  
It **must** be specified explicitly.

### `uw` mode (recommended)

- Gauge is specified using manufacturer **Unit Weight (UW)**
- `μ` is derived directly from UW
- Best match to real commercial strings where data is available from the manufacturer.

Example:

--uw-melody 0.00073039 lbf/in (D'addario PB060)
--uw-drone 0.00096833  lbf/in (D'addario PB070)

!IMORTANT!

The script uses metric values as output and you havew to specify the unit of the uw value to avoid sadness.

	
### `geom` mode

- `μ` is computed from a round-wound geometry model
- Requires diameter, core ratio, and winding pitch
- Intended for exploratory work

The script will refuse to run unless gauge parameters are provided.

---

## CLI usage examples

### Example 1 — Basic melody layout (default DEFGA scale, CSV output)

```bash
python kantele_calc.py --anchor D2:0.65 --uw-melody-lbf-per-in 0.00073039 
```

Generates a 5-string melody layout (D2 E2 F2 G2 A2) and outputs results as CSV.

### Example 2 — Same layout with end correction (TSV output)
```bash
python kantele_calc.py --anchor D2:0.65 --end-correction-mm 5 --uw-melody-lbf-per-in 0.00073039 --sep tsv
```
Applies a 5 mm end correction and outputs tab-separated values for easy import into spreadsheets.

### Example 3 — Add a drone with fixed physical length
```bash
python kantele_calc.py --anchor D2:0.65 --uw-melody-lbf-per-in 0.00073039 --uw-drone-lbf-per-in 0.00096833 --drone-length 0.8 --sep csv
```
Adds a drone string at D1 with a fixed length of 80 cm, of a thicker gague, and computes its tension.

### Example 4 — Traditional kantele-style major scale (μ only)
```bash
python kantele_calc.py --anchor D4:0.45 --scale "D4,E4,F#4,G4,A4" --mu-melody-kg-per-m 0.0045
```
Uses a D-major pentachord in a typical kantele register, anchored at D4 with a 45 cm reference length; assumes a plain steel (piano-wire–like) string with moderate linear density.


## Spreadsheet alternative


If you prefer not to use the command line, a LibreOffice Calc spreadsheet (kantele_calc.fods) is included in the repository.

The spreadsheet implements the same equations as the CLI tool and updates dynamically when you change:

the anchor length

the string linear density (μ)

the end correction

You can use it to explore string lengths and tensions interactively.
