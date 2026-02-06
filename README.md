# Kantele String & Layout Calculator

A **layout-first vibrating-string design tool** for kanteles and related plucked zithers.

This script is intended for **instrument design iteration**, not physics evaluation.  
Its primary output is **string speaking lengths** suitable for use as **parametric constraints in CAD (e.g. FreeCAD)**.

---

## Design Philosophy

The calculator is built around three practical assumptions:

1. **Melody strings share the same gauge**
   - Same material
   - Same linear density μ
   - Therefore, for equal tension:  

	<p align="left">
	  <img alt="L proportional to 1 over f" src="https://latex.codecogs.com/svg.image?L%20%5Cpropto%20%5Cfrac%7B1%7D%7Bf%7D" />
	</p>


2. **The drone can use a different gauge**
   - Different μ
   - Needs explicit handling to:
     - match melody feel, **or**
     - fit a fixed body length, **or**
     - hit a chosen target tension

3. **Real instruments are not ideal strings**
   - Terminations introduce end corrections
   - Geometry and stiffness matter
   - CAD layouts must reflect *physical* length, not purely acoustic length

This tool separates:
- **acoustic speaking length** (used in physics)
- **physical CAD length** (pin-to-bridge distance)

---

## What the Script Does

### Melody strings
- All melody strings are assumed to be **identical gauge**
- You specify:
  - a list of notes, like; D4, E4, F4, G4, A4 for a minor tuned kantele (F#4 for major)
  - one **anchor note + physical length**
- All other melody lengths are generated automatically by frequency ratios: <p align="left"> <img alt="L_i = L_anchor * f_anchor / f_i" src="https://latex.codecogs.com/svg.image?L_i%20%3D%20L_%7B%5Ctext%7Banchor%7D%7D%20%5Ccdot%20%5Cfrac%7Bf_%7B%5Ctext%7Banchor%7D%7D%7D%7Bf_i%7D" />
</p>

### Drone string
The drone can:
- use a **different gauge**
- be solved in one of three ways:
  1. Fixed physical length → compute resulting tension
  2. Fixed target tension → compute required length
  3. Match the **melody tension** implied by the anchor

---

## Modes: How μ Is Computed

### `uw` mode (recommended)
Uses **manufacturer Unit Weight (UW)** data.

- μ is derived directly from UW
- Matches real commercial strings well
- Best choice if you know the string model

### `geom` mode
Uses a **first-principles round-wound geometry model**.

- Requires assumptions about:
  - core diameter
  - wrap diameter
  - winding pitch
- Useful for exploration and sensitivity studies
- Less accurate than UW for real strings

---

## End Correction (Important)

Real strings do **not** vibrate over the full pin-to-bridge distance.

This tool models that explicitly:
<p align="left">
  <img alt="L_eff = L_physical - Delta_end" src="https://latex.codecogs.com/svg.image?L_%7B%5Ctext%7Beff%7D%7D%20%3D%20L_%7B%5Ctext%7Bphysical%7D%7D%20-%20%5CDelta_%7B%5Ctext%7Bend%7D%7D" />
</p>
Where:
- `L_eff` → used in all physics
- `L_physical` → what you put in CAD
- `Δ_end` → empirical correction (typically 2–10 mm)

---

## Typical Workflow (Recommended)

1. **Choose instrument size**
   - Decide where one melody string must land physically

2. **Set melody anchor**
   ```bash
   --melody-anchor "D2:0.651"
