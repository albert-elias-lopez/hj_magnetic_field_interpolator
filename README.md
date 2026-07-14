# Hot Jupiter Magnetic Field Interpolator

This repository contains Python tools to estimate the magnetic field strengths of giant exoplanets, particularly hot and warm Jupiters, using interpolation of one-dimensional planetary evolution models and analytical magnetic scaling laws.

The interpolator is based on a precomputed grid of MESA evolutionary models combined with the dynamo scaling law of Christensen et al. (2009). It predicts:

- the rms magnetic field at the dynamo surface, $B_{\rm rms,dyn}$;
- the surface dipolar magnetic field, $B_{\rm dip,surf}$;
- the heating efficiency required to reproduce the observed planetary radius.

As an alternative to interpolation, the repository also provides simple analytical scaling laws that only depend on planetary mass, radius, and (optionally) irradiation.

---

### Repository contents

The repository contains

- magnetic field interpolators for the published model grids;
- analytical scaling laws;
- uncertainty propagation using Monte Carlo sampling;
- scripts to reproduce the figures of Elias-López et al. (2026).

The planetary evolution calculations themselves are **not** included here. They were performed using the MESA stellar evolution code and are available in the repository **https://github.com/csic-ice-imagine/MESA_rossby_HJ**

---

### Required inputs

The interpolator requires

- Planet mass ($M_{\rm P}$)
- Planet radius ($R_{\rm P}$)
- Incident stellar flux ($F_{\rm irr}$)
- Orbital period ($P_{\rm orb}$)

By default, the code uses the rotation-corrected magnetic field tables (Fuentes et al. 2025). This correction can be disabled using the `--no-rotation` option, in which case there is no need to provide the orbital period.

---

## Running the code

The main script can be executed from the command line:

```
python main.py
```

or with additional options, for example

```
python main.py --save_plot_population
```

or

```bash
python main.py --no-HJ --which_planets seidel26
```

---

### Command-line options

| General |  |
|---------|-------------|
| `--save_plot_grid` | Save the interpolation grid figures (mainly for testing). |
| `--save_plot_3D_grid` | Save a schematic 3D representation of the interpolation grid. |
| `--save_plot_population` | Save the magnetic field plots for the hot Jupiter population. |
| `--print_interpolator` | Print interpolator results for every planet. |
| Rotation |   |
|---------|-------------|
| *(default)* | Use the rotation-corrected magnetic field tables. |
| `--no-rotation` | Disable the rotation correction. |

## Planet selection

By default the code evaluates the full hot Jupiter population.

Use

```bash
--no-HJ
```

to instead evaluate one of several predefined samples:

```
--which_planets
```

Available options are

- `cauley19`
- `cauley19nasa`
- `seidel26`
- `seidel26nasa`
- `hats53b`
- `my_own`
- `errors`

---

## Scaling-law fitting

Three fitting methods are available:

- `simple` (default)
- `gaussian`
- `odr`

Example

```bash
--fit_type gaussian
```

The number of Monte Carlo realizations can be controlled with

```bash
--N_gauss_fits 10000
```

Different cuts used in the published paper can also be selected through

```bash
--which_cut
```

Options include

- `radius cut` (default)
- `flux cut`
- `heat cut`
- `brms cut`
- `no cut`

---

# Outputs

The code produces

- heating efficiency estimates;
- $B_{\rm rms,dyn}$;
- $B_{\rm dip,surf}$;
- uncertainty estimates;
- publication-quality figures reproducing those of Elias-López et al. (2026).

---

# Model assumptions

The interpolator assumes

- one-dimensional planetary evolution models;
- adiabatic interiors;
- standard H-He equation of state;
- rapidly rotating planetary dynamos.

The results should therefore be regarded as order-of-magnitude magnetic field estimates.

---

# Citation

If you use this repository, please cite

> Elias-López et al. (2026), *Scaling laws for strong and weak hot Jupiter magnetism*.

