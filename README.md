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

By default, the code uses the rotation-corrected magnetic field tables (Fuentes et al. 2025). This correction can be disabled using the `--no_rotation` option, in which case the orbital period will not be used.

---

## Running the code

The main script can be executed from the command line:

```
python main.py
```

or with additional options, for example

```
python main.py --save_plot_population --fit_type simple
python main.py --no_rotation --no_HJ --which_planets seidel26
python main.py --no_HJ --which_planets my_own --my_planet_params 1.2 0.1 1.5 0.1 1 1e-6 1e9 1e8
```

To reproduce the results in Elias-López et al 2026:

```
python main.py --save_plot_population (by default fit_type=gaussian, N_gauss_fits=10000, which_cut=radius cut)
python main.py --no_HJ --which_planets seidel26 --add_systematic_error
python main.py --no_HJ --which_planets cauley19 --add_systematic_error
```

---

### Command-line options

|       Option              | Description |
|----------|-------------|
| General                   |       |
| `--save_plot_grid`        | Save the interpolation grid figures (mainly for testing). *(default)=False* |
| `--save_plot_3D_grid`     | Save a schematic 3D representation of the interpolation grid. *(default)=False* |
| `--save_plot_population`  | Save the magnetic field plots for the hot Jupiter population. *(default)=False* |
| `--print_planet_output`   | Print interpolator results for every planet. *(default)=False*, automatically enabled with `--no_HJ` |
| Rotation                  |       |
| *(default)*               | Use the rotation-corrected magnetic field tables. |
| `--no_rotation`           | Disable the rotation correction. |
| Scaling-law fitting       |       |
| `--fit_type`              | Fitting methods are available: `simple`, `gaussian` *(default)*, `odr`. |
| `--N_gauss_fits`          | Number of Monte Carlo realizations *(default)=10000* |
| `--print_scaling_law`     | Print the scaling-law fit results in the terminal (long output, mainly for testing). *(default)=False* |
| `--which_cut`             | Different population cuts before scaling law fits: `radius cut` *(default)*, `flux cut`, `heat cut`, `brms cut`, `no cut`  |
| Planet selection          |       |
| *(default)*               | Evaluates the full hot Jupiter population and calculates scaling law fits. |
| `--no_HJ`                 | Instead evaluates one of several predefined samples. |
| `--which_planets`         | Used if --no_HJ. It can be  `cauley19` *(default)*, `cauley19nasa`, `seidel26`, `seidel26nasa`, `hats53b`, `wasp12b`, `hd209458b`, `my_own`, `errors`|
| `--my_planet_params`      | Used if `--no_HJ --which_planets my_own`. Specifies your own planet parameters: $M_{\rm P}$, $\sigma_{M_{\rm P}}$, $R_{\rm P}$ , $\sigma_{R_{\rm P}}$, $P_{\rm orb}$, $\sigma_{P_{\rm orb}}$, $F_{\rm irr}$, $\sigma_{F_{\rm irr}}$. The default is `--my_planet_params 1.2 0.1 1.5 0.1 1 1e-6 1e9 1e8`|
| Uncertainties             |       |
| `--add_systematic_error`  | Add a 60 percent relative systematic error to the interpolated/scaling law magnetic field estimates. This is a rough estimate of the error associated with the interior structure and dynamo model assumptions. By default, and the uncertainty only comes from the propagation of observed data errors.|

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

