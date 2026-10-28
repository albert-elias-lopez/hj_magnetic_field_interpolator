# Hot Jupiter Magnetic Field Interpolator

This repository contains Python tools to estimate the magnetic field strengths of giant exoplanets, particularly hot and warm Jupiters, using interpolation of one-dimensional planetary evolution models and analytical magnetic scaling laws.

The interpolator is based on a precomputed grid of MESA evolutionary models combined with the dynamo scaling law of Christensen et al. (2009). For each planet it predicts:

- the heating efficiency $\epsilon$ required to reproduce the observed planetary radius;
- the rms magnetic field at the dynamo surface, $B_{\rm rms,dyn}$;
- the surface dipolar magnetic field, $B_{\rm dip,surf}$.

As an alternative to interpolation, the repository also provides simple analytical scaling laws, fitted to the interpolated population, that only depend on planetary mass, radius and (optionally) irradiation.

---

## Repository contents

- magnetic field interpolators for the published model grids (`<mass>_MJ/` folders);
- analytical scaling laws fitted to the hot Jupiter population;
- uncertainty propagation using Monte Carlo sampling, including asymmetric uncertainties;
- a Python interface (`hj_model.py`) to use the interpolator and the scaling laws from your own code;
- scripts to reproduce the results of Elias-López et al. (2026).

The planetary evolution calculations themselves are **not** included here. They were performed using the MESA stellar evolution code and are available in the repository **https://github.com/csic-ice-imagine/MESA_rossby_HJ**

The magnetic field tables in the `<mass>_MJ/` folders were generated from those runs with `table_generator.py` (`python table_generator.py --path_sims <MESA runs folder>` for the tables with rotation, adding `--no_rotation` for the tables without it). The script needs the MESA output and the module `radial_functions`, which are not included here but can be provided upon request.

---

## Required inputs

For each planet the interpolator requires

- planet mass $M_{\rm P}$ (in $M_{\rm J}$);
- planet radius $R_{\rm P}$ (in $R_{\rm J}$);
- incident stellar flux $F_{\rm irr}$ (in erg cm$^{-2}$ s$^{-1}$);
- orbital period $P_{\rm orb}$ (in days);

together with their uncertainties. The model grid covers $0.3 < M_{\rm P} < 12\,M_{\rm J}$, $10^7 < F_{\rm irr} < 1.58\times10^{10}$ erg cm$^{-2}$ s$^{-1}$ and $0.1 < P_{\rm orb} < 100$ days; planets outside it are not interpolated.

By default, the code uses the rotation-corrected magnetic field tables (Fuentes et al. 2025), which assume tidally locked planets. This correction can be disabled with the `--no_rotation` option, in which case the orbital period is not used.

For the hot Jupiter population, the inputs are read from the NASA Exoplanet Archive tables included in the repository (`PS_2026.02.18_*.csv`). Missing stellar temperatures and radii are estimated from a main-sequence relation (`main_sequence_luminosities.txt`), and a missing semi-major axis from Kepler's third law.

---

## Installation

You need Python 3.11 or 3.12 (check with `python3 --version`). These are the versions that work with the packages used. Any Python >= 3.10 also works with the `--latest` option below. LaTeX is optional: if it is installed, it is used for the plot labels; otherwise matplotlib's own math rendering is used.

**1. Download the code and create the environment** (only once):

```
git clone https://github.com/albert-elias-lopez/hj_magnetic_field_interpolator.git
cd hj_magnetic_field_interpolator
python3 set_up.py
```

`set_up.py` creates a virtual environment in the folder `.venv` of the repository, installs the package versions used in the paper (see [Package versions](#package-versions)) and checks that the code runs. With these versions, the default random seed reproduces the results of the paper exactly. It can be run from any folder, e.g. `python3 path/to/set_up.py`.

Options and common problems:

- `python3 set_up.py --latest` installs the newest package versions instead, for other Python versions. The last digits of the results may then differ slightly from the paper.
- If your default Python is not 3.11 or 3.12 but one of them is installed, run the script with it, e.g. `python3.12 set_up.py`.
- On Debian/Ubuntu, if creating the environment fails, install the venv module first (`sudo apt install python3.12-venv`, with your Python version) and run the script again.
- Running `set_up.py` again reuses the existing environment. To start from scratch, delete the folder `.venv` first.
- If you prefer to manage your own environment, install the packages listed in [Package versions](#package-versions).

**2. Activate the environment** (every time you open a new terminal):

```
source .venv/bin/activate        (Windows: .venv\Scripts\activate)
python main.py --help
```

When you are done, `deactivate` leaves the environment. Without activating it, you can also run the code directly with `.venv/bin/python main.py` (Windows: `.venv\Scripts\python main.py`).

### Package versions

`set_up.py` installs the package versions used for the results of the paper, which need Python 3.11 or 3.12. With these versions, the default random seed reproduces the results exactly.

| Package | Version |
|---|---|
| numpy | 1.26.4 |
| scipy | 1.16.2 |
| matplotlib | 3.10.6 |
| tqdm | 4.67.1 |
| emcee | 3.1.6 |
| odrpack | 0.5.0 |

To install them in your own environment instead:

```
pip install numpy==1.26.4 scipy==1.16.2 matplotlib==3.10.6 tqdm==4.67.1 emcee==3.1.6 odrpack==0.5.0
```

---

## Running the code

To reproduce the results of Elias-López et al. (2026), run these commands in this order (all but the first use the scaling-law fits saved by the first one):

```
python main.py --save_plot_population
python main.py --no_HJ --which_planets seidel26 --add_systematic_error
python main.py --no_HJ --which_planets cauley19 --add_systematic_error
python main.py --no_HJ --which_planets seidel26
python main.py --no_HJ --which_planets cauley19
python main.py --no_HJ --which_planets wasp12b
python main.py --no_HJ --which_planets hd209458b
```

The runs with `--add_systematic_error` include a factor-of-2 systematic uncertainty (×½ below, ×2 above), as in the literature comparison figure; the others only include the observational uncertainties, as in the literature comparison table.

The first command uses the defaults `--fit_type gaussian`, `--N_gauss_fits 10000` and `--which_cut "radius cut"`, and takes about 10 minutes. The Monte Carlo uncertainties and fits use a fixed random seed (`--seed`, default 42), so the same command always gives exactly the same results.

Other examples:

```
python main.py --save_plot_population --fit_type simple
python main.py --no_rotation --no_HJ --which_planets seidel26
python main.py --no_HJ --which_planets my_own --my_planet_params 1.2 0.1 1.5 0.1 1 1e-6 1e9 1e8
```

The last one estimates the magnetic field of your own planet (here $M_{\rm P} = 1.2 \pm 0.1\,M_{\rm J}$, $R_{\rm P} = 1.5 \pm 0.1\,R_{\rm J}$, $P_{\rm orb} = 1$ day, $F_{\rm irr} = (1.0 \pm 0.1)\times10^9$ erg cm$^{-2}$ s$^{-1}$) and prints the results in the terminal.

The code can be launched from any folder; the outputs are always written in the repository folders `predictions/` and `plots/`.

---

## Using the interpolator from Python

`hj_model.py` gives the same estimates as `main.py` from your own scripts or notebooks, without writing files. Run Python from the repository folder (or add it to `sys.path`):

```python
from hj_model import HJMagneticModel

model = HJMagneticModel()          # reads the grid tables once (a few seconds); rotation=False for the tables without rotation

# Incident flux from the stellar radius and temperature and the orbital distance (or the period and stellar mass)
flux, flux_err = model.get_flux(radius_star=1.57, radius_star_err=0.07, temp_star=6300, temp_star_err=150,
                                orbital_dist=0.0229, orbital_dist_err=0.0008)

# Interpolated heating efficiencies and magnetic fields, with Monte Carlo errors
result = model.estimate(mass=1.41, mass_err=0.10, radius=1.79, radius_err=0.09,
                        flux=flux, flux_err=flux_err, period=1.0914, period_err=3e-6)
print(result["brms_inner_surf"], result["brms_inner_surf_err"])   # B_dip,surf (G) for deep heating, (minus, plus) error

# Published scaling laws (MR fit without flux, MRF fit with flux)
b, b_err = model.scaling_law(mass=1.41, mass_err=0.10, radius=1.79, radius_err=0.09, heating="outer")
```

- **Units:** planet mass in $M_{\rm J}$, planet radius in $R_{\rm J}$, flux in erg cm$^{-2}$ s$^{-1}$, orbital period in days, stellar radius in $R_\odot$, stellar temperature in K, stellar mass in $M_\odot$, orbital distance in AU.
- **Several planets at once:** every value can be an array.
- **Errors:** a number or array for symmetric errors, or a tuple `(lower, upper)` for asymmetric ones. `np.nan` marks a missing error.
- **Results of `estimate`:** `eps_inner`, `eps_outer` (heating efficiencies, as fractions), `brms_inner`, `brms_inner_surf`, `brms_outer`, `brms_outer_surf` ($B_{\rm rms,dyn}$ and $B_{\rm dip,surf}$ in G for deep and shallow heating), `brms_irr`, `brms_irr_surf` and `radius_irr` (the non-inflated, $\epsilon = 0$, model with the same mass and flux). Each has its `(minus, plus)` error under `<name>_err`.
- **Planets without estimates:** planets outside the grid in mass, flux or period, smaller than their non-inflated model (only `radius_irr` is given), larger than the most inflated model, or with a missing value give `nan`, and `estimate` shows a `GridWarning` that lists each of them and the reason. Silence it with `warnings.simplefilter("ignore", hj_model.GridWarning)`. `main.py` prints how many planets have no estimates for each reason.
- **Random numbers:** the Monte Carlo errors use numpy's random numbers with `seed=42` by default, as `main.py`, so the same input always gives the same errors.
- **Check:** `python hj_model.py` (or `hj_model.check()`) verifies that the interface reproduces the committed predictions of `main.py` exactly. Run it after changing the code; the GitHub Action also runs it.

---

## Command-line options

| Option | Description |
|----------|-------------|
| **General** | |
| `--save_plot_grid` | Save the interpolation grid figures to `plots/<mass>_MJ/`, from the tables without rotation (mainly for testing). *(default: False)* |
| `--save_plot_3D_grid` | Save a schematic 3D plot of the grid ($F_{\rm irr}$, $M$, $\epsilon$) to `plots/population_grid*.png`, from the tables without rotation. *(default: False)* |
| `--save_plot_population` | Save the magnetic field plots for the hot Jupiter population. *(default: False)* |
| `--print_planet_output` | Print the interpolator results for every planet. *(default: False; always on with `--no_HJ`)* |
| **Rotation** | |
| *(default)* | Use the rotation-corrected magnetic field tables. |
| `--no_rotation` | Disable the rotation correction. |
| **Scaling-law fitting** | |
| `--fit_type` | Fitting method: `simple` (least squares), `gaussian` (Monte Carlo, *default*) or `odr` (orthogonal distance regression). |
| `--N_gauss_fits` | Number of Monte Carlo realizations of the `gaussian` fits. *(default: 10000)* |
| `--print_scaling_law` | Print the scaling-law fit results in the terminal (long output, mainly for testing). *(default: False)* |
| `--which_cut` | Population cut applied before the scaling-law fits: `radius cut` (*default*, $R_{\rm obs}/R_{\rm irr} > 1.05$), `flux cut`, `heat cut`, `brms cut` or `no cut`. |
| **Planet selection** | |
| *(default)* | Evaluate the full hot Jupiter population and calculate the scaling-law fits. |
| `--no_HJ` | Instead evaluate one of several predefined samples, using the previously saved scaling-law fits. |
| `--which_planets` | Sample used with `--no_HJ`: `cauley19` *(default)*, `cauley19nasa`, `seidel26`, `seidel26nasa`, `hats53b`, `wasp12b`, `hd209458b`, `my_own` or `errors`. |
| `--my_planet_params` | Used with `--no_HJ --which_planets my_own`: $M_{\rm P}$, $\sigma_{M_{\rm P}}$, $R_{\rm P}$, $\sigma_{R_{\rm P}}$, $P_{\rm orb}$, $\sigma_{P_{\rm orb}}$, $F_{\rm irr}$, $\sigma_{F_{\rm irr}}$ in the units above. Without it, `my_own` uses `1 0.1 1.5 0.1 1 1e-6 1e9 1e8`. |
| **Uncertainties** | |
| `--add_systematic_error` | Add a factor-of-2 systematic error (×½ below, ×2 above) in quadrature to the interpolated estimates, a rough estimate of the uncertainty of the interior structure and dynamo model assumptions. By default, the uncertainty only comes from the propagation of the observational errors. |
| `--seed` | Seed of the random numbers used for the Monte Carlo uncertainties and fits. The same seed gives exactly the same results. *(default: 42)* |
| `--population_file` | NASA Exoplanet Archive table (CSV) of the gas giants with complete stellar data, used for the population predictions and the fits. A path is relative to the folder you run from. *(default: `PS_2026.02.18_08.49.05.csv`)* |
| `--extra_population_file` | Archive table (CSV) of all gas giants, including those with incomplete stellar data, used for the `_extra` outputs. *(default: `PS_2026.02.18_07.09.03.csv`)* |

---

## Vocabulary

These names are used in the code, the options and the output files.

- **Heating efficiency $\epsilon$** (`eps`, `heat_eff`): fraction of the incident flux $F_{\rm irr}$ deposited in the interior, $Q_{\rm dep} = \epsilon\, \pi R_{\rm P}^2 F_{\rm irr}$, needed to reproduce the observed radius. It is a fraction inside the code (and in `hj_model.py`) and a percentage in the output files (`eps(%)`).
- **Inner and outer heating** (`inner`/`outer`, `up`/`low` in the output columns): the heat is deposited inside the dynamo region (deep heating) or above it (shallow heating). Deep heating gives the strong-field (upper) estimate and shallow heating the weak-field (lower) estimate.
- **Non-inflated model** (`irr`, `R_irr`): the model with only irradiation, $\epsilon = 0$, with the same mass and flux. Only planets larger than it get heating and inflated magnetic field estimates.
- **$B_{\rm rms,dyn}$** (`rms,dyn`, `bdyn`, `brms`): rms magnetic field in the dynamo region, from the scaling law of Christensen et al. (2009).
- **$B_{\rm dip,surf}$** (`dip,surf`, `bdipsurf`, `_surf`): its dipolar component extrapolated to the planetary surface.
- **Rotation correction** (default; `--no_rotation` to switch it off): the convective velocities and fields include the planet's rotation, assuming tidal locking (rotation period = orbital period), so the tables have an extra period axis. The tables without rotation are scaled by `b_factor` = 1/3 to match Jupiter's $B_{\rm dip,surf}$.
- **Scaling-law fits**: `MRL` ($B$ as a power law of $M$, $R$ and the deposited power $L_{\rm P} = \epsilon \pi R^2 F_{\rm irr}$), `MRF` ($M$, $R$ and $F_{\rm irr}$) and `MR` ($M$ and $R$ only); see [Output files](#output-files) for the equations.
- **Fit methods** (`--fit_type`): `simple` (least squares on the values), `odr` (orthogonal distance regression with the errors) and `gaussian` (Monte Carlo: the population is drawn from its errors many times and refitted; the default).
- **Radius cut** (`cut_R`, the default `--which_cut`): the fits only use the clearly inflated planets, $R_{\rm P} > 1.05\, R_{\epsilon=0}$.
- **R09, Th18**: the scaling law of Reiners et al. (2009) with fixed exponents, and the empirical heating efficiency of Thorngren & Fortney (2018).
- **Extra population** (`_extra`): all gas giants of the archive, including those with incomplete stellar data, whose missing stellar values come from main-sequence relations. It is not used in the fits.

---

## Output files

Values with uncertainties are written as `value$_{-lower}^{+upper}$` (or `value$\pm$error` for the observed values), ready to be used in LaTeX tables. Dashes mean that the value or its uncertainty is not available, for example when the observed radius is smaller than the radius of a purely irradiated planet, or when an observational uncertainty is missing in the archive.

The two heating cases bracket the magnetic field: heat deposited in the interior ("inner") gives an upper estimate and heat deposited in the outer layers ("outer") a lower estimate. "irr" refers to the purely irradiated models ($\epsilon = 0$).

**`predictions/predictions_<sample>.txt`** (one line per planet). For the population, `<sample>` is `hot_jupiter_populations` (planets with complete data) or `hot_jupiter_populations_extra` (all gas giants in the archive table, including those with incomplete stellar data, whose missing stellar radius and temperature are taken from main-sequence relations; not used in the fits); otherwise it is the `--which_planets` sample. The suffixes `_syst_err` and `_no_rot` mark runs with `--add_systematic_error` and `--no_rotation`.

| Column | Meaning |
|---|---|
| `M(MJ)`, `R(RJ)`, `F(erg cm-2 s-1)`, `P(days)` | observed mass, radius, irradiation flux and orbital period |
| `R_irr(RJ)` | radius of the purely irradiated planet ($\epsilon = 0$) |
| `eps_up(%)`, `eps_low(%)` | heating efficiency needed for the observed radius, inner and outer heating |
| `eps_th18(%)` | heating efficiency from the empirical relation of Thorngren & Fortney (2018) |
| `bdyn_up(G)`, `bdyn_irr(G)`, `bdyn_low(G)` | $B_{\rm rms,dyn}$ for inner heating, no heating and outer heating |
| `bdipsurf_up(G)`, `bdipsurf_irr(G)`, `bdipsurf_low(G)` | $B_{\rm dip,surf}$ for the same three cases |

**`predictions/predictions_cut_R_<fit_type>_fit[_<sample>]/<planet>.txt`** (one file per planet). Comparison of the interpolated magnetic fields with the scaling-law predictions, for inner heating, outer heating and no heating:

- `Interpolator`: interpolated value;
- `FitMRL`, `FitMRF`, `FitMR`: the scaling laws fitted in this work (see below);
- `R09`, `R09+Th18`: the scaling law of Reiners et al. (2009) with fixed exponents, using the interpolated heating efficiency or the Thorngren & Fortney (2018) one.

The folder `predictions_cut_R_extra/` contains the same files for the `_extra` population, for every planet with $R_{\rm P}/R_{\epsilon=0} > 0.75$ (a looser cut than the $R_{\rm P} > 1.05\,R_{\epsilon=0}$ used for the fits).

Files and folders ending in `_no_rot` contain the same outputs without the rotation correction (`python main.py --no_rotation`). They are not used in the paper and are included for transparency.

**`plots/fits_cut_R_<fit_type>_fit/`**: the fitted scaling laws for the magnetic field,

- `fit_general*` (MRL): $B = K\, M^\alpha R^\beta (\epsilon \pi R^2 F_{\rm irr} / 4.606\times10^{24}\,{\rm erg\,s^{-1}})^\gamma$;
- `fit_no_eps*` (MRF): $B = K\, M^\alpha R^\beta (F_{\rm irr}/10^9\,{\rm erg\,cm^{-2}\,s^{-1}})^\gamma$;
- `fit_no_flux*` (MR): $B = K\, M^\alpha R^\beta$;
- `fit_irr*`: MR fits for the planets that do not pass the cut (`fit_irr_brms_inner*`, `fit_irr_brms_outer*`) and for the purely irradiated estimates of all planets (`fit_irr_brms_irr*`);

with $M$ in $M_{\rm J}$, $R$ in $R_{\rm J}$ and $B$ in G. The files `fit_<name>.txt` give a readable summary of each group of fits. In the files `fit_<name>_<field>.txt` the first row contains the parameters ($\alpha$, $\beta$, $\gamma$, $K$) and the next two rows their lower and upper uncertainties; `*_cov_matrix.txt` contains the covariance matrices. These files are read by the `--no_HJ` runs.

**`plots/fits_eps_cut_R_<fit_type>_fit/`**: the same kind of fit for the heating efficiency, $\epsilon = K\, M^\alpha R^\beta (F_{\rm irr}/10^9)^\gamma$, and the related figures.

**`plots/general_predictions*/`**: figures of the population (with `--save_plot_population`).

---

## Model assumptions

The interpolator assumes

- one-dimensional planetary evolution models;
- adiabatic interiors;
- standard H-He equation of state;
- rapidly rotating planetary dynamos.

The results should therefore be regarded as order-of-magnitude magnetic field estimates.

---

## Citation

If you use this repository, please cite

> Elias-López et al. (2026), *Estimations for strong and weak hot Jupiter magnetism*.

---

## License

This code is released under the MIT License (see `LICENSE`).
