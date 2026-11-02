# Hot Jupiter Magnetic Field Interpolator

Python tools to estimate the magnetic field strength of giant exoplanets, particularly hot and warm Jupiters, from their observed mass, radius, incident stellar flux and orbital period (Elias-López et al. 2026).

The estimates come from a precomputed grid of MESA planetary evolution models combined with the dynamo scaling law of Christensen et al. (2009). For each planet, the interpolator gives:

- the heating efficiency $\epsilon$ needed to reproduce the observed radius;
- the rms magnetic field in the dynamo region, $B_{\rm rms,dyn}$;
- the dipolar magnetic field at the planetary surface, $B_{\rm dip,surf}$.

Each quantity is given for two limiting cases: heat deposited deep in the interior (strong field) or in the outer layers (weak field). As a quick alternative, the repository also provides analytical scaling laws fitted to the interpolated population, which only need the planetary mass, radius and (optionally) flux.

## Contents

- [Quick start](#quick-start)
- [Look up the saved estimates](#look-up-the-saved-estimates)
- [Using the interpolator from Python](#using-the-interpolator-from-python)
- [Inputs and grid limits](#inputs-and-grid-limits)
- [Installation](#installation)
- [Reproducing the paper](#reproducing-the-paper)
- [Command-line options](#command-line-options)
- [Output files](#output-files)
- [Vocabulary](#vocabulary)
- [Model assumptions](#model-assumptions)
- [Repository layout](#repository-layout)
- [Citation](#citation) and [License](#license)

---

## Quick start

```
git clone https://github.com/albert-elias-lopez/hj_magnetic_field_interpolator.git
cd hj_magnetic_field_interpolator
python3 set_up.py              # only once: creates the environment in .venv (needs Python 3.11 or 3.12)
source .venv/bin/activate      # in every new terminal (Windows: .venv\Scripts\activate)
python example.py              # estimates for WASP-12 b
```

To estimate your own planet, replace the values at the top of `example.py` and run it again.

| I want to... | Do this |
|---|---|
| look up a known planet | open `predictions/hot_jupiters.txt`, no installation needed ([details](#look-up-the-saved-estimates)) |
| estimate my own planet | edit the top of `example.py` and run `python example.py` |
| use the estimates in my own scripts | `from hj_model import HJMagneticModel` ([details](#using-the-interpolator-from-python)) |
| reproduce the results of the paper | run the `main.py` commands of [Reproducing the paper](#reproducing-the-paper) |

---

## Look up the saved estimates

The estimates of the paper are already in the folder `predictions/`, so you can use them without installing anything. Each table `<name>.txt` has one line per planet; the folder `<name>/` next to it has one file per planet, with the interpolated fields next to all the scaling laws. Planet file names replace spaces by `_` (e.g. `HD_189733_b.txt`).

| File or folder in `predictions/` | Contents |
|---|---|
| `hot_jupiters.txt` | the hot Jupiter population (planets with complete stellar data), with the planet name in the last column |
| `hot_jupiters/` | one file per inflated planet used in the fits ($R_{\rm P} > 1.05\,R_{\epsilon=0}$) |
| `hot_jupiters_extra.txt`, `hot_jupiters_extra/` | all gas giants of the archive, also those with incomplete stellar data; the folder has every planet with $R_{\rm P} > 0.75\,R_{\epsilon=0}$ |
| `seidel26`, `cauley19`, `wasp12b`, `hd209458b`, `hats53b` | the samples of the literature comparison (table and folder) |
| names ending in `_syst_err` | with a factor-of-2 systematic uncertainty, as in the literature comparison figure |
| names ending in `_no_rot` or `_simple_fit` | runs with other options, not used in the paper |

For example:

```
grep "HD 189733 b" predictions/hot_jupiters.txt
cat predictions/hot_jupiters/HD_189733_b.txt
```

The columns and the format of the values are explained in [Output files](#output-files).

---

## Using the interpolator from Python

`hj_model.py` gives the same estimates as `main.py`, from your own scripts and without writing files. Run Python from the repository folder, or add the folder to `sys.path`. The quickest start is `example.py`.

```python
from hj_model import HJMagneticModel

model = HJMagneticModel()          # reads the grid tables once (a few seconds); rotation=False for the tables without rotation

# Incident flux from the stellar radius and temperature and the orbital distance (or the period and stellar mass)
flux, flux_err = model.get_flux(radius_star=1.57, radius_star_err=0.07, temp_star=6300, temp_star_err=150,
                                orbital_dist=0.0229, orbital_dist_err=0.0008)

# Interpolated heating efficiencies and magnetic fields, with Monte Carlo errors
result = model.estimate(mass=1.41, mass_err=0.10, radius=1.79, radius_err=0.09,
                        flux=flux, flux_err=flux_err, period=1.0914, period_err=3e-6, names="WASP-12 b")
result.print_all()                                                 # all estimates, as main.py prints them
result.print_all_scaling_law()                                     # the estimates next to all the scaling laws
print(result["brms_inner_surf"], result["brms_inner_surf_err"])   # B_dip,surf (G) for deep heating, (minus, plus) error

# Published scaling laws (MR fit without flux, MRF fit with flux)
b, b_err = model.scaling_law(mass=1.41, mass_err=0.10, radius=1.79, radius_err=0.09, heating="outer")
```

**Inputs**

- **Units:** planet mass in $M_{\rm J}$, planet radius in $R_{\rm J}$, flux in erg cm$^{-2}$ s$^{-1}$, orbital period in days, stellar radius in $R_\odot$, stellar temperature in K, stellar mass in $M_\odot$, orbital distance in AU.
- **Errors:** a number or array for symmetric errors, or a tuple `(lower, upper)` for asymmetric ones. `np.nan` marks a missing error.
- **Several planets at once:** every value can be an array.

**Results of `estimate`**

| Key | Meaning | Unit |
|---|---|---|
| `eps_inner`, `eps_outer` | heating efficiency, for deep and shallow heating | fraction |
| `brms_inner`, `brms_outer` | $B_{\rm rms,dyn}$, for deep and shallow heating | G |
| `brms_inner_surf`, `brms_outer_surf` | $B_{\rm dip,surf}$, for deep and shallow heating | G |
| `brms_irr`, `brms_irr_surf` | $B_{\rm rms,dyn}$ and $B_{\rm dip,surf}$ of the non-inflated model ($\epsilon = 0$, same mass and flux) | G |
| `radius_irr` | radius of the non-inflated model | $R_{\rm J}$ |

Each key has its `(minus, plus)` error under `<key>_err`.

**Printing.** `result.print_all()` prints every estimate for each planet (heating efficiencies in %, fields in G), with the planet names given by `names` (default `planet 0`, `planet 1`, ...). `result.print_all_scaling_law()` prints them next to the scaling-law predictions, as in the per-planet files of `main.py`:

- the fits of this work (`FitMRL`, `FitMRF`, `FitMR`);
- `R09` and `R09+Th18`, the law of Reiners et al. (2009) with the interpolated heating efficiency or that of Thorngren & Fortney (2018), whose heating efficiency is also printed.

Their central values are those of `main.py`; their Monte Carlo errors can differ in the second digit, because `main.py` draws them after its other random numbers. The scaling laws without $\epsilon$ (`FitMRF`, `FitMR`, `R09+Th18`) are also given for planets without interpolated estimates; outside the grid they are extrapolations.

**Scaling laws.** `model.scaling_law(...)` gives the published laws directly: the MR fit (mass and radius), or the MRF fit if you also give `flux` and `flux_err`. Choose `heating="inner"` (strong field), `"outer"` (weak field) or `"none"` (non-inflated planets), and `field="dip,surf"` (default) or `"rms,dyn"`.

**Planets without estimates.** These planets give `nan`:

- outside the grid in mass, flux or period;
- smaller than their non-inflated model (only `radius_irr` is given);
- larger than the most inflated model;
- with a missing value.

`estimate` shows a `GridWarning` that lists each of them with the reason, and `print_all` prints the same reason instead of the estimates. To silence it: `warnings.simplefilter("ignore", hj_model.GridWarning)`.

**Reproducibility.** The Monte Carlo errors use numpy's random numbers with `seed=42` by default, as `main.py`, so the same input always gives the same errors. `python hj_model.py` (or `hj_model.check()`) verifies that the interface reproduces the committed predictions of `main.py` exactly; run it after changing the code. The GitHub Action also runs it.

---

## Inputs and grid limits

For each planet, the interpolator needs these values and their uncertainties:

| Input | Unit | Grid range |
|---|---|---|
| planet mass $M_{\rm P}$ | $M_{\rm J}$ | 0.3 to 12 |
| planet radius $R_{\rm P}$ | $R_{\rm J}$ | between the non-inflated and the most inflated model |
| incident stellar flux $F_{\rm irr}$ | erg cm$^{-2}$ s$^{-1}$ | $10^7$ to $1.58\times10^{10}$ |
| orbital period $P_{\rm orb}$ | days | 0.1 to 100 |

Planets outside the grid are not interpolated.

**Rotation.** By default, the code uses the rotation-corrected magnetic field tables (Fuentes et al. 2023), which assume tidally locked planets. The option `--no_rotation` (or `HJMagneticModel(rotation=False)`) disables the correction, and then the orbital period is not used.

**Population data.** For the hot Jupiter population, the inputs are read from the NASA Exoplanet Archive tables included in the repository (`PS_2026.02.18_*.csv`). Missing stellar temperatures and radii are estimated from a main-sequence relation (`main_sequence_luminosities.txt`), and a missing semi-major axis from Kepler's third law.

---

## Installation

You need Python 3.11 or 3.12 (check with `python3 --version`); the package versions of the paper need one of them. Any Python >= 3.10 also works with the `--latest` option below. LaTeX is optional: if it is installed, it is used for the plot labels; otherwise matplotlib's own math rendering is used.

**1. Download the code and create the environment** (only once):

```
git clone https://github.com/albert-elias-lopez/hj_magnetic_field_interpolator.git
cd hj_magnetic_field_interpolator
python3 set_up.py
```

`set_up.py` creates a virtual environment in the folder `.venv` of the repository, installs the package versions used in the paper (see [Package versions](#package-versions)) and checks that the code runs. With these versions, the default random seed reproduces the results of the paper exactly. It can be run from any folder, e.g. `python3 path/to/set_up.py`.

**2. Activate the environment** (in every new terminal):

```
source .venv/bin/activate        (Windows: .venv\Scripts\activate)
python main.py --help
```

When you are done, `deactivate` leaves the environment. Without activating it, you can also run the code with `.venv/bin/python main.py` (Windows: `.venv\Scripts\python main.py`).

**3. Try it on a planet:** `python example.py` prints the estimates for WASP-12 b. Replace the values at the top of `example.py` with your own planet.

**Common problems**

- **Other Python versions:** `python3 set_up.py --latest` installs the newest package versions instead. The last digits of the results may then differ slightly from the paper.
- **Default Python is not 3.11 or 3.12**, but one of them is installed: run the script with it, e.g. `python3.12 set_up.py`.
- **Creating the environment fails on Debian/Ubuntu:** install the venv module first (`sudo apt install python3.12-venv`, with your Python version) and run the script again.
- **Starting from scratch:** running `set_up.py` again reuses the existing environment; delete the folder `.venv` first.
- **Your own environment:** install the packages listed below.

### Package versions

`set_up.py` installs the package versions used for the results of the paper (Python 3.11 or 3.12). With these versions, the default random seed reproduces the results exactly.

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

## Reproducing the paper

Run these commands in this order. All but the first use the scaling-law fits saved by the first one.

```
python main.py --save_plot_population
python main.py --no_HJ --which_planets seidel26 --add_systematic_error
python main.py --no_HJ --which_planets cauley19 --add_systematic_error
python main.py --no_HJ --which_planets seidel26
python main.py --no_HJ --which_planets cauley19
python main.py --no_HJ --which_planets wasp12b
python main.py --no_HJ --which_planets hd209458b
python main.py --no_HJ --which_planets hats53b
```

- The first command estimates the whole population, fits the scaling laws and saves the figures (about 10 minutes). It uses the defaults `--fit_type gaussian`, `--N_gauss_fits 10000` and `--which_cut "radius cut"`.
- The runs with `--add_systematic_error` include a factor-of-2 systematic uncertainty (×½ below, ×2 above), as in the literature comparison figure. The others only include the observational uncertainties, as in the literature comparison table.
- The Monte Carlo uncertainties and fits use a fixed random seed (`--seed`, default 42), so the same command always gives exactly the same results.
- The code can be launched from any folder; the outputs are always written in the repository folders `predictions/` and `plots/`.

Other examples:

```
python main.py --save_plot_population --fit_type simple
python main.py --no_rotation --no_HJ --which_planets seidel26
python main.py --no_HJ --which_planets my_own --my_planet_params 1.2 0.1 1.5 0.1 1 1e-6 1e9 1e8
```

The last one estimates the magnetic field of your own planet ($M_{\rm P} = 1.2 \pm 0.1\,M_{\rm J}$, $R_{\rm P} = 1.5 \pm 0.1\,R_{\rm J}$, $P_{\rm orb} = 1$ day, $F_{\rm irr} = (1.0 \pm 0.1)\times10^9$ erg cm$^{-2}$ s$^{-1}$) and prints the results in the terminal.

---

## Command-line options

| Option | Description |
|----------|-------------|
| **General** | |
| `--save_plot_grid` | Save the interpolation grid figures to `plots/grid/<mass>_MJ/`, from the tables without rotation (mainly for testing). *(default: False)* |
| `--save_plot_3D_grid` | Save a schematic 3D plot of the grid ($F_{\rm irr}$, $M$, $\epsilon$) to `plots/grid/population_grid*.png`, from the tables without rotation. *(default: False)* |
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
| **Input files** | |
| `--population_file` | NASA Exoplanet Archive table (CSV) of the gas giants with complete stellar data, used for the population predictions and the fits. A path is relative to the folder you run from. *(default: `PS_2026.02.18_08.49.05.csv`)* |
| `--extra_population_file` | Archive table (CSV) of all gas giants, including those with incomplete stellar data, used for the `_extra` outputs. *(default: `PS_2026.02.18_07.09.03.csv`)* |

---

## Output files

### Names

The results of the paper use the default options and their names have no suffix. Runs with other options add a suffix to all their output names, so they never overwrite the default ones:

| Suffix | Run | Used in the paper |
|---|---|---|
| `_syst_err` | `--add_systematic_error` | yes, in the literature comparison figure |
| `_no_rot` | `--no_rotation` | no |
| `_simple_fit` | `--fit_type simple` | no |
| `_cut_epsF`, `_cut_Qdep`, `_cut_B`, `_no_cut` | the other `--which_cut` options | no |

The outputs that are not used in the paper are included for transparency. The folders in `plots/` follow the same rule: `plots/fits/` holds the default fits, `plots/fits_no_rot/` the fits without rotation, and so on.

### Values

Values with uncertainties are written as `value$_{-lower}^{+upper}$` (or `value$\pm$error` for the observed values), ready to be used in LaTeX tables. Dashes mean that the value or its uncertainty is not available, for example when the observed radius is smaller than the radius of a purely irradiated planet, or when an observational uncertainty is missing in the archive.

The two heating cases bracket the magnetic field: heat deposited in the interior ("inner", or "up" in the columns) gives an upper estimate, and heat deposited in the outer layers ("outer", or "low") a lower estimate. "irr" refers to the purely irradiated models ($\epsilon = 0$).

### `predictions/<sample>.txt`

One line per planet. `<sample>` is `hot_jupiters` (planets with complete stellar data), `hot_jupiters_extra` (all gas giants in the archive table, including those with incomplete stellar data, whose missing stellar radius and temperature are taken from main-sequence relations; not used in the fits) or the `--which_planets` sample.

| Column | Meaning |
|---|---|
| `M(MJ)`, `R(RJ)`, `F(erg cm-2 s-1)`, `P(days)` | observed mass, radius, irradiation flux and orbital period |
| `R_irr(RJ)` | radius of the purely irradiated planet ($\epsilon = 0$) |
| `eps_up(%)`, `eps_low(%)` | heating efficiency needed for the observed radius, inner and outer heating |
| `eps_th18(%)` | heating efficiency from the empirical relation of Thorngren & Fortney (2018) |
| `bdyn_up(G)`, `bdyn_irr(G)`, `bdyn_low(G)` | $B_{\rm rms,dyn}$ for inner heating, no heating and outer heating |
| `bdipsurf_up(G)`, `bdipsurf_irr(G)`, `bdipsurf_low(G)` | $B_{\rm dip,surf}$ for the same three cases |

### `predictions/<sample>/<planet>.txt`

One file per planet, comparing the interpolated magnetic fields with the scaling-law predictions, for inner heating, outer heating and no heating:

- `Interpolator`: interpolated value;
- `FitMRL`, `FitMRF`, `FitMR`: the scaling laws fitted in this work (see below);
- `R09`, `R09+Th18`: the scaling law of Reiners et al. (2009) with fixed exponents, using the interpolated heating efficiency or the Thorngren & Fortney (2018) one.

For the population, `hot_jupiters/` has the planets used in the fits ($R_{\rm P} > 1.05\,R_{\epsilon=0}$), and `hot_jupiters_extra/` every planet of the `_extra` population with $R_{\rm P} > 0.75\,R_{\epsilon=0}$ (a looser cut).

### `plots/fits/`

The fitted scaling laws for the magnetic field, with $M$ in $M_{\rm J}$, $R$ in $R_{\rm J}$ and $B$ in G:

| Files | Fit | Law |
|---|---|---|
| `fit_general*` | MRL | $B = K\, M^\alpha R^\beta (\epsilon \pi R^2 F_{\rm irr} / 4.606\times10^{24}\,{\rm erg\,s^{-1}})^\gamma$ |
| `fit_no_eps*` | MRF | $B = K\, M^\alpha R^\beta (F_{\rm irr}/10^9\,{\rm erg\,cm^{-2}\,s^{-1}})^\gamma$ |
| `fit_no_flux*` | MR | $B = K\, M^\alpha R^\beta$ |
| `fit_irr*` | MR | for the planets that do not pass the cut (`fit_irr_brms_inner*`, `fit_irr_brms_outer*`) and for the purely irradiated estimates of all planets (`fit_irr_brms_irr*`) |

- `fit_<name>.txt`: a readable summary of each group of fits.
- `fit_<name>_<field>.txt`: the parameters ($\alpha$, $\beta$, $\gamma$, $K$) in the first row, and their lower and upper uncertainties in the next two rows.
- `*_cov_matrix.txt`: the covariance matrices.

These files are read by the `--no_HJ` runs and by `hj_model.py`.

### Other folders in `plots/`

| Folder | Contents |
|---|---|
| `plots/fits_eps/` | the same kind of fit for the heating efficiency, $\epsilon = K\, M^\alpha R^\beta (F_{\rm irr}/10^9)^\gamma$, and the related figures |
| `plots/population/`, `plots/population_comparison/` | figures of the population (`--save_plot_population`) |
| `plots/grid/` | figures of the model grid: one folder per mass (`--save_plot_grid`) and the 3D plots `population_grid*.png` (`--save_plot_3D_grid`) |

---

## Vocabulary

These names are used in the code, the options and the output files.

- **Heating efficiency $\epsilon$** (`eps`, `heat_eff`): fraction of the incident flux $F_{\rm irr}$ deposited in the interior, $Q_{\rm dep} = \epsilon\, \pi R_{\rm P}^2 F_{\rm irr}$, needed to reproduce the observed radius. It is a fraction inside the code (and in `hj_model.py`) and a percentage in the output files (`eps(%)`).
- **Inner and outer heating** (`inner`/`outer`, `up`/`low` in the output columns): the heat is deposited inside the dynamo region (deep heating) or above it (shallow heating). Deep heating gives the strong-field (upper) estimate and shallow heating the weak-field (lower) estimate.
- **Non-inflated model** (`irr`, `R_irr`): the model with only irradiation, $\epsilon = 0$, with the same mass and flux. Only planets larger than it get heating and inflated magnetic field estimates.
- **Values and errors**: the interpolated quantities are the median of Monte Carlo samples of the observed mass, radius, flux (and period), with the 16th and 84th percentiles as lower and upper errors. Samples smaller than their non-inflated radius count as $\epsilon = 0$ and take the fields of the non-inflated model. Without observational errors, the value is interpolated at the observed values and has no error. The scaling-law predictions are the law at the observed values, with Monte Carlo errors.
- **$B_{\rm rms,dyn}$** (`rms,dyn`, `bdyn`, `brms`): rms magnetic field in the dynamo region, from the scaling law of Christensen et al. (2009).
- **$B_{\rm dip,surf}$** (`dip,surf`, `bdipsurf`, `_surf`): its dipolar component extrapolated to the planetary surface.
- **Rotation correction** (default; `--no_rotation` to switch it off): the convective velocities and fields include the planet's rotation, assuming tidal locking (rotation period = orbital period), so the tables have an extra period axis. The tables with rotation place the top of the dynamo at $r({\rm Rm} = 50)$ for inner heating and at $r(P = 1\,{\rm Mbar})$ for outer heating. The tables without rotation take $B_{\rm rms,dyn}$ and the dynamo radius from the MESA history, with the top of the dynamo at $r(P = 1\,{\rm Mbar})$ for both heating types, and are scaled by `b_factor` = 1/3 to match Jupiter's $B_{\rm dip,surf}$.
- **Scaling-law fits**: `MRL` ($B$ as a power law of $M$, $R$ and the deposited power $L_{\rm P} = \epsilon \pi R^2 F_{\rm irr}$), `MRF` ($M$, $R$ and $F_{\rm irr}$) and `MR` ($M$ and $R$ only); see [Output files](#output-files) for the equations.
- **Fit methods** (`--fit_type`): `simple` (least squares on the values), `odr` (orthogonal distance regression with the errors) and `gaussian` (Monte Carlo: the population is drawn from its errors many times and refitted; the default).
- **Radius cut** (the default `--which_cut`, `cut_R` in the figure names): the fits only use the clearly inflated planets, $R_{\rm P} > 1.05\, R_{\epsilon=0}$.
- **R09, Th18**: the scaling law of Reiners et al. (2009) with fixed exponents, and the empirical heating efficiency of Thorngren & Fortney (2018).
- **Extra population** (`_extra`): all gas giants of the archive, including those with incomplete stellar data, whose missing stellar values come from main-sequence relations. It is not used in the fits.

---

## Model assumptions

The interpolator assumes

- one-dimensional planetary evolution models;
- adiabatic interiors;
- standard H-He equation of state;
- rapidly rotating planetary dynamos.

The results should therefore be regarded as order-of-magnitude magnetic field estimates.

---

## Repository layout

```
hj_magnetic_field_interpolator/
├── example.py                start here: estimates for one planet
├── hj_model.py               Python interface (HJMagneticModel)
├── main.py                   command-line program: population, scaling-law fits and figures
├── set_up.py                 creates the Python environment
├── predictions/              saved estimates (see Output files)
├── plots/                    scaling-law fits and figures
├── tables/                   magnetic field tables of the model grid, one folder per mass (<mass>_MJ/)
├── PS_2026.02.18_*.csv       NASA Exoplanet Archive tables of the population
└── other .py files           interpolation, scaling-law fits, figures, table reading and generation
```

**Where the tables come from.** The planetary evolution calculations are **not** included here. They were performed with the MESA stellar evolution code and are available in the repository **https://github.com/csic-ice-imagine/MESA_rossby_HJ**. The magnetic field tables in `tables/` were generated from those runs with `table_generator.py` (`python table_generator.py --path_sims <MESA runs folder>` for the tables with rotation, adding `--no_rotation` for the tables without it). The script needs the MESA output and the module `radial_functions`, which are not included here but can be provided upon request.

---

## Citation

If you use this repository, please cite

> Elias-López et al. (2026), *Estimations for strong and weak hot Jupiter magnetism*.

---

## License

This code is released under the MIT License (see `LICENSE`).
