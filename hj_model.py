"""
Python interface to the hot Jupiter magnetic field interpolator (Elias-López et al. 2026).

    from hj_model import HJMagneticModel

    model = HJMagneticModel()      # reads the grid tables once (rotation=False: tables without the rotation correction)
    flux, flux_err = model.get_flux(radius_star=1.57, radius_star_err=0.07, temp_star=6300, temp_star_err=150,
                                    orbital_dist=0.0229, orbital_dist_err=0.0008)
    result = model.estimate(mass=1.41, mass_err=0.10, radius=1.79, radius_err=0.09,
                            flux=flux, flux_err=flux_err, period=1.0914, period_err=3e-6)
    result["brms_inner"], result["brms_inner_err"]     # B_rms,dyn (G) for deep heating, (minus, plus) error

Units: planet mass in M_J, planet radius in R_J, incident flux in erg cm^-2 s^-1, orbital period in
days, stellar radius in R_sun, stellar temperature in K, stellar mass in M_sun, orbital distance in AU.

Every value can be a number or an array (one entry per planet). An error is either symmetric (a number,
or an array like the values) or a tuple (lower, upper) of those. A missing error can be given as
np.nan; the corresponding error bars are then np.nan, as in the predictions of main.py.

python hj_model.py runs check(), which verifies that the interface still reproduces main.py exactly.
"""
import os
import warnings
import numpy as np
import grid
import interpolator
import planet_functions
import table_predictions_save
from scaling_law_fits import fit_object

class GridWarning(UserWarning):
    """Planets that get no estimates (outside the grid, not inflated or missing values)."""

# Shown every time, not only on the first call with the same planet (silence it with
# warnings.simplefilter("ignore", hj_model.GridWarning))
warnings.simplefilter("always", GridWarning)

# Estimates returned by HJMagneticModel.estimate and the planet_object arrays they come from
ESTIMATES = {
    "eps_inner":       "heat_eff_inner",    # heating efficiency (fraction) for heat deposited in the dynamo region
    "eps_outer":       "heat_eff_outer",    # heating efficiency (fraction) for heat deposited above it
    "brms_inner":      "brms_inner",        # B_rms,dyn (G), deep heating (strong field)
    "brms_inner_surf": "brms_inner_surf",   # B_dip,surf (G), deep heating
    "brms_outer":      "brms_outer",        # B_rms,dyn (G), shallow heating (weak field)
    "brms_outer_surf": "brms_outer_surf",   # B_dip,surf (G), shallow heating
    "brms_irr":        "brms_irr",          # B_rms,dyn (G) of the non-inflated (eps = 0) model with the same mass and flux
    "brms_irr_surf":   "brms_irr_surf",     # B_dip,surf (G) of that non-inflated model
    "radius_irr":      "radius_irr",        # radius (R_J) of that non-inflated model
}

def warn_not_interpolated(planets, rotation, scalar):
    # One GridWarning listing the planets without estimates and why
    lines = []
    for index_planet, planet_reasons in enumerate(interpolator.reasons_not_interpolated(planets, grid.limits, rotation)):
        for reason, value in planet_reasons:
            planet = "" if scalar else "planet " + str(index_planet) + ": "
            lines.append("  " + planet + (value + ": " if value else "") + reason)
    if lines:
        warnings.warn("no estimates (nan) for\n" + "\n".join(lines), GridWarning, stacklevel=3)

def split_error(error, n):
    # Symmetric error and (lower, upper) errors, both for n planets
    if isinstance(error, tuple):
        lower, upper = (np.broadcast_to(np.asarray(side, dtype=float), (n,)) for side in error)
    else:
        lower = upper = np.broadcast_to(np.asarray(error, dtype=float), (n,))
    return np.maximum(lower, upper), np.column_stack((lower, upper))

def output(values, scalar):
    # One value (or (minus, plus) pair) for a single planet given as numbers, arrays otherwise
    return values[0] if scalar else values


class HJMagneticModel:

    def __init__(self, rotation=True):
        # Read the grid tables and build the interpolators once; this takes a few seconds
        self.rotation = rotation
        table_inner, table_outer = grid.read_tables(rotation)
        if rotation:
            self.interpolators = [interpolator.build_interpolators(table_inner, table_outer, index_period)
                                  for index_period in range(len(grid.periods))]
        else:
            self.interpolators = interpolator.build_interpolators(table_inner, table_outer)

    def get_flux(self, radius_star, radius_star_err, temp_star, temp_star_err,
                 orbital_dist=None, orbital_dist_err=None,
                 period=None, period_err=None, mass_star=None, mass_star_err=None):
        """
        Incident stellar flux F = sigma_SB T_star^4 (R_star / a)^2 (erg cm^-2 s^-1) and its error.
        Give the orbital distance a (AU), or the orbital period (days) and the stellar mass (M_sun),
        from which a follows from Kepler's third law. Returns (flux, flux_err); flux_err is a
        (lower, upper) tuple if any error is given as one.
        """
        n = np.broadcast(radius_star, temp_star, orbital_dist if orbital_dist is not None else period).size
        scalar = all(np.ndim(value) == 0 for value in (radius_star, temp_star, orbital_dist, period))
        radius_star = np.broadcast_to(np.asarray(radius_star, dtype=float), (n,))
        temp_star = np.broadcast_to(np.asarray(temp_star, dtype=float), (n,))
        _, radius_star_err = split_error(radius_star_err, n)
        _, temp_star_err = split_error(temp_star_err, n)

        if orbital_dist is not None:
            orbital_dist = np.broadcast_to(np.asarray(orbital_dist, dtype=float), (n,))
            _, orbital_dist_err = split_error(orbital_dist_err, n)
        elif period is not None and mass_star is not None:
            period = np.broadcast_to(np.asarray(period, dtype=float), (n,))
            mass_star = np.broadcast_to(np.asarray(mass_star, dtype=float), (n,))
            period_err_sym, _ = split_error(period_err, n)
            mass_star_err_sym, _ = split_error(mass_star_err, n)
            orbital_dist, orbital_dist_err_sym = planet_functions.semi_major_axis(mass_star, period,
                                                                                 mass_star_err_sym, period_err_sym)
            orbital_dist_err = np.column_stack((orbital_dist_err_sym, orbital_dist_err_sym))
        else:
            raise ValueError("get_flux needs orbital_dist, or period and mass_star")

        flux = planet_functions.irradiation_flux(temp_star, radius_star, orbital_dist)
        # F ~ T^4 R_star^2 / a^2: the lower flux error comes from the lower T and R_star errors and the
        # upper a error (and the other way around for the upper flux error)
        flux_err_lower = planet_functions.flux_error(flux, temp_star, radius_star, orbital_dist,
                                                     temp_star_err[:,0], radius_star_err[:,0], orbital_dist_err[:,1])
        flux_err_upper = planet_functions.flux_error(flux, temp_star, radius_star, orbital_dist,
                                                     temp_star_err[:,1], radius_star_err[:,1], orbital_dist_err[:,0])

        # Symmetric errors give one flux error, asymmetric ones a (lower, upper) tuple
        if np.array_equal(flux_err_lower, flux_err_upper, equal_nan=True):
            return output(flux, scalar), output(flux_err_lower, scalar)
        return output(flux, scalar), (output(flux_err_lower, scalar), output(flux_err_upper, scalar))

    def estimate(self, mass, mass_err, radius, radius_err, flux, flux_err, period=None, period_err=0.0, seed=42):
        """
        Interpolated heating efficiencies and magnetic fields of the planets, with Monte Carlo errors
        (16th and 84th percentiles). Returns a dictionary with the keys of ESTIMATES, each with its
        value and, under <key>_err, its (minus, plus) error. Planets outside the grid or smaller than
        the non-inflated model get np.nan, with a GridWarning that says why. The period is needed for
        the tables with rotation.

        seed: seed of numpy's random numbers for the Monte Carlo errors (None to leave them as they
        are). With the default 42, the first planets give exactly the values of main.py.
        """
        if self.rotation and period is None:
            raise ValueError("estimate needs the orbital period with the rotation correction (or use rotation=False)")
        scalar = all(np.ndim(value) == 0 for value in (mass, radius, flux, period))
        n = np.broadcast(mass, radius, flux, period if period is not None else 0).size

        planets = planet_functions.planet_object(n, error_sym=False)
        planets.planet_names = ["planet " + str(index) for index in range(n)]
        planets.mass_obs = np.broadcast_to(np.asarray(mass, dtype=float), (n,)).copy()
        planets.radius_obs = np.broadcast_to(np.asarray(radius, dtype=float), (n,)).copy()
        planets.flux_obs = np.broadcast_to(np.asarray(flux, dtype=float), (n,)).copy()
        planets.mass_obs_err, planets.mass_obs_err_asym = split_error(mass_err, n)
        planets.radius_obs_err, planets.radius_obs_err_asym = split_error(radius_err, n)
        planets.flux_obs_err, planets.flux_obs_err_asym = split_error(flux_err, n)
        if period is not None:
            planets.orbital_period_obs = np.broadcast_to(np.asarray(period, dtype=float), (n,)).copy()
            planets.orbital_period_obs_err, _ = split_error(period_err, n)

        # Only the planets inside the grid are interpolated, as in main.py
        min_mass, max_mass, min_flux, max_flux, min_period, max_period = grid.limits
        inside = (planets.mass_obs > min_mass) & (planets.mass_obs < max_mass) & \
                 (planets.flux_obs > min_flux) & (planets.flux_obs < max_flux)
        if self.rotation:
            inside &= (planets.orbital_period_obs > min_period) & (planets.orbital_period_obs < max_period)

        if seed is not None: np.random.seed(seed)
        if self.rotation:
            interpolator.linear_interpolator_rot(None, None, planets, grid.periods, inside,
                                                 interpolators_s=self.interpolators)
        else:
            interpolator.linear_interpolator(None, None, planets, inside,
                                             interpolators=self.interpolators)

        warn_not_interpolated(planets, self.rotation, scalar)

        result = {}
        for key, name in ESTIMATES.items():
            value = np.where(inside, getattr(planets, name + "_est"), np.nan)
            error = np.where(inside[:,None], getattr(planets, name + "_est_err"), np.nan)
            result[key], result[key + "_err"] = output(value, scalar), output(error, scalar)
        return result

    def scaling_law(self, mass, mass_err, radius, radius_err, flux=None, flux_err=None,
                    heating="inner", field="dip,surf", seed=42):
        """
        Magnetic field (G) from the published scaling laws (Table 1 and Eqs. 13-15 of the paper) and
        its (minus, plus) error, which includes the uncertainties of the fit coefficients.
        Without flux the MR fit (mass and radius) is used, with flux the MRF fit (mass, radius, flux).
        heating: "inner" (deep heating, strong field), "outer" (shallow heating, weak field) or
        "none" (non-inflated planets, eps = 0; MR fit only).
        field: "dip,surf" (surface dipole, B_dip,surf) or "rms,dyn" (B_rms,dyn in the dynamo region).
        """
        if heating not in ("inner", "outer", "none") or field not in ("dip,surf", "rms,dyn"):
            raise ValueError('heating must be "inner", "outer" or "none", and field "dip,surf" or "rms,dyn"')
        if heating == "none" and flux is not None:
            raise ValueError('the eps = 0 fit (heating="none") only uses mass and radius')

        if heating == "none": file, header, fit_function_type = "fit_irr", "brms_irr", "MR"
        elif flux is None: file, header, fit_function_type = "fit_no_flux", "brms_" + heating, "MR"
        else: file, header, fit_function_type = "fit_no_eps", "brms_" + heating, "MRF"
        if field == "dip,surf": header += "_surf"

        fit = fit_object("gaussian", fit_function_type)
        suffix = "" if self.rotation else "_no_rot"
        fit.read_fit_data(os.path.join(grid.REPOSITORY, "plots", "fits_cut_R_gaussian_fit" + suffix, file), header)

        scalar = all(np.ndim(value) == 0 for value in (mass, radius, flux))
        n = np.broadcast(mass, radius, flux if flux is not None else 0).size
        mass = np.broadcast_to(np.asarray(mass, dtype=float), (n,))
        radius = np.broadcast_to(np.asarray(radius, dtype=float), (n,))
        _, mass_err = split_error(mass_err, n)
        _, radius_err = split_error(radius_err, n)
        if flux is None:
            # The MR fits are evaluated as in table_predictions_save.table_comparison (flux factor 1)
            flux_factor, flux_factor_err = np.ones(n), np.ones(n)
        else:
            flux_factor = np.broadcast_to(np.asarray(flux, dtype=float), (n,))/1e9
            flux_factor_err = split_error(flux_err, n)[0]/1e9

        if seed is not None: np.random.seed(seed)
        brms = table_predictions_save.scaling_law(fit, mass, radius, flux_factor)
        brms_err = np.array([table_predictions_save.get_scaling_law_err(fit, brms[index], mass[index], radius[index],
                                                                        flux_factor[index], mass_err[index],
                                                                        radius_err[index], flux_factor_err[index])
                             for index in range(n)])
        return output(brms, scalar), output(brms_err, scalar)


def check(samples=("hd209458b", "wasp12b", "seidel26", "cauley19")):
    """
    Checks that the interface reproduces main.py exactly: get_flux must give the fluxes of the archive
    table, and estimate the committed predictions/predictions_<sample>.txt. Prints every difference and
    returns their number (0 if all is fine). Run it after changing the code: python hj_model.py
    """
    model = HJMagneticModel()
    failures = 0

    # get_flux against the archive table read by main.py (planets with a, T_star and R_star given)
    filename = os.path.join(grid.REPOSITORY, "PS_2026.02.18_08.49.05.csv")
    rows = planet_functions.read_archive_csv(filename)
    planets = planet_functions.planet_object(len(rows), False)
    current = os.getcwd()
    os.chdir(grid.REPOSITORY)   # read_file finds the main-sequence table in the repository folder
    try: planets.read_file(filename)
    finally: os.chdir(current)
    given = np.array([row["pl_orbsmax"] != "" and row["st_teff"] != "" and row["st_rad"] != "" for row in rows])
    def errors(column):
        error = np.array([planet_functions.read_err_asym(row, column) for row in rows])[given]
        return error[:,0], error[:,1]
    flux, flux_err = model.get_flux(planets.radius_star_obs[given], errors("st_rad"),
                                    planets.temp_star_obs[given], errors("st_teff"),
                                    orbital_dist=planets.orbital_dist_obs[given], orbital_dist_err=errors("pl_orbsmax"))
    if not (np.array_equal(flux, planets.flux_obs[given]) and
            np.array_equal(np.column_stack(flux_err), planets.flux_obs_err_asym[given], equal_nan=True)):
        failures += 1
        print("get_flux: the fluxes differ from main.py")

    # estimate against the committed predictions, compared as written by main.py
    columns = {"eps_inner": 5, "eps_outer": 6, "brms_inner": 8, "brms_irr": 9, "brms_outer": 10,
               "brms_inner_surf": 11, "brms_irr_surf": 12, "brms_outer_surf": 13}
    for sample in samples:
        planets = planet_functions.planet_object(1, False)
        planets.my_planets(sample)
        result = model.estimate(planets.mass_obs, tuple(planets.mass_obs_err_asym.T),
                                planets.radius_obs, tuple(planets.radius_obs_err_asym.T),
                                planets.flux_obs, tuple(planets.flux_obs_err_asym.T),
                                planets.orbital_period_obs, planets.orbital_period_obs_err)
        with open(os.path.join(grid.REPOSITORY, "predictions", "predictions_" + sample + ".txt")) as f:
            lines = [line.split() for line in f if not line.startswith("#")]
        for index, cells in enumerate(lines):
            for key, column in columns.items():
                if "$_{" not in cells[column]: continue     # planet outside the grid
                value, (minus, plus) = result[key][index], result[key + "_err"][index]
                if key.startswith("eps"):   # written in %
                    mine = f"{100*value:.4f}$_{{-{100*minus:.4f}}}^{{+{100*plus:.4f}}}$"
                else:
                    mine = f"{value:.4e}$_{{-{minus:.1e}}}^{{+{plus:.1e}}}$"
                if mine != cells[column]:
                    failures += 1
                    print(f"estimate, {sample}, planet {index}, {key}: {mine} instead of {cells[column]}")

    print("check: OK" if failures == 0 else f"check: {failures} differences")
    return failures


if __name__ == "__main__":
    import sys
    sys.exit(1 if check() else 0)
