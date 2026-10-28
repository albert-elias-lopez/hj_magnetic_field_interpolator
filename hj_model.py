"""
Python interface to the hot Jupiter magnetic field interpolator (Elias-López et al. 2026).

    from hj_model import HJMagneticModel

    model = HJMagneticModel()      # reads the grid tables once (rotation=False: tables without the rotation correction)
    flux, flux_err = model.get_flux(radius_star=1.57, radius_star_err=0.07, temp_star=6300, temp_star_err=150,
                                    orbital_dist=0.0229, orbital_dist_err=0.0008)
    result = model.estimate(mass=1.41, mass_err=0.10, radius=1.79, radius_err=0.09,
                            flux=flux, flux_err=flux_err, period=1.0914, period_err=3e-6)
    result["brms_inner"], result["brms_inner_err"]     # B_rms,dyn (G) for deep heating, (minus, plus) error
    result.print_all()                                  # all estimates, as main.py prints them
    result.print_all_scaling_law()                      # the estimates next to the scaling laws (MRL, MRF, MR, R09, R09+Th18)

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
from scaling_law_fits import fit_object, reiners09_fit

class GridWarning(UserWarning):
    """Planets that get no estimates (outside the grid, not inflated or missing values)."""

# Shown every time, not only on the first call with the same planet (silence it with
# warnings.simplefilter("ignore", hj_model.GridWarning))
warnings.simplefilter("always", GridWarning)

DASHES = "-" * 70

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

def describe(planet_reasons):
    # The reasons of one planet without estimates as one sentence
    return "; ".join(reason + (" (" + value + ")" if value else "") for reason, value in planet_reasons)

def warn_not_interpolated(names, reasons):
    # One GridWarning with a line for each planet without estimates
    lines = ["  " + name + ": " + describe(planet_reasons)
             for name, planet_reasons in zip(names, reasons) if planet_reasons]
    if lines:
        warnings.warn("no estimates (nan) for\n" + "\n".join(lines), GridWarning, stacklevel=3)

def value_error(value, error, form, error_form=None):
    # A value with its (minus, plus) error, e.g. 6.1236e+01 (-2.5e+00 +1.5e+01); "-" if not available
    if np.isnan(value): return "-"
    error_form = error_form or form
    error = np.broadcast_to(error, (2,))
    return f"{value:{form}} (-{error[0]:{error_form}} +{error[1]:{error_form}})"

def field_row(label, b_rms, b_surf):
    # One row of print_all_scaling_law: B_rms,dyn and B_dip,surf (value, error) in G
    return f"  {label:<16}{value_error(*b_rms, '.4e', '.1e'):<36}{value_error(*b_surf, '.4e', '.1e')}"


class Estimates(dict):
    """
    Result of HJMagneticModel.estimate: a dictionary with the keys of ESTIMATES (and <key>_err), plus
    the input values (mass, radius, flux, period), the planet names and, for the planets without
    estimates, why (reasons). print_all() prints them as main.py does, and print_all_scaling_law()
    compares them with the scaling laws.
    """

    def print_all(self):
        # Inputs and estimates of each planet, as printed by main.py (eps in %, B in G)
        values = {key: np.atleast_1d(value) for key, value in self.items() if not key.endswith("_err")}
        errors = {key: np.atleast_1d(self[key + "_err"]).reshape(-1, 2) for key in values}
        mass, radius, flux, period = self.mass, self.radius, self.flux, self.period

        for index, name in enumerate(self.names):
            print(DASHES)
            period_text = "" if period is None else f"{period[index]:.3f} days, "
            print(f"   {name}: {mass[index]:.3f} MJ, {radius[index]:.3f} RJ, {period_text}"
                  f"{flux[index]:.4e} erg cm-2 s-1")
            print(" ")
            if self.reasons[index]:
                print("  no estimates: " + describe(self.reasons[index]) + ".")
            else:
                for heating in ("inner", "outer"):
                    eps = "eps_" + heating
                    print(f"  {eps}(%):     " + value_error(100*values[eps][index], 100*errors[eps][index], ".4f"))
                    print(f"            B_rms,{heating}      = "
                          + value_error(values["brms_" + heating][index], errors["brms_" + heating][index], ".4e", ".1e") + " G")
                    print(f"            B_dip,surf,{heating} = "
                          + value_error(values["brms_" + heating + "_surf"][index],
                                        errors["brms_" + heating + "_surf"][index], ".4e", ".1e") + " G")
                    print(" ")
            if not np.isnan(values["radius_irr"][index]):
                print("  eps(%) = 0 (radius of only irradiated planet: "
                      + value_error(values["radius_irr"][index], errors["radius_irr"][index], ".3f") + " R_J)")
            if not np.isnan(values["brms_irr"][index]):
                print("            B_rms,eps0       = "
                      + value_error(values["brms_irr"][index], errors["brms_irr"][index], ".4e", ".1e") + " G")
                print("            B_dip,surf,eps0  = "
                      + value_error(values["brms_irr_surf"][index], errors["brms_irr_surf"][index], ".4e", ".1e") + " G")
        print(DASHES)

    def print_all_scaling_law(self, seed=42):
        """
        Prints, for each planet, the interpolated magnetic fields next to the scaling-law predictions,
        as in the files predictions/predictions_cut_R_gaussian_fit*/<planet>.txt of main.py: the fits
        of this work (MRL, MRF and MR, Gaussian fits) and R09 (Reiners et al. 2009) with the interpolated
        heating efficiency or with the one of Thorngren & Fortney 2018 (Th18), for inner heating, outer
        heating and no heating (eps = 0). The errors are Monte Carlo errors that include the uncertainties
        of the fit coefficients; seed as in estimate. Values that cannot be computed are shown as "-".
        """
        values = {key: np.atleast_1d(value) for key, value in self.items() if not key.endswith("_err")}
        errors = {key: np.atleast_1d(self[key + "_err"]).reshape(-1, 2) for key in values}
        if seed is not None: np.random.seed(seed)

        for index, name in enumerate(self.names):
            laws = self.scaling_laws(index)
            def interpolated(key): return values[key][index], errors[key][index]

            print(DASHES)
            period_text = "" if self.period is None else f"{self.period[index]:.3f} days, "
            print(f"   {name}: {self.mass[index]:.3f} MJ, {self.radius[index]:.3f} RJ, {period_text}"
                  f"{self.flux[index]:.4e} erg cm-2 s-1")
            if self.reasons[index]:
                print("  no interpolated estimates: " + describe(self.reasons[index]) + ".")
            print(" ")
            print("  eps(%)")
            for label, (value, error) in (("Inner:", interpolated("eps_inner")), ("Outer:", interpolated("eps_outer")),
                                          ("Th18:", laws["eps_th18"])):
                print(f"  {label:<11}" + value_error(100*value, 100*np.asarray(error), ".4f"))
            for heating, rows in (("inner", ("MRL", "MRF", "MR", "R09", "R09+Th18")), ("outer", ("MRL", "MRF", "MR")),
                                  ("irr", ("MR",))):
                print(" ")
                print(f"  {'':<16}{'B_rms,' + heating + '(G)':<36}B_dip,surf,{heating}(G)")
                print(field_row("Interpolator:", interpolated("brms_" + heating), interpolated("brms_" + heating + "_surf")))
                for law in rows:
                    label = "Fit" + law if law.startswith("M") else law
                    print(field_row(label + ":", laws[law + "_" + heating], laws[law + "_" + heating + "_surf"]))
            radius_irr = value_error(*interpolated("radius_irr"), ".3f")
            print("  Radius of only irradiated planet: " + radius_irr + (" R_J" if radius_irr != "-" else ""))
        print(DASHES)

    def scaling_laws(self, index):
        """
        Scaling-law predictions for planet number index, as computed by main.py: a dictionary
        (see table_predictions_save.scaling_law_predictions) of (value, (minus, plus) error) pairs.
        """
        planets, fits = self.planets, self.model.scaling_law_fits()
        eps = [(np.atleast_1d(self[key])[index], np.mean(np.atleast_1d(self[key + "_err"]).reshape(-1, 2)[index]))
               for key in ("eps_inner", "eps_outer")]
        # Deposited power L_P = eps pi R^2 F_irr for the MRL and R09 laws, as in main.py
        powers = [table_predictions_save.deposited_power(planets.radius_obs[index], planets.radius_obs_err[index],
                                                         planets.flux_obs[index], planets.flux_obs_err[index], *heat_eff)
                  for heat_eff in eps]
        return table_predictions_save.scaling_law_predictions(
            fits, planets.mass_obs[index], planets.mass_obs_err[index], planets.mass_obs_err_asym[index],
            planets.radius_obs[index], planets.radius_obs_err[index], planets.radius_obs_err_asym[index],
            planets.flux_obs[index], planets.flux_obs_err[index], *powers[0], *powers[1])

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
        self.fits = None
        table_inner, table_outer = grid.read_tables(rotation)
        if rotation:
            self.interpolators = [interpolator.build_interpolators(table_inner, table_outer, index_period)
                                  for index_period in range(len(grid.periods))]
        else:
            self.interpolators = interpolator.build_interpolators(table_inner, table_outer)

    def scaling_law_fits(self):
        # The Gaussian fits read by print_all_scaling_law (as in main.py --no_HJ), read once
        if self.fits is None:
            files = {"MRL": ("fit_general", "MRL"), "MRF": ("fit_no_eps", "MRF"), "MR": ("fit_no_flux", "MR")}
            folder = os.path.join(grid.REPOSITORY, "plots", "fits_cut_R_gaussian_fit" + ("" if self.rotation else "_no_rot"))
            self.fits = {"R09": reiners09_fit("gaussian")}
            for name in table_predictions_save.SCALING_LAW_FITS:
                law, field = name.split("_", 1)     # e.g. "MRL", "inner_surf"
                file, fit_function_type = ("fit_irr", "MR") if field.startswith("irr") else files[law]
                self.fits[name] = fit_object("gaussian", fit_function_type)
                self.fits[name].read_fit_data(os.path.join(folder, file), "brms_" + field)
        return self.fits

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

    def estimate(self, mass, mass_err, radius, radius_err, flux, flux_err, period=None, period_err=0.0,
                 names=None, seed=42):
        """
        Interpolated heating efficiencies and magnetic fields of the planets, with Monte Carlo errors
        (16th and 84th percentiles). Returns an Estimates dictionary with the keys of ESTIMATES, each
        with its value and, under <key>_err, its (minus, plus) error; result.print_all() prints them.
        Planets outside the grid or smaller than the non-inflated model get np.nan, with a GridWarning
        that says why. The period is needed for the tables with rotation.

        names: the planet names used by print_all and the warnings (default "planet 0", "planet 1", ...)

        seed: seed of numpy's random numbers for the Monte Carlo errors (None to leave them as they
        are). With the default 42, the first planets give exactly the values of main.py.
        """
        if self.rotation and period is None:
            raise ValueError("estimate needs the orbital period with the rotation correction (or use rotation=False)")
        scalar = all(np.ndim(value) == 0 for value in (mass, radius, flux, period))
        n = np.broadcast(mass, radius, flux, period if period is not None else 0).size

        planets = planet_functions.planet_object(n, error_sym=False)
        if names is None: names = ["planet " + str(index) for index in range(n)] if not scalar else ["planet"]
        planets.planet_names = list(np.broadcast_to(np.asarray(names, dtype=object), (n,)))
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

        reasons = interpolator.reasons_not_interpolated(planets, grid.limits, self.rotation)
        warn_not_interpolated(planets.planet_names, reasons)

        result = Estimates()
        for key, name in ESTIMATES.items():
            value = np.where(inside, getattr(planets, name + "_est"), np.nan)
            error = np.where(inside[:,None], getattr(planets, name + "_est_err"), np.nan)
            result[key], result[key + "_err"] = output(value, scalar), output(error, scalar)
        result.names, result.reasons = planets.planet_names, reasons
        result.model, result.planets = self, planets
        result.mass, result.radius, result.flux = planets.mass_obs, planets.radius_obs, planets.flux_obs
        result.period = planets.orbital_period_obs if period is not None else None
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
        with warnings.catch_warnings():     # the samples include planets without estimates on purpose
            warnings.simplefilter("ignore", GridWarning)
            result = model.estimate(planets.mass_obs, tuple(planets.mass_obs_err_asym.T),
                                    planets.radius_obs, tuple(planets.radius_obs_err_asym.T),
                                    planets.flux_obs, tuple(planets.flux_obs_err_asym.T),
                                    planets.orbital_period_obs, planets.orbital_period_obs_err,
                                    names=planets.planet_names)
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

        # Scaling laws (central values; the Monte Carlo errors depend on the random numbers drawn before)
        rows = {"FitMRL:": "MRL", "FitMRF:": "MRF", "FitMR:": "MR", "R09:": "R09", "R09+Th18:": "R09+Th18"}
        for index, name in enumerate(planets.planet_names):
            filename = os.path.join(grid.REPOSITORY, "predictions", "predictions_cut_R_gaussian_fit_" + sample,
                                    name.replace(" ", "_").replace(".", "_") + ".txt")
            if not os.path.exists(filename): continue
            laws, heating = result.scaling_laws(index), None
            with open(filename) as f:
                for line in f:
                    cells = line.split()
                    if line.startswith(" ") and "B_rms," in line: heating = cells[0][6:-3]   # B_rms,<heating>(G)
                    elif cells and cells[0] in rows and heating is not None:
                        for column, suffix in ((1, ""), (2, "_surf")):
                            value = laws[rows[cells[0]] + '_' + heating + suffix][0]
                            mine = "-" if np.isnan(value) else f"{value:.4e}"
                            written = "-" if cells[column].startswith("-") else cells[column].split("$")[0]
                            if mine != written:
                                failures += 1
                                print(f"scaling_laws, {sample}, {name}, {rows[cells[0]]}_{heating}{suffix}: "
                                      f"{mine} instead of {cells[column]}")

    print("check: OK" if failures == 0 else f"check: {failures} differences")
    return failures


if __name__ == "__main__":
    import sys
    sys.exit(1 if check() else 0)
