"""
Command-line program of the hot Jupiter magnetic field interpolator (Elias-López et al. 2026).

For the archive population (default) or a preset sample (--no_HJ --which_planets <sample>) it
  1. reads the magnetic field tables of the MESA grid (grid.py, table_reader.py),
  2. interpolates the heating efficiency and the magnetic fields of each planet inside the grid,
     with Monte Carlo errors (interpolator.py),
  3. for the population, selects the inflated planets (radius cut) and fits the scaling laws to
     them (scaling_law_fits.py); the preset samples read the saved fits instead,
  4. writes the predictions and the comparison with the scaling laws (table_predictions_save.py)
     and, optionally, the figures (plots.py),
  5. for the population, repeats steps 1, 2 and 4 for the planets with incomplete stellar data.

    python main.py --help      # all options
"""
import os
import argparse
import numpy as np

import plots
import scaling_law_fits
from scaling_law_fits import fit_object
import planet_functions
import table_predictions_save
import interpolator
import grid

def parse_args():

    parser = argparse.ArgumentParser(
    description="Inflated HJ magnetic field estimator:",
    epilog="""
            Examples:
            python main.py --no_rotation
            python main.py --save_plot_population --fit_type simple
            python main.py --no_rotation --no_HJ --which_planets seidel26
            python main.py --fit_type simple
            python main.py --no_HJ --which_planets my_own --my_planet_params 1.2 0.1 1.5 0.1 1 1e-6 1e9 1e8

            To reproduce the results in Elias-López et al 2026:
            python main.py --save_plot_population (by default fit_type=gaussian, N_gauss_fits=10000, which_cut=radius cut)
            python main.py --no_HJ --which_planets seidel26 --add_systematic_error
            python main.py --no_HJ --which_planets cauley19 --add_systematic_error
            """,
    formatter_class=argparse.RawDescriptionHelpFormatter,)
    
    # General options
    parser.add_argument("--save_plot_grid", action="store_true",
                        help="Save the grid plots used for the interpolators, from the tables without rotation (very lengthy, only used for initial testing)")
    parser.add_argument("--save_plot_3D_grid", action="store_true",
                        help="Save a schematic 3D plot of the grid (F_irr, M, eps), from the tables without rotation")
    parser.add_argument("--save_plot_population", action="store_true",
                        help="Save the plots with interpolator and fits results for the HJ population")
    parser.add_argument("--print_planet_output", action="store_true",
                        help="Print the interpolator results for each planet (very long terminal output, only used for testing)")
    
    # By default, the program will use the magnetic field tables that use the rotation factor. 
    # If the flag --no-rotation is used, it will use the simpler tables.    
    parser.set_defaults(rotation=True)
    parser.add_argument("--no_rotation", dest="rotation", action="store_false",
                        help="Disable rotation.")
    
    # Which planets to evaluate. If the flag --no_HJ is not used, the program will evaluate the gas giant exoplanet population. 
    # If used, it will evaluate a predefined set of planets.
    parser.set_defaults(HJ_population=True)
    parser.add_argument("--no_HJ", dest="HJ_population", action="store_false",
                        help="Disable evaluation of HJ population. Instead, a predefined set of planets will be evaluated, and precomputed scaling laws will be used.")
    
    parser.add_argument("--which_planets", type=str, default="cauley19",
                        choices=["hats53b","wasp12b","hd209458b","cauley19","cauley19nasa","seidel26","seidel26nasa","my_own","errors"],
                        help="default='cauley19'. If --no_HJ, choose one option or enter your own planet")
    parser.add_argument("--my_planet_params", type=float, nargs=8, default=None,
                        help="default=None. If --no_HJ and --which_planets='my_own', " \
                        "enter your own planet parameters: mass (in Mjup), mass_err, radius (in Rjup), radius_err, period (in days), period_err, flux (in erg/s cm2), flux_err" \
                        "e.g. --my_planet_params 1.5 0.1 1 0.1 1 1e-6 1e9 1e8")
    
    parser.add_argument("--fit_type", type=str, default="gaussian",
                        choices=["simple", "gaussian", "odr"],
                        help="default='gaussian'. Type of fit to use for the scaling law.")
    parser.add_argument("--N_gauss_fits", type=int, default=10000,
                        help="default=10000. Number of gaussian fits to perform for the scaling law. Only used if fit_type='gaussian'.")
    parser.add_argument("--print_scaling_law", action="store_true",
                        help="Print the scaling law fit results on the terminal (lengthy output, mainly used for testing)")
    parser.add_argument("--which_cut", type=str, default="radius cut",
                        choices=["radius cut", "no cut", "flux cut", "heat cut", "brms cut"],
                        help="default='radius cut'. Which HJ population cut to use for the fits.")

    parser.add_argument("--add_systematic_error", action="store_true",
                        help="Add a factor-of-2 systematic error (x1/2 below, x2 above) to the interpolated estimates. " \
                        "This is a rough estimate of the error associated with the interior structure and dynamo model assumptions. " \
                        "By default, the uncertainty only comes from the propagation of observed data errors.")

    parser.add_argument("--seed", type=int, default=42,
                        help="default=42. Seed of the random numbers used for the Monte Carlo errors and fits. " \
                        "The same seed gives exactly the same results.")

    parser.add_argument("--population_file", type=str, default="PS_2026.02.18_08.49.05.csv",
                        help="default=PS_2026.02.18_08.49.05.csv. NASA Exoplanet Archive table (CSV) of the gas giants " \
                        "with complete stellar data, used for the population predictions and the fits.")
    parser.add_argument("--extra_population_file", type=str, default="PS_2026.02.18_07.09.03.csv",
                        help="default=PS_2026.02.18_07.09.03.csv. Archive table (CSV) of all gas giants, including those " \
                        "with incomplete stellar data, used for the _extra outputs.")

    args = parser.parse_args()
    return args


DASHES = "----------------------------------------------------------------------"

def rotation_suffix(rotation):
    # Outputs of the runs without the rotation correction end in _no_rot
    return "" if rotation else "_no_rot"

# Folder suffix and fit-file header of each --which_cut option
CUT_NAMES = {"no cut":     ("_no_cut",   "No cut (actually eps > 1e-5)"),
             "flux cut":   ("_cut_epsF", "Cut for eps*F_irr > 1e6"),
             "heat cut":   ("_cut_Qdep", "Cut for Q_dep = eps*F_irr*pi*R^2 > 1e6"),
             "radius cut": ("_cut_R",    "Cut for Robs/Rirr > {radius_ratio}"),
             "brms cut":   ("_cut_B",    "Cut for Bout/Bin < 0.25")}

def cut_name(which_cut, radius_ratio = 1.05):
    name_cut, header = CUT_NAMES[which_cut]
    return name_cut, header.format(radius_ratio=radius_ratio)

def cut_mask(planets, which_cut, radius_ratio = 1.05):
    """Planets used for the fits (or saved, for the extra population) after the chosen cut."""
    if which_cut == "no cut":     return planets.heat_eff_inner_est > 1e-5
    if which_cut == "flux cut":   return planets.flux_obs*planets.heat_eff_inner_est > 1e6
    if which_cut == "heat cut":   return planets.flux_obs*planets.heat_eff_inner_est*np.pi*planets.radius_obs**2 > 1e6
    if which_cut == "radius cut": return planets.radius_obs/planets.radius_irr_est > radius_ratio
    if which_cut == "brms cut":   return planets.brms_outer_est/planets.brms_inner_est < 0.25

def read_population(filename, error_sym):
    print("Reading gas giant exoplanet population " + filename + ": ", end="\r")
    planets = planet_functions.planet_object(len(planet_functions.read_archive_csv(filename)), error_sym)
    planets.read_file(filename)
    print("Reading gas giant exoplanet population " + filename + ": done")
    print(DASHES)
    return planets

def interpolate_planets(planets, table_inner, table_outer, periods, grid_limits, rotation, error_sym):
    """
    Interpolates the planets inside the grid (mass, flux and, with rotation, period within grid.limits)
    and stores the estimates in planets. Returns the mask of the interpolated planets.
    """
    min_mass, max_mass, min_flux, max_flux, min_period, max_period = grid_limits
    mask_predictions = (planets.mass_obs > min_mass) & (planets.mass_obs < max_mass) & \
                        (planets.flux_obs > min_flux) & (planets.flux_obs < max_flux)
    if rotation:
        mask_predictions &= (planets.orbital_period_obs > min_period) & (planets.orbital_period_obs < max_period)

    print("Interpolating exoplanets within simulation grid: ")
    if rotation:
        interpolator.linear_interpolator_rot(table_inner, table_outer, planets, periods,
                                             mask_predictions, error_sym=error_sym)
    else:
        interpolator.linear_interpolator(table_inner, table_outer, planets,
                                         mask_predictions, error_sym=error_sym)
    print("done")
    print_not_interpolated(planets, grid_limits, rotation)
    print(DASHES)
    return mask_predictions

def print_not_interpolated(planets, grid_limits, rotation):
    # Number of planets without estimates for each reason (a planet can have several)
    reasons = interpolator.reasons_not_interpolated(planets, grid_limits, rotation)
    n_without = sum(1 for planet_reasons in reasons if planet_reasons)
    if n_without == 0: return
    print(f"{n_without} of {planets.n_planets} planets have no heating efficiency or magnetic field estimates:")
    counts = {}
    for planet_reasons in reasons:
        for reason, _ in planet_reasons: counts[reason] = counts.get(reason, 0) + 1
    for reason, count in counts.items():
        print(f"{count:>6}  {reason}")

# The scaling-law fits, in the order they are made (the order matters: the Gaussian fits draw from
# the same random number generator) and passed to table_comparison and plot_population_fits.
# Each group is saved in plots/<folder><cut>/<file>.txt and has the entries
# (header, dyn_or_surf, heating_type, sample): sample "cut" is the fitted sample, "rest" the
# planets removed by the cut, and "all" every planet.
B_ENTRIES = [("brms_inner",      "rms,dyn",  "inner", "cut"),
             ("brms_inner_surf", "dip,surf", "inner", "cut"),
             ("brms_outer",      "rms,dyn",  "outer", "cut"),
             ("brms_outer_surf", "dip,surf", "outer", "cut")]
FIT_GROUPS = [
    # (folder, file, printed title, fit function, initial guess)
    (("fits", "fit_general", "Brms = K * M^alpha * R^beta * (eps*pi*R^2*F_irr)^gamma", "MRL", "MRL"), B_ENTRIES),
    (("fits", "fit_no_eps",  "Brms = K * M^alpha * R^beta * (F_irr)^gamma",            "MRF", "MRL"), B_ENTRIES),
    (("fits", "fit_no_flux", "Brms = K * M^alpha * R^beta",                            "MR",  "MR"),  B_ENTRIES),
    (("fits", "fit_irr",     "Brms, irr = K * M^alpha * R^beta",                       "MR",  "MR"),
        [(header, dyn_or_surf, heating_type, "rest") for header, dyn_or_surf, heating_type, _ in B_ENTRIES] +
        [("brms_irr", "rms,dyn", "irr", "all"), ("brms_irr_surf", "dip,surf", "irr", "all")]),
    (("fits_eps", "fit_eps", "eps = K * M^alpha * R^beta * F_irr^gamma",                "MRF", "MRL"),
        [("eps_inner", "eps", "inner", "cut"), ("eps_outer", "eps", "outer", "cut")]),
]

def fit_filename(folder, file, name_cut, rotation):
    return "plots/" + folder + name_cut + rotation_suffix(rotation) + "/" + file

def make_fits(planets, valid_mask, name_cut, name_cut_header, rotation, initial_guesses, fit_options):
    """
    Fits every scaling law of FIT_GROUPS to the planets of valid_mask (the sample of each entry), saves
    them in plots/<folder><cut>[_no_rot]/<file>*.txt and returns the fit_objects in the FIT_GROUPS order.
    """
    samples = {"cut": valid_mask, "rest": ~valid_mask, "all": []}
    fits = []
    for (folder, file, title, fit_function_type, guess), entries in FIT_GROUPS:
        filename = fit_filename(folder, file, name_cut, rotation)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename + ".txt", 'w') as f:
            f.writelines(name_cut_header + '\n')
        print(DASHES)
        print("   Fits for " + title)
        p0, bounds = initial_guesses[guess]
        group_fits = []
        for header, dyn_or_surf, heating_type, sample in entries:
            print(header + ":", end=" ")
            group_fits.append(scaling_law_fits.general_fit(planets, samples[sample],
                                                    dyn_or_surf = dyn_or_surf, heating_type = heating_type,
                                                    p0 = p0, bounds = bounds,
                                                    fit_function_type = fit_function_type, **fit_options))
        print("")
        with open(filename + ".txt", 'a') as f:
            for (header, *_), fit in zip(entries, group_fits):
                fit.print_fit_results(f, filename, header)
        fits += group_fits
    return fits

def read_fits(fit_type, name_cut, rotation):
    """Reads the fits saved by a population run, in the order of FIT_GROUPS (used by the --no_HJ runs)."""
    fits = []
    for (folder, file, title, fit_function_type, guess), entries in FIT_GROUPS:
        filename = fit_filename(folder, file, name_cut, rotation)
        for header, *_ in entries:
            fit = fit_object(fit_type, fit_function_type)
            fit.read_fit_data(filename, header)
            fits.append(fit)
    return fits


def main(args):
    """Runs the steps listed at the top of this file with the options of parse_args."""
    # A file given on the command line is relative to the folder main.py is launched from (the
    # defaults are in the repository folder), so resolve it before moving to the repository folder
    population_file       = os.path.abspath(args.population_file) if os.path.exists(args.population_file) else args.population_file
    extra_population_file = os.path.abspath(args.extra_population_file) if os.path.exists(args.extra_population_file) else args.extra_population_file

    # Folder of the repository
    current = os.path.dirname(os.path.realpath(__file__))
    # Work from the repository folder, so the relative paths to the tables, the CSV files and the
    # output folders (plots/, predictions/) are valid wherever main.py is launched from
    os.chdir(current)

    # Fix the random numbers of the Monte Carlo errors and fits, so every run with the same seed gives the same results
    np.random.seed(args.seed)

    # --------------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------- Saving and printing ------------------------------------------------
    # --------------------------------------------------------------------------------------------------------------------
    save_plots_grid = args.save_plot_grid
    save_plots_3D_grid = args.save_plot_3D_grid
    save_plots_population = args.save_plot_population
    print_planet_output = args.print_planet_output

    error_sym = False               # To use symmetric errors for the interpolator. Should be kept as False.
    add_systematic_error = args.add_systematic_error

    rotation = args.rotation    

    # If HJ_population = True there should be a file with all the gas giant exoplanet population. Then not all planets should
    # be evaluated, only the ones that can be interpolated by the grid.
    HJ_population = args.HJ_population
    fileHJ = population_file

    # In case of HJ_population = False, choose one option with the appropriate planet number.
    # If you choose your own, you should fill the values in 
    # errors,   hats53b,    cauley19,   cauley19nasa,   seidel26,   seidel26nasa, my_own
    # 11,       1,          4,          4,              7,          7,            1
    which_planets = args.which_planets
    if which_planets == "cauley19" or which_planets == "cauley19nasa": Npl = 4
    elif which_planets == "seidel26" or which_planets == "seidel26nasa": Npl = 7
    elif which_planets == "errors": Npl = 11
    elif which_planets == "hats53b" or which_planets == "wasp12b" or which_planets == "hd209458b" or which_planets == "my_own": Npl = 1
    if not HJ_population: print_planet_output = True  # To print the interpolator results for each planet.

    # Scaling law properties
    print_scaling_law = args.print_scaling_law
    fit_type = args.fit_type  # can be "simple", "odr", "gaussian"
    N_gauss_fits = args.N_gauss_fits  # Should use >1e4 for consistent results
    initial_guess = [4, 1/6, -7/6, 1/3]
    bounds=([1e-5, -np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf, np.inf])
    initial_guess_simple = [4, 1/6, -7/6]
    bounds_simple=([1e-5, -np.inf, -np.inf], [np.inf, np.inf, np.inf])
    # Which HJ population cut to use for the fits. The options are defined later
    which_cut = args.which_cut

    #if rotation: b_factor = 1
    #else: b_factor = 1/3 All factors to match the scaling law with Jupiter's B_dip,surf

    # The grid of models (masses, fluxes, heating efficiencies, periods) and its limits are in grid.py
    masses, fluxes, heat_effs, periods = grid.masses, grid.fluxes, grid.heat_effs, grid.periods

    print(DASHES)
    print("Reading data files: ", end="\r")

    table_inner, table_outer = grid.read_tables(rotation)

    print("Reading data files: done")
    print(f"----------------------------------------------------------------------")

    grid_limits = grid.limits
    initial_guesses = {"MRL": (initial_guess, bounds), "MR": (initial_guess_simple, bounds_simple)}

    if HJ_population:
        planets = read_population(fileHJ, error_sym)

    else:
        planets = planet_functions.planet_object(Npl,error_sym)
        planets.my_planets(which_planets)
        my_planet_params = args.my_planet_params
        if which_planets == "my_own" and my_planet_params is not None:
            my_planet_params = np.array(my_planet_params)
            planets.mass_obs                   = np.array([my_planet_params[0]])
            planets.mass_obs_err               = np.array([my_planet_params[1]])
            planets.radius_obs                 = np.array([my_planet_params[2]])
            planets.radius_obs_err             = np.array([my_planet_params[3]])
            planets.orbital_period_obs         = np.array([my_planet_params[4]])
            planets.orbital_period_obs_err     = np.array([my_planet_params[5]])
            planets.flux_obs                   = np.array([my_planet_params[6]])
            planets.flux_obs_err               = np.array([my_planet_params[7]])
            planets.set_symmetric_asym_errors()

    mask_predictions = interpolate_planets(planets, table_inner, table_outer, periods, grid_limits, rotation, error_sym)

    os.makedirs("./predictions", exist_ok=True)

    print("Saving file ", end="\r")
    table_predictions_save.table_predictions(planets,
                                            mask_predictions,
                                            HJ_population,
                                            which_planets,
                                            add_systematic_error,
                                            error_sym=error_sym,
                                            rotation = rotation)


    if print_planet_output: planets.print_output(error_sym)

    print(DASHES)

    # Decide which mask to use before performing the fits.
    name_cut, name_cut_header = cut_name(which_cut)
    name_cut_short = name_cut
    name_cut = name_cut + "_" + fit_type + "_fit"

    if HJ_population:

        print("Number of planets at each step:")
        print(table_inner.total_number_models, end="      ")
        print(f": total number of models for interpolator (inner)")
        print(table_outer.total_number_models, end="      ")
        print(f": total number of models for interpolator (outer)")

        # Masking the population for the fits. The number of planets at each step is printed.
        number_planets_total = len(planets.planet_names)

        planets.mask_population(mask_predictions)
        number_planets_grid_params = len(planets.planet_names)

        # This commented section prints the stats of the planets that have small radius, outside the grid. Then the code is stoped. 
        # mask_nans = (~np.isnan(planets.heat_eff_inner_est)) & (planets.heat_eff_inner_est >= 1e-5)
        # planets.mask_population(~mask_nans)
        # print(f"{'Planet':<17}{'Mass':>10}{'Radius':>10}{'Period':>12}{'Flux':>15}")
        # for index_planet in range(0,len(planets.planet_names)):
        #     print(f"{planets.planet_names[index_planet]:<15}"
        #         f"{planets.mass_obs[index_planet]:>10.3f} {planets.radius_obs[index_planet]:>10.3f}"
        #         f"{planets.orbital_period_obs[index_planet]:>12.3f} {planets.flux_obs[index_planet]:>15.3e}" )
        # sys.exit()

        mask_nans = (~np.isnan(planets.heat_eff_inner_est)) & (planets.heat_eff_inner_est >= 1e-5)
        planets.mask_population(mask_nans)
        number_planets_mask_nans = len(planets.planet_names)

        mask_ultra_heated_models = planets.heat_eff_inner_est < 0.1
        planets.mask_population(mask_ultra_heated_models)
        number_planets_extreme_heat = len(planets.planet_names)

        valid_mask = cut_mask(planets, which_cut)
        number_planets_after_mask = np.sum(valid_mask)

        print(number_planets_total, end="       ")
        print(f": total number of planets")
        print(number_planets_grid_params, end="       ")
        print(f": planets within grid parameters")
        print(number_planets_mask_nans, end="       ")
        print(f": planets with Brms estimates")
        print(number_planets_extreme_heat, end="       ")
        print(f": planets within eps < 10 %")
        print(number_planets_after_mask, end="       ")
        print(": " + name_cut_header + " (for fits)")
        print(DASHES)

    else:
        # Dummy mask to spit all possible magnetic field estimates. No fits are performed in this case, but the results are printed and saved.
        valid_mask = planets.mass_obs > 0

        number_planets_total = len(planets.planet_names)
        print(f"Number of interpolated planets: {number_planets_total}")
        print(DASHES)

    os.makedirs("./plots", exist_ok=True)
    os.makedirs("./plots/fits" + name_cut + rotation_suffix(rotation), exist_ok=True)

    fit_reiners09           = fit_object(fit_type, "reiners09")
    fit_reiners09.K         = 23
    if fit_type == "gaussian": 
        fit_reiners09.K_err = np.array([15,13])
    else:
        fit_reiners09.K_err = np.mean([15,13])
    fit_reiners09.alpha     = 1/6
    fit_reiners09.alpha_err = 0
    fit_reiners09.beta      = -7/6
    fit_reiners09.beta_err  = 0
    fit_reiners09.gamma     = 1/3
    fit_reiners09.gamma_err = 0

    if HJ_population:
        print("Fitting the different HJ magnetic field estimates:")
        fits = make_fits(planets, valid_mask, name_cut, name_cut_header, rotation, initial_guesses,
                         dict(print_all=print_scaling_law, error_sym=error_sym, fit_type=fit_type, N_gauss=N_gauss_fits))
        print(DASHES)
        print("Fitting the different HJ magnetic field estimates: done")
        print(DASHES)
        os.makedirs("./predictions/predictions" + name_cut + rotation_suffix(rotation), exist_ok=True)

    else:
        fits = read_fits(fit_type, name_cut, rotation)
        os.makedirs("./predictions/predictions" + name_cut + "_" + which_planets + rotation_suffix(rotation), exist_ok=True)

    # fits holds the 18 magnetic field fits (MRL, MRF, MR and MR for the planets removed by the cut and
    # for eps = 0) followed by the 2 eps fits, in the order table_comparison and the plots expect
    fits_B, fits_eps = fits[:-2], fits[-2:]

    print("Saving individual files for each planet:", end=" ")

    table_predictions_save.table_comparison(planets,
                                            valid_mask,
                                            name_cut,
                                            HJ_population,
                                            which_planets,
                                            fit_type, fit_reiners09,
                                            *fits,
                                            add_systematic_error,
                                            error_sym=error_sym,
                                            rotation = rotation)

    print("done")

    print(DASHES)

    # print(np.max(planets.radius_obs), np.min(planets.radius_obs), np.max(planets.mass_obs), np.min(planets.mass_obs), np.max(planets.flux_obs), np.min(planets.flux_obs))
    # Repeat the everything exoplanets with some missing parameters:
    # The grid plots have no period axis, so they show the tables without rotation (same models)
    if save_plots_grid or save_plots_3D_grid:
        if rotation: table_inner_grid, table_outer_grid = grid.read_tables(False)
        else: table_inner_grid, table_outer_grid = table_inner, table_outer

    if save_plots_grid:
        print(f"Saving plots for grid models:", end=" ")
        plots.plot_output_grid(masses,
                                fluxes,
                                heat_effs,
                                table_inner_grid.radius_heat,
                                table_outer_grid.radius_heat,
                                table_inner_grid.bscaling_heat,
                                table_outer_grid.bscaling_heat,
                                table_inner_grid.bscaling_surf_heat,
                                table_outer_grid.bscaling_surf_heat)
        print(f"done")
        print(f"----------------------------------------------------------------------")

    if save_plots_3D_grid:
        print(f"Saving 3D plots as the total grid:", end=" ")
        plots.plot_grid(masses,
                        fluxes,
                        heat_effs,
                        table_inner_grid.radius_heat,
                        table_outer_grid.radius_heat)
        print(f"done")
        print(f"----------------------------------------------------------------------")

    if save_plots_population and HJ_population:

        print(f"Saving plots for HJ population:", end=" ")

        plots.plot_population(planets,
                                error_sym=error_sym,
                                rotation=rotation)

        plots.plot_population(planets,
                              valid_mask=valid_mask,
                                name_cut = name_cut_short, 
                                error_sym=error_sym,
                                rotation=rotation)

        print(f"done")
        print(DASHES)
        print(f"Saving plots for HJ population fits:", end=" ")

        plots.plot_population_fits(planets,
                                name_cut,
                                valid_mask,
                                fit_type, fit_reiners09,
                                *fits_B,
                                rotation=rotation)

        plots.plot_population_fits_eps(planets,
                                        name_cut,
                                        valid_mask,
                                        fit_type, *fits_eps,
                                        rotation=rotation)

        print(f"done")
        print(DASHES)


    # Repeat everything and save the whole gas giant population
    if HJ_population:
        print(f"Repeat process for exoplanets with incomplete stellar data:")
        print(DASHES)

        planets = read_population(extra_population_file, error_sym)
        mask_predictions = interpolate_planets(planets, table_inner, table_outer, periods, grid_limits, rotation, error_sym)

        print("Saving file ", end=" ")
        table_predictions_save.table_predictions(planets,
                                                mask_predictions,
                                                HJ_population,
                                                which_planets,
                                                add_systematic_error,
                                                extra_HJ = True,
                                                error_sym=error_sym,
                                                rotation = rotation)

        planets.mask_population(mask_predictions)    

        # The extra population keeps the planets down to Robs/Rirr > 0.75 for the radius cut
        valid_mask = cut_mask(planets, which_cut, radius_ratio = 0.75)
        name_cut, name_cut_header = cut_name(which_cut, radius_ratio = 0.75)

        os.makedirs("./predictions/predictions" + name_cut + "_extra", exist_ok=True)
        if not rotation:
            os.makedirs("./predictions/predictions" + name_cut + "_extra_no_rot", exist_ok=True)


        print("Saving individual files for each planet:", end=" ")

        table_predictions_save.table_comparison(planets,
                                                valid_mask,
                                                name_cut,
                                                HJ_population,
                                                which_planets,
                                                fit_type, fit_reiners09,
                                                *fits,
                                                add_systematic_error,
                                                extra_HJ=True,
                                                error_sym=error_sym,
                                                rotation = rotation)

        print("done")

    print(f"----------------------------------------------------------------------")
    print(f"----------  Created by: A. Elias-López, Leiden University ------------")
    print(f"----------------------------------------------------------------------")


if __name__ == "__main__":
    main(parse_args())
