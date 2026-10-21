import sys
import os
import argparse
import numpy as np

import plots
import scaling_law_fits
from scaling_law_fits import fit_object
from table_reader import table_object
import planet_functions
import table_predictions_save
import interpolator

if __name__ == "__main__":

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
                        help="Save the grid plots used for the interpolators (very lengthy, only used for initial testing)")
    parser.add_argument("--save_plot_3D_grid", action="store_true",
                        help="Save a schematic 3D grid plot (Firr, M, R)")
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

    args = parser.parse_args()
    
# Name of the current directory
current = os.path.dirname(os.path.realpath(__file__))
# Get the parent directory name
parent = os.path.dirname(current)
# Add the parent directory to the sys.path
sys.path.append(parent)
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
fileHJ = 'PS_2026.02.18_08.49.05.csv'

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

# Grid limits which will be used to mask the exoplanet population before interpolating.
min_mass, max_mass = 0.3, 12
min_flux, max_flux = 1e7, 1.58e10
min_period, max_period = 0.1, 100.0

fluxes = np.array([1.00000000e+07, 1.25892541e+07, 1.58489319e+07, 1.99526231e+07,
                    2.51188643e+07, 3.16227766e+07, 3.98107171e+07, 5.01187234e+07,
                    6.30957344e+07, 7.94328235e+07, 1.00000000e+08, 1.25892541e+08,
                    1.58489319e+08, 1.99526231e+08, 2.51188643e+08, 3.16227766e+08,
                    3.98107171e+08, 5.01187234e+08, 6.30957344e+08, 7.94328235e+08,
                    1.00000000e+09, 1.25892541e+09, 1.58489319e+09, 1.99526231e+09,
                    2.51188643e+09, 3.16227766e+09, 3.98107171e+09, 5.01187234e+09,
                    6.30957344e+09, 7.94328235e+09, 1.00000000e+10, 1.25892541e+10,
                    1.58489319e+10])
n_fluxes    = len(fluxes)

heat_effs = np.array([0.0, 0.00001, 0.00002, 0.00005,
                        0.0001, 0.0002, 0.0005,
                        0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 
                        0.1, 0.2, 0.4, 0.8, 1. , 2. , 4. ])
n_heat_effs = len(heat_effs)

masses = np.array([0.3, 0.4, 0.5, 0.7, 1, 1.5, 2, 3, 4, 6, 8, 12])
n_mass      = len(masses)

periods = np.array([  0.1       ,   0.14384499,   0.20691381,   0.29763514,
                        0.42813324,   0.61584821,   0.88586679,   1.27427499,
                        1.83298071,   2.6366509 ,   3.79269019,   5.45559478,
                        7.8475997 ,  11.28837892,  16.23776739,  23.35721469,
                        33.59818286,  48.32930239,  69.51927962, 100.        ])
n_periods   = len(periods)

if rotation: shape_grid = (n_mass, n_fluxes, n_heat_effs, n_periods)
else: shape_grid = (n_mass, n_fluxes, n_heat_effs)

print(f"----------------------------------------------------------------------")
print("Reading data files: ", end="\r")

table_inner = table_object(shape_grid, "inner")
table_inner.read_table(rotation, masses, fluxes, heat_effs, periods)

table_outer = table_object(shape_grid, "outer")
table_outer.read_table(rotation, masses, fluxes, heat_effs, periods)

print("Reading data files: done")
print(f"----------------------------------------------------------------------")

if HJ_population:
    
    print("Reading gas giant exoplanet population " + fileHJ + ": ", end="\r")
    Npl = len(planet_functions.read_archive_csv(fileHJ))
    planets = planet_functions.planet_object(Npl,error_sym)
    planets.read_file(fileHJ)
    
    print("Reading gas giant exoplanet population " + fileHJ + ": done")
    print(f"----------------------------------------------------------------------")
    
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

if rotation:
    mask_predictions = (planets.mass_obs > min_mass) & (planets.mass_obs < max_mass) & \
                        (planets.flux_obs > min_flux) & (planets.flux_obs < max_flux) & \
                        (planets.orbital_period_obs > min_period) & (planets.orbital_period_obs < max_period)
else:
    mask_predictions = (planets.mass_obs > min_mass) & (planets.mass_obs < max_mass) & \
                        (planets.flux_obs > min_flux) & (planets.flux_obs < max_flux)

print("Interpolating exoplanets within simulation grid: ")
if rotation:
    interpolator.linear_interpolator_rot(table_inner, 
                                         table_outer, 
                                        planets,
                                        periods,
                                        mask_predictions,
                                        error_sym=error_sym)

else:
    interpolator.linear_interpolator(table_inner, table_outer,
                                    planets,
                                    mask_predictions,
                                    error_sym=error_sym)

                                    
print("done")
print(f"----------------------------------------------------------------------")

if not os.path.exists("./predictions"): 
    os.makedirs("./predictions")    

print("Saving file ", end="\r")
table_predictions_save.table_predictions(planets,
                                        mask_predictions,
                                        HJ_population,
                                        which_planets,
                                        add_systematic_error,
                                        error_sym=error_sym,
                                        rotation = rotation)


if print_planet_output: planets.print_output(error_sym)

print(f"----------------------------------------------------------------------")

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

    # Decide which mask to use before performing the fits. 
    if which_cut == "no cut":
        valid_mask = planets.heat_eff_inner_est > 1e-5
        name_cut, name_cut_header = "_no_cut", "No cut (actually eps > 1e-5)"
    elif which_cut == "flux cut":
        valid_mask = planets.flux_obs*planets.heat_eff_inner_est > 1e6
        name_cut, name_cut_header = "_cut_epsF", "Cut for eps*F_irr > 1e6"
    elif which_cut == "heat cut":
        valid_mask = planets.flux_obs*planets.heat_eff_inner_est*np.pi*planets.radius_obs**2 > 1e6
        name_cut, name_cut_header = "_cut_Qdep", "Cut for Q_dep = eps*F_irr*pi*R^2 > 1e6"
    elif which_cut == "radius cut":
        valid_mask = planets.radius_obs/planets.radius_irr_est > 1.05
        name_cut, name_cut_header = "_cut_R", "Cut for Robs/Rirr > 1.05"
    elif which_cut == "brms cut":
        valid_mask = planets.brms_outer_est/planets.brms_inner_est < 0.25
        name_cut, name_cut_header = "_cut_B", "Cut for Bout/Bin < 0.25"

    name_cut_short = name_cut
    name_cut = name_cut + "_" + fit_type + "_fit"

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
    print(f"----------------------------------------------------------------------")

else:

    # Decide which mask to use before performing the fits. 
    if which_cut == "no cut": name_cut, name_cut_header = "_no_cut", "No cut (actually eps > 1e-5)"
    elif which_cut == "flux cut": name_cut, name_cut_header = "_cut_epsF", "Cut for eps*F_irr > 1e6"
    elif which_cut == "heat cut": name_cut, name_cut_header = "_cut_Qdep", "Cut for Q_dep = eps*F_irr*pi*R^2 > 1e6"
    elif which_cut == "radius cut": name_cut, name_cut_header = "_cut_R", "Cut for Robs/Rirr > 1.05"
    elif which_cut == "brms cut": name_cut, name_cut_header = "_cut_B", "Cut for Bout/Bin < 0.25"

    name_cut_short = name_cut
    name_cut = name_cut + "_" + fit_type + "_fit"

    # Dummy mask to spit all possible magnetic field estimates. No fits are performed in this case, but the results are printed and saved.
    valid_mask = planets.mass_obs > 0

    number_planets_total = len(planets.planet_names)
    print(f"Number of interpolated planets: {number_planets_total}")
    print(f"----------------------------------------------------------------------")



if not os.path.exists("./plots"): os.makedirs("./plots")
if not os.path.exists("./predictions"): os.makedirs("./predictions")
if rotation:
    if not os.path.exists("./plots/fits" + name_cut): os.makedirs("./plots/fits" + name_cut)
else:
    if not os.path.exists("./plots/fits" + name_cut + "_no_rot"): os.makedirs("./plots/fits" + name_cut + "_no_rot")


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
    if rotation: filename = "plots/fits" + name_cut + "/fit_general"
    else: filename = "plots/fits" + name_cut + "_no_rot/fit_general"
    with open(filename + ".txt", 'w') as f:
        f.writelines(name_cut_header + '\n')

    print(f"----------------------------------------------------------------------")
    print(f"   Fits for Brms = K * M^alpha * R^beta * (eps*pi*R^2*F_irr)^gamma")
    print(f"brms_inner:", end=" ")
    fit_MRL_inner  = scaling_law_fits.general_fit(planets, valid_mask,
                                                dyn_or_surf = "rms,dyn", heating_type = "inner",
                                                print_all=print_scaling_law, error_sym=error_sym,
                                                p0 = initial_guess, bounds = bounds,
                                                fit_type = fit_type, fit_function_type = "MRL", N_gauss = N_gauss_fits)
    print(f"brms_inner_surf:", end=" ")
    fit_MRL_inner_surf = scaling_law_fits.general_fit(planets, valid_mask,
                                                dyn_or_surf = "dip,surf", heating_type = "inner",
                                                print_all=print_scaling_law, error_sym=error_sym,
                                                p0 = initial_guess, bounds = bounds,
                                                fit_type = fit_type, fit_function_type = "MRL", N_gauss = N_gauss_fits)
    print(f"brms_outer:", end=" ")
    fit_MRL_outer = scaling_law_fits.general_fit(planets, valid_mask,
                                                dyn_or_surf = "rms,dyn", heating_type = "outer",
                                                print_all=print_scaling_law, error_sym=error_sym,
                                                p0 = initial_guess, bounds = bounds,
                                                fit_type = fit_type, fit_function_type = "MRL", N_gauss = N_gauss_fits)
    print(f"brms_outer_surf:", end=" ")
    fit_MRL_outer_surf = scaling_law_fits.general_fit(planets, valid_mask,
                                                dyn_or_surf = "dip,surf", heating_type = "outer",
                                                print_all=print_scaling_law, error_sym=error_sym,
                                                p0 = initial_guess, bounds = bounds,
                                                fit_type = fit_type, fit_function_type = "MRL", N_gauss = N_gauss_fits)
    with open(filename + ".txt",'a') as f:
        fit_MRL_inner.print_fit_results(f, filename, "brms_inner")
        fit_MRL_inner_surf.print_fit_results(f, filename, "brms_inner_surf")
        fit_MRL_outer.print_fit_results(f, filename, "brms_outer")
        fit_MRL_outer_surf.print_fit_results(f, filename, "brms_outer_surf")
        
    if rotation: filename = "plots/fits" + name_cut + "/fit_no_eps"
    else: filename = "plots/fits" + name_cut + "_no_rot/fit_no_eps"
    with open(filename + ".txt", 'w') as f:
        f.writelines(name_cut_header + '\n')
    print(f"")
    print(f"----------------------------------------------------------------------")
    print(f"       Fits for Brms = K * M^alpha * R^beta * (F_irr)^gamma")
    print(f"brms_inner:", end=" ")
    fit_MRF_inner = scaling_law_fits.general_fit(planets, valid_mask,
                                                dyn_or_surf = "rms,dyn", heating_type = "inner",
                                                print_all=print_scaling_law, error_sym=error_sym,
                                                p0 = initial_guess, bounds = bounds,
                                                fit_type = fit_type, fit_function_type = "MRF", N_gauss = N_gauss_fits)
                                                        
    print(f"brms_inner_surf:", end=" ")
    fit_MRF_inner_surf = scaling_law_fits.general_fit(planets, valid_mask,
                                                dyn_or_surf = "dip,surf", heating_type = "inner",
                                                print_all=print_scaling_law, error_sym=error_sym,
                                                p0 = initial_guess, bounds = bounds,
                                                fit_type = fit_type, fit_function_type = "MRF", N_gauss = N_gauss_fits)
    print(f"brms_outer:", end=" ")
    fit_MRF_outer = scaling_law_fits.general_fit(planets, valid_mask,
                                                dyn_or_surf = "rms,dyn", heating_type = "outer",
                                                print_all=print_scaling_law, error_sym=error_sym,
                                                p0 = initial_guess, bounds = bounds,
                                                fit_type = fit_type, fit_function_type = "MRF", N_gauss = N_gauss_fits)
    print(f"brms_outer_surf:", end=" ")
    fit_MRF_outer_surf = scaling_law_fits.general_fit(planets, valid_mask,
                                                dyn_or_surf = "dip,surf", heating_type = "outer",
                                                print_all=print_scaling_law, error_sym=error_sym,
                                                p0 = initial_guess, bounds = bounds,
                                                fit_type = fit_type, fit_function_type = "MRF", N_gauss = N_gauss_fits)
    with open(filename + ".txt",'a') as f:
        fit_MRF_inner.print_fit_results(f, filename, "brms_inner")
        fit_MRF_inner_surf.print_fit_results(f, filename, "brms_inner_surf")
        fit_MRF_outer.print_fit_results(f, filename, "brms_outer")
        fit_MRF_outer_surf.print_fit_results(f, filename, "brms_outer_surf")


    if rotation: filename = "plots/fits" + name_cut + "/fit_no_flux"
    else: filename = "plots/fits" + name_cut + "_no_rot/fit_no_flux"
    with open(filename + ".txt", 'w') as f:
        f.writelines(name_cut_header + '\n')
    print(f"")
    print(f"----------------------------------------------------------------------")
    print(f"              Fits for Brms = K * M^alpha * R^beta")
    print(f"brms_inner:", end=" ")
    fit_MR_inner = scaling_law_fits.general_fit(planets, valid_mask,
                                                dyn_or_surf = "rms,dyn", heating_type = "inner",
                                                print_all=print_scaling_law, error_sym=error_sym,
                                                p0 = initial_guess_simple, bounds = bounds_simple,
                                                fit_type = fit_type, fit_function_type = "MR", N_gauss = N_gauss_fits)
    print(f"brms_inner_surf:", end=" ")
    fit_MR_inner_surf  = scaling_law_fits.general_fit(planets, valid_mask,
                                                dyn_or_surf = "dip,surf", heating_type = "inner",
                                                print_all=print_scaling_law, error_sym=error_sym,
                                                p0 = initial_guess_simple, bounds = bounds_simple,
                                                fit_type = fit_type, fit_function_type = "MR", N_gauss = N_gauss_fits)
    print(f"brms_outer:", end=" ")
    fit_MR_outer = scaling_law_fits.general_fit(planets, valid_mask,
                                                dyn_or_surf = "rms,dyn", heating_type = "outer",
                                                print_all=print_scaling_law, error_sym=error_sym,
                                                p0 = initial_guess_simple, bounds = bounds_simple,
                                                fit_type = fit_type, fit_function_type = "MR", N_gauss = N_gauss_fits)                                            
    print(f"brms_outer_surf:", end=" ")
    fit_MR_outer_surf = scaling_law_fits.general_fit(planets, valid_mask,
                                                dyn_or_surf = "dip,surf", heating_type = "outer",
                                                print_all=print_scaling_law, error_sym=error_sym,
                                                p0 = initial_guess_simple, bounds = bounds_simple,
                                                fit_type = fit_type, fit_function_type = "MR", N_gauss = N_gauss_fits)

    with open(filename + ".txt",'a') as f:
        fit_MR_inner.print_fit_results(f, filename, "brms_inner")
        fit_MR_inner_surf.print_fit_results(f, filename, "brms_inner_surf")
        fit_MR_outer.print_fit_results(f, filename, "brms_outer")
        fit_MR_outer_surf.print_fit_results(f, filename, "brms_outer_surf")

    if rotation: filename = "plots/fits" + name_cut + "/fit_irr"
    else: filename = "plots/fits" + name_cut + "_no_rot/fit_irr"
    with open(filename + ".txt", 'w') as f:
        f.writelines(name_cut_header + '\n')
    print(f"")
    print(f"----------------------------------------------------------------------")
    print(f"         Fits for Brms, irr = K * M^alpha * R^beta")
    print(f"brms_inner:", end=" ")
    fit_MR_irr_inner = scaling_law_fits.general_fit(planets, ~valid_mask,
                                                dyn_or_surf = "rms,dyn", heating_type = "inner",
                                                print_all=print_scaling_law, error_sym=error_sym,
                                                p0 = initial_guess_simple, bounds = bounds_simple,
                                                fit_type = fit_type, fit_function_type = "MR", N_gauss = N_gauss_fits)
    print(f"brms_inner_surf:", end=" ")
    fit_MR_irr_inner_surf = scaling_law_fits.general_fit(planets, ~valid_mask,
                                                dyn_or_surf = "dip,surf", heating_type = "inner",
                                                print_all=print_scaling_law, error_sym=error_sym,
                                                p0 = initial_guess_simple, bounds = bounds_simple,
                                                fit_type = fit_type, fit_function_type = "MR", N_gauss = N_gauss_fits)
    print(f"brms_outer:", end=" ")
    fit_MR_irr_outer = scaling_law_fits.general_fit(planets, ~valid_mask,
                                                dyn_or_surf = "rms,dyn", heating_type = "outer",
                                                print_all=print_scaling_law, error_sym=error_sym,
                                                p0 = initial_guess_simple, bounds = bounds_simple,
                                                fit_type = fit_type, fit_function_type = "MR", N_gauss = N_gauss_fits)
    print(f"brms_outer_surf:", end=" ")
    fit_MR_irr_outer_surf = scaling_law_fits.general_fit(planets, ~valid_mask,
                                                dyn_or_surf = "dip,surf", heating_type = "outer",
                                                print_all=print_scaling_law, error_sym=error_sym,
                                                p0 = initial_guess_simple, bounds = bounds_simple,
                                                fit_type = fit_type, fit_function_type = "MR", N_gauss = N_gauss_fits)
    
    print(f"brms_irr:", end=" ")
    fit_MR_irr_all = scaling_law_fits.general_fit(planets,
                                                dyn_or_surf = "rms,dyn", heating_type = "irr",
                                                print_all=print_scaling_law, error_sym=error_sym,
                                                p0 = initial_guess_simple, bounds = bounds_simple,
                                                fit_type = fit_type, fit_function_type = "MR", N_gauss = N_gauss_fits)
    print(f"brms_irr_surf:", end=" ")
    fit_MR_irr_all_surf = scaling_law_fits.general_fit(planets,
                                                dyn_or_surf = "dip,surf", heating_type = "irr",
                                                print_all=print_scaling_law, error_sym=error_sym,
                                                p0 = initial_guess_simple, bounds = bounds_simple,
                                                fit_type = fit_type, fit_function_type = "MR", N_gauss = N_gauss_fits)
    with open(filename + ".txt",'a') as f:
        fit_MR_irr_inner.print_fit_results(f, filename, "brms_inner")
        fit_MR_irr_inner_surf.print_fit_results(f, filename, "brms_inner_surf")
        fit_MR_irr_outer.print_fit_results(f, filename, "brms_outer")
        fit_MR_irr_outer_surf.print_fit_results(f, filename, "brms_outer_surf")
        fit_MR_irr_all.print_fit_results(f, filename, "brms_irr")
        fit_MR_irr_all_surf.print_fit_results(f, filename, "brms_irr_surf")

    if rotation:
        if not os.path.exists("./plots/fits_eps" + name_cut): os.makedirs("./plots/fits_eps" + name_cut)
        filename = "plots/fits_eps" + name_cut + "/fit_eps"
    else:
        if not os.path.exists("./plots/fits_eps" + name_cut + "_no_rot"): os.makedirs("./plots/fits_eps" + name_cut + "_no_rot")
        filename = "plots/fits_eps" + name_cut + "_no_rot/fit_eps"

    with open(filename + ".txt", 'w') as f:
        f.writelines(name_cut_header + '\n')
    print(f"")
    print(f"----------------------------------------------------------------------")
    print(f"           Fits for eps = K * M^alpha * R^beta * F_irr^gamma")
    print(f"eps_inner:", end=" ")    
    fit_MRL_eps_inner = scaling_law_fits.general_fit(planets, valid_mask,
                                                dyn_or_surf = "eps", heating_type = "inner",
                                                print_all=print_scaling_law, error_sym=error_sym,
                                                p0 = initial_guess, bounds = bounds,
                                                fit_type = fit_type, fit_function_type = "MRF", N_gauss = N_gauss_fits)
    print(f"eps_outer:", end=" ")    
    fit_MRL_eps_outer = scaling_law_fits.general_fit(planets, valid_mask,
                                                dyn_or_surf = "eps", heating_type = "outer",
                                                print_all=print_scaling_law, error_sym=error_sym,
                                                p0 = initial_guess, bounds = bounds,
                                                fit_type = fit_type, fit_function_type = "MRF", N_gauss = N_gauss_fits)
    print(f"")
    with open(filename + ".txt",'a') as f:
        fit_MRL_eps_inner.print_fit_results(f, filename, "eps_inner")
        fit_MRL_eps_outer.print_fit_results(f, filename, "eps_outer")
    print(f"----------------------------------------------------------------------")
    print("Fitting the different HJ magnetic field estimates: done")
    print(f"----------------------------------------------------------------------")

    if rotation:
        if not os.path.exists("./predictions/predictions" + name_cut): 
            os.makedirs("./predictions/predictions" + name_cut)
    else:
        if not os.path.exists("./predictions/predictions" + name_cut + "_no_rot"): 
            os.makedirs("./predictions/predictions" + name_cut + "_no_rot")

else:
    if rotation: filename = "plots/fits" + name_cut + "/fit_general"
    else: filename = "plots/fits" + name_cut + "_no_rot/fit_general"
    fit_MRL_inner = fit_object(fit_type, "MRL")
    fit_MRL_inner.read_fit_data(filename, "brms_inner")
    fit_MRL_inner_surf = fit_object(fit_type, "MRL")
    fit_MRL_inner_surf.read_fit_data(filename, "brms_inner_surf")
    fit_MRL_outer = fit_object(fit_type, "MRL")
    fit_MRL_outer.read_fit_data(filename, "brms_outer")
    fit_MRL_outer_surf = fit_object(fit_type, "MRL")
    fit_MRL_outer_surf.read_fit_data(filename, "brms_outer_surf")
    
    if rotation: filename = "plots/fits" + name_cut + "/fit_no_eps"
    else: filename = "plots/fits" + name_cut + "_no_rot/fit_no_eps"
    fit_MRF_inner = fit_object(fit_type, "MRF")
    fit_MRF_inner.read_fit_data(filename, "brms_inner")
    fit_MRF_inner_surf = fit_object(fit_type, "MRF")
    fit_MRF_inner_surf.read_fit_data(filename, "brms_inner_surf")
    fit_MRF_outer = fit_object(fit_type, "MRF")
    fit_MRF_outer.read_fit_data(filename, "brms_outer")
    fit_MRF_outer_surf = fit_object(fit_type, "MRF")
    fit_MRF_outer_surf.read_fit_data(filename, "brms_outer_surf")

    if rotation: filename = "plots/fits" + name_cut + "/fit_no_flux"
    else: filename = "plots/fits" + name_cut + "_no_rot/fit_no_flux"
    fit_MR_inner = fit_object(fit_type, "MR")
    fit_MR_inner.read_fit_data(filename, "brms_inner")
    fit_MR_inner_surf = fit_object(fit_type, "MR")
    fit_MR_inner_surf.read_fit_data(filename, "brms_inner_surf")
    fit_MR_outer = fit_object(fit_type, "MR")
    fit_MR_outer.read_fit_data(filename, "brms_outer")
    fit_MR_outer_surf = fit_object(fit_type, "MR")
    fit_MR_outer_surf.read_fit_data(filename, "brms_outer_surf")

    if rotation: filename = "plots/fits" + name_cut + "/fit_irr"
    else: filename = "plots/fits" + name_cut + "_no_rot/fit_irr"
    fit_MR_irr_inner = fit_object(fit_type, "MR")
    fit_MR_irr_inner.read_fit_data(filename, "brms_inner")
    fit_MR_irr_inner_surf = fit_object(fit_type, "MR")
    fit_MR_irr_inner_surf.read_fit_data(filename, "brms_inner_surf")
    fit_MR_irr_outer = fit_object(fit_type, "MR")
    fit_MR_irr_outer.read_fit_data(filename, "brms_outer")
    fit_MR_irr_outer_surf = fit_object(fit_type, "MR")
    fit_MR_irr_outer_surf.read_fit_data(filename, "brms_outer_surf")
    fit_MR_irr_all = fit_object(fit_type, "MR")
    fit_MR_irr_all.read_fit_data(filename, "brms_irr")
    fit_MR_irr_all_surf = fit_object(fit_type, "MR")
    fit_MR_irr_all_surf.read_fit_data(filename, "brms_irr_surf")

    if rotation: filename = "plots/fits_eps" + name_cut + "/fit_eps"
    else: filename = "plots/fits_eps" + name_cut + "_no_rot/fit_eps"
    fit_MRL_eps_inner = fit_object(fit_type, "MRF")
    fit_MRL_eps_inner.read_fit_data(filename, "eps_inner")
    fit_MRL_eps_outer = fit_object(fit_type, "MRF")
    fit_MRL_eps_outer.read_fit_data(filename, "eps_outer")

    if rotation:
        if not os.path.exists("./predictions/predictions" + name_cut + "_" + which_planets): 
            os.makedirs("./predictions/predictions" + name_cut + "_" + which_planets)
    else:
        if not os.path.exists("./predictions/predictions" + name_cut + "_" + which_planets + "_no_rot"): 
            os.makedirs("./predictions/predictions" + name_cut + "_" + which_planets + "_no_rot")

print("Saving individual files for each planet:", end=" ")

table_predictions_save.table_comparision(planets,
                                        valid_mask,
                                        name_cut,
                                        HJ_population,
                                        which_planets,
                                        fit_type, fit_reiners09,
                                        fit_MRL_inner, fit_MRL_inner_surf, fit_MRL_outer, fit_MRL_outer_surf,
                                        fit_MRF_inner, fit_MRF_inner_surf, fit_MRF_outer, fit_MRF_outer_surf,
                                        fit_MR_inner, fit_MR_inner_surf, fit_MR_outer, fit_MR_outer_surf,
                                        fit_MR_irr_inner, fit_MR_irr_inner_surf, fit_MR_irr_outer, fit_MR_irr_outer_surf,
                                        fit_MR_irr_all, fit_MR_irr_all_surf, fit_MRL_eps_inner, fit_MRL_eps_outer,
                                        add_systematic_error,
                                        error_sym=error_sym,
                                        rotation = rotation)

print("done")

print(f"----------------------------------------------------------------------")

# print(np.max(planets.radius_obs), np.min(planets.radius_obs), np.max(planets.mass_obs), np.min(planets.mass_obs), np.max(planets.flux_obs), np.min(planets.flux_obs))
# Repeat the everything exoplanets with some missing parameters:
if save_plots_grid:
    print(f"Saving plots for grid models:", end=" ")
    plots.plot_output_grid(masses,
                            fluxes,
                            heat_effs,
                            table_inner.radius_heat,
                            table_outer.radius_heat,
                            table_inner.bscaling_heat,
                            table_outer.bscaling_heat,
                            table_inner.bscaling_surf_heat,
                            table_outer.bscaling_surf_heat)
    print(f"done")
    print(f"----------------------------------------------------------------------")
    
if save_plots_3D_grid:
    print(f"Saving 3D plots as the total grid:", end=" ")
    plots.plot_grid(masses,
                    fluxes,
                    heat_effs,
                    table_inner.radius_heat,#[:,:,0:20],
                    table_outer.radius_heat)#[:,:,0:20])
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
    print(f"----------------------------------------------------------------------")
    print(f"Saving plots for HJ population fits:", end=" ")
    
    plots.plot_population_fits(planets,
                            name_cut,
                            valid_mask,
                            fit_type, fit_reiners09,
                            fit_MRL_inner, fit_MRL_inner_surf, fit_MRL_outer, fit_MRL_outer_surf,
                            fit_MRF_inner, fit_MRF_inner_surf, fit_MRF_outer, fit_MRF_outer_surf,
                            fit_MR_inner, fit_MR_inner_surf, fit_MR_outer, fit_MR_outer_surf,
                            fit_MR_irr_inner, fit_MR_irr_inner_surf, fit_MR_irr_outer, fit_MR_irr_outer_surf,
                            fit_MR_irr_all, fit_MR_irr_all_surf,
                            rotation=rotation)
    
    plots.plot_population_fits_eps(planets,
                                    name_cut,
                                    valid_mask,
                                    fit_type, fit_MRL_eps_inner, fit_MRL_eps_outer,
                                    rotation=rotation)
    
    print(f"done")
    print(f"----------------------------------------------------------------------")
    

# Repeat everything and save the whole gas giant population
if HJ_population:
    print(f"Repeat process for exoplanets with incomplete stellar data:")
    print(f"----------------------------------------------------------------------")

    fileHJ = 'PS_2026.02.18_07.09.03.csv'
    print("Reading gas giant exoplanet population " + fileHJ + ": ", end="\r")
    Npl = len(planet_functions.read_archive_csv(fileHJ))
    planets = planet_functions.planet_object(Npl,error_sym)
    planets.read_file(fileHJ)

    print("done")
    print(f"----------------------------------------------------------------------")

    
    if rotation:
        mask_predictions = (planets.mass_obs > min_mass) & (planets.mass_obs < max_mass) & \
                            (planets.flux_obs > min_flux) & (planets.flux_obs < max_flux) & \
                            (planets.orbital_period_obs > min_period) & (planets.orbital_period_obs < max_period)
    else:
        mask_predictions = (planets.mass_obs > min_mass) & (planets.mass_obs < max_mass) & \
                            (planets.flux_obs > min_flux) & (planets.flux_obs < max_flux)


    print("Interpolating exoplanets within simulation grid: ")
    if rotation:
        interpolator.linear_interpolator_rot(table_inner, 
                                            table_outer, 
                                            planets,
                                            periods,
                                            mask_predictions,
                                            error_sym=error_sym)
        

    else:
        interpolator.linear_interpolator(table_inner, table_outer,
                                        planets,
                                        mask_predictions,
                                        error_sym=error_sym)
    
    print("done")
    print(f"----------------------------------------------------------------------")

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
    
    if which_cut == "no cut":
        valid_mask = planets.heat_eff_inner_est > 1e-5
        name_cut, name_cut_header = "_no_cut", "No cut (actually eps > 1e-5)"

    elif which_cut == "flux cut":
        valid_mask = planets.flux_obs*planets.heat_eff_inner_est > 1e6
        name_cut, name_cut_header = "_cut_epsF", "Cut for eps*F_irr > 1e6"
        #print(flux_obs*heat_eff_inner_est)

    elif which_cut == "heat cut":
        valid_mask = planets.flux_obs*planets.heat_eff_inner_est*np.pi*planets.radius_obs**2 > 1e6
        name_cut, name_cut_header = "_cut_Qdep", "Cut for Q_dep = eps*F_irr*pi*R^2 > 1e6"
        #print(flux_obs*heat_eff_inner_est*np.pi*radius_obs**2)

    elif which_cut == "radius cut":
        valid_mask = planets.radius_obs/planets.radius_irr_est > 0.75
        name_cut, name_cut_header = "_cut_R", "Cut for Robs/Rirr > 0.75"
        #print(radius_obs/radius_irr_est)

    elif which_cut == "brms cut":
        valid_mask = planets.brms_outer_est/planets.brms_inner_est < 0.25
        name_cut, name_cut_header = "_cut_B", "Cut for Bout/Bin < 0.25"
        #print(brms_outer_est/brms_inner_est)
    
    if not os.path.exists("./predictions/predictions" + name_cut + "_extra"): 
        os.makedirs("./predictions/predictions" + name_cut + "_extra")
    if not rotation:
        if not os.path.exists("./predictions/predictions" + name_cut + "_extra_no_rot"): 
            os.makedirs("./predictions/predictions" + name_cut + "_extra_no_rot")


    print("Saving individual files for each planet:", end=" ")

    table_predictions_save.table_comparision(planets,
                                            valid_mask,
                                            name_cut,
                                            HJ_population,
                                            which_planets,
                                            fit_type, fit_reiners09,
                                            fit_MRL_inner, fit_MRL_inner_surf, fit_MRL_outer, fit_MRL_outer_surf,
                                            fit_MRF_inner, fit_MRF_inner_surf, fit_MRF_outer, fit_MRF_outer_surf,
                                            fit_MR_inner, fit_MR_inner_surf, fit_MR_outer, fit_MR_outer_surf,
                                            fit_MR_irr_inner, fit_MR_irr_inner_surf, fit_MR_irr_outer, fit_MR_irr_outer_surf,
                                            fit_MR_irr_all, fit_MR_irr_all_surf, fit_MRL_eps_inner, fit_MRL_eps_outer,
                                            add_systematic_error,
                                            extra_HJ=True,
                                            error_sym=error_sym,
                                            rotation = rotation)
    
    print("done")
    
print(f"----------------------------------------------------------------------")
print(f"----------  Created by: A. Elias-López, Leiden University ------------")
print(f"----------------------------------------------------------------------")
    
    
    