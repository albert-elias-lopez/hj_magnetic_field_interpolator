"""
Output files of main.py: the predictions of each planet (table_predictions,
predictions/predictions_<sample>.txt) and their comparison with the scaling laws (table_comparison,
one file per planet in predictions/predictions_cut_R_<fit_type>_fit[_<sample>]/). Also the scaling
laws with their Monte Carlo errors, the heating efficiency of Thorngren & Fortney (2018) and the
optional factor-of-2 systematic error.
"""
import numpy as np
from constants import R_J, L_J

def get_print_string(variable, variable_err, variable_type,error_sym=False):
    if error_sym:
        if np.isnan(variable):
            if variable_type == "heat": variable_print      = "-----------------"
            elif variable_type == "brms": variable_print    = "-----------------------"
            elif variable_type == "period": variable_print  = "-------------------------"
            elif variable_type == "flux": variable_print    = "-----------------------"
            elif variable_type in ["radius", "mass"]: variable_print = "---------------"
        else: 
            if variable_type == "heat":
                variable_print = r"{:.4f}$\pm${:.4f}".format(variable, variable_err)
                if variable_err > 10: variable_print = r"{:.4f}$\pm${:.3e}".format(variable, variable_err)
            elif variable_type == "brms": variable_print = r"{:.4e}$\pm${:.1e}".format(variable, variable_err)
            elif variable_type == "period": variable_print = r"{:.6e}$\pm${:.1e}".format(variable, variable_err)
            elif variable_type == "flux": variable_print = r"{:.3e}$\pm${:.1e}".format(variable, variable_err)
            elif variable_type in ["radius", "mass"]:
                variable_print = r"{:.3f}$\pm${:.3f}".format(variable, variable_err)
                if variable_err > 10: variable_print = r"{:.3f}$\pm${:.2f}".format(variable, variable_err)
            if np.isnan(variable_err):
                if variable_type == "heat": variable_print = r"{:.4f}$\pm$------".format(variable)
                elif variable_type == "brms": variable_print = r"{:.4e}$\pm$--------".format(variable)
                elif variable_type == "period": variable_print = r"{:.6e}$\pm$--------".format(variable)
                elif variable_type == "flux": variable_print = r"{:.3e}$\pm$-------".format(variable, variable_err)
                elif variable_type in ["radius", "mass"]: variable_print = r"{:.3f}$\pm$-----".format(variable, variable_err)
    else: 
        if np.isnan(variable):
            if variable_type == "heat":         variable_print = "----------------------------"
            elif variable_type == "brms":       variable_print = "----------------------------------"
            elif variable_type == "period":     variable_print = "------------------------------------"
            elif variable_type == "flux":       variable_print = "---------------------------------------"
            elif variable_type in ["radius", "mass"]: variable_print = "-----------------------------"
        else: 
            if variable_type == "heat":
                variable_print = r"{:.4f}$_{{-{:.4f}}}^{{+{:.4f}}}$".format(variable, variable_err[0], variable_err[1])
                if variable_err[0] > 10:
                    if variable_err[1] > 10: variable_print = r"{:.4f}$_{{-{:.3f}}}^{{+{:.3f}}}$".format(variable, variable_err[0], variable_err[1])
                    else: variable_print = r"{:.4f}$_{{-{:.3f}}}^{{+{:.4f}}}$".format(variable, variable_err[0], variable_err[1])
                elif variable_err[1] > 10: variable_print = r"{:.4f}$_{{-{:.4f}}}^{{+{:.3f}}}$".format(variable, variable_err[0], variable_err[1])
            elif variable_type == "brms": variable_print = r"{:.4e}$_{{-{:.1e}}}^{{+{:.1e}}}$".format(variable, variable_err[0], variable_err[1])
            elif variable_type == "period": variable_print = r"{:.6e}$_{{-{:.1e}}}^{{+{:.1e}}}$".format(variable, variable_err[0], variable_err[1])
            elif variable_type == "flux": variable_print = r"{:.3e}$_{{-{:.3e}}}^{{+{:.1e}}}$".format(variable, variable_err[0], variable_err[1])
            elif variable_type in ["radius", "mass"]: variable_print = r"{:.3f}$_{{-{:.3f}}}^{{+{:.3f}}}$".format(variable, variable_err[0], variable_err[1])
            if np.isnan(variable_err).any():
                if variable_type == "heat": variable_print = r"{:.4f}$_-----^-----$".format(variable)
                elif variable_type == "brms": variable_print = r"{:.4e}$_------^-------$".format(variable)
                elif variable_type == "period": variable_print = r"{:.6e}$_------^-------$".format(variable)
                elif variable_type == "flux": variable_print = r"{:.3e}$_-------^-------$".format(variable)
                elif variable_type in ["radius", "mass"]: variable_print = r"{:.3f}$\pm$----------".format(variable, variable_err)
    return variable_print

def sample_split_normal(mode, sigma_minus, sigma_plus, size=100000):
    z = np.random.normal(0, 1, size)
    return mode + np.where(z < 0, z * sigma_minus, z * sigma_plus)

def sample_obs(value, err, N):
    # err is one symmetric error or a (lower, upper) pair, sampled as a split normal
    if np.ndim(err) == 0: return np.random.normal(value, err, N)
    return sample_split_normal(value, err[0], err[1], N)

def scaling_law(fit_object, mass, radius, conv_or_flux):
    """Magnetic field (G) of a scaling-law fit_object: K M^alpha R^beta X^gamma, with X the luminosity or the flux factor."""
    brms = fit_object.K * mass ** fit_object.alpha * radius ** fit_object.beta * conv_or_flux ** fit_object.gamma
    return brms

def add_syst_error(mean, error):
    # Model systematic uncertainty of a factor of 2 (x1/2 below, x2 above), added in quadrature
    # to the (lower, upper) errors; a symmetric error gets the mean of both sides
    lower, upper = (error, error) if np.ndim(error) == 0 else (error[0], error[1])
    error_lower = np.sqrt(lower**2 + (0.5 * mean)**2)
    error_upper = np.sqrt(upper**2 + mean**2)
    if np.ndim(error) == 0: return (error_lower + error_upper) / 2
    return np.array([error_lower, error_upper])

def get_scaling_law_err(fit_object,
                        brms, 
                        mass,
                        radius,
                        conv_or_flux,
                        mass_err,
                        radius_err,
                        conv_or_flux_err,
                        Nerr = 10000):
    """
    Monte Carlo error of a scaling-law prediction brms, including the errors of the observed values and of
    the fit parameters (for the Gaussian fits). The R09 law has fixed exponents, so only K is sampled.
    """
    if fit_object.fit_type == "gaussian":

        K_s     = sample_split_normal(fit_object.K, fit_object.K_err[0], fit_object.K_err[1], Nerr)
        if fit_object.fit_function_type == "reiners09": # No error bars for the Reiners et al. 2009 law, since it is not a fit but a scaling law with fixed exponents
            alpha_s = fit_object.alpha
            beta_s  = fit_object.beta
            gamma_s = fit_object.gamma
        else:
            alpha_s = sample_split_normal(fit_object.alpha, fit_object.alpha_err[0], fit_object.alpha_err[1], Nerr)
            beta_s  = sample_split_normal(fit_object.beta, fit_object.beta_err[0], fit_object.beta_err[1], Nerr)
            if fit_object.fit_function_type == "MR":
                gamma_s = fit_object.gamma
            else:
                gamma_s = sample_split_normal(fit_object.gamma, fit_object.gamma_err[0], fit_object.gamma_err[1], Nerr)

        mass_s          = sample_obs(mass, mass_err, Nerr)
        radius_s        = sample_obs(radius, radius_err, Nerr)
        conv_or_flux_s  = np.random.normal(conv_or_flux, conv_or_flux_err, Nerr)

        with np.errstate(invalid='ignore'):
            brms_s = (K_s * \
                     np.power(mass_s, alpha_s) * \
                     np.power(radius_s, beta_s) * \
                     np.power(conv_or_flux_s, gamma_s))
            
        #brms_s = K_s * mass_s ** alpha_s * radius_s ** beta_s * conv_or_flux_s ** gamma_s
        valid_mask = (mass_s > 0) & (radius_s > 0) & (conv_or_flux_s > 0)
        brms_s = brms_s[valid_mask]
        if brms_s.size == 0:
            brms_err = np.array([np.nan, np.nan])
        else:
            p16, p50, p84 = np.percentile(brms_s, q = [16, 50, 84])
            brms_err = np.array([p50 - p16, p84 - p50])
    else:
        mass_err, radius_err = np.max(mass_err), np.max(radius_err)   # The analytic formula uses symmetric errors
        brms_err =  (fit_object.alpha*mass_err/mass)**2+\
                    (fit_object.beta*radius_err/radius)**2+\
                    (fit_object.gamma*conv_or_flux_err/conv_or_flux)**2+\
                    (fit_object.K_err/fit_object.K)**2+\
                    (np.log(mass)*fit_object.alpha_err)**2+\
                    (np.log(radius)*fit_object.beta_err)**2+\
                    (np.log(conv_or_flux)*fit_object.gamma_err)**2
        brms_err = brms*np.sqrt(brms_err)

    # Derivative vector
    # x = np.array([1,np.log(mass), np.log(radius), np.log(conv_or_flux)])
    # brms_err = x @ fit_object.cov_matrix @ x
    
    # brms_err += (fit_object.alpha*mass_err/mass)**2+\
    #              (fit_object.beta*radius_err/radius)**2+\
    #              (fit_object.gamma*conv_or_flux_err/conv_or_flux)**2
    # brms_err = brms*np.sqrt(brms_err)
    return brms_err

def get_thorn18_eff(flux, eps0 = 2.37/100, mu = 0.14, sigma =0.37):
    """Heating efficiency (fraction) of the Thorngren & Fortney (2018) relation for the flux F_irr (erg cm^-2 s^-1)."""
    return eps0*np.exp(-(np.log10(flux/1e9)-mu)**2/(2* sigma**2))

def get_thorn18_err(flux, Nerr, eps0 = 2.37/100, mu = 0.14, sigma =0.37):
    """Monte Carlo (minus, plus) error of get_thorn18_eff, sampling the published uncertainties of its parameters."""
    eps0_s   = sample_split_normal(eps0, 0.26/100, 1.3/100, Nerr)
    mu_s  = sample_split_normal(mu, 0.069, 0.060, Nerr)
    sig_s = sample_split_normal(sigma, 0.059, 0.038, Nerr)   
    mask = (eps0_s > 0) & (sig_s > 0)
    eps0_s, mu_s, sig_s = eps0_s[mask], mu_s[mask], sig_s[mask] 
    eps_s = eps0_s * np.exp(-((np.log10(flux/1e9) - mu_s)**2) / (2 * sig_s**2))
    eps_s = eps_s[np.isfinite(eps_s)]
    p16, p50, p84 = np.percentile(eps_s, [16, 50, 84])
    return np.array([p50 - p16, p84 - p50])

def table_predictions(planets,
                      mask_predictions,
                      HJ_population,
                      which_planets,
                      add_systematic_error,
                      extra_HJ=False,
                      error_sym=False,
                      N_err = 10000,
                      rotation = False):
    """
    Writes predictions/predictions_<sample>[_no_rot][_syst_err].txt: one line per planet in
    mask_predictions with the observed values, the heating efficiencies (in %) and the magnetic fields for
    inner heating, no heating (eps = 0) and outer heating (see the README for the columns).
    """
    if HJ_population:
        filename = "predictions/predictions_hot_jupiter_populations"
        if extra_HJ: filename = "predictions/predictions_hot_jupiter_populations_extra"
    else: filename = "predictions/predictions_" + which_planets
    if not rotation: filename += "_no_rot"
    if add_systematic_error: filename += "_syst_err"
    if error_sym:
        header = "#     M(MJ)               R(RJ)              R_irr(RJ)             F(erg cm-2 s-1)         P(days)             " + \
              "    eps_up(%)             eps_low(%)             eps_th18(%)                 bdyn_up(G)                  bdyn_irr(G)              " + \
              "    bdyn_low(G)                 bdipsurf_up(G)               bdipsurf_irr(G)              bdipsurf_low(G) "
    else:
        header = "#     M(MJ)               R(RJ)                   R_irr(RJ)                  F(erg cm-2 s-1)                P(days)                  " + \
              "      eps_up(%)                         eps_low(%)                        eps_th18(%)                           bdyn_up(G)                            bdyn_irr(G)                       " + \
              "      bdyn_low(G)                            bdipsurf_up(G)                          bdipsurf_irr(G)                        bdipsurf_low(G) "

    with open(filename + ".txt", 'w') as f:
        f.writelines(header + '\n')

    with open(filename + ".txt",'a') as f:
        for index_planet, name in enumerate(planets.planet_names):
            if mask_predictions[index_planet]:

                if add_systematic_error:
                    heat_eff_inner_est_err     = add_syst_error(planets.heat_eff_inner_est[index_planet],
                                                                planets.heat_eff_inner_est_err[index_planet])
                    brms_inner_est_err         = add_syst_error(planets.brms_inner_est[index_planet],
                                                                planets.brms_inner_est_err[index_planet])
                    brms_inner_surf_est_err    = add_syst_error(planets.brms_inner_surf_est[index_planet],
                                                                planets.brms_inner_surf_est_err[index_planet])
                    heat_eff_outer_est_err     = add_syst_error(planets.heat_eff_outer_est[index_planet],
                                                                planets.heat_eff_outer_est_err[index_planet])
                    brms_outer_est_err         = add_syst_error(planets.brms_outer_est[index_planet],
                                                                planets.brms_outer_est_err[index_planet])
                    brms_outer_surf_est_err    = add_syst_error(planets.brms_outer_surf_est[index_planet],
                                                                planets.brms_outer_surf_est_err[index_planet])
                    brms_irr_est_err           = add_syst_error(planets.brms_irr_est[index_planet],
                                                                planets.brms_irr_est_err[index_planet])
                    brms_irr_surf_est_err      = add_syst_error(planets.brms_irr_surf_est[index_planet],
                                                                planets.brms_irr_surf_est_err[index_planet])
                else:
                    heat_eff_inner_est_err     = planets.heat_eff_inner_est_err[index_planet]
                    brms_inner_est_err         = planets.brms_inner_est_err[index_planet]
                    brms_inner_surf_est_err    = planets.brms_inner_surf_est_err[index_planet]
                    heat_eff_outer_est_err     = planets.heat_eff_outer_est_err[index_planet]
                    brms_outer_est_err         = planets.brms_outer_est_err[index_planet]
                    brms_outer_surf_est_err    = planets.brms_outer_surf_est_err[index_planet]
                    brms_irr_est_err           = planets.brms_irr_est_err[index_planet]
                    brms_irr_surf_est_err      = planets.brms_irr_surf_est_err[index_planet]

                heat_eff_inner_est_print = get_print_string(100*planets.heat_eff_inner_est[index_planet],
                                                               100*heat_eff_inner_est_err, 
                                                               "heat",error_sym=error_sym)
                brms_inner_est_print = get_print_string(planets.brms_inner_est[index_planet],
                                                               brms_inner_est_err, 
                                                               "brms",error_sym=error_sym)
                brms_inner_surf_est_print = get_print_string(planets.brms_inner_surf_est[index_planet],
                                                                brms_inner_surf_est_err, 
                                                                "brms",error_sym=error_sym)
                

                heat_eff_outer_est_print = get_print_string(100*planets.heat_eff_outer_est[index_planet],
                                                               100*heat_eff_outer_est_err, 
                                                               "heat",error_sym=error_sym)
                brms_outer_est_print = get_print_string(planets.brms_outer_est[index_planet],
                                                               brms_outer_est_err, 
                                                               "brms",error_sym=error_sym)
                brms_outer_surf_est_print = get_print_string(planets.brms_outer_surf_est[index_planet],
                                                                brms_outer_surf_est_err, 
                                                                "brms",error_sym=error_sym)

            
                brms_irr_est_print = get_print_string(planets.brms_irr_est[index_planet],
                                                               brms_irr_est_err, 
                                                               "brms",error_sym=error_sym)
                brms_irr_surf_est_print = get_print_string(planets.brms_irr_surf_est[index_planet],
                                                                brms_irr_surf_est_err, 
                                                                "brms",error_sym=error_sym)
                radius_irr_surf_est_print = get_print_string(planets.radius_irr_est[index_planet],
                                                                planets.radius_irr_est_err[index_planet], 
                                                                "radius",error_sym=error_sym)

                thorn_eff_18 = get_thorn18_eff(planets.flux_obs[index_planet])
                thorn_eff_18_err = get_thorn18_err(planets.flux_obs[index_planet], N_err)
                heat_eff_thorn_print = get_print_string(100*thorn_eff_18, 100*thorn_eff_18_err, "heat",error_sym=False)
                    
                mass_obs_print = get_print_string(planets.mass_obs[index_planet],planets.mass_obs_err[index_planet],"mass",error_sym=True)
                radius_obs_print = get_print_string(planets.radius_obs[index_planet],planets.radius_obs_err[index_planet],"radius",error_sym=True)
                flux_obs_print = get_print_string(planets.flux_obs[index_planet],planets.flux_obs_err[index_planet],"flux",error_sym=True)
                orbital_period_obs_print = get_print_string(planets.orbital_period_obs[index_planet],planets.orbital_period_obs_err[index_planet],"period",error_sym=True)

                f.writelines(mass_obs_print                 + "      " + \
                            radius_obs_print                + "      " + \
                            radius_irr_surf_est_print       + "      " + \
                            flux_obs_print                  + "      " + \
                            orbital_period_obs_print        + "      " + \
                            heat_eff_inner_est_print        + "      " + \
                            heat_eff_outer_est_print        + "      " + \
                            heat_eff_thorn_print            + "      " + \
                            brms_inner_est_print            + "      " + \
                            brms_irr_est_print              + "      " + \
                            brms_outer_est_print            + "      " + \
                            brms_inner_surf_est_print       + "      " + \
                            brms_irr_surf_est_print         + "      " + \
                            brms_outer_surf_est_print       + "      " + \
                            name                            + "      " + '\n')
                
    print("Saving file " + filename + ".txt: done")

def deposited_power(radius, radius_err, flux, flux_err, heat_eff, heat_eff_err):
    """
    Power deposited in the interior, L_P = eps pi R^2 F_irr, in units of L_J, and its error (linear
    propagation with symmetric errors; heat_eff_err is the mean of the lower and upper errors).
    """
    power = np.pi*(radius*R_J)**2*flux*heat_eff/L_J
    power_err = power*np.sqrt((heat_eff_err/heat_eff)**2+\
                              (2*radius_err/radius)**2+\
                              (flux_err/flux)**2)
    return power, power_err

def reiners09_surf(brms, brms_err, mass, mass_err):
    # B_dip,surf of the Reiners et al. (2009) law from its B_rms,dyn: B (1 - 0.17/M)^3 / sqrt(2)
    brms_surf = brms*(1-0.17/mass)**3/np.sqrt(2)
    brms_surf_err = brms_surf*np.sqrt((brms_err/brms)**2+\
                                      (3*0.17*mass_err/mass**2/(1-0.17/mass))**2)
    return brms_surf, brms_surf_err

# Fits used by scaling_law_predictions: <law>_<heating>[_surf] (_surf: B_dip,surf, otherwise B_rms,dyn)
SCALING_LAW_FITS = ["MRL_inner", "MRL_inner_surf", "MRL_outer", "MRL_outer_surf",
                    "MRF_inner", "MRF_inner_surf", "MRF_outer", "MRF_outer_surf",
                    "MR_inner",  "MR_inner_surf",  "MR_outer",  "MR_outer_surf",
                    "MR_irr",    "MR_irr_surf"]

def scaling_law_predictions(fits, mass, mass_err, mass_err_asym, radius, radius_err, radius_err_asym,
                            flux, flux_err, power_inner, power_inner_err, power_outer, power_outer_err,
                            N_err = 10000):
    """
    Scaling-law predictions for one planet, each as (value, error): the magnetic fields (G) of the fits
    in SCALING_LAW_FITS and of R09 (Reiners et al. 2009) with the interpolated inner heating
    efficiency ("R09_inner") or the one of Thorngren & Fortney 2018 ("R09+Th18_inner"), plus that
    heating efficiency ("eps_th18", fraction). fits: dictionary with the fit_objects of
    SCALING_LAW_FITS and "R09". power_*: deposited power L_P / L_J for inner and outer heating.
    The values are computed in a fixed order, since the Monte Carlo errors share numpy's random numbers.
    """
    def predict(fit, x, x_err):
        value = scaling_law(fit, mass, radius, x)
        return value, get_scaling_law_err(fit, value, mass, radius, x, mass_err_asym, radius_err_asym, x_err)

    # Variable of each law (deposited power, flux factor, or 1 for the MR laws) and its error
    variables = {"MRL_inner": (power_inner, power_inner_err), "MRL_outer": (power_outer, power_outer_err),
                 "MRF_inner": ((flux/1e9), (flux_err/1e9)),   "MRF_outer": ((flux/1e9), (flux_err/1e9))}

    predictions = {}
    # Reiners et al. 2009 (the outer-heating prediction is not used, so it is not computed)
    predictions["R09_inner"] = predict(fits["R09"], power_inner, power_inner_err)
    predictions["R09_inner_surf"] = reiners09_surf(*predictions["R09_inner"], mass, mass_err)

    for name in SCALING_LAW_FITS:
        predictions[name] = predict(fits[name], *variables.get(name.replace("_surf", ""), (1, 1)))

    # R09 with the heating efficiency of Thorngren & Fortney 2018
    eps_th18 = get_thorn18_eff(flux)
    eps_th18_err = get_thorn18_err(flux, N_err)
    predictions["eps_th18"] = (eps_th18, eps_th18_err)
    power_th18, power_th18_err = deposited_power(radius, radius_err, flux, flux_err, eps_th18, np.mean(eps_th18_err))
    predictions["R09+Th18_inner"] = predict(fits["R09"], power_th18, power_th18_err)
    predictions["R09+Th18_inner_surf"] = reiners09_surf(*predictions["R09+Th18_inner"], mass, mass_err)
    return predictions

def table_comparison(planets,
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
                        extra_HJ=False,
                        error_sym=False,
                        N_err = 10000,
                        rotation = False):
    """
    Writes one file per planet in valid_mask with its interpolated magnetic fields and the predictions of
    the scaling laws (the fits MRL, MRF and MR of this work, and R09 with the interpolated or the
    Thorngren & Fortney 2018 heating efficiency), for inner heating, outer heating and no heating.
    """
    heat_eff_inner_est         = planets.heat_eff_inner_est[valid_mask]
    heat_eff_inner_est_err     = planets.heat_eff_inner_est_err[valid_mask]
    brms_inner_est             = planets.brms_inner_est[valid_mask]
    brms_inner_est_err         = planets.brms_inner_est_err[valid_mask]
    brms_inner_surf_est        = planets.brms_inner_surf_est[valid_mask]
    brms_inner_surf_est_err    = planets.brms_inner_surf_est_err[valid_mask]
    heat_eff_outer_est         = planets.heat_eff_outer_est[valid_mask]
    heat_eff_outer_est_err     = planets.heat_eff_outer_est_err[valid_mask]
    brms_outer_est             = planets.brms_outer_est[valid_mask]
    brms_outer_est_err         = planets.brms_outer_est_err[valid_mask]
    brms_outer_surf_est        = planets.brms_outer_surf_est[valid_mask]
    brms_outer_surf_est_err    = planets.brms_outer_surf_est_err[valid_mask]
    radius_irr_est             = planets.radius_irr_est[valid_mask]
    radius_irr_est_err         = planets.radius_irr_est_err[valid_mask]
    brms_irr_est               = planets.brms_irr_est[valid_mask]
    brms_irr_est_err           = planets.brms_irr_est_err[valid_mask]
    brms_irr_surf_est          = planets.brms_irr_surf_est[valid_mask]
    brms_irr_surf_est_err      = planets.brms_irr_surf_est_err[valid_mask]
    mass_obs                   = planets.mass_obs[valid_mask]
    mass_obs_err               = planets.mass_obs_err[valid_mask]
    flux_obs                   = planets.flux_obs[valid_mask]
    flux_obs_err               = planets.flux_obs_err[valid_mask]
    radius_obs                 = planets.radius_obs[valid_mask]
    radius_obs_err             = planets.radius_obs_err[valid_mask]
    # (lower, upper) errors, used to sample mass and radius in the Monte Carlo scaling-law errors
    mass_obs_err_asym          = planets.mass_obs_err_asym[valid_mask]
    radius_obs_err_asym        = planets.radius_obs_err_asym[valid_mask]
    orbital_period_obs         = planets.orbital_period_obs[valid_mask]
    orbital_period_obs_err     = planets.orbital_period_obs_err[valid_mask]

    planet_names               = planets.planet_names
    new_planet_names = []
    for index_planet, name in enumerate(planet_names):
        if valid_mask[index_planet]: new_planet_names += [planet_names[index_planet]]
    planet_names = new_planet_names

    if fit_type == "gaussian":
        error_sym_fits = False
    else: error_sym_fits = True
    
    # Deposited power L_P = eps pi R^2 F_irr (in L_J) and its error, for inner and outer heating
    if error_sym: heat_eff_inner_est_err_mean = heat_eff_inner_est_err
    else: heat_eff_inner_est_err_mean = np.mean(heat_eff_inner_est_err, axis=1)   # Mean of the lower and upper error of each planet
    conv_lumin_inner, conv_lumin_inner_err = deposited_power(radius_obs, radius_obs_err, flux_obs, flux_obs_err,
                                                             heat_eff_inner_est, heat_eff_inner_est_err_mean)
    if error_sym: heat_eff_outer_est_err_mean = heat_eff_outer_est_err
    else: heat_eff_outer_est_err_mean = np.mean(heat_eff_outer_est_err, axis=1)
    conv_lumin_outer, conv_lumin_outer_err = deposited_power(radius_obs, radius_obs_err, flux_obs, flux_obs_err,
                                                             heat_eff_outer_est, heat_eff_outer_est_err_mean)

    for index_planet, name in enumerate(planet_names):
        if add_systematic_error:
            heat_eff_inner_est_err[index_planet]     = add_syst_error(heat_eff_inner_est[index_planet],
                                                                        heat_eff_inner_est_err[index_planet])
            brms_inner_est_err[index_planet]         = add_syst_error(brms_inner_est[index_planet],
                                                                        brms_inner_est_err[index_planet])
            brms_inner_surf_est_err[index_planet]    = add_syst_error(brms_inner_surf_est[index_planet],
                                                                        brms_inner_surf_est_err[index_planet])
            heat_eff_outer_est_err[index_planet]     = add_syst_error(heat_eff_outer_est[index_planet],
                                                                        heat_eff_outer_est_err[index_planet])
            brms_outer_est_err[index_planet]         = add_syst_error(brms_outer_est[index_planet],
                                                                        brms_outer_est_err[index_planet])
            brms_outer_surf_est_err[index_planet]    = add_syst_error(brms_outer_surf_est[index_planet],
                                                                        brms_outer_surf_est_err[index_planet])
            brms_irr_est_err[index_planet]           = add_syst_error(brms_irr_est[index_planet],
                                                                        brms_irr_est_err[index_planet])
            brms_irr_surf_est_err[index_planet]      = add_syst_error(brms_irr_surf_est[index_planet],
                                                                        brms_irr_surf_est_err[index_planet])
    
    fits = {"R09": fit_reiners09,
            "MRL_inner": fit_MRL_inner, "MRL_inner_surf": fit_MRL_inner_surf, "MRL_outer": fit_MRL_outer, "MRL_outer_surf": fit_MRL_outer_surf,
            "MRF_inner": fit_MRF_inner, "MRF_inner_surf": fit_MRF_inner_surf, "MRF_outer": fit_MRF_outer, "MRF_outer_surf": fit_MRF_outer_surf,
            "MR_inner":  fit_MR_inner,  "MR_inner_surf":  fit_MR_inner_surf,  "MR_outer":  fit_MR_outer,  "MR_outer_surf":  fit_MR_outer_surf,
            "MR_irr":    fit_MR_irr_all, "MR_irr_surf":   fit_MR_irr_all_surf}

    for index_planet, name in enumerate(planet_names):

        predictions = scaling_law_predictions(fits, mass_obs[index_planet], mass_obs_err[index_planet], mass_obs_err_asym[index_planet],
                                              radius_obs[index_planet], radius_obs_err[index_planet], radius_obs_err_asym[index_planet],
                                              flux_obs[index_planet], flux_obs_err[index_planet],
                                              conv_lumin_inner[index_planet], conv_lumin_inner_err[index_planet],
                                              conv_lumin_outer[index_planet], conv_lumin_outer_err[index_planet], N_err)
        fits_print = {key: get_print_string(value, error, "brms", error_sym=error_sym_fits)
                      for key, (value, error) in predictions.items() if key != "eps_th18"}
        thorn_eff_18, thorn_eff_18_err = predictions["eps_th18"]
        heat_eff_thorn_print = get_print_string(100*thorn_eff_18, 100*thorn_eff_18_err, "heat",error_sym=False)

        mass_obs_print = get_print_string(mass_obs[index_planet],mass_obs_err[index_planet],"mass",error_sym=True)
        radius_obs_print = get_print_string(radius_obs[index_planet],radius_obs_err[index_planet],"radius",error_sym=True)
        flux_obs_print = get_print_string(flux_obs[index_planet],flux_obs_err[index_planet],"flux",error_sym=True)
        orbital_period_obs_print = get_print_string(orbital_period_obs[index_planet],orbital_period_obs_err[index_planet],"period",error_sym=True)

        heat_eff_inner_est_print = get_print_string(100*heat_eff_inner_est[index_planet],
                                                        100*heat_eff_inner_est_err[index_planet], 
                                                        "heat",error_sym=error_sym)
        brms_inner_est_print = get_print_string(brms_inner_est[index_planet],
                                                        brms_inner_est_err[index_planet], 
                                                        "brms",error_sym=error_sym)
        brms_inner_surf_est_print = get_print_string(brms_inner_surf_est[index_planet],
                                                        brms_inner_surf_est_err[index_planet], 
                                                        "brms",error_sym=error_sym)
        

        heat_eff_outer_est_print = get_print_string(100*heat_eff_outer_est[index_planet],
                                                        100*heat_eff_outer_est_err[index_planet], 
                                                        "heat",error_sym=error_sym)
        brms_outer_est_print = get_print_string(brms_outer_est[index_planet],
                                                        brms_outer_est_err[index_planet], 
                                                        "brms",error_sym=error_sym)
        brms_outer_surf_est_print = get_print_string(brms_outer_surf_est[index_planet],
                                                        brms_outer_surf_est_err[index_planet], 
                                                        "brms",error_sym=error_sym)

    
        brms_irr_est_print = get_print_string(brms_irr_est[index_planet],
                                                        brms_irr_est_err[index_planet], 
                                                        "brms",error_sym=error_sym)
        brms_irr_surf_est_print = get_print_string(brms_irr_surf_est[index_planet],
                                                        brms_irr_surf_est_err[index_planet], 
                                                        "brms",error_sym=error_sym)
        radius_irr_est_print = get_print_string(radius_irr_est[index_planet],
                                                        radius_irr_est_err[index_planet], 
                                                        "radius",error_sym=error_sym)
        

        if HJ_population:
            filename = "./predictions/predictions" + name_cut
        else: 
            filename = "./predictions/predictions" + name_cut + "_" + which_planets
        
        if rotation: 
            if extra_HJ: filename += "_extra/" 
            else: filename += "/" 
        else:
            if extra_HJ: filename += "_extra_no_rot/"
            else: filename += "_no_rot/" 
        filename += name.replace(" ", "_").replace(".", "_")
        if add_systematic_error: filename += "_syst_err"

        header = "      M(MJ)                R(RJ)              F(erg cm-2 s-1)            P(days)             Name"
        
        with open(filename + ".txt", 'w') as f:
            f.writelines(header + '\n')

        if error_sym: space_interpolator = "                     "
        else:  space_interpolator = "         "

        if error_sym_fits: space_fits = "                     "
        else:  space_fits = "         "
        
        with open(filename + ".txt",'a') as f:

            f.writelines(mass_obs_print                 + "      " + \
                        radius_obs_print                + "      " + \
                        flux_obs_print                  + "      " + \
                        orbital_period_obs_print        + "      " + \
                        name                            + "      ")
            
            f.writelines('\n')
            f.writelines('\n')

            f.writelines("eps(%)" + '\n')
            f.writelines("Inner:     " + heat_eff_inner_est_print + '\n')
            f.writelines("Outer:     " + heat_eff_outer_est_print + '\n')
            f.writelines("Th18:      " + heat_eff_thorn_print  +  '\n')
            
            f.writelines('\n')

            f.writelines("                   B_rms,inner(G)                              B_dip,surf,inner(G)"+ '\n')
            f.writelines("Interpolator:     " + brms_inner_est_print      + space_interpolator  + brms_inner_surf_est_print + '\n')
            f.writelines("FitMRL:           " + fits_print["MRL_inner"]   + space_fits          + fits_print["MRL_inner_surf"] + '\n')
            f.writelines("FitMRF:           " + fits_print["MRF_inner"]   + space_fits          + fits_print["MRF_inner_surf"] + '\n')
            f.writelines("FitMR:            " + fits_print["MR_inner"]    + space_fits          + fits_print["MR_inner_surf"] + '\n')
            f.writelines("R09:              " + fits_print["R09_inner"]  + space_fits          + fits_print["R09_inner_surf"] + '\n')
            f.writelines("R09+Th18:         " + fits_print["R09+Th18_inner"] + space_fits          + fits_print["R09+Th18_inner_surf"] + '\n')

            f.writelines('\n')

            f.writelines("                   B_rms,outer(G)                              B_dip,surf,outer(G)"+ '\n')
            f.writelines("Interpolator:     " + brms_outer_est_print      + space_interpolator  + brms_outer_surf_est_print + '\n')
            f.writelines("FitMRL:           " + fits_print["MRL_outer"]   + space_fits          + fits_print["MRL_outer_surf"] + '\n')
            f.writelines("FitMRF:           " + fits_print["MRF_outer"]   + space_fits          + fits_print["MRF_outer_surf"] + '\n')
            f.writelines("FitMR:            " + fits_print["MR_outer"]    + space_fits          + fits_print["MR_outer_surf"] + '\n')

            f.writelines('\n')

            f.writelines("                   B_rms,irr(G)                                B_dip,surf,irr(G)"+ '\n')
            f.writelines("Interpolator:     " + brms_irr_est_print        + space_interpolator                   + brms_irr_surf_est_print + '\n')
            f.writelines("FitMR:            " + fits_print["MR_irr"]      + space_fits               + fits_print["MR_irr_surf"] + '\n')
            f.writelines("Radius of only irradiated planet: " + radius_irr_est_print +    '\n')
            
        
