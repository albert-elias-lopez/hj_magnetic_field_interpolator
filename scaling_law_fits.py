import numpy as np
from scipy.optimize import curve_fit 
from scipy.stats import chisquare
from odrpack import odr_fit

def power_law(vars, K, alpha, beta, gamma):
    lnmass, lnradius, lnconv_lum = vars
    return np.log(K) + lnmass*alpha + lnradius*beta + lnconv_lum*gamma

def power_law_odr(vars, params):
    K, alpha, beta, gamma = params
    lnmass, lnradius, lnconv_lum = vars
    return np.log(K) + lnmass*alpha + lnradius*beta + lnconv_lum*gamma

def power_law_two(vars, K, alpha, beta):
    lnmass, lnradius = vars
    return np.log(K) + lnmass*alpha + lnradius*beta

def power_law_two_odr(vars, params):
    K, alpha, beta = params
    lnmass, lnradius = vars
    return np.log(K) + lnmass*alpha + lnradius*beta

class fit_object:
    def __init__(self, fit_type, fit_function_type):
        self.fit_type = fit_type
        self.fit_function_type = fit_function_type
        self.alpha = 0
        self.beta = 0
        self.gamma = 0
        self.K = 0
        if fit_type == "gaussian":
            self.alpha_err = np.zeros(2)
            self.beta_err = np.zeros(2)
            self.gamma_err = np.zeros(2)
            self.K_err = np.zeros(2)
        else:
            self.alpha_err = 0
            self.beta_err = 0
            self.gamma_err = 0
            self.K_err = 0
        self.chi_2 = 0

    # This funtion writes the lines about the whole fit expressions on the specific file.txt

    def print_fit_results(self, file, filename, header):
        file.writelines(header + '\n')
        if self.fit_function_type == "MRL":
            file.writelines(f"      Brms = {self.K:.4f} G * M^{self.alpha:.4f} * R^{self.beta:.4f} * (eps * pi * R^2 * F_irr)^{self.gamma:.4f}" + '\n')
            if self.fit_type == "gaussian":
                file.writelines(f"             -{self.K_err[0]:.4f}      -{self.alpha_err[0]:.4f}    -{self.beta_err[0]:.4f}                           -{self.gamma_err[0]:.4f}" + '\n')
                file.writelines(f"             +{self.K_err[1]:.4f}      +{self.alpha_err[1]:.4f}    +{self.beta_err[1]:.4f}                           +{self.gamma_err[1]:.4f}" + '\n')
            else:
                file.writelines(f"              {self.K_err:.4f}       {self.alpha_err:.4f}     {self.beta_err:.4f}                            {self.gamma_err:.4f}" + '\n')
            file.writelines(f"      chi_2 = {self.chi_2:.4f} " + '\n')
        elif self.fit_function_type == "MRF":
            file.writelines(f"      Brms = {self.K:.4f} G * M^{self.alpha:.4f} * R^{self.beta:.4f} * (F_irr/1e9)^{self.gamma:.4f}" + '\n')
            if self.fit_type == "gaussian":
                file.writelines(f"             -{self.K_err[0]:.4f}      -{self.alpha_err[0]:.4f}    -{self.beta_err[0]:.4f}               -{self.gamma_err[0]:.4f}" + '\n')
                file.writelines(f"             +{self.K_err[1]:.4f}      +{self.alpha_err[1]:.4f}    +{self.beta_err[1]:.4f}               +{self.gamma_err[1]:.4f}" + '\n')
            else:
                file.writelines(f"             {self.K_err:.4f}      {self.alpha_err:.4f}     {self.beta_err:.4f}                 {self.gamma_err:.4f}" + '\n')
            file.writelines(f"      chi_2 = {self.chi_2:.4f} " + '\n')
        elif self.fit_function_type == "MR":
            file.writelines(f"      Brms = {self.K:.4f} G * M^{self.alpha:.4f} * R^{self.beta:.4f}" + '\n')
            if self.fit_type == "gaussian":
                file.writelines(f"             -{self.K_err[0]:.4f}      -{self.alpha_err[0]:.4f}    -{self.beta_err[0]:.4f}" + '\n')
                file.writelines(f"             +{self.K_err[1]:.4f}      +{self.alpha_err[1]:.4f}    +{self.beta_err[1]:.4f}" + '\n')
            else:
                file.writelines(f"              {self.K_err:.4f}       {self.alpha_err:.4f}     {self.beta_err:.4f}" + '\n')
            file.writelines(f"      chi_2 = {self.chi_2:.4f} " + '\n')
         
        filename += "_" + header + ".txt"
        params  = np.array([self.alpha, self.beta, self.gamma, self.K])
        if self.fit_type == "gaussian":
            errors_minus    = np.array([self.alpha_err[0], self.beta_err[0], self.gamma_err[0], self.K_err[0]])
            errors_plus     = np.array([self.alpha_err[1], self.beta_err[1], self.gamma_err[1], self.K_err[1]])
            np.savetxt(filename, np.vstack((params, errors_minus, errors_plus)))
        else:
            errors          = np.array([self.alpha_err, self.beta_err, self.gamma_err, self.K_err])
            np.savetxt(filename, np.vstack((params, errors)))
    
       
    def read_fit_data(self, filename, header): 
        filename += "_" + header + ".txt"
        loaded = np.loadtxt(filename)
        if self.fit_type == "gaussian":
            params, errors_minus, errors_plus = loaded
            self.alpha, self.beta, self.gamma, self.K = params[0], params[1], params[2], params[3]
            self.alpha_err[0], self.beta_err[0], self.gamma_err[0], self.K_err[0] = errors_minus[0], errors_minus[1], errors_minus[2], errors_minus[3]
            self.alpha_err[1], self.beta_err[1], self.gamma_err[1], self.K_err[1] = errors_plus[0], errors_plus[1], errors_plus[2], errors_plus[3]
        else:
            params, errors = loaded
            self.alpha, self.beta, self.gamma, self.K = params[0], params[1], params[2], params[3]
            self.alpha_err, self.beta_err, self.gamma_err, self.K_err = errors[0], errors[1], errors[2], errors[3]

    

def general_fit(planets,
                valid_mask = [],
                dyn_or_surf = "rms,dyn",
                heating_type = "inner",
                print_all=False,
                error_sym=False,
                p0 = [4, 1/6, -7/6, 1/3],
                bounds=([1e-5, -np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf, np.inf]),
                fit_type = "simple", # can be "simple", "odr", "gaussian"
                fit_function_type = "MRL",
                N_gauss = 10000):
    
    if len(valid_mask) == 0: 
        ones = np.ones(planets.n_planets)
        valid_mask = [i < 2 for i in ones]
    
    fit_to_return = fit_object(fit_type, fit_function_type)
    
    if heating_type == "inner":
        heat_eff_est        = planets.heat_eff_inner_est[valid_mask]
        heat_eff_est_err    = planets.heat_eff_inner_est_err[valid_mask]
        if dyn_or_surf == "rms,dyn" or dyn_or_surf == "eps":
            brms_est            = planets.brms_inner_est[valid_mask]
            brms_est_err        = planets.brms_inner_est_err[valid_mask]
        elif dyn_or_surf == "dip,surf":
            brms_est            = planets.brms_inner_surf_est[valid_mask]
            brms_est_err        = planets.brms_inner_surf_est_err[valid_mask]
    elif heating_type == "outer":
        heat_eff_est        = planets.heat_eff_outer_est[valid_mask]
        heat_eff_est_err    = planets.heat_eff_outer_est_err[valid_mask]
        if dyn_or_surf == "rms,dyn" or dyn_or_surf == "eps":
            brms_est            = planets.brms_outer_est[valid_mask]
            brms_est_err        = planets.brms_outer_est_err[valid_mask]
        elif dyn_or_surf == "dip,surf":
            brms_est            = planets.brms_outer_surf_est[valid_mask]
            brms_est_err        = planets.brms_outer_surf_est_err[valid_mask]
    if heating_type == "irr":
        heat_eff_est        = planets.heat_eff_inner_est[valid_mask]
        heat_eff_est_err    = planets.heat_eff_inner_est_err[valid_mask]
        if dyn_or_surf == "rms,dyn":
            brms_est            = planets.brms_irr_est[valid_mask]
            brms_est_err        = planets.brms_irr_est_err[valid_mask]
        elif dyn_or_surf == "dip,surf":
            brms_est            = planets.brms_irr_surf_est[valid_mask]
            brms_est_err        = planets.brms_irr_surf_est_err[valid_mask]
    
    mass_obs        = planets.mass_obs[valid_mask]
    mass_obs_err    = planets.mass_obs_err[valid_mask]
    flux_obs        = planets.flux_obs[valid_mask]
    flux_obs_err    = planets.flux_obs_err[valid_mask]
    radius_obs      = planets.radius_obs[valid_mask]
    radius_obs_err  = planets.radius_obs_err[valid_mask]
    
    # Remove the predictions with Brms = 0, to not include them for the Brms_outer fits
    valid_mask = brms_est > 0
    heat_eff_est, heat_eff_est_err  = heat_eff_est[valid_mask], heat_eff_est_err[valid_mask]
    brms_est, brms_est_err          = brms_est[valid_mask], brms_est_err[valid_mask]
    mass_obs, mass_obs_err          = mass_obs[valid_mask], mass_obs_err[valid_mask]
    flux_obs, flux_obs_err          = flux_obs[valid_mask], flux_obs_err[valid_mask]
    radius_obs, radius_obs_err      = radius_obs[valid_mask], radius_obs_err[valid_mask]

    if not error_sym:
        heat_eff_est_err = np.mean(heat_eff_est_err, axis=1)
        brms_est_err = np.mean(brms_est_err, axis=1)
    conv_lumin = np.pi*(radius_obs*6.991e+9)**2*flux_obs*heat_eff_est/4.606e24
    conv_lumin_err = conv_lumin*np.sqrt((flux_obs_err/flux_obs)**2+\
                                        (2*radius_obs_err/radius_obs)**2+\
                                        (heat_eff_est_err/heat_eff_est)**2)

    # Define ln variables and their errors
    lnmass = np.log(mass_obs)
    lnmass_err = mass_obs_err / mass_obs

    lnradius = np.log(radius_obs)
    lnradius_err = radius_obs_err / radius_obs

    lnflux = np.log(flux_obs/1e9)
    lnflux_err = flux_obs_err / flux_obs

    lnconv_lum = np.log(conv_lumin)
    lnconv_lum_err = conv_lumin_err / conv_lumin

    lnbrms = np.log(brms_est)
    lnbrms_err = brms_est_err / brms_est

    lnheat = np.log(heat_eff_est)
    lnheat_err = heat_eff_est_err / heat_eff_est


    if fit_function_type == "MRL":
        if fit_type == "simple":
            params, cov = curve_fit(power_law, 
                                    (lnmass, lnradius, lnconv_lum),
                                    lnbrms,
                                    p0=p0,
                                    bounds=bounds)
            errors = np.sqrt(np.diag(cov))

        elif fit_type == "odr":
            valid_mask = (~np.isnan(lnmass)) & (~np.isnan(lnradius)) & (~np.isnan(lnconv_lum)) & (~np.isnan(lnbrms)) & \
                            (~np.isnan(lnmass_err)) & (~np.isnan(lnradius_err)) & (~np.isnan(lnconv_lum_err)) & (~np.isnan(lnbrms_err))

            out = odr_fit(power_law_odr, 
                            [lnmass[valid_mask], lnradius[valid_mask], lnconv_lum[valid_mask]],
                            lnbrms[valid_mask],
                            p0,
                            bounds=bounds,
                            weight_x= [lnmass_err[valid_mask],lnradius_err[valid_mask],lnconv_lum_err[valid_mask]],
                            weight_y= lnbrms_err[valid_mask])

            params = out.beta
            errors = out.sd_beta 

        elif fit_type == "gaussian":
            param_samples = []
            #cov_samples = []
            for _ in range(N_gauss):
                mass_s = np.random.normal(mass_obs, mass_obs_err)
                radius_s = np.random.normal(radius_obs, radius_obs_err)
                conv_lum_s = np.random.normal(conv_lumin, conv_lumin_err)
                brms_s = np.random.normal(brms_est, brms_est_err)

                valid_mask = (mass_s > 0) & (radius_s > 0) & (conv_lum_s > 0) & (brms_s > 0)

                lnmass_s = np.log(mass_s[valid_mask])
                lnradius_s = np.log(radius_s[valid_mask])
                lnconv_lum_s = np.log(conv_lum_s[valid_mask])
                lnbrms_s = np.log(brms_s[valid_mask])

                params, cov = curve_fit(power_law,
                                        (lnmass_s,lnradius_s,lnconv_lum_s),
                                        lnbrms_s,
                                        p0=p0,
                                        bounds=bounds)
                param_samples.append(params)

            param_samples = np.array(param_samples)

            params = param_samples.mean(axis=0)
            p16, p50, p84 = np.percentile(param_samples, q = [16, 50, 84], axis=0)
            errors = p50 - p16, p84 - p50
            errors = np.array(errors).transpose()
            #errors = param_samples.std(axis=0)

        K, alpha, beta, gamma = params

        brms_fit = power_law((lnmass, lnradius, lnconv_lum), K, alpha, beta, gamma)
        #brms_fit = np.exp(brms_fit)
        chi_2 = np.sum((np.log(brms_est)-brms_fit)**2/np.abs(brms_fit))#/len(brms_est)

        if print_all:
            print(f"")
            print(f"      Brms = {K:.4f} * M^{alpha:.4f} * R^{beta:.4f} * (eps * pi * R^2 * F_irr)^{gamma:.4f}")
            print(f"             {errors[0]:.4f}     {errors[1]:.4f}     {errors[2]:.4f}                 {errors[3]:.4f}")
            print(f"      chi^2 = {chi_2:.4f}")
            print(f"----------------------------------------------------------------------")
        else: print("done.  ", end=" ")

        fit_to_return.alpha = alpha
        fit_to_return.beta = beta
        fit_to_return.gamma = gamma
        fit_to_return.K = K
        fit_to_return.alpha_err = errors[1]
        fit_to_return.beta_err = errors[2]
        fit_to_return.gamma_err = errors[3]
        fit_to_return.chi_2 = chi_2
        fit_to_return.K_err = errors[0]
    
    if fit_function_type == "MRF" and dyn_or_surf != "eps":
        if fit_type == "simple":
            params, cov = curve_fit(power_law, 
                                    (lnmass, lnradius, lnflux),#temp_eff_obs/1000
                                    lnbrms,
                                    p0=p0,
                                    bounds=bounds)
            errors = np.sqrt(np.diag(cov))
        
        elif fit_type == "odr":
            valid_mask = (~np.isnan(lnmass)) & (~np.isnan(lnradius)) & (~np.isnan(lnflux)) & (~np.isnan(lnbrms)) & \
                            (~np.isnan(lnmass_err)) & (~np.isnan(lnradius_err)) & (~np.isnan(lnflux_err)) & (~np.isnan(lnbrms_err))

            out = odr_fit(power_law_odr, 
                        [lnmass[valid_mask], lnradius[valid_mask], lnflux[valid_mask]],
                        lnbrms[valid_mask],
                            p0,
                            bounds=bounds,
                            weight_x= [lnmass_err[valid_mask],lnradius_err[valid_mask],lnflux_err[valid_mask]],
                            weight_y= lnbrms_err[valid_mask])

            params = out.beta
            errors = out.sd_beta 

        elif fit_type == "gaussian":
            param_samples = []
            for _ in range(N_gauss):
                mass_s = np.random.normal(mass_obs, mass_obs_err)
                radius_s = np.random.normal(radius_obs, radius_obs_err)
                flux_s = np.random.normal(flux_obs, flux_obs_err)
                brms_s = np.random.normal(brms_est, brms_est_err)

                valid_mask = (mass_s > 0) & (radius_s > 0) & (flux_s > 0) & (brms_s > 0)

                lnmass_s = np.log(mass_s[valid_mask])
                lnradius_s = np.log(radius_s[valid_mask])
                lnflux_s = np.log(flux_s[valid_mask]/1e9)
                lnbrms_s = np.log(brms_s[valid_mask])

                params, cov = curve_fit(power_law,
                                        (lnmass_s,lnradius_s,lnflux_s),
                                        lnbrms_s,
                                        p0=p0,
                                        bounds=bounds)
                param_samples.append(params)

            param_samples = np.array(param_samples)
            params = param_samples.mean(axis=0)
            #errors = param_samples.std(axis=0)
            p16, p50, p84 = np.percentile(param_samples, q = [16, 50, 84], axis=0)
            errors = p50 - p16, p84 - p50
            errors = np.array(errors).transpose()

        K, alpha, beta, gamma = params

        brms_fit = power_law((np.log(mass_obs), np.log(radius_obs), np.log(flux_obs/1e9)), K, alpha, beta, gamma)
        #brms_fit = np.exp(brms_fit)
        chi_2 = np.sum((np.log(brms_est)-brms_fit)**2/np.abs(brms_fit))#/len(brms_est)

        if print_all:
            print(f"")
            print(f"      Brms = {K:.4f} * M^{alpha:.4f} * R^{beta:.4f} * (F_irr)^{gamma:.4f}")
            print(f"             {errors[0]:.4f}     {errors[1]:.4f}     {errors[2]:.4f}                 {errors[3]:.4f}")
            print(f"      chi^2 = {chi_2:.4f}")
            print(f"----------------------------------------------------------------------")
        else: print("done.  ", end=" ")

        fit_to_return.alpha = alpha
        fit_to_return.beta = beta
        fit_to_return.gamma = gamma
        fit_to_return.K = K
        fit_to_return.alpha_err = errors[1]
        fit_to_return.beta_err = errors[2]
        fit_to_return.gamma_err = errors[3]
        fit_to_return.K_err = errors[0]
        fit_to_return.chi_2 = chi_2
    
    if fit_function_type == "MR":
        if fit_type == "simple":
            params, cov = curve_fit(power_law_two, 
                                    (lnmass, lnradius),
                                    np.log(brms_est),
                                    p0=p0,
                                    bounds=bounds)
            errors = np.sqrt(np.diag(cov))
        
        elif fit_type == "odr":
            valid_mask = (~np.isnan(lnmass)) & (~np.isnan(lnradius)) & (~np.isnan(lnbrms)) & \
                            (~np.isnan(lnmass_err)) & (~np.isnan(lnradius_err)) & (~np.isnan(lnbrms_err))

            out = odr_fit(power_law_two_odr, 
                            [lnmass[valid_mask], lnradius[valid_mask]],
                            lnbrms[valid_mask],
                            p0,
                            bounds=bounds,
                            weight_x= [lnmass_err[valid_mask],lnradius_err[valid_mask]],
                            weight_y= lnbrms_err[valid_mask])

            params = out.beta
            errors = out.sd_beta 

        elif fit_type == "gaussian":
            param_samples = []
            for _ in range(N_gauss):
                mass_s = np.random.normal(mass_obs, mass_obs_err)
                radius_s = np.random.normal(radius_obs, radius_obs_err)
                brms_s = np.random.normal(brms_est, brms_est_err)

                valid_mask = (mass_s > 0) & (radius_s > 0) & (brms_s > 0)

                lnmass_s = np.log(mass_s[valid_mask])
                lnradius_s = np.log(radius_s[valid_mask])
                lnbrms_s = np.log(brms_s[valid_mask])

                params, cov = curve_fit(power_law_two,
                                        (lnmass_s,lnradius_s),
                                        lnbrms_s,
                                        p0=p0,
                                        bounds=bounds)
                param_samples.append(params)

            param_samples = np.array(param_samples)
            params = param_samples.mean(axis=0)
            #errors = param_samples.std(axis=0)
            p16, p50, p84 = np.percentile(param_samples, q = [16, 50, 84], axis=0)
            errors = p50 - p16, p84 - p50
            errors = np.array(errors).transpose()

        K, alpha, beta = params

        brms_fit = power_law_two((np.log(mass_obs), np.log(radius_obs)), K, alpha, beta)
        #brms_fit = np.exp(brms_fit)
        chi_2 = np.sum((np.log(brms_est)-brms_fit)**2/np.abs(brms_fit))#/len(brms_est)
        
        if print_all:
            print(f"")
            print(f"      Brms = {K:.4f} * M^{alpha:.4f} * R^{beta:.4f}")
            print(f"             {errors[0]:.4f}     {errors[1]:.4f}     {errors[2]:.4f} ")
            print(f"      chi^2 = {chi_2:.4f}")
            print(f"----------------------------------------------------------------------")
        else: print("done.  ", end=" ")

        fit_to_return.alpha = alpha
        fit_to_return.beta = beta
        fit_to_return.K = K
        fit_to_return.alpha_err = errors[1]
        fit_to_return.beta_err = errors[2]
        fit_to_return.K_err = errors[0]
        fit_to_return.chi_2 = chi_2

    if fit_function_type == "MRF" and dyn_or_surf == "eps":
        if fit_type == "simple":
            params, cov = curve_fit(power_law, 
                                    (lnmass, lnradius, lnflux),#temp_eff_obs/1000
                                    lnheat,
                                    p0=p0,
                                    bounds=bounds)
            errors = np.sqrt(np.diag(cov))
        
        elif fit_type == "odr":
            valid_mask = (~np.isnan(lnmass)) & (~np.isnan(lnradius)) & (~np.isnan(lnflux)) & (~np.isnan(lnheat)) & \
                            (~np.isnan(lnmass_err)) & (~np.isnan(lnradius_err)) & (~np.isnan(lnflux_err)) & (~np.isnan(lnheat_err))

            out = odr_fit(power_law_odr, 
                            [lnmass[valid_mask], lnradius[valid_mask], lnflux[valid_mask]],
                            lnheat[valid_mask],
                            p0,
                            bounds=bounds,
                            weight_x= [lnmass_err[valid_mask],lnradius_err[valid_mask],lnflux_err[valid_mask]],
                            weight_y= lnheat_err[valid_mask])

            params = out.beta
            errors = out.sd_beta 

        elif fit_type == "gaussian":
            param_samples = []
            for _ in range(N_gauss):
                mass_s = np.random.normal(mass_obs, mass_obs_err)
                radius_s = np.random.normal(radius_obs, radius_obs_err)
                flux_s = np.random.normal(flux_obs, flux_obs_err)
                heat_eff_s = np.random.normal(heat_eff_est, heat_eff_est_err)

                valid_mask = (mass_s > 0) & (radius_s > 0) & (flux_s > 0) & (heat_eff_s > 0)

                lnmass_s = np.log(mass_s[valid_mask])
                lnradius_s = np.log(radius_s[valid_mask])
                lnflux_s = np.log(flux_s[valid_mask]/1e9)
                lnheat_eff_s = np.log(heat_eff_s[valid_mask])

                params, cov = curve_fit(power_law,
                                        (lnmass_s,lnradius_s,lnflux_s),
                                        lnheat_eff_s,
                                        p0=p0,
                                        bounds=bounds)
                param_samples.append(params)

            param_samples = np.array(param_samples)
            params = param_samples.mean(axis=0)
            #errors = param_samples.std(axis=0)
            p16, p50, p84 = np.percentile(param_samples, q = [16, 50, 84], axis=0)
            errors = p50 - p16, p84 - p50
            errors = np.array(errors).transpose()

        K, alpha, beta, gamma = params

        eps_fit = power_law((np.log(mass_obs), np.log(radius_obs), np.log(flux_obs/1e9)), K, alpha, beta, gamma)
        #brms_fit = np.exp(brms_fit)
        chi_2 = np.sum((np.log(heat_eff_est)-eps_fit)**2/np.abs(eps_fit))#/len(brms_est)
        #eps_fit = np.exp(eps_fit)
        #chi_2 = np.sum((heat_eff_est-eps_fit)**2/eps_fit)/len(heat_eff_est)
        
        if print_all:
            print(f"")
            print(f"    eps_eff = {K:.4e} * M^{alpha:.4f} * R^{beta:.4f} * F_irr^{gamma:.4f}")
            print(f"             {errors[0]:.4e}     {errors[1]:.4f}     {errors[2]:.4f}          {errors[3]:.4f}")
            print(f"      chi^2 = {chi_2:.4f}")
            print(f"----------------------------------------------------------------------")
        else: print("done.  ", end=" ")

        fit_to_return.alpha = alpha
        fit_to_return.beta = beta
        fit_to_return.gamma = gamma
        fit_to_return.K = K
        fit_to_return.alpha_err = errors[1]
        fit_to_return.beta_err = errors[2]
        fit_to_return.gamma_err = errors[3]
        fit_to_return.K_err = errors[0]
        fit_to_return.chi_2 = chi_2

    return fit_to_return



def eps_fit(heat_eff_est, heat_eff_est_err, 
            brms_est, brms_est_err,
            mass_obs, mass_obs_err,
            flux_obs, flux_obs_err,
            radius_obs, radius_obs_err,
            print_all=False,
            error_sym=False,
            p0 = [4, 1/6, -7/6, 1/3],
            bounds=([1e-5, -np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf, np.inf]),
            fit_type = "simple",
            fit_function_type = "MRF",
            N_gauss = 10000):
    
    fit_to_return = fit_object(fit_type, fit_function_type)

    # Remove the predictions with Brms = 0, to not include them for the Brms_outer fits
    valid_mask = brms_est > 0
    heat_eff_est, heat_eff_est_err  = heat_eff_est[valid_mask], heat_eff_est_err[valid_mask]
    brms_est, brms_est_err          = brms_est[valid_mask], brms_est_err[valid_mask]
    mass_obs, mass_obs_err          = mass_obs[valid_mask], mass_obs_err[valid_mask]
    flux_obs, flux_obs_err          = flux_obs[valid_mask], flux_obs_err[valid_mask]
    radius_obs, radius_obs_err      = radius_obs[valid_mask], radius_obs_err[valid_mask]
    
    if not error_sym:
        heat_eff_est_err = np.mean(heat_eff_est_err, axis=1)
        brms_est_err = np.mean(brms_est_err, axis=1)

    # Define ln variables and their errors
    lnmass = np.log(mass_obs)
    lnradius = np.log(radius_obs)
    lnflux = np.log(flux_obs/1e9)
    lnheat = np.log(heat_eff_est)
    lnmass_err = mass_obs_err / mass_obs
    lnradius_err = radius_obs_err / radius_obs
    lnflux_err = flux_obs_err / flux_obs
    lnheat_err = heat_eff_est_err / heat_eff_est

    if fit_type == "simple":
        params, cov = curve_fit(power_law, 
                                (lnmass, lnradius, lnflux),#temp_eff_obs/1000
                                lnheat,
                                p0=p0,
                                bounds=bounds)
        errors = np.sqrt(np.diag(cov))
    
    elif fit_type == "odr":
        valid_mask = (~np.isnan(lnmass)) & (~np.isnan(lnradius)) & (~np.isnan(lnflux)) & (~np.isnan(lnheat)) & \
                        (~np.isnan(lnmass_err)) & (~np.isnan(lnradius_err)) & (~np.isnan(lnflux_err)) & (~np.isnan(lnheat_err))

        out = odr_fit(power_law_odr, 
                        [lnmass[valid_mask], lnradius[valid_mask], lnflux[valid_mask]],
                        lnheat[valid_mask],
                        p0,
                        bounds=bounds,
                        weight_x= [lnmass_err[valid_mask],lnradius_err[valid_mask],lnflux_err[valid_mask]],
                        weight_y= lnheat_err[valid_mask])

        params = out.beta
        errors = out.sd_beta 

    elif fit_type == "gaussian":
        param_samples = []
        for _ in range(N_gauss):
            mass_s = np.random.normal(mass_obs, mass_obs_err)
            radius_s = np.random.normal(radius_obs, radius_obs_err)
            flux_s = np.random.normal(flux_obs, flux_obs_err)
            heat_eff_s = np.random.normal(heat_eff_est, heat_eff_est_err)

            valid_mask = (mass_s > 0) & (radius_s > 0) & (flux_s > 0) & (heat_eff_s > 0)

            lnmass_s = np.log(mass_s[valid_mask])
            lnradius_s = np.log(radius_s[valid_mask])
            lnflux_s = np.log(flux_s[valid_mask]/1e9)
            lnheat_eff_s = np.log(heat_eff_s[valid_mask])

            params, cov = curve_fit(power_law,
                                    (lnmass_s,lnradius_s,lnflux_s),
                                    lnheat_eff_s,
                                    p0=p0,
                                    bounds=bounds)
            param_samples.append(params)

        param_samples = np.array(param_samples)
        params = param_samples.mean(axis=0)
        #errors = param_samples.std(axis=0)
        p16, p50, p84 = np.percentile(param_samples, q = [16, 50, 84], axis=0)
        errors = p50 - p16, p84 - p50
        errors = np.array(errors).transpose()

    K, alpha, beta, gamma = params

    eps_fit = power_law((np.log(mass_obs), np.log(radius_obs), np.log(flux_obs/1e9)), K, alpha, beta, gamma)
    #brms_fit = np.exp(brms_fit)
    chi_2 = np.sum((np.log(heat_eff_est)-eps_fit)**2/np.abs(eps_fit))#/len(brms_est)
    #eps_fit = np.exp(eps_fit)
    #chi_2 = np.sum((heat_eff_est-eps_fit)**2/eps_fit)/len(heat_eff_est)
    
    if print_all:
        print(f"")
        print(f"    eps_eff = {K:.4e} * M^{alpha:.4f} * R^{beta:.4f} * F_irr^{gamma:.4f}")
        print(f"             {errors[0]:.4e}     {errors[1]:.4f}     {errors[2]:.4f}          {errors[3]:.4f}")
        print(f"      chi^2 = {chi_2:.4f}")
        print(f"----------------------------------------------------------------------")
    else: print("done.  ", end=" ")

    fit_to_return.alpha = alpha
    fit_to_return.beta = beta
    fit_to_return.gamma = gamma
    fit_to_return.K = K
    fit_to_return.alpha_err = errors[1]
    fit_to_return.beta_err = errors[2]
    fit_to_return.gamma_err = errors[3]
    fit_to_return.K_err = errors[0]
    fit_to_return.chi_2 = chi_2

    return fit_to_return





def eps_exp_funct(flux, eps0, mu, sigma):
    return np.log(eps0) - (np.log10(flux/1e9)-mu)**2/2/sigma**2

def eps_fit_exp(heat_eff_est, heat_eff_est_err,
                    flux_obs, flux_obs_err,
                    print_all=False,
                    error_sym=False,
                    p0 = [0.023, 0.14, 0.37],
                    bounds=([1e-4, 1e-3, 0.01], [5, 5, 5]),
                    fit_type = "simple",
                    N_gauss = 10000):
    
    if not error_sym: heat_eff_est_err = np.mean(heat_eff_est_err, axis=1)

    # Define ln variables and their errors
    lnheat = np.log(heat_eff_est)
    lnheat_err = heat_eff_est_err / heat_eff_est

    if fit_type == "simple":
        valid_mask = (~np.isnan(lnheat)) & (~np.isnan(flux_obs)) & (flux_obs > 0)

        params, cov = curve_fit(eps_exp_funct, 
                                flux_obs[valid_mask],
                                lnheat[valid_mask],
                                p0=p0,
                                bounds=bounds)
        errors = np.sqrt(np.diag(cov))
    
    
    elif fit_type == "odr":
        valid_mask = (~np.isnan(lnheat)) & (~np.isnan(lnheat_err))

        out = odr_fit(eps_exp_funct, 
                        flux_obs[valid_mask],
                        lnheat[valid_mask],
                        p0,
                        bounds=bounds,
                        weight_x = [flux_obs_err[valid_mask]],
                        weight_y = lnheat_err[valid_mask])

        params = out.beta
        errors = out.sd_beta 

    elif fit_type == "gaussian":
        param_samples = []
        for _ in range(N_gauss):
            flux_s = np.random.normal(flux_obs, flux_obs_err)
            heat_eff_s = np.random.normal(heat_eff_est, heat_eff_est_err)

            valid_mask = (flux_s > 0) & (heat_eff_s > 0)

            params, cov = curve_fit(eps_exp_funct,
                                    flux_s[valid_mask],
                                    lnheat[valid_mask],
                                    p0=p0,
                                    bounds=bounds)
            param_samples.append(params)

        param_samples = np.array(param_samples)
        params = param_samples.mean(axis=0)
        #errors = param_samples.std(axis=0)
        p16, p50, p84 = np.percentile(param_samples, q = [16, 50, 84], axis=0)
        errors = p50 - p16, p84 - p50
        errors = np.array(errors).transpose()
    
    eps0, mu, sigma = params

    eps_fit = eps_exp_funct(flux_obs, eps0, mu, sigma)
    chi_2 = np.sum((np.log(heat_eff_est)-eps_fit)**2/np.abs(eps_fit))
    
    if print_all:
        print(f"")
        print(f"    eps_eff = {eps0:.4e} * exp^( log10(F_irr) - {mu:.4f})^2 / 2*{sigma:.4f}^2 )")
        print(f"             {errors[0]:.4e}           {errors[1]:.4f}     {errors[2]:.4f} ")
        print(f"      chi^2 = {chi_2:.4f}")
        print(f"----------------------------------------------------------------------")
    else: print("done.  ", end=" ")

    return mu, sigma, eps0, errors[1], errors[2], errors[0], chi_2









def power_law_exp(vars, K, alpha, beta, gamma, mu, sigma):
    lnmass, lnradius, lnflux = vars
    flux = np.exp(lnflux)
    return np.log(K) + lnmass*alpha + lnradius*beta + lnflux*gamma - gamma*(np.log10(flux)-mu)**2/2/sigma**2


def general_fit_exp(heat_eff_est, heat_eff_est_err,
                    brms_est,brms_est_err,
                    mass_obs, mass_obs_err,
                    flux_obs, flux_obs_err,
                    radius_obs, radius_obs_err,
                    print_all=False,
                    error_sym=False,
                    p0 = [4, 1/6, -7/6, 1/3, 0.14, 0.37],
                    bounds=([1e-5, -20, -20, -20, 1e-3, 0.01], [1e4, 20, 20, 20, 5, 5]),
                    fit_type = "simple",
                    N_gauss = 10000):

    # Remove the predictions with Brms = 0, to not include them for the Brms_outer fits
    valid_mask = brms_est > 0
    heat_eff_est, heat_eff_est_err  = heat_eff_est[valid_mask], heat_eff_est_err[valid_mask]
    brms_est, brms_est_err          = brms_est[valid_mask], brms_est_err[valid_mask]
    mass_obs, mass_obs_err          = mass_obs[valid_mask], mass_obs_err[valid_mask]
    flux_obs, flux_obs_err          = flux_obs[valid_mask], flux_obs_err[valid_mask]
    radius_obs, radius_obs_err      = radius_obs[valid_mask], radius_obs_err[valid_mask]
    
    if error_sym:
        brms_est_err = brms_est_err
    else: 
        heat_eff_est_err = np.mean(heat_eff_est_err, axis=1)
        brms_est_err = np.mean(brms_est_err, axis=1)

    # Define ln variables and their errors
    lnmass = np.log(mass_obs)
    lnradius = np.log(radius_obs)
    lnflux = np.log(flux_obs/1e9)
    lnheat = np.log(heat_eff_est)
    lnmass_err = mass_obs_err / mass_obs
    lnradius_err = radius_obs_err / radius_obs
    lnflux_err = flux_obs_err / flux_obs
    lnheat_err = heat_eff_est_err / heat_eff_est

    if fit_type == "simple":
        params, cov = curve_fit(power_law_exp, 
                                (np.log(mass_obs), np.log(radius_obs), np.log(flux_obs/1e9)),#temp_eff_obs/1000
                                np.log(heat_eff_est),
                                p0=p0,
                                bounds=bounds,
                                maxfev=10000)
        errors = np.sqrt(np.diag(cov))
    
    elif fit_type == "odr":
        valid_mask = (~np.isnan(lnmass)) & (~np.isnan(lnradius)) & (~np.isnan(lnflux)) & (~np.isnan(lnheat)) & \
                        (~np.isnan(lnmass_err)) & (~np.isnan(lnradius_err)) & (~np.isnan(lnflux_err)) & (~np.isnan(lnheat_err))

        out = odr_fit(power_law_exp, 
                        [lnmass[valid_mask], lnradius[valid_mask], lnflux[valid_mask]],
                        lnheat[valid_mask],
                        p0,
                        bounds=bounds,
                        weight_x= [lnmass_err[valid_mask],lnradius_err[valid_mask],lnflux_err[valid_mask]],
                        weight_y= lnheat_err[valid_mask])

        params = out.beta
        errors = out.sd_beta 

    elif fit_type == "gaussian":
        param_samples = []
        for _ in range(N_gauss):
            mass_s = np.random.normal(mass_obs, mass_obs_err)
            radius_s = np.random.normal(radius_obs, radius_obs_err)
            flux_s = np.random.normal(flux_obs, flux_obs_err)
            heat_eff_s = np.random.normal(heat_eff_est, heat_eff_est_err)

            valid_mask = (mass_s > 0) & (radius_s > 0) & (flux_s > 0) & (heat_eff_s > 0)

            lnmass_s = np.log(mass_s[valid_mask])
            lnradius_s = np.log(radius_s[valid_mask])
            lnflux_s = np.log(flux_s[valid_mask]/1e9)
            lnheat_eff_s = np.log(heat_eff_s[valid_mask])

            params, cov = curve_fit(power_law_exp,
                                    (lnmass_s,lnradius_s,lnflux_s),
                                    lnheat_eff_s,
                                    p0=p0,
                                    bounds=bounds,
                                    maxfev=10000)
            param_samples.append(params)

        param_samples = np.array(param_samples)
        params = param_samples.mean(axis=0)
        #errors = param_samples.std(axis=0)
        p16, p50, p84 = np.percentile(param_samples, q = [16, 50, 84], axis=0)
        errors = p50 - p16, p84 - p50
        errors = np.array(errors).transpose()

    K, alpha, beta, gamma, mu, sigma = params
        
    eps_fit = power_law_exp((lnmass, lnradius, lnflux), K, alpha, beta, gamma, mu, sigma)
    #brms_fit = np.exp(brms_fit)
    chi_2 = np.sum((np.log(heat_eff_est)-eps_fit)**2/np.abs(eps_fit))#/len(brms_est)
    #eps_fit = np.exp(eps_fit)
    #chi_2 = np.sum((heat_eff_est-eps_fit)**2/eps_fit)/len(heat_eff_est)
    
    if print_all:
        print(f"")
        print(f"    eps_eff = {K:.4e} * M^{alpha:.4f} * R^{beta:.4f} * (F_irr * exp^( log10(F_irr) - {mu:.4f})^2 / 2*{sigma:.4f}^2 ) *)^{gamma:.4f}")
        print(f"       {errors[0]:.4e}   {errors[1]:.4f}   {errors[2]:.4f}                      {errors[4]:.4f}    {errors[5]:.4f}      {errors[3]:.4f}")
        print(f"      chi^2 = {chi_2:.4f}")
        print(f"----------------------------------------------------------------------")
    else: print("done.  ", end=" ")

    return alpha, beta, gamma, mu, sigma, K, errors[1], errors[2], errors[3], errors[4], errors[5], errors[0], chi_2
