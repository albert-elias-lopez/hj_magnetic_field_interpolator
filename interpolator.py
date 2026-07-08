import sys
import os
import numpy as np
from scipy.interpolate import LinearNDInterpolator
import emcee
from tqdm import tqdm
import table_predictions_save

def get_error(interpolator,
              mass, mass_err,
              flux, flux_err,
              radius, radius_err,
              eps=1e-5):

    est_df_up    = interpolator(mass,np.log10(flux)+eps,radius)
    est_df_down  = interpolator(mass,np.log10(flux)-eps,radius)
    d_est_df = (est_df_up - est_df_down)/2/eps
    
    est_dm_up    = interpolator(mass+eps,np.log10(flux),radius)
    est_dm_down  = interpolator(mass-eps,np.log10(flux),radius)
    d_est_dm = (est_dm_up - est_dm_down)/2/eps
    
    est_dr_up    = interpolator(mass,np.log10(flux),radius+eps)
    est_dr_down  = interpolator(mass,np.log10(flux),radius-eps)
    d_est_dr = (est_dr_up - est_dr_down)/2/eps

    log10_flux_err = flux_err/np.log(10)/flux
    
    est_err = d_est_df**2*log10_flux_err**2+\
              d_est_dm**2*mass_err**2+\
              d_est_dr**2*radius_err**2
    
    return np.sqrt(est_err)

def get_error_gaussian(interpolator,
                        mass, mass_err,
                        flux, flux_err,
                        radius, radius_err,
                        radius_irr,
                        N = 10000,
                        min_mass = 0.3, max_mass = 12,
                        min_flux = 1e7, max_flux = 1.58e10):
    
    if np.isnan(mass_err):
        masses_s = mass*np.ones(N)
    else:
        masses_s = np.random.normal(mass, mass_err, N)

    if np.isnan(flux_err):
        fluxes_s = flux*np.ones(N)
    else:
        fluxes_s = np.random.normal(flux, flux_err, N)

    if np.isnan(radius_err):
        radii_s = radius*np.ones(N)
    else:
        radii_s = np.random.normal(radius, radius_err, N)

    mask = (masses_s > min_mass) & (masses_s < max_mass) & (fluxes_s > min_flux) & (fluxes_s < max_flux) & (radii_s > radius_irr)
    samples = interpolator(masses_s[mask],np.log10(fluxes_s[mask]),radii_s[mask])

    mask_nans = ~np.isnan(samples)
    samples = samples[mask_nans]
    p16, p50, p84 = np.percentile(samples, [16, 50, 84])

    #acceptance = np.sum(mask)/N
    #print("Acceptance: " + str(acceptance))
    
    return p50 - p16, p84 - p50 # sample estimate, estimate error minus, estimate error plus

def log_prior(theta,
            mass, mass_err,
            flux, flux_err,
            radius, radius_err,
            radius_irr,
            min_mass = 0.3, max_mass = 12,
            min_flux = 1e7, max_flux = 1.58e10):

    M, F, R = theta
    # physical + interpolator bounds
    if not (min_mass < M < max_mass):
        return -np.inf
    if not (min_flux < F < max_flux):
        return -np.inf
    if not (R > radius_irr):
        return -np.inf
    lp = 0.0

    # Gaussian priors (skip if no error)
    if not np.isnan(mass_err):
        lp += -0.5 * ((M - mass)/mass_err)**2

    if not np.isnan(flux_err):
        if F <= 0:
            return -np.inf
        lp += -0.5 * ((F - flux)/flux_err)**2

    if not np.isnan(radius_err):
        lp += -0.5 * ((R - radius)/radius_err)**2

    return lp

def get_error_mcmc(interpolator,
                    mass, mass_err,
                    flux, flux_err,
                    radius, radius_err,
                    radius_irr,
                    min_mass = 0.3, max_mass = 12,
                    min_flux = 1e7, max_flux = 1.58e10,
                    nwalkers=32,
                    nsteps=4000,
                    burnin=1000):

    def log_probability(theta):
        return log_prior(theta,
                        mass, mass_err,
                        flux, flux_err,
                        radius, radius_err,
                        radius_irr,
                        min_mass = min_mass, max_mass = max_mass,
                        min_flux = min_flux, max_flux = max_flux)

    # -------------------------------
    # Initial walker positions
    # -------------------------------
    ndim = 3

    pos = np.zeros((nwalkers, ndim))

    for i in range(nwalkers):
        pos[i,0] = mass   if np.isnan(mass_err)   else np.random.normal(mass, mass_err)
        pos[i,1] = flux   if np.isnan(flux_err)   else np.random.normal(flux, flux_err)
        pos[i,2] = radius if np.isnan(radius_err) else np.random.normal(radius, radius_err)

    # -------------------------------
    # Run MCMC
    # -------------------------------
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability)
    sampler.run_mcmc(pos, nsteps, progress=False)
    samples = sampler.get_chain(discard=burnin, flat=True)
    M_s = samples[:,0]
    F_s = samples[:,1]
    R_s = samples[:,2]

    # -------------------------------
    # Apply interpolator
    # -------------------------------
    valid = (F_s > 0)
    M_s = M_s[valid]
    F_s = F_s[valid]
    R_s = R_s[valid]

    B_s = interpolator(M_s, np.log10(F_s), R_s)
    B_s = B_s[np.isfinite(B_s)]

    p16, p50, p84 = np.percentile(B_s, [16, 50, 84])

    return p50 - p16, p84 - p50 # estimate error minus, estimate error plus


def get_error_irr(interpolator,
              mass, mass_err,
              flux, flux_err,
              eps=1e-5):

    est_df_up    = interpolator(mass,np.log10(flux)+eps)
    est_df_down  = interpolator(mass,np.log10(flux)-eps)
    d_est_df = (est_df_up - est_df_down)/2/eps
    
    est_dm_up    = interpolator(mass+eps,np.log10(flux))
    est_dm_down  = interpolator(mass-eps,np.log10(flux))
    d_est_dm = (est_dm_up - est_dm_down)/2/eps

    log10_flux_err = flux_err/np.log(10)/flux
    
    est_err = d_est_df**2*log10_flux_err**2+\
              d_est_dm**2*mass_err**2
    
    return np.sqrt(est_err)


def get_error_irr_gaussian(interpolator,
                        mass, mass_err,
                        flux, flux_err,
                        N = 10000,
                        min_mass = 0.3, max_mass = 12,
                        min_flux = 1e7, max_flux = 1.58e10):
    
    if np.isnan(mass_err):
        masses_s = mass*np.ones(N)
    else:
        masses_s = np.random.normal(mass, mass_err, N)

    if np.isnan(flux_err):
        fluxes_s = flux*np.ones(N)
    else:
        fluxes_s = np.random.normal(flux, flux_err, N)

    mask = (masses_s > min_mass) & (masses_s < max_mass) & (fluxes_s > min_flux) & (fluxes_s < max_flux)
    samples = interpolator(masses_s[mask],np.log10(fluxes_s[mask]))

    mask_nans = ~np.isnan(samples)
    samples = samples[mask_nans]
    p16, p50, p84 = np.percentile(samples, [16, 50, 84])

    return p50 - p16, p84 - p50 # estimate error minus, estimate error plus
 
def linear_interpolator(table_inner, 
                        table_outer,
                        planets,
                        mask_predictions,
                        print_all = True,
                        error_sym = False):
    
    # Save the epsilon = 0 data to interpolate and give estimates for the only irradiated models
    masses_irr_flat = table_inner.masses_grid_total[:,:,0].flatten()
    flux_irr_flat = table_inner.flux_grid_total[:,:,0].flatten()
    # It doesn't matter to take the radius and magnetic field estimation from inner or outer, as eps=0
    radii_irr_flat = table_inner.radius_heat[:,:,0].flatten()
    bscaling_irr_flat = table_inner.bscaling_heat[:,:,0].flatten()
    bscaling_surf_irr_flat = table_inner.bscaling_surf_heat[:,:,0].flatten()

    # Fatten the data for the interpolator
    masses_grid_total_flat              = table_inner.masses_grid_total.flatten()
    flux_grid_total_flat                = table_inner.flux_grid_total.flatten()
    heat_effs_grid_total_flat           = table_inner.heat_effs_grid_total.flatten()
    radii_heat_inner_flat               = table_inner.radius_heat.flatten()
    radii_heat_outer_flat               = table_outer.radius_heat.flatten()
    bscaling_heat_inner_flat            = table_inner.bscaling_heat.flatten()
    bscaling_heat_outer_flat            = table_outer.bscaling_heat.flatten()
    bscaling_surf_heat_inner_flat       = table_inner.bscaling_surf_heat.flatten()
    bscaling_surf_heat_outer_flat       = table_outer.bscaling_surf_heat.flatten()

    # Create a mask for valid simulation points (radius > 0)
    valid_mask_inner = radii_heat_inner_flat > 0
    masses_grid_inner_flat           = masses_grid_total_flat[valid_mask_inner]
    flux_grid_inner_flat             = flux_grid_total_flat[valid_mask_inner]
    heat_effs_grid_inner_flat        = heat_effs_grid_total_flat[valid_mask_inner]
    radii_heat_inner_flat            = radii_heat_inner_flat[valid_mask_inner]
    bscaling_heat_inner_flat         = bscaling_heat_inner_flat[valid_mask_inner]
    bscaling_surf_heat_inner_flat    = bscaling_surf_heat_inner_flat[valid_mask_inner]

    # Repeat the same for the outer heated models
    valid_mask_outer = radii_heat_outer_flat > 0
    masses_grid_outer_flat              = masses_grid_total_flat[valid_mask_outer]
    flux_grid_outer_flat                = flux_grid_total_flat[valid_mask_outer]
    heat_effs_grid_outer_flat           = heat_effs_grid_total_flat[valid_mask_outer]
    radii_heat_outer_flat               = radii_heat_outer_flat[valid_mask_outer]
    bscaling_heat_outer_flat            = bscaling_heat_outer_flat[valid_mask_outer]
    bscaling_surf_heat_outer_flat       = bscaling_surf_heat_outer_flat[valid_mask_outer]

    # Repeat the same of the irradiated models (some small planets with high irradiations explode, even without inflation)
    valid_mask_irr = radii_irr_flat > 0
    masses_irr_flat = masses_irr_flat[valid_mask_irr] 
    flux_irr_flat = flux_irr_flat[valid_mask_irr] 
    radii_irr_flat = radii_irr_flat[valid_mask_irr]  
    bscaling_irr_flat = bscaling_irr_flat[valid_mask_irr]  
    bscaling_surf_irr_flat = bscaling_surf_irr_flat[valid_mask_irr]

    linear_interp_heat_effs_inner = LinearNDInterpolator(list(zip(masses_grid_inner_flat, 
                                                                     np.log10(flux_grid_inner_flat), 
                                                                     radii_heat_inner_flat)), 
                                                                     heat_effs_grid_inner_flat)
    linear_interp_brms_inner = LinearNDInterpolator(list(zip(masses_grid_inner_flat, 
                                                                np.log10(flux_grid_inner_flat), 
                                                                radii_heat_inner_flat)), 
                                                                bscaling_heat_inner_flat)
    linear_interp_brms_surf_inner = LinearNDInterpolator(list(zip(masses_grid_inner_flat, 
                                                                     np.log10(flux_grid_inner_flat), 
                                                                     radii_heat_inner_flat)), 
                                                                     bscaling_surf_heat_inner_flat)

    linear_interp_heat_effs_outer = LinearNDInterpolator(list(zip(masses_grid_outer_flat, 
                                                                  np.log10(flux_grid_outer_flat), 
                                                                  radii_heat_outer_flat)), 
                                                                  heat_effs_grid_outer_flat)
    linear_interp_brms_outer = LinearNDInterpolator(list(zip(masses_grid_outer_flat, 
                                                             np.log10(flux_grid_outer_flat), 
                                                             radii_heat_outer_flat)), 
                                                             bscaling_heat_outer_flat)
    linear_interp_brms_surf_outer = LinearNDInterpolator(list(zip(masses_grid_outer_flat, 
                                                                  np.log10(flux_grid_outer_flat), 
                                                                  radii_heat_outer_flat)), 
                                                                  bscaling_surf_heat_outer_flat)
    

    linear_interp_radius_irr = LinearNDInterpolator(list(zip(masses_irr_flat, 
                                                                  np.log10(flux_irr_flat))), 
                                                                  radii_irr_flat)
    linear_interp_brms_irr = LinearNDInterpolator(list(zip(masses_irr_flat, 
                                                                  np.log10(flux_irr_flat))), 
                                                                  bscaling_irr_flat)
    linear_interp_brms_surf_irr = LinearNDInterpolator(list(zip(masses_irr_flat, 
                                                                  np.log10(flux_irr_flat))), 
                                                                  bscaling_surf_irr_flat)


    index_loop=0

    if print_all: print(f"----------------------------------------------------------------------")
    
    #for index_planet, name in enumerate(planet_names):
    for index_planet in tqdm(range(planets.n_planets)):
        name = planets.planet_names[index_planet]

        mass = planets.mass_obs[index_planet]
        mass_err = planets.mass_obs_err[index_planet]
        flux = planets.flux_obs[index_planet]
        flux_err = planets.flux_obs_err[index_planet]
        radius = planets.radius_obs[index_planet]
        radius_err = planets.radius_obs_err[index_planet]
        
        if mask_predictions[index_planet]:

            planets.radius_irr_est[index_planet]    = linear_interp_radius_irr(mass,np.log10(flux))
            if error_sym: planets.radius_irr_est_err[index_planet] =  get_error_irr(linear_interp_radius_irr, mass, mass_err, flux, flux_err)
            else: planets.radius_irr_est_err[index_planet, 0], planets.radius_irr_est_err[index_planet, 1]  =\
                                                        get_error_irr_gaussian(linear_interp_radius_irr, mass, mass_err,flux, flux_err)
            
            if radius < planets.radius_irr_est[index_planet]:
                planets.brms_irr_est[index_planet]                    = np.nan
                planets.brms_irr_surf_est[index_planet]         = np.nan
                planets.heat_eff_inner_est[index_planet]     = np.nan
                planets.brms_inner_est[index_planet]         = np.nan
                planets.brms_inner_surf_est[index_planet]    = np.nan
                planets.heat_eff_outer_est[index_planet]        = np.nan
                planets.brms_outer_est[index_planet]            = np.nan
                planets.brms_outer_surf_est[index_planet]       = np.nan

                if error_sym: 
                    planets.brms_irr_est_err[index_planet]  = np.nan
                    planets.brms_irr_surf_est_err[index_planet] = np.nan
                    planets.heat_eff_inner_est_err[index_planet] = np.nan
                    planets.brms_inner_est_err[index_planet]     = np.nan
                    planets.brms_inner_surf_est_err[index_planet]= np.nan
                    planets.heat_eff_outer_est_err[index_planet]    = np.nan
                    planets.brms_outer_est_err[index_planet]        = np.nan
                    planets.brms_outer_surf_est_err[index_planet]   = np.nan
                else: 
                    planets.brms_irr_est_err[index_planet, :] = np.nan
                    planets.brms_irr_surf_est_err[index_planet, :] = np.nan
                    planets.heat_eff_inner_est_err[index_planet, :] = np.nan
                    planets.brms_inner_est_err[index_planet, :]     = np.nan
                    planets.brms_inner_surf_est_err[index_planet, :]= np.nan
                    planets.heat_eff_outer_est_err[index_planet, :]    = np.nan
                    planets.brms_outer_est_err[index_planet, :]        = np.nan
                    planets.brms_outer_surf_est_err[index_planet, :]   = np.nan

            else:
                planets.brms_irr_est[index_planet]        = linear_interp_brms_irr(mass,np.log10(flux))
                planets.brms_irr_surf_est[index_planet]   = linear_interp_brms_surf_irr(mass,np.log10(flux))

                planets.heat_eff_inner_est[index_planet]     = linear_interp_heat_effs_inner(mass,np.log10(flux),radius) 
                planets.brms_inner_est[index_planet]         = linear_interp_brms_inner(mass,np.log10(flux),radius)
                planets.brms_inner_surf_est[index_planet]    = linear_interp_brms_surf_inner(mass,np.log10(flux),radius)

                planets.heat_eff_outer_est[index_planet]    = linear_interp_heat_effs_outer(mass,np.log10(flux),radius)
                planets.brms_outer_est[index_planet]        = linear_interp_brms_outer(mass,np.log10(flux),radius)
                planets.brms_outer_surf_est[index_planet]   = linear_interp_brms_surf_outer(mass,np.log10(flux),radius)

                if error_sym: 
                    planets.brms_irr_est_err[index_planet] =  get_error_irr(linear_interp_brms_irr, mass, mass_err, flux, flux_err)
                    planets.brms_irr_surf_est_err[index_planet] =  get_error_irr(linear_interp_brms_surf_irr, mass, mass_err, flux, flux_err)

                    planets.heat_eff_inner_est_err[index_planet] =  get_error(linear_interp_heat_effs_inner, mass, mass_err,flux, flux_err, radius, radius_err)
                    planets.brms_inner_est_err[index_planet] =  get_error(linear_interp_brms_inner, mass, mass_err,flux, flux_err, radius, radius_err)
                    planets.brms_inner_surf_est_err[index_planet] =  get_error(linear_interp_brms_surf_inner, mass, mass_err,flux, flux_err, radius, radius_err)  
                    
                    planets.heat_eff_outer_est_err[index_planet] =  get_error(linear_interp_heat_effs_outer, mass, mass_err,flux, flux_err, radius, radius_err)
                    planets.brms_outer_est_err[index_planet] =  get_error(linear_interp_brms_outer, mass, mass_err,flux, flux_err,radius, radius_err)
                    planets.brms_outer_surf_est_err[index_planet] =  get_error(linear_interp_brms_surf_outer, mass, mass_err,flux, flux_err, radius, radius_err)
                
                else:
                    planets.brms_irr_est_err[index_planet, 0], planets.brms_irr_est_err[index_planet, 1]  =\
                                                        get_error_irr_gaussian(linear_interp_brms_irr, mass, mass_err,flux, flux_err)  
                    planets.brms_irr_surf_est_err[index_planet, 0], planets.brms_irr_surf_est_err[index_planet, 1]  =\
                                                        get_error_irr_gaussian(linear_interp_brms_surf_irr, mass, mass_err,flux, flux_err)

                    planets.heat_eff_inner_est_err[index_planet, 0],planets.heat_eff_inner_est_err[index_planet, 1] =\
                                                        get_error_gaussian(linear_interp_heat_effs_inner, 
                                                        mass, mass_err,flux, flux_err, radius, radius_err, planets.radius_irr_est[index_planet])
                    planets.brms_inner_est_err[index_planet, 0],planets.brms_inner_est_err[index_planet, 1] =\
                                            get_error_gaussian(linear_interp_brms_inner, 
                                            mass, mass_err,flux, flux_err, radius, radius_err, planets.radius_irr_est[index_planet])
                    planets.brms_inner_surf_est_err[index_planet, 0],planets.brms_inner_surf_est_err[index_planet, 1] =\
                                            get_error_gaussian(linear_interp_brms_surf_inner, 
                                            mass, mass_err,flux, flux_err, radius, radius_err, planets.radius_irr_est[index_planet])
                    
                    planets.heat_eff_outer_est_err[index_planet, 0],planets.heat_eff_outer_est_err[index_planet, 1] =\
                                                        get_error_gaussian(linear_interp_heat_effs_outer, 
                                                        mass, mass_err,flux, flux_err, radius, radius_err, planets.radius_irr_est[index_planet])
                    planets.brms_outer_est_err[index_planet, 0],planets.brms_outer_est_err[index_planet, 1] =\
                                            get_error_gaussian(linear_interp_brms_outer, 
                                            mass, mass_err,flux, flux_err, radius, radius_err, planets.radius_irr_est[index_planet])
                    planets.brms_outer_surf_est_err[index_planet, 0],planets.brms_outer_surf_est_err[index_planet, 1] =\
                                            get_error_gaussian(linear_interp_brms_surf_outer, 
                                            mass, mass_err,flux, flux_err, radius, radius_err, planets.radius_irr_est[index_planet])
                    
                          

            if print_all:
                
                print("   " + name + ": " + fr"({mass:.3f}$\pm${mass_err:.3f})MJ, " + \
                                            fr"({radius:.3f}$\pm${radius:.3f})RJ, " + \
                                            fr"({flux:.4e}$\pm${flux_err:.4e})erg cm-2 s-1")
                print("                          eps (%)                    brms,dyn (G)                  bdip,surf (G)")

                heat_eff_inner_est_print = table_predictions_save.get_print_string(100*planets.heat_eff_inner_est[index_planet],
                                                        100*planets.heat_eff_inner_est_err[index_planet], 
                                                        "heat",error_sym=error_sym)
                brms_inner_est_print = table_predictions_save.get_print_string(planets.brms_inner_est[index_planet],
                                                        planets.brms_inner_est_err[index_planet], 
                                                        "brms",error_sym=error_sym)
                brms_inner_surf_est_print = table_predictions_save.get_print_string(planets.brms_inner_surf_est[index_planet],
                                                        planets.brms_inner_surf_est_err[index_planet], 
                                                        "brms",error_sym=error_sym)
                

                print("Inner heat:  " + heat_eff_inner_est_print + "    "  + \
                                        brms_inner_est_print + "    "  + \
                                        brms_inner_surf_est_print)
                
                heat_eff_irr_est = 0
                if error_sym: heat_eff_irr_est_err=0
                else: heat_eff_irr_est_err=np.array([0,0])
                
                heat_eff_irr_est_print = table_predictions_save.get_print_string(100*heat_eff_irr_est,
                                                                                    100*heat_eff_irr_est_err, 
                                                                                    "heat",error_sym=error_sym)
                
                brms_irr_est_print = table_predictions_save.get_print_string(planets.brms_irr_est[index_planet],
                                                               planets.brms_irr_est_err[index_planet], 
                                                               "brms",error_sym=error_sym)
                brms_irr_surf_est_print = table_predictions_save.get_print_string(planets.brms_irr_surf_est[index_planet],
                                                                planets.brms_irr_surf_est_err[index_planet], 
                                                                "brms",error_sym=error_sym)
                radius_irr_surf_est_print = table_predictions_save.get_print_string(planets.radius_irr_surf_est[index_planet],
                                                                planets.radius_irr_surf_est_err[index_planet], 
                                                                "radius",error_sym=error_sym)

                print("Only irr:    " + heat_eff_irr_est_print + "    "  + \
                                brms_irr_est_print + "    "  + \
                                brms_irr_surf_est_print + "    "  + \
                                radius_irr_surf_est_print)
                
                heat_eff_outer_est_print = table_predictions_save.get_print_string(100*planets.heat_eff_outer_est[index_planet],
                                                                        100*planets.heat_eff_outer_est_err[index_planet], 
                                                                        "heat",error_sym=error_sym)
                brms_outer_est_print = table_predictions_save.get_print_string(planets.brms_outer_est[index_planet],
                                                                planets.brms_outer_est_err[index_planet], 
                                                                "brms",error_sym=error_sym)
                brms_outer_surf_est_print = table_predictions_save.get_print_string(planets.brms_outer_surf_est[index_planet],
                                                        planets.brms_outer_surf_est_err[index_planet], 
                                                        "brms",error_sym=error_sym)
            
                print("Outer heat:  " + heat_eff_outer_est_print + "    "  + \
                                        brms_outer_est_print + "    "  + \
                                        brms_outer_surf_est_print)
                
                print(f"----------------------------------------------------------------------")
            #else:
            #    if index_loop % 12==0: print(name)
            #    else: print(name, end=" ")
            index_loop +=1

