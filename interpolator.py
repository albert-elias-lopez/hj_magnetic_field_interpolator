"""
Interpolation of the MESA grid: heating efficiency eps, radius of the non-inflated (eps = 0) model
and magnetic fields (B_rms,dyn and B_dip,surf) of each planet, with their errors.

The inflated quantities are interpolated linearly in (M, log10 F_irr, R), the eps = 0 ones in
(M, log10 F_irr). With rotation (linear_interpolator_rot) the magnetic fields are also interpolated
linearly in log10 P between the grids of the two closest periods; without rotation
(linear_interpolator) the grid has no period axis. The errors are Monte Carlo percentiles (16th and
84th) of samples of the observed values, or a linear propagation with symmetric errors (--error_sym).
The results are stored in the planet_object (<quantity>_est and <quantity>_est_err).
"""
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay
import emcee
from tqdm import tqdm
import table_predictions_save

def get_error(interpolator,
              mass, mass_err,
              flux, flux_err,
              radius, radius_err,
              eps=1e-5):
    """
    Symmetric error of an interpolated quantity by linear propagation of the mass, log10 flux and radius
    errors, with numerical derivatives (step eps).
    """
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
    """
    Monte Carlo (minus, plus) error of an interpolated quantity: N samples of the mass, flux and radius
    (split normals with their lower and upper errors) inside the grid and larger than the eps = 0 radius
    radius_irr; the errors are the 16th and 84th percentiles around the median. Missing (NaN) errors give
    NaN.
    """
    if np.any(np.isnan(mass_err)) or np.any(np.isnan(flux_err)) or np.any(np.isnan(radius_err)):
        return np.nan, np.nan

    masses_s = table_predictions_save.sample_obs(mass, mass_err, N)
    fluxes_s = table_predictions_save.sample_obs(flux, flux_err, N)
    radii_s = table_predictions_save.sample_obs(radius, radius_err, N)

    mask = (masses_s > min_mass) & (masses_s < max_mass) & (fluxes_s > min_flux) & (fluxes_s < max_flux) & (radii_s > radius_irr)
    samples = interpolator(masses_s[mask],np.log10(fluxes_s[mask]),radii_s[mask])

    mask_nans = ~np.isnan(samples)
    samples = samples[mask_nans]
    if samples.size == 0: return np.nan, np.nan   # No sample inside the grid: no error bar instead of a crash
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
    if B_s.size == 0: return np.nan, np.nan   # No sample inside the grid: no error bar instead of a crash

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
    
    # A missing (NaN) observational error gives a missing (NaN) error, as in get_scaling_law_err
    if np.any(np.isnan(mass_err)) or np.any(np.isnan(flux_err)):
        return np.nan, np.nan

    masses_s = table_predictions_save.sample_obs(mass, mass_err, N)
    fluxes_s = table_predictions_save.sample_obs(flux, flux_err, N)

    mask = (masses_s > min_mass) & (masses_s < max_mass) & (fluxes_s > min_flux) & (fluxes_s < max_flux)
    samples = interpolator(masses_s[mask],np.log10(fluxes_s[mask]))

    mask_nans = ~np.isnan(samples)
    samples = samples[mask_nans]
    if samples.size == 0: return np.nan, np.nan   # No sample inside the grid: no error bar instead of a crash
    p16, p50, p84 = np.percentile(samples, [16, 50, 84])

    return p50 - p16, p84 - p50 # estimate error minus, estimate error plus
 
# Estimates stored in planets as <name>_est and <name>_est_err, in the order they are computed
# (the order matters: the Monte Carlo errors draw from the same random number generator)
IRR_QUANTITIES  = ["brms_irr", "brms_irr_surf"]
HEAT_QUANTITIES = ["heat_eff_inner", "brms_inner", "brms_inner_surf",
                   "heat_eff_outer", "brms_outer", "brms_outer_surf"]

def get_error_irr_any(interpolator, mass, mass_err, flux, flux_err, error_sym):
    if error_sym: return get_error_irr(interpolator, mass, mass_err, flux, flux_err)
    return get_error_irr_gaussian(interpolator, mass, mass_err, flux, flux_err)

def get_error_any(interpolator, mass, mass_err, flux, flux_err, radius, radius_err, radius_irr, error_sym):
    if error_sym: return get_error(interpolator, mass, mass_err, flux, flux_err, radius, radius_err)
    return get_error_gaussian(interpolator, mass, mass_err, flux, flux_err, radius, radius_err, radius_irr)

def build_interpolators(table_inner, table_outer, index_period = None):
    """
    Linear interpolators of the tables, as a dictionary: heat_eff_<heating>, brms_<heating> and
    brms_<heating>_surf in (M, log10 F_irr, R) for the inner and outer heating, and radius_irr, brms_irr
    and brms_irr_surf in (M, log10 F_irr) from the eps = 0 models. Only the models that reached the age
    of the tables (R > 0) are used. index_period selects one period of the tables with rotation.
    """
    # The rotation tables have an extra last axis for the orbital period
    def grid(array):
        return array if index_period is None else array[..., index_period]

    masses_grid = grid(table_inner.masses_grid_total)
    flux_grid   = grid(table_inner.flux_grid_total)
    heat_effs_grid_flat = grid(table_inner.heat_effs_grid_total).flatten()

    interpolators = {}
    for heating, table in (("inner", table_inner), ("outer", table_outer)):
        radii_heat_flat = grid(table.radius_heat).flatten()
        # Create a mask for valid simulation points (radius > 0)
        valid_mask = radii_heat_flat > 0
        # One triangulation shared by the interpolators with the same points
        points = Delaunay(list(zip(masses_grid.flatten()[valid_mask],
                                   np.log10(flux_grid.flatten()[valid_mask]),
                                   radii_heat_flat[valid_mask])))
        interpolators["heat_eff_" + heating]    = LinearNDInterpolator(points, heat_effs_grid_flat[valid_mask])
        interpolators["brms_" + heating]        = LinearNDInterpolator(points, grid(table.bscaling_heat).flatten()[valid_mask])
        interpolators["brms_" + heating + "_surf"] = LinearNDInterpolator(points, grid(table.bscaling_surf_heat).flatten()[valid_mask])

    # The epsilon = 0 data give the estimates for the only irradiated models.
    # It doesn't matter to take the radius and magnetic field estimation from inner or outer, as eps=0
    radii_irr_flat = grid(table_inner.radius_heat)[:,:,0].flatten()
    # Some small planets with high irradiations explode, even without inflation
    valid_mask_irr = radii_irr_flat > 0
    points_irr = Delaunay(list(zip(masses_grid[:,:,0].flatten()[valid_mask_irr],
                                   np.log10(flux_grid[:,:,0].flatten()[valid_mask_irr]))))
    interpolators["radius_irr"]    = LinearNDInterpolator(points_irr, radii_irr_flat[valid_mask_irr])
    interpolators["brms_irr"]      = LinearNDInterpolator(points_irr, grid(table_inner.bscaling_heat)[:,:,0].flatten()[valid_mask_irr])
    interpolators["brms_irr_surf"] = LinearNDInterpolator(points_irr, grid(table_inner.bscaling_surf_heat)[:,:,0].flatten()[valid_mask_irr])
    return interpolators

def get_observed_values(planets, index_planet, error_sym):
    mass_err   = planets.mass_obs_err[index_planet]
    flux_err   = planets.flux_obs_err[index_planet]
    radius_err = planets.radius_obs_err[index_planet]
    if not error_sym:
        # The Monte Carlo errors sample the observed values with their (lower, upper) errors
        mass_err   = planets.mass_obs_err_asym[index_planet]
        flux_err   = planets.flux_obs_err_asym[index_planet]
        radius_err = planets.radius_obs_err_asym[index_planet]
    return (planets.mass_obs[index_planet], mass_err,
            planets.flux_obs[index_planet], flux_err,
            planets.radius_obs[index_planet], radius_err)

def set_radius_irr(planets, index_planet, interpolator_radius_irr, mass, mass_err, flux, flux_err, radius, error_sym):
    # Store the radius without heating. Returns False (with no B or heating estimates) if the planet is smaller
    planets.radius_irr_est[index_planet]     = interpolator_radius_irr(mass, np.log10(flux))
    planets.radius_irr_est_err[index_planet] = get_error_irr_any(interpolator_radius_irr, mass, mass_err, flux, flux_err, error_sym)

    if radius < planets.radius_irr_est[index_planet]:
        for name in IRR_QUANTITIES + HEAT_QUANTITIES:
            getattr(planets, name + "_est")[index_planet]     = np.nan
            getattr(planets, name + "_est_err")[index_planet] = np.nan
        return False
    return True

def linear_interpolator(table_inner,
                        table_outer,
                        planets,
                        mask_predictions,
                        error_sym = False,
                        interpolators = None):
    """
    Estimates of the planets in mask_predictions with the tables without rotation, stored in planets.
    Planets smaller than their eps = 0 model get only radius_irr (the other estimates are NaN).
    interpolators: the result of build_interpolators, to reuse it between calls.
    """

    if interpolators is None: interpolators = build_interpolators(table_inner, table_outer)

    for index_planet in tqdm(range(planets.n_planets)):
        if not mask_predictions[index_planet]: continue

        mass, mass_err, flux, flux_err, radius, radius_err = get_observed_values(planets, index_planet, error_sym)

        if not set_radius_irr(planets, index_planet, interpolators["radius_irr"],
                              mass, mass_err, flux, flux_err, radius, error_sym): continue
        radius_irr = planets.radius_irr_est[index_planet]

        for name in IRR_QUANTITIES:
            getattr(planets, name + "_est")[index_planet]     = interpolators[name](mass, np.log10(flux))
            getattr(planets, name + "_est_err")[index_planet] = get_error_irr_any(interpolators[name],
                                                                    mass, mass_err, flux, flux_err, error_sym)
        for name in HEAT_QUANTITIES:
            getattr(planets, name + "_est")[index_planet]     = interpolators[name](mass, np.log10(flux), radius)
            getattr(planets, name + "_est_err")[index_planet] = get_error_any(interpolators[name],
                                                                    mass, mass_err, flux, flux_err, radius, radius_err, radius_irr, error_sym)

def get_b_value(interpolator, index_period, fraction, *coordinates):
    brms_upper        = interpolator[index_period](*coordinates)
    brms_lower        = interpolator[index_period-1](*coordinates)
    brms_est          = brms_lower + fraction * (brms_upper - brms_lower)
    delta_brms_est    = brms_upper - brms_lower
    return brms_est, delta_brms_est

def get_error_gaussian_rot(interpolator, periods,
                           mass, mass_err,
                           flux, flux_err,
                           radius, radius_err,
                           radius_irr,
                           period, period_err,
                           only_irr = False,
                           N = 10000,
                           min_mass = 0.3, max_mass = 12,
                           min_flux = 1e7, max_flux = 1.58e10):
    """
    Monte Carlo (minus, plus) error of a magnetic field with rotation. interpolator has one interpolator
    per period of the grid. Mass, flux, radius and period are sampled together, and each sample is
    interpolated in log10 P between its own two grid periods. only_irr: an eps = 0 quantity, which does
    not depend on the radius.
    """
    # A missing (NaN) observational error gives a missing (NaN) error, as in get_scaling_law_err

    # A missing (NaN) observational error gives a missing (NaN) error, as in get_scaling_law_err
    if np.any(np.isnan(mass_err)) or np.any(np.isnan(flux_err)) or \
       (not only_irr and np.any(np.isnan(radius_err))):
        return np.nan, np.nan

    masses_s  = table_predictions_save.sample_obs(mass, mass_err, N)
    fluxes_s  = table_predictions_save.sample_obs(flux, flux_err, N)
    if not only_irr: radii_s = table_predictions_save.sample_obs(radius, radius_err, N)
    periods_s = np.random.normal(period, period_err, N)

    mask = (masses_s > min_mass) & (masses_s < max_mass) & (fluxes_s > min_flux) & (fluxes_s < max_flux) & \
           (periods_s > periods[0]) & (periods_s < periods[-1])
    if not only_irr: mask &= radii_s > radius_irr

    masses_s, log_fluxes_s, periods_s = masses_s[mask], np.log10(fluxes_s[mask]), periods_s[mask]
    if not only_irr: radii_s = radii_s[mask]

    idx_upper = np.searchsorted(periods, periods_s)
    fraction = (np.log10(periods_s) - np.log10(periods[idx_upper-1])) / (np.log10(periods[idx_upper]) - np.log10(periods[idx_upper-1]))

    samples = np.empty(periods_s.size)
    for index_period in np.unique(idx_upper):
        in_bracket = idx_upper == index_period
        coordinates = (masses_s[in_bracket], log_fluxes_s[in_bracket])
        if not only_irr: coordinates += (radii_s[in_bracket],)
        samples[in_bracket], _ = get_b_value(interpolator, index_period, fraction[in_bracket], *coordinates)

    samples = samples[~np.isnan(samples)]
    if samples.size == 0: return np.nan, np.nan   # No sample inside the grid: no error bar instead of a crash
    p16, p50, p84 = np.percentile(samples, [16, 50, 84])

    return p50 - p16, p84 - p50 # estimate error minus, estimate error plus

def linear_interpolator_rot(table_inner,
                            table_outer,
                            planets,
                            periods,
                            mask_predictions,
                            error_sym = False,
                            interpolators_s = None):
    """
    Estimates of the planets in mask_predictions with the tables with rotation, stored in planets: the
    magnetic fields are interpolated linearly in log10 P between the grids of the two closest periods
    (the heating efficiencies and radius_irr do not depend on the period).
    interpolators_s: the build_interpolators of every period, to reuse them between calls.
    """

    if interpolators_s is None:
        interpolators_s = [build_interpolators(table_inner, table_outer, index_period)
                           for index_period in range(len(periods))]
    # The heat efficiency and radius_irr interpolators are the same for all periods
    interpolators = interpolators_s[0]

    for index_planet in tqdm(range(planets.n_planets)):
        if not mask_predictions[index_planet]: continue

        mass, mass_err, flux, flux_err, radius, radius_err = get_observed_values(planets, index_planet, error_sym)
        period = planets.orbital_period_obs[index_planet]
        period_err = planets.orbital_period_obs_err[index_planet]
        # Some archive planets have no period error. Periods are known to better than ~1%,
        # so a missing error is taken as zero instead of removing the B error bars
        if np.isnan(period_err): period_err = 0.0

        if not set_radius_irr(planets, index_planet, interpolators["radius_irr"],
                              mass, mass_err, flux, flux_err, radius, error_sym): continue
        radius_irr = planets.radius_irr_est[index_planet]

        idx_upper = np.searchsorted(periods, period)
        idx_lower = idx_upper - 1
        period_upper = periods[idx_upper]
        period_lower = periods[idx_lower]
        fraction = (np.log10(period) - np.log10(period_lower)) / (np.log10(period_upper) - np.log10(period_lower))

        for name in IRR_QUANTITIES + [name for name in HEAT_QUANTITIES if name.startswith("brms")]:
            only_irr = name in IRR_QUANTITIES
            coordinates = (mass, np.log10(flux)) if only_irr else (mass, np.log10(flux), radius)

            interpolator = [interpolators_period[name] for interpolators_period in interpolators_s]
            getattr(planets, name + "_est")[index_planet], delta_brms_est = get_b_value(interpolator,
                                                                                idx_upper, fraction, *coordinates)
            if error_sym:
                # Linear propagation through B interpolated at the planet's period, plus the period term
                def interpolator_period(*coordinates):
                    return get_b_value(interpolator, idx_upper, fraction, *coordinates)[0]
                if only_irr: brms_err = get_error_irr(interpolator_period, mass, mass_err, flux, flux_err)
                else:        brms_err = get_error(interpolator_period, mass, mass_err, flux, flux_err, radius, radius_err)
                dbrms_dlogperiod = delta_brms_est/(np.log10(period_upper) - np.log10(period_lower))
                log_period_err = period_err/period/np.log(10)
                getattr(planets, name + "_est_err")[index_planet] = np.sqrt(brms_err**2 + dbrms_dlogperiod**2*log_period_err**2)
            else:
                getattr(planets, name + "_est_err")[index_planet] = get_error_gaussian_rot(interpolator, periods,
                                                                        mass, mass_err, flux, flux_err, radius, radius_err, radius_irr,
                                                                        period, period_err, only_irr = only_irr)

        for name in ["heat_eff_inner", "heat_eff_outer"]:
            getattr(planets, name + "_est")[index_planet]     = interpolators[name](mass, np.log10(flux), radius)
            getattr(planets, name + "_est_err")[index_planet] = get_error_any(interpolators[name],
                                                                    mass, mass_err, flux, flux_err, radius, radius_err, radius_irr, error_sym)

def reasons_not_interpolated(planets, grid_limits, rotation):
    """
    Why each planet has no heating efficiency or magnetic field estimates, to be called after
    linear_interpolator or linear_interpolator_rot. Returns, for each planet, a list of
    (reason, value) pairs, empty for the planets with estimates. The reason does not depend on
    the planet (it can be counted over the population); the value shows the planet's numbers.
    """
    min_mass, max_mass, min_flux, max_flux, min_period, max_period = grid_limits
    observed = [("mass", "M", planets.mass_obs, "M_J", min_mass, max_mass),
                ("incident flux", "F_irr", planets.flux_obs, "erg cm^-2 s^-1", min_flux, max_flux)]
    if rotation:
        observed.append(("orbital period", "P_orb", planets.orbital_period_obs, "days", min_period, max_period))

    reasons = []
    for index_planet in range(planets.n_planets):
        planet_reasons = []
        radius = planets.radius_obs[index_planet]
        if np.isnan(radius): planet_reasons.append(("missing radius", ""))
        for name, symbol, values, unit, minimum, maximum in observed:
            value = values[index_planet]
            if np.isnan(value):
                planet_reasons.append(("missing " + name, ""))
            elif not minimum < value < maximum:
                planet_reasons.append((f"{name} outside the grid ({minimum:.3g} < {symbol} < {maximum:.3g} {unit})",
                                       f"{symbol} = {value:.3g} {unit}"))

        # The planets inside the grid limits were interpolated: check that they got estimates
        if not planet_reasons:
            radius_irr = planets.radius_irr_est[index_planet]
            heat_effs = [planets.heat_eff_inner_est[index_planet], planets.heat_eff_outer_est[index_planet]]
            if np.isnan(radius_irr):
                planet_reasons.append(("no non-inflated (eps = 0) model at this mass and flux "
                                       "(the small, strongly irradiated models do not survive)", ""))
            elif radius < radius_irr:
                planet_reasons.append(("radius smaller than the non-inflated (eps = 0) model with the same "
                                       "mass and flux: not inflated, only radius_irr is estimated",
                                       f"R = {radius:.3g} R_J < radius_irr = {radius_irr:.3g} R_J"))
            elif np.any(np.isnan(heat_effs)):
                planet_reasons.append(("radius larger than the most inflated model of the grid "
                                       "with the same mass and flux", f"R = {radius:.3g} R_J"))
        reasons.append(planet_reasons)
    return reasons
