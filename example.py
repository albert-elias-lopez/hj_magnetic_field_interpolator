"""
Example: magnetic field estimates for one planet. Replace the values below with your own planet
and run (after activating the environment, see the README):

    python example.py
"""
from hj_model import HJMagneticModel

# ---------------- Your planet (errors are symmetric, or (lower, upper) tuples) ----------------
name                         = "WASP-12 b"
mass,         mass_err       = 1.41, 0.10       # planet mass (M_J)
radius,       radius_err     = 1.79, 0.09       # planet radius (R_J)
period,       period_err     = 1.0914, 3e-6     # orbital period (days)
orbital_dist, orbital_dist_err = 0.0229, 0.0008 # orbital distance (AU)
radius_star,  radius_star_err  = 1.57, 0.07     # stellar radius (R_sun)
temp_star,    temp_star_err    = 6300, 150      # stellar effective temperature (K)
flux, flux_err                 = None, None     # incident flux (erg cm^-2 s^-1), or None to compute from stellar parameters
# -----------------------------------------------------------------------------------------------

model = HJMagneticModel()      # reads the grid tables (a few seconds); option rotation=False: tables without rotation

# Incident flux on the planet (erg cm^-2 s^-1), unless it is given above. Without the orbital
# distance, give period=..., period_err=..., mass_star=..., mass_star_err=... (M_sun) instead.
if flux is None:
    flux, flux_err = model.get_flux(radius_star, radius_star_err, temp_star, temp_star_err,
                                    orbital_dist=orbital_dist, orbital_dist_err=orbital_dist_err)

# Heating efficiencies and magnetic fields interpolated in the grid, with Monte Carlo errors
result = model.estimate(mass, mass_err, radius, radius_err, flux, flux_err,
                        period=period, period_err=period_err, names=name)
result.print_all()
# The same estimates next to the scaling laws: the fits of the paper (MRL, MRF, MR) and
# Reiners et al. (2009) with the interpolated heating efficiency (R09) or that of Thorngren & Fortney 2018 (R09+Th18)
result.print_all_scaling_law()

# Single values, e.g. to use in your own code: value and (minus, plus) error
# print(result.keys())
# print("B_dip,surf for deep heating:", result["brms_inner_surf"], "G, error", result["brms_inner_surf_err"])

