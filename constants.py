"""
Physical constants and unit conversions used in the code (cgs units unless stated otherwise).
"""

SIGMA_SB = 5.670374419e-5   # Stefan-Boltzmann constant (erg cm^-2 s^-1 K^-4)
G        = 6.674e-8         # gravitational constant (cm^3 g^-1 s^-2)

M_J = 1.8981e30             # Jupiter mass (g)
R_J = 6.991e9               # Jupiter mean radius (cm)
L_J = 4.606e24              # Jupiter intrinsic luminosity (erg s^-1): 7.5 W m^-2 (Li et al. 2018) over 4 pi R_J^2;
                            # unit of the deposited heating power L_P = eps pi R^2 F_irr
M_J_EARTH = 317.8           # Jupiter mass in Earth masses

R_SUN = 6.957e10            # solar radius (cm)
L_SUN = 3.846e33            # solar luminosity (erg s^-1)
R_SUN_AU = 0.00465047       # solar radius in AU
R_SUN_RJ = 9.731            # solar radius in Jupiter equatorial radii (7.1492e9 cm), for the MESA radii

# Kepler's third law in practical units: a^3 (AU^3) = KEPLER * M_star (M_sun) * P^2 (days^2),
# with KEPLER = (1 day / 1 yr)^2 = 1/365.25^2
KEPLER = 7.496e-6
