"""
The grid of planetary evolution models used by the interpolator: planet masses (M_J), incident
fluxes (erg cm^-2 s^-1), heating efficiencies (fraction) and orbital periods (days), and the
limits of the grid used to select the planets that can be interpolated.
"""
import os
import numpy as np
from table_reader import table_object

# Folder of the repository, where the magnetic field tables (<mass>_MJ/ folders) are
REPOSITORY = os.path.dirname(os.path.realpath(__file__))

# Grid limits which will be used to mask the exoplanet population before interpolating.
min_mass, max_mass = 0.3, 12
min_flux, max_flux = 1e7, 1.58e10
min_period, max_period = 0.1, 100.0
limits = (min_mass, max_mass, min_flux, max_flux, min_period, max_period)

fluxes = np.array([1.00000000e+07, 1.25892541e+07, 1.58489319e+07, 1.99526231e+07,
                    2.51188643e+07, 3.16227766e+07, 3.98107171e+07, 5.01187234e+07,
                    6.30957344e+07, 7.94328235e+07, 1.00000000e+08, 1.25892541e+08,
                    1.58489319e+08, 1.99526231e+08, 2.51188643e+08, 3.16227766e+08,
                    3.98107171e+08, 5.01187234e+08, 6.30957344e+08, 7.94328235e+08,
                    1.00000000e+09, 1.25892541e+09, 1.58489319e+09, 1.99526231e+09,
                    2.51188643e+09, 3.16227766e+09, 3.98107171e+09, 5.01187234e+09,
                    6.30957344e+09, 7.94328235e+09, 1.00000000e+10, 1.25892541e+10,
                    1.58489319e+10])

heat_effs = np.array([0.0, 0.00001, 0.00002, 0.00005,
                        0.0001, 0.0002, 0.0005,
                        0.001, 0.002, 0.005, 0.01, 0.02, 0.05,
                        0.1, 0.2, 0.4, 0.8, 1. , 2. , 4. ])

masses = np.array([0.3, 0.4, 0.5, 0.7, 1, 1.5, 2, 3, 4, 6, 8, 12])

periods = np.array([  0.1       ,   0.14384499,   0.20691381,   0.29763514,
                        0.42813324,   0.61584821,   0.88586679,   1.27427499,
                        1.83298071,   2.6366509 ,   3.79269019,   5.45559478,
                        7.8475997 ,  11.28837892,  16.23776739,  23.35721469,
                        33.59818286,  48.32930239,  69.51927962, 100.        ])

def read_tables(rotation):
    # Read the inner- and outer-heating magnetic field tables of the grid
    if rotation: shape_grid = (len(masses), len(fluxes), len(heat_effs), len(periods))
    else: shape_grid = (len(masses), len(fluxes), len(heat_effs))

    # The table paths are relative to the repository folder
    current = os.getcwd()
    os.chdir(REPOSITORY)
    try:
        table_inner = table_object(shape_grid, "inner")
        table_inner.read_table(rotation, masses, fluxes, heat_effs, periods)
        table_outer = table_object(shape_grid, "outer")
        table_outer.read_table(rotation, masses, fluxes, heat_effs, periods)
    finally:
        os.chdir(current)
    return table_inner, table_outer
