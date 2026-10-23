"""
Reader of the magnetic field tables of the MESA grid (<mass>_MJ/ folders, written by
table_generator.py) into arrays over the grid (mass, flux, heating efficiency[, period]).
"""
import numpy as np

class table_object:
    """
    Tables of one heating type ("inner" or "outer") over the grid: radius (R_J), B_rms,dyn and B_dip,surf
    (G) and dynamo luminosity (erg s^-1) of each model (0 for the models that did not reach the age of the
    tables), and the mass, flux, heating efficiency and period of each grid point.
    """
    def __init__(self, shape_grid, heating_type):
        self.heating_type          = heating_type
        self.radius_heat            = np.zeros(shape_grid)
        self.bscaling_heat         = np.zeros(shape_grid)
        self.bscaling_surf_heat    = np.zeros(shape_grid)
        self.dyn_lum_heat          = np.zeros(shape_grid)
        self.masses_grid_total     = np.zeros(shape_grid)
        self.flux_grid_total       = np.zeros(shape_grid)
        self.period_grid_total     = np.zeros(shape_grid)
        self.heat_effs_grid_total  = np.zeros(shape_grid)
        self.total_number_models   = 0

    def read_table(self, rotation, masses, fluxes, heat_effs, periods):
        # Reads the tables <mass>_MJ/[P_<period>/]F_<flux>[_eps_outer].txt written by table_generator.py.
        # The arrays have the axes (mass, flux, heat efficiency), plus the period with rotation.

        names_flux = []
        for index_flux,flux in enumerate(fluxes):
            names_flux += ["F_{0:.2e}".format(flux).replace('.', '_').replace('+', '').replace('e07', 'e7').replace('e08', 'e8').replace('e09', 'e9')]

        names_mass = []
        for index_mass,mass in enumerate(masses):
            names_mass += ["{0:.2f}_MJ".format(mass).replace('.', '_')]

        names_period = []
        for index_period,period in enumerate(periods):
            names_period += ["P_{0:.3f}".format(period).replace('.', '_')]

        # Factor of the tables without rotation, to match the scaling law with Jupiter's B_dip,surf
        if rotation: b_factor = 1
        else: b_factor = 1/3

        # Without rotation there is no period axis: one pass with an empty period index
        if rotation: period_indices = [(index_period,) for index_period in range(len(periods))]
        else: period_indices = [()]

        n_heat_effs = len(heat_effs)
        for index_mass, mass in enumerate(masses):
            for index_flux, flux in enumerate(fluxes):
                for index_period in period_indices:

                    filename = names_mass[index_mass] + "/"
                    if rotation: filename += names_period[index_period[0]] + "/"
                    filename += names_flux[index_flux]
                    if self.heating_type == "inner": filename += ".txt"
                    elif self.heating_type == "outer": filename += "_eps_outer.txt"

                    # One row per model that reached the age of the tables; only the header if none did
                    with open(filename, "r") as f: lines = f.readlines()
                    if len(lines) <= 1: matrix_flux_loaded = np.zeros((0, 5))
                    else: matrix_flux_loaded = np.loadtxt(filename, skiprows=1, ndmin=2)

                    # The rows are in increasing heat efficiency: look for each one from its row number on
                    for index_row, row in enumerate(matrix_flux_loaded):
                        for index_heat_eff in range(index_row, n_heat_effs):
                            if row[0] == heat_effs[index_heat_eff]:
                                index = (index_mass, index_flux, index_heat_eff) + index_period
                                self.radius_heat[index]          = row[1]
                                self.bscaling_heat[index]       = row[2]*b_factor
                                self.bscaling_surf_heat[index]  = row[3]*b_factor
                                self.dyn_lum_heat[index]        = row[4]
                                self.total_number_models += 1
                                break

                    all_heat_effs = (index_mass, index_flux, slice(None)) + index_period
                    self.masses_grid_total[all_heat_effs] = np.ones(n_heat_effs)*mass
                    self.flux_grid_total[all_heat_effs] = np.ones(n_heat_effs)*flux
                    self.heat_effs_grid_total[all_heat_effs] = heat_effs
                    if rotation: self.period_grid_total[all_heat_effs] = np.ones(n_heat_effs)*periods[index_period[0]]
