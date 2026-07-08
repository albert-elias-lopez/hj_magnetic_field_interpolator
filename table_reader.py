import numpy as np

class table_object:
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

        names_flux = []
        for index_flux,flux in enumerate(fluxes):
            names_flux += ["F_{0:.2e}".format(flux).replace('.', '_').replace('+', '').replace('e07', 'e7').replace('e08', 'e8').replace('e09', 'e9')]

        names_mass = []
        for index_mass,mass in enumerate(masses):
            names_mass += ["{0:.2f}_MJ".format(mass).replace('.', '_')]

        n_heat_effs = len(heat_effs)
        names_period = []
        if rotation:
            for index_period,period in enumerate(periods):
                names_period += ["P_{0:.3f}".format(period).replace('.', '_')]

            b_factor = 1
        else:
            b_factor = 1/3

        if rotation:
            for index_mass, mass in enumerate(masses):
                for index_flux, flux in enumerate(fluxes):

                    for index_period, period in enumerate(periods):
                        
                        filename = names_mass[index_mass] + "/" + names_period[index_period] + "/" + names_flux[index_flux]
                        if self.heating_type == "inner": filename += ".txt"
                        elif self.heating_type == "outer": filename += "_eps_outer.txt"

                        # Check if the file is empty and read
                        with open(filename, "r") as f: lines = f.readlines()
                        if len(lines) <= 1: matrix_flux_loaded = np.array([])
                        else: matrix_flux_loaded = np.loadtxt(filename, skiprows=1)

                        for index_row in range(0,len(matrix_flux_loaded)):
                            for index_heat_eff in range(index_row,n_heat_effs):
                                heat_eff = heat_effs[index_heat_eff]
                                if matrix_flux_loaded.ndim == 2: 
                                    heat_eff_loaded = matrix_flux_loaded[index_row,0]
                                    if heat_eff_loaded == heat_eff:
                                        self.radius_heat[index_mass, index_flux, index_heat_eff, index_period]          = matrix_flux_loaded[index_row,1]
                                        self.bscaling_heat[index_mass, index_flux, index_heat_eff, index_period]       = matrix_flux_loaded[index_row,2]*b_factor
                                        self.bscaling_surf_heat[index_mass, index_flux, index_heat_eff, index_period]  = matrix_flux_loaded[index_row,3]*b_factor
                                        self.dyn_lum_heat[index_mass, index_flux, index_heat_eff, index_period]        = matrix_flux_loaded[index_row,4]
                                        self.total_number_models += 1
                                        break
                                elif matrix_flux_loaded.ndim == 1: 
                                    heat_eff_loaded = matrix_flux_loaded[0]
                                    if heat_eff_loaded == heat_eff:
                                        self.radius_heat[index_mass, index_flux, index_heat_eff, index_period]          = matrix_flux_loaded[1]
                                        self.bscaling_heat[index_mass, index_flux, index_heat_eff, index_period]       = matrix_flux_loaded[2]*b_factor
                                        self.bscaling_surf_heat[index_mass, index_flux, index_heat_eff, index_period]  = matrix_flux_loaded[3]*b_factor
                                        self.dyn_lum_heat[index_mass, index_flux, index_heat_eff, index_period]        = matrix_flux_loaded[4]
                                        self.total_number_models += 1
                                        break
                                else: break

                        self.masses_grid_total[index_mass, index_flux, :, index_period] = np.ones(n_heat_effs)*mass
                        self.flux_grid_total[index_mass, index_flux, :, index_period] = np.ones(n_heat_effs)*flux
                        self.heat_effs_grid_total[index_mass, index_flux, :, index_period] = heat_effs
                        self.period_grid_total[index_mass, index_flux, :, index_period] = np.ones(n_heat_effs)*period
        else:
            for index_mass, mass in enumerate(masses):
                for index_flux, flux in enumerate(fluxes):
                    
                    filename = names_mass[index_mass] + "/" + names_flux[index_flux]
                    if self.heating_type == "inner": filename += ".txt"
                    elif self.heating_type == "outer": filename += "_eps_outer.txt"

                    # Check if the file is empty and read
                    with open(filename, "r") as f: lines = f.readlines()
                    if len(lines) <= 1: matrix_flux_loaded = np.array([])
                    else: matrix_flux_loaded = np.loadtxt(filename, skiprows=1)

                    for index_row in range(0,len(matrix_flux_loaded)):
                        for index_heat_eff in range(index_row,n_heat_effs):
                            heat_eff = heat_effs[index_heat_eff]
                            if matrix_flux_loaded.ndim == 2: 
                                heat_eff_loaded = matrix_flux_loaded[index_row,0]
                                if heat_eff_loaded == heat_eff:
                                    self.radius_heat[index_mass, index_flux, index_heat_eff]          = matrix_flux_loaded[index_row,1]
                                    self.bscaling_heat[index_mass, index_flux, index_heat_eff]       = matrix_flux_loaded[index_row,2]*b_factor
                                    self.bscaling_surf_heat[index_mass, index_flux, index_heat_eff]  = matrix_flux_loaded[index_row,3]*b_factor
                                    self.dyn_lum_heat[index_mass, index_flux, index_heat_eff]        = matrix_flux_loaded[index_row,4]
                                    self.total_number_models += 1
                                    break
                            elif matrix_flux_loaded.ndim == 1: 
                                heat_eff_loaded = matrix_flux_loaded[0]
                                if heat_eff_loaded == heat_eff:
                                    self.radius_heat[index_mass, index_flux, index_heat_eff]          = matrix_flux_loaded[1]
                                    self.bscaling_heat[index_mass, index_flux, index_heat_eff]       = matrix_flux_loaded[2]*b_factor
                                    self.bscaling_surf_heat[index_mass, index_flux, index_heat_eff]  = matrix_flux_loaded[3]*b_factor
                                    self.dyn_lum_heat[index_mass, index_flux, index_heat_eff]        = matrix_flux_loaded[4]
                                    self.total_number_models += 1
                                    break
                            else: break

                    self.masses_grid_total[index_mass, index_flux, :] = np.ones(n_heat_effs)*mass
                    self.flux_grid_total[index_mass, index_flux, :] = np.ones(n_heat_effs)*flux
                    self.heat_effs_grid_total[index_mass, index_flux, :] = heat_effs