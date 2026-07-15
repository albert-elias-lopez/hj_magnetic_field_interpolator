import numpy as np

class planet_object:
    def __init__(self, N_planets, error_sym):

        if error_sym:
            error_array_shape = N_planets
        else:
            error_array_shape = (N_planets,2)

        # NASA data
        self.n_planets                  = N_planets
        self.planet_names               = []
        self.mass_obs                   = np.zeros(N_planets)
        self.mass_obs_err               = np.zeros(N_planets)
        self.temp_eff_obs               = np.zeros(N_planets)
        self.flux_obs                   = np.zeros(N_planets)
        self.flux_obs_err               = np.zeros(N_planets)
        self.radius_obs                 = np.zeros(N_planets)
        self.radius_obs_err             = np.zeros(N_planets)
        self.orbital_period_obs         = np.zeros(N_planets)
        self.orbital_period_obs_err     = np.zeros(N_planets)

        self.temp_star_obs              = np.zeros(N_planets)
        self.temp_star_obs_err          = np.zeros(N_planets)
        self.radius_star_obs            = np.zeros(N_planets)
        self.radius_star_obs_err        = np.zeros(N_planets)
        self.mass_star_obs              = np.zeros(N_planets)
        self.mass_star_obs_err          = np.zeros(N_planets)
        self.orbital_dist_obs           = np.zeros(N_planets)
        self.orbital_dist_obs_err       = np.zeros(N_planets)

        # Interpolated data for inner models
        self.heat_eff_inner_est         = np.zeros(N_planets)
        self.heat_eff_inner_est_err     = np.zeros(error_array_shape)
        self.brms_inner_est             = np.zeros(N_planets)
        self.brms_inner_est_err         = np.zeros(error_array_shape)
        self.brms_inner_surf_est        = np.zeros(N_planets)
        self.brms_inner_surf_est_err    = np.zeros(error_array_shape)

        # Interpolated data for outer models
        self.heat_eff_outer_est         = np.zeros(N_planets)
        self.heat_eff_outer_est_err     = np.zeros(error_array_shape)
        self.brms_outer_est             = np.zeros(N_planets)
        self.brms_outer_est_err         = np.zeros(error_array_shape)
        self.brms_outer_surf_est        = np.zeros(N_planets)
        self.brms_outer_surf_est_err    = np.zeros(error_array_shape)

        # Interpolated data for only irradiated models
        self.radius_irr_est             = np.zeros(N_planets)
        self.radius_irr_est_err         = np.zeros(error_array_shape)
        self.brms_irr_est               = np.zeros(N_planets)
        self.brms_irr_est_err           = np.zeros(error_array_shape)
        self.brms_irr_surf_est          = np.zeros(N_planets)
        self.brms_irr_surf_est_err      = np.zeros(error_array_shape)
    
    def get_flux(self, albedo= 0, sigma_SB = 5.670374419e-5):
        self.flux_obs = sigma_SB*self.temp_star_obs**4*(self.radius_star_obs*0.00465047/self.orbital_dist_obs)**2  # 0.00465047 is the solar radius in au
        self.flux_obs_err = 2*self.flux_obs* np.sqrt((2*self.temp_star_obs_err/self.temp_star_obs)**2 + \
                                                    (self.radius_star_obs_err/self.radius_star_obs)**2 + \
                                                    (self.orbital_dist_obs_err/self.orbital_dist_obs)**2)

    def read_file(self, filename, line_skip):
        # This will be used for intepolating missing stellar data from the exoplaent database
        filename_stars = './main_sequence_luminosities.txt'
        # Arrays to store values
        spectral_type = []
        star_masses = []
        star_temps = []
        star_radii = []
        star_lum = []

        with open(filename_stars, "r") as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and header lines
                if not line or line.startswith("#SpT"):
                    continue
                cols = line.split()
                try: spectral_type.append(cols[0])
                except: spectral_type.append(np.nan)

                try: star_temps.append(float(cols[1]))
                except: star_temps.append(np.nan) 

                #try: star_lum.append(float(cols[4]))
                #except: star_lum.append(np.nan) 

                try: star_radii.append(float(cols[6]))
                except: star_radii.append(np.nan)

                try: star_masses.append(float(cols[-2]))
                except: star_masses.append(np.nan)
            
        # Convert to numpy arrays
        log_star_temps = np.log(np.array(star_temps))
        log_star_radii = np.log(np.array(star_radii))
        log_star_masses = np.log(np.array(star_masses))




        file=open(filename, "r")
        lines = file.readlines()[line_skip:] #50
        Npl = len(lines) #50

        for index_planet in range(0, Npl):
            file_list = [i for i in lines[index_planet].split(',')]
            self.planet_names += [file_list[0]]

            self.orbital_period_obs[index_planet] = file_list[2]
            try: self.orbital_period_obs_err[index_planet] = max(abs(float(file_list[3])),abs(float(file_list[4])))
            except: self.orbital_period_obs_err[index_planet] = np.nan

            self.mass_obs[index_planet] = file_list[14]
            try: self.mass_obs_err[index_planet] = max(abs(float(file_list[15])),abs(float(file_list[16])))
            except: self.mass_obs_err[index_planet] = np.nan

            try:
                self.radius_obs[index_planet] = file_list[10]
                try: self.radius_obs_err[index_planet] = max(abs(float(file_list[11])),abs(float(file_list[12])))
                except: self.radius_obs_err[index_planet] = np.nan
            except:
                const, const_err = 18.6/11.2, 6.7/11.2
                alpha, alpha_err = -0.06, 0.07
                mass_planet, mass_planet_err = self.mass_obs[index_planet]*317.8, self.mass_obs_err[index_planet]*317.8
                self.radius_obs[index_planet] = const*(mass_planet)**alpha # Formula from Muller 2024
                self.radius_obs_err[index_planet] =self.radius_obs[index_planet]*np.sqrt((const_err/const)**2+\
                                                     (np.log(mass_planet)*alpha_err)**2+\
                                                     (alpha*mass_planet_err/mass_planet)**2)

            try:
                self.temp_star_obs[index_planet] = file_list[19]
                try: self.temp_star_obs_err[index_planet] = max(abs(float(file_list[20])),abs(float(file_list[21])))
                except: self.temp_star_obs_err[index_planet] = np.nan
            except: 
                log_temp = np.interp(np.log(self.mass_obs[index_planet]),np.flip(log_star_masses),np.flip(log_star_temps))
                self.temp_star_obs[index_planet] = np.exp(log_temp)
                self.temp_star_obs_err[index_planet] = np.nan

            try:
                self.radius_star_obs[index_planet] = file_list[23]
                try: self.radius_star_obs_err[index_planet] = max(abs(float(file_list[24])),abs(float(file_list[25])))
                except: self.radius_star_obs_err[index_planet] = np.nan
            except: 
                log_radius = np.interp(np.log(self.mass_obs[index_planet]),np.flip(log_star_masses),np.flip(log_star_radii))
                self.radius_star_obs[index_planet] = np.exp(log_radius)
                self.radius_star_obs_err[index_planet] = np.nan

            self.mass_star_obs[index_planet] = file_list[27]
            try: self.mass_star_obs_err[index_planet] = max(abs(float(file_list[28])),abs(float(file_list[29])))
            except: self.mass_star_obs_err[index_planet] = np.nan

            try: 
                self.orbital_dist_obs[index_planet] = file_list[6]
                self.orbital_dist_obs_err[index_planet] = max(abs(float(file_list[7])),abs(float(file_list[8])))
            except: 
                self.orbital_dist_obs[index_planet] = np.cbrt(7.496e-6*self.orbital_period_obs[index_planet]**2)
                self.orbital_dist_obs_err[index_planet] = (1/3)*np.cbrt(7.496e-6*self.mass_star_obs[index_planet]/self.orbital_period_obs[index_planet])*\
                                        np.sqrt(self.orbital_period_obs[index_planet]**2*self.mass_star_obs_err[index_planet]**2/self.mass_star_obs[index_planet]**2 + \
                                        4*self.orbital_period_obs_err[index_planet]**2)

        albedo = 0
        self.temp_eff_obs = self.temp_star_obs*np.sqrt(np.sqrt(1-albedo))*np.sqrt(self.radius_star_obs*0.00465047/self.orbital_dist_obs/2)
        sigma_SB = 5.670374419e-5
        self.flux_obs = sigma_SB*self.temp_star_obs**4*(self.radius_star_obs*0.00465047/self.orbital_dist_obs)**2  # 0.00465047 is the solar radius in au

        self.flux_obs_err = 2*self.flux_obs* np.sqrt((2*self.temp_star_obs_err/self.temp_star_obs)**2 + \
                                                    (self.radius_star_obs_err/self.radius_star_obs)**2 + \
                                                    (self.orbital_dist_obs_err/self.orbital_dist_obs)**2)
        
        #self.get_flux()
        
    # In case you want to interpolate your own inventet gas giant or an specific exoplanet you should put
    # HJ_population = False and choose one of the options and put the appropriate number of planet for each.    
    # You can also define your own option and fill the arrays here.
    # The flux can be calculated with host star's properties
    # Mass and radius are in Jovian units, flux in cgs and period in days.  

    def my_planets(self, which_planets):

        if which_planets == "my_own":
            # Example with inveneted exoplanets to check which error affects more significantly the magnetic stds.
            self.planet_names               = ["my_planet"]
            self.mass_obs                   = np.array([1])
            self.mass_obs_err               = np.array([0.1])
            self.radius_obs                 = np.array([1.5])
            self.radius_obs_err             = np.array([0.1])
            self.orbital_period_obs         = np.array([1])
            self.orbital_period_obs_err     = np.array([1e-6])
            self.flux_obs                   = np.array([1e9])
            self.flux_obs_err               = np.array([1e8])
        elif which_planets == "errors":
            # Example with inveneted exoplanets to check which error affects more significantly the magnetic stds.
            self.planet_names               = ["high_err_mass", "high_err_radius", "high_err_flux", "middle_err_mass", "middle_err_radius", "middle_err_flux", 
                                            "low_err_mass", "low_err_radius", "low_err_flux", "middle_err_all", "low_err_all"]
            self.mass_obs                   = np.array([1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      1,      ])
            self.mass_obs_err               = np.array([1,      0,      0,      0.1,    0,      0,      0.01,   0,      0,      0.1,    0.01,  ])
            self.radius_obs                 = np.array([1.5,    1.5,    1.5,    1.5,    1.5,    1.5,    1.5,    1.5,    1.5,    1.5,    1.5,    ])
            self.radius_obs_err             = np.array([0,      1.5,    0,      0,      0.15,   0,      0.,     0.015,  0,      0.15,   0.015,  ])
            self.orbital_period_obs         = np.array([2,      2,      2,      2,      2,      2,      2,      2,      2,      2,      2,      ])
            self.orbital_period_obs_err     = np.array([1e-5,   1e-5,   1e-5,   1e-5,   1e-5,   1e-5,   1e-5,   1e-5,   1e-5,   1e-5,   1e-5,   ])
            self.flux_obs                   = np.array([1e9,    1e9,    1e9,    1e9,    1e9,    1e9,    1e9,    1e9,    1e9,    1e9,    1e9,    ])
            self.flux_obs_err               = np.array([0,      0,      1e9,    0,      0,      1e8,    0,      0,      1e7,    1e8,    1e7,    ])
        elif which_planets == "hats53b":
            # Example with a single exoplanet
            self.planet_names               = ["HATS-53_b"]
            self.mass_obs                   = np.array([0.595])
            self.mass_obs_err               = np.array([0.089])
            self.radius_obs                 = np.array([1.340])
            self.radius_obs_err             = np.array([0.056])
            self.orbital_period_obs         = np.array([3.853777])
            self.orbital_period_obs_err     = np.array([3.8e-06])   
            self.flux_obs                   = np.array([6.677e+08])
            self.flux_obs_err               = np.array([6.1e+07])
        elif which_planets == "cauley19":
            # Cauley 19 estimates with their data
            self.planet_names               = ["HD 179949 b",  "HD 189733 b",    "tau Boo b",  "ups And b"]
            self.mass_obs                   = np.array([1.04,       1.14,       6.13,       0.69])
            self.mass_obs_err               = np.array([0.08,       0.03,       0.34,       0.06])
            self.radius_obs                 = np.array([1.22,       1.14,       1.13,       1.25])
            self.radius_obs_err             = np.array([0.18,       0.03,       0.17,       0.19])
            self.orbital_period_obs         = np.array([3.0925,     2.2186,     3.3124,     4.6170])
            self.orbital_period_obs_err     = np.array([3.2e-05,    1.5e-07,    6.9e-06,    2.3e-05]) 

            self.temp_star_obs              = np.array([6170,      5040,      6387,      6213     ])
            self.temp_star_obs_err          = np.array([50,        50,        50,        50       ])
            self.radius_star_obs            = np.array([1.23,      0.76,      1.48,      1.64     ])
            self.radius_star_obs_err        = np.array([0.03,      0.01,      0.08,      0.04     ])
            self.orbital_dist_obs           = np.array([0.0445,    0.0313,    0.0487,    0.0591    ])
            self.orbital_dist_obs_err       = np.array([0.0005,    0.0004,    0.0004,    0.0006   ])

            self.get_flux()

        elif which_planets == "cauley19nasa":
            # Cauley 19 estimates with NASA default data
            self.planet_names               = ["HD 179949 b",  "HD 189733 b",    "tau Boo b",  "ups And b"]
            self.mass_obs                   = np.array([0.916,      1.130,      4.320,      0.688     ])
            self.mass_obs_err               = np.array([0.076,      0.080,      0.040,      0.004     ])
            self.radius_obs                 = np.array([1.182,      1.130,      1.077,      1.202     ])
            self.radius_obs_err             = np.array([0.634,      0.010,      0.668,      0.627     ])
            self.orbital_period_obs         = np.array([3.092514,   2.218576,   3.312457,   4.617033  ])
            self.orbital_period_obs_err     = np.array([3.2e-05,    1.5e-07,    6.9e-06,    2.3e-05   ])
            self.flux_obs                   = np.array([0.696e9,    0.405e9,    1.815e9,    0.288e9   ])
            self.flux_obs_err               = np.array([0.035e9,    0.028e9,    0.30e9,     0.014e9   ]) # Errors for HD 179949 b adn ups And b are not on the NASA arxiv. I took an relative error of 5% arbitratritly
        elif which_planets == "seidel26":
            # Seidel 26 estimates
            self.planet_names               = ["WASP-76b",  "KELT-20b",  "WASP-121b",  "WASP-178b", "TOI-1518b", "HAT-P-70b",  "WASP-189b"]
            self.mass_obs                   = np.array([0.92,       2.16,       1.18,       1.41,       1.83,       1.66,       1.99])
            self.mass_obs_err               = np.array([0.03,       0.21,       0.07,       0.40,       0.47,       0.22,       0.16])
            self.radius_obs                 = np.array([1.83,       1.83,       1.87,       1.94,       1.88,       1.87,       1.62])
            self.radius_obs_err             = np.array([0.06,       0.07,       0.04,       0.06,       0.04,       0.15,       0.02])
            self.orbital_period_obs         = np.array([1.809882,   3.474108,   1.274925,   3.344829,   1.902611,   2.744325,   2.724033  ])
            self.orbital_period_obs_err     = np.array([6.4e-07,    1.9e-06,    1.5e-07,    1.2e-06,    4.3e-07,    7.9e-07,    2e-06     ]) # Error for WASP-189 b are not on the NASA arxiv. I took 2e-6, a hight value in comparison to others
            self.flux_obs                   = np.array([5.571e9,    5.912e9,    7.642e9,    8.431e9,    9.527e9,    9.611e9,    10.96e9   ])
            self.flux_obs_err               = np.array([0.51e9,     0.97e9,     0.31e9,     0.94e9,     0.79e9,     3.4e9,      0.67e9    ])
        elif which_planets == "seidel26nasa":
            # Seidel 26 estimates with NASA data
            self.planet_names               = ["WASP-76 b", "KELT-20b", "WASP-121 b", "WASP-178 b", "TOI-1518 b", "HAT-P-70 b", "WASP-189 b"]
            self.mass_obs                   = np.array([0.894,      3.382,      1.170,      1.660,      1.830,      6.780,      1.990     ])
            self.mass_obs_err               = np.array([0.014,      0.17,       0.043,      0.120,      0.470,      0.34,       0.160     ]) # Errors for KELT-20b and HAT-P-70 b are not in the NASA arxiv, as M is just an upper limit. I took an relative error of 5% arbitratritly
            self.radius_obs                 = np.array([1.854,      1.741,      1.742,      1.810,      1.878,      1.870,      1.619     ])
            self.radius_obs_err             = np.array([0.077,      0.074,      0.006,      0.090,      0.042,      0.150,      0.021     ])
            self.orbital_period_obs         = np.array([1.809882,   3.474108,   1.274925,   3.344829,   1.902611,   2.744325,   2.724033  ])
            self.orbital_period_obs_err     = np.array([6.4e-07,    1.9e-06,    1.5e-07,    1.2e-06,    4.3e-07,    7.9e-07,    2e-06     ]) # Errors for WASP-189 b are not on the NASA arxiv. I took 2e-6, a hight value in comparison to others
            self.flux_obs                   = np.array([5.571e9,    5.912e9,    7.642e9,    8.431e9,    9.527e9,    9.611e9,    10.96e9   ])
            self.flux_obs_err               = np.array([0.51e9,     0.97e9,     0.31e9,     0.94e9,     0.79e9,     3.4e9,      0.67e9    ])
        
        self.temp_eff_obs = np.ones(len(self.planet_names))


    def mask_population(self, mask):

        new_planet_names = []
        for index_planet, name in enumerate(self.planet_names):
            if mask[index_planet]: new_planet_names += [self.planet_names[index_planet]]
        self.planet_names = new_planet_names

        self.n_planets = len(self.planet_names)

        # NASA data
        self.mass_obs                    = self.mass_obs[mask]
        self.mass_obs_err                = self.mass_obs_err[mask]
        self.temp_eff_obs                = self.temp_eff_obs[mask]
        self.flux_obs                    = self.flux_obs[mask]
        self.flux_obs_err                = self.flux_obs_err[mask]
        self.radius_obs                  = self.radius_obs[mask]
        self.radius_obs_err              = self.radius_obs_err[mask]
        self.orbital_period_obs          = self.orbital_period_obs[mask]
        self.orbital_period_obs_err      = self.orbital_period_obs_err[mask]

        # Interpolated data for inner models
        self.heat_eff_inner_est          = self.heat_eff_inner_est[mask]
        self.heat_eff_inner_est_err      = self.heat_eff_inner_est_err[mask]
        self.brms_inner_est              = self.brms_inner_est[mask]
        self.brms_inner_est_err          = self.brms_inner_est_err[mask]
        self.brms_inner_surf_est         = self.brms_inner_surf_est[mask]
        self.brms_inner_surf_est_err     = self.brms_inner_surf_est_err[mask]

        # Interpolated data for outer models
        self.heat_eff_outer_est          = self.heat_eff_outer_est[mask]
        self.heat_eff_outer_est_err      = self.heat_eff_outer_est_err[mask]
        self.brms_outer_est              = self.brms_outer_est[mask]
        self.brms_outer_est_err          = self.brms_outer_est_err[mask]
        self.brms_outer_surf_est         = self.brms_outer_surf_est[mask]
        self.brms_outer_surf_est_err     = self.brms_outer_surf_est_err[mask]

        # Interpolated data for only irradiated models
        self.radius_irr_est              = self.radius_irr_est[mask]
        self.radius_irr_est_err          = self.radius_irr_est_err[mask]
        self.brms_irr_est                = self.brms_irr_est[mask]
        self.brms_irr_est_err            = self.brms_irr_est_err[mask]
        self.brms_irr_surf_est           = self.brms_irr_surf_est[mask]
        self.brms_irr_surf_est_err       = self.brms_irr_surf_est_err[mask]
