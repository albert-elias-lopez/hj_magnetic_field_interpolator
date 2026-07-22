from operator import index
import sys
import os
import numpy as np
import mesa_reader
# Name of the current directory
current = os.path.dirname(os.path.realpath(__file__))
# Getthe parent directory name
parent = os.path.dirname(current)
# Add the parent directory to the sys.path
sys.path.append(parent)

import radial_functions

# Hard-drive location (depends on the computer)
path_sims = "/home/albert/mesa_runs/population_grid/"
# Path to the run directories

fluxes = np.array([1.00000000e+07, 1.25892541e+07, 1.58489319e+07, 1.99526231e+07,
                    2.51188643e+07, 3.16227766e+07, 3.98107171e+07, 5.01187234e+07,
                    6.30957344e+07, 7.94328235e+07, 1.00000000e+08, 1.25892541e+08,
                    1.58489319e+08, 1.99526231e+08, 2.51188643e+08, 3.16227766e+08,
                    3.98107171e+08, 5.01187234e+08, 6.30957344e+08, 7.94328235e+08,
                    1.00000000e+09, 1.25892541e+09, 1.58489319e+09, 1.99526231e+09,
                    2.51188643e+09, 3.16227766e+09, 3.98107171e+09, 5.01187234e+09,
                    6.30957344e+09, 7.94328235e+09, 1.00000000e+10, 1.25892541e+10,
                    1.58489319e+10])

periods = np.array([  0.1       ,   0.14384499,   0.20691381,   0.29763514,
                    0.42813324,   0.61584821,   0.88586679,   1.27427499,
                    1.83298071,   2.6366509 ,   3.79269019,   5.45559478,
                    7.8475997 ,  11.28837892,  16.23776739,  23.35721469,
                    33.59818286,  48.32930239,  69.51927962, 100.        ])

heat_effs = np.array([0.0, 0.00001, 0.00002, 0.00005,
                        0.0001, 0.0002, 0.0005,
                        0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 
                        0.1, 0.2, 0.4, 0.8, 1. , 2. , 4. ])

alphas_MLT = 2

rm_threshold = 50
p_threshold = 1e6

ages_table = np.array([5])

profile_number = 6

masses = np.array([0.3,0.4,0.7,0.5,1,1.5,2,3,4,6,8,12])

matrix_flux_list_loaded = []

names_flux = []
for index_flux,flux in enumerate(fluxes):
    names_flux += ["F_{0:.2e}".format(flux).replace('.', '_').replace('+', '').replace('e07', 'e7').replace('e08', 'e8').replace('e09', 'e9')]

names_period = []
for index_period,period in enumerate(periods):
    names_period += ["P_{0:.3f}".format(period).replace('.', '_')]

names_mass = []
for index_mass,mass in enumerate(masses):
    names_mass += ["{0:.2f}_MJ".format(mass).replace('.', '_')]

for index_mass, mass in enumerate(masses):

    if not os.path.exists("./" + names_mass[index_mass]):
        os.makedirs("./" + names_mass[index_mass])

    for index_flux, flux in enumerate(fluxes):
        for index_period, period in enumerate(periods):

            if not os.path.exists("./" + names_mass[index_mass] + "/" + names_period[index_period]):
                os.makedirs("./" + names_mass[index_mass] + "/" + names_period[index_period])

            frequency = 2*np.pi/(np.array(period)*24*3600)

            matrix_flux = np.zeros((len(heat_effs), 5)) 
            matrix_flux_eps_outer = np.zeros((len(heat_effs), 5))  
            for index_heat_eff, heat_eff in enumerate(heat_effs):
                name = names_mass[index_mass] + "/" + names_flux[index_flux] + "/LOGS_" + names_flux[index_flux] + "_eps_{0:.5f}".format(heat_eff).replace('.', '_')
                run = path_sims + name
                print("Read M_" + f"{mass:.2f}_" + "F_" + f"{flux:.2e}_" + "eps_" + f"{heat_eff:.5f}" )

                try:
                    data = mesa_reader.MesaLogDir(log_path=run,
                                                profile_prefix='profile',
                                                profile_suffix='data',
                                                history_file='history.data',
                                                index_file='profiles.index',
                                                memoize_profiles=False)
                    
                    age = data.history_data.data("star_age")/1e9
                    age_loc = np.where(np.logical_and(age>=ages_table[0], age<=ages_table[0]*1.1))
                    age_loc = age_loc[0][0]
                    radius = 10**(data.history_data.data("log_R"))[age_loc]*9.731
                    dyn_radius = data.history_data.data("dyn_radius")[age_loc]/6.957e+10
                    radius_factor = dyn_radius/10**data.history_data.data("log_R")[age_loc]
                    dyn_lum = data.history_data.data("dyn_luminosity")[age_loc]*3.846e33

                    profile = data.profile_data(profile_number=profile_number)
                    if alphas_MLT == None: setattr(profile, "alpha_mlt", 2)
                    else: setattr(profile, "alpha_mlt", alphas_MLT)

                    conv_vel_rot  = radial_functions.get_v_conv_rot(profile, frequency)
                    rm_profile_rot = radial_functions.get_reynolds_mag(profile, conv_vel_rot, type_reynolds="local")
                    try: loc_dyn_rm50_rot = np.where(rm_profile_rot - rm_threshold < 0, rm_profile_rot - rm_threshold, -np.inf).argmax()
                    except: loc_dyn_rm50_rot = 0
                    
                    try: loc_dyn = np.where(profile.pressure/1e6 - p_threshold < 0, profile.pressure/1e6 - p_threshold, -np.inf).argmax()
                    except: loc_dyn = 0

                    bscaling = radial_functions.get_b_field(profile, loc_dyn_rm50_rot, frequency, mlt_conv=True)
                    bscaling_dip_surf = radial_functions.get_b_field_surf(profile, bscaling, loc_dyn_rm50_rot)

                except:
                    print("Planet died")
                    #print("----------------------------------------------------------------------------------------" )
                    radius = 0
                    bscaling = 0
                    bscaling_dip_surf = 0
                    dyn_lum = 0

                matrix_flux[index_heat_eff,0] = heat_eff
                matrix_flux[index_heat_eff,1] = radius
                matrix_flux[index_heat_eff,2] = bscaling
                matrix_flux[index_heat_eff,3] = bscaling_dip_surf
                matrix_flux[index_heat_eff,4] = dyn_lum


                name = names_mass[index_mass] + "_eps_outer/" + names_flux[index_flux] + "/LOGS_" + names_flux[index_flux] + "_eps_{0:.5f}".format(heat_eff).replace('.', '_')
                run = path_sims + name
                print("Read M_" + f"{mass:.2f}_" + "F_" + f"{flux:.2e}_" + "eps_outer_" + f"{heat_eff:.4f}" )
                try:
                    data = mesa_reader.MesaLogDir(log_path=run,
                                                profile_prefix='profile',
                                                profile_suffix='data',
                                                history_file='history.data',
                                                index_file='profiles.index',
                                                memoize_profiles=False)
                    
                    age = data.history_data.data("star_age")/1e9
                    age_loc = np.where(np.logical_and(age>=ages_table[0], age<=ages_table[0]*1.1))
                    age_loc = age_loc[0][0]
                    radius = 10**(data.history_data.data("log_R"))[age_loc]*9.731
                    dyn_radius = data.history_data.data("dyn_radius")[age_loc]/6.957e+10
                    radius_factor = dyn_radius/10**data.history_data.data("log_R")[age_loc]
                    dyn_lum = data.history_data.data("dyn_luminosity")[age_loc]*3.846e33
                    
                    profile = data.profile_data(profile_number=profile_number)
                    if alphas_MLT == None: setattr(profile, "alpha_mlt", 2)
                    else: setattr(profile, "alpha_mlt", alphas_MLT)

                    try: loc_dyn = np.where(profile.pressure/1e6 - p_threshold < 0, profile.pressure/1e6 - p_threshold, -np.inf).argmax()
                    except: loc_dyn = 0
                    bscaling = radial_functions.get_b_field(profile, loc_dyn, frequency, mlt_conv=True)
                    bscaling_dip_surf = radial_functions.get_b_field_surf(profile, bscaling, loc_dyn)
                    
                except:
                    print("Planet died")
                    #print("----------------------------------------------------------------------------------------" )
                    radius = 0
                    bscaling = 0
                    bscaling_dip_surf = 0
                    dyn_lum = 0

                if bscaling < 0:
                    bscaling = 0
                    bscaling_dip_surf = 0
                    dyn_lum = 0
                matrix_flux_eps_outer[index_heat_eff,0] = heat_eff
                matrix_flux_eps_outer[index_heat_eff,1] = radius
                matrix_flux_eps_outer[index_heat_eff,2] = bscaling
                matrix_flux_eps_outer[index_heat_eff,3] = bscaling_dip_surf
                matrix_flux_eps_outer[index_heat_eff,4] = dyn_lum

            header = "  eps_ext(%)         R(RJ)           Bdyn(G)        Bdipsurf(G)     Ldyn(erg s-1)"

            filename = names_mass[index_mass] + "/" + names_period[index_period] + "/" + names_flux[index_flux] + ".txt"
            #np.savetxt(filename, matrix_flux, fmt="%.10e", header = header)
            with open(filename, 'w') as f:
                f.writelines("# " + header + '\n')
            with open(filename, 'a') as f:
                for index_heat_eff, heat_eff in enumerate(heat_effs):
                    if matrix_flux[index_heat_eff,1] > 0:
                        f.writelines("{:.10e}".format(matrix_flux[index_heat_eff,0]) + " " + \
                                    "{:.10e}".format(matrix_flux[index_heat_eff,1]) + " " + \
                                    "{:.10e}".format(matrix_flux[index_heat_eff,2]) + " " + \
                                    "{:.10e}".format(matrix_flux[index_heat_eff,3]) + " " + \
                                    "{:.10e}".format(matrix_flux[index_heat_eff,4]) + " " + '\n')
            
            filename = names_mass[index_mass] + "/" + names_period[index_period] + "/" + names_flux[index_flux] + "_eps_outer.txt"
            #np.savetxt(filename, matrix_flux_eps_outer, fmt="%.10e", header = header)
            with open(filename, 'w') as f:
                f.writelines("# " + header + '\n')
            with open(filename, 'a') as f:
                for index_heat_eff, heat_eff in enumerate(heat_effs):
                    if matrix_flux_eps_outer[index_heat_eff,1] > 0:
                        f.writelines("{:.10e}".format(matrix_flux_eps_outer[index_heat_eff,0]) + " " + \
                                    "{:.10e}".format(matrix_flux_eps_outer[index_heat_eff,1]) + " " + \
                                    "{:.10e}".format(matrix_flux_eps_outer[index_heat_eff,2]) + " " + \
                                    "{:.10e}".format(matrix_flux_eps_outer[index_heat_eff,3]) + " " + \
                                    "{:.10e}".format(matrix_flux_eps_outer[index_heat_eff,4]) + " " + '\n')

