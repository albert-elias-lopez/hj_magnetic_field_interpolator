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

dip_factor = 1/np.sqrt(2)/2

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


ages_table = np.array([5])

profile_number = 6

masses = np.array([0.3,0.4,0.7,0.5,1,1.5,2,3,4,6,8,12])

matrix_flux_list_loaded = []

names_flux = []
for index_flux,flux in enumerate(fluxes):
    names_flux += ["F_{0:.2e}".format(flux).replace('.', '_').replace('+', '').replace('e07', 'e7').replace('e08', 'e8').replace('e09', 'e9')]

names_mass = []
for index_mass,mass in enumerate(masses):
    names_mass += ["{0:.2f}_MJ".format(mass).replace('.', '_')]

for index_mass, mass in enumerate(masses):

    if not os.path.exists("./" + names_mass[index_mass]):
        os.makedirs("./" + names_mass[index_mass])

    for index_flux, flux in enumerate(fluxes):
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
                
                profile = data.profile_data(profile_number=profile_number)
                #age_profile = data.profile_data(profile_number=profile_number).header("star_age")/1e9
                #print("Run " + name + ", profile " + str(profile_number) + " corresponds to age: " + f"{age_profile:.2f}" + " Gyr")
                age = data.history_data.data("star_age")/1e9
                age_loc = np.where(np.logical_and(age>=ages_table[0], age<=ages_table[0]*1.1))
                age_loc = age_loc[0][0]
                radius = 10**(data.history_data.data("log_R"))[age_loc]*9.731
                dyn_radius = data.history_data.data("dyn_radius")[age_loc]/6.957e+10
                radius_factor = dyn_radius/10**data.history_data.data("log_R")[age_loc]
                bscaling = data.history_data.data("b_scaling")[age_loc]
                bscaling_dip_surf = bscaling*radius_factor**3*dip_factor
                dyn_lum = data.history_data.data("dyn_luminosity")[age_loc]*3.846e33
                #print("R = " + f"{radius:.2f}" + " RJ, Bdyn = " + f"{bscaling:.2f}" + " G, Bdyn,surf = " + f"{bscaling_dip_surf:.2f}" + " G" )
                #print("----------------------------------------------------------------------------------------" )
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
                
                profile = data.profile_data(profile_number=profile_number)
                #age_profile = data.profile_data(profile_number=profile_number).header("star_age")/1e9
                #print("Run " + name + ", profile " + str(profile_number) + " corresponds to age: " + f"{age_profile:.2f}" + " Gyr")
                age = data.history_data.data("star_age")/1e9
                age_loc = np.where(np.logical_and(age>=ages_table[0], age<=ages_table[0]*1.1))
                age_loc = age_loc[0][0]
                radius = 10**(data.history_data.data("log_R"))[age_loc]*9.731
                dyn_radius = data.history_data.data("dyn_radius")[age_loc]/6.957e+10
                radius_factor = dyn_radius/10**data.history_data.data("log_R")[age_loc]
                bscaling = data.history_data.data("b_scaling")[age_loc]
                bscaling_dip_surf = bscaling*radius_factor**3*dip_factor
                dyn_lum = data.history_data.data("dyn_luminosity")[age_loc]*3.846e33
                #print("R = " + f"{radius:.2f}" + " RJ, Bdyn = " + f"{bscaling:.2f}" + " G, Bdyn,surf = " + f"{bscaling_dip_surf:.2f}" + " G" )
                #print("----------------------------------------------------------------------------------------" )
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

        filename = names_mass[index_mass] + "/" + names_flux[index_flux] + ".txt"
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
        
        filename = names_mass[index_mass] + "/" + names_flux[index_flux] + "_eps_outer.txt"
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

