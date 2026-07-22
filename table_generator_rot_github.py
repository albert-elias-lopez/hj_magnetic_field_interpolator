from operator import index
import sys
import os
import numpy as np
import mesa_reader

# Calculate the convective velocity when rotation is taken into account, following Fuentes 2022, i.e. v_conv_rot = v_conv * (v_conv/2/H_p/omega)**(1/5)
def get_v_conv_rot(profile, frequency):
    conv_vel_rot = profile.conv_vel*(profile.conv_vel/2/profile.scale_height/6.957e+10/frequency)**(1/5)
    return conv_vel_rot

# Calculate the convective flux
def get_conv_heat_flux(profile, conv_vel):
    np.seterr(divide='ignore')
    conv_heat_flux=2*profile.cp*profile.temperature*\
                    profile.rho**2*conv_vel**3/\
                    profile.pressure/profile.chiT*profile.chiRho*\
                    2/profile.alpha_mlt
    return conv_heat_flux

# Calculate the turnover timescale, i.e. t_turn = H_p/v_conv, where H_p is the pressure scale height
def get_t_turn(profile, conv_vel):
    np.seterr(divide='ignore')
    t_turn = np.divide(profile.scale_height,
                        conv_vel,
                        out=np.zeros_like(profile.scale_height),
                        where=conv_vel!=0)*6.957e+10/3600/24
    return t_turn

# Calculate the Reynolds magnetic number, i.e. Rm = sigma*mu0*L*v_conv, where L is the length scale of the system, 
# which we take as the density scale height H_rho or shell thickness, and sigma is the conductivity
def get_reynolds_mag(profile, conv_vel, type_reynolds="local", loc_rcb=None):
    if type_reynolds=="local":
        length = profile.scale_height*6.957e+10/1e2
    elif type_reynolds=="global":
        radius_convection = profile.radius[loc_rcb:]
        length = np.abs((radius_convection[-1]-radius_convection[0]))*6.957e+10/1e2
    reynolds_mag = profile.sigma*1.2566e-6*length*conv_vel/1e2
    return reynolds_mag

# Calculate the rms magnetic field using the Christensen 2009 scaling law, i.e. 
# B_rms = (2*mu0*0.63*1*(rho_mean)**(1/3)*F_factor*(q_0/1e3)**(2/3))**(1/2), where F_factor is a factor 
# that takes into account the radial distribution of the convective flux and the density, and q_0 is 
# the convective flux at the outer boundary of the dynamo.
def get_b_field(profile, loc_dyn, frequency, mlt_conv=True):

    radius_dynamo = profile.radius[loc_dyn:]
    r_dyn = radius_dynamo[0]*6.957e+10
    r_i = radius_dynamo[-1]*6.957e+10
    
    volume = 4 * np.pi * (r_dyn**3 - r_i**3) / 3 # dynamo shell volume
    rho_mean = (profile.mass[loc_dyn] - profile.mass[-1])*1.989e+33 / volume

    if mlt_conv == True:
        conv_vel = get_v_conv_rot(profile, frequency)
    elif mlt_conv == False:
        conv_vel = profile.conv_vel

    conv_heat_flux = get_conv_heat_flux(profile, conv_vel)
    conv_heat_flux_dyn_surf = 1e6 #conv_heat_flux[loc_dyn]

    dynamo_shell_thickness = r_dyn - r_i 
    scale_height_rho = np.minimum(profile.scale_height*6.957e+10,
                                  dynamo_shell_thickness*np.ones(len(profile.scale_height)))

    scale_height_t = profile.pressure / (profile.rho * profile.grav * profile.grada)
    
    # To calculate F_factor by integrating over the dynamo region it can be done by integrating the mass differential rather than the radius.
    dm = -np.ediff1d(profile.mass, to_end=0)*1.989e+33
    F_factor = (scale_height_rho/scale_height_t*conv_heat_flux/conv_heat_flux_dyn_surf)**(2/3)*(profile.rho/rho_mean)**(1/3)*dm/profile.rho
    # Alternative, it can be done with the radius, with almost the same result
    #dr = -np.ediff1d(profile.radius, to_end=0)*6.957e+10
    #F_factor = (scale_height_rho/scale_height_t*conv_heat_flux/conv_heat_flux_dyn_surf)**(2/3)*(profile.rho/rho_mean)**(1/3)*4*np.pi*profile.radius**2*6.957e+10**2*dr
    
    F_factor = np.sum(F_factor[loc_dyn:])/volume

    b_rms_dyn = 2 * 1.256637e-6 * 0.63 * 1 * (1e3*rho_mean)**(1.0/3.0) * F_factor * (conv_heat_flux_dyn_surf/1e3)**(2.0/3.0)
    b_rms_dyn = np.sqrt(b_rms_dyn)*1e4

    return b_rms_dyn


def get_b_field_surf(profile, b_rms_dyn, loc_dyn):

    radius_dynamo = profile.radius[loc_dyn]
    radius_surf = profile.radius[0]
    radius_factor = radius_dynamo/radius_surf
    b_dip_surf = b_rms_dyn*radius_factor**3/np.sqrt(2)

    #simple_factor = 1-.17/1
    #print(radius_factor**3, simple_factor**3)

    return b_dip_surf



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

                    conv_vel_rot  = get_v_conv_rot(profile, frequency)
                    rm_profile_rot = get_reynolds_mag(profile, conv_vel_rot, type_reynolds="local")
                    try: loc_dyn_rm50_rot = np.where(rm_profile_rot - rm_threshold < 0, rm_profile_rot - rm_threshold, -np.inf).argmax()
                    except: loc_dyn_rm50_rot = 0
                    
                    try: loc_dyn = np.where(profile.pressure/1e6 - p_threshold < 0, profile.pressure/1e6 - p_threshold, -np.inf).argmax()
                    except: loc_dyn = 0

                    bscaling = get_b_field(profile, loc_dyn_rm50_rot, frequency, mlt_conv=True)
                    bscaling_dip_surf = get_b_field_surf(profile, bscaling, loc_dyn_rm50_rot)

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
                    bscaling = get_b_field(profile, loc_dyn, frequency, mlt_conv=True)
                    bscaling_dip_surf = get_b_field_surf(profile, bscaling, loc_dyn)
                    
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

