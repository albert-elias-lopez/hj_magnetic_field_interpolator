"""
Generates the magnetic field tables read by table_reader.py from the MESA runs of the planetary
evolution grid (one table per mass and flux, plus one per period for the tables with rotation).

    python table_generator.py                  # tables with the rotation factor: <mass>_MJ/P_<period>/F_<flux>[_eps_outer].txt
    python table_generator.py --no_rotation    # tables without rotation: <mass>_MJ/F_<flux>[_eps_outer].txt

The MESA runs (path_sims) and the module radial_functions, which computes the rotating convective
velocities and magnetic fields from the MESA profiles, are not included in this repository. They can
be provided upon request.
"""
import sys
import os
import argparse
import numpy as np
import mesa_reader

# Hard-drive location of the MESA runs (depends on the computer)
path_sims = "/home/albert/mesa_runs/population_grid/"

# Dipolar factor for the surface field of the tables without rotation
dip_factor = 1/np.sqrt(2)

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

masses = np.array([0.3,0.4,0.7,0.5,1,1.5,2,3,4,6,8,12])

alphas_MLT = 2
# Top of the dynamo region in the tables with rotation. P = 1e6 bar lies deeper than Rm = 50, so its dipolar
# extrapolation to the surface gives a smaller B_dip,surf: Rm = 50 is used for the inner heating models
# (upper estimate) and P = 1e6 bar for the outer heating models (lower estimate)
rm_threshold = 50       # Magnetic Reynolds number threshold
p_threshold = 1e6       # Pressure threshold (bar)
ages_table = np.array([5])  # Age of the tables (Gyr)
profile_number = 6

names_flux = []
for index_flux,flux in enumerate(fluxes):
    names_flux += ["F_{0:.2e}".format(flux).replace('.', '_').replace('+', '').replace('e07', 'e7').replace('e08', 'e8').replace('e09', 'e9')]

names_period = []
for index_period,period in enumerate(periods):
    names_period += ["P_{0:.3f}".format(period).replace('.', '_')]

names_mass = []
for index_mass,mass in enumerate(masses):
    names_mass += ["{0:.2f}_MJ".format(mass).replace('.', '_')]


def read_run(index_mass, index_flux, heat_eff, heating_type):
    # Read the MESA run of one model at the age of the tables. Raises an error if the run does not exist
    # or did not reach that age (the planet "died")
    mass_folder = names_mass[index_mass] if heating_type == "inner" else names_mass[index_mass] + "_eps_outer"
    name = mass_folder + "/" + names_flux[index_flux] + "/LOGS_" + names_flux[index_flux] + "_eps_{0:.5f}".format(heat_eff).replace('.', '_')
    data = mesa_reader.MesaLogDir(log_path=path_sims + name,
                                profile_prefix='profile',
                                profile_suffix='data',
                                history_file='history.data',
                                index_file='profiles.index',
                                memoize_profiles=False)

    profile = data.profile_data(profile_number=profile_number)
    age = data.history_data.data("star_age")/1e9
    age_loc = np.where(np.logical_and(age>=ages_table[0], age<=ages_table[0]*1.1))
    age_loc = age_loc[0][0]
    radius = 10**(data.history_data.data("log_R"))[age_loc]*9.731
    dyn_radius = data.history_data.data("dyn_radius")[age_loc]/6.957e+10
    radius_factor = dyn_radius/10**data.history_data.data("log_R")[age_loc]
    dyn_lum = data.history_data.data("dyn_luminosity")[age_loc]*3.846e33
    return data, profile, age_loc, radius, radius_factor, dyn_lum


def print_reading(mass, flux, heat_eff, heating_type):
    if heating_type == "inner": print("Read M_" + f"{mass:.2f}_" + "F_" + f"{flux:.2e}_" + "eps_" + f"{heat_eff:.5f}" )
    else: print("Read M_" + f"{mass:.2f}_" + "F_" + f"{flux:.2e}_" + "eps_outer_" + f"{heat_eff:.4f}" )


def write_table(filename, matrix_flux):
    # One row per heating efficiency, only for the models that reached the age of the tables (R > 0)
    header = "  eps_ext(%)         R(RJ)           Bdyn(G)        Bdipsurf(G)     Ldyn(erg s-1)"
    with open(filename, 'w') as f:
        f.writelines("# " + header + '\n')
        for row in matrix_flux:
            if row[1] > 0:
                f.writelines(" ".join("{:.10e}".format(value) for value in row) + " " + '\n')


def make_tables_no_rotation():
    # B_dyn is the scaling-law field computed during the MESA run (history column b_scaling)
    for index_mass, mass in enumerate(masses):
        os.makedirs("./" + names_mass[index_mass], exist_ok=True)

        for index_flux, flux in enumerate(fluxes):
            matrices = {"inner": np.zeros((len(heat_effs), 5)), "outer": np.zeros((len(heat_effs), 5))}

            for index_heat_eff, heat_eff in enumerate(heat_effs):
                for heating_type in ["inner", "outer"]:
                    print_reading(mass, flux, heat_eff, heating_type)
                    try:
                        data, profile, age_loc, radius, radius_factor, dyn_lum = read_run(index_mass, index_flux, heat_eff, heating_type)
                        bscaling = data.history_data.data("b_scaling")[age_loc]
                        bscaling_dip_surf = bscaling*radius_factor**3*dip_factor
                    except:
                        print("Planet died")
                        radius, bscaling, bscaling_dip_surf, dyn_lum = 0, 0, 0, 0

                    if bscaling < 0:   # Unphysical
                        bscaling, bscaling_dip_surf, dyn_lum = 0, 0, 0
                    matrices[heating_type][index_heat_eff] = [heat_eff, radius, bscaling, bscaling_dip_surf, dyn_lum]

            write_table(names_mass[index_mass] + "/" + names_flux[index_flux] + ".txt", matrices["inner"])
            write_table(names_mass[index_mass] + "/" + names_flux[index_flux] + "_eps_outer.txt", matrices["outer"])


def make_tables_rotation():
    # B_dyn is recomputed from the MESA profile with the rotating convective velocity (Fuentes 2022)
    # for each orbital period, assuming tidal locking (rotation period = orbital period)
    import radial_functions  # Not public, see the docstring at the top

    for index_mass, mass in enumerate(masses):
        os.makedirs("./" + names_mass[index_mass], exist_ok=True)

        for index_flux, flux in enumerate(fluxes):
            for index_period, period in enumerate(periods):
                os.makedirs("./" + names_mass[index_mass] + "/" + names_period[index_period], exist_ok=True)
                frequency = 2*np.pi/(np.array(period)*24*3600)
                matrices = {"inner": np.zeros((len(heat_effs), 5)), "outer": np.zeros((len(heat_effs), 5))}

                for index_heat_eff, heat_eff in enumerate(heat_effs):
                    for heating_type in ["inner", "outer"]:
                        print_reading(mass, flux, heat_eff, heating_type)
                        try:
                            data, profile, age_loc, radius, radius_factor, dyn_lum = read_run(index_mass, index_flux, heat_eff, heating_type)
                            if alphas_MLT == None: setattr(profile, "alpha_mlt", 2)
                            else: setattr(profile, "alpha_mlt", alphas_MLT)

                            if heating_type == "inner":
                                # Top of the dynamo region where the rotating Rm drops below rm_threshold (upper estimate)
                                conv_vel_rot  = radial_functions.get_v_conv_rot(profile, frequency)
                                rm_profile_rot = radial_functions.get_reynolds_mag(profile, conv_vel_rot, type_reynolds="local")
                                try: loc_dyn = np.where(rm_profile_rot - rm_threshold < 0, rm_profile_rot - rm_threshold, -np.inf).argmax()
                                except: loc_dyn = 0
                            else:
                                # Top of the dynamo region where the pressure drops below p_threshold (deeper, lower estimate)
                                try: loc_dyn = np.where(profile.pressure/1e6 - p_threshold < 0, profile.pressure/1e6 - p_threshold, -np.inf).argmax()
                                except: loc_dyn = 0

                            bscaling = radial_functions.get_b_field(profile, loc_dyn, frequency, mlt_conv=True)
                            bscaling_dip_surf = radial_functions.get_b_field_surf(profile, bscaling, loc_dyn)
                        except:
                            print("Planet died")
                            radius, bscaling, bscaling_dip_surf, dyn_lum = 0, 0, 0, 0

                        if bscaling < 0:   # Unphysical
                            bscaling, bscaling_dip_surf, dyn_lum = 0, 0, 0
                        matrices[heating_type][index_heat_eff] = [heat_eff, radius, bscaling, bscaling_dip_surf, dyn_lum]

                folder = names_mass[index_mass] + "/" + names_period[index_period] + "/"
                write_table(folder + names_flux[index_flux] + ".txt", matrices["inner"])
                write_table(folder + names_flux[index_flux] + "_eps_outer.txt", matrices["outer"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate the magnetic field tables from the MESA runs.")
    parser.add_argument("--no_rotation", dest="rotation", action="store_false",
                        help="Generate the tables without the rotation factor (default: with rotation).")
    args = parser.parse_args()

    # radial_functions is looked for in the parent folder of the repository
    sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

    if args.rotation: make_tables_rotation()
    else: make_tables_no_rotation()
