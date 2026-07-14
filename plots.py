import sys
import os
import numpy as np
import pylab as plt
import matplotlib
import table_predictions_save

def plot_output_grid(masses,
                    fluxes,
                    heat_effs,
                    radii_heat_inner,
                    radii_heat_outer,
                    bscaling_heat_inner,
                    bscaling_heat_outer,
                    bscaling_surf_heat_inner,
                    bscaling_surf_heat_outer):
    
    names_mass = []
    for index_mass,mass in enumerate(masses):
        names_mass += ["{0:.2f}_MJ".format(mass).replace('.', '_')]
    
    flux_grid, heat_eff_grid = np.meshgrid(fluxes, heat_effs, indexing='ij')
    fluxes_flat_general = flux_grid.flatten()
    heat_effs_flat_general = heat_eff_grid.flatten()
    bscaling_heat_inner_flat = bscaling_heat_inner.flatten()
    bscaling_heat_outer_flat = bscaling_heat_outer.flatten()

    for index_mass, mass in enumerate(masses):

        if not os.path.exists("./plots/" + names_mass[index_mass]):
            os.makedirs("./plots/" + names_mass[index_mass])

        fontsize = 26
        plt.rcParams['text.usetex'] = True
        fig = plt.figure(figsize=(11, 8), constrained_layout=True)
        plt.rcParams["figure.figsize"] = (11, 8)
        plt.rcParams['font.size'] = fontsize
        plt.rcParams['lines.linewidth'] = 1
        plt.rcParams["figure.autolayout"] = True
        plt.rcParams["mathtext.fontset"] = "cm"

        # -------------------------------------------------------------------------------------------------------------
        # ------------------------------ Countourf for the eps_heat_inner models -----------------------------------
        # -------------------------------------------------------------------------------------------------------------


        valid_mask = radii_heat_inner[index_mass] == 0.0
        radii_heat_inner[index_mass][valid_mask] = np.nan
        bscaling_heat_inner[index_mass][valid_mask] = np.nan
        bscaling_surf_heat_inner[index_mass][valid_mask] = np.nan

        plt.clf()
        ax = plt.axes()
        cs = ax.contourf(flux_grid, heat_eff_grid, radii_heat_inner[index_mass], levels = np.linspace(.75,3.5,23))
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylim(1e-5,4)
        ax.set_xlabel(r"$F\mathrm{_{\rm irr} \, (erg \, cm^{-2} \, s^{-1})}$", fontsize=int(fontsize*1.3))
        ax.set_ylabel(r"$\mathrm{\epsilon (\%)}$", fontsize=int(fontsize*1.3))
        cbar = fig.colorbar(cs, aspect=20, fraction=0.06, shrink=0.8)
        cbar.set_label(r"$R_P \mathrm{(R_J)}$",fontsize=int(fontsize*1.3),labelpad=5)
        plt.savefig("plots/" + names_mass[index_mass] + "/" + names_mass[index_mass] + "_R.png")

        plt.clf()
        ax = plt.axes()
        cs = ax.contourf(flux_grid, heat_eff_grid, bscaling_heat_inner[index_mass])
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylim(1e-5,4)
        ax.set_xlabel(r"$F\mathrm{_{\rm irr} \, (erg \, cm^{-2} \, s^{-1})}$", fontsize=int(fontsize*1.3))
        ax.set_ylabel(r"$\mathrm{\epsilon (\%)}$", fontsize=int(fontsize*1.3))
        cbar = fig.colorbar(cs, aspect=20, fraction=0.06, shrink=0.8)
        cbar.set_label(r"$B_{\rm rms, \, dyn} \mathrm{(G)}$",fontsize=int(fontsize*1.3),labelpad=5)
        plt.savefig("plots/" + names_mass[index_mass] + "/" + names_mass[index_mass] + "_brms.png")

        plt.clf()
        ax = plt.axes()
        cs = ax.contourf(flux_grid, heat_eff_grid, bscaling_surf_heat_inner[index_mass])
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylim(1e-5,4)
        ax.set_xlabel(r"$F\mathrm{_{\rm irr} \, (erg \, cm^{-2} \, s^{-1})}$", fontsize=int(fontsize*1.3))
        ax.set_ylabel(r"$\mathrm{\epsilon (\%)}$", fontsize=int(fontsize*1.3))
        cbar = fig.colorbar(cs, aspect=20, fraction=0.06, shrink=0.8)
        cbar.set_label(r"$B_{\rm rms, \, surf} \mathrm{(G)}$",fontsize=int(fontsize*1.3),labelpad=5)
        plt.savefig("plots/" + names_mass[index_mass] + "/" + names_mass[index_mass] + "_brms_surf.png")

        # -------------------------------------------------------------------------------------------------------------
        # -------------------------------- Countourf for the eps_heat_outer models ------------------------------------
        # -------------------------------------------------------------------------------------------------------------

        valid_mask = radii_heat_outer[index_mass] == 0
        radii_heat_outer[index_mass][valid_mask] = np.nan
        bscaling_heat_outer[index_mass][valid_mask] = np.nan
        bscaling_surf_heat_outer[index_mass][valid_mask] = np.nan

        plt.clf()
        ax = plt.axes()
        cs = ax.contourf(flux_grid, heat_eff_grid, radii_heat_outer[index_mass], levels = np.linspace(.75,3.5,23))
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylim(1e-5,4)
        ax.set_xlabel(r"$F\mathrm{_{\rm irr} \, (erg \, cm^{-2} \, s^{-1})}$", fontsize=int(fontsize*1.3))
        ax.set_ylabel(r"$\mathrm{\epsilon (\%)}$", fontsize=int(fontsize*1.3))
        cbar = fig.colorbar(cs, aspect=20, fraction=0.06, shrink=0.8)
        cbar.set_label(r"$R_P \mathrm{(R_J)}$",fontsize=int(fontsize*1.3),labelpad=5)
        plt.savefig("plots/" + names_mass[index_mass] + "/" + names_mass[index_mass] + "_R_outer.png")

        plt.clf()
        ax = plt.axes()
        cs = ax.contourf(flux_grid, heat_eff_grid, bscaling_heat_outer[index_mass])
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylim(1e-5,4)
        ax.set_xlabel(r"$F\mathrm{_{\rm irr} \, (erg \, cm^{-2} \, s^{-1})}$", fontsize=int(fontsize*1.3))
        ax.set_ylabel(r"$\mathrm{\epsilon (\%)}$", fontsize=int(fontsize*1.3))
        cbar = fig.colorbar(cs, aspect=20, fraction=0.06, shrink=0.8)
        cbar.set_label(r"$B_{\rm rms, \, dyn} \mathrm{(G)}$",fontsize=int(fontsize*1.3),labelpad=5)
        plt.savefig("plots/" + names_mass[index_mass] + "/" + names_mass[index_mass] + "_brms_outer.png")

        plt.clf()
        ax = plt.axes()
        cs = ax.contourf(flux_grid, heat_eff_grid, bscaling_surf_heat_outer[index_mass])
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylim(1e-5,4)
        ax.set_xlabel(r"$F\mathrm{_{\rm irr} \, (erg \, cm^{-2} \, s^{-1})}$", fontsize=int(fontsize*1.3))
        ax.set_ylabel(r"$\mathrm{\epsilon (\%)}$", fontsize=int(fontsize*1.3))
        cbar = fig.colorbar(cs, aspect=20, fraction=0.06, shrink=0.8)
        cbar.set_label(r"$B_{\rm rms, \, surf} \mathrm{(G)}$",fontsize=int(fontsize*1.3),labelpad=5)
        plt.savefig("plots/" + names_mass[index_mass] + "/" + names_mass[index_mass] + "_brms_surf_outer.png")


        # -------------------------------------------------------------------------------------------------------------
        # ------------------------------ Scatter plots, for actual tri interpolation ----------------------------------
        # -------------------------------------------------------------------------------------------------------------

        plt.clf()
        ax = plt.axes()
        cs = ax.scatter(fluxes_flat_general,radii_heat_inner[index_mass].flatten(),c=heat_effs_flat_general,s=30, cmap='viridis')
        ax.set_xscale('log')
        ax.set_ylim(.75,3.5)
        ax.set_xlabel(r"$F\mathrm{_{\rm irr} \, (erg \, cm^{-2} \, s^{-1})}$", fontsize=int(fontsize*1.3))
        ax.set_ylabel(r"$R_P \mathrm{(R_J)}$", fontsize=int(fontsize*1.3))
        cbar = fig.colorbar(cs, aspect=20, fraction=0.06, shrink=0.8)
        cbar.set_label(r"$\mathrm{\epsilon (\%)}$",fontsize=int(fontsize*1.3),labelpad=5)
        plt.savefig("plots/" + names_mass[index_mass] + "/" + names_mass[index_mass] + "_eps.png")

        plt.clf()
        ax = plt.axes()
        cs = ax.scatter(fluxes_flat_general,radii_heat_outer[index_mass].flatten(),c=heat_effs_flat_general,s=30, cmap='viridis')
        ax.set_xscale('log')
        ax.set_ylim(.75,3.5)
        ax.set_xlabel(r"$F\mathrm{_{\rm irr} \, (erg \, cm^{-2} \, s^{-1})}$", fontsize=int(fontsize*1.3))
        ax.set_ylabel(r"$R_P \mathrm{(R_J)}$", fontsize=int(fontsize*1.3))
        cbar = fig.colorbar(cs, aspect=20, fraction=0.06, shrink=0.8)
        cbar.set_label(r"$\mathrm{\epsilon (\%)}$",fontsize=int(fontsize*1.3),labelpad=5)
        plt.savefig("plots/" + names_mass[index_mass] + "/" + names_mass[index_mass] + "_eps_outer.png")
        plt.savefig("plot.png")

        plt.clf()
        ax = plt.axes()
        cs = ax.scatter(fluxes_flat_general,radii_heat_inner[index_mass].flatten(),c=bscaling_heat_inner[index_mass].flatten(),s=30, cmap='viridis')
        ax.set_xscale('log')
        ax.set_ylim(.75,3.5)
        ax.set_xlabel(r"$F\mathrm{_{\rm irr} \, (erg \, cm^{-2} \, s^{-1})}$", fontsize=int(fontsize*1.3))
        ax.set_ylabel(r"$R_P \mathrm{(R_J)}$", fontsize=int(fontsize*1.3))
        cbar = fig.colorbar(cs, aspect=20, fraction=0.06, shrink=0.8)
        cbar.set_label(r"$B_{\rm rms, \, dyn} \mathrm{(G)}$",fontsize=int(fontsize*1.3),labelpad=5)
        plt.savefig("plots/" + names_mass[index_mass] + "/" + names_mass[index_mass] + "_brms_scatter.png")

        plt.clf()
        ax = plt.axes()
        cs = ax.scatter(fluxes_flat_general,radii_heat_outer[index_mass].flatten(),c=bscaling_heat_outer[index_mass].flatten(),s=30, cmap='viridis')
        ax.set_xscale('log')
        ax.set_ylim(.75,3.5)
        ax.set_xlabel(r"$F\mathrm{_{\rm irr} \, (erg \, cm^{-2} \, s^{-1})}$", fontsize=int(fontsize*1.3))
        ax.set_ylabel(r"$R_P \mathrm{(R_J)}$", fontsize=int(fontsize*1.3))
        cbar = fig.colorbar(cs, aspect=20, fraction=0.06, shrink=0.8)
        cbar.set_label(r"$B_{\rm rms, \, dyn} \mathrm{(G)}$",fontsize=int(fontsize*1.3),labelpad=5)
        plt.savefig("plots/" + names_mass[index_mass] + "/" + names_mass[index_mass] + "_brms_outer_scatter.png")

        plt.clf()
        ax = plt.axes()
        cs = ax.scatter(fluxes_flat_general,radii_heat_inner[index_mass].flatten(),c=bscaling_surf_heat_inner[index_mass].flatten(),s=30, cmap='viridis')
        ax.set_xscale('log')
        ax.set_ylim(.75,3.5)
        ax.set_xlabel(r"$F\mathrm{_{\rm irr} \, (erg \, cm^{-2} \, s^{-1})}$", fontsize=int(fontsize*1.3))
        ax.set_ylabel(r"$R_P \mathrm{(R_J)}$", fontsize=int(fontsize*1.3))
        cbar = fig.colorbar(cs, aspect=20, fraction=0.06, shrink=0.8)
        cbar.set_label(r"$B_{\rm dip, \, surf} \mathrm{(G)}$",fontsize=int(fontsize*1.3),labelpad=5)
        plt.savefig("plots/" + names_mass[index_mass] + "/" + names_mass[index_mass] + "_brms_surf_scatter.png")

        plt.clf()
        ax = plt.axes()
        cs = ax.scatter(fluxes_flat_general,radii_heat_outer[index_mass].flatten(),c=bscaling_surf_heat_outer[index_mass].flatten(),s=30, cmap='viridis')
        ax.set_xscale('log')
        ax.set_ylim(.75,3.5)
        ax.set_xlabel(r"$F\mathrm{_{\rm irr} \, (erg \, cm^{-2} \, s^{-1})}$", fontsize=int(fontsize*1.3))
        ax.set_ylabel(r"$R_P \mathrm{(R_J)}$", fontsize=int(fontsize*1.3))
        cbar = fig.colorbar(cs, aspect=20, fraction=0.06, shrink=0.8)
        cbar.set_label(r"$B_{\rm dip, \, surf} \mathrm{(G)}$",fontsize=int(fontsize*1.3),labelpad=5)
        plt.savefig("plots/" + names_mass[index_mass] + "/" + names_mass[index_mass] + "_brms_outer_surf_scatter.png")
        plt.clf()


def plot_grid(masses,
              fluxes,
              heat_effs,
              radii_heat_inner,
              radii_heat_outer):
    

    
    eps_min = 1e-6
    eps_global_max = 1e2
    
    heat_effs[heat_effs == 0] = eps_min
    heat_effs = 100*heat_effs

    fontsize = 28
    plt.rcParams['text.usetex'] = True
    fig = plt.figure(figsize=(10, 9.5))
    plt.rcParams["figure.figsize"] = (14, 10)
    plt.rcParams['font.size'] = fontsize
    plt.rcParams['lines.linewidth'] = 1
    #plt.rcParams["figure.autolayout"] = True
    plt.rcParams["mathtext.fontset"] = "cm"

    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(left=0.0, right=0.9,
                    bottom=0.05, top=1.05)
    
    facecolors = plt.colormaps['viridis_r'](np.flip(np.linspace(0, 1, len(masses))))

    for index_mass, mass in enumerate(masses):
        
        flux_grid, heat_eff_grid = np.meshgrid(fluxes, heat_effs, indexing='ij')
        fluxes_flat_general = flux_grid.flatten()
        heat_effs_flat_general = heat_eff_grid.flatten()

        valid_mask = (heat_effs_flat_general <= eps_global_max) & (radii_heat_inner[index_mass].flatten() != 0.0)

        ax.scatter(np.log10(fluxes_flat_general[valid_mask]),
                   np.log10(mass),
                   np.log10(heat_effs_flat_general[valid_mask]),
                   color=facecolors[index_mass],s=10)
        
        eps_max=np.zeros(len(fluxes))
        for index_flux, flux in enumerate(fluxes):
            try:
                valid_mask = radii_heat_inner[index_mass,index_flux] != 0.0
                eps_max[index_flux] = np.max(heat_effs[valid_mask])
            except:
                eps_max[index_flux] = 100*eps_min
            if eps_max[index_flux] > eps_global_max: eps_max[index_flux] = eps_global_max
            
        ax.fill_between(np.log10(fluxes),
                   np.log10(mass),
                   np.log10(eps_max),
                   np.log10(fluxes),
                   np.log10(mass),
                   -4,
                   facecolors=facecolors[index_mass], alpha=.7)
        
    ax.set_xlim(7, 10.3)
    ax.set_ylim(-0.55, 1.2)
    ax.set_zlim(-4, 2)
    ax.set_xlabel(r'$F_{\rm irr} \mathrm{(erg \, cm^{-2} \, s^{-1})}$', fontsize=int(fontsize*1.1), labelpad = 20)
    ax.set_ylabel(r'$M_{\rm P} (\mathrm{M_J})$', fontsize=int(fontsize*1.1), labelpad = 20)
    ax.set_zlabel(r'$\epsilon(\%)$', fontsize=int(fontsize*1.1), labelpad = 25)
    ax.set_yticks(np.log10(np.array([0.3, 0.5, 1, 2, 4, 8, 12])))
    ax.set_yticklabels([r"$\mathrm{0.3}$", 
                        r"$\mathrm{0.5}$",
                        r"$\mathrm{1}$",
                        r"$\mathrm{2}$",
                        r"$\mathrm{4}$",
                        r"$\mathrm{8}$",
                        r"$\mathrm{12}$"])
    ax.set_xticks([7,8,9,10])
    ax.set_xticklabels([r"$\mathrm{10^7}$",
                        r"$\mathrm{10^8}$",
                        r"$\mathrm{10^9}$",
                        r"$\mathrm{10^{10}}$"])
    ax.set_zticks([-4,-3,-2,-1,0,1,2])
    ax.tick_params(axis='z', which='major', pad=12)
    ax.set_zticklabels([r"$\mathrm{0}$",
                        r"$\mathrm{10^{-3}}$",
                        r"$\mathrm{10^{-2}}$",
                        r"$\mathrm{10^{-1}}$",
                        r"$\mathrm{10^0}$",
                        r"$\mathrm{10^1}$",
                        r"$\mathrm{10^2}$",])
    #ax.set_zscale('symlog')
    ax.view_init(25,-45)
    ax.set_box_aspect((1, 1, .85))
    plt.savefig("plots/population_grid.png")

    plt.clf()
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(left=0.0, right=0.9,
                    bottom=0.05, top=1.05)

    facecolors = plt.colormaps['viridis_r'](np.flip(np.linspace(0, 1, len(masses))))

    for index_mass, mass in enumerate(masses):
        
        flux_grid, heat_eff_grid = np.meshgrid(fluxes, heat_effs, indexing='ij')
        fluxes_flat_general = flux_grid.flatten()
        heat_effs_flat_general = heat_eff_grid.flatten()

        valid_mask = (heat_effs_flat_general <= eps_global_max) & (radii_heat_outer[index_mass].flatten() != 0.0)

        ax.scatter(np.log10(fluxes_flat_general[valid_mask]),
                   np.log10(mass),
                   np.log10(heat_effs_flat_general[valid_mask]),
                   color=facecolors[index_mass],s=10)
        
        eps_max=np.zeros(len(fluxes))
        for index_flux, flux in enumerate(fluxes):
            try:
                valid_mask = radii_heat_outer[index_mass,index_flux] != 0.0
                eps_max[index_flux] = np.max(heat_effs[valid_mask])
            except:
                eps_max[index_flux] = 100*eps_min
            if eps_max[index_flux] > eps_global_max: eps_max[index_flux] = eps_global_max
            
        ax.fill_between(np.log10(fluxes),
                   np.log10(mass),
                   np.log10(eps_max),
                   np.log10(fluxes),
                   np.log10(mass),
                   -4,
                   facecolors=facecolors[index_mass], alpha=.7)
        
    ax.set_xlim(7, 10.3)
    ax.set_ylim(-0.55, 1.2)
    ax.set_zlim(-4, 2)
    ax.set_xlabel(r'$F_{\rm irr} \mathrm{(erg \, cm^{-2} \, s^{-1})}$', fontsize=int(fontsize*1.1), labelpad = 20)
    ax.set_ylabel(r'$M_{\rm P} (\mathrm{M_J})$', fontsize=int(fontsize*1.1), labelpad = 20)
    ax.set_zlabel(r'$\epsilon(\%)$', fontsize=int(fontsize*1.1), labelpad = 30)
    ax.set_yticks(np.log10(np.array([0.3, 0.5, 1, 2, 4, 8, 12])))
    ax.set_yticklabels([r"$\mathrm{0.3}$", 
                        r"$\mathrm{0.5}$",
                        r"$\mathrm{1}$",
                        r"$\mathrm{2}$",
                        r"$\mathrm{4}$",
                        r"$\mathrm{8}$",
                        r"$\mathrm{12}$"])
    ax.set_xticks([7,8,9,10])
    ax.set_xticklabels([r"$\mathrm{10^7}$",
                        r"$\mathrm{10^8}$",
                        r"$\mathrm{10^9}$",
                        r"$\mathrm{10^{10}}$"])
    ax.set_zticks([-4,-3,-2,-1,0,1,2])
    ax.tick_params(axis='z', which='major', pad=12)
    ax.set_zticklabels([r"$\mathrm{0}$",
                        r"$\mathrm{10^{-3}}$",
                        r"$\mathrm{10^{-2}}$",
                        r"$\mathrm{10^{-1}}$",
                        r"$\mathrm{10^0}$",
                        r"$\mathrm{10^1}$",
                        r"$\mathrm{10^2}$",])
    #ax.set_zscale('symlog')
    ax.view_init(25,-45)
    ax.set_box_aspect((1, 1, .85))
    plt.savefig("plots/population_grid_eps_outer.png")


def plot_population(planets,
                valid_mask = [],
                name_cut = None,
                color_inner = "blue", color_inner_cut = "deepskyblue", color_inner_err = "darkblue",
                color_outer = "green", color_outer_cut = "limegreen", color_outer_err = "darkgreen", 
                color_irr = "k",
                error_sym=False,
                rotation=False):
    
    if len(valid_mask) == 0: 
        ones = np.ones(planets.n_planets)
        valid_mask = [i < 2 for i in ones]

    heat_eff_inner_est      = planets.heat_eff_inner_est[valid_mask]
    heat_eff_inner_est_err  = planets.heat_eff_inner_est_err[valid_mask]
    brms_inner_est          = planets.brms_inner_est[valid_mask]
    brms_inner_est_err      = planets.brms_inner_est_err[valid_mask]
    brms_inner_surf_est     = planets.brms_inner_surf_est[valid_mask]
    brms_inner_surf_est_err = planets.brms_inner_surf_est_err[valid_mask]
    heat_eff_outer_est      = planets.heat_eff_outer_est[valid_mask]
    heat_eff_outer_est_err  = planets.heat_eff_outer_est_err[valid_mask]
    brms_outer_est          = planets.brms_outer_est[valid_mask]
    brms_outer_est_err      = planets.brms_outer_est_err[valid_mask]
    brms_outer_surf_est     = planets.brms_outer_surf_est[valid_mask]
    brms_outer_surf_est_err = planets.brms_outer_surf_est_err[valid_mask]
    brms_irr_est            = planets.brms_irr_est[valid_mask]
    brms_irr_est_err        = planets.brms_irr_est_err[valid_mask]
    brms_irr_surf_est       = planets.brms_irr_surf_est[valid_mask]
    brms_irr_surf_est_err   = planets.brms_irr_surf_est_err[valid_mask]
    radius_irr_est          = planets.radius_irr_est[valid_mask]
    radius_irr_est_err      = planets.radius_irr_est_err[valid_mask]
    mass_obs                = planets.mass_obs[valid_mask]
    mass_obs_err            = planets.mass_obs_err[valid_mask]
    flux_obs                = planets.flux_obs[valid_mask]
    flux_obs_err            = planets.flux_obs_err[valid_mask]
    temp_eff_obs            = planets.temp_eff_obs[valid_mask]
    radius_obs              = planets.radius_obs[valid_mask]
    radius_obs_err          = planets.radius_obs_err[valid_mask]
    orbital_period_obs      = planets.orbital_period_obs[valid_mask]
    orbital_period_obs_err  = planets.orbital_period_obs_err[valid_mask]
    
    label_inner = r"$\epsilon_{\rm inner}$" #r"$B_{\rm dip, \, inner}$"
    label_outer = r"$\epsilon_{\rm outer}$" #r"$B_{\rm dip, \, outer}$"
    label_irr = r"$\epsilon = 0$" #r"$B_{\rm irr}$"

    if rotation:
        if name_cut == None: filename_ending = "_rot"
        else: filename_ending = name_cut + "_rot"
    else:
        if name_cut == None: filename_ending = ""
        else: filename_ending = name_cut
    filename_ending += ".png"

    if not error_sym:
        heat_eff_inner_est_err = np.transpose(heat_eff_inner_est_err)
        brms_inner_est_err = np.transpose(brms_inner_est_err)
        brms_inner_surf_est_err = np.transpose(brms_inner_surf_est_err)
        heat_eff_outer_est_err = np.transpose(heat_eff_outer_est_err)
        brms_outer_est_err = np.transpose(brms_outer_est_err)
        brms_outer_surf_est_err = np.transpose(brms_outer_surf_est_err)
        brms_irr_est_err = np.transpose(brms_irr_est_err)
        brms_irr_surf_est_err = np.transpose(brms_irr_surf_est_err)
    
    if rotation:
        if not os.path.exists("./plots/general_predictions_rot"):
            os.makedirs("./plots/general_predictions_rot")
        if not os.path.exists("./plots/general_predictions_comparison_rot"):
            os.makedirs("./plots/general_predictions_comparison_rot")
    else:
        if not os.path.exists("./plots/general_predictions"):
            os.makedirs("./plots/general_predictions")
        if not os.path.exists("./plots/general_predictions_comparison"):
            os.makedirs("./plots/general_predictions_comparison")

    fontsize = 32
    plt.rcParams['text.usetex'] = True
    fig = plt.figure(figsize=(11, 8), constrained_layout=True)
    plt.rcParams["figure.figsize"] = (11, 8)
    plt.rcParams['font.size'] = fontsize
    plt.rcParams['lines.linewidth'] = 1
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams["mathtext.fontset"] = "cm"

    if rotation: path_general = "plots/general_predictions_rot/"
    else: path_general = "plots/general_predictions/"
    if rotation: path_comparison = "plots/general_predictions_comparison_rot/"
    else: path_comparison = "plots/general_predictions_comparison/"



    filenames = ["bdyn_inner_vs_flux", 
                 "bdyn_outer_vs_flux", 
                 "bsurf_inner_vs_flux", 
                 "bsurf_outer_vs_flux"]
    variables = [brms_inner_est,
                 brms_outer_est,
                 brms_inner_surf_est,
                 brms_outer_surf_est]
    variables_err = [brms_inner_est_err,
                    brms_outer_est_err,
                    brms_inner_surf_est_err,
                    brms_outer_surf_est_err]

    for index_variable in range(0,len(variables)):

        plt.clf()
        ax = plt.axes()
        cs = ax.scatter(flux_obs,
                        variables[index_variable],
                        c=mass_obs,
                        s=100*radius_obs**2,
                        cmap='viridis',
                        alpha=.5)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r"$F\mathrm{_{\rm irr} \, (erg \, cm^{-2} \, s^{-1})}$", fontsize=int(fontsize))
        ax.set_ylabel(r"$B_{\rm rms, dyn} \mathrm{(G)}$", fontsize=int(fontsize))
        cbar = fig.colorbar(cs, aspect=20, fraction=0.06, shrink=0.8)
        cbar.set_label(r"$M \, (\mathrm{M_J})$",fontsize=int(fontsize),labelpad=5)
        plt.savefig(path_general + filenames[index_variable] + filename_ending)

        plt.clf()
        ax = plt.axes()
        ax.errorbar(flux_obs,
                    variables[index_variable],
                    xerr=flux_obs_err,
                    yerr=variables_err[index_variable],
                    zorder=1,alpha=.5,
                    ls="none",ecolor="k")
        cs = ax.scatter(flux_obs,
                        variables[index_variable],
                        c=mass_obs,
                        s=100*radius_obs**2,
                        cmap='viridis',
                        alpha=.5)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r"$F\mathrm{_{\rm irr} \, (erg \, cm^{-2} \, s^{-1})}$", fontsize=int(fontsize))
        ax.set_ylabel(r"$B_{\rm rms, dyn} \mathrm{(G)}$", fontsize=int(fontsize))
        cbar = fig.colorbar(cs, aspect=20, fraction=0.06, shrink=0.8)
        cbar.set_label(r"$M \, (\mathrm{M_J})$",fontsize=int(fontsize),labelpad=5)
        plt.savefig(path_general + filenames[index_variable] + "_err" + filename_ending)
    


    filenames = ["radius_vs_flux", 
                 "radius_vs_mass_inner", 
                 "radius_vs_mass_outer"]
    variables = [flux_obs,
                 mass_obs,
                 mass_obs]
    variables_err = [flux_obs_err,
                    mass_obs_err,
                    mass_obs_err]
    colors = [mass_obs,
                    brms_inner_est,
                    brms_outer_est]
    x_labels = [r"$F\mathrm{_{\rm irr} \, (erg \, cm^{-2} \, s^{-1})}$",
                  r"$M \, (\mathrm{M_J})$",
                  r"$M \, (\mathrm{M_J})$"]
    cbar_labels = [r"$M \, (\mathrm{M_J})$",
                    r"$B_{\rm rms, inner} \, \mathrm{(G)}$",
                    r"$B_{\rm rms, outer} \, \mathrm{(G)}$"]

    for index_variable in range(0,len(variables)):
        y_lims = [0.9,2.25]
        plt.clf()
        ax = plt.axes()
        cs = ax.scatter(variables[index_variable],
                        radius_obs,
                        c=colors[index_variable],
                        s=100*radius_obs**2,
                        cmap='viridis',
                        alpha=.7)
        ax.set_xscale('log')
        ax.set_ylim(y_lims)
        ax.set_xlabel(x_labels[index_variable], fontsize=int(fontsize))
        ax.set_ylabel(r"$R_P \, \mathrm{(R_J)}$", fontsize=int(fontsize))
        cbar = fig.colorbar(cs, aspect=20, fraction=0.06, shrink=0.8)
        cbar.set_label(cbar_labels[index_variable],fontsize=int(fontsize),labelpad=5)
        plt.savefig(path_general + filenames[index_variable] + filename_ending)

        plt.clf()
        ax = plt.axes()
        ax.errorbar(variables[index_variable],
                    radius_obs,
                    xerr=variables_err[index_variable],
                    yerr=radius_obs_err,
                    zorder=1,alpha=.5,
                    ls="none",ecolor="k")
        cs = ax.scatter(variables[index_variable],
                        radius_obs,
                        c=colors[index_variable],
                        s=100*radius_obs**2,
                        cmap='viridis',
                        alpha=.7)
        ax.set_xscale('log')
        ax.set_ylim(y_lims)
        ax.set_xlabel(x_labels[index_variable], fontsize=int(fontsize))
        ax.set_ylabel(r"$R_P \, \mathrm{(R_J)}$", fontsize=int(fontsize))
        cbar = fig.colorbar(cs, aspect=20, fraction=0.06, shrink=0.8)
        cbar.set_label(cbar_labels[index_variable],fontsize=int(fontsize),labelpad=5)
        plt.savefig(path_general + filenames[index_variable] + "_err" + filename_ending)


    filenames = ["bdyn_inner_vs_radius", 
                 "bdyn_inner_surf_vs_radius", 
                 "bdyn_outer_vs_radius", 
                 "bdyn_outer_surf_vs_radius"]
    variables = [brms_inner_est,
                 brms_inner_surf_est,
                 brms_outer_est,
                 brms_outer_surf_est]
    variables_err = [brms_inner_est_err,
                    brms_inner_surf_est_err,
                    brms_outer_est_err,
                    brms_outer_surf_est_err]
    y_labels = [r"$B_{\rm rms, \, dyn, \, inner} \, \mathrm{(G)}$",
                r"$B_{\rm dip, \, surf, \, inner} \mathrm{(G)}$",
                r"$B_{\rm rms, \, dyn, \, outer} \, \mathrm{(G)}$",
                r"$B_{\rm dip, \, surf, \, outer} \mathrm{(G)}$"]

    for index_variable in range(0,len(variables)):
        x_lims = [0.9,2.25]
        plt.clf()
        ax = plt.axes()
        cs = ax.scatter(radius_obs,
                        variables[index_variable],
                        c=mass_obs,
                        s=20*np.sqrt(flux_obs/1e7),
                        cmap='viridis',
                        alpha=.7)
        #ax.set_yscale('log')
        ax.set_xlim(x_lims)
        ax.set_xlabel(r"$R_P \, \mathrm{(R_J)}$", fontsize=int(fontsize))
        ax.set_ylabel(y_labels[index_variable], fontsize=int(fontsize))
        cbar = fig.colorbar(cs, aspect=20, fraction=0.06, shrink=0.8)
        cbar.set_label(r"$M \, (\mathrm{M_J})$",fontsize=int(fontsize),labelpad=5)
        plt.savefig(path_general + filenames[index_variable] + filename_ending)

        plt.clf()
        ax = plt.axes()
        ax.errorbar(radius_obs,
                    variables[index_variable],
                    xerr=radius_obs_err,
                    yerr=variables_err[index_variable],
                    zorder=1,alpha=.5,
                    ls="none",ecolor="k")
        cs = ax.scatter(radius_obs,
                        variables[index_variable],
                        c=mass_obs,
                        s=20*np.sqrt(flux_obs/1e7),
                        cmap='viridis',
                        alpha=.7)
        #ax.set_yscale('log')
        ax.set_xlim(x_lims)
        ax.set_xlabel(r"$R_P \, \mathrm{(R_J)}$", fontsize=int(fontsize))
        ax.set_ylabel(y_labels[index_variable], fontsize=int(fontsize))
        cbar = fig.colorbar(cs, aspect=20, fraction=0.06, shrink=0.8)
        cbar.set_label(r"$M \, (\mathrm{M_J})$",fontsize=int(fontsize),labelpad=5)
        plt.savefig(path_general + filenames[index_variable] + "_err" + filename_ending)




    plt.clf()
    ax = plt.axes()
    cs = ax.scatter(flux_obs,
                    100*heat_eff_inner_est,
                    c=mass_obs,
                    s=100*radius_obs**2,
                    cmap='viridis',
                    alpha=.7)
    ax.set_xscale('log')
    #ax.set_yscale('log')
    ax.set_ylim(-0.05,4)
    ax.set_xlabel(r"$F\mathrm{_{\rm irr} \, (erg \, cm^{-2} \, s^{-1})}$", fontsize=int(fontsize))
    ax.set_ylabel(r"$\mathrm{\epsilon (\%)}$", fontsize=int(fontsize))
    cbar = fig.colorbar(cs, aspect=20, fraction=0.06, shrink=0.8)
    cbar.set_label(r"$M \, (\mathrm{M_J})$",fontsize=int(fontsize),labelpad=5)
    plt.savefig(path_general + "/heat_eff_vs_flux" + filename_ending)

    plt.clf()
    ax = plt.axes()
    ax.errorbar(flux_obs,
                100*heat_eff_inner_est,
                xerr=flux_obs_err,
                yerr=100*heat_eff_inner_est_err, # [:, valid_mask]
                zorder=1,alpha=.5,
                ls="none",ecolor="k")
    cs = ax.scatter(flux_obs,
                    100*heat_eff_inner_est,
                    c=mass_obs,
                    s=100*radius_obs**2,
                    cmap='viridis',
                    alpha=.7)
    ax.set_xscale('log')
    #ax.set_yscale('log')
    ax.set_ylim(-0.05,4)
    ax.set_xlabel(r"$F\mathrm{_{\rm irr} \, (erg \, cm^{-2} \, s^{-1})}$", fontsize=int(fontsize))
    ax.set_ylabel(r"$\mathrm{\epsilon (\%)}$", fontsize=int(fontsize))
    cbar = fig.colorbar(cs, aspect=20, fraction=0.06, shrink=0.8)
    cbar.set_label(r"$M \, (\mathrm{M_J})$",fontsize=int(fontsize),labelpad=5)
    plt.savefig(path_general + "/heat_eff_vs_flux_err" + filename_ending)

    plt.clf()
    ax = plt.axes()
    cs = ax.scatter(mass_obs,
                    100*heat_eff_inner_est,
                    c=np.log10(flux_obs),
                    s=100*radius_obs**2,
                    cmap='viridis',
                    alpha=.7)
    ax.set_xscale('log')
    #ax.set_yscale('log')
    ax.set_ylim(-0.05,4)
    ax.set_xlabel(r"$M \, (\mathrm{M_J})$", fontsize=int(fontsize))
    ax.set_ylabel(r"$\mathrm{\epsilon (\%)}$", fontsize=int(fontsize))
    cbar = fig.colorbar(cs, aspect=20, fraction=0.06, shrink=0.8)
    cbar.set_label(r"$\log_{10}(F{\rm irr}\mathrm{(\mathrm{erg} \, \mathrm{cm}^{-2} \, \mathrm{s}^{-1}))}$",fontsize=int(fontsize),labelpad=5)
    plt.savefig(path_general + "/heat_eff_vs_mass" + filename_ending)

    plt.clf()
    ax = plt.axes()
    ax.errorbar(mass_obs,
                100*heat_eff_inner_est,
                xerr=mass_obs_err,
                yerr=100*heat_eff_inner_est_err, # [:, valid_mask]
                zorder=1,alpha=.5,
                ls="none",ecolor="k")
    cs = ax.scatter(mass_obs,
                    100*heat_eff_inner_est,
                    c=np.log10(flux_obs),
                    s=100*radius_obs**2,
                    cmap='viridis',
                    alpha=.7)
    ax.set_xscale('log')
    #ax.set_yscale('log')
    ax.set_ylim(-0.05,4)
    ax.set_xlabel(r"$M \, (\mathrm{M_J})$", fontsize=int(fontsize))
    ax.set_ylabel(r"$\mathrm{\epsilon (\%)}$", fontsize=int(fontsize))
    cbar = fig.colorbar(cs, aspect=20, fraction=0.06, shrink=0.8)
    cbar.set_label(r"$\log_{10}(F{\rm irr}\mathrm{_(1 \, \mathrm{erg} \, \mathrm{cm}^{-2} \, \mathrm{s}^{-1})}$",fontsize=int(fontsize),labelpad=5)
    plt.savefig(path_general + "/heat_eff_vs_mass_err" + filename_ending)




    surf_grav = 6.67e-11*1.8981e27/6.9911e7**2  *mass_obs/radius_obs**2 # N (m/kg)^2
    surf_grav_err = 6.67e-11*1.8981e27/6.9911e7**2 /radius_obs**2 * np.sqrt(mass_obs_err**2+(4*mass_obs*radius_obs_err/radius_obs)**2)


    conv_lumin_inner_reiners = np.pi*(radius_obs*6.991e+9)**2*flux_obs*heat_eff_inner_est/4.606e24
    conv_lumin_inner_reiners_err = conv_lumin_inner_reiners*np.sqrt((heat_eff_inner_est_err/heat_eff_inner_est)**2+\
                                                                          (2*radius_obs_err/radius_obs)**2+\
                                                                          (flux_obs_err/flux_obs)**2)
    brms_inner_reiners = 23*(mass_obs)**(1/6)*(radius_obs)**(-7/6)*conv_lumin_inner_reiners**(1/3)
    brms_inner_reiners_err = brms_inner_reiners*np.sqrt((mass_obs_err/mass_obs/6)**2+\
                                                              (7*radius_obs_err/radius_obs/6)**2+\
                                                              (conv_lumin_inner_reiners_err/conv_lumin_inner_reiners/3)**2)
    
    conv_lumin_outer_reiners = np.pi*(radius_obs*6.991e+9)**2*flux_obs*heat_eff_outer_est/4.606e24
    conv_lumin_outer_reiners_err = conv_lumin_outer_reiners*np.sqrt((heat_eff_outer_est_err/heat_eff_outer_est)**2+\
                                                                          (2*radius_obs_err/radius_obs)**2+\
                                                                          (flux_obs_err/flux_obs)**2)
    brms_outer_reiners = 23*(mass_obs)**(1/6)*(radius_obs)**(-7/6)*conv_lumin_outer_reiners**(1/3)
    brms_outer_reiners_err = brms_outer_reiners*np.sqrt((mass_obs_err/mass_obs/6)**2+\
                                                        (7*radius_obs_err/radius_obs/6)**2+\
                                                        (conv_lumin_outer_reiners_err/conv_lumin_outer_reiners/3)**2)
    

    thorn18_eff = np.zeros(len(flux_obs))
    thorn18_eff_err = np.zeros(len(flux_obs))
    for index_planet in range(0,len(flux_obs)):
        thorn18_eff[index_planet] = table_predictions_save.get_thorn18_eff(flux_obs[index_planet])
        thorn18_eff_err_nonsym = table_predictions_save.get_thorn18_err(flux_obs[index_planet], 10000)
        thorn18_eff_err[index_planet] = np.mean(thorn18_eff_err_nonsym)
    
    conv_lumin_thorn18 = np.pi*(radius_obs*6.991e+9)**2*flux_obs*thorn18_eff/4.606e24
    conv_lumin_thorn18_err = conv_lumin_thorn18*np.sqrt((thorn18_eff_err/thorn18_eff)**2+\
                                                                          (2*radius_obs_err/radius_obs)**2+\
                                                                          (flux_obs_err/flux_obs)**2)
    brms_thorn18 = 23*(mass_obs)**(1/6)*(radius_obs)**(-7/6)*conv_lumin_thorn18**(1/3)
    brms_thorn18_err = brms_thorn18*np.sqrt((mass_obs_err/mass_obs/6)**2+\
                                                        (7*radius_obs_err/radius_obs/6)**2+\
                                                        (conv_lumin_thorn18_err/conv_lumin_thorn18/3)**2)
    
    K_reiners = 1

    variables_inner = [flux_obs,
                       orbital_period_obs,
                        temp_eff_obs,
                        radius_obs,
                        mass_obs,
                        surf_grav,
                        flux_obs*heat_eff_inner_est,
                        K_reiners*brms_inner_reiners,
                        K_reiners*brms_thorn18]
    variables_outer = [flux_obs,
                       orbital_period_obs,
                        temp_eff_obs,
                        radius_obs,
                        mass_obs,
                        surf_grav,
                        flux_obs*heat_eff_outer_est,
                        K_reiners*brms_outer_reiners,
                        K_reiners*brms_thorn18]
    variables_inner_err  = [flux_obs_err,
                       orbital_period_obs_err,
                        temp_eff_obs,
                        radius_obs_err,
                        mass_obs_err,
                        surf_grav_err,
                        np.sqrt((flux_obs_err*heat_eff_inner_est)**2+(flux_obs*heat_eff_inner_est_err)**2),
                        K_reiners*brms_inner_reiners_err,
                        K_reiners*brms_thorn18_err]
    variables_outer_err  = [flux_obs_err,
                       orbital_period_obs_err,
                        temp_eff_obs,
                        radius_obs_err,
                        mass_obs_err,
                        surf_grav_err,
                        np.sqrt((flux_obs_err*heat_eff_outer_est)**2+(flux_obs*heat_eff_outer_est_err)**2),
                        K_reiners*brms_outer_reiners_err,
                        K_reiners*brms_thorn18_err]
    
    print_irr_models = [True,
                        True,
                        True,
                        False,
                        True,
                        False,
                        False,
                        False,
                        False]
    print_power_laws = [False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        False,
                        True,
                        True]
    variable_names = ["fluxes",
                      "period",
                      "temps",
                      "radius",
                      "mass",
                      "surf_grav",
                      "conv_lum",
                      "reiners09_law",
                      "thorn18"]
    variable_labels = [r"$F\mathrm{_{\rm irr} \, (erg \, cm^{-2} \, s^{-1})}$",
                       r"$P \, \mathrm{(days)}$",
                       r"$T_{\rm eq} \, \mathrm{(K)}$",
                       r"$R_P \, \mathrm{(R_J)}$",
                       r"$M_P \, \mathrm{(M_J)}$",
                       r"$g_{\rm surf} \, \mathrm{(m \, s^{-2})}$",
                       r"$\epsilon F_{\rm irr}$",
                       r"$\mathrm{23} \, M_P^{1/6} \, R_P^{-7/6} \, (\epsilon_{\rm inner} \pi R_P^2 F_{\rm irr})^{1/3}$",
                       r"$\mathrm{23} \, M_P^{1/6} \, R_P^{-7/6} \, (\epsilon_{\rm Th18} \pi R_P^2 F_{\rm irr})^{1/3}$"]

    for index_variable in range(0,len(variables_inner)):
        
        markersize = 100*radius_obs**2
        if variable_names[index_variable] == "radius": markersize = 20*np.sqrt(flux_obs/1e7)
        plt.clf()
        ax = plt.axes()
        for index_planet in range(0,len(radius_obs)):
            x = np.array([variables_inner[index_variable][index_planet],
                        variables_outer[index_variable][index_planet]])
            y = np.array([brms_inner_est[index_planet],
                        brms_outer_est[index_planet]])
            ax.plot(x,
                    y,
                    alpha=.5,
                    color = "k",
                    ls = ":")
            
        ax.scatter(variables_inner[index_variable],
                    brms_inner_est,
                    s=markersize,
                    alpha=.5,
                    color = color_inner,
                    label = label_inner)
        if print_irr_models[index_variable]:
            ax.scatter(variables_inner[index_variable],
                        brms_irr_est,
                        s=50,
                        alpha=.5,
                        color = color_irr,
                        label = label_irr)
        ax.scatter(variables_outer[index_variable],
                    brms_outer_est,
                    s=markersize,
                    alpha=.5,
                    color = color_outer,
                    label = label_outer)
        if variable_names[index_variable] != "radius": ax.set_xscale('log')
        ax.set_yscale('log')
        if print_power_laws[index_variable]:
            xmin, xmax = plt.xlim()
            ymin, ymax = plt.ylim()
            low = max(xmin, ymin)
            high = min(xmax, ymax)
            x = np.logspace(np.log10(1.1*low), np.log10(0.9*high), 20)
            if variable_names[index_variable] == "reiners09_law": y=x
            else: y=x
            plt.plot(x, y, '--', color = "k", linewidth=2, label=r"$x=y$")
        ax.set_xlabel(variable_labels[index_variable], fontsize=int(fontsize))
        ax.set_ylabel(r"$B_{\rm rms, dyn} \mathrm{(G)}$", fontsize=int(fontsize))
        plt.legend(fontsize=int(fontsize))
        plt.savefig(path_comparison + "/bdyn_vs_" + variable_names[index_variable] + filename_ending)


        plt.clf()
        ax = plt.axes()            

        ax.errorbar(variables_inner[index_variable],
                    brms_inner_est,
                    xerr=variables_inner_err[index_variable],
                    yerr=brms_inner_est_err,
                    zorder=1,alpha=.5,
                    ls="none",ecolor=color_inner_err)
        ax.scatter(variables_inner[index_variable],
                    brms_inner_est,
                    s=markersize,
                    alpha=.5,
                    color = color_inner,
                    label = label_inner)
        ax.errorbar(variables_outer[index_variable],
                    brms_outer_est,
                    xerr=variables_outer_err[index_variable],
                    yerr=brms_outer_est_err,
                    zorder=1,alpha=.5,
                    ls="none",ecolor=color_outer_err)
        ax.scatter(variables_outer[index_variable],
                    brms_outer_est,
                    s=markersize,
                    alpha=.5,
                    color = color_outer,
                    label = label_outer)
        if variable_names[index_variable] != "radius": ax.set_xscale('log')
        ax.set_yscale('log')
        if print_power_laws[index_variable]:
            xmin, xmax = plt.xlim()
            ymin, ymax = plt.ylim()
            low = max(xmin, ymin)
            high = min(xmax, ymax)
            x = np.logspace(np.log10(1.1*low), np.log10(0.9*high), 20)
            if variable_names[index_variable] == "reiners09_law": y=x
            else: y=x
            plt.plot(x, y, '--', color = "k", linewidth=2, label=r"$x=y$")
        ax.set_xlabel(variable_labels[index_variable], fontsize=int(fontsize))
        ax.set_ylabel(r"$B_{\rm rms, dyn} \mathrm{(G)}$", fontsize=int(fontsize))
        plt.legend(fontsize=int(fontsize))
        plt.savefig(path_comparison + "/bdyn_vs_" + variable_names[index_variable] + "_err" + filename_ending)

        plt.clf()
        ax = plt.axes()
        for index_planet in range(0,len(radius_obs)):
            x = np.array([variables_inner[index_variable][index_planet],
                        variables_outer[index_variable][index_planet]])
            y = np.array([brms_inner_surf_est[index_planet],
                        brms_outer_surf_est[index_planet]])
            ax.plot(x,
                    y,
                    alpha=.5,
                    color = "k",
                    ls = ":")
        ax.scatter(variables_inner[index_variable],
                    brms_inner_surf_est,
                    s=markersize,
                    alpha=.5,
                    color = color_inner,
                    label = label_inner)
        if print_irr_models[index_variable]:
            ax.scatter(variables_inner[index_variable],
                        brms_irr_surf_est,
                        s=50,
                        alpha=.5,
                        color = color_irr,
                        label = label_irr)
        ax.scatter(variables_outer[index_variable],
                    brms_outer_surf_est,
                    s=markersize,
                    alpha=.5,
                    color = color_outer,
                    label = label_outer)
        if variable_names[index_variable] != "radius": ax.set_xscale('log')
        ax.set_yscale('log')
        if print_power_laws[index_variable]:
            xmin, xmax = plt.xlim()
            ymin, ymax = plt.ylim()
            low = max(xmin, ymin)
            high = min(xmax, ymax)
            x = np.logspace(np.log10(1.1*low), np.log10(0.9*high), 20)
            if variable_names[index_variable] == "reiners09_law": y=x
            else: y=x
            plt.plot(x, y, '--', color = "k", linewidth=2, label=r"$x=y$")
        ax.set_xlabel(variable_labels[index_variable], fontsize=int(fontsize))
        ax.set_ylabel(r"$B_{\rm dip, surf} \mathrm{(G)}$", fontsize=int(fontsize))
        #plt.legend(fontsize=int(fontsize))
        plt.savefig(path_comparison + "/bsurf_vs_" + variable_names[index_variable] + filename_ending)

        plt.clf()
        ax = plt.axes()
        ax.errorbar(variables_inner[index_variable],
                    brms_inner_surf_est,
                    xerr=variables_inner_err[index_variable],
                    yerr=brms_inner_surf_est_err,
                    zorder=1,alpha=.5,
                    ls="none",ecolor=color_inner_err)
        ax.scatter(variables_inner[index_variable],
                    brms_inner_surf_est,
                    s=markersize,
                    alpha=.5,
                    color = color_inner,
                    label = label_inner)
        ax.errorbar(variables_outer[index_variable],
                    brms_outer_surf_est,
                    xerr=variables_outer_err[index_variable],
                    yerr=brms_outer_surf_est_err,
                    zorder=1,alpha=.5,
                    ls="none",ecolor=color_outer_err)
        ax.scatter(variables_outer[index_variable],
                    brms_outer_surf_est,
                    s=markersize,
                    alpha=.5,
                    color = color_outer,
                    label = label_outer)
        if variable_names[index_variable] != "radius": ax.set_xscale('log')
        ax.set_yscale('log')
        if print_power_laws[index_variable]:
            xmin, xmax = plt.xlim()
            ymin, ymax = plt.ylim()
            low = max(xmin, ymin)
            high = min(xmax, ymax)
            x = np.logspace(np.log10(1.1*low), np.log10(0.9*high), 20)
            if variable_names[index_variable] == "reiners09_law": y=x
            else: y=x
            plt.plot(x, y, '--', color = "k", linewidth=2, label=r"$x=y$")
        ax.set_xlabel(variable_labels[index_variable], fontsize=int(fontsize))
        ax.set_ylabel(r"$B_{\rm dip, surf} \mathrm{(G)}$", fontsize=int(fontsize))
        #plt.legend(fontsize=int(fontsize))
        plt.savefig(path_comparison + "/bsurf_vs_" + variable_names[index_variable] + "_err" + filename_ending)


def plot_population_fits(planets,
                        name_cut,
                        valid_mask,
                        fit_type, fit_reiners09,
                        fit_MRL_inner, fit_MRL_inner_surf, fit_MRL_outer, fit_MRL_outer_surf,
                        fit_MRF_inner, fit_MRF_inner_surf, fit_MRF_outer, fit_MRF_outer_surf,
                        fit_MR_inner, fit_MR_inner_surf, fit_MR_outer, fit_MR_outer_surf,
                        fit_MR_irr_inner, fit_MR_irr_inner_surf, fit_MR_irr_outer, fit_MR_irr_outer_surf,
                        fit_MR_irr_all, fit_MR_irr_all_surf,
                        rotation=False,
                        color_inner = "blue", color_inner_cut = "deepskyblue",
                        color_outer = "darkgreen", color_outer_cut = "limegreen", 
                        color_irr = "k"):
    
    heat_eff_inner_est  = planets.heat_eff_inner_est
    brms_inner_est      = planets.brms_inner_est
    brms_inner_surf_est = planets.brms_inner_surf_est
    heat_eff_outer_est  = planets.heat_eff_outer_est
    brms_outer_est      = planets.brms_outer_est
    brms_outer_surf_est = planets.brms_outer_surf_est
    brms_irr_est        = planets.brms_irr_est
    brms_irr_surf_est   = planets.brms_irr_surf_est
    mass_obs            = planets.mass_obs
    flux_obs            = planets.flux_obs
    radius_obs          = planets.radius_obs
    
    label_inner = r"$\epsilon_{\rm inner}$" #r"$B_{\rm dip, \, inner}$"
    label_inner_cut = r"$\epsilon_{\rm inner}, \, \mathrm{cut}$" #r"$B_{\rm dip, \, inner}$"
    label_outer = r"$\epsilon_{\rm outer}$" #r"$B_{\rm dip, \, outer}$"
    label_outer_cut = r"$\epsilon_{\rm outer}, \, \mathrm{cut}$" #r"$B_{\rm dip, \, outer}$"
    label_irr = r"$\epsilon = 0$" #r"$B_{\rm irr}$"
    
    fontsize = 32
    plt.rcParams['text.usetex'] = True
    fig = plt.figure(figsize=(11, 8), constrained_layout=True)
    plt.rcParams["figure.figsize"] = (11, 8)
    plt.rcParams['font.size'] = fontsize
    plt.rcParams['lines.linewidth'] = 1
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams["mathtext.fontset"] = "cm"

    conv_lumin_inner = np.pi*(radius_obs*6.991e+9)**2*flux_obs*heat_eff_inner_est/4.606e24
    conv_lumin_outer = np.pi*(radius_obs*6.991e+9)**2*flux_obs*heat_eff_outer_est/4.606e24

    fitMRL_brms_inner_inner = fit_MRL_inner.K   *mass_obs**fit_MRL_inner.alpha\
                                        *radius_obs**fit_MRL_inner.beta\
                                        *conv_lumin_inner**fit_MRL_inner.gamma
    fitMRL_brms_inner_outer = fit_MRL_inner.K  *mass_obs**fit_MRL_inner.alpha\
                                    *radius_obs**fit_MRL_inner.beta\
                                    *conv_lumin_outer**fit_MRL_inner.gamma
    
    fitMRL_brms_inner_inner_surf = fit_MRL_inner_surf.K *mass_obs**fit_MRL_inner_surf.alpha\
                                                *radius_obs**fit_MRL_inner_surf.beta\
                                                *conv_lumin_inner**fit_MRL_inner_surf.gamma
    fitMRL_brms_inner_outer_surf = fit_MRL_inner_surf.K*mass_obs**fit_MRL_inner_surf.alpha\
                                            *radius_obs**fit_MRL_inner_surf.beta\
                                            *conv_lumin_outer**fit_MRL_inner_surf.gamma
    
    fitMRL_brms_outer_inner = fit_MRL_outer.K*mass_obs**fit_MRL_outer.alpha\
                                        *radius_obs**fit_MRL_outer.beta\
                                        *conv_lumin_inner**fit_MRL_outer.gamma
    fitMRL_brms_outer_outer = fit_MRL_outer.K   *mass_obs**fit_MRL_outer.alpha\
                                        *radius_obs**fit_MRL_outer.beta\
                                        *conv_lumin_outer**fit_MRL_outer.gamma
    
    fitMRL_brms_outer_inner_surf = fit_MRL_outer_surf.K  *mass_obs**fit_MRL_outer_surf.alpha\
                                                    *radius_obs**fit_MRL_outer_surf.beta\
                                                    *conv_lumin_inner**fit_MRL_outer_surf.gamma
    fitMRL_brms_outer_outer_surf = fit_MRL_outer_surf.K *mass_obs**fit_MRL_outer_surf.alpha\
                                                *radius_obs**fit_MRL_outer_surf.beta\
                                                *conv_lumin_outer**fit_MRL_outer_surf.gamma
    
    fitMRF_brms_inner = fit_MRF_inner.K*mass_obs**fit_MRF_inner.alpha\
                                            *radius_obs**fit_MRF_inner.beta\
                                            *(flux_obs/1e9)**fit_MRF_inner.gamma
    fitMRF_brms_inner_surf = fit_MRF_inner_surf.K  *mass_obs**fit_MRF_inner_surf.alpha\
                                                        *radius_obs**fit_MRF_inner_surf.beta\
                                                        *(flux_obs/1e9)**fit_MRF_inner_surf.gamma
    
    fitMRF_brms_outer = fit_MRF_outer.K  *mass_obs**fit_MRF_outer.alpha\
                                        *radius_obs**fit_MRF_outer.beta\
                                        *(flux_obs/1e9)**fit_MRF_outer.gamma
    fitMRF_brms_outer_surf = fit_MRF_outer_surf.K    *mass_obs**fit_MRF_outer_surf.alpha\
                                                    *radius_obs**fit_MRF_outer_surf.beta\
                                                    *(flux_obs/1e9)**fit_MRF_outer_surf.gamma

    fitMR_brms_inner = fit_MR_inner.K   *mass_obs**fit_MR_inner.alpha\
                                                *radius_obs**fit_MR_inner.beta
    fitMR_brms_inner_surf = fit_MR_inner_surf.K *mass_obs**fit_MR_inner_surf.alpha\
                                                        *radius_obs**fit_MR_inner_surf.beta
    fitMR_brms_outer = fit_MR_outer.K *mass_obs**fit_MR_outer.alpha\
                                        *radius_obs**fit_MR_outer.beta
    fitMR_brms_outer_surf = fit_MR_outer_surf.K   *mass_obs**fit_MR_outer_surf.alpha\
                                                    *radius_obs**fit_MR_outer_surf.beta
    
    fitMR_brms_irr = fit_MR_irr_inner.K *mass_obs**fit_MR_irr_inner.alpha\
                                    *radius_obs**fit_MR_irr_inner.beta
    fitMR_brms_irr_surf = fit_MR_irr_inner_surf.K   *mass_obs**fit_MR_irr_inner_surf.alpha\
                                                *radius_obs**fit_MR_irr_inner_surf.beta
    fitMR_brms_irr_outer = fit_MR_irr_outer.K  *mass_obs**fit_MR_irr_outer.alpha\
                                        *radius_obs**fit_MR_irr_outer.beta
    fitMR_brms_irr_outer_surf = fit_MR_irr_outer_surf.K*mass_obs**fit_MR_irr_outer_surf.alpha\
                                                *radius_obs**fit_MR_irr_outer_surf.beta
    fitMR_brms_irr_all = fit_MR_irr_all.K  *mass_obs**fit_MR_irr_all.alpha\
                                        *radius_obs**fit_MR_irr_all.beta
    fitMR_brms_irr_all_surf = fit_MR_irr_all_surf.K*mass_obs**fit_MR_irr_all_surf.alpha\
                                                *radius_obs**fit_MR_irr_all_surf.beta

    variables_inner = [fitMRL_brms_inner_inner,
                        fitMRL_brms_inner_inner_surf,
                        fitMRL_brms_outer_inner,
                        fitMRL_brms_outer_inner_surf,
                        fitMRF_brms_inner,
                        fitMRF_brms_inner_surf,
                        fitMRF_brms_outer,
                        fitMRF_brms_outer_surf,
                        fitMR_brms_inner,
                        fitMR_brms_inner_surf,
                        fitMR_brms_outer,
                        fitMR_brms_outer_surf,
                        fitMR_brms_irr,
                        fitMR_brms_irr_surf,
                        fitMR_brms_irr_outer,
                        fitMR_brms_irr_outer_surf,
                        fitMR_brms_irr_all,
                        fitMR_brms_irr_all_surf]
    variables_outer = [fitMRL_brms_inner_outer,
                        fitMRL_brms_inner_outer_surf,
                        fitMRL_brms_outer_outer,
                        fitMRL_brms_outer_outer_surf,
                        fitMRF_brms_inner,
                        fitMRF_brms_inner_surf,
                        fitMRF_brms_outer,
                        fitMRF_brms_outer_surf,
                        fitMR_brms_inner,
                        fitMR_brms_inner_surf,
                        fitMR_brms_outer,
                        fitMR_brms_outer_surf,
                        fitMR_brms_irr,
                        fitMR_brms_irr_surf,
                        fitMR_brms_irr_outer,
                        fitMR_brms_irr_outer_surf,
                        fitMR_brms_irr_all,
                        fitMR_brms_irr_all_surf]
    print_irr_models = [True,True,True,True,
                        True,True,True,True,
                        True,True,True,True,
                        True,True,True,True,
                        True,True]
    print_power_laws = [True,True,True,True,
                        True,True,True,True,
                        True,True,True,True,
                        True,True,True,True,
                        True,True]
    valid_masks = [valid_mask,valid_mask,valid_mask,valid_mask,
                   valid_mask,valid_mask,valid_mask,valid_mask,
                   valid_mask,valid_mask,valid_mask,valid_mask,
                   ~valid_mask,~valid_mask,~valid_mask,~valid_mask,
                   valid_mask,valid_mask]
    variable_names = ["fit_general_inner" + name_cut,
                      "fit_general_inner_surf" + name_cut,
                      "fit_general_outer" + name_cut,
                      "fit_general_outer_surf" + name_cut,
                      "fit_no_eps_inner" + name_cut,
                      "fit_no_eps_inner_surf" + name_cut,
                      "fit_no_eps_outer" + name_cut,
                      "fit_no_eps_outer_surf" + name_cut,
                      "fit_no_flux_inner" + name_cut,
                      "fit_no_flux_inner_surf" + name_cut,
                      "fit_no_flux_outer" + name_cut,
                      "fit_no_flux_outer_surf" + name_cut,
                      "fit_irr_inner" + name_cut,
                      "fit_irr_inner_surf" + name_cut,
                      "fit_irr_outer" + name_cut,
                      "fit_irr_outer_surf" + name_cut,
                      "fit_irr_all" + name_cut,
                      "fit_irr_all_surf" + name_cut]
    
    variable_labels = [r"${{{:.1f}}}$".format(fit_MRL_inner.K) + r"$ \, \mathrm{G}$" + \
                         r"$\, M_P^{{{:.2f}}}$".format(fit_MRL_inner.alpha) + \
                         r"$\, R_P^{{{:.2f}}}$".format(fit_MRL_inner.beta) +  \
                         r"$ \, (\epsilon \pi R_P^2 F_{\rm irr})$" + r"$\, ^{{{:.3f}}}$".format(fit_MRL_inner.gamma),
                       r"${{{:.1f}}}$".format(fit_MRL_inner_surf.K) + r"$ \, \mathrm{G}$" + \
                        r"$\, M_P^{{{:.2f}}}$".format(fit_MRL_inner_surf.alpha) + \
                         r"$\, R_P^{{{:.2f}}}$".format(fit_MRL_inner_surf.beta) + \
                         r"$ \, (\epsilon \pi R_P^2 F_{\rm irr})$" + r"$\, ^{{{:.3f}}}$".format(fit_MRL_inner_surf.gamma),
                       r"${{{:.1f}}}$".format(fit_MRL_outer.K) + r"$ \, \mathrm{G}$" + \
                         r"$\, M_P^{{{:.2f}}}$".format(fit_MRL_outer.alpha) + \
                         r"$\, R_P^{{{:.2f}}}$".format(fit_MRL_outer.beta) + \
                         r"$ \, (\epsilon \pi R_P^2 F_{\rm irr})$" + r"$\, ^{{{:.3f}}}$".format(fit_MRL_outer.gamma),
                       r"${{{:.1f}}}$".format(fit_MRL_outer_surf.K) + r"$ \, \mathrm{G}$" + \
                         r"$\, M_P^{{{:.2f}}}$".format(fit_MRL_outer_surf.alpha) + \
                         r"$\, R_P^{{{:.2f}}}$".format(fit_MRL_outer_surf.beta) + \
                         r"$ \, (\epsilon \pi R_P^2 F_{\rm irr})$" + r"$\, ^{{{:.3f}}}$".format(fit_MRL_outer_surf.gamma),
                        
                        r"${{{:.1f}}}$".format(fit_MRF_inner.K) + r"$ \, \mathrm{G}$" + \
                         r"$\, M_P^{{{:.2f}}}$".format(fit_MRF_inner.alpha) + \
                         r"$\, R_P^{{{:.2f}}}$".format(fit_MRF_inner.beta) +  \
                         r"$ \, F_{\rm irr}$" + r"$\, ^{{{:.2f}}}$".format(fit_MRF_inner.gamma),
                       r"${{{:.1f}}}$".format(fit_MRF_inner_surf.K) + r"$ \, \mathrm{G}$" + \
                         r"$\, M_P^{{{:.2f}}}$".format(fit_MRF_inner_surf.alpha) + \
                         r"$\, R_P^{{{:.2f}}}$".format(fit_MRF_inner_surf.beta) + \
                         r"$ \, F_{\rm irr}$" + r"$\, ^{{{:.2f}}}$".format(fit_MRF_inner_surf.gamma),
                       r"${{{:.1f}}}$".format(fit_MRF_outer.K) + r"$ \, \mathrm{G}$" + \
                         r"$\, M_P^{{{:.2f}}}$".format(fit_MRF_outer.alpha) + \
                         r"$\, R_P^{{{:.2f}}}$".format(fit_MRF_outer.beta) + \
                         r"$ \, F_{\rm irr}$" + r"$\, ^{{{:.2f}}}$".format(fit_MRF_outer.gamma),
                       r"${{{:.1f}}}$".format(fit_MRF_outer_surf.K) + r"$ \, \mathrm{G}$" + \
                         r"$\, M_P^{{{:.2f}}}$".format(fit_MRF_outer_surf.alpha) + \
                         r"$\, R_P^{{{:.2f}}}$".format(fit_MRF_outer_surf.beta) + \
                         r"$ \, F_{\rm irr}$" + r"$\, ^{{{:.2f}}}$".format(fit_MRF_outer_surf.gamma),
                        
                        r"${{{:.1f}}}$".format(fit_MR_inner.K) + r"$ \, \mathrm{G}$" + \
                         r"$\, M_P^{{{:.2f}}}$".format(fit_MR_inner.alpha) + \
                         r"$\, R_P^{{{:.2f}}}$".format(fit_MR_inner.beta),
                       r"${{{:.1f}}}$".format(fit_MR_inner_surf.K) + r"$ \, \mathrm{G}$" + \
                         r"$\, M_P^{{{:.2f}}}$".format(fit_MR_inner_surf.alpha) + \
                         r"$\, R_P^{{{:.2f}}}$".format(fit_MR_inner_surf.beta),
                       r"${{{:.1f}}}$".format(fit_MR_outer.K) + r"$ \, \mathrm{G}$" + \
                         r"$\, M_P^{{{:.2f}}}$".format(fit_MR_outer.alpha) + \
                         r"$\, R_P^{{{:.2f}}}$".format(fit_MR_outer.beta),
                       r"${{{:.1f}}}$".format(fit_MR_outer_surf.K) + r"$ \, \mathrm{G}$" + \
                         r"$\, M_P^{{{:.2f}}}$".format(fit_MR_outer_surf.alpha) + \
                         r"$\, R_P^{{{:.2f}}}$".format(fit_MR_outer_surf.beta),

                        r"${{{:.1f}}}$".format(fit_MR_irr_inner.K) + r"$ \, \mathrm{G}$" + \
                         r"$\, M_P^{{{:.2f}}}$".format(fit_MR_irr_inner.alpha) + \
                         r"$\, R_P^{{{:.2f}}}$".format(fit_MR_irr_inner.beta),
                       r"${{{:.1f}}}$".format(fit_MR_irr_inner_surf.K) + r"$ \, \mathrm{G}$" + \
                         r"$\, M_P^{{{:.2f}}}$".format(fit_MR_irr_inner_surf.alpha) + \
                         r"$\, R_P^{{{:.2f}}}$".format(fit_MR_irr_inner_surf.beta),
                       r"${{{:.1f}}}$".format(fit_MR_irr_outer.K) + r"$ \, \mathrm{G}$" + \
                         r"$\, M_P^{{{:.2f}}}$".format(fit_MR_irr_outer.alpha) + \
                         r"$\, R_P^{{{:.2f}}}$".format(fit_MR_irr_outer.beta),
                       r"${{{:.1f}}}$".format(fit_MR_irr_outer_surf.K) + r"$ \, \mathrm{G}$" + \
                         r"$\, M_P^{{{:.2f}}}$".format(fit_MR_irr_outer_surf.alpha) + \
                         r"$\, R_P^{{{:.2f}}}$".format(fit_MR_irr_outer_surf.beta),
                       r"${{{:.1f}}}$".format(fit_MR_irr_all.K) + r"$ \, \mathrm{G}$" + \
                         r"$\, M_P^{{{:.2f}}}$".format(fit_MR_irr_all.alpha) + \
                         r"$\, R_P^{{{:.2f}}}$".format(fit_MR_irr_all.beta),
                       r"${{{:.1f}}}$".format(fit_MR_irr_all_surf.K) + r"$ \, \mathrm{G}$" + \
                         r"$\, M_P^{{{:.2f}}}$".format(fit_MR_irr_all_surf.alpha) + \
                         r"$\, R_P^{{{:.2f}}}$".format(fit_MR_irr_all_surf.beta)]

    if rotation: path = "plots/fits" + name_cut + "_rot/"
    else: path = "plots/fits" + name_cut + "/"

    for index_variable in range(0,len(variables_inner)):

        if variable_names[index_variable] != "fit_general_inner_surf" + name_cut and \
           variable_names[index_variable] != "fit_general_outer_surf" + name_cut and \
           variable_names[index_variable] != "fit_no_eps_inner_surf" + name_cut and \
           variable_names[index_variable] != "fit_no_eps_outer_surf" + name_cut and \
           variable_names[index_variable] != "fit_no_flux_inner_surf" + name_cut and \
           variable_names[index_variable] != "fit_no_flux_outer_surf" + name_cut and \
           variable_names[index_variable] != "fit_irr_inner_surf" + name_cut and \
           variable_names[index_variable] != "fit_irr_outer_surf" + name_cut and \
           variable_names[index_variable] != "fit_irr_all_surf" + name_cut:

            markersize = 100*radius_obs**2
            if variable_names[index_variable] == "radius": markersize = 20*np.sqrt(flux_obs/1e7)
            plt.clf()
            ax = plt.axes()
            for index_planet in range(0,len(radius_obs)):
                x = np.array([variables_inner[index_variable][index_planet],
                            variables_outer[index_variable][index_planet]])
                y = np.array([brms_inner_est[index_planet],
                            brms_outer_est[index_planet]])
                ax.plot(x,
                        y,
                        alpha=.5,
                        color = "k",
                        ls = ":")
            ax.scatter(variables_inner[index_variable][valid_masks[index_variable]],
                        brms_inner_est[valid_masks[index_variable]],
                        s=markersize[valid_masks[index_variable]], alpha=.5, color = color_inner,
                        label = label_inner)
            ax.scatter(variables_inner[index_variable][~valid_masks[index_variable]],
                        brms_inner_est[~valid_masks[index_variable]],
                        s=markersize[~valid_masks[index_variable]], alpha=.5, color = color_inner_cut)
            if print_irr_models[index_variable]:
                ax.scatter(variables_inner[index_variable],
                            brms_irr_est,
                            s=50, alpha=.5, color = color_irr,
                            label = label_irr)
            ax.scatter(variables_outer[index_variable][valid_masks[index_variable]],
                        brms_outer_est[valid_masks[index_variable]],
                        s=markersize[valid_masks[index_variable]], alpha=.5, color = color_outer,
                        label = label_outer)
            ax.scatter(variables_outer[index_variable][~valid_masks[index_variable]],
                        brms_outer_est[~valid_masks[index_variable]],
                        s=markersize[~valid_masks[index_variable]], alpha=.5, color = color_outer_cut)
            if variable_names[index_variable] != "radius": ax.set_xscale('log')
            ax.set_yscale('log')
            if print_power_laws[index_variable]:
                xmin, xmax = plt.xlim()
                ymin, ymax = plt.ylim()
                low = max(xmin, ymin)
                high = min(xmax, ymax)
                x = np.logspace(np.log10(1.1*low), np.log10(0.9*high), 20)
                if variable_names[index_variable] == "reiners09_law": y=x
                else: y=x
                plt.plot(x, y, '--', color = "k")
            ax.set_xlabel(variable_labels[index_variable], fontsize=int(fontsize))
            ax.set_ylabel(r"$B_{\rm rms, dyn} \mathrm{(G)}$", fontsize=int(fontsize))
            plt.legend(fontsize=int(fontsize))
            plt.savefig(path + variable_names[index_variable] + "_bdyn.png")

        if variable_names[index_variable] != "fit_general_inner" + name_cut and \
           variable_names[index_variable] != "fit_general_outer" + name_cut and \
           variable_names[index_variable] != "fit_no_eps_inner" + name_cut and \
           variable_names[index_variable] != "fit_no_eps_outer" + name_cut and \
           variable_names[index_variable] != "fit_no_flux_inner" + name_cut and \
           variable_names[index_variable] != "fit_no_flux_outer" + name_cut and \
           variable_names[index_variable] != "fit_irr_inner" + name_cut and \
           variable_names[index_variable] != "fit_irr_outer" + name_cut and  \
           variable_names[index_variable] != "fit_irr_all" + name_cut:

            plt.clf()
            ax = plt.axes()
            for index_planet in range(0,len(radius_obs)):
                x = np.array([variables_inner[index_variable][index_planet],
                            variables_outer[index_variable][index_planet]])
                y = np.array([brms_inner_surf_est[index_planet],
                            brms_outer_surf_est[index_planet]])
                ax.plot(x, y,
                        alpha=.5,
                        color = "k",
                        ls = ":")
            ax.scatter(variables_inner[index_variable][valid_masks[index_variable]],
                        brms_inner_surf_est[valid_masks[index_variable]],
                        s=markersize[valid_masks[index_variable]], alpha=.5, color = color_inner,
                        label = label_inner)
            ax.scatter(variables_inner[index_variable][~valid_masks[index_variable]],
                        brms_inner_surf_est[~valid_masks[index_variable]],
                        s=markersize[~valid_masks[index_variable]], alpha=.5, color = color_inner_cut)
            if print_irr_models[index_variable]:
                ax.scatter(variables_inner[index_variable],
                            brms_irr_surf_est,
                            s=50, alpha=.5, color = color_irr,
                            label = label_irr)
            ax.scatter(variables_outer[index_variable][valid_masks[index_variable]],
                        brms_outer_surf_est[valid_masks[index_variable]],
                        s=markersize[valid_masks[index_variable]], alpha=.5, color = color_outer,
                        label = label_outer)
            ax.scatter(variables_outer[index_variable][~valid_masks[index_variable]],
                        brms_outer_surf_est[~valid_masks[index_variable]],
                        s=markersize[~valid_masks[index_variable]], alpha=.5, color = color_outer_cut)
            if variable_names[index_variable] != "radius": ax.set_xscale('log')
            ax.set_yscale('log')
            if print_power_laws[index_variable]:
                xmin, xmax = plt.xlim()
                ymin, ymax = plt.ylim()
                low = max(xmin, ymin)
                high = min(xmax, ymax)
                x = np.logspace(np.log10(1.1*low), np.log10(0.9*high), 20)
                plt.plot(x, x, '--', color = "k")
            ax.set_xlabel(variable_labels[index_variable], fontsize=int(fontsize))
            ax.set_ylabel(r"$B_{\rm dip, surf} \mathrm{(G)}$", fontsize=int(fontsize))
            plt.legend(fontsize=int(fontsize))
            plt.savefig(path + variable_names[index_variable] + "_bsurf.png")


    ## General plot for comparing fits

    fontsize = 27
    plt.rcParams['text.usetex'] = True
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(22,15), sharey=True)
    plt.rcParams["figure.figsize"] = (22, 15)
    plt.rcParams['font.size'] = fontsize
    plt.rcParams['lines.linewidth'] = 1
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams["mathtext.fontset"] = "cm"

    axes[0,0].set_ylabel(r"$B_{\rm rms, \, dyn} \, \mathrm{(G)}$", fontsize=int(fontsize)*1.1)
    axes[1,0].set_ylabel(r"$B_{\rm rms, \, dyn} \, \mathrm{(G)}$", fontsize=int(fontsize)*1.1)

    for index_axis in range(0,3):
        if index_axis == 0:
            index_variable = 0
            label, label_cut = None, None
        elif index_axis == 1:
            index_variable = 4
            label, label_cut = None, None
        elif index_axis == 2:
            label = label_inner_cut
            label_cut = label_inner
            index_variable = 8
        axes[0,index_axis].scatter(variables_inner[index_variable][~valid_masks[index_variable]],
                            brms_inner_est[~valid_masks[index_variable]],
                            s=markersize[~valid_masks[index_variable]], alpha=.5, color = color_inner_cut,
                            label = label)
        axes[0,index_axis].scatter(variables_inner[index_variable][valid_masks[index_variable]],
                            brms_inner_est[valid_masks[index_variable]],
                            s=markersize[valid_masks[index_variable]], alpha=.5, color = color_inner,
                            label = label_cut)
        axes[0,index_axis].scatter(variables_outer[index_variable][~valid_masks[index_variable]],
                            brms_outer_est[~valid_masks[index_variable]],
                            s=markersize[~valid_masks[index_variable]], alpha=.15, color = color_outer_cut)
        axes[0,index_axis].scatter(variables_outer[index_variable][valid_masks[index_variable]],
                            brms_outer_est[valid_masks[index_variable]],
                            s=markersize[valid_masks[index_variable]], alpha=.15, color = color_outer)
        axes[0,index_axis].set_xlabel(variable_labels[index_variable], fontsize=int(fontsize))

        if index_axis == 0:
            index_variable = 2
            label, label_cut = None, None
        elif index_axis == 1:
            index_variable = 6
            label, label_cut = None, None
        elif index_axis == 2:
            label = label_inner_cut
            label_cut = label_inner
            index_variable = 10
        axes[1,index_axis].scatter(variables_inner[index_variable][~valid_masks[index_variable]],
                            brms_inner_est[~valid_masks[index_variable]],
                            s=markersize[~valid_masks[index_variable]], alpha=.15, color = color_inner_cut)
        axes[1,index_axis].scatter(variables_inner[index_variable][valid_masks[index_variable]],
                            brms_inner_est[valid_masks[index_variable]],
                            s=markersize[valid_masks[index_variable]], alpha=.15, color = color_inner)
        axes[1,index_axis].scatter(variables_outer[index_variable][~valid_masks[index_variable]],
                            brms_outer_est[~valid_masks[index_variable]],
                            s=markersize[~valid_masks[index_variable]], alpha=.5, color = color_outer_cut,
                            label = label)
        axes[1,index_axis].scatter(variables_outer[index_variable][valid_masks[index_variable]],
                            brms_outer_est[valid_masks[index_variable]],
                            s=markersize[valid_masks[index_variable]], alpha=.5, color = color_outer,
                            label = label_cut)
        axes[1,index_axis].set_xlabel(variable_labels[index_variable], fontsize=int(fontsize))

    for ax in [axes[0,0],axes[0,1],axes[0,2],axes[1,0],axes[1,1],axes[1,2]]:
        ax.set_xscale('log')
        ax.set_yscale('log')

        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
         
        #ax.set_ylim([1e-3,1e3])
        ax.xaxis.set_tick_params(labelsize=fontsize)
        ax.yaxis.set_tick_params(labelsize=fontsize)

        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        low = max(xmin, ymin)
        high = min(xmax, ymax)
        x = np.logspace(np.log10(1.1*low), np.log10(0.9*high), 20)
        ax.plot(x, x, '--', color = "k")
        
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_yticklabels([])
        ax.set_xticklabels([])


    for ax in [axes[0,0],axes[0,1],axes[0,2]]:
        ax.set_xticks([50,100,200])
        ax.set_xticklabels([r"$\mathrm{50}$",r"$\mathrm{100}$",r"$\mathrm{200}$"])

    for ax in [axes[1,0],axes[1,1],axes[1,2]]:
        ax.set_xticks([2,5,10,20])
        ax.set_xticklabels([r"$\mathrm{2}$",r"$\mathrm{5}$",r"$\mathrm{10}$",r"$\mathrm{20}$"])

    for ax in [axes[0,0],axes[1,0]]:
        ax.set_yticks([0.1,1,10,100,500])
        ax.set_yticklabels([r"$\mathrm{0.1}$",r"$\mathrm{1}$",r"$\mathrm{10}$",
                            r"$\mathrm{100}$",r"$\mathrm{500}$"])

    axes[0,2].legend(fontsize=int(fontsize))
    axes[1,2].legend(fontsize=int(fontsize))

    fig.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0.2)
    plt.savefig(path + "general_comparison_b_rms_dyn.png", bbox_inches='tight')


    plt.rcParams['text.usetex'] = True
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(22,15), sharey=True)
    plt.rcParams["figure.figsize"] = (22, 15)
    plt.rcParams['font.size'] = fontsize
    plt.rcParams['lines.linewidth'] = 1
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams["mathtext.fontset"] = "cm"

    axes[0,0].set_ylabel(r"$B_{\rm dip, \, surf} \, \mathrm{(G)}$", fontsize=int(fontsize)*1.1)
    axes[1,0].set_ylabel(r"$B_{\rm dip, \, surf} \, \mathrm{(G)}$", fontsize=int(fontsize)*1.1)

    for index_axis in range(0,3):
        if index_axis == 0:
            index_variable = 1
            label, label_cut = None, None
        elif index_axis == 1:
            index_variable = 5
            label, label_cut = None, None
        elif index_axis == 2:
            label = label_inner_cut
            label_cut = label_inner
            index_variable = 9
        axes[0,index_axis].scatter(variables_inner[index_variable][~valid_masks[index_variable]],
                            brms_inner_surf_est[~valid_masks[index_variable]],
                            s=markersize[~valid_masks[index_variable]], alpha=.5, color = color_inner_cut,
                            label = label)
        axes[0,index_axis].scatter(variables_inner[index_variable][valid_masks[index_variable]],
                            brms_inner_surf_est[valid_masks[index_variable]],
                            s=markersize[valid_masks[index_variable]], alpha=.5, color = color_inner,
                            label = label_cut)
        axes[0,index_axis].scatter(variables_outer[index_variable][~valid_masks[index_variable]],
                            brms_outer_surf_est[~valid_masks[index_variable]],
                            s=markersize[~valid_masks[index_variable]], alpha=.15, color = color_outer_cut)
        axes[0,index_axis].scatter(variables_outer[index_variable][valid_masks[index_variable]],
                            brms_outer_surf_est[valid_masks[index_variable]],
                            s=markersize[valid_masks[index_variable]], alpha=.15, color = color_outer)
        axes[0,index_axis].set_xlabel(variable_labels[index_variable], fontsize=int(fontsize))

        if index_axis == 0:
            index_variable = 3
            label, label_cut = None, None
        elif index_axis == 1:
            index_variable = 7
            label, label_cut = None, None
        elif index_axis == 2:
            label = label_inner_cut
            label_cut = label_outer
            index_variable = 11
        axes[1,index_axis].scatter(variables_inner[index_variable][~valid_masks[index_variable]],
                            brms_inner_surf_est[~valid_masks[index_variable]],
                            s=markersize[~valid_masks[index_variable]], alpha=.15, color = color_inner_cut)
        axes[1,index_axis].scatter(variables_inner[index_variable][valid_masks[index_variable]],
                            brms_inner_surf_est[valid_masks[index_variable]],
                            s=markersize[valid_masks[index_variable]], alpha=.15, color = color_inner)
        axes[1,index_axis].scatter(variables_outer[index_variable][~valid_masks[index_variable]],
                            brms_outer_surf_est[~valid_masks[index_variable]],
                            s=markersize[~valid_masks[index_variable]], alpha=.5, color = color_outer_cut,
                            label = label)
        axes[1,index_axis].scatter(variables_outer[index_variable][valid_masks[index_variable]],
                            brms_outer_surf_est[valid_masks[index_variable]],
                            s=markersize[valid_masks[index_variable]], alpha=.5, color = color_outer,
                            label = label_cut)
        axes[1,index_axis].set_xlabel(variable_labels[index_variable], fontsize=int(fontsize))

    for ax in [axes[0,0],axes[0,1],axes[0,2],axes[1,0],axes[1,1],axes[1,2]]:
        ax.set_xscale('log')
        ax.set_yscale('log')

        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
         
        #ax.set_ylim([1e-3,1e3])
        ax.xaxis.set_tick_params(labelsize=fontsize)
        ax.yaxis.set_tick_params(labelsize=fontsize)

        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        low = max(xmin, ymin)
        high = min(xmax, ymax)
        x = np.logspace(np.log10(1.1*low), np.log10(0.9*high), 20)
        ax.plot(x, x, '--', color = "k")
        
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_yticklabels([])
        ax.set_xticklabels([])


    for ax in [axes[0,0],axes[0,1],axes[0,2]]:
        ax.set_xticks([2,10,20,100])
        ax.set_xticklabels([r"$\mathrm{2}$",r"$\mathrm{10}$",r"$\mathrm{20}$",r"$\mathrm{100}$"])

    for ax in [axes[1,0],axes[1,1],axes[1,2]]:
        ax.set_xticks([0.1,1,10])
        ax.set_xticklabels([r"$\mathrm{0.1}$",r"$\mathrm{1}$",r"$\mathrm{10}$"])

    for ax in [axes[0,0],axes[1,0]]:
        ax.set_yticks([0.01,0.1,1,10,100])
        ax.set_yticklabels([r"$\mathrm{0.01}$",r"$\mathrm{0.1}$",r"$\mathrm{1}$",
                            r"$\mathrm{10}$",r"$\mathrm{100}$"])

    axes[0,2].legend(fontsize=int(fontsize))
    axes[1,2].legend(fontsize=int(fontsize))

    fig.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0.2)
    plt.savefig(path + "general_comparison_b_dip_surf.png", bbox_inches='tight')   

def plot_population_fits_eps(planets,
                name_cut,
                valid_mask,
                fit_type, fit_MRL_eps_inner, fit_MRL_eps_outer,
                color_inner = "blue", color_inner_cut = "deepskyblue",
                color_outer = "darkgreen", color_outer_cut = "limegreen", 
                color_irr = "k",
                rotation = False):
    
    heat_eff_inner_est      = planets.heat_eff_inner_est
    heat_eff_inner_est_err  = planets.heat_eff_inner_est_err
    heat_eff_outer_est      = planets.heat_eff_outer_est
    heat_eff_outer_est_err  = planets.heat_eff_outer_est_err
    mass_obs                = planets.mass_obs
    mass_obs_err            = planets.mass_obs_err
    flux_obs                = planets.flux_obs
    flux_obs_err            = planets.flux_obs_err
    temp_eff_obs            = planets.temp_eff_obs
    radius_obs              = planets.radius_obs
    radius_obs_err          = planets.radius_obs_err
    
    if not os.path.exists("./plots"): os.makedirs("./plots")

    if rotation: path = "plots/fits_eps_rot/"
    else: path = "plots/fits_eps/"

    fontsize = 32
    plt.rcParams['text.usetex'] = True
    fig = plt.figure(figsize=(11, 8), constrained_layout=True)
    plt.rcParams["figure.figsize"] = (11, 8)
    plt.rcParams['font.size'] = fontsize
    plt.rcParams['lines.linewidth'] = 1
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams["mathtext.fontset"] = "cm"

    surf_grav = 6.67e-11 * mass_obs * 1.8981e27 / (radius_obs * 6.9911e7)**2 # N (m/kg)^2

    variables_inner = [flux_obs,
                        temp_eff_obs,
                        radius_obs,
                        mass_obs,
                        surf_grav,
                        fit_MRL_eps_inner.K*mass_obs**fit_MRL_eps_inner.alpha*radius_obs**fit_MRL_eps_inner.beta*(flux_obs/1e9)**fit_MRL_eps_inner.gamma,
                        fit_MRL_eps_outer.K*mass_obs**fit_MRL_eps_outer.alpha*radius_obs**fit_MRL_eps_outer.beta*(flux_obs/1e9)**fit_MRL_eps_outer.gamma]
    variables_outer = [flux_obs,
                        temp_eff_obs,
                        radius_obs,
                        mass_obs,
                        surf_grav,
                        fit_MRL_eps_inner.K*mass_obs**fit_MRL_eps_inner.alpha*radius_obs**fit_MRL_eps_inner.beta*(flux_obs/1e9)**fit_MRL_eps_inner.gamma,
                        fit_MRL_eps_outer.K*mass_obs**fit_MRL_eps_outer.alpha*radius_obs**fit_MRL_eps_outer.beta*(flux_obs/1e9)**fit_MRL_eps_outer.gamma]
    print_power_laws = [False,
                        False,
                        False,
                        False,
                        False,
                        True,
                        True]
    variable_names = ["fluxes",
                      "temps",
                      "radius",
                      "mass",
                      "surf_grav",
                      "fit_eps_inner" + name_cut,
                      "fit_eps_outer" + name_cut]
    variable_labels = [r"$F\mathrm{_{\rm irr} \, (erg \, cm^{-2} \, s^{-1})}$",
                       r"$T_{\rm eq} \, \mathrm{(K)}$",
                       r"$R_P \, \mathrm{(R_J)}$",
                       r"$M_P \, \mathrm{(M_J)}$",
                       r"$g_{\rm surf} \, \mathrm{(m \, s^{-2})}$",
                       r"${{{:.3e}}}$".format(fit_MRL_eps_inner.K) + r"$ \, \mathrm{\%}$" + \
                         r"$\, M_P^{{{:.4f}}}$".format(fit_MRL_eps_inner.alpha) + \
                         r"$\, R_P^{{{:.4f}}}$".format(fit_MRL_eps_inner.beta) +  \
                         r"$\, F_{\rm irr}$" + r"$\, ^{{{:.4f}}}$".format(fit_MRL_eps_inner.gamma),
                       r"${{{:.3e}}}$".format(fit_MRL_eps_outer.K) + r"$ \, \mathrm{\%}$" + \
                         r"$\, M_P^{{{:.4f}}}$".format(fit_MRL_eps_outer.alpha) + \
                         r"$\, R_P^{{{:.4f}}}$".format(fit_MRL_eps_outer.beta) + \
                         r"$\, F_{\rm irr}$" + r"$\, ^{{{:.4f}}}$".format(fit_MRL_eps_outer.gamma)]

    plt.rcParams['text.usetex'] = True
    fig = plt.figure(figsize=(11, 8), constrained_layout=True)
    plt.rcParams["figure.figsize"] = (11, 8)
    plt.rcParams['font.size'] = fontsize
    plt.rcParams['lines.linewidth'] = 1
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams["mathtext.fontset"] = "cm"

    for index_variable in range(0,len(variables_inner)):
        plt.clf()
        ax = plt.axes()
        ax.scatter(variables_inner[index_variable][valid_mask],
                    heat_eff_inner_est[valid_mask],
                    s=100*radius_obs[valid_mask]**2, alpha=.4, color = color_inner,
                    label = r"$\epsilon_{\rm inner}$")
        ax.scatter(variables_inner[index_variable][~valid_mask],
                    heat_eff_inner_est[~valid_mask],
                    s=100*radius_obs[~valid_mask]**2, alpha=.4, color = color_inner_cut)
        ax.scatter(variables_outer[index_variable][valid_mask],
                    heat_eff_outer_est[valid_mask],
                    s=100*radius_obs[valid_mask]**2, alpha=.4, color = color_outer,
                    label = r"$\epsilon_{\rm outer}$")
        ax.scatter(variables_outer[index_variable][~valid_mask],
                    heat_eff_outer_est[~valid_mask],
                    s=100*radius_obs[~valid_mask]**2, alpha=.4, color = color_outer_cut)
        ax.set_xscale('log')
        ax.set_yscale('log')
        if print_power_laws[index_variable]:
            xmin, xmax = plt.xlim()
            ymin, ymax = plt.ylim()
            low = max(xmin, ymin)
            high = min(xmax, ymax)
            x = np.logspace(np.log10(1.1*low), np.log10(0.9*high), 20)
            plt.plot(x, x, '--', color = "k")
        ax.set_xlabel(variable_labels[index_variable], fontsize=int(fontsize))
        ax.set_ylabel(r"$\epsilon \mathrm{(\%)}$", fontsize=int(fontsize))
        plt.legend(fontsize=int(fontsize))
        plt.savefig(path + variable_names[index_variable] + "_heat_eff.png")