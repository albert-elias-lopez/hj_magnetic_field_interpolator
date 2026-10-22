"""
Checks that the Python interface (hj_model.py) gives exactly the committed predictions of main.py.

    python test_hj_model.py

It exits with an error if any value differs. The GitHub Action runs it on every push.
"""
import os
import re
import sys
import numpy as np
import planet_functions
from hj_model import HJMagneticModel

REPOSITORY = os.path.dirname(os.path.realpath(__file__))

def read_predictions(sample):
    # Values and (minus, plus) errors of predictions/predictions_<sample>.txt, as written by main.py
    columns = ["mass", "radius", "radius_irr", "flux", "period", "eps_inner", "eps_outer", "eps_th18",
               "brms_inner", "brms_irr", "brms_outer", "brms_inner_surf", "brms_irr_surf", "brms_outer_surf"]
    rows = []
    with open(os.path.join(REPOSITORY, "predictions", "predictions_" + sample + ".txt")) as f:
        for line in f:
            if line.startswith("#"): continue
            row = {}
            for name, cell in zip(columns, line.split()):
                match = re.match(r"([-\d.e+]+)\$_\{-([\d.e+-]+)\}\^\{\+([\d.e+-]+)\}\$", cell)
                row[name] = match.groups() if match else None
            rows.append(row)
    return rows

def check_estimates(model, sample):
    # The model with the default seed must reproduce the committed file of a --no_HJ sample
    planets = planet_functions.planet_object(1, False)
    planets.my_planets(sample)
    result = model.estimate(planets.mass_obs, tuple(planets.mass_obs_err_asym.T),
                            planets.radius_obs, tuple(planets.radius_obs_err_asym.T),
                            planets.flux_obs, tuple(planets.flux_obs_err_asym.T),
                            planets.orbital_period_obs, planets.orbital_period_obs_err)
    mismatches = 0
    for index, row in enumerate(read_predictions(sample)):
        for key in ["brms_inner", "brms_irr", "brms_outer", "brms_inner_surf", "brms_irr_surf", "brms_outer_surf",
                    "eps_inner", "eps_outer"]:
            if row[key] is None: continue     # planet outside the grid
            value, error = result[key][index], result[key + "_err"][index]
            if key.startswith("eps"):   # written in %
                mine = (f"{100*value:.4f}", f"{100*error[0]:.4f}", f"{100*error[1]:.4f}")
            else:
                mine = (f"{value:.4e}", f"{error[0]:.1e}", f"{error[1]:.1e}")
            if mine != row[key]:
                mismatches += 1
                print(f"  {sample}, planet {index}, {key}: {mine} instead of {row[key]}")
    print(f"estimate, {sample}: {'OK' if mismatches == 0 else str(mismatches) + ' values differ'}")
    return mismatches

def check_flux(model):
    # get_flux must give exactly the fluxes and errors of the archive table read by main.py
    filename = os.path.join(REPOSITORY, "PS_2026.02.18_08.49.05.csv")
    rows = planet_functions.read_archive_csv(filename)
    planets = planet_functions.planet_object(len(rows), False)
    current = os.getcwd()
    os.chdir(REPOSITORY)   # read_file finds the main-sequence table in the repository folder
    try: planets.read_file(filename)
    finally: os.chdir(current)

    def errors(column):
        return tuple(np.array([planet_functions.read_err_asym(row, column)[side] for row in rows]) for side in (0, 1))
    given = np.array([row["pl_orbsmax"] != "" and row["st_teff"] != "" and row["st_rad"] != "" for row in rows])
    flux, flux_err = model.get_flux(planets.radius_star_obs[given], tuple(e[given] for e in errors("st_rad")),
                                    planets.temp_star_obs[given], tuple(e[given] for e in errors("st_teff")),
                                    orbital_dist=planets.orbital_dist_obs[given],
                                    orbital_dist_err=tuple(e[given] for e in errors("pl_orbsmax")))
    flux_err = np.column_stack(flux_err if isinstance(flux_err, tuple) else (flux_err, flux_err))
    same = np.array_equal(flux, planets.flux_obs[given]) and \
           np.array_equal(flux_err, planets.flux_obs_err_asym[given], equal_nan=True)
    print(f"get_flux, {given.sum()} archive planets: {'OK' if same else 'values differ'}")
    return 0 if same else 1

if __name__ == "__main__":
    model = HJMagneticModel()
    failures = check_flux(model)
    for sample in ["hd209458b", "wasp12b", "seidel26", "cauley19"]:
        failures += check_estimates(model, sample)
    sys.exit(1 if failures else 0)
