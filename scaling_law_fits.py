"""
Scaling-law fits of the interpolated magnetic fields (or heating efficiencies) of the population.

A fit is defined by three pieces, each in a table below:
  - FIT_VARIABLES: the observables a fit can depend on (mass, radius, flux, luminosity),
  - FIT_TARGETS:   the fitted quantity (the magnetic field B or the heating efficiency eps),
  - FIT_FORMS:     the fit functions (MRL, MRF, MR): which variables, which parameters, the model
                   and how the result is written in the fit files.
general_fit puts them together with one of the fit methods ("simple", "odr" or "gaussian") and
returns a fit_object, which also writes and reads the fit files.

To add a new fit (for example eps = K M^alpha R^beta exp(-(log10 F - mu)^2 / (2 sigma^2))):
  1. write its model in ln space, model(ln_variables, *parameters), like power_law below;
  2. add a FitForm to FIT_FORMS with its variables, parameter names, initial guess, model and the
     lines written in the fit files (new parameters are saved after alpha, beta, gamma and K);
  3. if it needs a new observable, add a FitVariable to FIT_VARIABLES;
  4. add it to FIT_GROUPS in main.py, so it is fitted, saved and read back.
"""
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import pearsonr, spearmanr
from odrpack import odr_fit
from table_predictions_save import sample_split_normal
from constants import R_J, L_J


def power_law(ln_variables, K, *exponents):
    # ln of K * x1^exponent1 * x2^exponent2 * ..., from the ln of the variables
    result = np.log(K)
    for ln_variable, exponent in zip(ln_variables, exponents):
        result = result + ln_variable*exponent
    return result

def get_chi2(lnb, lnb_err, lnb_fit):
    residuals = lnb - lnb_fit
    chi2 = np.nansum((residuals / lnb_err)**2)
    return chi2


class FitData:
    # Values and errors of the planets used in a fit (masses in M_J, radii in R_J, fluxes in
    # erg cm^-2 s^-1, B in G, eps as a fraction). The *_err are symmetric errors and the
    # *_err_asym (lower, upper) errors, used to sample the observed values in the Gaussian fits.

    def __init__(self, planets, valid_mask, dyn_or_surf, heating_type, error_sym):
        if len(valid_mask) == 0: valid_mask = np.ones(planets.n_planets, dtype=bool)

        # The heating efficiency of the chosen heating type and the magnetic field (rms,dyn or dip,surf).
        # The eps fits use the planets of the B_rms,dyn sample of their heating type.
        heat_eff_name = "heat_eff_outer" if heating_type == "outer" else "heat_eff_inner"
        brms_name = "brms_" + heating_type + ("_surf" if dyn_or_surf == "dip,surf" else "")
        heat_eff, heat_eff_err = getattr(planets, heat_eff_name + "_est"), getattr(planets, heat_eff_name + "_est_err")
        brms, brms_err = getattr(planets, brms_name + "_est"), getattr(planets, brms_name + "_est_err")

        # Remove the predictions with Brms = 0, to not include them for the Brms_outer fits
        keep = np.where(valid_mask)[0]
        keep = keep[brms[keep] > 0]

        self.mass,   self.mass_err,   self.mass_err_asym   = planets.mass_obs[keep],   planets.mass_obs_err[keep],   planets.mass_obs_err_asym[keep]
        self.radius, self.radius_err, self.radius_err_asym = planets.radius_obs[keep], planets.radius_obs_err[keep], planets.radius_obs_err_asym[keep]
        self.flux,   self.flux_err,   self.flux_err_asym   = planets.flux_obs[keep],   planets.flux_obs_err[keep],   planets.flux_obs_err_asym[keep]
        self.heat_eff, self.heat_eff_err = heat_eff[keep], heat_eff_err[keep]
        self.brms,     self.brms_err     = brms[keep],     brms_err[keep]
        if not error_sym:   # symmetric errors: mean of the lower and upper errors
            self.heat_eff_err = np.mean(self.heat_eff_err, axis=1)
            self.brms_err = np.mean(self.brms_err, axis=1)

        # Deposited heating power L_P = eps pi R^2 F_irr, in units of L_J
        self.conv_lumin = np.pi*(self.radius*R_J)**2*self.flux*self.heat_eff/L_J
        self.conv_lumin_err = self.conv_lumin*np.sqrt((self.flux_err/self.flux)**2+\
                                                      (2*self.radius_err/self.radius)**2+\
                                                      (self.heat_eff_err/self.heat_eff)**2)


def log_positive(values):
    # ln of the values, with np.nan where they are not positive (draws that cannot be used)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(values > 0, np.log(values), np.nan)


class FitVariable:
    # An observable x a fit depends on, used in ln(x/scale). value, error and draw take a FitData;
    # draw gives one Monte Carlo realization of x for the Gaussian fits.
    def __init__(self, value, error, draw, scale = 1):
        self.value, self.error, self.draw, self.scale = value, error, draw, scale

    def ln(self, data):
        return np.log(self.value(data)/self.scale), self.error(data)/self.value(data)

    def ln_draw(self, data):
        return log_positive(self.draw(data)/self.scale)

def split_normal_draw(name):
    # Sample a value with its (lower, upper) errors
    return lambda data: sample_split_normal(getattr(data, name), getattr(data, name + "_err_asym")[:,0],
                                            getattr(data, name + "_err_asym")[:,1], len(getattr(data, name)))

FIT_VARIABLES = {
    "mass":       FitVariable(lambda data: data.mass, lambda data: data.mass_err, split_normal_draw("mass")),
    "radius":     FitVariable(lambda data: data.radius, lambda data: data.radius_err, split_normal_draw("radius")),
    # F_irr / (1e9 erg cm^-2 s^-1)
    "flux":       FitVariable(lambda data: data.flux, lambda data: data.flux_err, split_normal_draw("flux"), scale = 1e9),
    # L_P / L_J (eps is part of it, so it is drawn with a symmetric Gaussian)
    "luminosity": FitVariable(lambda data: data.conv_lumin, lambda data: data.conv_lumin_err,
                              lambda data: np.random.normal(data.conv_lumin, data.conv_lumin_err)),
}


class FitTarget:
    # The fitted quantity y, used in ln(y): ln(data) gives ln y and its error, ln_draw one Monte Carlo
    # realization of ln y (np.nan where it cannot be used).
    def __init__(self, ln, ln_draw, name):
        self.ln, self.ln_draw, self.name = ln, ln_draw, name

def ln_brms(data):
    # The model systematic uncertainty, a factor of 2 (x2 above, x1/2 below), is added in quadrature in ln B
    return np.log(data.brms), np.sqrt((data.brms_err / data.brms)**2 + np.log(2)**2)

FIT_TARGETS = {
    # B is sampled in ln B (log-normal), so the factor-of-2 uncertainty is symmetric in ln B
    "B":   FitTarget(ln_brms, lambda data: np.random.normal(*ln_brms(data)), "Brms"),
    # eps is sampled with a Gaussian in eps; the draws with eps <= 0 are not used
    "eps": FitTarget(lambda data: (np.log(data.heat_eff), data.heat_eff_err/data.heat_eff),
                     lambda data: log_positive(np.random.normal(data.heat_eff, data.heat_eff_err)), "eps"),
}


class FitForm:
    # A fit function: the variables it uses (in this order), its parameters (in the order of the model),
    # the model in ln space, and the lines written in the fit files. The lines are format strings with
    # the parameters, {quantity} and {unit} (Brms and G, or eps), {Kf} (format of K) and, in the
    # Gaussian error lines, {s} ("-" or "+").
    def __init__(self, variables, parameters, model, equation, errors_gaussian, errors_symmetric):
        self.variables, self.parameters, self.model = variables, parameters, model
        self.equation, self.errors_gaussian, self.errors_symmetric = equation, errors_gaussian, errors_symmetric

FIT_FORMS = {
    # B = K M^alpha R^beta (L_P/L_J)^gamma, with L_P = eps pi R^2 F_irr
    "MRL": FitForm(["mass", "radius", "luminosity"], ["K", "alpha", "beta", "gamma"], power_law,
                   "      {quantity} = {K:{Kf}}{unit} * M^{alpha:.4f} * R^{beta:.4f} * (eps * pi * R^2 * F_irr)^{gamma:.4f}",
                   "             {s}{K:{Kf}}      {s}{alpha:.4f}    {s}{beta:.4f}                           {s}{gamma:.4f}",
                   "              {K:{Kf}}       {alpha:.4f}     {beta:.4f}                            {gamma:.4f}"),
    # B = K M^alpha R^beta (F_irr/1e9)^gamma
    "MRF": FitForm(["mass", "radius", "flux"], ["K", "alpha", "beta", "gamma"], power_law,
                   "      {quantity} = {K:{Kf}}{unit} * M^{alpha:.4f} * R^{beta:.4f} * (F_irr/1e9)^{gamma:.4f}",
                   "             {s}{K:{Kf}}      {s}{alpha:.4f}    {s}{beta:.4f}               {s}{gamma:.4f}",
                   "             {K:{Kf}}      {alpha:.4f}     {beta:.4f}                 {gamma:.4f}"),
    # B = K M^alpha R^beta
    "MR":  FitForm(["mass", "radius"], ["K", "alpha", "beta"], power_law,
                   "      {quantity} = {K:{Kf}}{unit} * M^{alpha:.4f} * R^{beta:.4f}",
                   "             {s}{K:{Kf}}      {s}{alpha:.4f}    {s}{beta:.4f}",
                   "              {K:{Kf}}       {alpha:.4f}     {beta:.4f}"),
}

# Columns of the saved fit files: alpha, beta, gamma and K (as in the published files), followed by
# the other parameters of the fit form, if any
def file_columns(fit_function_type):
    columns = ["alpha", "beta", "gamma", "K"]
    if fit_function_type in FIT_FORMS:
        columns += [name for name in FIT_FORMS[fit_function_type].parameters if name not in columns]
    return columns


class fit_object:
    """
    Result of a scaling-law fit: the parameters (alpha, beta, gamma, K and those of new fit forms) and
    their errors ((minus, plus) for the Gaussian fits, symmetric otherwise), the covariance matrix and
    the quality of the fit. print_fit_results writes it to the fit files and read_fit_data reads it back.
    """
    def __init__(self, fit_type, fit_function_type):
        self.fit_type = fit_type
        self.fit_function_type = fit_function_type
        # Parameters (0 when a fit form does not use them, e.g. gamma in MR) and their errors:
        # (minus, plus) for the Gaussian fits, symmetric otherwise
        for name in file_columns(fit_function_type):
            setattr(self, name, 0)
            setattr(self, name + "_err", np.zeros(2) if fit_type == "gaussian" else 0)
        self.cov_matrix = np.zeros((4,4))
        self.chi_2 = 0
        self.chisquared = np.zeros(2)
        self.pearson = np.zeros(2)
        self.spearman = np.zeros(2)

    # This funtion writes the lines about the whole fit expressions on the specific file.txt

    def print_fit_results(self, file, filename, header):
        file.writelines(header + '\n')
        # The heating-efficiency fits (header "eps_...") are dimensionless; the other fits are Brms in G
        # Kf: the eps normalisation is ~1e-4, so it is written in scientific notation
        if header.startswith("eps"): text = dict(quantity="eps ", unit="  ", Kf=".4e")
        else: text = dict(quantity="Brms", unit=" G", Kf=".4f")
        form = FIT_FORMS[self.fit_function_type]
        columns = file_columns(self.fit_function_type)
        file.writelines(form.equation.format(**text, **{name: getattr(self, name) for name in columns}) + '\n')
        if self.fit_type == "gaussian":
            for side, sign in ((0, "-"), (1, "+")):
                errors = {name: getattr(self, name + "_err")[side] for name in columns}
                file.writelines(form.errors_gaussian.format(**text, s=sign, **errors) + '\n')
        else:
            errors = {name: getattr(self, name + "_err") for name in columns}
            file.writelines(form.errors_symmetric.format(**text, **errors) + '\n')
        #file.writelines(f"      chi_2 = {self.chi_2:.4f};   chi_squared = {self.chisquared[0]:.4f}, {self.chisquared[1]:.4f} " + '\n')
        file.writelines(f"      pearson = {self.pearson[0]:.4f}, {self.pearson[1]:.4e} " + '\n')
        #file.writelines(f"      spearman = {self.spearman[0]:.4f}, {self.spearman[1]:.4e} " + '\n')

        filename += "_" + header
        params  = np.array([getattr(self, name) for name in columns])
        if self.fit_type == "gaussian":
            errors_minus    = np.array([getattr(self, name + "_err")[0] for name in columns])
            errors_plus     = np.array([getattr(self, name + "_err")[1] for name in columns])
            np.savetxt(filename + ".txt", np.vstack((params, errors_minus, errors_plus)))
        else:
            errors          = np.array([getattr(self, name + "_err") for name in columns])
            np.savetxt(filename+ ".txt", np.vstack((params, errors)))

        filename += "_cov_matrix"
        np.savetxt(filename + ".txt", self.cov_matrix)


    def read_fit_data(self, filename, header):
        filename += "_" + header
        loaded = np.loadtxt(filename + ".txt")
        columns = file_columns(self.fit_function_type)
        if self.fit_type == "gaussian":
            params, errors_minus, errors_plus = loaded
            for index, name in enumerate(columns):
                setattr(self, name, params[index])
                getattr(self, name + "_err")[0], getattr(self, name + "_err")[1] = errors_minus[index], errors_plus[index]
        else:
            params, errors = loaded
            for index, name in enumerate(columns):
                setattr(self, name, params[index])
                setattr(self, name + "_err", errors[index])

        filename += "_cov_matrix"
        self.cov_matrix = np.loadtxt(filename + ".txt")


# ----------------------------------------------- Fit methods -----------------------------------------------
# Each takes the fit form, the data, the target and the initial guess and bounds, and returns the
# parameters, their errors and their covariance matrix.

def fit_simple(form, data, target, p0, bounds, N_gauss):
    # Least squares on the observed values, without their errors
    ln_variables = [FIT_VARIABLES[name].ln(data)[0] for name in form.variables]
    params, cov = curve_fit(form.model, tuple(ln_variables), target.ln(data)[0], p0=p0, bounds=bounds)
    return params, np.sqrt(np.diag(cov)), cov

def fit_odr(form, data, target, p0, bounds, N_gauss):
    # Orthogonal distance regression, weighted by the errors of the variables and of the target
    ln_variables, ln_variables_err = zip(*[FIT_VARIABLES[name].ln(data) for name in form.variables])
    ln_target, ln_target_err = target.ln(data)
    valid = ~np.isnan(ln_target) & ~np.isnan(ln_target_err)
    for ln_variable, ln_variable_err in zip(ln_variables, ln_variables_err):
        valid &= ~np.isnan(ln_variable) & ~np.isnan(ln_variable_err)

    out = odr_fit(lambda ln_x, params: form.model(ln_x, *params),
                  [ln_variable[valid] for ln_variable in ln_variables],
                  ln_target[valid],
                  p0,
                  bounds=bounds,
                  weight_x=[ln_variable_err[valid] for ln_variable_err in ln_variables_err],
                  weight_y=ln_target_err[valid])
    return out.beta, out.sd_beta, np.zeros((4,4))

def fit_gaussian(form, data, target, p0, bounds, N_gauss):
    # Monte Carlo: fit N_gauss realizations of the population, each drawn from the errors of the
    # variables and of the target. The parameters are the mean of the fits and the errors the 16th and
    # 84th percentiles. The draws are made in the order of the variables and then the target, so the
    # same seed gives the same fits.
    param_samples = []
    for _ in range(N_gauss):
        ln_draws = [FIT_VARIABLES[name].ln_draw(data) for name in form.variables]
        ln_target_draw = target.ln_draw(data)
        valid = np.isfinite(ln_target_draw)
        for ln_draw in ln_draws: valid &= np.isfinite(ln_draw)

        params, cov = curve_fit(form.model,
                                tuple(ln_draw[valid] for ln_draw in ln_draws),
                                ln_target_draw[valid],
                                p0=p0,
                                bounds=bounds)
        param_samples.append(params)

    param_samples = np.array(param_samples)
    params = param_samples.mean(axis=0)
    p16, p50, p84 = np.percentile(param_samples, q = [16, 50, 84], axis=0)
    errors = np.array((p50 - p16, p84 - p50)).transpose()
    cov_matrix = np.cov(param_samples, rowvar=False)
    if len(params) < 4:   # saved as a 4x4 matrix (K, alpha, beta, gamma), as the published files
        cov_matrix_small, cov_matrix = cov_matrix, np.zeros((4,4))
        cov_matrix[:len(params),:len(params)] = cov_matrix_small
    return params, errors, cov_matrix

FIT_METHODS = {"simple": fit_simple, "odr": fit_odr, "gaussian": fit_gaussian}


def general_fit(planets,
                valid_mask = [],
                dyn_or_surf = "rms,dyn",
                heating_type = "inner",
                print_all=False,
                error_sym=False,
                p0 = [4, 1/6, -7/6, 1/3],
                bounds=([1e-5, -np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf, np.inf]),
                fit_type = "simple", # can be "simple", "odr", "gaussian"
                fit_function_type = "MRL",
                N_gauss = 10000):
    """
    Fits the scaling law fit_function_type (a key of FIT_FORMS) to the planets in valid_mask (all if
    empty) with the method fit_type ("simple", "odr" or "gaussian") and returns a fit_object.
    dyn_or_surf: "rms,dyn" or "dip,surf" for the magnetic field, or "eps" for the heating efficiency;
    heating_type: "inner", "outer" or "irr" (the non-inflated eps = 0 models).
    """

    form = FIT_FORMS[fit_function_type]
    target = FIT_TARGETS["eps" if dyn_or_surf == "eps" else "B"]
    data = FitData(planets, valid_mask, dyn_or_surf, heating_type, error_sym)

    params, errors, cov_matrix = FIT_METHODS[fit_type](form, data, target, p0, bounds, N_gauss)

    # Quality of the fit, on the observed values
    ln_target, ln_target_err = target.ln(data)
    ln_fit = form.model(tuple(FIT_VARIABLES[name].ln(data)[0] for name in form.variables), *params)
    chi_2 = get_chi2(ln_target, ln_target_err, ln_fit)
    pearson_statistic, pearson_pvalue = pearsonr(ln_target, ln_fit)
    spearman_statistic, spearman_pvalue = spearmanr(ln_target, ln_fit)

    fit_to_return = fit_object(fit_type, fit_function_type)
    for index, name in enumerate(form.parameters):
        setattr(fit_to_return, name, params[index])
        setattr(fit_to_return, name + "_err", errors[index])
    fit_to_return.chi_2 = chi_2
    fit_to_return.cov_matrix = cov_matrix
    fit_to_return.pearson = np.array([pearson_statistic, pearson_pvalue])
    fit_to_return.spearman = np.array([spearman_statistic, spearman_pvalue])

    if print_all:
        print(f"")
        print(f"      {target.name}: " + ", ".join(f"{name} = {value:.4g}" for name, value in zip(form.parameters, params)))
        print(f"      chi^2 = {chi_2:.4f}")
        print(f"----------------------------------------------------------------------")
    else: print("done.  ", end=" ")

    return fit_to_return
