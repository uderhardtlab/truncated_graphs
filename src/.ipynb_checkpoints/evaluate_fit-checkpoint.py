import numpy as np 

def log_likelihood(C_true, C_pred):
    n = len(C_true)
    residuals = C_true - C_pred
    sigma2 = np.mean(residuals**2)  # MSE as variance estimate
    return -0.5 * n * (np.log(2 * np.pi * sigma2) + 1)


def relative_likelihood(C_true, C_model, C_baseline):
    log_likelihood_model = log_likelihood(C_true, C_model)
    log_likelihood_baseline = log_likelihood(C_true, C_baseline)
    return np.exp(log_likelihood_model - log_likelihood_baseline)


def akaike_information_criterion(num_params, log_likelihood_model):
    return 2 * num_params - 2 * log_likelihood_model


def calculate_fit_qualities(conc):
    conc["pieli_mon_incr"] = conc["pieli_m"] > 0
    conc["log_mon_incr"] = conc["log_a"] > 0
    conc["exp_mon_incr"] = (conc["exp_a"] > 0) == (conc["exp_b"] > 0)
    conc["const_mon_incr"] = False

    conc["measure"] = conc.index
    conc["dataset_type"] = conc["dataset"].apply(lambda x: x.split(":")[0])

    
    conc["const_aic"] = conc["const_ll"].apply(lambda x: akaike_information_criterion(2, x))
    conc["log_aic"] = conc["log_ll"].apply(lambda x: akaike_information_criterion(3, x))
    conc["exp_aic"] = conc["exp_ll"].apply(lambda x: akaike_information_criterion(4, x))
    conc["pieli_aic"] = conc["pieli_ll"].apply(lambda x: akaike_information_criterion(4, x))

    aics = [c for c in conc.columns if c.endswith("aic")] 
    best_fits = conc[aics].idxmin(axis=1)
    heatmap_matrix = pd.crosstab(best_fits.index, best_fits.values)

    conc["best_fit"] = best_fits
    conc["best_fit"] = conc["best_fit"].apply(lambda x: x.split("_aic")[0])

    conc["sign_border_effect"] = conc.apply(
    lambda row: 
            1
            if row[f"{row['best_fit']}_mon_incr"]
            else -1,
        axis=1
    )
    conc["sign_border_effect"] = np.where(conc["best_fit"] == "const", 0,  conc["sign_border_effect"])

    delta = conc[aics].sub(conc["const_aic"], axis=0)

    conc["d_aic"] = conc[["d_log_aic", "d_exp_aic", "d_pieli_aic"]].min(axis=1)
    conc["d_aic_per_node"] = - conc["d_aic"] / conc["num_nodes"]

    min_aic = conc[aics].min(axis=1)
    delta = conc[aics].sub(min_aic, axis=0)
    rel_likelihood = np.exp(-0.5 * delta)
    weight_sum = rel_likelihood.sum(axis=1)
    
    weights = rel_likelihood.div(weight_sum, axis=0)
    conc["const_akaike_weight"] = weights["const_aic"]
    conc["log_akaike_weight"]   = weights["log_aic"]
    conc["exp_akaike_weight"]   = weights["exp_aic"]
    conc["pieli_akaike_weight"] = weights["pieli_aic"]
    
    eps = 1e-15
    entropy = -(weights * np.log(weights + eps)).sum(axis=1)
    
    num_models = len(aics)
    conc["akaike_weights_entropy"] = entropy / np.log(num_models)
    return heatmap_matrix, conc
