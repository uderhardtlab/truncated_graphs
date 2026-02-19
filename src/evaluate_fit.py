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


def akaike_information_criterion(num_params, C_true, C_model):
    log_likelihood_model = log_likelihood(C_true, C_model)
    return 2 * num_params - 2 * log_likelihood_model


def support(b_opt, d):
    return b_opt / np.max(d)


def relative_slope(m_opt, C_true):
     return m_opt / (np.max(C_true) - np.min(C_true))


def mse(C_true, C_pred):
    return np.sum((C_true - C_pred) ** 2) / len(C_true)