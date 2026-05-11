from math import comb, gamma
import numpy as np
from tqdm.auto import tqdm
from scipy.special import gammaln


def ega_weight(n, s):
    return 1 if s == 0 else 0


def loo_weight(n, s):
    return 1 if s == n - 1 else 0


def shapley_weight(n, s):
    return 1 / n


def banzhaf_weight(n, s):
    return comb(n - 1, s) / (2 ** (n - 1))


def beta_weight(n, s, alpha, beta):
    beta_fn = lambda a, b: gamma(a) * gamma(b) / gamma(a + b)
    return comb(n-1, s) * beta_fn(beta + s, alpha + n - 1 - s) / beta_fn(alpha, beta)


def beta_weight_fast(n, s, alpha, beta):
    log_result = np.log(np.arange(beta, beta + s, 1.0)).sum() + np.log(np.arange(alpha, alpha + n - 1 - s, 1.0)).sum() - np.log(np.arange(alpha + beta, alpha + beta + n - 1, 1.0)).sum()
    log_result += gammaln(n) - gammaln(s + 1) - gammaln(n - s)
    return np.exp(log_result)


def get_weight(n: int, s: int, data_valuation_function: str="dummy", **kwargs):
    if data_valuation_function == "beta":
        alpha = kwargs["alpha"]
        beta = kwargs["beta"]
        return beta_weight_fast(n, s, alpha, beta)
    elif data_valuation_function == "loo":
        return loo_weight(n, s)
    elif data_valuation_function == "shapley":
        return shapley_weight(n, s)
    elif data_valuation_function == "banzhaf":
        return banzhaf_weight(n, s)
    else:
        return 0


def get_weights(n: int, data_valuation_function: str="dummy", **kwargs):
    values = np.zeros(n)
    for s in tqdm(range(n)):
        values[s] = get_weight(n, s, data_valuation_function, **kwargs)

    return values
