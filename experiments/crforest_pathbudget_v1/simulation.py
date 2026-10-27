"""Frozen known-CIF simulations; all original predictors are individual coordinates."""
import numpy as np

SCENARIOS = ['low_cif', 'high_cif', 'correlated', 'rare_target', 'heavy_censor', 'additive_log_hazard']
SEEDS = [42, 101, 202]


def covariates(n, seed, scenario):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 8))
    if scenario == 'correlated':
        z = rng.normal(size=n)
        X[:, :4] = np.sqrt(.8) * z[:, None] + np.sqrt(.2) * X[:, :4]
    return X


def parameters(X, scenario):
    if scenario == 'additive_log_hazard':
        return np.column_stack([.18 * np.exp(.6 * X[:, 0] + .5 * X[:, 1]),
                                .14 * np.exp(.6 * X[:, 2])])
    p1 = .30 + .10 * np.tanh(X[:, 0]) + .12 * np.tanh(X[:, 1]) * np.tanh(X[:, 2])
    if scenario == 'high_cif':
        p1 = .06 + .75 * ((X[:, 0] > 0) & (X[:, 1] > 0) & (X[:, 2] > 0))
    if scenario == 'rare_target':
        p1 *= .1
    p2 = .10 + .03 * np.tanh(X[:, 3])
    return np.column_stack([p1, p2])


def true_cif(X, times, scenario):
    p = parameters(X, scenario)
    if scenario == 'additive_log_hazard':
        rate = p.sum(1)
        return (p / rate[:, None])[:, None, :] * (-np.expm1(-rate[:, None] * np.asarray(times))[:, :, None])
    return p[:, None, :] * (-np.expm1(-.8 * np.asarray(times)))[None, :, None]


def sample(n, seed, scenario):
    X = covariates(n, seed, scenario)
    rng = np.random.default_rng(seed + 800000)
    p = parameters(X, scenario)
    if scenario == 'additive_log_hazard':
        latent = rng.exponential(1 / p.sum(1))
        event = np.where(rng.random(n) < p[:, 0]/p.sum(1), 1, 2)
    else:
        u = rng.random(n)
        event = np.where(u < p[:, 0], 1, np.where(u < p.sum(1), 2, 0))
        latent = np.where(event > 0, rng.exponential(1/.8, n), np.inf)
    rate = 1.4 if scenario == 'heavy_censor' else (.05 if scenario == 'low_cif' else .30)
    censor = np.minimum(2., rng.exponential(1/rate, n))
    observed = latent <= censor
    return dict(X=X, time=np.minimum(latent, censor), event=np.where(observed, event, 0), groups=np.arange(n))
