"""Fixed risk-index arithmetic. No fitting, clipping or outcome input."""
import numpy as np


def positive_denominator(mu):
    mu=np.asarray(mu,float)
    if not np.isfinite(mu).all() or (mu<=0).any():
        raise ValueError('Historical risk index undefined: denominator must be positive and finite')
    return mu


def risk_index(cif,mu):
    return 100*np.asarray(cif,float)/positive_denominator(mu)


def inverse_index(score,mu):
    return np.asarray(score,float)*positive_denominator(mu)/100


def point_components(baseline,components,mu):
    return risk_index(baseline,mu),risk_index(components,mu)


def historical_thresholds(scores):
    scores=np.asarray(scores,float)
    if scores.ndim!=1 or not np.isfinite(scores).all():
        raise ValueError('Only finite historical OOF indices accepted')
    if len(scores)<30:return None
    return np.quantile(scores,[.5,.8],method='linear')


def tiers(scores,thresholds):
    if thresholds is None:return np.full(np.shape(scores),'UNSUPPORTED')
    low,high=np.asarray(thresholds,float)
    if not np.isfinite([low,high]).all() or low>high:raise ValueError('Invalid thresholds')
    s=np.asarray(scores,float)
    if not np.isfinite(s).all():raise ValueError('Invalid scores')
    return np.where(s>=high,'high',np.where(s>=low,'medium','low'))
