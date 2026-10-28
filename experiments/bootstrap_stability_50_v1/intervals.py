"""Conditional Monte Carlo intervals; independent units are paired refits, not edges."""
import numpy as np
from scipy.stats import t


def paired_u_interval(kernel, confidence=.95):
    """Mean off-diagonal paired contrast and delete-one-replicate jackknife t CI.

    The original company sample, model seed, anchors and masker are conditioned
    on. This interval measures precision of the bootstrap stability functional,
    not its uncertainty across independent original research datasets.
    """
    h=np.asarray(kernel,dtype=float)
    if h.ndim!=2 or h.shape[0]!=h.shape[1] or len(h)<4:
        raise ValueError('Need a square kernel on at least four paired refits')
    if not np.isfinite(h).all() or not np.allclose(h,h.T,atol=1e-14,rtol=0):
        raise ValueError('Kernel must be finite and symmetric')
    if not np.allclose(np.diag(h),0,atol=1e-14,rtol=0):
        raise ValueError('Diagonal must be zero (unused self contrasts)')
    n=len(h);total=np.triu(h,1).sum();estimate=total/(n*(n-1)/2)
    deleted=(total-h.sum(axis=1))/((n-1)*(n-2)/2)
    se=np.sqrt((n-1)/n*np.sum((deleted-deleted.mean())**2))
    margin=t.ppf((1+confidence)/2,n-1)*se
    return dict(n_replicates=n,n_pairs=n*(n-1)//2,estimate=float(estimate),
                jackknife_se=float(se),ci_low=float(estimate-margin),ci_high=float(estimate+margin),
                confidence=confidence,interval='conditional_MC_delete_one_replicate_jackknife_t')


def anchored_interval(differences, confidence=.95):
    """Per-refit paired distances from each model's original full-data fit."""
    d=np.asarray(differences,dtype=float)
    if d.ndim!=1 or len(d)<4 or not np.isfinite(d).all():raise ValueError('Need finite paired differences')
    n=len(d);mean=d.mean();se=d.std(ddof=1)/np.sqrt(n);alpha=(1-confidence)/2
    lo,hi=np.quantile(d,[alpha,1-alpha]);margin=t.ppf(1-alpha,n-1)*se
    return dict(n_replicates=n,estimate=float(mean),mc_se=float(se),ci_low=float(mean-margin),ci_high=float(mean+margin),
                distribution_low=float(lo),distribution_high=float(hi),confidence=confidence,
                interval='conditional_MC_mean_t',distribution_interval='conditional_refit_percentiles_not_mean_CI')
