"""Competing-event controls; early censoring has zero endpoint metric weight."""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def endpoint_weights(time, event, horizon):
    time, event = np.asarray(time), np.asarray(event)
    times, inv, count = np.unique(time, return_inverse=True, return_counts=True)
    risk = np.cumsum(count[::-1])[::-1]
    # Reverse KM with observed failures removed before censoring at tied times.
    # For continuous simulated censoring ties occur only at administrative H.
    deaths = np.bincount(inv, weights=event > 0, minlength=len(times))
    censor = np.bincount(inv, weights=event == 0, minlength=len(times))
    g = np.cumprod(1 - np.divide(censor, risk - deaths, out=np.zeros(len(times)), where=risk > deaths))
    left = np.searchsorted(times, time, side='left') - 1
    gt = np.where(left >= 0, g[np.maximum(left, 0)], 1.)
    j = np.searchsorted(times, horizon, side='left') - 1
    gh = g[j] if j >= 0 else 1.
    observed_event = (event > 0) & (time <= horizon)
    known = observed_event | (time >= horizon)
    w = np.where(known, 1 / np.maximum(np.where(observed_event, gt, gh), 1e-12), 0.)
    return w, dict(G_horizon_minus=float(gh), max_ipcw=float(w.max()), known=int(known.sum()))


def evaluate(time, event, prediction, horizon, cause, truth=None):
    w, audit = endpoint_weights(time, event, horizon)
    y = ((event == cause) & (time <= horizon)).astype(int)
    keep = w > 0
    order = np.lexsort((np.arange(len(y)), -prediction))
    flags = np.zeros(len(y), bool); flags[order[:int(np.ceil(.2 * len(y)))]] = True
    unique, counts = np.unique(prediction, return_counts=True)
    row = dict(auc=float(roc_auc_score(y[keep], prediction[keep], sample_weight=w[keep])) if len(np.unique(y[keep]))==2 else None,
               ap=float(average_precision_score(y[keep], prediction[keep], sample_weight=w[keep])) if y.sum() else None,
               ipcw_brier=float(np.mean(w * (y-prediction)**2)),
               prediction_mean=float(prediction.mean()), observed_ipcw_incidence=float(np.mean(w*y)),
               calibration_bias=float(prediction.mean()-np.mean(w*y)),
               top20_captured_events=int((flags & (y==1)).sum()), observed_events=int(y.sum()),
               top20_event_capture=float((flags & (y==1)).sum()/y.sum()) if y.sum() else None,
               distinct_predictions=len(unique), tied_record_fraction=float(counts[counts>1].sum()/len(y)),
               pair_tie_fraction=float(np.sum(counts*(counts-1))/(len(y)*(len(y)-1))), **audit)
    if truth is not None: row['cif_mse'] = float(np.mean((prediction-truth)**2))
    calibration = []
    for b, ids in enumerate(np.array_split(np.argsort(prediction, kind='stable'), 10)):
        calibration.append(dict(bin=b+1, n=len(ids), mean_prediction=float(prediction[ids].mean()),
                                observed_ipcw=float(np.mean(w[ids]*y[ids])), known=int((w[ids]>0).sum())))
    return row, calibration


def probability_audit(p):
    result = dict(min_cif=float(p.min()), min_time_increment=float(np.diff(p, axis=1).min(initial=0)),
                  max_cause_sum=float(p.sum(2).max()), finite=bool(np.isfinite(p).all()))
    assert result['finite'] and result['min_cif'] >= -1e-12 and result['min_time_increment'] >= -1e-12 and result['max_cause_sum'] <= 1+1e-12
    return result
