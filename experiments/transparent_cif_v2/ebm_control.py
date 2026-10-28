"""Optional endpoint EBM control: choose rounds on TRAINING-INTERNAL group holdout.

No import-time InterpretML dependency. All candidate outcomes must precede the
same fit cutoff. This function never refits the selected model on the holdout.
It is an endpoint comparator, not a competing-risk curve estimator.
"""
import numpy as np
from crforest.transparent import disjoint_groups


def fit_ebm_control(X_structure, y_structure, w_structure, groups_structure,
                    X_select, y_select, w_select, groups_select,
                    interactions=0, seed=42, rounds=(50,200,500)):
    from interpret.glassbox import ExplainableBoostingClassifier
    isolation=disjoint_groups(groups_structure,groups_select)
    X=np.asarray(X_structure,float);V=np.asarray(X_select,float)
    y=np.asarray(y_structure);yv=np.asarray(y_select);w=np.asarray(w_structure,float);wv=np.asarray(w_select,float)
    if X.ndim!=2 or V.ndim!=2 or X.shape[1]!=V.shape[1] or np.isinf(X).any() or np.isinf(V).any():
        raise ValueError('Invalid aligned EBM predictors')
    for xx,yy,ww,gg in [(X,y,w,groups_structure),(V,yv,wv,groups_select)]:
        if yy.shape!=(len(xx),) or ww.shape!=yy.shape or len(gg)!=len(xx) or not np.isfinite(ww).all() or np.any(ww<0):
            raise ValueError('Invalid endpoint data/weights')
        if not np.isin(yy,[0,1]).all() or not (ww>0).any():raise ValueError('Binary known endpoint target required')
    if len(np.unique(y[w>0]))!=2:raise ValueError('Insufficient training event support')
    if not rounds or any(not isinstance(r,int) or r<1 for r in rounds):raise ValueError('Finite frozen rounds required')
    keep=w>0;logs=[];best=None
    for budget in rounds:
        model=ExplainableBoostingClassifier(feature_types=['continuous']*X.shape[1],
            interactions=interactions,max_bins=32,max_interaction_bins=16,
            max_rounds=budget,learning_rate=.03,greedy_ratio=0,
            smoothing_rounds=50,interaction_smoothing_rounds=50,
            min_samples_leaf=20,max_leaves=3,outer_bags=1,validation_size=0,
            early_stopping_rounds=0,n_jobs=1,random_state=seed)
        model.fit(X[keep],y[keep],sample_weight=w[keep])
        p=model.predict_proba(V)[:,1]
        # Same base-record denominator for all candidates; unknowns weight zero.
        score=float(np.sum(wv*(yv-p)**2)/len(V))
        row={'rounds':budget,'selection_ipcw_brier':score,'selection_gt99pct':int(np.sum(p>.99))}
        logs.append(row)
        if best is None or score<best[0]:best=(score,budget,model)
    return best[2],{'status':'INNER_HOLDOUT_SELECTED_NOT_TEST_TUNED','selected_rounds':best[1],
                    'isolation':isolation,'candidates':logs,'refit_on_holdout':False,
                    'scope':'12-month IPCW endpoint; log-odds components plus logistic link'}
