"""Predeclared descriptive ordering of exposed-test points; never adoption."""
import numpy as np
def rank_candidates(frame):
    z=frame[(frame.scope.astype(str)=='pooled')&(frame.seed.astype(str)=='ensemble')&
        (frame.version=='shrunk')&(frame.months==12)&(frame.cause==1)].copy()
    if z.empty or not z.supported.all() or z.arm.duplicated().any():raise ValueError('One supported pooled point per arm required')
    z['trees_per_seed']=z.arm.str[1:].astype(int);z['combined_trees']=z.trees_per_seed*3
    metrics=['within_year_auc','captured','ap','brier']
    if not np.isfinite(z[metrics].to_numpy()).all():raise ValueError('Undefined selection metric')
    for m in metrics:z['_'+m]=z[m].round(12)
    z=z.sort_values(['_'+m for m in metrics]+['trees_per_seed'],ascending=[False,False,False,True,True])
    z=z.drop(columns=['_'+m for m in metrics]);z['exploratory_rank']=range(1,len(z)+1)
    z['status']='EXPOSED_TEST_POINT_ORDER_NOT_ADOPTED'
    return z

