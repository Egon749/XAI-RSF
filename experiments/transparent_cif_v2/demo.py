"""Known-CIF finite synthetic audit. Never reads enterprise data or tunes to it."""
import argparse
from dataclasses import asdict
from itertools import product
from pathlib import Path
import time
import numpy as np
from crforest.core import CompetingRiskForest,ForestConfig
from crforest.transparent import (TransparentCIF,WeightConfig,group_roles,learn_tree_weights,
                                  tree_predictions,solve_simplex_brier,write_json)
from crforest.pair_candidates import PairConfig,augment_pair_bank,extend_ordinary_bank


def generate(n,seed):
    rng=np.random.default_rng(seed);X=rng.choice([-1.,1.],(n,8))
    p=.18+.04*X[:,0]+.05*X[:,1]*X[:,2];u=rng.random(n)
    e=np.where(u<p,1,np.where(u<p+.06,2,0))
    t=np.where(e>0,rng.uniform(.05,.95,n),1.)
    return X,t,e,np.arange(n)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    records=[];grid=np.array(list(product([-1.,1.],repeat=8)))
    truth=.18+.04*grid[:,0]+.05*grid[:,1]*grid[:,2]
    for seed in [42,101,202]:
        X,t,e,g=generate(3000,seed);a,b=group_roles(g,.2)
        cfg=ForestConfig(n_trees=32,path_budget=2,max_depth=6,min_leaf=30,min_groups=15,min_events=3,
            mtry=8,n_thresholds=5,seed=seed,cut_strategy='shared_quantile',tree_feature_count=3)
        f=CompetingRiskForest(cfg).fit(X[a],t[a],e[a],g[a])
        pair,pa=augment_pair_bank(f,X[a],t[a],e[a],g[a],PairConfig(max_pairs=28,max_trees=8,max_cuts=2))
        extra=extend_ordinary_bank(f,X[a],t[a],e[a],g[a],pa['added_trees'])
        write_json(args.output/f'pair_audit_{seed}.json',pa)
        for label,bank in [('base',f),('ordinary_extra',extra),('pair',pair)]:
            weights,receipt=learn_tree_weights(bank,X[b],t[b],e[b],np.ones((len(b),1)),np.ones((len(b),1),bool),
                                               g[a],g[b],t[a],e[a])
            write_json(args.output/f'weight_receipt_{seed}_{label}.json',receipt)
            # Oracle is an optimistic capacity diagnostic only, never a deployable fit.
            P=tree_predictions(bank,grid,np.ones((len(grid),1)))
            Y=np.stack([1-truth-.06,truth,np.full(len(grid),.06)],axis=1)[:,None,:]
            ow,oreceipt=solve_simplex_brier(P,Y,np.ones((len(grid),1)),WeightConfig(ridge_to_uniform=0))
            write_json(args.output/f'oracle_capacity_receipt_{seed}_{label}.json',oreceipt)
            for name,w in [('equal',None),('honest_weights',weights),('ORACLE_TRUE_CIF_CAPACITY',ow)]:
                m=TransparentCIF.from_forest(bank,X[a],w)
                p=m.predict(grid,[1.])[:,0,0]
                coefficient=float(np.mean(p*grid[:,1]*grid[:,2]))
                main_coefficient=float(np.mean(p*grid[:,0]))
                error=float(abs(m.predict(grid,[1.])-(P@m.weights)).max())
                assert error<1e-10 and m.explain(grid[:8],[1.])['max_error']<1e-10
                records.append({'seed':seed,'bank':label,'aggregation':name,'trees':len(bank.trees_),
                    'true_pair_coefficient':.05,'estimated_pair_coefficient':coefficient,
                    'true_main_coefficient':.04,'estimated_main_coefficient':main_coefficient,
                    'main_amplitude_ratio':main_coefficient/.04,
                    'pair_amplitude_ratio':coefficient/.05,'uniform_grid_mse':float(np.mean((p-truth)**2)),
                    'native_reconstruction_error':error,'effective_trees':float(1/(m.weights@m.weights))})
    write_json(args.output/'results.json',{'status':'SYNTHETIC_ENGINEERING_AND_CAPACITY_ONLY',
        'real_company_fits':0,'records':records,'seconds':time.perf_counter()-start,
        'limitations':['Three fixed seeds, one binary DGP, not evidence of enterprise performance',
                      'Oracle uses truth ONLY as capacity diagnostic; not comparable as a practical algorithm',
                      'A convex mixture cannot generally add several disjoint effects at full amplitude',
                      'No additional seed or configuration search after viewing outcomes']})
    print('Saved 27 synthetic comparisons; all native reconstruction checks passed.')

if __name__=='__main__':main()
