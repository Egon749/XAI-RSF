"""Independent arithmetic, join and saved-tree OOB checks; never fit models."""
from pathlib import Path
import argparse
import ast
import json
import numpy as np
import pandas as pd
from experiments.post_st_prognosis_v1.common import sha,arrays,write
from experiments.post_st_train2024_h12_v1.validate import direct_metrics


def validate(run):
    private,out=run/'private',run/'results'
    meta=pd.read_parquet(private/'aligned_cases_PRIVATE.parquet').set_index('sample_id')
    pred=pd.read_parquet(private/'fixed_forecast_audit_PRIVATE.parquet')
    assert len(meta)==81 and meta.index.is_unique and meta.group.is_unique
    assert pred.groupby('model').size().eq(81).all()
    errs=pd.read_csv(out/'event_error_segments.csv',dtype={'scope':str,'seed':str})
    parts=pd.read_csv(out/'brier_decomposition.csv',dtype={'scope':str,'seed':str})
    metrics=pd.read_csv(out/'metric_replay.csv',dtype={'scope':str,'seed':str})
    checked=0
    for model,p in pred.groupby('model',sort=False):
        p=p.set_index('sample_id').loc[meta.index]
        assert np.array_equal(p.event,meta.event)
        assert ((p.risk>=0)&(p.recovery_risk>=0)&(p.risk+p.recovery_risk<=1+1e-12)).all()
        for scope,mask in [('2023',meta.entry_year.eq(2023)),('2024',meta.entry_year.eq(2024)),('pooled',np.ones(len(meta),bool))]:
            m=meta.loc[mask];pp=p.loc[mask];y=m.event.eq(1).to_numpy();risk=pp.risk.to_numpy()
            values=direct_metrics(y,risk)
            if scope!='pooled':
                expected=metrics.query("scope==@scope and model==@model and seed=='ensemble' and cause=='termination'").iloc[0]
                np.testing.assert_allclose(values,[expected.auc,expected.ap,expected.brier],rtol=0,atol=1e-12)
            b=parts.query("scope==@scope and model==@model and seed=='ensemble'").iloc[0]
            squared=float(np.dot(risk-y,risk-y)/len(y))
            assert abs(squared-b.brier)<1e-12 and abs(b.brier-b.squared_mean_bias-b.centered_residual)<1e-12
            eventloss=0;captured=0;count=0
            for ground,g in m[m.event.eq(1)].groupby('ground'):
                q=pp.loc[g.index];n=len(g);capt=int(q.top20.sum());eventloss+=float(np.sum((q.risk-1)**2))/len(m)
                row=errs.query("scope==@scope and model==@model and seed=='ensemble' and axis=='ground' and segment==@ground").iloc[0]
                assert row.cases==n and row.captured_top20==capt and row.missed_top20==n-capt
                assert abs(row.mean_case_risk-q.risk.mean())<1e-12
                a=q.risk.to_numpy();c=pp.loc[m.event.ne(1),'risk'].to_numpy()
                rank=sum((float(v>w)+.5*float(v==w)) for v in a for w in c)/(len(a)*len(c))
                assert abs(row.auc_cases_vs_original_nontermination-rank)<1e-12
                captured+=capt;count+=n;checked+=1
            assert count==y.sum() and captured==int((pp.top20&m.event.eq(1)).sum())
            assert abs(eventloss-b.event_case_loss)<1e-12
    # Independently regenerate sampled GROUP SETS, not the helper's bincount
    # implementation. Traverse only excluded-company trees and compare every
    # stored OOB value and tree count for all 24 Python forests.
    protocol=json.loads((run/'audit/protocol.json').read_text());source=Path(protocol['source_run'])
    inherited=json.loads((source/'audit/protocol.json').read_text())
    gate=json.loads((source/'audit/formal_gate.json').read_text())
    models=[]
    for i,s in enumerate((source/'formal.jsonl').read_text().splitlines()):
        a=json.loads(s)['argv'];models.append((int(a[a.index('--year')+1]),a[a.index('--variant')+1],a[a.index('--seed')+1],source/'results/formal'/str(gate['job'])/f'task-{i}'/'output'))
    for year in protocol['test_years']:
        ref=Path(inherited['reference_runs'][str(year)])
        for s in (ref/'results/formal').glob('*/task-*/output/summary.json'):
            summary=json.loads(s.read_text())
            if summary['model']=='unrestricted':models.append((year,'unrestricted',str(summary['seed']),s.parent))
    n_oob=0;maxerror=0.
    for year,model,seed,folder in models:
        data=arrays(source/'input/folds'/str(year)/'equal/train_PRIVATE.npz')
        f=json.loads((folder/'forest_PRIVATE.json').read_text());groups=np.unique(data['groups'])
        sums=np.zeros((len(data['X']),2));counts=np.zeros(len(sums),int)
        for t,tree in enumerate(f['trees']):
            rng=np.random.default_rng(np.random.SeedSequence([int(seed),t,0]))
            drawn=set(groups[rng.integers(len(groups),size=len(groups))])
            for i,(x,g) in enumerate(zip(data['X'],data['groups'])):
                if g in drawn:continue
                node=tree[0]
                while 'cif' not in node:
                    v=x[node['feature']]
                    use_left=bool(node['missing_left']) if np.isnan(v) else bool(v<=node['threshold'])
                    node=tree[node['left'] if use_left else node['right']]
                times=np.asarray(f['grid']);index=np.flatnonzero(times<=data['h24'][i])
                sums[i]+=np.asarray(node['cif'])[index[-1]] if len(index) else 0.
                counts[i]+=1
        saved=arrays(private/f'oob_{year}_{model}_{seed}_PRIVATE.npz')
        np.testing.assert_array_equal(saved['oob_trees'],counts)
        assert (counts>0).all()
        error=float(np.max(abs(sums/counts[:,None]-saved['cif'])));maxerror=max(error,maxerror)
        assert error<1e-12;n_oob+=1
    hashes=json.loads((run/'audit/source_hashes.json').read_text())
    for name,digest in hashes.items():assert sha(name)==digest
    here=Path(__file__).parent
    for name in ['analyze.py','oob.py']:
        tree=ast.parse((here/name).read_text())
        calls=[n.func for n in ast.walk(tree) if isinstance(n,ast.Call)]
        assert not any(isinstance(n,ast.Attribute) and n.attr in ['fit','fit_transform','reference_fit'] for n in calls)
        assert sha(here/name)==sha(run/'audit'/name.replace('.py','_frozen.py'))
    write(out/'independent_validation.json',dict(status='PASS',new_fits=0,group_cells_checked=checked,
        independent_OOB_forests_checked=n_oob,independent_OOB_prediction_max_error=maxerror,
        all_source_hashes_unchanged=True,all_forecast_rows_aligned=True,probabilities_valid=True,
        unknown_test_outcomes=0,brier_identity_verified=True,other_termination_not_used_as_controls=True,
        diagnostic_code_matches_frozen_copies=True,fit_calls_absent_from_analysis=True,
        sha256={str(p.relative_to(run)):sha(p) for p in sorted(out.iterdir()) if p.is_file()}))
    print((out/'independent_validation.json').read_text())


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True,type=Path);validate(parser.parse_args().run)
