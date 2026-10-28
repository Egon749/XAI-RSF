"""New private runs only. Structure/weight fit before cutoff; prediction is separate.

Train never opens final_eval data. Model selection never uses 2023/2024 outcomes.
The existing source cohort and its provisional limitations are not repaired here.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
import json
from pathlib import Path
import platform
import time
import traceback
import numpy as np
import pandas as pd
import scipy
from crforest.core import CompetingRiskForest, ForestConfig
from crforest.pair_candidates import PairConfig, augment_pair_bank, extend_ordinary_bank
from crforest.transparent import (TransparentCIF, WeightConfig, learn_tree_weights,
    tree_predictions, group_roles, file_sha256, write_json, state_targets)


def read_npz(path, keys):
    with np.load(path, allow_pickle=False) as z:
        if not set(keys).issubset(z.files):
            raise ValueError(f'Missing declared source keys: {set(keys)-set(z.files)}')
        return {k: z[k].copy() for k in keys}


def _plain(value):
    if isinstance(value, np.ndarray): return _plain(value.tolist())
    if isinstance(value, np.generic): return _plain(value.item())
    if isinstance(value, dict): return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_plain(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def calendar_horizons(origins):
    dates = pd.to_datetime(origins, errors='raise')
    if dates.isna().any(): raise ValueError('Missing prediction origins')
    return np.array([(d + pd.DateOffset(months=12) - d).days for d in dates], float)[:, None]


def read_metadata(source, role, metadata_format='parquet', columns=None):
    if metadata_format not in ('parquet','csv'):
        raise ValueError('Metadata format must be explicitly parquet or csv')
    path=Path(source)/f'final_{role}_metadata_PRIVATE.{metadata_format}'
    if metadata_format=='parquet':
        return pd.read_parquet(path,columns=columns).reset_index(drop=True)
    return pd.read_csv(path,usecols=columns,dtype={'sample_id':'string','group':'string'}).reset_index(drop=True)


def training_contract(source, cutoff, metadata_format='parquet'):
    source = Path(source)
    a = read_npz(source/'final_train_PRIVATE.npz', ['X','time','event','groups','sample_ids','horizon'])
    d = read_metadata(source,'train',metadata_format)
    required = ['sample_id','group','origin','stop_date','outcome_available','event','label_eligible']
    if not set(required).issubset(d): raise ValueError('Missing source time/eligibility contract')
    np.testing.assert_array_equal(a['sample_ids'].astype(str), d.sample_id.astype(str))
    np.testing.assert_array_equal(a['groups'].astype(str), d.group.astype(str))
    np.testing.assert_array_equal(a['event'], d.event)
    if len(set(a['sample_ids'])) != len(d): raise ValueError('Duplicate training sample IDs')
    for name in ['origin','stop_date','outcome_available']:
        dates = pd.to_datetime(d[name], errors='raise')
        if dates.isna().any() or not (dates < pd.Timestamp(cutoff)).all():
            raise ValueError(f'{name}: historical cutoff unsupported; no fabricated date repair')
    elapsed = (pd.to_datetime(d.stop_date)-pd.to_datetime(d.origin)).dt.days.to_numpy(float)
    if np.any(elapsed <= 0) or not np.allclose(elapsed, a['time'], atol=1e-9, rtol=0):
        raise ValueError('Source time must be actual days, not the old normalized 12-month grid')
    h = calendar_horizons(d.origin)
    if not np.array_equal(h[:,0], a['horizon']): raise ValueError('12-calendar-month contract mismatch')
    if not d.label_eligible.isin([True,False]).all(): raise ValueError('Unknown endpoint eligibility encoding')
    # Conservative source guard: unknown censored rows have zero endpoint weight.
    eligible = ((d.event.to_numpy()>0) | d.label_eligible.to_numpy(bool))[:,None]
    return a, d, h, eligible


def feature_contract(path, p):
    obj = json.loads(Path(path).read_text(encoding='utf-8'))
    fields = obj['features'] if isinstance(obj, dict) else obj
    if (not isinstance(fields,list) or len(fields)!=p or
            any(not isinstance(v,dict) or not all(str(v.get(k,'')).strip() for k in
                ['name','unit','definition','transform']) for v in fields)):
        raise ValueError('One explicit name/unit/definition/transform per unchanged source column required')
    if len({v['name'] for v in fields})!=p: raise ValueError('Duplicate feature names')
    return fields


def train(args, out):
    start=time.perf_counter(); source=Path(args.source_input)
    protocol=json.loads(Path(args.config).read_text());source_plan=json.loads((source/'plan.json').read_text())
    if 'forest_configs' not in source_plan or 'q2' not in source_plan['forest_configs']:
        raise ValueError('Use the frozen interpretable_baselines input/plan.json with forest_configs.q2')
    base_cfg=ForestConfig(**dict(source_plan['forest_configs']['q2'],seed=args.seed))
    if base_cfg.path_budget!=2 or base_cfg.global_cell_budget is not None:
        raise ValueError('Expected frozen non-budget-ledger q2 source configuration')
    a,meta,h,eligible=training_contract(source,protocol['fit_cutoff'],args.metadata_format)
    features=feature_contract(args.features,a['X'].shape[1])
    structural,weight_rows=group_roles(a['groups'],protocol['weight_fraction'],protocol['role_seed'])
    if (not np.any(a['event'][structural]==1) or not np.any(a['event'][weight_rows]==1)):
        raise ValueError('Frozen role split has no target support; do not search splits for favorable outcomes')
    hashes={str(p):file_sha256(p) for p in [source/'final_train_PRIVATE.npz',
        source/f'final_train_metadata_PRIVATE.{args.metadata_format}',source/'plan.json',Path(args.config),Path(args.features)]}
    roles={'structure_rows':len(structural),'weight_rows':len(weight_rows),
        'structure_groups':len(set(a['groups'][structural])), 'weight_groups':len(set(a['groups'][weight_rows])),
        'role_seed':protocol['role_seed'],'weight_fraction':protocol['weight_fraction'],
        'label_use':'training pre-cutoff only; no final_eval files opened',
        'cause_rows_structure':[int(np.sum(a['event'][structural]==k)) for k in [1,2]],
        'cause_rows_weight':[int(np.sum(a['event'][weight_rows]==k)) for k in [1,2]]}
    write_json(out/'frozen_run.json',{'source_hashes':hashes,'protocol':protocol,
        'forest_config':vars(base_cfg),'roles':roles,'seed':args.seed,'features':features})
    np.savez_compressed(out/'roles_PRIVATE.npz',structure=structural,weight=weight_rows,
                        sample_ids=a['sample_ids'],groups=a['groups'])
    aa={k:a[k][structural] for k in ['X','time','event','groups']}
    forest=CompetingRiskForest(base_cfg).fit(aa['X'],aa['time'],aa['event'],aa['groups'],n_jobs=args.tree_workers)
    augmented,pa=augment_pair_bank(forest,aa['X'],aa['time'],aa['event'],aa['groups'],PairConfig(**protocol['pair']))
    extra=extend_ordinary_bank(forest,aa['X'],aa['time'],aa['event'],aa['groups'],pa['added_trees'])
    write_json(out/'pair_candidate_audit_PRIVATE.json',pa)
    model_index=[];tol=protocol['prediction_tolerance'];causes=['target_first_warning','competing_first_warning']
    if forest.config.n_causes!=len(causes): raise ValueError('Enterprise contract expects the frozen two causes')
    for bank_name,bank in [('base',forest),('ordinary_extra',extra),('pair',augmented)]:
        bank_path=out/f'{bank_name}_forest_PRIVATE.json';bank.save(bank_path)
        weights,wr=learn_tree_weights(bank,a['X'][weight_rows],a['time'][weight_rows],a['event'][weight_rows],
            h[weight_rows],eligible[weight_rows],a['groups'][structural],a['groups'][weight_rows],
            a['time'][structural],a['event'][structural],WeightConfig(**protocol['weights']))
        write_json(out/f'{bank_name}_weight_receipt.json',wr)
        for mode,w in [('equal',None),('weighted',weights)]:
            name=f'{bank_name}_{mode}';folder=out/name
            model=TransparentCIF.from_forest(bank,aa['X'],w,[v['name'] for v in features],causes,
                    wr if mode=='weighted' else None)
            model.audit['structure_data_sha256']=hashes[str(source/'final_train_PRIVATE.npz')]
            model.save(folder)
            # All outputs in this stage are training-only checks, not model selection evidence.
            anchors=weight_rows[:min(64,len(weight_rows))]
            direct=tree_predictions(bank,a['X'][anchors],h[anchors])@model.weights
            native=model.predict_horizons(a['X'][anchors],h[anchors])
            err=float(abs(direct-native).max())
            if err>tol: raise AssertionError('Native CIF compiler changed predictions')
            loaded=TransparentCIF.load(folder)
            np.testing.assert_array_equal(loaded.predict_horizons(a['X'][anchors],h[anchors]),native)
            times=[365.,366.]
            explanation=model.explain(a['X'][anchors[:6]],times)
            write_json(folder/'complete_cases_PRIVATE.json',_plain(explanation))
            tables=model.complete_tables(times,aa['X'])
            write_json(folder/'complete_tables_PRIVATE.json',tables)
            complexity=model.expansion.complexity(aa['X'])
            write_json(folder/'complexity_PRIVATE.json',complexity)
            model_index.append({'name':name,'path':name,'seed':args.seed,'bank':bank_name,
                'trees':len(bank.trees_),'native_error':err,'reload_error':0.,
                'functions':len(tables['functions']),'spatial_cells':tables['spatial_cells'],
                'effective_trees':float(1/(model.weights@model.weights)),
                'manifest_sha256':file_sha256(folder/'manifest_PRIVATE.json')})
    if args.ebm_control:
        import pickle
        from .ebm_control import fit_ebm_control
        ys,ws,_=state_targets(a['time'][structural],a['event'][structural],h[structural],eligible[structural],
                              a['time'][structural],a['event'][structural],2)
        yv,wv,_=state_targets(a['time'][weight_rows],a['event'][weight_rows],h[weight_rows],eligible[weight_rows],
                              a['time'][structural],a['event'][structural],2)
        for interaction in [0,10]:
            model,receipt=fit_ebm_control(a['X'][structural],ys[:,0,1].astype(int),ws[:,0],a['groups'][structural],
                a['X'][weight_rows],yv[:,0,1].astype(int),wv[:,0],a['groups'][weight_rows],
                interactions=interaction,seed=args.seed,rounds=tuple(protocol['ebm_rounds']))
            name=f'ebm_pair{interaction}';folder=out/name;folder.mkdir()
            with (folder/'model_PRIVATE.pkl').open('xb') as ff:pickle.dump(model,ff)
            write_json(folder/'selection_receipt.json',receipt)
            model_index.append({'name':name,'path':name,'seed':args.seed,'kind':'endpoint_ebm',
                                'pickle_sha256':file_sha256(folder/'model_PRIVATE.pkl'),
                                'native_scale':'log_odds','selected_rounds':receipt['selected_rounds']})
    for p,digest in hashes.items():
        if file_sha256(p)!=digest: raise AssertionError('Input changed during run')
    write_json(out/'model_index.json',model_index)
    write_json(out/'execution_receipt.json',{'status':'TRAINED_NOT_EXTERNALLY_VALIDATED',
        'data_status':'PROVISIONAL','seed':args.seed,'roles':roles,'source_hashes':hashes,
        'model_index':model_index,'base_forest_fits':1,'selected_pair_trees':pa['added_trees'],
        'ordinary_control_trees':pa['added_trees'],'weight_qp_solves':3,
        'ebm_candidate_fits':2*len(protocol['ebm_rounds']) if args.ebm_control else 0,
        'fit_cutoff':protocol['fit_cutoff'],'seconds':time.perf_counter()-start,
        'versions':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__},
        'limitations':['No enterprise advantage claimed','Group holdout reduces tree-training data',
                       'Global convex weights need not recover all disjoint signals',
                       'Joint pair search is finite and heuristic; source PIT remains provisional']})


def predict(args,out):
    src,trained=Path(args.source_input),Path(args.trained)
    # No event, stop date or outcome column is read here.
    a=read_npz(src/'final_eval_PRIVATE.npz',['X','sample_ids','groups','horizon'])
    d=read_metadata(src,'eval',args.metadata_format,columns=['sample_id','origin'])
    np.testing.assert_array_equal(a['sample_ids'].astype(str),d.sample_id.astype(str))
    h=calendar_horizons(d.origin)
    if not np.array_equal(h[:,0],a['horizon']):raise ValueError('Evaluation calendar horizons changed')
    index=json.loads((trained/'model_index.json').read_text());logs=[]
    for item in index:
        folder=trained/item['path']
        if item.get('kind')=='endpoint_ebm':
            import pickle
            from scipy.special import expit
            if file_sha256(folder/'model_PRIVATE.pkl')!=item['pickle_sha256']:raise ValueError('EBM hash changed')
            # Load only this run's trusted private artifact, never arbitrary external pickle.
            with (folder/'model_PRIVATE.pkl').open('rb') as ff:model=pickle.load(ff)
            risk=model.predict_proba(a['X'])[:,1]
            reconstruction=expit(model.intercept_[0]+model.eval_terms(a['X']).sum(1))
            err=float(abs(risk-reconstruction).max())
            if err>1e-10:raise AssertionError('EBM native logit reconstruction mismatch')
            np.savez_compressed(out/f'{item["name"]}_PRIVATE.npz',sample_ids=a['sample_ids'],groups=a['groups'],
                risk=risk,horizon=a['horizon'])
            logs.append({'model':item['name'],'seed':item['seed'],'records':len(risk),
                         'native_logit_to_probability_error':err,'endpoint_only':True,
                         'prediction_sha256':file_sha256(out/f'{item["name"]}_PRIVATE.npz')})
            continue
        if file_sha256(folder/'manifest_PRIVATE.json')!=item['manifest_sha256']:raise ValueError('Model hash changed')
        model=TransparentCIF.load(folder);p=model.predict_horizons(a['X'],h)[:,0,:]
        if not np.isfinite(p).all() or p.min() < -1e-10 or p.sum(1).max()>1+1e-10:
            raise ValueError('Native model violated probability constraints')
        # Independently check weighted original trees over the whole evaluation cohort.
        bank=CompetingRiskForest.load(trained/f'{item["bank"]}_forest_PRIVATE.json')
        error=0.
        for start in range(0,len(p),256):
            sl=slice(start,start+256)
            direct=(tree_predictions(bank,a['X'][sl],h[sl])@model.weights)[:,0,:]
            error=max(error,float(abs(direct-p[sl]).max()))
        if error>1e-10:raise AssertionError('Full evaluation native/weighted-tree mismatch')
        np.savez_compressed(out/f'{item["name"]}_PRIVATE.npz',sample_ids=a['sample_ids'],
             groups=a['groups'],cif12=p,risk=p[:,0],survival12=1-p.sum(1),horizon=a['horizon'])
        logs.append({'model':item['name'],'seed':item['seed'],'records':len(p),
                     'full_reconstruction_max_error':error,
                     'prediction_sha256':file_sha256(out/f'{item["name"]}_PRIVATE.npz')})
    write_json(out/'prediction_receipt.json',{'status':'PREDICTED_NOT_SELECTED',
        'evaluation':'POST_EXPOSURE_EXPLORATORY','labels_read':False,'models':logs,
        'training_receipt_sha256':file_sha256(trained/'execution_receipt.json'),
        'evaluation_source_sha256':file_sha256(src/'final_eval_PRIVATE.npz')})


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    tr=sub.add_parser('train');pr=sub.add_parser('predict')
    for p in [tr,pr]:
        p.add_argument('--source-input',required=True);p.add_argument('--output',required=True)
        p.add_argument('--metadata-format',choices=['parquet','csv'],default='parquet')
    tr.add_argument('--config',default=str(Path(__file__).with_name('config.json')))
    tr.add_argument('--features',required=True);tr.add_argument('--seed',type=int,default=42)
    tr.add_argument('--tree-workers',type=int,default=1);tr.add_argument('--ebm-control',action='store_true')
    pr.add_argument('--trained',required=True)
    args=parser.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    try:
        (train if args.command=='train' else predict)(args,out)
    except Exception:
        write_json(out/'FAILED.json',{'status':'FAILED','traceback':traceback.format_exc()});raise

if __name__=='__main__':main()
