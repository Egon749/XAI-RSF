from pathlib import Path
from dataclasses import asdict
import argparse,datetime,io,json,shutil,subprocess,tarfile
import numpy as np
import pandas as pd
from crforest import CompetingRiskForest
from crforest.partition import shared_quantile_cuts,tree_feature_subset
from experiments.post_st_leaf_shallow_v1.analyze import artifacts
from experiments.post_st_leaf_shallow_v1.explain_company import LABELS
from .common import *


def prepare(run,repo):
    for n in ['input','evaluation_input','audit','reports','results','private']:(run/n).mkdir(exist_ok=False)
    sources={}
    def copy(src,dest):
        sources[str(src)]=sha(src);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
    src=HISTORY/'input/folds/2024'
    plan=json.loads((src/'plan.json').read_text());sources[str(src/'plan.json')]=sha(src/'plan.json')
    names=plan['feature_names'];keep=kept_columns(names)
    selection=repo/'experiments/post_st_feature_permutation_v1/results/feature_classification.csv'
    chosen=pd.read_csv(selection);chosen=chosen[(chosen.outer_cutoff==2024)&(chosen.subtype=='suspected_redundant')]
    assert set(chosen.name)==set(DROP_NAMES) and len(chosen)==9
    sources[str(selection)]=sha(selection)
    for n in ['train_PRIVATE.npz','eval_PRIVATE.npz']:copy(src/n,run/'input'/('reference_'+n))
    tr,ev=arrays(src/'train_PRIVATE.npz'),arrays(src/'eval_PRIVATE.npz')
    assert tr['X'].shape==(345,83) and len(ev['X'])==45 and set(ev)=={'X','sample_ids'}
    assert (tr['event']==1).sum()==26 and tr['year'].max()==2023 and len(np.unique(tr['groups']))==345
    for stem,data in [('train',tr),('eval',ev)]:
        new=reduced(data,keep)
        for k in data:np.testing.assert_array_equal(new[k],data[k][:,keep] if k=='X' else data[k])
        np.savez_compressed(run/f'input/{stem}_PRIVATE.npz',**new)
    meta_src=HISTORY/'evaluation_input/2024_fit_metadata_PRIVATE.parquet';sources[str(meta_src)]=sha(meta_src)
    meta=pd.read_parquet(meta_src);np.testing.assert_array_equal(meta.sample_id,tr['sample_ids'])
    assert (meta.stop<pd.Timestamp('2024-01-01')).all() and (meta.origin<pd.Timestamp('2024-01-01')).all()
    for field in ['financial_available','market_available']:
        assert (meta[field].isna() | (meta[field]<=meta.origin)).all()
    for n in ['test_PRIVATE.npz','test_metadata_PRIVATE.parquet']:copy(HISTORY/'evaluation_input'/f'2024_{n}',run/'evaluation_input'/n)
    # Test labels copied for locked later evaluation; preparation never loads them.
    reference=json.loads((repo/'CURRENT_POST_ST_MODEL.json').read_text());reuse=[]
    for y,arm,seed,folder in artifacts(HISTORY):
        if y!=2024 or arm!='shallow_parent':continue
        entry=next(z for z in reference['saved_models'] if z['test_year']==y and z['seed']==int(seed))
        for n,key in [('forest_PRIVATE.json','forest_sha256'),('predictions_PRIVATE.npz','predictions_sha256')]:
            assert sha(folder/n)==entry[key];copy(folder/n,run/'input/reference'/str(seed)/n)
        f=CompetingRiskForest.load(folder/'forest_PRIVATE.json');assert f.config==configuration(int(seed),'shallow_parent')
        old=arrays(folder/'predictions_PRIVATE.npz');np.testing.assert_array_equal(old['sample_ids'],ev['sample_ids'])
        np.testing.assert_array_equal(f.predict(ev['X'],old['times']),old['cif'])
        reuse.append(dict(seed=int(seed),forest_sha256=entry['forest_sha256'],prediction_sha256=entry['predictions_sha256'],reference_source=str(folder)))
    assert len(reuse)==3 and sha(repo/'CURRENT_POST_ST_MODEL.json')==REFERENCE_SHA
    cuts=shared_quantile_cuts(tr['X'],5);newcuts=shared_quantile_cuts(tr['X'][:,keep],5)
    for i,j in enumerate(keep):np.testing.assert_array_equal(newcuts[i],cuts[j])
    featuremap=[dict(original_index=j,reduced_index=keep.index(j) if j in keep else -1,name=n,label=LABELS[n],removed=n in DROP_NAMES) for j,n in enumerate(names)]
    write(run/'input/feature_map.json',featuremap)
    pd.DataFrame(featuremap).to_csv(run/'reports/feature_map.csv',index=False)
    coverage=[];paired=[]
    for seed in SEEDS:
        cfg=configuration(seed,'shallow_parent');a=pairing(tr['groups'],cfg,83);b=pairing(tr['groups'],cfg,74)
        for i,(aa,bb) in enumerate(zip(a,b)):
            assert aa['bootstrap_sha256']==bb['bootstrap_sha256'] and aa['multiplicity_sha256']==bb['multiplicity_sha256']
            oldset=set(map(int,tree_feature_subset(83,25,seed,i)));newset={keep[int(k)] for k in tree_feature_subset(74,25,seed,i)}
            paired.append(dict(seed=seed,tree=i,bootstrap_sha256=aa['bootstrap_sha256'],multiplicity_sha256=aa['multiplicity_sha256'],old_eligible=25,new_eligible=25,
                common_eligible=len(oldset&newset),removed_old_eligible=len(oldset&set(chosen.feature)),old_feature_hash=aa['features_sha256'],new_feature_hash=bb['features_sha256']))
        for j,n in enumerate(names):
            coverage.append(dict(seed=seed,original_index=j,name=n,label=LABELS[n],removed=n in DROP_NAMES,
                reference_eligible=sum(j in tree_feature_subset(83,25,seed,i) for i in range(32)),
                reduced_eligible=sum(j in [keep[int(k)] for k in tree_feature_subset(74,25,seed,i)] for i in range(32))))
    pd.DataFrame(paired).to_csv(run/'reports/tree_pairing.csv',index=False)
    pd.DataFrame(coverage).to_csv(run/'reports/feature_eligibility.csv',index=False)
    cfg=asdict(configuration(42,'shallow_parent'));differences=[dict(parameter=k,reference=json.dumps(v),drop9=json.dumps(v),changed=False) for k,v in cfg.items()]
    differences += [dict(parameter='input_columns',reference='83',drop9='74',changed=True),dict(parameter='direct_parent_AJ_strength',reference='10',drop9='10',changed=False),
        dict(parameter='case_time_weight',reference='1',drop9='1',changed=False)]
    pd.DataFrame(differences).to_csv(run/'reports/configuration_diff.csv',index=False)
    protocol=dict(name=NAME,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True,cwd=repo).strip(),
        test_year=2024,training_entry_years='2012-2023',fit_information_before='2024-01-01',
        training_companies=345,training_terminations=26,old_input='frozen conservative C; same X values, timestamps, missingness and labels',
        removed_names=DROP_NAMES,selection_source_sha256=sha(selection),kept_original_indices=keep,
        seeds=SEEDS,configuration=cfg,shrinkage_strength=10,case_time_weight=1,
        forest_fits=3,reused_reference_forests=3,pilot_fits=0,
        main='24 calendar months, first final termination; mean three-seed company forecasts before scoring',
        secondary='12 calendar months and recovery; raw new forests retained, not a new fitted arm',
        metrics=['AUC','AP','IPCW Brier','Top20 captured/coverage','mean risk','observed incidence'],
        top20='ceil(.20*45)=9 companies; descending risk, sample_id tie-break. Unknown outcomes neither negatives nor false positives.',
        uncertainty=dict(repeats=1000,seed=202610319,unit='company',paired=True,scope='fixed forecasts, conditional descriptive95% intervals; no training/selection uncertainty'),
        coverage_caveat='83->74 changes balanced scheduling blocks4->3, eligible identities and inclusion fraction25/83->25/74. Same25 candidates per tree/mtry75; effects include this induced schedule change. No claim identical search identities or pure removal-only biological causation.',
        minimal_code_change='mtry is an upper bound; permit mtry75>p74, use existing min(mtry,len(eligible)) semantics. Test identical trees against mtry74; old valid configs unchanged.',
        resources='3 independent single-threaded CPU jobs in existing two-node Slurm pool, each memory cap4GiB; no GPU conversion',
        verification='Every train/eval row, every native time, both causes: exactq2 reconstruction<=1e-10; saved reload, CIF nonnegative/nondecreasing/sum<=1; paired company bootstrap, same retained cuts; raw direct-parentAJ10 replay.',
        status='FROZEN_BEFORE_NEW_FIT',evidence='PROVISIONAL/post-exposure exploratory; keep current reference; no tuning, calibration or2025 access')
    public_protocol=repo/'experiments'/NAME/'frozen_protocol.json'
    if public_protocol.exists():
        recorded=json.loads(public_protocol.read_text())
        # Reproduction in a fresh RUN must preserve the published protocol.
        # The checkout commit can differ after aggregate results are committed.
        assert {k:v for k,v in recorded.items() if k!='source_commit'}=={k:v for k,v in protocol.items() if k!='source_commit'}
    else:write(public_protocol,protocol)
    write(run/'input/protocol.json',protocol)
    pd.DataFrame(reuse).to_csv(run/'reports/reused_artifacts.csv',index=False)
    write(run/'audit/source_hashes.json',sources)
    write(run/'audit/data_alignment.json',dict(status='PASS',training_companies=345,training_terminations=26,training_recoveries=int((tr['event']==2).sum()),
        training_2023_companies=int((tr['year']==2023).sum()),evaluation_companies=45,old_columns=83,new_columns=74,deleted=9,
        retained_values_equal=True,all_non_X_arrays_equal=True,retained_shared_cuts_equal=True,paired_bootstrap_trees=96,
        feature_names_order_verified=True,labels_in_pool_input=False,reference_sha256=REFERENCE_SHA,original_disclosure_versions_independently_verified=False))
    print(json.dumps(dict(new_columns=74,training_companies=345,new_fits_planned=3)))


def freeze(run,repo):
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo).strip()
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    archive=subprocess.check_output(['git','archive',commit],cwd=repo)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tf:tf.extractall(run/'input/code',filter='data')
    tasks=[dict(kind='cpu',argv=['python3',str(run/'input/code/experiments'/NAME/'launch.py'),'--seed',str(s)],cwd=str(run/'input/code'),timeout=3600,
        storage=dict(source=str(run/'input'),mode='nfs',results=str(run/'results/formal'))) for s in SEEDS]
    with (run/'formal.jsonl').open('x') as f:f.write(''.join(json.dumps(t)+'\n' for t in tasks))
    (run/'formal.py').write_text('# Three independent seed fits; immutable shared inputs.\n')
    write(run/'formal.py.hetero.json',dict(independent=True,manifest='formal.jsonl'))
    write(run/'audit/freeze.json',dict(commit=commit,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),tasks=3,protocol_sha256=sha(run/'input/protocol.json'),manifest_sha256=sha(run/'formal.jsonl')))
    write(run/'audit/immutable_input_hashes.json',{str(p.relative_to(run/'input')):sha(p) for p in (run/'input').rglob('*') if p.is_file()})
    print(commit)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--stage',choices=['prepare','freeze'],required=True);a=p.parse_args()
    repo=Path(__file__).resolve().parents[2];prepare(a.run,repo) if a.stage=='prepare' else freeze(a.run,repo)
