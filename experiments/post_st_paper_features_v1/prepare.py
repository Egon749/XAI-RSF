from pathlib import Path
from dataclasses import asdict
import argparse,datetime,io,json,shutil,subprocess,tarfile
import numpy as np
import pandas as pd
from crforest import CompetingRiskForest
from crforest.partition import shared_quantile_cuts
from .common import *


def prepare(run,repo):
    for n in ['input','evaluation_input','audit','reports','private','results']:(run/n).mkdir(exist_ok=False)
    sources={}
    def source(p):sources[str(p)]=sha(p);return p
    def copy(src,dst):source(src);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    copy(PAPER,run/'private/source_paper.pdf')
    assert sha(source(repo/'CURRENT_POST_ST_MODEL.json'))==REFERENCE_SHA
    # Verify the same conservative entry-time83-vector and documented clocks.
    allmeta=pd.read_parquet(source(LATER/'private/shifted_cohort_PRIVATE.parquet'))
    clock=arrays(source(LATER/'private/financial_availability_PRIVATE.npz'))
    info=pd.read_parquet(source(LATER/'private/information_alignment_PRIVATE.parquet'))
    allparts=[arrays(source(LATER/f'input/folds/2025/{n}_PRIVATE.npz')) for n in ['train','eval']]
    allids=np.concatenate([z['sample_ids'] for z in allparts]);allX=np.vstack([z['X'] for z in allparts])
    order=pd.Index(allids).get_indexer(allmeta.sample_id);assert (order>=0).all();allX=allX[order]
    np.testing.assert_array_equal(clock['sample_ids'],allmeta.sample_id);np.testing.assert_array_equal(info.sample_id,allmeta.sample_id)
    assert (np.isfinite(allX[:,:48])<=~np.isnat(clock['dates'])).all()
    assert not ((~np.isnat(clock['dates']))&(clock['dates']>=allmeta.origin.to_numpy(dtype='datetime64[D]')[:,None])).any()
    assert (info.market_available.isna()|(info.market_available<allmeta.origin)).all()
    source(repo/'experiments/post_st_prognosis_v1/prepare.py')
    formula=Path('/home/oliver/LLMworkspace/early-warning-12m-server-isolated/experiments/early_warning_12m/source_reference/financial_formulas.py');source(formula)
    mapping=[]
    for i,name in enumerate(PAPER_NAMES,1):
        selected=i in SELECTED
        mapping.append(dict(paper_id=f'X{i}',paper_name=name,selected=selected,
            existing_name=SELECTED[i][0] if selected else '',operational_definition=SELECTED[i][1] if selected else '',
            status='AVAILABLE_ADAPTED_DEFINITION' if selected else 'NOT_IN_VERIFIED_FROZEN83',
            limitation='Paper table omits exact formula; inherited source definition is explicit, not certified author-code equivalence' if selected else
                'Requires a new audited input and exact formula/vintage/component availability; not filled with a proxy or all-NaN placeholder'))
    pd.DataFrame(mapping).to_csv(run/'reports/paper_variable_mapping.csv',index=False)
    pd.DataFrame(mapping).to_csv(repo/'experiments'/NAME/'paper_variable_mapping.csv',index=False)
    audits=[];reused=[];differences=[];paired=[];missing=[]
    for year in YEARS:
        src=history(year)/f'input/folds/{year}';dest=run/f'input/folds/{year}';dest.mkdir(parents=True)
        plan=json_load(source(src/'plan.json'));names=plan['feature_names'];idx=columns(names)
        tr=arrays(source(src/'train_PRIVATE.npz'));ev=arrays(source(src/'eval_PRIVATE.npz'))
        assert tr['X'].shape==({2023:309,2024:345,2025:390}[year],83)
        assert len(ev['X'])=={2023:36,2024:45,2025:53}[year] and set(ev)=={'X','sample_ids'}
        for d in [tr,ev]:
            ix=pd.Index(allmeta.sample_id).get_indexer(d['sample_ids']);assert (ix>=0).all()
            np.testing.assert_array_equal(d['X'],allX[ix])
        meta=pd.read_parquet(source(history(year)/f'evaluation_input/{year}_fit_metadata_PRIVATE.parquet'))
        np.testing.assert_array_equal(meta.sample_id,tr['sample_ids'])
        cutoff=pd.Timestamp(f'{year}-01-01');assert (meta.stop<cutoff).all() and (meta.origin<cutoff).all()
        for name,data in [('train',tr),('eval',ev)]:
            copy(src/f'{name}_PRIVATE.npz',dest/f'reference_{name}_PRIVATE.npz')
            z=transform(data,names);np.savez_compressed(dest/f'{name}_PRIVATE.npz',**z)
            for k in data:
                if k!='X':np.testing.assert_array_equal(z[k],data[k])
            np.testing.assert_array_equal(z['X'][:,0],z['X'][:,1])
            for j,k in enumerate(PAPER_IDS):missing.append(dict(year=year,role=name,paper_id=f'X{k}',name=SELECTED[k][0],
                rows=len(z['X']),missing_fraction=float(np.isnan(z['X'][:,j]).mean()),finite_unique=int(len(np.unique(z['X'][np.isfinite(z['X'][:,j]),j])))))
        for stem in ['test_PRIVATE.npz','test_metadata_PRIVATE.parquet']:
            copy(history(year)/f'evaluation_input/{year}_{stem}',run/f'evaluation_input/{year}_{stem}')
        copy(history(year)/f'evaluation_input/{year}_fit_metadata_PRIVATE.parquet',run/f'evaluation_input/{year}_fit_metadata_PRIVATE.parquet')
        write(dest/'plan.json',dict(year=year,feature_names=[f'X{k}_{SELECTED[k][0]}' for k in PAPER_IDS],original_indices=idx,
            paper_ids=PAPER_IDS,original_names=names,configuration=asdict(configuration(42))))
        for arm in ['reference83','drop9']:
            for seed,folder in references(year,arm,repo).items():
                # All reference versions are existing shrunk models; never fit again.
                for n in ['forest_PRIVATE.json','predictions_PRIVATE.npz']:
                    copy(folder/n,dest/'reference'/arm/str(seed)/n)
                source(folder/'functions_PRIVATE.npz')
                f=CompetingRiskForest.load(folder/'forest_PRIVATE.json');assert f.config==reference_configuration(seed,'shallow_parent')
                oldX=ev['X'] if arm=='reference83' else ev['X'][:,kept_columns(names)]
                z=arrays(folder/'predictions_PRIVATE.npz');np.testing.assert_array_equal(z['sample_ids'],ev['sample_ids'])
                np.testing.assert_array_equal(f.predict(oldX,TIMES),z['cif'])
                reused.append(dict(year=year,arm=arm,seed=seed,path=str(folder),forest_sha256=sha(folder/'forest_PRIVATE.json'),prediction_sha256=sha(folder/'predictions_PRIVATE.npz'),functions_sha256=sha(folder/'functions_PRIVATE.npz')))
        for seed in SEEDS:
            a=pairing(tr['groups'],reference_configuration(seed,'shallow_parent'),83);b=pairing(tr['groups'],configuration(seed),20)
            for j,(x,y) in enumerate(zip(a,b)):
                assert x['bootstrap_sha256']==y['bootstrap_sha256'] and x['multiplicity_sha256']==y['multiplicity_sha256']
                paired.append(dict(year=year,seed=seed,tree=j,bootstrap_sha256=x['bootstrap_sha256'],multiplicity_sha256=x['multiplicity_sha256'],paper_eligible=20,reference_eligible=25))
        before=shared_quantile_cuts(tr['X'],5);after=shared_quantile_cuts(transform(tr,names)['X'],5)
        for j,k in enumerate(idx):
            if PAPER_IDS[j]!=53:np.testing.assert_array_equal(before[k],after[j])
        audits.append(dict(year=year,fit_information_before=str(cutoff.date()),training_companies=len(tr['X']),training_terminations=int((tr['event']==1).sum()),
            training_recoveries=int((tr['event']==2).sum()),evaluation_companies=len(ev['X']),paper_listed=54,paper_available_columns=20,
            paper_unmapped=34,known_duplicate_pair='X1/X2',source_vector_values_verified=True,pre_origin_clock_checks_passed=True))
        for k,v in asdict(reference_configuration(42,'shallow_parent')).items():
            w=asdict(configuration(42))[k];differences.append(dict(year=year,parameter=k,reference=json.dumps(v),paper=json.dumps(w),changed=v!=w))
        differences.extend([dict(year=year,parameter=k,reference=a,paper=b,changed=a!=b) for k,a,b in
            [('input_columns','83','20'),('market_value_scale','natural log','original thousand CNY'),('shrinkage_strength','10','10'),('training_cap_calendar_months','24','24'),('evaluation_calendar_months','12','12')]])
    assert len(reused)==18
    for n,d in [('cohort_audit',audits),('reused_artifacts',reused),('configuration_diff',differences),('tree_pairing',paired),('input_quality',missing)]:pd.DataFrame(d).to_csv(run/f'reports/{n}.csv',index=False)
    protocol=dict(name=NAME,status='FROZEN_BEFORE_FIT',source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
        user_scope='Explicit reply authorizes available subset with missing variables and definition differences recorded',
        paper=dict(title='Progressive gradient boosted trees for imbalanced financial distress prediction',doi='10.1016/j.eswa.2026.132187',table=2,page=9,listed_variables=54,sha256=sha(PAPER)),
        experiment='Available-name20 adapted input experiment, NOT complete54 reconstruction or PROBoost reproduction',years=YEARS,seeds=SEEDS,
        new_forest_fits=9,derived_shrunk_models=9,reused_reference_forests=18,pilot_fits=0,paper_ids=PAPER_IDS,
        mapping='20 paper names mapped to frozen conservative inputs; X1/X2 identical;34 unmapped; author exact formulas not available; no outcome-based selection',
        configuration=asdict(configuration(42)),reference_configuration=asdict(reference_configuration(42,'shallow_parent')),shrinkage_strength=10,
        clocks='Same annual Jan1 exclusive training cutoffs, next-trading-day pre-open company origins, original conservative data and competing labels',
        training_cap_calendar_months=24,evaluation_calendar_months=12,observation_cutoff_2025='2026-06-30',
        aggregation='Average three seed company CIFs before scoring; cause1 first final termination primary, cause2 first full recovery secondary',
        metrics=['AUC','AP','IPCW Brier','mean risk','observed incidence','Top20 capture'],top20='ceil(.20*n), same annual companies, risk descending/sample_id ties',
        uncertainty=dict(repeats=1000,seed=2026110154,unit='company',paired=True,scope='each annual cause1 result; fixed forecasts; undefined counts retained'),
        representation='q2 complete functions all train/eval rows, native times and both causes, error<=1e-10; reload and CIF checks, raw parentAJ replay',
        coverage_caveat='Effective tree features20, all available; original25/83 and25/74. Input content, candidate coverage and market scale change jointly. Do not attribute all differences to variable identity.',
        report_constraints='No direct numerical comparison to paper ST classification AUC; no full54 efficacy claim; original model unchanged',
        resources='9 independent single-thread CPU tasks in two-node Slurm/NFS,4GiB caps; no new calibration/search')
    write(repo/'experiments'/NAME/'frozen_protocol.json',protocol);write(run/'input/protocol.json',protocol)
    write(run/'audit/source_hashes.json',sources)
    write(run/'audit/data_alignment.json',dict(status='PASS_PROVISIONAL_ADAPTED_SUBSET',cohorts=audits,reference_forests_reused=18,
        limitations=['exact paper formulas and54-variable reproduction unavailable','original disclosure-vintage PIT remains inherited/provisional','20columns include duplicateX1/X2; effective candidate budget20'],
        local_registry_connector='unavailable tunnel; local authorized schema and sources read',financial_dictionary_inspected='/home/oliver/pylib/csmar/盈利能力/FI_T5[DES][xlsx].txt',
        source_table_limitation='FI_T5 Accper is report period, not publication; no fabricated availability or use of unvalidated precomputed ratios'))
    print(json.dumps(audits))


def freeze(run,repo):
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo).strip()
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    with tarfile.open(fileobj=io.BytesIO(subprocess.check_output(['git','archive',commit],cwd=repo))) as f:f.extractall(run/'input/code',filter='data')
    tasks=[dict(kind='cpu',argv=['python3',str(run/'input/code/experiments'/NAME/'launch.py'),'--year',str(y),'--seed',str(s)],
        cwd=str(run/'input/code'),timeout=3600,storage=dict(source=str(run/'input'),mode='nfs',results=str(run/'results/formal'))) for y in YEARS for s in SEEDS]
    with (run/'formal.jsonl').open('x') as f:f.write(''.join(json.dumps(x)+'\n' for x in tasks))
    (run/'formal.py').write_text('# Nine independent available-paper subset fits.\n');write(run/'formal.py.hetero.json',dict(independent=True,manifest='formal.jsonl'))
    write(run/'audit/freeze.json',dict(commit=commit,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),tasks=9,protocol_sha256=sha(run/'input/protocol.json'),manifest_sha256=sha(run/'formal.jsonl')))
    write(run/'audit/immutable_input_hashes.json',{str(p.relative_to(run/'input')):sha(p) for p in (run/'input').rglob('*') if p.is_file()})
    print(commit)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--stage',choices=['prepare','freeze'],required=True);a=p.parse_args()
    (prepare if a.stage=='prepare' else freeze)(a.run,Path(__file__).resolve().parents[2])
