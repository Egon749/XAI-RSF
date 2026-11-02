"""Prepare immutable inputs, freeze code, and monitor paired two-node jobs."""
from dataclasses import asdict
from pathlib import Path
import argparse,datetime,io,json,shutil,subprocess,tarfile
import numpy as np
import pandas as pd
from crforest import CompetingRiskForest
from experiments.post_st_paper_features_v1.common import references,history
from experiments.post_st_drop9_transport_v1.common import horizon_support
from .common import *

def prepare():
    assert not (RUN/'input/protocol.json').exists()
    public=REPO/'experiments'/NAME;source_hashes={};cohorts=[];support=[];reused=[];diffs=[];features=[];clock_audit=[]
    def copy(src,dest):
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
        source_hashes[str(src)]=sha(src);assert sha(src)==sha(dest)
    copy(PARENT/'input/features.json',RUN/'input/features.json')
    # Metadata retains some pre-origin-shift report descriptors. The field-level
    # information alignment, produced with the ACTUAL refreshed X, is authoritative.
    clock_root=history(2025)
    for name in ['information_alignment_PRIVATE.parquet','financial_availability_PRIVATE.npz']:
        copy(clock_root/'private'/name,RUN/'audit'/name)
    info=pd.read_parquet(RUN/'audit/information_alignment_PRIVATE.parquet')
    clock=arrays(RUN/'audit/financial_availability_PRIVATE.npz')
    np.testing.assert_array_equal(clock['sample_ids'],info.sample_id)
    allparts=[arrays(clock_root/f'input/folds/2025/{role}_PRIVATE.npz') for role in ['train','eval']]
    allids=np.concatenate([p['sample_ids'] for p in allparts]);allX=np.vstack([p['X'] for p in allparts])
    for p in [clock_root/f'input/folds/2025/{r}_PRIVATE.npz' for r in ['train','eval']]:source_hashes[str(p)]=sha(p)
    for role in ['financial_available','market_available','available_at']:
        assert (info[role].isna()|info[role].lt(info.origin)).all()
    assert not ((~np.isnat(clock['dates']))&(clock['dates']>=info.origin.to_numpy(dtype='datetime64[D]')[:,None])).any()
    X=allX[pd.Index(allids).get_indexer(info.sample_id)]
    assert (np.isfinite(X[:,:48])<=~np.isnat(clock['dates'])).all()
    derivative_columns=[i for i,r in enumerate(load(RUN/'input/features.json')[:48]) if r['feature_name'].endswith(('_growth','_change'))]
    blocked=info.previous_component_available.isna()|info.previous_component_available.ge(info.origin)
    assert not np.isfinite(X[blocked][:,derivative_columns]).any()
    assert np.isnat(clock['dates'][blocked][:,derivative_columns]).all()
    for j in derivative_columns:
        finite=np.isfinite(X[:,j]);expected=info[['financial_available','previous_component_available']].max(axis=1).to_numpy(dtype='datetime64[D]')
        np.testing.assert_array_equal(clock['dates'][finite,j],expected[finite])
    for year in YEARS:
        src=PARENT/f'input/outer/{year}';dest=RUN/f'input/folds/{year}'
        for name in ['train_PRIVATE.npz','eval_PRIVATE.npz']:copy(src/name,dest/name)
        copy(PARENT/f'evaluation_input/{year}_test_PRIVATE.npz',RUN/f'evaluation_input/{year}_test_PRIVATE.npz')
        copy(META/f'{year}_metadata_PRIVATE.parquet',RUN/f'audit/{year}_training_metadata_PRIVATE.parquet')
        tr=arrays(dest/'train_PRIVATE.npz');ev=arrays(dest/'eval_PRIVATE.npz');lab=arrays(RUN/f'evaluation_input/{year}_test_PRIVATE.npz')
        meta=pd.read_parquet(META/f'{year}_metadata_PRIVATE.parquet');cut=pd.Timestamp(f'{year}-01-01')
        np.testing.assert_array_equal(meta.sample_id,tr['sample_ids']);np.testing.assert_array_equal(ev['X'],lab['X'])
        np.testing.assert_array_equal(ev['sample_ids'],lab['sample_ids'])
        assert (meta.origin<cut).all() and (meta.stop<cut).all() and (tr['year']<year).all()
        assert not set(tr['groups'])&set(lab['groups'])
        original=arrays(history(year)/f'input/folds/{year}/train_PRIVATE.npz')
        for k in ['X','time','event','groups','sample_ids']:np.testing.assert_array_equal(tr[k],original[k])
        for role,data in [('train',tr),('evaluation',ev)]:
            ix=pd.Index(info.sample_id).get_indexer(data['sample_ids']);assert (ix>=0).all()
            np.testing.assert_array_equal(data['X'],X[ix])
        ix=pd.Index(info.sample_id).get_indexer(meta.sample_id)
        np.testing.assert_array_equal(meta.origin.to_numpy(),info.origin.iloc[ix].to_numpy())
        for oldkey,newkey in [('financial_available','financial_available'),('prev_available','previous_component_available'),('market_available','market_available')]:
            a=pd.to_datetime(meta[oldkey]).reset_index(drop=True);b=info[newkey].iloc[ix].reset_index(drop=True)
            changed=~(a.eq(b)|(a.isna()&b.isna()))
            clock_audit.append(dict(year=year,metadata_field=oldkey,authoritative_field=newkey,
                stale_metadata_rows=int(changed.sum()),stale_metadata_after_origin=int((a>meta.origin.reset_index(drop=True)).sum()),
                authoritative_after_or_at_origin=int((b>=meta.origin.reset_index(drop=True)).sum()),
                actual_X_matches_authoritative=True,finite_financial_features_with_missing_clock=0))
        assert len(np.unique(tr['groups']))==len(tr['X']) and len(np.unique(lab['groups']))==len(lab['X'])
        assert (meta.origin>pd.to_datetime(meta.entry_announcement)).all()
        cohorts.append(dict(year=year,training_companies=len(tr['X']),training_termination=int((tr['event']==1).sum()),
            training_recovery=int((tr['event']==2).sum()),evaluation_companies=len(ev['X']),features=83,
            fit_information_before=str(cut.date()),latest_training_stop=str(meta.stop.max().date()),
            train_eval_company_overlap=0,input_identical_to_reference=True,next_open_origin_preserved=True,
            train_sha256=sha(dest/'train_PRIVATE.npz'),eval_X_sha256=sha(dest/'eval_PRIVATE.npz')))
        for h in [12,24]:support.append(dict(year=year,**horizon_support(lab,h)))
        for j,name in enumerate(load(RUN/'input/features.json')):
            features.append(dict(year=year,**name,training_missing=int(np.isnan(tr['X'][:,j]).sum()),evaluation_missing=int(np.isnan(ev['X'][:,j]).sum())))
        refs=references(year,'reference83',REPO)
        for seed,path in refs.items():
            old=CompetingRiskForest.load(path/'forest_PRIVATE.json')
            assert asdict(old.config)==asdict(configuration(seed,'baseline'))
            for name in ['forest_PRIVATE.json','predictions_PRIVATE.npz']:
                copy(path/name,RUN/f'input/reference/{year}/{seed}'/name)
            reused.append(dict(year=year,seed=seed,source=str(path),forest_sha256=sha(path/'forest_PRIVATE.json'),purpose='comparison only; baseline is refitted separately'))
    for arm in ARMS:
        baseline=asdict(configuration(42,'baseline'));candidate=asdict(configuration(42,arm))
        changed={k for k in baseline if baseline[k]!=candidate[k]}
        expected=({'split_score'} if 'quadratic' in arm else set())|({'split_min_target_groups'} if 'support' in arm else set())
        assert changed==expected
        for k,v in candidate.items():diffs.append(dict(arm=arm,parameter=k,baseline=json.dumps(baseline[k]),value=json.dumps(v),changed=k in changed))
    protocol=dict(name=NAME,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_commit='51e58f011dae19caba932659d98209a8658a0158',
        status='FROZEN_BEFORE_NEW_REAL_FITS',private_run=str(RUN),parent_private_run=str(PARENT),years=YEARS,seeds=SEEDS,
        arms={a:asdict(configuration(42,a)) for a in ARMS},primary_models='each arm after direct-parent AJ shrinkage10',
        secondary_models='raw same fitted forests, no additional fits',formal_raw_forests=36,baseline_reproductions=9,new_design_forests=27,
        pilot_raw_forests=2,pilot_design='2023 baseline seed42 and2025 quadratic_support seed42,4trees each; numerical/runtime only,never select by outer performance',
        score='Default abs(sum(w*U))/sqrt(sum(w^2*V)). Quadratic sqrt(sum(w^2*(U^2/V))/sum(w^2)); V<=1e-18 contributes0. Allweights1,1. This changes per-cause normalization AND combination,not an isolated cancellation-only effect; heuristic search score,not chi-square inference.',
        guard='After existing row/group/total-event guards: each child must contain either0 or at least2 DISTINCT companies with cause1. A positive child supported by exactly1 company is rejected. Both zero-event children are allowed. If no legal positive-score candidate remains,stop at a leaf. No required events in both daughters,no replacement by test labels,no artificial events.',
        guard_limit='May prevent true rare-subgroup splits; protects minimum positive support,not statistical reliability. Bootstrap repetitions never increase unique-company count. Root may lack target support. Zero-target leaves remain possible.',
        leaf='Raw AJ and same-bootstrap raw DIRECTparent AJ withalpha=uniqueleafcompanies/(uniqueleafcompanies+10),common across bothcauses/alltimes;rootleaf unchanged. No timeweighting,oversampling,calibration or probability clipping.',
        data='Exact frozen conservativeC83,Jan1cutoffs2023/24/25,first financial/auditST entry,next exchange trading-day pre-open,firstfinaltermination1 versusfirstrecovery2. Followup capped24calendar months andstrictlybeforecutoff. Financial-vintage PIT limitations inherited;no annual-source override.',
        test_labels='Stored outside worker input. Only same evalX/IDs on workers. Labels used after all configurations/fitsfixed.2023-2025previouslyexposed,exploratory.',
        pairing='Same companybootstrap indices/multiplicities,original83balanced25feature pools,sharedcutdictionary and node RNG. No new RNG consumption. Path/child correspondence not required after structure changes.',
        validation='All train+evalrows,all native times+0/365/366/730/731,2causes. q<=2,nonnegative,monotone,sum<=1. Fullbaseline+allmains+allpairs and reload error<=1e-10,no clipping. Replay all candidates/guards; independent selectedscore components and raw parentAJ. Baseline9 forests must match savedtrees/predictionsexactly.',
        metrics='Annual+pooled AUC/AP/IPCW Brier,meanrisk,IPCW observedincidence,Top20 ceil(.2*n) peryear then sum,calibrationbins. Mean3seed predictions before primary metrics; everyseed andrawsaved. Bothcauses,12/24months;unsupportedNA withreason.',
        primary='24month termination2023/24;2025 supported12month termination. Otherhorizons/recovery secondary,all reported. No performance threshold to drop arms.',
        unsupported='2025H24no horizonfollowupsupport:retainpredictions/supportcounts,do notcalculateperformance orcalibration;pooledH24uses2023/24only. Noobservedpositives meansAUC/APNA.',
        uncertainty=dict(repeats=1000,seed=ANALYSIS_SEED,unit='company',stratify='testyear',paired=True,
            scope='each supported horizon/cause,annualandpooled;fixedmodels,descriptive95percentile,differencesvsbaseline;undefineddrawcounts. Nottraining/selectionuncertainty.'),
        interpretation='Comparequadratic-baseline,support-baseline,both-baseline;also(both-quadratic)-(support-baseline)factorialinteraction. No automatic reference replacement.',
        complexity='Nonzeromains/pairs per existing all-cell/native-time test;structuralspatialcells includingmissingstates;scalarcells separately;ensemble96tree representationnotmeanseedcounts.',
        stability='Three-seed predictionSD andmain/pairJaccard,commonfunctioncorrelation/RMS on same annualtraining reference at365/730days,bothcauses. Algorithmicrandomness,notindependentcohorts.',
        resources='36 independent singlethreadCPU fits+2fourtree pilots,existingtwo-nodeSlurm,NFS,4GiB perprocess,timeout3600;noGPU,hardcodedCPUaffinity orautomaticextra search.',
        preserve='CURRENT_POST_ST_MODEL.json andall83/74historicalmodels unchanged;companydata/predictions/thresholds/parameters remainprivate.')
    write(public/'frozen_protocol.json',protocol);write(RUN/'input/protocol.json',protocol)
    write(public/'run_location.json',dict(private_run=str(RUN)));write(RUN/'audit/source_hashes.json',source_hashes)
    write(RUN/'audit/current_pointer.json',dict(path=str(REPO/'CURRENT_POST_ST_MODEL.json'),sha256=sha(REPO/'CURRENT_POST_ST_MODEL.json')))
    for name,rows in [('COHORT_AUDIT',cohorts),('FOLLOWUP_SUPPORT',support),('ARTIFACT_REFERENCES',reused),('PARAMETER_DIFFERENCES',diffs),('FEATURE_AUDIT',features),('CLOCK_METADATA_RECONCILIATION',clock_audit)]:
        pd.DataFrame(rows).to_csv(RUN/f'reports/{name}.csv',index=False)
        shutil.copy2(RUN/f'reports/{name}.csv',public/f'{name}.csv')
    print(json.dumps(cohorts,ensure_ascii=False))

def freeze():
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=REPO).strip()
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    tracked=subprocess.check_output(['git','ls-files'],cwd=REPO,text=True).splitlines();files=[p for p in tracked if p.endswith('.py')]
    with tarfile.open(fileobj=io.BytesIO(subprocess.check_output(['git','archive',commit,'--',*files],cwd=REPO))) as t:t.extractall(RUN/'input/code',filter='data')
    write(RUN/'audit/code_snapshot.json',dict(commit=commit,protocol_sha256=sha(RUN/'input/protocol.json')))
    write(RUN/'audit/immutable_input_hashes.json',{str(p.relative_to(RUN/'input')):sha(p) for p in (RUN/'input').rglob('*') if p.is_file()})

def freeze_analysis():
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=REPO).strip()
    dest=RUN/'analysis_input';dest.mkdir(exist_ok=False)
    shutil.copytree(RUN/'input/folds',dest/'folds');shutil.copy2(RUN/'audit/registry_formal.json',dest/'registry_formal.json')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    files=[p for p in subprocess.check_output(['git','ls-files'],cwd=REPO,text=True).splitlines() if p.endswith('.py')]
    with tarfile.open(fileobj=io.BytesIO(subprocess.check_output(['git','archive',commit,'--',*files],cwd=REPO))) as t:t.extractall(dest/'code',filter='data')
    write(RUN/'audit/analysis_code_snapshot.json',dict(commit=commit))
    write(RUN/'audit/immutable_analysis_hashes.json',{str(p.relative_to(dest)):sha(p) for p in dest.rglob('*') if p.is_file()})

def execute(stage):
    items=([dict(year=2023,seed=42,arm='baseline',trees=4),dict(year=2025,seed=42,arm='quadratic_support',trees=4)] if stage=='pilot'
           else [dict(year=y,seed=s,arm=a,trees=32) for y in YEARS for a in ARMS for s in SEEDS])
    if stage=='formal':assert load(RUN/'audit/gate_pilot.json')['status']=='PASS'
    if stage=='ensemble':
        assert load(RUN/'audit/gate_formal.json')['status']=='PASS'
        items=[dict(year=y,arm=a,version=v) for y in YEARS for a in ARMS for v in ['raw','shrunk']]
    input_root=RUN/('analysis_input' if stage=='ensemble' else 'input')
    tasks=[]
    for r in items:
        args=['--ensemble','--version',r['version']] if stage=='ensemble' else ['--seed',str(r['seed']),'--trees',str(r['trees'])]
        tasks.append(dict(kind='cpu',argv=['python3',str(input_root/f'code/experiments/{NAME}/launch.py'),
            '--year',str(r['year']),'--arm',r['arm'],*args],cwd=str(input_root/'code'),timeout=3600,
            storage=dict(source=str(input_root),mode='nfs',results=str(RUN/f'results/{stage}'))))
    (RUN/f'{stage}.jsonl').write_text(''.join(json.dumps(t)+'\n' for t in tasks))
    (RUN/f'{stage}.py').write_text('# Independent paired annual seed forests.\n')
    write(RUN/f'{stage}.py.hetero.json',dict(independent=True,manifest=f'{stage}.jsonl'))
    assert not (RUN/f'audit/launcher_{stage}.log').exists()
    receipt=None
    with (RUN/f'audit/launcher_{stage}.log').open('x') as log:
        p=subprocess.Popen(['/home/oliver/.local/bin/hetero-run',str(RUN/f'{stage}.py')],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        for line in p.stdout:
            log.write(line);log.flush();print(line,end='',flush=True)
            try:
                obj=json.loads(line)
                if isinstance(obj,dict) and 'job' in obj and 'result' in obj:receipt=obj
            except json.JSONDecodeError:pass
        status=p.wait()
    if status or receipt is None or not receipt['success']:raise RuntimeError('Retain failed compute/publication evidence; do not automatically refit')
    report=load(receipt['result']);assert report['expected']==len(items) and not report['pending']
    records=[]
    for task in report['completed']:
        assert task['returncode']==0 and task['compute_completed'] and task['publication_completed']
        index=int(task['id']);folder=RUN/f'results/{stage}/{receipt["job"]}/task-{index}/output'
        for name,h in load(folder/'hashes.json').items():assert sha(folder/name)==h
        result=load(folder/'summary.json');assert result['status']=='COMPLETE'
        records.append(dict(**items[index],path=str(folder),host=result['host'],job=receipt['job'],task=index))
    assert len(records)==len(items)
    write(RUN/f'audit/registry_{stage}.json',records)
    write(RUN/f'audit/gate_{stage}.json',dict(status='PASS',job=receipt['job'],tasks=len(items),seconds=receipt['execution_seconds'],report=receipt['result'],report_sha256=sha(Path(receipt['result']))))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','freeze','pilot','formal','freeze_analysis','ensemble']);a=p.parse_args()
    {'prepare':prepare,'freeze':freeze,'pilot':lambda:execute('pilot'),'formal':lambda:execute('formal'),
        'freeze_analysis':freeze_analysis,'ensemble':lambda:execute('ensemble')}[a.stage]()
