from dataclasses import asdict
import datetime,shutil,subprocess
import pandas as pd
from crforest import CompetingRiskForest,aalen_johansen
from experiments.post_st_ablation_v1.verification import merge
from experiments.post_st_level_coverage_v1.affine import forest_envelope,hull
from .common import *

def main():
    block_forest_fits();assert sha(REPO/'CURRENT_POST_ST_MODEL.json')==REFERENCE_SHA
    for name in ['input','audit','reports','results','private','evaluation_input']:(RUN/name).mkdir(exist_ok=False)
    sources={};reuse=[];counts=[];source_checks=[]
    def copy(p,q):
        sources[str(p)]=sha(p);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q);assert sha(q)==sha(p)
    # Reuse the completed authoritative clocks audit, preserving its inherited limits.
    for name in ['DATA_ALIGNMENT_AUDIT_ZH.md','COHORT_AUDIT.csv','CLOCK_METADATA_RECONCILIATION.csv','FOLLOWUP_SUPPORT.csv','FEATURE_AUDIT.csv']:
        copy(REPO/'experiments/post_st_cause_support_v1'/name,RUN/'audit'/name)
    for p,h in load(CAUSE/'audit/source_hashes.json').items():assert sha(Path(p))==h
    copy(CAUSE/'audit/source_hashes.json',RUN/'audit/inherited_source_hashes.json')
    copy(CAUSE/'audit/2025_training_metadata_PRIVATE.parquet',RUN/'audit/training_metadata_PRIVATE.parquet')
    meta=pd.read_parquet(RUN/'audit/training_metadata_PRIVATE.parquet');full=arrays(PARENT/'input/outer/2025/train_PRIVATE.npz')
    np.testing.assert_array_equal(meta.sample_id,full['sample_ids'])
    def labels_at(v,y):
        sel=meta.origin.dt.year.eq(v);mm=meta.loc[sel].copy();d={k:z[sel.to_numpy()].copy() for k,z in full.items() if k!='X'}
        cap=pd.Timestamp(f'{y}-01-01')-pd.Timedelta(days=1)
        d['event'][mm.stop.gt(cap).to_numpy()]=0;stop=mm.stop.where(mm.stop<=cap,cap)
        d['time']=(stop-mm.origin).dt.days.to_numpy();d['analysis_time']=np.array([calendar_elapsed(o,s) for o,s in zip(mm.origin,stop)])
        assert (mm.origin<cap).all() and (stop<=cap).all()
        return d,mm
    models={};preds={}
    for y in range(2019,2026):
        src=PARENT/f'input/folds/{y}' if y<2025 else PARENT/'input/outer/2025'
        for name in ['train_PRIVATE.npz','eval_PRIVATE.npz']:copy(src/name,RUN/f'input/folds/{y}'/name)
        tr=arrays(src/'train_PRIVATE.npz');ev=arrays(src/'eval_PRIVATE.npz')
        assert set(ev)=={'X','sample_ids'} and tr['X'].shape[1]==83 and (tr['year']<y).all()
        # Independently rebuild each historical fit's information truncation.
        i=pd.Index(meta.sample_id).get_indexer(tr['sample_ids']);assert (i>=0).all();mm=meta.iloc[i]
        cutoff=pd.Timestamp(f'{y}-01-01');stop=mm.stop.where(mm.stop<cutoff,cutoff-pd.Timedelta(days=1));events=full['event'][i].copy();events[mm.stop.ge(cutoff).to_numpy()]=0
        np.testing.assert_array_equal(tr['X'],full['X'][i]);np.testing.assert_array_equal(tr['event'],events)
        np.testing.assert_array_equal(tr['time'],(stop-mm.origin).dt.days.to_numpy())
        refs=references(y,'reference83',REPO) if y in YEARS else {s:src/f'reference/{s}' for s in SEEDS}
        fs=[];ps=[]
        for s,path in sorted(refs.items()):
            for name in ['forest_PRIVATE.json','predictions_PRIVATE.npz']:copy(path/name,RUN/f'input/reference/{y}/{s}'/name)
            f=CompetingRiskForest.load(path/'forest_PRIVATE.json');assert asdict(f.config)==asdict(configuration(s,'shallow_parent'))
            assert f.audit()['max_path_variables']<=2
            z=arrays(path/'predictions_PRIVATE.npz');np.testing.assert_array_equal(z['sample_ids'],ev['sample_ids']);np.testing.assert_array_equal(z['times'],TIMES)
            np.testing.assert_array_equal(f.predict(ev['X'],TIMES),z['cif']);fs.append(f);ps.append(z['cif'])
            reuse.append(dict(year=y,seed=s,role='current_reference' if y in YEARS else 'historical_OOF',source=str(path),forest_sha256=sha(path/'forest_PRIVATE.json')))
        models[y]=merge(fs);preds[y]=np.mean(ps,axis=0)
        # Precompute a deterministic, all-input terminal probability envelope.
        vertices=forest_envelope(models[y]);np.savez_compressed(RUN/f'input/folds/{y}/envelope_PRIVATE.npz',vertices=vertices)
        source_checks.append(dict(year=y,training_companies=len(tr['X']),training_termination=int((tr['event']==1).sum()),training_recovery=int((tr['event']==2).sum()),
            latest_training_origin=str(mm.origin.max().date()),latest_training_stop=str(stop.max().date()),cutoff_exclusive=str(cutoff.date()),X_time_event_rebuilt_equal=True))
        if y in YEARS:
            copy(PARENT/f'evaluation_input/{y}_test_PRIVATE.npz',RUN/f'evaluation_input/{y}_test_PRIVATE.npz')
    # Prepare each chronological correction using ONLY its own pre-cutoff data.
    for h in [12,24]:
        for y in periods(h):
            labs=[];pp=[];cohort_id=[]
            for v in cohorts(y,h):
                lab,mm=labels_at(v,y)
                assert ((mm.origin+pd.DateOffset(months=h))<pd.Timestamp(f'{y}-01-01')).all()
                assert ((lab['event']>0)|(lab['analysis_time']>=h)).all() and lab['label_eligible'].all()
                ev=arrays(RUN/f'input/folds/{v}/eval_PRIVATE.npz');np.testing.assert_array_equal(lab['sample_ids'],ev['sample_ids'])
                assert not set(lab['groups'])&set(arrays(RUN/f'input/folds/{v}/train_PRIVATE.npz')['groups'])
                if y in YEARS:
                    oldlab=arrays(PARENT/f'input/folds/{v}/labels_{y}_PRIVATE.npz')
                    for k in ['sample_ids','event','time','analysis_time','h12','h24','label_eligible']:np.testing.assert_array_equal(lab[k],oldlab[k])
                labs.append(lab);pp.append(horizon_predictions(preds[v],TIMES,lab[f'h{h}']));cohort_id.extend([v]*len(lab['groups']))
            lab={k:np.concatenate([d[k] for d in labs]) for k in labs[0]};p=np.concatenate(pp);event=np.column_stack([(lab['event']==c)&(lab['analysis_time']<=h) for c in [1,2]]).astype(float)
            assert len(np.unique(lab['groups']))==len(lab['groups'])
            folder=RUN/f'input/calibration/{h}/{y}';folder.mkdir(parents=True)
            vertex=hull(np.vstack([arrays(RUN/f'input/folds/{y}/envelope_PRIVATE.npz')['vertices'],p]))
            tr=arrays(RUN/f'input/folds/{y}/train_PRIVATE.npz');cap=365. if h==12 else 730.
            grid=np.unique(np.r_[0.,np.minimum(models[y].grid_,cap),cap]);aj=aalen_johansen(tr['time'],tr['event'],grid).sum(1);assert aj[-1]>0
            np.savez_compressed(folder/'calibration_PRIVATE.npz',p=p,y=event,groups=lab['groups'],sample_ids=lab['sample_ids'],cohort=np.array(cohort_id),vertices=vertex,
                shape_grid=grid,shape_values=aj/aj[-1])
            write(folder/'cohorts.json',cohorts(y,h))
            counts.append(dict(cutoff_year=y,months=h,cohorts=','.join(map(str,cohorts(y,h))),companies=len(p),termination=int(event[:,0].sum()),recovery=int(event[:,1].sum()),
                termination_slope_fixed=bool(event[:,0].sum()<10),recovery_slope_fixed=bool(event[:,1].sum()<10),initialization_only=y not in YEARS,
                previous_parameter_year=y-1 if y>periods(h)[0] else None,no_unknown_endpoints=True))
    # Resample companies WITHIN entry cohort, reusing the same draw in every
    # period/horizon/arm in which that cohort appears. No future outcomes used.
    drawdir=RUN/'input/resamples';drawdir.mkdir()
    for rep in range(50):
        draws={}
        for v in range(2019,2024):
            n=len(arrays(RUN/f'input/folds/{v}/eval_PRIVATE.npz')['X']);rng=np.random.default_rng(np.random.SeedSequence([RESAMPLE_SEED,rep,v]))
            draws[str(v)]=rng.integers(n,size=n)
        np.savez_compressed(drawdir/f'{rep}.npz',**draws)
    protocol=dict(name=NAME,source_commit='37be0ee05ed64f31ba18c905795403aa8b9012ed',status='FROZEN_BEFORE_CALIBRATOR_FITS',
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),years=YEARS,horizons=[12,24],arms=ARMS,forest_fits=0,
        reference='83 inputs,q2,depth4,shared_quantile,balanced25,three32-tree seeds42/101/202,direct-parentAJ10;all forests saved and immutable',
        source_policy='unchanged old conservativeC;next-trading-day pre-open;firsttermination1 versusfirstrecovery2;Jan1information cutoff;24calendar fittingcap;inheritedPROVISIONAL vintage limits',
        mapping='Separate horizon-specific full curves Fnew_k(t)=c_k*g_H(t)+b_k*F_k(t); c>=0,.5<=b<=4,timeconstantslopes. No claim12month and24month maps are one jointly calibrated curve;checkendpointcompatibility separately.',
        history='latest TWO fully horizon-mature entry years before cutoff, limited to available frozen2019+OOF. Startup uses one cohort;H12 chain2021..2025,H24chain2022..2025. Same companies/OOFpredictions/windows in bothcalibration arms.',
        historical_forests='saved q2 shallowparent models fitted before each OOF entryyear;no in-sample calibration,no forest refits',
        independent='joint two-cause squaredloss plus20/n times mean squared probability change towardidentity;equivalent to previous level_coverage_v1 control',
        temporal='same objective PLUS20/n times [(c-cprev)^2 summed + sum(s_k^2*(b-bprev)^2)];s_k=max(.05,RMS currenthistoricalOOFprobability). Previous comesONLYfrom preceding temporal calibrator, samehorizon. Firstperiod has no temporal penalty. No tuning.',
        sparse='fewerthan10 DISTINCT observed cause-event companies in calibration sample fixes that cause slopeexactly1,including companyresampling;repeated rows do not manufacture support. Operational gate,notpower guarantee.',
        dependency='each chain solved chronological;later outcomes never alter earlier parameter states. Shared overlapping windows do not create independent evidence;temporal penalty may delay adaptation to real shifts.',
        probability='same all-input terminal Minkowski convexhull certificate as prior experiment,plusOOFpoints. Nonnegative monotone g_H starts0,reaches1at365/730andstays1;positiveb. Two-cause sum<=1on allcertificatevertices. No clipping or nonlinear map.',
        optimization='SLSQP,activeKKTpolish,independentlinear-oraclegap<=1e-8,primal<=1e-10;alliterations/failuresrecorded;parametersprivate',
        formal_point_calibrators=18,point_outer_calibrators=12,point_initialization_calibrators=6,
        stability=dict(company_resamples=50,stratify='entrycohort',seed=RESAMPLE_SEED,paired=True,calibrator_fits=900,
            meaning='re-estimate complete chronological correction chains;forest/OOFpredictions fixed. Notforest-trainingorfullpipelineuncertainty. Same draws acrossoverlappingwindows/horizons/arms. No support-gate manipulation.'),
        workers='2 point horizon chains+100 resampled chains;point tasks export12 outer calibrators,alltrain/evalrows/allnativeandshape times/bothcauses,completeq2sum/reload<=1e-10. Two-nodeSlurm singlethreadCPU,NFS4GiBcap. No syntheticforestfitpilot;unit tests beforefreeze.',
        metrics='ensemble-first;annual/pooledAUC,AP,IPCW Brier,meanrisk,observedincidence,annualTop20ceilquota,calibrationquartiles;bothcauses12/24;2025H24performanceNA but mapping/functionchecked. Report originalpredictionperseed and fixedensemblecalibrators,no separateperseedfit.',
        uncertainty=dict(test_repeats=1000,seed=ANALYSIS_SEED,unit='company',stratify='testyear',paired=True,conditional_on='fixedforestandpointcalibrators;doesnotincludefulltraining/adaptivemethoduncertainty'),
        crossperiod='report actual parameter changes plus common-probability-probe correction changes;50paired calibrationresamples assess estimatespread,no assumed improvement',
        preserve='CURRENT_POST_ST_MODEL.json,original83/74models andoldreports unchanged;noautomaticpromotion;all exposed2023-25evaluation exploratory')
    public=REPO/'experiments'/NAME
    write(public/'frozen_protocol.json',protocol);write(RUN/'input/protocol.json',protocol);write(RUN/'audit/source_hashes.json',sources)
    write(public/'run_location.json',dict(private_run=str(RUN)))
    for name,rows in [('CALIBRATION_SUPPORT',counts),('MODEL_REUSE',reuse),('INFORMATION_CUTOFF_AUDIT',source_checks)]:
        pd.DataFrame(rows).to_csv(RUN/f'reports/{name}.csv',index=False);shutil.copy2(RUN/f'reports/{name}.csv',public/f'{name}.csv')
    print(pd.DataFrame(counts).to_string(index=False))

if __name__=='__main__':main()
