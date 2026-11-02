import datetime,shutil
import pandas as pd
from .common import *

def main():
    assert sha(REPO/'CURRENT_POST_ST_MODEL.json')==REFERENCE_SHA;block_forest_fits();RUN.mkdir(exist_ok=False)
    for n in ['input','audit','results','reports','private','evaluation_input']:(RUN/n).mkdir()
    sources={}
    def copy(p,q):
        sources[str(p)]=sha(p);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q);assert sha(q)==sources[str(p)]
    for name in ['source_hashes','inherited_source_hashes']:
        d=load(SOURCE/f'audit/{name}.json')
        for p,h in d.items():assert sha(Path(p))==h
        copy(SOURCE/f'audit/{name}.json',RUN/f'audit/{name}_prior.json')
    support=[]
    for y in range(2019,2026):
        copy(SOURCE/f'input/folds/{y}/train_PRIVATE.npz',RUN/f'input/folds/{y}/train_PRIVATE.npz')
        src=SOURCE/f'input/folds/{y}/eval_PRIVATE.npz';sources[str(src)]=sha(src);ev=arrays(src)
        src=PARENT/f'input/folds/{y}/labels_2025_PRIVATE.npz' if y<2024 else SOURCE/f'evaluation_input/{y}_test_PRIVATE.npz'
        sources[str(src)]=sha(src);metadata=arrays(src);np.testing.assert_array_equal(ev['sample_ids'],metadata['sample_ids'])
        np.savez_compressed(RUN/f'input/folds/{y}/eval_PRIVATE.npz',**ev,h12=metadata['h12'],h24=metadata['h24'])
        for seed in SEEDS:
            for name in ['forest_PRIVATE.json','predictions_PRIVATE.npz']:copy(SOURCE/f'input/reference/{y}/{seed}'/name,RUN/f'input/reference/{y}/{seed}'/name)
    for y in YEARS:
        copy(SOURCE/f'evaluation_input/{y}_test_PRIVATE.npz',RUN/f'evaluation_input/{y}_test_PRIVATE.npz')
        for h in [12,24]:
            src=SOURCE/f'input/calibration/{h}/{y}/calibration_PRIVATE.npz';sources[str(src)]=sha(src);d=arrays(src)
            folder=RUN/f'input/history/{y}/{h}';folder.mkdir(parents=True)
            np.savez_compressed(folder/'membership_PRIVATE.npz',sample_ids=d['sample_ids'],cohort=d['cohort'])
            np.savez_compressed(RUN/f'private/history_labels_{y}_{h}_PRIVATE.npz',sample_ids=d['sample_ids'],y=d['y'])
            assert set(np.unique(d['cohort']))==set(cohorts(y,h))
            support.append(dict(cutoff=y,months=h,entry_cohorts=','.join(map(str,cohorts(y,h))),companies=len(d['y']),target_events=int(d['y'][:,0].sum()),recovery_events=int(d['y'][:,1].sum()),all_endpoints_mature=True,labels_used_to_fit_gate=False))
    protocol=dict(name=NAME,status='FROZEN_BEFORE_NEW_DIAGNOSTICS_AND_SELECTION',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        reference='unchanged83/q2/depth4/shared/balanced25/3x32/direct-parentAJ10; no rescaling or forest/calibrator fit',
        source_commit='e78d287987ed468065bfc84a37098e58b59692ab',source_run=str(SOURCE),new_forests=0,new_calibrators=0,diagnostic_tasks=7,models_reused=21,
        years=YEARS,horizons=[12,24],causes=[1,2],coverage_targets=COVERAGES,primary_coverage=.8,
        history='latest two complete horizon-mature OOF entry cohorts before Jan1, same membership as frozen temporal_affine experiment',
        raw_features=RAW,definitions=dict(seed_sd='SD of3 saved seed endpointCIFs, algorithmic proxy not total uncertainty',
            function_sparse_share='fraction of absolute main+pair contributions lying in empirical JOINT cells with<10 distinct training companies;missingseparate;zero denominator=>0',
            leaf_company_penalty='mean over96 reached raw leaf memberships of1/(1+uniqueinbagcompanies)',
            leaf_cause_penalty='mean1/(1+distinct causeevents observedbyH) over96 reached leaves;parent borrowing adds no observed event',
            leaf_followup_penalty='mean fraction of reached inbag companies whose H-endpoint was unknown before forestfitcutoff'),
        arms=dict(combined='equal mean of5 historical midrank ECDF badness scores',support='equal mean of4 support ECDFs, excludingseedSD',seed_sd='raw seedSD only',low_risk_control='accept lowestpredictedrisk; case-mix negativecontrol, NOT reliability claim'),
        threshold='ceil(q*N_history)-th ordered (score,SHA256companyID tiekey); stored before outer labels read. Apply unchanged to futurecompany individually. Realizedcoverage maydiffer; empty acceptance staysNA.',
        no_supervised_gate_weights=True,no_error_guarantee=True,tie_seed=TIE_SEED,
        matched_analysis='secondary, same yearly acceptedcount as combined for each q; rank seed/support/lowrisk in testbatch without labels. Matchedrandom1000 permutations shared across rules and budgets. Thisbatch comparison is not the primary prospective gate.',
        primary_metrics='accepted/referred counts, Brier and calibration, AUC/AP, accepted target fraction among all targetevents, referred eventcounts, case-specific errors, balancedMSE, same originalceil20% budget within accepted;never assume humanreview succeeds',
        evaluation='full-cohort rawCIF baseline always reported; all five supportedyear-horizons complete endpoints (IPCW=1),2025H24 outcomesNA; both causes separately',
        case_mix='exact symmetric decomposition of accepted-minus-full Brier into event-prevalence mix and within-event-status loss differences;NA ifoneclassmissing',
        uncertainty=dict(test_bootstrap=1000,seed=BOOT_SEED,unit='company,stratifiedyear,fixedpredictionsandgates',history_bootstrap=50,history_seed=HISTORY_SEED,
            history_scope='resampleOOFcompanieswithineachentrycohort;recomputeECDFsandthresholds,notforests;paired acrossgates/horizons with identicalcohortmembership'),
        random_seed=RANDOM_SEED,full_representation='everytrain/evalrow/allnative+displaytimes/twocauses,error<=1e-10; preserve rejected predictions, never encode abstentionaszero',
        metadata='PROVISIONAL originalvintages;previouslyexposed2023–2025 exploration; no prospective independentvalidation',
        resources='7independentCPUdiagnostic jobs,two-nodeSlurm,NFS,4GiBcap;remaining small fixed-output analyses local;noGPUornewtraining',
        privacy='private IDs,thresholds,reliabilityvalues,companymasks,functioncoefficients stayserver;publicaggregates/code',adoption=False)
    write(RUN/'input/protocol.json',protocol);write(REPO/f'experiments/{NAME}/frozen_protocol.json',protocol)
    pd.DataFrame(support).to_csv(REPO/f'experiments/{NAME}/HISTORICAL_SUPPORT.csv',index=False)
    write(RUN/'audit/source_hashes.json',sources);write(REPO/f'experiments/{NAME}/run_location.json',dict(run=str(RUN)))
    print(pd.DataFrame(support).to_string(index=False))

if __name__=='__main__':main()
