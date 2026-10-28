"""Publish only aggregate diagnostics, suppressing recoverable small-cell values."""
from pathlib import Path
import argparse
import json
import platform
import re
import shutil
import numpy as np
import pandas as pd
import sklearn
import matplotlib
from experiments.post_st_prognosis_v1.common import sha,write


def deliver(run):
    here=Path(__file__).parent
    checks=json.loads((run/'results/independent_validation.json').read_text())
    assert checks['status']=='PASS' and checks['independent_OOB_forests_checked']==24
    text=(run/'audit/tests_final.log').read_text()
    assert re.search(r'111 passed, 1 warning',text) and 'failed' not in text.lower()
    target=here/'results';target.mkdir(exist_ok=False)
    publication=[]
    for p in sorted((run/'results').iterdir()):
        dest=target/p.name
        if p.suffix=='.csv':
            frame=pd.read_csv(p,dtype={'scope':str,'seed':str})
            forbidden={'sample_id','group','code','source_text','source_ids','origin','stop','threshold'}
            assert not forbidden.intersection(frame.columns)
            frame=frame.drop(columns=['risk_min','risk_max'],errors='ignore')
            sensitive=[];small=np.zeros(len(frame),bool)
            if p.name=='event_error_segments.csv':
                small=frame.cases<3
                sensitive=['mean_case_risk','mean_case_squared_error','loss_contribution_to_full_cohort',
                           'auc_cases_vs_original_nontermination','mean_reached_zero_target_leaf_fraction']
            elif p.name=='entry_information_subgroups.csv':
                small=frame.n<3;sensitive=['risk_mean','observed','brier','auc']
            if sensitive:
                frame.loc[small,sensitive]=np.nan
                frame['small_cell_detail_suppressed']=small
                assert frame.loc[small,sensitive].isna().all().all()
            frame.to_csv(dest,index=False)
            publication.append(dict(file=p.name,rows=len(frame),suppressed_rows=int(np.sum(small)),
                                    sensitive_columns=sensitive,private_original_sha256=sha(p),public_sha256=sha(dest)))
        elif p.name=='independent_validation.json':
            data=json.loads(p.read_text())
            data['validated_private_artifact_sha256']=data.pop('sha256')
            data['public_tables']='See publication_manifest.json; small-cell values suppressed and risk extrema omitted'
            write(dest,data)
        elif p.suffix=='.svg':
            # Matplotlib emits harmless trailing spaces in path data. Normalize
            # formatting only; path token values and the rendered figure agree.
            dest.write_text('\n'.join(line.rstrip() for line in p.read_text().splitlines())+'\n')
        else:
            assert p.suffix in ['.json','.png','.svg'];shutil.copy2(p,dest)
    shutil.copy2(run/'audit/oob_supplement.json',target/'oob_supplement.json')
    shutil.copy2(run/'audit/tests_final.log',target/'full_test.log')
    write(target/'publication_manifest.json',dict(policy='Suppress case/subgroup n<3 numeric risk, loss, ranking and leaf-support summaries; retain counts and fixed-budget captures for every variant/seed. Individual forecasts remain private.',
        no_model_parameters_or_company_rows_exported=True,files=publication))
    write(target/'environment.json',dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,
                                        sklearn=sklearn.__version__,matplotlib=matplotlib.__version__,
                                        layout='Small read-only CPU calculations on oliver; no new fit, GPU or duplicated two-node task'))
    # Frozen source artifacts are checked once more immediately before delivery.
    sources=json.loads((run/'audit/source_hashes.json').read_text())
    for name,digest in sources.items():assert sha(name)==digest
    receipt=dict(experiment='post_st_error_attribution_v1',status='COMPLETE_NO_NEW_TRAINING',decision='INCONCLUSIVE',
        decision_scope='Independent causal contribution of policy, sampling, input omissions and search restrictions remains unidentified',
        branch='research/xai-rsf-post-st-prognosis-v1',base_commit=json.loads((run/'audit/protocol.json').read_text())['base_commit'],
        source_fitting_commits={'q2_weight_fits':'2cf6934aab69c3a29276753944150bb6c5bd873b',
                               'reference_2023':'db8e154d217d12faeda03d9d9f83e0e1195ab460',
                               'reference_2024':'ce9f5b0f26078dc6df75ec4e7bf82d89f23fabc9'},
        delivery_revision='Commit containing this receipt; full remote SHA recorded separately after push',
        private_run=str(run),new_research_fits=0,new_calibration_fits=0,existing_forests_referenced=30,
        q2_complete_explanation_models_checked=18,q2_trees_checked=576,oob_python_forests=24,oob_trees=768,
        train_companies_by_fold={'2023':309,'2024':345},train_termination_by_fold={'2023':14,'2024':26},
        unique_test_companies=81,test_terminations=31,unresolved_test_terminal_ground=1,
        default_test_auc={'2023':.64,'2024':.56},
        historical_negative_prediction_decision_unchanged=True,
        failed_calculations=0,tests={'passed':111,'failed':0,'skipped':0,'warnings':1,'independent_R_checks':True},
        warnings='Existing EBM missing-value visualization warning; no feature values changed',
        no_2025_entry_test=True,no_original_artifact_changes=True,private_evidence_exported=False,
        original_source_files_verified=len(sources),source_manifest_sha256=sha(run/'audit/source_hashes.json'),
        max_complete_reconstruction_error=json.loads((run/'results/calculation_receipt.json').read_text())['max_function_reconstruction_error'],
        max_independent_oob_replay_error=checks['independent_OOB_prediction_max_error'],
        public_file_sha256={str(p.relative_to(here)):sha(p) for p in sorted(here.rglob('*')) if p.is_file() and '__pycache__' not in str(p)})
    oob=pd.read_csv(run/'results/oob_metrics.csv',dtype={'seed':str})
    receipt['default_oob_auc']={str(y):float(oob.query("test_year==@y and model=='equal' and seed=='ensemble' and cause=='termination'").auc.iloc[0]) for y in [2023,2024]}
    write(here/'execution_receipt.json',receipt);write(run/'execution_receipt.json',receipt)
    print(json.dumps({k:receipt[k] for k in ['status','decision','new_research_fits','tests','default_oob_auc']},ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True,type=Path);deliver(parser.parse_args().run)
