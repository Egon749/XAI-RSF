"""Post-run immutable-input, result, test and publication audit; no fitting."""
from pathlib import Path
import argparse,json,os,re,subprocess,xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from experiments.post_st_reporting_drift_v1.analyze import digest

PUBLIC=Path(__file__).resolve().parent;REPO=PUBLIC.parents[1]
SOURCE='fcf98aac4c68ac4ac1cadac4227262e0937e94a3'


def audit(run,test_xml=None):
    total=0
    for filename,base in [('input_sources_PRIVATE.json',None),('stage_input_hashes.json',run/'input'),('immutable_A_input_hashes.json',run/'input'),('immutable_B_input_hashes.json',run/'inputB')]:
        for name,h in json.loads((run/filename).read_text()).items():
            p=Path(name) if base is None else base/name
            assert digest(p)==h,f'Changed input: {p}';total+=1
    protected=['CURRENT_POST_ST_MODEL.json','crforest','experiments/post_st_risk_index_v1','experiments/post_st_cause_support_v1','experiments/post_st_reporting_drift_v1','experiments/post_st_calendar_level_v1']
    assert not subprocess.check_output(['git','diff',SOURCE,'--',*protected],cwd=REPO).strip()
    tasks=[]
    for part in ['A','B']:
        registry=json.loads((run/f'registry_{part}.json').read_text());assert len(registry)==3
        for r in registry:
            p=Path(r['path'])
            for name,h in json.loads((p/'hashes.json').read_text()).items():assert digest(p/name)==h
            s=json.loads((p/'summary.json').read_text());assert s['status']=='COMPLETE' and s['forest_fits']==s['calibrator_fits']==0
            tasks.append(s)
    exact=pd.read_csv(PUBLIC/'EXACTNESS_CHECKS.csv')
    error_columns=['full_reconstruction','explicit_component_sum','function_reload','inverse','forest_reload','merged_prediction','calendar_endpoint_error']
    assert exact[error_columns].to_numpy().max()<=1e-10 and exact.mu_positive_finite.all()
    rank=pd.read_csv(PUBLIC/'RANK_IDENTITY_CHECKS.csv');assert rank.order_equal.all() and rank.ties_equal.all()
    assert (rank.loc[rank.supported,'annual_metrics_exact_identity']==1).all()
    assert rank.loc[~rank.supported,['annual_metrics_exact_identity','pooled_auc_reference','pooled_auc_index','ap_reference','ap_index']].isna().all().all()
    for metric in ['pooled_auc','ap']:
        np.testing.assert_allclose(rank[metric+'_reference'],rank[metric+'_index'],atol=0,rtol=0,equal_nan=True)
    within=pd.read_csv(PUBLIC/'WITHIN_YEAR_IDENTITY.csv');assert within.exact_equal.all()
    support=pd.read_csv(PUBLIC/'OOB_SUPPORT.csv');assert support.supported.all() and support.label_poisoning_identical.all()
    assert support.oob_trees_min.min()>=10 and (support.max_leaf_parent_error<=1e-10).all()
    perf=pd.read_csv(PUBLIC/'TIER_PERFORMANCE.csv')
    assert perf.loc[(perf.year==2025)&(perf.months==24),['events','observed_rate','overall_observed','lift']].isna().all().all()
    for name in ['TIER_PERFORMANCE.csv','HIGH_REST_PERFORMANCE.csv']:
        d=pd.read_csv(PUBLIC/name);assert not d.events.between(1,4).any() and not d.companies.between(1,4).any()
        assert d.loc[d.publication_status.str.startswith('SUPPRESSED'),['events','observed_rate','lift']].isna().all().all()
    diff=pd.read_csv(PUBLIC/'TIER_DIFFERENCE_95.csv')
    unsupported=(diff.scope.astype(str)=='2025')&(diff.months==24)
    assert diff.loc[unsupported,['estimate','lower95','upper95','valid_repeats','undefined_repeats']].isna().all().all()
    assert (diff.loc[~unsupported,'valid_repeats']+diff.loc[~unsupported,'undefined_repeats']==1000).all()
    test_xml=Path(test_xml) if test_xml else run/'pytest_final_PRIVATE.xml'
    xml=ET.parse(test_xml);suite=xml.getroot().find('testsuite')
    assert suite is not None and int(suite.attrib['failures'])==int(suite.attrib['errors'])==int(suite.attrib['skipped'])==0
    expected_r={'test_weighted_split_against_independent_rfsrc[weights0]',
                'test_weighted_split_against_independent_rfsrc[weights1]',
                'test_r_survival_independent_aj','test_rfsrc_exhaustive_single_split_and_leaf_aj',
                'test_weighted_aj_independent_r_survival'}
    cases=suite.findall('testcase');rchecks=[x.attrib['name'] for x in cases if x.attrib['name'] in expected_r]
    assert set(rchecks)==expected_r and len(rchecks)==5
    # Scan only the newly authorized publication scope, never older rankings.
    identifiers=set()
    for year in [2023,2024,2025]:
        for kind in ['train','eval']:
            with np.load(run/f'input/{year}/{kind}_PRIVATE.npz') as z:
                for col in ['groups','sample_ids']:identifiers.update(str(v) for v in z[col] if len(str(v))>=6)
    pattern=re.compile(r'(?<![\w.])(?:'+'|'.join(re.escape(v) for v in sorted(identifiers,key=len,reverse=True))+r')(?!\w)')
    files=[f for f in PUBLIC.iterdir() if f.is_file()]
    forbidden_columns={'sample_id','sample_ids','company_id','company_code','mu','mu_star','threshold','thresholds','q50','q80','cif','score'}
    for f in files:
        assert f.suffix not in ['.npz','.parquet','.pkl','.pickle'] and '_PRIVATE' not in f.name
        if f.suffix=='.csv':assert not (set(pd.read_csv(f,nrows=0).columns)&forbidden_columns)
        if f.suffix=='.pdf':content=subprocess.check_output(['pdftotext',str(f),'-'],text=True)
        elif f.suffix in ['.md','.json','.csv','.log','.py']:content=f.read_text()
        else:continue
        assert not pattern.search(content),f'Company token in public {f.name}'
    return dict(status='PASS',immutable_file_checks=total,protected_paths_unchanged=protected,
                successful_real_validation_tasks=len(tasks),forest_fits=0,calibrator_fits=0,
                maximum_score_error=float(exact[error_columns].to_numpy().max()),
                all_supported_rank_metrics_identical=True,all_oob_memberships_passed=True,
                unsupported_2025_h24_performance_NA=True,public_company_tokens_found=0,
                public_raw_parameter_columns_found=0,small_cells_and_complements_suppressed=True,
                public_files_scanned=len(files),pytest=dict(tests=int(suite.attrib['tests']),failures=0,errors=0,skipped=0,independent_R_cases=rchecks,private_junit_sha256=digest(test_xml)),
                visual_review='PNG and rasterized PDF inspected; suppressed data shown S and unsupported data NA; no identifiers')


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--tests-xml',type=Path);a=p.parse_args()
    run=Path(os.environ.get('RISK_INDEX_RUN',json.loads((PUBLIC/'run_location.json').read_text())['private_run']))
    result=audit(run,a.tests_xml)
    with a.output.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
