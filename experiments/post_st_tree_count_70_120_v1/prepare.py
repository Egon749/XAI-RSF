import datetime,shutil,tempfile,subprocess
import pandas as pd
from .common import *

def main():
    old=previous_path();pub=REPO/'experiments'/NAME
    r=Path(tempfile.mkdtemp(prefix='tree_70_120_20261103_',dir=SOURCE/'private'))
    write(pub/'run_location.json',dict(private_run=str(r),source_run=str(old)))
    for n in ['input','audit','evaluation_input','private','reports','results']:(r/n).mkdir()
    sources={};manifest=load(old/'audit/input_hashes.json')
    def record(p):sources[str(p)]=sha(p);return p
    for name,h in manifest.items():
        if name.startswith('code/')or name in ['protocol.json','data_hashes.json']:continue
        p=record(old/'input'/name);assert sha(p)==h
        dest=r/'input'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    for y in YEARS:
        # Opaque label copy; no new evaluation before protocol commit.
        shutil.copy2(record(old/f'evaluation_input/{y}_test_PRIVATE.npz'),r/f'evaluation_input/{y}_test_PRIVATE.npz')
    shutil.copy2(record(old/'private/bootstrap_indices_PRIVATE.npz'),r/'private/bootstrap_indices_PRIVATE.npz')
    reused=load(record(old/'audit/registry_pilot.json'))+load(record(old/'audit/registry_formal.json'))
    assert len(reused)==156 and all(x['count']in range(27,41)and x['count']!=32 for x in reused)
    for rec in reused:
        p=Path(rec['path']);hs=load(record(p/'hashes.json'))
        for name,h in hs.items():assert sha(record(p/name))==h
    write(r/'input/reused_prefix_exports.json',reused)
    assert load(record(old/'reports/data_validation.json'))['status']=='PASS'
    record(REPO/'experiments/post_st_tree_count_local_v1/METRICS_ALL.csv')
    pilot=load(Path(load(old/'audit/registry_pilot.json')[0]['path'])/'summary.json')
    assert pilot['peak_rss_MiB']<=600 and pilot['forest_fits']==0
    shutil.copy2(REPO/'experiments/post_st_tree_count_local_v1/COHORT_AUDIT.csv',pub/'COHORT_AUDIT.csv')
    rows=[]
    for b in COUNTS:
        for k,v in asdict(configuration(42,b)).items():
            ref=asdict(configuration(42,32))[k];assert k=='n_trees'or v==ref
            rows.append(dict(trees_per_seed=b,combined_trees=3*b,parameter=k,reference=json.dumps(ref),actual=json.dumps(v),changed=v!=ref))
    pd.DataFrame(rows).to_csv(pub/'PARAMETER_DIFFERENCES.csv',index=False)
    protocol=dict(name=NAME,status='FROZEN_BEFORE_NEW72_75_78_RESULTS',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        years=YEARS,seeds=SEEDS,trees_per_seed=COUNTS,total_trees=[3*b for b in COUNTS],new_total_trees=[72,75,78],
        range_semantics='Total trees, equal size and weight of three seeds;70/71 are not admissible.72–120 step3,17counts.',
        configuration=asdict(configuration(42,32)),features=74,training_followup_calendar_months=24,
        emphasis='24month first termination,2023/2024 annual and pooled/within-year;2025H24 performanceNA',
        secondary='12month2023–2025,bothcauses,raw/shrunk,all seeds; mean seeds before metrics',
        selection='No new composite selection or adoption; report all metric-specific maxima. Previous12month ranking is not relabeled as24month selection.',
        reuse=dict(representations=336,source_forests=9,staged_trees_per_seed=40),
        new=dict(export_tasks=36,representations=72,forest_fits=0,new_trees=0,calibration_fits=0),total_representations=408,
        uncertainty=dict(repeats=1000,seed=ANALYSIS_SEED,unit='company',strata='testyear',draws='reuse original cause-support',contrast='every count minus32/seed;fixed predictions'),
        verification='Full train+evaluation/all native times/bothcauses,exact reconstruction<=1e-10,component sum,reload,CIFvalid,no clipping;reproduce every old81–120 metric and verify hashes',
        resources='Existing two-nodeSlurm,NFS,unchanged prefix exporter;24oliver/8lemon slots,2GiB process limit. Reuse successful120-tree peak450MiB resource pilot;no repeated pilot.',
        privacy='Companies,raw data,parameters,cutpoints,predictions,functions private;aggregates/code/hashes public',
        limitations='Post-exposure extension after prior broad andlocal tree-count results;exploratory only,not independent confirmation. Original PIT/74-feature provenance retained. No changes tocurrentpointer/mainbundle/96tree model.')
    write(pub/'frozen_protocol.json',protocol);write(r/'input/protocol.json',protocol)
    write(r/'audit/source_hashes.json',sources);write(r/'audit/current_pointer.json',dict(sha256=sha(REPO/'CURRENT_POST_ST_MODEL.json')))
    print(r,flush=True)

if __name__=='__main__':main()

