"""Stage verified saved artifacts; evaluation labels remain outside worker input."""
import datetime,shutil,tempfile,subprocess
from .common import *
def main():
    pub=REPO/'experiments'/NAME;s=source_path()
    assert not(pub/'run_location.json').exists()
    r=Path(tempfile.mkdtemp(prefix='seed_ensemble_20261103_',dir=s.parent))
    for d in ['input','evaluation_input','private','audit','reports','results']:(r/d).mkdir()
    sources={}
    def copy(src,dst):
        dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
        assert sha(src)==sha(dst);sources[str(src)]=sha(src)
    for y in YEARS:
        for n in ['train_PRIVATE.npz','eval_PRIVATE.npz']:copy(s/f'input/folds/{y}/{n}',r/f'input/folds/{y}/{n}')
        copy(s/f'evaluation_input/{y}_test_PRIVATE.npz',r/f'evaluation_input/{y}_test_PRIVATE.npz')
    copy(s/'private/bootstrap_indices_PRIVATE.npz',r/'private/bootstrap_indices_PRIVATE.npz')
    records=load(s/'audit/registry_pilot.json')+load(s/'audit/registry_formal.json')
    assert len(records)==24
    for row in records:
        src=Path(row['path']);assert load(src/'summary.json')['status']=='COMPLETE'
        for n,h in load(src/'hashes.json').items():assert sha(src/n)==h
        for n in ['forest_PRIVATE.json','predictions_PRIVATE.npz','verification.json']:
            copy(src/'shrunk'/n,r/f'input/members/{row["year"]}/{row["seed"]}/{n}')
    refs=[]
    for row in load(s/'input/old_ensemble_registry.json'):
        if row['version']!='shrunk':continue
        src=Path(row['path'])/'representation';refs.append(dict(year=row['year'],path=str(src)))
        for n in ['predictions_PRIVATE.npz','verification.json']:copy(src/n,r/f'input/reference/{row["year"]}/{n}')
    assert len(refs)==3
    protocol=dict(name=NAME,status='FROZEN_BEFORE_NEW_ENSEMBLE_RESULTS',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_commit=SOURCE_COMMIT,
        arms=ARMS,trees_per_seed=32,years=YEARS,forest_fits=0,calibration_fits=0,new_trees=0,derived_ensembles=6,
        fixed_model='74features,q2,depth4,support0or>=2,shared5,balanced25,rawdirectparentAJ10,24calendar-month training cap,ownpreJan1 snapshots',
        mapping='Equal company-level CIF average of already-shrunk members; equivalent to equal tree weights because each member has32trees. No shrinkage reapplication.',
        evaluation='Both causes,annual12/24months and pooled;2025H24performanceNA. Calendar endpoint per company;Top20ceil(.2n)withinyear. Recompute metrics from averaged predictions,never average member metrics.',
        uncertainty='1000original year-stratifiedcompanydraws seed2026110201;fixed-model paired differences toreference96;postselection exploratory intervals.',
        verification='Sample/time/hashalignment;meanpreservedbymergedforest;alltrain/evalrows,allnative times,bothcauses,fullcomponents/reload<=1e-10;CIFvalid;q<=2;independentsklearnandpairAUCchecks.',
        interpretation='User-selected memberships after exposed seed results. Different tree budgets,not independentvalidation or isolated seed effect. No search/adoption;preserve96/currentpointer/mainbundle/core/oldreports.',
        privacy='Onlycode/config/hash/aggregatespublic;identifiers,models,cuts,functions,predictionsprivate.',resources='Six independent CPU exports inexisting2nodeSlurm/NFS;noaffinityoverride;singlethread,4GiBaddresscap.')
    write(pub/'frozen_protocol.json',protocol);write(r/'input/protocol.json',protocol)
    write(r/'audit/source_hashes.json',sources);write(r/'audit/reference_paths_PRIVATE.json',refs)
    write(r/'audit/current_pointer.json',dict(sha256=sha(REPO/'CURRENT_POST_ST_MODEL.json')))
    write(pub/'run_location.json',dict(private_run=str(r),source_private_run=str(s)))
    print(json.dumps(dict(run=str(r),staged_sources=len(sources),forest_fits=0)))
if __name__=='__main__':main()
