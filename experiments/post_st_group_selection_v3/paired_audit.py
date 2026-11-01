"""Independent replay of schedules, bootstrap pairing and unaffected tree identity."""
from pathlib import Path
import gzip,json
import numpy as np
from .run import RUN,PUBLIC
from .store import previous_output
from .coverage import anchored_sets,original_tree
from .deliver import bundle_path,save
from experiments.post_st_dropcolumn_selection_v1.common import load,write,sha,array_hash


def main():
    records=load(RUN/'audit/registry_pilot.json')+load(RUN/'audit/registry_batch.json');rows=[];schedule_hashes=[]
    for r in records:
        out=Path(r['output']);s=load(out/'summary.json')
        base={b['cohort']:bundle_path(out,b) for b in s['bundles'] if len(b['keep'])==83 and b['mode']=='anchored'}
        entries=[(bundle_path(out,b),base[b['cohort']],b['mode']) for b in s['bundles'] if not b['reused'] and b['mode']=='anchored' and len(b['keep'])<83]
        entries.append((out/'final',previous_output(s['year'],s['repeat'])/'final','final'))
        for path,old,mode in entries:
            sm=load(path/'summary.json');keep=sm['keep']
            for seed in [42,101,202]:
                want,coverage=anchored_sets(keep,seed,groups=s['groups']);assert coverage==load(path/f'{seed}/coverage.json')
                new=json.loads(gzip.decompress((path/f'{seed}/raw/forest_PRIVATE.json.gz').read_bytes()))
                before=json.loads(gzip.decompress((old/f'{seed}/raw/forest_PRIVATE.json.gz').read_bytes()))
                assert want==new['tree_feature_sets']
                left=load(path/f'{seed}/pairing.json');right=load(old/f'{seed}/pairing.json')
                for a,b in zip(left,right):
                    assert all(a[k]==b[k] for k in ['tree','bootstrap_sha256','multiplicity_sha256'])
                # The frozen fitting helper inherited a counterfactual default
                # feature hash. coverage.json and saved model sets are actual.
                # Preserve that receipt and publish correctly named hashes here.
                for i,(a,cr) in enumerate(zip(left,coverage)):
                    schedule_hashes.append(dict(year=s['year'],repeat=s['repeat'],mode=mode,cohort=sm['year'],features=len(keep),seed=seed,tree=i,
                        bootstrap_sha256=a['bootstrap_sha256'],multiplicity_sha256=a['multiplicity_sha256'],
                        counterfactual_default_local_pool_sha256=a.get('features_sha256',a.get('counterfactual_default_local_pool_sha256')),
                        actual_local_pool_sha256=array_hash(np.asarray(want[i],dtype=int)),actual_original_pool_sha256=array_hash(np.asarray(cr['actual'],dtype=int))))
                unchanged=0
                for cr,a,b in zip(coverage,new['trees'],before['trees']):
                    if cr['unchanged']:
                        assert original_tree(a,keep)==b;unchanged+=1
                rows.append(dict(year=s['year'],repeat=s['repeat'],mode=mode,cohort=sm['year'],features=len(keep),seed=seed,
                    trees=len(coverage),unchanged_eligible_trees=unchanged,exactly_equal_unaffected_trees=unchanged,paired_bootstrap_trees=len(coverage),schedule_verified=True))
    save('PAIRED_COVERAGE_AUDIT.csv',rows)
    save('ACTUAL_SCHEDULE_HASHES.csv.gz',schedule_hashes)
    result=dict(status='PASS',forests=len(rows),paired_bootstrap_trees=sum(r['trees'] for r in rows),exactly_equal_unaffected_trees=sum(r['unchanged_eligible_trees'] for r in rows),schedule_replayed_independently=True)
    write(RUN/'audit/paired_coverage_validation.json',result);write(PUBLIC/'audit/paired_coverage_validation.json',result);print(result)

if __name__=='__main__':main()
