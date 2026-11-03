"""Independent date truncation and topology checks on saved artifacts; no fit."""
import pandas as pd
from .common import *

def main():
    run=run_path();checks=[];trees=[]
    records=load(run/'audit/registry_formal.json')
    paths={(r['year'],r['arm'],r['seed']):Path(r['path']) for r in records}
    for year in YEARS:
        un=pd.read_parquet(run/f'private/{year}_uncapped_training_metadata_PRIVATE.parquet')
        ev=arrays(run/f'evaluation_input/{year}_test_PRIVATE.npz')
        assert not set(un.group)&set(ev['groups'])
        assert len(np.unique(ev['groups']))==len(ev['groups'])
        cutoff=pd.Timestamp(f'{year}-01-01');end=cutoff-pd.Timedelta(days=1)
        for arm,months in ARM_CAPS.items():
            actual=arrays(run/f'input/folds/{year}/{arm}/train_PRIVATE.npz')
            independent_time=[];independent_event=[]
            for row in un.itertuples():
                limit=end if months is None else min(end,row.origin+pd.DateOffset(months=months))
                stop=min(row.stop,limit)
                independent_time.append((stop-row.origin).days)
                independent_event.append(row.event if row.stop<=limit else 0)
            np.testing.assert_array_equal(actual['time'],independent_time)
            np.testing.assert_array_equal(actual['event'],independent_event)
            checks.append(dict(year=year,arm=arm,companies=len(un),disjoint_train_test_groups=True,
                independent_date_truncation_match=True,all_training_stops_before_cutoff=bool((un.stop<cutoff).all()),
                same_first12_events=True,train_sha256=sha(run/f'input/folds/{year}/{arm}/train_PRIVATE.npz')))
            for seed in SEEDS:
                baseline=load(paths[year,'cap24',seed]/'raw/forest_PRIVATE.json')['trees']
                altered=load(paths[year,arm,seed]/'raw/forest_PRIVATE.json')['trees']
                def topology(tree):
                    return [{k:n.get(k) for k in ['depth','feature','threshold','missing_left','left','right']}for n in tree]
                equal=sum(topology(a)==topology(b)for a,b in zip(baseline,altered))
                trees.append(dict(year=year,arm=arm,seed=seed,trees=len(altered),identical_topology_to24=equal,
                    different_topology_to24=len(altered)-equal))
        for shorter in ['cap12','cap24','cap36']:
            short=arrays(run/f'input/folds/{year}/{shorter}/train_PRIVATE.npz')
            first=arrays(run/f'input/folds/{year}/cap12/train_PRIVATE.npz')
            at12=np.minimum(short['time'],first['h12'])
            ev12=np.where(short['time']<=first['h12'],short['event'],0)
            np.testing.assert_array_equal(first['time'],at12)
            np.testing.assert_array_equal(first['event'],ev12)
    for name,rows in [('INDEPENDENT_DATA_CHECKS',checks),('TOPOLOGY_CHANGE',trees)]:
        pd.DataFrame(rows).to_csv(run/'reports'/f'{name}.csv',index=False)
    pointer=load(run/'audit/current_pointer.json');assert sha(Path(pointer['path']))==pointer['sha256']
    for name,h in load(run/'audit/source_hashes.json').items():assert sha(Path(name))==h
    for root,manifest in [('input','immutable_input_hashes'),('analysis_input','immutable_analysis_hashes')]:
        for name,h in load(run/f'audit/{manifest}.json').items():assert sha(run/root/name)==h
    write(run/'reports/data_validation.json',dict(status='PASS',forest_fits=0,checks=len(checks),
        no_company_overlap=True,independent_date_truncation=True,all_sources_unchanged=True,
        frozen_training_and_analysis_snapshots_unchanged=True,current_pointer_unchanged=True))
    print(pd.DataFrame(trees).groupby('arm')[['trees','different_topology_to24']].sum().to_string())

if __name__=='__main__':main()
