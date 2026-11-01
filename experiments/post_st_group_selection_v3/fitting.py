"""Reuse the frozen fitting and verification interfaces with an anchored schedule."""
from experiments.post_st_dropcolumn_selection_h24_v1.fitting import *
from .coverage import anchored_sets, original_tree


def load_raw(path):
    return json.loads(gzip.decompress(path.read_bytes()))

def fit_bundle(parent,keep,year,out,draw_groups=None,outer_final=False,groups_spec=(),baseline_path=None):
    started=time.perf_counter();keep=tuple(keep);out.mkdir(parents=True,exist_ok=False)
    root=parent/'input'/('outer' if outer_final else 'folds')/str(year)
    original=arrays(root/'train_PRIVATE.npz');evaluation=reduced(arrays(root/'eval_PRIVATE.npz'),keep);reference=reduced(original,keep)
    if draw_groups is None:indices=np.arange(len(original['X']))
    else:
        where={g:i for i,g in enumerate(original['groups'])};indices=np.array([where[g] for g in draw_groups if g in where],int)
    train={k:v[indices] for k,v in reference.items()};np.savez_compressed(out/'training_rows_PRIVATE.npz',indices=indices)
    members=[];predictions=[];checks=[];pairsets=[];fit_times=[];shrink_times=[]
    for seed in SEEDS:
        folder=out/str(seed);folder.mkdir();cfg=configuration(seed,len(keep),'count');schedule,coverage=anchored_sets(keep,seed,groups=groups_spec);expected=pairing(train['groups'],cfg,len(keep))
        t=time.perf_counter();raw=CompetingRiskForest(cfg).fit(train['X'],train['time'],train['event'],train['groups'],feature_sources=[(j,) for j in keep],n_jobs=1,tree_feature_sets=schedule);fit_times.append(time.perf_counter()-t)
        for actual,want in zip(raw.inbag_,expected):assert array_hash(actual)==want['multiplicity_sha256']
        write(folder/'coverage.json',coverage)
        if baseline_path is not None:
            baseline=load_raw(Path(baseline_path)/str(seed)/'raw/forest_PRIVATE.json.gz')
            for row,nodes,old in zip(coverage,raw.trees_,baseline['trees']):
                if row['unchanged']:assert original_tree(nodes,keep)==old
        t=time.perf_counter();sh=shrink_leaf_cifs(raw,train['X'],train['time'],train['event'],train['groups'],10);shrink_times.append(time.perf_counter()-t)
        cc={}
        for label,model in [('raw',raw),('shrunk',sh)]:cc[label]=compact_verification(model,reference,evaluation,folder/label)
        ex=CIFExpansion.load(folder/'shrunk/functions_PRIVATE.npz');pairsets.append(set(ex.pairs));members.append(sh);predictions.append(sh.predict(evaluation['X'],TIMES));checks.append(cc)
        for row,local,actual in zip(expected,schedule,coverage):
            row['counterfactual_default_local_pool_sha256']=row.pop('features_sha256')
            row['actual_local_pool_sha256']=array_hash(np.asarray(local,dtype=int))
            row['actual_original_pool_sha256']=array_hash(np.asarray(actual['actual'],dtype=int))
        write(folder/'pairing.json',expected)
    forest=merge(members);forest.tree_feature_sets_=[s for m in members for s in m.tree_feature_sets_];ensemble_check=compact_verification(forest,reference,evaluation,out/'ensemble')
    pp=forest.predict(evaluation['X'],TIMES);difference=float(abs(pp-np.mean(predictions,axis=0)).max());assert difference<=1e-13
    overlaps=[len(a&b)/len(a|b) if a|b else 1. for a,b in itertools.combinations(pairsets,2)]
    summary=dict(year=year,keep=list(keep),subset_hash=subset_hash(keep),new_forests=3,training_rows=len(indices),training_companies=len(np.unique(train['groups'])),
        unique_training_termination=len(np.unique(train['groups'][train['event']==1])),fit_seconds=fit_times,shrink_seconds=shrink_times,seconds=time.perf_counter()-started,
        checks=checks,ensemble_check=ensemble_check,ensemble_mean_error=difference,cells=ensemble_check['spatial_table_cells'],pair_jaccard=float(np.mean(overlaps)))
    write(out/'summary.json',summary);write(out/'hashes.json',{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()})
    return dict(ensemble=pp,seeds=predictions,cells=summary['cells'],pair_jaccard=summary['pair_jaccard'])

