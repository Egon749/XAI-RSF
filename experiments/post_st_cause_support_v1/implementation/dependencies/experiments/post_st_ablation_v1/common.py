from dataclasses import asdict, replace
from pathlib import Path
import hashlib
import json
import numpy as np
from crforest import ForestConfig, cluster_bootstrap
from crforest.partition import tree_feature_subset
from experiments.post_st_prognosis_v1.common import arrays, sha, write

SOURCE_COMMIT = '261eb15e95b01e4527e07648216bbee3974f8444'
HISTORY = Path('/home/oliver/MainDiskData/xai_rsf_leaf_shallow_20261028_ep_9hazj')
YEARS = [2023, 2024]
SEEDS = [42, 101, 202]
NEW = ['q_unrestricted', 'local_cuts', 'iid_features']
OLD = ['baseline', 'parent_shrink', 'shallow', 'shallow_parent']
ARMS = OLD + NEW + [a+'_raw' for a in NEW]
TIMES = np.array([365., 366., 730., 731.])

def configuration(seed, arm):
    cfg = ForestConfig(n_trees=32, max_depth=4, path_budget=2, min_leaf=10,
        min_groups=5, min_events=3, mtry=75, n_thresholds=5, n_causes=2,
        seed=seed, cut_strategy='shared_quantile', tree_feature_count=25,
        cause_weights=(1., 1.), global_cell_budget=None, tree_feature_strategy='balanced')
    base = arm.removesuffix('_raw')
    if base in ('baseline', 'parent_shrink'): cfg = replace(cfg, max_depth=6)
    if base == 'q_unrestricted': cfg = replace(cfg, path_budget=None)
    if base == 'local_cuts': cfg = replace(cfg, cut_strategy='local_random')
    if base == 'iid_features': cfg = replace(cfg, tree_feature_strategy='iid')
    if arm not in ARMS: raise ValueError(arm)
    return cfg

def array_hash(value):
    a = np.ascontiguousarray(value, dtype='<i8')
    return hashlib.sha256(a.tobytes()).hexdigest()

def pairing(groups, cfg, p=83):
    return [dict(tree=i, bootstrap_sha256=array_hash(cluster_bootstrap(groups, cfg.seed, i)),
        multiplicity_sha256=array_hash(np.bincount(cluster_bootstrap(groups,cfg.seed,i),minlength=len(groups))),
        features_sha256=array_hash(tree_feature_subset(p,cfg.tree_feature_count,cfg.seed,i,cfg.tree_feature_strategy)))
        for i in range(cfg.n_trees)]

def registry(run):
    rows = json.loads((run/'audit/reused_artifacts.json').read_text())
    gate = run/'audit/formal_gate.json'
    if gate.exists():
        job = json.loads(gate.read_text())['job']
        for i,line in enumerate((run/'formal.jsonl').read_text().splitlines()):
            argv=json.loads(line)['argv']; get=lambda k:argv[argv.index(k)+1]
            y,s,a=int(get('--year')),int(get('--seed')),get('--arm')
            folder=run/'results/formal'/str(job)/f'task-{i}/output'
            for variant,child in [(a+'_raw','raw'),(a,'shrunk')]:
                rows.append(dict(year=y,arm=variant,seed=s,path=str(folder/child),reuse=False))
    return rows

def canonical(value):
    return json.loads(json.dumps(value))
