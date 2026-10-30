from dataclasses import asdict, replace
from pathlib import Path
import json
import numpy as np
from experiments.post_st_ablation_v1.common import configuration as original_configuration
from experiments.post_st_ablation_v1.common import pairing,array_hash
from experiments.post_st_prognosis_v1.common import arrays,sha,write

PREVIOUS=Path('/home/oliver/MainDiskData/xai_rsf_post_st_ablation_20261029_uDFKagf2')
SOURCE_COMMIT='ad49e66c25f6e76ad8206ddaf7f334ac4fcbd88c'
YEARS=[2023,2024];SEEDS=[42,101,202];STAGES=['s0','s1','s2','s3','s4'];NEW=['s2','s3']
LABELS={'s0':'完整模型','s1':'再解除q限制','s2':'再改局部随机切点','s3':'再改IID25覆盖','s4':'再取消收缩'}

def configuration(seed,stage):
    if stage not in STAGES:raise ValueError(stage)
    cfg=original_configuration(seed,'shallow_parent')
    if stage!='s0':cfg=replace(cfg,path_budget=None)
    if stage in ['s2','s3','s4']:cfg=replace(cfg,cut_strategy='local_random')
    if stage in ['s3','s4']:cfg=replace(cfg,tree_feature_strategy='iid')
    return cfg

def strength(stage):
    if stage not in STAGES:raise ValueError(stage)
    return 0 if stage=='s4' else 10

def registry(run):
    result=json.loads((run/'audit/reused_artifacts.json').read_text())
    gate=run/'audit/formal_gate.json'
    if gate.exists():
        job=json.loads(gate.read_text())['job']
        for i,line in enumerate((run/'formal.jsonl').read_text().splitlines()):
            argv=json.loads(line)['argv'];get=lambda k:argv[argv.index(k)+1]
            y,s,a=int(get('--year')),int(get('--seed')),get('--arm')
            root=run/'results/formal'/str(job)/f'task-{i}/output'
            result.append(dict(year=y,arm=a,seed=s,path=str(root/'shrunk'),reuse=False))
            if a=='s3':result.append(dict(year=y,arm='s4',seed=s,path=str(root/'raw'),reuse=False))
    return result

def contrasts():
    return {**{f'{STAGES[i]}_minus_{STAGES[i-1]}':{STAGES[i]:1,STAGES[i-1]:-1} for i in range(1,5)},
        **{f'{a}_minus_s0_cumulative':{a:1,'s0':-1} for a in STAGES[1:]}}

def verification_outputs(run):
    gate=json.loads((run/'audit/verify_gate.json').read_text());assert gate['status']=='PASS'
    for i,line in enumerate((run/'verify.jsonl').read_text().splitlines()):
        argv=json.loads(line)['argv'];get=lambda k:argv[argv.index(k)+1]
        yield int(get('--year')),get('--arm'),'ensemble' if get('--seed')=='0' else get('--seed'),run/'results/verify'/str(gate['job'])/f'task-{i}/output'
