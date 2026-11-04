from pathlib import Path
from dataclasses import replace
import json
import numpy as np
import pandas as pd
from experiments.post_st_tree_count_v1.common import configuration,arrays,sha,TIMES
from experiments.post_st_requested_seeds10_v1.common import run_path as seed_run
from experiments.post_st_cause_support_v1.common import RUN as CLOCK
from experiments.post_st_paper_features_v1.common import history
D=Path(__file__).resolve().parent
ROOT=D.parents[1]
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):
    with Path(p).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def run():return Path(read(D/'run_location.json')['private_run'])
def source():return seed_run()
YEARS=[2023,2024,2025];SEEDS=[42,101,202]
SETS={'M0':('F',),'M1':('F','A'),'M2':('F','M','Q'),'M3':('F','A','M','Q'),'M2_numeric':('F','M'),'M3_numeric':('F','A','M'),'M0_coverage':('F','Q')}
def features():
    d=pd.read_csv(ROOT/'experiments/post_st_grouped_explanation_v1/FEATURE_GROUP_MAPPING.csv')
    d['info']=np.select([d.group=='G7',d.group=='G6',d.english_name=='market_missing_coverage'],['A','M','Q'],default='F')
    d['reason']=np.select([d['info']=='A',d['info']=='M',d['info']=='Q',d.group=='G8'],['previous audit descriptors','market numerical signal','availability of market observations','common control'],default='financial measurement G1-G5')
    for arm,allowed in SETS.items():d[arm]=d['info'].isin(allowed)
    assert [d[a].sum() for a in SETS]==[45,52,67,74,66,73,46]
    return d
def pool_size(n,variant):return 25 if variant=='fixed25' else int(np.floor(n*25/74+.5))
def variants(arm):return ['fixed25','proportional'] if arm in ['M0','M1','M2'] else ['fixed25']
def cfg(seed,n,variant):return replace(configuration(seed,32),tree_feature_count=pool_size(n,variant))
def gate_arms(audit):
    if not audit['financial_clock_pass'] or not audit['market_clock_pass'] or not audit['matched_rows_pass']:raise ValueError('Frozen clock or common sample failed')
    return [arm for arm,rules in SETS.items() if arm!='M3' and ('A' not in rules or audit['audit_release_pass'])]
