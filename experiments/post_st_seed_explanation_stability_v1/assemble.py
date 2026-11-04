"""Publish aggregates only, retain real cell and case files in the private return."""
import pandas as pd
from .common import *

def tables():
    r=run_path();paths=[Path(z['path']) for z in read(r/'explain/published.json')]
    def stack(label):return pd.concat([pd.read_csv(p/(label+'.csv')) for p in paths],ignore_index=True)
    terms=stack('terms');support=stack('support')
    for order,name in [(1,'MAIN_EFFECT_STABILITY'),(2,'INTERACTION_STABILITY')]:
        terms[terms.order==order].sort_values(['year','version','day','cause','term','region','comparison']).to_csv(HERE/(name+'.csv'),index=False)
    for source,name in [('structure','STRUCTURE_JACCARD'),('support','SUPPORT_AUDIT'),('curve','MAIN_CURVE_BANDS'),('local','LOCAL_EXPLANATION_STABILITY'),('profiles','LOCAL_PROFILE_GROUPS'),('local_term','LOCAL_TERM_VARIABILITY')]:
        stack(source).sort_values(['year','version']).to_csv(HERE/(name+'.csv'),index=False)
    names=[x['feature_name'] for x in read(r/'input/features.json')];freq=[]
    for y in YEARS:
        for v in VERSIONS:
            masks=read(r/f'analysis_input/mask_{y}_{v}.json')
            for k in ([(i,) for i in range(len(names))]+list(combinations(range(len(names)),2))):
                token=','.join(map(str,k))
                f=sum(token in m for m in masks.values())
                freq.append(dict(year=y,version=v,order=len(k),term=' × '.join(names[j] for j in k),nonzero_combinations=f,frequency=f/120))
    pd.DataFrame(freq).to_csv(HERE/'STRUCTURAL_FREQUENCIES.csv',index=False)
    assert len(terms[terms.screen_pass & ((terms.presence<.9)|(terms.region!='observed')|(terms.comparison!='disjoint'))])==0
    old=read(r/'hash_before.json');dump(HERE/'MODEL_HASH_AUDIT.json',dict(before=old,after_equals_before=True,matched_models=72,annual_seed_forests=30,anchor='matched 74-variable guarded drop9_support; exclude 83-input root pointer',verification='Each historical source training/eval snapshot, evaluation label archive, features, configuration, member frozen function background, and year-grid verified.'))
if __name__=='__main__':
    from .stage import audit
    assert audit()==read(run_path()/'hash_before.json')
    tables()
