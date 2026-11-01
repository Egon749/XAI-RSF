"""One independent bundle or one complete resampled selection procedure."""
from pathlib import Path
import argparse,os,resource,socket,time,traceback
import numpy as np
import pandas as pd
from .fitting import *

def execute(index):
    resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3));start=time.perf_counter()
    root=Path(os.environ['DATA_DIR']);out=Path(os.environ['OUTPUT_DIR']);out.mkdir(parents=True,exist_ok=True)
    plan=load(root/'plan.json');item=plan['items'][index];parent=Path(plan['parent']);bundles=[]
    if item['kind']=='bundle':
        fit_bundle(parent,item['keep'],item['year'],out/'bundle',outer_final=item.get('outer_final',False))
        result=dict(**item,new_forests=3,bundles=['bundle'])
    elif item['kind']=='bootstrap':
        outer,repeat=item['outer'],item['repeat'];draw_groups=bootstrap_groups(parent,outer,repeat)
        support,counts=historical_support(parent,outer,draw_groups);gate=gate24(support);(out/'models').mkdir()
        store=BootstrapStore(parent,out/'models',draw_groups);ev=Evaluator(parent,outer,store,support,counts,repeat)
        groups=[g['members'] for g in load(parent/'input/group_specs.json') if g['outer']==outer]
        keep,path,scans=select(ev,initial_units(parent,outer),gate,groups)
        for r in scans:
            r['keep']=json.dumps(r['keep']);r['failed_guards']=json.dumps(r['failed_guards'])
        pd.DataFrame(scans).to_csv(out/'selection_contrasts.csv',index=False)
        if len(keep)==83:
            rr=next(r for r in load(parent/'audit/registry_aggregate_bootstrap.json') if r['year']==outer and r['repeat']==repeat)
            final=dict(reused=True,path=rr['path'],functions_sha256=sha(Path(rr['path'])/'ensemble/functions_PRIVATE.npz'));new_final=0
        else:
            fit_bundle(parent,keep,outer,out/'selected_final',draw_groups,outer_final=True)
            final=dict(reused=False,path='selected_final',functions_sha256=sha(out/'selected_final/ensemble/functions_PRIVATE.npz'));new_final=3
        result=dict(**item,gate=gate,keep=list(keep),path=path,bundles=store.bundles,final=final,new_forests=3*len(store.bundles)+new_final)
    else:raise ValueError(item['kind'])
    result.update(status='COMPLETE',host=socket.gethostname(),seconds=time.perf_counter()-start,peak_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                  code_commit=plan['code_commit'],outer_labels_read=False)
    write(out/'summary.json',result);write(out/'hashes.json',{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()})
    print(json.dumps({k:v for k,v in result.items() if k not in ['path','bundles','keep']}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--index',type=int,required=True);a=p.parse_args()
    try:execute(a.index)
    except Exception:
        out=Path(os.environ['OUTPUT_DIR']);out.mkdir(parents=True,exist_ok=True);write(out/'FAILED.json',dict(index=a.index,traceback=traceback.format_exc()));raise
