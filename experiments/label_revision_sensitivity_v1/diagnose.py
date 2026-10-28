"""Read-only structure/function perturbation diagnostics and artifact gates."""
from pathlib import Path
import sys,json
import numpy as np,pandas as pd
from support import sha,write,arrays
from orchestrate import outputs

def main(run):
 sys.path.insert(0,str(run/'input/frozen_code'));from crforest import CompetingRiskForest,CIFExpansion
 index=json.loads((run/'audit/models_index.json').read_text());expindex={};rows=[];function_changes=[]
 for p in outputs(run,'explain',8):
  s=json.loads((p/'summary.json').read_text());expindex[s['version'],s['seed']]=p
 for model in ['q2','unrestricted']:
  for seed in [42,101,202]:
   fs=[CompetingRiskForest.load(Path(index[f'{v}/{model}/{seed}'])/'forest_PRIVATE.json') for v in ['L0','L1']]
   def shape(tree):return [{k:v for k,v in n.items() if k in ['feature','threshold','missing_left','left','right','ancestors']} for n in tree]
   changed=sum(shape(x)!=shape(y) for x,y in zip(fs[0].trees_,fs[1].trees_))
   rows.append(dict(model=model,seed=seed,structurally_changed_trees=changed,trees=32,L0_nodes=sum(map(len,fs[0].trees_)),L1_nodes=sum(map(len,fs[1].trees_))))
 reference=arrays(run/'input/L0_ORIGINAL/final_train_PRIVATE.npz')['X'];xx=reference[np.linspace(0,len(reference)-1,512,dtype=int)]
 for seed in [42,101,202,-1]:
  a,b=[CIFExpansion.load(expindex[v,seed]/'functions_PRIVATE.npz') for v in ['L0','L1']]
  keys=sorted(set([(j,) for j in a.mains]+a.pairs+[(j,) for j in b.mains]+b.pairs),key=lambda x:(len(x),x))
  for cause in [0,1]:
   for order in [1,2]:
    vals=[];changes=0
    for key in keys:
     if len(key)!=order:continue
     delta=a.component(xx,key,[365,366])[:,:,cause]-b.component(xx,key,[365,366])[:,:,cause];vals.append(np.mean(abs(delta)));changes+=int(np.any(abs(delta)>1e-14))
    function_changes.append(dict(seed=seed,cause=cause+1,order=order,functions_in_union=len(vals),changed_functions_on_fixed_anchors=changes,mean_function_mean_abs_change=float(np.mean(vals)),max_function_mean_abs_change=float(np.max(vals)),anchors=512,scope='one adjudicated label perturbation; not general retraining stability'))
 pd.DataFrame(rows).to_csv(run/'reports/structure_changes.csv',index=False);pd.DataFrame(function_changes).to_csv(run/'reports/function_perturbation.csv',index=False)
 training=[]
 for p in outputs(run,'assemble',6):
  x=json.loads((p/'summary.json').read_text());training.append({k:v for k,v in x.items() if not isinstance(v,(list,dict))})
 pd.DataFrame(training).to_csv(run/'reports/training_diagnostics.csv',index=False)
 ledger=json.loads((run/'audit/source_hashes_before.json').read_text());bad=[p for p,h in ledger.items() if sha(p)!=h];assert not bad,bad
 write(run/'audit/immutability_verification.json',dict(source_files=len(ledger),all_original_sources_unchanged=True,mismatches=[]))
 # Every allocated task must have both successful computation and publication.
 gates=[]
 for stage,num in [('control',6),('train',48),('assemble',6),('explain',8),('bootstrap',10)]:
  report=next((run/'audit').glob(stage+'.jsonl.results-*.json'));x=json.loads(report.read_text());assert len(x['completed'])==num
  for r in x['completed']:
   assert r['returncode']==0 and r['compute_returncode']==0,r
   assert r['publication_completed'] is True and r['compute_completed'] is True,r
   assert r['storage']['publication_state']=='verified',r
  gates.append(dict(stage=stage,tasks=num,seconds=x['seconds'],hosts={h:sum(r['host']==h for r in x['completed']) for h in sorted({r['host'] for r in x['completed']})},report_sha256=sha(report),failed_tasks=0))
 write(run/'audit/publication_gates.json',gates);print(json.dumps(gates,indent=2))
if __name__=='__main__':main(Path(sys.argv[1]))
