"""Replay or execute the unchanged original worker; no new learning logic."""
import os,sys,json,time,socket,resource,argparse,shutil
from pathlib import Path
from dataclasses import replace,asdict
import numpy as np
from crforest import ForestConfig,CompetingRiskForest,CIFExpansion
from experiments.search_coverage_v1.worker import run as original_run,audit
from support import sha,write,arrays

def control(source,out,seed,variant):
 t=time.perf_counter();run=source.parent
 item=next(x for x in json.loads((source/'model_provenance_PRIVATE.json').read_text()) if x['seed']==seed and x['variant']==variant)
 folder=Path(item['source_folder']);f=CompetingRiskForest.load(folder/'forest_PRIVATE.json')
 assert sha(folder/'forest_PRIVATE.json')==item['model_sha256']
 config=json.loads((source/'plan.json').read_text())['configs'][f'final/{seed}/{variant}'];assert asdict(f.config)==config
 ev=arrays(source/'L0_ORIGINAL/final_eval_PRIVATE.npz');old=arrays(folder/'predictions_PRIVATE.npz')
 pred=f.predict(ev['X'],[365.,366.]);audit(pred);risk=pred[np.arange(len(pred)),ev['horizon']-365]
 assert np.array_equal(old['sample_ids'],ev['sample_ids']) and np.array_equal(old['groups'],ev['groups'])
 error=float(abs(risk-old['cif12']).max());assert error==0
 shutil.copy2(folder/'forest_PRIVATE.json',out/'forest_PRIVATE.json');np.savez_compressed(out/'predictions_PRIVATE.npz',**old)
 write(out/'summary.json',dict(status='L0_REPLAY_EXACT',seed=seed,variant=variant,reload_error=error,records=len(risk),model_sha256=sha(out/'forest_PRIVATE.json'),source=item,host=socket.gethostname(),seconds=time.perf_counter()-t,core_file=sys.modules['crforest.core'].__file__,numpy=np.__version__))

def explain(source,out,version,seed):
 t=time.perf_counter();run=source.parent;index=json.loads((run/'audit/models_index.json').read_text())
 seeds=[42,101,202] if seed==-1 else [seed]
 fs=[CompetingRiskForest.load(Path(index[f'{version}/q2/{s}'])/'forest_PRIVATE.json') for s in seeds]
 f=fs[0]
 if seed==-1:
  f.config=replace(f.config,n_trees=sum(len(g.trees_) for g in fs));f.trees_=sum([g.trees_ for g in fs],[]);f.audit_=f.audit()
 tr=arrays(source/'L0_ORIGINAL/final_train_PRIVATE.npz');ev=arrays(source/'L0_ORIGINAL/final_eval_PRIVATE.npz')
 exp=CIFExpansion.from_forest(f,tr['X']);exp.save(out/'functions_PRIVATE.npz');loaded=CIFExpansion.load(out/'functions_PRIVATE.npz')
 grid=np.unique(np.r_[0.,f.grid_,365.,366.]);err=0.;reloaderr=0.;scalars=0
 for i in range(0,len(ev['X']),128):
  xx=ev['X'][i:i+128];p=f.predict(xx,grid);e=loaded.predict(xx,grid);audit(p);audit(e)
  err=max(err,float(abs(p-e).max()));reloaderr=max(reloaderr,float(abs(exp.predict(xx,grid)-e).max()));scalars+=p.size
 assert err<1e-10 and reloaderr==0
 endpoint=loaded.predict(ev['X'],[365.,366.])[np.arange(len(ev['X'])),ev['horizon']-365]
 reference=np.mean([arrays(Path(index[f'{version}/q2/{s}'])/'predictions_PRIVATE.npz')['cif12'] for s in seeds],axis=0)
 assert np.max(abs(endpoint-reference))<1e-10
 cc=loaded.complexity(tr['X']);write(out/'complexity_PRIVATE.json',cc)
 support={};descriptors=[]
 for fi,key in enumerate([(j,) for j in loaded.mains]+loaded.pairs):
  cuts=[loaded.cuts(key,j) for j in key];shape=tuple(len(c)+2 for c in cuts);bins=[]
  for j,c in zip(key,cuts):
   b=np.searchsorted(c,tr['X'][:,j],side='left');b[np.isnan(tr['X'][:,j])]=len(c)+1;bins.append(b)
  support[f'function_{fi}']=np.bincount(np.ravel_multi_index(tuple(bins),shape),minlength=int(np.prod(shape))).reshape(shape)
  descriptors.append(dict(function_index=fi,features=list(map(int,key)),shape=shape,missing_cell_index=[n-1 for n in shape],cuts=[c.tolist() for c in cuts]))
 np.savez_compressed(out/'cell_support_PRIVATE.npz',**support);write(out/'cell_descriptor_PRIVATE.json',descriptors)
 cc['structural_main_functions']=len(loaded.mains);cc['structural_interaction_functions']=len(loaded.pairs)
 cc['input_features_used']=len(loaded.mains);cc['missing_cells_retained']=True
 cc['total_function_specific_cuts']=sum(sum(n-2 for n in x['shape']) for x in descriptors)
 cc['function_shape_partition_count']=sum(sum(n for n in x['shape']) for x in descriptors)
 # Full sum of named components independently checked at fixed anchors, every time/cause.
 xx=ev['X'][np.linspace(0,len(ev['X'])-1,16,dtype=int)];total=np.broadcast_to(loaded._coeff(grid)[1],(len(xx),len(grid),2)).copy()
 for key in [(j,) for j in loaded.mains]+loaded.pairs:total+=loaded.component(xx,key,grid)
 component_error=float(abs(total-f.predict(xx,grid)).max());assert component_error<1e-10
 write(out/'summary.json',dict(version=version,seed=seed,status='EXACT_COMPLETE',reference_rows=len(tr['X']),reference_weights='uniform frozen L0 X',reconstruction_error=err,component_sum_error=component_error,function_reload_error=reloaderr,scalar_checks=scalars,grid_count=len(grid),**{k:v for k,v in cc.items() if k!='functions'},host=socket.gethostname(),seconds=time.perf_counter()-t,peak_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,function_sha256=sha(out/'functions_PRIVATE.npz')))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--mode',required=True);p.add_argument('--seed',type=int,default=42);p.add_argument('--variant',default='coverage');p.add_argument('--tree-start',type=int);p.add_argument('--tree-count',type=int,default=4);p.add_argument('--index');p.add_argument('--version');p.add_argument('--batch',type=int);a=p.parse_args()
 source=Path(os.environ['DATA_DIR']);out=Path(os.environ['OUTPUT_DIR']);out.mkdir(parents=True,exist_ok=True)
 if a.mode=='control':control(source,out,a.seed,a.variant)
 elif a.mode=='explain':explain(source,out,a.version,a.seed)
 elif a.mode=='bootstrap':
  import bootstrap_worker;bootstrap_worker.run(source,out,a.batch)
 else:
  a.fold='final';a.pilot=False;os.environ['DATA_DIR']=str(source/'L1_ADJUDICATED');original_run(a)
 write(out/'output_hashes.json',{f.name:sha(f) for f in out.iterdir() if f.is_file() and f.name!='output_hashes.json'})
