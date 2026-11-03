"""Fixed74-input model entry: original engine, direct-parent AJ10, exact CIFs.

All outputs must use a new private directory. The bundle publication itself
only executes verification of existing models; it never invokes train.
"""
import argparse,json,hashlib,copy,time
from pathlib import Path
from dataclasses import asdict,replace
import numpy as np
from crforest import CompetingRiskForest,ForestConfig,CIFExpansion
from crforest.shrinkage import shrink_leaf_cifs

BUNDLE=Path(__file__).resolve().parents[1]
SPEC=json.loads((BUNDLE/'配置/MODEL_SPEC.json').read_text())
TIMES=np.array([365.,366.,730.,731.])

def arrays(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path,obj):
    with Path(path).open('x') as f:json.dump(obj,f,ensure_ascii=False,indent=2)

def data(path,training=False):
    z=arrays(path)
    if z['X'].ndim!=2 or z['X'].shape[1]!=74:raise ValueError('Expected frozen74 columns in INPUT_FEATURES.csv order')
    if training:
        for k in ['time','event','groups']:
            if k not in z or len(z[k])!=len(z['X']):raise ValueError('Aligned training outcomes/groups required')
    return z

def configuration(seed):
    if seed not in SPEC['seeds']:raise ValueError('Only the three frozen seeds are allowed')
    return ForestConfig(**{**SPEC['forest_config'],'seed':seed})

def check_model(f):
    if f.n_features_!=74:raise ValueError('Model is not74-input')
    expected=asdict(configuration(f.config.seed));actual=asdict(f.config)
    if actual['n_trees'] not in [32,96] or len(f.trees_)!=actual['n_trees']:raise ValueError('Expected32 or96 trees')
    actual['n_trees']=32
    if actual!=expected:raise ValueError('Saved configuration differs from this model')
    if f.audit()['max_path_variables']>2:raise ValueError('q2 constraint violated')

def export(f,tr,ev,out):
    check_model(f);out.mkdir(exist_ok=False)
    ex=CIFExpansion.from_forest(f,tr['X']);f.save(out/'forest_PRIVATE.json');ex.save(out/'functions_PRIVATE.npz')
    fr=CompetingRiskForest.load(out/'forest_PRIVATE.json');er=CIFExpansion.load(out/'functions_PRIVATE.npz')
    times=np.unique(np.r_[0.,f.grid_,TIMES]);X=np.vstack([tr['X'],ev['X']]);keys=[(j,) for j in ex.mains]+ex.pairs
    errors={k:0. for k in ['forest_reload','function_reload','reconstruction','explicit_sum']}
    minimum=0.;maximum=0.;increment=0.
    for start in range(0,len(X),32):
        xx=X[start:start+32];p=f.predict(xx,times)
        total=np.broadcast_to(ex._coeff(times)[1],p.shape).copy()
        for key in keys:total+=ex.component(xx,key,times)
        for name,value in [('forest_reload',fr.predict(xx,times)),('function_reload',er.predict(xx,times)),('reconstruction',ex.predict(xx,times)),('explicit_sum',total)]:errors[name]=max(errors[name],float(abs(value-p).max()))
        minimum=min(minimum,float(p.min()));maximum=max(maximum,float(p.sum(-1).max()));increment=min(increment,float(np.diff(p,axis=1).min()))
    if max(errors.values())>1e-10 or minimum < -1e-12 or maximum>1+1e-10 or increment < -1e-12:raise AssertionError('Exactness or probability check failed; no clipping allowed')
    np.savez_compressed(out/'predictions_PRIVATE.npz',cif=f.predict(ev['X'],TIMES),times=TIMES,sample_ids=ev.get('sample_ids',np.arange(len(ev['X']))))
    result=dict(status='PASS',rows=len(X),native_times=len(f.grid_),verified_times=len(times),**errors,probability_min=minimum,probability_sum_max=maximum,minimum_increment=increment,**f.audit())
    write(out/'verification.json',result);return result

def train(a,out):
    tr=data(a.train,True);ev=data(a.eval)
    f=CompetingRiskForest(configuration(a.seed)).fit(tr['X'],tr['time'],tr['event'],tr['groups'],n_jobs=1)
    sh=shrink_leaf_cifs(f,tr['X'],tr['time'],tr['event'],tr['groups'],10.)
    result={v:export(m,tr,ev,out/v) for v,m in [('raw',f),('shrunk',sh)]}
    return dict(forest_fits=1,derived_models=1,checks=result)

def verify(a,out):
    f=CompetingRiskForest.load(a.model);tr=data(a.train,True);ev=data(a.eval);check=export(f,tr,ev,out/'export')
    if a.saved_predictions:
        old=arrays(a.saved_predictions);new=arrays(out/'export/predictions_PRIVATE.npz')
        for k in ['times','sample_ids']:np.testing.assert_array_equal(old[k],new[k])
        error=float(abs(old['cif']-new['cif']).max());assert error<=1e-10;check['source_prediction_error']=error
    return dict(forest_fits=0,checks=check)

def predict(a,out):
    f=CompetingRiskForest.load(a.model);check_model(f);x=data(a.input)
    times=np.asarray(a.days,float)
    if not np.isfinite(times).all() or np.any(times<0):raise ValueError('Nonnegative finite day offsets required')
    np.savez_compressed(out/'predictions_PRIVATE.npz',times=times,sample_ids=x.get('sample_ids',np.arange(len(x['X']))),cif=f.predict(x['X'],times))
    return dict(forest_fits=0,rows=len(x['X']),day_offsets=times.tolist(),note='For calendar endpoints select365/366 or730/731 using each company origin; day365 alone is not always12calendar months')

def merge(a,out):
    members=[CompetingRiskForest.load(p) for p in a.models]
    if sorted(m.config.seed for m in members)!=[42,101,202] or any(len(m.trees_)!=32 for m in members):raise ValueError('Exactly three frozen32-tree seed models required')
    for m in members:
        check_model(m)
        if not all('leaf_shrinkage' in n and n['leaf_shrinkage']['strength']==10 for tree in m.trees_ for n in tree if 'cif' in n):raise ValueError('Merge requires the three shrunk models')
    first=members[0]
    if not all(np.array_equal(m.grid_,first.grid_) for m in members):raise ValueError('Models must use the same training snapshot/time grid')
    f=CompetingRiskForest(replace(first.config,n_trees=96));f.grid_=first.grid_.copy();f.n_features_=74
    f.trees_=copy.deepcopy([t for m in members for t in m.trees_]);f.candidate_evaluations_=sum(m.candidate_evaluations_ for m in members);f.audit_=f.audit()
    tr=data(a.train,True);ev=data(a.eval)
    error=float(abs(f.predict(ev['X'],TIMES)-np.mean([m.predict(ev['X'],TIMES) for m in members],axis=0)).max());assert error<=1e-10
    return dict(forest_fits=0,mean_prediction_error=error,checks=export(f,tr,ev,out/'export'))

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    for name in ['train','verify','predict','merge']:
        s=sub.add_parser(name);s.add_argument('--output',type=Path,required=True)
        if name in ['train','verify','merge']:s.add_argument('--train',type=Path,required=True);s.add_argument('--eval',type=Path,required=True)
        if name=='train':s.add_argument('--seed',type=int,choices=SPEC['seeds'],required=True)
        if name in ['predict','verify']:s.add_argument('--model',type=Path,required=True)
        if name=='verify':s.add_argument('--saved-predictions',type=Path)
        if name=='predict':s.add_argument('--input',type=Path,required=True);s.add_argument('--days',type=float,nargs='+',default=TIMES.tolist())
        if name=='merge':s.add_argument('--models',type=Path,nargs=3,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    result={'train':train,'verify':verify,'predict':predict,'merge':merge}[a.command](a,a.output)
    result.update(command=a.command,seconds=time.perf_counter()-start,input_sha256={str(v):sha(v) for k,v in vars(a).items() if k!='output' and isinstance(v,Path) and v.is_file()},output_sha256={str(p.relative_to(a.output)):sha(p) for p in a.output.rglob('*') if p.is_file()})
    if a.command=='merge':result['source_model_sha256']={str(p):sha(p) for p in a.models}
    write(a.output/'execution_receipt.json',result);print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
