"""No-fit standalone deployment and selected EBM training/holdout diagnostics."""
from pathlib import Path
import sys,json,shutil,subprocess,pickle
import numpy as np,pandas as pd
from sklearn.metrics import average_precision_score,roc_auc_score
from crforest.transparent import write_json,file_sha256 as sha,state_targets
from experiments.transparent_cif_v2.run import training_contract
from experiments.transparent_cif_l1_v1.orchestrate import gate
SOURCE=Path('/home/oliver/MainDiskData/label_revision_sensitivity_v1_20261028T151804Z/handoff_L1')
def main(run):
 deploy=run/'standalone_deployment';deploy.mkdir();(deploy/'crforest').mkdir();(deploy/'crforest/__init__.py').write_text('')
 for name in ['functions.py','transparent.py']:shutil.copy2(run/'input/code/crforest'/name,deploy/'crforest'/name)
 shutil.copy2(run/'input/code/experiments/transparent_cif_v2/server/deploy_verify.py',deploy/'verify.py')
 with np.load(SOURCE/'final_eval_PRIVATE.npz') as z:np.savez_compressed(deploy/'inputs_PRIVATE.npz',X=z['X'],sample_ids=z['sample_ids'],horizon=z['horizon'])
 index=[]
 for p in gate(run,'native_audit',24):
  r=json.loads((p/'native_audit.json').read_text());name=f"{r['model']}_{r['seed']}";shutil.copytree(p/'deployment',deploy/name);shutil.copy2(p/'expected_PRIVATE.npz',deploy/name/'expected_PRIVATE.npz');index.append(dict(model=r['model'],seed=r['seed'],path=name))
 write_json(deploy/'index.json',index)
 with (run/'audit/standalone.log').open('x') as f:subprocess.run([sys.executable,'-I',str(deploy/'verify.py')],cwd=deploy,stdout=f,stderr=subprocess.STDOUT,check=True)
 a,d,h,eligible=training_contract(SOURCE,'2022-01-01');paths=json.loads((run/'audit/fits_frozen.json').read_text())['seed_paths'];rows=[]
 for seed,path in paths.items():
  fit=Path(path);roles=np.load(fit/'roles_PRIVATE.npz');st=roles['structure'];wt=roles['weight'];
  for name in ['ebm_pair0','ebm_pair10']:
   with (fit/name/'model_PRIVATE.pkl').open('rb') as f:m=pickle.load(f)
   for role,ix in [('structure_training',st),('internal_holdout',wt)]:
    y,w,_=state_targets(a['time'][ix],a['event'][ix],h[ix],eligible[ix],a['time'][st],a['event'][st],2);target=y[:,0,1];weights=w[:,0];p=m.predict_proba(a['X'][ix])[:,1];ok=weights>0
    rows.append(dict(seed=int(seed),model=name,role=role,records=len(ix),ap=average_precision_score(target[ok],p[ok],sample_weight=weights[ok]),auc=roc_auc_score(target[ok],p[ok],sample_weight=weights[ok]),ipcw_brier=np.sum(weights*(target-p)**2)/len(ix),prediction_mean=float(p.mean()),gt99pct=int((p>.99).sum()),main_terms=sum(len(t)==1 for t in m.term_features_),pair_terms=sum(len(t)==2 for t in m.term_features_),logit_cells=sum(t.size for t in m.term_scores_)))
 pd.DataFrame(rows).to_csv(run/'reports/ebm_role_diagnostics.csv',index=False)
 write_json(run/'audit/standalone_summary.json',dict(models=24,forest_code_present=False,runtime_hashes={str(p.relative_to(deploy)):sha(p) for p in deploy.rglob('*.py')},receipt_sha256=sha(deploy/'deployment_receipt.json')))
if __name__=='__main__':main(Path(sys.argv[1]))
