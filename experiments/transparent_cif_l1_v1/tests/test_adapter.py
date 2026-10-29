import hashlib,json
from pathlib import Path
import numpy as np,pandas as pd,pytest
from experiments.transparent_cif_v2.evaluate import load_versioned_references

def test_versioned_reference_hash_and_identity(tmp_path):
 p=tmp_path/'pred.npz';np.savez(p,sample_ids=['a'],groups=['g'],cif12=[[.2,.1]])
 entry={'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'model':'L1_q2_fulltrain_reference','seed':42};m=tmp_path/'index.json';m.write_text(json.dumps([entry]));d=pd.DataFrame(dict(sample_id=['a'],group=['g']))
 a,h=load_versioned_references(m,d);assert a['L1_q2_fulltrain_reference',42][0]==.2
 with pytest.raises(AssertionError):load_versioned_references(m,pd.DataFrame(dict(sample_id=['b'],group=['g'])))
 entry['sha256']='bad';m.write_text(json.dumps([entry]))
 with pytest.raises(ValueError,match='hash'):load_versioned_references(m,d)

def test_real_interpret_round_selection():
 from experiments.transparent_cif_v2.ebm_control import fit_ebm_control
 rng=np.random.default_rng(17);x=rng.normal(size=(240,3));y=(x[:,0]+rng.normal(size=240)>.3).astype(int)
 model,r=fit_ebm_control(x[:180],y[:180],np.ones(180),np.arange(180),x[180:],y[180:],np.ones(60),np.arange(180,240),rounds=(2,3),seed=17)
 assert r['selected_rounds'] in [2,3] and len(r['candidates'])==2 and not r['refit_on_holdout'];assert np.isfinite(model.predict_proba(x)).all()

def test_bootstrap_wrapper_preserves_backend_summary(tmp_path,monkeypatch):
 import sys,types
 from experiments.transparent_cif_l1_v1 import worker
 from experiments.label_revision_sensitivity_v1 import bootstrap_worker
 source=tmp_path/'input';source.mkdir();out=tmp_path/'output';out.mkdir()
 monkeypatch.setenv('DATA_DIR',str(source));monkeypatch.setenv('OUTPUT_DIR',str(out))
 monkeypatch.setitem(sys.modules,'instrument',types.SimpleNamespace(install=lambda:None,save=lambda p:Path(p).write_text('[]')))
 def backend(source,output,batch):(output/'summary.json').write_text(json.dumps({'batch':batch,'status':'COMPLETE','replicates':100}))
 monkeypatch.setattr(bootstrap_worker,'run',backend)
 worker.main(types.SimpleNamespace(stage='bootstrap',seed=-1,model=None,batch=0))
 assert json.loads((out/'summary.json').read_text())['replicates']==100
 assert json.loads((out/'task_summary.json').read_text())['stage']=='bootstrap'
