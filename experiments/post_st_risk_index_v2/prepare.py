"""Stage immutable reference evidence after protocol commit; no fitting."""
from pathlib import Path
from dataclasses import asdict
import json,os,shutil,subprocess,traceback
import numpy as np
import pandas as pd
from crforest import CompetingRiskForest
from experiments.post_st_ablation_v1.common import configuration
from experiments.post_st_reporting_drift_v1.analyze import Reader,write_json,digest,block_fit
from experiments.post_st_prognosis_v1.common import horizon_predictions
from experiments.post_st_feature_permutation_v1.common import evaluation_context
from experiments.post_st_reporting_drift_v1.metrics import paired_draws

PUBLIC=Path(__file__).resolve().parent


def main():
    CompetingRiskForest.fit=block_fit
    loc=json.loads((PUBLIC/'run_location.json').read_text());source=Path(loc['source_run']);run=Path(os.environ.get('RISK_INDEX_RUN',loc['private_run']))
    out=run/'input';out.mkdir();reader=Reader();gates=[]
    frozen=subprocess.check_output(['git','show','7abd471f884b0c26592f9afe77650edc32e4a02c:experiments/post_st_risk_index_v2/frozen_protocol.json'])
    assert frozen==(PUBLIC/'frozen_protocol.json').read_bytes()
    reader.path(PUBLIC/'frozen_protocol.json');reader.path(PUBLIC.parents[1]/'CURRENT_POST_ST_MODEL.json')
    immutable=reader.json(source/'audit/immutable_input_hashes.json');original=reader.json(source/'audit/source_hashes.json');registry=reader.json(source/'audit/registry_formal.json')
    assert digest(PUBLIC.parents[1]/'CURRENT_POST_ST_MODEL.json')==reader.json(source/'audit/current_pointer.json')['sha256']
    def copy(p,q):
        reader.path(p);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q);assert digest(q)==digest(p)
    copy(source/'input/features.json',out/'features.json')
    sizes={};allgroups=[]
    for y in [2023,2024,2025]:
        folder=out/str(y);folder.mkdir()
        tr=reader.arrays(source/f'input/folds/{y}/train_PRIVATE.npz');ev=reader.arrays(source/f'evaluation_input/{y}_test_PRIVATE.npz')
        assert digest(source/f'input/folds/{y}/train_PRIVATE.npz')==immutable[f'folds/{y}/train_PRIVATE.npz']
        assert digest(source/f'evaluation_input/{y}_test_PRIVATE.npz') in original.values()
        assert tr['X'].shape[1]==83 and (tr['year']<y).all()
        assert len(set(tr['groups']))==len(tr['X']) and len(set(ev['groups']))==len(ev['X'])
        assert not set(tr['groups'])&set(ev['groups']);allgroups.extend(ev['groups'].tolist())
        features=reader.arrays(source/f'input/folds/{y}/eval_PRIVATE.npz')
        for k in ['X','sample_ids']:np.testing.assert_array_equal(features[k],ev[k])
        metadata=reader.parquet(source/f'audit/{y}_training_metadata_PRIVATE.parquet')
        np.testing.assert_array_equal(metadata.sample_id.astype(str),tr['sample_ids'].astype(str))
        assert (pd.to_datetime(metadata.origin)<pd.Timestamp(f'{y}-01-01')).all() and (pd.to_datetime(metadata.stop)<pd.Timestamp(f'{y}-01-01')).all()
        copy(source/f'input/folds/{y}/train_PRIVATE.npz',folder/'train_PRIVATE.npz')
        copy(source/f'evaluation_input/{y}_test_PRIVATE.npz',folder/'eval_PRIVATE.npz')
        rows=sorted([r for r in registry if r['year']==y and r['arm']=='baseline'],key=lambda r:r['seed']);assert [r['seed'] for r in rows]==[42,101,202]
        ps=[]
        for r in rows:
            root=Path(r['path']);hs=reader.json(root/'hashes.json')
            for name in ['forest_PRIVATE.json','predictions_PRIVATE.npz']:
                assert digest(reader.path(root/'shrunk'/name))==hs['shrunk/'+name]
                copy(root/'shrunk'/name,folder/str(r['seed'])/name)
            f=CompetingRiskForest.load(root/'shrunk/forest_PRIVATE.json');assert asdict(f.config)==asdict(configuration(r['seed'],'shallow_parent'))
            p=reader.arrays(root/'shrunk/predictions_PRIVATE.npz');np.testing.assert_array_equal(p['sample_ids'],ev['sample_ids']);ps.append(p['cif'])
        mean=np.mean(ps,axis=0);np.savez_compressed(folder/'reference_PRIVATE.npz',cif=mean,times=p['times'],sample_ids=ev['sample_ids'])
        for h in [12,24]:
            if y==2025 and h==24:continue
            yy,w,known,supported,_=evaluation_context(ev,h,1);assert supported and known.all() and (w==1).all()
            prob=horizon_predictions(mean,p['times'],ev[f'h{h}']);actual=float(yy.mean()/prob[:,0].mean())
            expected={12:[5.8229,10.9851,1.6114],24:[5.1535,3.8376]}[h][y-2023]
            gates.append(dict(year=y,months=h,expected=expected,actual=actual,passed=round(actual,4)==expected))
            assert round(actual,4)==expected,'STOP: reference O/E mismatch'
        sizes[y]=len(ev['X'])
    assert len(set(allgroups))==len(allgroups)
    draws=reader.arrays(source/'private/bootstrap_indices_PRIVATE.npz');new=paired_draws(sizes)
    for y in sizes:np.testing.assert_array_equal(draws[str(y)],new[y])
    copy(source/'private/bootstrap_indices_PRIVATE.npz',out/'bootstrap_indices_PRIVATE.npz')
    write_json(run/'REFERENCE_GATE.json',gates);reader.finish();write_json(run/'input_sources_PRIVATE.json',reader.hashes)
    write_json(run/'stage_input_hashes.json',{str(p.relative_to(out)):digest(p) for p in out.rglob('*') if p.is_file()})
    print('Five reference O/E gates passed; all original inputs aligned; no fits.')


if __name__=='__main__':
    try:main()
    except Exception:
        loc=json.loads((PUBLIC/'run_location.json').read_text());run=Path(os.environ.get('RISK_INDEX_RUN',loc['private_run']))
        (run/'PREPARE_FAILURE.txt').write_text(traceback.format_exc());raise
