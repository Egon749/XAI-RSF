"""Read and verify existing same-fold forecasts; no model fitting."""
from pathlib import Path
import json
import numpy as np
from experiments.post_st_prognosis_v1.common import arrays,sha


def forecasts(protocol):
    output={};details={};summaries=[]
    for year in protocol['test_years']:
        source=Path(protocol['reference_runs'][str(year)])
        gate=json.loads((source/'audit/formal_gate.json').read_text());assert gate['status']=='PASS'
        data=arrays(source/'input/eval_PRIVATE.npz');mask=data['year']==year
        files=list((source/'results/formal'/str(gate['job'])).glob('task-*/output/summary.json'));assert len(files)==9
        for path in files:
            folder=path.parent;s=json.loads(path.read_text());assert s['status']=='COMPLETE'
            for name,digest in json.loads((folder/'hashes.json').read_text()).items():assert sha(folder/name)==digest
            p=arrays(folder/'predictions_PRIVATE.npz')
            np.testing.assert_array_equal(p['sample_ids'],data['sample_ids'])
            selected={**p,'sample_ids':p['sample_ids'][mask],'cif':p['cif'][mask]}
            key=(year,s['model'],str(s['seed']));assert key not in output
            output[key]=selected;details[key]=folder;summaries.append(s)
    assert len(output)==18
    return output,details,summaries
