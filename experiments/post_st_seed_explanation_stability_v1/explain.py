import os,sys,socket
import pandas as pd
from crforest import CIFExpansion
from .common import *
from .geometry import geometry
from .local import local
from .similarity import *

def analyze(year,version,root,out):
    tr=arrays(root/f'folds/{year}/train_PRIVATE.npz');ev=arrays(root/f'folds/{year}/eval_PRIVATE.npz')
    labels=arrays(root/f'evaluation_input/{year}_test_PRIVATE.npz')
    np.testing.assert_array_equal(labels['sample_ids'],ev['sample_ids'])
    features=read(root/'features.json');names=[f['feature_name'] for f in features]
    masks=read(root/f'mask_{year}_{version}.json')
    ex=[CIFExpansion.load(root/str(year)/str(s)/version/'functions_PRIVATE.npz') for s in SEEDS]
    rows=[]
    for order in (1,2):
        allowed=sorted({k for mask in masks.values() for k in mask if len(k.split(','))==order})
        m=np.array([[k in masks[name] for k in allowed] for name in NAMES])
        rows+=similarity_rows(jaccards(m)[LEFT,RIGHT],year=year,version=version,order=order,metric='structure_jaccard')
    pd.DataFrame(rows).to_csv(out/'structure.csv',index=False)
    keys=geometry(year,version,ex,tr,names,masks,out)
    local(year,version,ex,labels,ev,sorted(keys),out,root)
    dump(out/'done.json',dict(host=socket.gethostname(),args=sys.argv[1:]))
if __name__=='__main__':
    out=Path(os.environ['OUTPUT_DIR']);out.mkdir(parents=True,exist_ok=True)
    analyze(int(sys.argv[1]),sys.argv[2],Path(os.environ['DATA_DIR']),out)
