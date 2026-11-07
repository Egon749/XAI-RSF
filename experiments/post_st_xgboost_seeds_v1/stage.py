import shutil
import pandas as pd
from .common import *

def main():
    inp=run()/'input';inp.mkdir();hashes={}
    old=ROOT/'experiments/post_st_c061_xgboost_v1'
    expected=read(old/'SOURCE_HASHES.json');params=pd.read_csv(old/'COMPUTE_AND_PARAMETERS.csv')
    for y in YEARS:
        (inp/str(y)).mkdir()
        for role in ('train','eval'):
            src=previous_run()/f'input/{y}/{role}.npz';dest=inp/f'{y}/{role}.npz'
            assert sha(src)==expected[f'{y}/{role}'];shutil.copyfile(src,dest)
            hashes[f'{y}/{role}']=sha(dest)
        for h in (12,24):
            for row in params[(params.year==y)&(params.months==h)].to_dict('records'):
                assert all(row[k]==v for k,v in config(y,h).items())
    write(HERE/'INPUT_HASHES.json',dict(inputs=hashes,parent_protocol=sha(old/'FROZEN_PROTOCOL.json'),parent_pipeline=sha(old/'pipeline.py'),parent_parameters=sha(old/'COMPUTE_AND_PARAMETERS.csv'),exact_parameter_inheritance=True))

if __name__=='__main__':main()
