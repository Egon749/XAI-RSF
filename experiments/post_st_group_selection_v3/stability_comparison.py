"""Compare matched complete-process functions with the unchanged83 fallback runs."""
import numpy as np
import pandas as pd
from .run import RUN,PUBLIC,V2
from .deliver import save
from experiments.post_st_dropcolumn_selection_v1.common import load,sha


def main():
    receipt=load(V2/'audit/execution_receipt.json');tables={};rows=[]
    for name in ['FUNCTION_STABILITY.csv','FUNCTION_SET_STABILITY.csv','PREDICTION_STABILITY.csv']:
        old=V2/'reports'/name;assert sha(old)==receipt['output_hashes'][name]
        tables[name]=(pd.read_csv(old),pd.read_csv(PUBLIC/name))
    for role,frame in zip(['matched83','new_selection'],tables['FUNCTION_STABILITY.csv']):
        for (year,h,c,o),z in frame.groupby(['year','horizon','cause','order']):
            rows.append(dict(model=role,year=year,horizon=h,cause=c,order=o,union_functions=len(z),mean_presence=z.structural_presence.mean(),
                sum_mean_abs_function=z.mean_abs_mean_function.sum(),rms_function_sd_norm=float(np.sqrt(np.sum(z.rms_function_sd**2)))))
    save('STABILITY_COMPARISON.csv',rows)
    for name in ['FUNCTION_SET_STABILITY.csv','PREDICTION_STABILITY.csv']:
        parts=[]
        for role,frame in zip(['matched83','new_selection'],tables[name]):parts.append(frame.assign(model=role))
        save('MATCHED_'+name,pd.concat(parts,ignore_index=True))

if __name__=='__main__':main()
