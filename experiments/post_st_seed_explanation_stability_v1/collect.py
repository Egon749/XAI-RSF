"""Gate and stage the explanation run after all exact combination exports."""
import shutil
import pandas as pd
from .common import *

def prepare():
    r=run_path();root=r/'analysis_input'; root.mkdir()
    for name in ['features.json','folds','evaluation_input',*map(str,YEARS)]:
        p=r/'input'/name
        if p.is_dir():shutil.copytree(p,root/name)
        else:shutil.copy2(p,root/name)
    rows=[];mask={}
    for item in read(r/'verify/published.json'):
        year,version,first,last=item['args'];p=Path(item['path'])
        frame=pd.read_csv(p/'verify.csv');expect=NAMES[int(first):int(last)]
        assert frame.model.tolist()==expect
        for row in frame.itertuples():assert sha(p/(row.model+'_PRIVATE.npz'))==row.function_sha256
        rows.append(frame);mask.setdefault((year,version),{}).update(dict(read(p/'masks.json')))
    df=pd.concat(rows,ignore_index=True)
    assert len(df)==720 and not df.duplicated(['year','version','model']).any()
    for (year,version),members in mask.items():
        assert set(members)==set(NAMES);dump(root/f'mask_{year}_{version}.json',members)
    err=['member_mean','reconstruction','reload','explicit_sum']
    audit=dict(status='PASS',year_version_combinations=len(df),distinct_memberships=len(TRIPLES),max_errors=df[err].max().to_dict(),max_path=int(df.max_path.max()),min_probability=float(df.probability_min.min()),max_sum=float(df.max_sum.max()),min_increment=float(df.min_increment.min()),
               before_and_after_reload=True,scope='all annual training/evaluation rows; all native times, zero and four horizon days; both causes; algebraically regrouped explicit component sums')
    dump(HERE/'RECONSTRUCTION_AUDIT.json',audit)
    old=pd.read_csv(BASE/'NUMERICAL_COMPLEXITY.csv').rename(columns={'seed':'model'});old['model']=old.model.astype(str)
    assert len(old)==72
    columns=['year','model','version','trees','main_functions','interaction_functions','spatial_table_cells','scalar_table_cells','low_support_cell_fraction','zero_support_cell_fraction','compile_seconds','verification_seconds','function_bytes']
    pd.concat([df[columns],old[columns]]).to_csv(HERE/'COMPLEXITY_TRADEOFF.csv',index=False)
    df.to_csv(HERE/'RECONSTRUCTION_DETAIL.csv',index=False)
if __name__=='__main__':prepare()
