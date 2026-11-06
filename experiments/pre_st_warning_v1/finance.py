"""Availability-masked current-vintage financial facts from local raw CSVs."""
from pathlib import Path
import argparse,json,re,time
import numpy as np
import pandas as pd
from .financial_formulas import add_financial_metrics,INCOME_VALUE_COLS,CASHFLOW_VALUE_COLS,BALANCE_VALUE_COLS
from .raw_scan import sha

def build(root,out):
    audit=[]
    def read(folder,stem,names=None,cols=None):
        p=root/folder/stem
        desc=next(p.parent.glob(p.stem+'[[]DES[]]*.txt'))
        meaning=dict(re.findall(r'^(\w+) \[([^]]+)\]',desc.read_text(),flags=re.M))
        mapping={k:v for k,v in meaning.items() if v in (names or [])}
        if names:
            assert set(mapping.values())==set(names),(stem,set(names)-set(mapping.values()))
            cols=['Stkcd','Accper','Typrep','IfCorrect','DeclareDate']+list(mapping)
        d=pd.read_csv(p,usecols=cols,dtype=str).fillna('')
        d.Stkcd=d.Stkcd.str.zfill(6)
        audit.append(dict(source=str(p),sha256=sha(p),dictionary_sha256=sha(desc),rows=len(d),mapping=mapping))
        return d.rename(columns=mapping)
    I=read('年、中、季报基本情况文件','IAR_Rept.csv',cols=['Stkcd','Accper','Annodt'])
    I['ann']=pd.to_datetime(I.Annodt,errors='coerce');I['period']=pd.to_datetime(I.Accper,errors='coerce')
    I=I.dropna(subset=['ann','period']).groupby(['Stkcd','Accper'],as_index=False).ann.min()
    parts=[]
    for k,(folder,stem,names) in enumerate([
        ('资产负债表','FS_Combas.csv',BALANCE_VALUE_COLS),
        ('利润表','FS_Comins.csv',INCOME_VALUE_COLS),
        ('现金流量表(直接法)','FS_Comscfd.csv',CASHFLOW_VALUE_COLS)]):
        d=read(folder,stem,names);d=d[d.Typrep.eq('A')].copy()
        du=d.duplicated(['Stkcd','Accper'],keep=False);audit[-1]['duplicate_excluded']=int(du.sum());d=d[~du].copy()
        corr=d.DeclareDate.map(lambda x:max(re.findall(r'\d{4}-\d{2}-\d{2}',x),default=''))
        good=d.IfCorrect.eq('0')|(d.IfCorrect.eq('1')&corr.ne(''))
        audit[-1]['uncertified_correction_excluded']=int((~good).sum());d=d[good].copy()
        d['correction_'+str(k)]=pd.to_datetime(corr[good],errors='coerce')
        for n in names:d[n]=pd.to_numeric(d[n],errors='coerce')
        parts.append(d[['Stkcd','Accper','correction_'+str(k)]+names])
    f=parts[0]
    for d in parts[1:]:f=f.merge(d,on=['Stkcd','Accper'],how='outer',validate='one_to_one')
    f=f.merge(I,on=['Stkcd','Accper'],validate='one_to_one')
    f['available']=f[['ann','correction_0','correction_1','correction_2']].max(axis=1)
    f['stock_code']=f.Stkcd;f['financial_period_end']=pd.to_datetime(f.Accper,errors='coerce')
    f=f.dropna(subset=['financial_period_end']);f=add_financial_metrics(f)
    f['log_cash_to_short_debt']=np.log1p(f.cash_to_short_debt.where(f.cash_to_short_debt.ge(0)))
    prev=f[['Stkcd','Accper','available']].rename(columns={'Accper':'prev_period','available':'prev_available'})
    f['prev_period']=(f.financial_period_end-pd.DateOffset(years=1)).dt.strftime('%Y-%m-%d')
    f=f.merge(prev,on=['Stkcd','prev_period'],how='left',validate='many_to_one')
    out.mkdir(parents=True,exist_ok=True);f.to_parquet(out/'financial_facts_PRIVATE.parquet',index=False)
    (out/'financial_sources.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
    print(json.dumps(dict(financial_rows=len(f),companies=f.Stkcd.nunique(),vintage='PROVISIONAL current-vintage'),ensure_ascii=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();build(a.root,a.out)
