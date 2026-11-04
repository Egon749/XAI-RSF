"""Secondary signed CIF accounting on the fixed annual companies, no subgroup selection."""
from crforest import CIFExpansion
from .common import *
def block(name,info):
    if info in ['F','A','Q']:return info
    if name.startswith(('log_price_last','log_total_mktcap','log_float_mktcap','ret_cum_')):return 'price_cap_return'
    if name.startswith(('ret_vol_','max_drawdown_','var_','es_','extreme_negative_','intraday_range_')):return 'volatility_tail'
    if name.startswith(('log_amihud_','turnover_mean_','log_amount_mean_','zero_amount_ratio_')):return 'liquidity_trading'
    raise ValueError(name)
def analyze():
    r=run();source_hashes=read(r/'sources_PRIVATE.json');f=features()
    f['subgroup']=[block(n,g) for n,g in zip(f.english_name,f['info'])]
    f[['input_position','english_name','info','subgroup']].to_csv(D/'MARKET_SUBGROUP_MAPPING.csv',index=False)
    files={(i['year'],i['arm'],i['variant']):Path(i['path'])/'shrunk/functions_PRIVATE.npz' for i in read(r/'ensemble/published.json')}
    anchor=read(D/'SOURCE_TIME_AUDIT.json')['anchor']
    for item in anchor['annual']:
        matches=[Path(p) for p,h in source_hashes.items() if h==item['source_function_sha256'] and p.endswith('/C039_PRIVATE.npz')]
        assert len(matches)==1;files[item['year'],'M3','fixed25']=matches[0]
    rows=[]
    for (y,a,v),path in files.items():
        if a not in ['M2','M3'] or v!='fixed25':continue
        ex=CIFExpansion.load(path);spec=f[f[a]]
        X=arrays(r/f'input/folds/{y}/eval_PRIVATE.npz')['X'][:,f[a].to_numpy()]
        lab=arrays(r/f'eval/{y}.npz');days=np.array(TIMES)
        values=np.zeros((len(X),len(days),2,len(spec)))
        for j in ex.mains:values[...,j]+=ex.component(X,(j,),days)
        for j,k in ex.pairs:
            t=ex.component(X,(j,k),days);values[...,j]+=.5*t;values[...,k]+=.5*t
        base=ex._coeff(days)[1];orig=ex.predict(X,days)
        err=float(abs(base+values.sum(-1)-orig).max());assert err<1e-10
        for h in [12,24]:
            loc=np.searchsorted(days,lab[f'h{h}']);terms=values[np.arange(len(X)),loc]
            for group in [*spec.subgroup.unique(),'M_numerical','M_broad']:
                mask=spec['info'].isin(['M','Q']) if group=='M_broad' else (spec['info']=='M' if group=='M_numerical' else spec.subgroup.eq(group))
                contrib=terms[...,mask.to_numpy()].sum(-1)
                for c in (1,2):rows.append(dict(year=y,arm=a,months=h,cause=c,subgroup=group,fields=int(mask.sum()),mean_absolute=float(abs(contrib[:,c-1]).mean()),signed_mean=float(contrib[:,c-1].mean()),reconstruction_error=err))
    table=pd.DataFrame(rows);table.to_csv(D/'SUBGROUP_ATTRIBUTION.csv',index=False)
    # Same audit and numerical market groups as the previous eight-group definition.
    ref=pd.read_csv(ROOT/'experiments/post_st_grouped_explanation_v1/GROUP_MODEL_DISTRIBUTIONS.csv').query("level=='C8' and model=='C039' and group in ['G6','G7']")
    for z in ref.itertuples():
        q=table[(table.year==z.year)&(table.arm=='M3')&(table.months==z.months)&(table.cause==z.cause)&(table.subgroup=={'G6':'M_numerical','G7':'A'}[z.group])]
        np.testing.assert_allclose([q.mean_absolute.item(),q.signed_mean.item()],[z.mean_abs,z.signed_mean],atol=1e-12,rtol=0)
if __name__=='__main__':analyze()
