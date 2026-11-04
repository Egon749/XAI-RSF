"""Link only the sealed legal register to unchanged predictions, by frozen evaluation row."""
from subprocess import check_output
from .common import *
from .integrity import sources,P,G
from crforest import CIFExpansion
from experiments.post_st_grouped_explanation_v1.common import layout,PAIRS,TRIPLES,SEEDS,mapping,bins
from experiments.post_st_ablation_v1.analyze import selected
SUPPLEMENT='614eb0be80ff5f7b8377ce7b4041cb884ebf98db'
def local_support(keys,terms,ex,X,tr,months,pair):
    enc=bins(keys,mapping());ix=[j for j in np.flatnonzero(enc==8+PAIRS.index(pair)) if tuple(keys[j]) in ex.pairs]
    n=len(tr['X']);target=(tr['event']==1)&(tr['time']<=tr[f'h{months}'])
    if not ix:return np.zeros(len(X)-n,bool)
    vv=terms[:,ix];dom=np.argmax(abs(vv[n:]),axis=1);ok=np.zeros(len(X)-n,bool)
    for ii,j in enumerate(ix):
        key=tuple(keys[j]);axes=[];dims=[]
        for a in key:
            cuts=ex.cuts(key,a);b=np.searchsorted(cuts,X[:,a]);b[np.isnan(X[:,a])]=len(cuts)+1;axes.append(b);dims.append(len(cuts)+2)
        cell=np.ravel_multi_index(axes,dims);nn=np.prod(dims);count=np.bincount(cell[:n],minlength=nn);deaths=np.bincount(cell[:n][target],minlength=nn)
        ok|=(dom==ii)&(vv[n:,ii]!=0)&(count[cell[n:]]>=10)&(deaths[cell[n:]]>=3)
    return ok
def main():
    gate=load(D/'SOURCE_AND_HASH_AUDIT.json');assert gate['classification_gate']
    p=run()/'DELISTING_CAUSE_REGISTRY_PRIVATE.csv';assert sha(p)==gate['private_registry_hash']
    z=pd.read_csv(p,dtype={'code':str});cases=z.set_index('code');companies=pd.read_csv(run()/'companies_PRIVATE.csv',dtype={'code':str})
    feat=pd.read_csv(G/'FEATURE_GROUP_MAPPING.csv');names=feat.english_name.tolist();market_audit=PAIRS.index((5,6))
    suppath=Path(json.loads(check_output(['git','show',SUPPLEMENT+':experiments/post_st_information_increment_best3_v1/run_location.json'],cwd=ROOT))['private_run'])
    extra=np.load(suppath/'risk_PRIVATE.npz');rank=pd.read_parquet(suppath/'rankings_PRIVATE.parquet')
    facts=[];modelsets={};sources_hashes={};unit_ok=os.environ.get('CAUSE_CAP_DICTIONARY')
    if unit_ok:assert re.search(r'Dsmvtll.{0,60}千元',Path(unit_ok).read_text(errors='replace'))
    for y in YEARS:
        inp,blocks,cp=sources(y);tr,ev,l=layout(inp,y);n=len(tr['X']);X=np.vstack([tr['X'],ev['X']]);cf=CIFExpansion.load(cp)
        com=companies[companies.year==y].sort_values('row');np.testing.assert_array_equal(com.sample_id,l['sample_ids'])
        source=np.load(inp/f'{y}_terms.npy',mmap_mode='r');keys=np.load(inp/f'{y}_keys.npy');assert len(np.unique(tr['groups']))==n
        sources_hashes[str(y)]={f:sha(inp/f'{y}_{f}.npy') for f in ('terms','keys','base')}
        for u,h in enumerate((12,24)):
            if (y,h)==(2025,24):continue
            dates=l[f'h{h}'];known=(l['time']>=dates)|(l['event']>0);term=(l['event']==1)&(l['time']<=dates);recover=(l['event']==2)&(l['time']<=dates);assert known.all()
            q=arrays(blocks/f'{h}_1_grouped_PRIVATE.npz');cs=q['C'][:,n:];inter=q['I'][:,n:];mu=q['mu'][:,n:];table=cs[50];pp=mu+cs.sum(-1)
            rr=rank.query("year==@y and months==@h and cause==1 and arm=='M3' and variant=='fixed25' and seed==0").sort_values('row');np.testing.assert_array_equal(rr.sample_id,l['sample_ids'])
            m={a:extra[f'{y}|{a}|fixed25|0|{h}|1'] for a in ('M0','M1','M2','M3')}
            np.testing.assert_allclose(pp[50],m['M3'],rtol=0,atol=1e-10);top={a:selected(v,np.full(len(v),y),l['sample_ids']) for a,v in m.items()};np.testing.assert_array_equal(top['M3'],rr.top20)
            raw=source[TRIPLES[38],u,0].mean(axis=0);V=np.zeros((len(X),74));mainV=V.copy()
            for j,(a,b) in enumerate(keys):
                if b<0:V[:,a]+=raw[:,j];mainV[:,a]+=raw[:,j]
                else:V[:,a]+=raw[:,j]/2;V[:,b]+=raw[:,j]/2
            gr=mapping()
            np.testing.assert_allclose(np.stack([V[:,gr==g].sum(1) for g in range(8)],1)[n:],table,rtol=0,atol=1e-10)
            support={p:local_support(keys,raw,cf,X,tr,h,p) for p in PAIRS}
            rm=np.exp(ev['X'][:,names.index('log_total_mktcap_last')]);price=np.exp(ev['X'][:,names.index('log_price_last')])
            for j,code in enumerate(com.code):
                true=cases.loc[code] if term[j] else None
                cat=true.category if term[j] else ('recovered' if recover[j] else 'event_free')
                val=dict(year=y,months=h,row=j,code=code,category=cat,dual=bool(term[j] and cat=='C2' and true.multiple_grounds),grounds=true.all_documented_grounds if term[j] else '',price_yuan=price[j],price_floor_ratio=price[j],market_cap_floor_ratio=np.nan)
                if unit_ok:
                    main=not code.startswith(('30','688'));floor=5e8 if main and com.origin.iloc[j]>='2024-10-30' else 3e8
                    val['market_cap_floor_ratio']=rm[j]*1000/floor
                val.update({f'G{g+1}':v*100 for g,v in enumerate(table[j])});val['four_capabilities']=table[j,:4].sum()*100
                for b,name in enumerate(names):
                    val['main_'+name]=mainV[n+j,b]*100;val['alloc_'+name]=V[n+j,b]*100
                for pn,pair in enumerate(PAIRS):
                    tag=f'I{pair[0]+1}x{pair[1]+1}';val[tag]=inter[50,j,pn]*100;val['support_'+tag]=support[pair][j]
                for a in m:val['CIF_'+a]=m[a][j]*100;val['Top_'+a]=top[a][j]
                facts.append(val)
            modelsets[y,h]=np.stack([cs[:,:,5],cs[:,:,6],cs[:,:,:4].sum(-1),inter[:,:,market_audit]],axis=-1)*100
    df=pd.DataFrame(facts);df.to_parquet(run()/'linked_channels_PRIVATE.parquet',index=False)
    A=[];B=[];T=[];S=[];J=[];M=[];shape=[]
    fields=[f'G{i}' for i in range(1,9)]+['four_capabilities','CIF_M3','price_floor_ratio','market_cap_floor_ratio']+[c for c in df if c.startswith(('main_','alloc_'))]
    for h,dh in df.groupby('months'):
        Y=sorted(dh.year.unique());types={'C1':dh.category=='C1','C2_pure':(dh.category=='C2')&(~dh.dual),'C3':dh.category=='C3','C4':dh.category=='C4','nontrading':dh.category.isin(['C1','C3','C4']),'known_noncase':dh.category.isin(['recovered','event_free']),'recovered':dh.category=='recovered','event_free':dh.category=='event_free'}
        for name,mask in types.items():
            sub=dh[mask];sparse_year=any(0<len(sub[sub.year==y])<5 for y in Y)
            for scope,ss in [('pooled',sub)]+[(str(y),sub[sub.year==y]) for y in Y]:
                nn=len(ss);publish=nn>=5 and (scope=='pooled' or not sparse_year)
                base=dict(months=h,scope=scope,group=name,n=nn,status='descriptive' if publish else 'insufficient or complementary suppression')
                for v in fields:
                    if not publish:continue
                    data=ss[v].dropna()
                    if len(data)<5:continue
                    unit='multiple of statutory threshold at origin' if v.endswith('floor_ratio') else 'CIF percentage points'
                    A.append(dict(**base,signal=v,unit=unit,nonmissing=len(data),signed_mean=data.mean(),mean_abs=data.abs().mean(),median=data.median()))
                T.append(dict(**base,eligible_years='/'.join(map(str,Y)) if scope=='pooled' else scope,captured=int(ss.Top_M3.sum()) if publish else None,coverage=ss.Top_M3.mean() if publish else None))
                if not publish:continue
                for a in ('M0','M1','M2','M3'):M.append(dict(**base,arm=a,captured=int(ss['Top_'+a].sum()),mean_overall_cif_pct=ss['CIF_'+a].mean()))
                if scope!='pooled':continue
                allv=np.concatenate([modelsets[y,h][:,ss[ss.year==y].row.to_numpy(int)] for y in Y],axis=1)
                for vi,label in enumerate(['market','audit','four_capabilities','market_x_audit']):
                    signed=allv[:,:,vi].mean(1);ab=np.abs(allv[:,:,vi]).mean(1)
                    for fam,sl in [('10 singles',slice(0,10)),('120 overlapping triples',slice(12,None)),('anchor96',slice(10,11)),('ensemble320',slice(11,12)),('C039',slice(50,51))]:
                        x=signed[sl];S.append(dict(**base,signal=label,family=fam,models=len(x),signed_p05=np.quantile(x,.05),signed_median=np.median(x),signed_p95=np.quantile(x,.95),fraction_positive=np.mean(x>0),fraction_abs_mean_ge_0_1pp=np.mean(ab[sl]>=.1),abs_median=np.median(ab[sl])))
                for x,y in PAIRS:
                    v=f'I{x+1}x{y+1}';good=ss[ss['support_'+v]];J.append(dict(**base,interaction=v,supported_firms=len(good),supported_fraction=len(good)/nn,signed_supported=good[v].mean() if len(good)>=5 else None,mean_abs_supported=good[v].abs().mean() if len(good)>=5 else None))
        for v in ['G6','G7','four_capabilities','alloc_log_price_last','alloc_log_total_mktcap_last','CIF_M3']:
            aa=dh[types['C2_pure']];bb=dh[types['nontrading']];cc=dh[dh.category=='C2']
            if min(len(aa),len(bb))<5:continue
            paired=[y for y in Y if min(sum(aa.year==y),sum(bb.year==y))>=5]
            for mode,fn in [('signed_mean',lambda x:x.mean()),('mean_abs',lambda x:x.abs().mean())]:
                a,b,c=[fn(f[v]) for f in (aa,bb,cc)]
                B.append(dict(months=h,signal=v,statistic=mode,pure_trading_n=len(aa),nontrading_n=len(bb),pure_trading=a,nontrading=b,difference=a-b,dual_inclusion_same_difference_direction=bool(np.sign(c-b)==np.sign(a-b)),common_years='/'.join(map(str,paired)),common_year_equal_weight_difference=np.mean([fn(aa.loc[aa.year==y,v])-fn(bb.loc[bb.year==y,v]) for y in paired]) if paired else None))
        # Within-training percentile groups of entry price, each company contributes to one bin only.
        for y in Y:
            inp,_,cp=sources(y);tr,ev,lab=layout(inp,y);j=names.index('log_price_last');xx=tr['X'][:,j];xx=xx[np.isfinite(xx)]
            s=dh[dh.year==y].copy();s['bin']=np.searchsorted(np.quantile(xx,[1/3,2/3]),ev['X'][:,j]);s.loc[~np.isfinite(ev['X'][:,j]),'bin']=-1
            for b,bb in s.groupby('bin'):
                if len(bb)>=5:shape.append(dict(year=y,months=h,training_price_tertile=int(b),n=len(bb),mean_price_main_pp=bb.main_log_price_last.mean(),mean_allocated_price_pp=bb.alloc_log_price_last.mean(),scope='known evaluation companies'))
    for name,rows in [('CAUSE_SIGNAL_COMPARISON',A),('TRADING_VS_NONTRADING',B),('CAUSE_TOP20_CAPTURE',T),('CAUSE_EXPLANATION_STABILITY',S),('JOINT_SUPPORT',J),('FIXED_INFORMATION_SET_CAPTURE',M),('PRICE_FUNCTION_BY_TRAINING_TERTILE',shape)]:pd.DataFrame(rows).to_csv(D/(name+'.csv'),index=False)
    dump(run()/'additional_inputs.json',dict(cached_components=sources_hashes,supplement_commit=SUPPLEMENT,supplement_risk_hash=sha(suppath/'risk_PRIVATE.npz'),supplement_id_alignment_hash=sha(suppath/'rankings_PRIVATE.parquet'),cap_unit_source_hash=sha(Path(unit_ok)) if unit_ok else None))
if __name__=='__main__':main()
