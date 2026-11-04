"""Grouped attribution on real rows; no prediction models or calibrators are fitted."""
from scipy.stats import rankdata
from .common import *
from .stats import meanfinite,quant,summary

def triple(x):
    return (x[TRIPLES[:,0]]+x[TRIPLES[:,1]]+x[TRIPLES[:,2]])/3

def labelset(level):
    return INTER if level=='I28' else LABELS[:4] if level=='C4' else LABELS

def details(allvals,axis,level,ntrain,events,context):
    """132 model-wise distributions and within-company replication by group."""
    labs=labelset(level);tx=allvals[12:,:ntrain];x=allvals[12:,ntrain:]
    ev=allvals[:,ntrain:];mabs=abs(ev).mean(1)
    ranks=rankdata(-mabs,axis=1)
    pos=(tx>=.001).mean(0)>=.9;neg=(tx<=-.001).mean(0)>=.9
    records=[];model_rows=[];support_rows=[]
    for j,label in enumerate(labs):
        base=dict(**context,level=level,group=label)
        a=x[:,:,j];amp=mabs[12:,j];company_ranks=rankdata(-abs(x),axis=-1)[:,:,j]
        for i,arm in enumerate(ARMS+COMB):
            val=ev[i,:,j]
            model_rows.append(dict(**base,model=arm,mean_abs=mabs[i,j],signed_mean=val.mean(),rank_by_mean_abs=ranks[i,j],
                firm_p05=np.quantile(val,.05),firm_median=np.median(val),firm_p95=np.quantile(val,.95),fraction_positive_material=(val>=.001).mean(),fraction_negative_material=(val<=-.001).mean()))
        npos,nneg=int(pos[:,j].sum()),int(neg[:,j].sum())
        cp,cn=int(events[pos[:,j]].sum()),int(events[neg[:,j]].sum())
        train_amp=np.median(abs(tx[:,:,j]).mean(1))
        support_rows.append(dict(**base,train_companies=ntrain,consensus_positive_firms=npos,consensus_negative_firms=nneg,positive_events=cp,negative_events=cn,
                                 any_consensus_firms=npos+nneg,any_consensus_events=cp+cn,train_mean_abs_median=train_amp,
                                 support_pass=bool(npos+nneg>=10 and cp+cn>=3 and train_amp>=.001)))
        for s,total in enumerate((2100,3780,1260)):
            agree=axis[:,s,j,0];material=axis[:,s,j,1]
            p=np.divide(agree,material,out=np.full_like(agree,np.nan),where=material>0)
            records.append(dict(**base,shared=s,mean_abs_median=np.median(amp),amp_p05=np.quantile(amp,.05),amp_p95=np.quantile(amp,.95),amp_sd=amp.std(ddof=1),amp_range=np.ptp(amp),signed_mean_median=np.median(a.mean(1)),median_rank=np.median(ranks[12:,j]),
                first_rank_frequency=(ranks[12:,j]==1).mean(),median_company_rank_range=np.median(np.ptp(company_ranks,axis=0)),
                mean_company_sd=a.std(0,ddof=1).mean(),mean_company_range=np.ptp(a,axis=0).mean(),
                sign_agreement=meanfinite(p),sign_evaluable_firms=np.isfinite(p).sum(),both_material_fraction=material.mean()/total,
                positive_material_fraction=(a>=.001).mean(),negative_material_fraction=(a<=-.001).mean(),near_zero_fraction=(abs(a)<.001).mean(),
                ref96_mean_abs=mabs[10,j],ensemble320_mean_abs=mabs[11,j]))
    return records,model_rows,support_rows

def cancellation(raw_triples,encoded,net,context):
    """Compute denominator after forming each triple, before grouped summation."""
    rows=[]
    for g in range(8):
        ids=np.where(encoded==g)[0];gross=np.abs(raw_triples[...,ids]).sum(-1)
        ratio=np.divide(abs(net[...,g]),gross+1e-15,out=np.full_like(gross,np.nan),where=gross>0)
        cancel=1-ratio;by_combo=meanfinite(cancel,axis=1)
        rows.append(dict(**context,group=LABELS[g],gross_mean=gross.mean(),net_signed_mean=net[...,g].mean(),abs_net_mean=abs(net[...,g]).mean(),
                         retention_mean=meanfinite(ratio),cancellation_mean=meanfinite(cancel),cancellation_p05=quant(by_combo,.05),cancellation_p95=quant(by_combo,.95),
                         gross_combo_sd=gross.mean(1).std(ddof=1),net_combo_sd=net[...,g].mean(1).std(ddof=1),valid_fraction=np.isfinite(cancel).mean()))
    return rows

def feature_comparators(raw,keys,context):
    """All exact native terms, variable mains, symmetric full variable attribution."""
    var=np.zeros((*raw.shape[:-1],74));main=np.zeros_like(var)
    for i,(j,k) in enumerate(keys):
        if k<0:
            var[...,j]+=raw[...,i];main[...,j]=raw[...,i]
        else:
            var[...,j]+=.5*raw[...,i];var[...,k]+=.5*raw[...,i]
    active=np.any(raw!=0,axis=(0,1))
    info=[]
    for name,val in [('native_terms',raw[...,active]),('variable_main74',main),('variable_full74',var),('financial_variables35',var[...,mapping()<4])]:
        res,_,_=summary(val.transpose(1,0,2),5,name,False,context);info+=res
    return info,var

def worker(year,first,last,out,root):
    keys=np.load(root/f'{year}_keys.npy');source=np.load(root/f'{year}_terms.npy',mmap_mode='r')
    bases=np.load(root/f'{year}_base.npy')
    tr,ev,lab=layout(root,year);nt=len(tr['X']);n=len(ev['X'])
    maps=np.vstack([mapping(),pd.read_csv(D/'RANDOM_MAPS.csv',index_col=0).to_numpy()-1])
    folds=range(first,last);metrics=[];items=[];models=[];supports=[];cancels=[];profiles=[];random_items=[]
    for u,h in enumerate((12,24)):
        for k in (0,1):
            values=source[:,u,k]
            ec=tr['event']==k+1
            events=ec & (tr['time']<=tr[f'h{h}'])
            raw=triple(values[:10,nt:]);info=dict(year=year,months=h,cause=k+1)
            if first==0:
                controls,var=feature_comparators(raw,keys,info|dict(map_id=0));metrics.extend(controls)
            for mid in folds:
                context=info|dict(map_id=mid);groups=maps[mid];enc=bins(keys,groups)
                ai=blocksum(values,enc,36); c=allocate(ai)
                np.testing.assert_allclose(ai.sum(-1),c.sum(-1),atol=1e-10,rtol=0)
                channels=[('A8',ai[...,:8],3),('C8',c,3),('C4',c[...,:4],1),('I28',ai[...,8:],3)]
                for level,val,topk in channels:
                    triples=triple(val);x=triples[:,nt:]
                    if mid==0 and level=='C8':
                        np.testing.assert_allclose(blocksum(var,groups,8),x,atol=1e-10,rtol=0)
                    rows,axis,scored=summary(x.transpose(1,0,2),topk,level,mid==0,context)
                    metrics.extend(rows)
                    if mid==0:
                        v=np.concatenate((val,triples),axis=0)
                        a,b,s=details(v,axis,level,nt,events,info);items.extend(a);models.extend(b);supports.extend(s)
                    else:
                        # Descriptive per-slot amplitude, including the pseudo-four random slots.
                        amp=abs(x).mean(1)
                        for j,label in enumerate(labelset(level)):
                            random_items.append(dict(**context,level=level,group=label,mean_abs_median=np.median(amp[:,j]),signed_mean=x[:,:,j].mean(),
                                                     mean_company_sd=x[:,:,j].std(0,ddof=1).mean()))
                cc=triple(c[:,nt:]);aai=triple(ai[:,nt:])
                cancels+=cancellation(raw,enc,aai[...,:8],context)
                if mid==0:
                    # Private endpoint export. All 132 models, all actual firms, both group accounting conventions.
                    allai=np.concatenate([ai,triple(ai)]);allc=allocate(allai);allmu=np.concatenate([bases[:,u,k],triple(bases[:10,u,k])])
                    np.savez_compressed(out/f'{h}_{k+1}_grouped_PRIVATE.npz',A=allai[...,:8],I=allai[...,8:],C=allc,mu=allmu)
                    previous=pd.read_csv(root/f'cases/{year}/firms_PRIVATE.csv')
                    prev=previous[(previous.months==h)&(previous.cause==k+1)].sort_values('row')
                    assert np.array_equal(prev.row,np.arange(n))
                    size=max(5,int(np.ceil(.2*n)));sort=np.argsort(prev.median_l1,kind='stable')
                    for band,idx in [('low',sort[:size]),('high',sort[-size:])]:
                        profile=cc[:,idx].mean(1)
                        for j,g in enumerate(LABELS):profiles.append(dict(**info,group=g,band=band,companies=size,signed_median=np.median(profile[:,j]),p05=np.quantile(profile[:,j],.05),p95=np.quantile(profile[:,j],.95)))
                    for q in [.1,.9]:
                        row=int(arrays(root/f'cases/{year}/case_{h}_{k+1}_{q}_PRIVATE.npz')['row'])
                        np.savez_compressed(out/f'case_{h}_{k+1}_{q}_PRIVATE.npz',row=row,C=cc[:,row],A=aai[:,row,:8],I=aai[:,row,8:],reference=c[10,nt+row],case_rule=q)
    for name,rows in [('metrics',metrics),('items',items),('model_wise',models),('support_PRIVATE',supports),('cancel',cancels),('profiles',profiles),('random_amplitudes',random_items)]:
        if rows:pd.DataFrame(rows).to_csv(out/(name+'.csv'),index=False)
