from pathlib import Path
from dataclasses import asdict,replace
import json,hashlib
import numpy as np
from experiments.post_st_ablation_v1.common import configuration as existing_configuration,pairing,array_hash
from experiments.post_st_prognosis_v1.common import sha,write,arrays

SEEDS=[42,101,202];ARMS=['q2','unrestricted'];TIMES=np.array([365.,730.]);P=83;REPETITIONS=50
SCENARIOS={
 'additive_small':dict(n=350,kind='additive',distribution='binary',censor=.2),
 'pair_small':dict(n=350,kind='pair',distribution='binary',censor=.2),
 'pair_large':dict(n=3000,kind='pair',distribution='binary',censor=.2),
 'pure_pair_large':dict(n=3000,kind='pure_pair',distribution='binary',censor=.2),
 'rare_pair_small':dict(n=350,kind='rare_pair',distribution='binary',censor=.2),
 'correlated_pair_large':dict(n=3000,kind='pair',distribution='correlated_binary',censor=.2),
 'heavy_censor_pair_large':dict(n=3000,kind='pair',distribution='binary',censor=1.5),
 'triple_large':dict(n=3000,kind='triple',distribution='binary',censor=.2),
 'continuous_pair_large':dict(n=3000,kind='pair',distribution='uniform',censor=.2),
}

def configuration(seed,arm):
    if arm not in ARMS:raise ValueError(arm)
    return existing_configuration(seed,'shallow_parent' if arm=='q2' else 'q_unrestricted')

def data_hash(data):
    h=hashlib.sha256()
    for k in sorted(data):
        a=np.ascontiguousarray(data[k]);h.update(k.encode());h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()

def signals(rep):
    return np.random.default_rng(np.random.SeedSequence([20261030,rep,99])).choice(P,7,replace=False)

def truth_terms(scenario,rep,cause=1):
    s=list(map(int,signals(rep)));kind=SCENARIOS[scenario]['kind']
    if cause==2:return {():.40,(s[3],):.04}
    if kind=='rare_pair':return {():.06,(s[0],):.01,(s[1],):.01,tuple(sorted(s[:2])):.02}
    if kind=='pure_pair':return {():.18,(s[2],):.04,tuple(sorted(s[:2])):.05}
    answer={():.18,(s[0],):.04,(s[1],):.03}
    if kind!='additive':answer[tuple(sorted(s[:2]))]=.05
    if kind=='triple':answer[tuple(sorted(s[4:7]))]=.04
    return answer

def time_scale(times):
    return -np.expm1(-np.minimum(np.maximum(np.asarray(times,float),0),730)/450)/(-np.expm1(-730/450))

def truth(X,scenario,rep,times=TIMES):
    X=np.asarray(X);value=np.zeros((len(X),2))
    for c in [1,2]:
        for key,b in truth_terms(scenario,rep,c).items():value[:,c-1]+=b*np.prod(X[:,list(key)],axis=1)
    assert value.min()>=0 and value.sum(1).max()<=1
    return value[:,None,:]*time_scale(times)[None,:,None]

def covariates(n,scenario,rep,stream):
    rng=np.random.default_rng(np.random.SeedSequence([20261030,rep,stream]));spec=SCENARIOS[scenario]
    # No outcome-guided location: signal coordinates change independently across repetitions.
    u=rng.random((n,P));X=2*u-1 if spec['distribution']=='uniform' else np.where(u<.5,-1.,1.)
    if spec['distribution']=='correlated_binary':
        j,k=signals(rep)[:2];X[:,k]=np.where(rng.random(n)<.7,X[:,j],X[:,k])
    return X

def generate(scenario,rep,n_test=5000):
    spec=SCENARIOS[scenario];X=covariates(spec['n'],scenario,rep,0)
    rng=np.random.default_rng(np.random.SeedSequence([20261030,rep,1]));p=truth(X,scenario,rep,[730])[:,0,:]
    u=rng.random(len(X));latent_event=np.where(u<p[:,0],1,np.where(u<p.sum(1),2,0))
    latent_time=np.ceil(-450*np.log1p(-rng.random(len(X))*(-np.expm1(-730/450))))
    latent_time=np.where(latent_event>0,latent_time,np.inf)
    censor=np.minimum(730.,np.ceil(rng.exponential(730/spec['censor'],len(X))))
    event=np.where(latent_time<=censor,latent_event,0)
    train=dict(X=X,time=np.minimum(latent_time,censor),event=event,groups=np.arange(len(X)))
    test=covariates(n_test,scenario,rep,2)
    return train,test

def population_kind(scenario):return 'uniform' if SCENARIOS[scenario]['distribution']=='uniform' else 'binary'

def interval_mass(lo,hi,kind):
    lo=np.asarray(lo);hi=np.asarray(hi)
    if kind=='binary':return .5*((lo<-1)&(hi>=-1)).astype(float)+.5*((lo<1)&(hi>=1)).astype(float)
    return np.maximum(0,np.minimum(hi,1)-np.maximum(lo,-1))/2

def interval_first_moment(lo,hi,kind):
    lo=np.asarray(lo);hi=np.asarray(hi)
    if kind=='binary':return -.5*((lo<-1)&(hi>=-1)).astype(float)+.5*((lo<1)&(hi>=1)).astype(float)
    a=np.maximum(lo,-1);b=np.minimum(hi,1)
    return np.where(b>a,(b*b-a*a)/4,0.)

def config_differences():
    a,b=[asdict(configuration(42,k)) for k in ARMS]
    differences={k:[a[k],b[k]] for k in a if a[k]!=b[k]};assert set(differences)=={'path_budget'}
    return differences
