"""Frozen endpoint baselines and independent exact finite-background Shapley audit."""
import math
from itertools import product
import numpy as np
from scipy.special import expit,logit
from crforest.functions import region_indicator


def endpoint_weights(time,event,horizon):
    """Reverse KM: left limits at events and administrative endpoint."""
    time=np.asarray(time,float);event=np.asarray(event);h=np.broadcast_to(horizon,time.shape)
    tt,inv,count=np.unique(time,return_inverse=True,return_counts=True)
    risk=np.cumsum(count[::-1])[::-1];cens=np.bincount(inv,weights=event==0)
    g=np.cumprod(1-cens/risk)
    def left(t):
        ix=np.searchsorted(tt,t,side='left')-1
        return np.where(ix>=0,g[np.maximum(ix,0)],1.)
    known=(event>0)|(time>=h-1e-9)
    den=np.where(event>0,left(time),left(h))
    return np.where(known,1/np.maximum(den,1e-8),0.)


def resample_groups(groups,seed):
    """Materialize a shared company bootstrap; repeated draws get clone IDs."""
    groups=np.asarray(groups);unique,inv=np.unique(groups,return_inverse=True)
    rng=np.random.default_rng(seed);draw=rng.integers(len(unique),size=len(unique))
    rows=[np.flatnonzero(inv==j) for j in draw]
    return np.concatenate(rows),np.concatenate([np.full(len(x),i) for i,x in enumerate(rows)])


def group_reference(X,groups,n,seed):
    rng=np.random.default_rng(seed);u=np.unique(groups);chosen=rng.choice(u,min(n,len(u)),replace=False)
    return np.asarray([X[rng.choice(np.flatnonzero(groups==g))] for g in chosen])


def fit_ebm(X,event,weights,interactions,seed,rounds=2000):
    from interpret.glassbox import ExplainableBoostingClassifier
    # No random row-level internal validation: fixed rounds and external common
    # company resampling avoid leakage through quarter records of a company.
    model=ExplainableBoostingClassifier(feature_types=['continuous']*X.shape[1],interactions=interactions,
        max_bins=32,max_interaction_bins=16,max_rounds=rounds,learning_rate=.03,
        greedy_ratio=0,smoothing_rounds=50,interaction_smoothing_rounds=50,
        min_samples_leaf=20,max_leaves=3,validation_size=0,outer_bags=1,
        early_stopping_rounds=0,n_jobs=1,random_state=seed)
    keep=weights>0
    model.fit(X[keep],(event[keep]==1).astype(int),sample_weight=weights[keep])
    return model


def ebm_complexity(model,X):
    terms=model.eval_terms(X);logits=model.intercept_[0]+terms.sum(1)
    error=float(abs(expit(logits)-model.predict_proba(X)[:,1]).max())
    nonzero=[bool(np.any(abs(v)>1e-14)) for v in model.term_scores_]
    return dict(reconstruction_error=error,main_functions=sum(z and len(k)==1 for z,k in zip(nonzero,model.term_features_)),
        pair_functions=sum(z and len(k)==2 for z,k in zip(nonzero,model.term_features_)),
        spatial_cells=sum(v.size for v in model.term_scores_),native_scale='log_odds',link='logistic',
        trained_rounds=np.asarray(model.best_iteration_).tolist(),endpoint_only=True)


class ScalarForest:
    """Exact cached vector traversal for the same fitted forest and one CIF output."""
    def __init__(self,forest,time=365.,cause=0):
        self.forest=forest;self.time=time;self.cause=cause
        self.T=len(forest.trees_);size=max(map(len,forest.trees_));shape=(self.T,size)
        self.feature=np.zeros(shape,int);self.cut=np.zeros(shape);self.missing=np.zeros(shape,bool)
        self.left=np.zeros(shape,int);self.right=np.zeros(shape,int);self.leaf=np.ones(shape,bool);self.value=np.zeros(shape)
        self.depth=max(n['depth'] for t in forest.trees_ for n in t)
        ti=np.searchsorted(forest.grid_,time,side='right')-1;self.regions=[]
        for t,tree in enumerate(forest.trees_):
            for i,n in enumerate(tree):
                if 'cif' in n:self.value[t,i]=n['cif'][ti][cause] if ti>=0 else 0.
                else:
                    self.leaf[t,i]=False;self.feature[t,i]=n['feature'];self.cut[t,i]=n['threshold'];self.missing[t,i]=n['missing_left'];self.left[t,i]=n['left'];self.right[t,i]=n['right']
            def visit(i,region):
                n=tree[i]
                if 'cif' in n:self.regions.append((region,self.value[t,i]/self.T));return
                j=n['feature'];cut=n['threshold'];lo,hi,miss=region.get(j,(-np.inf,np.inf,True))
                a,b=dict(region),dict(region);a[j]=(lo,min(hi,cut),miss and n['missing_left']);b[j]=(max(lo,cut),hi,miss and not n['missing_left'])
                visit(n['left'],a);visit(n['right'],b)
            visit(0,{})
    def __call__(self,X):
        X=np.asarray(X,float);out=np.empty(len(X));trees=np.arange(self.T)[:,None]
        for start in range(0,len(X),4096):
            xx=X[start:start+4096];nodes=np.zeros((self.T,len(xx)),int)
            for _ in range(self.depth):
                terminal=self.leaf[trees,nodes];values=xx[np.arange(len(xx))[None,:],self.feature[trees,nodes]]
                go=(values<=self.cut[trees,nodes])|(np.isnan(values)&self.missing[trees,nodes])
                nodes=np.where(terminal,nodes,np.where(go,self.left[trees,nodes],self.right[trees,nodes]))
            out[start:start+len(xx)]=self.value[trees,nodes].mean(0)
        return out
    def exact_shap(self,X,background):
        """Exact interventional Shapley values of the finite JOINT row masker.

        A leaf is a product of indicators. For each (x,b), a coordinates change
        0->1, b change 1->0; beta-integral permutation weights give exact values.
        A constant-zero coordinate makes that whole leaf game identically zero.
        No independence/product-marginal approximation of the background rows.
        """
        X=np.asarray(X);B=np.asarray(background);phi=np.zeros_like(X,dtype=float)
        for region,value in self.regions:
            if value==0 or not region:continue
            key=list(region);ix=np.column_stack([region_indicator(X[:,j],*region[j]) for j in key]).astype(bool)
            ib=np.column_stack([region_indicator(B[:,j],*region[j]) for j in key]).astype(bool)
            pos=ix[:,None,:]&~ib[None,:,:];neg=~ix[:,None,:]&ib[None,:,:]
            valid=~((~ix[:,None,:]&~ib[None,:,:]).any(2));a=pos.sum(2);b=neg.sum(2)
            wp=np.zeros(a.shape);wn=np.zeros(a.shape)
            for aa in range(len(key)+1):
                for bb in range(len(key)+1-aa):
                    mask=valid&(a==aa)&(b==bb)
                    if aa:wp[mask]=math.factorial(aa-1)*math.factorial(bb)/math.factorial(aa+bb)
                    if bb:wn[mask]=-math.factorial(aa)*math.factorial(bb-1)/math.factorial(aa+bb)
            phi[:,key]+=value*(pos*wp[:,:,None]+neg*wn[:,:,None]).mean(1)
        baseline=float(self(B).mean())
        assert abs(baseline+phi.sum(1)-self(X)).max()<1e-10
        return baseline,phi


def binary_grid(p=8):return np.array(list(product([-1.,1.],repeat=p)))
def walsh(X):
    p=X.shape[1];return np.column_stack([np.prod(X[:,[j for j in range(p) if mask>>j&1]],axis=1) for mask in range(2**p)])

SCENARIOS=['additive_cif','pair_cif','triple_cif','additive_logit','pair_logit','correlated_pair','heavy_censor_pair']
def truth(X,scenario):
    if scenario=='additive_logit':return expit(-2+.6*X[:,0]+.6*X[:,1])
    if scenario=='pair_logit':return expit(-2+.6*X[:,0]+.8*X[:,1]*X[:,2])
    if scenario=='additive_cif':return .18+.05*X[:,0]+.04*X[:,1]
    p=.18+.04*X[:,0]+.05*X[:,1]*X[:,2]
    if scenario=='triple_cif':p+=.04*X[:,4]*X[:,5]*X[:,6]
    return p

def simulation(n,seed,scenario):
    rng=np.random.default_rng(seed);X=rng.choice([-1.,1.],size=(n,8))
    if scenario=='correlated_pair':X[:,1]=np.where(rng.random(n)<.85,X[:,0],X[:,1])
    target=truth(X,scenario)/(-np.expm1(-1.6));other=(.08+.02*X[:,3])/(-np.expm1(-1.6))
    u=rng.random(n);event=np.where(u<target,1,np.where(u<target+other,2,0))
    latent=np.where(event>0,rng.exponential(1/.8,n),np.inf)
    censor=np.minimum(2.,rng.exponential(1/(.9 if scenario=='heavy_censor_pair' else .15),n))
    observed=latent<=censor
    return dict(X=X,time=np.minimum(latent,censor),event=np.where(observed,event,0),groups=np.arange(n))
