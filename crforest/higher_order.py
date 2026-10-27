"""Exact order<=4 CIF functional expansion; no truncation or tree traversal.

For each leaf rectangle, expand product(m_j + (I_j-m_j)). Each nonempty
subset is a centered component, multiplied by the other coordinates' masses.
Storage stays factored; logical full-table complexity is audited separately.
The original order-two exporter is unchanged.
"""
from itertools import combinations, product
from pathlib import Path
import numpy as np
from .functions import CIFExpansion, region_indicator


class HigherOrderCIFExpansion(CIFExpansion):
    @classmethod
    def from_forest(cls, forest, reference):
        reference=np.asarray(reference,float)
        if reference.ndim!=2 or reference.shape[1]!=forest.n_features_ or not len(reference) or np.isinf(reference).any():
            raise ValueError('finite-or-missing nonempty reference matrix required')
        order=forest.audit()['max_path_variables']
        if order>4:raise ValueError('exact higher-order export supports at most four path variables')
        obj=cls();obj.grid=forest.grid_.copy();obj.n_features=forest.n_features_
        regions=[];coefficients=[]
        for tree in forest.trees_:
            def visit(index,region):
                node=tree[index]
                if 'cif' in node:
                    regions.append(region);coefficients.append(np.asarray(node['cif'])/len(forest.trees_));return
                j,cut,ml=node['feature'],node['threshold'],node['missing_left']
                lo,hi,missing=region.get(j,(-np.inf,np.inf,True))
                left,right=dict(region),dict(region)
                left[j]=(lo,min(hi,cut),missing and ml);right[j]=(max(lo,cut),hi,missing and not ml)
                visit(node['left'],left);visit(node['right'],right)
            visit(0,{})
        shape=(len(regions),max(1,order))
        obj.features=np.full(shape,-1,int);obj.low=np.full(shape,-np.inf);obj.high=np.full(shape,np.inf)
        obj.missing=np.ones(shape,bool);obj.mass=np.ones(shape)
        for i,region in enumerate(regions):
            for a,(j,(lo,hi,missing)) in enumerate(sorted(region.items())):
                obj.features[i,a]=j;obj.low[i,a]=lo;obj.high[i,a]=hi;obj.missing[i,a]=missing
                obj.mass[i,a]=region_indicator(reference[:,j],lo,hi,missing).mean()
        obj.coefficients=np.asarray(coefficients);obj.mu=np.einsum('i,itk->tk',obj.mass.prod(1),obj.coefficients)
        obj.reference_n=len(reference);obj._index();return obj

    def _index(self):
        self.max_order=self.features.shape[1]
        if not 1<=self.max_order<=4:raise ValueError('invalid exported order')
        terms={}
        for i,features in enumerate(self.features):
            actual=features[features>=0]
            for r in range(1,len(actual)+1):
                for axes in combinations(range(len(actual)),r):
                    key=tuple(map(int,actual[list(axes)]));terms.setdefault(key,[]).append((i,axes))
        self.keys=sorted(terms,key=lambda k:(len(k),k))
        self._terms={k:(np.array([x[0] for x in v],int),np.array([x[1] for x in v],int)) for k,v in terms.items()}
        self.mains=[k[0] for k in self.keys if len(k)==1];self.pairs=[k for k in self.keys if len(k)==2]

    def _validate(self,X):
        X=np.asarray(X,float)
        if X.ndim!=2 or X.shape[1]!=self.n_features or np.isinf(X).any():raise ValueError('invalid prediction matrix')
        return X

    def _indicators(self,X,ids=None):
        ids=np.arange(len(self.features)) if ids is None else np.asarray(ids)
        ans=np.ones((len(X),len(ids),self.max_order))
        for axis in range(self.max_order):
            f=self.features[ids,axis];valid=f>=0
            ans[:,valid,axis]=region_indicator(X[:,f[valid]],self.low[ids[valid],axis],self.high[ids[valid],axis],self.missing[ids[valid],axis])
        return ans

    def _order_weights(self,X):
        delta=self._indicators(X)-self.mass
        w=np.zeros((len(X),len(self.features),self.max_order+1));w[:,:,0]=1.
        for axis in range(self.max_order):
            for r in range(axis+1,0,-1):
                w[:,:,r]=w[:,:,r]*self.mass[:,axis]+w[:,:,r-1]*delta[:,:,axis]
            w[:,:,0]*=self.mass[:,axis]
        return w

    def predict(self,X,times=None):
        X=self._validate(X);c,mu=self._coeff(times);out=np.empty((len(X),*mu.shape))
        for start in range(0,len(X),128):
            ix=slice(start,start+128);w=self._order_weights(X[ix])[:,:,1:].sum(2)
            out[ix]=mu+(w@c.reshape(len(c),-1)).reshape(-1,*mu.shape)
        return out

    def predict_by_order(self,X,times=None):
        """Return baseline and every order, including signed higher-order terms."""
        X=self._validate(X);c,mu=self._coeff(times)
        out=np.empty((len(X),self.max_order+1,*mu.shape));out[:,0]=mu
        for start in range(0,len(X),128):
            ix=slice(start,start+128);w=self._order_weights(X[ix])
            for r in range(1,self.max_order+1):out[ix,r]=(w[:,:,r]@c.reshape(len(c),-1)).reshape(-1,*mu.shape)
        return out

    def component(self,X,key,times=None):
        X=self._validate(X);key=tuple(key)
        if not key or tuple(sorted(set(key)))!=key or len(key)>4 or any(not isinstance(j,(int,np.integer)) or j<0 or j>=self.n_features for j in key):
            raise ValueError('component requires one to four sorted distinct valid feature indices')
        times=self.grid if times is None else np.asarray(times,float)
        if times.ndim!=1:raise ValueError('times must be a vector')
        shape=(len(X),len(times),self.coefficients.shape[-1])
        if key not in self._terms:return np.zeros(shape)
        ids,axes=self._terms[key];z=self._indicators(X,ids);selected=np.zeros((len(ids),self.max_order),bool)
        selected[np.arange(len(ids))[:,None],axes]=True
        w=np.where(selected[None],z-self.mass[ids],self.mass[ids]).prod(2)
        ti=np.searchsorted(self.grid,times,side='right')-1
        c=self.coefficients[ids][:,np.maximum(ti,0)].copy();c[:,ti<0]=0.
        return (w@c.reshape(len(ids),-1)).reshape(shape)

    def cuts(self,key,j):
        key=tuple(key)
        if j not in key:raise ValueError('cut coordinate must belong to the component')
        if key not in self._terms:return np.array([])
        ids,_=self._terms[key];axis=np.argmax(self.features[ids]==j,axis=1)
        values=np.r_[self.low[ids,axis],self.high[ids,axis]]
        return np.unique(values[np.isfinite(values)])

    def complexity(self,reference,support_min=10):
        reference=self._validate(reference)
        if not len(reference):raise ValueError('nonempty support reference required')
        records=[]
        for key in self.keys:
            cuts=[self.cuts(key,j) for j in key];shape=tuple(len(c)+2 for c in cuts)
            bins=[];prob=np.array(1.)
            for j,cc in zip(key,cuts):
                b=np.searchsorted(cc,reference[:,j],side='left');b[np.isnan(reference[:,j])]=len(cc)+1;bins.append(b)
                margin=np.bincount(b,minlength=len(cc)+2)/len(reference);prob=np.multiply.outer(prob,margin)
            support=np.bincount(np.ravel_multi_index(tuple(bins),shape),minlength=int(np.prod(shape))).reshape(shape)
            nonzero=False;buffer=[]
            for values in product(*[self.representatives(c) for c in cuts]):
                buffer.append(values)
                if len(buffer)==128:
                    xx=np.zeros((len(buffer),self.n_features));xx[:,list(key)]=buffer
                    nonzero=bool(np.any(abs(self.component(xx,key))>1e-14));buffer=[]
                    if nonzero:break
            if buffer and not nonzero:
                xx=np.zeros((len(buffer),self.n_features));xx[:,list(key)]=buffer
                nonzero=bool(np.any(abs(self.component(xx,key))>1e-14))
            records.append(dict(features=list(key),order=len(key),cells=int(np.prod(shape)),nonzero=nonzero,
                low_support_cells=int((support<support_min).sum()),zero_support_cells=int((support==0).sum()),
                low_support_product_mass=float(prob[support<support_min].sum())))
        cells=sum(r['cells'] for r in records)
        result=dict(spatial_table_cells=cells,scalar_table_cells=cells*len(self.grid)*self.coefficients.shape[-1],
            factored_leaf_coefficients=int(self.coefficients.size),low_support_cell_fraction=sum(r['low_support_cells'] for r in records)/max(cells,1),
            zero_support_cell_fraction=sum(r['zero_support_cells'] for r in records)/max(cells,1),support_threshold=support_min,functions=records)
        for order in range(1,5):
            rr=[r for r in records if r['order']==order]
            result[f'order{order}_functions']=sum(r['nonzero'] for r in rr)
            result[f'order{order}_structural_functions']=len(rr)
            result[f'order{order}_spatial_cells']=sum(r['cells'] for r in rr)
        result.update(main_functions=result['order1_functions'],interaction_functions=result['order2_functions'])
        return result

    def save(self,path):
        with Path(path).open('xb') as f:
            np.savez_compressed(f,format_version=np.array('higher_order_v1'),grid=self.grid,features=self.features,
                low=self.low,high=self.high,missing=self.missing,mass=self.mass,coefficients=self.coefficients,mu=self.mu,
                n_features=np.array(self.n_features),reference_n=np.array(self.reference_n))

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as a:
            if str(a['format_version'])!='higher_order_v1':raise ValueError('unknown expansion format')
        return super().load(path)
