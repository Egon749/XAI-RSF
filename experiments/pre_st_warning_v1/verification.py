"""Same exhaustive checks with read-only caching of fixed time coefficients.

Caching only avoids rebuilding identical coefficient arrays for every function
and every row batch; it changes no forest, reference mass, term or prediction.
"""
from pathlib import Path
import time
import numpy as np
from crforest import CIFExpansion,CompetingRiskForest
from experiments.post_st_prognosis_v1.common import write

class CachedExpansion(CIFExpansion):
    def _coeff(self,times):
        key=tuple(self.grid if times is None else np.asarray(times,float))
        if not hasattr(self,'_coefficient_cache'):self._coefficient_cache={}
        if key not in self._coefficient_cache:
            value=super()._coeff(times)
            for x in value:x.setflags(write=False)
            self._coefficient_cache[key]=value
        return self._coefficient_cache[key]

def verify(f,tr,ev,out):
    out.mkdir();start=time.perf_counter();ex=CachedExpansion.from_forest(f,tr['X'])
    compile_seconds=time.perf_counter()-start
    ex.save(out/'functions_PRIVATE.npz');again=CachedExpansion.load(out/'functions_PRIVATE.npz')
    # Ordinary class loading must also produce exactly the same representation.
    ordinary=CIFExpansion.load(out/'functions_PRIVATE.npz')
    f.save(out/'forest_PRIVATE.json');loaded=CompetingRiskForest.load(out/'forest_PRIVATE.json')
    X=np.vstack([tr['X'],ev['X']]);times=np.unique(np.r_[0.,f.grid_,365.,366.,730.,731.]);keys=[(j,) for j in ex.mains]+ex.pairs
    errors=dict(full_reconstruction=0.,explicit_component_sum=0.,function_reload=0.,forest_reload=0.,ordinary_cache_agreement=0.)
    minimum=0.;maximum=0.;increment=0.;start=time.perf_counter()
    for begin in range(0,len(X),128):
        xx=X[begin:begin+128];pred=f.predict(xx,times)
        values=ex.predict(xx,times);original=ordinary.predict(xx,times)
        errors['ordinary_cache_agreement']=max(errors['ordinary_cache_agreement'],float(abs(values-original).max()))
        errors['forest_reload']=max(errors['forest_reload'],float(abs(loaded.predict(xx,times)-pred).max()))
        errors['full_reconstruction']=max(errors['full_reconstruction'],float(abs(values-pred).max()))
        errors['function_reload']=max(errors['function_reload'],float(abs(again.predict(xx,times)-pred).max()))
        total=np.broadcast_to(ex._coeff(times)[1],pred.shape).copy()
        for key in keys:total+=ex.component(xx,key,times)
        errors['explicit_component_sum']=max(errors['explicit_component_sum'],float(abs(total-pred).max()))
        minimum=min(minimum,float(pred.min()));maximum=max(maximum,float(pred.sum(-1).max()));increment=min(increment,float(np.diff(pred,axis=1).min()))
    assert max(errors.values())<=1e-10 and minimum>=-1e-12 and maximum<=1+1e-10 and increment>=-1e-12
    prediction_times=np.array([365.,366.,730.,731.]);p=f.predict(ev['X'],prediction_times)
    np.savez_compressed(out/'predictions_PRIVATE.npz',times=prediction_times,sample_ids=ev['sample_ids'],cif=p)
    complexity=ex.complexity(tr['X'],support_min=10);write(out/'complexity_PRIVATE.json',complexity)
    summary=dict(**errors,**f.audit(),**{k:v for k,v in complexity.items() if k!='functions'},verified_rows=len(X),verified_times=len(times),native_times=len(f.grid_),
        probability_min=minimum,probability_sum_max=maximum,minimum_increment=increment,compile_seconds=compile_seconds,
        verification_seconds=time.perf_counter()-start,function_bytes=(out/'functions_PRIVATE.npz').stat().st_size,candidate_attempts=f.candidate_evaluations_)
    write(out/'verification.json',summary);return summary
