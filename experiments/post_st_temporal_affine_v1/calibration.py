"""Convex affine CIF calibration, optionally anchored to the preceding period.

The temporal penalty is diagonal in parameters with probability-scale units:
tau/n * [||c-cprev||² + sum(s_k²*(b_k-bprev_k)²)],
s_k=max(.05,RMS of current historical OOF risk). It is not extra outcome data.
"""
import numpy as np
from itertools import combinations
from scipy.optimize import minimize,linprog
from experiments.post_st_level_coverage_v1.affine import hull

def _polish(z,H,linear,allG,allh,fun):
    active=np.flatnonzero(abs(allG@z-allh)<1e-7)
    # Also polish an interior optimum: an empty active set is the ordinary
    # unconstrained quadratic system, not a reason to skip refinement.
    # A nearly binding hull facet need not be truly active. In four dimensions
    # enumerate the local independent active subsets (at most four faces),
    # rather than force every near-zero slack to equality. All proposals must
    # satisfy the full original constraints and the original objective.
    polished=[]
    for size in range(min(4,len(active))+1):
        for subset in combinations(active,size):
            ix=np.asarray(subset,dtype=int);A=allG[ix]
            K=np.block([[H,A.T],[A,np.zeros((len(ix),len(ix)))]])
            q=np.linalg.lstsq(K,np.r_[-linear,allh[ix]],rcond=None)[0][:4]
            if np.max(allG@q-allh)<=1e-11:polished.append(q)
    if polished:
        q=min(polished,key=fun)
        if fun(q)<=fun(z)+1e-13:z=q
    return z

def fit(p,y,vertices,unique_events,previous=None,temporal_companies=20.,identity_companies=20.):
    p=np.asarray(p,float);y=np.asarray(y,float);vertices=hull(vertices);events=np.asarray(unique_events)
    if p.ndim!=2 or p.shape[1]!=2 or y.shape!=p.shape or not len(p):raise ValueError('Nonempty two-cause arrays required')
    if not np.isfinite(p).all() or (p<0).any() or (p.sum(1)>1+1e-12).any():raise ValueError('Illegal CIF')
    if not np.isin(y,[0,1]).all() or (y.sum(1)>1).any():raise ValueError('Invalid competing labels')
    if events.shape!=(2,) or (events<0).any() or (events>y.sum(0)).any() or (events!=np.floor(events)).any():raise ValueError('Invalid distinct event support')
    if identity_companies<0 or temporal_companies<0:raise ValueError('Nonnegative penalties required')
    n=len(p);identity=np.array([0.,0.,1.,1.]);lam=identity_companies/n
    target=identity if previous is None else np.r_[previous['c'],previous['b']].astype(float)
    if target.shape!=(4,) or not np.isfinite(target).all():raise ValueError('Invalid preceding parameters')
    tau=0. if previous is None else temporal_companies/n
    scale=np.r_[1.,1.,np.maximum(.05,np.sqrt(np.mean(p*p,axis=0)))];P=np.diag(scale*scale)
    design=np.zeros((n,2,4));design[:,0,0]=1;design[:,1,1]=1;design[:,0,2]=p[:,0];design[:,1,3]=p[:,1]
    D=design.reshape(-1,4);yy=y.ravel();original=p.ravel()
    H=2*(1+lam)*(D.T@D)/n+2*tau*P
    linear=-2*D.T@(yy+lam*original)/n-2*tau*P@target
    fun=lambda z:float((np.sum((D@z-yy)**2)+lam*np.sum((D@z-original)**2))/n+tau*((z-target)@P@(z-target)))
    jac=lambda z:H@z+linear
    lo=np.array([0.,0.,.5,.5]);hi=np.array([1.,1.,4.,4.]);fixed=events<10;lo[2:][fixed]=1.;hi[2:][fixed]=1.
    G=np.c_[np.ones((len(vertices),2)),vertices];h=np.ones(len(vertices));assert np.max(G@identity-h)<=1e-12
    opt=minimize(fun,identity,jac=jac,bounds=list(zip(lo,hi)),constraints=[dict(type='ineq',fun=lambda z:h-G@z,jac=lambda z:-G)],method='SLSQP',options=dict(ftol=1e-14,maxiter=2000))
    z=opt.x;allG=np.vstack([G,-np.eye(4),np.eye(4)]);allh=np.r_[h,-lo,hi]
    z=_polish(z,H,linear,allG,allh,fun)
    for j in range(4):
        if abs(z[j]-lo[j])<1e-12:z[j]=lo[j]
        if abs(z[j]-hi[j])<1e-12:z[j]=hi[j]
    oracle=linprog(jac(z),A_ub=G,b_ub=h,bounds=list(zip(lo,hi)),method='highs')
    if not oracle.success:raise RuntimeError('No independent linear-oracle certificate')
    gap=float(jac(z)@(z-oracle.x));violation=float(max(0,np.max(allG@z-allh)))
    if gap>1e-8 or gap< -1e-10 or violation>1e-10 or fun(z)>fun(identity)+1e-12:raise RuntimeError(f'Calibration failed certificate gap={gap} violation={violation}')
    data_loss=float(np.mean(np.sum((D@z-yy).reshape(-1,2)**2,axis=1)))
    return dict(c=z[:2].tolist(),b=z[2:].tolist(),gap=gap,primal_violation=violation,iterations=int(opt.nit),optimizer_success=bool(opt.success),solver_message=str(opt.message),
        objective_before=fun(identity),objective_after=fun(z),data_loss=data_loss,
        identity_penalty=float(lam*np.sum((D@z-original)**2)/n),temporal_penalty=float(tau*((z-target)@P@(z-target))),
        frozen_slope_for_sparse_cause=fixed.tolist(),unique_event_counts=events.astype(int).tolist(),resampled_event_counts=y.sum(0).astype(int).tolist(),
        identity_companies=float(identity_companies),temporal_companies=float(temporal_companies if previous is not None else 0),parameter_scales=scale.tolist(),previous_parameters=target.tolist(),
        certificate_vertices=len(vertices),maximum_terminal_sum_bound=float((G@z).max()),lower=lo.tolist(),upper=hi.tolist())
