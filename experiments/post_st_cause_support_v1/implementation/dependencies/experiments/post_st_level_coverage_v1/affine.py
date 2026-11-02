"""Affine CIF updates, with a certificate for ALL inputs, not only observed rows.

The Minkowski sum of the terminal leaf convex hulls bounds every possible
forest prediction, including incompatible cross-tree leaf combinations. A
nonnegative offset curve, positive time-constant slopes and this terminal
polytope certificate guarantee valid two-cause CIFs for every x and time.
"""
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, linprog

def hull(points):
    x=np.unique(np.asarray(points,float),axis=0)
    if len(x)<=2:return x
    if np.linalg.matrix_rank(x-x[0],tol=1e-12)<2:
        direction=x[np.argmax(np.linalg.norm(x-x[0],axis=1))]-x[0]
        z=x@direction
        return x[[np.argmin(z),np.argmax(z)]]
    return x[ConvexHull(x).vertices]

def forest_envelope(forest):
    vertices=np.zeros((1,2)); count=len(forest.trees_)
    for tree in forest.trees_:
        for node in tree:
            if 'cif' not in node:continue
            values=np.asarray(node['cif'])
            if not np.isfinite(values).all() or values.min()< -1e-12 or values.sum(1).max()>1+1e-12 or (len(values)>1 and np.diff(values,axis=0).min()< -1e-12):
                raise ValueError('Invalid native leaf CIF')
        leaf=hull([np.asarray(n['cif'])[-1] for n in tree if 'cif' in n])
        vertices=hull((vertices[:,None,:]+leaf[None,:,:]/count).reshape(-1,2))
    return vertices

def at(times,grid,values):
    ix=np.searchsorted(grid,np.asarray(times,float),side='right')-1
    result=np.asarray(values)[np.maximum(ix,0)].copy();result[ix<0]=0.
    return result

def transform(pred,times,cal):
    g=at(times,cal['shape_grid'],cal['shape_values'])
    return np.asarray(pred)*np.asarray(cal['b'])+g[...,None]*np.asarray(cal['c'])

def transformed_expansion(original,cal):
    from copy import deepcopy
    obj=deepcopy(original)
    grid=np.unique(np.r_[original.grid,cal['shape_grid']])
    coefficients,mu=original._coeff(grid)
    obj.grid=grid;obj.coefficients=coefficients*np.asarray(cal['b'])
    obj.mu=mu*np.asarray(cal['b'])+at(grid,cal['shape_grid'],cal['shape_values'])[:,None]*np.asarray(cal['c'])
    return obj

def fit_affine(p,y,vertices,event_counts,prior_companies=20.):
    p=np.asarray(p,float);y=np.asarray(y,float);vertices=hull(vertices)
    if p.ndim!=2 or p.shape[1]!=2 or p.shape!=y.shape or len(p)==0:
        raise ValueError('Compatible nonempty two-cause arrays required')
    if not np.isfinite(p).all() or not np.isfinite(y).all() or (p<0).any() or (p.sum(1)>1+1e-12).any():
        raise ValueError('Invalid predictions')
    if not np.isin(y,[0.,1.]).all() or (y.sum(1)>1).any():raise ValueError('Competing outcomes required')
    if not np.array_equal(y.sum(0),event_counts):raise ValueError('Event support mismatch')
    # Penalize the *probability change* toward identity, equivalent to20 pseudo-companies.
    # This is frozen regularization, not a result-selected penalty.
    n=len(p);identity=np.array([0.,0.,1.,1.]);lam=prior_companies/n
    design=np.zeros((n,2,4));design[:,0,0]=1;design[:,1,1]=1
    design[:,0,2]=p[:,0];design[:,1,3]=p[:,1]
    D=design.reshape(-1,4);yy=y.ravel();original=p.ravel()
    H=2*(1+lam)*(D.T@D)/n
    linear=-2*D.T@(yy+lam*original)/n
    fun=lambda z:float((np.sum((D@z-yy)**2)+lam*np.sum((D@z-original)**2))/n)
    jac=lambda z:H@z+linear
    lower=np.array([0.,0.,.5,.5]);upper=np.array([1.,1.,4.,4.])
    fixed=np.asarray(event_counts)<10
    lower[2:][fixed]=1.;upper[2:][fixed]=1.
    G=np.c_[np.ones((len(vertices),2)),vertices];h=np.ones(len(vertices))
    assert np.max(G@identity-h)<=1e-12
    opt=minimize(fun,identity,jac=jac,bounds=list(zip(lower,upper)),
        constraints=[dict(type='ineq',fun=lambda z:h-G@z,jac=lambda z:-G)],
        method='SLSQP',options=dict(ftol=1e-14,maxiter=2000))
    z=opt.x
    # KKT polishing uses the active constraints, then a linear-oracle gap certifies
    # the convex objective; optimizer success alone is not accepted.
    allG=np.vstack([G,-np.eye(4),np.eye(4)]);allh=np.r_[h,-lower,upper]
    active=np.flatnonzero(abs(allG@z-allh)<1e-7)
    if len(active):
        A=allG[active];b=allh[active]
        K=np.block([[H,A.T],[A,np.zeros((len(active),len(active)))]])
        sol=np.linalg.lstsq(K,np.r_[-linear,b],rcond=None)[0][:4]
        if np.max(allG@sol-allh)<=1e-11 and fun(sol)<=fun(z)+1e-13:z=sol
    # Exact declared active bounds remove only solver roundoff, never clip CIFs.
    for j in range(4):
        if abs(z[j]-lower[j])<1e-12:z[j]=lower[j]
        if abs(z[j]-upper[j])<1e-12:z[j]=upper[j]
    lp=linprog(jac(z),A_ub=G,b_ub=h,bounds=list(zip(lower,upper)),method='highs')
    if not lp.success:raise RuntimeError('Cannot certify affine optimum')
    gap=float(jac(z)@(z-lp.x));violation=float(max(0,np.max(allG@z-allh)))
    if gap>1e-8 or gap< -1e-10 or violation>1e-10 or fun(z)>fun(identity)+1e-12:
        raise RuntimeError(f'Uncertified affine optimum: gap={gap}, violation={violation}')
    return dict(c=z[:2].tolist(),b=z[2:].tolist(),gap=gap,primal_violation=violation,
        objective_before=fun(identity),objective_after=fun(z),iterations=int(opt.nit),
        optimizer_success=bool(opt.success),solver_message=str(opt.message),
        frozen_slope_for_sparse_cause=fixed.tolist(),event_counts=list(map(int,event_counts)),
        prior_companies=float(prior_companies),certificate_vertices=len(vertices),
        maximum_terminal_sum_bound=float((G@z).max()),lower=lower.tolist(),upper=upper.tolist())
