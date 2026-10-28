"""Constrained affine calibration, closed under exact order-two CIF expansion.

For every input probability vector in the full simplex, c>=0 and
sum(c)+max(b)<=1 guarantee a valid transformed vector. The time intercept
is c_k*g_k(t), with a fixed nondecreasing training-only curve g in [0,1].
No test observations, probability clipping, or forest fitting occur here.
"""
import numpy as np
from scipy.optimize import linprog
from itertools import combinations

B_MIN=1e-6
A=np.array([[1.,1.,1.,0.],[1.,1.,0.,1.]])


def fit_affine(p,y,w):
    p=np.asarray(p,float);y=np.asarray(y,float);w=np.asarray(w,float)
    if p.ndim!=2 or p.shape[1]!=2 or y.shape!=p.shape or w.shape!=(len(p),):
        raise ValueError('Two compatible causes and one weight per company required')
    if not np.isfinite(p).all() or not np.isfinite(y).all() or not np.isfinite(w).all() or (w<0).any() or w.sum()<=0:
        raise ValueError('Finite data and positive observed endpoint weight required')
    if (p<0).any() or (p.sum(1)>1+1e-12).any() or not np.isin(y,[0,1]).all() or (y.sum(1)>1).any():
        raise ValueError('Invalid competing-risk endpoint data')
    def value(z):
        residual=z[:2]+p*z[2:]-y
        return float(np.sum(w[:,None]*residual**2)/len(p))
    def jac(z):
        residual=z[:2]+p*z[2:]-y
        return 2*np.r_[np.sum(w[:,None]*residual,0),np.sum(w[:,None]*residual*p,0)]/len(p)
    initial=np.array([0.,0.,1.,1.]);lower=np.array([0.,0.,B_MIN,B_MIN])
    # Four-dimensional convex QP: enumerate all <=4 independent active
    # inequalities. 57 small KKT systems avoid optimizer success flags that
    # do not certify gradient accuracy. The objective/constraints are unchanged.
    H=np.zeros((4,4));H[:2,:2]=np.eye(2)*2*w.sum()/len(p)
    H[2:,2:]=np.diag(2*np.sum(w[:,None]*p*p,0)/len(p))
    H[:2,2:]=H[2:,:2]=np.diag(2*np.sum(w[:,None]*p,0)/len(p))
    q=jac(np.zeros(4));G=np.vstack([-np.eye(4),A]);h=np.r_[-lower,[1.,1.]]
    candidates=[];systems=0
    for nactive in range(5):
        for active in combinations(range(6),nactive):
            systems+=1;ga=G[list(active)];ha=h[list(active)]
            K=np.block([[H,ga.T],[ga,np.zeros((nactive,nactive))]])
            rhs=np.r_[-q,ha];sol=np.linalg.lstsq(K,rhs,rcond=None)[0]
            if np.max(abs(K@sol-rhs))>1e-10:continue
            z=sol[:4].copy()
            # An active lower-bound equality has its exact declared value.
            for j in active:
                if j<4:z[j]=lower[j]
            if (z<lower).any() or np.max(G@z-h)>1e-10:continue
            if nactive and sol[4:].min() < -1e-10:continue
            candidates.append((value(z),z))
    if not candidates:raise RuntimeError('No feasible KKT point found for convex affine QP')
    _,z=min(candidates,key=lambda item:item[0])
    violation=float(max(0.,np.max(A@z-1),np.max(lower-z),np.max(z-1)))
    linear=linprog(jac(z),A_ub=A,b_ub=np.ones(2),bounds=list(zip(lower,np.ones(4))),method='highs')
    if not linear.success:raise RuntimeError('Unable to certify convex optimization gap')
    gap=float(jac(z)@(z-linear.x))
    if violation>1e-10 or abs(min(gap,0))>1e-10 or gap>1e-8:
        raise RuntimeError(f'Affine fit failed: feasibility={violation}; gap={gap}')
    if value(z)>value(initial)+1e-10:raise RuntimeError('Constrained solution worse than feasible identity')
    return dict(c=z[:2].tolist(),b=z[2:].tolist(),iterations=systems,solver='enumerated_convex_KKT',success=True,
        objective_before=value(initial),objective_after=value(z),gap=gap,primal_violation=violation,
        slope_floor_active=bool(np.any(z[2:]<=B_MIN+1e-8)),
        simplex_constraint_active=bool(np.any(A@z>=1-1e-8)),
        identity_change_max=float(np.max(abs(z-initial))))


def transform(cif,g,cal):
    return np.asarray(g)*np.asarray(cal['c'])+np.asarray(cif)*np.asarray(cal['b'])


def shape_at(times,grid,values):
    """Training AJ normalized at day 365; constant thereafter (12m=365/366d).

    This curve is an assumed intercept shape, not evidence of calibration at
    intermediate times. At every individual 12-calendar-month horizon g=1.
    """
    times=np.asarray(times,float);grid=np.asarray(grid,float);values=np.asarray(values,float)
    denominator=values[np.searchsorted(grid,365.,side='right')-1]
    if (denominator<=0).any():raise ValueError('Both training causes need support by day 365')
    ix=np.searchsorted(grid,np.minimum(times,365.),side='right')-1
    out=values[np.maximum(ix,0)]/denominator
    out[ix<0]=0.
    return out
