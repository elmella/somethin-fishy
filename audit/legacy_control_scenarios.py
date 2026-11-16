"""Replay the pinned notebook's one active control update, then its state."""
import json
import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from diagnostics import legacy_functions
from compare_visuals import sample
from scenario_experiments import CASES,problem_for
from pontryagin_controller import OUT


def replay(case):
    dest=OUT/'scenarios';path=dest/(case+'_legacy_control.npz')
    if path.exists():return dict(np.load(path))
    p=problem_for(case);N,K=p.N,p.K;steps=200
    ns=legacy_functions();deadline=time.monotonic()+90
    def bounded_solve(fun,*args,**kwargs):
        def bounded(t,y):
            if time.monotonic()>deadline:raise RuntimeError('Legacy replay exceeded 90 seconds')
            return fun(t,y)
        return solve_ivp(bounded,*args,**kwargs)
    ns['solve_ivp']=bounded_solve
    U=np.ones((steps,N,K+2));initial=U.copy()
    params=[p.alpha,p.sigma,p.beta,p.rate,p.grounds,np.array([100.,0.]),2*N,K,p.ships.shape]
    adjparams=[p.alpha,p.sigma,p.beta,p.epsilon,p.rate,p.grounds,np.array([100.,0.]),2*N,K,p.ships.shape]
    y0=np.r_[p.ships.ravel(),np.full(K,p.F0)]
    state=ns['solve_state_equations'](ns['state'],U,params,p.T,y0,steps)
    backward=ns['solve_costate_equations'](ns['costate'],state,U,adjparams,p.T,np.r_[np.zeros(2*N),np.full(K,-p.delta)],2*N,K,steps)
    if not backward.success or backward.y.shape[1]!=steps:raise RuntimeError(backward.message)
    pv=backward.y[:2*N,::-1].reshape(N,2,-1);x=state.y[:2*N].reshape(N,2,-1);F=state.y[2*N:]
    failures=0
    for t in range(steps):
        D=ns['d_matrix'](x[:,:,t],p.grounds);G=ns['ship_ground_gaussians'](x[:,:,t],p.grounds,p.alpha)
        for boat in range(N):
            def objective(candidate):
                force=np.sum(candidate[:-2,None]*G[boat,:,None]*D[boat]*F[:,t,None],axis=0)
                cross=pv[boat,:,t]@np.array([force[1],-force[0]])
                return cross**2+(0 if t==0 else 100*np.sum((U[t-1,boat]-candidate)**2))
            result=minimize(objective,U[t,boat],bounds=[(0,10)]*(K+2))
            U[t,boat]=result.x;failures+=int(not result.success)
    # Unlike the old plot, replay after the update so it actually shows its effect.
    state=ns['solve_state_equations'](ns['state'],U,params,p.T,y0,steps)
    if not state.success:raise RuntimeError(state.message)
    times=np.linspace(0,p.T,16001);run=sample(state,U,p,times)
    data=dict(times=times,**{k:run[k] for k in ['x','fish','catch','speed','travel']})
    np.savez_compressed(path,**data)
    metadata=dict(source_commit='23043fe5525d3b27dc0401f733fc8a9ec169ca26',source='random/updated_sam_fishin.ipynb, active cell 23',
                  control_updates=1,max_control_change=float(np.max(abs(U-initial))),optimizer_failures=failures,
                  harvested=float(run['catch'][-1]),note='Original surrogate, terminal sign, bounds and inactive speed assignment retained. Post-update state replayed instead of stale pre-update plot. Scenario geometry and T adapted; coefficients held equal across comparisons.')
    path.with_suffix('.json').write_text(json.dumps(metadata,indent=2))
    print(case,metadata,flush=True)
    return data


if __name__=='__main__':
    for case in CASES:replay(case)
