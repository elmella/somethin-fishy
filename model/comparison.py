"""Sample fleet positions and fish stocks under two steering policies."""
from pathlib import Path
import numpy as np
from regularized_pmp import radial_flow
from fishing import scenario
from scenario_experiments import CASES, problem_for, policy_name
from pontryagin_controller import load_saved

RESULTS=Path(__file__).resolve().parents[1]/'results'
MAPS={**CASES, 'dispersed_400':('Dispersed fleet · 400 s','5 boats · 7 grounds · dispersed starts'),
      'dispersed_600':('Dispersed fleet · 600 s','5 boats · 7 grounds · dispersed starts')}


def setting(case,beta):
    if case in CASES:
        return problem_for(case,beta),policy_name(case,beta)+'_verified'
    t=int(case.split('_')[-1])
    p=scenario(T=t);p.beta=beta
    return p,f'coverage_{t}'+('_beta0' if beta==0 else '')+'_verified'


def equal_attraction_target(x,logfish,p):
    """Weighted ground centroid for unit attraction at every ground.

    At a ground center take the continuous limit (that ground's position).
    """
    q=p.grounds[None]-x[:,None]
    d2=np.sum(q*q,axis=-1)
    centered=(d2==0).any(axis=1)
    logweight=logfish[None]-p.alpha*d2-.5*np.log(np.where(d2==0,1.,d2))
    w=np.exp(logweight-logweight.max(axis=1,keepdims=True))
    displacement=np.sum(w[...,None]*q,axis=1)/w.sum(axis=1,keepdims=True)
    displacement[centered]=0
    return x+displacement


def rollout(solver,control=None):
    """Sample at half seconds. An unplanned boat chooses a target every second;
    an optimized boat follows its saved knots. Integrate both with radial flow
    and the same 10-point exposure quadrature, splitting at all control changes.
    """
    p=solver.p
    times=np.arange(0,p.T+.25,.5)
    knots=np.arange(0,p.T+1) if control is None else np.linspace(0,p.T,solver.steps+1)
    if control is not None:
        _,_,targets,fractions=solver.unpack(control)
    x=p.ships.copy();lf=np.full(p.K,np.log(p.F0));distance=0.
    xs=[x.copy()];stocks=[np.exp(lf)];distances=[0.]
    boundaries=np.union1d(knots,times)
    for t,end in zip(boundaries[:-1],boundaries[1:]):
        k=np.searchsorted(knots,t,side='right')-1
        if t==knots[k]:
            if control is None:
                z=equal_attraction_target(x,lf,p);a=np.ones(p.N)
            else:
                z=targets[k];a=fractions[k]
        h=end-t
        nodes=radial_flow(x[:,None],z[:,None],a[:,None],h*solver.nodes,p.V,solver.eta)[0]
        diff=nodes[:,:,None]-p.grounds
        exposure=np.einsum('iqj,q->j',np.exp(-p.sigma*np.sum(diff*diff,axis=-1)),solver.weights)
        lf-=p.rate*h*exposure
        nxt=radial_flow(x,z,a,h,p.V,solver.eta)[0]
        distance+=np.linalg.norm(nxt-x,axis=-1).sum();x=nxt
        output_index=np.searchsorted(times,end)
        if output_index<len(times) and times[output_index]==end:
            xs.append(x.copy());stocks.append(np.exp(lf));distances.append(distance)
    stocks=np.array(stocks)
    return dict(times=times,x=np.array(xs),fish=stocks,catch=p.K*p.F0-stocks.sum(axis=1),distance=distances)


def comparison(case):
    betas=[5,0] if case!='dispersed_600' else [5]
    runs=[]
    for beta in betas:
        p,name=setting(case,beta);solver,u,report=load_saved(name,p)
        if not runs:runs.append(('Without optimal control',rollout(solver),None))
        run=rollout(solver,u)
        assert abs(run['catch'][-1]-report['harvested'])<1e-7
        assert abs(run['distance'][-1]-sum(report['distance_per_boat']))<1e-6
        runs.append((f'Optimal control · β = {beta}',run,report))
    return p,runs
