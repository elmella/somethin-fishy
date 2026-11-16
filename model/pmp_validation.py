"""Check discrete adjoint trajectories against the physical continuous equations."""
import time
import numpy as np
from scipy.integrate import solve_ivp
from fishing import stable_rhs, running_cost

# At almost stationary ground arrivals, direction is dominated by coordinate
# roundoff. Execute a feasible stop instead of normalizing a ~1e-12 velocity.
# This bounds the velocity approximation by 1e-10; it is not a new cost term.
CONTROL_SPEED_ZERO_TOLERANCE=1e-10
CONTROL_VELOCITY_RECONSTRUCTION_TOLERANCE=1e-10


def reconstruct_attractions(x,fish,velocity,problem):
    """Well-conditioned one/two-ground controls for a feasible physical velocity."""
    pair_a,pair_b=np.triu_indices(problem.K,1)
    speed=np.linalg.norm(velocity,axis=-1)
    speed=np.where(speed<=CONTROL_SPEED_ZERO_TOLERANCE,0.,speed)
    heading=np.divide(velocity,speed[:,None],out=np.zeros_like(velocity),where=speed[:,None]>0)
    rho=np.linalg.norm(problem.grounds[None]-x[:,None],axis=-1)
    if rho.min()==0:raise ValueError('Trajectory hits a ground-center singularity.')
    d=(problem.grounds[None]-x[:,None])/rho[...,None]
    a=d[:,pair_a];b=d[:,pair_b];h=heading[:,None]
    det=a[:,:,0]*b[:,:,1]-a[:,:,1]*b[:,:,0]
    numerator=np.stack((h[:,:,0]*b[:,:,1]-h[:,:,1]*b[:,:,0],a[:,:,0]*h[:,:,1]-a[:,:,1]*h[:,:,0]),axis=-1)
    all_weights=np.divide(numerator,det[...,None],out=np.full_like(numerator,np.nan),where=np.abs(det[...,None])>1e-12)
    # Assess clipped, nonnegative controls by their actual velocity error.
    # An angle/weight tolerance alone is ill-conditioned near a stopped hull
    # arrival, where subtraction roundoff can put the heading just outside it.
    clipped=np.maximum(all_weights,0)
    force=clipped[:,:,:1]*a+clipped[:,:,1:]*b
    norm=np.linalg.norm(force,axis=-1)
    direction=np.divide(force,norm[...,None],out=np.zeros_like(force),where=norm[...,None]>0)
    error=np.linalg.norm(speed[:,None,None]*direction-velocity[:,None],axis=-1)
    valid=(error<=CONTROL_VELOCITY_RECONSTRUCTION_TOLERANCE)&np.all(np.isfinite(all_weights),axis=-1)
    score=np.where(valid,clipped.sum(axis=-1),np.inf)
    selected=score.argmin(axis=-1)
    if np.any(~np.isfinite(score[np.arange(problem.N),selected])):raise ValueError('No admissible two-ray representation.')
    pairs=np.stack((pair_a[selected],pair_b[selected]),axis=-1)
    weights=np.maximum(all_weights[np.arange(problem.N),selected],0)
    stopped=speed==0
    weights[stopped]=[1.,0.]
    logs=np.log(np.maximum(weights,1e-300))+problem.alpha*np.take_along_axis(rho**2,pairs,axis=1)-np.log(fish[pairs])
    values=10*np.exp(logs-logs.max(axis=1,keepdims=True));values[weights==0]=0
    u=np.zeros((problem.N,problem.K));np.put_along_axis(u,pairs,values,axis=1)
    return u,speed


class RegularizedSchedule:
    """Sample a target schedule and check its physical realization."""
    def __init__(self,solver,control):
        self.solver=solver;self.p=solver.p;self.dt=solver.dt;self.steps=solver.steps
        self.state=solver.forward(control)
        exposure=self.dt*np.sum(self.state['C']*solver.weights[None,None,:,None],axis=(1,2))
        self.prefix=np.vstack((np.zeros(self.p.K),np.cumsum(exposure,axis=0)))

    def nominal(self,k,t):
        from regularized_pmp import radial_flow
        s=self.state;p=self.p;elapsed=np.clip(t-k*self.dt,0,self.dt)
        x=radial_flow(s['x'][k],s['target'][k],s['a'][k],elapsed,p.V,self.solver.eta)[0]
        nodes=radial_flow(s['x'][k,:,None],s['target'][k,:,None],s['a'][k,:,None],elapsed*self.solver.nodes[None],p.V,self.solver.eta)[0]
        C=np.exp(-p.sigma*np.sum((p.grounds[None,None]-nodes[:,:,None])**2,axis=-1))
        exposure=elapsed*np.sum(C*self.solver.weights[None,:,None],axis=(0,1))
        return x,p.F0*np.exp(-p.rate*(self.prefix[k]+exposure))

    def controls(self,k,t):
        x,fish=self.nominal(k,t);q=self.state['target'][k]-x
        v=self.state['a'][k,:,None]*self.p.V*q/np.sqrt(np.sum(q*q,axis=-1)+self.solver.eta**2)[:,None]
        return reconstruct_attractions(x,fish,v,self.p)

    def checks(self):
        p=self.p;s=self.state;q=p.grounds[None,None]-s['x'][:-1,:,None];d=s['dx']
        dd=np.sum(d*d,axis=-1)
        t=np.clip(np.divide(np.sum(q*d[:,:,None],axis=-1),dd[:,:,None],out=np.zeros(q.shape[:-1]),where=dd[:,:,None]>0),0,1)
        clearance=float(np.linalg.norm(q-t[...,None]*d[:,:,None],axis=-1).min());error=0.;min_u=10.;max_u=0.;max_speed=0.
        for k in range(self.steps):
            for fraction in [.1,.5,.9]:
                t=(k+fraction)*self.dt;x,F=self.nominal(k,t);u,v=self.controls(k,t)
                actual=stable_rhs(np.r_[x.ravel(),F],u,v,p)[:2*p.N].reshape(-1,2)
                q=s['target'][k]-x;expected=s['a'][k,:,None]*p.V*q/np.sqrt(np.sum(q*q,axis=-1)+self.solver.eta**2)[:,None]
                error=max(error,float(np.max(abs(actual-expected))));min_u=min(min_u,float(u.min()));max_u=max(max_u,float(u.max()));max_speed=max(max_speed,float(v.max()))
        return dict(minimum_center_clearance=clearance,max_velocity_reconstruction_error=error,attraction_min=min_u,attraction_max=max_u,speed_max=max_speed)


def replay_feedback(schedule,rtol=1e-8,max_step=.5):
    """Execute the target policy through the physical u/v equations.

    """
    p=schedule.p;s=schedule.state;solver=schedule.solver
    z=np.r_[p.ships.ravel(),np.full(p.K,p.F0),0.]
    deadline=time.monotonic()+55;max_position=0.;states=[z.copy()]
    for k in range(schedule.steps):
        def rhs(t,state):
            if time.monotonic()>deadline:raise RuntimeError('Physical feedback replay exceeded 55 seconds.')
            x=state[:2*p.N].reshape(p.N,2);F=state[2*p.N:2*p.N+p.K]
            q=s['target'][k]-x;velocity=s['a'][k,:,None]*p.V*q/np.sqrt(np.sum(q*q,axis=-1)+solver.eta**2)[:,None]
            u,v=reconstruct_attractions(x,F,velocity,p)
            return np.r_[stable_rhs(state[:-1],u,v,p),running_cost(state[:-1],v,p,include_self=False)]
        sol=solve_ivp(rhs,[k*schedule.dt,(k+1)*schedule.dt],z,method='DOP853',rtol=rtol,atol=rtol*1e-3,max_step=max_step)
        if not sol.success:raise RuntimeError(sol.message)
        z=sol.y[:,-1];states.append(z.copy())
        max_position=max(max_position,float(np.max(abs(z[:2*p.N].reshape(-1,2)-s['x'][k+1]))))
    return dict(rtol=rtol,max_step=max_step,max_position_error=max_position,
                max_fish_error=float(np.max(abs(z[2*p.N:2*p.N+p.K]-s['fish']))),
                distance_objective=float(z[-1]+p.delta*z[2*p.N:2*p.N+p.K].sum()))
