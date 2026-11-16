"""Check discrete adjoint trajectories against the original continuous equations."""
import time
import numpy as np
from scipy.special import erf
from scipy.integrate import solve_ivp
from fishing import stable_rhs, running_cost

# At almost stationary ground arrivals, direction is dominated by coordinate
# roundoff. Execute a feasible stop instead of normalizing a ~1e-12 velocity.
# This bounds the velocity approximation by 1e-10; it is not a new cost term.
CONTROL_SPEED_ZERO_TOLERANCE=1e-10
CONTROL_VELOCITY_RECONSTRUCTION_TOLERANCE=1e-10


def exposure(x, velocity, duration, grounds, sigma):
    """Exact Gaussian exposure for all ships on one straight segment."""
    q=x[:,None]-grounds
    speed=np.linalg.norm(velocity,axis=-1)
    direction=np.divide(velocity,speed[:,None],out=np.zeros_like(velocity),where=speed[:,None]>0)
    parallel=np.sum(q*direction[:,None],axis=-1)
    perpendicular=q[:,:,0]*direction[:,None,1]-q[:,:,1]*direction[:,None,0]
    factor=np.divide(np.sqrt(np.pi),2*speed*np.sqrt(sigma),out=np.zeros_like(speed),where=speed>0)
    integral=np.exp(-sigma*perpendicular**2)*factor[:,None]*(erf(np.sqrt(sigma)*(parallel+speed[:,None]*duration))-erf(np.sqrt(sigma)*parallel))
    integral[speed==0]=duration*np.exp(-sigma*np.sum(q[speed==0]**2,axis=-1))
    return integral.sum(axis=0)


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


class OriginalSchedule:
    """Time-only raw attraction schedule reconstructed from a nominal path.

    During replay, controls use the stored nominal state, NEVER the simulated
    state. This checks open-loop execution of the original equations.
    """
    def __init__(self,solver,control):
        self.p=solver.p;self.dt=solver.dt;self.steps=solver.steps
        self.state=solver.forward(control);s=self.state
        self.velocity=s['dx']/self.dt
        self.pair_a,self.pair_b=np.triu_indices(self.p.K,1)
        speed=np.linalg.norm(self.velocity,axis=-1)
        self.heading=np.divide(self.velocity,speed[...,None],out=np.zeros_like(self.velocity),where=speed[...,None]>0)
        self.prefix=np.zeros((self.steps+1,self.p.K))
        for k in range(self.steps):
            self.prefix[k+1]=self.prefix[k]+exposure(s['x'][k],self.velocity[k],self.dt,self.p.grounds,self.p.sigma)

    def nominal(self,k,t):
        elapsed=np.clip(t-k*self.dt,0,self.dt)
        x=self.state['x'][k]+elapsed*self.velocity[k]
        integral=self.prefix[k]+exposure(self.state['x'][k],self.velocity[k],elapsed,self.p.grounds,self.p.sigma)
        fish=self.p.F0*np.exp(-self.p.rate*integral)
        return x,fish

    def controls(self,k,t):
        x,fish=self.nominal(k,t)
        return reconstruct_attractions(x,fish,self.velocity[k],self.p)

    def checks(self):
        p=self.p;error=0.;clearance=np.inf
        for k in range(self.steps):
            v=self.velocity[k];q=self.state['x'][k,:,None]-p.grounds
            vv=np.sum(v*v,axis=-1)
            closest=np.clip(np.divide(-np.sum(q*v[:,None],axis=-1),vv[:,None],out=np.zeros((p.N,p.K)),where=vv[:,None]>0),0,self.dt)
            clearance=min(clearance,float(np.linalg.norm(q+closest[...,None]*v[:,None],axis=-1).min()))
            for fraction in [.1,.5,.9]:
                t=(k+fraction)*self.dt;x,F=self.nominal(k,t);u,speed=self.controls(k,t)
                actual=stable_rhs(np.r_[x.ravel(),F],u,speed,p)[:2*p.N].reshape(-1,2)
                error=max(error,float(np.max(abs(actual-v))))
        final=self.nominal(self.steps-1,p.T)[1]
        return dict(max_velocity_reconstruction_error=error,minimum_center_clearance=clearance,
                    exact_final_fish=final.tolist(),max_fish_quadrature_error=float(np.max(abs(final-self.state['fish']))),
                    exact_original_objective=float(self.state['original_J']+p.delta*(final-self.state['fish']).sum()))

    def replay(self,rtol=1e-8,max_step=.5):
        p=self.p;z=np.r_[p.ships.ravel(),np.full(p.K,p.F0),0.]
        rows=[z.copy()];deadline=time.monotonic()+55.;max_error=0.
        for k in range(self.steps):
            def rhs(t,state):
                if time.monotonic()>deadline:raise RuntimeError('Original-equation replay exceeded 55 seconds.')
                u,v=self.controls(k,t)
                return np.r_[stable_rhs(state[:-1],u,v,p),running_cost(state[:-1],v,p,include_self=False)]
            sol=solve_ivp(rhs,[k*self.dt,(k+1)*self.dt],z,method='DOP853',rtol=rtol,atol=rtol*1e-3,max_step=max_step)
            if not sol.success:raise RuntimeError(sol.message)
            z=sol.y[:,-1];rows.append(z.copy())
            max_error=max(max_error,float(np.max(abs(z[:2*p.N].reshape(-1,2)-self.state['x'][k+1]))))
        rows=np.array(rows)
        return dict(rtol=rtol,max_step=max_step,max_position_error=max_error,
                    max_fish_error=float(np.max(abs(z[2*p.N:2*p.N+p.K]-self.state['fish']))),
                    original_objective=float(z[-1]+p.delta*z[2*p.N:2*p.N+p.K].sum()),
                    x=rows[:,:2*p.N].reshape(-1,p.N,2),fish=rows[:,2*p.N:2*p.N+p.K])


def continuous_pmp_check(schedule, rtol=1e-7):
    """Original raw-u adjoint in log-stock coordinates; original travel cost.

    This is a separate continuous necessary-condition diagnostic. It does not
    reuse the discretized optimizer's adjoint or its smoothed travel penalty.
    """
    from fishing import maximize_hamiltonian
    p=schedule.p;N=p.N;K=p.K;end=schedule.nominal(schedule.steps-1,p.T)[1]
    terminal=np.r_[np.zeros(2*N),-p.delta*end]
    deadline=time.monotonic()+50.;costates=np.empty((schedule.steps,N*2+K))
    samples=(np.arange(schedule.steps)+.5)*schedule.dt
    def rhs(t,costate,k):
        if time.monotonic()>deadline:raise RuntimeError('Continuous PMP diagnostic exceeded 50 seconds.')
        x,F=schedule.nominal(k,t);u,v=schedule.controls(k,t)
        q=p.grounds[None]-x[:,None];rho=np.linalg.norm(q,axis=-1);d=q/rho[...,None]
        logs=np.log(np.maximum(u,1e-300))+np.log(F)[None]-p.alpha*rho**2
        logs[u==0]=-np.inf;weights=np.exp(logs-logs.max(axis=-1,keepdims=True))
        field=np.sum(weights[...,None]*d,axis=1);norm=np.linalg.norm(field,axis=-1)
        direction=field/norm[:,None];weights/=norm[:,None]
        px=costate[:2*N].reshape(N,2);pz=costate[2*N:]
        Qp=px-direction*np.sum(px*direction,axis=-1)[:,None]
        B=2*p.alpha*q[:,:,:,None]*d[:,:,None,:]-(np.eye(2)[None,None]-d[:,:,:,None]*d[:,:,None,:])/rho[...,None,None]
        B=np.sum(weights[...,None,None]*B,axis=1)
        C=np.exp(-p.sigma*rho**2)
        difference=x[None]-x[:,None];distance=p.epsilon+np.sum(difference*difference,axis=-1)
        Lx=4*p.beta*np.sum(difference/distance[...,None]**2,axis=1)
        dx=Lx+2*p.rate*p.sigma*np.sum((C*pz)[...,None]*q,axis=1)-v[:,None]*np.einsum('nji,nj->ni',B,Qp)
        dz=-np.sum(v[:,None]*weights*np.sum(d*Qp[:,None],axis=-1),axis=0)
        return np.r_[dx.ravel(),dz]
    for k in range(schedule.steps-1,-1,-1):
        sol=solve_ivp(lambda t,c:rhs(t,c,k),[(k+1)*schedule.dt,k*schedule.dt],terminal,method='DOP853',rtol=rtol,atol=rtol*1e-3,dense_output=True,max_step=.5)
        if not sol.success:raise RuntimeError(sol.message)
        terminal=sol.y[:,-1];costates[k]=sol.sol(samples[k])
    gaps=[]
    for k,t in enumerate(samples):
        x,F=schedule.nominal(k,t);y=np.r_[x.ravel(),F];px=costates[k,:2*N].reshape(N,2)
        u,v=maximize_hamiltonian(y,np.r_[px.ravel(),np.zeros(K)],p)
        best=stable_rhs(y,u,v,p)[:2*N].reshape(N,2);actual=schedule.velocity[k]
        gaps.append(np.sum(px*(best-actual),axis=-1)-p.gamma*(np.linalg.norm(best,axis=-1)-np.linalg.norm(actual,axis=-1)))
    gaps=np.array(gaps)
    return dict(integrated_hamiltonian_gap=float(schedule.dt*gaps.sum()),max_hamiltonian_gap=float(gaps.max()),
                midpoint_gaps=gaps,costates=costates,rtol=rtol)


class RegularizedSchedule(OriginalSchedule):
    """Replay a regularized-target solution using time-only ORIGINAL u and v."""
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
    """Execute the target policy through the ORIGINAL physical u/v equations.

    Unlike OriginalSchedule.replay, this explicitly recomputes raw u and v from
    the evolving simulated state. The distinction is part of the model report.
    """
    p=schedule.p;s=schedule.state;solver=schedule.solver
    z=np.r_[p.ships.ravel(),np.full(p.K,p.F0),0.]
    deadline=time.monotonic()+55;max_position=0.;states=[z.copy()]
    for k in range(schedule.steps):
        def rhs(t,state):
            if time.monotonic()>deadline:raise RuntimeError('Original-equation feedback replay exceeded 55 seconds.')
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
                original_objective=float(z[-1]+p.delta*z[2*p.N:2*p.N+p.K].sum()))
