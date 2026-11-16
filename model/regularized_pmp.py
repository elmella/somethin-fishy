"""Forward/backward solver with an explicit effective-target regularization.

x' = a V (z-x) / sqrt(|z-x|^2 + eta^2), z in hull(grounds).
The travel integrand is
 gamma (sqrt(|x'|^2 + nu^2) - nu).
Both eta and nu are mathematical regularizations, not hidden solver tolerances.
"""
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.spatial import ConvexHull


def radial_flow(x,z,a,t,V,eta):
    """Exact constant-target flow and derivatives, including zero displacement."""
    q=z-x;r=np.linalg.norm(q,axis=-1);R=np.sqrt(r*r+eta*eta)
    e=np.divide(q,r[...,None],out=np.zeros_like(q),where=r[...,None]>0)
    log_initial=np.log(np.maximum(r/eta,1e-300))
    initial=R/eta+log_initial-np.log1p(R/eta)
    target=initial-a*V*t/eta
    w=log_initial-a*V*t/R
    for _ in range(8):
        rr=np.exp(np.clip(w,-745,700));root=np.hypot(rr,1)
        w-=(root+w-np.log1p(root)-target)/root
    end_r=eta*np.exp(np.clip(w,-745,700));end_r=np.where(r>0,end_r,0)
    end_R=np.hypot(end_r,eta)
    ratio=np.divide(end_r,r,out=np.exp(-a*V*t/eta),where=r>0)
    radial=ratio*R/end_R
    A=ratio[...,None,None]*np.eye(2)+(radial-ratio)[...,None,None]*e[...,None]*e[...,None,:]
    B=(t*V*end_r/end_R)[...,None]*e
    position=z-ratio[...,None]*q
    velocity=(a*V*end_r/end_R)[...,None]*e
    return position,A,B,velocity


class RegularizedSweep:
    target_scale=20.

    def __init__(self,problem,steps=400,eta=.1,travel_smoothing=.001):
        if eta<=0 or travel_smoothing<=0:raise ValueError('Specify positive eta and travel smoothing explicitly.')
        if steps<1 or problem.T<=0:raise ValueError('The horizon and step count must be positive.')
        self.p=problem;self.steps=steps;self.dt=problem.T/steps
        self.eta=eta;self.travel_smoothing=travel_smoothing;self.calls=0
        self.ii,self.jj=np.triu_indices(problem.N,1)
        hull=ConvexHull(problem.grounds)
        self.vertices=problem.grounds[hull.vertices]/self.target_scale
        self.halfspaces=hull.equations
        z,w=leggauss(10);self.nodes=(z+1)/2;self.weights=w/2

    def unpack(self,control):
        c=control.reshape(self.steps,self.p.N,3)
        return c,None,c[:,:,:2]*self.target_scale,c[:,:,-1]

    def project(self,control):
        out=control.reshape(self.steps,self.p.N,3).copy()
        points=out[:,:,:2].reshape(-1,2)
        outside=np.any(np.einsum('ni,ji->nj',points*self.target_scale,self.halfspaces[:,:2])+self.halfspaces[:,2]>1e-10,axis=-1)
        if np.any(outside):
            starts=self.vertices;edges=np.roll(starts,-1,axis=0)-starts
            q=points[outside,None]-starts[None]
            fraction=np.clip(np.sum(q*edges[None],axis=-1)/np.sum(edges*edges,axis=-1),0,1)
            projections=starts[None]+fraction[...,None]*edges[None]
            index=np.argmin(np.sum((projections-points[outside,None])**2,axis=-1),axis=1)
            points[outside]=projections[np.arange(len(index)),index]
        out[:,:,-1]=np.clip(out[:,:,-1],0,1-1e-7)
        return out.ravel()

    def seed(self,kind='coverage',mix=.002):
        """Generate a feasible coverage assignment and dwell schedule."""
        from scipy.optimize import linear_sum_assignment
        from fishing import route_segments
        p=self.p
        if kind in ('coverage','coverage-matched','coverage-deadline'):
            if p.N>p.K:raise ValueError('This coverage initializer expects no more ships than grounds.')
            distances=np.linalg.norm(p.ships[:,None]-p.grounds,axis=-1)
            _,assigned=linear_sum_assignment(distances)
            routes=[[int(j)] for j in assigned];dwells=np.zeros(p.N);offset=.5
            remaining=sorted(set(range(p.K))-set(assigned))
            if len(remaining)>p.N:raise ValueError('This initializer supports at most two assigned grounds per ship.')
            if kind in ('coverage-matched','coverage-deadline'):
                # Jointly match second visits, avoiding the greedy final boat
                # receiving a cross-map trip after nearer boats are committed.
                costs=distances[np.arange(p.N),assigned,None]+np.linalg.norm(p.grounds[assigned,None]-p.grounds[remaining],axis=-1)
                boats,columns=linear_sum_assignment(costs)
                assignments=[(int(i),remaining[j]) for i,j in zip(boats,columns)]
            else:
                assignments=[];available=list(range(p.N))
                for j in remaining:
                    i=min(available,key=lambda i:distances[i,assigned[i]]+np.linalg.norm(p.grounds[assigned[i]]-p.grounds[j]))
                    assignments.append((i,j));available.remove(i)
            for i,j in assignments:
                routes[i].append(j);dwells[i]=p.T/4
                if kind=='coverage-deadline':
                    travel=(distances[i,assigned[i]]+np.linalg.norm(p.grounds[assigned[i]]-p.grounds[j]))/p.V
                    dwells[i]=max(0,min(p.T/4,(p.T-travel)/2))
        else:raise ValueError('Choose coverage, coverage-matched, or coverage-deadline initialization.')
        trajectories=[route_segments(p,i,route,dwells[i],offset) for i,route in enumerate(routes)]
        control=np.empty((self.steps,p.N,3))
        for k in range(self.steps):
            t=(k+.5)*self.dt
            for i,path in enumerate(trajectories):
                segment=next(segment for segment in path if segment.start<=t<segment.end)
                target=(1-mix)*p.grounds[segment.target]+mix*p.grounds.mean(axis=0)
                control[k,i,:2]=target/self.target_scale
                control[k,i,-1]=np.linalg.norm(segment.velocity)/p.V*(1-1e-7)
        return control.ravel()

    def forward(self,control):
        p=self.p;_,_,target,a=self.unpack(control)
        x=np.empty((self.steps+1,p.N,2));x[0]=p.ships
        A=np.empty((self.steps,p.N,2,2));B=np.empty((self.steps,p.N,2))
        for k in range(self.steps):
            x[k+1],A[k],B[k],_=radial_flow(x[k],target[k],a[k],self.dt,p.V,self.eta)
        node_x,An,Bn,velocity=radial_flow(x[:-1,:,None],target[:,:,None],a[:,:,None],self.dt*self.nodes[None,None],p.V,self.eta)
        q=p.grounds[None,None,None]-node_x[:,:,:,None]
        C=np.exp(-p.sigma*np.sum(q*q,axis=-1));weights=self.dt*self.weights
        exposure=np.sum(C*weights[None,None,:,None],axis=(0,1,2))
        fish=p.F0*np.exp(-p.rate*exposure)
        pair=node_x[:,self.ii]-node_x[:,self.jj]
        denominators=p.epsilon+np.sum(pair*pair,axis=-1)
        crowd=2*p.beta*np.sum(weights[None,None]/denominators)
        speed=np.linalg.norm(velocity,axis=-1)
        travel=p.gamma*np.sum(weights[None,None]*(np.sqrt(speed*speed+self.travel_smoothing**2)-self.travel_smoothing))
        distance_travel=p.gamma*np.linalg.norm(np.diff(x,axis=0),axis=-1).sum()
        J=crowd+travel+p.delta*fish.sum()
        return dict(x=x,dx=np.diff(x,axis=0),A=A,B=B,An=An,Bn=Bn,node_x=node_x,q=q,C=C,fish=fish,
                    J=float(J),distance_J=float(J-travel+distance_travel),crowd=float(crowd),travel=float(travel),target=target,a=a,
                    pair=pair,denominators=denominators,velocity=velocity,speed=speed)

    def value_gradient(self,control):
        self.calls+=1;p=self.p;s=self.forward(control);weights=self.dt*self.weights
        G=-2*p.delta*p.rate*p.sigma*np.sum((s['C']*s['fish'][None,None,None])[...,None]*s['q'],axis=3)
        for k,(i,j) in enumerate(zip(self.ii,self.jj)):
            derivative=-4*p.beta*s['pair'][:,k]/s['denominators'][:,k,:,None]**2
            G[:,i]+=derivative;G[:,j]-=derivative
        q=s['target'][:,:,None]-s['node_x'];r=np.linalg.norm(q,axis=-1);R=np.hypot(r,self.eta)
        e=np.divide(q,r[...,None],out=np.zeros_like(q),where=r[...,None]>0)
        phi=s['speed']/np.sqrt(s['speed']**2+self.travel_smoothing**2)
        # abs/sign also make gradient probes just outside a=0 consistent with
        # the smooth speed norm used by finite-difference Hessian products.
        direct_z=p.gamma*phi[...,None]*np.abs(s['a'])[:,:,None,None]*p.V*self.eta**2/R[...,None]**3*e
        direct_a=p.gamma*phi*np.sign(s['a'])[:,:,None]*p.V*r/R
        G-=direct_z
        G*=weights[None,None,:,None]
        direct_z*=weights[None,None,:,None];direct_a*=weights[None,None]
        adj=np.zeros_like(s['x']);grad=np.zeros((self.steps,p.N,3))
        for k in range(self.steps-1,-1,-1):
            future=adj[k+1];Af=np.einsum('nji,nj->ni',s['A'][k],future)
            Ag=np.einsum('nqji,nqj->nqi',s['An'][k],G[k])
            adj[k]=Af+Ag.sum(axis=1)
            grad[k,:,:2]=self.target_scale*(future-Af+(G[k]-Ag+direct_z[k]).sum(axis=1))
            grad[k,:,-1]=np.sum(s['B'][k]*future,axis=-1)+np.sum(s['Bn'][k]*G[k],axis=(1,2))+direct_a[k].sum(axis=1)
        s['costate']=-adj;self.last=s
        return s['J'],grad.ravel()


def verify_continuous(solver,control,rtol=1e-8):
    """Independent full ODE replay and continuous regularized adjoint check."""
    import time
    from scipy.integrate import solve_ivp
    p=solver.p;s=solver.forward(control);dt=solver.dt;eta=solver.eta;nu=solver.travel_smoothing
    deadline=time.monotonic()+55
    def nominal(k,t):
        return radial_flow(s['x'][k],s['target'][k],s['a'][k],t-k*dt,p.V,eta)[0]
    z=np.r_[p.ships.ravel(),np.full(p.K,p.F0),0.,0.];max_position=0.
    for k in range(solver.steps):
        def rhs(t,z):
            if time.monotonic()>deadline:raise RuntimeError('Regularized replay exceeded 55 seconds.')
            x=z[:2*p.N].reshape(p.N,2);F=z[2*p.N:2*p.N+p.K]
            q=s['target'][k]-x;R=np.sqrt(np.sum(q*q,axis=-1)+eta**2)
            velocity=s['a'][k,:,None]*p.V*q/R[:,None];speed=np.linalg.norm(velocity,axis=-1)
            C=np.exp(-p.sigma*np.sum((p.grounds[None]-x[:,None])**2,axis=-1))
            dif=x[solver.ii]-x[solver.jj]
            crowd=2*p.beta*np.sum(1/(p.epsilon+np.sum(dif*dif,axis=-1)))
            smooth=crowd+p.gamma*np.sum(np.sqrt(speed*speed+nu*nu)-nu)
            distance_cost=crowd+p.gamma*speed.sum()
            return np.r_[velocity.ravel(),-p.rate*F*C.sum(axis=0),smooth,distance_cost]
        sol=solve_ivp(rhs,[k*dt,(k+1)*dt],z,method='DOP853',rtol=rtol,atol=rtol*1e-3,max_step=min(.5,dt))
        if not sol.success:raise RuntimeError(sol.message)
        z=sol.y[:,-1];max_position=max(max_position,float(np.max(abs(z[:2*p.N].reshape(-1,2)-s['x'][k+1]))))
    terminal=p.delta*z[2*p.N:2*p.N+p.K].sum()
    replay=dict(max_position_error=max_position,max_fish_error=float(np.max(abs(z[2*p.N:2*p.N+p.K]-s['fish']))),
                J=float(z[-2]+terminal),distance_J=float(z[-1]+terminal),rtol=rtol)
    # The transformed motion is independent of F, so its log-stock costate is
    # constant. The state Jacobian here is that of the CONTINUOUS radial field.
    pz=-p.delta*s['fish'];terminal=np.zeros((p.N,2));costates=[]
    for k in range(solver.steps-1,-1,-1):
        def adjoint(t,flat):
            if time.monotonic()>deadline:raise RuntimeError('Regularized adjoint check exceeded 55 seconds.')
            x=nominal(k,t);px=flat.reshape(p.N,2);q=s['target'][k]-x;r=np.linalg.norm(q,axis=-1);R=np.hypot(r,eta)
            a=s['a'][k];speed=a*p.V*r/R
            phi=speed/np.sqrt(speed**2+nu**2)
            travel=-p.gamma*phi[:,None]*a[:,None]*p.V*eta**2/R[:,None]**3*np.divide(q,r[:,None],out=np.zeros_like(q),where=r[:,None]>0)
            difference=x[None]-x[:,None];dist=p.epsilon+np.sum(difference*difference,axis=-1)
            crowd=4*p.beta*np.sum(difference/dist[...,None]**2,axis=1)
            qf=p.grounds[None]-x[:,None];C=np.exp(-p.sigma*np.sum(qf*qf,axis=-1))
            fish=2*p.rate*p.sigma*np.sum((C*pz)[...,None]*qf,axis=1)
            jac=-a[:,None,None]*p.V*(np.eye(2)[None]/R[:,None,None]-q[:,:,None]*q[:,None,:]/R[:,None,None]**3)
            return (travel+crowd+fish-np.einsum('nji,nj->ni',jac,px)).ravel()
        sol=solve_ivp(adjoint,[(k+1)*dt,k*dt],terminal.ravel(),method='DOP853',rtol=rtol,atol=rtol*1e-3,dense_output=True,max_step=min(.5,dt))
        if not sol.success:raise RuntimeError(sol.message)
        terminal=sol.y[:,-1].reshape(p.N,2);costates.append(sol.sol((k+.5)*dt).reshape(p.N,2))
    costates=np.array(costates[::-1]);gaps=[];aa,bb=np.triu_indices(p.K,1)
    for k,px in enumerate(costates):
        x=nominal(k,(k+.5)*dt);qf=p.grounds[None]-x[:,None]
        rho=np.linalg.norm(qf,axis=-1);d=np.divide(qf,rho[...,None],out=np.zeros_like(qf),where=rho[...,None]>0)
        length=np.linalg.norm(px,axis=-1);h=np.divide(px,length[:,None],out=np.zeros_like(px),where=length[:,None]>0)
        a=d[:,aa];b=d[:,bb];det=a[:,:,0]*b[:,:,1]-a[:,:,1]*b[:,:,0]
        numerator=np.stack((h[:,None,0]*b[:,:,1]-h[:,None,1]*b[:,:,0],a[:,:,0]*h[:,None,1]-a[:,:,1]*h[:,None,0]),axis=-1)
        w=np.divide(numerator,det[...,None],out=np.full_like(numerator,-np.inf),where=abs(det[...,None])>1e-12)
        feasible=np.any(np.all(w>=-1e-10,axis=-1),axis=-1)
        coefficient=np.where(feasible,length,np.max(np.sum(d*px[:,None],axis=-1),axis=-1))
        velocity=np.where(coefficient>=p.gamma,p.V,np.where(coefficient>0,nu*np.maximum(coefficient,0)/np.sqrt(np.maximum(p.gamma**2-coefficient**2,1e-300)),0))
        velocity=np.clip(velocity,0,p.V)
        best=velocity*coefficient-p.gamma*(np.sqrt(velocity**2+nu**2)-nu)
        q=s['target'][k]-x;actual=s['a'][k,:,None]*p.V*q/np.sqrt(np.sum(q*q,axis=-1)+eta**2)[:,None]
        speed=np.linalg.norm(actual,axis=-1)
        current=np.sum(px*actual,axis=-1)-p.gamma*(np.sqrt(speed*speed+nu*nu)-nu)
        gaps.append(best-current)
    gaps=np.array(gaps)
    return dict(replay=replay,hamiltonian_gap_upper_bound=float(dt*gaps.sum()),max_gap_upper_bound=float(gaps.max()),
                midpoint_costates=costates,midpoint_gaps=gaps)


def hamiltonian_gaps(solver,control,costates):
    """Search all hull edges for the regularized Hamiltonian maximum.

    Radial extension cannot reduce optimized H, so an edge contains a maximizer.
    Dense edge samples plus bounded refinements check each sampled local peak.
    """
    from scipy.optimize import minimize_scalar
    s=solver.forward(control);p=solver.p;nu=solver.travel_smoothing;eta=solver.eta
    vertices=solver.vertices*solver.target_scale;edges=np.roll(vertices,-1,axis=0)-vertices
    all_gaps=[];best_targets=[];best_speeds=[]
    def values(q,px):
        r=np.linalg.norm(q,axis=-1);R=np.hypot(r,eta)
        coefficient=np.divide(np.sum(q*px,axis=-1),r,out=np.zeros_like(r),where=r>0)
        cap=p.V*r/R
        v=np.where(coefficient>=p.gamma,cap,np.where(coefficient>0,nu*np.maximum(coefficient,0)/np.sqrt(np.maximum(p.gamma**2-coefficient**2,1e-300)),0))
        v=np.clip(v,0,cap)
        return v*coefficient-p.gamma*(np.sqrt(v*v+nu*nu)-nu)
    for k,px in enumerate(costates):
        x=radial_flow(s['x'][k],s['target'][k],s['a'][k],solver.dt/2,p.V,eta)[0]
        maxima=[];targets=[];fractions=[]
        for i in range(p.N):
            best=0.;best_point=s['target'][k,i].copy()
            for start,edge in zip(vertices,edges):
                base=start-x[i];near=np.clip(-np.sum(base*edge)/np.sum(edge*edge),0,1)
                width=eta/np.linalg.norm(edge)
                grid=np.unique(np.r_[np.linspace(0,1,101),np.clip(near+width*np.array([-1,-.1,-.01,-.001,0,.001,.01,.1,1]),0,1)])
                val=values(base+grid[:,None]*edge,px[i]);index=int(val.argmax())
                if val[index]>best:best=float(val[index]);best_point=start+grid[index]*edge
                peaks=np.flatnonzero((val[1:-1]>=val[:-2])&(val[1:-1]>=val[2:]))+1
                for j in peaks:
                    if val[j]<=0:continue
                    result=minimize_scalar(lambda t:-float(values(base+t*edge,px[i])),bounds=(grid[j-1],grid[j+1]),method='bounded',options={'xatol':1e-10})
                    if -result.fun>best:best=-result.fun;best_point=start+result.x*edge
            maxima.append(best)
            targets.append(best_point)
            q=best_point-x[i];r=np.linalg.norm(q);cap=p.V*r/np.hypot(r,eta)
            coefficient=0 if r==0 else np.sum(px[i]*q)/r
            v=cap if coefficient>=p.gamma else (nu*max(coefficient,0)/np.sqrt(max(p.gamma**2-coefficient**2,1e-300)))
            fractions.append(np.clip(v/cap,0,1-1e-7) if cap>0 else 0.)
        q=s['target'][k]-x;actual=s['a'][k,:,None]*p.V*q/np.sqrt(np.sum(q*q,axis=-1)+eta**2)[:,None]
        speed=np.linalg.norm(actual,axis=-1)
        current=np.sum(px*actual,axis=-1)-p.gamma*(np.sqrt(speed*speed+nu*nu)-nu)
        all_gaps.append(np.array(maxima)-current)
        best_targets.append(targets);best_speeds.append(fractions)
    gaps=np.array(all_gaps)
    best_control=np.concatenate((np.array(best_targets)/solver.target_scale,np.array(best_speeds)[...,None]),axis=-1).ravel()
    return dict(integrated_gap=float(solver.dt*gaps.sum()),max_gap=float(gaps.max()),gaps=gaps,best_control=best_control)
