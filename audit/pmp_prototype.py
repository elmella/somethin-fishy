"""Discrete forward/backward adjoint solver; original heading via effective targets.

Experimental until gradient, reconstruction, and refinement checks pass.
"""
import json
from pathlib import Path
import numpy as np
from scipy.special import softmax
from scipy.optimize import minimize
from numpy.polynomial.legendre import leggauss
from fishing import scenario


def pair_integral(a, b, epsilon):
    """Integral over normalized time [0,1], analytically complex-step safe."""
    A=np.sum(b*b,axis=-1); B=np.sum(a*b,axis=-1); C=epsilon+np.sum(a*a,axis=-1)
    small=A.real < .01*C.real
    result=np.empty_like(A,dtype=np.result_type(a,b))
    if np.any(small):
        nodes,weights=leggauss(8); nodes=(nodes+1)/2;weights=weights/2
        result[small]=np.sum(weights/(C[small,None]+2*B[small,None]*nodes+A[small,None]*nodes**2),axis=-1)
    if np.any(~small):
        aa=A[~small]; bb=B[~small]
        cross=a[~small,0]*b[~small,1]-a[~small,1]*b[~small,0]
        root=np.sqrt(epsilon*aa+cross**2)
        result[~small]=(np.arctan((aa+bb)/root)-np.arctan(bb/root))/root
    return result


class Sweep:
    def __init__(self,problem,steps=400,travel_smoothing=0.):
        self.p=problem; self.steps=steps; self.dt=problem.T/steps
        self.travel_smoothing=travel_smoothing
        z,w=leggauss(6);self.nodes=(z+1)/2;self.weights=w/2
        self.ii,self.jj=np.triu_indices(problem.N,1)
        self.calls=0

    def unpack(self,control):
        c=control.reshape(self.steps,self.p.N,self.p.K+1)
        theta=softmax(c[:,:,:self.p.K],axis=-1)
        return c,theta,theta@self.p.grounds,c[:,:,-1]

    def forward(self,control):
        p=self.p;c,theta,target,a=self.unpack(control)
        x=np.empty((self.steps+1,p.N,2));x[0]=p.ships
        A=np.empty((self.steps,p.N,2,2)); B=np.empty((self.steps,p.N,2))
        for t in range(self.steps):
            q=target[t]-x[t];r=np.linalg.norm(q,axis=-1)
            D=p.V*self.dt;R=np.sqrt(r*r+D*D)
            B[t]=D*q/R[:,None]
            x[t+1]=x[t]+a[t,:,None]*B[t]
            A[t]=np.eye(2)[None]-a[t,:,None,None]*D*(np.eye(2)[None]/R[:,None,None]-q[:,:,None]*q[:,None,:]/R[:,None,None]**3)
        dx=np.diff(x,axis=0)
        node_x=x[:-1,:,None,:]+self.nodes[None,None,:,None]*dx[:,:,None,:]
        q=p.grounds[None,None,None]-node_x[:,:,:,None]
        C=np.exp(-p.sigma*np.sum(q*q,axis=-1))
        exposure=self.dt*np.sum(C*self.weights[None,None,:,None],axis=(0,1,2))
        fish=p.F0*np.exp(-p.rate*exposure)
        r0=x[:-1,self.ii]-x[:-1,self.jj];d=dx[:,self.ii]-dx[:,self.jj]
        crowd=2*p.beta*self.dt*pair_integral(r0,d,p.epsilon).sum()
        distance=np.linalg.norm(dx,axis=-1)
        original_travel=p.gamma*distance.sum()
        travel=p.gamma*np.sum(np.sqrt(distance**2+(self.travel_smoothing*self.dt)**2)-self.travel_smoothing*self.dt)
        J=crowd+travel+p.delta*fish.sum()
        return dict(x=x,dx=dx,A=A,B=B,q=q,C=C,fish=fish,theta=theta,r0=r0,d=d,J=float(J),original_J=float(J-travel+original_travel),crowd=float(crowd),travel=float(travel),target=target,a=a)

    def value_gradient(self,control):
        self.calls+=1;p=self.p;s=self.forward(control)
        x,dx=s['x'],s['dx'];g0=np.zeros_like(dx);g1=np.zeros_like(dx)
        # Terminal fish cost transported backward through log-stock dynamics.
        G=-2*p.delta*p.rate*p.sigma*self.dt*np.sum((s['C']*s['fish'][None,None,None])[...,None]*s['q'],axis=3)
        g0+=np.sum(G*(self.weights*(1-self.nodes))[None,None,:,None],axis=2)
        g1+=np.sum(G*(self.weights*self.nodes)[None,None,:,None],axis=2)
        # Exact sharp pair-crowding integral and its complex-step partials.
        ga=np.empty_like(s['r0']);gb=np.empty_like(s['d'])
        for j in range(2):
            aa=s['r0'].astype(complex);aa[:,:,j]+=1e-20j
            bb=s['d'].astype(complex);bb[:,:,j]+=1e-20j
            ga[:,:,j]=pair_integral(aa,s['d'],p.epsilon).imag/1e-20
            gb[:,:,j]=pair_integral(s['r0'],bb,p.epsilon).imag/1e-20
        scale=2*p.beta*self.dt
        for k,(i,j) in enumerate(zip(self.ii,self.jj)):
            g0[:,i]+=scale*(ga[:,k]-gb[:,k]);g0[:,j]-=scale*(ga[:,k]-gb[:,k])
            g1[:,i]+=scale*gb[:,k];g1[:,j]-=scale*gb[:,k]
        # Travel is linear in the nonnegative distance-fraction control.
        # Keep its one-sided speed derivative even at a=0 (a stationary boat).
        target=s['target'];theta=s['theta'];a=s['a']
        q0=target-x[:-1];r=np.linalg.norm(q0,axis=-1);e=q0/np.maximum(r[...,None],1e-300)
        near=r<=p.V*self.dt
        s['stage_g0']=g0.copy();s['stage_g1']=g1.copy()
        if self.travel_smoothing:
            length=np.sqrt(np.sum(dx*dx,axis=-1)+(self.travel_smoothing*self.dt)**2)
            travel_grad=p.gamma*dx/length[...,None]
            g0-=travel_grad;g1+=travel_grad
            direct_target=np.zeros_like(g0);direct_a=np.zeros_like(a)
        else:
            D=p.V*self.dt;R=np.sqrt(r*r+D*D)
            direct_target=p.gamma*a[...,None]*D**3*e/R[...,None]**3
            g0-=direct_target
            direct_a=p.gamma*D*r/R
        adj=np.zeros_like(x);gz=np.empty_like(dx);grad=np.zeros((self.steps,p.N,p.K+1))
        for t in range(self.steps-1,-1,-1):
            z=g1[t]+adj[t+1]
            Az=np.einsum('nji,nj->ni',s['A'][t],z)
            adj[t]=g0[t]+Az
            gz[t]=z-Az+direct_target[t]
            grad[t,:,-1]=np.sum(z*s['B'][t],axis=-1)+direct_a[t]
        if theta is None:
            grad=np.concatenate((gz*self.target_scale,grad[:,:,-1,None]),axis=-1)
        else:
            rewards=gz@p.grounds.T
            grad[:,:,:p.K]=theta*(rewards-np.sum(theta*rewards,axis=-1,keepdims=True))
        s['costate']=-adj
        self.last=s
        return s['J'],grad.ravel()

    def pmp_residual(self,control):
        """Original unit-heading H gap, using fixed-speed effective-target adjoints."""
        self.value_gradient(control);s=self.last;p=self.p
        q=s['target']-s['x'][:-1];r=np.linalg.norm(q,axis=-1);e=q/np.maximum(r[...,None],1e-300)
        distance=np.linalg.norm(s['dx'],axis=-1)
        ratio=distance/np.maximum(r,1e-300)
        A=np.eye(2)[None,None]-ratio[...,None,None]*(np.eye(2)[None,None]-e[...,None]*e[...,None,:])
        lam=np.zeros_like(s['x'])
        for t in range(self.steps-1,-1,-1):
            lam[t]=s['stage_g0'][t]+np.einsum('nji,nj->ni',A[t],s['stage_g1'][t]+lam[t+1])
        from fishing import maximize_hamiltonian,stable_rhs
        gaps=[]
        for t in range(self.steps):
            x=(s['x'][t]+s['x'][t+1])/2;px=-(lam[t]+lam[t+1])/2
            y=np.r_[x.ravel(),np.ones(p.K)]
            u,v=maximize_hamiltonian(y,np.r_[px.ravel(),np.zeros(p.K)],p)
            best=stable_rhs(y,u,v,p)[:2*p.N].reshape(-1,2)
            actual=s['dx'][t]/self.dt
            gap=np.sum(px*(best-actual),axis=-1)-p.gamma*(np.linalg.norm(best,axis=-1)-np.linalg.norm(actual,axis=-1))
            gaps.append(gap)
        gaps=np.array(gaps)
        return dict(integrated_gap=float(self.dt*gaps.sum()),max_gap=float(gaps.max()),costate=-lam,gaps=gaps)

    def seed(self,kind='benchmark',mix=.002):
        p=self.p
        if kind in ['benchmark','coverage']:
            from fishing import evaluate_routes
            if kind=='benchmark':
                data=json.loads((Path(__file__).parent/'results/working/notebook_400.json').read_text())['solution']
                routes,dwells,offset=data['routes'],data['dwells'],data['standoff']
            else:
                from scipy.optimize import linear_sum_assignment
                distances=np.linalg.norm(p.ships[:,None]-p.grounds,axis=-1)
                _,assigned=linear_sum_assignment(distances)
                routes=[[int(j)] for j in assigned];dwells=np.zeros(p.N);offset=.5
                for j in sorted(set(range(p.K))-set(assigned)):
                    candidates=[i for i,r in enumerate(routes) if len(r)==1]
                    i=min(candidates,key=lambda i:distances[i,assigned[i]]+np.linalg.norm(p.grounds[assigned[i]]-p.grounds[j]))
                    routes[i].append(j);dwells[i]=p.T/4
            _,trajectories=evaluate_routes(p,routes,dwells,offset)
            target=np.zeros((self.steps,p.N),int);speed=np.empty((self.steps,p.N))
            for t in range(self.steps):
                time=(t+.5)*self.dt
                for i,tr in enumerate(trajectories):
                    seg=next(seg for seg in tr if seg.start<=time<seg.end)
                    target[t,i]=seg.target;speed[t,i]=np.linalg.norm(seg.velocity)/p.V
        else:
            from scipy.optimize import linear_sum_assignment
            dist=np.linalg.norm(p.ships[:,None]-p.grounds,axis=-1)
            _,indices=linear_sum_assignment(dist)
            target=np.tile(indices,(self.steps,1));speed=np.ones((self.steps,p.N))
        theta=np.full((self.steps,p.N,p.K),mix/p.K)
        np.put_along_axis(theta,target[...,None],1-mix+mix/p.K,axis=-1)
        logits=np.log(theta);logits-=logits.mean(axis=-1,keepdims=True)
        return np.concatenate((logits,speed[...,None]*(1-1e-7)),axis=-1).ravel()

    def optimize(self,control,maxiter=300):
        history=[];count=0
        def callback(c):
            nonlocal count
            count+=1
            s=self.forward(c);history.append(dict(iteration=count,J=s['J'],harvested=float(self.p.K*self.p.F0-s['fish'].sum())))
            if count==1 or count%10==0:print(history[-1],flush=True)
        bounds=[(-25.,25.)]*self.p.K+[(0.,1-1e-7)]
        initial=self.forward(control)
        result=minimize(self.value_gradient,control,jac=True,method='L-BFGS-B',bounds=bounds*(self.steps*self.p.N),callback=callback,
                        options=dict(maxiter=maxiter,maxls=30,ftol=1e-11,gtol=1e-5,maxcor=20))
        J,g=self.value_gradient(result.x)
        return result,dict(initial_J=initial['J'],initial_catch=float(self.p.K*self.p.F0-initial['fish'].sum()),history=history,J=J,harvested=float(self.p.K*self.p.F0-self.last['fish'].sum()),gradient_max=float(abs(g).max()),message=str(result.message),success=bool(result.success),calls=self.calls)


class TargetSweep(Sweep):
    target_scale=20.

    def __init__(self,problem,steps=200,travel_smoothing=0.):
        super().__init__(problem,steps,travel_smoothing)
        from scipy.spatial import ConvexHull
        hull=ConvexHull(problem.grounds)
        self.vertices=problem.grounds[hull.vertices]/self.target_scale
        self.halfspaces=hull.equations

    def unpack(self,control):
        c=control.reshape(self.steps,self.p.N,3)
        return c,None,c[:,:,:2]*self.target_scale,c[:,:,-1]

    def seed(self,kind='benchmark',mix=.002):
        raw=Sweep.seed(self,kind,mix)
        _,_,target,a=Sweep.unpack(self,raw)
        return np.concatenate((target/self.target_scale,a[...,None]),axis=-1).ravel()

    def project(self,u):
        out=u.reshape(self.steps,self.p.N,3).copy()
        points=out[:,:,:2].reshape(-1,2)
        outside=np.any(np.einsum('ni,ji->nj',points*self.target_scale,self.halfspaces[:,:2])+self.halfspaces[:,2]>1e-10,axis=-1)
        if np.any(outside):
            starts=self.vertices;edges=np.roll(starts,-1,axis=0)-starts
            q=points[outside,None]-starts[None]
            t=np.clip(np.sum(q*edges[None],axis=-1)/np.sum(edges*edges,axis=-1),0,1)
            projections=starts[None]+t[...,None]*edges[None]
            index=np.argmin(np.sum((projections-points[outside,None])**2,axis=-1),axis=1)
            points[outside]=projections[np.arange(len(index)),index]
        out[:,:,-1]=np.clip(out[:,:,-1],0,1-1e-7)
        return out.ravel()

    def canonicalize(self,u):
        """Remove target-distance redundancy while preserving every displacement.

        Extend each target ray to the far hull edge and reduce its distance
        fraction to keep the physical speed exactly unchanged. This is a
        control-coordinate change, not a trajectory correction.
        """
        s=self.forward(u);q=s['target']-s['x'][:-1]
        r=np.linalg.norm(q,axis=-1)
        center=self.p.grounds.mean(axis=0)
        fallback=center-s['x'][:-1]
        q=np.where((r>1e-10)[...,None],q,fallback)
        e=q/np.linalg.norm(q,axis=-1)[...,None]
        h=self.halfspaces
        numerator=-(np.einsum('tni,ji->tnj',s['x'][:-1],h[:,:2])+h[:,2])
        denominator=np.einsum('tni,ji->tnj',e,h[:,:2])
        upper=np.full_like(denominator,np.inf)
        np.divide(numerator,denominator,out=upper,where=denominator>1e-12)
        radius=upper.min(axis=-1)
        if np.any(radius<=0):raise ValueError('No positive target ray in hull.')
        target=s['x'][:-1]+radius[...,None]*e
        distance=np.linalg.norm(s['dx'],axis=-1);D=self.p.V*self.dt
        a=distance*np.sqrt(radius**2+D**2)/(D*radius)
        result=np.concatenate((target/self.target_scale,a[...,None]),axis=-1).ravel()
        np.testing.assert_allclose(self.forward(result)['x'],s['x'],atol=1e-8,rtol=0)
        return result

    def optimize(self,u,maxiter=2000):
        from types import SimpleNamespace
        u=self.project(u);J,g=self.value_gradient(u);initial_J=J;initial_catch=float(self.p.K*self.p.F0-self.last['fish'].sum())
        history=[];pairs=[];status='Iteration limit';residual=float('inf')
        for iteration in range(maxiter):
            pg=self.project(u-g)-u;residual=float(abs(pg).max())
            if residual<1e-5:
                status='Projected adjoint residual below tolerance';break
            # Restrict curvature updates to the current free face. A plain
            # Euclidean projection of an unconstrained L-BFGS direction uses
            # the wrong metric and can stagnate on an active hull edge.
            c=u.reshape(self.steps,self.p.N,3)
            pg3=pg.reshape(c.shape)
            face_distance=np.einsum('tni,ji->tnj',c[:,:,:2]*self.target_scale,self.halfspaces[:,:2])+self.halfspaces[:,2]
            face_step=np.einsum('tni,ji->tnj',pg3[:,:,:2],self.halfspaces[:,:2])
            active=(face_distance>-1e-7)&(face_step>-1e-9)
            normals=self.halfspaces[:,:2]
            count=active.sum(axis=-1)
            normal=np.einsum('tnj,ji->tni',active,normals)
            free_speed=(np.abs(pg3[:,:,-1])>1e-12)|((c[:,:,-1]>1e-10)&(c[:,:,-1]<1-1e-7-1e-10))
            def tangent(vector):
                v=vector.reshape(c.shape).copy()
                component=np.sum(v[:,:,:2]*normal,axis=-1)
                v[:,:,:2]-=np.where((count==1)[...,None],component[...,None]*normal,0)
                v[:,:,:2]=np.where((count>1)[...,None],0,v[:,:,:2])
                v[:,:,-1]*=free_speed
                return v.ravel()
            free_pairs=[]
            for ss,yy,_ in pairs:
                ss=tangent(ss);yy=tangent(yy);curvature=np.sum(ss*yy)
                if curvature>1e-10*np.linalg.norm(ss)*np.linalg.norm(yy):
                    free_pairs.append((ss,yy,1/curvature))
            q=tangent(g);alphas=[]
            for ss,yy,rho in reversed(free_pairs):
                alpha=rho*np.sum(ss*q);alphas.append(alpha);q-=alpha*yy
            scale=1. if not free_pairs else np.clip(np.sum(free_pairs[-1][0]*free_pairs[-1][1])/np.sum(free_pairs[-1][1]**2),1e-5,1e4)
            direction=scale*q
            for (ss,yy,rho),alpha in zip(free_pairs,reversed(alphas)):
                direction+=ss*(alpha-rho*np.sum(yy*direction))
            d=self.project(u-direction)-u
            if np.sum(g*d)>=-1e-12*np.sum(pg*pg):
                d=pg;pairs=[]
            slope=np.sum(g*d);accepted=False
            for step in 2.**-np.arange(30):
                trial=u+step*d
                value=self.forward(trial)['J']
                if value <= J+1e-4*step*slope:
                    accepted=True;break
            if not accepted:
                status='Line search failed';break
            value,new_g=self.value_gradient(trial)
            ss=trial-u;yy=new_g-g;curvature=np.sum(ss*yy)
            if curvature>1e-10*np.linalg.norm(ss)*np.linalg.norm(yy):
                pairs.append((ss,yy,1/curvature));pairs=pairs[-15:]
            u,J,g=trial,value,new_g
            history.append(dict(iteration=iteration+1,J=J,harvested=float(self.p.K*self.p.F0-self.last['fish'].sum()),residual=residual,step=float(step)))
            if iteration==0 or (iteration+1)%50==0:print(history[-1],flush=True)
        J,g=self.value_gradient(u)
        report=dict(initial_J=initial_J,initial_catch=initial_catch,history=history,J=J,original_J=self.last['original_J'],travel_smoothing=self.travel_smoothing,harvested=float(self.p.K*self.p.F0-self.last['fish'].sum()),
                    projected_residual=float(abs(self.project(u-g)-u).max()),message=status,success=status.startswith('Projected'),calls=self.calls)
        return SimpleNamespace(x=u),report


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--steps',type=int,default=200);parser.add_argument('--iterations',type=int,default=300);parser.add_argument('--seed',default='benchmark');parser.add_argument('--target',action='store_true');parser.add_argument('--smoothing',type=float,default=0.)
    args=parser.parse_args();p=scenario();s=(TargetSweep if args.target else Sweep)(p,args.steps,args.smoothing);u=s.seed(args.seed)
    value,grad=s.value_gradient(u)
    rng=np.random.default_rng(21);direction=rng.normal(size=u.shape);direction.reshape(args.steps,p.N,-1)[:,:,-1]=0
    direction/=np.linalg.norm(direction)
    for eps in [1e-3,1e-4,1e-5]:
        fd=(s.forward(u+eps*direction)['J']-s.forward(u-eps*direction)['J'])/(2*eps)
        print('gradient check',eps,fd,np.sum(grad*direction),flush=True)
    result,report=s.optimize(u,args.iterations)
    out=Path(__file__).parent/'results/pmp';out.mkdir(exist_ok=True)
    stem=f'{"target_" if args.target else ""}{args.seed}_{args.steps}_{args.smoothing:g}'
    (out/(stem+'.json')).write_text(json.dumps(report,indent=2))
    np.savez_compressed(out/(stem+'.npz'),control=result.x,**s.last)
    print({k:v for k,v in report.items() if k!='history'},flush=True)

class BoxSweep(TargetSweep):
    target_scale=1.
    def __init__(self,problem,steps=200,travel_smoothing=.001):
        super().__init__(problem,steps,travel_smoothing)
        self.xmin=problem.grounds[:,0].min();self.xspan=np.ptp(problem.grounds[:,0])
        low=self.halfspaces[self.halfspaces[:,1]<-1e-10];high=self.halfspaces[self.halfspaces[:,1]>1e-10]
        self.low_slope=-low[:,0]/low[:,1];self.low_intercept=-low[:,2]/low[:,1]
        self.high_slope=-high[:,0]/high[:,1];self.high_intercept=-high[:,2]/high[:,1]

    def envelope(self,x):
        lows=x[...,None]*self.low_slope+self.low_intercept; highs=x[...,None]*self.high_slope+self.high_intercept
        li=lows.argmax(axis=-1);hi=highs.argmin(axis=-1)
        return lows.max(axis=-1),highs.min(axis=-1),self.low_slope[li],self.high_slope[hi]

    def unpack(self,u):
        c=u.reshape(self.steps,self.p.N,3);x=self.xmin+self.xspan*c[:,:,0]
        low,high,_,_=self.envelope(x)
        target=np.stack((x,low+c[:,:,1]*(high-low)),axis=-1)
        return c,None,target,c[:,:,-1]

    def seed(self,kind='coverage',mix=.002):
        raw=Sweep.seed(self,kind,mix);_,_,target,a=Sweep.unpack(self,raw)
        low,high,_,_=self.envelope(target[:,:,0])
        return np.stack(((target[:,:,0]-self.xmin)/self.xspan,(target[:,:,1]-low)/np.maximum(high-low,1e-20),a),axis=-1).ravel()

    def value_gradient(self,u):
        J,g=Sweep.value_gradient(self,u);c=u.reshape(self.steps,self.p.N,3);g=g.reshape(c.shape)
        _,_,target,_=self.unpack(u);low,high,dl,dh=self.envelope(target[:,:,0])
        grad=g.copy();grad[:,:,0]=self.xspan*(g[:,:,0]+g[:,:,1]*(dl+c[:,:,1]*(dh-dl)));grad[:,:,1]=g[:,:,1]*(high-low)
        self.raw_gradient=g
        return J,grad.ravel()

    def optimize(self,u,maxiter=3000):
        history=[];start=self.forward(u)
        def callback(c):
            s=self.forward(c);history.append(dict(iteration=len(history)+1,J=s['J'],original_J=s['original_J'],harvested=float(self.p.K*self.p.F0-s['fish'].sum())))
            if len(history)==1 or len(history)%100==0: print(history[-1],flush=True)
        result=minimize(self.value_gradient,u,jac=True,method='L-BFGS-B',bounds=[(1e-9,1-1e-9),(0.,1.),(0.,1-1e-7)]*(self.steps*self.p.N),callback=callback,
                        options=dict(maxiter=maxiter,maxls=40,ftol=1e-13,gtol=1e-5,maxcor=30))
        J,g=self.value_gradient(result.x);pg=np.clip(result.x-g,0,1)-result.x
        report=dict(initial_J=start['J'],initial_catch=float(self.p.K*self.p.F0-start['fish'].sum()),history=history,J=J,original_J=self.last['original_J'],travel_smoothing=self.travel_smoothing,
                    harvested=float(self.p.K*self.p.F0-self.last['fish'].sum()),projected_residual=float(abs(pg).max()),message=str(result.message),success=bool(result.success),calls=self.calls)
        return result,report
