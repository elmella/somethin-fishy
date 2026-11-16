"""Conditioned hull-edge coordinates for Pontryagin control polishing."""
import numpy as np
from scipy.optimize import minimize
from regularized_pmp import RegularizedSweep,radial_flow


def edge_initialization(solver,control):
    """Extend target rays to the hull; preserve each interval's end position.

    The within-interval speed profile changes, so its cost must be re-evaluated.
    This is a numerical control parameterization, not an identity of policies.
    """
    s=solver.forward(control)
    faces=np.einsum('tni,ji->tnj',s['target'],solver.halfspaces[:,:2])+solver.halfspaces[:,2]
    # Already-canonical edge targets need no ray extension. In particular,
    # re-extending a near-stopped ray at a vertex amplifies coordinate roundoff.
    if np.max(abs(faces.max(axis=-1)))<1e-6:return control.copy()
    q=s['target']-s['x'][:-1];r=np.linalg.norm(q,axis=-1)
    rays=solver.p.grounds[None,None]-s['x'][:-1,:,None]
    farthest=np.argmax(np.sum(rays*rays,axis=-1),axis=-1)
    fallback=np.take_along_axis(rays,farthest[...,None,None],axis=2)[:,:,0]
    q=np.where((r>1e-9)[...,None],q,fallback)
    e=q/np.linalg.norm(q,axis=-1)[...,None];h=solver.halfspaces
    numerator=-(np.einsum('tni,ji->tnj',s['x'][:-1],h[:,:2])+h[:,2]);denominator=np.einsum('tni,ji->tnj',e,h[:,:2])
    upper=np.divide(numerator,denominator,out=np.full_like(denominator,np.inf),where=denominator>1e-12)
    radius=upper.min(axis=-1)
    if np.any(radius<=0):raise ValueError('No positive far-hull target.')
    target=s['x'][:-1]+radius[...,None]*e;distance=np.linalg.norm(s['dx'],axis=-1)
    end=np.maximum(radius-distance,1e-300);R0=np.hypot(radius,solver.eta);R1=np.hypot(end,solver.eta)
    deltaR=distance*(radius+end)/(R0+R1)
    integral=deltaR+solver.eta*(np.log1p(distance/end)+np.log1p(-deltaR/(R0+solver.eta)))
    a=integral/(solver.p.V*solver.dt);a[distance<1e-10]=0
    if a.max()>1+1e-6:raise ValueError('Canonicalized speed exceeds its bound.')
    result=np.concatenate((target/solver.target_scale,np.clip(a,0,1-1e-7)[...,None]),axis=-1).ravel()
    np.testing.assert_allclose(solver.forward(result)['x'],s['x'],atol=1e-6,rtol=0)
    return result


class EdgeSweep(RegularizedSweep):
    def __init__(self,solver,control):
        super().__init__(solver.p,solver.steps,solver.eta,solver.travel_smoothing)
        c=control.reshape(self.steps,self.p.N,3)
        vertices=self.vertices*self.target_scale;edges=np.roll(vertices,-1,axis=0)-vertices
        q=c[:,:,:2]*self.target_scale
        delta=q[:,:,None]-vertices[None,None]
        t=np.clip(np.sum(delta*edges[None,None],axis=-1)/np.sum(edges*edges,axis=-1),0,1)
        projections=vertices[None,None]+t[...,None]*edges[None,None]
        index=np.argmin(np.sum((projections-q[:,:,None])**2,axis=-1),axis=-1)
        self.starts=vertices[index];self.edges=edges[index];self.length=np.linalg.norm(self.edges,axis=-1)
        self.direction=self.edges/self.length[...,None]
        arc=np.sum((q-self.starts)*self.direction,axis=-1)/self.target_scale
        self.initial=np.stack((np.clip(arc,0,self.length/self.target_scale),c[:,:,-1]),axis=-1).ravel()

    def unpack(self,control):
        c=control.reshape(self.steps,self.p.N,2)
        return c,None,self.starts+self.target_scale*c[:,:,:1]*self.direction,c[:,:,-1]

    def value_gradient(self,control):
        J,g=RegularizedSweep.value_gradient(self,control)
        g=g.reshape(self.steps,self.p.N,3)
        return J,np.stack((np.sum(g[:,:,:2]*self.direction,axis=-1),g[:,:,-1]),axis=-1).ravel()

    def full_control(self,control):
        _,_,target,a=self.unpack(control)
        return np.concatenate((target/self.target_scale,a[...,None]),axis=-1).ravel()

    def optimize_edges(self,iterations=5000,tolerance=1e-5,method='TNC'):
        history=[]
        def callback(c):
            value=self.forward(c)['J']
            if len(history)%100==0:print('Edge adjoint sweep',len(history),value,flush=True)
            history.append(value)
        bounds=[]
        for length in self.length.ravel():bounds.extend([(0,length/self.target_scale),(0,1-1e-7)])
        state=self.forward(self.initial)
        middle=radial_flow(state['x'][:-1],state['target'],state['a'],self.dt/2,self.p.V,self.eta)[0]
        q=state['target']-middle;R=np.sqrt(np.sum(q*q,axis=-1)+self.eta**2)
        target_gain=state['a'][...,None]*self.p.V*self.target_scale*(self.direction/R[...,None]-q*np.sum(q*self.direction,axis=-1)[...,None]/R[...,None]**3)
        gains=np.stack((np.linalg.norm(target_gain,axis=-1),self.p.V*np.linalg.norm(q,axis=-1)/R),axis=-1)
        scales=np.clip(1/np.maximum(gains,.05),.01,20).ravel()
        options=(dict(maxfun=iterations,gtol=tolerance,ftol=1e-12,xtol=1e-10,maxCGit=80,scale=scales,offset=self.initial.copy())
                 if method=='TNC' else dict(maxiter=iterations,maxls=50,ftol=1e-14,gtol=tolerance,maxcor=80))
        result=minimize(self.value_gradient,self.initial,jac=True,method=method,bounds=bounds,callback=callback,options=options)
        if method=='TNC' and 'Linear search failed' in str(result.message):
            # TNC's finite-difference Hessian can fail near a nearly stopped
            # target. The validated first derivative still supplies a descent
            # direction; switch to a bounded first-derivative line search.
            result=minimize(self.value_gradient,result.x,jac=True,method='L-BFGS-B',bounds=bounds,callback=callback,
                            options=dict(maxiter=min(3000,iterations),maxls=50,ftol=1e-14,gtol=tolerance,maxcor=80))
        # A few nearly stopped intervals can dominate the remaining gradient
        # with very different curvature scales. Polish their largest bounded
        # residuals one coordinate at a time, using the verified adjoint and
        # an objective-decreasing line search. This does not alter the cost.
        upper=np.array([bound[1] for bound in bounds])
        current=result.x.copy();value,gradient=self.value_gradient(current)
        coordinate_steps=0
        for _ in range(500):
            projected=np.clip(current-gradient,0,upper)-current
            if np.max(abs(projected))<1e-3:break
            index=int(np.argmax(abs(projected)));epsilon=1e-7
            plus=current.copy();minus=current.copy();plus[index]+=epsilon;minus[index]-=epsilon
            curvature=(self.value_gradient(plus)[1][index]-self.value_gradient(minus)[1][index])/(2*epsilon)
            displacement=np.clip(-gradient[index]/max(curvature,1e-3),-1,1)
            displacement=np.clip(current[index]+displacement,0,upper[index])-current[index]
            if gradient[index]*displacement>=0:displacement=projected[index]
            accepted=False
            for fraction in 2.**-np.arange(25):
                trial=current.copy();trial[index]+=fraction*displacement
                if self.forward(trial)['J']<=value+1e-4*fraction*gradient[index]*displacement:
                    accepted=True;break
            if not accepted:break
            current=trial;value,gradient=self.value_gradient(current)
            history.append(value);coordinate_steps+=1
        result.x=current
        J,g=self.value_gradient(result.x)
        lower=np.zeros_like(result.x);upper=np.array([b[1] for b in bounds])
        residual=float(np.max(abs(np.clip(result.x-g,lower,upper)-result.x)))
        return self.full_control(result.x),dict(initial_J=self.forward(self.initial)['J'],J=J,original_J=self.last['original_J'],projected_residual=residual,status=str(result.message),coordinate_polish_steps=coordinate_steps,history=history,history_is_accepted_iterates=True)
