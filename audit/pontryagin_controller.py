"""Forward states, backward adjoints, and joint Hamiltonian control updates.

The route optimizer is only a separately labeled initialization/benchmark.
Regularizations eta and nu are explicit and reported with every result.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from fishing import scenario
from regularized_pmp import RegularizedSweep, verify_continuous, hamiltonian_gaps

OUT=Path(__file__).parent/'results/pontryagin'


class FeedbackPolicy:
    """Executable original-model controls; raw attractions are state dependent.

    Integrators must split at ``times``. Display samples of u are not a
    substitute for calling controls(t, current_state) during execution.
    """
    def __init__(self,solver,control):
        self.problem=solver.p;self.eta=solver.eta;self.dt=solver.dt
        self.steps=solver.steps;self.times=np.linspace(0,solver.p.T,solver.steps+1)
        _,_,target,a=solver.unpack(control)
        self.targets=target.copy();self.fractions=a.copy()

    def controls(self,t,state):
        from pmp_validation import reconstruct_attractions
        p=self.problem;k=min(max(int(t/self.dt),0),self.steps-1)
        x=np.asarray(state[:2*p.N]).reshape(p.N,2);fish=np.asarray(state[2*p.N:2*p.N+p.K])
        q=self.targets[k]-x
        velocity=self.fractions[k,:,None]*p.V*q/np.sqrt(np.sum(q*q,axis=-1)+self.eta**2)[:,None]
        return reconstruct_attractions(x,fish,velocity,p)

    def rhs(self,t,state):
        from fishing import stable_rhs
        u,v=self.controls(t,state)
        return stable_rhs(state,u,v,self.problem)


def load_saved(name,problem):
    """Load an explicitly selected result, rejecting mismatched configurations."""
    report=json.loads((OUT/(name+'.json')).read_text())
    for key,value in vars(problem).items():
        np.testing.assert_allclose(value,report['problem'][key],rtol=0,atol=0,
                                   err_msg='Saved PMP result does not match '+key)
    data=np.load(OUT/(name+'.npz'))
    solver=RegularizedSweep(problem,int(data['steps']),float(data['eta']),float(data['nu']))
    control=data['control'].copy()
    if not np.all(np.isfinite(control)):raise ValueError('Nonfinite saved controls.')
    np.testing.assert_allclose(solver.project(control),control,rtol=0,atol=1e-8,
                               err_msg='Saved controls violate their bounds.')
    np.testing.assert_allclose(solver.forward(control)['original_J'],report['original_J'],rtol=1e-9,atol=1e-8,
                               err_msg='Saved controls and reported objective do not match.')
    return solver,control,report


def polish(solver,control,iterations=1200,tolerance=1e-3):
    """Adjoint-gradient control update with an augmented-Lagrangian hull constraint.

    Every objective/gradient call performs the forward and backward passes.
    Intermediate AL iterates may be infeasible; only projected controls are
    returned or scored as fishing policies. L-BFGS-B enforces speed bounds.
    """
    s=solver;u=s.project(control);initial=s.forward(u);best=u.copy();best_J=initial['J']
    normals=s.halfspaces[:,:2];offset=s.halfspaces[:,2]/s.target_scale
    multipliers=np.zeros((s.steps,s.p.N,len(normals)));history=[]
    bounds=[(None,None),(None,None),(0,1-1e-7)]*(s.steps*s.p.N)
    def constraints(c):
        return np.einsum('tni,ji->tnj',c.reshape(s.steps,s.p.N,3)[:,:,:2],normals)+offset
    for penalty in [1.,10.,100.,1000.,10000.,10000.]:
        def fg(c):
            J,g=s.value_gradient(c);w=np.maximum(0,multipliers+penalty*constraints(c))
            value=J+np.sum(w*w-multipliers*multipliers)/(2*penalty)
            gradient=g.reshape(s.steps,s.p.N,3).copy()
            gradient[:,:,:2]+=np.einsum('tnj,ji->tni',w,normals)
            return value,gradient.ravel()
        result=minimize(fg,u,jac=True,method='L-BFGS-B',bounds=bounds,
                        options=dict(maxiter=iterations,maxls=40,ftol=1e-13,gtol=1e-6,maxcor=30))
        u=result.x;multipliers=np.maximum(0,multipliers+penalty*constraints(u))
        candidate=s.project(u);J,g=s.value_gradient(candidate)
        residual=float(np.max(abs(s.project(candidate-g)-candidate)))
        row=dict(penalty=penalty,J=J,original_J=s.last['original_J'],projected_residual=residual,
                 max_constraint_violation=float(max(0,constraints(u).max())),inner_status=str(result.message),iterations=int(result.nit))
        history.append(row);print('Adjoint polish',row,flush=True)
        if J<=best_J:
            best=candidate.copy();best_J=J
        if residual<tolerance and J<=best_J+1e-7:
            best=candidate.copy();break
    J,g=s.value_gradient(best)
    return best,dict(initial_J=initial['J'],J=J,original_J=s.last['original_J'],
                     projected_residual=float(np.max(abs(s.project(best-g)-best))),history=history)


def reset_stopped_headings(solver,control,best_control):
    """A joint PMP correction that a local heading/speed gradient can miss.

    At a=0, heading is a flat direction. Replace its target by the Hamiltonian
    maximizer without changing speed or trajectory. The next speed update can
    then exploit a genuinely beneficial direction.
    """
    c=control.reshape(solver.steps,solver.p.N,3).copy();maximum=best_control.reshape(c.shape)
    stopped=c[:,:,-1]<1e-14
    changed=stopped&(np.linalg.norm(c[:,:,:2]-maximum[:,:,:2],axis=-1)>1e-8)
    c[:,:,:2][changed]=maximum[:,:,:2][changed]
    candidate=solver.project(c.ravel())
    np.testing.assert_allclose(solver.forward(candidate)['x'],solver.forward(control)['x'],rtol=0,atol=1e-8)
    return candidate,int(changed.sum())


def needle_updates(solver,control,hamiltonian,limit=12):
    """Try the full joint PMP update on the largest-gap control intervals."""
    c=control.reshape(solver.steps,solver.p.N,3).copy();best=hamiltonian['best_control'].reshape(c.shape)
    J=solver.forward(c.ravel())['J'];accepted=0
    for flat in np.argsort(hamiltonian['gaps'].ravel())[-limit:][::-1]:
        k,i=np.unravel_index(flat,hamiltonian['gaps'].shape)
        trial=c.copy();trial[k,i]=best[k,i]
        candidate=solver.project(trial.ravel());value=solver.forward(candidate)['J']
        if value<J-1e-9:c=candidate.reshape(c.shape);J=value;accepted+=1
    return c.ravel(),accepted


def fit(solver,control,cycles=6,iterations=1200,gradient_tolerance=1e-3,hamiltonian_tolerance=.005,edge_evaluations=3000):
    from pmp_edge import EdgeSweep,edge_initialization
    initial=solver.forward(control);history=[];u=control
    for cycle in range(cycles):
        u,optimization=polish(solver,u,iterations,gradient_tolerance)
        # All continuously attainable headings have a hull-edge representative.
        # Edge coordinates remove the redundant target-radius variable. At a
        # finite control interval they change its speed profile, so acceptance
        # is based on a fresh complete objective evaluation.
        edge=EdgeSweep(solver,edge_initialization(solver,u))
        candidate,edge_report=edge.optimize_edges(edge_evaluations)
        if solver.forward(candidate)['J']<=optimization['J']:
            u=candidate;optimization.update(J=edge_report['J'],original_J=edge_report['original_J'],
                                            projected_residual=edge_report['projected_residual'],edge_polish=edge_report)
        verification=verify_continuous(solver,u)
        ham=hamiltonian_gaps(solver,u,verification['midpoint_costates'])
        row=dict(cycle=cycle,**optimization,integrated_hamiltonian_gap=ham['integrated_gap'],max_hamiltonian_gap=ham['max_gap'])
        history.append(row)
        print('Forward-backward cycle', {k:v for k,v in row.items() if k not in ['history','edge_polish']},flush=True)
        if optimization['projected_residual']<gradient_tolerance and ham['integrated_gap']<hamiltonian_tolerance:break
        if cycle+1<cycles:
            u,count=reset_stopped_headings(solver,u,ham['best_control'])
            u,needles=needle_updates(solver,u,ham)
            print('Joint Hamiltonian updates:',count,'stopped headings;',needles,'accepted intervals',flush=True)
    state=solver.forward(u)
    converged=optimization['projected_residual']<gradient_tolerance and ham['integrated_gap']<hamiltonian_tolerance
    report=dict(method='Forward/backward adjoints with joint Hamiltonian updates; regularized effective targets',
                eta=solver.eta,travel_smoothing=solver.travel_smoothing,steps=solver.steps,
                initial_J=initial['J'],initial_original_J=initial['original_J'],initial_catch=float(solver.p.K*solver.p.F0-initial['fish'].sum()),
                J=state['J'],original_J=state['original_J'],harvested=float(solver.p.K*solver.p.F0-state['fish'].sum()),
                converged=converged,status='Numerical residual tolerances met' if converged else 'Iteration limit; residual tolerances not met',
                gradient_tolerance=gradient_tolerance,hamiltonian_tolerance=hamiltonian_tolerance,
                projected_residual=optimization['projected_residual'],integrated_hamiltonian_gap=ham['integrated_gap'],max_hamiltonian_gap=ham['max_gap'],
                replay=verification['replay'],history=history,global_optimality_claim=False,
                original_unregularized_PMP_convergence_claim=False)
    return u,report


def save(name,solver,control,report):
    OUT.mkdir(exist_ok=True,parents=True)
    report=dict(report,problem={key:(value.tolist() if isinstance(value,np.ndarray) else value) for key,value in vars(solver.p).items()})
    (OUT/(name+'.json')).write_text(json.dumps(report,indent=2))
    state=solver.forward(control)
    np.savez_compressed(OUT/(name+'.npz'),control=control,x=state['x'],fish_final=state['fish'],target=state['target'],a=state['a'],
                        grounds=solver.p.grounds,ships=solver.p.ships,eta=solver.eta,nu=solver.travel_smoothing,T=solver.p.T,steps=solver.steps)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',choices=['coverage','coverage-matched','coverage-deadline','benchmark'],default='coverage')
    parser.add_argument('--steps',type=int,default=200);parser.add_argument('--eta',type=float,default=.5)
    parser.add_argument('--nu',type=float,default=.001);parser.add_argument('--iterations',type=int,default=1200)
    parser.add_argument('--cycles',type=int,default=6);parser.add_argument('--horizon',type=float,default=400.)
    parser.add_argument('--beta',type=float,default=5.,help='Crowding weight; zero removes crowding from the objective.')
    parser.add_argument('--warm-file');parser.add_argument('--name',default='coverage_400')
    args=parser.parse_args();problem=scenario(T=args.horizon);problem.beta=args.beta
    s=RegularizedSweep(problem,args.steps,args.eta,args.nu)
    if args.warm_file:
        data=np.load(args.warm_file);u=data['control'];old_steps=len(u)//(s.p.N*3)
        if s.steps%old_steps:raise ValueError('Refinement must be an integer multiple of the saved grid.')
        u=np.repeat(u.reshape(old_steps,s.p.N,3),s.steps//old_steps,axis=0).ravel()
    else:u=s.seed(args.seed)
    control,report=fit(s,u,args.cycles,args.iterations);report['initialization']=args.warm_file or args.seed
    save(args.name,s,control,report)
    print({k:v for k,v in report.items() if k!='history'},flush=True)

if __name__=='__main__':main()
