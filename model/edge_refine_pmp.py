"""Resume a saved PMP solution with joint Hamiltonian and bounded Newton updates."""
import argparse,json
from pathlib import Path
import numpy as np
from fishing import scenario,Problem
from regularized_pmp import RegularizedSweep,verify_continuous,hamiltonian_gaps
from pmp_edge import EdgeSweep,edge_initialization
from pontryagin_controller import reset_stopped_headings,needle_updates,save


def refine(source,name,horizon=400.,steps=None,cycles=3,evaluations=8000):
    data=np.load(source);u=data['control'];p=scenario(T=horizon)
    report_path=Path(source).with_suffix('.json')
    report=json.loads(report_path.read_text()) if report_path.exists() else {}
    if 'problem' in report:
        values=dict(report['problem'])
        for key in ('grounds','ships'):values[key]=np.asarray(values[key])
        p=Problem(**values)
        if p.T!=horizon:raise ValueError('Requested horizon does not match saved problem.')
    if len(u)%(3*p.N):raise ValueError('Control dimensions do not match saved fleet size.')
    old_steps=len(u)//(3*p.N);steps=steps or old_steps
    if steps%old_steps:raise ValueError('Refinement factor must be an integer.')
    u=np.repeat(u.reshape(old_steps,p.N,3),steps//old_steps,axis=0).ravel()
    s=RegularizedSweep(p,steps,eta=float(data['eta']),travel_smoothing=float(data['nu']));initial=s.forward(u);history=[]
    for cycle in range(cycles):
        verify=verify_continuous(s,u);ham=hamiltonian_gaps(s,u,verify['midpoint_costates'])
        u,count=reset_stopped_headings(s,u,ham['best_control']);u,needles=needle_updates(s,u,ham)
        print('Joint update',cycle,ham['integrated_gap'],count,needles,flush=True)
        edge=EdgeSweep(s,edge_initialization(s,u));u,fit=edge.optimize_edges(evaluations)
        verify=verify_continuous(s,u);ham=hamiltonian_gaps(s,u,verify['midpoint_costates'])
        row=dict(cycle=cycle,J=fit['J'],projected_residual=fit['projected_residual'],integrated_hamiltonian_gap=ham['integrated_gap'],max_hamiltonian_gap=ham['max_gap'])
        history.append(row);converged=fit['projected_residual']<1e-3 and ham['integrated_gap']<.005
        report=dict(initial_J=initial['J'],J=fit['J'],distance_J=s.forward(u)['distance_J'],history=history,converged=converged,
                    status='Numerical tolerances met' if converged else 'Residual tolerances not met',optimizer_termination=fit['status'],
                    projected_residual=fit['projected_residual'],integrated_hamiltonian_gap=ham['integrated_gap'],max_hamiltonian_gap=ham['max_gap'],
                    initialization=str(source),method='Forward/backward adjoints, joint Hamiltonian updates, bounded Newton polishing')
        save(name,s,u,report);print('REFINED',row,flush=True)
        if converged:break
    return s,u,report

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('source');parser.add_argument('name');parser.add_argument('--horizon',type=float,default=400.);parser.add_argument('--steps',type=int);parser.add_argument('--cycles',type=int,default=3);parser.add_argument('--evaluations',type=int,default=8000)
    args=parser.parse_args();refine(args.source,args.name,args.horizon,args.steps,args.cycles,args.evaluations)
