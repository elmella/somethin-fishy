"""Validate and publish local PMP artifacts; never hides a failed check."""
import json
from pathlib import Path
import numpy as np
from fishing import scenario
from regularized_pmp import RegularizedSweep,verify_continuous,hamiltonian_gaps
from pmp_validation import RegularizedSchedule,replay_feedback,CONTROL_SPEED_ZERO_TOLERANCE,CONTROL_VELOCITY_RECONSTRUCTION_TOLERANCE
from pmp_edge import EdgeSweep
from pontryagin_controller import save,OUT


def stop_roundoff_arrivals(solver,control):
    """Replace sub-1e-10 interval displacements by feasible stops if a saved
    trajectory numerically reaches a singular ground center. Recheck the
    perturbed policy in full; this does not redefine the state equations.
    """
    from regularized_pmp import radial_flow
    before=solver.forward(control)
    clearance=np.linalg.norm(before['x'][:,:,None]-solver.p.grounds,axis=-1).min()
    if clearance>1e-12:return control,dict(changed_intervals=0)
    candidate=control.reshape(solver.steps,solver.p.N,3).copy()
    x=solver.p.ships.copy();changes=0
    for k in range(solver.steps):
        z=candidate[k,:,:2]*solver.target_scale;a=candidate[k,:,-1]
        end=radial_flow(x,z,a,solver.dt,solver.p.V,solver.eta)[0]
        tiny=(np.linalg.norm(end-x,axis=-1)<=1e-10)&(a>0)
        changes+=int(tiny.sum());a[tiny]=0
        x=radial_flow(x,z,a,solver.dt,solver.p.V,solver.eta)[0]
    candidate=candidate.ravel();after=solver.forward(candidate)
    info=dict(changed_intervals=changes,displacement_threshold=1e-10,
              max_position_change=float(np.max(abs(after['x']-before['x']))),
              distance_objective_change=float(after['distance_J']-before['distance_J']))
    if info['max_position_change']>1e-7 or abs(info['distance_objective_change'])>1e-5:
        raise AssertionError('Roundoff stop changed the policy materially.')
    return candidate,info


def finalize(source,name,T=400.,eta=.5,nu=.001,initialization='Saved adjoint continuation'):
    path=Path(source);u=np.load(path)['control'];p=scenario(T=T)
    source_report=json.loads(path.with_suffix('.json').read_text()) if path.with_suffix('.json').exists() else {}
    # Preserve the actual saved problem, including ablations such as beta=0.
    if 'problem' in source_report:
        from fishing import Problem
        values=dict(source_report['problem'])
        for key in ('grounds','ships'):values[key]=np.asarray(values[key])
        p=Problem(**values)
        if p.T!=T:raise ValueError('Requested horizon does not match saved problem.')
    if len(u)%(3*p.N):raise ValueError('Control dimensions do not match saved fleet size.')
    steps=len(u)//(3*p.N)
    s=RegularizedSweep(p,steps,eta,nu)
    u,roundoff_stops=stop_roundoff_arrivals(s,u);state=s.forward(u)
    face_distance=np.einsum('tni,ji->tnj',state['target'],s.halfspaces[:,:2])+s.halfspaces[:,2]
    on_edges=bool(np.max(-face_distance.max(axis=-1))<1e-6)
    if on_edges:
        edge=EdgeSweep(s,u);_,gradient=edge.value_gradient(edge.initial)
        upper=np.stack((edge.length/edge.target_scale,np.full_like(edge.length,1-1e-7)),axis=-1).ravel()
        residual=float(np.max(abs(np.clip(edge.initial-gradient,0,upper)-edge.initial)))
    else:
        _,gradient=s.value_gradient(u)
        residual=float(np.max(abs(s.project(u-gradient)-u)))
    dense=RegularizedSweep(p,steps*3,eta,nu);repeated=np.repeat(u.reshape(steps,p.N,3),3,axis=0).ravel()
    continuous=verify_continuous(dense,repeated,rtol=1e-9)
    ham=hamiltonian_gaps(dense,repeated,continuous['midpoint_costates'])
    schedule=RegularizedSchedule(s,u);checks=schedule.checks()
    replay=[replay_feedback(schedule,tolerance,step) for tolerance,step in [(1e-8,.5),(1e-10,.25)]]
    for row in replay:
        if row['max_position_error']>1e-5 or row['max_fish_error']>1e-7 or abs(row['distance_objective']-state['distance_J'])>1e-5:
            raise AssertionError('Physical feedback replay did not meet accuracy requirements.')
    if checks['minimum_center_clearance']<=0 or checks['max_velocity_reconstruction_error']>1e-9:raise AssertionError('Invalid physical control reconstruction.')
    history=source_report.get('history',[])
    # Only record objective values known to come from accepted iterates.
    if history and isinstance(history[0],(float,int)) and not source_report.get('history_is_accepted_iterates',False):
        history=[]
    initial_J=source_report.get('initial_J')
    converged=residual<1e-3 and ham['integrated_gap']<.005
    sensitivity=[]
    for radius in [.1,.25,.5,1.]:
        trial=RegularizedSweep(p,steps,radius,nu).forward(u)
        sensitivity.append(dict(eta=radius,distance_J=trial['distance_J'],harvested=float(p.K*p.F0-trial['fish'].sum()),reoptimized=False))
    report=dict(method='Pontryagin forward/backward adjoints and joint Hamiltonian control updates',
                residual_coordinates='Hull-edge arc lengths / 20 and speed fractions' if on_edges else 'Hull targets / 20 and speed fractions',
                initialization=initialization,execution='State-feedback attraction reconstruction; NOT a fixed raw-u table',
                source=path.name,eta=eta,travel_smoothing=nu,steps=steps,J=state['J'],distance_J=state['distance_J'],
                harvested=float(p.K*p.F0-state['fish'].sum()),crowding=state['crowd'],travel=float(p.gamma*np.linalg.norm(state['dx'],axis=-1).sum()),terminal_fish=float(p.delta*state['fish'].sum()),
                distance_per_boat=np.linalg.norm(state['dx'],axis=-1).sum(axis=0).tolist(),
                initial_J=initial_J,history=history,history_scope=('Saved accepted optimization cycles/iterates' if history else 'No verified per-iteration history retained; use validated endpoint comparisons'),
                projected_residual=residual,gradient_tolerance=1e-3,integrated_hamiltonian_gap=ham['integrated_gap'],
                max_hamiltonian_gap=ham['max_gap'],hamiltonian_tolerance=.005,hamiltonian_time_samples=steps*3,
                converged=converged,status='Numerical regularized-PMP tolerances met' if converged else 'Residual tolerances not met',
                optimizer_termination=source_report.get('optimizer_termination',source_report.get('status',source_report.get('message','See source report'))),
                continuous_replay=continuous['replay'],feedback_replays=replay,control_checks=checks,
                travel_smoothing_uniform_cost_bound=p.gamma*p.N*p.T*nu,regularization_sensitivity=sensitivity,
                executable_control_speed_zero_tolerance=CONTROL_SPEED_ZERO_TOLERANCE,
                executable_control_velocity_reconstruction_tolerance=CONTROL_VELOCITY_RECONSTRUCTION_TOLERANCE,
                roundoff_arrival_stops=roundoff_stops,
                global_optimality_claim=False)
    save(name,s,u,report)
    print({k:v for k,v in report.items() if k not in ['history','regularization_sensitivity']},flush=True)
    return s,u,report

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('source');parser.add_argument('name');parser.add_argument('--horizon',type=float,default=400.);parser.add_argument('--initialization',default='Adjoint continuation')
    args=parser.parse_args();finalize(args.source,args.name,args.horizon,initialization=args.initialization)
