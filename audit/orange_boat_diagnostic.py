"""Compare feasible timing changes along the saved orange boat's exact path.

Other boats retain their original controls and timing. Constant-target intervals
are shortened by increasing a, preserving their exact radial-flow endpoints.
Time saved is added as a stationary interval at the end. No solver weights change.
"""
import json
from pathlib import Path
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import minimize_scalar
from fishing import scenario
from pontryagin_controller import load_saved
from regularized_pmp import radial_flow

OUT=Path(__file__).parent/'results/pontryagin'


def timing_variant(solver,control,windows,minimum_fraction):
    c=control.reshape(solver.steps,solver.p.N,3)
    targets=[];fractions=[];durations=[]
    for k in range(solver.steps):
        a=float(c[k,4,-1]);new_a=a;duration=solver.dt
        if any(lo<=k*solver.dt<hi for lo,hi in windows):
            new_a=max(a,minimum_fraction)
            duration=solver.dt*a/new_a if new_a>0 else solver.dt
        if duration>1e-11:
            targets.append(c[k,4,:2]*solver.target_scale);fractions.append(new_a);durations.append(duration)
    saved=solver.p.T-sum(durations)
    if saved>1e-10:
        targets.append(targets[-1]);fractions.append(0.);durations.append(saved)
    return np.array(durations),np.array(targets),np.array(fractions)


def evaluate(solver,control,orange,order=10):
    p=solver.p;_,_,targets,a=solver.unpack(control);durations,orange_targets,orange_a=orange
    orange_times=np.r_[0.,np.cumsum(durations)];orange_times[-1]=p.T
    times=np.unique(np.clip(np.r_[np.arange(solver.steps+1)*solver.dt,orange_times],0,p.T))
    x=p.ships.copy();exposure=np.zeros((p.N,p.K));pair_cost=np.zeros(len(solver.ii));distance=np.zeros(p.N)
    nodes,weights=leggauss(order);nodes=(nodes+1)/2;weights=weights/2
    trajectory=[x.copy()]
    for lo,hi in zip(times[:-1],times[1:]):
        dt=hi-lo
        if dt<1e-12:continue
        mid=(lo+hi)/2;k=min(int(mid/solver.dt),solver.steps-1);j=min(np.searchsorted(orange_times,mid,side='right')-1,len(orange_a)-1)
        z=targets[k].copy();fraction=a[k].copy();z[4]=orange_targets[j];fraction[4]=orange_a[j]
        nx=radial_flow(x[:,None],z[:,None],fraction[:,None],dt*nodes[None],p.V,solver.eta)[0]
        C=np.exp(-p.sigma*np.sum((p.grounds[None,None]-nx[:,:,None])**2,axis=-1))
        exposure+=dt*np.sum(C*weights[None,:,None],axis=1)
        pair=nx[solver.ii]-nx[solver.jj]
        pair_cost+=2*p.beta*dt*np.sum(weights[None]/(p.epsilon+np.sum(pair*pair,axis=-1)),axis=1)
        end=radial_flow(x,z,fraction,dt,p.V,solver.eta)[0]
        distance+=np.linalg.norm(end-x,axis=-1);x=end;trajectory.append(x.copy())
    fish=p.F0*np.exp(-p.rate*exposure.sum(axis=0));crowd=float(pair_cost.sum());travel=float(p.gamma*distance.sum());terminal=float(p.delta*fish.sum())
    return dict(original_cost=crowd+travel+terminal,crowding=crowd,travel=travel,terminal_fish_penalty=terminal,
                total_catch=float(p.K*p.F0-fish.sum()),remaining_by_ground=fish.tolist(),distance_per_boat=distance.tolist(),
                orange_pair_costs={f'boat_{i+1}':float(pair_cost[j]) for j,(i,l) in enumerate(zip(solver.ii,solver.jj)) if l==4},
                final_positions=x.tolist(),quadrature_order=order)


def main():
    p=scenario();s,u,report=load_saved('coverage_400_verified',p)
    cases=[('Saved policy',[],0.)]
    for name,windows in [('Initial wait',[(0,28)]),('Middle wait',[(88,126)]),('Later wait',[(172,206)]),('All three waits',[(0,28),(88,126),(172,206)]),('All travel before G5',[(0,252)])]:
        for fraction in [.1,.5,1.]:cases.append((f'{name}: minimum a={fraction:g}',windows,fraction))
    rows=[]
    for name,windows,fraction in cases:
        orange=timing_variant(s,u,windows,fraction);row=evaluate(s,u,orange)
        row.update(case=name,windows=windows,minimum_fraction=fraction)
        row['advance_of_final_rest']=float(orange[0][-1]) if windows else 0.
        rows.append(row)
        print(name,{key:round(row[key],7) for key in ['original_cost','crowding','travel','terminal_fish_penalty','total_catch']},flush=True)
    np.testing.assert_allclose(rows[0]['original_cost'],report['original_J'],atol=1e-8,rtol=0)
    baseline=np.array(rows[0]['final_positions']);distances=np.array(rows[0]['distance_per_boat'])
    for row in rows:
        np.testing.assert_allclose(row['final_positions'],baseline,atol=1e-7,rtol=0)
        np.testing.assert_allclose(row['distance_per_boat'],distances,atol=1e-7,rtol=0)
    checks=[]
    for name,windows,fraction in [cases[0],cases[-1],cases[10]]:
        fine=evaluate(s,u,timing_variant(s,u,windows,fraction),order=20)
        coarse=next(row for row in rows if row['case']==name)
        checks.append(dict(case=name,objective_difference=fine['original_cost']-coarse['original_cost']))
    # Test whether a single initial delay, followed by uninterrupted motion,
    # replaces the repeated waits. These are local one-parameter searches,
    # not joint reoptimization of the fleet.
    endpoint=s.forward(u)['x'][-1,4]
    assert np.max(s.halfspaces[:,:2]@endpoint+s.halfspaces[:,2])<1e-8
    def direct(delay):
        return np.array([delay,p.T-delay]),np.repeat(endpoint[None],2,axis=0),np.array([0.,1.])
    fast=timing_variant(s,u,[(0,400)],1.);available=float(fast[0][-1])
    def single_wait(delay):
        return (np.r_[delay,fast[0][:-1],available-delay],
                np.vstack([fast[1][0],fast[1][:-1],fast[1][-1]]),
                np.r_[0.,fast[2][:-1],0.])
    alternatives=[]
    for label,make,upper in [('Straight to same final position',direct,230.),
                              ('Same original path, one initial wait',single_wait,available-1)]:
        for delay in [0.,28.,80.,120.,150.]:
            row=evaluate(s,u,make(delay));row.update(case=label,delay=delay);alternatives.append(row)
        fit=minimize_scalar(lambda delay:evaluate(s,u,make(delay))['original_cost'],bounds=(0.,upper),
                            method='bounded',options=dict(xatol=1e-4))
        row=evaluate(s,u,make(fit.x),order=20)
        row.update(case=label+' (locally optimized departure)',delay=float(fit.x),scalar_search_success=bool(fit.success))
        alternatives.append(row);print(row['case'],row['delay'],row['original_cost'],flush=True)
    from pontryagin_visuals import sample_run
    run=sample_run(s,u,count=801);snapshots=[]
    targets=s.forward(u)['target'];fractions=s.forward(u)['a']
    for t in [0.,20.,100.,190.,252.,400.]:
        j=int(t*2);k=min(int(t/s.dt),s.steps-1);x=run['x'][j,4]
        distances=np.linalg.norm(p.grounds-x,axis=-1)
        snapshots.append(dict(time=t,position=x.tolist(),speed=float(run['speed'][j,4]),
                              commanded_fraction=float(fractions[k,4]),target_distance=float(np.linalg.norm(targets[k,4]-x)),
                              nearest_ground_distance=float(distances.min()),
                              instantaneous_catch=float(p.rate*np.sum(run['fish'][j]*np.exp(-p.sigma*distances**2)))))
    result=dict(policy='coverage_400_verified',boat=5,color='#D55E00',method=__doc__,cases=rows,quadrature_checks=checks,
                alternative_schedules=alternatives,snapshots=snapshots,
                unchanged_endpoints_and_path_lengths_verified=True,global_optimality_claim=False)
    (OUT/'orange_boat_diagnostic.json').write_text(json.dumps(result,indent=2))
    print('Quadrature checks',checks,flush=True)
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(10,5.5),sharex=True)
    axes[0].plot(run['times'],run['speed'][:,4],color='#D55E00',lw=2)
    nearest=np.linalg.norm(p.grounds[None]-run['x'][:,4,None],axis=-1).min(axis=-1)
    axes[1].plot(run['times'],nearest,color='#D55E00',lw=2)
    for ax in axes:
        for lo,hi in [(0,28),(88,126),(172,206)]:ax.axvspan(lo,hi,color='#E69F00',alpha=.15)
        ax.axvline(252,color='#777777',ls='--',lw=1);ax.grid(alpha=.2);ax.set_xlim(0,p.T)
    axes[0].set(ylabel='Actual speed',ylim=(-.03,1.08),title='Orange boat toward G5: shaded intervals are near-stops away from fishing grounds')
    axes[1].set(ylabel='Distance to nearest ground',xlabel='Time',ylim=(-2,75))
    axes[1].annotate('Fishing at G5',xy=(260,.5),xytext=(285,32),arrowprops=dict(arrowstyle='->',color='#555555'))
    fig.tight_layout();fig.savefig(OUT/'orange_boat_waits.png',dpi=150);plt.close(fig)

if __name__=='__main__':main()
