"""Explicit revised problem: add a harbor target and require every boat home at T."""
import csv
import json
import numpy as np
from scipy.optimize import linear_sum_assignment
from fishing import scenario,evaluate_return_routes,return_segments
from experiments import search,quadrature_check,make_plots,OUT


def main():
    p=scenario(harbor=True)
    distance=np.linalg.norm(p.ships[:,None]-p.grounds,axis=2)
    cost=distance.copy()
    cost[2*(distance-.5)/p.V>p.T]=1e9
    _,assigned=linear_sum_assignment(cost)
    distinct=dict(routes=[[int(j)] for j in assigned],dwells=[0.]*p.N)
    nearest=dict(routes=[[int(j)] for j in distance.argmin(axis=1)],dwells=[0.]*p.N)
    for start in [distinct,nearest]:
        start['metrics']=evaluate_return_routes(p,start['routes'],start['dwells'])[0]
    fits=[search(p,start['routes'],start['dwells'],return_home=True) for start in [distinct,nearest]]
    best,trajectories=min(fits,key=lambda pair:pair[0]['metrics']['J_reduced'])
    check=quadrature_check(p,trajectories)
    np.testing.assert_allclose(check['remaining'],best['metrics']['remaining'],rtol=2e-9,atol=2e-9)
    np.testing.assert_allclose(check['crowding'],best['metrics']['crowding'],rtol=2e-9,atol=1e-5)
    assert best['metrics']['J_reduced']<distinct['metrics']['J_reduced']
    max_terminal_error=0.
    max_direction_error=0.
    max_speed=0.
    for i,tr in enumerate(trajectories):
        np.testing.assert_allclose(tr[0].x,p.ships[i],atol=1e-12)
        assert tr[0].start==0 and tr[-1].end==p.T
        for a,b in zip(tr[:-1],tr[1:]):
            assert abs(a.end-b.start)<1e-10
            np.testing.assert_allclose(a.position(a.end),b.x,atol=1e-10)
        for s in tr:
            speed=np.linalg.norm(s.velocity)
            max_speed=max(max_speed,speed)
            assert speed<=p.V+1e-12
            if speed>0:
                target=p.ships[i] if s.target==-1 else p.grounds[s.target]
                for frac in [.1,.5,.9]:
                    point=s.position(s.start+frac*(s.end-s.start))
                    d=target-point;d/=np.linalg.norm(d)
                    max_direction_error=max(max_direction_error,float(np.linalg.norm(s.velocity-speed*d)))
        error=np.linalg.norm(tr[-1].position(p.T)-p.ships[i])
        max_terminal_error=max(max_terminal_error,float(error))
    assert max_terminal_error<1e-10 and max_direction_error<1e-10
    data=dict(model_change='Add non-depleting harbor attraction u_ih*d_ih; require x_i(T)=h; set velocity=0 when v_i=0.',
              dwell_parameter_units='fraction of available dwell budget at the first of two stops',
              parameters={k:v for k,v in vars(p).items() if k not in ['grounds','ships']},
              grounds=p.grounds.tolist(),ships=p.ships.tolist(),self_cost=p.self_cost,
              distinct=distinct,nearest=nearest,optimized=best,all_starts=[pair[0] for pair in fits],
              quadrature=check,verification=dict(max_terminal_error=max_terminal_error,
                  max_direction_error=max_direction_error,max_speed=max_speed),
              standoff_sensitivity={str(s):evaluate_return_routes(p,best['routes'],best['dwells'],s)[0]
                                    for s in [.25,.5,1.]})
    (OUT/'return_experiment.json').write_text(json.dumps(data,indent=2))
    with (OUT/'return_schedule.csv').open('w',newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(['ship','start','end','target (-1=harbor)','u_target','speed','start_x','start_y','end_x','end_y'])
        for i,tr in enumerate(trajectories):
            for s in tr:
                writer.writerow([i,s.start,s.end,s.target,1,np.linalg.norm(s.velocity),*s.x,*s.position(s.end)])
    make_plots(p,distinct,best,OUT/'return_to_harbor.png',return_home=True)
    print(json.dumps(data,indent=2),flush=True)


if __name__=='__main__':
    main()
