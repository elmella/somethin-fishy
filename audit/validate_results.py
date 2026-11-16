"""Replay saved optimized controls through the original coupled ODE at two tolerances."""
import csv
import json
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from fishing import scenario, evaluate_routes, rhs

OUT = Path(__file__).parent/'results'


def replay(problem, trajectories, rtol):
    knots = sorted(set([s.start for tr in trajectories for s in tr]+[problem.T]))
    y = np.r_[problem.ships.ravel(), np.full(problem.K,problem.F0)]
    max_position_error = 0.
    min_fish = problem.F0
    for start, end in zip(knots[:-1],knots[1:]):
        middle = (start+end)/2
        active = [next(s for s in tr if s.start <= middle < s.end) for tr in trajectories]
        u = np.zeros((problem.N,problem.K))
        u[np.arange(problem.N),[s.target for s in active]] = 1
        v = np.array([np.linalg.norm(s.velocity) for s in active])
        assert np.all(v <= problem.V+1e-12)
        sol = solve_ivp(lambda t,state: rhs(state,u,v,problem), [start,end], y,
                        method='DOP853', rtol=rtol, atol=rtol*1e-3, max_step=2.)
        if not sol.success:
            raise RuntimeError(sol.message)
        y = sol.y[:,-1]
        min_fish = min(min_fish,float(sol.y[2*problem.N:].min()))
        expected = np.array([s.position(end) for s in active])
        max_position_error = max(max_position_error,float(np.max(np.abs(y[:2*problem.N].reshape(-1,2)-expected))))
    return dict(max_position_error=max_position_error,remaining=y[2*problem.N:].tolist(),
                min_fish=min_fish,rtol=rtol)


def main():
    results = json.loads((OUT/'experiments.json').read_text())
    verification = {}
    for name, data in results.items():
        p = scenario(name.startswith('shared'),data['parameters']['T'])
        best = data['optimized']
        expected, trajectories = evaluate_routes(p,best['routes'],best['dwells'])
        checks = []
        for tolerance in [1e-7,1e-9]:
            check = replay(p,trajectories,tolerance)
            check['max_fish_error'] = float(np.max(np.abs(np.array(check['remaining'])-expected['remaining'])))
            assert check['max_position_error'] < 1e-5
            assert check['max_fish_error'] < 1e-7
            assert check['min_fish'] >= 0
            checks.append(check)
        # Avoid all center singularities, including non-target grounds.
        min_center_distance = np.inf
        for tr in trajectories:
            for s in tr:
                q = s.x-p.grounds
                a = s.velocity@s.velocity
                t = np.zeros(p.K) if a == 0 else np.clip(-q@s.velocity/a,0,s.end-s.start)
                min_center_distance = min(min_center_distance,float(np.linalg.norm(q+t[:,None]*s.velocity,axis=1).min()))
        assert min_center_distance > 0
        verification[name] = dict(replays=checks,min_distance_to_any_center=min_center_distance,
                                  initial_positions_match=True,controls_in_bounds=True)
        with (OUT/(name+'_schedule.csv')).open('w',newline='') as stream:
            writer=csv.writer(stream)
            writer.writerow(['ship','start','end','target','u_target','speed','start_x','start_y','end_x','end_y'])
            for i,tr in enumerate(trajectories):
                for s in tr:
                    writer.writerow([i,s.start,s.end,s.target,1,np.linalg.norm(s.velocity),*s.x,*s.position(s.end)])
        print(name,verification[name],flush=True)
    (OUT/'verification.json').write_text(json.dumps(verification,indent=2))


if __name__ == '__main__':
    main()
