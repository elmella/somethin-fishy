"""Working bounded target/switch-time controller for the paper's fishing model.

Search one/two-ground schedules; independently replay the coupled equations.
This is direct optimization in a restricted feasible control family, not PMP
convergence or a global optimality certificate. No objective weights are tuned.
"""
import csv
import json
import time
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import linear_sum_assignment

from fishing import scenario, stable_rhs, evaluate_routes, running_cost
from experiments import search, quadrature_check

OUT = Path(__file__).parent/'results/working'


def optimize(problem, standoff=.5, max_passes=6):
    if not 0 < standoff <= 1:
        raise ValueError('Choose a small positive stopping offset (at most 1 spatial unit).')
    distances = np.linalg.norm(problem.ships[:, None]-problem.grounds, axis=2)
    _, assigned = linear_sum_assignment(distances)
    if len(assigned) != problem.N:
        raise ValueError('This distinct-ground initialization requires N <= K.')
    seeds = [[[int(j)] for j in assigned], [[int(j)] for j in distances.argmin(axis=1)]]
    fits = [search(problem, routes, np.zeros(problem.N), standoff, max_passes)[0] for routes in seeds]
    best = min(fits, key=lambda fit: fit['metrics']['J_reduced'])
    metrics, trajectories = evaluate_routes(problem, best['routes'], best['dwells'], standoff)
    best = dict(best, metrics=metrics, standoff=standoff,
                method='Direct one/two-ground route and first-dwell optimization',
                all_start_costs=[fit['metrics']['J_reduced'] for fit in fits],
                initial_metrics=evaluate_routes(problem, seeds[0], np.zeros(problem.N), standoff)[0],
                status=('No improving tested route/dwell change' if best['stopped_no_improvement']
                        else 'Pass limit reached'),
                global_optimality_claim=False)
    return best, trajectories


def replay(problem, trajectories, rtol=1e-9, max_step=2., sample_count=1601):
    """Full state + objective ODE, exact control switches, bounded Python solve."""
    deadline = time.monotonic()+40.
    knots = sorted({s.start for tr in trajectories for s in tr} | {problem.T})
    z = np.r_[problem.ships.ravel(), np.full(problem.K, problem.F0), 0., 0.]
    times = np.linspace(0, problem.T, sample_count)
    values = np.empty((len(z), len(times)))
    controls = np.zeros((len(times), problem.N, problem.K+2))
    max_position_error = 0.
    for start, end in zip(knots[:-1], knots[1:]):
        middle = (start+end)/2
        active = [next(s for s in tr if s.start <= middle < s.end) for tr in trajectories]
        u = np.zeros((problem.N, problem.K))
        u[np.arange(problem.N), [s.target for s in active]] = 1.
        v = np.array([np.linalg.norm(s.velocity) for s in active])
        def augmented(t, state):
            if time.monotonic() > deadline:
                raise RuntimeError('Forward validation exceeded its 40-second bound.')
            derivative = stable_rhs(state[:-2], u, v, problem)
            cost = running_cost(state[:-2], v, problem, include_self=False)
            return np.r_[derivative, cost-problem.gamma*v.sum(), problem.gamma*v.sum()]
        sol = solve_ivp(augmented, [start, end], z, method='DOP853', rtol=rtol,
                        atol=rtol*1e-3, max_step=max_step, dense_output=True)
        if not sol.success:
            raise RuntimeError(sol.message)
        z = sol.y[:, -1]
        expected = np.array([s.position(end) for s in active])
        max_position_error = max(max_position_error,
                                 float(np.max(np.abs(z[:2*problem.N].reshape(-1, 2)-expected))))
        selected = (times >= start) & ((times < end) | ((end == problem.T) & (times == end)))
        # A short control interval need not contain a display sample. It still
        # must be integrated; SciPy's dense interpolant rejects an empty input.
        if np.any(selected):
            values[:, selected] = sol.sol(times[selected])
        controls[selected, :, :problem.K] = u
        controls[selected, :, -1] = v
    fish = values[2*problem.N:2*problem.N+problem.K].T
    if not np.all(np.isfinite(values)) or fish.min() < 0 or fish.max() > problem.F0+1e-9:
        raise AssertionError('Invalid replay state.')
    if np.max(np.diff(fish, axis=0)) > 1e-8:
        raise AssertionError('Replay fish stock increased.')
    return dict(times=times, x=values[:2*problem.N].T.reshape(-1, problem.N, 2),
                fish=fish, controls=controls, crowd=values[-2], travel=values[-1],
                catch=problem.K*problem.F0-fish.sum(axis=1),
                objective=float(z[-2]+z[-1]+problem.delta*fish[-1].sum()),
                max_position_error=max_position_error, rtol=rtol, max_step=max_step)


def validate(problem, best, trajectories):
    checks = []
    expected = best['metrics']
    for tolerance, step in [(1e-7, 2.), (1e-9, .5)]:
        data = replay(problem, trajectories, tolerance, step)
        error = float(np.max(np.abs(data['fish'][-1]-expected['remaining'])))
        np.testing.assert_allclose(data['objective'], expected['J_reduced'], rtol=1e-7, atol=1e-5)
        assert error < 1e-7 and data['max_position_error'] < 1e-6
        checks.append(dict(rtol=tolerance, max_step=step, objective=data['objective'],
                           harvested=float(data['catch'][-1]), max_fish_error=error,
                           max_position_error=data['max_position_error']))
    quad = quadrature_check(problem, trajectories)
    np.testing.assert_allclose(quad['remaining'], expected['remaining'], atol=2e-9)
    np.testing.assert_allclose(quad['crowding'], expected['crowding'], rtol=1e-8, atol=1e-5)
    clearance = float('inf')
    for tr in trajectories:
        for segment in tr:
            q = segment.x-problem.grounds
            speed2 = segment.velocity@segment.velocity
            t = np.zeros(problem.K) if speed2 == 0 else np.clip(-q@segment.velocity/speed2, 0, segment.end-segment.start)
            clearance = min(clearance, float(np.linalg.norm(q+t[:, None]*segment.velocity, axis=1).min()))
    assert clearance > 0
    assert np.all(np.diff(best['history']) <= 0)
    assert data['controls'][:, :, :problem.K].max() <= 10
    assert data['controls'][:, :, -1].max() <= problem.V+1e-12
    return dict(replays=checks, quadrature=quad, minimum_center_clearance=clearance), data


def save(name, problem, best, trajectories, verification, data):
    OUT.mkdir(parents=True, exist_ok=True)
    report = dict(problem={k:(v.tolist() if isinstance(v, np.ndarray) else v) for k, v in vars(problem).items()},
                  solution=best, verification=verification)
    (OUT/(name+'.json')).write_text(json.dumps(report, indent=2))
    np.savez_compressed(OUT/(name+'.npz'), **data, grounds=problem.grounds, ships=problem.ships)
    with (OUT/(name+'_schedule.csv')).open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['boat_1_based', 'start', 'end', 'ground_1_based', 'attraction', 'speed', 'start_x', 'start_y', 'end_x', 'end_y'])
        for i, tr in enumerate(trajectories):
            for s in tr:
                writer.writerow([i+1, s.start, s.end, s.target+1, 1., np.linalg.norm(s.velocity), *s.x, *s.position(s.end)])


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario', choices=['notebook_400', 'notebook_600', 'harbor_400', 'all'], default='notebook_400')
    parser.add_argument('--standoff', type=float, default=.5)
    args = parser.parse_args()
    for name, harbor, T in [('notebook_400', False, 400.), ('notebook_600', False, 600.), ('harbor_400', True, 400.)]:
        if args.scenario not in ['all', name]:
            continue
        p = scenario(harbor, T)
        print('Optimizing', name, flush=True)
        best, trajectories = optimize(p, standoff=args.standoff)
        verification, data = validate(p, best, trajectories)
        save(name, p, best, trajectories, verification, data)
        print(name, best['status'], best['metrics'], flush=True)


if __name__ == '__main__':
    main()
