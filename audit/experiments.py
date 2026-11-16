"""Run with python audit/experiments.py; write exact, reproducible comparisons.

Search is coordinate descent over one/two-ground routes and first-stop dwell.
This is a restricted-control improvement experiment, NOT a global PMP solver.
"""
import json
from pathlib import Path
from itertools import permutations
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment
from scipy.integrate import quad
from fishing import (scenario, evaluate_routes, evaluate_return_routes, gaussian_exposure, pair_crowding)

OUT = Path(__file__).parent/'results'


def search(problem, initial_routes, initial_dwells, standoff=.5, max_passes=6, return_home=False):
    evaluator = evaluate_return_routes if return_home else evaluate_routes
    routes = [list(r) for r in initial_routes]
    dwells = np.array(initial_dwells, dtype=float)
    result, _ = evaluator(problem, routes, dwells, standoff)
    history = [result['J_reduced']]
    candidates = [[j] for j in range(problem.K)] + list(map(list, permutations(range(problem.K), 2)))
    dwell_grid = [0., .25, .5, .75, 1.] if return_home else [0., 20., 50., 100., 180., 280., problem.T]
    for iteration in range(max_passes):
        before = history[-1]
        for ship in range(problem.N):
            best = history[-1]
            best_route, best_dwell = routes[ship], dwells[ship]
            for route in candidates:
                trial_routes = routes.copy(); trial_routes[ship] = route
                for dwell in (dwell_grid if len(route) == 2 else [0.]):
                    trial_dwells = dwells.copy(); trial_dwells[ship] = dwell
                    value = evaluator(problem, trial_routes, trial_dwells, standoff)[0]['J_reduced']
                    if value < best-1e-9:
                        best, best_route, best_dwell = value, route, dwell
            routes[ship], dwells[ship] = best_route, best_dwell
            if best < history[-1]:
                history.append(best)
        # Continuous dwell refinement leaves the control bounds and J unchanged.
        fit = minimize(lambda z: evaluator(problem, routes, z, standoff)[0]['J_reduced'],
                       dwells, method='L-BFGS-B', bounds=[(0., 1. if return_home else problem.T)]*problem.N,
                       options={'ftol':1e-12, 'gtol':1e-6, 'maxiter':150, 'maxls':30})
        if fit.fun < history[-1]:
            dwells = fit.x
            history.append(float(fit.fun))
        print('pass', iteration+1, 'J without constant', history[-1], flush=True)
        if before-history[-1] < 1e-7:
            break
    result, trajectories = evaluator(problem, routes, dwells, standoff)
    return dict(metrics=result, routes=routes, dwells=dwells.tolist(), history=history,
                passes=iteration+1, stopped_no_improvement=bool(before-history[-1] < 1e-7),
                dwell_optimizer_success=bool(fit.success), dwell_optimizer_message=str(fit.message)), trajectories


def quadrature_check(problem, trajectories):
    # Separate adaptive quadrature, not the closed-form evaluator used by search.
    exposure = np.zeros(problem.K)
    for trajectory in trajectories:
        for s in trajectory:
            for j, ground in enumerate(problem.grounds):
                exposure[j] += quad(lambda t: np.exp(-problem.sigma*np.sum((s.position(t)-ground)**2)),
                                    s.start, s.end, epsabs=1e-10, epsrel=1e-10,
                                    points=np.linspace(s.start, s.end, 10)[1:-1], limit=200)[0]
    crowd = 0.
    for i in range(problem.N):
        for j in range(i):
            for a in trajectories[i]:
                for b in trajectories[j]:
                    lo, hi = max(a.start,b.start), min(a.end,b.end)
                    if hi <= lo:
                        continue
                    def integrand(t):
                        return 1/(problem.epsilon+np.sum((a.position(t)-b.position(t))**2))
                    relative = a.velocity-b.velocity
                    start = a.position(lo)-b.position(lo)
                    speed2 = relative@relative
                    # Resolve the narrow crowding peak, including t=0 at harbor.
                    closest = lo if speed2 == 0 else np.clip(lo-start@relative/speed2, lo, hi)
                    width = np.sqrt(problem.epsilon/max(speed2, 1e-30))
                    points = sorted(set([float(z) for z in [closest, closest-width, closest+width,
                                     closest-10*width, closest+10*width] if lo < z < hi]))
                    crowd += 2*problem.beta*quad(integrand, lo, hi, points=points,
                                                epsabs=1e-8, epsrel=1e-10, limit=200)[0]
    return dict(remaining=(problem.F0*np.exp(-problem.rate*exposure)).tolist(), crowding=crowd)


def make_plots(problem, initial, optimized, path, return_home=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = plt.get_cmap('tab10')
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    for ax, data, title in zip(axes[:2], [initial, optimized], ['Distinct-target baseline', 'Optimized target / dwell schedule']):
        evaluator = evaluate_return_routes if return_home else evaluate_routes
        result, trajectories = evaluator(problem, data['routes'], data['dwells'])
        ax.scatter(*problem.grounds.T, marker='x', s=65, color='black')
        for j, point in enumerate(problem.grounds):
            ax.annotate(str(j), point+np.array([2, 2]), fontsize=9)
        for i, trajectory in enumerate(trajectories):
            points = np.array([s.x for s in trajectory]+[trajectory[-1].position(problem.T)])
            ax.plot(*points.T, color=colors(i), lw=2)
            ax.scatter(*points[0], marker='^', color=colors(i), s=45)
            ax.scatter(*points[-1], marker='o', color=colors(i), s=25)
        ax.set(xlim=(0, 205), ylim=(-5, 205), xlabel='x', ylabel='y', aspect='equal',
               title=title+'\nCatch {:.3f} / 7; J* {:.2f}'.format(result['harvested'],result['J_reduced']))
    axes[2].plot(optimized['history'], marker='.', color='#007c91')
    same_start = abs(optimized['history'][0]-initial['metrics']['J_reduced']) < 1e-6
    start_label = 'distinct-target' if same_start else 'nearest-target'
    axes[2].set(xlabel='Accepted update', ylabel='J* (self-crowding constant removed)',
                title='Actual objective: best run\n'+start_label+' initialization')
    if max(optimized['history'])/min(optimized['history']) > 100:
        axes[2].set_yscale('log')
    axes[2].grid(alpha=.2)
    start_kind = 'Shared harbor' if np.all(problem.ships == problem.ships[0]) else 'Notebook starting positions'
    suffix='; mandatory return' if return_home else '; free endpoints'
    fig.suptitle(start_kind+': 5 ships, 7 grounds, T = '+str(int(problem.T))+suffix, fontsize=14)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    results = {}
    for name, harbor, T in [('notebook_400',False,400.), ('notebook_600',False,600.), ('shared_harbor_400',True,400.)]:
        p = scenario(harbor, T)
        distances = np.linalg.norm(p.ships[:,None]-p.grounds, axis=2)
        _, assigned = linear_sum_assignment(distances)
        nearest = dict(routes=[[int(j)] for j in distances.argmin(axis=1)],dwells=[0.]*p.N)
        distinct = dict(routes=[[int(j)] for j in assigned],dwells=[0.]*p.N)
        nearest['metrics'] = evaluate_routes(p, nearest['routes'], nearest['dwells'])[0]
        distinct['metrics'] = evaluate_routes(p, distinct['routes'], distinct['dwells'])[0]
        print(name, 'baseline', distinct['metrics'], flush=True)
        starts = [distinct, nearest]
        fits = [search(p, start['routes'], start['dwells'])[0] for start in starts]
        best = min(fits, key=lambda fit: fit['metrics']['J_reduced'])
        trajectories = evaluate_routes(p,best['routes'],best['dwells'])[1]
        check = quadrature_check(p, trajectories)
        np.testing.assert_allclose(check['remaining'], best['metrics']['remaining'], atol=2e-9, rtol=2e-9)
        np.testing.assert_allclose(check['crowding'], best['metrics']['crowding'], atol=1e-5, rtol=2e-9)
        assert all(np.all(np.diff(fit['history']) <= 0) for fit in fits)
        assert best['metrics']['J_reduced'] < distinct['metrics']['J_reduced']
        sensitivity = {str(s):evaluate_routes(p,best['routes'],best['dwells'],s)[0] for s in [.25,.5,1.]}
        results[name] = dict(parameters={k:v for k,v in vars(p).items() if k not in ['grounds','ships']},
                             grounds=p.grounds.tolist(), ships=p.ships.tolist(), self_cost=p.self_cost,
                             nearest=nearest, distinct=distinct, optimized=best, all_starts=fits,
                             quadrature=check, standoff_sensitivity=sensitivity)
        (OUT/'experiments.json').write_text(json.dumps(results,indent=2))
        make_plots(p,distinct,best,OUT/(name+'.png'))
        print(name, 'FINAL', best['metrics'], flush=True)


if __name__ == '__main__':
    main()
