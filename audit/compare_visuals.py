"""Matched visual comparison of the original displayed run and saved repair.

Only forward replay: no optimization, adjoint integration, or LSODA calls.
Run with an Agg-capable environment: python audit/compare_visuals.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.integrate import cumulative_trapezoid, solve_ivp
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image

from diagnostics import legacy_functions
from fishing import scenario
from test_notebooks import load_functions

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'audit/results/comparison'
COLORS = ['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#D55E00']
RUN_COLORS = ['#737373', '#B77620', '#0072B2']
LABELS = ['Original: displayed run', 'Repair: starting guess', 'Repair: accepted update']


def sample(sol, controls, p, times):
    y = sol.sol(times)
    x = y[:2*p.N].T.reshape(-1, p.N, 2)
    fish = y[2*p.N:2*p.N+p.K].T
    idx = np.round(times / (p.T/(len(controls)-1))).astype(int)
    U = controls[idx]
    q = p.grounds[None, None] - x[:, :, None]
    rho = np.linalg.norm(q, axis=-1)
    s = np.sum(U[:, :, :p.K, None]*np.exp(-p.alpha*rho**2)[..., None]
               * fish[:, None, :, None]*q/(rho[..., None]+1e-10), axis=2)
    sn = np.linalg.norm(s, axis=-1)
    speed = U[:, :, -1]*sn/(sn+1e-10)
    pair = x[:, :, None] - x[:, None]
    d2 = np.sum(pair**2, axis=-1)
    inv = 1/(p.epsilon+d2)
    inv[:, np.arange(p.N), np.arange(p.N)] = 0
    crowd_rate = p.beta*inv.sum(axis=(1, 2))
    crowd = cumulative_trapezoid(crowd_rate, times, initial=0)
    travel = p.gamma*cumulative_trapezoid(speed.sum(axis=1), times, initial=0)
    d2[:, np.arange(p.N), np.arange(p.N)] = np.inf
    return dict(x=x, fish=fish, catch=p.K*p.F0-fish.sum(axis=1), speed=speed,
                command=U[:, :, -1], crowd=crowd, travel=travel,
                separation=np.sqrt(d2.min(axis=(1, 2))),
                objective=crowd[-1]+travel[-1]+p.delta*fish[-1].sum())


def replay():
    p = scenario()
    archive = np.load(ROOT/'audit/results/updated_sam_fishin_controls.npz')
    np.testing.assert_array_equal(archive['grounds'], p.grounds)
    np.testing.assert_array_equal(archive['ships'], p.ships)
    accepted = archive['controls']
    old_controls = np.ones_like(accepted)
    seed = np.zeros_like(accepted)
    grid = np.linspace(0, p.T, len(seed))
    for ship in range(p.N):
        distances = np.linalg.norm(p.grounds-p.ships[ship], axis=1)
        target = int(np.argmin(distances))
        seed[:, ship, target] = 10.
        seed[:, ship, -1] = p.V*(grid+(grid[1]-grid[0])/2 <= max(0., distances[target]-.5)/p.V)
    params = [p.alpha, p.sigma, p.beta, p.rate, p.grounds, np.array([100., 0.]),
              2*p.N, p.K, p.ships.shape]
    y0 = np.r_[p.ships.ravel(), np.full(p.K, p.F0)]
    legacy = legacy_functions()
    # A Python callback deadline prevents an unbounded legacy replay.
    def bounded_solve(fun, *args, **kwargs):
        deadline = time.monotonic()+20
        def bounded(t, y):
            if time.monotonic() > deadline:
                raise TimeoutError('Legacy forward replay exceeded 20 seconds.')
            return fun(t, y)
        return solve_ivp(bounded, *args, **kwargs)
    legacy['solve_ivp'] = bounded_solve
    old = legacy['solve_state_equations'](legacy['state'], old_controls, params, p.T, y0, len(grid))
    if not old.success:
        raise RuntimeError(old.message)
    fixed = load_functions()
    sols = [old]
    for label, U in [('starting guess', seed), ('accepted update', accepted)]:
        print('Forward replay:', label, flush=True)
        sols.append(fixed['solve_state_equations'](fixed['state'], U,
                    params+[p.epsilon, p.gamma], p.T, y0, len(grid)))
    times = np.linspace(0, p.T, 16001)
    runs = [sample(sol, U, p, times) for sol, U in zip(sols, [old_controls, seed, accepted])]
    saved = json.loads((ROOT/'audit/results/updated_sam_fishin_repair.json').read_text())
    reference = json.loads((ROOT/'audit/results/diagnostics.json').read_text())
    np.testing.assert_allclose(runs[0]['catch'][-1], reference['legacy_initial']['harvested'], atol=1e-8)
    np.testing.assert_allclose(runs[2]['catch'][-1], saved['harvested'], atol=1e-8)
    for i, sol in enumerate(sols[1:]):
        np.testing.assert_allclose(fixed['fishing_objective'](sol, 2*p.N, p.K, p.delta),
                                   saved['J_history'][i], rtol=1e-8)
    for label, run in zip(LABELS, runs):
        print(label, 'stock range', run['fish'].min(), run['fish'].max(),
              'largest interpolated increase', np.max(np.diff(run['fish'], axis=0)), flush=True)
        assert np.all(np.isfinite(run['x']))
        assert run['fish'].min() >= -1e-6
        assert run['speed'].max() <= p.V+1e-12
        assert np.allclose(run['x'][0], p.ships)
    report = dict(T=p.T, original_commit='23043fe5525d3b27dc0401f733fc8a9ec169ca26',
                  notebook='random/updated_sam_fishin.ipynb',
                  description='Original trajectory displayed before its sole control update versus saved accepted repaired control. Same dynamics, map, weights and horizon; initialization and numerics differ.',
                  caution='Neither trajectory establishes convergence or mesh-independent accuracy. Crowding quadrature is sampled at 16001 times. Original displayed state was stale after its update.',
                  accepted_control_status=saved['stop_reason'], runs=[])
    for label, run in zip(LABELS, runs):
        report['runs'].append(dict(label=label, harvested=float(run['catch'][-1]),
                                  sampled_J=float(run['objective']), sampled_crowding=float(run['crowd'][-1]),
                                  sampled_travel=float(run['travel'][-1]),
                                  terminal_fish=float(p.delta*run['fish'][-1].sum()),
                                  largest_interpolated_stock_increase=float(np.max(np.diff(run['fish'], axis=0))),
                                  minimum_sampled_separation=float(run['separation'].min())))
    report['repair_integrated_J'] = saved['J_history']
    (OUT/'comparison.json').write_text(json.dumps(report, indent=2))
    np.savez_compressed(OUT/'trajectories.npz', times=times, grounds=p.grounds, ships=p.ships,
                        **{f'run{i}_{key}': value for i, run in enumerate(runs) for key, value in run.items()})
    print(json.dumps(report, indent=2), flush=True)
    return p, times, runs, saved


def style():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.titleweight': 'bold', 'figure.facecolor': '#fafbfc',
                         'axes.facecolor': 'white', 'savefig.facecolor': '#fafbfc'})


def map_axes(ax, p):
    ax.set(xlim=(-5, 205), ylim=(-5, 205), xlabel='x', ylabel='y', aspect='equal')
    ax.grid(alpha=.15)
    for j, (gx, gy) in enumerate(p.grounds):
        ax.annotate(f'G{j+1}', (gx, gy), (7, 6), textcoords='offset points', fontsize=8)
    ax.scatter(p.ships[:, 0], p.ships[:, 1], c=COLORS, marker='x', s=35, zorder=5)


def maps(p, times, runs):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6.8))
    fig.subplots_adjust(top=.79, bottom=.19, left=.06, right=.88, wspace=.18)
    fig.suptitle('Same fishing problem, before and after code repairs', fontsize=17, y=.97)
    fig.text(.5, .915, 'Five boats • seven grounds • T = 400 • identical map, dynamics and cost weights', ha='center')
    fig.text(.5, .875, 'Initialization and solver settings differ. Neither run demonstrates convergence.', ha='center', color='#9b3f22')
    artists = []
    for ax, index in zip(axs, [0, 2]):
        run = runs[index]
        map_axes(ax, p)
        ax.set_title(LABELS[index], fontsize=13)
        grounds = ax.scatter(p.grounds[:, 0], p.grounds[:, 1], s=220,
                             c=run['fish'][0], cmap='Blues', vmin=0, vmax=p.F0,
                             edgecolor='#344050', linewidth=.9, zorder=2)
        lines = [ax.plot([], [], color=color, lw=1.8, alpha=.8)[0] for color in COLORS]
        boats = ax.scatter(p.ships[:, 0], p.ships[:, 1], c=COLORS, s=55, marker='^', zorder=6,
                           edgecolor='white', linewidth=.5)
        caption = ax.text(.02, .02, '', transform=ax.transAxes, fontsize=10,
                          bbox=dict(facecolor='white', alpha=.85, edgecolor='none'))
        artists.append((run, grounds, lines, boats, caption))
    cax = fig.add_axes([.91, .26, .012, .44])
    fig.colorbar(grounds, cax=cax, label='Fish remaining / initial stock')
    fig.legend(handles=[Line2D([0], [0], color=c, label=f'Boat {i+1}') for i, c in enumerate(COLORS)],
               loc='lower center', bbox_to_anchor=(.5, .075), ncol=5, frameon=False)
    fig.text(.5, .035, '× = start; ▲ = boat; trails show travel; pale grounds are depleted. Both panels use the same scales.',
             ha='center', fontsize=9)
    clock = fig.text(.5, .135, '', ha='center', fontsize=12, weight='bold')
    def update(k):
        for run, grounds, lines, boats, caption in artists:
            grounds.set_array(run['fish'][k])
            boats.set_offsets(run['x'][k])
            for i, line in enumerate(lines):
                line.set_data(run['x'][:k+1:8, i, 0], run['x'][:k+1:8, i, 1])
            caption.set_text(f"Fish harvested: {run['catch'][k]:.3f} / 7")
        clock.set_text(f'Time {times[k]:.1f} / 400')
    update(len(times)-1)
    fig.savefig(OUT/'trajectories.png', dpi=160)
    # Quantize each frame immediately; keep memory bounded while creating the GIF.
    fig.set_dpi(90)
    frames = []
    indices = np.linspace(0, len(times)-1, 121).astype(int)
    for k in indices:
        update(k)
        fig.canvas.draw()
        frame = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
        frames.append(frame.quantize(colors=128))
    durations = [100]*len(frames)
    durations[-1] = 2200
    frames[0].save(OUT/'before_after.gif', save_all=True, append_images=frames[1:],
                   duration=durations, loop=0, optimize=False, disposal=2)
    plt.close(fig)


def plots(p, times, runs, saved):
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('What changed — harvest, crowding, travel and accepted cost', fontsize=16)
    for label, color, run in zip(LABELS, RUN_COLORS, runs):
        axs[0, 0].plot(times, run['catch'], color=color, label=label, lw=2)
        axs[0, 1].plot(times, run['crowd'], color=color, lw=2)
        axs[1, 0].plot(times, run['travel']/p.gamma, color=color, lw=2)
    axs[0, 0].set(title='Total fish harvested (higher is better)', ylim=(0, 7), ylabel='Fish stock units')
    axs[0, 0].legend(fontsize=9)
    axs[0, 1].set(title='Accumulated crowding penalty (lower is better)', yscale='symlog', ylabel='Cost; log scale above 1')
    axs[1, 0].set(title='Total distance traveled by all boats', ylabel='Distance')
    bars = [runs[0]['objective'], *saved['J_history']]
    axs[1, 1].bar(range(3), bars, color=RUN_COLORS, width=.6)
    axs[1, 1].set(yscale='log', ylim=(1, 2e8), title='Weighted objective J* (lower is better)',
                  xticks=range(3), xticklabels=['Original\ndisplayed', 'Repair\nstarting guess', 'Repair\naccepted update'],
                  ylabel='Objective; log scale')
    for i, value in enumerate(bars):
        axs[1, 1].text(i, value*1.5, f'{value:,.1f}', ha='center', fontsize=10)
    for ax in axs.flat:
        ax.grid(axis='y', alpha=.2)
    for ax in [axs[0, 0], axs[0, 1], axs[1, 0]]:
        ax.set(xlabel='Time', xlim=(0, p.T))
    fig.text(.5, .045, 'J* = crowding + 0.01 × distance + 100 × remaining fish; constant self-interaction cost removed.', ha='center', fontsize=10)
    fig.text(.5, .018, 'Original cost: sampled quadrature. Repair cost: forward-integrated. One accepted update; next adjoint failed.', ha='center', fontsize=9, color='#9b3f22')
    fig.tight_layout(rect=(0, .08, 1, .95))
    fig.savefig(OUT/'outcomes.png', dpi=150)
    plt.close(fig)

    fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey='row')
    ground_colors = plt.get_cmap('tab10').colors[:p.K]
    for col, index in enumerate([0, 2]):
        run = runs[index]
        for j, color in enumerate(ground_colors):
            axs[0, col].plot(times, run['fish'][:, j], color=color, label=f'G{j+1}')
        for i, color in enumerate(COLORS):
            axs[1, col].plot(times, run['speed'][:, i], color=color, label=f'Boat {i+1}', lw=1.2)
            axs[1, col].plot(times, run['command'][:, i], color=color, ls=':', alpha=.4, lw=1)
        axs[0, col].set(title=LABELS[index], ylim=(-.03, 1.05), ylabel='Fish remaining / initial stock')
        axs[1, col].set(ylim=(-.03, 1.05), xlabel='Time', ylabel='Boat speed')
        for row in range(2):
            axs[row, col].set_xlim(0, p.T)
            axs[row, col].grid(alpha=.15)
    axs[0, 0].legend(ncol=4, fontsize=8)
    axs[1, 0].legend(ncol=3, fontsize=8)
    fig.suptitle('Ground-by-ground stocks and boat speeds', fontsize=16)
    fig.text(.5, .025, 'Speed: solid = actual movement; dotted = commanded speed. Overlapping curves can hide individual boats.', ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .05, 1, .95))
    fig.savefig(OUT/'stocks_and_speeds.png', dpi=150)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    style()
    if '--cached' in sys.argv:
        p = scenario()
        data = np.load(OUT/'trajectories.npz')
        times = data['times']
        runs = [{key[len(f'run{i}_'):]: data[key] for key in data.files if key.startswith(f'run{i}_')}
                for i in range(3)]
        saved = json.loads((ROOT/'audit/results/updated_sam_fishin_repair.json').read_text())
    else:
        p, times, runs, saved = replay()
    plots(p, times, runs, saved)
    maps(p, times, runs)
    print('Saved comparison assets to', OUT, flush=True)


if __name__ == '__main__':
    main()
