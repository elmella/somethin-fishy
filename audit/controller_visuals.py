"""Scientific plots and a synchronized original/working-controller animation."""
import json
import time
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image

from fishing import scenario
from diagnostics import legacy_functions
from compare_visuals import sample, style, COLORS

OUT = Path(__file__).parent/'results/working'


def original_run(problem):
    """Pinned original displayed all-on policy, with its original ODE settings."""
    ns = legacy_functions()
    deadline = time.monotonic()+20.
    def bounded_solve(fun, *args, **kwargs):
        def bounded(t, y):
            if time.monotonic() > deadline:
                raise RuntimeError('Original forward comparison exceeded 20 seconds.')
            return fun(t, y)
        return solve_ivp(bounded, *args, **kwargs)
    ns['solve_ivp'] = bounded_solve
    U = np.ones((200, problem.N, problem.K+2))
    params = [problem.alpha, problem.sigma, problem.beta, problem.rate, problem.grounds,
              np.array([100., 0.]), 2*problem.N, problem.K, problem.ships.shape]
    sol = ns['solve_state_equations'](ns['state'], U, params, problem.T,
                                     np.r_[problem.ships.ravel(), np.ones(problem.K)], 200)
    if not sol.success:
        raise RuntimeError(sol.message)
    times = np.linspace(0, problem.T, 16001)
    return times, sample(sol, U, problem, times)


def outcome_figure(problem, solution, data, before=None):
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Working controller: fishing results and optimization', fontsize=17)
    t = data['times']
    if before is not None:
        old_t, old = before
        axs[0, 0].plot(old_t, old['catch'], color='#777777', lw=2, label='Original displayed run')
        axs[1, 0].plot(old_t, old['crowd'], color='#777777', lw=2, label='Original')
    axs[0, 0].plot(t, data['catch'], color='#0072B2', lw=2.5, label='Working controller')
    axs[0, 0].set(title='Total harvest', xlabel='Time', ylabel='Fish stock units', ylim=(0, 7.15))
    axs[0, 0].legend(loc='lower right')
    for j in range(problem.K):
        axs[0, 1].plot(t, data['fish'][:, j], label=f'G{j+1}')
    axs[0, 1].set(title='Every ground is harvested', xlabel='Time', ylabel='Remaining / initial stock', ylim=(-.03, 1.03))
    axs[0, 1].legend(ncol=4, fontsize=8)
    axs[1, 0].plot(t, data['crowd'], color='#0072B2', lw=2.5, label='Working controller')
    axs[1, 0].set(title='Accumulated crowding penalty', xlabel='Time', ylabel='Cost; log scale above 1', yscale='symlog')
    axs[1, 0].legend()
    history = solution['history']
    axs[1, 1].plot(range(len(history)), history, 'o-', color='#0072B2')
    axs[1, 1].set(title='Actual objective during search', xlabel='Accepted change', ylabel='J* (lower is better)', yscale='log')
    axs[1, 1].annotate(f'{history[-1]:.3f}', (len(history)-1, history[-1]), xytext=(-35, 12), textcoords='offset points')
    for ax in axs.flat:
        ax.grid(alpha=.2)
    for ax in [axs[0, 0], axs[0, 1], axs[1, 0]]:
        ax.set_xlim(0, problem.T)
    fig.text(.5, .04, 'Same objective weights and control bounds. Paper heading restored; direct target/timing optimization.', ha='center', fontsize=10)
    fig.text(.5, .015, 'J* omits constant self-interaction. Restricted search; no global optimum or PMP convergence claimed.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .07, 1, .95))
    return fig


def animation(problem, data, path, before=None, label='Working controller'):
    # Every frame uses a single physical time; stocks have a fixed initial scale.
    pairs = [(data['times'], data, label)] if before is None else [(*before, 'Original code: displayed run'), (data['times'], data, label)]
    fig, axs = plt.subplots(1, len(pairs), figsize=(12 if len(pairs) == 2 else 7, 6.6), squeeze=False)
    fig.subplots_adjust(left=.07, right=.88, top=.77, bottom=.23 if before is None else .19, wspace=.19)
    fig.suptitle('Original vs working fishing controller' if before else label, fontsize=17, y=.97)
    fig.text(.5, .91, f'Five boats • seven grounds • T = {problem.T:g} • unchanged objective weights and bounds', ha='center', fontsize=10)
    fig.text(.5, .865, 'Correction: stable unit heading. Search: choose grounds and travel/stop timings.', ha='center', fontsize=10)
    clock = fig.text(.5, .13, '', ha='center', fontsize=12, weight='bold')
    artists = []
    for ax, (t, run, title) in zip(axs[0], pairs):
        ax.set(xlim=(-5, 205), ylim=(-5, 205), aspect='equal', xlabel='x', ylabel='y', title=title)
        ax.grid(alpha=.15)
        grounds = ax.scatter(*problem.grounds.T, s=220, c=run['fish'][0], cmap='Blues', vmin=0, vmax=1,
                             edgecolor='#344050', linewidth=.9, zorder=2)
        for j, point in enumerate(problem.grounds):
            ax.annotate(f'G{j+1}', point, (7, 6), textcoords='offset points', fontsize=8)
        ax.scatter(*problem.ships.T, marker='x', c=COLORS, s=40, zorder=4)
        boats = ax.scatter(*problem.ships.T, marker='^', c=COLORS, s=55, edgecolor='white', linewidth=.5, zorder=6)
        lines = [ax.plot([], [], color=c, lw=1.8)[0] for c in COLORS]
        caption = ax.text(.02, .02, '', transform=ax.transAxes, fontsize=10,
                          bbox=dict(facecolor='white', alpha=.9, edgecolor='none'))
        artists.append((t, run, grounds, boats, lines, caption))
    cax = fig.add_axes([.91, .28, .012, .40])
    fig.colorbar(grounds, cax=cax, label='Fish remaining / initial stock')
    fig.legend(handles=[Line2D([0], [0], color=c, label=f'Boat {i+1}') for i, c in enumerate(COLORS)],
               ncol=5, loc='lower center', bbox_to_anchor=(.5, .065), frameon=False)
    fig.text(.5, .025, '× = start; ▲ = boat; pale = depleted. Boats stop to fish, then selected boats visit another ground.', ha='center', fontsize=9)
    def update(physical_time):
        for t, run, grounds, boats, lines, caption in artists:
            k = int(np.argmin(abs(t-physical_time)))
            grounds.set_array(run['fish'][k])
            boats.set_offsets(run['x'][k])
            for i, line in enumerate(lines):
                step = max(1, len(t)//2000)
                line.set_data(run['x'][:k+1:step, i, 0], run['x'][:k+1:step, i, 1])
            caption.set_text(f"Harvested {run['catch'][k]:.3f} / 7")
        clock.set_text(f'Time {physical_time:.1f} / {problem.T:g}')
    update(problem.T)
    fig.savefig(path.with_suffix('.png'), dpi=160)
    fig.set_dpi(95)
    frames = []
    for t in np.linspace(0, problem.T, 161):
        update(t)
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()).quantize(colors=128))
    durations = [80]*len(frames); durations[-1] = 2200
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=False, disposal=2)
    plt.close(fig)


def main():
    style()
    problem = scenario()
    data = dict(np.load(OUT/'notebook_400.npz'))
    solution = json.loads((OUT/'notebook_400.json').read_text())['solution']
    old = original_run(problem)
    np.testing.assert_allclose(old[1]['catch'][-1], 3.981605146181011, atol=1e-8)
    figure = outcome_figure(problem, solution, data, old)
    figure.savefig(OUT/'outcomes.png', dpi=150); plt.close(figure)
    animation(problem, data, OUT/'original_vs_working.gif', old)
    harbor = dict(np.load(OUT/'harbor_400.npz'))
    animation(scenario(harbor=True), harbor, OUT/'shared_harbor.gif', label='Working controller: shared harbor')
    print('Saved working-controller plots and animations:', OUT, flush=True)


if __name__ == '__main__':
    main()
