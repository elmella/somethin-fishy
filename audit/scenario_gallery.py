"""Render synchronized uncontrolled / legacy update / PMP beta=5 / PMP beta=0 scenario animations."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from PIL import Image
from scenario_experiments import CASES,problem_for,policy_name
from controller_visuals import original_run
from legacy_control_scenarios import replay as legacy_replay
from pontryagin_controller import OUT,load_saved
from pontryagin_visuals import sample_run
from compare_visuals import COLORS

HERE=Path(__file__).resolve().parent
DEST=OUT/'scenarios'
PALETTE=COLORS+['#6A51A3','#17BECF','#8C564B']
LABELS=['No optimal control','Original broken optimal control','Repaired PMP · β = 5','Repaired PMP · β = 0']


def original(case):
    path=DEST/(case+'_original.npz');p=problem_for(case)
    if path.exists():return dict(np.load(path))
    times,run=original_run(p)
    data=dict(times=times,**{key:run[key] for key in ['x','fish','catch','speed','travel']})
    np.savez_compressed(path,**data)
    print('Original replay',case,float(data['catch'][-1]),flush=True)
    return data


def render(case):
    p=problem_for(case);runs=[original(case),legacy_replay(case)];reports=[None,None]
    for beta in [5,0]:
        solver,u,report=load_saved(policy_name(case,beta)+'_verified',problem_for(case,beta))
        runs.append(sample_run(solver,u,count=1201));reports.append(report)
    colors=PALETTE[:p.N]
    fig,axes=plt.subplots(2,2,figsize=(11,12));artists=[]
    fig.suptitle(f'{CASES[case][0]} · T = 300 seconds',fontsize=16)
    for ax,data,label in zip(axes.flat,runs,LABELS):
        ax.set(xlim=(-5,205),ylim=(-5,205),aspect='equal',title=label,xlabel='x',ylabel='y');ax.grid(alpha=.15)
        grounds=ax.scatter(*p.grounds.T,s=160,c=np.ones(p.K),cmap='Blues',vmin=0,vmax=1,edgecolor='#344050')
        for j,g in enumerate(p.grounds):ax.annotate(f'G{j+1}',g,(5,5),textcoords='offset points',fontsize=8)
        ax.scatter(*p.ships.T,c=colors,marker='x',s=30)
        boats=ax.scatter(*p.ships.T,c=colors,marker='^',s=45,edgecolor='white',zorder=5)
        lines=[ax.plot([],[],c=c,lw=1.7)[0] for c in colors]
        caption=ax.text(.02,.02,'',transform=ax.transAxes,fontsize=10,bbox=dict(facecolor='white',alpha=.85,edgecolor='none'))
        artists.append((data,grounds,boats,lines,caption))
    clock=fig.text(.5,.09,'',ha='center',fontsize=12)
    fig.text(.5,.065,CASES[case][1]+' · Same clock and map in all four panels',ha='center',fontsize=10)
    status='Both PMP runs pass numerical residual and physical replay checks.' if all(r['converged'] for r in reports[2:]) else 'Some PMP residual checks remain unmet; see gallery diagnostics.'
    fig.text(.5,.042,status,ha='center',fontsize=9)
    fig.text(.5,.018,'Original broken solver: one historical sweep, then replay. Repaired: η = 0.5, ν = 0.001; no global optimum claimed.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.11,1,.96),h_pad=3.0)
    def draw(time):
        for data,grounds,boats,lines,caption in artists:
            k=int(np.argmin(abs(data['times']-time)));grounds.set_array(data['fish'][k]);boats.set_offsets(data['x'][k])
            stride=max(1,len(data['times'])//1200)
            for i,line in enumerate(lines):line.set_data(data['x'][:k+1:stride,i,0],data['x'][:k+1:stride,i,1])
            caption.set_text(f"Catch: {data['catch'][k]:.3f} / {p.K}")
        clock.set_text(f'Time {time:.1f} / 300 s')
    draw(300);fig.savefig(DEST/(case+'.png'),dpi=130)
    fig.set_dpi(90);frames=[]
    for time in np.linspace(0,300,151):
        draw(time);fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()).quantize(colors=128))
    frames[0].save(DEST/(case+'.gif'),save_all=True,append_images=frames[1:],duration=[100]*150+[2500],loop=0,disposal=2)
    plt.close(fig)
    times=np.linspace(0,300,601);packed=[]
    for label,data,report in zip(LABELS,runs,reports):
        def interp(array):
            array=np.asarray(array);shape=array.shape[1:];flat=array.reshape(len(array),-1)
            return np.stack([np.interp(times,data['times'],flat[:,i]) for i in range(flat.shape[1])],axis=-1).reshape((len(times),)+shape)
        positions=interp(data['x']);stocks=interp(data['fish'])
        assert np.all(np.isfinite(positions)) and np.min(stocks)>-1e-6
        np.testing.assert_allclose(positions[0],p.ships,atol=1e-8)
        travel=float(data['travel'][-1]/p.gamma)
        entry=dict(label=label,x=np.round(positions,6).tolist(),fish=np.round(stocks,8).tolist(),
                   harvested=float(data['catch'][-1]),distance=travel,
                   converged=report['converged'] if report else None,
                   projected_residual=report['projected_residual'] if report else None,
                   integrated_hamiltonian_gap=report['integrated_hamiltonian_gap'] if report else None,
                   control_intervals=report['steps'] if report else None)
        packed.append(entry)
    result=dict(id=case,title=CASES[case][0],description=CASES[case][1],N=p.N,K=p.K,T=p.T,
                grounds=p.grounds.tolist(),ships=p.ships.tolist(),colors=colors,times=times.tolist(),runs=packed,
                legacy_update=json.loads((DEST/(case+'_legacy_control.json')).read_text()),
                gif=f'results/pontryagin/scenarios/{case}.gif')
    (DEST/(case+'_display.json')).write_text(json.dumps(result,separators=(',',':')))
    print('Rendered',case,[(r['harvested'],r['distance'],r['converged']) for r in packed],flush=True)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--originals-only',action='store_true');parser.add_argument('--case',choices=CASES)
    args=parser.parse_args();DEST.mkdir(exist_ok=True,parents=True)
    if args.originals_only:
        for case in CASES:original(case)
        return
    if args.case:render(args.case);return
    cases=[render(case) for case in CASES]
    template=(HERE/'scenario_gallery_template.html').read_text()
    (HERE/'scenario_gallery.html').write_text(template.replace('__SCENARIO_DATA__',json.dumps(cases,separators=(',',':'))))
    summary=[dict(scenario=c['id'],description=c['description'],runs=[{k:v for k,v in r.items() if k not in ['x','fish']} for r in c['runs']]) for c in cases]
    (DEST/'summary.json').write_text(json.dumps(summary,indent=2))


if __name__=='__main__':main()
