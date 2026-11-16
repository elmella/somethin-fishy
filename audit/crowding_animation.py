"""Compare the current notebook policy against a reoptimized beta=0 policy."""
import json
import re
from dataclasses import replace
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from PIL import Image
from fishing import scenario
from pontryagin_controller import load_saved
from pontryagin_visuals import sample_run
from compare_visuals import COLORS

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'audit/results/pontryagin'


def main():
    book=json.loads((ROOT/'random/updated_sam_fishin.ipynb').read_text())
    source='\n'.join(''.join(cell['source']) for cell in book['cells'])
    current=re.search(r'^RESULT_NAME = "([^"]+)"',source,re.M).group(1)
    names=[current,'coverage_400_beta0_verified']
    runs=[];metrics=[];solvers=[]
    for name,beta in zip(names,[5.,0.]):
        solver,control,report=load_saved(name,replace(scenario(),beta=beta))
        if not report['converged']:raise ValueError(f'{name}: residual checks have not passed')
        solvers.append(solver);data=sample_run(solver,control)
        state=solver.forward(control)
        assert abs(data['catch'][-1]-report['harvested'])<1e-9
        runs.append(data)
        metrics.append(dict(policy=name,beta=beta,harvest=report['harvested'],
                            total_distance=float(np.linalg.norm(state['dx'],axis=-1).sum()),
                            crowding=state['crowd'],remaining_fish_cost=float(solver.p.delta*state['fish'].sum()),
                            projected_residual=report['projected_residual'],
                            integrated_hamiltonian_gap=report['integrated_hamiltonian_gap']))
    for key,value in vars(solvers[0].p).items():
        if key!='beta':np.testing.assert_array_equal(value,getattr(solvers[1].p,key))
    for key in ['steps','eta','travel_smoothing']:
        assert getattr(solvers[0],key)==getattr(solvers[1],key)
    p=solvers[0].p
    fig,axes=plt.subplots(1,2,figsize=(11,6.1));artists=[]
    fig.suptitle('Crowding penalty: current policy versus reoptimized without crowding',fontsize=14)
    for ax,data,title in zip(axes,runs,['Current: crowding weight β = 5','Reoptimized: crowding weight β = 0']):
        ax.set(xlim=(-5,205),ylim=(-5,205),aspect='equal',title=title,xlabel='x',ylabel='y')
        ax.grid(alpha=.15)
        grounds=ax.scatter(*p.grounds.T,s=190,c=np.ones(p.K),cmap='Blues',vmin=0,vmax=1,edgecolor='#344050')
        for j,g in enumerate(p.grounds):ax.annotate(f'G{j+1}',g,(6,5),textcoords='offset points',fontsize=8)
        ax.scatter(*p.ships.T,c=COLORS,marker='x',s=30)
        boats=ax.scatter(*p.ships.T,c=COLORS,marker='^',s=45,edgecolor='white',zorder=5)
        lines=[ax.plot([],[],c=c,lw=1.8)[0] for c in COLORS]
        caption=ax.text(.02,.02,'',transform=ax.transAxes,fontsize=10,bbox=dict(facecolor='white',alpha=.85,edgecolor='none'))
        artists.append((data,grounds,boats,lines,caption))
    clock=fig.text(.5,.105,'',ha='center',fontsize=12)
    fig.text(.5,.055,'Same map, initial coverage seed, horizon, bounds and regularizations; only β changes.',ha='center',fontsize=9)
    fig.text(.5,.018,'Dark grounds: fish remaining. Trails: boat paths. Regularized PMP solutions; no global-optimality claim.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.19,1,.94))
    def draw(time):
        for data,grounds,boats,lines,caption in artists:
            k=int(np.argmin(abs(data['times']-time)));grounds.set_array(data['fish'][k]);boats.set_offsets(data['x'][k])
            for i,line in enumerate(lines):line.set_data(data['x'][:k+1,i,0],data['x'][:k+1,i,1])
            caption.set_text(f"Harvest: {data['catch'][k]:.4f} / 7")
        clock.set_text(f'Time {time:.1f} / {p.T:g}')
    draw(p.T);fig.savefig(OUT/'crowding_comparison.png',dpi=150)
    fig.set_dpi(90);frames=[]
    for time in np.linspace(0,p.T,161):
        draw(time);fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()).quantize(colors=128))
    frames[0].save(OUT/'crowding_comparison.gif',save_all=True,append_images=frames[1:],duration=[80]*160+[2500],loop=0,disposal=2)
    plt.close(fig)
    (OUT/'crowding_comparison.json').write_text(json.dumps(dict(runs=metrics,frames=161,horizon=p.T,only_problem_parameter_changed='beta'),indent=2))
    print(json.dumps(metrics,indent=2))


if __name__=='__main__':main()
