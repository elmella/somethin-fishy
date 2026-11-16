"""Two-panel animation of the original run and the notebook's selected policy."""
import json
import re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from PIL import Image
from fishing import scenario
from pontryagin_controller import load_saved
from pontryagin_visuals import sample_run
from controller_visuals import original_run
from compare_visuals import COLORS

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'audit/results/pontryagin'
book=json.loads((ROOT/'random/updated_sam_fishin.ipynb').read_text())
source=''.join(book['cells'][23]['source'])
name=re.search(r'^RESULT_NAME = "([^"]+)"',source,re.M).group(1)
p=scenario();solver,control,report=load_saved(name,p)
final=sample_run(solver,control);times,original=original_run(p)
runs=[(times,original,'Original'),(final['times'],final,'Final: current notebook policy')]
fig,axes=plt.subplots(1,2,figsize=(11,6.1));artists=[]
fig.suptitle('Original and final fishing control',fontsize=17)
for ax,(t,data,title) in zip(axes,runs):
    ax.set(xlim=(-5,205),ylim=(-5,205),aspect='equal',title=title,xlabel='x',ylabel='y')
    ax.grid(alpha=.15)
    grounds=ax.scatter(*p.grounds.T,s=190,c=np.ones(p.K),cmap='Blues',vmin=0,vmax=1,edgecolor='#344050')
    for j,g in enumerate(p.grounds):ax.annotate(f'G{j+1}',g,(6,5),textcoords='offset points',fontsize=8)
    ax.scatter(*p.ships.T,c=COLORS,marker='x',s=30)
    boats=ax.scatter(*p.ships.T,c=COLORS,marker='^',s=45,edgecolor='white',zorder=5)
    lines=[ax.plot([],[],c=c,lw=1.8)[0] for c in COLORS]
    caption=ax.text(.02,.02,'',transform=ax.transAxes,fontsize=10,bbox=dict(facecolor='white',alpha=.85,edgecolor='none'))
    artists.append((t,data,grounds,boats,lines,caption))
clock=fig.text(.5,.105,'',ha='center',fontsize=12)
fig.text(.5,.055,'Dark grounds: fish remaining. Colored trails: boat paths. Both panels use the same clock.',ha='center',fontsize=9)
fig.text(.5,.018,'Final: regularized PMP feedback, coverage initialization; original map and control bounds.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.19,1,.94))
def draw(time):
    for t,data,grounds,boats,lines,caption in artists:
        k=int(np.argmin(abs(t-time)));grounds.set_array(data['fish'][k]);boats.set_offsets(data['x'][k])
        stride=max(1,len(t)//2000)
        for i,line in enumerate(lines):line.set_data(data['x'][:k+1:stride,i,0],data['x'][:k+1:stride,i,1])
        caption.set_text(f"Harvest: {data['catch'][k]:.4f} / 7")
    clock.set_text(f'Time {time:.1f} / {p.T:g}')
draw(p.T);fig.savefig(OUT/'original_final.png',dpi=150)
fig.set_dpi(90);frames=[]
for t in np.linspace(0,p.T,161):
    draw(t);fig.canvas.draw();frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()).quantize(colors=128))
frames[0].save(OUT/'original_final.gif',save_all=True,append_images=frames[1:],duration=[80]*160+[2500],loop=0,disposal=2)
plt.close(fig)
(OUT/'original_final_manifest.json').write_text(json.dumps(dict(notebook='random/updated_sam_fishin.ipynb',policy=name,original_harvest=float(original['catch'][-1]),final_harvest=float(final['catch'][-1]),original_cost_of_final=report['original_J'],frames=len(frames),horizon=p.T),indent=2))
print(name,original['catch'][-1],final['catch'][-1])
