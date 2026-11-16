"""Build the offline gallery, animated trajectories, and harvest plots."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.font_manager import findfont, FontProperties
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from comparison import RESULTS, MAPS, comparison

BOATS=['#2779B5','#DE792D','#358855','#BB4050','#7758A6','#167D84','#9C6B4C','#BF5594']
POLICIES=['#495E71','#147C77','#BA631F']
INK='#183443'
def font(size,bold=False):
    return ImageFont.truetype(findfont(FontProperties(family='DejaVu Sans',weight='bold' if bold else 'normal')),size)


def render(case):
    p,runs=comparison(case)
    times=runs[0][1]['times']; title,description=MAPS[case]
    labels=[label for label,_,_ in runs]
    width=350*len(runs);height=500
    base=Image.new('RGB',(width,height),'#F3F7F8')
    d=ImageDraw.Draw(base);heading,sub,small=font(21,True),font(16,True),font(13)
    d.text((18,10),title,fill=INK,font=heading)
    d.text((18,40),description,fill=INK,font=small)
    def points(coords,i):
        return np.round(np.asarray(coords)*[1.42,-1.42]+[350*i+32,402]).astype(int)
    for i,label in enumerate(labels):
        left=350*i
        d.rounded_rectangle((left+6,70,left+344,477),8,fill='white',outline='#D0DFE2')
        d.text((left+17,80),label,font=sub,fill=POLICIES[i])
        for tick in range(0,201,50):
            x,y=points([tick,tick],i)
            d.line((x,118,x,402),fill='#E5EBEF');d.line((left+32,y,left+316,y),fill='#E5EBEF')
        for x,y in points(p.ships,i):
            d.line((x-3,y-3,x+3,y+3),fill=INK);d.line((x-3,y+3,x+3,y-3),fill=INK)
    paths=[points(run['x'],i) for i,(_,run,_) in enumerate(runs)]
    grounds=[points(p.grounds,i) for i in range(len(runs))]
    d.text((18,481),'× start    ▲ boat    Circle shade: dark = full, light = depleted',font=small,fill=INK)
    frames=[]
    for k in np.linspace(0,len(times)-1,121,dtype=int):
        img=base.copy();d=ImageDraw.Draw(img)
        d.text((width-15,17),f'{times[k]:.0f} / {p.T:g} s',fill=INK,font=sub,anchor='ra')
        for i,(_,r,_) in enumerate(runs):
            for j,((x,y),fraction) in enumerate(zip(grounds[i],r['fish'][k]/p.F0)):
                color=tuple(round(a+(b-a)*fraction) for a,b in zip((245,248,250),(31,83,127)))
                d.ellipse((x-6,y-6,x+6,y+6),fill=color,outline=INK)
            for boat in range(p.N):
                line=[tuple(point) for point in paths[i][:k+1,boat]]
                if len(line)>1:d.line(line,fill=BOATS[boat],width=2)
                x,y=line[-1];d.polygon([(x,y-5),(x-5,y+4),(x+5,y+4)],fill=BOATS[boat],outline='white')
            left=i*350+17
            d.text((left,420),f"Harvest: {r['catch'][k]:.3f} / {p.K*p.F0:g}",font=sub,fill=POLICIES[i])
            d.text((left,448),f"Fleet distance: {r['distance'][k]:.1f}",font=small,fill=INK)
        frames.append(img.quantize(colors=128))
    frames[-1].convert('RGB').save(RESULTS/(case+'_paths.png'))
    frames[0].save(RESULTS/(case+'.gif'),save_all=True,append_images=frames[1:],duration=[90]*120+[2000],loop=0,disposal=2)
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for i,(label,r,_) in enumerate(runs):
        axes[0].plot(times,r['catch'],color=POLICIES[i],label=label,ls=['-','-','--'][i])
        axes[1].plot(times,r['distance'],color=POLICIES[i],ls=['-','-','--'][i])
    axes[0].set(ylabel='Fish units',title='Cumulative catch',ylim=(0,p.K*1.04))
    axes[1].set(ylabel='Map units',title='Fleet distance')
    axes[2].plot(times,runs[1][1]['fish']);axes[2].set(ylabel='Fish units',title='Ground stocks · optimal control (β = 5)')
    for ax in axes:ax.set(xlabel='Time (s)',xlim=(0,p.T));ax.grid(alpha=.2)
    axes[0].legend(fontsize=8);axes[2].legend([f'G{i+1}' for i in range(p.K)],ncol=3,fontsize=7)
    fig.suptitle(title+'  |  '+description);fig.savefig(RESULTS/(case+'_outcomes.png'),dpi=120);plt.close(fig)
    entries=[]
    for label,r,report in runs:
        metrics={k:report[k] if report else None for k in ('projected_residual','integrated_hamiltonian_gap','converged','steps')}
        entries.append(dict(label=label,x=r['x'].round(6).tolist(),fish=r['fish'].round(10).tolist(),harvested=r['catch'][-1],
                            distance=r['distance'][-1],gain=r['catch'][-1]-runs[0][1]['catch'][-1],**metrics))
    print(case,flush=True)
    return dict(id=case,title=title,description=description,N=p.N,K=p.K,T=p.T,grounds=p.grounds.tolist(),ships=p.ships.tolist(),
                colors=BOATS[:p.N],times=times.tolist(),runs=entries,gif=case+'.gif')


def movement_page():
    from comparison import setting, rollout
    from pontryagin_controller import load_saved
    p,name=setting('dispersed_400',5)
    solver,u,_=load_saved(name,p)
    run=rollout(solver,u)
    _,_,targets,throttle=solver.unpack(u)
    k=np.minimum((run['times']/solver.dt).astype(int),solver.steps-1)
    data=dict(x=run['x'][:,0].tolist(),z=targets[k,0].tolist(),a=throttle[k,0].tolist(),
              fish=run['fish'].tolist(),grounds=p.grounds.tolist(),eta=solver.eta,V=p.V)
    dest=RESULTS.parent/'audit'/'movement_explained.html'
    dest.parent.mkdir(exist_ok=True)
    template=Path(__file__).with_name('movement_template.html').read_text()
    dest.write_text(template.replace('__DATA__',json.dumps(data,separators=(',',':'),allow_nan=False)))


def main():
    RESULTS.mkdir(exist_ok=True)
    movement_page()
    cases=[render(case) for case in MAPS]
    template=Path(__file__).with_name('gallery_template.html').read_text()
    (RESULTS/'gallery.html').write_text(template.replace('__SCENARIO_DATA__',json.dumps(cases,separators=(',',':'),allow_nan=False)))
    summary=[dict(scenario=c['id'],seconds=c['T'],boats=c['N'],grounds=c['K'],runs=[
        {k:v for k,v in row.items() if k not in ('x','fish')} for row in c['runs']]) for c in cases]
    (RESULTS/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
