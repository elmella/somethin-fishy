"""Comparable scientific plots; each animation frame shares one physical time."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from PIL import Image
from regularized_pmp import RegularizedSweep,radial_flow
from fishing import scenario
from controller_visuals import original_run
from compare_visuals import COLORS

OUT=Path(__file__).parent/'results/pontryagin'


def sample_run(solver,control,count=1601):
    p=solver.p;s=solver.forward(control);dt=solver.dt
    times=np.linspace(0,p.T,count);x=np.empty((count,p.N,2));fish=np.empty((count,p.K));speed=np.empty((count,p.N))
    crowd=np.empty(count);travel=np.empty(count)
    exposure=dt*np.sum(s['C']*solver.weights[None,None,:,None],axis=(1,2))
    prefixE=np.vstack((np.zeros(p.K),np.cumsum(exposure,axis=0)))
    prefixC=np.r_[0,np.cumsum(2*p.beta*dt*np.sum(solver.weights[None,None]/s['denominators'],axis=(1,2)))]
    prefixL=np.r_[0,np.cumsum(p.gamma*np.linalg.norm(s['dx'],axis=-1).sum(axis=-1))]
    for k in range(solver.steps):
        mask=(times>=k*dt)&((times<(k+1)*dt)|((k==solver.steps-1)&(times<=p.T)))
        elapsed=times[mask]-k*dt
        if not len(elapsed):continue
        positions,_,_,velocities=radial_flow(s['x'][k,:,None],s['target'][k,:,None],s['a'][k,:,None],elapsed[None],p.V,solver.eta)
        x[mask]=positions.transpose(1,0,2);speed[mask]=np.linalg.norm(velocities,axis=-1).T
        node_x=radial_flow(s['x'][k,:,None,None],s['target'][k,:,None,None],s['a'][k,:,None,None],elapsed[None,:,None]*solver.nodes[None,None],p.V,solver.eta)[0]
        C=np.exp(-p.sigma*np.sum((p.grounds[None,None,None]-node_x[:,:,:,None])**2,axis=-1))
        E=np.sum(C*solver.weights[None,None,:,None],axis=(0,2))*elapsed[:,None]
        fish[mask]=p.F0*np.exp(-p.rate*(prefixE[k]+E))
        pair=node_x[solver.ii]-node_x[solver.jj]
        partial=2*p.beta*elapsed*np.sum(solver.weights[None,None]/(p.epsilon+np.sum(pair*pair,axis=-1)),axis=(0,2))
        crowd[mask]=prefixC[k]+partial
        travel[mask]=prefixL[k]+p.gamma*np.linalg.norm(positions-s['x'][k,:,None],axis=-1).sum(axis=0)
    from pmp_validation import reconstruct_attractions
    controls=np.zeros((count,p.N,p.K+2));effective=np.zeros((count,p.N,p.K))
    for row,t in enumerate(times):
        k=min(int(t/dt),solver.steps-1);q=s['target'][k]-x[row]
        velocity=s['a'][k,:,None]*p.V*q/np.sqrt(np.sum(q*q,axis=-1)+solver.eta**2)[:,None]
        u,v=reconstruct_attractions(x[row],fish[row],velocity,p)
        controls[row,:,:p.K]=u;controls[row,:,-1]=v
        distance2=np.sum((p.grounds[None]-x[row,:,None])**2,axis=-1)
        logs=np.log(np.maximum(u,1e-300))+np.log(fish[row])[None]-p.alpha*distance2
        logs[u==0]=-np.inf;weights=np.exp(logs-logs.max(axis=-1,keepdims=True))
        effective[row]=weights/weights.sum(axis=-1,keepdims=True)
    return dict(times=times,x=x,fish=fish,catch=p.K*p.F0-fish.sum(axis=1),crowd=crowd,travel=travel,speed=speed,controls=controls,effective_attractions=effective)


def figures(problem,pmp,benchmark,original,reports,label):
    OUT.mkdir(parents=True,exist_ok=True)
    runs=[(original[0],original[1],'Original code'),(benchmark['times'],benchmark,'Route benchmark'),(pmp['times'],pmp,label)]
    fig,axes=plt.subplots(1,3,figsize=(15,6.3));artists=[]
    fig.suptitle('Fishing control: original code, benchmark, and forward–backward solver',fontsize=15)
    for ax,(t,data,title) in zip(axes,runs):
        ax.set(xlim=(-5,205),ylim=(-5,205),aspect='equal',title=title,xlabel='x',ylabel='y');ax.grid(alpha=.15)
        grounds=ax.scatter(*problem.grounds.T,s=190,c=np.ones(problem.K),cmap='Blues',vmin=0,vmax=1,edgecolor='#344050')
        for j,g in enumerate(problem.grounds):ax.annotate(f'G{j+1}',g,(6,5),textcoords='offset points',fontsize=8)
        ax.scatter(*problem.ships.T,c=COLORS,marker='x',s=30)
        boats=ax.scatter(*problem.ships.T,c=COLORS,marker='^',s=45,edgecolor='white',zorder=5)
        lines=[ax.plot([],[],c=c,lw=1.8)[0] for c in COLORS]
        caption=ax.text(.02,.02,'',transform=ax.transAxes,fontsize=9,bbox=dict(facecolor='white',alpha=.85,edgecolor='none'))
        artists.append((t,data,grounds,boats,lines,caption))
    clock=fig.text(.5,.085,'',ha='center',fontsize=11)
    fig.text(.5,.025,'Same map, five boats, seven grounds. PMP regularizations are explicitly documented; benchmark retained unchanged.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.19,1,.94))
    def draw(time):
        for t,data,grounds,boats,lines,caption in artists:
            k=int(np.argmin(abs(t-time)));grounds.set_array(data['fish'][k]);boats.set_offsets(data['x'][k])
            stride=max(1,len(t)//2000)
            for i,line in enumerate(lines):line.set_data(data['x'][:k+1:stride,i,0],data['x'][:k+1:stride,i,1])
            caption.set_text(f"Harvest: {data['catch'][k]:.4f} / 7")
        clock.set_text(f'Time {time:.1f} / {problem.T:g}')
    draw(problem.T);fig.savefig(OUT/'comparison.png',dpi=150)
    fig.set_dpi(80);frames=[]
    for t in np.linspace(0,problem.T,161):
        draw(t);fig.canvas.draw();frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()).quantize(colors=128))
    duration=[80]*len(frames);duration[-1]=2500
    frames[0].save(OUT/'comparison.gif',save_all=True,append_images=frames[1:],duration=duration,loop=0,disposal=2)
    plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(12,8))
    for t,data,name in runs:axs[0,0].plot(t,data['catch'],label=name,lw=2)
    axs[0,0].set(title='Harvest',xlabel='Time',ylabel='Fish units',ylim=(0,7.15));axs[0,0].legend(fontsize=8)
    for j in range(problem.K):axs[0,1].plot(pmp['times'],pmp['fish'][:,j],label=f'G{j+1}')
    axs[0,1].set(title='PMP: remaining stock by ground',xlabel='Time',ylabel='Fraction of initial stock');axs[0,1].legend(ncol=4,fontsize=8)
    benchmark_report=json.loads((Path(__file__).parent/'results/working/notebook_400.json').read_text())['solution']['metrics']
    coverage=json.loads((OUT/'coverage_400_verified.json').read_text())
    best=json.loads((OUT/'best_400.json').read_text())
    labels=['Route\nbenchmark','Coverage\nPMP','Benchmark-seeded\nPMP']
    bottom=np.zeros(3)
    for key,name,color in [('crowding','Crowding','#E69F00'),('travel','Travel','#0072B2'),('terminal_fish','Remaining fish','#D55E00')]:
        values=np.array([benchmark_report[key],coverage[key],best[key]])
        axs[1,0].bar(labels,values,bottom=bottom,label=name,color=color,width=.6);bottom+=values
    for i,value in enumerate(bottom):axs[1,0].text(i,value+.15,f'{value:.3f}',ha='center',fontsize=9)
    axs[1,0].set(title='Original objective: same weights, actual travel cost',ylabel='J*; lower is better',ylim=(0,15));axs[1,0].legend(fontsize=8)
    for i in range(problem.N):axs[1,1].plot(pmp['times'],pmp['speed'][:,i],c=COLORS[i],label=f'Boat {i+1}')
    axs[1,1].set(title='Actual boat speed',xlabel='Time',ylabel='Speed',ylim=(-.03,1.05));axs[1,1].legend(ncol=3,fontsize=8)
    for ax in axs.flat:ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'outcomes.png',dpi=150);plt.close(fig)
    run=json.loads((OUT/'coverage_400.json').read_text());history=run['history']
    fig,axes=plt.subplots(1,2,figsize=(11,4.2))
    axes[0].plot(range(len(history)+1),[run['initial_J']]+[row['J'] for row in history],'o-')
    axes[0].set(title='Coverage initialization: forward–backward improvement',xlabel='Completed cycle',ylabel='Regularized J*')
    axes[1].semilogy(range(1,len(history)+1),[row['integrated_hamiltonian_gap'] for row in history],'o-',label='Continuous midpoint check')
    axes[1].axhline(.005,ls='--',c='#777777',label='Acceptance tolerance')
    axes[1].set(title='Hamiltonian residual across cycles',xlabel='Completed cycle',ylabel='Integrated gap (200 time samples)');axes[1].legend(fontsize=8)
    for ax in axes:ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'convergence.png',dpi=150);plt.close(fig)


def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('file');parser.add_argument('--eta',type=float,default=.5)
    args=parser.parse_args();data=np.load(args.file);steps=len(data['control'])//15;p=scenario();s=RegularizedSweep(p,steps,eta=args.eta)
    result=sample_run(s,data['control']);benchmark=dict(np.load(Path(__file__).parent/'results/working/notebook_400.npz'))
    report_file=Path(args.file).with_suffix('.json');reports=[]
    if report_file.exists():reports=[('PMP refinement',json.loads(report_file.read_text()))]
    initialization=reports[0][1].get('initialization','') if reports else ''
    label='PMP: benchmark warm start' if 'benchmark' in initialization.lower() else 'PMP: coverage initialization'
    figures(p,result,benchmark,original_run(p),reports,label)
    np.savez_compressed(OUT/'display_run.npz',**result)

if __name__=='__main__':main()
