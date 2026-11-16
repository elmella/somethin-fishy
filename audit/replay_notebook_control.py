"""Tighter replay of saved final controls, splitting at every control switch."""
import json
import time
import hashlib
import sys
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from test_notebooks import load_functions

OUT=Path(__file__).parent/'results'


def main():
    ns=load_functions()
    records={}
    for name,T in [('updated_sam_fishin',400.),('igofishing',600.)]:
        if sys.argv[1:] and name not in sys.argv[1:]:
            continue
        path=OUT/(name+'_controls.npz')
        if not path.exists():
            continue
        data=np.load(path)
        controls,grounds,ships=data['controls'],data['grounds'],data['ships']
        N,K=len(ships),len(grounds)
        dt=T/(len(controls)-1)
        boundaries=np.r_[0.,(np.arange(len(controls)-1)+.5)*dt,T]
        # Merge consecutive equal controls. This is exact, not downsampling.
        changes=np.r_[0,np.flatnonzero(np.any(controls[1:]!=controls[:-1],axis=(1,2)))+1,len(controls)]
        checks=[]
        for tolerance in [1e-6,1e-8]:
            deadline=time.monotonic()+20.
            failed=False
            y=np.r_[ships.ravel(),np.ones(K),0.,0.]
            for first,last in zip(changes[:-1],changes[1:]):
                U=controls[first]
                params=[U,.005,.05,5.,.05,grounds,np.array([100.,0.]),2*N,K,ships.shape]
                def augmented(t,z):
                    if time.monotonic()>deadline:
                        raise RuntimeError('Replay reached its 20-second total limit.')
                    derivative=ns['state'](t,z[:-2],params)
                    x=z[:2*N].reshape(-1,2)
                    inv=1/(.0001+np.sum((x[:,None]-x[None])**2,axis=2))
                    np.fill_diagonal(inv,0.)
                    return np.r_[derivative,5*inv.sum(),.01*np.linalg.norm(derivative[:2*N].reshape(-1,2),axis=1).sum()]
                try:
                    sol=solve_ivp(augmented,[boundaries[first],boundaries[last]],y,
                                  method='DOP853',rtol=tolerance,atol=tolerance*1e-3,max_step=1.)
                except (RuntimeError,ValueError) as error:
                    checks.append(dict(rtol=tolerance,status='failed',reason=str(error)))
                    failed=True
                    break
                if not sol.success:
                    raise RuntimeError(sol.message)
                y=sol.y[:,-1]
            if failed:
                break
            checks.append(dict(rtol=tolerance,status='completed',harvested=float(K-y[2*N:2*N+K].sum()),
                               J_reduced=float(y[-2]+y[-1]+100*y[2*N:2*N+K].sum()),
                               crowding=float(y[-2]),travel=float(y[-1])))
        records[name]=dict(controls_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),checks=checks)
        print(name,checks,flush=True)
    (OUT/'notebook_replay.json').write_text(json.dumps(records,indent=2))


if __name__=='__main__':
    main()
