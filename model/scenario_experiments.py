"""Reproducible 300-second comparisons; compare controls across maps and crowding weights."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import numpy as np
from fishing import scenario
from regularized_pmp import RegularizedSweep
from pontryagin_controller import fit,save,OUT,load_saved
from finalize_pmp import finalize

CASES={
    'baseline_300':('Dispersed fleet','5 boats · 7 grounds · dispersed starts'),
    'grounds_300':('More fishing grounds','5 boats · 10 grounds · dispersed starts'),
    'fleet_300':('Larger fleet and fishing area coverage','8 boats · 12 grounds · dispersed starts'),
    'harbor_300':('Harbor departure line','5 boats · 7 grounds · starts spaced along y = 10'),
}


def problem_for(case,beta=5.):
    p=scenario(T=300.)
    # Keep the base map points; additions use separate fixed random streams.
    extra_grounds=np.random.RandomState(123).uniform(15,185,(5,2))
    extra_ships=np.random.RandomState(456).uniform(15,185,(3,2))
    if case=='grounds_300':p=replace(p,grounds=np.vstack((p.grounds,extra_grounds[:3])))
    elif case=='fleet_300':p=replace(p,grounds=np.vstack((p.grounds,extra_grounds)),ships=np.vstack((p.ships,extra_ships)))
    elif case=='harbor_300':p=replace(p,ships=np.column_stack((np.linspace(80,120,5),np.full(5,10.))))
    elif case!='baseline_300':raise ValueError(case)
    return replace(p,beta=float(beta))


def policy_name(case,beta):
    prefix={'grounds_300':'grounds_matched_300','harbor_300':'harbor_deadline_300'}.get(case,case)
    return f'{prefix}_beta{beta:g}'


def run(case,beta):
    name=policy_name(case,beta);p=problem_for(case,beta)
    seed={'grounds_300':'coverage-matched','harbor_300':'coverage-deadline'}.get(case,'coverage')
    if (OUT/(name+'_verified.json')).exists():
        _,_,previous=load_saved(name+'_verified',p)
        if previous['converged']:
            print('Already verified',name,flush=True);return
    s=RegularizedSweep(p,150,eta=.5,travel_smoothing=.001)
    if (OUT/(name+'.json')).exists():
        s,u,report=load_saved(name,p)
        print('Resuming saved policy',name,flush=True)
    else:
        u,report=fit(s,s.seed(seed),cycles=6,iterations=1000)
        report.update(initialization=f'Deterministic {seed} assignment; same seed for both beta values',scenario=case)
        save(name,s,u,report)
    s,u,verified=finalize(OUT/(name+'.npz'),name+'_verified',T=p.T,
                          initialization=f'Deterministic {seed} assignment; adjoint optimization')
    # A denser diagnostic can reveal failure missed by midpoint checks. Retain
    # both reports; refine without changing equations or acceptance thresholds.
    for attempt in range(2):
        if verified['converged']:break
        from edge_refine_pmp import refine
        # After the coarse solve, allow finer switching whenever the continuous
        # Hamiltonian check fails; do not relax the acceptance threshold.
        steps=2*s.steps if verified['integrated_hamiltonian_gap']>=.005 else s.steps
        s,u,report=refine(OUT/(name+'.npz'),name,horizon=p.T,steps=steps,cycles=3,evaluations=8000)
        s,u,verified=finalize(OUT/(name+'.npz'),name+'_verified',T=p.T,
                              initialization='Coverage initialization followed by adjoint refinement')
    print('SCENARIO_DONE',case,beta,'converged',verified['converged'],flush=True)
    if not verified['converged']:raise RuntimeError('PMP thresholds unmet; saved candidate retained for refinement.')


def batch():
    OUT.mkdir(parents=True,exist_ok=True)
    logdir=OUT/'scenarios';logdir.mkdir(exist_ok=True)
    jobs=[(case,beta) for case in CASES for beta in (5.,0.)]
    def execute(job):
        case,beta=job;path=logdir/f'{case}_beta{beta:g}.log'
        print('START',case,beta,flush=True)
        with path.open('w') as log:
            result=subprocess.run([sys.executable,__file__,'--case',case,'--beta',str(beta)],stdout=log,stderr=subprocess.STDOUT)
        print('FINISHED',case,beta,'exit',result.returncode,flush=True)
        return dict(case=case,beta=beta,returncode=result.returncode,log=path.name)
    with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(execute,jobs))
    (logdir/'batch.json').write_text(json.dumps(results,indent=2))
    if any(row['returncode'] for row in results):raise RuntimeError('Some numerical runs failed; inspect saved logs.')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--case',choices=CASES);parser.add_argument('--beta',type=float,default=5.)
    args=parser.parse_args()
    if args.case:run(args.case,args.beta)
    else:batch()
