"""Reproduce legacy behavior without executing plotting or stale notebook cells."""
import ast
import json
import subprocess
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from fishing import scenario, adjoint, hamiltonian, geometry
from test_fishing import derivative_fixture, complex_gradient

ROOT = Path(__file__).resolve().parents[1]


def legacy_functions():
    # Pin the pre-repair source so reproductions remain valid after fixing it.
    notebook = json.loads(subprocess.check_output(
        ['git','show','23043fe5525d3b27dc0401f733fc8a9ec169ca26:random/updated_sam_fishin.ipynb'],cwd=ROOT))
    namespace = dict(np=np, solve_ivp=solve_ivp, NORMALIZATOR=1e-10)
    for cell in notebook['cells'][1:20]:
        if cell['cell_type'] != 'code':
            continue
        source = ''.join(cell['source'])
        tree = ast.parse(source)
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        exec(compile(ast.Module(body=functions, type_ignores=[]), '<legacy notebook functions>', 'exec'), namespace)
    return namespace


def metrics(sol, U_tensor, problem, namespace):
    # Diagnostic quadrature of the NOTEBOOK dynamics, not the paper's exact model.
    times = np.linspace(0, problem.T, 16001)
    ys = sol.sol(times)
    x = ys[:2*problem.N].T.reshape(-1,problem.N,2)
    pair = x[:,:,None]-x[:,None]
    inv = 1/(problem.epsilon+np.sum(pair**2,axis=-1))
    inv[:,np.arange(problem.N),np.arange(problem.N)] = 0
    crowd = np.trapezoid(problem.beta*inv.sum(axis=(1,2)),times)
    q = problem.grounds[None,None]-x[:,:,None]
    rho = np.linalg.norm(q,axis=-1)
    idx = np.round(times/(problem.T/(len(U_tensor)-1))).astype(int)
    s = np.sum(U_tensor[idx,:,:-2,None]*np.exp(-problem.alpha*rho**2)[...,None]
               *q/(rho[...,None]+1e-10)*ys[2*problem.N:].T[:,None,:,None],axis=2)
    sn = np.linalg.norm(s,axis=-1)
    speed = U_tensor[idx,:,-1]*sn/(sn+1e-10)
    distance = np.trapezoid(speed.sum(axis=1),times)
    remaining = ys[2*problem.N:,-1]
    return dict(harvested=float(problem.K*problem.F0-remaining.sum()),
                crowding=float(crowd),travel=float(problem.gamma*distance),
                terminal_fish=float(problem.delta*remaining.sum()),
                J_reduced=float(crowd+problem.gamma*distance+problem.delta*remaining.sum()),
                quadrature_points=len(times), solver_success=bool(sol.success))


def main():
    legacy = legacy_functions()
    problem, y, p, u, v = derivative_fixture()
    x, F = y[:2*problem.N].reshape(-1,2), y[2*problem.N:]
    U = np.column_stack((u,np.zeros(problem.N),v))
    params = [x,F,U,problem.alpha,problem.sigma,problem.beta,problem.epsilon,problem.rate,
              problem.grounds,np.zeros(2),2*problem.N,problem.K,x.shape]
    old = legacy['costate'](0,p,params)
    correct = adjoint(y,p,u,v,problem)
    numerical = -complex_gradient(lambda z: hamiltonian(z,p,u,v,problem),y)
    results = dict(adjoint=dict(original=old.tolist(),corrected=correct.tolist(),
                   numerical=numerical.tolist(),
                   old_relative_error=float(np.linalg.norm(old-numerical)/np.linalg.norm(numerical)),
                   corrected_max_abs_error=float(np.max(np.abs(correct-numerical)))))
    # Isolate errors in the printed formulas, not just the combined notebook error.
    q,rho,d,g,C,s = geometry(x,F,u,problem)
    sn = np.linalg.norm(s,axis=1); n=s/sn[:,None]
    Q=np.eye(2)[None]-n[:,:,None]*n[:,None,:]
    px=p[:2*problem.N].reshape(-1,2);pf=p[2*problem.N:]
    def Dxs(alpha_factor):
        return np.sum((u*g*F/rho)[...,None,None]*((alpha_factor*problem.alpha+1/rho**2)[...,None,None]
                      *q[..., :,None]*q[...,None,:]-np.eye(2)),axis=1)
    Dgood,Dold=Dxs(2),Dxs(1)
    true_move=-(v/sn)[:,None]*np.einsum('nji,njk,nk->ni',Dgood,Q,px)
    wrong_order=-(v/sn)[:,None]*np.einsum('nij,njk,nk->ni',Q,Dgood,px)
    true_fish=2*problem.rate*problem.sigma*np.sum((pf*F*C)[...,None]*q,axis=1)
    old_fish=-2*problem.rate*problem.sigma*np.sum((pf*C)[...,None]*q,axis=1)
    results['isolated_formula_errors'] = dict(Dxs_max_abs=float(np.max(np.abs(Dgood-Dold))),
         matrix_order_max_abs=float(np.max(np.abs(true_move-wrong_order))),
         fish_sign_and_F_max_abs=float(np.max(np.abs(true_fish-old_fish))),
         omitted_pf_term_max_abs=float(np.max(np.abs(correct[2*problem.N:]-problem.rate*pf*C.sum(axis=0)))))
    print('derivative audit',results,flush=True)
    problem = scenario()
    N,K = problem.N,problem.K
    y0=np.r_[problem.ships.ravel(),np.ones(K)*problem.F0]
    pT=np.r_[np.zeros(2*N),np.full(K,-problem.delta)]
    U_tensor=np.ones((200,N,K+2))
    state_params=[problem.alpha,problem.sigma,problem.beta,problem.rate,problem.grounds,
                  np.array([100.,0.]),2*N,K,problem.ships.shape]
    costate_params=[problem.alpha,problem.sigma,problem.beta,problem.epsilon,problem.rate,
                    problem.grounds,np.array([100.,0.]),2*N,K,problem.ships.shape]
    sol=legacy['solve_state_equations'](legacy['state'],U_tensor,state_params,problem.T,y0,200)
    results['legacy_initial']=metrics(sol,U_tensor,problem,legacy)
    print('legacy initial',results['legacy_initial'],flush=True)
    backward=legacy['solve_costate_equations'](legacy['costate'],sol,U_tensor,costate_params,
                                              problem.T,pT,2*N,K,200)
    if not backward.success or backward.y.shape[1] != 200:
        raise RuntimeError('Legacy backward integration failed: '+backward.message)
    pv=backward.y[:2*N,::-1].reshape(N,2,-1)
    x=sol.y[:2*N].reshape(N,2,-1);F=sol.y[2*N:]
    initial_controls=U_tensor.copy()
    failures=0
    for t in range(200):
        D=legacy['d_matrix'](x[:,:,t],problem.grounds)
        G=legacy['ship_ground_gaussians'](x[:,:,t],problem.grounds,problem.alpha)
        for ship in range(N):
            def old_surrogate(candidate):
                s=np.sum(candidate[:-2,None]*G[ship,:,None]*D[ship]*F[:,t,None],axis=0)
                cross=pv[ship,:,t]@np.array([s[1],-s[0]])
                reg=0 if t==0 else 100*np.sum((U_tensor[t-1,ship]-candidate)**2)
                return cross**2+reg
            fit=minimize(old_surrogate,U_tensor[t,ship],bounds=[(0,10)]*(K+2))
            failures+=int(not fit.success)
            U_tensor[t,ship]=fit.x
    replay=legacy['solve_state_equations'](legacy['state'],U_tensor,state_params,problem.T,y0,200)
    results['legacy_after_one_update']=metrics(replay,U_tensor,problem,legacy)
    results['legacy_update']=dict(max_control_change=float(np.max(np.abs(U_tensor-initial_controls))),
                                  optimizer_failures=failures,
                                  note='Exact active cell 23 update; gamma switch is computed there but never assigned.')
    print('legacy replay',results['legacy_after_one_update'],results['legacy_update'],flush=True)
    results['harbor_initial_speed']={}
    for name,pr in [('notebook',scenario()),('harbor',scenario(True))]:
        _,_,_,_,_,s=geometry(pr.ships,np.ones(K),np.ones((N,K)),pr)
        sn=np.linalg.norm(s,axis=1)
        results['harbor_initial_speed'][name]=(sn/(sn+1e-10)).tolist()
    out=ROOT/'audit/results/diagnostics.json'
    out.write_text(json.dumps(results,indent=2))


if __name__ == '__main__':
    main()
