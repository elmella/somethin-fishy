"""Regression checks against the actual repaired notebook functions."""
import ast
import json
from pathlib import Path
import unittest
import numpy as np
from scipy.integrate import solve_ivp
from test_fishing import derivative_fixture

ROOT=Path(__file__).resolve().parents[1]


def load_functions(filename='updated_sam_fishin.ipynb'):
    notebook=json.loads((ROOT/'random'/filename).read_text())
    namespace=dict(np=np,solve_ivp=solve_ivp,NORMALIZATOR=1e-10)
    for cell in notebook['cells'][1:20]:
        if cell['cell_type']!='code':
            continue
        tree=ast.parse(''.join(cell['source']))
        functions=[n for n in tree.body if isinstance(n,ast.FunctionDef)]
        exec(compile(ast.Module(body=functions,type_ignores=[]),filename,'exec'),namespace)
    return namespace


class NotebookTests(unittest.TestCase):
    def test_control_switches_are_integrated_at_exact_boundaries(self):
        ns=load_functions()
        U=np.zeros((3,1,3));U[:,0,-1]=[1.,0.,1.]
        def state(t,y,params):
            return np.array([params[0][0,-1],0.,0.])
        params=[.005,.05,0.,.05,np.array([[4.,4.]]),np.zeros(2),2,1,(1,2),.1,0.]
        sol=ns['solve_state_equations'](state,U,params,2.,np.array([0.,0.,1.]),3)
        np.testing.assert_allclose(sol.y[0],[0.,.5,1.],atol=1e-10)

    def test_all_active_cells_parse(self):
        for filename in ['updated_sam_fishin.ipynb','igofishing.ipynb']:
            book=json.loads((ROOT/'random'/filename).read_text())
            for cell in book['cells']:
                if cell['cell_type']=='code':
                    source='\n'.join(line for line in ''.join(cell['source']).splitlines() if not line.startswith('%'))
                    ast.parse(source)

    def test_regularized_adjoint_matches_implemented_H(self):
        problem,y,p,u,v=derivative_fixture()
        for filename in ['updated_sam_fishin.ipynb','igofishing.ipynb']:
            ns=load_functions(filename)
            for strength in [1.,1e-9]:  # Includes the denominator-floor regime.
                U=np.column_stack((strength*u,np.zeros(problem.N),v))
                state_params=[U,problem.alpha,problem.sigma,problem.beta,problem.rate,
                              problem.grounds,np.zeros(2),2*problem.N,problem.K,problem.ships.shape]
                def H(z):
                    f=ns['state'](0,z,state_params)
                    x=z[:2*problem.N].reshape(-1,2)
                    diff=x[:,None]-x[None]
                    L=problem.beta*np.sum(1/(problem.epsilon+np.sum(diff**2,axis=2)))
                    L+=problem.gamma*np.linalg.norm(f[:2*problem.N].reshape(-1,2),axis=1).sum()
                    return p@f-L
                numerical=[]
                for k in range(len(y)):
                    perturb=np.zeros_like(y);perturb[k]=1e-5
                    numerical.append(-(H(y+perturb)-H(y-perturb))/(2e-5))
                params=[y[:2*problem.N],y[2*problem.N:],U,problem.alpha,problem.sigma,
                        problem.beta,problem.epsilon,problem.rate,problem.gamma,problem.grounds,
                        np.zeros(2),2*problem.N,problem.K,problem.ships.shape]
                actual=ns['costate'](0,p,params)
                np.testing.assert_allclose(actual,numerical,atol=2e-8,rtol=2e-7)

    def test_backward_uses_same_physical_time(self):
        ns=load_functions()
        class Forward:
            def sol(self,t):
                return np.array([t,2*t,1.,0.,0.])
        observed=[]
        def inspect(t,p,params):
            observed.append((t,params[0][0]))
            return np.zeros_like(p)
        ns['solve_costate_equations'](inspect,Forward(),np.ones((5,1,3)),[0.]*6,
                                      2.,np.zeros(3),2,1,5)
        self.assertTrue(all(abs(t-x)<1e-12 for t,x in observed))


if __name__=='__main__':
    unittest.main()
