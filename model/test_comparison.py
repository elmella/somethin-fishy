import unittest
import numpy as np
from scipy.integrate import solve_ivp
from comparison import equal_attraction_target, rollout
from fishing import Problem
from regularized_pmp import RegularizedSweep


class ComparisonTests(unittest.TestCase):
    def test_equal_attractions_halfway_and_at_ground(self):
        p=Problem(np.array([[0.,0.],[2.,0.]]),np.array([[1.,2.],[0.,0.]]))
        targets=equal_attraction_target(p.ships,np.zeros(2),p)
        np.testing.assert_allclose(targets,[[1,0],[0,0]])
        self.assertGreater(equal_attraction_target(p.ships,np.log([.2,1.]),p)[0,0],1)

    def test_unplanned_exposure_and_distance_against_separate_ode(self):
        p=Problem(np.array([[0.,0.],[2.,0.],[1.,3.]]),np.array([[1.,-2.]]),T=4.)
        s=RegularizedSweep(p,2,eta=.5)
        curve=rollout(s)
        state=np.r_[p.ships.ravel(),np.ones(3),0.]
        for k in range(4):
            x=state[:2];q=p.grounds-x;rho=np.linalg.norm(q,axis=1)
            target=np.average(p.grounds,axis=0,weights=np.exp(-p.alpha*rho**2)*state[2:5]/rho)
            def rhs(t,y):
                d=target-y[:2];vel=p.V*d/np.sqrt(d@d+s.eta**2)
                c=np.exp(-p.sigma*np.sum((p.grounds-y[:2])**2,axis=1))
                return np.r_[vel,-p.rate*y[2:5]*c,np.linalg.norm(vel)]
            sol=solve_ivp(rhs,[k,k+1],state,rtol=1e-11,atol=1e-12)
            self.assertTrue(sol.success)
            state=sol.y[:,-1]
            np.testing.assert_allclose(curve['x'][2*k+2,0],state[:2],atol=1e-9)
            np.testing.assert_allclose(curve['fish'][2*k+2],state[2:5],atol=1e-9)
            self.assertAlmostEqual(curve['distance'][2*k+2],state[-1],places=9)
