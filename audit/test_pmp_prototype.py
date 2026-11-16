"""Numerical contracts for the forward/backward repair and raw-control replay."""
import unittest
import numpy as np
from scipy.integrate import quad
from fishing import Problem, scenario
from pmp_prototype import TargetSweep, pair_integral
from pmp_validation import OriginalSchedule


class ForwardBackwardTests(unittest.TestCase):
    def setUp(self):
        p=Problem(np.array([[0.,0.],[10.,0.],[5.,10.]]),np.array([[2.,3.],[7.,3.]]),T=4.)
        self.solver=TargetSweep(p,20,.001)
        rng=np.random.default_rng(4)
        theta=rng.dirichlet(np.ones(p.K)*3,size=(20,p.N))
        target=np.einsum('tnj,ji->tni',theta,p.grounds)
        a=rng.uniform(.2,.7,(20,p.N))
        self.control=np.concatenate((target/self.solver.target_scale,a[...,None]),axis=-1).ravel()

    def test_backward_gradient_matches_target_and_speed_perturbations(self):
        s=self.solver;u=self.control;J,g=s.value_gradient(u)
        rng=np.random.default_rng(13)
        for _ in range(4):
            d=rng.normal(size=u.shape);d/=np.linalg.norm(d);h=1e-6
            fd=(s.forward(u+h*d)['J']-s.forward(u-h*d)['J'])/(2*h)
            self.assertAlmostEqual(fd,float(np.sum(g*d)),delta=2e-5)
        np.testing.assert_array_equal(s.last['costate'][-1],0)

    def test_original_linear_speed_cost_gradient_including_rest(self):
        s=self.solver;s.travel_smoothing=0;u=self.control.copy()
        u.reshape(s.steps,s.p.N,3)[4,0,-1]=0
        J,g=s.value_gradient(u);d=np.zeros_like(u);d.reshape(s.steps,s.p.N,3)[4,0,-1]=1
        fd=(s.forward(u+1e-6*d)['J']-J)/1e-6
        self.assertAlmostEqual(fd,float(np.sum(g*d)),delta=2e-5)

    def test_smoothing_error_has_declared_uniform_bound(self):
        s=self.solver;state=s.forward(self.control)
        error=state['original_J']-state['J']
        self.assertGreaterEqual(error,0)
        self.assertLessEqual(error,s.p.gamma*s.p.N*s.p.T*s.travel_smoothing+1e-12)

    def test_hull_projection_and_gauge_preserve_feasibility(self):
        s=self.solver;u=s.project(self.control+2)
        _,_,targets,a=s.unpack(u)
        distances=np.einsum('tni,ji->tnj',targets,s.halfspaces[:,:2])+s.halfspaces[:,2]
        self.assertLess(distances.max(),1e-9)
        self.assertGreaterEqual(a.min(),0);self.assertLessEqual(a.max(),1)
        v=s.canonicalize(self.control)
        np.testing.assert_allclose(s.forward(v)['x'],s.forward(self.control)['x'],atol=1e-10)
        self.assertAlmostEqual(s.forward(v)['J'],s.forward(self.control)['J'],places=9)

    def test_reconstructed_time_only_controls_replay_original_equations(self):
        schedule=OriginalSchedule(self.solver,self.control)
        checks=schedule.checks()
        self.assertLess(checks['max_velocity_reconstruction_error'],1e-9)
        self.assertLess(checks['max_fish_quadrature_error'],1e-10)
        replay=schedule.replay(1e-10,.1)
        self.assertLess(replay['max_position_error'],1e-7)
        self.assertLess(replay['max_fish_error'],1e-7)
        self.assertAlmostEqual(replay['original_objective'],schedule.state['original_J'],delta=1e-6)
        for k in range(self.solver.steps):
            u,v=schedule.controls(k,(k+.5)*self.solver.dt)
            self.assertGreaterEqual(u.min(),0);self.assertLessEqual(u.max(),10)
            self.assertGreaterEqual(v.min(),0);self.assertLessEqual(v.max(),self.solver.p.V)
            self.assertTrue(np.all((u>0).sum(axis=1)<=2))

    def test_sharp_pair_crowding_integral_and_derivative(self):
        for a,b in [(np.array([-1.,0.]),np.array([2.,0.])),(np.array([0.,0.]),np.array([.2,.3])),(np.array([3.,4.]),np.zeros(2))]:
            expected=quad(lambda t:1/(1e-4+np.sum((a+t*b)**2)),0,1,epsabs=1e-9)[0]
            actual=pair_integral(a[None],b[None],1e-4)[0]
            self.assertAlmostEqual(actual,expected,delta=1e-8)
        a=np.array([[3.,4.]]);b=np.array([[1.,-2.]])
        shifted=a.astype(complex);shifted[0,0]+=1e-20j
        derivative=pair_integral(shifted,b,1e-4)[0].imag/1e-20
        expected=quad(lambda t:-2*(3+t)/(1e-4+np.sum((a[0]+t*b[0])**2))**2,0,1)[0]
        self.assertAlmostEqual(derivative,expected,places=11)

if __name__=='__main__':unittest.main()
