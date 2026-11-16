"""Checks the math against numerical derivatives and independent quadrature."""
import unittest
import numpy as np
from scipy.integrate import quad, solve_ivp
from fishing import (Problem, Segment, rhs, hamiltonian, adjoint, control_gradient,
                     gaussian_exposure, reciprocal_distance_integral, scenario,
                     evaluate_routes, pair_crowding, maximize_hamiltonian, running_cost,
                     return_segments, evaluate_return_routes)


def derivative_fixture():
    problem = Problem(np.array([[3., 2.], [-1., 4.], [4., -2.]]),
                      np.array([[.2, -.3], [1., .7]]), alpha=.12, sigma=.2,
                      rate=.3, beta=.7, gamma=.2, epsilon=.1)
    y = np.r_[problem.ships.ravel(), [.8, .3, .6]]
    p = np.array([.7, -.4, -.6, .8, -.9, -.5, -.3])
    u = np.array([[.3, .8, .2], [.9, .4, .7]])
    v = np.array([.6, .9])
    return problem, y, p, u, v


def complex_gradient(function, x):
    grad = np.zeros(x.size)
    for k in range(x.size):
        z = x.astype(complex).copy().reshape(-1)
        z[k] += 1e-25j
        grad[k] = np.imag(function(z.reshape(x.shape)))/1e-25
    return grad.reshape(x.shape)


class FishingTests(unittest.TestCase):
    def test_adjoint_is_negative_H_state_derivative(self):
        problem, y, p, u, v = derivative_fixture()
        reference = -complex_gradient(lambda z: hamiltonian(z, p, u, v, problem), y)
        np.testing.assert_allclose(adjoint(y, p, u, v, problem), reference, atol=1e-11, rtol=1e-11)

    def test_control_gradient(self):
        problem, y, p, u, v = derivative_fixture()
        Hu, Hv = control_gradient(y, p, u, v, problem)
        np.testing.assert_allclose(Hu, complex_gradient(lambda z: hamiltonian(y, p, z, v, problem), u), atol=1e-11)
        np.testing.assert_allclose(Hv, complex_gradient(lambda z: hamiltonian(y, p, u, z, problem), v), atol=1e-11)

    def test_backward_adjoint_predicts_actual_objective_change(self):
        problem,y0,_,u,v=derivative_fixture()
        T=1.2
        def forward(control,speed):
            sol=solve_ivp(lambda t,z: np.r_[rhs(z[:-1],control,speed,problem),
                          running_cost(z[:-1],speed,problem)], [0,T],np.r_[y0,0.],
                          rtol=1e-10,atol=1e-12,dense_output=True)
            return sol,sol.y[-1,-1]+problem.delta*sol.y[2*problem.N:-1,-1].sum()
        state,_=forward(u,v)
        pT=np.r_[np.zeros(2*problem.N),np.full(problem.K,-problem.delta)]
        backward=solve_ivp(lambda t,p: adjoint(state.sol(t)[:-1],p,u,v,problem),
                           [T,0],pT,rtol=1e-10,atol=1e-12,dense_output=True)
        du=np.array([[.2,-.3,.1],[-.1,.2,.3]])
        dv=np.array([.1,-.2])
        def integrand(t):
            Hu,Hv=control_gradient(state.sol(t)[:-1],backward.sol(t),u,v,problem)
            return -np.sum(Hu*du)-Hv@dv
        predicted=quad(integrand,0,T,epsabs=1e-9)[0]
        eps=1e-5
        observed=(forward(u+eps*du,v+eps*dv)[1]-forward(u-eps*du,v-eps*dv)[1])/(2*eps)
        np.testing.assert_allclose(predicted,observed,rtol=2e-7,atol=2e-7)

    def test_speed_rule_maximizes_H(self):
        problem, y, p, u, v = derivative_fixture()
        _, switch = control_gradient(y, p, u, v, problem)
        selected = problem.V*(switch > 0)
        best = hamiltonian(y, p, u, selected, problem)
        for a in np.linspace(0, problem.V, 7):
            for b in np.linspace(0, problem.V, 7):
                self.assertGreaterEqual(best+1e-12, hamiltonian(y, p, u, np.array([a, b]), problem))

    def test_zero_costate_speed_is_zero_with_positive_gamma(self):
        problem, y, p, u, v = derivative_fixture()
        _, switch = control_gradient(y, np.zeros_like(p), u, v, problem)
        np.testing.assert_allclose(switch, -problem.gamma)

    def test_positive_rescaling_keeps_original_motion(self):
        problem, y, p, u, v = derivative_fixture()
        np.testing.assert_allclose(rhs(y, .001*u, v, problem), rhs(y, u, v, problem), atol=1e-12)

    def test_exact_pointwise_maximum_beats_random_controls(self):
        problem, y, p, u, v = derivative_fixture()
        selected, speed = maximize_hamiltonian(y,p,problem)
        self.assertTrue(np.all((selected >= 0) & (selected <= 10)))
        best = hamiltonian(y,p,selected,speed,problem)
        rng = np.random.default_rng(8)
        for _ in range(1000):
            candidate = rng.uniform(0,10,u.shape)
            _, switch = control_gradient(y,p,candidate,v,problem)
            candidate_v = problem.V*(switch>0)
            self.assertGreaterEqual(best+1e-12,hamiltonian(y,p,candidate,candidate_v,problem))

    def test_cone_boundary_does_not_require_zero_control_derivative(self):
        problem = Problem(np.array([[1.,0.],[0.,1.]]),np.zeros((1,2)),gamma=.1)
        y = np.r_[0.,0.,1.,1.]
        p = np.array([1.,-1.,-1.,-1.])
        u,v = maximize_hamiltonian(y,p,problem)
        self.assertEqual(u[0,1],0)
        self.assertEqual(v[0],problem.V)
        Hu,_=control_gradient(y,p,u,v,problem)
        self.assertLess(Hu[0,1],0)  # Correct bound optimum, not Hu = 0.

    def test_exact_integrals_against_quadrature(self):
        rng = np.random.default_rng(4)
        for _ in range(30):
            a, b = rng.normal(size=(2, 2))
            L, epsilon = 2.3, .03
            expected = quad(lambda t: 1/(epsilon+np.sum((a+b*t)**2)), 0, L, epsabs=1e-10)[0]
            self.assertAlmostEqual(reciprocal_distance_integral(a, b, L, epsilon), expected, places=9)
            segment = Segment(0, L, a, b, 0)
            ground = rng.normal(size=2)
            expected = quad(lambda t: np.exp(-.4*np.sum((segment.position(t)-ground)**2)), 0, L, epsabs=1e-10)[0]
            self.assertAlmostEqual(gaussian_exposure(segment, ground[None], .4)[0], expected, places=10)

    def test_sharp_crossing_and_shared_harbor(self):
        eps = 1e-4
        expected = 2/np.sqrt(eps)*np.arctan(1/np.sqrt(eps))
        self.assertAlmostEqual(reciprocal_distance_integral(np.array([-1., 0.]), np.array([1., 0.]), 2, eps), expected, places=8)
        expected = 1/np.sqrt(eps)*np.arctan(400/np.sqrt(eps))
        self.assertAlmostEqual(reciprocal_distance_integral(np.zeros(2), np.array([1., 0.]), 400, eps), expected, places=8)

    def test_feasible_controls_and_full_ODE_replay(self):
        # Compact non-singular route; directly replay the original coupled ODE.
        p = Problem(np.array([[4., 2.], [-2., 4.]]), np.array([[0., 0.]]),
                    T=12., alpha=.2, sigma=.4, beta=.1, rate=.3)
        result, trajectories = evaluate_routes(p, [[0, 1]], [1.])
        y = np.r_[p.ships.ravel(), np.ones(p.K)*p.F0]
        for segment in trajectories[0]:
            u = np.zeros((1, p.K)); u[0, segment.target] = 1
            v = np.array([np.linalg.norm(segment.velocity)])
            self.assertLessEqual(v[0], p.V+1e-12)
            sol = solve_ivp(lambda t, state: rhs(state, u, v, p),
                            [segment.start, segment.end], y, rtol=2e-10, atol=1e-12)
            self.assertTrue(sol.success)
            y = sol.y[:, -1]
            np.testing.assert_allclose(y[:2], segment.position(segment.end), atol=2e-8)
        np.testing.assert_allclose(y[2:], result['remaining'], atol=2e-9)

    def test_fish_monotonic_and_constant_self_term(self):
        p = scenario()
        result, trajectories = evaluate_routes(p, [[i] for i in range(p.N)], np.zeros(p.N))
        F = np.array(result['remaining'])
        self.assertTrue(np.all((F >= 0) & (F <= p.F0)))
        self.assertAlmostEqual(result['J_full']-result['J_reduced'], p.self_cost)

    def test_return_constraint_and_infeasible_route(self):
        p=Problem(np.array([[4.,0.],[0.,4.],[100.,100.]]),np.zeros((1,2)),T=20.)
        for fraction in [0.,.3,1.]:
            result,trajectories=evaluate_return_routes(p,[[0,1]],[fraction])
            tr=trajectories[0]
            self.assertEqual(tr[-1].end,p.T)
            np.testing.assert_allclose(tr[-1].position(p.T),p.ships[0],atol=1e-12)
            self.assertTrue(all(np.linalg.norm(s.velocity)<=p.V+1e-12 for s in tr))
        self.assertIsNone(return_segments(p,0,[2],0.))
        result,tr=evaluate_return_routes(p,[[2]],[0.])
        self.assertTrue(np.isinf(result['J_reduced']))


if __name__ == '__main__':
    unittest.main()
