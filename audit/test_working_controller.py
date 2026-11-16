"""Regression tests for the observed stalling and feasible controller replay."""
import unittest
import numpy as np
from fishing import Problem, scenario, rhs, stable_rhs, evaluate_routes
from working_controller import replay


class WorkingControllerTests(unittest.TestCase):
    def test_distant_attraction_does_not_reduce_commanded_speed(self):
        p = scenario()
        y = np.r_[p.ships.ravel(), np.ones(p.K)]
        u = np.zeros((p.N, p.K)); u[:, 6] = 10.
        velocity = stable_rhs(y, u, np.ones(p.N), p)[:2*p.N].reshape(-1, 2)
        expected = p.grounds[6]-p.ships
        expected /= np.linalg.norm(expected, axis=1)[:, None]
        np.testing.assert_allclose(velocity, expected, atol=1e-14)

    def test_underflow_and_positive_rescaling_preserve_heading(self):
        p = Problem(np.array([[10000., 10000.]]), np.zeros((1, 2)))
        y = np.array([0., 0., 1e-300])
        for weight in [1., 1e-300]:
            velocity = stable_rhs(y, np.array([[weight]]), np.ones(1), p)[:2]
            np.testing.assert_allclose(velocity, np.ones(2)/np.sqrt(2), atol=1e-14)

    def test_matches_written_equations_in_regular_region(self):
        p = Problem(np.array([[3., 2.], [-1., 4.]]), np.array([[.2, -.3]]), alpha=.12)
        y = np.r_[p.ships.ravel(), [.8, .3]]
        u, v = np.array([[.3, .8]]), np.array([.6])
        np.testing.assert_allclose(stable_rhs(y, u, v, p), rhs(y, u, v, p), atol=1e-14)

    def test_stopped_boat_and_undefined_moving_heading(self):
        p = Problem(np.array([[1., 0.], [-1., 0.]]), np.zeros((1, 2)))
        y = np.r_[0., 0., 1., 1.]
        np.testing.assert_array_equal(stable_rhs(y, np.zeros((1, 2)), np.zeros(1), p)[:2], [0., 0.])
        for u in [np.zeros((1, 2)), np.ones((1, 2))]:
            with self.assertRaises(ValueError):
                stable_rhs(y, u, np.ones(1), p)
        with self.assertRaises(ValueError):
            stable_rhs(y, np.array([[1., 0.]]), np.array([1.1]), p)

    def test_full_ode_replay_matches_analytic_objective_and_switches(self):
        p = Problem(np.array([[4., 2.], [-2., 4.], [2., -3.]]),
                    np.array([[0., 0.], [1., -.5]]), T=25.)
        expected, tr = evaluate_routes(p, [[0, 1], [2]], [3.25, 0.], .5)
        data = replay(p, tr, rtol=1e-9, max_step=.5, sample_count=101)
        np.testing.assert_allclose(data['objective'], expected['J_reduced'], atol=1e-7)
        np.testing.assert_allclose(data['fish'][-1], expected['remaining'], atol=1e-9)
        np.testing.assert_allclose(data['x'][-1], [t[-1].position(p.T) for t in tr], atol=1e-9)
        self.assertLess(data['max_position_error'], 1e-9)

    def test_control_interval_shorter_than_display_spacing(self):
        p = Problem(np.array([[4., 2.], [-2., 4.]]), np.zeros((1, 2)), T=20.)
        expected, tr = evaluate_routes(p, [[0, 1]], [1e-8])
        data = replay(p, tr, sample_count=21)
        np.testing.assert_allclose(data['objective'], expected['J_reduced'], atol=1e-7)
        self.assertTrue(np.all(np.isfinite(data['x'])))


if __name__ == '__main__':
    unittest.main()
