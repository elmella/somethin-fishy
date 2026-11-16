import unittest
import numpy as np
from scipy.integrate import solve_ivp
from fishing import Problem
from regularized_pmp import RegularizedSweep, radial_flow, verify_continuous
from pmp_edge import EdgeSweep, edge_initialization
from pontryagin_controller import reset_stopped_headings,FeedbackPolicy


class PontryaginTests(unittest.TestCase):
    def setUp(self):
        self.p=Problem(np.array([[0.,0.],[10.,0.],[5.,10.]]),np.array([[2.,3.],[7.,3.]]),T=4.)
        self.s=RegularizedSweep(self.p,20,eta=.5)
        rng=np.random.default_rng(4);theta=rng.dirichlet(np.ones(3)*3,size=(20,2))
        target=np.einsum('tnj,ji->tni',theta,self.p.grounds)
        self.u=np.concatenate((target/20,rng.uniform(.2,.7,(20,2,1))),axis=-1).ravel()

    def test_exact_radial_flow_and_derivative(self):
        for radius in [0.,.001,.1,1.,100.]:
            x=np.array([[0.,0.]]);target=np.array([[radius,0.]])
            end,A,B,_=radial_flow(x,target,np.array([.6]),2.,1.,.5)
            numerical=solve_ivp(lambda t,z:.6*(target[0]-z)/np.sqrt(np.sum((target[0]-z)**2)+.25),[0,2],x[0],rtol=1e-11,atol=1e-13)
            np.testing.assert_allclose(end[0],numerical.y[:,-1],atol=1e-9)
            for j in range(2):
                h=np.zeros_like(x);h[0,j]=1e-6
                fd=(radial_flow(x+h,target,np.array([.6]),2.,1.,.5)[0]-radial_flow(x-h,target,np.array([.6]),2.,1.,.5)[0])/(2e-6)
                np.testing.assert_allclose(fd[0],A[0,:,j],atol=2e-8)

    def test_gradient_including_fish_crowding_speed_and_heading(self):
        J,g=self.s.value_gradient(self.u);rng=np.random.default_rng(10)
        for _ in range(4):
            d=rng.normal(size=self.u.shape);d/=np.linalg.norm(d);h=1e-6
            fd=(self.s.forward(self.u+h*d)['J']-self.s.forward(self.u-h*d)['J'])/(2*h)
            self.assertAlmostEqual(fd,float(np.sum(g*d)),delta=3e-5)

    def test_constant_controls_have_exact_mesh_refinement(self):
        fine=RegularizedSweep(self.p,40,eta=.5)
        repeated=np.repeat(self.u.reshape(20,2,3),2,axis=0).ravel()
        coarse=self.s.forward(self.u);refined=fine.forward(repeated)
        np.testing.assert_allclose(refined['x'][::2],coarse['x'],atol=1e-11)
        self.assertAlmostEqual(coarse['J'],refined['J'],places=7)

    def test_zero_beta_removes_only_crowding_and_retains_correct_gradient(self):
        from dataclasses import replace
        no_crowd=RegularizedSweep(replace(self.p,beta=0.),20,eta=.5)
        before=self.s.forward(self.u);after=no_crowd.forward(self.u)
        np.testing.assert_array_equal(before['x'],after['x'])
        np.testing.assert_array_equal(before['fish'],after['fish'])
        self.assertEqual(after['crowd'],0.)
        self.assertAlmostEqual(before['J']-after['J'],before['crowd'],places=10)
        _,gradient=no_crowd.value_gradient(self.u)
        direction=np.random.default_rng(29).normal(size=self.u.shape)
        direction/=np.linalg.norm(direction);h=1e-6
        fd=(no_crowd.forward(self.u+h*direction)['J']-no_crowd.forward(self.u-h*direction)['J'])/(2*h)
        self.assertAlmostEqual(fd,float(gradient@direction),delta=3e-5)

    def test_finalizer_preserves_custom_fleet_and_zero_beta(self):
        import contextlib,json,io,tempfile
        from pathlib import Path
        from dataclasses import replace
        from unittest.mock import patch
        from finalize_pmp import finalize
        p=replace(self.p,beta=0.)
        values={k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in vars(p).items()}
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'custom.npz'
            np.savez(path,control=self.u)
            path.with_suffix('.json').write_text(json.dumps(dict(problem=values)))
            with patch('finalize_pmp.save') as saved,contextlib.redirect_stdout(io.StringIO()):
                solver,control,report=finalize(path,'custom_verified',T=p.T)
            self.assertEqual(solver.p.N,2)
            self.assertEqual(solver.steps,20)
            self.assertEqual(solver.p.beta,0.)
            self.assertEqual(report['crowding'],0.)
            np.testing.assert_array_equal(control,self.u)
            self.assertEqual(saved.call_count,1)

    def test_refinement_preserves_custom_fleet_and_zero_beta(self):
        import contextlib,json,io,tempfile
        from pathlib import Path
        from dataclasses import replace
        from unittest.mock import patch
        from edge_refine_pmp import refine
        p=replace(self.p,beta=0.)
        values={k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in vars(p).items()}
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'custom.npz'
            np.savez(path,control=self.u,eta=.5,nu=.001)
            path.with_suffix('.json').write_text(json.dumps(dict(problem=values)))
            with patch('edge_refine_pmp.save'),contextlib.redirect_stdout(io.StringIO()):
                solver,control,report=refine(path,'refined',horizon=p.T,steps=40,cycles=1,evaluations=10)
            self.assertEqual(solver.p.N,2)
            self.assertEqual(solver.p.beta,0.)
            self.assertEqual(solver.steps,40)
            self.assertEqual(len(control),40*2*3)
            self.assertEqual(solver.forward(control)['crowd'],0.)

    def test_matched_seed_reaches_all_ten_grounds_without_changing_old_seed(self):
        from scenario_experiments import problem_for
        baseline=RegularizedSweep(problem_for('baseline_300'),150,eta=.5)
        self.assertAlmostEqual(baseline.forward(baseline.seed())['J'],22.48028081705239,places=8)
        expanded=RegularizedSweep(problem_for('grounds_300',0),150,eta=.5)
        old=expanded.forward(expanded.seed('coverage'))
        improved=expanded.forward(expanded.seed('coverage-matched'))
        self.assertGreater(old['fish'].max(),.99)
        self.assertLess(improved['fish'].max(),.2)

    def test_roundoff_speed_can_stop_but_infeasible_moving_heading_is_rejected(self):
        from fishing import stable_rhs
        from pmp_validation import reconstruct_attractions
        x=self.p.ships.copy();x[0]=[-1e-12,0.]
        velocity=np.array([[1e-12,-1e-13],[0.,0.]])
        fish=np.ones(self.p.K)
        u,v=reconstruct_attractions(x,fish,velocity,self.p)
        np.testing.assert_array_equal(v,0.)
        actual=stable_rhs(np.r_[x.ravel(),fish],u,v,self.p)[:4]
        np.testing.assert_array_equal(actual,0.)
        self.assertTrue(np.all((u>=0)&(u<=10)))
        velocity[0]=[.1,-.01]
        with self.assertRaisesRegex(ValueError,'No admissible'):
            reconstruct_attractions(x,fish,velocity,self.p)

    def test_deadline_seed_reserves_time_to_fish_at_both_harbor_visits(self):
        from scenario_experiments import problem_for
        solver=RegularizedSweep(problem_for('harbor_300'),150,eta=.5)
        old=solver.forward(solver.seed('coverage'))
        improved=solver.forward(solver.seed('coverage-deadline'))
        self.assertGreater(old['fish'].max(),.99)
        self.assertLess(improved['fish'].max(),.3)
        self.assertLess(improved['J'],old['J'])

    def test_nearly_feasible_heading_is_judged_by_velocity_error(self):
        from fishing import stable_rhs
        from pmp_validation import reconstruct_attractions
        x=self.p.ships.copy();x[0]=[2.,0.]
        velocity=np.array([[1e-4,-1e-13],[0.,0.]])
        fish=np.ones(self.p.K)
        u,v=reconstruct_attractions(x,fish,velocity,self.p)
        actual=stable_rhs(np.r_[x.ravel(),fish],u,v,self.p)[:4].reshape(2,2)
        self.assertLess(np.linalg.norm(actual-velocity),1e-10)
        self.assertTrue(np.all((u>=0)&(u<=10)))
        velocity[0]=[.1,-.01]
        with self.assertRaisesRegex(ValueError,'No admissible'):
            reconstruct_attractions(x,fish,velocity,self.p)

    def test_roundoff_arrival_stops_avoid_exact_center_without_changing_path(self):
        from dataclasses import replace
        from finalize_pmp import stop_roundoff_arrivals
        solver=RegularizedSweep(replace(self.p,T=40.),40,eta=.5)
        control=np.zeros((40,2,3));control[:,:,-1]=.9
        control[:,:,:2]=self.p.grounds[2]/solver.target_scale
        before=solver.forward(control.ravel())
        self.assertEqual(np.linalg.norm(before['x'][-1]-self.p.grounds[2],axis=-1).min(),0.)
        fixed,info=stop_roundoff_arrivals(solver,control.ravel())
        after=solver.forward(fixed)
        self.assertGreater(info['changed_intervals'],0)
        self.assertGreater(np.linalg.norm(after['x'][-1]-self.p.grounds[2],axis=-1).min(),0.)
        np.testing.assert_allclose(after['x'],before['x'],atol=1e-9)

    def test_existing_edge_targets_are_not_extended_again_at_a_vertex(self):
        from dataclasses import replace
        solver=RegularizedSweep(replace(self.p,T=40.),40,eta=.5)
        control=np.zeros((40,2,3));control[:,:,-1]=.9
        control[:,:,:2]=self.p.grounds[2]/solver.target_scale
        np.testing.assert_array_equal(edge_initialization(solver,control.ravel()),control.ravel())

    def test_stopped_heading_update_preserves_path_but_changes_speed_gradient(self):
        u=self.u.copy();u.reshape(20,2,3)[0,0,-1]=0
        proposal=u.copy();proposal.reshape(20,2,3)[0,0,:2]=self.p.grounds[0]/20
        _,before=self.s.value_gradient(u)
        updated,count=reset_stopped_headings(self.s,u,proposal)
        _,after=self.s.value_gradient(updated)
        self.assertEqual(count,1)
        self.assertAlmostEqual(before.reshape(20,2,3)[0,0,0],0,places=12)
        self.assertNotAlmostEqual(before.reshape(20,2,3)[0,0,-1],after.reshape(20,2,3)[0,0,-1],places=6)

    def test_edge_coordinates_gradient_and_endpoint_preservation(self):
        canonical=edge_initialization(self.s,self.u)
        np.testing.assert_allclose(self.s.forward(canonical)['x'],self.s.forward(self.u)['x'],atol=1e-8)
        edge=EdgeSweep(self.s,canonical);J,g=edge.value_gradient(edge.initial)
        rng=np.random.default_rng(13);d=rng.normal(size=g.shape);d/=np.linalg.norm(d);h=1e-6
        fd=(edge.forward(edge.initial+h*d)['J']-edge.forward(edge.initial-h*d)['J'])/(2*h)
        self.assertAlmostEqual(fd,float(np.sum(g*d)),delta=3e-5)

    def test_independent_continuous_replay(self):
        report=verify_continuous(self.s,self.u,1e-10)['replay']
        self.assertLess(report['max_position_error'],1e-7)
        self.assertLess(report['max_fish_error'],1e-7)
        self.assertAlmostEqual(report['J'],self.s.forward(self.u)['J'],delta=1e-6)

    def test_feedback_realization_uses_original_physical_equations(self):
        from pmp_validation import RegularizedSchedule,replay_feedback
        schedule=RegularizedSchedule(self.s,self.u)
        checks=schedule.checks()
        self.assertLess(checks['max_velocity_reconstruction_error'],1e-10)
        self.assertGreater(checks['minimum_center_clearance'],0)
        self.assertLessEqual(checks['attraction_max'],10)
        self.assertLessEqual(checks['speed_max'],self.p.V)
        replay=replay_feedback(schedule,1e-10,.1)
        self.assertLess(replay['max_position_error'],1e-7)
        self.assertAlmostEqual(replay['original_objective'],self.s.forward(self.u)['original_J'],delta=1e-6)

    def test_executable_policy_reconstructs_from_current_state(self):
        policy=FeedbackPolicy(self.s,self.u)
        state=np.r_[(self.p.ships+.1).ravel(),np.full(self.p.K,.7)]
        actual=policy.rhs(.1,state)
        q=policy.targets[0]-state[:2*self.p.N].reshape(-1,2)
        expected=policy.fractions[0,:,None]*self.p.V*q/np.sqrt(np.sum(q*q,axis=-1)+self.s.eta**2)[:,None]
        np.testing.assert_allclose(actual[:2*self.p.N].reshape(-1,2),expected,atol=1e-12)
        self.assertTrue(np.all(actual[2*self.p.N:]<=0))

if __name__=='__main__':unittest.main()
