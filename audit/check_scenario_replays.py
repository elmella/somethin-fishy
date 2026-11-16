"""Replay every selected scenario with the current executable controller."""
import argparse,json
from scenario_experiments import CASES,problem_for,policy_name
from pontryagin_controller import OUT,load_saved
from pmp_validation import RegularizedSchedule,replay_feedback


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--case',choices=CASES)
    args=parser.parse_args()
    path=OUT/'scenarios/replay_checks.json'
    records=json.loads(path.read_text()) if args.case and path.exists() else {}
    for case in ([args.case] if args.case else CASES):
        for beta in (5,0):
            name=policy_name(case,beta)+'_verified'
            if not (OUT/(name+'.json')).exists():continue
            s,u,report=load_saved(name,problem_for(case,beta))
            schedule=RegularizedSchedule(s,u);checks=schedule.checks()
            replay=replay_feedback(schedule,1e-10,.25)
            assert replay['max_position_error']<1e-5
            assert replay['max_fish_error']<1e-7
            assert abs(replay['original_objective']-report['original_J'])<1e-5
            assert checks['minimum_center_clearance']>0
            assert checks['max_velocity_reconstruction_error']<1e-9
            records[name]=dict(controls=checks,replay=replay)
            path.write_text(json.dumps(records,indent=2))
            print(name,'physical feedback replay passed',flush=True)


if __name__=='__main__':main()
