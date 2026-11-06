"""Fresh PPO, progressively harder initial falls, and unchanged strict acceptance."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import sys

RL_DIR = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--blocks', type=int, default=18)
    parser.add_argument('--chunk', type=int, default=32768)
    parser.add_argument('--envs', type=int, default=8)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--resume', type=Path, help='Continue this experimental learner, not the author policy')
    parser.add_argument('--start-angle', type=int, default=5)
    args = parser.parse_args()
    resume = args.resume.resolve() if args.resume else None
    import torch
    import numpy as np
    import mujoco
    import gymnasium as gym
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.logger import configure
    torch.set_num_threads(1)
    run = RL_DIR / 'runs' / ('curriculum_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    snapshot = run / 'model'
    snapshot.parent.mkdir(parents=True)
    shutil.copytree(RL_DIR, snapshot,
                    ignore=shutil.ignore_patterns('*.zip','*.h','__pycache__','runs','.venv'))
    shutil.copy2(__file__, run / 'training_source.py')
    os.chdir(snapshot)
    sys.path.insert(0, str(snapshot))
    from robo1_env import Robo1GetupEnv, FALLEN_POSES, roll_pitch_to_quat
    levels = [5,10,15,20,30,35,40,45,55,65,75,90]

    class CurriculumEnv(Robo1GetupEnv):
        def __init__(self):
            super().__init__('robo1.xml')
            self.level = 5
            self.max_steps = 400
        def reset(self, seed=None, options=None):
            gym.Env.reset(self, seed=seed)
            self.step_count = self.success_count = 0
            options = options or {}
            self.current_pose_name = options.get('pose', self.np_random.choice(self.fallen_poses))
            angle = options.get('angle', self.np_random.uniform(0.7*self.level,self.level))
            roll,pitch = FALLEN_POSES[self.current_pose_name]
            ratio = angle/90.0
            mujoco.mj_resetData(self.model,self.data)
            self.data.qpos[:3] = [0,0,.08]
            self.data.qpos[3:7] = roll_pitch_to_quat(roll*ratio,pitch*ratio)
            self.target[:] = 0
            self.data.ctrl[:] = 0
            mujoco.mj_forward(self.model,self.data)
            for _ in range(self.settle_steps):
                mujoco.mj_step(self.model,self.data)
            self.prev_upright = self._upright()
            self.potential = self.measure_potential()
            return self._get_obs(), {'pose':self.current_pose_name}
        def measure_potential(self):
            upright = self._upright()
            gate = np.clip((upright-.65)/.35,0,1)
            return float(upright - .15*gate*np.sum(self.target**2))
        def step(self,action):
            old = self.potential
            obs,_,terminated,truncated,info = super().step(action)
            self.potential = self.measure_potential()
            # No positive survival reward for holding a partial standing pose.
            reward = 20*(.995*self.potential-old) - .2 - .8*(1-info['upright'])
            reward -= .001*float(np.sum(np.asarray(action)**2))
            if info['goal_pose']:
                reward += 8
            if terminated:
                reward += 100
            return obs,float(reward),bool(terminated),bool(truncated),info

    env = DummyVecEnv([lambda:Monitor(CurriculumEnv()) for _ in range(args.envs)])
    model = PPO('MlpPolicy',env,device='cpu',seed=args.seed,verbose=0,
                n_steps=512,batch_size=512,learning_rate=2e-4,gamma=.995,
                ent_coef=.002,use_sde=True,sde_sample_freq=16,
                policy_kwargs={'log_std_init':-.5})
    if resume:
        model = PPO.load(str(resume),env=env,device='cpu')
    model.set_logger(configure(str(run/'logs'),['csv']))
    model.save(str(run/('initial_policy' if resume else 'initial_random_policy')))
    (run/'configuration.json').write_text(json.dumps({'resume':str(resume) if resume else None,
        'seed':args.seed,'teacher_data_used':False,'author_policy_used':False,
        'start_angle':args.start_angle,'chunk':args.chunk,'blocks':args.blocks,'envs':args.envs},indent=2),encoding='utf-8')

    def evaluate(angle=None):
        evaluation = CurriculumEnv() if angle is not None else Robo1GetupEnv('robo1.xml')
        results=[]
        try:
            for pose in FALLEN_POSES:
                options={'pose':pose}
                if angle is not None:
                    options['angle']=angle
                obs,_=evaluation.reset(seed=123,options=options)
                best=-1.
                for step in range(evaluation.max_steps):
                    action,_=model.predict(obs,deterministic=True)
                    obs,_,done,timeout,info=evaluation.step(action)
                    best=max(best,float(info['upright']))
                    if done or timeout:
                        break
                results.append({'pose':pose,'success':bool(done),'steps':step+1,'best_upright':best})
        finally:
            evaluation.close()
        return results

    history=[]
    level_index=levels.index(args.start_angle)
    best_success=-1
    print(f'RUN={run}',flush=True)
    try:
        for block in range(args.blocks):
            angle=levels[level_index]
            for instance in env.envs:
                instance.unwrapped.level=angle
            model.learn(total_timesteps=args.chunk,reset_num_timesteps=False)
            stage=evaluate(angle)
            full=evaluate()
            passed=sum(r['success'] for r in full)
            row={'block':block+1,'steps':model.num_timesteps,'angle':angle,
                 'stage':stage,'full_falls':full}
            history.append(row)
            (run/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
            model.save(str(run/'latest_policy'))
            if passed>best_success:
                best_success=passed
                model.save(str(run/'best_policy'))
            print(json.dumps({'block':block+1,'steps':model.num_timesteps,'angle':angle,
                              'stage_success':sum(r['success'] for r in stage),'full_success':passed}),flush=True)
            if passed==4:
                print('PASS: 4/4 original fixed-fall tests',flush=True)
                break
            if sum(r['success'] for r in stage)>=3 and level_index<len(levels)-1:
                level_index+=1
    finally:
        model.save(str(run/'latest_policy'))
        (run/'training_result.json').write_text(json.dumps({'total_timesteps':model.num_timesteps,
            'best_fixed_fall_successes':best_success,'tested_poses':4},indent=2),encoding='utf-8')
        env.close()


if __name__=='__main__':
    main()
