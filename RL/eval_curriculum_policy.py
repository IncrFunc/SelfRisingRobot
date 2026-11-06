"""Runs inside a snapshotted RL directory using the project's Python environment."""
import argparse
import json
from pathlib import Path
import sys
import time
import itertools

sys.path.insert(0, str(Path.cwd()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['evaluate', 'play'])
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--pose', default='roll_pos')
    args = parser.parse_args()
    import numpy as np
    from stable_baselines3 import PPO
    from robo1_env import DEFAULT_FALLEN_POSES, Robo1GetupEnv
    model = PPO.load(args.model, device='cpu')
    if args.command == 'play':
        import mujoco.viewer
        poses = DEFAULT_FALLEN_POSES if args.pose == 'all' else (args.pose,)
        pose_cycle = itertools.cycle(poses)
        env = Robo1GetupEnv('robo1.xml', fallen_poses=poses)
        try:
            pose = next(pose_cycle)
            obs, _ = env.reset(options={'pose': pose})
            with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
                while viewer.is_running():
                    start = time.perf_counter()
                    action, _ = model.predict(obs, deterministic=True)
                    obs, _, terminated, truncated, _ = env.step(action)
                    viewer.sync()
                    time.sleep(max(0, env.model.opt.timestep * env.frame_skip - (time.perf_counter()-start)))
                    if terminated or truncated:
                        pose = next(pose_cycle)
                        obs, _ = env.reset(options={'pose': pose})
        finally:
            env.close()
        return
    rows = []
    for pose in DEFAULT_FALLEN_POSES:
        env = Robo1GetupEnv('robo1.xml', fallen_poses=(pose,))
        try:
            obs, _ = env.reset(seed=0, options={'pose':pose})
            total_reward = 0.0
            best_upright = -1.0
            for step in range(env.max_steps):
                action, _ = model.predict(obs, deterministic=True)
                if not np.isfinite(action).all():
                    raise RuntimeError('Policy produced non-finite actions')
                obs, reward, terminated, truncated, info = env.step(action)
                if not np.isfinite(env.data.qpos).all():
                    raise RuntimeError('Simulation produced a non-finite state')
                total_reward += float(reward)
                best_upright = max(best_upright, float(info['upright']))
                if terminated or truncated:
                    break
            row = {'pose':pose, 'success':bool(terminated), 'steps':step+1,
                   'reward':total_reward, 'best_upright':best_upright,
                   'success_count':int(info['success_count']),
                   'final_target':np.asarray(info['target']).tolist()}
            rows.append(row)
            print(json.dumps(row), flush=True)
        finally:
            env.close()
    result = {'model':args.model, 'total':len(rows),
              'successes':sum(row['success'] for row in rows), 'poses':rows}
    if args.output:
        args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')


if __name__ == '__main__':
    main()
