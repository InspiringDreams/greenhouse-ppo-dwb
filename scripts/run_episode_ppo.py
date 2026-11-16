#!/usr/bin/env python3
"""Runs the trained PPO policy for one episode in one world and returns its result."""

import argparse
import json
import signal
import subprocess
import sys
import time
from pathlib import Path

WS = Path.home() / 'greenhouse_ws'
sys.path.insert(0, str(WS / 'scripts'))

from goal_frame import goal_in_odom
from run_episode import kill_all, sh, wait_for_data

STALL_WINDOW_S = 15.0
STALL_PROGRESS_M = 0.05


def count_stalls(dists, hz):
    w = int(STALL_WINDOW_S * hz)
    if len(dists) <= w:
        return 0
    events, i = 0, 0
    while i + w < len(dists):
        if dists[i] - dists[i + w] < STALL_PROGRESS_M:
            events += 1
            i += w
        else:
            i += 1
    return events


def run(world_path, model_path, timeout_s, goal_radius, traj_path):
    world_path = str(world_path)
    meta_path = Path(world_path).with_suffix('.json')
    meta = json.loads(meta_path.read_text())
    gx, gy = goal_in_odom(meta_path)

    base = {
        'world': meta['world_name'], 'curvature': meta['curvature'],
        'density': meta['density'], 'cell_id': meta['cell_id'],
        'seed': meta['seed'], 'held_out': meta['held_out'],
        'goal_odom_x': round(gx, 4), 'goal_odom_y': round(gy, 4),
        'straight_line_distance_m': meta['straight_line_distance_m'],
        'recovery_count': 0,
    }

    kill_all()
    metrics, env = None, None
    metrics_json = Path('/tmp/episode_metrics_ppo.json')

    try:
        sh(f'ros2 launch greenhouse_sim spawn_robot.launch.py '
           f'world:={world_path} gui:=false',
           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if not wait_for_data('/scan'):
            return {**base, 'outcome': 'setup_failed', 'success': 0,
                    'detail': 'no scan data'}
        if not wait_for_data('/odom'):
            return {**base, 'outcome': 'setup_failed', 'success': 0,
                    'detail': 'no odom data'}
        time.sleep(3)

        metrics_json.unlink(missing_ok=True)
        metrics = sh(f'python3 {WS}/scripts/metrics_node.py '
                     f'{metrics_json} {traj_path}',
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(1)

        from stable_baselines3 import PPO
        import greenhouse_env as ge
        from greenhouse_env import GreenhouseEnv

        ge.GOAL_RADIUS = goal_radius
        ge.STALL_LIMIT_S = 1e9

        env = GreenhouseEnv(meta['world_name'],
                            worlds_dir=meta_path.parent,
                            episode_limit_s=timeout_s,
                            start_x=None,
                            terminate_on_collision=False,
                            success_from_truth=True)
        model = PPO.load(model_path)

        t0 = time.time()
        obs, _ = env.reset()
        dists, steps, reached = [], 0, False
        while True:
            act, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(act)
            dists.append(info['distance'])
            steps += 1
            if term and info.get('outcome') == 'success':
                reached = True
                break
            if trunc or steps >= env.max_steps:
                break
        wall = time.time() - t0

        if metrics is not None:
            metrics.send_signal(signal.SIGINT)
            time.sleep(2)
        collected = json.loads(metrics_json.read_text()) \
            if metrics_json.exists() else {}

        result = {
            **base,
            'outcome': ('success' if reached else
                        'stalled' if info.get('true_stalled')
                        else 'timeout'),
            'success': 1 if reached else 0,
            'time_to_goal_s': round(steps / ge.CONTROL_HZ, 3),
            'wall_time_s': round(wall, 2),
            'stall_events': count_stalls(dists, ge.CONTROL_HZ),
            'contact_steps': getattr(env, 'eval_contacts', 0),
            'goal_radius_m': ge.GOAL_RADIUS,
            'truth_dist_m': (round(info['truth_dist'], 3)
                             if info.get('truth_dist') is not None else None),
            'odom_dist_m': round(info['distance'], 3),
            'odom_claimed_goal': int(info['distance'] < goal_radius),
        }
        result.update(collected)
        return result

    except RuntimeError as e:
        return {**base, 'outcome': 'setup_failed', 'success': 0,
                'detail': str(e)[:200]}
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        kill_all()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('world')
    ap.add_argument('--model', default=str(WS / 'models' / 'ppo_seed0'))
    ap.add_argument('--timeout', type=float, default=240.0)
    ap.add_argument('--goal-radius', type=float, default=0.25)
    ap.add_argument('--traj', default='/tmp/episode_traj_ppo.csv')
    ap.add_argument('--json-out')
    a = ap.parse_args()
    res = run(a.world, a.model, a.timeout, a.goal_radius, a.traj)
    if a.json_out:
        Path(a.json_out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
