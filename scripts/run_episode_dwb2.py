#!/usr/bin/env python3
"""Runs one DWB (Nav2) episode in one world and returns its result."""

import argparse
import csv
import json
import math
import signal
import subprocess
import sys
import time
from pathlib import Path

WS = Path.home() / 'greenhouse_ws'
sys.path.insert(0, str(WS / 'scripts'))

from goal_frame import goal_in_odom
from run_episode import (kill_all, sh, wait_for_data, wait_for_nav2,
                         send_goal, parse_feedback)

ALLOWANCE_S = 240.0


def score_from_truth(traj_path, goal_xy, goal_radius, allowance_s):
    p = Path(traj_path)
    if not p.exists():
        return None, None, 0
    gx, gy = goal_xy
    best, best_t, n = float('inf'), None, 0
    with p.open() as f:
        rd = csv.DictReader(f)
        if not rd.fieldnames or 'tx' not in rd.fieldnames:
            return None, None, 0
        for row in rd:
            try:
                t = float(row.get('t', 0.0))
                tx, ty = float(row['tx']), float(row['ty'])
            except (TypeError, ValueError):
                continue
            if t > allowance_s:
                break
            n += 1
            d = math.hypot(gx - tx, gy - ty)
            if d < best:
                best, best_t = d, t
    if n == 0:
        return None, None, 0
    return best, best_t, n


def run(world_path, timeout_s, goal_radius, traj_path):
    meta_path = Path(world_path).with_suffix('.json')
    meta = json.loads(meta_path.read_text())
    gx_odom, gy_odom = goal_in_odom(meta_path)
    goal_world = (meta['goal_pose']['x'], meta['goal_pose']['y'])

    kill_all()
    metrics = None
    metrics_json = Path('/tmp/episode_metrics_dwb2.json')

    base = {
        'world': meta['world_name'], 'curvature': meta['curvature'],
        'density': meta['density'], 'cell_id': meta['cell_id'],
        'seed': meta['seed'], 'held_out': meta['held_out'],
        'goal_odom_x': round(gx_odom, 4), 'goal_odom_y': round(gy_odom, 4),
        'straight_line_distance_m': meta['straight_line_distance_m'],
        'goal_radius_m': goal_radius,
    }

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

        sh('ros2 launch greenhouse_sim nav2.launch.py',
           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if not wait_for_nav2():
            return {**base, 'outcome': 'setup_failed', 'success': 0,
                    'detail': 'nav2 never became active'}

        wait_for_data('/global_costmap/costmap', timeout=30)
        time.sleep(3)

        Path(traj_path).unlink(missing_ok=True)
        metrics_json.unlink(missing_ok=True)
        metrics = sh(f'python3 {WS}/scripts/metrics_node.py '
                     f'{metrics_json} {traj_path}',
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(1)

        text, wall, timed_out = send_goal(gx_odom, gy_odom, timeout_s)

        if timed_out:
            nav2_outcome = 'timeout'
        elif 'SUCCEEDED' in text:
            nav2_outcome = 'success'
        elif 'ABORTED' in text:
            nav2_outcome = 'aborted'
        elif 'rejected' in text.lower():
            nav2_outcome = 'setup_failed'
        else:
            nav2_outcome = 'unknown'

        if metrics is not None:
            metrics.send_signal(signal.SIGINT)
            time.sleep(2)

        collected = {}
        if metrics_json.exists():
            collected = json.loads(metrics_json.read_text())

        best, best_t, n = score_from_truth(traj_path, goal_world,
                                           goal_radius, ALLOWANCE_S)

        if n == 0:
            return {**base, 'outcome': 'setup_failed', 'success': 0,
                    'nav2_outcome': nav2_outcome,
                    'detail': 'no ground-truth samples in trajectory'}

        reached = best < goal_radius
        if reached:
            outcome = 'success'
        elif nav2_outcome == 'timeout':
            outcome = 'timeout'
        else:
            outcome = nav2_outcome if nav2_outcome != 'success' else 'aborted'

        result = {
            **base,
            'outcome': outcome,
            'success': 1 if reached else 0,
            'nav2_outcome': nav2_outcome,
            'nav2_claimed_goal': 1 if nav2_outcome == 'success' else 0,
            'truth_min_dist_m': round(best, 4),
            'truth_min_dist_time_s': (round(best_t, 2)
                                      if best_t is not None else None),
            'truth_samples': n,
            'wall_time_s': round(wall, 2),
        }
        result.update(parse_feedback(text))
        result.update(collected)
        if reached and best_t is not None:
            result['time_to_goal_s'] = round(best_t, 3)
        return result

    finally:
        kill_all()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('world')
    ap.add_argument('--timeout', type=float, default=ALLOWANCE_S)
    ap.add_argument('--goal-radius', type=float, default=0.25)
    ap.add_argument('--traj', default='/tmp/episode_traj_dwb2.csv')
    ap.add_argument('--json-out')
    a = ap.parse_args()
    res = run(a.world, a.timeout, a.goal_radius, a.traj)
    if a.json_out:
        Path(a.json_out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
