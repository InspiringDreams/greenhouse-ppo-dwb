#!/usr/bin/env python3
"""Runs PPO on all 270 test worlds and writes one CSV row per episode."""

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

WS = Path.home() / 'greenhouse_ws'
WORLDS = WS / 'worlds' / 'generated'
RUNS = WS / 'runs'
RESULT_JSON = Path('/tmp/episode_result_ppo.json')

COLUMNS = ['episode_id', 'timestamp', 'git_sha', 'planner', 'world', 'cell_id',
           'curvature', 'density', 'seed', 'held_out', 'outcome', 'success',
           'collisions', 'contact_time_s', 'recovery_count', 'time_to_goal_s',
           'wall_time_s', 'path_length_m', 'straight_line_distance_m',
           'min_clearance_m', 'mean_clearance_m', 'mean_linear_vel',
           'mean_abs_angular_vel', 'odom_samples', 'goal_odom_x', 'goal_odom_y',
           'detail', 'stall_events', 'contact_steps', 'goal_radius_m',
           'truth_dist_m', 'odom_dist_m', 'odom_claimed_goal',
           'path_length_truth_m', 'truth_final_x', 'truth_final_y']

CURVATURES = ['straight', 'low', 'high']
DENSITIES = ['sparse', 'medium', 'cluttered']


def git_sha():
    try:
        r = subprocess.run(['git', '-C', str(WS), 'rev-parse', '--short', 'HEAD'],
                           capture_output=True, text=True, timeout=10)
        return r.stdout.strip() or 'nogit'
    except Exception:
        return 'nogit'


def episode_order(seeds):
    for s in seeds:
        for c in CURVATURES:
            for d in DENSITIES:
                yield f'greenhouse_{c}_{d}_s{s:02d}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=str(WS / 'models' / 'ppo_seed0'))
    ap.add_argument('--out', default=str(RUNS / 'episodes_ppo.csv'))
    ap.add_argument('--seeds', type=int, default=30)
    ap.add_argument('--timeout', type=float, default=240.0)
    ap.add_argument('--goal-radius', type=float, default=0.25)
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()

    out = Path(a.out)
    traj_dir = RUNS / 'trajectories_ppo'
    traj_dir.mkdir(parents=True, exist_ok=True)

    done, next_id = set(), 1
    if out.exists():
        with open(out) as f:
            for row in csv.DictReader(f):
                done.add(row['world'])
                next_id = max(next_id, int(row['episode_id']) + 1)
        print(f'resuming: {len(done)} episodes already recorded', flush=True)
    else:
        with open(out, 'w', newline='') as f:
            csv.DictWriter(f, COLUMNS).writeheader()

    sha = git_sha()
    todo = [w for w in episode_order(range(a.seeds)) if w not in done]
    if a.limit:
        todo = todo[:a.limit]
    print(f'{len(todo)} episodes to run -> {out}', flush=True)

    t_start = time.time()
    for i, world in enumerate(todo, 1):
        sdf = WORLDS / f'{world}.sdf'
        if not sdf.exists():
            print(f'  [{i}/{len(todo)}] {world}: NO SDF, skipped', flush=True)
            continue

        t0 = time.time()
        RESULT_JSON.unlink(missing_ok=True)
        p = subprocess.run(
            [sys.executable, str(WS / 'scripts' / 'run_episode_ppo.py'), str(sdf),
             '--model', a.model, '--timeout', str(a.timeout),
             '--goal-radius', str(a.goal_radius),
             '--traj', str(traj_dir / f'{world}.csv'),
             '--json-out', str(RESULT_JSON)],
            capture_output=True, text=True, timeout=a.timeout * 3 + 300)

        try:
            res = json.loads(RESULT_JSON.read_text())
        except (OSError, ValueError):
            res = {'world': world, 'outcome': 'harness_error', 'success': 0,
                   'detail': (p.stderr or p.stdout)[-200:].replace('\n', ' ')}

        res.update({'episode_id': next_id, 'planner': 'ppo', 'git_sha': sha,
                    'timestamp': datetime.now(timezone.utc).isoformat()})
        next_id += 1

        with open(out, 'a', newline='') as f:
            csv.DictWriter(f, COLUMNS, extrasaction='ignore',
                           restval='').writerow(res)

        el = time.time() - t_start
        print(f'  [{i}/{len(todo)}] {world}: {res.get("outcome")} '
              f'({time.time() - t0:.0f}s)  eta {el / i * (len(todo) - i) / 3600:.1f}h',
              flush=True)

    print('done', flush=True)


if __name__ == '__main__':
    main()
