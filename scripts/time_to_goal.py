#!/usr/bin/env python3
"""Works out time to goal from the recorded paths (first time within 0.75 m)."""

import csv, json, math, statistics as st
from pathlib import Path
WS = Path.home() / 'greenhouse_ws'
WORLDS = WS / 'worlds' / 'generated'
DIRS = {'PPO': WS / 'runs' / 'trajectories_ppo',
        'DWB': WS / 'runs' / 'trajectories_dwb_v2'}
TOL = 0.75
LIMIT = 240.0

def first_entry(path, goal, tol):
    t0, p0 = None, None
    with open(path) as f:
        for row in csv.DictReader(f):
            try:
                t, x, y = float(row['t']), float(row['tx']), float(row['ty'])
            except (KeyError, ValueError):
                continue
            if not (math.isfinite(t) and math.isfinite(x) and math.isfinite(y)):
                continue
            if t > LIMIT:
                break
            if p0 is None:
                p0 = (x, y)
            if t0 is None and math.hypot(x - p0[0], y - p0[1]) > 0.05:
                t0 = t
            if t0 is not None and math.hypot(goal[0] - x, goal[1] - y) < tol:
                return t - t0
    return None

res = {}
for pl, d in DIRS.items():
    for p in sorted(d.glob('*.csv')):
        meta = json.loads((WORLDS / f'{p.stem}.json').read_text())
        g = (meta['goal_pose']['x'], meta['goal_pose']['y'])
        res[(pl, p.stem)] = first_entry(p, g, TOL)

print(f'--- time from first movement to first within {TOL} m of the goal ---')
for pl in DIRS:
    v = [t for (q, w), t in res.items() if q == pl and t is not None]
    n = sum(1 for (q, w) in res if q == pl)
    print(f'{pl}: reached in {len(v)}/{n} episodes, median {st.median(v):.1f} s')
both = [(res[('PPO', w)], res[('DWB', w)]) for (q, w) in res
        if q == 'PPO' and res[('PPO', w)] is not None and res.get(('DWB', w)) is not None]
print(f'same worlds, both reached ({len(both)}): PPO median {st.median(a for a, b in both):.1f} s, '
      f'DWB median {st.median(b for a, b in both):.1f} s')

with open(WS / 'runs' / 'ttg_times.csv', 'w') as f:
    f.write('planner,world,seconds\n')
    for (pl, w), t in res.items():
        if t is not None:
            f.write(f'{pl},{w},{t}\n')
