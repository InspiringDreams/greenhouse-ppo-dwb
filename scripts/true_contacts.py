"""Checks every recorded pose for contact between the robot footprint and the world."""

import sys, math, csv, glob, os
from pathlib import Path
WS = Path.home() / 'greenhouse_ws'
sys.path.insert(0, str(WS / 'scripts'))
from astar_ground_truth import read_footprints

FOOT_X, FOOT_HX, FOOT_HY = -0.064, 0.1330, 0.1530

def pt_to_aabb(dx, dy, hx, hy):
    return math.hypot(max(abs(dx) - hx, 0.0), max(abs(dy) - hy, 0.0))

def true_gap(px, py, yaw, shapes):
    c, s = math.cos(yaw), math.sin(yaw)
    corners = [(px + lx*c - ly*s, py + lx*s + ly*c)
               for lx in (FOOT_X - FOOT_HX, FOOT_X + FOOT_HX) for ly in (-FOOT_HY, FOOT_HY)]
    best = float('inf')
    for sh in shapes:
        if sh['kind'] == 'cyl':
            dx, dy = sh['x'] - px, sh['y'] - py
            best = min(best, pt_to_aabb(dx*c + dy*s - FOOT_X, -dx*s + dy*c, FOOT_HX, FOOT_HY) - sh['r'])
            continue
        hx, hy = sh['sx']/2.0, sh['sy']/2.0
        d = float('inf')
        for wx, wy in corners:
            d = min(d, pt_to_aabb(wx - sh['x'], wy - sh['y'], hx, hy))
        for ox in (sh['x'] - hx, sh['x'] + hx):
            for oy in (sh['y'] - hy, sh['y'] + hy):
                dx, dy = ox - px, oy - py
                d = min(d, pt_to_aabb(dx*c + dy*s - FOOT_X, -dx*s + dy*c, FOOT_HX, FOOT_HY))
        best = min(best, d)
    return best

def find(name):
    hits = glob.glob(str(WS / '**' / name), recursive=True)
    if not hits: sys.exit(f'cannot find {name}')
    return Path(hits[0])

worlds_dir = WS / 'worlds' / 'generated'
sets = {'PPO': (find('episodes_ppo.csv'), find('trajectories_ppo')),
        'DWB': (find('episodes_dwb_v2.csv'), find('trajectories_dwb_v2'))}
out = {}
for name, (csvp, tdir) in sets.items():
    rows = list(csv.DictReader(open(csvp)))
    n_ep = n_contact = n_missing = 0
    per = {}
    for r in rows:
        w = r['world']; tp = tdir / f'{w}.csv'
        if not tp.exists(): n_missing += 1; continue
        shapes = read_footprints(worlds_dir / f'{w}.sdf')
        hit = 0
        for t in csv.DictReader(open(tp)):
            try:
                px, py, yaw = float(t['tx']), float(t['ty']), float(t['tyaw'])
            except (ValueError, KeyError):
                continue
            if not all(map(math.isfinite, (px, py, yaw))): continue
            if true_gap(px, py, yaw, shapes) <= 0.0: hit += 1
        n_ep += 1
        if hit: n_contact += 1
        per[w] = hit
    out[name] = per
    print(f'{name}: {n_contact}/{n_ep} episodes with TRUE contact  (missing trajectory files: {n_missing})')
with open('true_contacts.csv', 'w', newline='') as f:
    wr = csv.writer(f); wr.writerow(['planner', 'world', 'true_contact_samples'])
    for k, per in out.items():
        for w, h in per.items(): wr.writerow([k, w, h])
print('saved true_contacts.csv in', os.getcwd())
