#!/usr/bin/env python3
"""Finds the shortest drivable path in each world, used for SPL."""

import argparse
import csv
import heapq
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

WS = Path.home() / 'greenhouse_ws'
WORLDS = WS / 'worlds' / 'generated'
OUT_CSV = WS / 'runs' / 'optimal_paths.csv'

RES = 0.05
ROBOT_RADIUS = 0.22
SQRT2 = math.sqrt(2.0)


def read_footprints(sdf_path):
    root = ET.parse(sdf_path).getroot()
    shapes = []
    for model in root.find('world').findall('model'):
        name = model.get('name')
        if name == 'ground_plane':
            continue
        px, py = (float(v) for v in model.find('pose').text.split()[:2])
        geom = model.find('link/collision/geometry')
        box, cyl = geom.find('box'), geom.find('cylinder')
        if box is not None:
            sx, sy, _ = (float(v) for v in box.find('size').text.split())
            shapes.append({'name': name, 'kind': 'box', 'x': px, 'y': py,
                           'sx': sx, 'sy': sy})
        elif cyl is not None:
            shapes.append({'name': name, 'kind': 'cyl', 'x': px, 'y': py,
                           'r': float(cyl.find('radius').text)})
    return shapes


def build_grid(shapes, pad=ROBOT_RADIUS):
    xs = [s['x'] - (s.get('sx', 0) / 2 + s.get('r', 0)) for s in shapes]
    xs += [s['x'] + (s.get('sx', 0) / 2 + s.get('r', 0)) for s in shapes]
    ys = [s['y'] - (s.get('sy', 0) / 2 + s.get('r', 0)) for s in shapes]
    ys += [s['y'] + (s.get('sy', 0) / 2 + s.get('r', 0)) for s in shapes]
    x0, y0 = min(xs) - RES, min(ys) - RES
    nx = int((max(xs) + RES - x0) / RES) + 1
    ny = int((max(ys) + RES - y0) / RES) + 1

    grid = np.zeros((ny, nx), dtype=bool)

    for s in shapes:
        if s['kind'] == 'box':
            hx, hy = s['sx'] / 2 + pad, s['sy'] / 2 + pad
            ix0 = max(0, int((s['x'] - hx - x0) / RES))
            ix1 = min(nx - 1, int(math.ceil((s['x'] + hx - x0) / RES)))
            iy0 = max(0, int((s['y'] - hy - y0) / RES))
            iy1 = min(ny - 1, int(math.ceil((s['y'] + hy - y0) / RES)))
            grid[iy0:iy1 + 1, ix0:ix1 + 1] = True
        else:
            rad = s['r'] + pad
            ix0 = max(0, int((s['x'] - rad - x0) / RES))
            ix1 = min(nx - 1, int(math.ceil((s['x'] + rad - x0) / RES)))
            iy0 = max(0, int((s['y'] - rad - y0) / RES))
            iy1 = min(ny - 1, int(math.ceil((s['y'] + rad - y0) / RES)))
            for iy in range(iy0, iy1 + 1):
                for ix in range(ix0, ix1 + 1):
                    cx, cy = x0 + ix * RES, y0 + iy * RES
                    if (cx - s['x']) ** 2 + (cy - s['y']) ** 2 <= rad * rad:
                        grid[iy, ix] = True

    return grid, x0, y0


def seal_row_ends(grid, shapes, x0):
    xs = [s['x'] for s in shapes if s['name'].startswith('plant_')]
    g = grid.copy()
    i_lo = max(0, int((min(xs) - 0.30 - x0) / RES))
    i_hi = min(grid.shape[1], int((max(xs) + 0.30 - x0) / RES))
    g[:, :i_lo] = True
    g[:, i_hi:] = True
    return g


def to_cell(x, y, x0, y0):
    return int(round((x - x0) / RES)), int(round((y - y0) / RES))


def to_world(ix, iy, x0, y0):
    return x0 + ix * RES, y0 + iy * RES


NEIGHBOURS = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
              (1, 1, SQRT2), (1, -1, SQRT2), (-1, 1, SQRT2), (-1, -1, SQRT2)]


def astar(grid, start, goal):
    ny, nx = grid.shape
    gx, gy = goal

    def h(ix, iy):
        return math.hypot(gx - ix, gy - iy) * RES

    best = {start: 0.0}
    came = {}
    open_set = [(h(*start), 0.0, start)]

    while open_set:
        _, cost, cur = heapq.heappop(open_set)
        if cur == goal:
            path = [cur]
            while path[-1] in came:
                path.append(came[path[-1]])
            return path[::-1]
        if cost > best.get(cur, float('inf')):
            continue
        cx, cy = cur
        for dx, dy, step in NEIGHBOURS:
            nxt = (cx + dx, cy + dy)
            if not (0 <= nxt[0] < nx and 0 <= nxt[1] < ny) or grid[nxt[1], nxt[0]]:
                continue
            new = cost + step * RES
            if new < best.get(nxt, float('inf')):
                best[nxt] = new
                came[nxt] = cur
                heapq.heappush(open_set, (new + h(*nxt), new, nxt))
    return None


def visible(grid, a, b):
    dist = math.hypot(b[0] - a[0], b[1] - a[1])
    steps = max(1, int(dist * 2))
    for i in range(steps + 1):
        t = i / steps
        ix = int(round(a[0] + t * (b[0] - a[0])))
        iy = int(round(a[1] + t * (b[1] - a[1])))
        if grid[iy, ix]:
            return False
    return True


def shortcut(grid, path):
    out = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not visible(grid, path[i], path[j]):
            j -= 1
        out.append(path[j])
        i = j
    return out


def length_m(cells):
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) * RES
               for a, b in zip(cells, cells[1:]))


def solve(sdf_path):
    meta = json.loads(Path(sdf_path).with_suffix('.json').read_text())
    shapes = read_footprints(sdf_path)
    grid, x0, y0 = build_grid(shapes)

    start = to_cell(meta['start_pose']['x'], meta['start_pose']['y'], x0, y0)
    goal = to_cell(meta['goal_pose']['x'], meta['goal_pose']['y'], x0, y0)

    row = {'world': meta['world_name'], 'curvature': meta['curvature'],
           'density': meta['density'], 'seed': meta['seed'],
           'cell_id': meta['cell_id'], 'held_out': meta['held_out'],
           'straight_line_distance_m': meta['straight_line_distance_m'],
           'grid_res_m': RES, 'inflation_m': ROBOT_RADIUS}

    if grid[start[1], start[0]]:
        return {**row, 'solvable': 0, 'detail': 'start blocked'}, None, None
    if grid[goal[1], goal[0]]:
        return {**row, 'solvable': 0, 'detail': 'goal blocked'}, None, None

    raw = astar(grid, start, goal)
    if raw is None:
        return {**row, 'solvable': 0, 'aisle_passable': 0,
                'detail': 'no path'}, grid, None

    aisle_grid = seal_row_ends(grid, shapes, x0)
    aisle_path = astar(aisle_grid, start, goal)

    smooth = shortcut(grid, raw)
    row.update({
        'solvable': 1,
        'aisle_passable': 1 if aisle_path else 0,
        'aisle_length_m': round(length_m(shortcut(aisle_grid, aisle_path)), 4)
                          if aisle_path else '',
        'optimal_length_grid_m': round(length_m(raw), 4),
        'optimal_length_m': round(length_m(smooth), 4),
        'detour_ratio': round(length_m(smooth) / meta['straight_line_distance_m'], 4),
        'waypoints': len(smooth),
        'detail': '',
    })
    return row, grid, (raw, smooth, x0, y0)


def plot(world_name, grid, extra, out_png):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    raw, smooth, x0, y0 = extra
    ny, nx = grid.shape
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.imshow(grid, origin='lower', cmap='Greys', alpha=0.85,
              extent=[x0, x0 + nx * RES, y0, y0 + ny * RES])

    rx = [to_world(*c, x0, y0)[0] for c in raw]
    ry = [to_world(*c, x0, y0)[1] for c in raw]
    sx = [to_world(*c, x0, y0)[0] for c in smooth]
    sy = [to_world(*c, x0, y0)[1] for c in smooth]
    ax.plot(rx, ry, lw=1, color='tab:blue', label=f'A* grid  {length_m(raw):.2f} m')
    ax.plot(sx, sy, lw=2, color='tab:red', label=f'shortcut  {length_m(smooth):.2f} m')
    ax.plot(sx[0], sy[0], 'go', ms=8, label='start')
    ax.plot(sx[-1], sy[-1], 'r*', ms=14, label='goal')

    ax.set_title(f'{world_name}  -  free space for robot centre '
                 f'({ROBOT_RADIUS} m inflation, {RES} m cells)')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.legend(loc='upper right', fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f'wrote {out_png}')


COLUMNS = ['world', 'curvature', 'density', 'seed', 'cell_id', 'held_out',
           'solvable', 'aisle_passable', 'optimal_length_m', 'aisle_length_m',
           'optimal_length_grid_m',
           'straight_line_distance_m', 'detour_ratio', 'waypoints',
           'grid_res_m', 'inflation_m', 'detail']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--worlds', type=Path, default=WORLDS)
    ap.add_argument('--out', type=Path, default=OUT_CSV)
    ap.add_argument('--one', help='solve a single world by name and print it')
    ap.add_argument('--plot', action='store_true', help='with --one, save a PNG')
    args = ap.parse_args()

    if args.one:
        sdf = args.worlds / f'{args.one}.sdf'
        row, grid, extra = solve(sdf)
        print(json.dumps(row, indent=2))
        if args.plot and extra:
            plot(args.one, grid, extra, WS / 'runs' / f'astar_{args.one}.png')
        return

    sdfs = sorted(args.worlds.glob('*.sdf'))
    rows = []
    for i, sdf in enumerate(sdfs, 1):
        row, _, _ = solve(sdf)
        rows.append(row)
        mark = ''
        if not row['solvable']:
            mark = f"   UNSOLVABLE: {row['detail']}"
        elif not row['aisle_passable']:
            mark = '   AISLE BLOCKED (only route is around the rows)'
        print(f"[{i}/{len(sdfs)}] {row['world']:<34} "
              f"{row.get('optimal_length_m', 0):>7.2f} m{mark}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)

    bad = [r for r in rows if not r['solvable']]
    blocked = [r for r in rows if r['solvable'] and not r['aisle_passable']]
    ok = [r for r in rows if r['solvable']]
    print(f'\n{len(ok)}/{len(rows)} solvable -> {args.out}')
    if ok:
        lens = [r['optimal_length_m'] for r in ok]
        print(f'optimal length: min {min(lens):.2f}  '
              f'mean {sum(lens) / len(lens):.2f}  max {max(lens):.2f} m')
    print(f"{len(rows) - len(blocked)}/{len(rows)} passable through the aisle")
    for r in bad:
        print(f"  UNSOLVABLE {r['world']}: {r['detail']}")
    for r in blocked:
        print(f"  AISLE BLOCKED {r['world']}")


if __name__ == '__main__':
    main()
