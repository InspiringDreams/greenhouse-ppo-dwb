#!/usr/bin/env python3
"""Builds a greenhouse world (SDF + JSON) from a curvature level, a density level and a seed."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path


ROW_LENGTH_M = 12.0
N_ROWS = 4
PLANT_W = 0.30
PLANT_H = 0.80
PLANT_SPACING_M = 0.45
PLANT_JITTER_M = 0.030
AISLE_W = 1.80
ROW_PITCH = PLANT_W + AISLE_W

START_X = 0.5
GOAL_X = ROW_LENGTH_M - 0.5

OBST_X_RANGE = (1.5, 10.5)
MIN_PASSABLE_GAP = 0.80
MIN_OBST_SPACING = 0.85

WALL_MARGIN_X = 0.35
WALL_MARGIN_Y = 0.35
WALL_H = 2.0
WALL_T = 0.10

PHYSICS_STEP_S = 0.004
TIMEOUT_S = 120.0

N_HARMONICS = 3

PLANT_SIZE_MAX = PLANT_W * 1.12
CURVE_MISMATCH_MARGIN = 0.12
USABLE_HALF_AISLE = (ROW_PITCH / 2.0
                     - PLANT_SIZE_MAX / 2.0
                     - PLANT_JITTER_M
                     - CURVE_MISMATCH_MARGIN)


TARGET_MAX_ANGLE_DEG = {
    "straight": 0.0,
    "low": 16.0,
    "high": 32.0,
}

BASE_CYCLES = {
    "straight": 0.0,
    "low": 0.9,
    "high": 0.9,
}

DENSITY_COUNT = {
    "sparse": 2,
    "medium": 5,
    "cluttered": 9,
}

HELD_OUT_CELLS = {("high", "medium"), ("low", "cluttered")}


class Centreline:

    def __init__(self, level: str, rng: random.Random):
        self.level = level
        self.target_angle_rad = math.radians(TARGET_MAX_ANGLE_DEG[level])
        self.components: list[tuple[float, float, float]] = []
        self.scale = 0.0

        if TARGET_MAX_ANGLE_DEG[level] == 0.0:
            return

        base = BASE_CYCLES[level]
        for k in range(N_HARMONICS):
            cycles = base * rng.uniform(0.55, 1.60) * (1.0 + 0.30 * k)
            phase = rng.uniform(0.0, 2.0 * math.pi)
            weight = rng.uniform(0.45, 1.0) / (k + 1.0) ** 1.80
            self.components.append((cycles, phase, weight))

        samples = 3000
        raw_max = max(abs(self._raw_slope(i * ROW_LENGTH_M / samples))
                      for i in range(samples + 1))
        self.scale = math.tan(self.target_angle_rad) / raw_max if raw_max > 0 else 0.0

    def _raw_y(self, x: float) -> float:
        return sum(w * math.sin(2.0 * math.pi * c * x / ROW_LENGTH_M + p)
                   for c, p, w in self.components)

    def _raw_slope(self, x: float) -> float:
        return sum(w * (2.0 * math.pi * c / ROW_LENGTH_M)
                   * math.cos(2.0 * math.pi * c * x / ROW_LENGTH_M + p)
                   for c, p, w in self.components)

    def y(self, x: float) -> float:
        return self.scale * self._raw_y(x)

    def slope(self, x: float) -> float:
        return self.scale * self._raw_slope(x)

    def angle(self, x: float) -> float:
        return math.atan(self.slope(x))

    def max_excursion(self) -> float:
        n = 1200
        return max(abs(self.y(i * ROW_LENGTH_M / n)) for i in range(n + 1))

    def describe(self) -> list[dict]:
        return [{"cycles": round(c, 4), "phase_rad": round(p, 4),
                 "weight": round(w * self.scale, 5)}
                for c, p, w in self.components]


def row_centre_offsets() -> list[float]:
    mid = (N_ROWS - 1) / 2.0
    return [(i - mid) * ROW_PITCH for i in range(N_ROWS)]


def plant_positions_x(curve: Centreline) -> list[float]:
    xs = [0.0]
    step = 0.001
    x = 0.0
    travelled = 0.0
    while x < ROW_LENGTH_M:
        nxt = x + step
        travelled += math.hypot(step, curve.y(nxt) - curve.y(x))
        if travelled >= PLANT_SPACING_M:
            xs.append(round(nxt, 4))
            travelled = 0.0
        x = nxt
    return xs


def build_plants(curve: Centreline, rng: random.Random) -> list[dict]:
    plants = []
    xs = plant_positions_x(curve)
    for row_i, row_off in enumerate(row_centre_offsets()):
        for j, x in enumerate(xs):
            jx = rng.uniform(-PLANT_JITTER_M, PLANT_JITTER_M)
            jy = rng.uniform(-PLANT_JITTER_M, PLANT_JITTER_M)
            size = PLANT_W * rng.uniform(0.88, 1.12)
            height = PLANT_H * rng.uniform(0.85, 1.15)
            plants.append({
                "name": f"plant_r{row_i}_p{j}",
                "x": round(x + jx, 4),
                "y": round(row_off + curve.y(x) + jy, 4),
                "z": round(height / 2.0, 4),
                "size": round(size, 4),
                "height": round(height, 4),
            })
    return plants


def _random_object(rng: random.Random):
    if rng.random() < 0.5:
        radius = rng.uniform(0.08, 0.18)
        height = rng.uniform(0.30, 0.60)
        return "cylinder", radius, None, height, radius
    side = rng.uniform(0.20, 0.36)
    height = rng.uniform(0.25, 0.50)
    return "box", None, side, height, side / 2.0


def _spaced_positions(n: int, rng: random.Random) -> list[float]:
    x_lo, x_hi = OBST_X_RANGE
    slack = (x_hi - x_lo) - (n - 1) * MIN_OBST_SPACING
    if slack < 0:
        raise RuntimeError(
            f"{n} obstacles need {(n - 1) * MIN_OBST_SPACING:.2f} m but only "
            f"{x_hi - x_lo:.2f} m is available. Widen OBST_X_RANGE or lower the count."
        )

    cuts = sorted(rng.random() for _ in range(n))
    fractions = ([cuts[0]]
                 + [cuts[i] - cuts[i - 1] for i in range(1, n)]
                 + [1.0 - cuts[-1]])

    positions = []
    pos = x_lo
    for i in range(n):
        pos += fractions[i] * slack
        positions.append(pos)
        pos += MIN_OBST_SPACING
    return positions


def build_obstacles(n: int, curve: Centreline, rng: random.Random) -> list[dict]:
    half_aisle = USABLE_HALF_AISLE
    xs = _spaced_positions(n, rng)

    obstacles = []
    for i, x in enumerate(xs):
        kind, radius, side_len, height, half_w = _random_object(rng)

        cos_t = math.cos(curve.angle(x))
        required_lateral = MIN_PASSABLE_GAP / cos_t

        clear_side = rng.choice([-1, 1])
        max_off = half_aisle - half_w
        if clear_side > 0:
            y_off = rng.uniform(-max_off, max_off - required_lateral)
            gap = half_aisle - (y_off + half_w)
        else:
            y_off = rng.uniform(-max_off + required_lateral, max_off)
            gap = (y_off - half_w) + half_aisle

        obstacles.append({
            "name": f"obstacle_{i}",
            "aisle": "middle",
            "kind": kind,
            "x": round(x, 4),
            "y": round(y_off + curve.y(x), 4),
            "z": round(height / 2.0, 4),
            "radius": round(radius, 4) if radius is not None else None,
            "side": round(side_len, 4) if side_len is not None else None,
            "height": round(height, 4),
            "clear_side": "left" if clear_side > 0 else "right",
            "clear_gap_m": round(gap, 4),
            "clear_gap_perp_m": round(gap * cos_t, 4),
            "row_angle_deg": round(math.degrees(curve.angle(x)), 2),
        })
    return obstacles


def build_distractors(n: int, curve: Centreline, rng: random.Random) -> list[dict]:
    offsets = row_centre_offsets()
    side_centres = [(offsets[0] + offsets[1]) / 2.0,
                    (offsets[2] + offsets[3]) / 2.0]
    half_aisle = USABLE_HALF_AISLE

    items = []
    for a_i, centre in enumerate(side_centres):
        for i in range(n):
            kind, radius, side_len, height, half_w = _random_object(rng)
            x = rng.uniform(*OBST_X_RANGE)
            y_off = rng.uniform(-(half_aisle - half_w), half_aisle - half_w)
            items.append({
                "name": f"distractor_a{a_i}_{i}",
                "aisle": "side",
                "kind": kind,
                "x": round(x, 4),
                "y": round(centre + y_off + curve.y(x), 4),
                "z": round(height / 2.0, 4),
                "radius": round(radius, 4) if radius is not None else None,
                "side": round(side_len, 4) if side_len is not None else None,
                "height": round(height, 4),
            })
    return items


def _box_model(name, x, y, z, sx, sy, sz, rgba) -> str:
    return f"""
    <model name="{name}">
      <static>true</static>
      <pose>{x} {y} {z} 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry><box><size>{sx} {sy} {sz}</size></box></geometry>
        </collision>
        <visual name="visual">
          <geometry><box><size>{sx} {sy} {sz}</size></box></geometry>
          <material><ambient>{rgba}</ambient><diffuse>{rgba}</diffuse></material>
        </visual>
      </link>
    </model>"""


def _cylinder_model(name, x, y, z, radius, length, rgba) -> str:
    return f"""
    <model name="{name}">
      <static>true</static>
      <pose>{x} {y} {z} 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry><cylinder><radius>{radius}</radius><length>{length}</length></cylinder></geometry>
        </collision>
        <visual name="visual">
          <geometry><cylinder><radius>{radius}</radius><length>{length}</length></cylinder></geometry>
          <material><ambient>{rgba}</ambient><diffuse>{rgba}</diffuse></material>
        </visual>
      </link>
    </model>"""


def _object_model(o: dict, rgba: str) -> str:
    if o["kind"] == "cylinder":
        return _cylinder_model(o["name"], o["x"], o["y"], o["z"],
                               o["radius"], o["height"], rgba)
    return _box_model(o["name"], o["x"], o["y"], o["z"],
                      o["side"], o["side"], o["height"], rgba)


def build_sdf(world_name, plants, obstacles, distractors, y_extent) -> str:
    parts = []
    for p in plants:
        parts.append(_box_model(p["name"], p["x"], p["y"], p["z"],
                                p["size"], p["size"], p["height"], "0.15 0.45 0.15 1"))
    for o in obstacles:
        parts.append(_object_model(o, "0.85 0.45 0.10 1"))
    for d in distractors:
        parts.append(_object_model(d, "0.55 0.50 0.45 1"))

    x_min, x_max = -WALL_MARGIN_X, ROW_LENGTH_M + WALL_MARGIN_X
    x_span, x_mid = x_max - x_min, (x_max + x_min) / 2.0
    grey = "0.7 0.7 0.72 1"
    parts.append(_box_model("wall_west", x_min, 0, WALL_H / 2, WALL_T, 2 * y_extent, WALL_H, grey))
    parts.append(_box_model("wall_east", x_max, 0, WALL_H / 2, WALL_T, 2 * y_extent, WALL_H, grey))
    parts.append(_box_model("wall_south", x_mid, -y_extent, WALL_H / 2, x_span, WALL_T, WALL_H, grey))
    parts.append(_box_model("wall_north", x_mid, y_extent, WALL_H / 2, x_span, WALL_T, WALL_H, grey))

    models = "\n".join(parts)

    return f"""<?xml version="1.0" ?>
<sdf version="1.10">
  <world name="{world_name}">

    <physics name="step_{int(PHYSICS_STEP_S * 1000)}ms" type="ignored">
      <max_step_size>{PHYSICS_STEP_S}</max_step_size>
      <real_time_factor>1.0</real_time_factor>
    </physics>

    <plugin filename="gz-sim-physics-system"
            name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system"
            name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system"
            name="gz::sim::systems::SceneBroadcaster"/>
    <plugin filename="gz-sim-sensors-system"
            name="gz::sim::systems::Sensors">
      <render_engine>ogre2</render_engine>
    </plugin>
    <plugin filename="gz-sim-imu-system"
            name="gz::sim::systems::Imu"/>
    <!-- Required for robot-mounted contact sensors: this is how collisions
         are counted, rather than a LiDAR minimum-range proxy. -->
    <plugin filename="gz-sim-contact-system"
            name="gz::sim::systems::Contact"/>

    <light type="directional" name="sun">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.9 0.9 0.9 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <direction>-0.4 0.3 -0.9</direction>
    </light>

    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry><plane><normal>0 0 1</normal><size>100 100</size></plane></geometry>
        </collision>
        <visual name="visual">
          <geometry><plane><normal>0 0 1</normal><size>100 100</size></plane></geometry>
          <material><ambient>0.35 0.32 0.28 1</ambient><diffuse>0.35 0.32 0.28 1</diffuse></material>
        </visual>
      </link>
    </model>
{models}

  </world>
</sdf>
"""


def make_world(curvature: str, density: str, seed: int) -> tuple[str, str, dict]:
    if curvature not in TARGET_MAX_ANGLE_DEG:
        raise ValueError(f"curvature must be one of {list(TARGET_MAX_ANGLE_DEG)}")
    if density not in DENSITY_COUNT:
        raise ValueError(f"density must be one of {list(DENSITY_COUNT)}")

    n_obst = DENSITY_COUNT[density]

    rng = random.Random(f"{curvature}|{density}|{seed}")

    curve = Centreline(curvature, rng)
    plants = build_plants(curve, rng)
    obstacles = build_obstacles(n_obst, curve, rng)
    distractors = build_distractors(n_obst, curve, rng)

    outer = (N_ROWS - 1) / 2.0 * ROW_PITCH
    y_extent = outer + curve.max_excursion() + WALL_MARGIN_Y

    world_name = f"greenhouse_{curvature}_{density}_s{seed:02d}"
    sdf = build_sdf(world_name, plants, obstacles, distractors, y_extent)

    start_y, goal_y = curve.y(START_X), curve.y(GOAL_X)

    heading = curve.angle(START_X)

    n = 1200
    angles = [abs(math.degrees(curve.angle(i * ROW_LENGTH_M / n))) for i in range(n + 1)]

    meta = {
        "world_name": world_name,
        "curvature": curvature,
        "density": density,
        "seed": seed,
        "cell_id": f"{curvature}_{density}",
        "held_out": (curvature, density) in HELD_OUT_CELLS,

        "target_max_angle_deg": TARGET_MAX_ANGLE_DEG[curvature],
        "actual_max_angle_deg": round(max(angles), 2),
        "mean_abs_angle_deg": round(sum(angles) / len(angles), 2),
        "max_lateral_excursion_m": round(curve.max_excursion(), 4),
        "centreline_harmonics": curve.describe(),

        "n_obstacles": n_obst,
        "n_distractors": len(distractors),
        "n_plants": len(plants),
        "plants_per_row": len(plants) // N_ROWS,

        "aisle_width_m": AISLE_W,
        "min_aisle_width_perp_m": round(AISLE_W * math.cos(math.radians(max(angles))), 4),
        "row_length_m": ROW_LENGTH_M,
        "n_rows": N_ROWS,
        "plant_spacing_arc_m": PLANT_SPACING_M,
        "plant_jitter_m": PLANT_JITTER_M,

        "start_pose": {"x": START_X, "y": round(start_y, 4), "yaw": round(heading, 4)},
        "goal_pose": {"x": GOAL_X, "y": round(goal_y, 4)},
        "straight_line_distance_m": round(math.hypot(GOAL_X - START_X, goal_y - start_y), 4),

        "timeout_s": TIMEOUT_S,
        "physics_step_s": PHYSICS_STEP_S,
        "obstacles": obstacles,
        "distractors": distractors,
    }
    return world_name, sdf, meta


def write_world(curvature: str, density: str, seed: int, out_dir: Path) -> Path:
    world_name, sdf, meta = make_world(curvature, density, seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    sdf_path = out_dir / f"{world_name}.sdf"
    sdf_path.write_text(sdf)
    (out_dir / f"{world_name}.json").write_text(json.dumps(meta, indent=2))
    return sdf_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate greenhouse worlds for Gazebo Harmonic.")
    ap.add_argument("--curvature", choices=list(TARGET_MAX_ANGLE_DEG), default="straight")
    ap.add_argument("--density", choices=list(DENSITY_COUNT), default="sparse")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--all", action="store_true", help="generate every cell of the 3x3 grid")
    ap.add_argument("--seeds", type=int, default=30, help="with --all: seeds per cell")
    ap.add_argument("--out", type=Path,
                    default=Path.home() / "greenhouse_ws" / "worlds" / "generated")
    ap.add_argument("--dry-run", action="store_true", help="summarise without writing files")
    args = ap.parse_args()

    if args.all:
        count = 0
        for curvature in TARGET_MAX_ANGLE_DEG:
            for density in DENSITY_COUNT:
                for seed in range(args.seeds):
                    if args.dry_run:
                        make_world(curvature, density, seed)
                    else:
                        write_world(curvature, density, seed, args.out)
                    count += 1
        verb = "would generate" if args.dry_run else "wrote"
        print(f"{verb} {count} worlds "
              f"({len(TARGET_MAX_ANGLE_DEG)} curvature x {len(DENSITY_COUNT)} density "
              f"x {args.seeds} seeds) -> {args.out}")
        return

    world_name, sdf, meta = make_world(args.curvature, args.density, args.seed)

    print(f"world            : {world_name}")
    print(f"held out (RQ3)   : {meta['held_out']}")
    print(f"max row angle    : {meta['actual_max_angle_deg']} deg "
          f"(target {meta['target_max_angle_deg']})")
    print(f"mean row angle   : {meta['mean_abs_angle_deg']} deg")
    print(f"lateral swing    : {meta['max_lateral_excursion_m']} m")
    print(f"aisle, perp min  : {meta['min_aisle_width_perp_m']} m")
    print(f"plants per row   : {meta['plants_per_row']}")
    print(f"obstacles/distr  : {meta['n_obstacles']} / {meta['n_distractors']}")
    print(f"start            : x={meta['start_pose']['x']} "
          f"y={meta['start_pose']['y']} yaw={meta['start_pose']['yaw']}")
    print(f"goal             : x={meta['goal_pose']['x']} y={meta['goal_pose']['y']}")
    for o in meta["obstacles"]:
        print(f"  {o['name']:<12} {o['kind']:<9} x={o['x']:>6.2f} y={o['y']:>6.2f}"
              f"  row {o['row_angle_deg']:>6.1f} deg  perp gap {o['clear_gap_perp_m']:.2f} m")

    if args.dry_run:
        print("\n(dry run - nothing written)")
        return

    path = write_world(args.curvature, args.density, args.seed, args.out)
    print(f"\nwrote {path}")
    print(f"wrote {path.with_suffix('.json')}")
    print(f"\nopen it with:\n  gz sim -v 4 {path}")


if __name__ == "__main__":
    main()
