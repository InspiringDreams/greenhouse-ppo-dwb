#!/usr/bin/env python3
"""Gymnasium environment used to train and test the PPO policy in Gazebo."""

import itertools
import json
import math
import os
import re
import subprocess
import time
from pathlib import Path

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from tf2_msgs.msg import TFMessage

WS = Path.home() / 'greenhouse_ws'

N_SECTORS = 36
LIDAR_MAX = 3.5
MAX_LIN = 0.26
MAX_ANG = 1.82
CONTROL_HZ = 10.0
GOAL_RADIUS = 0.25

STALL_LIMIT_S = 20.0
STALL_PROGRESS_M = 0.05
COLLISION_RANGE = 0.25
TRUE_STALL_S = 45.0
TRUE_STALL_M = 0.10

W_PROGRESS = 8.0
W_STEP = -0.02
W_SPIN = -0.02
R_GOAL = 200.0
R_COLLISION = -200.0
R_TIMEOUT = -100.0
CLEAR_START = 0.25
W_CLEAR = -0.10

FOOT_X, FOOT_HX, FOOT_HY = -0.064, 0.1330, 0.1530

DF_RES = 0.05
DF_PAD = 0.10
DF_SMEAR = 8


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def _pt_to_aabb(dx, dy, hx, hy):
    return math.hypot(max(abs(dx) - hx, 0.0), max(abs(dy) - hy, 0.0))

_node_seq = itertools.count()


class RosLink(Node):

    def __init__(self):
        super().__init__(f'ppo_env_{next(_node_seq)}')
        self.scan = None
        self.odom = None
        self.truth = None
        self.scan_count = 0
        self.create_subscription(LaserScan, '/scan', self._on_scan, 1)
        self.create_subscription(Odometry, '/odom', self._on_odom, 1)
        self.create_subscription(TFMessage, '/truth_tf', self._on_truth, 1)
        self.cmd = self.create_publisher(Twist, '/cmd_vel', 1)

    def _on_scan(self, msg):
        self.scan = msg
        self.scan_count += 1

    def _on_odom(self, msg):
        self.odom = msg

    def _on_truth(self, msg):
        if msg.transforms:
            t = msg.transforms[0].transform
            self.truth = (t.translation.x, t.translation.y,
                          yaw_of(t.rotation))

    def publish(self, lin, ang):
        t = Twist()
        t.linear.x = float(lin)
        t.angular.z = float(ang)
        self.cmd.publish(t)

    def wait_for_new_scan(self, timeout=5.0):
        target = self.scan_count + 1
        deadline = time.time() + timeout
        while self.scan_count < target and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
        return self.scan_count >= target


class GreenhouseEnv(gym.Env):

    def __init__(self, world_name, worlds_dir=WS / 'worlds' / 'generated',
                 episode_limit_s=240.0, start_x=None,
                 terminate_on_collision=True, success_from_truth=True):
        super().__init__()
        self.world_name = world_name
        self._assert_world()
        self.terminate_on_collision = terminate_on_collision
        self.success_from_truth = success_from_truth
        meta = json.loads((Path(worlds_dir) / f'{world_name}.json').read_text())
        self.meta = meta
        self.row_length = meta['row_length_m']
        self.harmonics = meta['centreline_harmonics']
        self.goal = (meta['goal_pose']['x'], meta['goal_pose']['y'])
        self.default_start_x = meta['start_pose']['x']
        from astar_ground_truth import read_footprints
        self.shapes = read_footprints(Path(worlds_dir) / f'{world_name}.sdf')
        self.start_x = self.default_start_x if start_x is None else float(start_x)

        self.geo_fallbacks = 0
        self._build_distance_field()
        if self.dfield is None:
            raise RuntimeError(
                f'{world_name}: could not build the geodesic distance field. '
                'Refusing to run: the straight-line fallback would undo the '
                'progress-reward fix with nothing to show that it had.')

        try:
            rclpy.init()
        except Exception:
            pass
        self.ros = RosLink()

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,),
                                       dtype=np.float32)
        self.observation_space = spaces.Box(low=-1.0, high=1.0,
                                            shape=(N_SECTORS + 4,),
                                            dtype=np.float32)
        self.max_steps = int(episode_limit_s * CONTROL_HZ)

        self.steps = 0
        self.last_action = np.zeros(2, dtype=np.float32)
        self.prev_dist = None
        self.origin = None
        self.goal_rel = None


    def _assert_world(self, attempts=3):
        seen = []
        for i in range(attempts):
            try:
                out = subprocess.run(['gz', 'topic', '-l'],
                                     capture_output=True, text=True,
                                     timeout=20).stdout
            except Exception as e:
                raise RuntimeError(f'cannot list Gazebo topics: {e}')
            seen = sorted(set(re.findall(r'/world/([^/]+)/', out)))
            if seen:
                break
            time.sleep(2)

        part = os.environ.get('GZ_PARTITION', '(unset)')
        dom = os.environ.get('ROS_DOMAIN_ID', '(unset)')
        if not seen:
            raise RuntimeError(
                f'no Gazebo world is running on GZ_PARTITION={part} '
                f'(ROS_DOMAIN_ID={dom}). Launch the simulator first and wait '
                f'for its "simulator up" line.')
        if seen != [self.world_name]:
            raise RuntimeError(
                f'the simulator on GZ_PARTITION={part} is running {seen}, '
                f'not [{self.world_name}]. A leftover gz sim is holding this '
                f'partition - kill it first:\n'
                f'    pkill -9 -f "gz sim"; pkill -9 -f parameter_bridge; '
                f'sleep 4')


    def _build_distance_field(self):
        import heapq
        from astar_ground_truth import build_grid

        self.dfield = None
        try:
            grid, x0, y0 = build_grid(self.shapes, pad=DF_PAD)
        except Exception:
            return
        ny, nx = grid.shape
        gx = int(round((self.goal[0] - x0) / DF_RES))
        gy = int(round((self.goal[1] - y0) / DF_RES))
        if not (0 <= gx < nx and 0 <= gy < ny):
            return

        INF = np.inf
        dist = np.full((ny, nx), INF, dtype=np.float64)
        s2 = math.sqrt(2.0)
        nbrs = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
                (1, 1, s2), (1, -1, s2), (-1, 1, s2), (-1, -1, s2)]

        iy_, ix_ = np.nonzero(~grid)
        if len(ix_) == 0:
            return
        d2 = (ix_ - gx) ** 2 + (iy_ - gy) ** 2
        k = int(np.argmin(d2))
        off = math.sqrt(float(d2[k])) * DF_RES
        if off > 1.0:
            return
        sx_, sy_ = int(ix_[k]), int(iy_[k])
        dist[sy_, sx_] = off
        pq = [(off, sx_, sy_)]
        self.df_goal_offset = off
        while pq:
            d, ix, iy = heapq.heappop(pq)
            if d > dist[iy, ix]:
                continue
            for dx, dy, stp in nbrs:
                jx, jy = ix + dx, iy + dy
                if not (0 <= jx < nx and 0 <= jy < ny) or grid[jy, jx]:
                    continue
                nd = d + stp * DF_RES
                if nd < dist[jy, jx]:
                    dist[jy, jx] = nd
                    heapq.heappush(pq, (nd, jx, jy))

        if not np.isfinite(dist).any():
            return

        for _ in range(DF_SMEAR):
            nb = np.full_like(dist, INF)
            nb[1:, :] = np.minimum(nb[1:, :], dist[:-1, :] + DF_RES)
            nb[:-1, :] = np.minimum(nb[:-1, :], dist[1:, :] + DF_RES)
            nb[:, 1:] = np.minimum(nb[:, 1:], dist[:, :-1] + DF_RES)
            nb[:, :-1] = np.minimum(nb[:, :-1], dist[:, 1:] + DF_RES)
            dist = np.minimum(dist, nb)

        self.dfield = dist
        self.dfx0 = x0
        self.dfy0 = y0

    def _geo_dist(self, x, y):
        euclid = math.hypot(self.goal[0] - x, self.goal[1] - y)
        if self.dfield is None:
            self.geo_fallbacks += 1
            return euclid
        f = self.dfield
        ny, nx = f.shape
        fx = (x - self.dfx0) / DF_RES
        fy = (y - self.dfy0) / DF_RES

        ix, iy = int(math.floor(fx)), int(math.floor(fy))
        if 0 <= ix < nx - 1 and 0 <= iy < ny - 1:
            v00, v10 = float(f[iy, ix]), float(f[iy, ix + 1])
            v01, v11 = float(f[iy + 1, ix]), float(f[iy + 1, ix + 1])
            if all(math.isfinite(v) for v in (v00, v10, v01, v11)):
                tx, ty = fx - ix, fy - iy
                return (v00 * (1 - tx) * (1 - ty) + v10 * tx * (1 - ty)
                        + v01 * (1 - tx) * ty + v11 * tx * ty)

        jx = int(round(fx))
        jy = int(round(fy))
        if not (0 <= jx < nx and 0 <= jy < ny):
            self.geo_fallbacks += 1
            return euclid
        v = float(f[jy, jx])
        if not math.isfinite(v):
            self.geo_fallbacks += 1
            return euclid
        return v


    def _centre_y(self, x):
        return sum(h['weight'] * math.sin(2 * math.pi * h['cycles'] * x
                                          / self.row_length + h['phase_rad'])
                   for h in self.harmonics)

    def _centre_slope(self, x):
        return sum(h['weight'] * (2 * math.pi * h['cycles'] / self.row_length)
                   * math.cos(2 * math.pi * h['cycles'] * x / self.row_length
                              + h['phase_rad'])
                   for h in self.harmonics)

    def _clear_at(self, x):
        y = self._centre_y(x)
        best = 99.0
        for o in self.meta.get('obstacles', []):
            half = o['radius'] if o.get('radius') else (o.get('side') or 0) / 2.0
            d = math.hypot(o['x'] - x, o['y'] - y) - half
            best = min(best, d)
        return best

    def _safe_start_x(self, want, need=0.5):
        if self._clear_at(want) >= need:
            return want
        for step in [0.25 * k for k in range(1, 13)]:
            for cand in (want + step, want - step):
                if 0.5 <= cand <= self.goal[0] - 0.6 and self._clear_at(cand) >= need:
                    return cand
        return want

    def _start_pose(self):
        x = self._safe_start_x(self.start_x)
        return x, self._centre_y(x), math.atan(self._centre_slope(x))


    def _teleport(self):
        x, y, yaw = self._start_pose()
        req = (f'name: "waffle", position: {{x: {x}, y: {y}, z: 0.02}}, '
               f'orientation: {{z: {math.sin(yaw / 2)}, w: {math.cos(yaw / 2)}}}')
        try:
            subprocess.run(
                ['gz', 'service', '-s', f'/world/{self.world_name}/set_pose',
                 '--reqtype', 'gz.msgs.Pose', '--reptype', 'gz.msgs.Boolean',
                 '--timeout', '2000', '--req', req],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
        except subprocess.TimeoutExpired:
            raise RuntimeError('gz service did not answer - simulator running?')

    def _pose(self):
        o = self.ros.odom.pose.pose
        x, y, yaw = o.position.x, o.position.y, yaw_of(o.orientation)
        if self.origin is None:
            return 0.0, 0.0, 0.0
        x0, y0, yaw0 = self.origin
        dx, dy = x - x0, y - y0
        c, s = math.cos(yaw0), math.sin(yaw0)
        return dx * c + dy * s, -dx * s + dy * c, wrap(yaw - yaw0)

    def _truth_goal_dist(self):
        if self.ros.truth is None:
            return None
        tx, ty, _ = self.ros.truth
        return math.hypot(self.goal[0] - tx, self.goal[1] - ty)

    def _query_truth(self):
        for attempt in range(4):
            try:
                out = subprocess.run(['gz', 'model', '-m', 'waffle', '-p'],
                                     capture_output=True, text=True,
                                     timeout=15).stdout
            except Exception:
                out = ''
            t = re.findall(r'\[\s*(-?[\d.eE+-]+)\s+(-?[\d.eE+-]+)\s+'
                           r'(-?[\d.eE+-]+)\s*\]', out)
            if len(t) >= 2:
                return (float(t[0][0]), float(t[0][1]), float(t[1][2]))
            time.sleep(1.0 + attempt)
        return None

    def _await_truth(self, timeout=1.5):
        deadline = time.time() + timeout
        while self.ros.truth is None and time.time() < deadline:
            rclpy.spin_once(self.ros, timeout_sec=0.05)
        if self.ros.truth is None:
            for _ in range(8):
                self.ros.publish(0.02, 0.0)
                rclpy.spin_once(self.ros, timeout_sec=0.05)
                if self.ros.truth is not None:
                    break
            self.ros.publish(0.0, 0.0)
        return self.ros.truth

    def _true_gap(self):
        if self.ros.truth is None:
            return None
        px, py, yaw = self.ros.truth
        c, s = math.cos(yaw), math.sin(yaw)

        corners = []
        for lx in (FOOT_X - FOOT_HX, FOOT_X + FOOT_HX):
            for ly in (-FOOT_HY, FOOT_HY):
                corners.append((px + lx * c - ly * s, py + lx * s + ly * c))

        best = float('inf')
        for sh in self.shapes:
            if sh['kind'] == 'cyl':
                dx, dy = sh['x'] - px, sh['y'] - py
                rx = dx * c + dy * s
                ry = -dx * s + dy * c
                best = min(best, _pt_to_aabb(rx - FOOT_X, ry,
                                             FOOT_HX, FOOT_HY) - sh['r'])
                continue

            hx, hy = sh['sx'] / 2.0, sh['sy'] / 2.0
            d = float('inf')
            for (wx, wy) in corners:
                d = min(d, _pt_to_aabb(wx - sh['x'], wy - sh['y'], hx, hy))
            for ox in (sh['x'] - hx, sh['x'] + hx):
                for oy in (sh['y'] - hy, sh['y'] + hy):
                    dx, dy = ox - px, oy - py
                    rx = dx * c + dy * s
                    ry = -dx * s + dy * c
                    d = min(d, _pt_to_aabb(rx - FOOT_X, ry, FOOT_HX, FOOT_HY))
            best = min(best, d)
        return best

    def _observe(self):
        r = np.array(self.ros.scan.ranges, dtype=np.float32)
        r = np.nan_to_num(r, nan=LIDAR_MAX, posinf=LIDAR_MAX)
        r = np.clip(r, 0.0, LIDAR_MAX)
        sectors = np.min(r[:len(r) // N_SECTORS * N_SECTORS]
                         .reshape(N_SECTORS, -1), axis=1) / LIDAR_MAX

        x, y, yaw = self._pose()
        gx, gy = self.goal_rel
        dist = math.hypot(gx - x, gy - y)
        bearing = wrap(math.atan2(gy - y, gx - x) - yaw)

        obs = np.concatenate([
            sectors,
            [min(dist / 12.0, 1.0), bearing / math.pi],
            [self.last_action[0] / MAX_LIN, self.last_action[1] / MAX_ANG],
        ]).astype(np.float32)
        return obs, dist, float(sectors.min() * LIDAR_MAX)

    def _wait_for_data(self, timeout=60.0):
        deadline = time.time() + timeout
        while (self.ros.scan is None or self.ros.odom is None) and time.time() < deadline:
            rclpy.spin_once(self.ros, timeout_sec=0.1)
        if self.ros.scan is None or self.ros.odom is None:
            raise RuntimeError(f'no scan or odom within {timeout:.0f}s')


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.ros.truth = None
        self._wait_for_data()
        self.ros.publish(0.0, 0.0)
        time.sleep(0.2)
        self._teleport()
        time.sleep(0.5)
        for _ in range(5):
            self.ros.wait_for_new_scan()
        rclpy.spin_once(self.ros, timeout_sec=0.2)

        o = self.ros.odom.pose.pose
        self.origin = (o.position.x, o.position.y, yaw_of(o.orientation))

        sx, sy, syaw = self._start_pose()
        dx, dy = self.goal[0] - sx, self.goal[1] - sy
        c, s = math.cos(syaw), math.sin(syaw)
        self.goal_rel = (dx * c + dy * s, -dx * s + dy * c)

        if self.success_from_truth:
            if self.ros.truth is None:
                self._await_truth(1.5)
            if self.ros.truth is None:
                self.ros.truth = self._query_truth()
            for _ in range(2):
                if self.ros.truth is not None:
                    break
                self._teleport()
                time.sleep(1.0)
                self.ros.wait_for_new_scan()
                self.ros.truth = self._await_truth(1.5) or self._query_truth()
            if self.ros.truth is None:
                raise RuntimeError('cannot read ground-truth pose from Gazebo '
                                   'after 3 attempts')
            sx, sy, _ = self._start_pose()
            tx, ty, _ = self.ros.truth
            if math.hypot(tx - sx, ty - sy) > 0.5:
                raise RuntimeError(
                    f'ground truth ({tx:.2f},{ty:.2f}) does not match spawn '
                    f'({sx:.2f},{sy:.2f}) - wrong entity in /truth_tf')

        self.steps = 0
        self.stalled_steps = 0
        self.truth_anchor = None
        self.truth_stall_steps = 0
        self.last_action = np.zeros(2, dtype=np.float32)
        obs, dist, nearest = self._observe()
        td0 = self._truth_goal_dist()
        gap0 = self._true_gap()
        if td0 is None or gap0 is None:
            raise RuntimeError('no ground-truth pose on /truth_tf at reset')
        if gap0 <= 0.0:
            raise RuntimeError(f'episode would start in collision '
                               f'(gap {gap0:.3f} m at x={self.start_x})')
        tx, ty, _ = self.ros.truth
        before = self.geo_fallbacks
        gd0 = self._geo_dist(tx, ty)
        if self.geo_fallbacks > before:
            raise RuntimeError(
                f'start ({tx:.2f},{ty:.2f}) is not on the geodesic field, so '
                f'progress would be shaped by straight-line distance - the '
                f'very thing the field replaces')
        self.best_dist = gd0
        self.prev_dist = gd0
        return obs, {}

    def step(self, action):
        a = np.clip(action, -1.0, 1.0)
        lin = float(a[0]) * MAX_LIN
        ang = float(a[1] * MAX_ANG)
        self.ros.publish(lin, ang)
        if not self.ros.wait_for_new_scan():
            raise RuntimeError('no new LiDAR scan within 5 s - simulator stalled')
        self.last_action = np.array([lin, ang], dtype=np.float32)
        self.steps += 1

        obs, dist, nearest = self._observe()

        td = self._truth_goal_dist()
        gap = self._true_gap()
        if td is None or gap is None:
            raise RuntimeError('no ground-truth pose on /truth_tf')
        tx, ty, _ = self.ros.truth
        gd = self._geo_dist(tx, ty)

        reward = (W_PROGRESS * (self.prev_dist - gd)
                  + W_STEP
                  + W_SPIN * abs(ang)
                  + W_CLEAR * max(0.0, (CLEAR_START - gap) / CLEAR_START))
        self.prev_dist = gd

        if gd < self.best_dist - STALL_PROGRESS_M:
            self.best_dist = gd
            self.stalled_steps = 0
        else:
            self.stalled_steps += 1

        terminated = False
        info = {'distance': dist, 'nearest': nearest, 'start_x': self.start_x,
                'truth_dist': td, 'geo_dist': gd, 'true_gap': gap,
                'truth_x': tx, 'truth_y': ty,
                'geo_fallbacks': self.geo_fallbacks}

        if td < GOAL_RADIUS:
            reward += R_GOAL
            terminated = True
            info['outcome'] = 'success'
        elif gap <= 0.0:
            reward += R_COLLISION
            if self.terminate_on_collision:
                terminated = True
                info['outcome'] = 'collision'
            else:
                self.eval_contacts = getattr(self, 'eval_contacts', 0) + 1

        true_stalled = False
        if self.ros.truth is not None:
            if (self.truth_anchor is None
                    or math.dist((tx, ty), self.truth_anchor) > TRUE_STALL_M):
                self.truth_anchor = (tx, ty)
                self.truth_stall_steps = 0
            else:
                self.truth_stall_steps += 1
            true_stalled = (self.truth_stall_steps
                            >= int(TRUE_STALL_S * CONTROL_HZ))
        info['true_stalled'] = true_stalled

        stalled = (self.stalled_steps >= int(STALL_LIMIT_S * CONTROL_HZ)
                   or true_stalled)
        truncated = self.steps >= self.max_steps or stalled
        if truncated and not terminated:
            info['outcome'] = 'stalled' if stalled else 'timeout'
            reward += R_TIMEOUT + W_STEP * max(0, self.max_steps - self.steps)

        return obs, float(reward), terminated, truncated, info

    def close(self):
        try:
            self.ros.publish(0.0, 0.0)
            self.ros.destroy_node()
        except Exception:
            pass
