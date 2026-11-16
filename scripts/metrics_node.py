#!/usr/bin/env python3
"""Records the robot path and episode metrics while an episode runs."""

import json
import math
import sys

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from tf2_msgs.msg import TFMessage

COLLISION_RANGE = 0.25
COLLISION_CLEAR = 0.35
DEBOUNCE_S = 1.0
SCAN_PERIOD_S = 0.1


class Metrics(Node):

    def __init__(self, out_path, traj_path=None):
        super().__init__('metrics')
        self.out_path = out_path
        self.traj_path = traj_path
        self.traj = []
        self.traj_last_t = None

        self.path_length = 0.0
        self.last = None
        self.steps = 0

        self.min_clearance = float('inf')
        self.clearance_sum = 0.0
        self.scan_count = 0

        self.collisions = 0
        self.in_collision = False
        self.contact_time = 0.0
        self.last_collision_t = -999.0

        self.lin_sum = 0.0
        self.ang_sum = 0.0

        self.truth = None
        self.truth_path = 0.0
        self.truth_last = None

        self.create_subscription(Odometry, '/odom', self.on_odom, 10)
        self.create_subscription(LaserScan, '/scan', self.on_scan, 10)
        self.create_subscription(TFMessage, '/truth_tf', self.on_truth, 10)

    def on_truth(self, msg):
        if not msg.transforms:
            return
        t = msg.transforms[0].transform
        x, y = t.translation.x, t.translation.y
        q = t.rotation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        if self.truth_last is not None:
            self.truth_path += math.dist((x, y), self.truth_last)
        self.truth_last = (x, y)
        self.truth = (x, y, yaw)

    def on_odom(self, msg):
        p = msg.pose.pose.position
        if self.last is not None:
            self.path_length += math.dist((p.x, p.y), self.last)
        self.last = (p.x, p.y)

        if self.traj_path is not None:
            t = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
            if self.traj_last_t is None or t - self.traj_last_t >= 0.1:
                q = msg.pose.pose.orientation
                yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                                 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
                tr = self.truth or (float('nan'),) * 3
                self.traj.append((t, p.x, p.y, yaw, tr[0], tr[1], tr[2]))
                self.traj_last_t = t

        self.lin_sum += abs(msg.twist.twist.linear.x)
        self.ang_sum += abs(msg.twist.twist.angular.z)
        self.steps += 1

    def on_scan(self, msg):
        floor = msg.range_min + 0.02
        valid = [r for r in msg.ranges
                 if floor < r < msg.range_max
                 and not math.isinf(r) and not math.isnan(r)]
        if not valid:
            return

        lo = min(valid)
        self.min_clearance = min(self.min_clearance, lo)
        self.clearance_sum += lo
        self.scan_count += 1

        now = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9

        if lo < COLLISION_RANGE:
            self.contact_time += SCAN_PERIOD_S
            if not self.in_collision and (now - self.last_collision_t) > DEBOUNCE_S:
                self.collisions += 1
                self.last_collision_t = now
                self.in_collision = True
        elif lo > COLLISION_CLEAR:
            self.in_collision = False

    def result(self):
        n = max(self.steps, 1)
        s = max(self.scan_count, 1)
        return {
            'path_length_m': round(self.path_length, 4),
            'collisions': self.collisions,
            'contact_time_s': round(self.contact_time, 2),
            'min_clearance_m': round(self.min_clearance, 4)
                               if self.min_clearance != float('inf') else None,
            'mean_clearance_m': round(self.clearance_sum / s, 4),
            'mean_linear_vel': round(self.lin_sum / n, 4),
            'mean_abs_angular_vel': round(self.ang_sum / n, 4),
            'odom_samples': self.steps,
            'path_length_truth_m': round(self.truth_path, 4),
            'truth_final_x': round(self.truth[0], 4) if self.truth else None,
            'truth_final_y': round(self.truth[1], 4) if self.truth else None,
        }

    def write(self):
        with open(self.out_path, 'w') as f:
            json.dump(self.result(), f, indent=2)
        if self.traj_path is not None:
            with open(self.traj_path, 'w') as f:
                f.write('t,x,y,yaw,tx,ty,tyaw\n')
                for t, x, y, yaw, tx, ty, tyaw in self.traj:
                    f.write(f'{t:.3f},{x:.4f},{y:.4f},{yaw:.4f},'
                            f'{tx:.4f},{ty:.4f},{tyaw:.4f}\n')


def main():
    out_path = sys.argv[1]
    traj_path = sys.argv[2] if len(sys.argv) > 2 else None
    rclpy.init()
    node = Metrics(out_path, traj_path)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.write()
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
