#!/usr/bin/env python3
"""Converts the goal from world coordinates into the odometry frame the robot starts in."""

import json
import math
from pathlib import Path


def world_to_odom(point, start):
    dx = point[0] - start['x']
    dy = point[1] - start['y']
    yaw = start['yaw']
    x = dx * math.cos(yaw) + dy * math.sin(yaw)
    y = -dx * math.sin(yaw) + dy * math.cos(yaw)
    return x, y


def goal_in_odom(meta_path):
    meta = json.loads(Path(meta_path).read_text())
    g = meta['goal_pose']
    return world_to_odom((g['x'], g['y']), meta['start_pose'])


if __name__ == '__main__':
    import sys
    x, y = goal_in_odom(sys.argv[1])
    print(f"{x:.4f} {y:.4f}")
