#!/usr/bin/env python3
"""Shared helpers for running one episode: start/stop the simulator, wait for data, send a Nav2 goal."""

import re
import os
import signal
import subprocess
import time
from pathlib import Path


WS = Path.home() / 'greenhouse_ws'
SETUP = f'source /opt/ros/jazzy/setup.bash && source {WS}/install/setup.bash'


def sh(cmd, **kw):
    return subprocess.Popen(['bash', '-c', f'{SETUP} && {cmd}'],
                            start_new_session=True, **kw)


def kill_tree(p, grab=False):
    try:
        os.killpg(os.getpgid(p.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        p.kill()
    try:
        out, _ = p.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        out = b''
    return out if grab else b''


def run_capture(cmd, timeout=15):
    p = sh(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    try:
        out, _ = p.communicate(timeout=timeout)
        return out.decode(errors='replace')
    except subprocess.TimeoutExpired:
        kill_tree(p)
        return ''


def kill_all():
    for pat in ['gz sim', 'parameter_bridge', 'robot_state_publisher',
                'controller_server', 'planner_server', 'behavior_server',
                'bt_navigator', 'lifecycle_manager', 'metrics_node']:
        subprocess.run(['pkill', '-f', pat], stderr=subprocess.DEVNULL)
    subprocess.run(['bash', '-c', f'{SETUP} && ros2 daemon stop'],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(3)
    for pat in ['gz sim', 'parameter_bridge', 'metrics_node']:
        subprocess.run(['pkill', '-9', '-f', pat], stderr=subprocess.DEVNULL)
    time.sleep(2)


def wait_for_data(topic, timeout=60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if run_capture(f'timeout 5 ros2 topic echo {topic} --once', timeout=10).strip():
            return True
        time.sleep(1)
    return False


def wait_for_nav2(timeout=90):
    deadline = time.time() + timeout
    while time.time() < deadline:
        out = run_capture('ros2 service call /bt_navigator/get_state '
                          'lifecycle_msgs/srv/GetState "{}"', timeout=20)
        if "label='active'" in out or 'id=3' in out:
            return True
        time.sleep(2)
    return False


def parse_feedback(text):
    recoveries = re.findall(r'number_of_recoveries: (\d+)', text)
    secs = re.findall(r'navigation_time:\s*\n\s*sec: (\d+)\s*\n\s*nanosec: (\d+)', text)
    out = {}
    if recoveries:
        out['recovery_count'] = int(recoveries[-1])
    if secs:
        s, ns = secs[-1]
        out['time_to_goal_s'] = round(int(s) + int(ns) / 1e9, 3)
    return out


def send_goal(gx, gy, timeout_s):
    goal = (f'{{pose: {{header: {{frame_id: odom}}, pose: {{'
            f'position: {{x: {gx:.4f}, y: {gy:.4f}, z: 0.0}}, '
            f'orientation: {{w: 1.0}}}}}}}}')
    text, wall, timed_out = '', 0.0, False
    for attempt in range(3):
        t0 = time.time()
        p = sh(f'ros2 action send_goal /navigate_to_pose '
               f'nav2_msgs/action/NavigateToPose "{goal}" --feedback',
               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        try:
            out, _ = p.communicate(timeout=timeout_s * 2)
            text = out.decode(errors='replace')
            timed_out = False
        except subprocess.TimeoutExpired:
            text = kill_tree(p, grab=True).decode(errors='replace')
            timed_out = True
        wall = time.time() - t0
        if 'rejected' in text.lower() and attempt < 2:
            print(f'    goal rejected, retry {attempt + 1}/2')
            time.sleep(8)
            continue

        fb = parse_feedback(text)
        if ('ABORTED' in text and fb.get('time_to_goal_s', 99) < 1.0
                and attempt < 2):
            print(f'    instant abort ({fb.get("time_to_goal_s")}s), '
                  f'retry {attempt + 1}/2')
            time.sleep(8)
            continue
        break
    return text, wall, timed_out
