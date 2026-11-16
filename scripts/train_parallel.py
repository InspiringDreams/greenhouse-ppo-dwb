#!/usr/bin/env python3
"""Trains the PPO policy with a curriculum, using several simulators at once."""

import argparse
import json
import os
import random
import subprocess
import threading
import time
from pathlib import Path

WS = Path.home() / 'greenhouse_ws'
TRAIN_WORLDS = WS / 'worlds' / 'training'
OUT = WS / 'models'
LOGS = WS / 'runs' / 'training'
BLOCKS = LOGS / 'blocks'

LEVELS = [11.0, 10.0, 8.5, 7.0, 5.5, 4.0, 2.5, 1.8, 1.2, 0.5]
ADVANCE_RATE = 0.70
REGRESS_RATE = 0.35
REGRESS_AFTER = 60_000
WINDOW = 100
LEVEL_CAP = 150_000

EPISODE_LIMIT_S = 240.0

PPO_KWARGS = dict(
    learning_rate=3e-4,
    n_steps=1024,
    batch_size=256,
    n_epochs=10,
    gamma=0.998,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.005,
    vf_coef=0.5,
    max_grad_norm=0.5,
)
POLICY_KWARGS = dict(log_std_init=-1.0, net_arch=[256, 256])


def worlds_for_level(level, worlds):
    if level <= 1:
        return [w for w in worlds if '_sparse_' in w]
    if level <= 3:
        return [w for w in worlds if '_cluttered_' not in w]
    return worlds


def pick_worlds(level, worlds, n, rng):
    pool = worlds_for_level(level, worlds)
    buckets = [b for b in ([w for w in pool if c in w]
                           for c in ('_sparse_', '_medium_', '_cluttered_')) if b]
    return [rng.choice(buckets[i % len(buckets)]) for i in range(n)]


def _safe_close(venv):
    try:
        venv.close()
    except Exception:
        pass


def _hard_kill(venv):
    inner = getattr(venv, 'venv', venv)
    for p in getattr(inner, 'processes', []):
        try:
            p.terminate()
        except Exception:
            pass


def kill_sims():
    for pat in ['gz sim', 'parameter_bridge', 'robot_state_publisher']:
        subprocess.run(['pkill', '-9', '-f', pat], stderr=subprocess.DEVNULL)
    time.sleep(4)


def launch_all(worlds):
    kill_sims()
    procs = [subprocess.Popen([str(WS / 'scripts' / 'launch_sim.sh'), w, str(i)],
                              stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
             for i, w in enumerate(worlds)]
    ok = True
    for p in procs:
        try:
            out, _ = p.communicate(timeout=240)
            if b'simulator up' not in out:
                ok = False
        except subprocess.TimeoutExpired:
            p.kill()
            ok = False
    time.sleep(4)
    return ok


def make_env(world, inst, start_x):
    def _f():
        os.environ['ROS_DOMAIN_ID'] = str(40 + inst)
        os.environ['GZ_PARTITION'] = f'sim{inst}'
        from greenhouse_env import GreenhouseEnv
        return GreenhouseEnv(world, worlds_dir=TRAIN_WORLDS,
                             episode_limit_s=EPISODE_LIMIT_S, start_x=start_x)
    return _f


def episode_rows(seed):
    rows = []
    for f in sorted(BLOCKS.glob(f'seed{seed}_b*.monitor.csv')):
        for line in open(f):
            line = line.strip()
            if not line or not (line[0].isdigit() or line[0] == '-'):
                continue
            rows.append(line.split(','))
    return rows


def success_rate_since(seed, mark, window=WINDOW):
    rows = episode_rows(seed)[mark:][-window:]
    if len(rows) < window:
        return None, len(rows)
    hits = sum(1 for r in rows if len(r) > 3 and r[3] == 'success')
    return hits / len(rows), len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--total-steps', type=int, default=1_000_000)
    ap.add_argument('--steps-per-block', type=int, default=10_000)
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--start-level', type=int, default=None,
                    help='override the checkpointed curriculum level')
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    BLOCKS.mkdir(parents=True, exist_ok=True)
    model_path = OUT / f'ppo_seed{args.seed}'
    state_path = LOGS / f'curriculum_seed{args.seed}.json'

    all_worlds = sorted(p.stem for p in TRAIN_WORLDS.glob('*.sdf'))
    rng = random.Random(args.seed)

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

    done = 0
    level = 0
    level_steps = 0
    level_mark = len(episode_rows(args.seed))
    if state_path.exists() and args.start_level is None:
        try:
            st = json.loads(state_path.read_text())
            done = st.get('done', 0)
            level = st.get('level', 0)
            level_steps = st.get('level_steps', 0)
            level_mark = st.get('level_mark', level_mark)
            print(f'resuming: {done} steps done, level {level + 1}'
                  f'/{len(LEVELS)} (start x={LEVELS[level]})', flush=True)
        except Exception as e:
            print(f'could not read {state_path}: {e}', flush=True)
    if args.start_level is not None:
        level = args.start_level

    def save_state():
        state_path.write_text(json.dumps(
            {'done': done, 'level': level, 'level_steps': level_steps,
             'level_mark': level_mark, 'updated': time.strftime('%F %T')},
            indent=2))

    block = max((int(f.name.split('_b')[1][:4])
                 for f in BLOCKS.glob(f'seed{args.seed}_b*.monitor.csv')),
                default=0)
    t0 = time.time()
    print(f'{args.workers} workers, seed {args.seed}, '
          f'{len(all_worlds)} worlds, episode allowance {EPISODE_LIMIT_S:.0f}s',
          flush=True)

    while done < args.total_steps:
        worlds = pick_worlds(level, all_worlds, args.workers, rng)
        print(f'\n[{done}/{args.total_steps}] level {level + 1}/{len(LEVELS)} '
              f'start x={LEVELS[level]}  worlds={[w[11:] for w in worlds]}',
              flush=True)

        if not launch_all(worlds):
            print('  launch failed, retrying with different worlds', flush=True)
            continue

        block += 1
        venv = None
        try:
            venv = SubprocVecEnv([make_env(w, i, LEVELS[level])
                                  for i, w in enumerate(worlds)])
            venv = VecMonitor(
                venv,
                filename=str(BLOCKS / f'seed{args.seed}_b{block:04d}'),
                info_keywords=('outcome',))

            if (model_path.with_suffix('.zip')).exists():
                model = PPO.load(str(model_path), env=venv)
            else:
                model = PPO('MlpPolicy', venv, seed=args.seed, verbose=0,
                            policy_kwargs=POLICY_KWARGS, **PPO_KWARGS)
                import torch
                with torch.no_grad():
                    model.policy.action_net.bias[0].fill_(1.0)
                print('  init: linear action mean biased to full forward',
                      flush=True)

            model.learn(total_timesteps=args.steps_per_block,
                        reset_num_timesteps=False)
            model.save(str(model_path))
        except Exception as e:
            print(f'  BLOCK FAILED: {type(e).__name__}: {e}', flush=True)
            continue
        finally:
            if venv is not None:
                closer = threading.Thread(target=_safe_close, args=(venv,),
                                          daemon=True)
                closer.start()
                closer.join(30)
                if closer.is_alive():
                    print('  venv.close() hung (dead worker) - abandoning it',
                          flush=True)
                    _hard_kill(venv)
            kill_sims()

        done += args.steps_per_block
        level_steps += args.steps_per_block
        rate, n = success_rate_since(args.seed, level_mark)
        shown = 'n/a' if rate is None else f'{rate:.0%}'
        print(f'  {done / (time.time() - t0):.1f} steps/sec overall, '
              f'success {shown} of {n} episodes at this level', flush=True)

        advance = rate is not None and rate >= ADVANCE_RATE
        collapse = (rate is not None and rate < REGRESS_RATE
                    and level_steps >= REGRESS_AFTER)
        stuck = level_steps >= LEVEL_CAP
        top = len(LEVELS) - 1

        def checkpoint():
            subprocess.run(['cp', f'{model_path}.zip',
                            str(OUT / f'ppo_seed{args.seed}_level{level}.zip')])

        if advance and level < top:
            checkpoint()
            level += 1
            level_steps = 0
            level_mark = len(episode_rows(args.seed))
            print(f'  >>> MASTERED at {shown}. Advancing to level '
                  f'{level + 1}: start x={LEVELS[level]}', flush=True)

        elif collapse and level > 0:
            level -= 1
            level_steps = 0
            level_mark = len(episode_rows(args.seed))
            print(f'  <<< COLLAPSED to {shown}. Dropping back to level '
                  f'{level + 1}: start x={LEVELS[level]}', flush=True)

        elif stuck and level < top:
            checkpoint()
            level += 1
            level_steps = 0
            level_mark = len(episode_rows(args.seed))
            print(f'  !!! FORCE-ADVANCED at {shown} after {LEVEL_CAP} steps '
                  f'without reaching {ADVANCE_RATE:.0%}. Level {level + 1}: '
                  f'start x={LEVELS[level]}',
                  flush=True)

        save_state()

    kill_sims()
    print(f'\nstopped at {done} steps, level {level + 1}/{len(LEVELS)}')


if __name__ == '__main__':
    main()
