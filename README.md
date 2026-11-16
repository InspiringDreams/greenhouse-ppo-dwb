# PPO vs DWB in a simulated greenhouse

Code for the Honours study *Simulation-Based Reinforcement Learning for Low-Cost Autonomous Navigation in Greenhouse Environments*.

A trained PPO policy and Nav2's DWB planner each drive a TurtleBot3 down greenhouse rows, using only a 2D LiDAR and wheel odometry.

## Needs

Ubuntu 24.04, ROS 2 Jazzy, Gazebo Harmonic, Nav2, TurtleBot3, Python 3.12, Stable-Baselines3.

## Run

```bash
python3 scripts/generate_world.py --all --seeds 30   # make the test worlds
python3 scripts/train_parallel.py                    # train PPO
python3 scripts/run_eval_ppo.py                      # test PPO
python3 scripts/run_eval_dwb2.py                     # test DWB
python3 scripts/astar_ground_truth.py                # shortest paths (for SPL)
python3 scripts/true_contacts.py                     # contact check
python3 scripts/time_to_goal.py                      # time to goal
```

## Figures

The `notebooks` folder has one Jupyter notebook per figure and test in the dissertation. Run them after the commands above.
