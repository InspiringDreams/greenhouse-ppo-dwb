#!/bin/bash
# Starts one headless simulator: launch_sim.sh <world> [instance]
source /opt/ros/jazzy/setup.bash && source $HOME/greenhouse_ws/install/setup.bash
WORLD=${1:-greenhouse_straight_sparse_s00}
INST=${2:-0}
export ROS_DOMAIN_ID=$((40 + INST))
export GZ_PARTITION=sim$INST
SDF=$(ls $HOME/greenhouse_ws/worlds/*/$WORLD.sdf 2>/dev/null | head -1)
[ -f "$SDF" ] || { echo "no such world: $WORLD"; exit 1; }

GUIARG=false; [ "${GUI:-0}" = "1" ] && GUIARG=true
ros2 launch greenhouse_sim spawn_robot.launch.py world:=$SDF gui:=$GUIARG \
  > /tmp/sim_$INST.log 2>&1 &

for i in $(seq 40); do
  timeout 3 ros2 topic echo /scan --once > /dev/null 2>&1 && {
    echo "simulator up: $WORLD (instance $INST, domain $ROS_DOMAIN_ID)"; exit 0; }
  sleep 2
done
echo "FAILED to start - last lines of /tmp/sim_$INST.log:"; tail -15 /tmp/sim_$INST.log
exit 1
