#!/bin/bash

echo "🚀 运行 detection_node..."

# Conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rel_env

# ROS 环境
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash

# Run detection node
ros2 run vision_ai detection_node
