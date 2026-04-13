#!/bin/bash
source /opt/ros/humble/setup.bash
cd ~/文档/real/ros2_workspace/
colcon build --packages-select vision_ai_interfaces
source install/setup.bash
echo "接口编译完成！"
