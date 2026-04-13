#!/bin/bash

# 根据参数决定显示模式
DISPLAY_MODE=${1:-"off"}  # 默认关闭显示

export VISION_AI_DISPLAY=$DISPLAY_MODE
export QT_QPA_PLATFORM=offscreen
export MPLBACKEND=Agg

# 如果要开启显示
if [ "$DISPLAY_MODE" = "on" ]; then
    export QT_QPA_PLATFORM=xcb
    export QT_QPA_PLATFORM_PLUGIN_PATH="/usr/lib/x86_64-linux-gnu/qt5/plugins"
    export MPLBACKEND=TkAgg
fi

# 激活环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rel_env
source /opt/ros/humble/setup.bash
cd ~/ros2_ws
source install/setup.bash

# 运行节点
ros2 run vision_ai detection_node
EOF