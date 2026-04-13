#!/bin/bash

# 彩色输出定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 解析参数
DISPLAY_MODE=${1:-"on"}  # 默认开启显示

echo -e "${YELLOW}🔄 Quick restart with all nodes...${NC}"

# 如果传入参数控制显示
if [ "$1" = "--no-display" ] || [ "$1" = "off" ]; then
    DISPLAY_MODE="off"
    echo -e "${BLUE}📺 显示功能已禁用${NC}"
elif [ "$1" = "--display" ] || [ "$1" = "on" ]; then
    DISPLAY_MODE="on"
    echo -e "${BLUE}📺 显示功能已启用${NC}"
else
    echo -e "${BLUE}📺 默认启用显示功能 (使用 --no-display 禁用)${NC}"
    DISPLAY_MODE="on"
fi

# 杀死旧的 tmux 会话
tmux kill-session -t vision_ai 2>/dev/null

# 激活 Conda 和工作区
source ~/miniconda3/etc/profile.d/conda.sh
cd ~/ros2_ws || exit 1
eval "$(conda shell.bash hook)"
conda activate rel_env

# 正确 source ROS2 和当前工作区（顺序不能反）
source /opt/ros/humble/setup.bash

# 🔧 设置所有必要的环境变量
echo -e "${YELLOW}🔧 设置编译和运行环境...${NC}"

# NumPy 头文件路径（用于编译）
NUMPY_INCLUDE="/home/qi/.local/lib/python3.10/site-packages/numpy/core/include"
export CPATH="${NUMPY_INCLUDE}:$CPATH"
export C_INCLUDE_PATH="${NUMPY_INCLUDE}:$C_INCLUDE_PATH"
export CPLUS_INCLUDE_PATH="${NUMPY_INCLUDE}:$CPLUS_INCLUDE_PATH"

# 设置显示相关环境变量
export VISION_AI_DISPLAY=$DISPLAY_MODE

if [ "$DISPLAY_MODE" = "off" ]; then
    # 禁用显示模式
    export QT_QPA_PLATFORM=offscreen
    export MPLBACKEND=Agg
    echo -e "${BLUE}🚫 设置为无显示模式${NC}"
else
    # 启用显示模式
    export QT_QPA_PLATFORM=xcb
    export QT_QPA_PLATFORM_PLUGIN_PATH="/usr/lib/x86_64-linux-gnu/qt5/plugins"
    export QT_X11_NO_MITSHM=1
    export DISPLAY=${DISPLAY:-:0}
    export MPLBACKEND=TkAgg
    echo -e "${BLUE}✅ 设置为显示模式${NC}"
fi

# 库路径
export LD_LIBRARY_PATH="/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"

echo -e "${YELLOW}🔧 正在编译 vision_ai 系统...${NC}"

if colcon build \
    --packages-select vision_ai_interfaces vision_ai xarm_controller camera_node \
    --cmake-args -DPYTHON_EXECUTABLE=$(which python); then
    
    echo -e "${GREEN}✅ 编译成功${NC}"
    source install/setup.bash
    
    # 检查 tracking_node 是否成功构建
    if [ -f "install/vision_ai/lib/vision_ai/tracking_node" ]; then
        echo -e "${GREEN}✅ tracking_node 已编译${NC}"
    else
        echo -e "${YELLOW}⚠️  tracking_node 未找到，可能未包含或构建失败${NC}"
    fi
    
    ./dev.sh full_system
else
    echo -e "${RED}❌ 编译失败，请检查错误信息${NC}"
    exit 1
fi