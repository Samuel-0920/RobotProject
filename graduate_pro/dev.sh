#!/bin/bash

source install/setup.bash

if [ "$1" = "full_system" ]; then
    echo "🚀 Starting full vision AI system with 3D detection..."

    # 创建tmux会话，包含所有必需的窗口
    tmux new-session -d -s vision_ai

    # # 窗口0：硬件节点（相机+机械臂）
    tmux send-keys -t vision_ai:0 'ros2 run camera_node realsense_publisher' Enter
    tmux split-window -h -t vision_ai:0
    tmux send-keys -t vision_ai:0.1 'ros2 run xarm_controller xarm_controller' Enter

    # 窗口1：AI处理节点（规划器+执行器）
    tmux new-window -t vision_ai:1
    tmux send-keys -t vision_ai:1 'ros2 run vision_ai scan_planner_node' Enter
    tmux split-window -h -t vision_ai:1
    tmux send-keys -t vision_ai:1.1 'ros2 run vision_ai scan_executor_node' Enter

    # 窗口2：拼接和检测 🆕
    tmux new-window -t vision_ai:2
    tmux send-keys -t vision_ai:2 'ros2 run vision_ai smart_stitcher_node' Enter
    tmux split-window -h -t vision_ai:2
    tmux send-keys -t vision_ai:2.1 'ros2 run vision_ai detection_node' Enter

    # 窗口3：GUI和监控 + 自动实验记录
    tmux new-window -t vision_ai:3
    tmux send-keys -t vision_ai:3 'ros2 run vision_ai gui_config_node' Enter
    tmux split-window -h -t vision_ai:3

    # 右侧启动自动记录器
    tmux send-keys -t vision_ai:3.1 'echo "Starting automated experiment logger..."' Enter
    tmux send-keys -t vision_ai:3.1 'sleep 5' Enter  # 等待其他节点启动
    tmux send-keys -t vision_ai:3.1 'cd ~/vision_ai_experiments && source /opt/ros/humble/setup.bash && source ~/ros2_ws/install/setup.bash' Enter
    tmux send-keys -t vision_ai:3.1 './auto_logger.sh' Enter
    # 🆕 窗口4：追踪系统主控制台
    tmux new-window -t vision_ai:4
    tmux send-keys -t vision_ai:4 'echo "🎯 追踪系统主控制台"' Enter
    tmux send-keys -t vision_ai:4 'echo "═══════════════════════════════════════"' Enter
    tmux send-keys -t vision_ai:4 'echo "正在启动追踪节点..."' Enter
    tmux send-keys -t vision_ai:4 'cd ~/ros2_ws && source install/setup.bash' Enter
    tmux send-keys -t vision_ai:4 'echo ""' Enter
    tmux send-keys -t vision_ai:4 'echo "🚀 启动追踪节点..."' Enter
    tmux send-keys -t vision_ai:4 'sleep 2' Enter
    tmux send-keys -t vision_ai:4 'ros2 run vision_ai tracking_node' Enter

    # 右侧分屏：追踪状态监控和控制
    tmux split-window -h -t vision_ai:4
    echo -e "${GREEN}🎯 准备静态抓取系统...${NC}"
    tmux new-window -t vision_ai:4
    tmux send-keys -t vision_ai:4 'cd ~/ros2_ws && source install/setup.bash' Enter
    tmux send-keys -t vision_ai:4 'echo "抓取系统"' Enter
    tmux send-keys -t vision_ai:4 'echo "等待detection完成后自动启动..."' Enter
    tmux send-keys -t vision_ai:4 'echo "监听 /detection_complete 话题..."' Enter
    
    # # 启动完全自动化静态抓取系统
    tmux send-keys -t vision_ai:4 'echo "🤖 Starting Automated Static Grasp System..."' Enter
    tmux send-keys -t vision_ai:4 'python3 /home/qi/ros2_ws/src/vision_ai/vision_ai/simple_grasp_validator.py' Enter
    # 连接到会话，默认显示第1个窗口（规划+执行）
    tmux select-window -t vision_ai:2
    tmux attach-session -t vision_ai

elif [ "$1" = "both" ]; then
    echo "🔧 Starting basic nodes with detection..."
    tmux new-session -d -s vision_ai
    tmux send-keys -t vision_ai:0 'ros2 run vision_ai scan_planner_node' Enter
    tmux split-window -h -t vision_ai:0
    tmux send-keys -t vision_ai:0.1 'ros2 run vision_ai gui_config_node' Enter
    # 🆕 添加检测节点到基本模式
    tmux split-window -v -t vision_ai:0
    tmux send-keys -t vision_ai:0.2 'ros2 run vision_ai detection_node' Enter
    tmux attach-session -t vision_ai
else
    echo "Usage: ./dev.sh [both|full_system]"
    echo "  both: Basic nodes (planner + GUI + detection)"
    echo "  full_system: All nodes with hardware + 3D detection"
fi