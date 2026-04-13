#!/bin/bash

source install/setup.bash

if [ "$1" = "full_system" ]; then
    echo "🚀 Starting full vision AI system with 3D detection..."

    # 创建tmux会话，包含所有必需的窗口
    tmux new-session -d -s vision_ai

    # 窗口0：硬件节点（相机+机械臂）
    tmux send-keys -t vision_ai:0 'echo "📷 启动相机节点..."' Enter
    tmux send-keys -t vision_ai:0 'ros2 run camera_node realsense_publisher' Enter
    tmux split-window -h -t vision_ai:0
    tmux send-keys -t vision_ai:0.1 'echo "🦾 启动机械臂控制器..."' Enter
    tmux send-keys -t vision_ai:0.1 'ros2 run xarm_controller xarm_controller' Enter

    # 窗口1：AI处理节点（规划器+执行器）
    tmux new-window -t vision_ai:1
    tmux send-keys -t vision_ai:1 'echo "🧠 启动扫描规划器..."' Enter
    tmux send-keys -t vision_ai:1 'ros2 run vision_ai scan_planner_node' Enter
    tmux split-window -h -t vision_ai:1
    tmux send-keys -t vision_ai:1.1 'echo "⚡启动扫描执行器..."' Enter
    tmux send-keys -t vision_ai:1.1 'ros2 run vision_ai scan_executor_node' Enter

    # 窗口2：拼接和增强检测 🆕
    tmux new-window -t vision_ai:2
    tmux send-keys -t vision_ai:2 'echo "🔗 启动智能拼接器..."' Enter
    tmux send-keys -t vision_ai:2 'ros2 run vision_ai smart_stitcher_node' Enter
    tmux split-window -h -t vision_ai:2
    tmux send-keys -t vision_ai:2.1 'echo "🎯 启动增强检测节点..."' Enter
    tmux send-keys -t vision_ai:2.1 'sleep 3' Enter
    tmux send-keys -t vision_ai:2.1 'ros2 run vision_ai enhanced_detection_node' Enter

    # 🆕 窗口3：3窗口可视化GUI
    tmux new-window -t vision_ai:3
    tmux send-keys -t vision_ai:3 'echo "🖼️ 启动3窗口可视化界面..."' Enter
    tmux send-keys -t vision_ai:3 'echo "等待其他节点启动完成..."' Enter
    tmux send-keys -t vision_ai:3 'sleep 5' Enter
    tmux send-keys -t vision_ai:3 'ros2 run vision_ai visualization_trigger' Enter
    tmux split-window -h -t vision_ai:3
    tmux send-keys -t vision_ai:3.1 'echo "📊 监控窗口 - 使用以下命令:"' Enter
    tmux send-keys -t vision_ai:3.1 'echo "ros2 topic echo /detection_result --once"' Enter
    tmux send-keys -t vision_ai:3.1 'echo "ros2 topic echo /detection_complete --once"' Enter
    tmux send-keys -t vision_ai:3.1 'echo "ros2 topic echo /detection_visualization --once"' Enter
    tmux send-keys -t vision_ai:3.1 'echo "ros2 topic list | grep detection"' Enter
    tmux send-keys -t vision_ai:3.1 'echo ""' Enter
    tmux send-keys -t vision_ai:3.1 'echo "🔍 实时监控检测状态:"' Enter

    # 窗口4：系统状态和调试
    tmux new-window -t vision_ai:4
    tmux send-keys -t vision_ai:4 'echo "🔧 系统状态监控"' Enter
    tmux send-keys -t vision_ai:4 'echo "═══════════════════════════════════════"' Enter
    tmux send-keys -t vision_ai:4 'sleep 8' Enter
    tmux send-keys -t vision_ai:4 'echo "📋 活跃节点:"' Enter
    tmux send-keys -t vision_ai:4 'ros2 node list' Enter
    tmux send-keys -t vision_ai:4 'echo ""' Enter
    tmux send-keys -t vision_ai:4 'echo "📡 活跃话题:"' Enter
    tmux send-keys -t vision_ai:4 'ros2 topic list | grep camera' Enter
    tmux split-window -h -t vision_ai:4
    tmux send-keys -t vision_ai:4.1 'echo "📊 话题监控 - 选择要监控的话题:"' Enter
    tmux send-keys -t vision_ai:4.1 'echo "  1. ros2 topic hz /camera/color/image_raw"' Enter
    tmux send-keys -t vision_ai:4.1 'echo "  2. ros2 topic hz /camera/depth/image_raw"' Enter
    tmux send-keys -t vision_ai:4.1 'echo "  3. ros2 topic echo /stitching_complete"' Enter
    tmux send-keys -t vision_ai:4.1 'echo "  4. ros2 topic echo /detection_complete"' Enter
    tmux send-keys -t vision_ai:4.1 'echo ""' Enter
    tmux send-keys -t vision_ai:4.1 'echo "🎮 手动触发测试:"' Enter
    tmux send-keys -t vision_ai:4.1 'echo "  ros2 topic pub /start_scan std_msgs/Empty"' Enter

    # 连接到会话，默认显示可视化窗口
    tmux select-window -t vision_ai:3
    tmux attach-session -t vision_ai

elif [ "$1" = "scan_detect_test" ]; then
    echo "🧪 Starting scan-detect test mode with visualization..."
    
    # 创建测试模式的tmux会话
    tmux new-session -d -s scan_detect_test

    # 窗口0：相机节点
    tmux send-keys -t scan_detect_test:0 'echo "📷 启动相机节点 (scan-detect测试模式)..."' Enter
    tmux send-keys -t scan_detect_test:0 'ros2 run camera_node realsense_publisher' Enter
    
    # 窗口1：扫描相关节点
    tmux new-window -t scan_detect_test:1
    tmux send-keys -t scan_detect_test:1 'echo "🧠 启动扫描规划器..."' Enter
    tmux send-keys -t scan_detect_test:1 'ros2 run vision_ai scan_planner_node' Enter
    tmux split-window -h -t scan_detect_test:1
    tmux send-keys -t scan_detect_test:1.1 'echo "⚡ 启动扫描执行器..."' Enter
    tmux send-keys -t scan_detect_test:1.1 'ros2 run vision_ai scan_executor_node' Enter
    tmux split-window -v -t scan_detect_test:1.1
    tmux send-keys -t scan_detect_test:1.2 'echo "🔗 启动智能拼接器..."' Enter
    tmux send-keys -t scan_detect_test:1.2 'ros2 run vision_ai smart_stitcher_node' Enter

    # 窗口2：增强检测节点
    tmux new-window -t scan_detect_test:2
    tmux send-keys -t scan_detect_test:2 'echo "🎯 启动增强检测节点..."' Enter
    tmux send-keys -t scan_detect_test:2 'echo "等待其他节点启动..."' Enter
    tmux send-keys -t scan_detect_test:2 'sleep 4' Enter
    tmux send-keys -t scan_detect_test:2 'ros2 run vision_ai enhanced_detection_node' Enter

    # 窗口3：3窗口可视化GUI
    tmux new-window -t scan_detect_test:3
    tmux send-keys -t scan_detect_test:3 'echo "🖼️ 启动3窗口可视化界面..."' Enter
    tmux send-keys -t scan_detect_test:3 'echo "等待所有节点启动完成..."' Enter
    tmux send-keys -t scan_detect_test:3 'sleep 6' Enter
    tmux send-keys -t scan_detect_test:3 'ros2 run vision_ai visualization_trigger' Enter

    # 窗口4：测试控制台
    tmux new-window -t scan_detect_test:4
    tmux send-keys -t scan_detect_test:4 'echo "🎮 测试控制台"' Enter
    tmux send-keys -t scan_detect_test:4 'echo "═══════════════════════════════════════"' Enter
    tmux send-keys -t scan_detect_test:4 'sleep 8' Enter
    tmux send-keys -t scan_detect_test:4 'echo "🚀 准备就绪！使用以下命令开始测试:"' Enter
    tmux send-keys -t scan_detect_test:4 'echo ""' Enter
    tmux send-keys -t scan_detect_test:4 'echo "1️⃣ 手动触发扫描:"' Enter
    tmux send-keys -t scan_detect_test:4 'echo "   ros2 topic pub /start_scan std_msgs/Empty --once"' Enter
    tmux send-keys -t scan_detect_test:4 'echo ""' Enter
    tmux send-keys -t scan_detect_test:4 'echo "2️⃣ 检查系统状态:"' Enter
    tmux send-keys -t scan_detect_test:4 'echo "   ros2 node list"' Enter
    tmux send-keys -t scan_detect_test:4 'echo "   ros2 topic list"' Enter
    tmux send-keys -t scan_detect_test:4 'echo ""' Enter
    tmux send-keys -t scan_detect_test:4 'echo "3️⃣ 实时监控:"' Enter
    tmux send-keys -t scan_detect_test:4 'echo "   ros2 topic echo /stitching_complete"' Enter
    tmux send-keys -t scan_detect_test:4 'echo "   ros2 topic echo /detection_complete"' Enter
    tmux send-keys -t scan_detect_test:4 'echo ""' Enter
    tmux send-keys -t scan_detect_test:4 'echo "📊 系统应该显示3个窗口: 彩色图像、深度图像、检测结果"' Enter
    
    # 连接到会话，默认显示测试控制台
    tmux select-window -t scan_detect_test:4
    tmux attach-session -t scan_detect_test

elif [ "$1" = "both" ]; then
    echo "🔧 Starting basic nodes with enhanced detection..."
    tmux new-session -d -s vision_ai
    
    # 添加相机节点
    tmux send-keys -t vision_ai:0 'echo "📷 启动相机节点..."' Enter
    tmux send-keys -t vision_ai:0 'ros2 run camera_node realsense_publisher' Enter
    tmux split-window -h -t vision_ai:0
    tmux send-keys -t vision_ai:0.1 'ros2 run vision_ai scan_planner_node' Enter
    tmux split-window -v -t vision_ai:0.1
    tmux send-keys -t vision_ai:0.2 'ros2 run vision_ai gui_config_node' Enter
    
    # 🆕 添加增强检测节点到基本模式
    tmux split-window -v -t vision_ai:0
    tmux send-keys -t vision_ai:0.3 'echo "🎯 启动增强检测节点..."' Enter
    tmux send-keys -t vision_ai:0.3 'sleep 3' Enter
    tmux send-keys -t vision_ai:0.3 'ros2 run vision_ai enhanced_detection_node' Enter
    
    # 🆕 添加可视化
    tmux new-window -t vision_ai:1
    tmux send-keys -t vision_ai:1 'echo "🖼️ 启动可视化界面..."' Enter
    tmux send-keys -t vision_ai:1 'sleep 5' Enter
    tmux send-keys -t vision_ai:1 'ros2 run vision_ai visualization_trigger' Enter
    
    tmux attach-session -t vision_ai
else
    echo "Usage: ./dev.sh [both|full_system|scan_detect_test]"
    echo "  both: Basic nodes (planner + GUI + enhanced detection + visualization)"
    echo "  full_system: All nodes with hardware + 3D detection + visualization"
    echo "  scan_detect_test: Focused scan-detect testing with 3-window visualization"
fi