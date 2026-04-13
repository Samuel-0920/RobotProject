#!/bin/bash

source install/setup.bash

if [ "$1" = "track_test" ]; then
    echo "🎯 Starting track testing environment..."

    # 创建tmux会话用于追踪测试
    tmux new-session -d -s track_test

    # 窗口0：硬件节点（相机+机械臂）
    tmux send-keys -t track_test:0 'echo "📷 启动相机节点..."' Enter
    tmux send-keys -t track_test:0 'ros2 run camera_node realsense_publisher' Enter
    tmux split-window -h -t track_test:0
    tmux send-keys -t track_test:0.1 'echo "🤖 启动机械臂控制器..."' Enter
    tmux send-keys -t track_test:0.1 'sleep 2' Enter
    tmux send-keys -t track_test:0.1 'ros2 run xarm_controller xarm_controller' Enter

    # 窗口1：追踪节点
    tmux new-window -t track_test:1
    tmux send-keys -t track_test:1 'echo "🎯 启动追踪节点..."' Enter
    tmux send-keys -t track_test:1 'echo "等待硬件初始化..."' Enter
    tmux send-keys -t track_test:1 'sleep 5' Enter
    tmux send-keys -t track_test:1 'ros2 run vision_ai tracking_node' Enter

    # 窗口2：追踪测试触发器
    tmux new-window -t track_test:2
    tmux send-keys -t track_test:2 'echo "🚀 追踪测试触发器"' Enter
    tmux send-keys -t track_test:2 'echo "══════════════════════════════════════"' Enter
    tmux send-keys -t track_test:2 'cd ~/ros2_ws && source install/setup.bash' Enter
    tmux send-keys -t track_test:2 'echo ""' Enter
    tmux send-keys -t track_test:2 'echo "等待追踪节点启动完成..."' Enter
    tmux send-keys -t track_test:2 'sleep 8' Enter
    tmux send-keys -t track_test:2 'python3 -c "
import sys
sys.path.append(\"/home/qi/ros2_ws/src/vision_ai\")

def main():
    print(\"\\n🎯 追踪节点测试触发器\")
    print(\"═\" * 50)
    print(\"请输入要测试的扫描目录路径:\")
    print(\"例如: /home/qi/ros2_ws/scan_output_20250728_214915\")
    print(\"═\" * 50)
    
    while True:
        try:
            scan_dir = input(\"\\n📁 扫描目录路径: \").strip()
            
            if not scan_dir:
                print(\"❌ 路径不能为空，请重新输入\")
                continue
                
            if scan_dir.lower() in [\"quit\", \"exit\", \"q\"]:
                print(\"👋 退出测试触发器\")
                break
                
            # 导入所需模块
            import rclpy
            from rclpy.node import Node
            from std_msgs.msg import String
            import json
            import os
            import time
            
            # 验证目录存在
            if not os.path.exists(scan_dir):
                print(f\"❌ 目录不存在: {scan_dir}\")
                continue
                
            # 检查必需文件
            required_files = [
                os.path.join(scan_dir, \"enhanced_detection_results\", \"reference_features_library.json\"),
                os.path.join(scan_dir, \"enhanced_detection_results\", \"tracking_selection.txt\")
            ]
            
            missing_files = []
            for file_path in required_files:
                if not os.path.exists(file_path):
                    missing_files.append(os.path.basename(file_path))
            
            if missing_files:
                print(f\"❌ 缺少必需文件: {missing_files}\")
                print(\"请确保该目录包含完整的检测结果\")
                continue
            
            print(f\"✅ 目录验证通过: {scan_dir}\")
            
            # 初始化ROS2
            rclpy.init()
            
            class TrackTrigger(Node):
                def __init__(self):
                    super().__init__(\"track_trigger\")
                    self.publisher = self.create_publisher(String, \"/detection_complete\", 10)
                    
                def send_trigger(self, scan_directory):
                    # 构建检测完成消息
                    detection_data = {
                        \"enhanced_detection\": True,
                        \"scan_directory\": scan_directory,
                        \"reference_library_file\": os.path.join(scan_directory, \"enhanced_detection_results\", \"reference_features_library.json\"),
                        \"timestamp\": time.time(),
                        \"trigger_source\": \"manual_test\"
                    }
                    
                    msg = String()
                    msg.data = json.dumps(detection_data)
                    
                    print(f\"\\n📤 发送追踪触发信号...\")
                    print(f\"📁 扫描目录: {scan_directory}\")
                    print(f\"📋 特征库: {detection_data[\\\"reference_library_file\\\"]}\")
                    
                    # 发送消息
                    self.publisher.publish(msg)
                    print(f\"✅ 追踪触发信号已发送!\")
                    print(f\"🎯 请检查追踪节点窗口查看追踪进度\")
                    
                    return True
            
            # 创建触发器并发送消息
            trigger_node = TrackTrigger()
            
            if trigger_node.send_trigger(scan_dir):
                print(f\"\\n🎉 测试触发成功!\")
                print(f\"📋 后续操作:\")
                print(f\"   1. 切换到追踪节点窗口 (Ctrl+B, 1)\")
                print(f\"   2. 观察追踪过程和日志输出\")
                print(f\"   3. 根据提示进行追踪确认\")
                print(f\"\\n💡 提示: 输入 \\'quit\\' 退出触发器\")
            
            # 清理
            trigger_node.destroy_node()
            rclpy.shutdown()
            
        except KeyboardInterrupt:
            print(\"\\n🛑 用户中断\")
            break
        except Exception as e:
            print(f\"❌ 错误: {e}\")
            continue

if __name__ == \"__main__\":
    main()
"' Enter
    
    # 右侧分屏：监控和调试
    tmux split-window -h -t track_test:2
    tmux send-keys -t track_test:2.1 'echo "📊 追踪监控面板"' Enter
    tmux send-keys -t track_test:2.1 'echo "══════════════════════════════════════"' Enter
    tmux send-keys -t track_test:2.1 'cd ~/ros2_ws && source install/setup.bash' Enter
    tmux send-keys -t track_test:2.1 'echo ""' Enter
    tmux send-keys -t track_test:2.1 'echo "📋 监控命令:"' Enter
    tmux send-keys -t track_test:2.1 'echo "  ros2 topic echo /tracking_status"' Enter
    tmux send-keys -t track_test:2.1 'echo "  ros2 topic echo /xarm/target_pose"' Enter
    tmux send-keys -t track_test:2.1 'echo "  ros2 topic echo /detection_complete"' Enter
    tmux send-keys -t track_test:2.1 'echo "  ros2 node list"' Enter
    tmux send-keys -t track_test:2.1 'echo ""' Enter
    tmux send-keys -t track_test:2.1 'echo "🎮 实时监控追踪状态..."' Enter
    tmux send-keys -t track_test:2.1 'sleep 10' Enter
    tmux send-keys -t track_test:2.1 'watch -n 1 "echo \"节点状态:\"; ros2 node list 2>/dev/null | grep -E \"(tracking|camera|xarm)\" || echo \"等待节点启动...\""' Enter

    # 窗口3：系统状态和日志
    tmux new-window -t track_test:3
    tmux send-keys -t track_test:3 'echo "📋 系统状态监控"' Enter
    tmux send-keys -t track_test:3 'echo "══════════════════════════════════════"' Enter
    tmux send-keys -t track_test:3 'cd ~/ros2_ws && source install/setup.bash' Enter
    tmux send-keys -t track_test:3 'echo ""' Enter
    tmux send-keys -t track_test:3 'echo "等待系统启动..."' Enter
    tmux send-keys -t track_test:3 'sleep 12' Enter
    tmux send-keys -t track_test:3 'echo "🔧 ROS2 话题列表:"' Enter
    tmux send-keys -t track_test:3 'ros2 topic list | grep -E "(camera|xarm|track|detection)"' Enter

    # 连接到会话，默认显示触发器窗口
    tmux select-window -t track_test:2
    tmux attach-session -t track_test

elif [ "$1" = "full_system" ]; then
    echo "🚀 Starting full vision AI system with 3D detection..."

    # 创建tmux会话，包含所有必需的窗口
    tmux new-session -d -s vision_ai

    # 窗口0：硬件节点（相机+机械臂）
    tmux send-keys -t vision_ai:0 'ros2 run camera_node realsense_publisher' Enter
    tmux split-window -h -t vision_ai:0
    tmux send-keys -t vision_ai:0.1 'ros2 run xarm_controller xarm_controller' Enter

    # 窗口1：AI处理节点（规划器+执行器）
    tmux new-window -t vision_ai:1
    tmux send-keys -t vision_ai:1 'ros2 run vision_ai scan_planner_node' Enter
    tmux split-window -h -t vision_ai:1
    tmux send-keys -t vision_ai:1.1 'ros2 run vision_ai scan_executor_node' Enter

    # 窗口2：拼接和检测
    tmux new-window -t vision_ai:2
    tmux send-keys -t vision_ai:2 'ros2 run vision_ai smart_stitcher_node' Enter
    tmux split-window -h -t vision_ai:2
    tmux send-keys -t vision_ai:2.1 'ros2 run vision_ai detection_node' Enter

    # 窗口3：GUI和监控
    tmux new-window -t vision_ai:3
    tmux send-keys -t vision_ai:3 'ros2 run vision_ai gui_config_node' Enter
    tmux split-window -h -t vision_ai:3
    tmux send-keys -t vision_ai:3.1 'echo "📊 监控窗口 - 使用以下命令:"' Enter
    tmux send-keys -t vision_ai:3.1 'echo "ros2 topic echo /detection_result"' Enter
    tmux send-keys -t vision_ai:3.1 'echo "ros2 topic echo /stitching_complete"' Enter
    tmux send-keys -t vision_ai:3.1 'echo "ros2 node list"' Enter

    # 窗口4：追踪系统主控制台
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
    tmux send-keys -t vision_ai:4.1 'echo "📊 追踪状态监控与控制"' Enter
    tmux send-keys -t vision_ai:4.1 'echo "═══════════════════════════════════════"' Enter
    tmux send-keys -t vision_ai:4.1 'cd ~/ros2_ws && source install/setup.bash' Enter
    tmux send-keys -t vision_ai:4.1 'echo ""' Enter
    tmux send-keys -t vision_ai:4.1 'echo "📋 快捷命令："' Enter
    tmux send-keys -t vision_ai:4.1 'echo "  📊 追踪状态: ros2 topic echo /tracking_status --once"' Enter
    tmux send-keys -t vision_ai:4.1 'echo "  🎮 目标位姿: ros2 topic echo /xarm/target_pose --once"' Enter
    tmux send-keys -t vision_ai:4.1 'echo "  🤏 夹爪控制: ros2 topic echo /xarm/gripper_target --once"' Enter
    tmux send-keys -t vision_ai:4.1 'echo "  🖼️ 可视化流: ros2 topic echo /tracking/visualization --once"' Enter
    tmux send-keys -t vision_ai:4.1 'echo "  🚨 紧急停止: ros2 topic pub /tracking/emergency_stop std_msgs/Empty"' Enter
    tmux send-keys -t vision_ai:4.1 'echo ""' Enter
    tmux send-keys -t vision_ai:4.1 'echo "等待追踪节点启动..."' Enter
    tmux send-keys -t vision_ai:4.1 'sleep 3' Enter

    # 连接到会话，默认显示第1个窗口（规划+执行）
    tmux select-window -t vision_ai:1
    tmux attach-session -t vision_ai

elif [ "$1" = "both" ]; then
    echo "🔧 Starting basic nodes with detection..."
    tmux new-session -d -s vision_ai
    tmux send-keys -t vision_ai:0 'ros2 run vision_ai scan_planner_node' Enter
    tmux split-window -h -t vision_ai:0
    tmux send-keys -t vision_ai:0.1 'ros2 run vision_ai gui_config_node' Enter
    # 添加检测节点到基本模式
    tmux split-window -v -t vision_ai:0
    tmux send-keys -t vision_ai:0.2 'ros2 run vision_ai detection_node' Enter
    tmux attach-session -t vision_ai
else
    echo "Usage: ./dev.sh [track_test|both|full_system]"
    echo "  track_test: Track testing environment (camera + xarm + tracking + trigger)"
    echo "  both: Basic nodes (planner + GUI + detection)"
    echo "  full_system: All nodes with hardware + 3D detection"
    echo ""
    echo "🎯 Track Testing Mode:"
    echo "  - Starts camera and xarm controllers"
    echo "  - Launches tracking node"
    echo "  - Provides interactive trigger to test any scan directory"
    echo "  - Includes monitoring and debugging tools"
fi