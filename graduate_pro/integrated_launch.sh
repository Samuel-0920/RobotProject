#!/bin/bash
# integrated_launch.sh
# 简化的集成启动脚本 - 只保留full模式，detection→静态抓取

echo "🎯 启动完整视觉AI系统 (Detection → 静态抓取)"
echo "============================================="

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

# 检查环境
check_environment() {
    echo -e "${BLUE}🔧 检查环境...${NC}"
    
    # 检查是否在正确的工作空间
    if [ ! -f "install/setup.bash" ]; then
        echo -e "${RED}❌ 请在ROS2工作空间根目录运行此脚本${NC}"
        echo -e "${YELLOW}💡 应该在 ~/ros2_ws 目录下运行${NC}"
        exit 1
    fi
    
    # 检查conda环境
    if [ -z "$CONDA_DEFAULT_ENV" ]; then
        echo -e "${YELLOW}⚠️ 未检测到conda环境，尝试激活rel_env...${NC}"
        source ~/miniconda3/etc/profile.d/conda.sh
        conda activate rel_env
    fi
    
    # Source ROS2环境
    source install/setup.bash
    echo -e "${GREEN}✅ 环境检查完成${NC}"
    echo -e "${BLUE}📦 当前conda环境: $CONDA_DEFAULT_ENV${NC}"
}

# 启动完整系统
start_full_system() {
    echo -e "${PURPLE}🚀 启动完整视觉AI系统...${NC}"
    
    # 杀死可能存在的旧会话
    tmux kill-session -t vision_ai_full 2>/dev/null
    
    # 创建新的tmux会话
    tmux new-session -d -s vision_ai_full
    
    # 窗口0：硬件节点
    echo -e "${BLUE}🔧 启动硬件节点...${NC}"
    tmux send-keys -t vision_ai_full:0 'cd ~/ros2_ws && source install/setup.bash' Enter
    tmux send-keys -t vision_ai_full:0 'ros2 run camera_node realsense_publisher' Enter
    tmux split-window -h -t vision_ai_full:0
    tmux send-keys -t vision_ai_full:0.1 'cd ~/ros2_ws && source install/setup.bash' Enter
    tmux send-keys -t vision_ai_full:0.1 'ros2 run xarm_controller xarm_controller' Enter
    
    # 窗口1：AI处理节点（扫描规划+执行）
    echo -e "${BLUE}🧠 启动AI处理节点...${NC}"
    tmux new-window -t vision_ai_full:1
    tmux send-keys -t vision_ai_full:1 'cd ~/ros2_ws && source install/setup.bash' Enter
    tmux send-keys -t vision_ai_full:1 'ros2 run vision_ai scan_planner_node' Enter
    tmux split-window -h -t vision_ai_full:1
    tmux send-keys -t vision_ai_full:1.1 'cd ~/ros2_ws && source install/setup.bash' Enter
    tmux send-keys -t vision_ai_full:1.1 'ros2 run vision_ai scan_executor_node' Enter
    
    # 窗口2：拼接和检测
    echo -e "${BLUE}🔍 启动拼接和检测节点...${NC}"
    tmux new-window -t vision_ai_full:2
    tmux send-keys -t vision_ai_full:2 'cd ~/ros2_ws && source install/setup.bash' Enter
    tmux send-keys -t vision_ai_full:2 'ros2 run vision_ai smart_stitcher_node' Enter
    tmux split-window -h -t vision_ai_full:2
    tmux send-keys -t vision_ai_full:2.1 'cd ~/ros2_ws && source install/setup.bash' Enter
    tmux send-keys -t vision_ai_full:2.1 'ros2 run vision_ai detection_node' Enter
    
    # 窗口3：🆕 静态抓取系统
    echo -e "${GREEN}🎯 准备静态抓取系统...${NC}"
    tmux new-window -t vision_ai_full:3
    tmux send-keys -t vision_ai_full:3 'cd ~/ros2_ws && source install/setup.bash' Enter
    tmux send-keys -t vision_ai_full:3 'echo "🎯 静态抓取系统"' Enter
    tmux send-keys -t vision_ai_full:3 'echo "等待detection完成后自动启动..."' Enter
    tmux send-keys -t vision_ai_full:3 'echo "监听 /detection_complete 话题..."' Enter
    
    # 启动完全自动化静态抓取系统
    tmux send-keys -t vision_ai_full:3 'echo "🤖 Starting Automated Static Grasp System..."' Enter
    tmux send-keys -t vision_ai_full:3 'python3 /home/qi/ros2_ws/src/vision_ai/vision_ai/static_grasp_system.py' Enter
    
    # 右侧：抓取系统监控
    tmux split-window -h -t vision_ai_full:3
    tmux send-keys -t vision_ai_full:3.1 'cd ~/ros2_ws && source install/setup.bash' Enter
    tmux send-keys -t vision_ai_full:3.1 'echo "📊 抓取系统监控"' Enter
    tmux send-keys -t vision_ai_full:3.1 'echo ""' Enter
    tmux send-keys -t vision_ai_full:3.1 'echo "可用监控命令："' Enter
    tmux send-keys -t vision_ai_full:3.1 'echo "  ros2 topic echo /detection_complete --once"' Enter
    tmux send-keys -t vision_ai_full:3.1 'echo "  ros2 node list | grep -E (grasp|static)"' Enter
    tmux send-keys -t vision_ai_full:3.1 'echo "  ros2 topic list | grep -E (xarm|gripper)"' Enter
    tmux send-keys -t vision_ai_full:3.1 'echo ""' Enter
    tmux send-keys -t vision_ai_full:3.1 'echo "系统状态："' Enter
    
    # 窗口4：GUI和系统监控
    echo -e "${BLUE}🖥️ 启动GUI监控...${NC}"
    tmux new-window -t vision_ai_full:4
    tmux send-keys -t vision_ai_full:4 'cd ~/ros2_ws && source install/setup.bash' Enter
    tmux send-keys -t vision_ai_full:4 'ros2 run vision_ai gui_config_node' Enter
    tmux split-window -h -t vision_ai_full:4
    tmux send-keys -t vision_ai_full:4.1 'cd ~/ros2_ws && source install/setup.bash' Enter
    tmux send-keys -t vision_ai_full:4.1 'echo "📊 系统监控窗口"' Enter
    tmux send-keys -t vision_ai_full:4.1 'echo ""' Enter
    tmux send-keys -t vision_ai_full:4.1 'echo "系统流程："' Enter
    tmux send-keys -t vision_ai_full:4.1 'echo "  1. 扫描规划 → 2. 扫描执行 → 3. 图像拼接"' Enter
    tmux send-keys -t vision_ai_full:4.1 'echo "  4. 目标检测 → 5. 🆕 静态抓取"' Enter
    tmux send-keys -t vision_ai_full:4.1 'echo ""' Enter
    tmux send-keys -t vision_ai_full:4.1 'echo "监控命令："' Enter
    tmux send-keys -t vision_ai_full:4.1 'echo "  ros2 topic echo /detection_result --once"' Enter
    tmux send-keys -t vision_ai_full:4.1 'echo "  ros2 topic echo /stitching_complete --once"' Enter
    tmux send-keys -t vision_ai_full:4.1 'echo "  ros2 node list"' Enter
    
    # 显示启动完成信息
    echo -e "${GREEN}✅ 系统启动完成！${NC}"
    echo -e "${YELLOW}💡 tmux窗口说明：${NC}"
    echo "  窗口0: 硬件节点 (相机 | 机械臂)"
    echo "  窗口1: 扫描系统 (规划器 | 执行器)"
    echo "  窗口2: 处理系统 (拼接器 | 检测器)"
    echo "  窗口3: 🆕 静态抓取系统 (主程序 | 监控)"
    echo "  窗口4: GUI监控 (配置界面 | 系统监控)"
    echo ""
    echo -e "${BLUE}🔧 tmux操作提示：${NC}"
    echo "  Ctrl+B 然后 0-4    : 切换窗口"
    echo "  Ctrl+B 然后 d      : 脱离会话"
    echo "  tmux attach -t vision_ai_full  : 重新连接"
    echo "  tmux kill-session -t vision_ai_full : 终止会话"
    echo ""
    echo -e "${GREEN}🎯 系统工作流程：${NC}"
    echo "  1. 硬件节点启动完成"
    echo "  2. 扫描系统开始工作"
    echo "  3. 拼接系统处理图像"
    echo "  4. 检测系统分析目标"
    echo "  5. 🆕 静态抓取系统自动启动"
    echo ""
    echo -e "${YELLOW}⏳ 等待系统完全启动...${NC}"
    sleep 3
    
    # 默认显示静态抓取窗口
    tmux select-window -t vision_ai_full:3
    tmux attach-session -t vision_ai_full
}

# 显示帮助信息
show_help() {
    cat << EOF
🎯 集成ROS2静态抓取系统启动脚本

用法: $0 [选项]

这个脚本启动完整的系统流程：
  相机 → 扫描规划 → 扫描执行 → 图像拼接 → 检测分析 → 静态抓取

启动后的系统包含：
  - 硬件节点: realsense_publisher, xarm_controller
  - AI处理: scan_planner_node, scan_executor_node
  - 图像处理: smart_stitcher_node, detection_node
  - 🆕 静态抓取: ros2_static_grasp_system.py (自动启动)
  - GUI监控: gui_config_node

选项:
  -h, --help     显示此帮助信息
  full          启动完整系统 (默认)

系统要求:
  - ROS2 Humble环境
  - conda环境 rel_env
  - 在 ~/ros2_ws 目录下运行

推荐使用方式:
  cd ~/ros2_ws
  ./restart.sh        # 编译并启动完整系统
  
  或者直接：
  ./integrated_launch.sh
EOF
}

# 清理函数
cleanup() {
    echo -e "\n${YELLOW}🔄 正在关闭系统...${NC}"
    tmux kill-session -t vision_ai_full 2>/dev/null
    echo -e "${GREEN}✅ 系统已安全关闭${NC}"
}

# 信号处理
trap cleanup SIGINT SIGTERM

# 主函数
main() {
    echo -e "${BLUE}🚀 开始启动完整视觉AI系统...${NC}"
    
    # 检查参数
    case "${1:-full}" in
        "full"|"")
            check_environment
            start_full_system
            ;;
        "-h"|"--help"|"help")
            show_help
            ;;
        *)
            echo -e "${RED}❌ 未知选项: $1${NC}"
            echo -e "${YELLOW}💡 使用 $0 --help 查看帮助${NC}"
            exit 1
            ;;
    esac
}

# 检查脚本权限
if [ ! -x "$0" ]; then
    echo -e "${YELLOW}⚠️ 脚本没有执行权限，正在修复...${NC}"
    chmod +x "$0"
    echo -e "${GREEN}✅ 权限已修复${NC}"
fi

# 执行主函数
main "$@"