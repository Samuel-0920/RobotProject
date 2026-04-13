#!/bin/bash

# 扫描系统测试启动脚本
# 使用方法: ./test_scan_system.sh [模式]
# 模式: auto (自动测试), interactive (交互测试), benchmark (性能测试), analyze (分析报告)

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 启动系统节点
# 启动系统节点
start_system_nodes() {
    log_info "启动扫描系统节点..."
    
    local tmux_session="scan_test_session"
    
    # 强制杀死旧的会话，以确保干净启动
    if tmux has-session -t "$tmux_session" 2>/dev/null; then
        log_warning "发现旧的tmux会话'$tmux_session'，正在终止..."
        tmux kill-session -t "$tmux_session"
    fi
    
    # 直接根据dev.sh的逻辑启动节点
    log_info "启动系统节点..."
    
    # 使用 tmux 启动各个节点
    tmux new-session -d -s "$tmux_session"
    
    # 窗口0：AI处理节点（规划器+执行器）
    tmux rename-window -t "$tmux_session:0" "planner_executor"
    tmux send-keys -t "$tmux_session:0" 'ros2 run vision_ai scan_planner_node' Enter
    tmux split-window -h -t "$tmux_session:0"
    tmux send-keys -t "$tmux_session:0.1" 'ros2 run vision_ai scan_executor_node' Enter
    
    # 窗口1：拼接
    tmux new-window -t "$tmux_session:1" -n "stitcher"
    tmux send-keys -t "$tmux_session:1" 'ros2 run vision_ai smart_stitcher_node' Enter
    
    # 窗口2：GUI
    tmux new-window -t "$tmux_session:2" -n "gui_config"
    tmux send-keys -t "$tmux_session:2" 'ros2 run vision_ai gui_config_node' Enter
    
    log_success "系统节点启动完成"
}

# 清理节点
cleanup_nodes() {
    log_info "清理系统节点..."
    local tmux_session="scan_test_session"
    
    # 强制清理可能残留的节点
    pkill -f "gui_config_node\|scan_planner_node\|scan_executor_node\|smart_stitcher_node" 2>/dev/null || true
    
    if tmux has-session -t "$tmux_session" 2>/dev/null; then
        log_info "终止 tmux 会话 '$tmux_session'..."
        tmux kill-session -t "$tmux_session"
    fi
    
    log_success "节点清理完成"
}

# 运行测试
run_test() {
    local mode=$1
    log_info "运行测试模式: $mode"
    
    case $mode in
        "auto")
            log_info "执行自动化测试套件..."
            python3 /home/qi/ros2_ws/src/vision_ai/vision_ai/scan_system_tester.py auto
            ;;
        "interactive")
            log_info "启动交互式测试..."
            python3 /home/qi/ros2_ws/src/vision_ai/vision_ai/scan_system_tester.py interactive
            ;;
        "benchmark")
            log_info "执行性能基准测试..."
            python3 /home/qi/ros2_ws/src/vision_ai/vision_ai/scan_system_tester.py benchmark
            ;;
        *)
            log_error "未知测试模式: $mode"
            echo "可用模式: auto, interactive, benchmark"
            exit 1
            ;;
    esac
}

# 分析测试报告
analyze_reports() {
    log_info "分析测试报告..."
    
    REPORT_DIR="scan_test_reports"
    
    if [ ! -d "$REPORT_DIR" ]; then
        log_error "报告目录不存在: $REPORT_DIR"
        exit 1
    fi
    
    # 找到最新的报告文件
    LATEST_REPORT=$(ls -t $REPORT_DIR/scan_test_report_*.json 2>/dev/null | head -n 1)
    
    if [ -z "$LATEST_REPORT" ]; then
        log_error "未找到测试报告文件"
        exit 1
    fi
    
    log_info "分析报告: $LATEST_REPORT"
    
    # 生成简单的分析报告
    python3 -c "
import json
import sys
from datetime import datetime

try:
    with open('$LATEST_REPORT', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print('\\n' + '='*60)
    print('测试报告分析')
    print('='*60)
    
    session = data['test_session']
    print(f\"测试时间: {session['timestamp']}\")
    print(f\"总测试数: {session['total_tests']}\")
    print(f\"通过测试: {session['passed_tests']}\")
    print(f\"失败测试: {session['failed_tests']}\")
    
    success_rate = session['passed_tests'] / session['total_tests'] * 100
    print(f\"成功率: {success_rate:.1f}%\")
    
    # 分析具体结果
    print('\\n详细结果:')
    for result in data['test_results']:
        status = '✅' if result['success'] else '❌'
        print(f\"  {status} {result['test_name']}\")
        
        if result['success'] and 'phases' in result:
            if 'planning' in result['phases']:
                p = result['phases']['planning']
                if 'strategy' in p and 'waypoint_count' in p:
                    print(f\"      规划: {p['strategy']}, {p['waypoint_count']}航点\")
            
            if 'execution' in result['phases']:
                e = result['phases']['execution']
                if 'total_estimated_time' in e:
                    print(f\"      执行: 预计{e['total_estimated_time']:.1f}s\")
        elif not result['success']:
            print(f\"      错误: {result.get('error', '未知错误')}\")
    
    # 性能统计
    if 'summary' in data and 'planning_performance' in data['summary']:
        perf = data['summary']['planning_performance']
        print('\\n性能统计:')
        print(f\"  规划平均耗时: {perf.get('avg_calculation_time_ms', 0):.1f}ms\")
        print(f\"  规划成功率: {perf.get('success_rate', 0)*100:.1f}%\")
    
    print('='*60)
    
except Exception as e:
    print(f'分析报告时出错: {e}', file=sys.stderr)
    sys.exit(1)
"
    
    log_success "报告分析完成"
}

# 显示使用说明
show_help() {
    cat << EOF
扫描系统测试脚本

使用方法:
  $0 [模式] [选项]

模式:
  auto        - 运行自动化测试套件
  interactive - 运行交互式测试
  benchmark   - 运行性能基准测试  
  analyze     - 分析最新的测试报告
  cleanup     - 清理系统节点
  help        - 显示此帮助信息

选项:
  --no-nodes  - 不启动系统节点 (假设节点已运行)
  --keep      - 测试完成后保持节点运行

示例:
  $0 auto                    # 运行自动化测试
  $0 interactive --keep      # 交互式测试并保持节点
  $0 benchmark --no-nodes    # 基准测试(节点已运行)  
  $0 analyze                 # 分析测试报告
  $0 cleanup                 # 清理节点

EOF
}

# 主函数
main() {
    local mode=${1:-"help"}
    local no_nodes=false
    local keep_nodes=false
    
    # 解析选项
    while [[ $# -gt 0 ]]; do
        case $1 in
            --no-nodes)
                no_nodes=true
                shift
                ;;
            --keep)
                keep_nodes=true
                shift
                ;;
            *)
                if [ -z "${mode##*help*}" ] || [ -z "${mode##*auto*}" ] || [ -z "${mode##*interactive*}" ] || [ -z "${mode##*benchmark*}" ] || [ -z "${mode##*analyze*}" ] || [ -z "${mode##*cleanup*}" ]; then
                    shift
                else
                    log_error "未知选项: $1"
                    show_help
                    exit 1
                fi
                ;;
        esac
    done
    
    # 设置清理陷阱
    if [ "$keep_nodes" = false ]; then
        trap cleanup_nodes EXIT
    fi
    
    case $mode in
        "help")
            show_help
            ;;
        "cleanup")
            cleanup_nodes
            ;;
        "analyze")
            analyze_reports
            ;;
        "auto"|"interactive"|"benchmark")
            if [ "$no_nodes" = false ]; then
                start_system_nodes
            fi
            
            run_test $mode
            
            if [ $? -eq 0 ]; then
                log_success "测试完成!"
                
                # 如果是自动化测试，自动显示分析
                if [ "$mode" = "auto" ]; then
                    log_info "自动分析测试结果..."
                    analyze_reports
                fi
            else
                log_error "测试失败!"
                exit 1
            fi
            ;;
        *)
            log_error "未知模式: $mode"
            show_help
            exit 1
            ;;
    esac
}

# 检查必要文件
if [ ! -f "/home/qi/ros2_ws/src/vision_ai/vision_ai/scan_system_tester.py" ]; then
    log_error "找不到 /home/qi/ros2_ws/src/vision_ai/vision_ai/scan_system_tester.py 文件"
    log_info "请确保在正确的目录中运行此脚本"
    exit 1
fi

# 运行主函数
main "$@"