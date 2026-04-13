#!/bin/bash

YELLOW='\033[1;33m'
GREEN='\033[0;32m'
NC='\033[0m'

echo -e "${YELLOW}🧹 Cleaning up vision AI system...${NC}"

# 杀死所有相关进程
pkill -f "scan_planner_node\|gui_config_node\|scan_executor_node\|smart_stitcher_node\|detection_node" 2>/dev/null
tmux kill-session -t vision_ai 2>/dev/null

# 清理编译缓存
cd ~/ros2_ws
rm -rf build/vision_ai install/vision_ai log/

# 🆕 清理扫描输出目录（可选）
if [ "$1" = "full" ]; then
    echo -e "${YELLOW}🗑️  Cleaning scan output directories...${NC}"
    rm -rf scan_output_* 2>/dev/null
    echo -e "${GREEN}✅ Scan outputs cleaned!${NC}"
fi

echo -e "${GREEN}✅ Cleanup complete!${NC}"
echo -e "${GREEN}🚀 Ready to restart with: ./restart.sh${NC}"