#!/bin/bash
# detection_monitor.sh - 检测结果监控工具

source ~/ros2_ws/install/setup.bash

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🔍 Vision AI 检测结果监控器${NC}"
echo -e "${BLUE}================================${NC}"

if [ "$1" = "live" ]; then
    echo -e "${YELLOW}📡 实时监控检测结果...${NC}"
    echo -e "${GREEN}按 Ctrl+C 退出${NC}"
    echo ""
    ros2 topic echo /detection_result | while read line; do
        if [[ $line == *"data:"* ]]; then
            echo -e "${GREEN}[$(date '+%H:%M:%S')] 🎯 检测结果:${NC}"
            echo "$line" | python3 -c "
import sys, json
try:
    data = json.loads(sys.stdin.read().split('data: ')[1].strip('\\'\"'))
    print(f'  📊 检测到 {data.get(\"detection_count\", 0)} 个对象')
    for i, obj in enumerate(data.get('objects', []), 1):
        print(f'  {i}. {obj.get(\"description\", \"N/A\")}')
except:
    pass
"
            echo ""
        fi
    done

elif [ "$1" = "latest" ]; then
    echo -e "${YELLOW}📄 获取最新检测结果...${NC}"
    ros2 topic echo /detection_result --once | python3 -c "
import sys, json
try:
    content = sys.stdin.read()
    if 'data:' in content:
        data_str = content.split('data: ')[1].strip()
        data = json.loads(data_str.strip('\"'))
        print(f'🎯 检测摘要: {data.get(\"detection_count\", 0)} 个对象')
        print(f'⏱️  处理时间: {data.get(\"processing_time\", 0):.2f}s')
        print(f'📁 输出目录: {data.get(\"output_directory\", \"N/A\")}')
        print('\\n🧬 检测对象:')
        for i, obj in enumerate(data.get('objects', []), 1):
            print(f'  {i}. {obj.get(\"description\", \"N/A\")}')
            print(f'     ID: {obj.get(\"object_id\", \"N/A\")}')
            print(f'     置信度: {obj.get(\"confidence\", 0):.3f}')
            if 'world_x' in obj:
                print(f'     世界坐标: ({obj.get(\"world_x\", 0):.1f}, {obj.get(\"world_y\", 0):.1f}, {obj.get(\"world_z\", 0):.1f})')
    else:
        print('❌ 没有检测结果数据')
except Exception as e:
    print(f'❌ 解析检测结果失败: {e}')
"

elif [ "$1" = "topics" ]; then
    echo -e "${YELLOW}📡 检查相关话题状态...${NC}"
    echo ""
    echo -e "${GREEN}可用话题:${NC}"
    ros2 topic list | grep -E "(detection|stitching|scan)" | while read topic; do
        echo -e "  📢 $topic"
    done
    echo ""
    echo -e "${GREEN}节点状态:${NC}"
    ros2 node list | grep -E "(detection|stitcher|scan)" | while read node; do
        echo -e "  🤖 $node"
    done

else
    echo -e "${GREEN}用法: ./detection_monitor.sh [live|latest|topics]${NC}"
    echo ""
    echo -e "${YELLOW}选项:${NC}"
    echo -e "  ${GREEN}live${NC}    - 实时监控检测结果"
    echo -e "  ${GREEN}latest${NC}  - 获取最新检测结果"
    echo -e "  ${GREEN}topics${NC}  - 检查话题和节点状态"
    echo ""
    echo -e "${BLUE}示例:${NC}"
    echo -e "  ./detection_monitor.sh live     # 实时监控"
    echo -e "  ./detection_monitor.sh latest   # 查看最新结果"
    echo -e "  ./detection_monitor.sh topics   # 检查系统状态"
fi