#!/bin/bash

echo "🔍 分析当前ROS2架构..."
echo "=================================="

echo -e "\n📋 1. 列出所有活跃的ROS2节点:"
ros2 node list

echo -e "\n📨 2. 列出所有ROS2话题:"
ros2 topic list

echo -e "\n🔧 3. 列出所有ROS2服务:"
ros2 service list

echo -e "\n⚙️ 4. 检查xarm_controller相关的话题:"
echo "xarm_controller话题:"
ros2 topic list | grep -i xarm

echo -e "\n🤖 5. 检查机械臂相关的话题:"
echo "机械臂相关话题:"
ros2 topic list | grep -E "(arm|robot|joint|gripper)"

echo -e "\n📷 6. 检查相机/图像相关的话题:"
echo "图像相关话题:"
ros2 topic list | grep -E "(image|camera|depth)"

echo -e "\n🎯 7. 检查检测相关的话题:"
echo "检测相关话题:"
ros2 topic list | grep -E "(detection|tracking|reference)"

echo -e "\n📡 8. 获取关键话题的详细信息:"
echo "xarm_controller话题详情:"
if ros2 topic list | grep -q "/xarm_controller"; then
    ros2 topic info /xarm_controller 2>/dev/null || echo "话题不存在或无权限"
else
    echo "未找到/xarm_controller话题"
fi

echo -e "\n检测结果话题详情:"
if ros2 topic list | grep -q "/detection_result"; then
    ros2 topic info /detection_result
    echo "话题类型:"
    ros2 topic type /detection_result
else
    echo "未找到/detection_result话题"
fi

echo -e "\n📊 9. 检查参数服务器:"
echo "所有参数:"
ros2 param list 2>/dev/null | head -20

echo -e "\n🕐 10. 获取系统时间信息:"
echo "ROS2时间:"
ros2 topic echo /clock --once 2>/dev/null || echo "无法获取时间信息"

echo -e "\n🔍 11. 检查可能的gripper/夹爪相关话题:"
echo "夹爪相关话题:"
ros2 topic list | grep -E "(gripper|grip|hand|finger|clamp)"

echo -e "\n⚡ 12. 检查动作服务器 (Action Servers):"
echo "动作服务器:"
ros2 action list 2>/dev/null || echo "无动作服务器或ros2 action命令不可用"

echo -e "\n📋 13. 获取特定节点的详细信息:"
echo "detection_node详情:"
if ros2 node list | grep -q "detection_node"; then
    ros2 node info /detection_node 2>/dev/null || echo "无法获取节点信息"
else
    echo "detection_node未运行"
fi

echo -e "\n📋 14. 查看包结构:"
echo "可用的ROS2包:"
ros2 pkg list | grep -E "(xarm|vision|detection|tracking)" | head -10

echo -e "\n💻 15. 系统资源信息:"
echo "CPU核心数: $(nproc)"
echo "内存信息: $(free -h | grep Mem)"
echo "ROS_DOMAIN_ID: ${ROS_DOMAIN_ID:-未设置}"

echo -e "\n=================================="
echo "✅ ROS2架构分析完成!"