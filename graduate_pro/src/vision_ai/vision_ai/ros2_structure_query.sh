#!/bin/bash

echo "=== ROS2 项目结构查询 ==="
echo

# 1. 查看整体项目结构
echo "📁 项目整体结构:"
echo "=================="
find . -type d -name "vision*" -o -name "*vision*" | head -20
echo

# 2. 查看当前所有ROS2包
echo "📦 当前ROS2包列表:"
echo "=================="
find . -name "package.xml" -exec dirname {} \; | sort
echo

# 3. 查看vision相关的包结构
echo "👁️ Vision相关包结构:"
echo "===================="
find . -path "*/vision*" -type f -name "*.py" | head -20
echo

# 4. 查看现有的ROS2节点
echo "🤖 现有ROS2节点:"
echo "==============="
find . -name "*.py" -exec grep -l "class.*Node\|rclpy.init\|Node(" {} \; | head -25
echo

# 5. 查看现有的话题定义
echo "📡 现有话题和消息类型:"
echo "===================="
find . -name "*.py" -exec grep -h "create_publisher\|create_subscription" {} \; | head -120
echo

# 6. 查看launch文件
echo "🚀 Launch文件:"
echo "============="
find . -name "*.launch.py" -o -name "*.launch" | head -20
echo

# 7. 查看配置文件
echo "⚙️ 配置文件:"
echo "============"
find . -name "*.yaml" -o -name "*.yml" -o -name "config.py" | head -10
echo

# 8. 查看CMakeLists和package.xml
echo "🔧 构建文件:"
echo "============"
find . -name "CMakeLists.txt" -o -name "package.xml" | head -10
echo

# 9. 查看当前工作空间结构
echo "🏗️ 工作空间结构:"
echo "================"
ls -la
echo

# 10. 查看src目录结构
echo "📂 src目录结构:"
echo "=============="
if [ -d "src" ]; then
    tree src -d -L 3 2>/dev/null || find src -type d | head -20
fi
echo

echo "=== 附加信息查询 ==="
echo

# 11. 查看ROS2相关的Python导入
echo "🐍 Python导入信息:"
echo "=================="
find . -name "*.py" -exec grep -h "^import\|^from.*import" {} \; | grep -E "rclpy|std_msgs|sensor_msgs|geometry_msgs" | sort | uniq | head -10
echo

# 12. 查看现有的服务和动作
echo "🔧 服务和动作:"
echo "============="
find . -name "*.py" -exec grep -h "create_service\|create_client\|create_action" {} \; | head -30
echo

# 13. 查看参数配置
echo "📋 参数配置:"
echo "============"
find . -name "*.py" -exec grep -h "declare_parameter\|get_parameter" {} \; | head -10
echo

echo "请将以上输出结果提供给我，我会基于此设计tracking系统的代码结构。"