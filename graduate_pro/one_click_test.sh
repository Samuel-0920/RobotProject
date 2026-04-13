#!/bin/bash
set -e

echo "🎯 一键测试追踪系统"
echo "=================="

# 1. 创建__init__.py文件
echo "📁 创建__init__.py文件..."
touch ~/ros2_ws/src/vision_ai/vision_ai/tracking_system/__init__.py
touch ~/ros2_ws/src/vision_ai/vision_ai/tracking_system/utils/__init__.py
touch ~/ros2_ws/src/vision_ai/vision_ai/tracking_system/core/__init__.py
touch ~/ros2_ws/src/vision_ai/vision_ai/tracking_system/detection/__init__.py
echo "✅ __init__.py文件创建完成"

# 2. 设置Python路径
echo "🔧 设置Python路径..."
export PYTHONPATH=$PYTHONPATH:~/ros2_ws/src/vision_ai/vision_ai
echo "✅ Python路径设置完成"

# 3. 切换到工作空间
echo "📂 切换到工作空间..."
cd ~/ros2_ws

# 4. 检查文件是否存在
echo "🔍 检查关键文件..."
required_files=(
    "src/vision_ai/vision_ai/tracking_system/utils/config.py"
    "src/vision_ai/vision_ai/tracking_system/core/state_machine.py"
    "src/vision_ai/vision_ai/tracking_system/utils/data_structures.py"
    "src/vision_ai/vision_ai/tracking_system/detection/realtime_detector.py"
    "src/vision_ai/vision_ai/tracking_system/detection/id_matcher.py"
    "src/vision_ai/vision_ai/tracking_system/tests/quick_test.py"
)

missing_files=()
for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        missing_files+=("$file")
    fi
done

if [ ${#missing_files[@]} -ne 0 ]; then
    echo "❌ 缺少以下文件："
    printf '%s\n' "${missing_files[@]}"
    echo "请确保所有文件都已创建并放置在正确位置"
    exit 1
fi

echo "✅ 所有必需文件都存在"

# 5. 运行测试
echo "🚀 运行快速测试..."
echo "==================="
python3 src/vision_ai/vision_ai/tracking_system/tests/quick_test.py

echo ""
echo "🎉 测试完成!"
echo "============"
