#!/bin/bash

# SAM2模型下载脚本
# 保存为：~/ros2_ws/src/vision_ai/scripts/download_sam2_models.sh

echo "开始下载SAM2模型文件..."

# 创建模型目录
mkdir -p ~/ros2_ws/src/vision_ai/models/sam2

cd ~/ros2_ws/src/vision_ai/models/sam2

# 下载SAM2模型检查点
echo "下载 SAM2 Hiera Large 模型..."
wget -O sam2_hiera_large.pt https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt

# 下载配置文件
echo "下载配置文件..."
wget -O sam2_hiera_l.yaml https://raw.githubusercontent.com/facebookresearch/segment-anything-2/main/sam2_configs/sam2_hiera_l.yaml

# 可选：下载其他大小的模型
echo "下载 SAM2 Hiera Base 模型 (可选)..."
wget -O sam2_hiera_base_plus.pt https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt
wget -O sam2_hiera_b_plus.yaml https://raw.githubusercontent.com/facebookresearch/segment-anything-2/main/sam2_configs/sam2_hiera_b%2B.yaml

echo "模型下载完成！"
echo "文件位置：~/ros2_ws/src/vision_ai/models/sam2/"
ls -la ~/ros2_ws/src/vision_ai/models/sam2/