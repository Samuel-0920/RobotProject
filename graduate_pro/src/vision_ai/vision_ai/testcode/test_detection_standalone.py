#!/usr/bin/env python3
# test_detection_standalone.py - 独立测试检测节点

import os
import sys
import cv2
import numpy as np
from pathlib import Path

# 添加项目路径
project_path = Path(__file__).parent
sys.path.insert(0, str(project_path))

def test_detection_pipeline_standalone():
    """独立测试检测管道"""
    try:
        print("🔍 测试检测管道独立功能...")
        
        # 导入检测管道
        from detection.detection_pipeline import DetectionPipeline
        
        # 创建检测管道
        pipeline = DetectionPipeline()
        print("✅ 检测管道创建成功")
        
        # 创建测试图像
        test_image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        test_depth = np.random.randint(500, 1500, (720, 1280), dtype=np.uint16)
        
        print("🖼️ 创建测试图像完成")
        
        # 测试检测流程
        result = pipeline.process_reference_image(
            test_image, 
            test_depth, 
            generate_visualization=True
        )
        
        if result['success']:
            print(f"✅ 检测测试成功！")
            print(f"   检测到 {result['detection_count']} 个目标")
            print(f"   处理时间: {result['processing_time']:.2f}秒")
            print(f"   输出目录: {pipeline.output_dir}")
        else:
            print(f"⚠️ 检测测试完成，但未检测到目标")
            print(f"   消息: {result.get('message', '无')}")
        
        return True
        
    except Exception as e:
        print(f"❌ 检测管道测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_detection_node_import():
    """测试detection_node导入"""
    try:
        print("📦 测试detection_node导入...")
        
        # 测试导入
        import detection_node
        print("✅ detection_node导入成功")
        
        return True
        
    except Exception as e:
        print(f"❌ detection_node导入失败: {e}")
        print("请检查文件是否正确保存")
        return False

def main():
    print("🧪 开始独立检测系统测试...")
    
    # 测试1：检测管道
    test1 = test_detection_pipeline_standalone()
    
    # 测试2：检测节点导入
    test2 = test_detection_node_import()
    
    if test1 and test2:
        print("\n🎉 所有测试通过！检测系统准备就绪")
        print("\n📋 下一步:")
        print("1. 修复vision_ai包构建错误")
        print("2. 更新vision_ai_interfaces包添加新消息")
        print("3. 修改smart_stitcher_node添加发布功能")
        return True
    else:
        print("\n❌ 部分测试失败，请检查上述错误")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)