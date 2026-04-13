#!/usr/bin/env python3
# comprehensive_fix.py - 全面修复检测模块问题

import os
import sys
import numpy as np
import cv2
from pathlib import Path

def test_opencv_histogram():
    """测试OpenCV直方图计算"""
    print("=== 测试OpenCV直方图计算 ===")
    
    try:
        # 创建测试数据
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        test_mask = np.zeros((100, 100), dtype=np.uint8)
        test_mask[20:80, 30:70] = 255
        
        test_image = np.ascontiguousarray(test_image)
        test_mask = np.ascontiguousarray(test_mask)

        print(f"图像形状: {test_image.shape}, 类型: {test_image.dtype}")
        print(f"掩码形状: {test_mask.shape}, 类型: {test_mask.dtype}")
        print(f"图像连续性: {test_image.flags['C_CONTIGUOUS']}")
        print(f"掩码连续性: {test_mask.flags['C_CONTIGUOUS']}")

        # 推荐方式：使用整图+通道索引
        hist_r = cv2.calcHist([test_image], [0], test_mask, [64], [0, 256])
        print(f"✅ 方法1成功 - 使用原图第0通道: {hist_r.shape}")
        
        hist_rgb = cv2.calcHist([test_image], [0, 1, 2], test_mask, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        print(f"✅ 方法2成功 - 多通道联合直方图: {hist_rgb.shape}")

        return True

    except Exception as e:
        print(f"❌ OpenCV测试失败: {e}")
        return False


def update_color_features_file():
    """更新颜色特征文件"""
    print("\n=== 更新颜色特征文件 ===")
    
    color_features_path = Path.home() / "ros2_ws/src/vision_ai/vision_ai/detection/features/color_features.py"
    
    if not color_features_path.exists():
        print(f"❌ 文件不存在: {color_features_path}")
        return False
    
    # 新的compute_color_histogram方法
    new_method = '''    def compute_color_histogram(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        计算mask区域的颜色直方图
        
        Args:
            image: RGB图像 (H, W, 3)
            mask: 二值掩码 (H, W)
            
        Returns:
            hist: 归一化的颜色直方图 (bins*3,)
        """
        if np.sum(mask) == 0:
            return np.zeros((self.bins * 3,))
        
        # 确保image是正确的数据类型和内存布局
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        
        # 确保image是连续的内存布局
        if not image.flags['C_CONTIGUOUS']:
            image = np.ascontiguousarray(image)
        
        # 确保mask是正确的数据类型
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)
        
        # 创建mask用于calcHist（必须是uint8，值为0或255）
        mask_uint8 = np.zeros_like(mask, dtype=np.uint8)
        mask_uint8[mask > 0] = 255
        
        # 确保mask也是连续的
        if not mask_uint8.flags['C_CONTIGUOUS']:
            mask_uint8 = np.ascontiguousarray(mask_uint8)
        
        # 分别提取每个通道，确保是连续的
        r_channel = np.ascontiguousarray(image[:, :, 0])
        g_channel = np.ascontiguousarray(image[:, :, 1])
        b_channel = np.ascontiguousarray(image[:, :, 2])
        
        # 计算每个通道的直方图
        hist_r = cv2.calcHist([r_channel], [0], mask_uint8, [self.bins], [0, 256])
        hist_g = cv2.calcHist([g_channel], [0], mask_uint8, [self.bins], [0, 256])
        hist_b = cv2.calcHist([b_channel], [0], mask_uint8, [self.bins], [0, 256])
        
        # 合并直方图
        hist = np.concatenate([hist_r.flatten(), hist_g.flatten(), hist_b.flatten()])
        
        # 归一化
        hist = hist / (np.sum(hist) + 1e-10)
        
        return hist'''
    
    try:
        # 读取原文件
        with open(color_features_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 查找并替换方法
        import re
        
        # 查找compute_color_histogram方法
        pattern = r'    def compute_color_histogram\(self, image: np\.ndarray, mask: np\.ndarray\) -> np\.ndarray:.*?(?=    def |\n    @|\nclass |\Z)'
        
        if re.search(pattern, content, re.DOTALL):
            content = re.sub(pattern, new_method, content, flags=re.DOTALL)
            
            # 写回文件
            with open(color_features_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print("✅ 颜色特征文件已更新")
            return True
        else:
            print("❌ 未找到compute_color_histogram方法")
            return False
            
    except Exception as e:
        print(f"❌ 更新颜色特征文件失败: {e}")
        return False

def update_config_manager_file():
    """更新配置管理器文件"""
    print("\n=== 更新配置管理器文件 ===")
    
    config_manager_path = Path.home() / "ros2_ws/src/vision_ai/vision_ai/detection/utils/config_manager.py"
    
    if not config_manager_path.exists():
        print(f"❌ 文件不存在: {config_manager_path}")
        return False
    
    # 新的validate_config方法
    new_method = '''    def validate_config(self) -> bool:
        """验证配置的有效性"""
        try:
            # 检查必要的配置项
            required_sections = ['detector', 'segmentor', 'camera', 'class_names']
            for section in required_sections:
                if section not in self.config:
                    print(f"[CONFIG] 缺少必要配置段: {section}")
                    return False
            
            # 检查关键模型文件是否存在
            detector_config = self.get_detector_config()
            yolo_path = detector_config.get('model_path', '')
            
            segmentor_config = self.get_segmentor_config()
            sam2_checkpoint = segmentor_config.get('checkpoint', '')
            
            # 只检查重要的文件路径
            critical_files = {
                'yolo_model': yolo_path,
                'sam2_checkpoint': sam2_checkpoint
            }
            
            for name, path in critical_files.items():
                if path and not os.path.exists(path):
                    print(f"[CONFIG] 关键模型文件不存在: {name} = {path}")
                    return False
            
            print(f"[CONFIG] 配置验证通过")
            return True
            
        except Exception as e:
            print(f"[CONFIG] 配置验证失败: {e}")
            return False'''
    
    try:
        # 读取原文件
        with open(config_manager_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 查找并替换方法
        import re
        
        pattern = r'    def validate_config\(self\) -> bool:.*?(?=    def |\n    @|\nclass |\Z)'
        
        if re.search(pattern, content, re.DOTALL):
            content = re.sub(pattern, new_method, content, flags=re.DOTALL)
            
            # 写回文件
            with open(config_manager_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print("✅ 配置管理器文件已更新")
            return True
        else:
            print("❌ 未找到validate_config方法")
            return False
            
    except Exception as e:
        print(f"❌ 更新配置管理器文件失败: {e}")
        return False

def main():
    """主函数"""
    print("开始全面修复检测模块...")
    
    # 测试OpenCV
    opencv_ok = test_opencv_histogram()
    
    if not opencv_ok:
        print("❌ OpenCV测试失败，无法继续修复")
        return False
    
    # 更新文件
    fixes = [
        update_color_features_file,
        update_config_manager_file
    ]
    
    results = []
    for fix in fixes:
        result = fix()
        results.append(result)
    
    if all(results):
        print("\n🎉 所有修复都完成了！")
        print("请重新运行测试: python test_detection_modules.py")
        return True
    else:
        print("\n❌ 部分修复失败")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)