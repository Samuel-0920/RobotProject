#!/usr/bin/env python3
# setup_enhanced_config.py - 创建增强配置文件的脚本

import os
import json
import sys
from pathlib import Path

def create_default_config():
    """创建默认的增强配置"""
    config = {
        # 检测器配置
        "detector": {
            "type": "yolo",
            "model_path": "models/yolo11n.pt",
            "confidence_threshold": 0.5,
            "iou_threshold": 0.45,
            "device": "cuda",
            "classes": None
        },
        
        # 分割器配置
        "segmentor": {
            "type": "sam2",
            "model_type": "sam2_hiera_tiny",
            "device": "cuda"
        },
        
        # 相机配置（增强版）
        "camera": {
            "intrinsics": {
                "fx": 912.694580078125,
                "fy": 910.309814453125,
                "cx": 624.051574707031,
                "cy": 320.749542236328
            },
            "hand_eye_calibration": {
                "translation": [0.06637719970480799, -0.032133912794949385, 0.02259679892714925],
                "quaternion": [0.0013075170827278532, -0.0024892917521336377, 0.7106597907015502, 0.7035302095188808]
            },
            "depth_scale": 1000.0,
            "depth_range": [0.01, 2.0]
        },
        
        # 特征提取配置（新增）
        "features": {
            # 颜色特征
            "color": {
                "histogram_bins": 32,
                "color_spaces": ["RGB", "HSV"],
                "enabled": True
            },
            
            # 3D几何特征（新增）
            "geometric": {
                "enabled": True,
                "fpfh": {
                    "radius": 0.05,
                    "max_neighbors": 100,
                    "enabled": True
                },
                "pca": {
                    "enabled": True
                },
                "shape_context_3d": {
                    "bins_r": 5,
                    "bins_theta": 8,
                    "bins_phi": 6,
                    "enabled": True
                },
                "density_features": {
                    "k_neighbors": 10,
                    "enabled": True
                }
            },
            
            # 2D形状特征
            "shape": {
                "enabled": True,
                "hu_moments": {
                    "enabled": True,
                    "use_robust": True,
                    "smooth_kernel_size": 7
                },
                "fourier_descriptors": {
                    "enabled": True,
                    "n_descriptors": 8
                },
                "contour_features": {
                    "enabled": True
                }
            },
            
            # 空间特征
            "spatial": {
                "enabled": True,
                "region_grid": [3, 3],
                "include_3d_coords": True
            }
        },
        
        # 相似度计算配置（新增）
        "similarity": {
            "feature_weights": {
                "geometric": 0.4,
                "appearance": 0.2,
                "shape": 0.3,
                "spatial": 0.1
            },
            "geometric_weights": {
                "fpfh": 0.4,
                "pca": 0.3,
                "bbox": 0.2,
                "shape_context_3d": 0.1
            },
            "thresholds": {
                "overall_match": 0.75,
                "high_confidence": 0.85,
                "low_confidence": 0.60
            }
        },
        
        # 自适应学习配置（新增）
        "adaptive_learning": {
            "enabled": True,
            "learning_rate": 0.1,
            "min_samples_for_optimization": 50,
            "optimization_interval": 100,
            "data_retention_days": 30,
            "auto_save": True,
            "learning_data_file": "data/adaptive_learning.json"
        },
        
        # 输出配置
        "output": {
            "save_visualizations": True,
            "save_features": True,
            "save_masks": True,
            "visualization_quality": "high",
            "feature_format": "json"
        },
        
        # 性能配置
        "performance": {
            "point_cloud_sampling": 5,
            "max_features_per_object": 1000,
            "parallel_processing": False,
            "memory_optimization": True
        },
        
        # 调试配置
        "debug": {
            "enabled": False,
            "verbose_logging": False,
            "save_intermediate_results": False,
            "timing_analysis": False
        }
    }
    
    return config

def setup_directories_and_config():
    """设置目录结构和配置文件"""
    try:
        print("🔧 开始设置增强检测配置...")
        
        # 找到当前脚本的位置
        script_dir = Path(__file__).parent
        print(f"📁 脚本位置: {script_dir}")
        
        # 尝试找到vision_ai目录
        vision_ai_dir = None
        
        # 方法1: 检查是否在vision_ai包内
        if (script_dir / "detection").exists():
            vision_ai_dir = script_dir
            print(f"✅ 找到vision_ai目录 (方法1): {vision_ai_dir}")
        
        # 方法2: 检查父目录
        elif (script_dir.parent / "vision_ai" / "detection").exists():
            vision_ai_dir = script_dir.parent / "vision_ai"
            print(f"✅ 找到vision_ai目录 (方法2): {vision_ai_dir}")
        
        # 方法3: 检查当前目录的子目录
        elif (script_dir / "vision_ai" / "detection").exists():
            vision_ai_dir = script_dir / "vision_ai"
            print(f"✅ 找到vision_ai目录 (方法3): {vision_ai_dir}")
        
        if not vision_ai_dir:
            print("❌ 未找到vision_ai目录，请确认脚本位置")
            print("💡 请将此脚本放在以下位置之一:")
            print("   1. vision_ai/detection/ 目录内")
            print("   2. vision_ai/ 目录内") 
            print("   3. 包含vision_ai/的父目录内")
            return False
        
        # 创建必要的目录
        config_dir = vision_ai_dir / "detection" / "config"
        data_dir = vision_ai_dir / "detection" / "data"
        
        config_dir.mkdir(parents=True, exist_ok=True)
        data_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 创建配置目录: {config_dir}")
        print(f"📁 创建数据目录: {data_dir}")
        
        # 创建配置文件
        config_file = config_dir / "enhanced_detection_config.json"
        config_data = create_default_config()
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 配置文件已创建: {config_file}")
        
        # 创建自适应学习数据文件的占位符
        learning_file = data_dir / "adaptive_learning.json"
        if not learning_file.exists():
            with open(learning_file, 'w', encoding='utf-8') as f:
                json.dump({}, f)
            print(f"✅ 学习数据文件已创建: {learning_file}")
        
        # 显示配置摘要
        print(f"\n{'='*50}")
        print("🚀 增强检测系统配置摘要")
        print(f"{'='*50}")
        print(f"检测器: {config_data['detector']['type']}")
        print(f"分割器: {config_data['segmentor']['type']}")
        print("\n启用的特征:")
        for feature_type, feature_config in config_data['features'].items():
            if feature_config.get('enabled', True):
                print(f"  ✅ {feature_type}")
            else:
                print(f"  ❌ {feature_type}")
        
        print(f"\n特征权重:")
        for feature, weight in config_data['similarity']['feature_weights'].items():
            print(f"  {feature}: {weight:.1f}")
        
        adaptive_enabled = config_data['adaptive_learning']['enabled']
        print(f"\n自适应学习: {'✅ 启用' if adaptive_enabled else '❌ 禁用'}")
        
        print(f"\n🎉 设置完成！现在可以使用增强检测系统了。")
        print(f"📄 配置文件位置: {config_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ 设置失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_basic_imports():
    """测试基础导入是否正常"""
    print("\n🧪 测试基础功能...")
    
    try:
        import numpy as np
        print("✅ numpy 导入成功")
        
        import cv2
        print("✅ opencv 导入成功")
        
        try:
            import open3d as o3d
            print("✅ open3d 导入成功")
        except ImportError:
            print("❌ open3d 未安装，请运行: pip install open3d")
        
        try:
            from sklearn.neighbors import NearestNeighbors
            print("✅ scikit-learn 导入成功")
        except ImportError:
            print("❌ scikit-learn 未安装，请运行: pip install scikit-learn")
        
        print("✅ 基础依赖检查完成")
        return True
        
    except Exception as e:
        print(f"❌ 基础导入测试失败: {e}")
        return False

if __name__ == "__main__":
    print("🔧 增强检测系统配置设置工具")
    print("=" * 50)
    
    # 测试基础导入
    if not test_basic_imports():
        print("\n❌ 请先安装必要的依赖包")
        sys.exit(1)
    
    # 设置配置文件
    if setup_directories_and_config():
        print("\n🎉 配置设置成功！")
        print("\n📋 下一步:")
        print("1. 检查配置文件并根据需要调整")
        print("2. 测试增强检测功能")
        print("3. 集成到现有系统中")
    else:
        print("\n❌ 配置设置失败")
        sys.exit(1)