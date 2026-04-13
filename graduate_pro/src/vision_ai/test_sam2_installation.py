#!/usr/bin/env python3
# test_sam2_installation_fixed.py - 修正的SAM2测试

import os
import sys
import torch
import numpy as np

def test_sam2_installation():
    """测试SAM2安装和模型加载"""
    
    print("测试SAM2安装...")
    print(f"Python版本: {sys.version}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA设备: {torch.cuda.get_device_name()}")
    
    try:
        # 测试导入SAM2
        print("\n1. 测试SAM2导入...")
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        print("✅ SAM2库导入成功")
        
        # 测试模型文件
        print("\n2. 检查模型文件...")
        model_dir = os.path.expanduser("~/ros2_ws/src/vision_ai/models/sam2")
        checkpoint_path = os.path.join(model_dir, "sam2_hiera_large.pt")
        
        if os.path.exists(checkpoint_path):
            size_mb = os.path.getsize(checkpoint_path) / 1024 / 1024
            print(f"✅ 检查点文件存在: {checkpoint_path} ({size_mb:.1f}MB)")
        else:
            print(f"❌ 检查点文件不存在: {checkpoint_path}")
            return False
        
        # 测试模型加载 - 使用相对配置名称
        print("\n3. 测试模型加载...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 方法1: 使用预定义的配置名称
        try:
            print("尝试使用预定义配置名称...")
            config_name = "sam2_hiera_l.yaml"  # 相对路径，让Hydra自动搜索
            model = build_sam2(config_name, checkpoint_path, device=device)
            predictor = SAM2ImagePredictor(model)
            print(f"✅ SAM2模型加载成功 (设备: {device})")
            
        except Exception as e1:
            print(f"❌ 预定义配置加载失败: {e1}")
            
            # 方法2: 尝试其他可能的配置名称
            try:
                print("尝试使用sam2/sam2_hiera_l.yaml...")
                config_name = "sam2/sam2_hiera_l.yaml"
                model = build_sam2(config_name, checkpoint_path, device=device)
                predictor = SAM2ImagePredictor(model)
                print(f"✅ SAM2模型加载成功 (配置: {config_name})")
                
            except Exception as e2:
                print(f"❌ sam2/配置加载失败: {e2}")
                
                # 方法3: 尝试不带.yaml后缀
                try:
                    print("尝试使用sam2_hiera_l...")
                    config_name = "sam2_hiera_l"
                    model = build_sam2(config_name, checkpoint_path, device=device)
                    predictor = SAM2ImagePredictor(model)
                    print(f"✅ SAM2模型加载成功 (配置: {config_name})")
                    
                except Exception as e3:
                    print(f"❌ 所有配置方法都失败了")
                    print(f"错误1: {e1}")
                    print(f"错误2: {e2}")
                    print(f"错误3: {e3}")
                    return False
        
        # 测试简单推理
        print("\n4. 测试简单推理...")
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        predictor.set_image(test_image)
        
        test_box = np.array([100, 100, 200, 200])
        mask, _, _ = predictor.predict(box=test_box, multimask_output=False)
        
        print(f"✅ 推理测试成功，输出mask形状: {mask.shape}")
        
        print("\n🎉 SAM2安装和配置完全正常！")
        return True
        
    except ImportError as e:
        print(f"❌ SAM2导入失败: {e}")
        print("请确保已正确安装SAM2:")
        print("pip install git+https://github.com/facebookresearch/segment-anything-2.git")
        return False
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        return False

def show_sam2_config_info():
    """显示SAM2配置信息"""
    try:
        import sam2
        sam2_path = os.path.dirname(sam2.__file__)
        print(f"SAM2包路径: {sam2_path}")
        
        # 查找配置目录
        possible_config_dirs = [
            os.path.join(sam2_path, "configs"),
            os.path.join(sam2_path, "configs", "sam2"),
            os.path.join(sam2_path, "sam2_configs")
        ]
        
        print("\n查找配置目录:")
        for config_dir in possible_config_dirs:
            if os.path.exists(config_dir):
                print(f"✅ 找到配置目录: {config_dir}")
                configs = [f for f in os.listdir(config_dir) if f.endswith('.yaml')]
                if configs:
                    print(f"   可用配置: {configs}")
            else:
                print(f"❌ 配置目录不存在: {config_dir}")
        
    except Exception as e:
        print(f"❌ 获取SAM2配置信息失败: {e}")

if __name__ == "__main__":
    print("=== SAM2配置信息 ===")
    show_sam2_config_info()
    
    print("\n=== SAM2功能测试 ===")
    success = test_sam2_installation()
    sys.exit(0 if success else 1)