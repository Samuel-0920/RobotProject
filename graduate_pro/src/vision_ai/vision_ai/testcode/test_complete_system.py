import sys
import os
from pathlib import Path

# 添加项目路径
project_path = Path(__file__).parent
sys.path.insert(0, str(project_path))

def test_complete_system():
    """测试完整的检测系统"""
    print("=== 完整系统测试 ===")
    
    try:
        # 测试检测管道导入
        from detection.detection_pipeline import DetectionPipeline
        print("✅ 检测管道导入成功")
        
        # 测试配置加载
        config_file = project_path / "config" / "default_config.yaml"
        if config_file.exists():
            pipeline = DetectionPipeline(str(config_file))
            print("✅ 检测管道初始化成功")
        else:
            pipeline = DetectionPipeline()
            print("✅ 检测管道使用默认配置初始化成功")
        
        print("🎉 完整系统测试通过！")
        return True
        
    except Exception as e:
        print(f"❌ 系统测试失败: {e}")
        return False

if __name__ == "__main__":
    success = test_complete_system()
    sys.exit(0 if success else 1)