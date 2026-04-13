# detection/utils/config_manager.py
import os
import yaml
from typing import Dict, Any

class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_file: str = None):
        """
        初始化配置管理器
        
        Args:
            config_file: 配置文件路径
        """
        self.config_file = config_file
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        if self.config_file and os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                print(f"[CONFIG] 从文件加载配置: {self.config_file}")
                self._resolve_model_paths(config)
                return config
            except Exception as e:
                print(f"[CONFIG] 配置文件加载失败: {e}")
        
        # 返回默认配置
        return self._get_default_config()

    def _resolve_model_paths(self, config: Dict[str, Any]) -> None:
        """
        将相对 model_path / checkpoint 解析为绝对路径。

        约定：JSON 里写 `models/yolo/...`、`models/sam2/...` 时，锚点为 **内层 vision_ai 包根目录**
        （即 `.../vision_ai/vision_ai/`，与 `detection/` 同级），而不是 `detection/config/`。
        这样权重放在 `vision_ai/models/` 即可，与 download_vision_weights.py 一致。
        """
        if not self.config_file:
            return
        config_dir = os.path.dirname(os.path.abspath(self.config_file))
        # enhanced_detection_config.json 位于 .../detection/config/
        package_root = os.path.dirname(os.path.dirname(config_dir))

        det = config.get("detector") or {}
        mp = det.get("model_path", "")
        if mp and not os.path.isabs(mp):
            det["model_path"] = os.path.normpath(os.path.join(package_root, mp))
            config["detector"] = det

        seg = config.get("segmentor") or {}
        for key in ("checkpoint_path", "checkpoint"):
            ck = seg.get(key)
            if ck and not os.path.isabs(ck):
                seg[key] = os.path.normpath(os.path.join(package_root, ck))
        config["segmentor"] = seg
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        # 获取模型路径
        home_dir = os.path.expanduser("~")
        model_dir = os.path.join(home_dir, "ros2_ws", "src", "vision_ai", "models")
        
        return {
            'detector': {
                'type': 'yolo',
                'model_path': '/home/qi/下载/best2.pt',
                'confidence_threshold': 0.7
            },
            'segmentor': {
                'type': 'sam2',
                'checkpoint': os.path.expanduser(os.path.join(model_dir, 'sam2', 'sam2_hiera_large.pt')),
                'config': 'sam2_hiera_l.yaml',
                'device': 'cuda'
            },
            'features': {
                'color': {
                    'histogram_bins': 64
                },
                'shape': {
                    'enabled': True
                },
                'spatial': {
                    'enabled': True
                }
            },
            'camera': {
                'intrinsics': {
                    'fx': 912.694580078125,
                    'fy': 910.309814453125,
                    'cx': 640,
                    'cy': 360
                }
            },
            'class_names': {
                1: 'banana',
                2: 'carrot', 
                3: 'corn',
                4: 'lemon',
                5: 'greenlemon',
                6: 'strawberry',
                7: 'tomato',
                8: 'potato',
                9: 'redpepper'
            }
        }
    
    def get_detector_config(self) -> Dict[str, Any]:
        """获取检测器配置"""
        return self.config.get('detector', {})
    
    def get_segmentor_config(self) -> Dict[str, Any]:
        """获取分割器配置"""
        return self.config.get('segmentor', {})
    
    def get_features_config(self) -> Dict[str, Any]:
        """获取特征提取配置"""
        return self.config.get('features', {})
    
    def get_camera_config(self) -> Dict[str, Any]:
        """获取相机配置"""
        return self.config.get('camera', {})
    
    def get_class_names(self) -> Dict[int, str]:
        """获取类别名称映射"""
        return self.config.get('class_names', {})
    
    def save_config(self, output_path: str):
        """保存配置到文件"""
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
            print(f"[CONFIG] 配置已保存到: {output_path}")
        except Exception as e:
            print(f"[CONFIG] 配置保存失败: {e}")
    
    def update_config(self, updates: Dict[str, Any]):
        """更新配置"""
        def deep_update(base_dict, update_dict):
            for key, value in update_dict.items():
                if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                    deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value
        
        deep_update(self.config, updates)
        print(f"[CONFIG] 配置已更新")
    
    def get_model_paths(self) -> Dict[str, str]:
        """获取模型路径信息"""
        detector_config = self.get_detector_config()
        segmentor_config = self.get_segmentor_config()
        
        return {
            'yolo_model_path': detector_config.get('model_path', ''),
            'sam2_checkpoint': segmentor_config.get('checkpoint', ''),
            'sam2_config': segmentor_config.get('config', '')
        }
    
    def validate_config(self) -> bool:
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
            return False