# utils/enhanced_config_manager.py
import json
import os
from typing import List, Dict, Any, Optional, Tuple
class EnhancedConfigManager:
    """增强的配置管理器 - 支持新的3D特征和自适应学习配置"""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        初始化配置管理器
        
        Args:
            config_file: 配置文件路径
        """
        self.config_file = config_file or "config/enhanced_detection_config.json"
        self.config = self._load_default_config()
        
        # 如果存在配置文件，加载并合并
        if os.path.exists(self.config_file):
            self.load_config()
    
    def _load_default_config(self) -> Dict[str, Any]:
        """加载默认配置"""
        return {
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
                "depth_scale": 1000.0,  # 深度图缩放因子
                "depth_range": [0.01, 2.0]  # 有效深度范围（米）
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
                    "region_grid": [3, 3],  # 3x3网格划分
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
                "optimization_interval": 100,  # 每100次匹配后优化一次
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
                "point_cloud_sampling": 5,  # 每N个像素采样一次
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
    
    def load_config(self):
        """从文件加载配置"""
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                file_config = json.load(f)
            
            # 递归合并配置
            self.config = self._merge_configs(self.config, file_config)
            print(f"[CONFIG] 配置文件加载成功: {self.config_file}")
            
        except Exception as e:
            print(f"[CONFIG] 加载配置文件失败: {e}")
            print("[CONFIG] 使用默认配置")
    
    def save_config(self):
        """保存配置到文件"""
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            
            print(f"[CONFIG] 配置已保存到: {self.config_file}")
            
        except Exception as e:
            print(f"[CONFIG] 保存配置失败: {e}")
    
    def _merge_configs(self, default: Dict, custom: Dict) -> Dict:
        """递归合并配置字典"""
        merged = default.copy()
        
        for key, value in custom.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._merge_configs(merged[key], value)
            else:
                merged[key] = value
        
        return merged
    
    def get_detector_config(self) -> Dict:
        """获取检测器配置"""
        return self.config.get("detector", {})
    
    def get_segmentor_config(self) -> Dict:
        """获取分割器配置"""
        return self.config.get("segmentor", {})
    
    def get_camera_config(self) -> Dict:
        """获取相机配置"""
        return self.config.get("camera", {})
    
    def get_features_config(self) -> Dict:
        """获取特征提取配置"""
        return self.config.get("features", {})
    
    def get_similarity_config(self) -> Dict:
        """获取相似度计算配置"""
        return self.config.get("similarity", {})
    
    def get_adaptive_learning_config(self) -> Dict:
        """获取自适应学习配置"""
        return self.config.get("adaptive_learning", {})
    
    def get_performance_config(self) -> Dict:
        """获取性能配置"""
        return self.config.get("performance", {})
    
    def get_debug_config(self) -> Dict:
        """获取调试配置"""
        return self.config.get("debug", {})
    
    def update_feature_weights(self, new_weights: Dict[str, float]):
        """更新特征权重"""
        if "similarity" not in self.config:
            self.config["similarity"] = {}
        
        self.config["similarity"]["feature_weights"] = new_weights
        
        # 自动保存
        if self.config.get("adaptive_learning", {}).get("auto_save", True):
            self.save_config()
    
    def update_thresholds(self, new_thresholds: Dict[str, float]):
        """更新相似度阈值"""
        if "similarity" not in self.config:
            self.config["similarity"] = {}
        
        self.config["similarity"]["thresholds"].update(new_thresholds)
        
        # 自动保存
        if self.config.get("adaptive_learning", {}).get("auto_save", True):
            self.save_config()
    
    def enable_feature(self, feature_type: str, sub_feature: str = None):
        """启用特定特征"""
        if feature_type in self.config["features"]:
            if sub_feature:
                if sub_feature in self.config["features"][feature_type]:
                    self.config["features"][feature_type][sub_feature]["enabled"] = True
            else:
                self.config["features"][feature_type]["enabled"] = True
    
    def disable_feature(self, feature_type: str, sub_feature: str = None):
        """禁用特定特征"""
        if feature_type in self.config["features"]:
            if sub_feature:
                if sub_feature in self.config["features"][feature_type]:
                    self.config["features"][feature_type][sub_feature]["enabled"] = False
            else:
                self.config["features"][feature_type]["enabled"] = False
    
    def get_enabled_features(self) -> Dict[str, List[str]]:
        """获取启用的特征列表"""
        enabled_features = {}
        
        for feature_type, config in self.config.get("features", {}).items():
            if config.get("enabled", True):
                enabled_features[feature_type] = []
                
                # 检查子特征
                for key, value in config.items():
                    if isinstance(value, dict) and value.get("enabled", True):
                        enabled_features[feature_type].append(key)
        
        return enabled_features
    
    def validate_config(self) -> Dict[str, List[str]]:
        """验证配置的有效性"""
        warnings = []
        errors = []
        
        # 验证相机内参
        intrinsics = self.config.get("camera", {}).get("intrinsics", {})
        required_intrinsics = ["fx", "fy", "cx", "cy"]
        for param in required_intrinsics:
            if param not in intrinsics:
                errors.append(f"缺少相机内参: {param}")
        
        # 验证特征权重
        weights = self.config.get("similarity", {}).get("feature_weights", {})
        if weights:
            total_weight = sum(weights.values())
            if abs(total_weight - 1.0) > 0.01:
                warnings.append(f"特征权重总和不为1.0: {total_weight}")
        
        # 验证阈值范围
        thresholds = self.config.get("similarity", {}).get("thresholds", {})
        for name, threshold in thresholds.items():
            if not 0.0 <= threshold <= 1.0:
                errors.append(f"阈值超出范围 [0,1]: {name} = {threshold}")
        
        # 验证文件路径
        learning_file = self.config.get("adaptive_learning", {}).get("learning_data_file")
        if learning_file:
            learning_dir = os.path.dirname(learning_file)
            if learning_dir and not os.path.exists(learning_dir):
                warnings.append(f"学习数据目录不存在: {learning_dir}")
        
        return {"warnings": warnings, "errors": errors}
    
    def create_sample_config_file(self, output_path: str):
        """创建示例配置文件"""
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            
            print(f"[CONFIG] 示例配置文件已创建: {output_path}")
            
        except Exception as e:
            print(f"[CONFIG] 创建示例配置文件失败: {e}")
    
    def get_config_summary(self) -> str:
        """获取配置摘要"""
        enabled_features = self.get_enabled_features()
        
        summary = []
        summary.append("=== 增强检测系统配置摘要 ===")
        summary.append(f"检测器: {self.config.get('detector', {}).get('type', 'unknown')}")
        summary.append(f"分割器: {self.config.get('segmentor', {}).get('type', 'unknown')}")
        
        summary.append("\n启用的特征:")
        for feature_type, sub_features in enabled_features.items():
            summary.append(f"  {feature_type}: {', '.join(sub_features) if sub_features else 'all'}")
        
        weights = self.config.get("similarity", {}).get("feature_weights", {})
        if weights:
            summary.append("\n特征权重:")
            for feature, weight in weights.items():
                summary.append(f"  {feature}: {weight:.2f}")
        
        adaptive_enabled = self.config.get("adaptive_learning", {}).get("enabled", False)
        summary.append(f"\n自适应学习: {'启用' if adaptive_enabled else '禁用'}")
        
        return "\n".join(summary)