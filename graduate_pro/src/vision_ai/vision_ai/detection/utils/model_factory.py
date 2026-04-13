# detection/utils/model_factory.py
from typing import Dict, Any
from ..interfaces.detector_interface import ObjectDetector
from ..interfaces.segmentor_interface import ObjectSegmentor
from ..detectors.yolo_detector import YOLODetector
from ..segmentors.sam2_segmentor import SAM2Segmentor

class ModelFactory:
    """模型工厂类"""
    
    # 注册的检测器类型
    DETECTOR_REGISTRY = {
        'yolo': YOLODetector,
        # 未来可以添加更多检测器
        # 'detr': DetrDetector,
        # 'rtdetr': RTDetrDetector,
    }
    
    # 注册的分割器类型
    SEGMENTOR_REGISTRY = {
        'sam2': SAM2Segmentor,
        # 未来可以添加更多分割器
        # 'grounded_sam': GroundedSAMSegmentor,
        # 'fastsam': FastSAMSegmentor,
    }
    
    @classmethod
    def create_detector(cls, detector_type: str, **kwargs) -> ObjectDetector:
        """
        创建检测器实例
        
        Args:
            detector_type: 检测器类型
            **kwargs: 检测器初始化参数
            
        Returns:
            detector: 检测器实例
            
        Raises:
            ValueError: 不支持的检测器类型
        """
        detector_type = detector_type.lower()
        
        if detector_type not in cls.DETECTOR_REGISTRY:
            available_types = list(cls.DETECTOR_REGISTRY.keys())
            raise ValueError(
                f"不支持的检测器类型: {detector_type}. "
                f"可用类型: {available_types}"
            )
        
        detector_class = cls.DETECTOR_REGISTRY[detector_type]
        
        try:
            detector = detector_class(**kwargs)
            print(f"[FACTORY] {detector_type.upper()}检测器创建成功")
            return detector
        except Exception as e:
            raise RuntimeError(f"检测器创建失败: {e}") from e
    
    @classmethod
    def create_segmentor(cls, segmentor_type: str, **kwargs) -> ObjectSegmentor:
        """
        创建分割器实例
        
        Args:
            segmentor_type: 分割器类型
            **kwargs: 分割器初始化参数
            
        Returns:
            segmentor: 分割器实例
            
        Raises:
            ValueError: 不支持的分割器类型
        """
        segmentor_type = segmentor_type.lower()
        
        if segmentor_type not in cls.SEGMENTOR_REGISTRY:
            available_types = list(cls.SEGMENTOR_REGISTRY.keys())
            raise ValueError(
                f"不支持的分割器类型: {segmentor_type}. "
                f"可用类型: {available_types}"
            )
        
        segmentor_class = cls.SEGMENTOR_REGISTRY[segmentor_type]
        
        try:
            segmentor = segmentor_class(**kwargs)
            print(f"[FACTORY] {segmentor_type.upper()}分割器创建成功")
            return segmentor
        except Exception as e:
            raise RuntimeError(f"分割器创建失败: {e}") from e
    
    @classmethod
    def create_from_config(cls, detector_config: Dict[str, Any], 
                          segmentor_config: Dict[str, Any]) -> tuple:
        """
        根据配置创建检测器和分割器
        
        Args:
            detector_config: 检测器配置
            segmentor_config: 分割器配置
            
        Returns:
            detector, segmentor: 检测器和分割器实例
        """
        # 创建检测器
        detector_config_copy = detector_config.copy()
        detector_type = detector_config_copy.pop('type', 'yolo')
        detector = cls.create_detector(detector_type, **detector_config_copy)
        
        # 创建分割器 - 修复参数映射
        segmentor_config_copy = segmentor_config.copy()
        segmentor_type = segmentor_config_copy.pop('type', 'sam2')
        
        # 修复SAM2参数映射
        if segmentor_type.lower() == 'sam2':
            # 将'checkpoint'映射为'checkpoint_path'
            if 'checkpoint' in segmentor_config_copy:
                segmentor_config_copy['checkpoint_path'] = segmentor_config_copy.pop('checkpoint')
            # 将'config'映射为'config_name'
            if 'config' in segmentor_config_copy:
                segmentor_config_copy['config_name'] = segmentor_config_copy.pop('config')
        
        segmentor = cls.create_segmentor(segmentor_type, **segmentor_config_copy)
        
        return detector, segmentor
    
    @classmethod
    def register_detector(cls, name: str, detector_class):
        """
        注册新的检测器类型
        
        Args:
            name: 检测器名称
            detector_class: 检测器类
        """
        cls.DETECTOR_REGISTRY[name.lower()] = detector_class
        print(f"[FACTORY] 已注册检测器: {name}")
    
    @classmethod
    def register_segmentor(cls, name: str, segmentor_class):
        """
        注册新的分割器类型
        
        Args:
            name: 分割器名称
            segmentor_class: 分割器类
        """
        cls.SEGMENTOR_REGISTRY[name.lower()] = segmentor_class
        print(f"[FACTORY] 已注册分割器: {name}")
    
    @classmethod
    def list_available_models(cls) -> Dict[str, list]:
        """
        列出所有可用的模型类型
        
        Returns:
            available_models: 可用模型字典
        """
        return {
            'detectors': list(cls.DETECTOR_REGISTRY.keys()),
            'segmentors': list(cls.SEGMENTOR_REGISTRY.keys())
        }
    
    @classmethod
    def validate_config(cls, detector_config: Dict[str, Any], 
                       segmentor_config: Dict[str, Any]) -> bool:
        """
        验证配置的有效性
        
        Args:
            detector_config: 检测器配置
            segmentor_config: 分割器配置
            
        Returns:
            valid: 配置是否有效
        """
        try:
            # 检查检测器类型
            detector_type = detector_config.get('type', 'yolo').lower()
            if detector_type not in cls.DETECTOR_REGISTRY:
                print(f"[FACTORY] 无效的检测器类型: {detector_type}")
                return False
            
            # 检查分割器类型
            segmentor_type = segmentor_config.get('type', 'sam2').lower()
            if segmentor_type not in cls.SEGMENTOR_REGISTRY:
                print(f"[FACTORY] 无效的分割器类型: {segmentor_type}")
                return False
            
            print(f"[FACTORY] 配置验证通过")
            return True
            
        except Exception as e:
            print(f"[FACTORY] 配置验证失败: {e}")
            return False