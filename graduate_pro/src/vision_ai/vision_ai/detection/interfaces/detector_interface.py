# detection/interfaces/detector_interface.py
from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, List

class ObjectDetector(ABC):
    """目标检测器抽象基类"""
    
    @abstractmethod
    def detect(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        检测图像中的目标
        
        Args:
            image: RGB图像 (H, W, 3)
            
        Returns:
            boxes: 检测框 (N, 4) [x1, y1, x2, y2]
            class_ids: 类别ID (N,)
            confidences: 置信度 (N,)
        """
        pass
    
    @abstractmethod
    def get_class_names(self) -> dict:
        """
        获取类别名称映射
        
        Returns:
            class_names: {class_id: class_name}
        """
        pass
    
    def set_confidence_threshold(self, threshold: float):
        """设置置信度阈值"""
        self.confidence_threshold = threshold