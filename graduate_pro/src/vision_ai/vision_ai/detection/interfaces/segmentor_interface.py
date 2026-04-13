# detection/interfaces/segmentor_interface.py
from abc import ABC, abstractmethod
import numpy as np
from typing import List

class ObjectSegmentor(ABC):
    """目标分割器抽象基类"""
    
    @abstractmethod
    def segment(self, image: np.ndarray, boxes: np.ndarray) -> List[np.ndarray]:
        """
        对检测到的目标进行分割
        
        Args:
            image: RGB图像 (H, W, 3)
            boxes: 检测框 (N, 4) [x1, y1, x2, y2]
            
        Returns:
            masks: 分割掩码列表，每个掩码为 (H, W) 的boolean数组
        """
        pass
    
    @abstractmethod
    def set_image(self, image: np.ndarray):
        """
        设置当前处理的图像
        
        Args:
            image: RGB图像 (H, W, 3)
        """
        pass