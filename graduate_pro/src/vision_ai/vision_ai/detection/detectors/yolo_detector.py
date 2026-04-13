# detection/detectors/yolo_detector.py
import cv2
import numpy as np
from ultralytics import YOLO
from typing import Tuple
from ..interfaces.detector_interface import ObjectDetector

class YOLODetector(ObjectDetector):
    
    def __init__(self, model_path: str, confidence_threshold: float = 0.5):
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold

        # 优先使用权重内嵌类别名（COCO / 自定义训练集）；兼容 ultralytics 不同版本
        names = getattr(self.model, "names", None)
        if isinstance(names, dict):
            self.class_names = {int(k): str(v) for k, v in names.items()}
        elif isinstance(names, (list, tuple)):
            self.class_names = {i: str(n) for i, n in enumerate(names)}
        else:
            self.class_names = {
                1: "banana",
                2: "carrot",
                3: "corn",
                4: "lemon",
                5: "greenlemon",
                6: "strawberry",
                7: "tomato",
                8: "potato",
                9: "redpepper",
            }
        
        print(f"[YOLO] Model loaded successfully: {model_path}")
        print(f"[YOLO] Confidence threshold: {confidence_threshold}")
    
    def detect(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        results = self.model(image_bgr)[0]
        boxes = results.boxes.xyxy.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy().astype(int)
        confidences = results.boxes.conf.cpu().numpy()
        
        valid_indices = confidences >= self.confidence_threshold
        boxes = boxes[valid_indices]
        class_ids = class_ids[valid_indices]
        confidences = confidences[valid_indices]
                
        print(f"[YOLO] Detected {len(boxes)} objects (confidence >= {self.confidence_threshold})")
        
        return boxes, class_ids, confidences
    
    def get_class_names(self) -> dict:
        return self.class_names
    
    def set_confidence_threshold(self, threshold: float):
        self.confidence_threshold = threshold
        print(f"[YOLO] confidence threshold updated to: {threshold}")