# detection/detection_pipeline.py
import os
import cv2
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import time
import json
from .utils.config_manager import ConfigManager
from .utils.model_factory import ModelFactory
from .features.color_features import ColorFeatureExtractor
from .features.shape_features import EnhancedShapeFeatureExtractor
from .features.spatial_features import SpatialFeatureExtractor

class DetectionPipeline:
    """完整的检测管道，集成YOLO检测 + SAM2分割 + 四维特征提取"""
    
    def __init__(self, config_file: str = None, output_dir: str = None):
        """
        初始化检测管道
        
        Args:
            config_file: 配置文件路径
            output_dir: 输出目录
        """
        # 配置管理
        self.config_manager = ConfigManager(config_file)
        self.config = self.config_manager.config
        
        # 输出目录
        self.output_dir = output_dir or f"detection_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 创建检测器和分割器
        self._initialize_models()
        
        # 创建特征提取器
        self._initialize_feature_extractors()
        
        # 数据存储
        self.reference_data = {}
        self.selected_tracking_ids = []
        self.next_object_id = {}  # 每个类别的下一个ID
        
        print(f"[PIPELINE] 检测管道初始化完成")
        print(f"[PIPELINE] 输出目录: {self.output_dir}")
    
    def _initialize_models(self):
        """初始化检测和分割模型"""
        try:
            detector_config = self.config_manager.get_detector_config().copy()
            segmentor_config = self.config_manager.get_segmentor_config().copy()
            
            # 创建模型
            self.detector, self.segmentor = ModelFactory.create_from_config(
                detector_config, segmentor_config
            )
            
            print(f"[PIPELINE] 模型初始化成功")
            
        except Exception as e:
            raise RuntimeError(f"模型初始化失败: {e}") from e
    
    def _initialize_feature_extractors(self):
        """初始化特征提取器"""
        features_config = self.config_manager.get_features_config()
        camera_config = self.config_manager.get_camera_config()
        
        # 颜色特征提取器
        color_config = features_config.get('color', {})
        self.color_extractor = ColorFeatureExtractor(
            bins=color_config.get('histogram_bins', 64)
        )
        
        # 形状特征提取器
        self.shape_extractor = EnhancedShapeFeatureExtractor()
        
        # 空间特征提取器
        camera_intrinsics = camera_config.get('intrinsics', {})
        self.spatial_extractor = SpatialFeatureExtractor(camera_intrinsics)
        
        print(f"[PIPELINE] 特征提取器初始化完成")
    
    def process_single_image(self, image_rgb: np.ndarray, 
                           depth_image: Optional[np.ndarray] = None,
                           camera_pose: Optional[Dict] = None) -> Dict:
        """
        处理单张图像
        
        Args:
            image_rgb: RGB图像 (H, W, 3)
            depth_image: 深度图像 (H, W)，可选
            camera_pose: 相机位姿，可选
            
        Returns:
            result: 检测结果字典
        """
        print(f"[PIPELINE] 开始处理图像...")
        
        # 1. YOLO检测
        boxes, class_ids, confidences = self.detector.detect(image_rgb)
        print(f"[PIPELINE] YOLO检测到 {len(boxes)} 个目标")
        
        if len(boxes) == 0:
            return {
                'objects': [],
                'detection_count': 0,
                'processing_time': 0
            }
        
        # 2. SAM2分割
        masks = self.segmentor.segment(image_rgb, boxes)
        print(f"[PIPELINE] SAM2分割完成")
        
        # 3. 提取特征并生成结果
        objects = []
        class_names = self.detector.get_class_names()
        
        for i, (box, class_id, confidence, mask) in enumerate(zip(boxes, class_ids, confidences, masks)):
            try:
                # 分配对象ID
                object_id = self._assign_object_id(class_id)
                class_name = class_names.get(class_id, f'class_{class_id}')
                
                # 提取四维特征
                features = self._extract_all_features(
                    image_rgb, mask, depth_image, camera_pose
                )
                
                # 生成语义描述
                description = self._generate_description(
                    class_name, features, object_id
                )
                
                # 构建对象信息
                obj_info = {
                    'object_id': f"{class_name}_{object_id}",
                    'class_id': int(class_id),
                    'class_name': class_name,
                    'confidence': float(confidence),
                    'bounding_box': box.tolist(),
                    'mask': mask,
                    'features': features,
                    'description': description
                }
                
                objects.append(obj_info)
                
            except Exception as e:
                print(f"[PIPELINE] 处理第 {i} 个对象时出错: {e}")
                continue
        
        result = {
            'objects': objects,
            'detection_count': len(objects),
            'processing_time': 0  # 可以添加计时
        }
        
        print(f"[PIPELINE] 图像处理完成，检测到 {len(objects)} 个有效目标")
        return result
    
    def _assign_object_id(self, class_id: int) -> int:
        """为对象分配ID"""
        if class_id not in self.next_object_id:
            self.next_object_id[class_id] = 0
        else:
            self.next_object_id[class_id] += 1
        
        return self.next_object_id[class_id]
    
    def _extract_all_features(self, image: np.ndarray, mask: np.ndarray,
                            depth_image: Optional[np.ndarray] = None,
                            camera_pose: Optional[Dict] = None) -> Dict:
        """提取所有四维特征 - 改进版本，保存原始mask"""
        features = {}
        
        try:
            # 🆕 保存原始mask数据用于改进的相似度计算
            if mask.dtype != bool:
                mask_bool = mask > 0.5
            else:
                mask_bool = mask
            
            # 1. 颜色特征 - 使用改进的颜色检测
            color_stats = self.color_extractor.get_color_statistics(image, mask)
            
            # 🆕 使用改进的颜色直方图
            improved_histogram = self._extract_color_histogram_improved(image, mask_bool, bins=32)
            color_stats['histogram'] = improved_histogram
            
            # 使用改进的颜色检测覆盖原有结果
            improved_color = self._improve_color_detection(image, mask_bool)
            color_stats['color_name'] = improved_color
            
            features['color'] = color_stats
            
            # 2. 形状特征
            shape_features = self.shape_extractor.extract_all_features(mask)
            
            # 🆕 保存原始mask用于改进的Hu矩计算
            shape_features['raw_mask'] = mask_bool.copy()
            
            # 🆕 添加鲁棒形状特征
            robust_features = self._extract_robust_shape_features(mask_bool)
            shape_features.update(robust_features)
            
            features['shape'] = shape_features
            
            # 3. 空间特征（保持原有逻辑）
            if depth_image is not None and camera_pose is not None:
                spatial_features = self.spatial_extractor.extract_all_features(
                    mask, depth_image, camera_pose
                )
                features['spatial'] = spatial_features
            else:
                # 基本的2D空间信息
                ys, xs = np.where(mask > 0)
                if len(xs) > 0:
                    centroid_x, centroid_y = np.mean(xs), np.mean(ys)
                    image_height, image_width = mask.shape
                    norm_x = centroid_x / image_width
                    norm_y = centroid_y / image_height
                    
                    region_position = self.spatial_extractor._determine_region_position(norm_x, norm_y)
                    
                    features['spatial'] = {
                        'centroid_2d': (centroid_x, centroid_y),
                        'normalized_coords': (norm_x, norm_y),
                        'region_position': region_position
                    }
            
        except Exception as e:
            print(f"[PIPELINE] 特征提取失败: {e}")
            features = {'error': str(e)}
        
        return features
    def _extract_color_histogram_improved(self, image, mask, bins=32):
        """改进的颜色直方图提取 - 与charttest.py保持一致"""
        if mask.dtype != bool:
            mask = mask > 0.5
        
        masked_pixels = image[mask]
        if len(masked_pixels) == 0:
            return [0.0] * (bins * 3)
        
        # 分别计算RGB三个通道的直方图
        hist_r = np.histogram(masked_pixels[:, 0], bins=bins, range=(0, 256))[0]
        hist_g = np.histogram(masked_pixels[:, 1], bins=bins, range=(0, 256))[0]
        hist_b = np.histogram(masked_pixels[:, 2], bins=bins, range=(0, 256))[0]
        
        # 归一化直方图
        hist_r = hist_r / (np.sum(hist_r) + 1e-10)
        hist_g = hist_g / (np.sum(hist_g) + 1e-10)
        hist_b = hist_b / (np.sum(hist_b) + 1e-10)
        
        return np.concatenate([hist_r, hist_g, hist_b]).tolist()

    def _extract_robust_shape_features(self, mask, smooth_kernel_size=7):
        """提取鲁棒形状特征 - 与charttest.py保持一致"""
        if mask.dtype == bool:
            mask_uint8 = mask.astype(np.uint8) * 255
        else:
            mask_uint8 = (mask * 255).astype(np.uint8)
        
        # 形态学处理
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (smooth_kernel_size, smooth_kernel_size))
        mask_smooth = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
        mask_smooth = cv2.morphologyEx(mask_smooth, cv2.MORPH_OPEN, kernel)
        
        features = {}
        
        # 鲁棒Hu矩
        moments = cv2.moments(mask_smooth)
        if moments['m00'] > 0:
            hu_moments = cv2.HuMoments(moments).flatten()
            hu_log = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-8)
            features['hu_moments_robust'] = hu_log.tolist()
        else:
            features['hu_moments_robust'] = [0.0] * 7
        
        # 轮廓特征
        contours, _ = cv2.findContours(mask_smooth, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            contour_area = cv2.contourArea(largest_contour)
            hull_area = cv2.contourArea(cv2.convexHull(largest_contour))
            solidity = contour_area / hull_area if hull_area > 0 else 0
            
            perimeter = cv2.arcLength(largest_contour, True)
            circularity = 4 * np.pi * contour_area / (perimeter * perimeter) if perimeter > 0 else 0
            
            rect = cv2.minAreaRect(largest_contour)
            width, height = rect[1]
            aspect_ratio = max(width, height) / (min(width, height) + 1e-8)
            
            if len(largest_contour) >= 5:
                ellipse = cv2.fitEllipse(largest_contour)
                ellipse_aspect = max(ellipse[1]) / (min(ellipse[1]) + 1e-8)
            else:
                ellipse_aspect = aspect_ratio
                
            features['shape_descriptors'] = [solidity, circularity, aspect_ratio, ellipse_aspect]
        else:
            features['shape_descriptors'] = [0.0, 0.0, 0.0, 0.0]
        
        # 傅里叶描述子
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if len(largest_contour) > 10:
                contour_points = largest_contour.reshape(-1, 2)
                cx, cy = np.mean(contour_points, axis=0)
                distances = np.sqrt((contour_points[:, 0] - cx)**2 + (contour_points[:, 1] - cy)**2)
                
                if len(distances) > 1:
                    fft_desc = np.fft.fft(distances)
                    fourier_desc = np.abs(fft_desc[1:9])
                    if fourier_desc[0] > 1e-8:
                        fourier_desc = fourier_desc / fourier_desc[0]
                    features['fourier_descriptors'] = fourier_desc.tolist()
                else:
                    features['fourier_descriptors'] = [0.0] * 8
            else:
                features['fourier_descriptors'] = [0.0] * 8
        else:
            features['fourier_descriptors'] = [0.0] * 8
        
        return features    
    def _generate_description(self, class_name: str, features: Dict, object_id: int) -> str:
        """生成英文描述"""
        try:
            # 获取颜色信息
            color_name = features.get('color', {}).get('color_name', 'unknown')
            
            # 获取位置信息
            spatial_features = features.get('spatial', {})
            region_position = spatial_features.get('region_position', 'unknown')
            
            # 生成描述
            if region_position != 'unknown':
                # 转换区域名称为更自然的描述
                region_desc = region_position.replace('-', ' ').replace('_', ' ')
                if region_desc == 'center':
                    region_desc = 'center area'
                elif 'center' in region_desc:
                    region_desc = region_desc.replace('center', 'center')
                else:
                    region_desc = f"{region_desc} corner"
                
                description = f"{color_name.capitalize()} {class_name} in {region_desc}"
            else:
                description = f"{color_name.capitalize()} {class_name}"
            
            return description
            
        except Exception as e:
            print(f"[PIPELINE] 描述生成失败: {e}")
            return f"{class_name}_{object_id}"

    def process_reference_image(self, image_rgb: np.ndarray, 
                              depth_image: Optional[np.ndarray] = None,
                              camera_pose: Optional[Dict] = None,
                              generate_visualization: bool = True,
                              auto_display: bool = True) -> Dict:
        """
        处理单张参考图像（拼接后的最终图像）
        
        Args:
            image_rgb: RGB参考图像 (H, W, 3)
            depth_image: 深度图像 (H, W)，可选
            camera_pose: 相机位姿，可选
            generate_visualization: 是否生成可视化
            
        Returns:
            result: 检测结果字典
        """
        print(f"[PIPELINE] 开始处理参考图像...")
        start_time = time.time()
        
        try:
            # 1. YOLO检测
            boxes, class_ids, confidences = self.detector.detect(image_rgb)
            print(f"[PIPELINE] YOLO检测到 {len(boxes)} 个目标")
            
            if len(boxes) == 0:
                return {
                    'success': True,
                    'objects': [],
                    'detection_count': 0,
                    'processing_time': time.time() - start_time,
                    'message': '未检测到任何目标'
                }
            
            # 2. SAM2分割
            masks = self.segmentor.segment(image_rgb, boxes)
            print(f"[PIPELINE] SAM2分割完成")
            
            # 3. 提取特征并生成结果
            objects = []
            class_names = self.detector.get_class_names()
            
            for i, (box, class_id, confidence, mask) in enumerate(zip(boxes, class_ids, confidences, masks)):
                try:
                    # 分配对象ID
                    object_id = self._assign_object_id(class_id)
                    class_name = class_names.get(class_id, f'class_{class_id}')
                    
                    # 提取四维特征
                    features = self._extract_all_features(
                        image_rgb, mask, depth_image, camera_pose
                    )
                    
                    # 生成语义描述
                    description = self._generate_description(
                        class_name, features, object_id
                    )
                    
                    # 构建对象信息
                    obj_info = {
                        'object_id': f"{class_name}_{object_id}",
                        'class_id': int(class_id),
                        'class_name': class_name,
                        'confidence': float(confidence),
                        'bounding_box': box.tolist(),
                        'mask': mask,
                        'features': features,
                        'description': description
                    }
                    
                    objects.append(obj_info)
                    
                except Exception as e:
                    print(f"[PIPELINE] 处理第 {i} 个对象时出错: {e}")
                    continue
            
            # 4. 生成可视化
            visualization_image = None
            if generate_visualization and len(objects) > 0:
                visualization_image = self._generate_visualization(image_rgb, objects)
                
                if visualization_image is not None:
                    # 自动显示可视化
                    if auto_display:
                        self._auto_display_visualization(visualization_image, self.output_dir)
                    else:
                        # 只保存，不显示
                        vis_filename = os.path.join(self.output_dir, "detection_visualization.jpg")
                        cv2.imwrite(vis_filename, cv2.cvtColor(visualization_image, cv2.COLOR_RGB2BGR))
                        print(f"[PIPELINE] 可视化图像已保存: {vis_filename}")
            
                # 保存objects到实例变量，供选择功能使用
                self.latest_objects = objects
                
                # 保存可视化图像
                if visualization_image is not None:
                    vis_filename = os.path.join(self.output_dir, "detection_visualization.jpg")
                    cv2.imwrite(vis_filename, cv2.cvtColor(visualization_image, cv2.COLOR_RGB2BGR))
                    print(f"[PIPELINE] 可视化图像已保存: {vis_filename}")
            
            # 5. 保存检测结果
            self._save_detection_results(objects)
            
            result = {
                'success': True,
                'objects': objects,
                'detection_count': len(objects),
                'processing_time': time.time() - start_time,
                'visualization_image': visualization_image,
                'message': f'成功检测到 {len(objects)} 个目标'
            }
            
            print(f"[PIPELINE] 参考图像处理完成，检测到 {len(objects)} 个有效目标")
            return result
            
        except Exception as e:
            print(f"[PIPELINE] 参考图像处理失败: {e}")
            return {
                'success': False,
                'objects': [],
                'detection_count': 0,
                'processing_time': time.time() - start_time,
                'message': f'检测失败: {str(e)}'
            }

    def _generate_visualization(self, image_rgb: np.ndarray, objects: List[Dict]) -> np.ndarray:
        """
        生成检测可视化图像 - 修复mask重叠问题
        
        Args:
            image_rgb: 原始图像
            objects: 检测到的对象列表
            
        Returns:
            visualization: 可视化图像
        """
        # 增强图像亮度
        vis_image = image_rgb.copy().astype(np.float32)
        
        # 自适应亮度增强
        mean_brightness = np.mean(vis_image)
        if mean_brightness < 100:  # 如果图像较暗
            brightness_factor = 1.5
            vis_image = np.clip(vis_image * brightness_factor, 0, 255)
        
        vis_image = vis_image.astype(np.uint8)
        
        # 🔧 创建一个单独的mask覆盖层，避免累积效应
        mask_overlay = np.zeros_like(vis_image, dtype=np.float32)
        
        # 设置可视化参数
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_thickness = 2
        
        # 预定义亮色调色板
        bright_colors = [
            (255, 100, 100),  # 亮红
            (100, 255, 100),  # 亮绿
            (100, 150, 255),  # 亮蓝
            (255, 255, 100),  # 亮黄
            (255, 100, 255),  # 亮紫
            (100, 255, 255),  # 亮青
            (255, 150, 100),  # 亮橙
            (200, 255, 150),  # 浅绿
            (255, 200, 150),  # 浅橙
            (150, 200, 255),  # 浅蓝
        ]
        
        # 存储已使用的标签位置，避免重叠
        used_label_areas = []
        
        # 🎨 第一步：收集所有mask并创建统一的覆盖层
        valid_masks = []
        for i, obj in enumerate(objects):
            try:
                mask = obj['mask']
                if isinstance(mask, np.ndarray):
                    # 确保mask是boolean类型
                    if mask.dtype != bool:
                        mask_bool = mask > 0.5 if mask.dtype == np.float32 else mask > 0
                    else:
                        mask_bool = mask
                    
                    # 确保mask尺寸匹配
                    if mask_bool.shape[:2] == vis_image.shape[:2]:
                        color = bright_colors[i % len(bright_colors)]
                        valid_masks.append((mask_bool, color, i, obj))
                    else:
                        print(f"[PIPELINE] 警告: mask尺寸不匹配，跳过对象 {i}")
            except Exception as e:
                print(f"[PIPELINE] 处理mask {i} 时出错: {e}")
        
        # 🎨 创建统一的mask覆盖层
        for mask_bool, color, i, obj in valid_masks:
            # 只在当前mask区域添加颜色，不影响其他区域
            mask_overlay[mask_bool] = color
        
        # 🎨 一次性应用所有mask到原图像
        if len(valid_masks) > 0:
            # 创建一个mask指示哪些像素有覆盖
            any_mask = np.any(mask_overlay > 0, axis=2)
            
            # 只在有mask的区域进行混合
            vis_image_float = vis_image.astype(np.float32)
            vis_image_float[any_mask] = (vis_image_float[any_mask] * 0.6 + 
                                       mask_overlay[any_mask] * 0.4)
            vis_image = np.clip(vis_image_float, 0, 255).astype(np.uint8)
        
        # 🖼️ 第二步：绘制边框、中心点和标签
        for mask_bool, color, i, obj in valid_masks:
            try:
                bbox = obj['bounding_box']
                object_id = obj['object_id']
                description = obj['description']
                confidence = obj['confidence']
                
                # 2. 绘制边界框（加粗）
                try:
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 4)  # 更粗的边框
                except Exception as e:
                    print(f"[PIPELINE] 绘制边界框失败: {e}")
                    continue
                
                # 3. 计算并绘制中心点
                try:
                    if 'spatial' in obj.get('features', {}):
                        centroid = obj['features']['spatial'].get('centroid_2d', ((x1+x2)/2, (y1+y2)/2))
                    else:
                        centroid = ((x1+x2)/2, (y1+y2)/2)
                    
                    center_x, center_y = map(int, centroid)
                    
                    # 确保中心点在图像范围内
                    center_x = max(0, min(vis_image.shape[1]-1, center_x))
                    center_y = max(0, min(vis_image.shape[0]-1, center_y))
                    
                    # 绘制更明显的中心点
                    cv2.circle(vis_image, (center_x, center_y), 12, color, -1)
                    cv2.circle(vis_image, (center_x, center_y), 16, (255, 255, 255), 3)
                    
                    # 4. 绘制编号（更大更明显）
                    number_text = str(i + 1)
                    cv2.putText(vis_image, number_text, (center_x - 8, center_y + 6), 
                               font, 1.0, (0, 0, 0), 3)  # 黑色边框
                    cv2.putText(vis_image, number_text, (center_x - 8, center_y + 6), 
                               font, 1.0, (255, 255, 255), 2)  # 白色文字
                except Exception as e:
                    print(f"[PIPELINE] 绘制中心点失败: {e}")
                
                # 5. 智能放置描述标签，避免重叠
                try:
                    label_text = f"{i+1}. {description} ({confidence:.2f})"
                    
                    # 计算文本尺寸
                    (text_width, text_height), baseline = cv2.getTextSize(
                        label_text, font, font_scale, font_thickness
                    )
                    
                    # 寻找不重叠的标签位置
                    label_positions = [
                        (x1, y1 - 15),  # 框上方
                        (x1, y2 + text_height + 15),  # 框下方
                        (x2 + 10, y1 + text_height),  # 框右侧
                        (max(0, x1 - text_width - 10), y1 + text_height),  # 框左侧
                    ]
                    
                    label_x, label_y = label_positions[0]  # 默认位置
                    
                    # 检查每个位置是否与已有标签重叠
                    for pos_x, pos_y in label_positions:
                        # 确保位置在图像范围内
                        if (pos_x >= 0 and pos_y >= text_height and 
                            pos_x + text_width < vis_image.shape[1] and 
                            pos_y < vis_image.shape[0]):
                            
                            # 检查是否与已有标签重叠
                            current_area = (pos_x, pos_y - text_height, pos_x + text_width, pos_y)
                            
                            overlap = False
                            for used_area in used_label_areas:
                                if (current_area[0] < used_area[2] and current_area[2] > used_area[0] and
                                    current_area[1] < used_area[3] and current_area[3] > used_area[1]):
                                    overlap = True
                                    break
                            
                            if not overlap:
                                label_x, label_y = pos_x, pos_y
                                used_label_areas.append(current_area)
                                break
                    
                    # 🎨 绘制标签背景（使用单独的overlay避免影响原图）
                    label_overlay = vis_image.copy()
                    cv2.rectangle(label_overlay, 
                                (label_x - 5, label_y - text_height - 5), 
                                (label_x + text_width + 5, label_y + 5), 
                                color, -1)
                    
                    # 只在标签区域进行混合
                    label_mask = np.zeros(vis_image.shape[:2], dtype=bool)
                    label_mask[label_y - text_height - 5:label_y + 5, 
                             label_x - 5:label_x + text_width + 5] = True
                    
                    vis_image[label_mask] = (vis_image[label_mask] * 0.7 + 
                                           label_overlay[label_mask] * 0.3).astype(np.uint8)
                    
                    # 绘制文字（带阴影效果）
                    # 阴影
                    cv2.putText(vis_image, label_text, (label_x + 2, label_y - 2), 
                               font, font_scale, (0, 0, 0), font_thickness + 1)
                    # 主文字
                    cv2.putText(vis_image, label_text, (label_x, label_y), 
                               font, font_scale, (255, 255, 255), font_thickness)
                    
                except Exception as e:
                    print(f"[PIPELINE] 绘制标签失败: {e}")
                
            except Exception as e:
                print(f"[PIPELINE] 绘制第 {i} 个对象可视化时出错: {e}")
                continue
        
        return vis_image

    def _improve_color_detection(self, image: np.ndarray, mask: np.ndarray) -> str:
        """
        改进的颜色检测方法
        
        Args:
            image: RGB图像
            mask: 对象mask
            
        Returns:
            color_name: 改进的颜色名称
        """
        if np.sum(mask) == 0:
            return "unknown"
        
        # 提取mask区域的像素
        masked_pixels = image[mask > 0]
        
        if len(masked_pixels) == 0:
            return "unknown"
        
        # 计算主色调 - 使用中位数而不是平均值，更抗噪声
        median_color = np.median(masked_pixels, axis=0)
        
        # 改进的颜色映射 - 更宽松的阈值
        r, g, b = median_color
        
        # 计算颜色的HSV值来更好地识别
        hsv_color = cv2.cvtColor(np.uint8([[median_color]]), cv2.COLOR_RGB2HSV)[0][0]
        h, s, v = hsv_color
        
        # 基于HSV的颜色分类
        if s < 30:  # 低饱和度
            if v > 200:
                return "white"
            elif v < 60:
                return "black"
            else:
                return "gray"
        
        # 高饱和度颜色分类
        if h < 15 or h > 165:  # 红色范围
            if r > g and r > b:
                return "red"
        elif 15 <= h < 35:  # 橙色/黄色范围
            if r > 180 and g > 150:
                return "yellow"
            else:
                return "orange"
        elif 35 <= h < 85:  # 绿色范围
            return "green"
        elif 85 <= h < 125:  # 青色范围
            return "cyan"
        elif 125 <= h < 165:  # 蓝色范围
            return "blue"
        
        # 特殊情况处理
        if r > 150 and g > 100 and b < 100:  # 偏橙色
            return "orange"
        elif g > 120 and r < 100 and b < 100:  # 偏绿色
            return "green"
        elif r > 100 and g < 80 and b < 80:  # 偏红色
            return "red"
        elif r > 120 and g > 120 and b < 80:  # 偏黄色
            return "yellow"
        
        return "mixed"

    def _save_detection_results(self, objects: List[Dict]):
        """保存检测结果到文件 - 改进版本，支持mask数据保存"""
        try:
            # 🆕 创建mask数据目录
            mask_data_dir = os.path.join(self.output_dir, 'mask_data')
            os.makedirs(mask_data_dir, exist_ok=True)
            
            # 创建可序列化的数据
            serializable_objects = []
            
            for obj in objects:
                obj_copy = obj.copy()
                
                # 🆕 处理mask数据
                if 'features' in obj_copy and 'shape' in obj_copy['features']:
                    shape_features = obj_copy['features']['shape']
                    
                    # 保存原始mask到.npy文件
                    if 'raw_mask' in shape_features:
                        raw_mask = shape_features['raw_mask']
                        mask_filename = f"mask_{obj_copy['object_id']}.npy"
                        mask_filepath = os.path.join(mask_data_dir, mask_filename)
                        
                        # 保存mask文件
                        np.save(mask_filepath, raw_mask)
                        
                        # 在JSON中记录mask文件信息
                        shape_features['raw_mask_file'] = f'mask_data/{mask_filename}'
                        shape_features['raw_mask_shape'] = raw_mask.shape
                        shape_features['raw_mask_dtype'] = str(raw_mask.dtype)
                        
                        # 从JSON数据中移除原始mask数组
                        del shape_features['raw_mask']
                        
                        print(f"[PIPELINE] 保存mask文件: {mask_filepath}")
                
                # 移除不可序列化的mask
                if 'mask' in obj_copy:
                    del obj_copy['mask']
                
                # 处理features中的numpy数组
                if 'features' in obj_copy:
                    features = obj_copy['features']
                    for feature_type, feature_data in features.items():
                        if isinstance(feature_data, dict):
                            features[feature_type] = self._sanitize_for_json(feature_data)
                
                serializable_objects.append(obj_copy)
            
            result_data = {
                'detection_summary': {
                    'total_objects': len(objects),
                    'detection_time': datetime.now().isoformat(),
                    'objects_by_class': self._group_objects_by_class(objects),
                    'mask_data_directory': 'mask_data'  # 🆕 记录mask目录
                },
                'objects': serializable_objects
            }
            
            # 保存JSON文件
            json_file = os.path.join(self.output_dir, "detection_results.json")
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, indent=2, ensure_ascii=False)
            
            print(f"[PIPELINE] 检测结果已保存到: {json_file}")
            print(f"[PIPELINE] Mask数据已保存到: {mask_data_dir}")
            
            # 生成可读的摘要文件
            self._save_detection_summary(objects)
            
        except Exception as e:
            print(f"[PIPELINE] 保存检测结果失败: {e}")
            import traceback
            traceback.print_exc()


    def _sanitize_for_json(self, data):
        """递归清理数据中的numpy数组，使其可JSON序列化"""
        if isinstance(data, dict):
            return {k: self._sanitize_for_json(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._sanitize_for_json(item) for item in data]
        elif isinstance(data, tuple):
            return list(data)  # 将tuple转为list
        elif isinstance(data, np.ndarray):
            return data.tolist()  # 将numpy数组转为list
        elif isinstance(data, (np.integer, np.floating)):
            return float(data)  # 将numpy数值类型转为Python原生类型
        else:
            return data
    def _group_objects_by_class(self, objects: List[Dict]) -> Dict:
        """按类别分组对象"""
        groups = {}
        for obj in objects:
            class_name = obj['class_name']
            if class_name not in groups:
                groups[class_name] = []
            groups[class_name].append(obj['object_id'])
        return groups

    def _save_detection_summary(self, objects: List[Dict]):
        """保存检测摘要"""
        try:
            summary_file = os.path.join(self.output_dir, "detection_summary.txt")
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write("Detection Results Summary\n")
                f.write("=" * 50 + "\n\n")
                
                f.write(f"Total Objects Detected: {len(objects)}\n")
                f.write(f"Detection Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                f.write("Detected Objects:\n")
                f.write("-" * 30 + "\n")
                
                for i, obj in enumerate(objects, 1):
                    f.write(f"{i:2d}. {obj['description']}\n")
                    f.write(f"    ID: {obj['object_id']}\n")
                    f.write(f"    Confidence: {obj['confidence']:.3f}\n")
                    
                    # 添加位置信息
                    if 'spatial' in obj.get('features', {}):
                        spatial = obj['features']['spatial']
                        region = spatial.get('region_position', 'unknown')
                        f.write(f"    Region: {region}\n")
                    
                    f.write("\n")
            
            print(f"[PIPELINE] 检测摘要已保存到: {summary_file}")
            
        except Exception as e:
            print(f"[PIPELINE] 保存检测摘要失败: {e}")
    
    def _save_reference_data(self):
        """保存参考数据"""
        import json
        
        # 创建可序列化的数据
        serializable_data = []

        for obj in self.reference_data['objects']:
            obj_copy = obj.copy()
            # 移除不可序列化的mask
            if 'mask' in obj_copy:
                del obj_copy['mask']
            serializable_data.append(obj_copy)
        
        save_data = {
            'objects': serializable_data,
            'total_images': self.reference_data['total_images'],
            'total_objects': self.reference_data['total_objects'],
            'created_at': self.reference_data['created_at']
        }
        
        # 保存JSON文件
        json_file = os.path.join(self.output_dir, "reference_data.json")
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        print(f"[PIPELINE] 参考数据已保存到: {json_file}")
    
    def _display_detection_results(self):
        """显示检测结果"""
        objects = self.reference_data['objects']
        
        print(f"\n{'='*60}")
        print("Detection Results Summary")
        print(f"{'='*60}")
        
        # 按类别统计
        class_counts = {}
        for obj in objects:
            class_name = obj['class_name']
            if class_name not in class_counts:
                class_counts[class_name] = []
            class_counts[class_name].append(obj['object_id'])
        
        print(f"\nDetected Objects by Category:")
        for class_name, obj_ids in class_counts.items():
            unique_ids = list(set(obj_ids))
            print(f"  {class_name}: {len(unique_ids)} unique objects")
            for obj_id in sorted(unique_ids):
                count = obj_ids.count(obj_id)
                print(f"    - {obj_id} (seen {count} times)")
        
        print(f"\nDetailed Detection List:")
        for i, obj in enumerate(objects, 1):
            print(f"{i:2d}. {obj['description']} (ID: {obj['object_id']})")
        
        print(f"\nTotal: {len(objects)} detections from {self.reference_data['total_images']} images")
        print(f"{'='*60}")
    
    def select_tracking_targets(self) -> bool:
        """选择要追踪的目标"""
        if not hasattr(self, 'latest_objects') or not self.latest_objects:
            print("[PIPELINE] 没有检测数据，请先进行检测")
            return False
        
        objects = self.latest_objects
        
        print(f"\n{'='*50}")
        print("Select Tracking Targets")
        print(f"{'='*50}")
        
        print("\nDetected objects:")
        for i, obj in enumerate(objects, 1):
            print(f"{i:2d}. {obj['description']} (ID: {obj['object_id']}, Conf: {obj['confidence']:.3f})")
        
        print("\nPlease select objects to track:")
        print("Format: Enter numbers separated by commas (e.g., 1,3,5)")
        
        while True:
            try:
                user_input = input("\nYour selection: ").strip()
                
                if user_input.lower() == 'all':
                    self.selected_tracking_ids = [obj['object_id'] for obj in objects]
                    break
                
                indices = [int(x.strip()) for x in user_input.split(',')]
                selected_ids = []
                
                for idx in indices:
                    if 1 <= idx <= len(objects):
                        selected_ids.append(objects[idx-1]['object_id'])
                    else:
                        print(f"Warning: Index {idx} is out of range")
                
                if selected_ids:
                    self.selected_tracking_ids = selected_ids
                    break
                else:
                    print("No valid selections. Please try again.")
                    
            except ValueError:
                print("Invalid input format. Please enter numbers separated by commas.")
            except KeyboardInterrupt:
                print("\nSelection cancelled.")
                return False
        
        print(f"\nSelected tracking targets:")
        for obj_id in self.selected_tracking_ids:
            obj = next((o for o in objects if o['object_id'] == obj_id), None)
            if obj:
                print(f"  - {obj['description']} (ID: {obj_id})")
        
        # 保存选择
        self._save_tracking_selection()
        
        return True

    def _save_tracking_selection(self):
        """保存追踪选择"""
        try:
            selection_file = os.path.join(self.output_dir, "tracking_selection.txt")
            
            with open(selection_file, 'w', encoding='utf-8') as f:
                f.write("Selected Tracking Targets:\n")
                f.write("=" * 50 + "\n")
                
                for obj_id in self.selected_tracking_ids:
                    obj = next((o for o in self.latest_objects if o['object_id'] == obj_id), None)
                    if obj:
                        f.write(f"- {obj['description']} (ID: {obj_id})\n")
            
            print(f"[PIPELINE] 追踪选择已保存到: {selection_file}")
            
        except Exception as e:
            print(f"[PIPELINE] 保存追踪选择失败: {e}")

    def get_selected_tracking_ids(self) -> List[str]:
        """获取选择的追踪目标ID"""
        return getattr(self, 'selected_tracking_ids', [])
    
    def get_reference_data(self) -> Dict:
        """获取参考数据"""
        return self.reference_data
    
    def get_output_dir(self) -> str:
        """获取输出目录"""
        return self.output_dir
    
    def _auto_display_visualization(self, vis_image: np.ndarray, output_dir: str):
        """自动显示可视化结果"""
        try:
            import os
            
            # 保存可视化图像（这部分总是执行）
            vis_file = os.path.join(output_dir, "detection_visualization.jpg")
            vis_bgr = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(vis_file, vis_bgr)
            print(f"[PIPELINE] 可视化图像已保存: {vis_file}")
            
            # 检查是否应该尝试显示
            display_mode = os.environ.get('VISION_AI_DISPLAY', 'auto')
            
            if display_mode == 'off':
                print("[PIPELINE] 显示功能已禁用（VISION_AI_DISPLAY=off）")
                return
            
            # 设置安全的显示环境
            original_platform = os.environ.get('QT_QPA_PLATFORM')
            try:
                # 首先尝试设置正确的环境
                os.environ['QT_QPA_PLATFORM'] = 'xcb'
                os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = '/usr/lib/x86_64-linux-gnu/qt5/plugins'
                
                # 尝试 matplotlib 显示
                import matplotlib
                matplotlib.use('Agg')  # 先使用无GUI后端，避免Qt冲突
                import matplotlib.pyplot as plt
                
                plt.figure(figsize=(15, 10))
                plt.imshow(vis_image)
                plt.title('Detection Results Visualization', fontsize=16, fontweight='bold')
                plt.axis('off')
                
                plt.figtext(0.5, 0.02, 
                        'Detection completed! Each detected object is marked with bounding box, mask, center point and ID number.',
                        ha='center', fontsize=12, style='italic')
                
                # 保存matplotlib图像而不是显示
                plt_file = os.path.join(output_dir, "detection_plot.png")
                plt.savefig(plt_file, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"[PIPELINE] matplotlib图像已保存: {plt_file}")
                
                # 如果有DISPLAY环境且不是offscreen模式，尝试显示
                if os.environ.get('DISPLAY') and os.environ.get('QT_QPA_PLATFORM') != 'offscreen':
                    try:
                        matplotlib.use('TkAgg')  # 尝试使用TkAgg后端
                        plt.figure(figsize=(15, 10))
                        plt.imshow(vis_image)
                        plt.title('Detection Results Visualization', fontsize=16, fontweight='bold')
                        plt.axis('off')
                        plt.tight_layout()
                        plt.show(block=False)  # 非阻塞显示
                        plt.pause(0.1)  # 短暂暂停确保显示
                        print("[PIPELINE] ✅ matplotlib显示成功")
                    except Exception as e:
                        print(f"[PIPELINE] matplotlib显示失败: {e}")
                
            except Exception as e:
                print(f"[PIPELINE] matplotlib处理失败: {e}")
            
            finally:
                # 恢复原始环境
                if original_platform:
                    os.environ['QT_QPA_PLATFORM'] = original_platform
                elif 'QT_QPA_PLATFORM' in os.environ:
                    del os.environ['QT_QPA_PLATFORM']
            
            print("[PIPELINE] ✅ 可视化处理完成")
            
        except Exception as e:
            print(f"[PIPELINE] 显示函数整体失败: {e}")
            print(f"[PIPELINE] 检测结果已保存，请手动查看: {output_dir}")