# detection/enhanced_detection_pipeline.py
import os
import cv2
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import time
import json

from .features.shape_features import EnhancedShapeFeatureExtractor, FeatureQualityAssessor
from .features.color_features import ColorFeatureExtractor
from .features.spatial_features import SpatialFeatureExtractor
from .utils.coordinate_calculator import CoordinateCalculator, ObjectAnalyzer, AdaptiveThresholdManager
from .utils.config_manager import ConfigManager
from .utils.model_factory import ModelFactory
from .utils.detection_post_processor import Detection3DPostProcessor
class EnhancedDetectionPipeline:
    """增强的检测管道 - 集成3D点云特征和自适应学习"""
    
    def __init__(self, config_file: str = None, output_dir: str = None):
        """初始化增强检测管道"""
        # 配置管理
        self.config_manager = ConfigManager(config_file)
        self.config = self.config_manager.config
        
        # 输出目录
        self.output_dir = output_dir or f"detection_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 初始化核心组件
        self._initialize_models()
        self._initialize_enhanced_components()
        
        # 数据存储
        self.reference_library = {}
        self.selected_tracking_ids = []
        self.next_object_id = {}
        self.post_processor = Detection3DPostProcessor(self.coordinate_calculator)
        self.performance_data = []
        self.current_test_metadata = None
        #  后处理配置
        self.enable_3d_post_processing = True  # 可以通过配置文件控制
        
        print(f"[ENHANCED_PIPELINE] Enhanced detection pipeline initialization completed")
    
    def _initialize_models(self):
        try:
            detector_config = self.config_manager.get_detector_config().copy()
            segmentor_config = self.config_manager.get_segmentor_config().copy()
            
            self.detector, self.segmentor = ModelFactory.create_from_config(
                detector_config, segmentor_config
            )
            
            print("[ENHANCED_PIPELINE] Model initialization successful")
            
        except Exception as e:
            raise RuntimeError(f"Model initialization failed: {e}") from e
    
    def _initialize_enhanced_components(self):
        """初始化增强组件"""
        # 获取配置
        features_config = self.config_manager.get_features_config()
        camera_config = self.config_manager.get_camera_config()
        
        # 坐标计算器
        calibration_data = {
            'camera_intrinsics': camera_config.get('intrinsics', {}),
            'hand_eye_translation': camera_config.get('hand_eye_translation'),
            'hand_eye_quaternion': camera_config.get('hand_eye_quaternion')
        }
        self.coordinate_calculator = CoordinateCalculator(calibration_data)
        
        # 物体分析器
        self.object_analyzer = ObjectAnalyzer(self.coordinate_calculator)
        
        # 增强特征提取器
        camera_intrinsics = camera_config.get('intrinsics', {})
        self.enhanced_shape_extractor = EnhancedShapeFeatureExtractor(camera_intrinsics)
        
        # 传统特征提取器（保留）
        color_config = features_config.get('color', {})
        self.color_extractor = ColorFeatureExtractor(
            bins=color_config.get('histogram_bins', 32)
        )
        self.spatial_extractor = SpatialFeatureExtractor(camera_intrinsics)
        
        # 特征质量评估器
        self.quality_assessor = FeatureQualityAssessor()
        
        # 自适应阈值管理器
        self.threshold_manager = AdaptiveThresholdManager()
    
    def set_test_metadata(self, metadata: Dict):
        """设置当前测试的元数据"""
        self.current_test_metadata = metadata
    
    def build_reference_library(self, image_rgb: np.ndarray, 
                                depth_image: np.ndarray,
                                waypoint_data: Dict,
                                generate_visualization: bool = True) -> Dict:
        """构建参考特征库 - 添加性能数据收集"""
        start_time = time.time()
        start_timestamp = datetime.now()
        
        # 记录各阶段的时间
        stage_times = {}
        
        try:
            # 1. YOLO检测
            yolo_start = time.time()
            boxes, class_ids, confidences = self.detector.detect(image_rgb)
            stage_times['yolo_detection'] = time.time() - yolo_start
            
            if len(boxes) == 0:
                return self._create_empty_result(start_time, stage_times)
            
            # 2. SAM2分割
            sam_start = time.time()
            masks = self.segmentor.segment(image_rgb, boxes)
            stage_times['sam_segmentation'] = time.time() - sam_start
            
            # 3. 构建初始检测结果
            preprocessing_start = time.time()
            initial_detections = self._build_initial_detections(boxes, class_ids, confidences, masks)
            stage_times['preprocessing'] = time.time() - preprocessing_start
            
            # 4. 3D后处理
            postprocessing_start = time.time()
            if self.enable_3d_post_processing and depth_image is not None and len(initial_detections) > 1:
                filtered_detections = self.post_processor.process_detections(
                    initial_detections, image_rgb, depth_image, waypoint_data
                )
            else:
                filtered_detections = initial_detections
            stage_times['3d_post_processing'] = time.time() - postprocessing_start
            
            # 5. 特征提取
            feature_start = time.time()
            reference_library, visualization_objects = self._extract_features_and_build_library(
                filtered_detections, image_rgb, depth_image, waypoint_data
            )
            stage_times['feature_extraction'] = time.time() - feature_start
            
            # 6. 可视化生成
            viz_start = time.time()
            visualization_image = None
            if generate_visualization and len(visualization_objects) > 0:
                visualization_image = self._generate_enhanced_visualization(image_rgb, visualization_objects)
                if visualization_image is not None:
                    vis_filename = os.path.join(self.output_dir, "detection_visualization.jpg")
                    cv2.imwrite(vis_filename, cv2.cvtColor(visualization_image, cv2.COLOR_RGB2BGR))
            stage_times['visualization'] = time.time() - viz_start
            
            # 7. 保存参考库
            save_start = time.time()
            self._save_reference_library(reference_library)
            stage_times['save_library'] = time.time() - save_start
            
            total_time = time.time() - start_time
            
            # 收集性能数据
            performance_data = self._collect_performance_data(
                start_timestamp, total_time, stage_times, 
                len(initial_detections), len(filtered_detections),
                reference_library, visualization_objects
            )
            
            result = {
                'success': True,
                'reference_library': reference_library,
                'detection_count': len(reference_library),
                'processing_time': total_time,
                'stage_times': stage_times,
                'visualization_image': visualization_image,
                'post_processing_stats': { 
                    'original_detections': len(initial_detections),
                    'filtered_detections': len(filtered_detections),
                    'duplicates_removed': len(initial_detections) - len(filtered_detections),
                    'post_processing_enabled': self.enable_3d_post_processing
                },
                'performance_data': performance_data,
                'message': f'Successfully built {len(reference_library)} reference features'
            }
            # 在3D后处理后添加
            if hasattr(self.post_processor, 'detailed_processing_stats'):
                result['detailed_3d_processing'] = self.post_processor.detailed_processing_stats

            if hasattr(self.post_processor, 'similarity_analyses'):
                result['merge_decision_data'] = self.post_processor.similarity_analyses

            print(f"[DEBUG] Result keys: {list(result.keys())}")
            if 'detailed_3d_processing' in result:
                print("[DEBUG] 成功添加 detailed_3d_processing")
            if 'merge_decision_data' in result:
                print("[DEBUG] 成功添加 merge_decision_data")
            self.reference_library = reference_library
            return result
            
        except Exception as e:
            total_time = time.time() - start_time
            print(f"[ENHANCED_PIPELINE] Reference library construction failed: {e}")
            import traceback
            traceback.print_exc()
            
            # 即使失败也收集性能数据
            performance_data = self._collect_error_performance_data(start_timestamp, total_time, str(e))
            
            return {
                'success': False,
                'reference_library': {},
                'detection_count': 0,
                'processing_time': total_time,
                'performance_data': performance_data,
                'message': f'Construction failed: {str(e)}'
            }

    def _log_post_processing_summary(self, stats: Dict, filtered_detections: List[Dict]):
        """记录后处理摘要"""
        try:
            print(f"\n[POST_PROCESSING_SUMMARY]")
            print(f"Original YOLO detections: {stats['original_detections']}")
            print(f"After 3D post-processing: {stats['filtered_detections']}")
            print(f"Duplicates removed: {stats['duplicates_removed']}")
            print(f"Post-processing enabled: {stats['post_processing_enabled']}")
            
            if stats['duplicates_removed'] > 0:
                print(f" Successfully removed {stats['duplicates_removed']} duplicate detection(s)")
                
                # 显示合并的检测信息
                merged_count = 0
                for detection in filtered_detections:
                    if detection.get('merged_from', 1) > 1:
                        merged_count += 1
                        print(f"   -> {detection['class_name']} (merged from {detection['merged_from']} detections)")
                
                if merged_count > 0:
                    print(f" {merged_count} detection(s) were merged from duplicates")
            
            print(f"[POST_PROCESSING_SUMMARY] Complete\n")
            
        except Exception as e:
            print(f"[POST_PROCESSING_SUMMARY] Logging failed: {e}")

    def _generate_enhanced_visualization(self, image_rgb: np.ndarray, objects: List[Dict]) -> np.ndarray:
        """生成增强可视化 - 添加详细调试信息"""
        try:
            
            # 检查输入对象数据
            for i, obj in enumerate(objects):
                features = obj.get('features', {})
                if 'appearance' in features:
                    color_name = features['appearance'].get('color_name', 'MISSING')
                    print(f"  - color_name: {color_name}")
                else:
                    print(f"  - appearance features: MISSING")
                
                if 'spatial' in features:
                    height_mm = features['spatial'].get('height_mm', 'MISSING')
                    print(f"  - height_mm: {height_mm}")
                else:
                    print(f"  - spatial features: MISSING")
            
            vis_image = image_rgb.copy().astype(np.float32)
            mean_brightness = np.mean(vis_image)
            if mean_brightness < 120:
                vis_image = np.clip(vis_image * 1.2, 0, 255)
            vis_image = vis_image.astype(np.uint8)
            
            mask_overlay = np.zeros_like(vis_image, dtype=np.float32)
            combined_mask = np.zeros((vis_image.shape[0], vis_image.shape[1]), dtype=bool)
            
            colors = [
                (255, 120, 120), (120, 255, 120), (120, 180, 255), (255, 255, 120),
                (255, 120, 255), (120, 255, 255), (255, 180, 120), (200, 255, 180)
            ]
            
            valid_objects = []
            for i, obj in enumerate(objects):
                mask = obj.get('mask')
                if mask is not None and isinstance(mask, np.ndarray):
                    mask_bool = mask > 0.5 if mask.dtype != bool else mask
                    if mask_bool.shape[:2] == vis_image.shape[:2]:
                        post_info = obj.get('post_processing_info', {})
                        if post_info.get('was_merged', False):
                            color = tuple(min(255, c + 30) for c in colors[i % len(colors)])
                        else:
                            color = colors[i % len(colors)]
                        
                        mask_overlay[mask_bool] = color
                        combined_mask = combined_mask | mask_bool
                        valid_objects.append((i, obj, mask_bool, color))
            
            # 应用mask覆盖
            if len(valid_objects) > 0:
                alpha = 0.3
                vis_image_float = vis_image.astype(np.float32)
                vis_image_float[combined_mask] = (vis_image_float[combined_mask] * (1 - alpha) + 
                                                mask_overlay[combined_mask] * alpha)
                vis_image = np.clip(vis_image_float, 0, 255).astype(np.uint8)
            
            # 绘制对象 - 增强调试版本
            font = cv2.FONT_HERSHEY_SIMPLEX
            for obj_idx, (i, obj, mask_bool, color) in enumerate(valid_objects):
                
                # 计算mask中心点
                center_x, center_y = self._calculate_mask_center(mask_bool)
                bbox = obj['bounding_box']
                x1, y1, x2, y2 = map(int, bbox)
                
                # 绘制边框和中心点
                post_info = obj.get('post_processing_info', {})
                if post_info.get('was_merged', False):
                    cv2.rectangle(vis_image, (x1-2, y1-2), (x2+2, y2+2), (255, 255, 255), 2)
                    cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 3)
                else:
                    cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
                
                # 绘制中心点
                cv2.circle(vis_image, (center_x, center_y), 12, color, -1)
                cv2.circle(vis_image, (center_x, center_y), 15, (255, 255, 255), 2)
                
                # 绘制编号
                number_text = str(obj_idx + 1)
                text_size = cv2.getTextSize(number_text, font, 1.2, 2)[0]
                text_x = center_x - text_size[0] // 2
                text_y = center_y + text_size[1] // 2
                
                # 白色背景 + 黑色文字
                cv2.circle(vis_image, (center_x, center_y), 18, (255, 255, 255), -1)
                cv2.circle(vis_image, (center_x, center_y), 18, color, 3)
                cv2.putText(vis_image, number_text, (text_x, text_y), font, 1.2, (0, 0, 0), 2)
                
                # 生成英文描述 - 添加调试
                try:
                    description = self._generate_english_description(obj)
                except Exception as desc_error:
                    description = f"Object {obj_idx+1} (conf:{obj.get('confidence', 0):.2f})"
                
                # 为合并的检测添加标记
                if post_info.get('was_merged', False):
                    description += f" [M{post_info['merged_from_count']}]"
                
                # 智能标签位置
                label_positions = [(x1, y1 + 5), (x1, y2 - 25), (x2 - 10, y1 - 20)]
                label_x, label_y = label_positions[0]
                for pos_x, pos_y in label_positions:
                    if pos_x >= 0 and pos_y >= 20 and pos_x + 400 < vis_image.shape[1]:
                        label_x, label_y = pos_x, pos_y
                        break
                label_width = max(200, len(description) * 8)  # 确保最小宽度
                label_height = 25
                
                # 确保标签在图像范围内
                if label_x + label_width > vis_image.shape[1]:
                    label_x = vis_image.shape[1] - label_width - 10
                if label_y < 25:
                    label_y = 25
                
                try:
                    # 绘制白色背景
                    cv2.rectangle(vis_image, (label_x - 5, label_y - 20), 
                                (label_x + label_width, label_y + 5), (255, 255, 255), -1)
                    
                    # 绘制彩色边框
                    cv2.rectangle(vis_image, (label_x - 5, label_y - 20), 
                                (label_x + label_width, label_y + 5), color, 2)
                    
                    # 绘制黑色文字
                    cv2.putText(vis_image, description, (label_x, label_y), 
                                font, 0.6, (0, 0, 0), 2) 
                
                    
                except Exception as draw_error:
                    # 备用简化绘制
                    cv2.putText(vis_image, f"Obj{obj_idx+1}", (label_x, label_y), 
                                font, 0.6, (0, 0, 0), 2)
            
            # 添加信息面板
            self._add_post_processing_info_panel(vis_image, objects)
            description_lines = []
            for obj_idx, (i, obj, mask_bool, color) in enumerate(valid_objects):
                features = obj.get('features', {})
                class_name = obj['class_name']
                confidence = obj['confidence']
                object_id = obj.get('object_id', f"{class_name}_{obj_idx}")
                
                color_name = features.get('appearance', {}).get('color_name', '')
                height_mm = features.get('spatial', {}).get('height_mm', 0)
                distance = features.get('spatial', {}).get('distance_to_camera')
                
                desc_parts = []
                if color_name and color_name != 'unknown':
                    desc_parts.append(color_name.capitalize())
                desc_parts.append(class_name)
                
                if height_mm > 0:
                    if height_mm < 20:
                        desc_parts.append("(flat)")
                    elif height_mm < 80:
                        desc_parts.append("(low)")
                    elif height_mm < 120:
                        desc_parts.append("(medium height)")
                    else:
                        desc_parts.append("(tall)")
                
                if distance is not None:
                    if distance < 0.3:
                        desc_parts.append("nearby")
                    elif distance < 0.6:
                        desc_parts.append("at medium distance")
                    else:
                        desc_parts.append("far")
                
                full_description = f"{obj_idx + 1}. {' '.join(desc_parts)} (ID: {object_id}, confidence: {confidence:.3f})"
                description_lines.append(full_description)

            if description_lines:
                line_height = 25
                padding = 20
                bottom_space = len(description_lines) * line_height + padding * 2
                
                # 创建新的扩展画布
                extended_height = vis_image.shape[0] + bottom_space
                extended_image = np.ones((extended_height, vis_image.shape[1], 3), dtype=np.uint8) * 240  # 浅灰背景
                
                # 复制原图到新画布
                extended_image[:vis_image.shape[0], :] = vis_image
                
                # 绘制描述文本
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.8
                font_thickness = 2
                
                for i, desc in enumerate(description_lines):
                    y_pos = vis_image.shape[0] + padding + (i + 1) * line_height
                    cv2.putText(extended_image, desc, (10, y_pos), 
                            font, font_scale, (0, 0, 0), font_thickness)
                
                vis_image = extended_image
            return vis_image
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return None
        
    def _add_post_processing_info_panel(self, vis_image: np.ndarray, objects: List[Dict]):
        """添加后处理信息面板"""
        try:
            # 统计后处理信息
            total_objects = len(objects)
            merged_objects = sum(1 for obj in objects 
                            if obj.get('post_processing_info', {}).get('was_merged', False))
            
            if merged_objects > 0:
                # 在图像底部添加信息条
                panel_height = 40
                panel_y = vis_image.shape[0] - panel_height
                
                # 创建半透明背景
                overlay = vis_image.copy()
                cv2.rectangle(overlay, (0, panel_y), (vis_image.shape[1], vis_image.shape[0]), 
                            (50, 50, 50), -1)
                cv2.addWeighted(overlay, 0.8, vis_image, 0.2, 0, vis_image)
                
                # 添加文字信息
                font = cv2.FONT_HERSHEY_SIMPLEX
                info_text = f"3D Post-Processing: {merged_objects}/{total_objects} objects merged from duplicates"
                
                cv2.putText(vis_image, info_text, (10, panel_y + 25), 
                        font, 0.6, (255, 255, 255), 2)
                
                # 添加图例
                legend_text = "[M#] = Merged from # detections"
                cv2.putText(vis_image, legend_text, (vis_image.shape[1] - 300, panel_y + 25), 
                        font, 0.5, (200, 200, 200), 1)
            
        except Exception as e:
            print(f"[POST_PROCESSING_INFO] Adding info panel failed: {e}")

    # 4. 🆕 添加配置方法
    def set_post_processing_config(self, config: Dict):
        """设置后处理配置"""
        try:
            if hasattr(self, 'post_processor'):
                # 更新后处理器参数
                if 'spatial_distance_threshold' in config:
                    self.post_processor.spatial_distance_threshold = config['spatial_distance_threshold']
                if 'depth_similarity_threshold' in config:
                    self.post_processor.depth_similarity_threshold = config['depth_similarity_threshold']
                if 'mask_overlap_threshold' in config:
                    self.post_processor.mask_overlap_threshold = config['mask_overlap_threshold']
                if 'height_similarity_threshold' in config:
                    self.post_processor.height_similarity_threshold = config['height_similarity_threshold']
                
                print(f"[ENHANCED_PIPELINE] Post-processing config updated: {config}")
        
        except Exception as e:
            print(f"[ENHANCED_PIPELINE] Post-processing config update failed: {e}")
    def _calculate_mask_center(self, mask: np.ndarray) -> Tuple[int, int]:
        """计算mask的几何中心点"""
        try:
            # 找到mask的所有像素
            mask_indices = np.where(mask > 0)
            if len(mask_indices[0]) == 0:
                return 100, 100
            
            # 计算质心
            center_y = int(np.mean(mask_indices[0]))
            center_x = int(np.mean(mask_indices[1]))
            
            # 确保在图像范围内
            center_x = max(10, min(mask.shape[1] - 10, center_x))
            center_y = max(10, min(mask.shape[0] - 10, center_y))
            
            return center_x, center_y
            
        except Exception:
            return 100, 100

    def _generate_english_description(self, obj: Dict) -> str:
        """生成简洁的英文描述"""
        try:
            features = obj.get('features', {})
            class_name = obj['class_name']
            
            # 颜色信息
            color_name = features.get('appearance', {}).get('color_name', '')
            if color_name and color_name != 'unknown':
                color_desc = color_name.capitalize()
            else:
                color_desc = ''
            
            # 组合描述
            parts = []
            if color_desc:
                parts.append(color_desc)
            parts.append(class_name)
            
            return ' '.join(parts)
            
        except Exception:
            return f"{obj.get('class_name', 'object')} conf:{obj.get('confidence', 0):.2f}"
    
    def _extract_enhanced_features(self, image_rgb: np.ndarray, mask: np.ndarray,
                                depth_image: np.ndarray, waypoint_data: Dict,
                                bbox: List[int]) -> Dict:
        """提取增强的多层次特征 - 调试版本"""
        features = {}
        
        try:
            if depth_image is not None:
                try:
                    geometric_features = self.enhanced_shape_extractor.extract_all_features(
                        mask, depth_image, waypoint_data
                    )
                    features['geometric'] = geometric_features
                            
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    features['geometric'] = {}
            else:
                features['geometric'] = {}
            
            # 2. 形状特征（中等稳定）
            try:
                shape_features = self.enhanced_shape_extractor._extract_2d_shape_features(mask)
                features['shape'] = shape_features
            except Exception as e:
                features['shape'] = {}
            
            # 3. 外观特征（较不稳定 - 权重较低）
            try:
                color_stats = self.color_extractor.get_color_statistics(image_rgb, mask)
                # 使用改进的颜色直方图
                improved_histogram = self._extract_improved_color_histogram(image_rgb, mask)
                color_stats['histogram'] = improved_histogram
                color_stats['color_name'] = self._improve_color_detection(image_rgb, mask)
                features['appearance'] = color_stats
            except Exception as e:
                features['appearance'] = {}
            
            # 4. 空间上下文特征（辅助信息）
            try:
                if depth_image is not None and waypoint_data is not None:
                    spatial_features = self.object_analyzer.calculate_3d_spatial_features(
                        mask, depth_image, waypoint_data, bbox
                    )
                    
                    # 添加高度和背景信息
                    height_info = self.object_analyzer.calculate_object_height_and_background(
                        mask, depth_image, bbox, waypoint_data
                    )
                    spatial_features.update(height_info)
                    
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
                import traceback
                traceback.print_exc()
                features['spatial'] = {}
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            features['extraction_error'] = str(e)
        
        return features
    
    def _extract_improved_color_histogram(self, image: np.ndarray, mask: np.ndarray, bins: int = 32) -> List[float]:
        """改进的颜色直方图提取"""
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
    
    def _improve_color_detection(self, image: np.ndarray, mask: np.ndarray) -> str:
        """改进的颜色检测"""
        if np.sum(mask) == 0:
            return "unknown"
        
        masked_pixels = image[mask > 0]
        if len(masked_pixels) == 0:
            return "unknown"
        
        # 使用中位数而不是平均值
        median_color = np.median(masked_pixels, axis=0)
        
        # 转换为HSV进行颜色分类
        hsv_color = cv2.cvtColor(np.uint8([[median_color]]), cv2.COLOR_RGB2HSV)[0][0]
        h, s, v = hsv_color
        
        # 改进的颜色分类逻辑
        if s < 30:  # 低饱和度
            if v > 200:
                return "white"
            elif v < 60:
                return "black"
            else:
                return "gray"
        
        # 高饱和度颜色分类
        if h < 15 or h > 165:
            return "red"
        elif 15 <= h < 35:
            return "yellow" if median_color[0] > 180 and median_color[1] > 150 else "orange"
        elif 35 <= h < 85:
            return "green"
        elif 85 <= h < 125:
            return "cyan"
        elif 125 <= h < 165:
            return "blue"
        
        return "mixed"
    
    def _assign_object_id(self, class_id: int) -> int:
        """为对象分配ID"""
        if class_id not in self.next_object_id:
            self.next_object_id[class_id] = 0
        else:
            self.next_object_id[class_id] += 1
        
        return self.next_object_id[class_id]
    
    def _generate_enhanced_description(self, class_name: str, features: Dict, object_id: int) -> str:
        """生成增强描述"""
        try:
            # 基础信息
            description_parts = []
            
            # 颜色信息
            color_name = features.get('appearance', {}).get('color_name', 'unknown')
            if color_name and color_name != 'unknown':
                description_parts.append(color_name.capitalize())
            
            # 对象名称
            description_parts.append(class_name)
            
            # 高度信息
            height_mm = features.get('spatial', {}).get('height_mm')
            if height_mm is not None:
                if height_mm < 20:
                    description_parts.append("(flat)")
                elif height_mm < 80:
                    description_parts.append("(low)")
                elif height_mm < 120:
                    description_parts.append("(medium height)")
                else:
                    description_parts.append("(tall)")
            
            # 距离信息
            distance = features.get('spatial', {}).get('distance_to_camera')
            if distance is not None:
                if distance < 0.3:
                    description_parts.append("nearby")
                elif distance < 0.6:
                    description_parts.append("at medium distance")
                else:
                    description_parts.append("far")
            
            enhanced_description = ' '.join(description_parts)
            
            # 添加技术信息
            tech_info = []
            if height_mm is not None:
                tech_info.append(f"h={height_mm:.1f}mm")
            if distance is not None:
                tech_info.append(f"d={distance*1000:.0f}mm")
            
            # 添加质量信息
            quality_score = self.quality_assessor.assess_feature_quality(features)
            tech_info.append(f"q={quality_score:.0f}%")
            
            if tech_info:
                enhanced_description += f" [{', '.join(tech_info)}]"
            
            return enhanced_description
            
        except Exception as e:
            return f"{class_name}_{object_id}"
    
    def _save_reference_library(self, reference_library: Dict):
        """保存参考特征库"""
        try:
            # 创建可序列化的数据
            serializable_library = {}
            
            for obj_id, entry in reference_library.items():
                # 深拷贝并处理numpy数组
                serializable_entry = {
                    'features': self._sanitize_features_for_json(entry['features']),
                    'quality_score': float(entry['quality_score']),
                    'metadata': entry['metadata']
                }
                serializable_library[obj_id] = serializable_entry
            
            # 保存主要数据文件
            library_file = os.path.join(self.output_dir, "reference_library.json")
            with open(library_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_library, f, indent=2, ensure_ascii=False)
            
            # 保存摘要文件
            self._save_library_summary(reference_library)
            
            print(f"[ENHANCED_PIPELINE] Reference feature library has been saved to: {library_file}")
            
        except Exception as e:
            print(f"[ENHANCED_PIPELINE] Failed to save reference feature library: {e}")
    
    def _sanitize_features_for_json(self, features: Dict) -> Dict:
        """清理特征数据用于JSON序列化"""
        sanitized = {}
        
        for feature_type, feature_data in features.items():
            if isinstance(feature_data, dict):
                sanitized[feature_type] = {}
                for key, value in feature_data.items():
                    if isinstance(value, np.ndarray):
                        sanitized[feature_type][key] = value.tolist()
                    elif isinstance(value, (np.integer, np.floating)):
                        sanitized[feature_type][key] = float(value)
                    elif isinstance(value, tuple):
                        sanitized[feature_type][key] = list(value)
                    else:
                        sanitized[feature_type][key] = value
            elif isinstance(feature_data, np.ndarray):
                sanitized[feature_type] = feature_data.tolist()
            elif isinstance(feature_data, (np.integer, np.floating)):
                sanitized[feature_type] = float(feature_data)
            else:
                sanitized[feature_type] = feature_data
        
        return sanitized
    
    def _save_library_summary(self, reference_library: Dict):
        """保存特征库摘要"""
        try:
            summary_file = os.path.join(self.output_dir, "library_summary.txt")
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write("Reference Feature Library Summary\n")
                f.write("=" * 50 + "\n\n")
                
                f.write(f"Total Objects: {len(reference_library)}\n")
                f.write(f"Creation Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # 按类别分组
                by_class = {}
                for obj_id, entry in reference_library.items():
                    class_name = entry['metadata']['class_name']
                    if class_name not in by_class:
                        by_class[class_name] = []
                    by_class[class_name].append((obj_id, entry['quality_score']))
                
                f.write("Objects by Class:\n")
                f.write("-" * 30 + "\n")
                for class_name, objects in by_class.items():
                    avg_quality = np.mean([q for _, q in objects])
                    f.write(f"{class_name}: {len(objects)} objects (avg quality: {avg_quality:.1f}%)\n")
                    for obj_id, quality in objects:
                        f.write(f"  - {obj_id} (quality: {quality:.1f}%)\n")
                    f.write("\n")
            
            print(f"[ENHANCED_PIPELINE] Feature library summary saved to: {summary_file}")
            
        except Exception as e:
            print(f"[ENHANCED_PIPELINE] Failed to save summary: {e}")

    
    def get_reference_library(self) -> Dict:
        """获取参考特征库"""
        return self.reference_library
    
    def get_output_dir(self) -> str:
        """获取输出目录"""
        return self.output_dir
    
    def select_tracking_targets(self) -> bool:
        """Selects the targets to be tracked"""
        if not hasattr(self, 'reference_library') or not self.reference_library:
            print("[ENHANCED_PIPELINE] No reference to feature library data, please perform a test first")
            return False
        
        objects = list(self.reference_library.values())
        
        print(f"\n{'='*50}")
        print("Select Tracking Targets")
        print(f"{'='*50}")
        
        print("\nDetected Objects:")
        for i, (obj_id, entry) in enumerate(self.reference_library.items(), 1):
            metadata = entry['metadata']
            quality = entry['quality_score']
            print(f"{i:2d}. {metadata['description']} (ID: {obj_id}, Confidence: {metadata['confidence']:.3f}, Quality: {quality:.1f}%)")
        
        print("\nPlease select the objects to track:")
        print("Format: Enter numbers separated by commas (e.g., 1,3,5)")
        
        while True:
            try:
                user_input = input("\nYour selection: ").strip()
                
                if user_input.lower() == 'all':
                    self.selected_tracking_ids = list(self.reference_library.keys())
                    break
                
                indices = [int(x.strip()) for x in user_input.split(',')]
                selected_ids = []
                
                obj_list = list(self.reference_library.keys())
                for idx in indices:
                    if 1 <= idx <= len(obj_list):
                        selected_ids.append(obj_list[idx-1])
                    else:
                        print(f"Warning: Index {idx} is out of range")
                
                if selected_ids:
                    self.selected_tracking_ids = selected_ids
                    break
                else:
                    print("No valid selection, please try again.")
                    
            except ValueError:
                print("Invalid input format. Please enter numbers separated by commas.")
            except KeyboardInterrupt:
                print("\nSelection canceled.")
                return False
        
        print(f"\nSelected tracking targets:")
        for obj_id in self.selected_tracking_ids:
            entry = self.reference_library[obj_id]
            metadata = entry['metadata']
            print(f"  - {metadata['description']} (ID: {obj_id})")
        
        # Save the selection
        self._save_tracking_selection()
        
        return True

    def _save_tracking_selection(self):
        """Save tracking selection"""
        try:
            selection_file = os.path.join(self.output_dir, "tracking_selection.txt")
            
            with open(selection_file, 'w', encoding='utf-8') as f:
                f.write("Selected tracking targets:\n")
                f.write("=" * 50 + "\n")
                
                for obj_id in getattr(self, 'selected_tracking_ids', []):
                    if obj_id in self.reference_library:
                        entry = self.reference_library[obj_id]
                        metadata = entry['metadata']
                        f.write(f"- {metadata['description']} (ID: {obj_id})\n")
            
            print(f"[ENHANCED_PIPELINE] Tracking selection saved to: {selection_file}")
            
        except Exception as e:
            print(f"[ENHANCED_PIPELINE] Failed to save tracking selection: {e}")

    def get_selected_tracking_ids(self) -> List[str]:
        """Get selected tracking target IDs"""
        return getattr(self, 'selected_tracking_ids', [])
    
    def _display_and_publish_visualization(self, vis_image: np.ndarray, output_dir: str):
        """Display and publish visualization image"""
        try:
            if vis_image is None:
                print("[DISPLAY] Visualization image is empty, skipping display")
                return
            
            # Save image
            vis_file = os.path.join(output_dir, "detection_visualization.jpg")
            success = cv2.imwrite(vis_file, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
            
            if success:
                print(f"[DISPLAY] Visualization image saved: {vis_file}")
            else:
                print(f"[DISPLAY] Failed to save visualization image: {vis_file}")
                return
            
            # 🖼️ Try to display image
            self._show_visualization_popup(vis_image)
            
            # 📡 Publish ROS visualization message
            self._publish_visualization_message(vis_image)
            
        except Exception as e:
            print(f"[DISPLAY] Failed to display and publish visualization: {e}")

    def _show_visualization_popup(self, vis_image: np.ndarray):
        """Show visualization popup"""
        try:
            import os
            
            # Check display environment
            display_mode = os.environ.get('VISION_AI_DISPLAY', 'auto')
            
            if display_mode == 'off':
                print("[DISPLAY] Display function disabled")
                return
            
            # Try matplotlib display
            try:
                import matplotlib
                matplotlib.use('TkAgg')  # Use TkAgg backend
                import matplotlib.pyplot as plt
                
                plt.figure(figsize=(16, 12))
                plt.imshow(vis_image)
                plt.title('Enhanced Detection Results', fontsize=26, fontweight='bold')
                plt.axis('off')
                
                # Add description text
                plt.figtext(0.5, 0.02, 
                        'Enhanced detection with 3D point cloud features. Numbers indicate object centers in 3D space.',
                        ha='center', fontsize=20, style='italic')
                
                plt.tight_layout()
                plt.show(block=False)  # Non-blocking display
                plt.pause(0.1)
                print("[DISPLAY] matplotlib popup display successful")
                
            except Exception as e:
                print(f"[DISPLAY] matplotlib display failed: {e}")
                
                # Try OpenCV display
                try:
                    cv2.namedWindow('Enhanced Detection Results', cv2.WINDOW_NORMAL)
                    cv2.resizeWindow('Enhanced Detection Results', 1200, 800)
                    cv2.imshow('Enhanced Detection Results', cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
                    cv2.waitKey(1)  # Non-blocking
                    print("[DISPLAY] OpenCV window display successful")
                    
                except Exception as e2:
                    print(f"[DISPLAY] OpenCV display also failed: {e2}")
                    print("[DISPLAY] Unable to display visualization window, only saving file")
            
        except Exception as e:
            print(f"[DISPLAY] Failed to show visualization popup: {e}")

    def _collect_performance_data(self, start_timestamp, total_time, stage_times, 
                                  original_count, filtered_count, reference_library, 
                                  visualization_objects):
        """收集详细的性能数据"""
        # 计算质量分数统计
        quality_scores = [entry['quality_score'] for entry in reference_library.values()]
        
        # 计算置信度统计
        confidences = []
        feature_stats = {'geometric': 0, 'shape': 0, 'appearance': 0, 'spatial': 0}
        
        for viz_obj in visualization_objects:
            confidences.append(viz_obj['confidence'])
            features = viz_obj.get('features', {})
            for feature_type in feature_stats.keys():
                if feature_type in features and features[feature_type]:
                    feature_stats[feature_type] += 1
        
        performance_data = {
            'timestamp': start_timestamp.isoformat(),
            'test_metadata': self.current_test_metadata,
            'timing': {
                'total_time': total_time,
                'stage_breakdown': stage_times,
                'fps': 1.0 / total_time if total_time > 0 else 0
            },
            'detection_stats': {
                'original_detections': original_count,
                'filtered_detections': filtered_count,
                'final_objects': len(reference_library),
                'duplicates_removed': original_count - filtered_count,
                'removal_rate': (original_count - filtered_count) / original_count if original_count > 0 else 0
            },
            'quality_stats': {
                'mean_quality': np.mean(quality_scores) if quality_scores else 0,
                'std_quality': np.std(quality_scores) if quality_scores else 0,
                'min_quality': np.min(quality_scores) if quality_scores else 0,
                'max_quality': np.max(quality_scores) if quality_scores else 0,
                'quality_distribution': self._calculate_quality_distribution(quality_scores)
            },
            'confidence_stats': {
                'mean_confidence': np.mean(confidences) if confidences else 0,
                'std_confidence': np.std(confidences) if confidences else 0,
                'min_confidence': np.min(confidences) if confidences else 0,
                'max_confidence': np.max(confidences) if confidences else 0
            },
            'feature_extraction_success': feature_stats,
            'post_processing': {
                'enabled': self.enable_3d_post_processing,
                'effectiveness': (original_count - filtered_count) / original_count if original_count > 1 else 0
            }
        }
        
        # 保存到实例变量中
        self.performance_data.append(performance_data)
        
        return performance_data
    def _collect_error_performance_data(self, start_timestamp, total_time, error_message):
        """收集错误情况下的性能数据"""
        return {
            'timestamp': start_timestamp.isoformat(),
            'test_metadata': self.current_test_metadata,
            'timing': {'total_time': total_time, 'fps': 0},
            'detection_stats': {'error': True, 'error_message': error_message},
            'success': False
        }
    
    def _calculate_quality_distribution(self, quality_scores):
        """计算质量分数分布"""
        if not quality_scores:
            return {'excellent': 0, 'good': 0, 'fair': 0, 'poor': 0}
        
        distribution = {'excellent': 0, 'good': 0, 'fair': 0, 'poor': 0}
        for score in quality_scores:
            if score >= 90:
                distribution['excellent'] += 1
            elif score >= 75:
                distribution['good'] += 1
            elif score >= 60:
                distribution['fair'] += 1
            else:
                distribution['poor'] += 1
        
        return distribution
    
    def _create_empty_result(self, start_time, stage_times):
        """创建空结果"""
        total_time = time.time() - start_time
        performance_data = {
            'timestamp': datetime.now().isoformat(),
            'test_metadata': self.current_test_metadata,
            'timing': {'total_time': total_time, 'stage_breakdown': stage_times},
            'detection_stats': {'original_detections': 0, 'filtered_detections': 0, 'final_objects': 0},
            'no_detections': True
        }
        
        return {
            'success': True,
            'reference_library': {},
            'detection_count': 0,
            'processing_time': total_time,
            'performance_data': performance_data,
            'message': 'No targets detected'
        }
    
    def _build_initial_detections(self, boxes, class_ids, confidences, masks):
        """构建初始检测结果"""
        initial_detections = []
        class_names = self.detector.get_class_names()
        
        for i, (box, class_id, confidence, mask) in enumerate(zip(boxes, class_ids, confidences, masks)):
            class_name = class_names.get(class_id, f'class_{class_id}')
            detection = {
                'bounding_box': box.tolist(),
                'class_id': int(class_id),
                'class_name': class_name,
                'confidence': float(confidence),
                'mask': mask,
                'original_index': i
            }
            initial_detections.append(detection)
        
        return initial_detections
    
    def _extract_features_and_build_library(self, filtered_detections, image_rgb, depth_image, waypoint_data):
        """提取特征并构建参考库"""
        reference_library = {}
        visualization_objects = []
        
        for i, detection in enumerate(filtered_detections):
            try:
                object_id = self._assign_object_id(detection['class_id'])
                class_name = detection['class_name']
                full_object_id = f"{class_name}_{object_id}"
                
                features = self._extract_enhanced_features(
                    image_rgb, detection['mask'], depth_image, waypoint_data, detection['bounding_box']
                )
                
                quality_score = self.quality_assessor.assess_feature_quality(features)
                description = self._generate_enhanced_description(class_name, features, object_id)
                
                post_processing_info = {
                    'was_merged': detection.get('merged_from', 1) > 1,
                    'merged_from_count': detection.get('merged_from', 1),
                    'original_confidences': detection.get('original_confidences', [detection['confidence']]),
                    'post_processed': self.enable_3d_post_processing
                }
                
                reference_entry = {
                    'features': features,
                    'quality_score': quality_score,
                    'post_processing_info': post_processing_info,
                    'metadata': {
                        'object_id': full_object_id,
                        'class_id': int(detection['class_id']),
                        'class_name': class_name,
                        'confidence': float(detection['confidence']),
                        'bounding_box': detection['bounding_box'],
                        'description': description,
                        'detection_timestamp': datetime.now().isoformat(),
                        'waypoint_info': {
                            'world_pos': waypoint_data['world_pos'],
                            'orientation': [waypoint_data['roll'], waypoint_data['pitch'], waypoint_data['yaw']]
                        }
                    }
                }
                
                reference_library[full_object_id] = reference_entry
                
                viz_obj = {
                    'object_id': full_object_id,
                    'class_id': int(detection['class_id']),
                    'class_name': class_name,
                    'confidence': float(detection['confidence']),
                    'bounding_box': detection['bounding_box'],
                    'mask': detection['mask'],
                    'features': features,
                    'description': description,
                    'quality_score': quality_score,
                    'post_processing_info': post_processing_info
                }
                visualization_objects.append(viz_obj)
                
            except Exception as e:
                print(f"[ENHANCED_PIPELINE] Error processing detection {i}: {e}")
                continue
        
        return reference_library, visualization_objects
    
    def get_all_performance_data(self):
        """获取所有收集的性能数据"""
        return self.performance_data
    
    def clear_performance_data(self):
        """清空性能数据"""
        self.performance_data = []
    
    def save_performance_data(self, filename):
        """保存性能数据到文件"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.performance_data, f, indent=2, ensure_ascii=False)
            print(f"[ENHANCED_PIPELINE] Performance data saved to: {filename}")
        except Exception as e:
            print(f"[ENHANCED_PIPELINE] Failed to save performance data: {e}")

    def _publish_visualization_message(self, vis_image: np.ndarray):
        """Publish ROS visualization message"""
        try:
            print("[PUBLISH] Visualization message publishing interface reserved")
            
        except Exception as e:
            print(f"[PUBLISH] Failed to publish visualization message: {e}")
    
    def process_single_image(self, image_rgb: np.ndarray, 
                        depth_image: Optional[np.ndarray] = None,
                        camera_pose: Optional[Dict] = None) -> Dict:
        try:
            # print(f"[ENHANCED_PIPELINE] Starting to process single image...")
            
            # 🔧 Ensure RGB and depth image resolutions match
            if image_rgb.shape[:2] != depth_image.shape:
                print(f'[ENHANCED_PIPELINE] Adjusting depth image resolution: {depth_image.shape} -> {image_rgb.shape[:2]}')
                depth_image = cv2.resize(
                    depth_image, 
                    (image_rgb.shape[1], image_rgb.shape[0]), 
                    interpolation=cv2.INTER_LINEAR
                )
            
            # 1. YOLO detection
            boxes, class_ids, confidences = self.detector.detect(image_rgb)
            
            if len(boxes) == 0:
                print("[ENHANCED_PIPELINE] No targets detected")
                return {'objects': []}
            
            # print(f"[ENHANCED_PIPELINE] YOLO detected {len(boxes)} targets")
            
            # 2. SAM2 segmentation
            masks = self.segmentor.segment(image_rgb, boxes)
            # print("[ENHANCED_PIPELINE] SAM2 segmentation completed")
            
            # 3. Process detection results (no 3D post-processing to avoid merging)
            objects = []
            class_names = self.detector.get_class_names()
            
            for i, (box, class_id, confidence, mask) in enumerate(zip(boxes, class_ids, confidences, masks)):
                try:
                    class_name = class_names.get(class_id, f'class_{class_id}')
                    
                    # 🔧 Safe feature extraction, ensure mask and image dimensions match
                    if mask.shape != image_rgb.shape[:2]:
                        print(f'[ENHANCED_PIPELINE] Adjusting mask size: {mask.shape} -> {image_rgb.shape[:2]}')
                        mask = cv2.resize(
                            mask.astype(np.uint8), 
                            (image_rgb.shape[1], image_rgb.shape[0]), 
                            interpolation=cv2.INTER_NEAREST
                        ).astype(bool)
                    
                    # 🔧 Build waypoint data (if camera_pose is not provided)
                    if camera_pose is None:
                        camera_pose = {
                            'world_pos': [0, 0, 350],
                            'roll': 179, 'pitch': 0, 'yaw': 0
                        }
                    
                    # Extract enhanced features
                    features = self._extract_enhanced_features(
                        image_rgb, mask, depth_image, camera_pose, box.tolist()
                    )
                    
                    obj = {
                        'bounding_box': box.tolist(),
                        'confidence': float(confidence),
                        'class_id': int(class_id),
                        'class_name': class_name,
                        'mask': mask,
                        'features': features
                    }
                    
                    objects.append(obj)
                    
                except Exception as e:
                    print(f'[ENHANCED_PIPELINE] Error processing object {i}: {e}')
                    continue
            
            # print(f"[ENHANCED_PIPELINE] Single image processing completed, detected {len(objects)} valid targets")
            
            return {
                'objects': objects,
                'image_shape': image_rgb.shape,
                'depth_shape': depth_image.shape if depth_image is not None else None
            }
            
        except Exception as e:
            print(f"[ENHANCED_PIPELINE] Single image processing failed: {e}")
            import traceback
            traceback.print_exc()
            return {'objects': []}