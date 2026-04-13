# detection/utils/detection_post_processor.py
import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist

class Detection3DPostProcessor:
    """基于3D/深度信息的检测后处理器"""
    
    def __init__(self, coordinate_calculator):
        """
        初始化后处理器
        
        Args:
            coordinate_calculator: 坐标计算器实例
        """
        self.coord_calc = coordinate_calculator
        
        # 重复检测过滤参数
        self.spatial_distance_threshold = 50.0  # 3D空间距离阈值(mm)
        self.depth_similarity_threshold = 20.0  # 深度相似性阈值(mm)
        self.mask_overlap_threshold = 0.3      # mask重叠阈值
        self.height_similarity_threshold = 15.0 # 高度相似性阈值(mm)
        
        # DBSCAN聚类参数
        self.dbscan_eps = 30.0  # 聚类距离阈值(mm)
        self.dbscan_min_samples = 1  # 最小样本数
        
        print("[POST_PROCESSOR] Detection 3D post-processor initialized")
    
    def process_detections(self, detections: List[Dict], image_rgb: np.ndarray, 
                          depth_image: np.ndarray, waypoint_data: Dict) -> List[Dict]:
        """
        处理检测结果，过滤重复检测
        
        Args:
            detections: 原始检测结果列表
            image_rgb: RGB图像
            depth_image: 深度图像  
            waypoint_data: waypoint数据
            
        Returns:
            filtered_detections: 过滤后的检测结果
        """
        try:
            if len(detections) <= 1:
                print(f"[POST_PROCESSOR] Only {len(detections)} detection(s), no filtering needed")
                return detections
            
            print(f"[POST_PROCESSOR] Processing {len(detections)} detections")
            self.similarity_analyses = []
            # Step 1: 提取3D特征
            detection_features = []
            for i, detection in enumerate(detections):
                features = self._extract_3d_features(detection, depth_image, waypoint_data)
                features['original_index'] = i
                features['detection'] = detection
                detection_features.append(features)
            
            # Step 2: 计算相似性矩阵
            similarity_matrix = self._compute_similarity_matrix(detection_features)
            
            # Step 3: 识别重复组
            duplicate_groups = self._identify_duplicate_groups(similarity_matrix, detection_features)
            
            # Step 4: 合并重复检测
            filtered_detections = self._merge_duplicate_detections(duplicate_groups, detection_features)
            
            print(f"[POST_PROCESSOR] Filtered {len(detections)} -> {len(filtered_detections)} detections")
            self._log_filtering_results(detections, filtered_detections)
            self.last_processing_stats = {
                'original_detections': len(detections),
                'filtered_detections': len(filtered_detections),
                'merge_operations': [],
                'filter_reasons': {
                    'low_depth_consistency': 0,
                    'insufficient_spatial_support': 0,
                    'geometric_outlier': 0
                },
                'depth_consistency_analysis': self._analyze_batch_depth_consistency(detection_features)
            }
            self.detailed_processing_stats = {
                'input_summary': {
                    'total_detections': len(detections),
                    'classes_detected': list(set(d['class_name'] for d in detections)),
                    'confidence_range': [min(d['confidence'] for d in detections), 
                                    max(d['confidence'] for d in detections)] if detections else [0, 0]
                },
                'filtering_process': {
                    'similarity_comparisons': len(self.similarity_analyses),
                    'merge_operations': len([g for g in duplicate_groups if len(g) > 1]),
                    'preserved_detections': len([g for g in duplicate_groups if len(g) == 1])
                },
                'output_summary': {
                    'final_detections': len(filtered_detections),
                    'reduction_rate': (len(detections) - len(filtered_detections)) / len(detections) if detections else 0,
                    'avg_confidence_change': self._calculate_confidence_change(detections, filtered_detections)
                },
                'physical_validation_summary': self._summarize_physical_validation(detection_features)
            }
            # 在 return filtered_detections 前
            print(f"[POST_PROCESSOR] 设置统计数据: detailed_processing_stats={hasattr(self, 'detailed_processing_stats')}")
            if hasattr(self, 'detailed_processing_stats'):
                print(f"[POST_PROCESSOR] 统计数据内容: {self.detailed_processing_stats}")
            return filtered_detections
            
        except Exception as e:
            print(f"[POST_PROCESSOR] Processing failed: {e}")
            import traceback
            traceback.print_exc()
            return detections  # 返回原始检测结果
    

    def _summarize_physical_validation(self, detection_features: List[Dict]) -> Dict:
        """总结物理验证结果"""
        validations = [feat.get('physical_validation', {}) for feat in detection_features]
        
        return {
            'valid_3d_positions': sum(v.get('has_valid_3d_position', False) for v in validations),
            'reliable_depth_measurements': sum(v.get('has_reliable_depth', False) for v in validations),
            'sufficient_coverage': sum(v.get('sufficient_mask_coverage', False) for v in validations),
            'reasonable_heights': sum(v.get('height_within_reasonable_range', False) for v in validations)
        }

    def _calculate_confidence_change(self, original: List[Dict], filtered: List[Dict]) -> float:
        """计算置信度变化"""
        if not original or not filtered:
            return 0.0
        
        orig_avg = np.mean([d['confidence'] for d in original])
        filt_avg = np.mean([d['confidence'] for d in filtered])
        
        return float(filt_avg - orig_avg)

    def _analyze_batch_depth_consistency(self, detection_features: List[Dict]) -> Dict:
        """分析整批检测的深度一致性"""
        try:
            consistency_scores = []
            depth_variances = []
            valid_ratios = []
            peak_counts = []
            
            for feat in detection_features:
                if 'depth_consistency_score' in feat:
                    consistency_scores.append(feat['depth_consistency_score'])
                    depth_variances.append(feat.get('depth_variance', 0))
                    valid_ratios.append(feat.get('valid_depth_ratio', 0))
                    peak_counts.append(feat.get('depth_peaks', 0))
            
            if not consistency_scores:
                return {'batch_consistency': 'no_data'}
            
            return {
                'mean_consistency_score': float(np.mean(consistency_scores)),
                'std_consistency_score': float(np.std(consistency_scores)),
                'high_quality_detections': sum(1 for s in consistency_scores if s > 0.8),
                'low_quality_detections': sum(1 for s in consistency_scores if s < 0.4),
                'mean_depth_variance': float(np.mean(depth_variances)),
                'multi_peak_detections': sum(1 for p in peak_counts if p > 1)
            }
            
        except Exception as e:
            print(f"[POST_PROCESSOR] Batch depth consistency analysis failed: {e}")
            return {'batch_consistency': 'analysis_failed'}
    
    def _extract_3d_features(self, detection: Dict, depth_image: np.ndarray, 
                            waypoint_data: Dict) -> Dict:
        """提取检测的3D特征"""
        try:
            bbox = detection['bounding_box']
            mask = detection['mask']
            
            # 基本3D特征
            features = {
                'bbox': bbox,
                'class_id': detection['class_id'],
                'class_name': detection['class_name'],
                'confidence': detection['confidence']
            }
            
            # 计算3D质心
            world_coords = self._calculate_3d_centroid(mask, depth_image, waypoint_data)
            features['world_centroid'] = world_coords
            
            # 计算平均深度
            avg_depth = self._calculate_average_depth(mask, depth_image)
            features['average_depth'] = avg_depth
            
            # 计算物体高度
            height = self._estimate_object_height(mask, depth_image, bbox)
            features['estimated_height'] = height
            
            # 计算mask面积
            mask_area = np.sum(mask > 0) if mask is not None else 0
            features['mask_area'] = mask_area
            
            # 计算深度变化
            depth_variance = self._calculate_depth_variance(mask, depth_image)
            features['depth_variance'] = depth_variance
            depth_analysis = self._analyze_depth_consistency(mask, depth_image)
            features.update(depth_analysis)
            # 添加物理属性验证标记
            features['physical_validation'] = {
                'has_valid_3d_position': features['world_centroid'] != [0, 0, 0],
                'has_reliable_depth': features['depth_variance'] < 0.05,
                'sufficient_mask_coverage': features['mask_area'] > 100,
                'height_within_reasonable_range': 5 < features['estimated_height'] < 200
            }
            
            return features
        except Exception as e:
            print(f"[POST_PROCESSOR] 3D feature extraction failed: {e}")
            return {
                'bbox': detection.get('bounding_box', [0, 0, 100, 100]),
                'class_id': detection.get('class_id', 0),
                'class_name': detection.get('class_name', 'unknown'),
                'confidence': detection.get('confidence', 0),
                'world_centroid': [0, 0, 0],
                'average_depth': 0.5,
                'estimated_height': 30.0,
                'mask_area': 0,
                'depth_variance': 0.0
            }
    
    def _analyze_depth_consistency(self, mask: np.ndarray, depth_image: np.ndarray) -> Dict:
        """分析深度一致性，用于理论依据"""
        if mask is None or np.sum(mask) == 0:
            return {
                'depth_consistency_score': 0.0,
                'depth_uniformity': 'invalid',
                'reflection_likelihood': 'unknown'
            }
        
        masked_depth = depth_image[mask > 0] / 1000.0  # 转换为米
        valid_depths = masked_depth[masked_depth > 0.01]
        
        if len(valid_depths) < 10:
            return {
                'depth_consistency_score': 0.0,
                'depth_uniformity': 'insufficient_data'
            }
        
        # 计算深度统计
        depth_median = np.median(valid_depths)
        depth_std = np.std(valid_depths)
        depth_range = np.max(valid_depths) - np.min(valid_depths)
        
        # 计算一致性分数
        consistency_score = max(0, 1 - (depth_std / (depth_median + 1e-6)) * 2)
        
        # 判断深度均匀性
        if depth_std < 0.01:  # 1cm以内
            uniformity = 'highly_uniform'
        elif depth_std < 0.03:  # 3cm以内  
            uniformity = 'moderately_uniform'
        else:
            uniformity = 'non_uniform'
        
        # 反射可能性判断
        if depth_median > 2.0 or depth_std > 0.1:
            reflection_risk = 'high'
        elif depth_std > 0.05:
            reflection_risk = 'medium' 
        else:
            reflection_risk = 'low'
        
        return {
            'depth_consistency_score': float(consistency_score),
            'depth_std_mm': float(depth_std * 1000),
            'depth_range_mm': float(depth_range * 1000),
            'depth_uniformity': uniformity,
            'reflection_likelihood': reflection_risk,
            'valid_depth_points': len(valid_depths),
            'depth_coverage_ratio': len(valid_depths) / np.sum(mask > 0)
        }

    def _calculate_3d_centroid(self, mask: np.ndarray, depth_image: np.ndarray, 
                              waypoint_data: Dict) -> List[float]:
        """计算3D质心"""
        try:
            if mask is None:
                return [0, 0, 0]
            
            mask_indices = np.where(mask > 0)
            if len(mask_indices[0]) == 0:
                return [0, 0, 0]
            
            # 采样计算3D点
            valid_points = []
            for i in range(0, len(mask_indices[0]), 10):  # 每10个像素采样一次
                y, x = mask_indices[0][i], mask_indices[1][i]
                depth_val = depth_image[y, x] / 1000.0  # 转换为米
                
                if depth_val > 0.01:
                    # 转换为相机坐标
                    cam_point = self.coord_calc.pixel_to_camera_coordinates(x, y, depth_val)
                    if cam_point is not None:
                        # 转换为世界坐标
                        tcp_pose = [
                            waypoint_data['world_pos'][0],
                            waypoint_data['world_pos'][1],
                            waypoint_data['world_pos'][2],
                            waypoint_data['roll'],
                            waypoint_data['pitch'],
                            waypoint_data['yaw']
                        ]
                        world_point = self.coord_calc.camera_to_world_coordinates(cam_point, tcp_pose)
                        if world_point is not None:
                            valid_points.append(world_point)
            
            if len(valid_points) > 0:
                centroid = np.mean(valid_points, axis=0)
                return centroid.tolist()
            else:
                return [0, 0, 0]
                
        except Exception as e:
            print(f"[POST_PROCESSOR] 3D centroid calculation failed: {e}")
            return [0, 0, 0]
    
    def _calculate_average_depth(self, mask: np.ndarray, depth_image: np.ndarray) -> float:
        """计算平均深度"""
        try:
            if mask is None:
                return 0.5
            
            masked_depth = depth_image[mask > 0] / 1000.0
            valid_depths = masked_depth[masked_depth > 0.01]
            
            if len(valid_depths) > 0:
                return float(np.median(valid_depths))
            else:
                return 0.5
                
        except Exception:
            return 0.5
    
    def _estimate_object_height(self, mask: np.ndarray, depth_image: np.ndarray, 
                               bbox: List[int]) -> float:
        """估计物体高度"""
        try:
            if mask is None:
                return 30.0
            
            x1, y1, x2, y2 = map(int, bbox)
            
            # 在bbox区域内分别计算物体和背景深度
            object_depths = []
            background_depths = []
            
            for y in range(y1, y2, 3):
                for x in range(x1, x2, 3):
                    if 0 <= y < depth_image.shape[0] and 0 <= x < depth_image.shape[1]:
                        depth_val = depth_image[y, x] / 1000.0
                        if depth_val > 0.01:
                            if mask[y, x] > 0:
                                object_depths.append(depth_val)
                            else:
                                background_depths.append(depth_val)
            
            if len(object_depths) > 5 and len(background_depths) > 5:
                obj_median = np.median(object_depths)
                bg_median = np.median(background_depths)
                height_mm = abs(obj_median - bg_median) * 1000
                return max(5.0, min(500.0, height_mm))  # 限制在合理范围
            else:
                return 30.0
                
        except Exception:
            return 30.0
    
    def _calculate_depth_variance(self, mask: np.ndarray, depth_image: np.ndarray) -> float:
        """计算深度变化"""
        try:
            if mask is None:
                return 0.0
            
            masked_depth = depth_image[mask > 0] / 1000.0
            valid_depths = masked_depth[masked_depth > 0.01]
            
            if len(valid_depths) > 10:
                return float(np.std(valid_depths))
            else:
                return 0.0
                
        except Exception:
            return 0.0
    
    def _compute_similarity_matrix(self, detection_features: List[Dict]) -> np.ndarray:
        """计算相似性矩阵"""
        try:
            n = len(detection_features)
            similarity_matrix = np.zeros((n, n))
            
            for i in range(n):
                for j in range(i + 1, n):
                    similarity = self._calculate_detection_similarity(
                        detection_features[i], detection_features[j]
                    )
                    similarity_matrix[i, j] = similarity
                    similarity_matrix[j, i] = similarity
            
            return similarity_matrix
            
        except Exception as e:
            print(f"[POST_PROCESSOR] Similarity matrix computation failed: {e}")
            return np.zeros((len(detection_features), len(detection_features)))
    
    def _calculate_detection_similarity(self, feat1: Dict, feat2: Dict) -> float:
        """计算相似性并记录详细分析数据"""
        # 初始化相似性分析记录
        if not hasattr(self, 'similarity_analyses'):
            self.similarity_analyses = []
        
        similarity_breakdown = {
            'detection_pair': f"{feat1.get('class_name', 'unknown')}_{feat2.get('class_name', 'unknown')}",
            'individual_similarities': {},
            'decision_factors': {},
            'physical_evidence': {}
        }
        
        similarity_score = 0.0
        
        # 1. 3D空间距离分析
        pos1 = np.array(feat1['world_centroid'])
        pos2 = np.array(feat2['world_centroid'])
        spatial_distance = 0.0
        
        if np.any(pos1) and np.any(pos2):
            spatial_distance = np.linalg.norm(pos1 - pos2)
            spatial_similarity = max(0, 1 - spatial_distance / self.spatial_distance_threshold)
            similarity_score += spatial_similarity * 0.4
            
            similarity_breakdown['individual_similarities']['spatial'] = {
                'distance_mm': float(spatial_distance),
                'similarity_score': float(spatial_similarity),
                'weight': 0.4,
                'threshold_mm': self.spatial_distance_threshold
            }
            
            # 物理证据：如果距离小于物体尺寸，很可能是同一物体
            object_size_estimate = max(feat1.get('estimated_height', 30), feat2.get('estimated_height', 30))
            similarity_breakdown['physical_evidence']['likely_same_object_by_distance'] = spatial_distance < object_size_estimate
        
        # 2. 深度相似性分析
        depth_diff = abs(feat1['average_depth'] - feat2['average_depth']) * 1000
        depth_similarity = max(0, 1 - depth_diff / self.depth_similarity_threshold)
        similarity_score += depth_similarity * 0.25
        
        similarity_breakdown['individual_similarities']['depth'] = {
            'difference_mm': float(depth_diff),
            'similarity_score': float(depth_similarity),
            'weight': 0.25,
            'threshold_mm': self.depth_similarity_threshold
        }
        
        # 3. 高度相似性分析
        height_diff = abs(feat1['estimated_height'] - feat2['estimated_height'])
        height_similarity = max(0, 1 - height_diff / self.height_similarity_threshold)
        similarity_score += height_similarity * 0.2
        
        similarity_breakdown['individual_similarities']['height'] = {
            'difference_mm': float(height_diff),
            'similarity_score': float(height_similarity), 
            'weight': 0.2,
            'threshold_mm': self.height_similarity_threshold
        }
        
        # 4. Mask重叠分析
        mask_overlap = self._calculate_mask_overlap(feat1, feat2)
        similarity_score += mask_overlap * 0.15
        
        similarity_breakdown['individual_similarities']['mask_overlap'] = {
            'overlap_ratio': float(mask_overlap),
            'similarity_score': float(mask_overlap),
            'weight': 0.15,
            'threshold': self.mask_overlap_threshold
        }
        
        # 综合决策分析
        similarity_breakdown['decision_factors'] = {
            'combined_similarity': float(similarity_score),
            'passes_threshold': similarity_score > 0.6,
            'same_class': feat1.get('class_name') == feat2.get('class_name'),
            'confidence_diff': abs(feat1.get('confidence', 0) - feat2.get('confidence', 0))
        }
        
        # 物理合理性判断
        very_close_position = spatial_distance < 20 if np.any(pos1) and np.any(pos2) else False
        very_similar_depth = depth_diff < 10
        very_similar_height = height_diff < 10
        
        similarity_breakdown['physical_evidence'].update({
            'identical_physical_properties': very_close_position and very_similar_depth and very_similar_height,
            'likely_yolo_duplicate': similarity_score > 0.8 and feat1.get('class_name') == feat2.get('class_name'),
            'depth_consistency_match': (feat1.get('depth_uniformity') == 'highly_uniform' and 
                                    feat2.get('depth_uniformity') == 'highly_uniform')
        })
        
        # 只记录可能合并的相似性分析
        if similarity_score > 0.3:  # 降低阈值以记录更多数据
            self.similarity_analyses.append(similarity_breakdown)
        
        return similarity_score
    
    def _calculate_mask_overlap(self, feat1: Dict, feat2: Dict) -> float:
        """计算mask重叠度"""
        try:
            detection1 = feat1['detection']
            detection2 = feat2['detection']
            
            mask1 = detection1.get('mask')
            mask2 = detection2.get('mask')
            
            if mask1 is None or mask2 is None:
                return 0.0
            
            # 计算重叠区域
            overlap = np.logical_and(mask1 > 0, mask2 > 0)
            union = np.logical_or(mask1 > 0, mask2 > 0)
            
            overlap_area = np.sum(overlap)
            union_area = np.sum(union)
            
            if union_area > 0:
                iou = overlap_area / union_area
                return float(iou)
            else:
                return 0.0
                
        except Exception:
            return 0.0
    
    def _identify_duplicate_groups(self, similarity_matrix: np.ndarray, 
                                  detection_features: List[Dict]) -> List[List[int]]:
        """识别重复检测组"""
        try:
            n = len(detection_features)
            duplicate_groups = []
            processed = set()
            
            # 基于相似性矩阵进行聚类
            for i in range(n):
                if i in processed:
                    continue
                
                # 找到与当前检测相似的其他检测
                current_group = [i]
                processed.add(i)
                
                for j in range(i + 1, n):
                    if j in processed:
                        continue
                    
                    # 检查是否满足重复条件
                    if self._is_duplicate_pair(similarity_matrix[i, j], 
                                             detection_features[i], detection_features[j]):
                        current_group.append(j)
                        processed.add(j)
                
                # 如果组内有多个检测，认为是重复组
                if len(current_group) > 1:
                    duplicate_groups.append(current_group)
                    print(f"[POST_PROCESSOR] Found duplicate group: {current_group}")
                    self._log_duplicate_group(current_group, detection_features)
                else:
                    # 单独的检测，创建单个检测的组
                    duplicate_groups.append(current_group)
            
            return duplicate_groups
            
        except Exception as e:
            print(f"[POST_PROCESSOR] Duplicate group identification failed: {e}")
            # 返回每个检测单独一组
            return [[i] for i in range(len(detection_features))]
    
    def _is_duplicate_pair(self, similarity_score: float, feat1: Dict, feat2: Dict) -> bool:
        """判断两个检测是否为重复"""
        try:
            # 基本相似性阈值
            if similarity_score < 0.6:
                return False
            
            # 额外的重复判断条件
            
            # 1. 如果是相同类别且空间距离很近
            if (feat1['class_name'] == feat2['class_name'] and 
                similarity_score > 0.8):
                return True
            
            # 2. 如果mask重叠度很高且空间距离近
            mask_overlap = self._calculate_mask_overlap(feat1, feat2)
            if mask_overlap > self.mask_overlap_threshold and similarity_score > 0.7:
                return True
            
            # 3. 如果深度、高度都很相似且有一定的mask重叠
            depth_diff = abs(feat1['average_depth'] - feat2['average_depth']) * 1000
            height_diff = abs(feat1['estimated_height'] - feat2['estimated_height'])
            
            if (depth_diff < self.depth_similarity_threshold and 
                height_diff < self.height_similarity_threshold and 
                mask_overlap > 0.2):
                return True
            
            return False
            
        except Exception:
            return False
    
    def _merge_duplicate_detections(self, duplicate_groups: List[List[int]], 
                                   detection_features: List[Dict]) -> List[Dict]:
        """合并重复检测"""
        try:
            merged_detections = []
            
            for group in duplicate_groups:
                if len(group) == 1:
                    # 单个检测，直接添加
                    merged_detections.append(detection_features[group[0]]['detection'])
                else:
                    # 多个重复检测，进行合并
                    merged_detection = self._merge_detection_group(group, detection_features)
                    merged_detections.append(merged_detection)
                    print(f"[POST_PROCESSOR] Merged {len(group)} duplicate detections")
            
            return merged_detections
            
        except Exception as e:
            print(f"[POST_PROCESSOR] Merging duplicate detections failed: {e}")
            return [feat['detection'] for feat in detection_features]
    
    def _merge_detection_group(self, group_indices: List[int], 
                              detection_features: List[Dict]) -> Dict:
        """合并一组重复检测"""
        try:
            group_detections = [detection_features[i]['detection'] for i in group_indices]
            group_features = [detection_features[i] for i in group_indices]
            
            # 选择置信度最高的检测作为基础
            best_idx = max(range(len(group_detections)), 
                          key=lambda i: group_detections[i]['confidence'])
            base_detection = group_detections[best_idx].copy()
            
            # 更新置信度（使用加权平均）
            confidences = [det['confidence'] for det in group_detections]
            weights = np.array(confidences) / sum(confidences)
            merged_confidence = np.average(confidences, weights=weights)
            base_detection['confidence'] = float(merged_confidence)
            
            # 合并边界框（使用所有检测的并集）
            all_bboxes = [det['bounding_box'] for det in group_detections]
            merged_bbox = self._merge_bounding_boxes(all_bboxes)
            base_detection['bounding_box'] = merged_bbox
            
            # 合并mask（使用最大的mask）
            all_masks = [det.get('mask') for det in group_detections if det.get('mask') is not None]
            if all_masks:
                merged_mask = self._merge_masks(all_masks)
                base_detection['mask'] = merged_mask
            
            # 添加合并信息
            base_detection['merged_from'] = len(group_detections)
            base_detection['original_confidences'] = confidences
            
            print(f"[POST_PROCESSOR] Merged detection: {base_detection['class_name']} "
                  f"conf:{merged_confidence:.3f} from {len(group_detections)} detections")
            
            merge_detail = {
                'kept_detection_idx': group_indices[best_idx],
                'merged_detection_indices': [idx for i, idx in enumerate(group_indices) if i != best_idx],
                'spatial_distances': [np.linalg.norm(np.array(group_features[i]['world_centroid']) - 
                                                np.array(group_features[best_idx]['world_centroid'])) 
                                    for i in range(len(group_features)) if i != best_idx],
                'depth_similarities': [abs(group_features[i]['average_depth'] - 
                                        group_features[best_idx]['average_depth']) * 1000
                                    for i in range(len(group_features)) if i != best_idx],
                'mask_overlaps': [self._calculate_mask_overlap(group_features[best_idx], group_features[i])
                                for i in range(len(group_features)) if i != best_idx]
            }
            
            # 添加到处理统计中
            if hasattr(self, 'last_processing_stats'):
                self.last_processing_stats['merge_operations'].append(merge_detail)
            
            return base_detection
            
        except Exception as e:
            print(f"[POST_PROCESSOR] Group merging failed: {e}")
            return detection_features[group_indices[0]]['detection']
    
    def _merge_bounding_boxes(self, bboxes: List[List[int]]) -> List[int]:
        """合并多个边界框"""
        try:
            if not bboxes:
                return [0, 0, 100, 100]
            
            x1_min = min(bbox[0] for bbox in bboxes)
            y1_min = min(bbox[1] for bbox in bboxes)
            x2_max = max(bbox[2] for bbox in bboxes)
            y2_max = max(bbox[3] for bbox in bboxes)
            
            return [x1_min, y1_min, x2_max, y2_max]
            
        except Exception:
            return bboxes[0] if bboxes else [0, 0, 100, 100]
    
    def _merge_masks(self, masks: List[np.ndarray]) -> np.ndarray:
        """合并多个mask"""
        try:
            if not masks:
                return None
            
            # 使用逻辑或合并所有mask
            merged_mask = masks[0].copy()
            for mask in masks[1:]:
                if mask.shape == merged_mask.shape:
                    merged_mask = np.logical_or(merged_mask > 0, mask > 0)
            
            return merged_mask.astype(np.uint8)
            
        except Exception:
            return masks[0] if masks else None
    
    def _log_duplicate_group(self, group_indices: List[int], detection_features: List[Dict]):
        """记录重复组信息"""
        try:
            print(f"[POST_PROCESSOR] Duplicate group details:")
            for i, idx in enumerate(group_indices):
                feat = detection_features[idx]
                print(f"  {i+1}. {feat['class_name']} conf:{feat['confidence']:.3f} "
                      f"pos:{feat['world_centroid']} depth:{feat['average_depth']:.3f}m "
                      f"height:{feat['estimated_height']:.1f}mm")
        except Exception:
            pass
    
    def _log_filtering_results(self, original_detections: List[Dict], 
                              filtered_detections: List[Dict]):
        """记录过滤结果"""
        try:
            print(f"[POST_PROCESSOR] Filtering summary:")
            print(f"  Original detections: {len(original_detections)}")
            print(f"  Filtered detections: {len(filtered_detections)}")
            print(f"  Removed duplicates: {len(original_detections) - len(filtered_detections)}")
            
            # 显示保留的检测
            for i, det in enumerate(filtered_detections):
                merged_info = f" (merged from {det.get('merged_from', 1)})" if det.get('merged_from', 1) > 1 else ""
                print(f"  Final {i+1}: {det['class_name']} conf:{det['confidence']:.3f}{merged_info}")
                
        except Exception:
            pass