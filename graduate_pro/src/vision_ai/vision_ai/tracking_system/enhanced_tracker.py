# tracking/enhanced_tracker.py
import os
import json
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import time
import cv2
# 导入增强组件
from ..detection.features.similarity_calculator import FeatureSimilarityCalculator
from ..detection.utils.coordinate_calculator import CoordinateCalculator, ObjectAnalyzer
from ..detection.utils.adaptive_learner import AdaptiveThresholdManager
from ..detection.utils.enhanced_config_manager import EnhancedConfigManager
from .adaptive_learning.online_learner import OnlineLearner
class EnhancedTracker:
    """增强Track模块 - 独立的特征匹配追踪器"""
    
    def __init__(self, config_path: str):
        """初始化增强追踪器 - 修复版"""
        try:
            # 加载配置
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
            
            tracking_config = self.config.get('tracking', {})
            
            # 🔧 确保设置相似度阈值
            self.similarity_threshold = tracking_config.get('similarity_threshold', 0.6)  # 降低默认阈值
            self.confidence_threshold = tracking_config.get('confidence_threshold', 0.6)
            self.max_tracking_distance = tracking_config.get('max_tracking_distance', 600.0)
            
            print(f"[ENHANCED_TRACKER] 配置加载完成:")
            print(f"  - 相似度阈值: {self.similarity_threshold}")
            print(f"  - 置信度阈值: {self.confidence_threshold}")
            print(f"  - 最大追踪距离: {self.max_tracking_distance}mm")
            
            # 初始化组件
            self.similarity_calculator = FeatureSimilarityCalculator()
            self.coordinate_calculator = None  # 将在需要时初始化
            learning_file = "/home/qi/ros2_ws/src/vision_ai/vision_ai/detection/data/adaptive_learning.json"
            
            self.adaptive_manager = AdaptiveThresholdManager(learning_file)
            self.similarity_calculator = FeatureSimilarityCalculator()
            self.online_learner = OnlineLearner(
                user_id='default_user',
                similarity_calculator=self.similarity_calculator
            )
            # 学习状态跟踪
            self.learning_context = {
                'distance_history': [],
                'confidence_history': [],
                'lighting_conditions': []
            }            
            # 数据存储
            self.reference_library = {}
            self.current_target_id = None
            self.current_target_class = None
            self.config_manager = EnhancedConfigManager(config_path)
            # 🆕 添加ID映射缓存
            self.id_to_class_mapping = {}
            self.class_to_ids_mapping = {}
            self.successful_detection_history = {}  # 成功检测历史
            self.confidence_based_success = {}      # 基于置信度的自动成功判断
            self.auto_success_enabled = False      # 是否启用自动成功判断
            self.auto_success_threshold = 0.80     # 自动成功判断阈值
            
            # 🆕 历史管理配置
            self.max_history_length = 5           # 最大历史记录数
            self.min_confidence_for_history = 0.7  # 加入历史的最小置信度
        
            print("[HYBRID_TRACKER] 混合相似度追踪器初始化完成")
            print("[ENHANCED_TRACKER] 增强追踪器初始化完成")
            self._initialize_components()

        except Exception as e:
            print(f"[ENHANCED_TRACKER] 初始化失败: {e}")
            # 设置默认值
            self.similarity_threshold = 0.3
            self.confidence_threshold = 0.5
            self.max_tracking_distance = 500.0
            self.similarity_calculator = FeatureSimilarityCalculator()
            self.reference_library = {}
            self.current_target_id = None
            self.current_target_class = None
            self.id_to_class_mapping = {}
            self.class_to_ids_mapping = {}
    
    def _initialize_components(self):
        """初始化核心组件"""
        try:
            # 相似度计算器
            similarity_config = self.config_manager.get_similarity_config()
            feature_weights = similarity_config.get('feature_weights', {})
            self.similarity_calculator = FeatureSimilarityCalculator(feature_weights)
            
            # 坐标计算器
            camera_config = self.config_manager.get_camera_config()
            calibration_data = {
                'camera_intrinsics': camera_config.get('intrinsics', {}),
                'hand_eye_translation': camera_config.get('hand_eye_calibration', {}).get('translation'),
                'hand_eye_quaternion': camera_config.get('hand_eye_calibration', {}).get('quaternion')
            }
            self.coordinate_calculator = CoordinateCalculator(calibration_data)
            self.object_analyzer = ObjectAnalyzer(self.coordinate_calculator)
            
            # 自适应学习管理器
            adaptive_config = self.config_manager.get_adaptive_learning_config()
            if adaptive_config.get('enabled', True):
                learning_file = adaptive_config.get('learning_data_file', 'data/tracking_adaptive_learning.json')
                self.adaptive_manager = AdaptiveThresholdManager(learning_file)
                print("[ENHANCED_TRACKER] 自适应学习已启用")
            else:
                self.adaptive_manager = None
                
            print("[ENHANCED_TRACKER] 核心组件初始化成功")
            
        except Exception as e:
            print(f"[ENHANCED_TRACKER] 组件初始化失败: {e}")
            raise
    
    def load_reference_features(self, library_file: str) -> bool:
        """加载参考特征库 - 优化版，构建ID映射"""
        try:
            with open(library_file, 'r', encoding='utf-8') as f:
                library_data = json.load(f)
            
            self.reference_library = {}
            
            # 🆕 构建ID映射
            for obj_id, entry in library_data.items():
                self.reference_library[obj_id] = entry
                
                # 提取类别信息
                class_name = obj_id.split('_')[0]
                self.id_to_class_mapping[obj_id] = class_name
                
                # 构建类别到ID的映射
                if class_name not in self.class_to_ids_mapping:
                    self.class_to_ids_mapping[class_name] = []
                self.class_to_ids_mapping[class_name].append(obj_id)
            
            print(f"[ENHANCED_TRACKER] 参考特征库加载完成")
            print(f"[ENHANCED_TRACKER] 总对象数: {len(self.reference_library)}")
            print(f"[ENHANCED_TRACKER] 类别统计:")
            for class_name, ids in self.class_to_ids_mapping.items():
                print(f"  - {class_name}: {len(ids)} 个对象 {ids}")
            
            return True
            
        except Exception as e:
            print(f"[ENHANCED_TRACKER] 加载参考特征库失败: {e}")
            return False
    
    def _display_reference_library_summary(self):
        """显示参考特征库摘要"""
        try:
            print(f"\n{'='*50}")
            print("参考特征库摘要")
            print(f"{'='*50}")
            
            # 按类别统计
            by_class = {}
            quality_scores = []
            
            for obj_id, entry in self.reference_library.items():
                if isinstance(entry, dict) and 'metadata' in entry:
                    class_name = entry['metadata'].get('class_name', 'unknown')
                    quality_score = entry.get('quality_score', 0)
                    
                    if class_name not in by_class:
                        by_class[class_name] = []
                    by_class[class_name].append((obj_id, quality_score))
                    quality_scores.append(quality_score)
            
            print(f"总计: {len(self.reference_library)} 个参考特征")
            
            if quality_scores:
                print(f"平均特征质量: {np.mean(quality_scores):.1f}%")
                print(f"质量范围: {np.min(quality_scores):.1f}% - {np.max(quality_scores):.1f}%")
            
            print("\n按类别分布:")
            for class_name, objects in by_class.items():
                avg_quality = np.mean([q for _, q in objects])
                print(f"  {class_name}: {len(objects)} 个对象 (平均质量: {avg_quality:.1f}%)")
            
            print(f"{'='*50}\n")
            
        except Exception as e:
            print(f"[ENHANCED_TRACKER] 显示摘要失败: {e}")
    
    def select_tracking_target(self, target_id: str) -> bool:
        """选择追踪目标 - 添加验证"""
        try:
            if target_id not in self.reference_library:
                print(f"[ENHANCED_TRACKER] 错误: 目标 {target_id} 不存在于参考库中")
                print(f"[ENHANCED_TRACKER] 可用目标: {list(self.reference_library.keys())}")
                return False
            
            self.current_target_id = target_id
            self.current_target_class = self.id_to_class_mapping[target_id]
            
            print(f"[ENHANCED_TRACKER] 已选择追踪目标: {target_id}")
            print(f"[ENHANCED_TRACKER] 目标类别: {self.current_target_class}")
            print(f"[ENHANCED_TRACKER] 同类别其他对象: {[id for id in self.class_to_ids_mapping[self.current_target_class] if id != target_id]}")
            
            return True
            
        except Exception as e:
            print(f"[ENHANCED_TRACKER] 选择追踪目标失败: {e}")
            return False
    
    def track_target(self, target_id: str, image_rgb: np.ndarray, 
                    depth_image: np.ndarray, waypoint_data: Dict,
                    candidate_detections: List[Dict]) -> Optional[Dict]:
        """追踪目标 - 混合相似度版本，支持自动化过渡"""
        try:
            print(f"[HYBRID_TRACKER] 开始混合相似度追踪: {target_id}")
            
            # Layer 1: 实时自适应权重调整
            current_context = self._build_tracking_context(waypoint_data, candidate_detections)
            adaptive_weights = self._get_realtime_adaptive_weights(current_context)
            
            original_weights = self.similarity_calculator.feature_weights.copy()
            current_weights_used = original_weights.copy()
            
            if adaptive_weights:
                print(f"[LAYER1] 应用实时自适应权重: {adaptive_weights}")
                self.similarity_calculator.update_weights(adaptive_weights)
                current_weights_used = adaptive_weights

            # 获取参考特征和历史特征
            if target_id not in self.reference_library:
                print(f"[HYBRID_TRACKER] 目标不在参考库中: {target_id}")
                return None
                
            reference_features = self.reference_library[target_id]['features']
            historical_features = self._get_successful_detection_history(target_id)
            
            print(f"[HYBRID_TRACKER] 参考库特征已加载，历史特征数量: {len(historical_features)}")
            
            try:
                # 过滤同类别候选
                target_class = target_id.split('_')[0]
                same_class_candidates = [(i, detection) for i, detection in enumerate(candidate_detections)
                                       if detection['class_name'] == target_class]
                
                if not same_class_candidates:
                    print(f"[HYBRID_TRACKER] 无同类别候选")
                    return None
                
                # 🆕 混合相似度计算
                all_candidates_analysis = []
                best_match = None
                best_detailed_similarity = None
                best_similarity = 0.0
                
                for original_index, detection in same_class_candidates:
                    try:
                        # 计算混合相似度
                        hybrid_similarity = self._calculate_hybrid_similarity(
                            reference_features,
                            historical_features,
                            detection,
                            waypoint_data
                        )
                        
                        similarity_score = hybrid_similarity['final_score']
                        
                        # 构建候选分析数据
                        candidate_analysis = {
                            'candidate_index': original_index,
                            'detection_data': {
                                'bounding_box': detection.get('bounding_box', []),
                                'confidence': detection.get('confidence', 0.0),
                                'class_name': detection.get('class_name', ''),
                                'mask_area': self._calculate_mask_area(detection.get('mask'))
                            },
                            'features': {
                                'geometric': detection.get('geometric_features', {}),
                                'appearance': detection.get('appearance_features', {}),
                                'shape': detection.get('shape_features', {}),
                                'spatial': detection.get('spatial_features', {})
                            },
                            'similarity_to_target': {
                                'hybrid_breakdown': hybrid_similarity,
                                'final_score': similarity_score,
                                'reference_score': hybrid_similarity.get('reference_score', 0.0),
                                'historical_score': hybrid_similarity.get('historical_score', 0.0),
                                'history_count': hybrid_similarity.get('history_count', 0)
                            },
                            'is_best_match': False,
                            'meets_threshold': similarity_score >= self.similarity_threshold
                        }
                        
                        all_candidates_analysis.append(candidate_analysis)
                        
                        print(f"[HYBRID_TRACKER] 候选 {original_index+1} 混合相似度: {similarity_score:.3f} "
                              f"(参考:{hybrid_similarity.get('reference_score', 0):.3f}, "
                              f"历史:{hybrid_similarity.get('historical_score', 0):.3f})")
                        
                        if similarity_score > best_similarity:
                            best_similarity = similarity_score
                            best_match = detection
                            best_detailed_similarity = hybrid_similarity
                            
                            # 更新最佳匹配标记
                            for analysis in all_candidates_analysis:
                                analysis['is_best_match'] = False
                            candidate_analysis['is_best_match'] = True
                                
                    except Exception as e:
                        print(f"[HYBRID_TRACKER] 计算候选 {original_index} 混合相似度失败: {e}")
                        continue
                
                print(f"[HYBRID_TRACKER] 最佳混合相似度: {best_similarity:.3f} (阈值: {self.similarity_threshold:.3f})")
                
                if best_match is not None and best_similarity >= self.similarity_threshold:
                    print(f"[HYBRID_TRACKER] 匹配成功！")
                    
                    # 🆕 自动成功判断（如果启用）
                    auto_success = self._auto_judge_success(target_id, best_detailed_similarity, best_similarity)
                    
                    tracking_result = self._build_tracking_result_with_detailed_features(
                        target_id, best_match, best_detailed_similarity, waypoint_data, image_rgb, depth_image
                    )
                    
                    # 添加混合相似度特有信息
                    tracking_result['all_candidates_analysis'] = all_candidates_analysis
                    tracking_result['adaptive_weights_used'] = current_weights_used
                    tracking_result['total_candidates_analyzed'] = len(all_candidates_analysis)
                    tracking_result['hybrid_similarity_info'] = {
                        'reference_score': best_detailed_similarity.get('reference_score', 0.0),
                        'historical_score': best_detailed_similarity.get('historical_score', 0.0),
                        'history_count': best_detailed_similarity.get('history_count', 0),
                        'fusion_weights': {
                            'reference_weight': best_detailed_similarity.get('reference_weight', 1.0),
                            'historical_weight': best_detailed_similarity.get('historical_weight', 0.0)
                        }
                    }
                    
                    # 🆕 如果启用自动判断且判断为成功，自动记录为成功
                    if auto_success:
                        print(f"[AUTO_SUCCESS] 自动判断为成功，置信度: {best_similarity:.3f}")
                        tracking_result['auto_success'] = True
                        self._record_successful_detection(target_id, best_match, best_detailed_similarity)
                    else:
                        tracking_result['auto_success'] = False
                    
                    return tracking_result
                else:
                    print(f"[HYBRID_TRACKER] 相似度低于阈值，匹配失败")
                    return None
                    
            finally:
                # 恢复原始权重
                self.similarity_calculator.update_weights(original_weights)
                
        except Exception as e:
            print(f"[HYBRID_TRACKER] 追踪过程失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _calculate_mask_area(self, mask):
        """计算mask区域面积"""
        try:
            if mask is not None and hasattr(mask, 'sum'):
                return int(mask.sum())
            return 0
        except:
            return 0
    def _get_successful_detection_history(self, target_id: str, max_history: int = None) -> List[Dict]:
        """获取成功检测历史特征"""
        try:
            if max_history is None:
                max_history = self.max_history_length
                
            history = self.successful_detection_history.get(target_id, [])
            
            # 返回最近N次成功检测的特征
            recent_history = history[-max_history:] if len(history) > max_history else history
            
            print(f"[HISTORY] {target_id} 历史特征数量: {len(recent_history)} / {len(history)}")
            return recent_history
            
        except Exception as e:
            print(f"[HISTORY] 获取历史特征失败: {e}")
            return []
    def _get_successful_detection_history(self, target_id: str, max_history: int = None) -> List[Dict]:
        """获取成功检测历史特征"""
        try:
            if max_history is None:
                max_history = self.max_history_length
                
            history = self.successful_detection_history.get(target_id, [])
            
            # 返回最近N次成功检测的特征
            recent_history = history[-max_history:] if len(history) > max_history else history
            
            print(f"[HISTORY] {target_id} 历史特征数量: {len(recent_history)} / {len(history)}")
            return recent_history
            
        except Exception as e:
            print(f"[HISTORY] 获取历史特征失败: {e}")
            return []

    def _calculate_hybrid_similarity(self, reference_features: Dict, historical_features: List[Dict],
                                   candidate: Dict, waypoint_data: Dict) -> Dict:
        """计算混合相似度"""
        try:
            # 提取候选特征
            candidate_features_dict = {
                'geometric': candidate.get('geometric_features', {}),
                'appearance': candidate.get('appearance_features', {}),
                'shape': candidate.get('shape_features', {}),
                'spatial': candidate.get('spatial_features', {})
            }
            
            # 1. 计算与参考库的相似度
            reference_similarity = self.similarity_calculator.calculate_detailed_similarity_breakdown(
                reference_features, 
                candidate_features_dict, 
                waypoint_data
            )
            reference_score = reference_similarity['final_score']
            
            # 2. 计算与历史特征的相似度
            historical_score = 0.0
            historical_details = []
            
            if historical_features:
                historical_scores = []
                for i, hist_features in enumerate(historical_features):
                    try:
                        hist_similarity = self.similarity_calculator.calculate_detailed_similarity_breakdown(
                            hist_features,
                            candidate_features_dict,
                            waypoint_data
                        )
                        score = hist_similarity['final_score']
                        historical_scores.append(score)
                        historical_details.append({
                            'index': i,
                            'score': score,
                            'timestamp': hist_features.get('timestamp', 'unknown')
                        })
                        
                    except Exception as e:
                        print(f"[HYBRID_SIMILARITY] 计算历史 {i} 相似度失败: {e}")
                        continue
                
                if historical_scores:
                    # 🆕 使用加权平均 - 更近的历史权重更高
                    weights = self._calculate_temporal_weights(len(historical_scores))
                    historical_score = np.average(historical_scores, weights=weights)
                    
                    print(f"[HYBRID_SIMILARITY] 历史相似度详情: {[f'{s:.3f}' for s in historical_scores]} -> {historical_score:.3f}")
            
            # 3. 动态权重计算
            history_count = len(historical_features)
            reference_weight, historical_weight = self._calculate_fusion_weights(
                history_count, reference_score, historical_score
            )
            
            # 4. 计算最终混合分数
            final_score = reference_weight * reference_score + historical_weight * historical_score
            
            print(f"[HYBRID_SIMILARITY] 参考库={reference_score:.3f}({reference_weight:.2f}), "
                  f"历史={historical_score:.3f}({historical_weight:.2f}), "
                  f"混合={final_score:.3f}")
            
            return {
                'final_score': final_score,
                'reference_score': reference_score,
                'historical_score': historical_score,
                'reference_weight': reference_weight,
                'historical_weight': historical_weight,
                'history_count': history_count,
                'historical_details': historical_details,
                'detailed_breakdown': reference_similarity.get('detailed_breakdown', {}),
                'feature_contributions': reference_similarity.get('feature_contributions', {}),
                'feature_weights_used': reference_similarity.get('feature_weights_used', {})
            }
            
        except Exception as e:
            print(f"[HYBRID_SIMILARITY] 计算混合相似度失败: {e}")
            return {
                'final_score': reference_score if 'reference_score' in locals() else 0.0,
                'reference_score': reference_score if 'reference_score' in locals() else 0.0,
                'historical_score': 0.0,
                'reference_weight': 1.0,
                'historical_weight': 0.0,
                'history_count': 0,
                'detailed_breakdown': {},
                'feature_contributions': {},
                'feature_weights_used': {}
            }

    def _calculate_temporal_weights(self, count: int) -> np.ndarray:
        """计算时序权重 - 越近的历史权重越高"""
        if count <= 0:
            return np.array([])
        
        # 线性递增权重：最新的权重最高
        weights = np.arange(1, count + 1, dtype=float)
        return weights / weights.sum()

    def _calculate_fusion_weights(self, history_count: int, reference_score: float, 
                                historical_score: float) -> Tuple[float, float]:
        """动态计算融合权重"""
        try:
            if history_count == 0:
                return 1.0, 0.0
            
            # 基础权重策略
            if history_count <= 1:
                base_ref_weight = 0.85
                base_hist_weight = 0.15
            elif history_count <= 3:
                base_ref_weight = 0.70
                base_hist_weight = 0.30
            else:
                base_ref_weight = 0.60
                base_hist_weight = 0.40
            
            # 🆕 基于相对性能的动态调整
            if historical_score > reference_score + 0.1:
                # 历史表现明显更好，增加历史权重
                adjustment = min(0.1, (historical_score - reference_score) * 0.5)
                base_hist_weight += adjustment
                base_ref_weight -= adjustment
            elif reference_score > historical_score + 0.1:
                # 参考库表现更好，增加参考权重
                adjustment = min(0.1, (reference_score - historical_score) * 0.5)
                base_ref_weight += adjustment
                base_hist_weight -= adjustment
            
            # 确保权重和为1
            total = base_ref_weight + base_hist_weight
            return base_ref_weight / total, base_hist_weight / total
            
        except Exception as e:
            print(f"[FUSION_WEIGHTS] 计算融合权重失败: {e}")
            return 0.7, 0.3

    def _auto_judge_success(self, target_id: str, similarity_result: Dict, 
                           final_confidence: float) -> bool:
        """自动判断追踪是否成功"""
        try:
            if not self.auto_success_enabled:
                return False
            
            # 条件1: 置信度足够高
            if final_confidence < self.auto_success_threshold:
                return False
            
            # 条件2: 混合相似度各组成部分都合理
            reference_score = similarity_result.get('reference_score', 0.0)
            historical_score = similarity_result.get('historical_score', 0.0)
            history_count = similarity_result.get('history_count', 0)
            
            # 参考库相似度必须超过基础阈值
            if reference_score < 0.6:
                return False
            
            # 如果有历史特征，历史相似度也应该合理
            if history_count > 0 and historical_score < 0.5:
                return False
            
            # 条件3: 检查一致性（各特征类型相似度不能差距过大）
            feature_contributions = similarity_result.get('feature_contributions', {})
            if feature_contributions:
                scores = [score for score in feature_contributions.values() if isinstance(score, (int, float))]
                if scores:
                    score_std = np.std(scores)
                    if score_std > 0.3:  # 各特征相似度标准差过大
                        print(f"[AUTO_JUDGE] 特征相似度不一致，标准差: {score_std:.3f}")
                        return False
            
            print(f"[AUTO_JUDGE] 自动判断成功: 置信度={final_confidence:.3f}, "
                  f"参考={reference_score:.3f}, 历史={historical_score:.3f}")
            return True
            
        except Exception as e:
            print(f"[AUTO_JUDGE] 自动判断失败: {e}")
            return False

    def record_successful_detection_manual(self, target_id: str, detection: Dict, 
                                        similarity_result: Dict, user_confirmed: bool = True):
        """手动记录成功检测（用于人工标注阶段）"""
        try:
            if user_confirmed:
                # 🔧 确保similarity_result包含正确的置信度
                if not similarity_result or similarity_result.get('final_score', 0.0) == 0.0:
                    # 从其他地方获取置信度
                    tracking_confidence = detection.get('tracking_confidence', 0.0)
                    if tracking_confidence > 0:
                        similarity_result = similarity_result.copy() if similarity_result else {}
                        similarity_result['final_score'] = tracking_confidence
                        print(f"[MANUAL_RECORD] 修正置信度: {tracking_confidence:.3f}")
                
                self._record_successful_detection(target_id, detection, similarity_result)
                print(f"[MANUAL_RECORD] 用户确认成功，已记录到历史: {target_id}")
                
        except Exception as e:
            print(f"[MANUAL_RECORD] 手动记录失败: {e}")

    def _record_successful_detection(self, target_id: str, detection: Dict, similarity_result: Dict):
        """记录成功检测到历史"""
        try:
            # 检查置信度是否足够记录
            final_score = similarity_result.get('final_score', 0.0)
            if final_score < self.min_confidence_for_history:
                print(f"[RECORD] 置信度过低，不记录历史: {final_score:.3f} < {self.min_confidence_for_history}")
                return
            
            if target_id not in self.successful_detection_history:
                self.successful_detection_history[target_id] = []
            
            # 提取特征
            successful_features = {
                'geometric': detection.get('geometric_features', {}),
                'appearance': detection.get('appearance_features', {}),
                'shape': detection.get('shape_features', {}),
                'spatial': detection.get('spatial_features', {}),
                'timestamp': datetime.now().isoformat(),
                'similarity_to_reference': similarity_result.get('reference_score', final_score),
                'final_confidence': final_score,
                'history_length_at_time': len(self.successful_detection_history[target_id])
            }
            
            self.successful_detection_history[target_id].append(successful_features)
            
            # 保持历史长度限制
            if len(self.successful_detection_history[target_id]) > self.max_history_length:
                removed = self.successful_detection_history[target_id].pop(0)
                print(f"[RECORD] 移除最旧历史记录: {removed.get('timestamp', 'unknown')}")
            
            current_length = len(self.successful_detection_history[target_id])
            print(f"[RECORD] ✅ 成功记录历史: {target_id}, 当前历史长度: {current_length}")
            
            # 🆕 自动保存历史（可选）
            self._save_detection_history()
            
        except Exception as e:
            print(f"[RECORD] 记录成功检测失败: {e}")

    def enable_auto_success_mode(self, enabled: bool = True, threshold: float = 0.85):
        """启用/禁用自动成功判断模式"""
        self.auto_success_enabled = enabled
        self.auto_success_threshold = threshold
        
        mode_str = "启用" if enabled else "禁用"
        print(f"[AUTO_MODE] {mode_str}自动成功判断，阈值: {threshold:.3f}")

    def get_detection_history_stats(self, target_id: str = None) -> Dict:
        """获取检测历史统计"""
        try:
            if target_id:
                if target_id in self.successful_detection_history:
                    history = self.successful_detection_history[target_id]
                    confidences = [h.get('final_confidence', 0) for h in history]
                    
                    return {
                        'target_id': target_id,
                        'history_length': len(history),
                        'avg_confidence': np.mean(confidences) if confidences else 0.0,
                        'min_confidence': np.min(confidences) if confidences else 0.0,
                        'max_confidence': np.max(confidences) if confidences else 0.0,
                        'latest_timestamp': history[-1].get('timestamp', 'unknown') if history else None
                    }
                else:
                    return {'target_id': target_id, 'history_length': 0}
            else:
                # 返回所有目标的统计
                stats = {}
                for tid in self.successful_detection_history:
                    stats[tid] = self.get_detection_history_stats(tid)
                return stats
                
        except Exception as e:
            print(f"[HISTORY_STATS] 获取历史统计失败: {e}")
            return {}

    def _save_detection_history(self):
        """保存检测历史到文件"""
        try:
            if hasattr(self, 'current_scan_dir') and self.current_scan_dir:
                history_dir = os.path.join(self.current_scan_dir, 'detection_history')
                os.makedirs(history_dir, exist_ok=True)
                
                history_file = os.path.join(history_dir, 'successful_detection_history.json')
                
                # 转换为可序列化格式
                serializable_history = {}
                for target_id, history in self.successful_detection_history.items():
                    serializable_history[target_id] = []
                    for record in history:
                        # 移除不能序列化的numpy数组等
                        clean_record = {}
                        for key, value in record.items():
                            if isinstance(value, (str, int, float, bool, list, dict)):
                                clean_record[key] = value
                            else:
                                clean_record[key] = str(value)
                        serializable_history[target_id].append(clean_record)
                
                with open(history_file, 'w', encoding='utf-8') as f:
                    json.dump(serializable_history, f, indent=2, ensure_ascii=False)
                    
                print(f"[SAVE_HISTORY] 检测历史已保存: {history_file}")
                
        except Exception as e:
            print(f"[SAVE_HISTORY] 保存检测历史失败: {e}")

    def load_detection_history(self, history_file: str = None):
        """加载检测历史"""
        try:
            if history_file is None and hasattr(self, 'current_scan_dir'):
                history_file = os.path.join(self.current_scan_dir, 'detection_history', 'successful_detection_history.json')
            
            if history_file and os.path.exists(history_file):
                with open(history_file, 'r', encoding='utf-8') as f:
                    loaded_history = json.load(f)
                
                self.successful_detection_history = loaded_history
                print(f"[LOAD_HISTORY] ✅ 检测历史已加载: {len(loaded_history)} 个目标")
                
                for target_id, history in loaded_history.items():
                    print(f"  - {target_id}: {len(history)} 条历史记录")
                
                return True
            else:
                print(f"[LOAD_HISTORY] 历史文件不存在: {history_file}")
                return False
                
        except Exception as e:
            print(f"[LOAD_HISTORY] 加载检测历史失败: {e}")
            return False
        
    def _build_tracking_result_with_detailed_features(self, target_id: str, detection: Dict, 
                                                    detailed_similarity: Dict, waypoint_data: Dict,
                                                    image_rgb: np.ndarray, depth_image: np.ndarray) -> Dict:
        """构建包含详细特征的追踪结果"""
        try:
            print("[ENHANCED_TRACKER] 构建详细追踪结果...")
            
            # 🆕 计算最小外接矩形
            bounding_rect = self._calculate_minimum_bounding_rect(detection)
            
            # 计算3D抓取坐标
            grasp_coord = self._calculate_grasp_coordinate(detection, waypoint_data, depth_image)
            
            # 分析物体信息
            features = detection.get('features', {})
            object_info = self._analyze_object_for_grasping_fixed(detection, waypoint_data, features, depth_image)
            
            # 构建追踪结果
            tracking_result = {
                'target_id': target_id,
                'detection_confidence': float(detection.get('confidence', 0.0)),
                'tracking_confidence': float(detailed_similarity['final_score']),
                
                # 抓取坐标
                'grasp_coordinate': grasp_coord,
                
                # 物体信息
                'object_info': object_info,
                
                # 🆕 最小外接矩形信息
                'bounding_rect': bounding_rect,
                
                # 🆕 详细相似度分解 - 确保传递
                'detailed_similarity_breakdown': detailed_similarity.get('detailed_breakdown', {}),
                'similarity_breakdown': detailed_similarity.get('feature_contributions', {}),
                'feature_weights_used': detailed_similarity.get('feature_weights_used', {}),
                
                # 元数据
                'timestamp': datetime.now().isoformat(),
                'waypoint_info': waypoint_data
            }
            
            print(f"[ENHANCED_TRACKER] 详细追踪结果构建完成:")
            print(f"  抓取坐标: {grasp_coord}")
            print(f"  详细特征数量: {len(detailed_similarity.get('detailed_breakdown', {}))}")
            print(f"  相似度分解数量: {len(detailed_similarity.get('feature_contributions', {}))}")
            
            return tracking_result
            
        except Exception as e:
            print(f"[ENHANCED_TRACKER] 构建详细追踪结果失败: {e}")
            import traceback
            traceback.print_exc()
            return None
        
    def _calculate_minimum_bounding_rect(self, detection: Dict) -> Dict:
        """基于mask计算最小外接矩形"""
        try:
            import cv2
            import numpy as np
            
            mask = detection.get('mask')
            if mask is None or not isinstance(mask, np.ndarray):
                print("[ENHANCED_TRACKER] 无效的mask数据")
                return {'width': 0, 'height': 0, 'angle': 0, 'center': [0, 0]}
            
            # 确保mask是正确的格式
            if mask.dtype != np.uint8:
                if mask.dtype == bool:
                    mask = mask.astype(np.uint8) * 255
                else:
                    mask = mask.astype(np.uint8)
            
            # 找到轮廓
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                print("[ENHANCED_TRACKER] 未找到有效轮廓")
                return {'width': 0, 'height': 0, 'angle': 0, 'center': [0, 0]}
            
            # 获取最大轮廓的最小外接矩形
            largest_contour = max(contours, key=cv2.contourArea)
            rect = cv2.minAreaRect(largest_contour)
            center, (width, height), angle = rect

            print(f"[ENHANCED_TRACKER] 最小外接矩形: 中心{center}, 尺寸({width:.1f}x{height:.1f}), 角度{angle:.1f}°")
            
            return {
                'width': float(width),
                'height': float(height), 
                'angle': float(angle),
                'center': [float(center[0]), float(center[1])]
            }
            
        except Exception as e:
            print(f"[ENHANCED_TRACKER] 最小外接矩形计算失败: {e}")
            return {'width': 0, 'height': 0, 'angle': 0, 'center': [0, 0]}        
    
    def _calculate_grasp_coordinate(self, detection: Dict, waypoint_data: Dict, depth_image: np.ndarray) -> Dict:
        """通过ObjectAnalyzer计算3D抓取坐标"""
        try:
            mask = detection.get('mask')
            bbox = detection.get('bounding_box', [0, 0, 100, 100])
            
            if mask is None or depth_image is None:
                raise ValueError("缺少mask或depth图")

            spatial_features = self.object_analyzer.calculate_3d_spatial_features(
                mask, depth_image, waypoint_data, bbox
            )

            coords = spatial_features.get('world_coordinates', [0, 0, 350])
            return {
                'x': float(coords[0]),
                'y': float(coords[1]),
                'z': float(coords[2])
            }
        except Exception as e:
            print(f"[ENHANCED_TRACKER] 统一坐标转换失败: {e}")
            return {'x': 0.0, 'y': 0.0, 'z': 350.0}

    def _analyze_object_for_grasping_fixed(self, detection: Dict, waypoint_data, features: Dict, 
                                        depth_image: np.ndarray) -> Dict:
        """分析物体抓取信息 - 修复版，正确获取宽度"""
        try:
            print(f"[ENHANCED_TRACKER] 开始物体分析，目标ID: {self.current_target_id}")
            
            # 🆕 首先尝试从当前检测的spatial_features获取宽度信息
            estimated_width = 50.0  # 默认值
            
            # 🔧 方法1：从当前检测的spatial_features获取gripper_width_info
            spatial_features = detection.get('spatial_features', {})
            gripper_info = spatial_features.get('gripper_width_info', {})
            
            if gripper_info and 'real_width_mm' in gripper_info:
                estimated_width = float(gripper_info['real_width_mm'])
                print(f"[ENHANCED_TRACKER] ✅ 从spatial_features获取宽度: {estimated_width:.1f}mm")
            else:
                # 🔧 方法2：从features字典中获取
                if 'spatial' in features and 'gripper_width_info' in features['spatial']:
                    gripper_info = features['spatial']['gripper_width_info']
                    if 'real_width_mm' in gripper_info:
                        estimated_width = float(gripper_info['real_width_mm'])
                        print(f"[ENHANCED_TRACKER] ✅ 从features获取宽度: {estimated_width:.1f}mm")
                else:
                    # 🔧 方法3：尝试从参考库获取（备用）
                    if (hasattr(self, 'current_target_id') and 
                        self.current_target_id and 
                        hasattr(self, 'reference_library') and
                        self.current_target_id in self.reference_library):
                        
                        ref_entry = self.reference_library[self.current_target_id]
                        spatial_ref = ref_entry.get('features', {}).get('spatial', {})
                        gripper_ref = spatial_ref.get('gripper_width_info', {})
                        
                        if 'real_width_mm' in gripper_ref:
                            estimated_width = float(gripper_ref['real_width_mm'])
                            print(f"[ENHANCED_TRACKER] ✅ 从参考库获取宽度: {estimated_width:.1f}mm")
                        else:
                            print(f"[ENHANCED_TRACKER] ⚠️ 参考库中无宽度信息，使用默认值: {estimated_width:.1f}mm")
                    else:
                        print(f"[ENHANCED_TRACKER] ⚠️ 无法获取宽度信息，使用默认值: {estimated_width:.1f}mm")
            
            # 🔧 从参考库获取其他数据（高度、背景等）
            if (hasattr(self, 'current_target_id') and 
                self.current_target_id and 
                hasattr(self, 'reference_library') and
                self.current_target_id in self.reference_library):
                
                ref_entry = self.reference_library[self.current_target_id]
                spatial_features_ref = ref_entry.get('features', {}).get('spatial', {})
                metadata = ref_entry.get('metadata', {})
                
                # 从参考库读取高度和背景信息
                height_mm = spatial_features_ref.get('height_mm')
                background_world_z = spatial_features_ref.get('background_world_z')
                class_name = metadata.get('class_name', 'unknown')
                
                print(f"[ENHANCED_TRACKER] 参考库数据:")
                print(f"  height_mm: {height_mm}")
                print(f"  background_world_z: {background_world_z}")
                print(f"  class_name: {class_name}")
                print(f"  estimated_width: {estimated_width}")
                
                if height_mm is not None and background_world_z is not None:
                    object_info = {
                        'object_id': self.current_target_id,
                        'class_name': class_name,
                        'estimated_height': float(height_mm),
                        'background_z': float(background_world_z),
                        'estimated_width': float(estimated_width),  # ← 🔧 关键修复
                        'recommended_gripper_width': 500,
                        'grasp_angle': waypoint_data.get('yaw', 0.0)
                    }
                    
                    print(f"[ENHANCED_TRACKER] ✅ 使用完整数据:")
                    print(f"  高度: {object_info['estimated_height']:.1f}mm")
                    print(f"  宽度: {object_info['estimated_width']:.1f}mm")  # ← 🔧 显示宽度
                    print(f"  背景: {object_info['background_z']:.1f}mm")
                    print(f"  类别: {object_info['class_name']}")
                    
                    return object_info
            
            # 如果参考库数据不可用，使用默认值但保持计算得到的宽度
            print(f"[ENHANCED_TRACKER] ⚠️ 使用默认值，但保持计算宽度: {estimated_width:.1f}mm")
            return {
                'object_id': 'unknown',
                'class_name': 'unknown',
                'estimated_height': 30.0,
                'background_z': 300.0,
                'estimated_width': float(estimated_width),  # ← 🔧 使用计算出的宽度而非默认值
                'recommended_gripper_width': 500,
                'grasp_angle': 0.0
            }
            
        except Exception as e:
            print(f"[ENHANCED_TRACKER] 物体分析失败: {e}")
            import traceback
            traceback.print_exc()
            return {
                'object_id': 'unknown',
                'class_name': 'unknown',
                'estimated_height': 30.0,
                'background_z': 300.0,
                'estimated_width': 50.0,  # 只有在异常情况下才使用硬编码默认值
                'recommended_gripper_width': 500,
                'grasp_angle': 0.0
            }

    def _get_fixed_object_info_with_cache(self, detection: Dict) -> Dict:
        """从固定检测结果获取物体信息 - 优先使用参考库数据"""
        try:
            object_id = detection.get('object_id')
            class_name = detection.get('class_name', 'unknown')
            
            # 默认值
            default_info = {
                'object_id': object_id or 'unknown',
                'class_name': class_name,
                'estimated_height': 30.0,
                'background_z': 300.0,
                'recommended_gripper_width': 500,
            }
            
            print(f"[ENHANCED_TRACKER] 获取物体信息: 目标ID={self.current_target_id}")
            
            # 🔧 关键修复：优先使用参考库数据
            if hasattr(self, 'current_target_id') and self.current_target_id:
                if self.current_target_id in self.reference_library:
                    ref_entry = self.reference_library[self.current_target_id]
                    if 'features' in ref_entry and 'spatial' in ref_entry['features']:
                        spatial_ref = ref_entry['features']['spatial']
                        depth_info = spatial_ref.get('depth_info', {})
                        
                        if 'height_mm' in depth_info:
                            default_info['estimated_height'] = float(depth_info['height_mm'])
                            print(f"[ENHANCED_TRACKER] 📚 使用参考库高度: {default_info['estimated_height']:.1f}mm")
                        if 'background_world_z' in depth_info:
                            default_info['background_z'] = float(depth_info['background_world_z'])
                            print(f"[ENHANCED_TRACKER] 📚 使用参考库背景: {default_info['background_z']:.1f}mm")
                        
                        # 如果参考库中有完整数据，直接返回
                        if 'height_mm' in depth_info and 'background_world_z' in depth_info:
                            print(f"[ENHANCED_TRACKER] ✅ 使用参考库完整数据 (优先级1)")
                            return default_info
            
            # 🔧 备用方案：使用检测结果缓存
            if hasattr(self, 'detection_results_cache') and self.current_target_id:
                print(f"[ENHANCED_TRACKER] 检查缓存，缓存大小: {len(getattr(self, 'detection_results_cache', {}))}")
                
                if self.current_target_id in self.detection_results_cache:
                    cache_data = self.detection_results_cache[self.current_target_id]
                    
                    # 🆕 只在参考库数据不完整时使用缓存数据
                    if default_info['estimated_height'] == 30.0:  # 参考库没有高度数据
                        default_info['estimated_height'] = cache_data['height_mm']
                        print(f"[ENHANCED_TRACKER] 💾 补充缓存高度: {default_info['estimated_height']:.1f}mm")
                    
                    if default_info['background_z'] == 300.0:  # 参考库没有背景数据
                        default_info['background_z'] = cache_data['background_world_z']
                        print(f"[ENHANCED_TRACKER] 💾 补充缓存背景: {default_info['background_z']:.1f}mm")
                    
                    if default_info['class_name'] == 'unknown':  # 补充类别信息
                        default_info['class_name'] = cache_data['class_name']
                    
                    print(f"[ENHANCED_TRACKER] ✅ 使用参考库+缓存组合数据 (优先级2)")
                    return default_info
                else:
                    print(f"[ENHANCED_TRACKER] ⚠️ 目标 {self.current_target_id} 不在缓存中")
                    print(f"[ENHANCED_TRACKER] 可用缓存键: {list(self.detection_results_cache.keys())}")
            else:
                print(f"[ENHANCED_TRACKER] ⚠️ 缓存不可用: hasattr={hasattr(self, 'detection_results_cache')}, target_id={self.current_target_id}")
            
            print(f"[ENHANCED_TRACKER] ⚠️ 使用默认值 (优先级3): 高度={default_info['estimated_height']:.1f}mm, 背景={default_info['background_z']:.1f}mm")
            return default_info
            
        except Exception as e:
            print(f"[ENHANCED_TRACKER] 获取固定物体信息失败: {e}")
            import traceback
            traceback.print_exc()
            return {
                'object_id': 'unknown',
                'class_name': 'unknown', 
                'estimated_height': 30.0,
                'background_z': 300.0,
                'recommended_gripper_width': 500,
                'grasp_angle': 0.0
            }
        
    def set_detection_results_cache(self, cache: Dict):
        """设置检测结果缓存"""
        self.detection_results_cache = cache
        print(f"[ENHANCED_TRACKER] 检测结果缓存已设置: {len(cache)} 个目标")

    def _calculate_detailed_similarity(self, reference_features: Dict, 
                                    candidate_detection: Dict, 
                                    waypoint_data: Dict) -> Dict:
        """计算详细相似度 - 新增方法"""
        try:
            # 提取候选特征
            candidate_features = {}
            
            # 尝试多种特征提取方式
            if 'geometric_features' in candidate_detection:
                candidate_features['geometric'] = candidate_detection['geometric_features']
            if 'appearance_features' in candidate_detection:
                candidate_features['appearance'] = candidate_detection['appearance_features']
            if 'shape_features' in candidate_detection:
                candidate_features['shape'] = candidate_detection['shape_features']
            if 'spatial_features' in candidate_detection:
                candidate_features['spatial'] = candidate_detection['spatial_features']
            
            # 如果上面没有找到，尝试从features字段提取
            if not candidate_features and 'features' in candidate_detection:
                features = candidate_detection['features']
                candidate_features['geometric'] = features.get('geometric', {})
                candidate_features['appearance'] = features.get('appearance', features.get('color', {}))
                candidate_features['shape'] = features.get('shape', {})
                candidate_features['spatial'] = features.get('spatial', {})
            
            print(f"[DETAILED_SIMILARITY] 参考特征类型: {list(reference_features.keys())}")
            print(f"[DETAILED_SIMILARITY] 候选特征类型: {list(candidate_features.keys())}")
            
            # 使用详细相似度计算方法
            detailed_result = self.similarity_calculator.calculate_detailed_similarity_breakdown(
                reference_features, 
                candidate_features, 
                waypoint_data
            )
            
            print(f"[DETAILED_SIMILARITY] 详细相似度计算完成: {detailed_result['final_score']:.3f}")
            
            return detailed_result
            
        except Exception as e:
            print(f"[DETAILED_SIMILARITY] 详细相似度计算失败: {e}")
            import traceback
            traceback.print_exc()
            
            # 返回基本结构，避免错误
            return {
                'final_score': 0.0,
                'detailed_breakdown': {},
                'feature_contributions': {},
                'feature_weights_used': {}
            }
        
    def _extract_candidate_features(self, candidate: Dict, image: np.ndarray,
                                   depth: np.ndarray, waypoint_data: Dict) -> Dict:
        """为候选检测提取特征 - 优化版本"""
        try:
            # 🆕 如果candidate已经包含detection_pipeline提取的特征，直接使用
            if 'geometric_features' in candidate:
                features = {
                    'geometric': candidate.get('geometric_features', {}),
                    'appearance': candidate.get('appearance_features', {}),
                    'shape': candidate.get('shape_features', {}),
                    'spatial': candidate.get('spatial_features', {})
                }
                
                # 验证空间特征是否完整，如果不完整则重新计算
                if not features['spatial'].get('world_coordinates'):
                    bbox = candidate.get('bounding_box', [0, 0, 100, 100])
                    mask = candidate.get('mask', np.zeros((image.shape[0], image.shape[1]), dtype=bool))
                    
                    spatial_features = self.object_analyzer.calculate_3d_spatial_features(
                        mask, depth, waypoint_data, bbox
                    )
                    features['spatial'] = spatial_features
                
                print(f"[ENHANCED_TRACKER] 使用detection_pipeline提取的特征")
                return features
            
            # 🆕 如果没有预提取特征，执行完整特征提取（备用方案）
            else:
                bbox = candidate.get('bounding_box', [0, 0, 100, 100])
                mask = candidate.get('mask', np.zeros((image.shape[0], image.shape[1]), dtype=bool))
                
                # 调用完整的特征提取管道
                spatial_features = self.object_analyzer.calculate_3d_spatial_features(
                    mask, depth, waypoint_data, bbox
                )
                
                # 这里可以调用其他特征提取器
                # geometric_features = self.extract_geometric_features(mask, depth)
                # appearance_features = self.extract_appearance_features(image, mask)
                # shape_features = self.extract_shape_features(mask)
                
                features = {
                    'geometric': {},  # 暂时为空，可以后续添加
                    'appearance': {},
                    'shape': {},
                    'spatial': spatial_features
                }
                
                print(f"[ENHANCED_TRACKER] 执行完整特征提取")
                return features
                
        except Exception as e:
            print(f"[ENHANCED_TRACKER] 特征提取失败: {e}")
            return {
                'geometric': {},
                'appearance': {},
                'shape': {},
                'spatial': {}
            }
    
    def _get_adaptive_threshold(self, target_info: Dict, confidence: float) -> float:
        """获取自适应阈值"""
        if not self.adaptive_manager:
            # 使用配置文件中的默认阈值
            similarity_config = self.config_manager.get_similarity_config()
            return similarity_config.get('thresholds', {}).get('overall_match', 0.75)
        
        # 使用自适应阈值
        feature_quality = target_info.get('quality_score', 75.0)
        context = {
            'tracking_state': target_info.get('tracking_state', 'unknown'),
            'confidence_history': target_info.get('confidence_history', [])
        }
        
        return self.adaptive_manager.get_adaptive_threshold(
            'overall', 'match', feature_quality, context
        )
    
    def _temporal_consistency_check(self, target_id: str, best_match: Dict) -> bool:
        """时序一致性检查"""
        try:
            target_info = self.current_targets[target_id]
            position_history = target_info.get('position_history', [])
            
            if len(position_history) < 2:
                return True  # 历史数据不足，跳过检查
            
            # 获取当前位置
            current_spatial = best_match['candidate_data']['features'].get('spatial', {})
            current_position = current_spatial.get('world_coordinates', [0, 0, 0])
            
            # 计算与最近位置的距离
            last_position = position_history[-1]
            distance = np.linalg.norm(np.array(current_position) - np.array(last_position))
            
            # 最大允许移动距离（毫米）
            max_movement = 200.0  # 200mm
            
            if distance > max_movement:
                print(f"[ENHANCED_TRACKER] 时序一致性检查失败: 移动距离 {distance:.1f}mm > {max_movement}mm")
                return False
            
            return True
            
        except Exception as e:
            print(f"[ENHANCED_TRACKER] 时序一致性检查异常: {e}")
            return True  # 异常时允许通过
    
    def _generate_tracking_result(self, target_id: str, best_match: Dict, 
                                waypoint_data: Dict) -> Dict:
        """生成追踪结果"""
        try:
            candidate_data = best_match['candidate_data']
            similarity_result = best_match['similarity_result']
            
            # 获取3D坐标和抓取信息
            spatial_features = candidate_data['features'].get('spatial', {})
            world_coordinates = spatial_features.get('world_coordinates', [0, 0, 0])
            gripper_info = spatial_features.get('gripper_width_info', {})
            angle = gripper_info.get('angle', 0.0)
            # 构建追踪结果
            tracking_result = {
                'target_id': target_id,
                'detection_confidence': candidate_data['original_data'].get('confidence', 0.0),
                'tracking_confidence': similarity_result['overall_similarity'],
                'match_confidence': similarity_result['confidence'],
                
                # 抓取坐标信息
                'grasp_coordinate': {
                    'x': float(world_coordinates[0]),
                    'y': float(world_coordinates[1]),
                    'z': float(world_coordinates[2])
                },
               

                # 物体抓取信息
                'object_info': {
                    'object_id': target_id,
                    'class_name': self.current_targets[target_id]['metadata'].get('class_name', 'unknown'),
                    'estimated_height': spatial_features.get('height_mm', 30.0),
                    'background_z': spatial_features.get('background_world_z', 300.0),
                    'recommended_gripper_width': gripper_info.get('recommended_gripper_width', 150),
                    'grasp_angle': gripper_info.get('angle', 0.0)
                },
                
                # 详细相似度信息
                'similarity_breakdown': similarity_result['feature_similarities'],
                'valid_features': similarity_result['valid_features'],
                
                # 追踪元数据
                'timestamp': datetime.now().isoformat(),
                'frame_count': self.current_frame_count,
                'waypoint_info': waypoint_data
            }
            
            return tracking_result
            
        except Exception as e:
            print(f"[ENHANCED_TRACKER] 生成追踪结果失败: {e}")
            return None
    
    def _update_tracking_state(self, target_id: str, tracking_result: Dict):
        """更新追踪状态"""
        try:
            target_info = self.current_targets[target_id]
            
            # 更新基本状态
            target_info['tracking_state'] = 'tracking'
            target_info['last_detection'] = tracking_result
            
            # 更新置信度历史
            confidence_history = target_info.get('confidence_history', [])
            confidence_history.append(tracking_result['tracking_confidence'])
            if len(confidence_history) > 10:  # 保持最近10个记录
                confidence_history = confidence_history[-10:]
            target_info['confidence_history'] = confidence_history
            
            # 更新位置历史
            position_history = target_info.get('position_history', [])
            current_position = [
                tracking_result['grasp_coordinate']['x'],
                tracking_result['grasp_coordinate']['y'],
                tracking_result['grasp_coordinate']['z']
            ]
            position_history.append(current_position)
            if len(position_history) > 5:  # 保持最近5个位置
                position_history = position_history[-5:]
            target_info['position_history'] = position_history
            
            # 更新帧计数
            self.current_frame_count += 1
            
            print(f"[ENHANCED_TRACKER] 已更新追踪状态: {target_id}")
            
        except Exception as e:
            print(f"[ENHANCED_TRACKER] 更新追踪状态失败: {e}")
    
    def _handle_tracking_failure(self, target_id: str) -> Optional[Dict]:
        """处理追踪失败"""
        try:
            target_info = self.current_targets[target_id]
            
            # 更新失败状态
            target_info['tracking_state'] = 'lost'
            
            # 检查是否有最近的检测结果可以使用
            last_detection = target_info.get('last_detection')
            if last_detection:
                # 返回最后一次成功的检测结果，但标记为不确定
                failure_result = last_detection.copy()
                failure_result['tracking_confidence'] = 0.0
                failure_result['match_confidence'] = 0.0
                failure_result['status'] = 'lost'
                failure_result['timestamp'] = datetime.now().isoformat()
                
                print(f"[ENHANCED_TRACKER] 追踪失败，返回最后已知位置: {target_id}")
                return failure_result
            
            print(f"[ENHANCED_TRACKER] 追踪彻底失败: {target_id}")
            return None
            
        except Exception as e:
            print(f"[ENHANCED_TRACKER] 处理追踪失败异常: {e}")
            return None
    
    def _update_adaptive_learning(self, target_id: str, match_result: Dict, is_correct: bool):
        """更新自适应学习"""
        try:
            if not self.adaptive_manager:
                return
            
            target_info = self.current_targets[target_id]
            similarity_result = match_result['similarity_result']
            feature_similarities = similarity_result['feature_similarities']
            feature_quality = target_info.get('quality_score', 75.0)
            
            # 为每个特征类型更新学习历史
            for feature_type, similarity in feature_similarities.items():
                self.adaptive_manager.update_learning_history(
                    feature_type=feature_type,
                    sub_feature='overall',
                    similarity=similarity,
                    feature_quality=feature_quality,
                    is_correct_match=is_correct,
                    context={
                        'target_id': target_id,
                        'frame_count': self.current_frame_count,
                        'tracking_state': target_info.get('tracking_state', 'unknown')
                    }
                )
            
            print(f"[ENHANCED_TRACKER] 已更新自适应学习: {target_id} ({'成功' if is_correct else '失败'})")
            
        except Exception as e:
            print(f"[ENHANCED_TRACKER] 更新自适应学习失败: {e}")
    
    def get_tracking_status(self) -> Dict:
        """获取追踪状态"""
        try:
            status = {
                'tracking_active': self.tracking_active,
                'current_frame_count': self.current_frame_count,
                'total_targets': len(self.current_targets),
                'targets_status': {}
            }
            
            for target_id, target_info in self.current_targets.items():
                confidence_history = target_info.get('confidence_history', [])
                status['targets_status'][target_id] = {
                    'state': target_info.get('tracking_state', 'unknown'),
                    'quality_score': target_info.get('quality_score', 0),
                    'avg_confidence': np.mean(confidence_history) if confidence_history else 0.0,
                    'last_confidence': confidence_history[-1] if confidence_history else 0.0,
                    'detection_count': len(confidence_history)
                }
            
            return status
            
        except Exception as e:
            print(f"[ENHANCED_TRACKER] 获取追踪状态失败: {e}")
            return {'error': str(e)}
    
    def optimize_tracking_parameters(self) -> Dict:
        """优化追踪参数"""
        try:
            if not self.adaptive_manager:
                return {'message': 'adaptive learning disabled'}
            
            # 执行阈值优化
            optimization_results = self.adaptive_manager.optimize_thresholds(min_samples=20)
            
            if optimization_results:
                # 更新相似度计算器的权重
                updated_weights = {}
                for feature_key, result in optimization_results.items():
                    if '.' in feature_key:
                        main_type = feature_key.split('.')[0]
                        if main_type not in updated_weights:
                            updated_weights[main_type] = 0.0
                        # 根据优化结果调整权重
                        improvement = result.get('improvement', 0)
                        updated_weights[main_type] += improvement * 0.1  # 小幅调整
                
                if updated_weights:
                    # 归一化权重
                    total_weight = sum(abs(w) for w in updated_weights.values())
                    if total_weight > 0:
                        current_weights = self.similarity_calculator.feature_weights.copy()
                        for feature_type, adjustment in updated_weights.items():
                            if feature_type in current_weights:
                                current_weights[feature_type] += adjustment * 0.1
                        
                        # 重新归一化
                        total = sum(current_weights.values())
                        if total > 0:
                            current_weights = {k: v/total for k, v in current_weights.items()}
                            self.similarity_calculator.update_weights(current_weights)
                
                print(f"[ENHANCED_TRACKER] 参数优化完成: {len(optimization_results)} 个特征已优化")
                return {
                    'optimized_features': len(optimization_results),
                    'results': optimization_results
                }
            else:
                return {'message': 'insufficient data for optimization'}
                
        except Exception as e:
            print(f"[ENHANCED_TRACKER] 参数优化失败: {e}")
            return {'error': str(e)}
    
    def save_tracking_session(self, output_path: str):
        """保存追踪会话数据"""
        try:
            session_data = {
                'session_info': {
                    'start_time': datetime.now().isoformat(),
                    'frame_count': self.current_frame_count,
                    'total_targets': len(self.current_targets)
                },
                'targets': {},
                'tracking_history': self.tracking_history,
                'adaptive_learning_stats': self.adaptive_manager.get_performance_report() if self.adaptive_manager else None
            }
            
            # 保存目标信息（移除不能序列化的数据）
            for target_id, target_info in self.current_targets.items():
                session_data['targets'][target_id] = {
                    'metadata': target_info['metadata'],
                    'quality_score': target_info['quality_score'],
                    'tracking_state': target_info['tracking_state'],
                    'confidence_history': target_info['confidence_history'],
                    'position_history': target_info['position_history']
                }
            
            # 保存到文件
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False)
            
            print(f"[ENHANCED_TRACKER] 追踪会话已保存: {output_path}")
            
        except Exception as e:
            print(f"[ENHANCED_TRACKER] 保存追踪会话失败: {e}")
    
    def reset_tracking(self):
        """重置追踪状态"""
        try:
            self.current_targets.clear()
            self.tracking_history.clear()
            self.current_frame_count = 0
            self.tracking_active = False
            
            print("[ENHANCED_TRACKER] 追踪状态已重置")
            
        except Exception as e:
            print(f"[ENHANCED_TRACKER] 重置追踪状态失败: {e}")
    
    def get_performance_report(self) -> Dict:
        """获取性能报告"""
        try:
            report = {
                'session_stats': {
                    'total_frames': self.current_frame_count,
                    'active_targets': len([t for t in self.current_targets.values() 
                                         if t.get('tracking_state') == 'tracking']),
                    'lost_targets': len([t for t in self.current_targets.values() 
                                       if t.get('tracking_state') == 'lost'])
                },
                'feature_performance': {},
                'adaptive_learning': None
            }
            
            # 计算各特征的平均性能
            all_confidences = {}
            for target_info in self.current_targets.values():
                for confidence in target_info.get('confidence_history', []):
                    if 'overall' not in all_confidences:
                        all_confidences['overall'] = []
                    all_confidences['overall'].append(confidence)
            
            for feature_type, confidences in all_confidences.items():
                if confidences:
                    report['feature_performance'][feature_type] = {
                        'mean_confidence': float(np.mean(confidences)),
                        'std_confidence': float(np.std(confidences)),
                        'min_confidence': float(np.min(confidences)),
                        'max_confidence': float(np.max(confidences)),
                        'sample_count': len(confidences)
                    }
            
            # 自适应学习报告
            if self.adaptive_manager:
                report['adaptive_learning'] = self.adaptive_manager.get_performance_report()
            
            return report
            
        except Exception as e:
            print(f"[ENHANCED_TRACKER] 获取性能报告失败: {e}")
            return {'error': str(e)}
        
    def _build_tracking_context(self, waypoint_data: Dict, candidates: List[Dict]) -> Dict:
        """构建追踪上下文 - 修复版"""
        try:
            # 🔧 修复距离计算
            current_pos = waypoint_data.get('world_pos', [0, 0, 350])
            # 计算到相机的距离（毫米转米）
            distance_to_camera = np.sqrt(current_pos[0]**2 + current_pos[1]**2) / 1000.0
            
            print(f"[CONTEXT] 当前位置: {current_pos}")
            print(f"[CONTEXT] 距离相机: {distance_to_camera:.3f}m")
            
            # 更新历史记录
            self.learning_context['distance_history'].append(distance_to_camera)
            if len(self.learning_context['distance_history']) > 10:
                self.learning_context['distance_history'] = self.learning_context['distance_history'][-10:]
            
            return {
                'distance_to_target': distance_to_camera,
                'num_candidates': len(candidates),
                'current_position': current_pos,
                'distance_history': self.learning_context['distance_history']
            }
        except Exception as e:
            print(f"[CONTEXT] 构建上下文失败: {e}")
            return {'distance_to_target': 0.5, 'num_candidates': len(candidates)}

    def _get_realtime_adaptive_weights(self, context: Dict) -> Optional[Dict]:
        """Layer 1: 实时自适应权重调整 - 修复版"""
        try:
            distance = context.get('distance_to_target', 1.0)
            num_candidates = context.get('num_candidates', 1)
            
            print(f"[LAYER1] 实时调参检查: 距离={distance:.3f}m, 候选数={num_candidates}")
            
            # 🔧 修复条件判断
            need_adjustment = False
            adjustment_reason = ""
            adaptive_weights = None
            
            # 条件1: 距离过近，可能遮挡 (修复阈值)
            if distance < 0.3:  # 50cm以内就开始调整
                need_adjustment = True
                adjustment_reason = "close_distance_occlusion"
                adaptive_weights = {
                    'geometric': 0.35,    # 增加几何特征权重
                    'appearance': 0.2,   # 大幅降低颜色权重
                    'shape': 0.35,        # 保持形状权重
                    'spatial': 0.1       # 降低空间权重（因为在移动）
                }
            
            # 条件2: 多候选且相似度接近时，降低空间权重
            elif num_candidates > 1:
                need_adjustment = True
                adjustment_reason = "multiple_candidates_dynamic"
                adaptive_weights = {
                    'geometric': 0.35,   # 稍微增加几何权重
                    'appearance': 0.30,  # 稍微降低颜色权重
                    'shape': 0.30,       # 增加形状权重
                    'spatial': 0.05      # 大幅降低空间权重（动态场景）
                }
            
            # 条件3: 置信度急剧下降
            elif self._detect_confidence_degradation():
                need_adjustment = True
                adjustment_reason = "confidence_degradation"
                adaptive_weights = {
                    'geometric': 0.45,
                    'appearance': 0.15,
                    'shape': 0.35,
                    'spatial': 0.05
                }
            
            if need_adjustment:
                print(f"[LAYER1] ✅ 触发实时调整: {adjustment_reason}")
                print(f"[LAYER1] 新权重: {adaptive_weights}")
                return adaptive_weights
            else:
                print(f"[LAYER1] ❌ 无需调整")
                return None
            
        except Exception as e:
            print(f"[LAYER1] 实时权重调整失败: {e}")
            return None

    def _detect_confidence_degradation(self) -> bool:
        """检测置信度急剧下降"""
        try:
            history = self.learning_context.get('confidence_history', [])
            if len(history) < 3:
                return False
            
            recent = history[-3:]
            # 如果从高置信度急剧下降到低置信度
            if recent[0] > 0.7 and recent[-1] < 0.65:
                return True
            
            return False
        except:
            return False

    def _process_user_feedback_learning(self, learning_data: Dict, feedback: str):
        """Layer 2: 基于用户反馈的学习 - 修复版"""
        try:
            if not hasattr(self, 'online_learner'):
                print("[LAYER2] online_learner 不存在")
                return
            
            # 🔧 安全地提取tracking_result
            if isinstance(learning_data, dict):
                tracking_result = learning_data.get('tracking_result')
                target_id = learning_data.get('target_id', self.current_target_id or 'unknown')
            else:
                # 如果learning_data就是tracking_result
                tracking_result = learning_data
                target_id = self.current_target_id or 'unknown'
            
            if not tracking_result:
                print("[LAYER2] tracking_result 为空，跳过学习")
                return
            
            print(f"[LAYER2] 开始处理用户反馈: {feedback} for {target_id}")
            
            # 🆕 构建完整的学习记录
            complete_learning_record = {
                'tracking_result': tracking_result,
                'target_id': target_id,
                'step_number': learning_data.get('step_number', 0) if isinstance(learning_data, dict) else 0,
                'timestamp': datetime.now().isoformat(),
                'user_feedback': feedback,
                # 🆕 添加当前特征权重信息
                'current_weights': self.similarity_calculator.feature_weights.copy(),
                # 🆕 添加相似度分解信息
                'similarity_breakdown': tracking_result.get('similarity_breakdown', {}),
                'tracking_confidence': tracking_result.get('tracking_confidence', 0.0)
            }
            
            # 更新在线学习器
            is_correct = (feedback == 'success')
            self.online_learner.update_with_feedback(
                target_id, 
                complete_learning_record, 
                is_correct
            )
            
            # 更新置信度历史
            confidence = tracking_result.get('tracking_confidence', 0.0)
            self.learning_context['confidence_history'].append(confidence)
            if len(self.learning_context['confidence_history']) > 10:
                self.learning_context['confidence_history'] = self.learning_context['confidence_history'][-10:]
            
            print(f"[LAYER2] ✅ 用户反馈学习完成: {feedback}")
            
            # 🆕 如果连续失败，立即进行微调
            if not is_correct:
                self._check_immediate_adjustment(target_id)
            
        except Exception as e:
            print(f"[LAYER2] 用户反馈学习失败: {e}")
            import traceback
            traceback.print_exc()

    def _check_immediate_adjustment(self, target_id: str):
        """检查是否需要立即调整权重"""
        try:
            if not hasattr(self, 'online_learner'):
                return
            
            class_name = target_id.split('_')[0] if target_id else 'unknown'
            class_report = self.online_learner.get_class_performance_report(class_name)
            
            # 如果最近准确率很低，立即触发微调
            recent_accuracy = class_report.get('recent_accuracy', 1.0)
            total_feedback = class_report.get('total_feedback', 0)
            
            print(f"[IMMEDIATE_CHECK] {class_name}: 准确率={recent_accuracy:.3f}, 反馈数={total_feedback}")
            
            if total_feedback >= 3 and recent_accuracy < 0.5:  # 最近准确率低于50%
                print(f"[IMMEDIATE_CHECK] ⚠️ 触发立即微调: {class_name}")
                self.online_learner._perform_micro_update(class_name)
                
        except Exception as e:
            print(f"[IMMEDIATE_CHECK] 立即调整检查失败: {e}")

    def _trigger_long_term_optimization(self):
        """Layer 3: 长期全局优化"""
        try:
            if not hasattr(self, 'adaptive_manager') or not self.adaptive_manager:
                return False
            
            # 执行阈值优化
            optimization_results = self.adaptive_manager.optimize_thresholds(min_samples=15)
            
            if optimization_results:
                print(f"[LAYER3] 长期优化完成: 优化了 {len(optimization_results)} 个特征")
                
                # 保存学习数据
                if hasattr(self, 'online_learner'):
                    self.online_learner.save_learning_data()
                self.adaptive_manager.save_learning_data()
                
                return True
            
            return False
            
        except Exception as e:
            print(f"[LAYER3] 长期优化失败: {e}")
            return False