# features/similarity_calculator.py
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.spatial.distance import cosine, euclidean
from scipy.stats import wasserstein_distance
import cv2
import math
class FeatureSimilarityCalculator:
    """特征相似度计算器 - 用于Track模块的特征匹配"""
    
    def __init__(self, feature_weights: Optional[Dict[str, float]] = None):
        """
        初始化相似度计算器
        
        Args:
            feature_weights: 各类特征的权重配置
        """
        self.feature_weights = feature_weights or {
            'geometric': 0.4,     # 3D几何特征权重最高
            'appearance': 0.2,    # 颜色特征权重较低
            'shape': 0.3,         # 2D形状特征中等权重
            'spatial': 0.1        # 空间上下文辅助权重
        }
        
        # 各特征类型的相似度计算方法
        self.similarity_methods = {
            'geometric': self._calculate_geometric_similarity,
            'appearance': self._calculate_appearance_similarity,
            'shape': self._calculate_shape_similarity,
            'spatial': self._calculate_spatial_similarity
        }
    
    def calculate_overall_similarity(self, ref_features: Dict, candidate_features: Dict) -> Dict:
        """
        计算整体相似度
        
        Args:
            ref_features: 参考特征
            candidate_features: 候选特征
            
        Returns:
            similarity_result: 包含各项相似度和最终分数的字典
        """
        similarities = {}
        valid_weights_sum = 0.0
        
        # 计算各类特征的相似度
        for feature_type, weight in self.feature_weights.items():
            if feature_type in ref_features and feature_type in candidate_features:
                try:
                    similarity_method = self.similarity_methods[feature_type]
                    similarity = similarity_method(
                        ref_features[feature_type], 
                        candidate_features[feature_type]
                    )
                    similarities[feature_type] = similarity
                    valid_weights_sum += weight
                except Exception as e:
                    print(f"[SIMILARITY] {feature_type}Similarity calculation failed: {e}")
                    similarities[feature_type] = 0.0
            else:
                similarities[feature_type] = 0.0
        
        # 计算加权总相似度
        if valid_weights_sum > 0:
            weighted_similarity = sum(
                similarities[ft] * self.feature_weights[ft] 
                for ft in similarities.keys()
            ) / valid_weights_sum
        else:
            weighted_similarity = 0.0
        
        return {
            'overall_similarity': weighted_similarity,
            'feature_similarities': similarities,
            'confidence': self._calculate_confidence(similarities),
            'valid_features': list(similarities.keys())
        }
    
    def _calculate_geometric_similarity(self, ref_geometric: Dict, candidate_geometric: Dict) -> float:
        """计算3D几何特征相似度"""
        similarities = []
        
        # 1. FPFH特征相似度
        if 'fpfh' in ref_geometric and 'fpfh' in candidate_geometric:
            fpfh_sim = self._calculate_fpfh_similarity(
                ref_geometric['fpfh'], candidate_geometric['fpfh']
            )
            similarities.append(fpfh_sim * 0.4)  # FPFH权重40%
        
        # 2. PCA特征相似度
        if 'pca_features' in ref_geometric and 'pca_features' in candidate_geometric:
            pca_sim = self._calculate_pca_similarity(
                ref_geometric['pca_features'], candidate_geometric['pca_features']
            )
            similarities.append(pca_sim * 0.3)  # PCA权重30%
        
        # 3. 包围盒相似度
        if 'bbox_dimensions' in ref_geometric and 'bbox_dimensions' in candidate_geometric:
            bbox_sim = self._calculate_bbox_similarity(
                ref_geometric['bbox_dimensions'], candidate_geometric['bbox_dimensions']
            )
            similarities.append(bbox_sim * 0.2)  # 包围盒权重20%
        
        # 4. 3D形状上下文相似度
        if 'shape_context_3d' in ref_geometric and 'shape_context_3d' in candidate_geometric:
            context_sim = self._calculate_histogram_similarity(
                ref_geometric['shape_context_3d'], candidate_geometric['shape_context_3d']
            )
            similarities.append(context_sim * 0.1)  # 形状上下文权重10%
        
        return np.mean(similarities) if similarities else 0.0
    
    def _calculate_fpfh_similarity(self, ref_fpfh: List[float], candidate_fpfh: List[float]) -> float:
        """计算FPFH特征相似度"""
        try:
            if len(ref_fpfh) != len(candidate_fpfh):
                return 0.0
            
            ref_array = np.array(ref_fpfh)
            candidate_array = np.array(candidate_fpfh)
            
            # 使用余弦相似度，对幅度变化更鲁棒
            similarity = 1.0 - cosine(ref_array, candidate_array)
            return max(0.0, similarity)  # 确保非负
            
        except Exception:
            return 0.0
    
    def _calculate_pca_similarity(self, ref_pca: Dict, candidate_pca: Dict) -> float:
        """计算PCA特征相似度"""
        try:
            similarities = []
            
            # 形状特征相似度
            shape_features = ['linearity', 'planarity', 'sphericity', 'anisotropy']
            for feature in shape_features:
                if feature in ref_pca and feature in candidate_pca:
                    # 使用相对差异计算相似度
                    diff = abs(ref_pca[feature] - candidate_pca[feature])
                    similarity = max(0.0, 1.0 - diff)
                    similarities.append(similarity)
            
            return np.mean(similarities) if similarities else 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_bbox_similarity(self, ref_bbox: List[float], candidate_bbox: List[float]) -> float:
        """计算3D包围盒相似度"""
        try:
            if len(ref_bbox) != len(candidate_bbox) or len(ref_bbox) != 3:
                return 0.0
            
            ref_array = np.array(ref_bbox)
            candidate_array = np.array(candidate_bbox)
            
            # 计算尺寸比例相似度（对绝对尺寸变化鲁棒）
            ref_ratios = ref_array / (np.sum(ref_array) + 1e-8)
            candidate_ratios = candidate_array / (np.sum(candidate_array) + 1e-8)
            
            # 使用L2距离的相似度
            distance = np.linalg.norm(ref_ratios - candidate_ratios)
            similarity = max(0.0, 1.0 - distance)
            
            return similarity
            
        except Exception:
            return 0.0
    
    def _calculate_appearance_similarity(self, ref_appearance: Dict, candidate_appearance: Dict) -> float:
        """计算外观特征相似度"""
        similarities = []
        
        # 1. 颜色直方图相似度
        if 'histogram' in ref_appearance and 'histogram' in candidate_appearance:
            hist_sim = self._calculate_histogram_similarity(
                ref_appearance['histogram'], candidate_appearance['histogram']
            )
            similarities.append(hist_sim * 0.7)  # 直方图权重70%
        
        # 2. 颜色名称匹配
        if 'color_name' in ref_appearance and 'color_name' in candidate_appearance:
            color_match = 1.0 if ref_appearance['color_name'] == candidate_appearance['color_name'] else 0.0
            similarities.append(color_match * 0.3)  # 颜色名称权重30%
        
        return np.mean(similarities) if similarities else 0.0
    
    def _calculate_histogram_similarity(self, ref_hist: List[float], candidate_hist: List[float]) -> float:
        """计算直方图相似度"""
        try:
            if len(ref_hist) != len(candidate_hist):
                return 0.0
            
            ref_array = np.array(ref_hist)
            candidate_array = np.array(candidate_hist)
            
            # 使用多种相似度度量的组合
            # 1. 巴氏系数
            bc_similarity = np.sum(np.sqrt(ref_array * candidate_array))
            
            # 2. 相关系数
            correlation = np.corrcoef(ref_array, candidate_array)[0, 1]
            correlation = 0.0 if np.isnan(correlation) else correlation
            
            # 3. 卡方距离的相似度
            chi_square = np.sum((ref_array - candidate_array) ** 2 / (ref_array + candidate_array + 1e-8))
            chi_similarity = max(0.0, 1.0 - chi_square / len(ref_hist))
            
            # 组合相似度
            combined_similarity = (bc_similarity * 0.5 + 
                                 max(0.0, correlation) * 0.3 + 
                                 chi_similarity * 0.2)
            
            return min(1.0, combined_similarity)
            
        except Exception:
            return 0.0
    
    def _calculate_shape_similarity(self, ref_shape: Dict, candidate_shape: Dict) -> float:
        """计算2D形状特征相似度"""
        similarities = []
        
        # 1. Hu矩相似度
        if 'hu_moments' in ref_shape and 'hu_moments' in candidate_shape:
            hu_sim = self._calculate_hu_moments_similarity(
                ref_shape['hu_moments'], candidate_shape['hu_moments']
            )
            similarities.append(hu_sim * 0.4)  # Hu矩权重40%
        
        # 2. 鲁棒Hu矩相似度（如果有）
        if 'hu_moments_robust' in ref_shape and 'hu_moments_robust' in candidate_shape:
            robust_hu_sim = self._calculate_hu_moments_similarity(
                ref_shape['hu_moments_robust'], candidate_shape['hu_moments_robust']
            )
            similarities.append(robust_hu_sim * 0.3)  # 鲁棒Hu矩权重30%
        
        # 3. 形状描述符相似度
        if 'shape_descriptors' in ref_shape and 'shape_descriptors' in candidate_shape:
            desc_sim = self._calculate_shape_descriptors_similarity(
                ref_shape['shape_descriptors'], candidate_shape['shape_descriptors']
            )
            similarities.append(desc_sim * 0.2)  # 形状描述符权重20%
        
        # 4. 傅里叶描述子相似度
        if 'fourier_descriptors' in ref_shape and 'fourier_descriptors' in candidate_shape:
            fourier_sim = self._calculate_fourier_similarity(
                ref_shape['fourier_descriptors'], candidate_shape['fourier_descriptors']
            )
            similarities.append(fourier_sim * 0.1)  # 傅里叶描述子权重10%
        
        return np.mean(similarities) if similarities else 0.0
    
    def _calculate_hu_moments_similarity(self, ref_hu: List[float], candidate_hu: List[float]) -> float:
        """计算Hu矩相似度"""
        try:
            if len(ref_hu) != len(candidate_hu):
                return 0.0
            
            ref_array = np.array(ref_hu)
            candidate_array = np.array(candidate_hu)
            
            # 使用对数空间的欧几里得距离
            distances = np.abs(ref_array - candidate_array)
            # 将距离转换为相似度（距离越小，相似度越高）
            similarities = np.exp(-distances)
            
            return np.mean(similarities)
            
        except Exception:
            return 0.0
    
    def _calculate_shape_descriptors_similarity(self, ref_desc: List[float], candidate_desc: List[float]) -> float:
        """计算形状描述符相似度"""
        try:
            if len(ref_desc) != len(candidate_desc):
                return 0.0
            
            ref_array = np.array(ref_desc)
            candidate_array = np.array(candidate_desc)
            
            # 计算各个描述符的相似度
            similarities = []
            for i in range(len(ref_array)):
                if ref_array[i] > 0 and candidate_array[i] > 0:
                    # 使用比例相似度
                    ratio = min(ref_array[i], candidate_array[i]) / max(ref_array[i], candidate_array[i])
                    similarities.append(ratio)
                elif ref_array[i] == 0 and candidate_array[i] == 0:
                    similarities.append(1.0)
                else:
                    similarities.append(0.0)
            
            return np.mean(similarities) if similarities else 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_fourier_similarity(self, ref_fourier: List[float], candidate_fourier: List[float]) -> float:
        """计算傅里叶描述子相似度"""
        try:
            if len(ref_fourier) != len(candidate_fourier):
                return 0.0
            
            ref_array = np.array(ref_fourier)
            candidate_array = np.array(candidate_fourier)
            
            # 使用归一化的欧几里得距离
            if np.sum(ref_array) > 0 and np.sum(candidate_array) > 0:
                ref_normalized = ref_array / np.sum(ref_array)
                candidate_normalized = candidate_array / np.sum(candidate_array)
                
                distance = np.linalg.norm(ref_normalized - candidate_normalized)
                similarity = max(0.0, 1.0 - distance)
                return similarity
            else:
                return 0.0
                
        except Exception:
            return 0.0
    
    def _calculate_spatial_similarity(self, ref_spatial: Dict, candidate_spatial: Dict) -> float:
        """计算空间特征相似度"""
        similarities = []
        
        # 1. 相对位置相似度
        if 'region_position' in ref_spatial and 'region_position' in candidate_spatial:
            position_match = 1.0 if ref_spatial['region_position'] == candidate_spatial['region_position'] else 0.0
            similarities.append(position_match * 0.4)
        
        # 2. 尺寸相似度
        if 'mask_area_pixels' in ref_spatial and 'mask_area_pixels' in candidate_spatial:
            area_ratio = min(ref_spatial['mask_area_pixels'], candidate_spatial['mask_area_pixels']) / \
                        max(ref_spatial['mask_area_pixels'], candidate_spatial['mask_area_pixels'])
            similarities.append(area_ratio * 0.3)
        
        # 3. 距离相似度
        if 'distance_to_camera' in ref_spatial and 'distance_to_camera' in candidate_spatial:
            ref_dist = ref_spatial['distance_to_camera']
            candidate_dist = candidate_spatial['distance_to_camera']
            if ref_dist > 0 and candidate_dist > 0:
                dist_ratio = min(ref_dist, candidate_dist) / max(ref_dist, candidate_dist)
                similarities.append(dist_ratio * 0.3)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _calculate_confidence(self, similarities: Dict[str, float]) -> float:
        """计算匹配置信度"""
        # 基于相似度分布计算置信度
        sim_values = list(similarities.values())
        if not sim_values:
            return 0.0
        
        mean_sim = np.mean(sim_values)
        std_sim = np.std(sim_values)
        
        # 高相似度且低方差 = 高置信度
        confidence = mean_sim * (1.0 - min(std_sim, 0.5))
        
        return max(0.0, min(1.0, confidence))
    
    def find_best_match(self, target_features: Dict, candidate_list: List[Dict]) -> Optional[Dict]:
        """
        在候选列表中找到最佳匹配
        
        Args:
            target_features: 目标特征
            candidate_list: 候选特征列表
            
        Returns:
            best_match: 最佳匹配结果，包含候选索引和相似度信息
        """
        if not candidate_list:
            return None
        
        best_match = None
        best_similarity = 0.0
        
        for i, candidate in enumerate(candidate_list):
            if 'features' in candidate:
                similarity_result = self.calculate_overall_similarity(
                    target_features, candidate['features']
                )
                
                overall_sim = similarity_result['overall_similarity']
                
                if overall_sim > best_similarity:
                    best_similarity = overall_sim
                    best_match = {
                        'candidate_index': i,
                        'candidate_data': candidate,
                        'similarity_result': similarity_result
                    }
        
        return best_match
    
    def update_weights(self, new_weights: Dict[str, float]):
        """更新特征权重"""
        self.feature_weights.update(new_weights)
        
        # 归一化权重
        total_weight = sum(self.feature_weights.values())
        if total_weight > 0:
            self.feature_weights = {
                k: v / total_weight for k, v in self.feature_weights.items()
            }

    def calculate_comprehensive_similarity(self, reference_features: Dict, 
                                         candidate_features: Dict, 
                                         waypoint_data: Dict = None) -> Dict:
        """
        计算综合相似度 - 新增方法
        
        Args:
            reference_features: 参考特征
            candidate_features: 候选特征
            waypoint_data: waypoint数据（可选）
        
        Returns:
            result: 包含final_score和breakdown的结果字典
        """
        try:
            print(f"[SIMILARITY_CALC] Start comprehensive similarity calculation")
            
            breakdown = {}
            total_score = 0.0
            total_weight = 0.0
            
            # 🔧 几何特征相似度 (权重最高)
            geometric_weight = 0.4
            if 'geometric' in reference_features and 'geometric' in candidate_features:
                try:
                    geometric_sim = self.calculate_geometric_similarity(
                        reference_features['geometric'], 
                        candidate_features['geometric']
                    )
                    breakdown['geometric'] = geometric_sim
                    total_score += geometric_sim * geometric_weight
                    total_weight += geometric_weight
                    print(f"[SIMILARITY_CALC] Geometric similarity: {geometric_sim:.3f}")
                except Exception as e:
                    print(f"[SIMILARITY_CALC] Geometric similarity calculation failed: {e}")    
                    breakdown['geometric'] = 0.0
            else:
                print("[SIMILARITY_CALC] Missing geometric features")
                breakdown['geometric'] = 0.0
            
            # 🔧 形状特征相似度
            shape_weight = 0.3
            if 'shape' in reference_features and 'shape' in candidate_features:
                try:
                    shape_sim = self.calculate_shape_similarity(
                        reference_features['shape'], 
                        candidate_features['shape']
                    )
                    breakdown['shape'] = shape_sim
                    total_score += shape_sim * shape_weight
                    total_weight += shape_weight
                    print(f"[SIMILARITY_CALC] Shape similarity: {shape_sim:.3f}")
                except Exception as e:
                    print(f"[SIMILARITY_CALC] Shape similarity calculation failed: {e}")
                    breakdown['shape'] = 0.0
            else:
                print("[SIMILARITY_CALC] Missing shape features")
                breakdown['shape'] = 0.0
            appearance_weight = 0.2
            ref_appearance = reference_features.get('appearance', reference_features.get('color', {}))
            cand_appearance = candidate_features.get('appearance', candidate_features.get('color', {}))
            
            if ref_appearance and cand_appearance:
                try:
                    appearance_sim = self.calculate_appearance_similarity(
                        ref_appearance, cand_appearance
                    )
                    breakdown['appearance'] = appearance_sim
                    total_score += appearance_sim * appearance_weight
                    total_weight += appearance_weight
                    print(f"[SIMILARITY_CALC] Appearance similarity: {appearance_sim:.3f}")
                except Exception as e:
                    print(f"[SIMILARITY_CALC] Appearance similarity failed: {e}")
                    breakdown['appearance'] = 0.0
            else:
                print("[SIMILARITY_CALC] Missing Appearance features")
                breakdown['appearance'] = 0.0
            
            # 🔧 空间特征相似度
            spatial_weight = 0.1
            if 'spatial' in reference_features and 'spatial' in candidate_features:
                try:
                    spatial_sim = self.calculate_spatial_similarity(
                        reference_features['spatial'], 
                        candidate_features['spatial']
                    )
                    breakdown['spatial'] = spatial_sim
                    total_score += spatial_sim * spatial_weight
                    total_weight += spatial_weight
                    print(f"[SIMILARITY_CALC] spatial similarity: {spatial_sim:.3f}")
                except Exception as e:
                    print(f"[SIMILARITY_CALC] spatial similarity calculation failed: {e}")
                    breakdown['spatial'] = 0.0
            else:
                print("[SIMILARITY_CALC] Missing spatial features")
                breakdown['spatial'] = 0.0
            
            # 计算最终得分
            if total_weight > 0:
                final_score = total_score / total_weight
            else:
                final_score = 0.0
            
            result = {
                'final_score': final_score,
                'breakdown': breakdown,
                'total_weight': total_weight,
                'raw_score': total_score
            }
            
            print(f"[SIMILARITY_CALC] Comprehensive similarity calculation completed: {final_score:.3f}")
            
            return result
            
        except Exception as e:
            print(f"[SIMILARITY_CALC] Comprehensive similarity calculation failed: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                'final_score': 0.0,
                'breakdown': {'error': str(e)},
                'total_weight': 0.0,
                'raw_score': 0.0
            }

    # 🆕 确保基础相似度计算方法存在
    def calculate_geometric_similarity(self, ref_geometric: Dict, cand_geometric: Dict) -> float:
        try:
            if not ref_geometric or not cand_geometric:
                return 0.0
            
            similarities = []
            
            # 🔧 Hu矩相似度 (最重要的形状描述符)
            if 'hu_moments' in ref_geometric and 'hu_moments' in cand_geometric:
                hu_sim = self._calculate_hu_moments_similarity(
                    ref_geometric['hu_moments'], 
                    cand_geometric['hu_moments']
                )
                similarities.append(hu_sim * 0.4)  # 高权重
            
            # 🔧 面积比较
            if 'area' in ref_geometric and 'area' in cand_geometric:
                area_sim = self._calculate_ratio_similarity(
                    ref_geometric['area'], 
                    cand_geometric['area']
                )
                similarities.append(area_sim * 0.2)
            
            # 🔧 周长比较
            if 'perimeter' in ref_geometric and 'perimeter' in cand_geometric:
                perimeter_sim = self._calculate_ratio_similarity(
                    ref_geometric['perimeter'], 
                    cand_geometric['perimeter']
                )
                similarities.append(perimeter_sim * 0.1)
            
            # 🔧 圆形度比较
            if 'circularity' in ref_geometric and 'circularity' in cand_geometric:
                circ_sim = self._calculate_ratio_similarity(
                    ref_geometric['circularity'], 
                    cand_geometric['circularity']
                )
                similarities.append(circ_sim * 0.15)
            
            # 🔧 纵横比比较
            if 'aspect_ratio' in ref_geometric and 'aspect_ratio' in cand_geometric:
                aspect_sim = self._calculate_ratio_similarity(
                    ref_geometric['aspect_ratio'], 
                    cand_geometric['aspect_ratio']
                )
                similarities.append(aspect_sim * 0.15)
            
            # 计算加权平均
            if similarities:
                return sum(similarities) / len(similarities) * len(similarities)  # 归一化
            else:
                return 0.0
                
        except Exception as e:
            return 0.0

    def _calculate_hu_moments_similarity(self, hu1: List[float], hu2: List[float]) -> float:
        """计算Hu矩相似度"""
        try:
            if len(hu1) != len(hu2):
                return 0.0
            
            # 使用欧几里得距离的倒数作为相似度
            distance = 0.0
            for i in range(len(hu1)):
                diff = abs(hu1[i] - hu2[i])
                distance += diff * diff
            
            distance = math.sqrt(distance)
            
            # 转换为相似度 (0-1范围)
            similarity = 1.0 / (1.0 + distance)
            return similarity
            
        except Exception:
            return 0.0

    def _calculate_ratio_similarity(self, val1: float, val2: float, tolerance: float = 0.3) -> float:
        """计算比值相似度"""
        try:
            if val1 <= 0 or val2 <= 0:
                return 0.0
            
            ratio = min(val1, val2) / max(val1, val2)
            
            # 如果比值在容忍范围内，返回高相似度
            if ratio >= (1.0 - tolerance):
                return ratio
            else:
                return ratio * 0.5  # 降权处理
                
        except Exception:
            return 0.0

    def calculate_shape_similarity(self, ref_shape: Dict, cand_shape: Dict) -> float:
        """计算形状特征相似度 - 简化版"""
        try:
            if not ref_shape or not cand_shape:
                return 0.0
            
            similarities = []
            
            # 基本形状特征
            shape_features = ['circularity', 'solidity', 'aspect_ratio', 'extent']
            
            for feature in shape_features:
                if feature in ref_shape and feature in cand_shape:
                    sim = self._calculate_ratio_similarity(
                        ref_shape[feature], 
                        cand_shape[feature]
                    )
                    similarities.append(sim)
            
            return np.mean(similarities) if similarities else 0.0
            
        except Exception:
            return 0.0

    def calculate_appearance_similarity(self, ref_appearance: Dict, cand_appearance: Dict) -> float:
        """计算外观相似度 - 简化版"""
        try:
            if not ref_appearance or not cand_appearance:
                return 0.0
            
            # 颜色名称匹配
            ref_color = ref_appearance.get('color_name', '')
            cand_color = cand_appearance.get('color_name', '')
            
            if ref_color and cand_color:
                if ref_color == cand_color:
                    return 0.8  # 颜色匹配给高分
                else:
                    return 0.2  # 颜色不匹配给低分
            
            return 0.5  # 无法比较时给中等分
            
        except Exception:
            return 0.0

    def calculate_spatial_similarity(self, ref_spatial: Dict, cand_spatial: Dict) -> float:
        """计算空间相似度 - 简化版"""
        try:
            # 空间特征在追踪时权重较低，简单处理
            return 0.5  # 给一个中等分数
            
        except Exception:
            return 0.0
        
    def calculate_detailed_similarity_breakdown(self, reference_features: Dict, 
                                            candidate_features: Dict, 
                                            waypoint_data: Dict = None) -> Dict:
        try:
            
            detailed_breakdown = {}
            
            # 🆕 几何特征细分
            if 'geometric' in reference_features and 'geometric' in candidate_features:
                detailed_breakdown['geometric_features'] = self._analyze_geometric_features_detailed(
                    reference_features['geometric'], candidate_features['geometric']
                )
            
            # 🆕 外观特征细分
            ref_appearance = reference_features.get('appearance', reference_features.get('color', {}))
            cand_appearance = candidate_features.get('appearance', candidate_features.get('color', {}))
            if ref_appearance and cand_appearance:
                detailed_breakdown['appearance_features'] = self._analyze_appearance_features_detailed(
                    ref_appearance, cand_appearance
                )
            
            # 🆕 形状特征细分
            if 'shape' in reference_features and 'shape' in candidate_features:
                detailed_breakdown['shape_features'] = self._analyze_shape_features_detailed(
                    reference_features['shape'], candidate_features['shape']
                )
            
            # 🆕 空间特征细分
            if 'spatial' in reference_features and 'spatial' in candidate_features:
                detailed_breakdown['spatial_features'] = self._analyze_spatial_features_detailed(
                    reference_features['spatial'], candidate_features['spatial']
                )
            
            # 🆕 计算加权总分和贡献度
            final_result = self._calculate_weighted_final_score(detailed_breakdown)
            
            return {
                'detailed_breakdown': detailed_breakdown,
                'final_score': final_result['final_score'],
                'feature_contributions': final_result['contributions'],
                'feature_weights_used': self.feature_weights.copy()
            }
            
        except Exception as e:
            return {'detailed_breakdown': {}, 'final_score': 0.0, 'feature_contributions': {}}

    def _analyze_geometric_features_detailed(self, ref_geom: Dict, cand_geom: Dict) -> Dict:
        """分析几何特征的细粒度相似度"""
        detailed = {}
        
        # 具体几何特征匹配度
        geometric_features = [
            'area', 'perimeter', 'circularity', 'aspect_ratio', 
            'solidity', 'extent', 'convex_area'
        ]
        
        for feature in geometric_features:
            if feature in ref_geom and feature in cand_geom:
                detailed[feature] = self._calculate_ratio_similarity(
                    ref_geom[feature], cand_geom[feature]
                )
            else:
                detailed[feature] = 0.0
        
        # Hu矩详细分析
        if 'hu_moments' in ref_geom and 'hu_moments' in cand_geom:
            hu_ref, hu_cand = ref_geom['hu_moments'], cand_geom['hu_moments']
            for i, (h1, h2) in enumerate(zip(hu_ref, hu_cand)):
                detailed[f'hu_moment_{i+1}'] = max(0.0, 1.0 - abs(h1 - h2))
        
        return detailed

    def _analyze_appearance_features_detailed(self, ref_app: Dict, cand_app: Dict) -> Dict:
        """分析外观特征的细粒度相似度"""
        detailed = {}
        
        # 颜色统计特征
        color_stats = ['mean_r', 'mean_g', 'mean_b', 'std_r', 'std_g', 'std_b']
        for stat in color_stats:
            if stat in ref_app and stat in cand_app:
                detailed[stat] = self._calculate_ratio_similarity(
                    ref_app[stat], cand_app[stat], tolerance=0.4
                )
            else:
                detailed[stat] = 0.0
        
        # 颜色直方图分段比较
        if 'histogram' in ref_app and 'histogram' in cand_app:
            ref_hist, cand_hist = ref_app['histogram'], cand_app['histogram']
            if len(ref_hist) == len(cand_hist):
                # 分为RGB三段分析
                segment_size = len(ref_hist) // 3
                for i, color in enumerate(['r', 'g', 'b']):
                    start_idx = i * segment_size
                    end_idx = (i + 1) * segment_size
                    ref_segment = ref_hist[start_idx:end_idx]
                    cand_segment = cand_hist[start_idx:end_idx]
                    
                    detailed[f'histogram_{color}_channel'] = self._calculate_histogram_similarity(
                        ref_segment, cand_segment
                    )
        
        # 颜色名称匹配
        if 'color_name' in ref_app and 'color_name' in cand_app:
            detailed['color_name_match'] = 1.0 if ref_app['color_name'] == cand_app['color_name'] else 0.0
        
        return detailed

    def _analyze_shape_features_detailed(self, ref_shape: Dict, cand_shape: Dict) -> Dict:
        """分析形状特征的细粒度相似度"""
        detailed = {}
        
        # 基本形状描述符
        shape_descriptors = ['circularity', 'solidity', 'aspect_ratio', 'extent']
        for desc in shape_descriptors:
            if desc in ref_shape and desc in cand_shape:
                detailed[desc] = self._calculate_ratio_similarity(
                    ref_shape[desc], cand_shape[desc]
                )
            else:
                detailed[desc] = 0.0
        
        # Hu矩逐个分析
        if 'hu_moments' in ref_shape and 'hu_moments' in cand_shape:
            hu_ref, hu_cand = ref_shape['hu_moments'], cand_shape['hu_moments']
            for i, (h1, h2) in enumerate(zip(hu_ref, hu_cand)):
                detailed[f'hu_moment_{i+1}'] = max(0.0, 1.0 - abs(h1 - h2))
        
        return detailed

    def _analyze_spatial_features_detailed(self, ref_spatial: Dict, cand_spatial: Dict) -> Dict:
        """分析空间特征的细粒度相似度"""
        detailed = {}
        
        # 3D位置比较
        if 'world_coordinates' in ref_spatial and 'world_coordinates' in cand_spatial:
            ref_coord = ref_spatial['world_coordinates']
            cand_coord = cand_spatial['world_coordinates']
            
            if len(ref_coord) >= 3 and len(cand_coord) >= 3:
                for i, axis in enumerate(['x', 'y', 'z']):
                    distance = abs(ref_coord[i] - cand_coord[i])
                    detailed[f'position_{axis}_similarity'] = max(0.0, 1.0 - distance / 1000.0)  # 归一化到米
        
        # 距离相似度
        if 'distance_to_camera' in ref_spatial and 'distance_to_camera' in cand_spatial:
            detailed['camera_distance_similarity'] = self._calculate_ratio_similarity(
                ref_spatial['distance_to_camera'], cand_spatial['distance_to_camera']
            )
        
        # 面积相似度
        if 'mask_area_pixels' in ref_spatial and 'mask_area_pixels' in cand_spatial:
            detailed['mask_area_similarity'] = self._calculate_ratio_similarity(
                ref_spatial['mask_area_pixels'], cand_spatial['mask_area_pixels']
            )
        
        return detailed

    def _calculate_weighted_final_score(self, detailed_breakdown: Dict) -> Dict:
        """计算加权最终分数和贡献度"""
        try:
            contributions = {}
            total_score = 0.0
            total_weight = 0.0
            
            # 为每个特征类型计算贡献度
            for feature_type, weight in self.feature_weights.items():
                feature_key = f'{feature_type}_features'
                if feature_key in detailed_breakdown:
                    feature_scores = detailed_breakdown[feature_key]
                    if feature_scores:
                        # 计算该特征类型的平均分数
                        avg_score = np.mean(list(feature_scores.values()))
                        contribution = avg_score * weight
                        
                        contributions[feature_type] = {
                            'average_score': avg_score,
                            'weight': weight,
                            'contribution': contribution,
                            'individual_scores': feature_scores
                        }
                        
                        total_score += contribution
                        total_weight += weight
            
            final_score = total_score / total_weight if total_weight > 0 else 0.0
            
            return {
                'final_score': final_score,
                'contributions': contributions
            }
            
        except Exception as e:
            print(f"[WEIGHTED_SCORE] Failed to calculate weighted score: {e}")
            return {'final_score': 0.0, 'contributions': {}}