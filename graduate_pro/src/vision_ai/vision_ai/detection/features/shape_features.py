# features/shape_features.py - 增强版本
import cv2
import numpy as np
from typing import Dict, Optional, Tuple, List
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    print("[WARNING] Open3D not available, some 3D features will be disabled")

try:
    from sklearn.neighbors import NearestNeighbors
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("[WARNING] scikit-learn not available, some features will be disabled")

class EnhancedShapeFeatureExtractor:
    def __init__(self, camera_intrinsics: Optional[Dict] = None):
        self.camera_intrinsics = camera_intrinsics or {
            'fx': 912.694580078125,
            'fy': 910.309814453125, 
            'cx': 624.051574707031,
            'cy': 320.749542236328
        }
    
    def extract_all_features(self, mask: np.ndarray, 
                           depth_data: Optional[np.ndarray] = None,
                           waypoint_data: Optional[Dict] = None) -> Dict:
        features = {}
        
        features.update(self._extract_2d_shape_features(mask))
        
        if depth_data is not None:
            pointcloud_features = self._extract_pointcloud_features(mask, depth_data)
            features.update(pointcloud_features)
            
            # 3D几何特征
            geometric_features = self._extract_3d_geometric_features(mask, depth_data)
            features.update(geometric_features)
        
        return features
    
    def _extract_3d_geometric_features(self, mask: np.ndarray, depth_data: np.ndarray) -> Dict:
        """提取3D几何特征"""
        features = {}
        
        try:
            # 生成点云
            points_3d = self._mask_to_pointcloud(mask, depth_data)
            
            if len(points_3d) < 10:
                return {'geometric_error': 'insufficient_points'}
            
            # 1. 3D包围盒特征
            min_coords = np.min(points_3d, axis=0)
            max_coords = np.max(points_3d, axis=0)
            bbox_dimensions = max_coords - min_coords
            bbox_volume = np.prod(bbox_dimensions)
            
            features['bbox_dimensions'] = bbox_dimensions.tolist()
            features['bbox_volume'] = float(bbox_volume)
            
            # 2. 点云分布特征
            centroid = np.mean(points_3d, axis=0)
            distances = np.linalg.norm(points_3d - centroid, axis=1)
            
            features['centroid_3d'] = centroid.tolist()
            features['point_spread'] = {
                'mean_distance': float(np.mean(distances)),
                'std_distance': float(np.std(distances)),
                'max_distance': float(np.max(distances))
            }
            
            # 3. 表面法向量估计
            if len(points_3d) >= 20:
                surface_normals = self._estimate_surface_normals(points_3d)
                features['surface_normal_stats'] = surface_normals
            
            # 4. 点云密度特征
            density_stats = self._compute_local_density_stats(points_3d)
            features.update(density_stats)
            
        except Exception as e:
            features['geometric_error'] = str(e)
        
        return features
    
    def _estimate_surface_normals(self, points_3d: np.ndarray) -> Dict:
        """估计表面法向量"""
        try:
            if not SKLEARN_AVAILABLE:
                return {'normal_error': 'sklearn_not_available'}
                
            # 使用简单的PCA方法估计法向量
            # 对于每个点，找到邻近点并计算局部法向量
            
            # 找到每个点的k个最近邻
            k = min(10, len(points_3d) - 1)
            nbrs = NearestNeighbors(n_neighbors=k).fit(points_3d)
            
            normals = []
            for i in range(0, len(points_3d), 5):  # 采样计算以提高效率
                _, indices = nbrs.kneighbors([points_3d[i]])
                neighbors = points_3d[indices[0]]
                
                # 对邻域进行PCA
                centered = neighbors - np.mean(neighbors, axis=0)
                cov_matrix = np.cov(centered.T)
                eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
                
                # 最小特征值对应的特征向量就是法向量
                normal = eigenvectors[:, 0]  # 最小特征值的特征向量
                normals.append(normal)
            
            normals = np.array(normals)
            
            # 统计法向量的分布
            normal_variance = np.var(normals, axis=0)
            
            return {
                'normal_variance': normal_variance.tolist(),
                'surface_complexity': float(np.mean(normal_variance)),
                'sample_count': len(normals)
            }
            
        except Exception:
            return {'normal_error': 'computation_failed'}
    
    def _compute_local_density_stats(self, points_3d: np.ndarray) -> Dict:
        """计算局部密度统计"""
        try:
            if not SKLEARN_AVAILABLE or len(points_3d) < 10:
                return {'density_error': 'insufficient_points_or_sklearn'}
            
            # 计算每个点到其k个最近邻的平均距离
            k = min(8, len(points_3d) - 1)
            nbrs = NearestNeighbors(n_neighbors=k).fit(points_3d)
            distances, _ = nbrs.kneighbors(points_3d)
            
            # 计算局部密度（距离的倒数）
            mean_distances = np.mean(distances[:, 1:], axis=1)  # 排除自己（距离=0）
            local_densities = 1.0 / (mean_distances + 1e-8)
            
            return {
                'density_stats': {
                    'mean_density': float(np.mean(local_densities)),
                    'std_density': float(np.std(local_densities)),
                    'min_density': float(np.min(local_densities)),
                    'max_density': float(np.max(local_densities))
                },
                'density_uniformity': float(np.std(local_densities) / (np.mean(local_densities) + 1e-8))
            }
            
        except Exception:
            return {'density_error': 'computation_failed'}
    

    def _extract_2d_shape_features(self, mask: np.ndarray) -> Dict:
        """提取2D形状特征（保留作为补充）"""
        features = {}
        
        if mask.dtype != bool:
            mask_bool = mask > 0.5
        else:
            mask_bool = mask
            
        mask_uint8 = mask_bool.astype(np.uint8) * 255
        
        # 形态学处理
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask_smooth = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
        mask_smooth = cv2.morphologyEx(mask_smooth, cv2.MORPH_OPEN, kernel)
        
        # Hu矩（保留）
        moments = cv2.moments(mask_smooth)
        if moments['m00'] > 0:
            hu_moments = cv2.HuMoments(moments).flatten()
            hu_log = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-8)
            features['hu_moments'] = hu_log.tolist()
        else:
            features['hu_moments'] = [0.0] * 7
        
        # 轮廓特征
        contours, _ = cv2.findContours(mask_smooth, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            features.update(self._extract_contour_features(largest_contour))
            
            # 🆕 添加凸包面积计算
            try:
                convex_hull = cv2.convexHull(largest_contour)
                convex_area = cv2.contourArea(convex_hull)
                features['convex_area'] = float(convex_area)
            except Exception as e:
                print(f"[SHAPE_FEATURES] Convex hull area calculation failed: {e}")
                features['convex_area'] = 0.0
        else:
            # 🆕 如果没有找到轮廓，确保 convex_area 存在
            features['convex_area'] = 0.0
        
        return features
    
    def _extract_pointcloud_features(self, mask: np.ndarray, depth_data: np.ndarray) -> Dict:
        features = {}
        
        try:
            # 生成点云
            points_3d = self._mask_to_pointcloud(mask, depth_data)
            
            if len(points_3d) < 10:
                return {'pointcloud_error': 'insufficient_points'}
            
            # 1. FPFH特征（快速点特征直方图）
            fpfh_features = self._compute_fpfh_features(points_3d)
            features['fpfh'] = fpfh_features
            
            # 2. 几何描述符
            geometric_desc = self._compute_geometric_descriptors(points_3d)
            features.update(geometric_desc)
            
            # 3. 3D形状上下文
            shape_context_3d = self._compute_3d_shape_context(points_3d)
            features['shape_context_3d'] = shape_context_3d
            
            # 4. 点云密度特征
            density_features = self._compute_density_features(points_3d)
            features.update(density_features)
            
        except Exception as e:
            features['pointcloud_error'] = str(e)
        
        return features
    
    def _mask_to_pointcloud(self, mask: np.ndarray, depth_data: np.ndarray) -> np.ndarray:
        """将mask区域转换为3D点云"""
        mask_indices = np.where(mask > 0)
        points_3d = []
        
        fx, fy = self.camera_intrinsics['fx'], self.camera_intrinsics['fy']
        cx, cy = self.camera_intrinsics['cx'], self.camera_intrinsics['cy']
        
        # 每5个像素采样一次，提高效率
        for i in range(0, len(mask_indices[0]), 5):
            y, x = mask_indices[0][i], mask_indices[1][i]
            depth_val = depth_data[y, x] / 1000.0  # 转换为米
            
            if depth_val > 0.01:  # 有效深度
                # 转换为3D坐标
                z = depth_val
                x_3d = (x - cx) * z / fx
                y_3d = (y - cy) * z / fy
                points_3d.append([x_3d, y_3d, z])
        
        return np.array(points_3d)
    
    def _compute_fpfh_features(self, points_3d: np.ndarray, radius: float = 0.05) -> List[float]:
        """计算FPFH特征"""
        try:
            if not OPEN3D_AVAILABLE or len(points_3d) < 10:
                return [0.0] * 33  # FPFH特征维度
            
            # 创建Open3D点云
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_3d)
            
            # 估计法向量
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30)
            )
            
            # 计算FPFH特征
            fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                pcd,
                o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 5, max_nn=100)
            )
            
            # 返回特征的平均值和标准差
            fpfh_data = np.asarray(fpfh.data)
            if fpfh_data.size > 0:
                mean_features = np.mean(fpfh_data, axis=1)
                return mean_features.tolist()
            else:
                return [0.0] * 33
                
        except Exception:
            return [0.0] * 33
    
    def _compute_geometric_descriptors(self, points_3d: np.ndarray) -> Dict:
        """计算3D几何描述符"""
        try:
            # 1. 3D包围盒特征
            min_coords = np.min(points_3d, axis=0)
            max_coords = np.max(points_3d, axis=0)
            bbox_size = max_coords - min_coords
            
            # 2. 主成分分析
            pca_features = self._compute_pca_features(points_3d)
            
            # 3. 点云散布特征
            centroid = np.mean(points_3d, axis=0)
            distances = np.linalg.norm(points_3d - centroid, axis=1)
            
            return {
                'bbox_dimensions': bbox_size.tolist(),
                'bbox_volume': float(np.prod(bbox_size)),
                'pca_features': pca_features,
                'point_spread': {
                    'mean_distance': float(np.mean(distances)),
                    'std_distance': float(np.std(distances)),
                    'max_distance': float(np.max(distances))
                }
            }
        except Exception:
            return {'geometric_error': 'computation_failed'}
    
    def _compute_pca_features(self, points_3d: np.ndarray) -> Dict:
        """计算PCA特征"""
        try:
            centered_points = points_3d - np.mean(points_3d, axis=0)
            covariance = np.cov(centered_points.T)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            
            # 按特征值排序
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            # 计算形状特征
            e1, e2, e3 = eigenvalues + 1e-8  # 避免除零
            
            return {
                'eigenvalues': eigenvalues.tolist(),
                'linearity': float((e1 - e2) / e1),
                'planarity': float((e2 - e3) / e1), 
                'sphericity': float(e3 / e1),
                'anisotropy': float((e1 - e3) / e1)
            }
        except Exception:
            return {'pca_error': 'computation_failed'}
    
    def _compute_3d_shape_context(self, points_3d: np.ndarray, bins_r: int = 5, bins_theta: int = 8, bins_phi: int = 6) -> List[float]:
        """计算3D形状上下文"""
        try:
            if len(points_3d) < 2:
                return [0.0] * (bins_r * bins_theta * bins_phi)
            
            centroid = np.mean(points_3d, axis=0)
            relative_points = points_3d - centroid
            
            # 转换为球坐标
            r = np.linalg.norm(relative_points, axis=1)
            theta = np.arccos(np.clip(relative_points[:, 2] / (r + 1e-8), -1, 1))
            phi = np.arctan2(relative_points[:, 1], relative_points[:, 0])
            
            # 归一化
            r_max = np.max(r) + 1e-8
            r_norm = r / r_max
            theta_norm = theta / np.pi
            phi_norm = (phi + np.pi) / (2 * np.pi)
            
            # 创建直方图
            hist, _ = np.histogramdd(
                np.column_stack([r_norm, theta_norm, phi_norm]),
                bins=[bins_r, bins_theta, bins_phi],
                range=[[0, 1], [0, 1], [0, 1]]
            )
            
            # 归一化直方图
            hist_flat = hist.flatten()
            hist_norm = hist_flat / (np.sum(hist_flat) + 1e-8)
            
            return hist_norm.tolist()
            
        except Exception:
            return [0.0] * (bins_r * bins_theta * bins_phi)
    
    def _compute_density_features(self, points_3d: np.ndarray) -> Dict:
        """计算点云密度特征"""
        try:
            if not SKLEARN_AVAILABLE or len(points_3d) < 3:
                return {'density_error': 'insufficient_points_or_sklearn'}
            
            # 计算局部密度
            nbrs = NearestNeighbors(n_neighbors=min(10, len(points_3d))).fit(points_3d)
            distances, _ = nbrs.kneighbors(points_3d)
            
            # 密度特征
            local_densities = 1.0 / (np.mean(distances[:, 1:], axis=1) + 1e-8)
            
            return {
                'mean_density': float(np.mean(local_densities)),
                'std_density': float(np.std(local_densities)),
                'density_variation': float(np.std(local_densities) / (np.mean(local_densities) + 1e-8))
            }
            
        except Exception:
            return {'density_error': 'computation_failed'}
    
    def _extract_contour_features(self, contour: np.ndarray) -> Dict:
        """提取轮廓特征（保留作为2D补充）"""
        features = {}
        
        try:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            if area > 0 and perimeter > 0:
                # 基本几何特征
                features['area'] = float(area)
                features['perimeter'] = float(perimeter)
                features['circularity'] = float(4 * np.pi * area / (perimeter * perimeter))
                
                # 凸包特征
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                features['solidity'] = float(area / hull_area) if hull_area > 0 else 0.0
                
                # 最小外接矩形
                rect = cv2.minAreaRect(contour)
                width, height = rect[1]
                features['aspect_ratio'] = float(max(width, height) / (min(width, height) + 1e-8))
                features['extent'] = float(area / (width * height)) if width * height > 0 else 0.0
        
        except Exception:
            features['contour_error'] = 'extraction_failed'
        
        return features

# 特征质量评估器
class FeatureQualityAssessor:
    """特征质量评估器 - 修复版本"""
    
    @staticmethod
    def assess_feature_quality(features: Dict) -> float:
        """评估特征质量，返回0-100的分数"""
        quality_score = 0.0
        max_score = 0.0
        
        # 评估几何特征质量
        if 'geometric' in features:
            geometric_features = features['geometric']
            
            geometric_quality = 0.0
            geometric_max = 0.0
            
            # 评估FPFH特征
            if 'fpfh' in geometric_features and isinstance(geometric_features['fpfh'], list):
                fpfh_data = geometric_features['fpfh']
                if len(fpfh_data) > 0:
                    non_zero_count = sum(1 for x in fpfh_data if abs(x) > 1e-6)
                    fpfh_quality = non_zero_count / len(fpfh_data) * 100
                    geometric_quality += fpfh_quality * 0.4
                geometric_max += 40
            
            # 评估PCA特征
            if 'pca_features' in geometric_features:
                pca_data = geometric_features['pca_features']
                if isinstance(pca_data, dict) and 'eigenvalues' in pca_data:
                    eigenvals = pca_data['eigenvalues']
                    if isinstance(eigenvals, list) and len(eigenvals) == 3 and eigenvals[0] > 1e-6:
                        # 计算特征值的分布均匀性
                        e1, e2, e3 = eigenvals[0], eigenvals[1], eigenvals[2]
                        balance = (e2 / e1 + e3 / e1) / 2 * 100  # 归一化到0-100
                        geometric_quality += balance * 0.3
                geometric_max += 30
            
            # 评估包围盒特征
            if 'bbox_dimensions' in geometric_features:
                bbox_dims = geometric_features['bbox_dimensions']
                if isinstance(bbox_dims, list) and len(bbox_dims) == 3:
                    # 检查包围盒是否合理
                    if all(d > 0.001 for d in bbox_dims):  # 至少1mm
                        bbox_quality = min(100, sum(bbox_dims) * 1000)  # 转换为mm并限制
                        geometric_quality += bbox_quality * 0.2
                geometric_max += 20
            
            # 评估3D形状上下文
            if 'shape_context_3d' in geometric_features:
                context_data = geometric_features['shape_context_3d']
                if isinstance(context_data, list) and len(context_data) > 0:
                    non_zero_ratio = sum(1 for x in context_data if abs(x) > 1e-6) / len(context_data)
                    context_quality = non_zero_ratio * 100
                    geometric_quality += context_quality * 0.1
                geometric_max += 10
            
            if geometric_max > 0:
                quality_score += geometric_quality
                max_score += geometric_max
        
        # 评估外观特征质量
        if 'appearance' in features:
            appearance_features = features['appearance']
            appearance_quality = 0.0
            appearance_max = 20
            
            # 评估颜色直方图
            if 'histogram' in appearance_features:
                hist_data = appearance_features['histogram']
                if isinstance(hist_data, list) and len(hist_data) > 0:
                    non_zero_ratio = sum(1 for x in hist_data if abs(x) > 1e-6) / len(hist_data)
                    hist_quality = non_zero_ratio * 100
                    appearance_quality += hist_quality * 0.7
            
            # 评估颜色名称
            if 'color_name' in appearance_features:
                color_name = appearance_features['color_name']
                if color_name and color_name != 'unknown':
                    appearance_quality += 30  # 有效颜色名称给30分
            
            quality_score += appearance_quality
            max_score += appearance_max
        
        # 评估形状特征质量
        if 'shape' in features:
            shape_features = features['shape']
            shape_quality = 0.0
            shape_max = 20
            
            # 评估Hu矩
            if 'hu_moments' in shape_features:
                hu_data = shape_features['hu_moments']
                if isinstance(hu_data, list) and len(hu_data) == 7:
                    valid_hu = sum(1 for x in hu_data if not (np.isnan(x) or np.isinf(x)))
                    hu_quality = (valid_hu / 7) * 100
                    shape_quality += hu_quality * 0.5
            
            # 评估轮廓特征
            if 'area' in shape_features:
                area = shape_features['area']
                if area > 0:
                    shape_quality += 50  # 有效面积给50分
            
            quality_score += shape_quality
            max_score += shape_max
        
        # 评估空间特征质量
        if 'spatial' in features:
            spatial_features = features['spatial']
            spatial_quality = 0.0
            spatial_max = 10
            
            # 检查3D坐标
            if 'world_coordinates' in spatial_features:
                coords = spatial_features['world_coordinates']
                if isinstance(coords, (list, tuple)) and len(coords) == 3:
                    if all(isinstance(c, (int, float)) and not np.isnan(c) for c in coords):
                        spatial_quality += 100  # 有效3D坐标给满分
            
            quality_score += spatial_quality
            max_score += spatial_max
        
        # 计算最终质量分数
        final_quality = (quality_score / max_score * 100) if max_score > 0 else 0.0
        # 确保返回合理的分数
        return max(0.0, min(100.0, final_quality))