# detection/features/spatial_features.py
import numpy as np
import math
from typing import Dict, Tuple, Optional

class SpatialFeatureExtractor:
    def __init__(self, camera_intrinsics: Dict, scan_region_bounds: Optional[Dict] = None):

        self.camera_intrinsics = camera_intrinsics
        self.scan_region_bounds = scan_region_bounds
        
        # 空间区域定义（9宫格）
        self.spatial_regions = {
            'top-left': (0, 1/3, 0, 1/3),
            'top-center': (1/3, 2/3, 0, 1/3),
            'top-right': (2/3, 1, 0, 1/3),
            'center-left': (0, 1/3, 1/3, 2/3),
            'center': (1/3, 2/3, 1/3, 2/3),
            'center-right': (2/3, 1, 1/3, 2/3),
            'bottom-left': (0, 1/3, 2/3, 1),
            'bottom-center': (1/3, 2/3, 2/3, 1),
            'bottom-right': (2/3, 1, 2/3, 1)
        }
    
    def pixel_to_world_coordinates(self, pixel_x: float, pixel_y: float, 
                                 depth: float, camera_pose: Dict) -> Tuple[float, float, float]:
        # 像素坐标转相机坐标
        fx, fy = self.camera_intrinsics['fx'], self.camera_intrinsics['fy']
        cx, cy = self.camera_intrinsics['cx'], self.camera_intrinsics['cy']
        
        # 转换为米（假设深度以毫米为单位）
        depth_m = depth / 1000.0
        
        # 相机坐标系下的3D点
        x_cam = (pixel_x - cx) * depth_m / fx
        y_cam = (pixel_y - cy) * depth_m / fy
        z_cam = depth_m
        
        # 相机坐标转世界坐标（简化版，假设只有平移）
        # 在实际应用中，需要使用完整的旋转矩阵
        world_x = camera_pose['x'] + x_cam
        world_y = camera_pose['y'] + y_cam
        world_z = camera_pose['z'] + z_cam
        
        return world_x, world_y, world_z
    
    def compute_spatial_position(self, mask: np.ndarray, depth_image: np.ndarray, 
                               camera_pose: Dict) -> Dict:
        if np.sum(mask) == 0:
            return {
                'world_coordinates': (0, 0, 0),
                'normalized_coords': (0, 0),
                'region_position': 'unknown',
                'centroid_2d': (0, 0),
                'bounding_box_2d': (0, 0, 0, 0),
                'average_depth': 0,
                'depth_variance': 0
            }
        
        # 计算2D质心
        ys, xs = np.where(mask > 0)
        centroid_x = np.mean(xs)
        centroid_y = np.mean(ys)
        
        # 计算边界框
        x_min, x_max = np.min(xs), np.max(xs)
        y_min, y_max = np.min(ys), np.max(ys)
        bounding_box_2d = (x_min, y_min, x_max, y_max)
        
        # 计算深度统计
        mask_depths = depth_image[mask > 0]
        valid_depths = mask_depths[mask_depths > 0]  # 过滤无效深度
        
        if len(valid_depths) > 0:
            average_depth = np.mean(valid_depths)
            depth_variance = np.var(valid_depths)
        else:
            average_depth = 0
            depth_variance = 0
        
        # 计算3D世界坐标
        world_x, world_y, world_z = self.pixel_to_world_coordinates(
            centroid_x, centroid_y, average_depth, camera_pose
        )
        
        # 计算归一化坐标
        image_height, image_width = mask.shape
        norm_x = centroid_x / image_width
        norm_y = centroid_y / image_height
        
        # 确定区域位置
        region_position = self._determine_region_position(norm_x, norm_y)
        
        return {
            'world_coordinates': (world_x, world_y, world_z),
            'normalized_coords': (norm_x, norm_y),
            'region_position': region_position,
            'centroid_2d': (centroid_x, centroid_y),
            'bounding_box_2d': bounding_box_2d,
            'average_depth': average_depth,
            'depth_variance': depth_variance
        }
    
    def _determine_region_position(self, norm_x: float, norm_y: float) -> str:
        for region_name, (x_min, x_max, y_min, y_max) in self.spatial_regions.items():
            if x_min <= norm_x < x_max and y_min <= norm_y < y_max:
                return region_name
        
        return 'center'  # 默认返回center
    
    def compute_relative_positions(self, current_features: Dict, 
                                 all_objects_features: list) -> Dict:
        current_coords = current_features['world_coordinates']
        
        if not all_objects_features:
            return {
                'nearest_neighbor_distance': float('inf'),
                'average_distance_to_others': float('inf'),
                'relative_position_desc': 'isolated'
            }
        
        distances = []
        for other_features in all_objects_features:
            other_coords = other_features['world_coordinates']
            distance = self._compute_3d_distance(current_coords, other_coords)
            if distance > 0:  # 排除自己
                distances.append(distance)
        
        if not distances:
            return {
                'nearest_neighbor_distance': float('inf'),
                'average_distance_to_others': float('inf'),
                'relative_position_desc': 'isolated'
            }
        
        nearest_distance = min(distances)
        average_distance = np.mean(distances)
        
        # 生成相对位置描述
        relative_desc = self._generate_relative_position_desc(nearest_distance, average_distance)
        
        return {
            'nearest_neighbor_distance': nearest_distance,
            'average_distance_to_others': average_distance,
            'relative_position_desc': relative_desc
        }
    
    def _compute_3d_distance(self, coords1: Tuple[float, float, float], 
                           coords2: Tuple[float, float, float]) -> float:
        x1, y1, z1 = coords1
        x2, y2, z2 = coords2
        
        return math.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)
    
    def _generate_relative_position_desc(self, nearest_distance: float, 
                                       average_distance: float) -> str:
        if nearest_distance < 50:  # 50mm
            return "clustered"
        elif nearest_distance < 100:  # 100mm
            return "close"
        elif nearest_distance < 200:  # 200mm
            return "moderate"
        else:
            return "isolated"
    
    def compute_scan_region_position(self, world_coords: Tuple[float, float, float]) -> Dict:
        if not self.scan_region_bounds:
            return {
                'in_scan_region': True,
                'region_position': 'unknown',
                'distance_to_boundary': 0
            }
        
        x, y, z = world_coords
        bounds = self.scan_region_bounds
        
        in_region = (bounds['x_min'] <= x <= bounds['x_max'] and 
                    bounds['y_min'] <= y <= bounds['y_max'])
        
        if in_region:
            dist_to_x_min = x - bounds['x_min']
            dist_to_x_max = bounds['x_max'] - x
            dist_to_y_min = y - bounds['y_min']
            dist_to_y_max = bounds['y_max'] - y
            
            distance_to_boundary = min(dist_to_x_min, dist_to_x_max, 
                                     dist_to_y_min, dist_to_y_max)
        else:
            dx = max(bounds['x_min'] - x, 0, x - bounds['x_max'])
            dy = max(bounds['y_min'] - y, 0, y - bounds['y_max'])
            distance_to_boundary = -math.sqrt(dx**2 + dy**2)  # 负值表示在区域外
        
        if in_region:
            norm_x = (x - bounds['x_min']) / (bounds['x_max'] - bounds['x_min'])
            norm_y = (y - bounds['y_min']) / (bounds['y_max'] - bounds['y_min'])
            region_position = self._determine_region_position(norm_x, norm_y)
        else:
            region_position = 'outside'
        
        return {
            'in_scan_region': in_region,
            'region_position': region_position,
            'distance_to_boundary': distance_to_boundary
        }
    
    def compute_position_similarity(self, pos1: Dict, pos2: Dict) -> float:
        # 3D坐标距离相似度
        coords1 = pos1['world_coordinates']
        coords2 = pos2['world_coordinates']
        distance_3d = self._compute_3d_distance(coords1, coords2)
        
        # 距离相似度（距离越近相似度越高）
        distance_similarity = math.exp(-distance_3d / 100.0)  # 100mm为缩放因子
        
        # 区域位置相似度
        region_similarity = 1.0 if pos1['region_position'] == pos2['region_position'] else 0.0
        
        # 归一化坐标相似度
        norm1 = pos1['normalized_coords']
        norm2 = pos2['normalized_coords']
        norm_distance = math.sqrt((norm1[0] - norm2[0])**2 + (norm1[1] - norm2[1])**2)
        norm_similarity = math.exp(-norm_distance * 5.0)  # 5.0为缩放因子
        
        # 加权平均
        overall_similarity = (
            0.4 * distance_similarity +
            0.3 * region_similarity +
            0.3 * norm_similarity
        )
        
        return overall_similarity
    
    def extract_all_features(self, mask: np.ndarray, depth_image: np.ndarray, 
                           camera_pose: Dict, all_objects_features: list = None) -> Dict:
        # 基本空间特征
        spatial_features = self.compute_spatial_position(mask, depth_image, camera_pose)
        
        # 相对位置特征
        if all_objects_features:
            relative_features = self.compute_relative_positions(
                spatial_features, all_objects_features
            )
            spatial_features.update(relative_features)
        
        # 扫描区域位置特征
        region_features = self.compute_scan_region_position(
            spatial_features['world_coordinates']
        )
        spatial_features.update(region_features)
        
        return spatial_features
    
    def generate_position_description(self, spatial_features: Dict, class_name: str, 
                                    color_name: str) -> str:
        region = spatial_features['region_position']
        
        # 构建描述字符串
        if region == 'unknown':
            description = f"{color_name.capitalize()} {class_name}"
        else:
            # 转换区域名称为更自然的描述
            region_desc = region.replace('-', ' ').replace('_', ' ')
            if region_desc == 'center':
                region_desc = 'center area'
            elif 'center' in region_desc:
                region_desc = region_desc.replace('center', 'center')
            else:
                region_desc = f"{region_desc} corner"
            
            description = f"{color_name.capitalize()} {class_name} in {region_desc}"
        
        return description