# utils/coordinate_calculator.py - 重构版本 + 动态补偿
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d
from typing import Dict, Optional, Tuple, List
import cv2

class CoordinateCalculator:
    """统一的坐标计算器 - 支持动态补偿"""
    
    def __init__(self, calibration_data: Optional[Dict] = None):
        """
        初始化坐标计算器
        
        Args:
            calibration_data: 标定数据，包含相机内参和手眼标定参数
        """
        # 设置默认参数
        self.camera_intrinsics = {
            'fx': 912.694580078125,
            'fy': 910.309814453125,
            'cx': 624.051574707031,
            'cy': 320.749542236328
        }
        
        # 手眼标定参数
        self.hand_eye_translation = np.array([
            0.06637719970480799,
            -0.032133912794949385,
            0.02259679892714925
        ])
        
        self.hand_eye_quaternion = np.array([
            0.0013075170827278532,
            -0.0024892917521336377,
            0.7106597907015502,
            0.7035302095188808
        ])
        
        # 从标定数据更新参数
        if calibration_data:
            self.update_calibration(calibration_data)
        
        # 🆕 安全检查：确保四元数有效
        if (self.hand_eye_quaternion is None or 
            len(self.hand_eye_quaternion) != 4 or
            np.any(np.isnan(self.hand_eye_quaternion))):
            print("[WARNING] 无效的手眼标定四元数，使用默认值")
            self.hand_eye_quaternion = np.array([0.0, 0.0, 0.0, 1.0])  # 单位四元数
        
        # 构建变换矩阵
        self.camera_to_tcp_transform = self._build_transform_matrix()
        
        # 🆕 设置动态补偿系统
        self._setup_dynamic_compensation()
    
    def update_calibration(self, calibration_data: Dict):
        """更新标定参数"""
        if 'camera_intrinsics' in calibration_data:
            self.camera_intrinsics.update(calibration_data['camera_intrinsics'])
        
        if 'hand_eye_translation' in calibration_data and calibration_data['hand_eye_translation']:
            self.hand_eye_translation = np.array(calibration_data['hand_eye_translation'])
        
        if 'hand_eye_quaternion' in calibration_data and calibration_data['hand_eye_quaternion']:
            self.hand_eye_quaternion = np.array(calibration_data['hand_eye_quaternion'])
            
            # 🆕 验证四元数
            if len(self.hand_eye_quaternion) != 4:
                print(f"[WARNING] 四元数长度错误: {len(self.hand_eye_quaternion)}，使用默认值")
                self.hand_eye_quaternion = np.array([0.0, 0.0, 0.0, 1.0])
    
    def _setup_dynamic_compensation(self):
        """设置基于yaw角度的动态补偿"""
        # 基于你观察到的数据点设置补偿
        self.yaw_angles = np.array([-180, 0, 180])  # yaw角度 (度)
        
        # 对应的TCP补偿 [x, y, z] (米)
        self.compensation_x = np.array([-0.005, 0.0, 0.005])
        self.compensation_y = np.array([0.005, 0.002, -0.003])
        self.compensation_z = np.array([0.0, 0.0, 0.0])
        
        # 创建插值函数
        self.interp_x = interp1d(self.yaw_angles, self.compensation_x, 
                                kind='linear', fill_value='extrapolate')
        self.interp_y = interp1d(self.yaw_angles, self.compensation_y, 
                                kind='linear', fill_value='extrapolate')
        self.interp_z = interp1d(self.yaw_angles, self.compensation_z, 
                                kind='linear', fill_value='extrapolate')
    
    
    def get_dynamic_compensation(self, yaw_deg: float) -> np.ndarray:
        """
        根据yaw角度计算动态补偿
        
        Args:
            yaw_deg: yaw角度 (度)
            
        Returns:
            compensation: TCP坐标系中的补偿向量 [x, y, z] (米)
        """
        # 将yaw角度标准化到-180到180范围
        yaw_normalized = yaw_deg
        
        comp_x = float(self.interp_x(yaw_normalized))
        comp_y = float(self.interp_y(yaw_normalized))
        comp_z = float(self.interp_z(yaw_normalized))
        
        compensation = np.array([comp_x, comp_y, comp_z])
        
        return compensation
    
    def _build_transform_matrix(self) -> np.ndarray:
        """构建4x4变换矩阵"""
        rotation = R.from_quat(self.hand_eye_quaternion)
        rotation_matrix = rotation.as_matrix()
        
        transform = np.eye(4)
        transform[:3, :3] = rotation_matrix
        transform[:3, 3] = self.hand_eye_translation
        
        return transform
    
    def pixel_to_camera_coordinates(self, pixel_x: float, pixel_y: float, depth_m: float) -> np.ndarray:
        """像素坐标转相机坐标"""
        fx, fy = self.camera_intrinsics['fx'], self.camera_intrinsics['fy']
        cx, cy = self.camera_intrinsics['cx'], self.camera_intrinsics['cy']
        
        cam_x = (pixel_x - cx) * depth_m / fx
        cam_y = (pixel_y - cy) * depth_m / fy
        cam_z = depth_m
        
        return np.array([cam_x, cam_y, cam_z])
    
    def camera_to_world_coordinates(self, camera_point: np.ndarray, tcp_pose: List[float]) -> np.ndarray:
        """
        相机坐标转世界坐标 - 使用动态补偿
        
        Args:
            camera_point: 相机坐标系中的点 [x, y, z] (米)
            tcp_pose: TCP当前位姿 [x, y, z, roll, pitch, yaw] (mm, degree)
        
        Returns:
            world_point: 世界坐标系中的点 [x, y, z] (mm)
        """
        try:            
            # 相机坐标转TCP坐标
            camera_point_homo = np.array([camera_point[0], camera_point[1], camera_point[2], 1.0])
            tcp_point_homo = self.camera_to_tcp_transform @ camera_point_homo
            tcp_point = tcp_point_homo[:3]
            
            # ✅ 根据yaw角度动态计算补偿
            yaw_angle = tcp_pose[5]  # yaw是tcp_pose的第6个元素
            dynamic_compensation = self.get_dynamic_compensation(yaw_angle)
            
            # 应用动态补偿
            tcp_point += dynamic_compensation
            
            # TCP到世界坐标的变换矩阵
            tcp_x, tcp_y, tcp_z = tcp_pose[:3]  # mm
            tcp_roll, tcp_pitch, tcp_yaw = tcp_pose[3:]  # degree
            
            tcp_rotation = R.from_euler('xyz', [tcp_roll, tcp_pitch, tcp_yaw], degrees=True)
            tcp_to_world_matrix = np.eye(4)
            tcp_to_world_matrix[:3, :3] = tcp_rotation.as_matrix()
            tcp_to_world_matrix[:3, 3] = [tcp_x / 1000.0, tcp_y / 1000.0, tcp_z / 1000.0]  # 转为米
            
            # TCP坐标 → 世界坐标
            tcp_point_homo_world = np.array([tcp_point[0], tcp_point[1], tcp_point[2], 1.0])
            world_point_homo = tcp_to_world_matrix @ tcp_point_homo_world
            world_point = world_point_homo[:3] * 1000  # 转为 mm
            
            return world_point
            
        except Exception as e:
            print(f"[ERROR] 坐标转换失败: {e}")
            return camera_point * 1000  # 返回默认值，避免程序崩溃


class ObjectAnalyzer:
    """物体分析器 - 高度、体积、抓夹等计算"""
    
    def __init__(self, coordinate_calculator: CoordinateCalculator):
        self.coord_calc = coordinate_calculator
    
    def calculate_object_height_and_background(self, mask: np.ndarray, depth_data: np.ndarray, 
                                             bbox: List[int], waypoint_data: Dict) -> Dict:
        """计算物体高度和背景深度"""
        x1, y1, x2, y2 = map(int, bbox)
        h, w = depth_data.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        # 收集深度数据
        object_depths = []
        background_depths = []
        
        # 方法1: bbox内采样
        for y in range(y1, y2, 3):
            for x in range(x1, x2, 3):
                if 0 <= y < h and 0 <= x < w:
                    depth_val = depth_data[y, x] / 1000.0
                    if depth_val > 0.01:
                        if mask[y, x] > 0:
                            object_depths.append(depth_val)
                        else:
                            background_depths.append(depth_val)
        
        # 方法2: 扩展背景采样
        expand_pixels = 30
        for y in range(y1 - expand_pixels, y2 + expand_pixels, 5):
            for x in range(x1 - expand_pixels, x2 + expand_pixels, 5):
                if 0 <= y < h and 0 <= x < w:
                    if not (x1 <= x < x2 and y1 <= y < y2):
                        depth_val = depth_data[y, x] / 1000.0
                        if depth_val > 0.01:
                            background_depths.append(depth_val)
        
        # 计算结果
        if len(object_depths) < 10 or len(background_depths) < 10:
            height_mm = 30.0
            background_depth_m = 0.3
            confidence = 0.2
        else:
            object_median = np.median(object_depths)
            background_median = np.median(background_depths)
            height_mm = abs(object_median - background_median) * 1000
            background_depth_m = background_median
            confidence = 0.8
            
            if not (5 <= height_mm <= 500):
                height_mm = 30.0
                confidence = 0.3
        
        # 计算背景世界坐标 - 使用动态补偿
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        background_camera_point = self.coord_calc.pixel_to_camera_coordinates(
            center_x, center_y, background_depth_m
        )
        
        tcp_pose = [
            waypoint_data['world_pos'][0],
            waypoint_data['world_pos'][1], 
            waypoint_data['world_pos'][2],
            waypoint_data['roll'],
            waypoint_data['pitch'],
            waypoint_data['yaw']
        ]
        
        background_world = self.coord_calc.camera_to_world_coordinates(
            background_camera_point, tcp_pose
        )
        
        return {
            'height_mm': height_mm,
            'background_world_z': background_world[2],
            'background_depth_m': background_depth_m,
            'confidence': confidence,
            'sample_counts': {
                'object_samples': len(object_depths),
                'background_samples': len(background_depths)
            }
        }
    
    def calculate_3d_spatial_features(self, mask: np.ndarray, depth_data: np.ndarray,
                                    waypoint_data: Dict, bbox: List[int]) -> Dict:
        """计算3D空间特征 - 使用动态补偿"""
        # 计算物体中心的3D坐标
        center_3d_camera = self._calculate_object_center_3d(mask, depth_data)
        if center_3d_camera is None:
            return {'error': 'cannot_calculate_center'}
        
        # TCP位姿
        tcp_pose = [
            waypoint_data['world_pos'][0],
            waypoint_data['world_pos'][1],
            waypoint_data['world_pos'][2],
            waypoint_data['roll'],
            waypoint_data['pitch'], 
            waypoint_data['yaw']
        ]
        
        # 转换到世界坐标 - 使用动态补偿
        world_coordinates = self.coord_calc.camera_to_world_coordinates(
            center_3d_camera, tcp_pose
        )
        
        # 计算抓夹信息
        gripper_info = self._calculate_gripper_width(mask, depth_data)
        
        # 计算其他空间特征
        bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        mask_area = np.sum(mask > 0)
        coverage_ratio = mask_area / bbox_area if bbox_area > 0 else 0
        
        result = {
            'centroid_3d_camera': tuple(center_3d_camera),
            'world_coordinates': tuple(world_coordinates),
            'distance_to_camera': center_3d_camera[2],
            'mask_area_pixels': int(mask_area),
            'bbox_area_pixels': int(bbox_area),
            'mask_coverage_ratio': float(coverage_ratio),
            'estimated_real_area_mm2': float(mask_area * (center_3d_camera[2] * 1000) ** 2 / 1000000),
            'scan_detail': (waypoint_data['world_pos'], [waypoint_data['roll'], waypoint_data['pitch'], waypoint_data['yaw']]),
            'dynamic_compensation_used': self.coord_calc.get_dynamic_compensation(waypoint_data['yaw'])
        }
        
        if gripper_info:
            result['gripper_width_info'] = gripper_info
        
        return result
    
    def _calculate_object_center_3d(self, mask: np.ndarray, depth_data: np.ndarray) -> Optional[np.ndarray]:
        """计算物体的3D中心点"""
        mask_indices = np.where(mask > 0)
        if len(mask_indices[0]) == 0:
            return None
        
        valid_points = []
        
        for i in range(0, len(mask_indices[0]), 5):
            y, x = mask_indices[0][i], mask_indices[1][i]
            depth_val = depth_data[y, x] / 1000.0
            
            if depth_val > 0.01:
                cam_point = self.coord_calc.pixel_to_camera_coordinates(x, y, depth_val)
                valid_points.append(cam_point)
        
        if len(valid_points) < 5:
            return None
        
        points_array = np.array(valid_points)
        center_3d = np.median(points_array, axis=0)
        
        return center_3d
    
    def _calculate_gripper_width(self, mask: np.ndarray, depth_data: np.ndarray) -> Optional[Dict]:
        """计算推荐抓夹宽度"""
        try:
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return None
            
            largest_contour = max(contours, key=cv2.contourArea)
            rect = cv2.minAreaRect(largest_contour)
            _, (width, height), angle = rect
            # 选择较短的边作为抓夹宽度参考
            shorter_side_pixels = min(width, height)
            
            # 获取物体中心深度
            center_depth = self._get_object_center_depth(mask, depth_data)
            if center_depth is None:
                return None
            
            # 像素到毫米转换
            fx = self.coord_calc.camera_intrinsics['fx']
            real_width_mm = shorter_side_pixels * (center_depth / fx) * 1000
            
            # 添加安全余量
            recommended_width = real_width_mm * 10
            recommended_width = max(100, min(800, recommended_width))
            
            return {
                'real_width_mm': real_width_mm,
                'recommended_gripper_width': int(recommended_width),
                'pixel_width': shorter_side_pixels,
                'depth_used': center_depth,
                'angle': angle
            }
            
        except Exception as e:
            print(f"[ERROR] 抓夹宽度计算失败: {e}")
            return None
    
    def _get_object_center_depth(self, mask: np.ndarray, depth_data: np.ndarray) -> Optional[float]:
        """获取物体中心的深度值"""
        try:
            mask_indices = np.where(mask > 0)
            if len(mask_indices[0]) == 0:
                return None
            
            center_y = int(np.median(mask_indices[0]))
            center_x = int(np.median(mask_indices[1]))
            
            depth_samples = []
            for dy in range(-5, 6):
                for dx in range(-5, 6):
                    y, x = center_y + dy, center_x + dx
                    if 0 <= y < depth_data.shape[0] and 0 <= x < depth_data.shape[1]:
                        if mask[y, x] > 0:
                            depth_val = depth_data[y, x] / 1000.0
                            if depth_val > 0.01:
                                depth_samples.append(depth_val)
            
            return np.median(depth_samples) if depth_samples else None
            
        except Exception:
            return None


class AdaptiveThresholdManager:
    """自适应阈值管理器"""
    
    def __init__(self):
        self.threshold_history = {}
        self.feature_weights = {
            'fpfh': 0.4,
            'geometric': 0.3,
            'shape_context_3d': 0.2,
            'hu_moments': 0.1
        }
        self.base_thresholds = {
            'fpfh': 0.85,
            'geometric': 0.75,
            'shape_context_3d': 0.8,
            'hu_moments': 0.7
        }
    
    def get_adaptive_threshold(self, feature_type: str, feature_quality: float) -> float:
        """根据特征质量动态调整阈值"""
        base_threshold = self.base_thresholds.get(feature_type, 0.75)
        quality_factor = feature_quality / 100.0
        
        # 质量越高，阈值越严格
        adaptive_threshold = base_threshold * (0.7 + 0.3 * quality_factor)
        
        return adaptive_threshold
    
    def update_threshold_history(self, feature_type: str, similarity: float, 
                               is_same_object: bool, feature_quality: float):
        """更新阈值历史，用于在线学习"""
        if feature_type not in self.threshold_history:
            self.threshold_history[feature_type] = {
                'successes': [],
                'failures': [],
                'qualities': []
            }
        
        history = self.threshold_history[feature_type]
        
        if is_same_object:
            history['successes'].append((similarity, feature_quality))
        else:
            history['failures'].append((similarity, feature_quality))
        
        # 保持历史记录在合理范围内
        max_history = 100
        for key in ['successes', 'failures']:
            if len(history[key]) > max_history:
                history[key] = history[key][-max_history:]
    
    def get_optimal_threshold(self, feature_type: str) -> float:
        """基于历史数据计算最优阈值"""
        if feature_type not in self.threshold_history:
            return self.base_thresholds.get(feature_type, 0.75)
        
        history = self.threshold_history[feature_type]
        successes = history['successes']
        failures = history['failures']
        
        if len(successes) < 5 or len(failures) < 5:
            return self.base_thresholds.get(feature_type, 0.75)
        
        # 计算最优分离阈值
        success_similarities = [s[0] for s in successes]
        failure_similarities = [f[0] for f in failures]
        
        # 寻找最佳分离点
        all_similarities = sorted(success_similarities + failure_similarities)
        best_threshold = self.base_thresholds.get(feature_type, 0.75)
        best_accuracy = 0.0
        
        for threshold in all_similarities:
            tp = sum(1 for s in success_similarities if s >= threshold)
            tn = sum(1 for f in failure_similarities if f < threshold)
            total = len(successes) + len(failures)
            accuracy = (tp + tn) / total
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold
        
        return best_threshold


# 测试和使用示例
if __name__ == "__main__":
    # 创建坐标计算器
    coord_calc = CoordinateCalculator()
    
    # 测试动态补偿在不同yaw角度下的效果
    test_yaws = [-180, -90, -45, 0, 45, 90, 180]
    
    print("\n=== 动态补偿测试 ===")
    for yaw in test_yaws:
        compensation = coord_calc.get_dynamic_compensation(yaw)
        print(f"Yaw={yaw:4d}°: 补偿=[{compensation[0]:.4f}, {compensation[1]:.4f}, {compensation[2]:.4f}] (米)")
    
    # 测试角度标准化
    print("\n=== 角度标准化测试 ===")
    test_angles = [-270, -180, -90, 0, 90, 180, 270, 360, 450]
    for angle in test_angles:
        compensation = coord_calc.get_dynamic_compensation(angle)
        normalized = ((angle + 180) % 360) - 180
        print(f"原始角度={angle:4d}° -> 标准化={normalized:4d}° -> 补偿X={compensation[0]:.4f}")