import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2

class HandEyeCalibratedCoordinateCalculator:
    """使用官方手眼标定的坐标计算器"""
    
    def __init__(self):
        # 官方手眼标定参数
        self.translation = np.array([
            0.06637719970480799,      # x (米)
            -0.032133912794949385,    # y (米)  
            0.02259679892714925       # z (米)
        ])
        
        self.rotation_quaternion = np.array([
            0.0013075170827278532,    # x
            -0.0024892917521336377,   # y
            0.7106597907015502,       # z
            0.7035302095188808        # w
        ])
        
        # 构建变换矩阵 (从相机坐标系到末端执行器坐标系)
        self.camera_to_tcp_transform = self._build_transform_matrix()
        
        # RealSense相机内参 (根据你的实际相机调整)
        self.camera_intrinsics = {
            'fx': 912.694580078125,
            'fy': 910.309814453125,
            'cx': 624.051574707031,  # 更新
            'cy': 320.749542236328   # 更新
        }
        
    def _build_transform_matrix(self):
        """构建4x4变换矩阵"""
        # 四元数转旋转矩阵
        rotation = R.from_quat(self.rotation_quaternion)
        rotation_matrix = rotation.as_matrix()
        
        # 构建4x4变换矩阵
        transform = np.eye(4)
        transform[:3, :3] = rotation_matrix
        transform[:3, 3] = self.translation
        
        return transform
    
    def camera_to_world_coordinates(self, camera_point_m, tcp_pose, verbose=True):
        """
        使用官方手眼标定将相机坐标转换为世界坐标
        
        Args:
            camera_point_m: 相机坐标系中的点 [x, y, z] (米)
            tcp_pose: TCP当前位姿 [x, y, z, roll, pitch, yaw] (mm, degree)
        
        Returns:
            world_point: 世界坐标系中的点 [x, y, z] (mm)
        """
        try:
            if verbose:
                print(f"[CALIB] 输入相机坐标: {camera_point_m} (米)")
                print(f"[CALIB] 当前TCP位姿: {tcp_pose}")
            
            # 转换相机点到齐次坐标
            camera_point_homo = np.array([camera_point_m[0], camera_point_m[1], camera_point_m[2], 1.0])
            
            # 相机坐标 -> TCP坐标
            tcp_point_homo = self.camera_to_tcp_transform @ camera_point_homo
            tcp_point = tcp_point_homo[:3]  # 米
            
            if verbose:
                print(f"[CALIB] TCP坐标系中的点: {tcp_point} (米)")
            
            # TCP当前位姿
            tcp_x, tcp_y, tcp_z = tcp_pose[:3]  # mm
            tcp_roll, tcp_pitch, tcp_yaw = tcp_pose[3:]  # degree
            
            # TCP到世界坐标的变换矩阵
            tcp_rotation = R.from_euler('xyz', [tcp_roll, tcp_pitch, tcp_yaw], degrees=True)
            tcp_to_world_matrix = np.eye(4)
            tcp_to_world_matrix[:3, :3] = tcp_rotation.as_matrix()
            tcp_to_world_matrix[:3, 3] = [tcp_x/1000, tcp_y/1000, tcp_z/1000]  # 转换为米
            
            # TCP坐标 -> 世界坐标
            tcp_point_homo_world = np.array([tcp_point[0], tcp_point[1], tcp_point[2], 1.0])
            world_point_homo = tcp_to_world_matrix @ tcp_point_homo_world
            world_point = world_point_homo[:3] * 1000  # 转换为mm
            
            if verbose:
                print(f"[CALIB] 世界坐标: {world_point} (mm)")
            
            return world_point
            
        except Exception as e:
            print(f"[ERROR] 坐标转换失败: {e}")
            return None
    
    def pixel_to_camera_coordinates(self, pixel_x, pixel_y, depth_m):
        """将像素坐标转换为相机坐标"""
        try:
            fx = self.camera_intrinsics['fx']
            fy = self.camera_intrinsics['fy']
            cx = self.camera_intrinsics['cx']
            cy = self.camera_intrinsics['cy']
            
            # 像素坐标到相机坐标的转换
            cam_x = (pixel_x - cx) * depth_m / fx
            cam_y = (pixel_y - cy) * depth_m / fy
            cam_z = depth_m
            
            return np.array([cam_x, cam_y, cam_z])
            
        except Exception as e:
            print(f"[ERROR] 像素到相机坐标转换失败: {e}")
            return None
    
    def calculate_object_center_3d(self, mask, depth_data):
        """计算对象的3D中心点（相机坐标系）"""
        try:
            # 找到mask中的有效像素
            mask_indices = np.where(mask > 0)
            if len(mask_indices[0]) == 0:
                return None
            
            # 收集有效的3D点
            valid_points = []
            
            for i in range(0, len(mask_indices[0]), 5):  # 每5个像素采样一次，提高效率
                y, x = mask_indices[0][i], mask_indices[1][i]
                depth_val = depth_data[y, x] / 1000.0  # 转换为米
                
                if depth_val > 0.01:  # 有效深度
                    cam_point = self.pixel_to_camera_coordinates(x, y, depth_val)
                    if cam_point is not None:
                        valid_points.append(cam_point)
            
            if len(valid_points) < 5:
                return None
            
            # 计算3D中心点（中位数，更抗噪声）
            points_array = np.array(valid_points)
            center_3d = np.median(points_array, axis=0)
            
            return center_3d
            
        except Exception as e:
            print(f"[ERROR] 3D中心点计算失败: {e}")
            return None

# 修改后的坐标计算函数
def calculate_3d_spatial_features_calibrated(mask, depth_data, waypoint_data, bbox, calculator):
    """使用官方标定计算3D空间特征"""
    try:
        # 计算相机坐标系中的物体中心
        center_3d_camera = calculator.calculate_object_center_3d(mask, depth_data)
        if center_3d_camera is None:
            print("[WARN] 无法计算3D中心点")
            return {}
        
        # 当前TCP位姿
        tcp_pose = [
            waypoint_data['world_pos'][0],  # x (mm)
            waypoint_data['world_pos'][1],  # y (mm)
            waypoint_data['world_pos'][2],  # z (mm)
            waypoint_data['roll'],          # roll (degree)
            waypoint_data['pitch'],         # pitch (degree)
            waypoint_data['yaw']            # yaw (degree)
        ]
        
        # 使用官方标定将相机坐标转换为世界坐标
        world_coordinates = calculator.camera_to_world_coordinates(
            center_3d_camera, tcp_pose, verbose=True
        )
        
        if world_coordinates is None:
            print("[ERROR] 世界坐标转换失败")
            return {}
        
        # 计算抓夹宽度信息
        gripper_info = calculate_real_gripper_width_calibrated(mask, depth_data, calculator)
        
        # 计算其他空间特征
        bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        mask_area = np.sum(mask > 0)
        coverage_ratio = mask_area / bbox_area if bbox_area > 0 else 0
        
        result = {
            'centroid_3d_camera': tuple(center_3d_camera),
            'world_coordinates': tuple(world_coordinates),
            'scan_detail': (waypoint_data['world_pos'], [waypoint_data['roll'], waypoint_data['pitch'], waypoint_data['yaw']]),
            'distance_to_camera': center_3d_camera[2],
            'mask_area_pixels': int(mask_area),
            'bbox_area_pixels': int(bbox_area),
            'mask_coverage_ratio': float(coverage_ratio),
            'estimated_real_area_mm2': float(mask_area * (center_3d_camera[2] * 1000) ** 2 / 1000000),
        }
        
        # 添加抓夹信息
        if gripper_info:
            result['gripper_width_info'] = gripper_info
        
        print(f"[CALIB] 最终世界坐标: {world_coordinates}")
        return result
        
    except Exception as e:
        print(f"[ERROR] 3D空间特征计算失败: {e}")
        return {}

def calculate_real_gripper_width_calibrated(mask, depth_data, calculator):
    """使用校准后的方法计算真实抓夹宽度"""
    try:
        # 找到mask的最小外接矩形
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        
        largest_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest_contour)
        _, (width, height), angle = rect
        
        # 选择较短的边作为抓夹宽度参考
        shorter_side_pixels = min(width, height)
        
        # 获取物体中心深度
        center_depth = get_object_center_depth_calibrated(mask, depth_data)
        if center_depth is None:
            return None
        
        # 使用相机内参进行像素到毫米转换
        fx = calculator.camera_intrinsics['fx']
        
        # 像素到毫米转换
        real_width_mm = shorter_side_pixels * (center_depth / fx) * 1000
        
        # 添加安全余量（建议比实际宽度大25%）
        safety_margin = 1.25
        recommended_width = real_width_mm * safety_margin
        
        # 限制在合理范围内 (100mm - 800mm)
        recommended_width = max(100, min(800, recommended_width))
        
        print(f"[CALIB] 抓夹宽度: 像素={shorter_side_pixels:.1f}, 深度={center_depth:.3f}m")
        print(f"[CALIB] 实际宽度={real_width_mm:.1f}mm, 推荐={recommended_width:.1f}mm")
        
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

def get_object_center_depth_calibrated(mask, depth_data):
    """获取物体中心的深度值（改进版）"""
    try:
        # 找到mask中心
        mask_indices = np.where(mask > 0)
        if len(mask_indices[0]) == 0:
            return None
        
        center_y = int(np.median(mask_indices[0]))
        center_x = int(np.median(mask_indices[1]))
        
        # 获取中心点周围的深度值
        depth_samples = []
        for dy in range(-5, 6):  # 稍微扩大采样范围
            for dx in range(-5, 6):
                y, x = center_y + dy, center_x + dx
                if 0 <= y < depth_data.shape[0] and 0 <= x < depth_data.shape[1]:
                    if mask[y, x] > 0:  # 确保在物体内
                        depth_val = depth_data[y, x] / 1000.0  # 转换为米
                        if depth_val > 0.01:
                            depth_samples.append(depth_val)
        
        return np.median(depth_samples) if depth_samples else None
        
    except Exception:
        return None

def calculate_object_height_with_background_depth(mask, depth_data, bbox, waypoint_data, calculator):
    """计算物体高度和背景深度（使用标定）"""
    try:
        x1, y1, x2, y2 = map(int, bbox)
        
        # 确保bbox在图像范围内
        h, w = depth_data.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        print(f"[CALIB] 计算高度: bbox=[{x1},{y1},{x2},{y2}]")
        
        # 收集物体和背景的深度值
        object_depths = []
        background_depths = []
        
        # 方法1: bbox内采样
        for y in range(y1, y2, 2):  # 每2个像素采样一次
            for x in range(x1, x2, 2):
                if 0 <= y < h and 0 <= x < w:
                    depth_val = depth_data[y, x] / 1000.0  # 转换为米
                    if depth_val > 0.01:
                        if mask[y, x] > 0:
                            object_depths.append(depth_val)
                        else:
                            background_depths.append(depth_val)
        
        # 方法2: 扩展背景采样
        expand_pixels = 30
        for y in range(y1 - expand_pixels, y2 + expand_pixels, 3):
            for x in range(x1 - expand_pixels, x2 + expand_pixels, 3):
                if 0 <= y < h and 0 <= x < w:
                    if not (x1 <= x < x2 and y1 <= y < y2):  # 在扩展区域但不在原bbox内
                        depth_val = depth_data[y, x] / 1000.0
                        if depth_val > 0.01:
                            background_depths.append(depth_val)
        
        # 计算高度
        if len(object_depths) < 10 or len(background_depths) < 10:
            print("[WARN] 深度采样不足，使用默认高度")
            background_depth_m = 0.3 if not background_depths else np.median(background_depths)
        else:
            object_median = np.median(object_depths)
            background_median = np.median(background_depths)
            height_mm = abs(object_median - background_median) * 1000
            background_depth_m = background_median
            
            print(f"[CALIB] 物体深度中位数: {object_median:.3f}m")
            print(f"[CALIB] 背景深度中位数: {background_median:.3f}m")
            print(f"[CALIB] 计算高度: {height_mm:.1f}mm")
            
            if not (5 <= height_mm <= 500):
                print("[WARN] 计算高度超出合理范围，使用默认值")
                height_mm = 30.0
        
        # 将背景深度转换为世界坐标Z
        if background_depth_m > 0.01:
            # 使用bbox中心点作为参考
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            background_camera_point = calculator.pixel_to_camera_coordinates(center_x, center_y, background_depth_m)
            
            if background_camera_point is not None:
                tcp_pose = [
                    waypoint_data['world_pos'][0],
                    waypoint_data['world_pos'][1],
                    waypoint_data['world_pos'][2],
                    waypoint_data['roll'],
                    waypoint_data['pitch'],
                    waypoint_data['yaw']
                ]
                
                background_world = calculator.camera_to_world_coordinates(
                    background_camera_point, tcp_pose, verbose=False
                )
                
                if background_world is not None:
                    background_world_z = background_world[2]
                else:
                    background_world_z = waypoint_data['world_pos'][2] - background_depth_m * 1000
            else:
                background_world_z = waypoint_data['world_pos'][2] - background_depth_m * 1000
        else:
            background_world_z = waypoint_data['world_pos'][2] - 300  # 默认
        
        print(f"[CALIB] 背景世界Z坐标: {background_world_z:.1f}mm")
        
        return {
            'height_mm': height_mm if 'height_mm' in locals() else 30.0,
            'background_world_z': background_world_z,
            'background_depth_m': background_depth_m,
            'confidence': 0.8 if len(object_depths) >= 10 and len(background_depths) >= 10 else 0.5
        }
        
    except Exception as e:
        print(f"[ERROR] 高度和背景深度计算失败: {e}")
        return {
            'height_mm': 30.0,
            'background_world_z': waypoint_data['world_pos'][2] - 300,
            'background_depth_m': 0.3,
            'confidence': 0.2
        }