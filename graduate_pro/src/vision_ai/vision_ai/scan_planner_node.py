#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from vision_ai_interfaces.srv import PlanScan
from vision_ai_interfaces.msg import ScanPlan, Waypoint, GridPosition, RelativePosition
from geometry_msgs.msg import Point, Pose, Quaternion
import math
import numpy as np

class ScanPlannerNode(Node):
    """
    A ROS2 node for planning scan paths based on input points and object height.
    It calculates optimal scanning strategies (single-point or multi-point)
    considering camera intrinsics, workspace limits, and object geometry.
    """
    def __init__(self):
        super().__init__('scan_planner_node')
    
        # Robot and workspace parameters
        self.safe_pose = [200, 0, 250, 179, 0, 0]  # mm + deg (x, y, z, roll, pitch, yaw)
        self.workspace_limit = 800  # mm
        self.obstacle_radius = 150  # mm
        self.max_height = 550  # mm (Maximum reachable height for scanning)
        
        # Camera parameters (1280x720 resolution)
        self.camera_width = 1280
        self.camera_height = 720
        self.fx = 912.694580078125  # Focal length in pixels (x-axis)
        self.fy = 910.309814453125  # Focal length in pixels (y-axis)
        
        # Calculate Field of View (FOV)
        self.fov_h = 2 * math.atan(self.camera_width / (2 * self.fx)) # Horizontal FOV in radians
        self.fov_v = 2 * math.atan(self.camera_height / (2 * self.fy)) # Vertical FOV in radians
        
        # Create the planning service
        self.plan_service = self.create_service(
            PlanScan, 
            'plan_scan', 
            self.plan_scan_callback
        )
        
        self.get_logger().info('Scan planning node has started.')
        self.get_logger().info(f'FOV: {math.degrees(self.fov_h):.1f}° × {math.degrees(self.fov_v):.1f}°')
        # Verify FOV calculation
        self.get_logger().info(f'Camera intrinsics verification:')
        self.get_logger().info(f'  fx={self.fx:.1f}, fy={self.fy:.1f}')
        self.get_logger().info(f'  FOV H: {math.degrees(self.fov_h):.1f}°')
        self.get_logger().info(f'  FOV V: {math.degrees(self.fov_v):.1f}°')
        
        # Verification: Coverage at 1 meter height
        test_height = 1000  # 1 meter
        coverage_h = 2 * test_height * math.tan(self.fov_h / 2)
        coverage_v = 2 * test_height * math.tan(self.fov_v / 2)
        self.get_logger().info(f'  At 1m height: {coverage_h:.0f}×{coverage_v:.0f}mm coverage')

    def create_enhanced_waypoint_msg(self, position, waypoint_index, grid_info=None, relative_info=None, overlap_info=None):
        """
        创建增强的Waypoint消息，包含grid和相对位置信息
        """
        waypoint = Waypoint()
        
        # 解析位置信息
        if len(position) >= 4:
            x, y, z, yaw = position[:4]
        else:
            x, y, z = position[:3]
            yaw = 0.0
        
        # 设置基本位置信息
        waypoint.pose = Pose()
        waypoint.pose.position.x = float(x)
        waypoint.pose.position.y = float(y) 
        waypoint.pose.position.z = float(z)

        # 设置yaw角度为四元数
        yaw_rad = math.radians(yaw)
        waypoint.pose.orientation = Quaternion()
        waypoint.pose.orientation.x = 0.0
        waypoint.pose.orientation.y = 0.0
        waypoint.pose.orientation.z = math.sin(yaw_rad / 2)
        waypoint.pose.orientation.w = math.cos(yaw_rad / 2)
        
        waypoint.waypoint_index = int(waypoint_index)
        
        # 🆕 添加grid信息
        if grid_info:
            waypoint.grid_position = GridPosition()
            waypoint.grid_position.grid_x = int(grid_info.get('grid_x', 0))
            waypoint.grid_position.grid_y = int(grid_info.get('grid_y', 0))
            waypoint.grid_position.total_x = int(grid_info.get('total_x', 1))
            waypoint.grid_position.total_y = int(grid_info.get('total_y', 1))
        
        # 🆕 添加相对位置信息
        if relative_info:
            waypoint.relative_position = RelativePosition()
            waypoint.relative_position.camera_x = float(relative_info.get('camera_x', 0.0))
            waypoint.relative_position.camera_y = float(relative_info.get('camera_y', 0.0))
            waypoint.relative_position.neighbors = relative_info.get('neighbors', [])
        
        # 🆕 添加重叠度信息
        if overlap_info:
            waypoint.overlap_x_ratio = float(overlap_info.get('overlap_x', 0.0))
            waypoint.overlap_y_ratio = float(overlap_info.get('overlap_y', 0.0))
        else:
            waypoint.overlap_x_ratio = 0.0
            waypoint.overlap_y_ratio = 0.0
        
        # 日志输出
        self.get_logger().info(f'🎯 Enhanced Waypoint {waypoint_index}: ({x:.1f}, {y:.1f}, {z:.1f}) yaw={yaw:.1f}°')
        if grid_info:
            self.get_logger().info(f'   Grid: ({grid_info.get("grid_x")}, {grid_info.get("grid_y")}) of {grid_info.get("total_x")}×{grid_info.get("total_y")}')
        if relative_info:
            self.get_logger().info(f'   Camera: ({relative_info.get("camera_x"):.1f}, {relative_info.get("camera_y"):.1f})')
            self.get_logger().info(f'   Neighbors: {relative_info.get("neighbors")}')
        
        return waypoint
    def calculate_enhanced_zigzag_path(self, coordinates, object_height, strategy):
        """
        增强的zigzag路径生成，包含完整的相对位置信息
        """
        scan_height = strategy['scan_height']
        yaw = strategy['yaw']
        long_dim = strategy['long_dim']
        short_dim = strategy['short_dim']
        overlap_ratio = strategy['overlap_ratio']
        
        self.get_logger().info(f'Generating enhanced path for {strategy["points"]} points:')
        self.get_logger().info(f'  Scan height: {scan_height:.1f}mm')
        self.get_logger().info(f'  Yaw angle: {yaw:.1f}°')
        self.get_logger().info(f'  Overlap ratio: {overlap_ratio:.1%}')
        
        # 单点扫描
        if strategy['points'] == 1:
            points = np.array(coordinates)
            centroid = np.mean(points, axis=0)
            
            # 创建单点waypoint，没有grid信息
            waypoint_data = {
                'position': (centroid[0], centroid[1], scan_height, yaw),
                'waypoint_index': 0,
                'grid_info': None,
                'relative_info': None,
                'overlap_info': {'overlap_x': 0.0, 'overlap_y': 0.0}
            }
            
            return [waypoint_data]
        
        # 多点扫描：重新计算准确的网格
        fov_h_deg, fov_v_deg = self.calculate_fov_from_intrinsics()
        fov_h_rad = math.radians(fov_h_deg / 2)
        fov_v_rad = math.radians(fov_v_deg / 2)
        effective_height = scan_height - object_height
        
        # 计算步长
        coverage_1280 = 2 * effective_height * math.tan(fov_h_rad)
        coverage_720 = 2 * effective_height * math.tan(fov_v_rad)
        step_1280 = coverage_1280 * (1 - overlap_ratio)
        step_720 = coverage_720 * (1 - overlap_ratio)
        
        # 根据相机方向计算网格
        if strategy['camera_1280_along'] == 'long':
            num_y = max(1, int(math.ceil(long_dim / step_1280)))   
            num_x = max(1, int(math.ceil(short_dim / step_720)))   
            y_step = step_1280
            x_step = step_720
        else:
            num_y = max(1, int(math.ceil(short_dim / step_1280)))  
            num_x = max(1, int(math.ceil(long_dim / step_720)))    
            y_step = step_1280
            x_step = step_720
        
        total_points = num_x * num_y
        self.get_logger().info(f'  Calculated grid: {num_x}×{num_y} = {total_points} points')
        
        # 计算起始位置
        x_start = -(num_x - 1) * x_step / 2 if num_x > 1 else 0
        y_start = -(num_y - 1) * y_step / 2 if num_y > 1 else 0
        
        # 生成路径点
        path = []
        points = np.array(coordinates)
        centroid = np.mean(points, axis=0)
        
        waypoint_index = 0
        for i in range(num_x):
            camera_x = x_start + i * x_step
            y_range = range(num_y) if i % 2 == 0 else range(num_y - 1, -1, -1)
            
            for j in y_range:
                camera_y = y_start + j * y_step
                
                # 转换到世界坐标
                yaw_rad = math.radians(yaw)
                world_x = centroid[0] + camera_x * math.cos(yaw_rad) - camera_y * math.sin(yaw_rad)
                world_y = centroid[1] + camera_x * math.sin(yaw_rad) + camera_y * math.cos(yaw_rad)
                
                # 🆕 计算邻居关系
                neighbors = []
                if i > 0:  # 有左邻居
                    neighbors.append("left")
                if i < num_x - 1:  # 有右邻居
                    neighbors.append("right")
                if j > 0:  # 有上邻居
                    neighbors.append("top")
                if j < num_y - 1:  # 有下邻居
                    neighbors.append("bottom")
                
                # 🆕 构建完整的waypoint信息
                waypoint_data = {
                    'position': (world_x, world_y, scan_height, yaw),
                    'waypoint_index': waypoint_index,
                    'grid_info': {
                        'grid_x': i,
                        'grid_y': j,
                        'total_x': num_x,
                        'total_y': num_y
                    },
                    'relative_info': {
                        'camera_x': camera_x,
                        'camera_y': camera_y,
                        'neighbors': neighbors
                    },
                    'overlap_info': {
                        'overlap_x': overlap_ratio if ('left' in neighbors or 'right' in neighbors) else 0.0,
                        'overlap_y': overlap_ratio if ('top' in neighbors or 'bottom' in neighbors) else 0.0
                    }
                }
                
                path.append(waypoint_data)
                
                self.get_logger().info(f'  WP{waypoint_index}: World({world_x:.1f},{world_y:.1f}) Camera({camera_x:.1f},{camera_y:.1f}) Neighbors{neighbors}')
                waypoint_index += 1
        
        self.get_logger().info(f'Generated {len(path)} enhanced waypoints')
        return path

    def plan_scan_callback(self, request, response):
        """
        处理扫描规划请求 - 增强版
        """
        try:
            # 提取参数
            points = [(p.x, p.y) for p in request.points]
            object_height = request.object_height
            mode = request.mode
            
            self.get_logger().info(f'Enhanced planning request: mode={mode}, height={object_height}mm, {len(points)} points')
            
            # 计算最佳扫描策略
            best_strategy = self.calculate_scanning_strategies(points, object_height)
            
            # 创建ScanPlan消息
            scan_plan = ScanPlan()
            scan_plan.object_height = float(object_height)
            scan_plan.mode = mode
            scan_plan.required_height = float(best_strategy['height'])
            scan_plan.scan_height = float(best_strategy.get('scan_height', best_strategy['height']))
            scan_plan.overlap_ratio = float(best_strategy.get('overlap_ratio', 0.25))
            
            # 设置扫描区域
            for point in request.points:
                scan_plan.scan_region.append(point)
            
            # 🆕 使用增强的路径生成
            if best_strategy['points'] == 1:
                scan_plan.strategy = "single_point"
                path_data = self.calculate_enhanced_zigzag_path(points, object_height, best_strategy)
                
                for wp_data in path_data:
                    waypoint = self.create_enhanced_waypoint_msg(
                        wp_data['position'], 
                        wp_data['waypoint_index'],
                        wp_data['grid_info'],
                        wp_data['relative_info'],
                        wp_data['overlap_info']
                    )
                    scan_plan.waypoints.append(waypoint)
                
                self.get_logger().info(f'Enhanced single-point scan created')
            else:
                scan_plan.strategy = "multi_point"
                path_data = self.calculate_enhanced_zigzag_path(points, object_height, best_strategy)
                
                for wp_data in path_data:
                    waypoint = self.create_enhanced_waypoint_msg(
                        wp_data['position'], 
                        wp_data['waypoint_index'],
                        wp_data['grid_info'],
                        wp_data['relative_info'],
                        wp_data['overlap_info']
                    )
                    scan_plan.waypoints.append(waypoint)
                
                self.get_logger().info(f'Enhanced multi-point scan: {len(scan_plan.waypoints)} waypoints with grid info')
            
            response.success = True
            response.message = f"Enhanced planning success: {scan_plan.strategy} strategy, {len(scan_plan.waypoints)} waypoints with positioning data"
            response.scan_plan = scan_plan
            
        except Exception as e:
            response.success = False
            response.message = f"Enhanced planning failed: {str(e)}"
            response.scan_plan = ScanPlan()
            self.get_logger().error(f'Enhanced planning error: {e}')
        
        return response
    


    def calculate_fov_from_intrinsics(self):
        """
        Calculates FOV in degrees (already done in __init__, just returns the values).
        """
        fov_h_deg = math.degrees(self.fov_h)
        fov_v_deg = math.degrees(self.fov_v)
        return fov_h_deg, fov_v_deg

    def validate_scan_height(self, calculated_height, object_height):
        """确保扫描高度在350-550mm范围内"""
        effective_height = calculated_height - object_height
        if effective_height < 350:
            return object_height + 350
        elif effective_height > 550:
            return object_height + 550
        return calculated_height

    def create_waypoint_msg(self, position, waypoint_index):
        """
        Creates a Waypoint message with position (x, y, z) and orientation (yaw).
        Yaw is converted to a Quaternion representing rotation around the Z-axis.
        """
        waypoint = Waypoint()
        
        # Extract x, y, z, and yaw from the position tuple
        if len(position) >= 4:
            x, y, z, yaw = position[:4]
        else:
            x, y, z = position[:3]
            yaw = 0.0 # Default yaw if not provided
        
        # Set position
        waypoint.pose = Pose()
        waypoint.pose.position.x = float(x)
        waypoint.pose.position.y = float(y) 
        waypoint.pose.position.z = float(z)

        # Correctly set yaw into a Quaternion (rotation only around Z-axis)
        yaw_rad = math.radians(yaw)
        waypoint.pose.orientation = Quaternion()
        waypoint.pose.orientation.x = 0.0
        waypoint.pose.orientation.y = 0.0
        waypoint.pose.orientation.z = math.sin(yaw_rad / 2)
        waypoint.pose.orientation.w = math.cos(yaw_rad / 2)
        
        waypoint.waypoint_index = int(waypoint_index)
        waypoint.coverage_rect = [] # Placeholder for coverage rectangle, if needed
        
        self.get_logger().info(f'🎯 Waypoint {waypoint_index}: ({x:.1f}, {y:.1f}, {z:.1f}) yaw={yaw:.1f}°')
        
        return waypoint
    
    def analyze_region_geometry(self, coordinates):
        """
        Analyzes the geometry of the input region (a set of 2D points).
        It finds the minimum bounding box and its orientation, returning
        the yaw angle (for aligning the 1280px camera axis with the long edge)
        and the dimensions of the bounding box.
        """
        points = np.array(coordinates)
        
        try:
            # Use ConvexHull for more robust bounding box calculation if scipy is available
            from scipy.spatial import ConvexHull
            hull = ConvexHull(points)
            hull_points = points[hull.vertices]
        except ImportError:
            # Fallback to using all points if scipy is not installed
            self.get_logger().warn('Scipy not found. Using all points for geometry analysis, ConvexHull not applied.')
            hull_points = points
        
        best_area = float('inf')
        best_orientation = None
        best_dims = None
        
        # Iterate through angles to find the minimum area bounding box
        angles = np.linspace(0, np.pi, 180) # 0 to 180 degrees in radians
        
        for angle in angles:
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            # Rotation matrix for rotating points by 'angle'
            rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            rotated_points = hull_points @ rotation_matrix.T # Apply rotation
            
            # Find min/max x and y after rotation
            x_min, x_max = np.min(rotated_points[:, 0]), np.max(rotated_points[:, 0])
            y_min, y_max = np.min(rotated_points[:, 1]), np.max(rotated_points[:, 1])
            
            # Note: Due to coordinate system definition, dim1 corresponds to Y-direction, dim2 to X-direction
            # This is based on the rotation logic where X becomes Y and Y becomes X after rotation for bounding box.
            y_dim = x_max - x_min  # Dimension along the rotated Y-axis
            x_dim = y_max - y_min  # Dimension along the rotated X-axis
            area = y_dim * x_dim
            
            # Update best bounding box if current area is smaller
            if area < best_area:
                best_area = area
                best_orientation = angle # Angle of the rotated X-axis with the original X-axis
                best_dims = (y_dim, x_dim)  # (Y-direction dimension, X-direction dimension)
        
        y_dim, x_dim = best_dims
        
        # Determine the long edge direction and the yaw angle for aligning 1280px with the long edge
        if y_dim > x_dim:
            # Y-direction is the long edge, 1280px camera axis aligns with this Y-direction
            long_dim, short_dim = y_dim, x_dim
            # The yaw for the long edge is the angle of the original Y-axis (which is now rotated by best_orientation)
            # plus 90 degrees to align the camera's 1280px axis (which is usually horizontal) with this long edge.
            yaw_for_long_edge = (best_orientation + np.pi/2) % (2*np.pi) 
        else:
            # X-direction is the long edge, 1280px camera axis aligns with this X-direction
            long_dim, short_dim = x_dim, y_dim
            # The yaw for the long edge is simply the best_orientation itself.
            yaw_for_long_edge = best_orientation % (2*np.pi) 
        
        # Convert to degrees and normalize to [-180, 180]
        yaw_for_long_edge_deg = np.degrees(yaw_for_long_edge)
        if yaw_for_long_edge_deg > 180:
            yaw_for_long_edge_deg -= 360
        
        self.get_logger().info(f'Region geometry analysis:')
        self.get_logger().info(f'  Long edge: {long_dim:.1f}mm, Short edge: {short_dim:.1f}mm')
        self.get_logger().info(f'  1280px aligned with long edge yaw: {yaw_for_long_edge_deg:.1f}°')
        
        # Return negative yaw because the rotation matrix used for analysis is for rotating points,
        # but for robot yaw, we need the opposite rotation to align the robot's frame.
        return -yaw_for_long_edge_deg, long_dim, short_dim

    def calculate_scanning_strategies(self, coordinates, object_height):
        """
        自适应扫描策略：修复面积计算和策略传递
        """
        yaw_for_long_edge, long_dim, short_dim = self.analyze_region_geometry(coordinates)
        
        # 修复：正确的面积计算 mm² -> cm²
        region_area = long_dim * short_dim  # mm²
        region_area_cm2 = region_area / 10000  # 正确转换：1cm² = 100mm²
        self.get_logger().info(f'Target region: {long_dim:.1f}×{short_dim:.1f}mm, Area: {region_area_cm2:.1f} cm²')
        
        strategies = []
        fov_h_deg, fov_v_deg = self.calculate_fov_from_intrinsics()
        fov_h_rad = math.radians(fov_h_deg / 2)
        fov_v_rad = math.radians(fov_v_deg / 2)
        
        # 自适应重叠率
        overlap_ratio = self.calculate_adaptive_overlap(region_area_cm2, long_dim, short_dim)
        
        for strategy_name, camera_1280_along, base_yaw in [
            ('1280px_along_long_edge', 'long', yaw_for_long_edge),
            ('1280px_along_short_edge', 'short', yaw_for_long_edge + 90)
        ]:
            
            strategy_yaw = self.find_closest_yaw(base_yaw, coordinates)
            
            # 计算所需高度
            if camera_1280_along == 'long':
                height_from_1280 = (long_dim / 2) / math.tan(fov_h_rad) + object_height
                height_from_720 = (short_dim / 2) / math.tan(fov_v_rad) + object_height
            else:
                height_from_1280 = (short_dim / 2) / math.tan(fov_h_rad) + object_height
                height_from_720 = (long_dim / 2) / math.tan(fov_v_rad) + object_height
            
            required_height = max(height_from_1280, height_from_720)
            required_height = self.validate_scan_height(required_height, object_height)
            
            # 智能点数计算
            scan_height, num_points, optimized_overlap = self.calculate_smart_scan_points(
                long_dim, short_dim, object_height, required_height, 
                camera_1280_along, overlap_ratio, fov_h_rad, fov_v_rad
            )
            
            strategies.append({
                'name': strategy_name,
                'yaw': strategy_yaw,
                'height': required_height,
                'scan_height': scan_height,
                'points': num_points,
                'long_dim': long_dim,
                'short_dim': short_dim,
                'camera_1280_along': camera_1280_along,
                'overlap_ratio': optimized_overlap,  # 使用优化后的重叠率
                'grid_x': 0,  # 将在路径生成时计算
                'grid_y': 0   # 将在路径生成时计算
            })
        
        # 选择最优策略
        best_strategy = min(strategies, key=lambda x: (x['points'], x['height']))
        
        self.get_logger().info(f'Selected strategy: {best_strategy["name"]}')
        self.get_logger().info(f'Final result: {best_strategy["points"]} points @ {best_strategy["scan_height"]:.0f}mm, overlap {best_strategy["overlap_ratio"]:.1%}')
        
        return best_strategy
    def calculate_adaptive_overlap(self, region_area_cm2, long_dim, short_dim):
        """
        根据区域大小和形状自适应计算重叠率
        """
        # 基础重叠率
        base_overlap = 0.25  # 降低基础重叠率到25%
        
        # 根据区域面积调整
        if region_area_cm2 < 100:  # 小于100cm²
            area_factor = 0.0
            self.get_logger().info(f'Small region ({region_area_cm2:.1f} cm²): minimal overlap')
        elif region_area_cm2 < 500:  # 100-500cm²
            area_factor = 0.05  # 减少增量
            self.get_logger().info(f'Medium region ({region_area_cm2:.1f} cm²): moderate overlap')
        else:  # 大于500cm²
            area_factor = 0.15  # 减少增量
            self.get_logger().info(f'Large region ({region_area_cm2:.1f} cm²): higher overlap')
        
        # 根据长宽比调整
        aspect_ratio = max(long_dim, short_dim) / min(long_dim, short_dim)
        if aspect_ratio > 3:
            aspect_factor = 0.1  # 减少
            self.get_logger().info(f'Very elongated object (ratio {aspect_ratio:.1f}): extra overlap')
        elif aspect_ratio > 2:
            aspect_factor = 0.05  # 减少
            self.get_logger().info(f'Elongated object (ratio {aspect_ratio:.1f}): slight extra overlap')
        else:
            aspect_factor = 0.0
            self.get_logger().info(f'Regular shape (ratio {aspect_ratio:.1f}): no extra overlap')
        
        final_overlap = base_overlap + area_factor + aspect_factor
        final_overlap = min(final_overlap, 0.5)  # 最大不超过50%
        
        return final_overlap


    def calculate_smart_scan_points(self, long_dim, short_dim, object_height, 
                                required_height, camera_1280_along, overlap_ratio,
                                fov_h_rad, fov_v_rad):
        """
        智能计算扫描点数：返回优化后的重叠率
        """
        # 单点扫描检查
        if required_height <= self.max_height:
            self.get_logger().info(f'Single point scan possible at {required_height:.0f}mm')
            return required_height, 1, overlap_ratio
        
        # 多点扫描：固定为430mm高度
        scan_height = 430  # 硬编码为430mm
        self.get_logger().info(f'Multi-point scan: using fixed height {scan_height}mm')
        
        def calculate_points(scan_height, overlap):
            effective_height = scan_height - object_height
            coverage_1280 = 2 * effective_height * math.tan(fov_h_rad) * (1 - overlap)
            coverage_720 = 2 * effective_height * math.tan(fov_v_rad) * (1 - overlap)
            
            if camera_1280_along == 'long':
                grid_long = max(1, int(math.ceil(long_dim / coverage_1280)))
                grid_short = max(1, int(math.ceil(short_dim / coverage_720)))
            else:
                grid_long = max(1, int(math.ceil(long_dim / coverage_720)))
                grid_short = max(1, int(math.ceil(short_dim / coverage_1280)))
            
            return grid_long * grid_short, (grid_long, grid_short)
        
        # 尝试优化重叠率
        best_points = float('inf')
        best_overlap = overlap_ratio
        best_grid = (1, 1)
        
        # 尝试不同的重叠率
        for test_overlap in [0.1, 0.15, 0.2, 0.25, 0.3, overlap_ratio]:
            points, grid = calculate_points(scan_height, test_overlap)
            if points < best_points:
                best_points = points
                best_overlap = test_overlap
                best_grid = grid
                self.get_logger().info(f'Better overlap {test_overlap:.1%}: {points} points (grid {grid[0]}×{grid[1]})')
        
        self.get_logger().info(f'Final optimization: {best_points} points @ {scan_height:.0f}mm, overlap {best_overlap:.1%}')
        
        return scan_height, best_points, best_overlap


    def find_closest_yaw(self, target_angle, coordinates):
        """
        Optimizes the yaw angle based on the robot's coordinate system and quadrant.
        It finds the closest valid yaw angle within the robot's reachable range
        for the given target region.
        """
        
        # Calculate the centroid of the input coordinates
        x_coords = [coord[0] for coord in coordinates]  # X-direction (forward/backward)
        y_coords = [coord[1] for coord in coordinates]  # Y-direction (left/right)
        x_mean = sum(x_coords) / len(x_coords)
        y_mean = sum(y_coords) / len(y_coords)
        
        # Determine the quadrant based on the centroid for robot arm's yaw range
        # This assumes specific robot arm kinematics or preferred operating zones.
        if x_mean >= 0 and y_mean <= 0:
            # Quadrant 1: Front-Right area
            yaw_range = (-180, 0)
            quadrant = "Front-Right"
        elif x_mean >= 0 and y_mean >= 0:
            # Quadrant 2: Front-Left area
            yaw_range = (0, 180)
            quadrant = "Front-Left"
        elif x_mean <= 0 and y_mean >= 0:
            # Quadrant 3: Rear-Left area
            yaw_range = (0, 180)
            quadrant = "Rear-Left"
        else:  # x_mean <= 0 and y_mean <= 0
            # Quadrant 4: Rear-Right area
            yaw_range = (-180, 0)
            quadrant = "Rear-Right"
        
        self.get_logger().info(f'Quadrant for yaw optimization: {quadrant} (Centroid: ({x_mean:.1f}, {y_mean:.1f}))')
        self.get_logger().info(f'Allowed yaw range: {yaw_range[0]}° to {yaw_range[1]}°')

        # Normalize the target angle to [-180, 180]
        target_angle = target_angle % 360
        if target_angle > 180:
            target_angle -= 360
        
        self.get_logger().info(f'Normalized target yaw: {target_angle:.1f}°')

        # Generate equivalent angles (e.g., target_angle, target_angle + 180, target_angle - 180)
        # because a 180-degree rotation of the camera can still cover the same area.
        equivalent_targets = [
            target_angle,
            target_angle + 180,
            target_angle - 180
        ]
        
        # Deduplicate and normalize equivalent targets to [-180, 180]
        normalized_targets = []
        for t in equivalent_targets:
            norm_t = t % 360
            if norm_t > 180:
                norm_t -= 360
            if norm_t not in normalized_targets:
                normalized_targets.append(norm_t)
        
        self.get_logger().info(f'Equivalent normalized target yaws: {", ".join([f"{t:.1f}°" for t in normalized_targets])}')

        # Filter for angles that fall within the allowed yaw range for the robot's quadrant
        valid_targets = [t for t in normalized_targets if yaw_range[0] <= t <= yaw_range[1]]
        
        if not valid_targets:
            # If no equivalent target angle is directly within the range,
            # select the closest angle to the range boundaries.
            self.get_logger().warn(f'No valid yaw found within range. Selecting closest boundary.')
            distances = []
            for t in normalized_targets:
                if t < yaw_range[0]:
                    dist = yaw_range[0] - t
                    closest = yaw_range[0]
                elif t > yaw_range[1]:
                    dist = t - yaw_range[1]
                    closest = yaw_range[1]
                else: # Should not happen if valid_targets is empty
                    dist = 0
                    closest = t
                distances.append((dist, closest))
            
            best_yaw = min(distances)[1]
            self.get_logger().info(f'Selected closest yaw: {best_yaw:.1f}°')
        else:
            # If valid targets exist, choose the one closest to the original target_angle
            best_yaw = min(valid_targets, key=lambda x: abs(x - target_angle))
            self.get_logger().info(f'Selected valid yaw closest to target: {best_yaw:.1f}°')
        
        return float(best_yaw)

    
def main(args=None):
    """
    Main function to initialize and run the ScanPlannerNode.
    """
    rclpy.init(args=args) # Initialize ROS2 client library
    node = ScanPlannerNode() # Create an instance of the node
    rclpy.spin(node) # Keep the node alive and processing callbacks
    node.destroy_node() # Destroy the node gracefully
    rclpy.shutdown() # Shutdown ROS2 client library

if __name__ == '__main__':
    main()