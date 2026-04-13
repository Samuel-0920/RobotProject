#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from vision_ai_interfaces.srv import ExecuteScan, ProcessStitching
from vision_ai_interfaces.msg import ImageData
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger, SetBool 
import cv2
import numpy as np
from cv_bridge import CvBridge
import time
import os
from datetime import datetime
import math
from std_msgs.msg import String

class ScanExecutorNode(Node):
    """
    A ROS2 node responsible for executing a scan plan, moving the robot arm
    to specified waypoints, capturing color and depth images, and
    initiating an image stitching process.
    """
    def __init__(self):
        super().__init__('scan_executor_node')
        
        # State management variables
        self.current_scan_plan = None
        self.execution_active = False
        self.captured_images = []
        self.current_waypoint_index = 0
        
        # CV Bridge for converting ROS Image messages to OpenCV images
        self.bridge = CvBridge()
        self.current_arm_position = None # Current position of the robot arm
        self.target_position = None      # Target position for movement
        self.movement_completed = False  # Flag to indicate movement completion
        self.output_dir = None           # Directory to save captured images
        
        # Camera intrinsic parameters (obtained from scan_planner_node)
        self.camera_intrinsics = {
            'fx': 912.694580078125,
            'fy': 910.309814453125,
            'cx': 640,  # Center X (1280/2)
            'cy': 360   # Center Y (720/2)
        }
        
        # Create the service for executing scans
        self.execute_service = self.create_service(
            ExecuteScan, 
            'execute_scan', 
            self.execute_scan_callback
        )
        
        # Subscriber for robot arm's current pose
        self.current_pose_sub = self.create_subscription(
            PoseStamped, 
            '/xarm/current_pose', 
            self.pose_callback, 
            10
        )
        
        # Subscriber for color images from the camera
        self.color_image_sub = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.color_image_callback,
            10
        )
        
        # Subscriber for depth images from the camera
        self.depth_image_sub = self.create_subscription(
            Image,
            '/camera/depth/image_raw',
            self.depth_image_callback,
            10
        )
        
        # Publisher for sending target poses to the robot arm
        self.target_pose_pub = self.create_publisher(
            PoseStamped,
            '/xarm/target_pose',
            10
        )
        
        # Service client for robot arm control (e.g., enable/disable)
        self.arm_enable_client = self.create_client(
            SetBool, 
            '/xarm/enable'
        )
        
        # Store the latest received images
        self.latest_color_image = None
        self.latest_depth_image = None
        
        self.get_logger().info('The scan execution node has been started (supports depth map acquisition)')
        
        # Wait for necessary hardware services to be available
        self._wait_for_hardware_services()

    def _wait_for_hardware_services(self):
        """Waits for the necessary hardware services (e.g., robot arm enable) to be available."""
        self.get_logger().info('Waiting for hardware service...')
        
        # Wait for the robot arm enable service
        if not self.arm_enable_client.wait_for_service(timeout_sec=15.0):
            self.get_logger().warn('The robot enabling service connection timed out, but continues to run.')
        else:
            self.get_logger().info('Robot arm enable service connected.')
        
        self.get_logger().info('All hardware services are ready.')

    def execute_scan_callback(self, request, response):
        """
        Callback for the execute_scan service.
        Initiates the scanning process based on the provided scan plan.
        """
        try:
            if self.execution_active:
                response.success = False
                response.message = "Scan is already in progress, please wait for completion."
                return response
            
            # Store the received scan plan
            self.current_scan_plan = request.scan_plan
            
            # Create a unique output directory for this scan
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.output_dir = f"scan_output_{timestamp}"
            os.makedirs(self.output_dir, exist_ok=True)
            
            self.get_logger().info(f'Starting scan execution: {len(self.current_scan_plan.waypoints)} waypoints.')
            self.get_logger().info(f'Output directory: {self.output_dir}')
            
            # Start asynchronous execution
            self.execution_active = True
            self.captured_images = []
            self.current_waypoint_index = 0
            
            # Create a timer to step through the scan sequence
            self.scan_timer = self.create_timer(0.5, self.scan_sequence_step)
            
            response.success = True
            response.message = f"Scan execution started, {len(self.current_scan_plan.waypoints)} waypoints expected."
            
        except Exception as e:
            response.success = False
            response.message = f"Scan execution failed: {str(e)}"
            self.get_logger().error(f'Scan execution error: {e}')
        
        return response

    def color_image_callback(self, msg):
        """Callback for receiving color image messages."""
        self.latest_color_image = msg

    def depth_image_callback(self, msg):
        """Callback for receiving depth image messages."""
        self.latest_depth_image = msg

    def scan_sequence_step(self):
        """
        Executes a single step in the scanning sequence.
        Moves to the next waypoint, waits for movement, then captures images.
        """
        if not self.execution_active or not self.current_scan_plan:
            return
        
        # Check if all waypoints have been processed
        if self.current_waypoint_index >= len(self.current_scan_plan.waypoints):
            self._finish_scan_execution()
            return
        
        # Only execute the next waypoint if not currently busy with a waypoint
        if not getattr(self, 'waypoint_busy', False):
            waypoint = self.current_scan_plan.waypoints[self.current_waypoint_index]
            self.waypoint_busy = True # Set flag to indicate busy
            self._execute_waypoint(waypoint)

    def _execute_waypoint(self, waypoint):
        """
        Executes movement to a single waypoint and sets up a timer
        to check for movement completion.
        """
        try:
            self.get_logger().info(f'🎯 Executing waypoint {self.current_waypoint_index + 1}/{len(self.current_scan_plan.waypoints)}')
            
            # Store the target position for movement completion detection
            self.target_position = [
                waypoint.pose.position.x,
                waypoint.pose.position.y, 
                waypoint.pose.position.z
            ]
            
            target_pose = PoseStamped()
            target_pose.header.stamp = self.get_clock().now().to_msg()
            target_pose.header.frame_id = "base_link" # Assuming "base_link" as the robot's base frame
            target_pose.pose = waypoint.pose
            
            self.target_pose_pub.publish(target_pose) # Publish the target pose
            
            # Start movement completion detection
            self.waypoint_data = waypoint
            self.movement_start_time = time.time()
            self.movement_completed = False
            
            # Create a timer to check movement completion every 0.5 seconds
            self.movement_check_timer = self.create_timer(0.5, self.check_movement_completion)
            
        except Exception as e:
            self.get_logger().error(f'Waypoint execution failed: {e}')
            self.waypoint_busy = False # Release busy flag on error

    def check_movement_completion(self):
        """
        Checks if the robot arm has reached the target position.
        Includes a timeout mechanism.
        """
        if not hasattr(self, 'target_position') or self.target_position is None:
            return
            
        # Timeout check (50 seconds)
        if time.time() - self.movement_start_time > 50:
            self.get_logger().warn('Movement timed out, forcing continuation.')
            self._on_movement_completed()
            return
        
        # Check if the target position has been reached
        if self.current_arm_position is not None:
            distance = math.sqrt(
                (self.current_arm_position[0] - self.target_position[0])**2 +
                (self.current_arm_position[1] - self.target_position[1])**2 +
                (self.current_arm_position[2] - self.target_position[2])**2
            )
            
            # Arrival threshold: 5mm
            if distance < 5.0:
                self.get_logger().info(f'✅ Reached target position, distance: {distance:.1f}mm.')
                self._on_movement_completed()
            else:
                elapsed = time.time() - self.movement_start_time
                self.get_logger().info(f'🚀 Moving... Distance to target: {distance:.1f}mm, Time elapsed: {elapsed:.1f}s.')

    def _on_movement_completed(self):
        """
        Handles actions to be performed once movement to a waypoint is complete,
        such as stopping the movement check timer and initiating image capture.
        """
        # Stop the movement check timer
        if hasattr(self, 'movement_check_timer'):
            self.movement_check_timer.destroy()
        
        self.movement_completed = True
        
        self.get_logger().info('Movement completed, starting image capture.')
        self._capture_image_at_waypoint(self.waypoint_data)
        
        # Advance to the next waypoint
        self.current_waypoint_index += 1
        self.waypoint_busy = False # Release busy flag
        self.target_position = None # Clear target position

    def pose_callback(self, msg):
        """
        Callback for robot arm pose updates.
        Used to track the current position for movement completion detection.
        """
        try:
            # Save the current position for movement completion detection
            self.current_arm_position = [
                msg.pose.position.x,
                msg.pose.position.y, 
                msg.pose.position.z
            ]
        except Exception as e:
            self.get_logger().error(f'Pose callback error: {e}')

    def _capture_image_at_waypoint(self, waypoint):
        """
        Captures color and depth images at the current waypoint.
        """
        try:
            self.get_logger().info(f'📸 Starting image capture (Color + Depth).')
            
            # Check if both color and depth images are available
            if not self.latest_color_image or not self.latest_depth_image:
                self.get_logger().error('Color or depth image not available.')
                return
            
            self.current_waypoint_for_image = waypoint
            self.process_captured_images(self.latest_color_image, self.latest_depth_image)
                
        except Exception as e:
            self.get_logger().error(f'Image capture failed: {e}')


    def process_captured_images(self, color_msg, depth_msg):
            """
            Processes the captured color and depth images, converting them
            to OpenCV format, saving them, and storing relevant metadata.
            Includes error handling for cv_bridge conversions.
            """
            try:
                # Fix: Safe image conversion
                try:
                    # For bgr8 encoding, convert directly then transform color
                    color_cv_image = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding="passthrough")
                    
                    # Check image channels and convert
                    if len(color_cv_image.shape) == 3 and color_cv_image.shape[2] == 3:
                        # BGR -> RGB
                        color_rgb = cv2.cvtColor(color_cv_image, cv2.COLOR_BGR2RGB)
                    else:
                        self.get_logger().error(f'Unsupported color image format: {color_cv_image.shape}')
                        return
                        
                    self.get_logger().info(f'Color image conversion successful: {color_rgb.shape}')
                        
                except Exception as color_error:
                    self.get_logger().error(f'Color image conversion failed: {color_error}')
                    # Fallback: Try a different conversion method
                    try:
                        color_cv_image = self.bridge.imgmsg_to_cv2(color_msg)
                        color_rgb = cv2.cvtColor(color_cv_image, cv2.COLOR_BGR2RGB)
                        self.get_logger().info('Color image conversion successful using fallback.')
                    except Exception as fallback_error:
                        self.get_logger().error(f'All color image conversion attempts failed: {fallback_error}')
                        return
                
                # 深度图像处理 - 修复数据截断问题
                try:
                    self.get_logger().info(f'Received depth message encoding: {depth_msg.encoding}')
                    self.get_logger().info(f'Received depth message step: {depth_msg.step}')
                    self.get_logger().info(f'Expected step for 16UC1: {depth_msg.width * 2}')
                    
                    # 直接从原始字节数据重建16位深度图像
                    if depth_msg.step == depth_msg.width * 2:  # 正确的16位步长
                        raw_data = np.frombuffer(depth_msg.data, dtype=np.uint16)
                        depth_raw = raw_data.reshape((depth_msg.height, depth_msg.width))
                        self.get_logger().info('Using direct uint16 reconstruction')
                    else:
                        # 如果步长不对，可能需要重新解释数据
                        raw_data = np.frombuffer(depth_msg.data, dtype=np.uint8)
                        # 将连续的两个uint8重新组合为uint16
                        depth_raw = raw_data.view(np.uint16).reshape((depth_msg.height, depth_msg.width))
                        self.get_logger().info('Using uint8->uint16 reinterpretation')
                    
                    self.get_logger().info(f'Fixed depth image: {depth_raw.shape}, dtype: {depth_raw.dtype}')
                    self.get_logger().info(f'Fixed depth range: min={depth_raw.min()}, max={depth_raw.max()}')
                    
                except Exception as depth_error:
                    self.get_logger().error(f'Depth image reconstruction failed: {depth_error}')
                    # 降级到cv_bridge方法
                    try:
                        depth_raw = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
                        if depth_raw.dtype != np.uint16:
                            depth_raw = depth_raw.astype(np.uint16)
                        self.get_logger().info('Fallback to cv_bridge conversion')
                    except Exception as fallback_error:
                        self.get_logger().error(f'All depth conversion methods failed: {fallback_error}')
                        return
                
                # 验证深度数据合理性
                if depth_raw.max() < 50:  # 深度值应该至少有几百毫米
                    self.get_logger().warn(f'Depth values seem too small: max={depth_raw.max()}')
                
                # 确保有效的图像数据
                if color_rgb is None or color_rgb.size == 0:
                    self.get_logger().error('Invalid color image data.')
                    return
                    
                if depth_raw is None or depth_raw.size == 0:
                    self.get_logger().error('Invalid depth image data.')
                    return
                
                # Create a depth heatmap for visualization and saving
                depth_normalized = cv2.normalize(depth_raw, None, 0, 255, cv2.NORM_MINMAX)
                depth_heatmap = cv2.applyColorMap(depth_normalized.astype(np.uint8), cv2.COLORMAP_JET)

                
                # Define filenames for saving
                color_filename = os.path.join(self.output_dir, f"color_waypoint_{self.current_waypoint_index+1:03d}.jpg")
                depth_heatmap_filename = os.path.join(self.output_dir, f"depth_heatmap_waypoint_{self.current_waypoint_index+1:03d}.jpg")
                depth_raw_filename = os.path.join(self.output_dir, f"depth_raw_waypoint_{self.current_waypoint_index+1:03d}.npy")
                
                # Save images (OpenCV saves BGR by default for JPG)
                cv2.imwrite(color_filename, cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR))
                cv2.imwrite(depth_heatmap_filename, depth_heatmap)
                np.save(depth_raw_filename, depth_raw)
                
                # Construct waypoint pose information in 6D format
                waypoint_pose_6d = {
                    'x': self.current_waypoint_for_image.pose.position.x,
                    'y': self.current_waypoint_for_image.pose.position.y, 
                    'z': self.current_waypoint_for_image.pose.position.z,
                    'qx': self.current_waypoint_for_image.pose.orientation.x,
                    'qy': self.current_waypoint_for_image.pose.orientation.y,
                    'qz': self.current_waypoint_for_image.pose.orientation.z,
                    'qw': self.current_waypoint_for_image.pose.orientation.w
                }
                
                # Store enhanced image data with metadata
                image_data = {
                    'waypoint_index': self.current_waypoint_index,
                    'waypoint': self.current_waypoint_for_image,
                    'color_filename': color_filename,
                    'depth_heatmap_filename': depth_heatmap_filename,
                    'depth_raw_filename': depth_raw_filename,
                    'timestamp': self.get_clock().now().to_msg(),
                    
                    # Image data (in RGB format for color, raw for depth)
                    'color_image': color_rgb,
                    'depth_raw': depth_raw,
                    'depth_heatmap': cv2.cvtColor(depth_heatmap, cv2.COLOR_BGR2RGB), # Store RGB for consistency
                    
                    # Camera and geometric information
                    'camera_intrinsics': self.camera_intrinsics,
                    'object_height': self.current_scan_plan.object_height,
                    'scan_height': self.current_scan_plan.scan_height,
                    'scan_area_corners': [
                        {'x': p.x, 'y': p.y} for p in self.current_scan_plan.scan_region
                    ],
                    
                    # Full 6D pose of the waypoint
                    'waypoint_pose_6d': waypoint_pose_6d
                }
                
                self.captured_images.append(image_data)
                
                self.get_logger().info(f'✅ Image saved: waypoint_{self.current_waypoint_index+1:03d} (Color + Depth).')
                self.get_logger().info(f'📁 Color: {color_filename}')
                self.get_logger().info(f'🔥 Depth Heatmap: {depth_heatmap_filename}')
                self.get_logger().info(f'🗂️  Raw Depth: {depth_raw_filename}')
                
            except Exception as e:
                self.get_logger().error(f'Image processing failed: {e}')
                import traceback # Print full traceback for debugging
                traceback.print_exc()

    def _calculate_grid_layout(self):
        """
        Calculates and returns information about the grid layout of waypoints,
        if the scan plan indicates a multi-point scan.
        """
        try:
            if not self.current_scan_plan.waypoints:
                return None
            
            # Extract all waypoint positions
            positions = []
            for wp in self.current_scan_plan.waypoints:
                positions.append((wp.pose.position.x, wp.pose.position.y))
            
            positions = np.array(positions)
            
            # Analyze grid structure by finding unique X and Y coordinates
            unique_x = np.unique(np.round(positions[:, 0], 1))
            unique_y = np.unique(np.round(positions[:, 1], 1))
            
            grid_layout = {
                'num_rows': len(unique_x),
                'num_cols': len(unique_y),
                'x_positions': sorted(unique_x.tolist()),
                'y_positions': sorted(unique_y.tolist()),
                'is_regular_grid': len(positions) == len(unique_x) * len(unique_y),
                'total_waypoints': len(self.current_scan_plan.waypoints)
            }
            
            # Analyze for zigzag pattern if it's a regular grid
            if grid_layout['is_regular_grid']:
                grid_layout['snake_pattern'] = True # Assuming zigzag for regular grids
                grid_layout['current_grid_pos'] = self._find_grid_position(
                    self.current_waypoint_for_image.pose.position.x,
                    self.current_waypoint_for_image.pose.position.y,
                    grid_layout
                )
            
            return grid_layout
            
        except Exception as e:
            self.get_logger().error(f'Grid layout calculation failed: {e}')
            return None

    def _find_grid_position(self, x, y, grid_layout):
        """
        Finds the grid position (row, column) of a given waypoint.
        """
        try:
            # Find the closest grid position
            x_idx = None
            y_idx = None
            
            min_x_dist = float('inf')
            for i, grid_x in enumerate(grid_layout['x_positions']):
                dist = abs(x - grid_x)
                if dist < min_x_dist:
                    min_x_dist = dist
                    x_idx = i
            
            min_y_dist = float('inf')
            for i, grid_y in enumerate(grid_layout['y_positions']):
                dist = abs(y - grid_y)
                if dist < min_y_dist:
                    min_y_dist = dist
                    y_idx = i
            
            return (x_idx, y_idx) if x_idx is not None and y_idx is not None else None
            
        except Exception as e:
            self.get_logger().error(f'Grid position calculation failed: {e}')
            return None

    def _calculate_overlap_info(self):
        """
        Calculates and returns information about camera overlap between waypoints.
        """
        try:
            # Calculate actual FOV coverage dimensions
            scan_height = self.current_scan_plan.scan_height
            object_height = self.current_scan_plan.object_height
            effective_height = scan_height - object_height
            
            # Camera FOV (from scan_planner)
            fx, fy = self.camera_intrinsics['fx'], self.camera_intrinsics['fy']
            camera_width, camera_height = 1280, 720
            
            fov_h = 2 * math.atan(camera_width / (2 * fx))
            fov_v = 2 * math.atan(camera_height / (2 * fy))
            
            # Actual coverage dimensions in millimeters
            fov_coverage_width = 2 * effective_height * math.tan(fov_h / 2)
            fov_coverage_height = 2 * effective_height * math.tan(fov_v / 2)
            
            overlap_info = {
                'effective_height': effective_height,
                'fov_coverage_width_mm': fov_coverage_width,
                'fov_coverage_height_mm': fov_coverage_height,
                'fov_h_degrees': math.degrees(fov_h),
                'fov_v_degrees': math.degrees(fov_v)
            }
            
            # If there are adjacent waypoints, calculate actual overlap ratio (simplified)
            current_pos = (
                self.current_waypoint_for_image.pose.position.x,
                self.current_waypoint_for_image.pose.position.y
            )
            
            adjacent_overlaps = []
            for wp in self.current_scan_plan.waypoints:
                wp_pos = (wp.pose.position.x, wp.pose.position.y)
                if wp_pos != current_pos: # Don't compare with itself
                    distance = math.sqrt((wp_pos[0] - current_pos[0])**2 + (wp_pos[1] - current_pos[1])**2)
                    
                    # Simplified overlap ratio calculation (assuming movement primarily along one axis)
                    if distance < fov_coverage_width: # Check for overlap in the dominant direction
                        x_overlap = max(0, fov_coverage_width - abs(wp_pos[1] - current_pos[1])) # Simplified to Y-axis distance
                        overlap_ratio_x = x_overlap / fov_coverage_width
                        adjacent_overlaps.append({
                            'waypoint_pos': wp_pos,
                            'distance': distance,
                            'overlap_ratio_x': overlap_ratio_x
                        })
            
            overlap_info['adjacent_overlaps'] = adjacent_overlaps
            
            return overlap_info
            
        except Exception as e:
            self.get_logger().error(f'Overlap information calculation failed: {e}')
            return None

    def _finish_scan_execution(self):
        """
        Finalizes the scan execution process, stopping timers and
        initiating the stitching process.
        """
        try:
            # Stop the scan timer
            if hasattr(self, 'scan_timer'):
                self.scan_timer.destroy()
            
            self.execution_active = False
            
            self.get_logger().info(f'Scan execution completed!')
            self.get_logger().info(f'Number of images captured: {len(self.captured_images)}')
            self.get_logger().info(f'Output directory: {self.output_dir}')
            
            # 🆕 Fix: Process all scans (single or multi-point) through the stitching service
            self._start_stitching_process()
                
        except Exception as e:
            self.get_logger().error(f'Scan completion handling failed: {e}')

    def _start_stitching_process(self):
        """修复图像数据传递，确保深度文件路径正确"""
        try:
            self.get_logger().info('Starting smart image stitching process...')
            
            stitch_client = self.create_client(ProcessStitching, 'process_stitching')
            
            if not stitch_client.wait_for_service(timeout_sec=5.0):
                self.get_logger().error('Stitching service not available.')
                self._trigger_detection_for_fallback()
                return
            
            request = ProcessStitching.Request()
            request.scan_plan = self.current_scan_plan
            request.output_directory = self.output_dir
            
            # 🆕 改进的图像数据准备 - 确保深度文件信息正确传递
            for img_data in self.captured_images:
                image_msg = ImageData()
                
                # 🆕 只传递文件名，不传递完整路径
                color_full_path = img_data.get('color_filename', '')
                depth_full_path = img_data.get('depth_raw_filename', '')
                
                # 提取文件名（去掉路径）
                color_filename = os.path.basename(color_full_path) if color_full_path else f'color_waypoint_{img_data["waypoint_index"]:03d}.jpg'
                depth_filename = os.path.basename(depth_full_path) if depth_full_path else f'depth_raw_waypoint_{img_data["waypoint_index"]:03d}.npy'
                
                image_msg.filename = color_filename
                image_msg.timestamp = img_data['timestamp']
                image_msg.waypoint = img_data['waypoint']
                
                # 🆕 添加自定义字段来传递深度文件名（如果接口支持）
                # 如果不支持，需要修改接口定义
                # image_msg.depth_filename = depth_filename
                
                # Convert OpenCV image to ROS Image message
                img_rgb = img_data['color_image']
                img_msg = self.bridge.cv2_to_imgmsg(img_rgb, "rgb8")
                image_msg.image = img_msg
                
                request.image_data.append(image_msg)
                
                self.get_logger().info(f'📤 准备图像数据: waypoint_{img_data["waypoint_index"]}')
                self.get_logger().info(f'  - Color文件: {color_filename}')
                self.get_logger().info(f'  - Depth文件: {depth_filename}')
                self.get_logger().info(f'  - Depth完整路径: {depth_full_path}')
                self.get_logger().info(f'  - 文件存在: {os.path.exists(depth_full_path)}')
            
            scan_type_msg = "Single-point scan" if len(self.captured_images) == 1 else "Multi-point scan"
            self.get_logger().info(f'📤 发送{scan_type_msg}数据到拼接服务: {len(request.image_data)} 张图像')
            
            future = stitch_client.call_async(request)
            future.add_done_callback(self.stitching_response_callback)
            
        except Exception as e:
            self.get_logger().error(f'拼接进程启动失败: {e}')
            import traceback
            traceback.print_exc()
            self._trigger_detection_for_fallback()

    def _trigger_detection_for_fallback(self):
        """
        Fallback method to trigger detection directly if stitching service is unavailable or fails.
        (Implementation for this fallback is not provided in the original code,
        this is a placeholder for future expansion if needed).
        """
        self.get_logger().warn("Stitching service unavailable or failed. Triggering direct detection (fallback not fully implemented).")
        # TODO: Implement actual direct detection logic here if stitching fails
        pass


    def stitching_response_callback(self, future):
        """
        Callback for the stitching service response.
        Logs the success or failure of the stitching process.
        """
        try:
            response = future.result()
            if response.success:
                self.get_logger().info(f'Stitching completed: {response.message}')
                self.get_logger().info(f'Result file: {response.result.output_path}')
            else:
                self.get_logger().error(f'Stitching failed: {response.message}')
        except Exception as e:
            self.get_logger().error(f'Stitching response processing failed: {e}')


def main(args=None):
    """
    Main function to initialize and run the ScanExecutorNode.
    """
    rclpy.init(args=args) # Initialize ROS2 client library
    node = ScanExecutorNode() # Create an instance of the node
    rclpy.spin(node) # Keep the node alive and processing callbacks
    node.destroy_node() # Destroy the node gracefully
    rclpy.shutdown() # Shutdown ROS2 client library

if __name__ == '__main__':
    main()
