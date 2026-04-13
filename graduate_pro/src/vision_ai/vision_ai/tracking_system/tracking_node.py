#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
import json
import os
import cv2
import numpy as np
from datetime import datetime
import threading
import time
import math
# 导入Track模块组件
from .enhanced_tracker import EnhancedTracker
from .utils.feedback_collector import FeedbackCollector
from .utils.data_recorder import DataRecorder
from .utils.user_profile_manager import UserProfileManager
from .utils.tracking_visualizer import TrackingVisualizer
from .adaptive_learning.online_learner import OnlineLearner
from ..detection.enhanced_detection_pipeline import EnhancedDetectionPipeline
from typing import Dict, List, Optional, Tuple
from .filters.kalman_tracker import TrackingStabilityManager
class TrackingNode(Node):
    def __init__(self):
        super().__init__('tracking_node')
        self.tracking_active = False
        self.current_target_id = None
        self.current_scan_dir = None
        self.current_user_id = None
        self.movement_complete = True
        self.tracking_count = 0
        self.max_tracking_steps = 30
        
        self.waiting_for_user_confirmation = False
        self.current_tracking_result = None
        self.user_feedback_received = False
        self.command_input_thread = None
        
        self.initial_yaw = None  # 记录初始yaw角度
        self.max_yaw_change = 90.0  # 最大yaw变化限制（度）
        
        #  追踪参数记录
        self.tracking_parameters_history = []
        
        # imagedata缓存
        self.latest_rgb_image = None
        self.latest_depth_image = None
        self.latest_camera_info = None
        
        # 核心组件（Initialized时都为None）
        self.tracker = None
        self.feedback_collector = None
        self.data_recorder = None
        self.user_profile_manager = None
        self.visualizer = None
        self.online_learner = None
        self.detection_pipeline = None
        self._last_waypoint_data = None
        # 线程锁和TCP位姿
        self.tracking_lock = threading.Lock()
        self.current_tcp_pose = None
        self.previous_yaw = None  #  用于yaw角度连续性处理
        # 系统状态标志
        self.system_initialized = False
        self.waiting_for_detection_complete = True
        self.hybrid_mode_enabled = True  # 启用混合相似度模式
        self.auto_success_transition_step = 5  # 第10步开始考虑自动判断
        self.consecutive_high_confidence = 0    # 连续高置信度计数
        self.debug_data_flow = True  # 启用data流调试
        # Initialized基础组件
        from cv_bridge import CvBridge
        self.bridge = CvBridge()
        
        # 订阅机械臂状态获取TCP位姿
        self.tcp_pose_sub = self.create_subscription(
            PoseStamped, '/xarm/current_pose', self.tcp_pose_callback, 10
        )
        
        # ROS接口设置
        self._setup_ros_interfaces()
        
        # 系统启动检查
        self._perform_startup_checks()
        
        self.get_logger().info('Track node startup completed - waiting for detection completion signal')
    
    def _setup_ros_interfaces(self):
        """设置ROS接口"""
        # 订阅检测完成信号
        self.detection_complete_sub = self.create_subscription(
            String, '/detection_complete', self.detection_complete_callback, 10
        )
        
        # 订阅相机data
        self.rgb_image_sub = self.create_subscription(
            Image, '/camera/color/image_raw', self.rgb_image_callback, 10
        )
        self.depth_image_sub = self.create_subscription(
            Image, '/camera/depth/image_raw', self.depth_image_callback, 10
        )
        
        # 订阅Robotic arm movement completed信号
        self.movement_complete_sub = self.create_subscription(
            String, '/xarm/movement_complete', self.movement_complete_callback, 10
        )
        
        # 发布追踪结果给机械臂
        self.tracking_result_pub = self.create_publisher(
            PoseStamped, '/xarm/target_pose', 10
        )
        self.tracking_status_pub = self.create_publisher(
            String, '/tracking_status', 10
        )
        
        # 发布追踪完成信号
        self.tracking_complete_pub = self.create_publisher(
            String, '/tracking_complete', 10
        )
        #  发布grasp触发信号
        self.grasp_trigger_pub = self.create_publisher(
            String, '/grasp_trigger', 10
        )

    def _perform_startup_checks(self):
        """执行启动检查"""
        try:
            self.get_logger().info(' Performing system startup checks...')
            
            # 检查关键文件路径
            key_files = [
                '/home/qi/下载/best2.pt',
                '/home/qi/ros2_ws/src/vision_ai/models/sam2/sam2_hiera_large.pt',
                '/home/qi/ros2_ws/src/vision_ai/vision_ai/detection/config/enhanced_detection_config.json'
            ]
            
            all_files_exist = True
            for file_path in key_files:
                if os.path.exists(file_path):
                    self.get_logger().info(f'   {os.path.basename(file_path)}')
                else:
                    self.get_logger().warn(f'   {file_path} not exist')
                    all_files_exist = False
            
            # 检查ROS话题连接
            self.get_logger().info(' Checking ROS topic connection...')
            self.get_logger().info(' - Waiting for detection completion signal: /detection_complete')
            self.get_logger().info(' - Waiting for camera data: /camera/color/image_raw, /camera/depth/image_raw')
            self.get_logger().info(' - Waiting for robot arm pose: /xarm/current_pose')
            
            if all_files_exist:
                self.get_logger().info('Boot checks complete - all required files present')
            else:
                self.get_logger().warn('Startup check complete - some files are missing, which may affect functionality')
            
            self.get_logger().info('The system is ready, waiting for the detection completion signal to start tracking...')
            
        except Exception as e:
            self.get_logger().error(f'Startup check failed: {e}')

    def _extract_user_id(self, scan_dir_path: str) -> str:
        """从扫描目录路径提取UserID"""
        try:
            # /home/qi/ros2_ws/scan_output_20250728_214915 -> qi
            path_parts = scan_dir_path.split('/')
            for i, part in enumerate(path_parts):
                if part == 'home' and i + 1 < len(path_parts):
                    return path_parts[i + 1]
            
            # 如果提取 fail ，使用默认值
            return 'default_user'
            
        except Exception as e:
            self.get_logger().error(f'Failed to retrieve user ID: {e}')
            return 'default_user'
    
    def detection_complete_callback(self, msg):
        try:
            data = json.loads(msg.data)
            
            self.get_logger().info('After receiving the detection completion signal, the tracking system started to be initialized...')
            
            # 检查是否是增强检测结果
            if not data.get('enhanced_detection', False):
                self.get_logger().info('Receive normal detection results and skip enhanced tracking')
                return
            
            self.current_scan_dir = data.get('scan_directory')
            if not self.current_scan_dir:
                self.get_logger().error('Unable to obtain scan directory')
                return

            # 提取UserID
            self.current_user_id = self._extract_user_id(self.current_scan_dir)
            self.get_logger().info(f' UserID: {self.current_user_id}')
            
            #  标记不再 wait 检测完成信号
            self.waiting_for_detection_complete = False
            
            # Initialized追踪系统
            success = self._initialize_tracking_system(data)
            
            if success:
                #  只有在系统完全Initialized成功后才激活追踪
                self.system_initialized = True
                self.tracking_active = True
                self.get_logger().info(' The tracking system has been initialized successfully and tracking has started.')
            else:
                self.get_logger().error('Tracking system initialization failed')
                
        except Exception as e:
            self.get_logger().error(f'Failed to process the detection completion signal: {e}')
            import traceback
            traceback.print_exc()

    def _publish_grasp_trigger(self, filename: str, coordinate: dict):
        """发布grasp触发信号"""
        try:
            grasp_data = {
                'filename': filename,
                'coordinate': coordinate,
                'timestamp': datetime.now().isoformat(),
                'target_id': self.current_target_id,
                'trigger_reason': 'conditions_met'
            }
            
            trigger_msg = String()
            trigger_msg.data = json.dumps(grasp_data)
            self.grasp_trigger_pub.publish(trigger_msg)
            
            self.get_logger().info(f'Crawl trigger signal issued: {filename}')            
        except Exception as e:
            self.get_logger().error(f'Failed to issue the crawl trigger signal: {e}')

    def _initialize_tracking_system(self, detection_data: dict) -> bool:
        """Initialized追踪系统 - 完全使用参考库data"""
        try:
            self.get_logger().info('Starting initialization of tracking system...')
            
            # 1. 检查参考特征库文件
            reference_library_file = detection_data.get('reference_library_file')
            if not reference_library_file or not os.path.exists(reference_library_file):
                self.get_logger().error(f'The reference signature library file does not exist: {reference_library_file}')
                return False
            
            self.get_logger().info(f'Reference signature library file: {reference_library_file}')
            
            # 2.  配置文件路径处理（保持不变）
            possible_config_paths = [
                '/home/qi/ros2_ws/src/vision_ai/vision_ai/tracking_system/config/tracking_config.json',
                os.path.join(os.path.dirname(__file__), 'config', 'tracking_config.json'),
                os.path.join(os.path.expanduser('~'), 'ros2_ws', 'src', 'vision_ai', 'vision_ai', 'tracking_system', 'config', 'tracking_config.json')
            ]
            
            config_path = None
            for path in possible_config_paths:
                if os.path.exists(path):
                    config_path = path
                    self.get_logger().info(f'Find the tracking configuration file: {config_path}')
                    break
            
            # 3. Initialized追踪器
            self.tracker = EnhancedTracker(config_path)
            
            # 4. 加载参考特征库
            if not self.tracker.load_reference_features(reference_library_file):
                self.get_logger().error('Failed to load reference feature library')
                return False
            
            self.get_logger().info(' Reference feature library loading completed')
            
            # 5. 读取User选择的Tracking Target
            target_id = self._load_tracking_target()
            if not target_id:
                self.get_logger().error(' Unable to load tracking target')
                return False
            
            self.current_target_id = target_id
            self.get_logger().info(f' Tracking Target: {target_id}')
            
            # 6.  验证参考库中的高度data（不需要缓存）
            if target_id in self.tracker.reference_library:
                ref_entry = self.tracker.reference_library[target_id]
                spatial_features = ref_entry.get('features', {}).get('spatial', {})
                
                height_mm = spatial_features.get('height_mm')
                background_z = spatial_features.get('background_world_z')
            else:
                self.get_logger().error(f'Target {target_id} is not in the reference library')
                return False
            
            # 7. 选择Tracking Target
            if not self.tracker.select_tracking_target(target_id):
                self.get_logger().error(f'Failed to select tracking target: {target_id}')
                return False
            
            self.get_logger().info('Tracking target selection completed')
            
            # 8. Initialized其他组件
            self._initialize_auxiliary_components()
            
            # 9. Initialized检测管道
            self._initialize_detection_pipeline_fixed()

            if self.tracker and hasattr(self.tracker, 'load_detection_history'):
                history_loaded = self.tracker.load_detection_history()
                if history_loaded:
                    self.get_logger().info('Detection history loaded, mixed similarity mode enabled')
                else:
                    self.get_logger().info('The detection history is empty and will be accumulated from the beginning')
            
            #  设置tracker的scan目录（用于保存histroy）
            if hasattr(self.tracker, 'current_scan_dir'):
                self.tracker.current_scan_dir = self.current_scan_dir
            
            return True
            
        except Exception as e:
            self.get_logger().error(f"Failed to initialize tracking system: {e}")
            return False
        
    def _verify_system_components_no_cache(self) -> bool:
        """验证系统组件是否正常 - 不检查缓存"""
        try:
            
            # 检查核心组件
            components = [
                ('tracker', self.tracker),
                ('detection_pipeline', self.detection_pipeline),
                ('data_recorder', self.data_recorder),
            ]
            
            all_ok = True
            for name, component in components:
                if component is not None:
                    self.get_logger().info(f'   {name}: Initialized')
                else:
                    self.get_logger().error(f'   {name}: no Initialized')
                    all_ok = False
            
            # 检查Tracking Target
            if self.current_target_id:
                self.get_logger().info(f'   Tracking Target: {self.current_target_id}')
            else:
                self.get_logger().error('   Tracking Target: Not set')
                all_ok = False
            
            #  验证参考库data而不是缓存
            if (hasattr(self.tracker, 'reference_library') and 
                self.current_target_id in self.tracker.reference_library):
                ref_entry = self.tracker.reference_library[self.current_target_id]
                spatial_data = ref_entry.get('features', {}).get('spatial', {})
                if 'height_mm' in spatial_data and 'background_world_z' in spatial_data:
                    self.get_logger().info(f'   Reference library data: Height and background data available')
                else:
                    self.get_logger().warn(f'   Reference library data: missing height or background data')
            else:
                self.get_logger().error('   Reference Library Data: Not Available')
                all_ok = False
            
            # 检查imagedata
            if self.latest_rgb_image is not None:
                self.get_logger().info(f'   RGB image: {self.latest_rgb_image.shape}')
            else:
                self.get_logger().warn('   RGB image: No data yet')
            
            if self.latest_depth_image is not None:
                self.get_logger().info(f'   depth image: {self.latest_depth_image.shape}')
            else:
                self.get_logger().warn('   depth image: No data yet')
            
            return all_ok
            
        except Exception as e:
            self.get_logger().error(f'系统组件 fail: {e}')
            return False
    
    def _initialize_detection_pipeline_fixed(self):
        """Initialized检测管道 - 修复YOLO参数问题"""
        try:
            #find 正确的配置文件
            possible_config_paths = [
                '/home/qi/ros2_ws/src/vision_ai/vision_ai/detection/config/enhanced_detection_config.json',
                os.path.join(os.path.dirname(__file__), '..', 'detection', 'config', 'enhanced_detection_config.json'),
                os.path.join(os.path.expanduser('~'), 'ros2_ws', 'src', 'vision_ai', 'vision_ai', 'detection', 'config', 'enhanced_detection_config.json')
            ]
            
            config_file = None
            for path in possible_config_paths:
                if os.path.exists(path):
                    config_file = path
                    break
            
            # 创建增强检测管道
            from ..detection.enhanced_detection_pipeline import EnhancedDetectionPipeline
            
            self.detection_pipeline = EnhancedDetectionPipeline(
                config_file=config_file,
                output_dir=None
            )
            
            # 禁用3D后处理
            self.detection_pipeline.enable_3d_post_processing = False
            
            self.get_logger().info(' The enhanced detection pipeline was successfully initialized.')
            
        except Exception as e:
            self.get_logger().error(f'Initialized detection pipeline failed: {e}')
    
    def tcp_pose_callback(self, msg):
        """TCP位姿回调 - 缓存当前机械臂位姿"""
        try:
            #  修复：xarm_controller发布的位置已经是mm单位，无需再转换
            self.current_tcp_pose = {
                'position': {
                    'x': msg.pose.position.x,  # 已经是mm单位
                    'y': msg.pose.position.y,
                    'z': msg.pose.position.z
                },
                'orientation': {
                    'x': msg.pose.orientation.x,
                    'y': msg.pose.orientation.y,
                    'z': msg.pose.orientation.z,
                    'w': msg.pose.orientation.w
                },
                'timestamp': msg.header.stamp
            }
            
        except Exception as e:
            self.get_logger().error(f'Handling TCP pose failure: {e}')
    
    def _get_candidate_detections(self) -> list:
        """获取候选检测结果 - 使用增强管道版本"""
        try:
            if not self.detection_pipeline:
                self.get_logger().error('Detection pipeline not initialized')
                return []
            
            if self.latest_rgb_image is None or self.latest_depth_image is None:
                self.get_logger().warn('image data Incomplete')
                return []
            
            #  额外的image预处理确保格式正确
            rgb_processed = self.latest_rgb_image.copy()
            depth_processed = self.latest_depth_image.copy()
            
            # 确保image尺寸匹配
            if rgb_processed.shape[:2] != depth_processed.shape:
                self.get_logger().warn('Adjust the depth image size again before detection')
                depth_processed = cv2.resize(
                    depth_processed, 
                    (rgb_processed.shape[1], rgb_processed.shape[0]), 
                    interpolation=cv2.INTER_LINEAR
                )
            
            # 确保data类型正确
            if rgb_processed.dtype != np.uint8:
                rgb_processed = rgb_processed.astype(np.uint8)
            if depth_processed.dtype != np.uint16:
                depth_processed = depth_processed.astype(np.uint16)
            
            #  使用增强管道的process_single_image方法
            detection_result = self.detection_pipeline.process_single_image(
                image_rgb=rgb_processed,
                depth_image=depth_processed,
                camera_pose=self._get_current_camera_pose()
            )
            
            if not detection_result.get('objects'):
                self.get_logger().warn('No targets were found in real-time detection')
                return []
            
            #  转换为追踪器期望的格式
            candidates = []
            for obj in detection_result['objects']:
                candidate = {
                    'bounding_box': obj['bounding_box'],
                    'confidence': obj['confidence'],
                    'class_id': obj['class_id'],
                    'class_name': obj['class_name'],
                    'mask': obj['mask'],
                    
                    #  使用增强管道提取的特征（格式可能不同）
                    'geometric_features': obj['features'].get('geometric', {}),
                    'appearance_features': obj['features'].get('appearance', obj['features'].get('color', {})),
                    'shape_features': obj['features'].get('shape', {}),
                    'spatial_features': obj['features'].get('spatial', {})
                }
                candidates.append(candidate)
            
            return candidates
            
        except Exception as e:
            self.get_logger().error(f'real-time Detection failure: {e}')
            import traceback
            traceback.print_exc()
            return []
    
    def _get_current_camera_pose(self) -> dict:
        """获取当前相机位姿（用于detection_pipeline）"""
        try:
            if not self.current_tcp_pose:
                # 使用默认位姿
                return {
                    'world_pos': [0, 0, 350],
                    'roll': 179,
                    'pitch': 0,
                    'yaw': 0
                }
            
            # 从四元数转换为欧拉角
            orientation = self.current_tcp_pose['orientation']
            roll, pitch, yaw = self._quat_to_euler(
                orientation['x'], orientation['y'], 
                orientation['z'], orientation['w']
            )
            # self.get_logger().info(f'当前位姿状态 :roll:{roll},pitch: {pitch}, yaw:{yaw} ')
            #  修复：位置已经是mm单位，直接使用
            return {
                'world_pos': [
                    self.current_tcp_pose['position']['x'],
                    self.current_tcp_pose['position']['y'],
                    self.current_tcp_pose['position']['z']
                ],
                'roll': roll,
                'pitch': pitch,
                'yaw': yaw
            }
            
        except Exception as e:
            self.get_logger().error(f'Failed to obtain camera pose: {e}')
            return {'world_pos': [0, 0, 350], 'roll': 179, 'pitch': 0, 'yaw': 0}
    
    def _quat_to_euler(self, qx, qy, qz, qw):
        """四元数转欧拉角（度）- 修复版，增强稳定性"""
        import math
        
        #  使用更稳定的转换算法，避免万向节锁问题
        # 参考ROS tf2和scipy的实现
        
        # 预处理：确保四元数归一化
        norm = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
        if norm > 0:
            qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm
        
        #  使用更稳定的atan2计算，适合xarm坐标系
        # Roll (X轴旋转)
        sinr_cosp = 2 * (qw * qx + qy * qz)
        cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        
        # Pitch (Y轴旋转) - 使用clamp避免数值误差
        sinp = 2 * (qw * qy - qz * qx)
        sinp = max(-1.0, min(1.0, sinp))  # clamp到[-1,1]
        pitch = math.asin(sinp)
        
        # Yaw (Z轴旋转) -  关键修复点
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        # 转换为度
        roll_deg = math.degrees(roll)
        pitch_deg = math.degrees(pitch)
        yaw_deg = math.degrees(yaw)
        
        #  yaw角度连续性处理，避免-180°到180°的跳跃
        if self.previous_yaw is not None:
            yaw_diff = yaw_deg - self.previous_yaw
            
            # 处理跨越±180°的情况
            if yaw_diff > 180:
                yaw_deg -= 360
            elif yaw_diff < -180:
                yaw_deg += 360
            
            #  如果角度变化过大（>90°），可能是计算错误，使用上一次的值
            if abs(yaw_deg - self.previous_yaw) > 90:
                yaw_deg = self.previous_yaw
        
        # 更新previous_yaw
        self.previous_yaw = yaw_deg
        
        #  角度范围标准化到[-180, 180]
        roll_deg = ((roll_deg + 180) % 360) - 180
        pitch_deg = ((pitch_deg + 180) % 360) - 180  
        yaw_deg = ((yaw_deg + 180) % 360) - 180
        
        return roll_deg, pitch_deg, yaw_deg
    
    def _get_current_waypoint_data(self) -> dict:
        """获取当前waypointdata - 基于实际TCP位姿"""
        try:
            if not self.current_tcp_pose:
                self.get_logger().warn('TCP pose data is not available, use default value')
                return {
                    'world_pos': [0, 0, 350],
                    'roll': 179,
                    'pitch': 0,
                    'yaw': 0
                }
            
            # 从缓存的TCP位姿构建waypointdata
            camera_pose = self._get_current_camera_pose()
            
            waypoint_data = {
                'world_pos': camera_pose['world_pos'],
                'roll': camera_pose['roll'],
                'pitch': camera_pose['pitch'],
                'yaw': camera_pose['yaw'],
                'timestamp': self.current_tcp_pose['timestamp']
            }
            
            return waypoint_data
            
        except Exception as e:
            self.get_logger().error(f'Failed to obtain waypoint data: {e}')
            return {'world_pos': [0, 0, 350], 'roll': 179, 'pitch': 0, 'yaw': 0}
    
    
    def _load_tracking_target(self) -> str:
        """从tracking_selection.txt加载User选择的Tracking Target - 简化版"""
        try:
            selection_file = os.path.join(
                self.current_scan_dir, 
                'enhanced_detection_results', 
                'tracking_selection.txt'
            )
            
            if not os.path.exists(selection_file):
                self.get_logger().error(f'Tracking selection file does not exist: {selection_file}')
                return None
            
            with open(selection_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 解析文件内容，提取Target ID
            target_id = None
            lines = content.strip().split('\n')
            for line in lines:
                if 'ID:' in line and ')' in line:
                    # 格式：- Description (ID: target_id)
                    start = line.find('ID:') + 3
                    end = line.find(')', start)
                    if start < end:
                        target_id = line[start:end].strip()
                        self.get_logger().info(f'find Tracking Target: {target_id}')
                        break
            
            if not target_id:
                self.get_logger().error('Unable to parse Target ID from selection file')
                return None
            
            return target_id
            
        except Exception as e:
            self.get_logger().error(f'Failed to load Tracking Target: {e}')
            return None

    def _initialize_auxiliary_components(self):
        """Initialized辅助组件 - 详细版本"""
        try:
            self.get_logger().info(' Initialized auxiliary components...')
            
            # User配置管理器
            self.user_profile_manager = UserProfileManager(self.current_user_id)
            
            # data记录器
            self._initialize_data_recorder()
            
            # 反馈收集器
            self.feedback_collector = FeedbackCollector(self.get_logger())
            
            # 追踪可视化器
            self.visualizer = TrackingVisualizer()
            
            # 在线学习器
            self.online_learner = OnlineLearner(
                self.current_user_id or "default_user",
                self.tracker.similarity_calculator
            )
        
            
        except Exception as e:
            self.get_logger().error(f'Auxiliary component Initialized failed: {e}')
            self._create_fallback_components()

    def _initialize_data_recorder(self) -> bool:
        """Initializeddata记录器"""
        try:
            if self.current_scan_dir and self.current_user_id:
                self.data_recorder = DataRecorder(
                    self.current_user_id, 
                    self.current_scan_dir
                )
                return True
            else:
                # 创建备用data记录器
                self.data_recorder = self._create_fallback_recorder()
                return False
                
        except Exception as e:
            self.data_recorder = self._create_fallback_recorder()
            return False

    def _create_fallback_components(self):
        """创建备用组件"""
        try:
            if self.data_recorder is None:
                self.data_recorder = self._create_fallback_recorder()
            
            if self.feedback_collector is None:
                class FallbackFeedbackCollector:
                    def __init__(self, logger):
                        self.logger = logger
                    def collect_feedback(self, prompt):
                        self.logger.info(f'alternative feedback collector: {prompt}')
                        return True  # 默认正确
                
                self.feedback_collector = FallbackFeedbackCollector(self.get_logger())
                
        except Exception as e:
            self.get_logger().error(f'创建备用组件 fail : {e}')


    def rgb_image_callback(self, msg):
        """RGBimage回调 - 修复确认 wait 逻辑"""
        try:
            # 处理RGBimage
            try:
                color_cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
                
                if len(color_cv_image.shape) == 3 and color_cv_image.shape[2] == 3:
                    color_rgb = cv2.cvtColor(color_cv_image, cv2.COLOR_BGR2RGB)
                else:
                    self.get_logger().error(f'Unsupported color image format: {color_cv_image.shape}')
                    return
                    
                self.latest_rgb_image = color_rgb
                    
            except Exception as color_error:
                self.get_logger().error(f'RGBimage conversion failed: {color_error}')
                return
            
            #  修复：只有在系统完全Initialized且不在 wait 确认状态时才执行追踪
            if (self.system_initialized and 
                self.tracking_active and 
                not self.waiting_for_user_confirmation and  #  关键修复
                self.latest_depth_image is not None and 
                self.tracking_count < self.max_tracking_steps and
                self.movement_complete):  #  确保Robotic arm movement completed
                
                current_time = time.time()
                if not hasattr(self, '_last_tracking_time'):
                    self._last_tracking_time = 0
                
                if current_time - self._last_tracking_time >= 0.5:
                    self._last_tracking_time = current_time
                    self._execute_tracking_step()
            
            # 检查是否达到最大步数
            elif self.tracking_count >= self.max_tracking_steps and self.tracking_active:
                self.get_logger().info(' The maximum Tracking steps has been reached, stopping tracking')
                self._finish_tracking_session()
            
            # 显示系统状态（仅在 wait 时）
            elif self.waiting_for_detection_complete:
                current_time = time.time()
                if not hasattr(self, '_last_status_time'):
                    self._last_status_time = current_time
                
                if current_time - self._last_status_time >= 10.0:
                    self._last_status_time = current_time
                    self.get_logger().info('The system is ready, waiting for the detection completion signal...')
                    
        except Exception as e:
            self.get_logger().error(f'RGBimage fail: {e}')

    def depth_image_callback(self, msg):
        """depth image回调 - 总是处理depth image用于系统监控"""
        try:
            #  完全采用scan_executor的depth 处理逻辑
            try:
                # 直接从原始字节data重建16位depth image
                if msg.step == msg.width * 2:  # 正确的16位步长
                    raw_data = np.frombuffer(msg.data, dtype=np.uint16)
                    depth_raw = raw_data.reshape((msg.height, msg.width))
                else:
                    raw_data = np.frombuffer(msg.data, dtype=np.uint8)
                    depth_raw = raw_data.view(np.uint16).reshape((msg.height, msg.width))
            except Exception as depth_error:
                self.get_logger().error(f'Direct depth reconstruction fails: {depth_error}')
                try:
                    depth_raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
                    if depth_raw.dtype != np.uint16:
                        depth_raw = depth_raw.astype(np.uint16)
                except Exception as fallback_error:
                    self.get_logger().error(f'All depth conversion methods fail: {fallback_error}')
                    return
            
            # 检查并修正depth image分辨率
            expected_width, expected_height = 1280, 720
            if depth_raw.shape != (expected_height, expected_width):
                depth_raw = cv2.resize(
                    depth_raw, 
                    (expected_width, expected_height), 
                    interpolation=cv2.INTER_LINEAR
                )
            
            #  修复：正确的depth data合理性验证
            depth_min, depth_max = depth_raw.min(), depth_raw.max()
            
            # 合理的depth 范围验证（静默处理）
            if depth_max == 0:
                self.get_logger().debug('depth data all 0')
            elif depth_max > 10000:
                self.get_logger().debug(f'depth too big : max={depth_max}mm')
            elif depth_max < 50:
                self.get_logger().debug(f'depth too small: max={depth_max}mm')
            else:
                self.get_logger().debug(f'depth current: {depth_min}mm - {depth_max}mm')
            
            self.latest_depth_image = depth_raw
            
        except Exception as e:
            self.get_logger().error(f'depth image fail: {e}')
    
    def movement_complete_callback(self, msg):
        """Robotic arm movement completed回调 - 修复版"""
        try:
            self.get_logger().info(' Robotic arm movement completed')
            self.movement_complete = True
            
            #  移动完成后，如果追踪仍然激活，准备下一步
            if self.tracking_count >= self.max_tracking_steps:
                self.get_logger().info(' All tracking steps completed骤')
                self._finish_tracking_session()
                
        except Exception as e:
            self.get_logger().error(f'Failed to process the move completion signal: {e}')
    
    def _execute_tracking_step(self):
        """执行单步追踪 - 修复版：只在需要时保存data"""
        try:
            self.get_logger().info(f' Execute {self.tracking_count + 1} tracking steps')
            
            # 验证imagedata
            if not self._validate_image_data():
                self.get_logger().error('image data fail, skip tracking')
                return
            
            # 获取候选检测结果
            candidate_detections = self._get_candidate_detections()
            
            if not candidate_detections:
                self.get_logger().warn(' No candidate target detected')
                self._handle_detection_failure_with_retry()
                return
            
            # 保存候选检测的完整data（在追踪前）
            candidates_backup = self._prepare_candidates_for_detailed_save(candidate_detections)
            
            # 获取当前waypointdata
            waypoint_data = self._get_current_waypoint_data()
            
            # 保存waypointdata供后续使用
            self._last_waypoint_data = waypoint_data.copy()
            
            tracking_result = self.tracker.track_target(
                self.current_target_id,
                self.latest_rgb_image,
                self.latest_depth_image,
                waypoint_data,
                candidate_detections
            )
            
            if tracking_result:
                # 🔧 只保存详细data，不保存image（image将在确认时保存）
                self._save_detailed_step_data(
                    self.tracking_count + 1,
                    candidates_backup, 
                    tracking_result, 
                    waypoint_data
                )
                
                # 处理追踪成功， wait User确认（确认时会保存image）
                self._handle_tracking_success_with_confirmation(tracking_result)
            else:
                self._handle_detection_failure_with_retry()
                
        except Exception as e:
            self.get_logger().error(f'Tracking step execution failed: {e}')
            import traceback
            traceback.print_exc()

    # tracking_node.py
    def _handle_tracking_success_with_confirmation(self, tracking_result: dict):
        """处理追踪成功 - 混合相似度版本 - 修复auto模式data丢失"""
        try:
            self.get_logger().info(' Tracking success (mixed similarity), analysis of grasp conditions')
            
            # 🔧 关键修复：确保current_tracking_result总是被设置
            self.current_tracking_result = tracking_result.copy()
            
            # 显示混合相似度详细信息
            self._log_hybrid_tracking_details(tracking_result)
            
            # 计算移动策略
            movement_strategy = self._calculate_adaptive_movement_strategy(
                self.current_tcp_pose, 
                tracking_result['grasp_coordinate'], 
                tracking_result['object_info']
            )
            
            # 评估grasp条件
            grasp_conditions = self._evaluate_grasp_conditions(tracking_result, movement_strategy)
            
            # 检查是否可以自动判断成功
            auto_success = tracking_result.get('auto_success', False)
            
            # 🔧 添加置信度检查调试
            tracking_confidence = tracking_result.get('tracking_confidence', 0.0)
            self.get_logger().info(f'  Tracking confidence check: {tracking_confidence:.3f} (auto_success: {auto_success})')
            
            if grasp_conditions['grasp_ready']:
                # 满足grasp条件
                if auto_success:
                    # 自动判断成功且满足grasp条件，直接grasp
                    self.get_logger().info(' Automatically judge success and meet the grab conditions, and execute grab directly')
                    self._execute_direct_grasp(tracking_result)
                else:
                    # 需要User确认grasp
                    self.get_logger().info(' Meet the grab condition，请求User确认grasp')
                    self._start_grasp_confirmation_with_auto_info(tracking_result)
            else:
                # 不满足grasp条件
                if auto_success:
                    # 自动判断成功，直接continue
                    self.get_logger().info(' Automatically judge success and continue tracking')
                    self._handle_auto_success_continue(tracking_result)
                else:
                    # 需要User确认
                    self.get_logger().info('Request User to confirm tracking results')
                    self._start_command_input_thread_with_auto_info(tracking_result)
                        
        except Exception as e:
            self.get_logger().error(f'Processing tracking success fail : {e}')

    def _perform_yaw_alignment_detection(self, tracking_result: dict) -> bool:
        """执行yaw角度对齐检测 - 简化版，不 wait 移动完成"""
        try:
            self.get_logger().info(' 开始yaw角度对齐检测...')
            
            #  确保current_tracking_result不为空
            if self.current_tracking_result is None:
                self.get_logger().warn(' current_tracking_result is None,use tracking_result')
                self.current_tracking_result = tracking_result.copy()
            
            # 获取目标grasp坐标和角度信息
            grasp_coord = tracking_result['grasp_coordinate']
            target_yaw = self._get_or_calculate_target_yaw(tracking_result)
            
            self.get_logger().info(f' Target yaw angle: {target_yaw:.1f}°')
            
            # 发布yaw对齐位置
            if not self._publish_yaw_alignment_pose(grasp_coord, target_yaw):
                self.get_logger().error(' Publish yaw alignment position fail ')
                return False
            
            #  不 wait 移动完成，直接 wait image稳定后进行检测
            self.get_logger().info(' wait image稳定...')
            time.sleep(1.0)  # 短暂 wait ，让机械臂开始移动
            
            # 进行二次检测验证
            self.get_logger().info(' Perform secondary detection verification after yaw alignment...')
            return self._perform_secondary_detection_verification(tracking_result)
            
        except Exception as e:
            self.get_logger().error(f'Yaw angle alignment detection failure: {e}')
            return False

    def _get_or_calculate_target_yaw(self, tracking_result: dict) -> float:
        """获取or计算Target yaw angle - 确保在auto模式下也能正确计算"""
        try:
            # 方法1：从tracking_result的grasp_angles获取
            grasp_angles = tracking_result.get('grasp_angles', {})
            if grasp_angles and 'recommended_yaw' in grasp_angles:
                recommended_yaw = grasp_angles['recommended_yaw']
                if recommended_yaw != 0.0:  # 如果不是默认值
                    self.get_logger().info(f' 使用tracking_result中的推荐yaw: {recommended_yaw:.1f}°')
                    return recommended_yaw
            
            # 方法2：如果上面没有or是默认值，real-time 计算
            self.get_logger().info(' tracking_result中无有效yaw信息，real-time 计算...')
            
            # 获取当前yaw和扫描yaw
            current_yaw = self._get_current_yaw()
            scan_yaw = 0.0
            if hasattr(self, '_last_waypoint_data') and self._last_waypoint_data:
                scan_yaw = self._last_waypoint_data.get('yaw', 0.0)
            
            # 获取bounding_rect信息
            bounding_rect = tracking_result.get('bounding_rect', {})
            object_info = tracking_result.get('object_info', {})
            
            if bounding_rect and bounding_rect.get('width', 0) > 0:
                # real-time 计算推荐yaw
                calculated_yaw = self._calculate_enhanced_grasp_yaw(
                    current_yaw, scan_yaw, bounding_rect, object_info
                )
                self.get_logger().info(f' real-time 计算推荐yaw: {calculated_yaw:.1f}°')
                return calculated_yaw
            else:
                # 方法3：使用当前yaw作为备用
                self.get_logger().warn(f' 无法获取足够信息计算yaw，使用当前yaw: {current_yaw:.1f}°')
                return current_yaw
                
        except Exception as e:
            self.get_logger().error(f'获取Target yaw angle fail : {e}')
            return 0.0

    def _get_current_yaw(self) -> float:
        """获取当前yaw角度"""
        try:
            if self.current_tcp_pose:
                orientation = self.current_tcp_pose.get('orientation', {})
                if orientation:
                    qx, qy, qz, qw = orientation.get('x', 0), orientation.get('y', 0), orientation.get('z', 0), orientation.get('w', 1)
                    _, _, current_yaw = self._quat_to_euler(qx, qy, qz, qw)
                    return current_yaw
            return 0.0
        except:
            return 0.0
    
    def _handle_auto_success_continue(self, tracking_result: dict):
        """处理自动成功continue追踪 - 完整修复版"""
        try:
            # 🔧 确保current_tracking_result被正确保存
            if not self.current_tracking_result:
                self.current_tracking_result = tracking_result.copy()
                self.get_logger().warn('🔧 修复：current_tracking_result为空，重新设置')
            
            # 记录成功检测到混合相似度histroy
            if hasattr(self.tracker, 'record_successful_detection_manual'):
                detection_data = self._extract_detection_from_tracking_result(tracking_result)
                similarity_data = tracking_result.get('hybrid_similarity_info', {})
                self.tracker.record_successful_detection_manual(
                    self.current_target_id, detection_data, similarity_data, user_confirmed=True
                )
                self._record_tracking_success_to_history(tracking_result)
                
            # 确保参数histroy记录更新
            tracking_params = self._extract_tracking_parameters(tracking_result)
            tracking_params['user_feedback'] = 'auto_success'
            tracking_params['feedback_timestamp'] = datetime.now().isoformat()
            self.tracking_parameters_history.append(tracking_params)
            
            # 保存imagedata（包含auto_success反馈）
            self.get_logger().info('💾 自动判断成功，保存追踪 step imagedata...')
            enhanced_result = tracking_result.copy()
            enhanced_result['user_feedback'] = 'auto_success'
            enhanced_result['feedback_timestamp'] = datetime.now().isoformat()
            
            self._save_tracking_step_data(enhanced_result, True)
            
            # 🔧 更新详细data文件
            step_number = self.tracking_count + 1
            self._update_detailed_file_feedback(step_number, 'auto_success')
            
            # continue追踪流程
            self._publish_target_pose(tracking_result)
            self.tracking_count += 1
            self.movement_complete = False
            
            # 更新连续成功计数
            self.consecutive_high_confidence += 1
            
            # 🔧 重置状态，为下一步准备
            self.waiting_for_user_confirmation = False
            self.current_tracking_result = None  # 在保存后重置
            
            self.get_logger().info(f'  自动continue追踪 (步数: {self.tracking_count}/{self.max_tracking_steps})')
            
        except Exception as e:
            self.get_logger().error(f'自动成功continue fail: {e}')

    def _start_command_input_thread_with_auto_info(self, tracking_result: dict):
        """启动命令输入（包含自动判断信息）"""
        try:
            self.current_tracking_result = tracking_result
            self.waiting_for_user_confirmation = True
            self.user_feedback_received = False
            self._grasp_confirmation_mode = False
            
            # 启动增强版命令输入线程
            self._start_enhanced_command_input_thread()
            
        except Exception as e:
            self.get_logger().error(f'启动增强命令输入 fail : {e}')

    def _start_enhanced_grasp_confirmation_thread(self):
        """启动增强版grasp确认线程"""
        try:
            if hasattr(self, 'enhanced_grasp_thread') and self.enhanced_grasp_thread and self.enhanced_grasp_thread.is_alive():
                return
            
            self.enhanced_grasp_thread = threading.Thread(
                target=self._enhanced_grasp_confirmation_worker,
                daemon=True
            )
            self.enhanced_grasp_thread.start()
            
        except Exception as e:
            self.get_logger().error(f'Start enhanced grab confirmation thread fail : {e}')

    def _enhanced_grasp_confirmation_worker(self):
        """增强版grasp确认工作线程"""
        try:
            print(f"\n{'='*80}")
            print(f" 混合相似度grasp确认 -  step  {self.tracking_count + 1}")
            print(f"{'='*80}")
            
            if self.current_tracking_result:
                grasp_coord = self.current_tracking_result['grasp_coordinate']
                object_info = self.current_tracking_result['object_info']
                hybrid_info = self.current_tracking_result.get('hybrid_similarity_info', {})
                auto_success = self.current_tracking_result.get('auto_success', False)
                
                print(f" Grasp information:")
                print(f" Target coordinates: ({grasp_coord['x']: 0.1f}, {grasp_coord['y']: 0.1f}, {grasp_coord['z']: 0.1f}mm)")
                print(f" Object height: {object_info.get('estimated_height', 30): 0.1f}mm")
                print(f" Background height: {object_info.get('background_z', 300): 0.1f}mm")
                print(f" Recommended gripper width: {object_info.get('recommended_gripper_width', 150)}")

                print(f" Hybrid similarity information:")
                print(f" Reference library similarity: {hybrid_info.get('reference_score', 0): 0.3f}")
                print(f" Historical similarity: {hybrid_info.get('historical_score', 0):.3f}")
                print(f" Historical score: {hybrid_info.get('history_count', 0)}")
                print(f" Final confidence: {self.current_tracking_result['tracking_confidence']:.3f}")
                
                if auto_success:
                    print(f"   System automatic judgment: Success")
                else:
                    print(f"   System judgment: Manual confirmation is required")
            
                print(f"\n Command options:")
                print(f" 'g' or 'grasp' - Confirm grab execution")
                print(f" 'c' or 'continue' - Cancel grab and continue tracking")
                if not self.current_tracking_result.get('auto_success', False):
                    print(f" 'a' or 'auto' - Mark as successful and enable auto-tracking for this target")
                print(f" 'q' or 'quit' - Quit tracking")
            print(f"{'='*80}")
            
            while self.waiting_for_user_confirmation and not self.user_feedback_received:
                try:
                    user_input = input("please enter (g/c/a/q): ").strip().lower()
                    
                    if user_input in ['g', 'grasp']:
                        self._handle_enhanced_grasp_confirmation('grasp')
                        break
                    elif user_input in ['c', 'continue']:
                        self._handle_enhanced_grasp_confirmation('continue')
                        break
                    elif user_input in ['a', 'auto'] and not self.current_tracking_result.get('auto_success', False):
                        self._handle_enhanced_grasp_confirmation('auto_success')
                        break
                    elif user_input in ['q', 'quit']:
                        self._handle_quit_request()
                        break
                    else:
                        available_commands = ['g', 'c', 'q']
                        if not self.current_tracking_result.get('auto_success', False):
                            available_commands.insert(-1, 'a')
                        print(f" Invalid input: {'/'.join(available_commands)}")
                        
                except (EOFError, KeyboardInterrupt):
                    print("\n User interrupt input")
                    self._handle_quit_request()
                    break
                except Exception as e:
                    print(f" Input processing errors: {e}")
                    
        except Exception as e:
            self.get_logger().error(f'增强grasp确认工作线程 fail : {e}')

    def _start_enhanced_command_input_thread(self):
        """启动增强版命令输入线程"""
        try:
            if hasattr(self, 'enhanced_command_thread') and self.enhanced_command_thread and self.enhanced_command_thread.is_alive():
                return
            
            self.enhanced_command_thread = threading.Thread(
                target=self._enhanced_command_input_worker,
                daemon=True
            )
            self.enhanced_command_thread.start()
            
        except Exception as e:
            self.get_logger().error(f'Start enhanced command input thread fail : {e}')

    def _enhanced_command_input_worker(self):
        """增强版命令输入工作线程"""
        try:
            print(f"\n{'='*80}")
            print(f" Hybrid Similarity Tracking Confirmation - Steps {self.tracking_count + 1}")
            print(f"{'='*80}")
            
            if self.current_tracking_result:
                grasp_coord = self.current_tracking_result['grasp_coordinate']
                hybrid_info = self.current_tracking_result.get('hybrid_similarity_info', {})
                auto_success = self.current_tracking_result.get('auto_success', False)
                
                print(f" Tracking result:")
                print(f" Target coordinates: ({grasp_coord['x']:.1f}, {grasp_coord['y']:.1f}, {grasp_coord['z']:.1f}mm)")
                print(f" Final confidence: {self.current_tracking_result['tracking_confidence']:.3f}")

                print(f"\n Hybrid similarity details:")
                print(f" Reference similarity: {hybrid_info.get('reference_score', 0):.3f}")
                print(f" Historical similarity: {hybrid_info.get('historical_score', 0):.3f}")
                print(f" History count: {hybrid_info.get('history_count', 0)}")
                
                if auto_success:
                    print(f"Automatic system determination: Success")
                else:
                    print(f" System determination: Manual confirmation required")
            
            print(f"\n Command options:")
            print(f"  's' or 'success' - Confirm that tracking is successful and continue moving")
            print(f"  'f' or 'fail'    -Marker tracking failed")
            if not self.current_tracking_result.get('auto_success', False):
                print(f"  'a' or 'auto'    - Mark as successful and enable automatic judgment for this goal")
            print(f"  'q' or 'quit'    -Opt-out of tracking")
            print(f"{'='*80}")
            
            while self.waiting_for_user_confirmation and not self.user_feedback_received:
                try:
                    user_input = input("Please enter the command (s/f/a/q): ").strip().lower()
                    
                    if user_input in ['s', 'success']:
                        self._handle_enhanced_user_confirmation('success')
                        break
                    elif user_input in ['f', 'fail']:
                        self._handle_enhanced_user_confirmation('failure')
                        break
                    elif user_input in ['a', 'auto'] and not self.current_tracking_result.get('auto_success', False):
                        self._handle_enhanced_user_confirmation('auto_success')
                        break
                    elif user_input in ['q', 'quit']:
                        self._handle_quit_request()
                        break
                    else:
                        available_commands = ['s', 'f', 'q']
                        if not self.current_tracking_result.get('auto_success', False):
                            available_commands.insert(-1, 'a')
                        print(f" Invalid input, please enter: {'/'.join(available_commands)}")
                        
                except (EOFError, KeyboardInterrupt):
                    print("\n User interrupt input")
                    self._handle_quit_request()
                    break
                except Exception as e:
                    print(f" Input processing errors: {e}")
                    
        except Exception as e:
            self.get_logger().error(f'Enhanced command input worker thread failed: {e}')

    def _handle_enhanced_grasp_confirmation(self, action: str):
        """处理增强版grasp确认"""
        try:
            self.user_feedback_received = True
            current_result_backup = self.current_tracking_result
            
            if action == 'grasp':
                self.get_logger().info(' The user confirms the execution of the grab and starts the secondary confirmation of the yaw angle...')
                
                #  yaw角度二次确认
                if self._perform_yaw_alignment_detection(current_result_backup):
                    # yaw对齐检测成功，执行grasp
                    self._execute_confirmed_grasp(current_result_backup)
                else:
                    # yaw对齐Detection failure，询问User
                    self._handle_yaw_detection_failure(current_result_backup)
                    return  # 不重置状态， wait User选择
                
            elif action == 'continue':
                self.get_logger().info(' User cancels crawling and continues tracking')
                
                # 记录成功到histroy
                self._record_tracking_success_to_history(current_result_backup)
                
                # countinue追踪流程
                self._continue_tracking_after_confirmation(current_result_backup, True)
                
            elif action == 'auto_success':
                self.get_logger().info(' User enables automatic judgment mode')
                
                # 启用自动模式
                if hasattr(self.tracker, 'enable_auto_success_mode'):
                    self.tracker.enable_auto_success_mode(True, threshold=0.8)
                
                # 记录成功到histroy
                self._record_tracking_success_to_history(current_result_backup)
                
                # countinue追踪流程
                self._continue_tracking_after_confirmation(current_result_backup, True)
            
            # 重置状态
            self.waiting_for_user_confirmation = False
            self.current_tracking_result = None
            if hasattr(self, '_grasp_confirmation_mode'):
                delattr(self, '_grasp_confirmation_mode')
                
        except Exception as e:
            self.get_logger().error(f'处理增强grasp确认 fail : {e}')

    def _execute_confirmed_grasp(self, tracking_result: dict):
        """执行确认后的grasp"""
        try:
            self.get_logger().info(' Yaw alignment verification passed, execute crawl')
            
            # 记录成功到histroy
            self._record_tracking_success_to_history(tracking_result)
            
            # 生成grasp文件
            bounding_rect = tracking_result.get('bounding_rect', {})
            grasp_filename = self._generate_grasp_info_file(tracking_result, bounding_rect)
            
            if grasp_filename:
                self._publish_grasp_trigger(grasp_filename, tracking_result['grasp_coordinate'])
                self.tracking_active = False
                self._finish_tracking_session()
            else:
                self.get_logger().error(' Failed to generate crawl file')
                
        except Exception as e:
            self.get_logger().error(f'Execution confirmation capture failed: {e}')

    def _handle_yaw_detection_failure(self, tracking_result: dict):
        """处理yawDetection failure"""
        try:
            self.get_logger().warn(' aw angle secondary detection failed')
            
            # 询问User是否countinue
            self._start_yaw_failure_confirmation_thread(tracking_result)
            
        except Exception as e:
            self.get_logger().error(f'处理yawDetection failure fail : {e}')

    def _start_yaw_failure_confirmation_thread(self, tracking_result: dict):
        """启动yaw fail 确认线程"""
        try:
            self.current_tracking_result = tracking_result
            self.waiting_for_user_confirmation = True
            self.user_feedback_received = False
            
            yaw_failure_thread = threading.Thread(
                target=self._yaw_failure_confirmation_worker,
                daemon=True
            )
            yaw_failure_thread.start()
            
        except Exception as e:
            self.get_logger().error(f'启动yaw fail 确认线程 fail : {e}')

    def _yaw_failure_confirmation_worker(self):
        """yaw fail 确认工作线程"""
        try:
            print(f"\n{'='*80}")
            print(f"  aw angle secondary detection failed")
            print(f"{'='*80}")
            print(f"\n Please select:")
            print(f"  'g' or 'grasp',force_grasp")
            print(f"  'c' or 'continue'")
            print(f"  'r' or 'retry' ")
            print(f"  'q' or 'quit'")
            print(f"{'='*80}")
            
            while self.waiting_for_user_confirmation and not self.user_feedback_received:
                try:
                    user_input = input("Please enter the command (g/c/r/q): ").strip().lower()
                    
                    if user_input in ['g', 'grasp']:
                        self._handle_yaw_failure_decision('force_grasp')
                        break
                    elif user_input in ['c', 'continue']:
                        self._handle_yaw_failure_decision('continue')
                        break
                    elif user_input in ['r', 'retry']:
                        self._handle_yaw_failure_decision('retry')
                        break
                    elif user_input in ['q', 'quit']:
                        self._handle_quit_request()
                        break
                    else:
                        print(" Invalid input, please enter: g/c/r/q")
                        
                except (EOFError, KeyboardInterrupt):
                    print("\n User interrupt input")
                    self._handle_quit_request()
                    break
                    
        except Exception as e:
            self.get_logger().error(f'Yaw failed to confirm the worker thread failed: {e}')
    def _handle_auto_yaw_failure(self, tracking_result: dict):
        """处理auto模式下的yawDetection failure"""
        try:
            self.get_logger().warn('The yaw detection fails in auto mode, switch to manual confirmation mode')
            
            # 设置为手动确认模式
            self.current_tracking_result = tracking_result
            self.waiting_for_user_confirmation = True
            self.user_feedback_received = False
            
            # 启动手动yaw fail 确认线程
            auto_yaw_failure_thread = threading.Thread(
                target=self._auto_yaw_failure_confirmation_worker,
                daemon=True
            )
            auto_yaw_failure_thread.start()
            
        except Exception as e:
            self.get_logger().error(f'处理auto模式yaw fail  fail : {e}')

    def _auto_yaw_failure_confirmation_worker(self):
        """auto模式yaw fail 确认工作线程"""
        try:
            print(f"\n{'='*80}")
            print(f" AUTO模式 - aw angle secondary detection failed")
            print(f"{'='*80}")
            print(f"The system automatically judges that it is successful, but the yaw angle verification fails.")
            print(f"\n Please select:")
            print(f"  'g' or 'grasp'")
            print(f"  'c' or 'continue'")
            print(f"  'r' or 'retry'  ")
            print(f"  'd' or 'disable' ")
            print(f"  'q' or 'quit'")
            print(f"{'='*80}")
            
            while self.waiting_for_user_confirmation and not self.user_feedback_received:
                try:
                    user_input = input("Please enter the command (g/c/r/d/q): ").strip().lower()
                    
                    if user_input in ['g', 'grasp']:
                        self._handle_auto_yaw_failure_decision('force_grasp')
                        break
                    elif user_input in ['c', 'continue']:
                        self._handle_auto_yaw_failure_decision('continue')
                        break
                    elif user_input in ['r', 'retry']:
                        self._handle_auto_yaw_failure_decision('retry')
                        break
                    elif user_input in ['d', 'disable']:
                        self._handle_auto_yaw_failure_decision('disable_auto')
                        break
                    elif user_input in ['q', 'quit']:
                        self._handle_quit_request()
                        break
                    else:
                        print(" Invalid input, please enter: g/c/r/d/q")
                        
                except (EOFError, KeyboardInterrupt):
                    print("\n User interrupt input")
                    self._handle_quit_request()
                    break
                    
        except Exception as e:
            self.get_logger().error(f'auto Yaw failed to confirm the worker thread failed: {e}')

    def _handle_auto_yaw_failure_decision(self, decision: str):
        """处理auto模式yaw fail 决策"""
        try:
            self.user_feedback_received = True
            current_result = self.current_tracking_result
            
            if decision == 'force_grasp':
                self.get_logger().info('Users trust the original detection and enforce crawling')
                self._execute_confirmed_grasp(current_result)
                
            elif decision == 'continue':
                self.get_logger().info(' User cancels crawling and continues tracking')
                self._record_tracking_success_to_history(current_result)
                self._continue_tracking_after_confirmation(current_result, True)
                
            elif decision == 'retry':
                self.get_logger().info(' User chooses to re-perform yaw detection')
                # 重新开始yaw检测流程
                if self._perform_yaw_alignment_detection(current_result):
                    self._execute_confirmed_grasp(current_result)
                else:
                    # 再次 fail ，回到选择界面
                    self._handle_auto_yaw_failure(current_result)
                    return  # 不重置状态
                    
            elif decision == 'disable_auto':
                self.get_logger().info(' The user disables the auto mode and switches to full manual mode')
                # 禁用auto模式
                if hasattr(self.tracker, 'enable_auto_success_mode'):
                    self.tracker.enable_auto_success_mode(False)
                
                # 转为手动grasp确认
                self._start_grasp_confirmation_with_auto_info(current_result)
                return  # 不重置状态，让手动确认接管
            
            # 重置状态
            self.waiting_for_user_confirmation = False
            self.current_tracking_result = None
            
        except Exception as e:
            self.get_logger().error(f'处理auto yaw fail 决策 fail : {e}')
    def _handle_yaw_failure_decision(self, decision: str):
        """处理yaw fail 决策"""
        try:
            self.user_feedback_received = True
            current_result = self.current_tracking_result
            
            if decision == 'force_grasp':
                self._execute_confirmed_grasp(current_result)
                
            elif decision == 'continue':
                self._record_tracking_success_to_history(current_result)
                self._continue_tracking_after_confirmation(current_result, True)
                
            elif decision == 'retry':
                self.get_logger().info(' User chooses to re-perform yaw detection')
                # 重新开始yaw检测流程
                if self._perform_yaw_alignment_detection(current_result):
                    self._execute_confirmed_grasp(current_result)
                else:
                    # 再次 fail ，回到选择界面
                    self._handle_yaw_detection_failure(current_result)
                    return  # 不重置状态
            
            # 重置状态
            self.waiting_for_user_confirmation = False
            self.current_tracking_result = None
            
        except Exception as e:
            self.get_logger().error(f'处理yaw fail 决策 fail : {e}')
            
    def _execute_direct_grasp(self, tracking_result: dict):
        """直接执行grasp（自动判断成功且满足grasp条件）- 修复版"""
        try:
            self.get_logger().info(' 自动执行grasp流程，开始yaw角度二次确认...')
            
            #  在auto模式下，确保tracking_result包含完整的角度信息
            if not tracking_result.get('bounding_rect'):
                self.get_logger().warn(' Auto mode yaw verification passed, automatic crawling performed')
                # 可以在这里添加重新计算bounding_rect的逻辑
            
            #  auto模式下也需要yaw二次验证
            if self._perform_yaw_alignment_detection(tracking_result):
                # yaw对齐检测成功，countinue自动grasp流程
                self.get_logger().info(' autoAuto mode yaw verification passed, automatic crawling performed')
                
                # 记录自动成功到histroy
                if hasattr(self.tracker, 'record_successful_detection_manual'):
                    detection_data = self._extract_detection_from_tracking_result(tracking_result)
                    similarity_data = tracking_result.get('hybrid_similarity_info', {})
                    self.tracker.record_successful_detection_manual(
                        self.current_target_id, detection_data, similarity_data, user_confirmed=True
                    )
                
                # 生成grasp文件 - 这里会计算完整的角度信息
                bounding_rect = tracking_result.get('bounding_rect', {})
                grasp_filename = self._generate_grasp_info_file(tracking_result, bounding_rect)
                
                if grasp_filename:
                    self._publish_grasp_trigger(grasp_filename, tracking_result['grasp_coordinate'])
                    self.tracking_active = False
                    self._finish_tracking_session()
                else:
                    self.get_logger().error(' 自动Failed to generate crawl file，转为手动确认')
                    self._start_grasp_confirmation_with_auto_info(tracking_result)
            else:
                # yaw对齐Detection failure，转为手动模式
                self.get_logger().warn(' auto模式yaw fail，转为手动确认模式')
                self._handle_auto_yaw_failure(tracking_result)
                    
        except Exception as e:
            self.get_logger().error(f'Direct crawling fails: {e}')

    def _publish_yaw_alignment_pose(self, grasp_coord: dict, target_yaw: float) -> bool:
        """发布yaw对齐位置"""
        try:
            from tf2_ros import TransformException
            import tf_transformations
            
            # 使用Current Location，只改变yaw角度
            if not self.current_tcp_pose:
                self.get_logger().error('无法获取当前TCP位置')
                return False
            
            current_pos = self.current_tcp_pose['position']
            
            # 构建目标位姿消息
            pose_msg = PoseStamped()
            pose_msg.header.stamp = self.get_clock().now().to_msg()
            pose_msg.header.frame_id = "base_link"
            
            # 位置保持Current Location（高度稍微抬高一点以确保安全）
            pose_msg.pose.position.x = float(current_pos['x'])
            pose_msg.pose.position.y = float(current_pos['y'])
            pose_msg.pose.position.z = float(current_pos['z']) + 30.0  # 抬高20mm
            
            # 计算目标四元数（只改变yaw）
            # 保持当前的roll和pitch，只修改yaw
            current_orientation = self.current_tcp_pose['orientation']
            current_roll, current_pitch, _ = self._quat_to_euler(
                current_orientation['x'], current_orientation['y'], 
                current_orientation['z'], current_orientation['w']
            )
            
            # 转换Target yaw angle为四元数
            target_quat = tf_transformations.quaternion_from_euler(
                np.radians(current_roll), 
                np.radians(current_pitch), 
                np.radians(target_yaw)
            )
            
            pose_msg.pose.orientation.x = float(target_quat[0])
            pose_msg.pose.orientation.y = float(target_quat[1])
            pose_msg.pose.orientation.z = float(target_quat[2])
            pose_msg.pose.orientation.w = float(target_quat[3])
            
            # 发布位姿
            self.tracking_result_pub.publish(pose_msg)
            
            self.get_logger().info(f' Published yaw alignment position: yaw={target_yaw:.1f}°, 高度={current_pos["z"]+20:.1f}mm')
            
            return True
            
        except Exception as e:
            self.get_logger().error(f'发布yaw对齐位置 fail : {e}')
            return False

    def _perform_secondary_detection_verification(self, original_result: dict) -> bool:
        """执行二次检测验证 - 重新用tracker检测验证grasp条件"""
        try:
            #  短暂 wait 让image稳定
            self.get_logger().info('  wait imagedata稳定...')
            time.sleep(0.5)
            
            # 验证imagedata
            if not self._validate_image_data():
                self.get_logger().error(' 二次检测imagedata fail')
                return False
            
            # 获取新的候选检测
            self.get_logger().info(' 获取yaw对齐后的候选检测...')
            candidate_detections = self._get_candidate_detections()
            
            if not candidate_detections:
                self.get_logger().warn(' 二次检测未发现候选目标')
                return False
            
            # 获取当前waypointdata
            waypoint_data = self._get_current_waypoint_data()
            
            # 执行tracker追踪
            self.get_logger().info(' 执行yaw对齐后的tracker验证...')
            tracking_result = self.tracker.track_target(
                self.current_target_id,
                self.latest_rgb_image,
                self.latest_depth_image,
                waypoint_data,
                candidate_detections
            )
            
            if not tracking_result:
                self.get_logger().warn(' 二次tracker fail')
                return False
            
            #  验证二次检测是否仍然满足grasp条件
            return self._verify_secondary_detection_for_grasp(original_result, tracking_result)
            
        except Exception as e:
            self.get_logger().error(f'二次检测 fail: {e}')
            return False

    def _verify_secondary_detection_for_grasp(self, original_result: dict, secondary_result: dict) -> bool:
        """验证二次检测是否仍然满足grasp条件"""
        try:
            self.get_logger().info(' 验证二次检测的grasp条件...')
            
            # 1. 检查置信度
            original_confidence = original_result.get('tracking_confidence', 0.0)
            secondary_confidence = secondary_result.get('tracking_confidence', 0.0)
            
            # 2. 检查位置稳定性
            orig_coord = original_result['grasp_coordinate']
            sec_coord = secondary_result['grasp_coordinate']
            
            position_diff = math.sqrt(
                (orig_coord['x'] - sec_coord['x'])**2 + 
                (orig_coord['y'] - sec_coord['y'])**2
            )
            
            # 3. 重新评估grasp条件 -  修复TCP位姿格式
            if not self.current_tcp_pose:
                self.get_logger().warn(' 无法获取当前TCP位姿，使用默认评估')
                # 简化的grasp条件检查
                confidence_stable = secondary_confidence >= 0.7
                position_stable = position_diff <= 40.0
                confidence_consistent = abs(secondary_confidence - original_confidence) <= 0.4
                grasp_still_ready = True  # 如果无法获取位姿，就信任tracker的结果
            else:
                #  使用正确的TCP位姿格式
                object_info = secondary_result['object_info']
                movement_strategy = self._calculate_adaptive_movement_strategy(
                    self.current_tcp_pose,  # 直接使用current_tcp_pose，不要包装
                    sec_coord, 
                    object_info
                )
                grasp_conditions = self._evaluate_grasp_conditions(secondary_result, movement_strategy)
                grasp_still_ready = grasp_conditions.get('grasp_ready', False)
                
                # 基本验证条件
                confidence_stable = secondary_confidence >= 0.7
                position_stable = position_diff <= 40.0
                confidence_consistent = abs(secondary_confidence - original_confidence) <= 0.4
            
                self.get_logger().info(f' Secondary detection verification result:')
                self.get_logger().info(f' Original confidence: {original_confidence: .3f}')
                self.get_logger().info(f' Secondary confidence: {secondary_confidence: .3f} (>= 0.6: {confidence_stable})')
                self.get_logger().info(f' Position deviation: {position_diff: .1f}mm (<= 40mm: {position_stable})')
                self.get_logger().info(f' Confidence difference: {abs(secondary_confidence - original_confidence): .3f} (<= 0.4: {confidence_consistent})')
                self.get_logger().info(f' Grasp condition: {grasp_still_ready}')
            
            # 综合判断
            verification_passed = confidence_stable and position_stable and confidence_consistent and grasp_still_ready
            
            if verification_passed:
                self.get_logger().info(' 二次检测验证通过，确认可以grasp')
                
                # 安全地更新tracking_result
                if self.current_tracking_result is not None:
                    original_bounding_rect = self.current_tracking_result.get('bounding_rect', {})
                    original_object_info = self.current_tracking_result.get('object_info', {})

                    self.current_tracking_result['grasp_coordinate'] = sec_coord
                    self.current_tracking_result['bounding_rect'] = original_bounding_rect
                    self.current_tracking_result['object_info'] = original_object_info
                    self.current_tracking_result['secondary_verification'] = {
                        'passed': True,
                        'secondary_confidence': secondary_confidence,
                        'position_difference': position_diff,
                        'grasp_conditions_met': grasp_still_ready
                    }
                    self.get_logger().info(' 已更新current_tracking_result')
                else:
                    self.get_logger().warn(' current_tracking_result为None，无法更新坐标')
                    # 作为备用，直接使用secondary_result
                    self.current_tracking_result = secondary_result.copy()
                    self.current_tracking_result['secondary_verification'] = {
                        'passed': True,
                        'secondary_confidence': secondary_confidence,
                        'position_difference': position_diff,
                        'grasp_conditions_met': grasp_still_ready
                    }
                    self.get_logger().info(' 已用secondary_result重建current_tracking_result')
            else:
                self.get_logger().warn(' 二次检测 fail，存在以下问题:')
            
            return verification_passed
            
        except Exception as e:
            self.get_logger().error(f'验证二次检测结果 fail : {e}')
            import traceback
            traceback.print_exc()
            return False
    
    def _handle_enhanced_user_confirmation(self, action: str):
        """处理增强版User确认 - 完整修复版"""
        try:
            self.user_feedback_received = True
            current_result_backup = self.current_tracking_result
            
            if action in ['success', 'auto_success']:
                is_success = True
                
                # 更新参数histroy记录
                tracking_params = self._extract_tracking_parameters(current_result_backup)
                tracking_params['user_feedback'] = action
                tracking_params['feedback_timestamp'] = datetime.now().isoformat()
                self.tracking_parameters_history.append(tracking_params)
                
                # 保存imagedata
                enhanced_result = current_result_backup.copy()
                enhanced_result['user_feedback'] = action  # 'success' 或 'auto_success'
                enhanced_result['feedback_timestamp'] = datetime.now().isoformat()
                
                self._save_tracking_step_data(enhanced_result, True)
                
                # 🔧 关键修复：更新已保存的详细data文件
                step_number = self.tracking_count + 1
                self._update_detailed_file_feedback(step_number, action)
                
                # 记录成功到histroy
                self._record_tracking_success_to_history(current_result_backup)
                
                if action == 'auto_success':
                    if hasattr(self.tracker, 'enable_auto_success_mode'):
                        self.tracker.enable_auto_success_mode(True, threshold=0.8)
                
            elif action == 'failure':
                is_success = False
                
                # 更新参数histroy
                tracking_params = self._extract_tracking_parameters(current_result_backup)
                tracking_params['user_feedback'] = 'failure'
                tracking_params['feedback_timestamp'] = datetime.now().isoformat()
                self.tracking_parameters_history.append(tracking_params)
                
                # 保存imagedata
                enhanced_result = current_result_backup.copy()
                enhanced_result['user_feedback'] = 'failure'
                enhanced_result['feedback_timestamp'] = datetime.now().isoformat()
                
                self._save_tracking_step_data(enhanced_result, False)
                
                # 🔧 更新详细data文件
                step_number = self.tracking_count + 1
                self._update_detailed_file_feedback(step_number, 'failure')
                
                self._record_tracking_failure_info(current_result_backup)
            
            # continue追踪流程
            self._continue_tracking_after_confirmation(current_result_backup, is_success)
            
            # 重置状态
            self.waiting_for_user_confirmation = False
            self.current_tracking_result = None
            
        except Exception as e:
            self.get_logger().error(f'处理增强User确认 fail : {e}')

    def _update_detailed_file_feedback(self, step_number: int, feedback: str):
        """更新详细data文件中的User反馈"""
        try:
            if not self.data_recorder:
                return
                
            # 调用data_recorder的更新方法
            timestamp = datetime.now().isoformat()
            self.data_recorder.update_step_detailed_feedback(step_number, feedback, timestamp)
            
            self.get_logger().info(f'  更新详细data文件反馈:  step {step_number} -> {feedback}')
            
        except Exception as e:
            self.get_logger().error(f'更新详细data文件反馈 fail : {e}')
    def _record_tracking_success_to_history(self, tracking_result: dict):
        """记录追踪成功到混合相似度histroy"""
        try:
            if hasattr(self.tracker, 'record_successful_detection_manual'):
                detection_data = self._extract_detection_from_tracking_result(tracking_result)
                similarity_data = tracking_result.get('hybrid_similarity_info', {})
                
                self.tracker.record_successful_detection_manual(
                    self.current_target_id, 
                    detection_data, 
                    similarity_data, 
                    user_confirmed=True
                )
                
                self.get_logger().info(' 成功记录已添加到混合相似度histroy')
                
        except Exception as e:
            self.get_logger().error(f'记录追踪成功到histroy fail : {e}')

    def _record_tracking_failure_info(self, tracking_result: dict):
        """记录追踪 fail 信息（不添加到成功histroy）"""
        try:
            # 可以记录 fail 统计，但不影响成功histroy
            failure_info = {
                'timestamp': datetime.now().isoformat(),
                'confidence': tracking_result.get('tracking_confidence', 0),
                'reason': 'user_marked_failure'
            }
            
            self.get_logger().info(' 追踪 fail 信息已记录')
            
        except Exception as e:
            self.get_logger().error(f'记录追踪 fail 信息 fail : {e}')

    def _extract_detection_from_tracking_result(self, tracking_result: dict) -> dict:
        """从追踪结果中提取检测data"""
        try:
            # 这个方法需要根据tracking_result的实际结构来实现
            # 提取检测相关的特征data
            return {
                'geometric_features': tracking_result.get('geometric_features', {}),
                'appearance_features': tracking_result.get('appearance_features', {}),
                'shape_features': tracking_result.get('shape_features', {}),
                'spatial_features': tracking_result.get('spatial_features', {}),
                'confidence': tracking_result.get('detection_confidence', 0.0),
                'class_name': tracking_result.get('object_info', {}).get('class_name', 'unknown')
            }
            
        except Exception as e:
            self.get_logger().error(f'提取检测data fail : {e}')
            return {}

    def _continue_tracking_after_confirmation(self, tracking_result: dict, success: bool):
        """确认后continue追踪流程 - 修复版，避免重复保存"""
        try:
            if success:
                # 发布移动指令
                if self._validate_pose_change(tracking_result):
                    self._publish_target_pose(tracking_result)
                    self.tracking_count += 1
                    self.movement_complete = False
                    
                    # 更新连续成功计数
                    self.consecutive_high_confidence += 1
                    
                    # 🔧 不再重复调用保存data，因为已经在确认时保存了
                    self.get_logger().info(f'   step  {self.tracking_count} 已完成data保存')
                else:
                    self.get_logger().warn(' 姿态变化超过限制，跳过移动')
            else:
                #  fail 情况也增加计数，但重置连续成功
                self.tracking_count += 1
                self.consecutive_high_confidence = 0
            
            # 检查是否完成
            if self.tracking_count >= self.max_tracking_steps:
                self._finish_tracking_session()
                
        except Exception as e:
            self.get_logger().error(f'确认后continue追踪 fail : {e}')

    def _finish_tracking_session(self):
        """完成追踪会话 - 混合相似度版本"""
        try:
            
            # 显示混合相似度统计
            if hasattr(self.tracker, 'get_detection_history_stats'):
                stats = self.tracker.get_detection_history_stats(self.current_target_id)
                if stats.get('history_length', 0) > 0:
                    self.get_logger().info(f' {self.current_target_id} Historical Statistics:')
                    self.get_logger().info(f' Number of historical records: {stats["history_length"]}')
                    self.get_logger().info(f'   Average confidence: {stats["avg_confidence"]:.3f}')
                    self.get_logger().info(f'   Confidence Range: {stats["min_confidence"]:.3f} - {stats["max_confidence"]:.3f}')
            
            # 保存混合相似度histroy
            if hasattr(self.tracker, '_save_detection_history'):
                self.tracker._save_detection_history()
            
            # 原有的完成逻辑
            self.tracking_active = False
            
            complete_data = {
                'status': 'completed',
                'total_steps': self.tracking_count,
                'target_id': self.current_target_id,
                'user_id': self.current_user_id,
                'consecutive_high_confidence': self.consecutive_high_confidence,
                'hybrid_mode_used': True
            }
            
            complete_msg = String()
            complete_msg.data = json.dumps(complete_data)
            self.tracking_complete_pub.publish(complete_msg)
            
            self.get_logger().info(f' Tracking completed: total steps = {self.tracking_count}, continuous high confidence = {self.consecutive_high_confidence}')
            
        except Exception as e:
            self.get_logger().error(f'Completed hybrid similarity tracking session fail : {e}')

    def _start_grasp_confirmation_with_auto_info(self, tracking_result: dict):
        """启动grasp确认（包含自动判断信息）"""
        try:
            self.current_tracking_result = tracking_result
            self.waiting_for_user_confirmation = True
            self.user_feedback_received = False
            self._grasp_confirmation_mode = True
            
            # 启动增强版grasp确认线程
            self._start_enhanced_grasp_confirmation_thread()
            
        except Exception as e:
            self.get_logger().error(f'启动增强grasp确认 fail : {e}')

    def _log_hybrid_tracking_details(self, tracking_result: dict):
        """Log hybrid similarity tracking details"""
        try:
            self.get_logger().info(f' Hybrid Similarity Tracking Details:')
            
            # Basic information
            self.get_logger().info(f'  Target ID: {tracking_result["target_id"]}')
            self.get_logger().info(f'  Final Confidence: {tracking_result["tracking_confidence"]:.3f}')
            
            # Hybrid similarity information
            hybrid_info = tracking_result.get('hybrid_similarity_info', {})
            if hybrid_info:
                ref_score = hybrid_info.get('reference_score', 0)
                hist_score = hybrid_info.get('historical_score', 0)
                hist_count = hybrid_info.get('history_count', 0)
                
                self.get_logger().info(f'   Reference Library Similarity: {ref_score:.3f}')
                self.get_logger().info(f'   Historical Similarity: {hist_score:.3f} (based on {hist_count} history entries)')
                
                fusion_weights = hybrid_info.get('fusion_weights', {})
                ref_weight = fusion_weights.get('reference_weight', 1.0)
                hist_weight = fusion_weights.get('historical_weight', 0.0)
                self.get_logger().info(f'  ⚖️ Fusion Weights: Reference {ref_weight:.2f}, Historical {hist_weight:.2f}')
            
            # Auto-judgment information
            auto_success = tracking_result.get('auto_success', False)
            if auto_success:
                self.get_logger().info(f'   Auto-Judgment: Succeeded')
            else:
                self.get_logger().info(f'   Manual Confirmation Required')
                
            # Grasp coordinates
            grasp_coord = tracking_result['grasp_coordinate']
            self.get_logger().info(f'   Grasp Coordinates: ({grasp_coord["x"]:.1f}, {grasp_coord["y"]:.1f}, {grasp_coord["z"]:.1f}mm)')
            
        except Exception as e:
            self.get_logger().error(f'Failed to log hybrid tracking details: {e}')

    def _grasp_confirmation_worker(self):
        """Grasp confirmation worker thread"""
        try:
            print(f"\n{'='*80}")
            print(f" The grasp condition has been met - Please confirm whether to execute grasp")
            print(f"{'='*80}")
            
            if self.current_tracking_result:
                grasp_coord = self.current_tracking_result['grasp_coordinate']
                object_info = self.current_tracking_result['object_info']
                
                print(f" Grasp Information:")
                print(f"   Target coordinates: ({grasp_coord['x']:.1f}, {grasp_coord['y']:.1f}, {grasp_coord['z']:.1f}mm)")
                print(f"   Object height: {object_info.get('estimated_height', 30):.1f}mm")
                print(f"   Background height: {object_info.get('background_z', 300):.1f}mm")
                print(f"   Recommended jaw width: {object_info.get('recommended_gripper_width', 150)}")
                print(f"   Tracking confidence: {self.current_tracking_result['tracking_confidence']:.3f}")
                print(f"   Step Number: {self.tracking_count + 1}")
            
            print(f"\n Command Options:")
            print(f"  'g' or 'grasp'    - Confirm to execute the crawl")
            print(f"  'c' or 'continue' - Cancel crawling and continue tracking")
            print(f"  'q' or 'quit'     - Opt-out of tracking")
            print(f"{'='*80}")
            
            while self.waiting_for_user_confirmation and not self.user_feedback_received:
                try:
                    user_input = input("Please enter the command (g/c/q): ").strip().lower()
                    
                    if user_input in ['g', 'grasp', 'grasp']:
                        self._handle_grasp_confirmation(True)
                        break
                    elif user_input in ['c', 'continue', 'countinue']:
                        self._handle_grasp_confirmation(False)
                        break
                    elif user_input in ['q', 'quit', 'quit']:
                        self._handle_quit_request()
                        break
                    else:
                        print(" Invalid input, please enter 'g'(grasp), 'c'(continue), or 'q'(quit)")
                        
                except (EOFError, KeyboardInterrupt):
                    print("\n User interrupt input")
                    self._handle_quit_request()
                    break
                except Exception as e:
                    print(f" Input processing errors: {e}")
                    
        except Exception as e:
            self.get_logger().error(f'Grasp confirmation worker failed: {e}')

    def _handle_grasp_confirmation(self, execute_grasp: bool):
        """Handle grasp confirmation result"""
        try:
            self.user_feedback_received = True
            current_result_backup = self.current_tracking_result
            
            if execute_grasp:
                self.get_logger().info(' User confirmed to execute the crawl')
                
                # Generate grasp info file
                bounding_rect = current_result_backup.get('bounding_rect', {})
                grasp_filename = self._generate_grasp_info_file(current_result_backup, bounding_rect)
                
                if grasp_filename:
                    # Publish grasp trigger signal
                    self._publish_grasp_trigger(grasp_filename, current_result_backup['grasp_coordinate'])
                    
                    # Save grasp confirmed tracking data
                    current_result_backup['user_feedback'] = 'grasp_confirmed'
                    current_result_backup['feedback_timestamp'] = datetime.now().isoformat()
                    
                    self._save_tracking_step_data(current_result_backup, True)
                    
                    # Complete tracking session
                    self.tracking_active = False
                    self._finish_tracking_session()
                else:
                    self.get_logger().error('Failed to generate grasp info file, continuing tracking')
                    self._handle_grasp_confirmation(False)
            else:
                self.get_logger().info(' User cancels crawling and continues tracking')
                
                # Mark to continue tracking
                current_result_backup['user_feedback'] = 'grasp_cancelled'
                current_result_backup['feedback_timestamp'] = datetime.now().isoformat()
                
                # Continue normal tracking confirmation process
                self._grasp_confirmation_mode = False
                self.waiting_for_user_confirmation = True
                self.user_feedback_received = False
                
                # Start normal tracking confirmation
                self._start_command_input_thread()
            
            # Reset grasp confirmation status
            if hasattr(self, '_grasp_confirmation_mode'):
                delattr(self, '_grasp_confirmation_mode')
                
        except Exception as e:
            self.get_logger().error(f'Handling grasp confirmation failed: {e}')

    def _generate_grasp_info_file(self, tracking_result: dict, bounding_rect: dict) -> str:
        """Generate grasp info file - enhanced yaw angle calculation"""
        try:
            # Improved width acquisition logic (unchanged)
            object_info = tracking_result.get('object_info', {})
            object_width = object_info.get('estimated_width')
            
            if object_width is None:
                gripper_info = object_info.get('gripper_width_info', {})
                object_width = gripper_info.get('real_width_mm', 50.0)
                self.get_logger().warn(f'Acquiring width from gripper_width_info: {object_width}mm')
            else:
                self.get_logger().info(f'Acquiring width from estimated_width: {object_width}mm')
            
            # Get current yaw angle
            current_yaw = 0.0
            if self.current_tcp_pose:
                orientation = self.current_tcp_pose.get('orientation', {})
                if orientation:
                    qx, qy, qz, qw = orientation.get('x', 0), orientation.get('y', 0), orientation.get('z', 0), orientation.get('w', 1)
                    _, _, current_yaw = self._quat_to_euler(qx, qy, qz, qw)
                    self.get_logger().info(f' Recording current yaw angle: {current_yaw:.1f}°')
                else:
                    self.get_logger().warn(' Unable to get orientation info, using default yaw=0°')
            else:
                self.get_logger().warn(' Unable to get current TCP pose, using default yaw=0°')
            
            # Get scan angle information from waypoint_data
            scan_yaw = 0.0
            if hasattr(self, '_last_waypoint_data') and self._last_waypoint_data:
                scan_yaw = self._last_waypoint_data.get('yaw', 0.0)
                self.get_logger().info(f' Scan yaw angle: {scan_yaw:.1f}°')
            
            # Enhanced yaw angle calculation (referencing logic from simple_grasp_validator.py)
            recommended_yaw = self._calculate_enhanced_grasp_yaw(
                current_yaw, scan_yaw, bounding_rect, object_info
            )
            
            grasp_info = {
                'target_id': tracking_result['target_id'],
                'object_coordinate': tracking_result['grasp_coordinate'],
                'background_height': tracking_result['object_info']['background_z'],
                'object_height': tracking_result['object_info']['estimated_height'],
                'object_width': float(object_width),
                'bounding_rect': bounding_rect,
                'recommended_gripper_width': tracking_result['object_info']['recommended_gripper_width'],
                'tracking_confidence': tracking_result['tracking_confidence'],
                'detection_confidence': tracking_result['detection_confidence'],
                
                # Enhanced angle info
                'grasp_angles': {
                    'current_yaw': float(current_yaw),
                    'scan_yaw': float(scan_yaw),
                    'recommended_yaw': float(recommended_yaw),
                    'bounding_rect_angle': float(bounding_rect.get('angle', 0.0)),
                    'calculation_method': 'enhanced_rect_analysis'  # Mark as using enhanced calculation
                }
            }
            
            # Unify filename to grasp_info.json
            filename = "grasp_info.json"
            filepath = os.path.join(self.current_scan_dir, 'grasp_commands', filename)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(grasp_info, f, indent=2, ensure_ascii=False)
            
            self.get_logger().info(f' Grasp info file generated: {filename}')            
            return filename
            
        except Exception as e:
            self.get_logger().error(f'Failed to generate grasp info file: {e}')
            import traceback
            traceback.print_exc()
            return None

    def _calculate_enhanced_grasp_yaw(self, current_yaw: float, scan_yaw: float, 
                                    bounding_rect: dict, object_info: dict) -> float:
        """Calculate enhanced grasp yaw angle - based on logic from simple_grasp_validator.py"""
        try:
            self.get_logger().info(' Starting enhanced yaw angle calculation...')
            
            # Get bounding rectangle info
            rect_width = bounding_rect.get('width', 0)
            rect_height = bounding_rect.get('height', 0)
            bounding_rect_angle = bounding_rect.get('angle', 0.0)
            
            if rect_width <= 0 or rect_height <= 0:
                self.get_logger().warn(' Invalid bounding rectangle size, using current yaw angle')
                return current_yaw
            
            # Calculate yaw_angle based on width-height relationship (grasp along the short side principle)
            if rect_width < rect_height:
                # Width is less than height, grasp along the width direction (short side)
                yaw_angle = -bounding_rect_angle
                self.get_logger().info(f' Width < Height, grasp along short side: yaw_angle = -{bounding_rect_angle:.1f}° = {yaw_angle:.1f}°')
            else:
                # Height is less than or equal to width, grasp along the height direction (short side)
                yaw_angle = -bounding_rect_angle + 90
                self.get_logger().info(f' Height <= Width, grasp along short side: yaw_angle = -{bounding_rect_angle:.1f}° + 90° = {yaw_angle:.1f}°')
            
            # Calculate the final yaw
            final_yaw = yaw_angle + current_yaw
            self.get_logger().info(f' Final yaw calculation: {yaw_angle:.1f}° + {current_yaw:.1f}° = {final_yaw:.1f}°')
            
            # Normalize angle to [-180, 180]
            final_yaw_normalized = self._normalize_angle(final_yaw)
            self.get_logger().info(f' Normalized yaw: {final_yaw_normalized:.1f}°')
            
            # Find the equivalent angle closest to current_yaw
            optimized_yaw, angle_diff = self._find_closest_equivalent_angle(final_yaw_normalized, current_yaw)
            
            self.get_logger().info(f' Optimized yaw angle:')
            self.get_logger().info(f'   Original calculated yaw: {final_yaw_normalized:.1f}°')
            self.get_logger().info(f'   Optimized yaw: {optimized_yaw:.1f}°')
            self.get_logger().info(f'   Difference from current yaw: {angle_diff:.1f}°')
            
            return optimized_yaw
            
        except Exception as e:
            self.get_logger().error(f'Enhanced yaw angle calculation failed: {e}')
            return current_yaw

    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to the range [-180, 180]"""
        while angle > 180:
            angle -= 360
        while angle < -180:
            angle += 360
        return angle

    def _find_closest_equivalent_angle(self, target_angle: float, reference_angle: float) -> tuple:
        """Find the equivalent angle closest to the reference angle"""
        try:
            # Generate all equivalent angles (a difference of 180 degrees is equivalent for grasping)
            candidates = [
                target_angle,
                target_angle + 180,  # 180-degree equivalence (for grasping)
                target_angle - 180
            ]
            
            # Normalize all candidate angles
            candidates = [self._normalize_angle(angle) for angle in candidates]
            
            # Find the one closest to the reference angle
            min_diff = float('inf')
            best_angle = target_angle
            
            for candidate in candidates:
                diff = abs(candidate - reference_angle)
                # Consider the cyclical nature of angles
                if diff > 180:
                    diff = 360 - diff
                
                if diff < min_diff:
                    min_diff = diff
                    best_angle = candidate
            
            return best_angle, min_diff
            
        except Exception as e:
            self.get_logger().error(f'Finding closest equivalent angle failed: {e}')
            return target_angle, 0.0
        
    def _extract_tracking_parameters(self, tracking_result: dict) -> dict:
        """Extract and log tracking parameters - fixed version, ensuring feature data is passed"""
        try:
            grasp_coord = tracking_result['grasp_coordinate']
            object_info = tracking_result['object_info']
            
            # Correctly extract fine-grained feature data
            detailed_features = tracking_result.get('detailed_similarity_breakdown', {})
            similarity_breakdown = tracking_result.get('similarity_breakdown', {})
            feature_weights = tracking_result.get('feature_weights_used', {})
            
            # Debugging output
            print(f"[TRACKING_PARAMS] Extracting feature data:")
            print(f"  detailed_features keys: {list(detailed_features.keys())}")
            print(f"  similarity_breakdown keys: {list(similarity_breakdown.keys())}")
            print(f"  feature_weights keys: {list(feature_weights.keys())}")
            
            # Calculate minimum bounding rectangle
            bbox_info = tracking_result.get('bounding_rect', {})
            
            # Calculate current pose
            current_pose = self._get_current_camera_pose()
            
            tracking_params = {
                'step': self.tracking_count + 1,
                'timestamp': datetime.now().isoformat(),
                'target_coordinate': {
                    'x': grasp_coord['x'],
                    'y': grasp_coord['y'], 
                    'z': grasp_coord['z']
                },
                'current_tcp_pose': current_pose,
                'object_properties': {
                    'height': object_info.get('estimated_height', 30.0),
                    'recommended_gripper_width': object_info.get('recommended_gripper_width', 150),
                    'class_name': object_info.get('class_name', 'unknown')
                },
                'bounding_rect': bbox_info,
                'tracking_confidence': tracking_result['tracking_confidence'],
                'detection_confidence': tracking_result['detection_confidence'],
                
                # Ensure feature data is correctly passed
                'detailed_features': detailed_features,
                'similarity_breakdown': similarity_breakdown,
                'feature_weights_used': feature_weights
            }
            
            print(f"[TRACKING_PARAMS] Final parameters include detailed_features: {len(tracking_params['detailed_features'])} items")
            print(f"[TRACKING_PARAMS] Final parameters include similarity_breakdown: {len(tracking_params['similarity_breakdown'])} items")
            
            return tracking_params
            
        except Exception as e:
            self.get_logger().error(f'Failed to extract tracking parameters: {e}')
            import traceback
            traceback.print_exc()
            return {}

    def _display_tracking_parameters(self, params: dict):
        """Display tracking parameters"""
        try:
            self.get_logger().info(' Current tracking parameters:')
            self.get_logger().info(f'   step : {params.get("step", "Unknown")}')
            
            target_coord = params.get('target_coordinate', {})
            self.get_logger().info(f'  Target coordinates: ({target_coord.get("x", 0):.1f}, {target_coord.get("y", 0):.1f}, {target_coord.get("z", 0):.1f}mm)')
            
            obj_props = params.get('object_properties', {})
            self.get_logger().info(f'  Object height: {obj_props.get("height", 0):.1f}mm')
            self.get_logger().info(f'  Recommended jaw width: {obj_props.get("recommended_gripper_width", 0)}')
            
            self.get_logger().info(f'  Tracking confidence: {params.get("tracking_confidence", 0):.3f}')
            self.get_logger().info(f'  Detection confidence: {params.get("detection_confidence", 0):.3f}')
            
        except Exception as e:
            self.get_logger().error(f'Failed to display tracking parameters: {e}')

    def _validate_pose_change(self, tracking_result: dict) -> bool:
        """Validate if the pose change is within the allowed range"""
        try:
            current_pose = self._get_current_camera_pose()
            current_yaw = current_pose.get('yaw', 0)
            
            # Record initial yaw angle
            if self.initial_yaw is None:
                self.initial_yaw = current_yaw
                self.get_logger().info(f' Recording initial yaw angle: {self.initial_yaw:.1f}°')
                return True
            
            # Check if yaw change exceeds the limit
            yaw_change = abs(current_yaw - self.initial_yaw)
            
            # Handle cases where the angle crosses the 180-degree boundary
            if yaw_change > 180:
                yaw_change = 360 - yaw_change
            
            if yaw_change > self.max_yaw_change:
                self.get_logger().warn(
                    f' Yaw angle change is too large: {yaw_change:.1f}° > {self.max_yaw_change}° '
                    f'(Initial: {self.initial_yaw:.1f}°, Current: {current_yaw:.1f}°)'
                )
                return False
            else:
                self.get_logger().info(f' Yaw angle change is normal: {yaw_change:.1f}°')
                return True
                
        except Exception as e:
            self.get_logger().error(f'Failed to validate pose change: {e}')
            return True  # Allow movement in case of exception

    def _save_tracking_step_data(self, tracking_result: dict, success: bool):
        """Save tracking step data - fixed version, ensures image saving"""
        try:
            if not self.data_recorder:
                self.get_logger().warn('Data recorder not initialized, unable to save data')
                return
            
            # Ensure tracking_result is not empty
            if not tracking_result:
                self.get_logger().warn('tracking_result is empty, unable to save data')
                return
            
            # 🔧 Key fix: check image data
            if self.latest_rgb_image is None:
                self.get_logger().error('RGB image data is empty, skipping image save')
                return
            
            if self.latest_depth_image is None:
                self.get_logger().error('Depth image data is empty, skipping image save')
                return
            
            # Merge user feedback into tracking_result
            enhanced_tracking_result = tracking_result.copy()
            enhanced_tracking_result['user_feedback'] = 'success' if success else 'failure'
            enhanced_tracking_result['feedback_timestamp'] = datetime.now().isoformat()
            
            # 🔧 Fix: Ensure image data is correctly passed, add debug info
            self.get_logger().info(f'Saving step data - RGB: {self.latest_rgb_image.shape}, Depth: {self.latest_depth_image.shape}')
            
            # Call the correct method signature (4 parameters)
            self.data_recorder.record_tracking_step(
                self.current_target_id,
                enhanced_tracking_result,
                self.latest_rgb_image,
                self.latest_depth_image
            )
            
            # Save tracking parameter history to file
            self._save_parameters_history()
            
            self.get_logger().info(f'   Step {self.tracking_count + 1} data and images saved')
            
        except Exception as e:
            self.get_logger().error(f'Failed to save tracking step data: {e}')
            import traceback
            traceback.print_exc()

    def _save_parameters_history(self):
        """Save tracking parameter history to file"""
        try:
            if not self.current_scan_dir:
                return
            
            output_dir = os.path.join(self.current_scan_dir, 'tracking_results')
            os.makedirs(output_dir, exist_ok=True)
            
            params_file = os.path.join(output_dir, 'tracking_parameters_history.json')
            
            with open(params_file, 'w', encoding='utf-8') as f:
                json.dump(self.tracking_parameters_history, f, indent=2, ensure_ascii=False)
            
            self.get_logger().info(f' Tracking parameter history saved: {params_file}')
            
        except Exception as e:
            self.get_logger().error(f'Failed to save parameter history: {e}')

    def _handle_quit_request(self):
        """Handle quit request"""
        try:
            self.get_logger().info(' User requested to quit tracking')
            
            # Save current data
            if self.current_tracking_result:
                tracking_params = self._extract_tracking_parameters(self.current_tracking_result)
                tracking_params['user_feedback'] = 'quit'
                self.tracking_parameters_history.append(tracking_params)
            
            self._save_parameters_history()
            
            # Stop tracking
            self.tracking_active = False
            self.waiting_for_user_confirmation = False
            self.user_feedback_received = True
            
            # Finish the session
            self._finish_tracking_session()
            
        except Exception as e:
            self.get_logger().error(f'Failed to handle quit request: {e}')


    def _validate_image_data(self):
        """Validate image data integrity and format - enhanced version"""
        try:
            if self.latest_rgb_image is None:
                self.get_logger().warn('RGB image data is empty')
                return False
                
            if self.latest_depth_image is None:
                self.get_logger().warn('Depth image data is empty')
                return False
            
            # Check RGB image format
            if len(self.latest_rgb_image.shape) != 3 or self.latest_rgb_image.shape[2] != 3:
                self.get_logger().error(f'Incorrect RGB image format: {self.latest_rgb_image.shape}')
                return False
            
            # Check depth image format
            if len(self.latest_depth_image.shape) != 2:
                self.get_logger().error(f'Incorrect depth image format: {self.latest_depth_image.shape}')
                return False
            
            # Strictly check if image dimensions match
            rgb_height, rgb_width = self.latest_rgb_image.shape[:2]
            depth_height, depth_width = self.latest_depth_image.shape
            
            if (rgb_height, rgb_width) != (depth_height, depth_width):
                self.get_logger().error(f' RGB and depth image dimensions do not match: RGB({rgb_height}, {rgb_width}) vs Depth({depth_height}, {depth_width})')
                
                # Attempt to resize the depth image here as an emergency fix
                try:
                    self.latest_depth_image = cv2.resize(
                        self.latest_depth_image, 
                        (rgb_width, rgb_height), 
                        interpolation=cv2.INTER_LINEAR
                    )
                    self.get_logger().info(f' Depth image emergency resized to: {self.latest_depth_image.shape}')
                except Exception as resize_error:
                    self.get_logger().error(f'Failed to emergency resize depth image: {resize_error}')
                    return False
            
            # Check depth data plausibility
            if self.latest_depth_image.max() < 100:
                self.get_logger().warn(f'Depth too small: max={self.latest_depth_image.max()}')
            
            return True
            
        except Exception as e:
            self.get_logger().error(f'Image data validation failed: {e}')
            return False
 
    
    def _setup_movement_monitoring(self):
        """Set up movement monitoring - referencing scan_executor's logic"""
        try:
            # Mimic scan_executor's movement monitoring
            self.movement_start_time = time.time()
            self.movement_completed = False
            
            # Start the movement completion check timer (similar to scan_executor)
            if hasattr(self, 'movement_check_timer'):
                self.movement_check_timer.destroy()
            
            self.movement_check_timer = self.create_timer(0.5, self.check_movement_completion)
            
            self.get_logger().info(' Starting to monitor robot arm movement completion...')
            
        except Exception as e:
            self.get_logger().error(f'Failed to set up movement monitoring: {e}')   

    def check_movement_completion(self):
        """Check for movement completion - referencing scan_executor's check logic"""
        try:
            # Timeout check (50 seconds, the timeout for scan_executor)
            if time.time() - self.movement_start_time > 50:
                self.get_logger().warn('Movement timed out, forcing continue')
                self._on_movement_completed()
                return
            if self.current_tcp_pose is not None:
                elapsed = time.time() - self.movement_start_time
                
                if elapsed > 2.0:  # Assume movement is complete after 2 seconds (simplified)
                    self.get_logger().info(' Movement completed (simplified detection)')
                    self._on_movement_completed()
                else:
                    self.get_logger().debug(f'  Waiting for movement completion... Elapsed time: {elapsed:.1f}s')
            
        except Exception as e:
            self.get_logger().error(f'Failed to check for movement completion: {e}')

    def _on_movement_completed(self):
        """Movement completion handling - referencing scan_executor's handling"""
        try:
            # Stop the movement check timer (scan_executor style)
            if hasattr(self, 'movement_check_timer'):
                self.movement_check_timer.destroy()
            
            self.movement_completed = True
            
            self.get_logger().info(' Robot arm movement completed, preparing for next tracking step')
            
            # Publish movement completion signal (simulating scan_executor's completion signal)
            movement_complete_msg = String()
            movement_complete_msg.data = "movement_completed"
            
            # A movement completion publisher should exist here, if not, create one
            if not hasattr(self, 'movement_complete_pub'):
                self.movement_complete_pub = self.create_publisher(String, '/xarm/movement_complete', 10)
            
            self.movement_complete_pub.publish(movement_complete_msg)
            
        except Exception as e:
            self.get_logger().error(f'Movement completion failed: {e}')

    def _publish_target_pose(self, tracking_result: dict):
        """Publish target pose to the robot arm - using adaptive movement strategy"""
        try:
            grasp_coord = tracking_result['grasp_coordinate']
            object_info = tracking_result['object_info']
            
            # Get current TCP position
            if not self.current_tcp_pose:
                self.get_logger().error('Unable to get current TCP position')
                return
            
            current_pos = self.current_tcp_pose['position']
            
            # Calculate adaptive movement strategy
            movement_strategy = self._calculate_adaptive_movement_strategy(
                self.current_tcp_pose, grasp_coord, object_info
            )
            
            # Use adaptive ratio instead of a fixed 25%
            target_x = current_pos['x'] + (grasp_coord['x'] - current_pos['x']) * movement_strategy['xy_movement_ratio']
            target_y = current_pos['y'] + (grasp_coord['y'] - current_pos['y']) * movement_strategy['xy_movement_ratio']
            
            # Use safe movement height
            target_z = movement_strategy['safe_movement_z']
            
            # Build pose message
            pose_msg = PoseStamped()
            pose_msg.header.stamp = self.get_clock().now().to_msg()
            pose_msg.header.frame_id = "base_link"
            
            # 🔧 Key fix: ensure all position data are of float type
            pose_msg.pose.position.x = float(target_x)
            pose_msg.pose.position.y = float(target_y)
            pose_msg.pose.position.z = float(target_z)
            
            # Keep orientation unchanged
            if self.current_tcp_pose and 'orientation' in self.current_tcp_pose:
                orientation = self.current_tcp_pose['orientation']
                pose_msg.pose.orientation.x = float(orientation['x'])
                pose_msg.pose.orientation.y = float(orientation['y'])
                pose_msg.pose.orientation.z = float(orientation['z'])
                pose_msg.pose.orientation.w = float(orientation['w'])
            else:
                # Default orientation
                pose_msg.pose.orientation.x = 0.0
                pose_msg.pose.orientation.y = 1.0
                pose_msg.pose.orientation.z = 0.0
                pose_msg.pose.orientation.w = 0.0
            
            # Publish target pose
            self.tracking_result_pub.publish(pose_msg)
            
            # Mark to wait for movement completion
            self.movement_complete = False
            
        except Exception as e:
            self.get_logger().error(f'Failed to publish target pose: {e}')

    
    def _publish_tracking_status(self, status: str, result: dict):
        """Publish tracking status"""
        try:
            status_data = {
                'status': status,
                'step': self.tracking_count,
                'target_id': self.current_target_id,
                'timestamp': datetime.now().isoformat()
            }
            
            if result:
                status_data['confidence'] = result.get('tracking_confidence', 0.0)
                status_data['position'] = result.get('grasp_coordinate', {})
            
            status_msg = String()
            status_msg.data = json.dumps(status_data)
            self.tracking_status_pub.publish(status_msg)
            
        except Exception as e:
            self.get_logger().error(f'Failed to publish tracking status: {e}')
    
    def _finish_tracking_session(self):
        """Finish the tracking session - fixed version, ensures feedback synchronization"""
        try:
            self.get_logger().info(' Tracking session completed, starting to collect feedback and synchronize data')
            
            # Immediately stop tracking to prevent continued execution
            self.tracking_active = False
            
            # Stop image processing
            if hasattr(self, '_last_tracking_time'):
                del self._last_tracking_time
            
            # Check feedback consistency
            if self.data_recorder:
                self.data_recorder.check_feedback_consistency()
            
            # Generate detailed data summary report
            self._generate_detailed_session_summary()
            
            # Collect user feedback
            self._collect_user_feedback()
            
            # Ensure feedback is synchronized before final save
            if self.data_recorder:
                self.get_logger().info(' Synchronizing feedback before final save...')
                self.data_recorder.update_session_feedback_from_details()
            
            # Publish tracking completion signal
            complete_data = {
                'status': 'completed',
                'total_steps': self.tracking_count,
                'target_id': self.current_target_id,
                'user_id': self.current_user_id
            }
            
            complete_msg = String()
            complete_msg.data = json.dumps(complete_data)
            self.tracking_complete_pub.publish(complete_msg)
            
            self.get_logger().info(f' Tracking completion stats: Total steps={self.tracking_count}, Target={self.current_target_id}')
            
            # Trigger optimization (if there are enough steps)
            if self.tracking_count >= 8:
                if hasattr(self.tracker, 'adaptive_manager') and self.tracker.adaptive_manager:
                    self.tracker.adaptive_manager.optimize_thresholds()
                    
        except Exception as e:
            self.get_logger().error(f'Failed to finish tracking session: {e}')

    def _generate_detailed_session_summary(self):
        """生成详细data的汇总报告"""
        try:
            if not self.data_recorder:
                return
            
            summary_data = {
                'session_overview': {
                    'target_id': self.current_target_id,
                    'total_steps': self.tracking_count,
                    'user_id': self.current_user_id,
                    'session_start': getattr(self.data_recorder, 'session_start_time', datetime.now()).isoformat(),
                    'session_end': datetime.now().isoformat()
                },
                'detailed_analysis_summary': self._analyze_detailed_steps(),
                'adaptive_weights_evolution': self._analyze_adaptive_weights_evolution(),
                'candidates_analysis_summary': self._analyze_candidates_patterns()
            }
            
            # 保存汇总报告
            summary_dir = os.path.join(self.data_recorder.data_dir, 'step_details')
            summary_file = os.path.join(summary_dir, 'session_detailed_summary.json')
            
            os.makedirs(summary_dir, exist_ok=True)
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, indent=2, ensure_ascii=False)
            
            self.get_logger().info(f' 详细汇总报告已生成: {summary_file}')
            
        except Exception as e:
            self.get_logger().error(f'生成详细汇总报告 fail : {e}')

    # 10. 新增方法：分析详细 step data
    def _analyze_detailed_steps(self) -> Dict:
        """分析所有详细 step data - 修复版，增强错误处理"""
        try:
            step_details_dir = os.path.join(self.data_recorder.data_dir, 'step_details')
            
            if not os.path.exists(step_details_dir):
                self.get_logger().warn('step_details 目录不存在，创建空分析结果')
                return {'error': 'step_details_dir_not_found', 'total_steps_analyzed': 0}
            
            analysis = {
                'total_steps_analyzed': 0,
                'successful_matches': 0,
                'user_feedback_stats': {'success': 0, 'failure': 0, 'no_feedback': 0},
                'confidence_evolution': [],
                'grasp_conditions_met': 0,
                'average_candidates_per_step': 0,
                'parsing_errors': 0
            }
            
            total_candidates = 0
            
            for step_num in range(1, self.tracking_count + 1):
                detail_file = os.path.join(step_details_dir, f'step_{step_num:02d}_detailed.json')
                
                if os.path.exists(detail_file):
                    try:
                        with open(detail_file, 'r', encoding='utf-8') as f:
                            step_data = json.load(f)
                        
                        analysis['total_steps_analyzed'] += 1
                        
                        # 安全地分析追踪结果
                        tracking_result = step_data.get('tracking_result', {})
                        confidence = tracking_result.get('tracking_confidence', 0)
                        
                        if confidence > 0:
                            analysis['successful_matches'] += 1
                            analysis['confidence_evolution'].append({
                                'step': step_num,
                                'confidence': confidence
                            })
                        
                        # User反馈统计
                        feedback = step_data.get('user_feedback')
                        if feedback == 'success':
                            analysis['user_feedback_stats']['success'] += 1
                        elif feedback == 'failure':
                            analysis['user_feedback_stats']['failure'] += 1
                        else:
                            analysis['user_feedback_stats']['no_feedback'] += 1
                        
                        # grasp条件统计
                        grasp_conditions = step_data.get('grasp_conditions', {})
                        if grasp_conditions.get('grasp_ready', False):
                            analysis['grasp_conditions_met'] += 1
                        
                        # 候选数量统计
                        candidates = step_data.get('all_candidates', [])
                        total_candidates += len(candidates)
                        
                    except json.JSONDecodeError as je:
                        self.get_logger().warn(f' step  {step_num} JSON解析 fail : {je}')
                        analysis['parsing_errors'] += 1
                    except Exception as e:
                        self.get_logger().warn(f'分析 step  {step_num} 详细data fail : {e}')
                        analysis['parsing_errors'] += 1
            
            if analysis['total_steps_analyzed'] > 0:
                analysis['average_candidates_per_step'] = total_candidates / analysis['total_steps_analyzed']
            
            return analysis
            
        except Exception as e:
            self.get_logger().error(f'分析详细 step  fail : {e}')
            return {'error': str(e), 'total_steps_analyzed': 0}

    # 11. 新增方法：分析自适应权重演化
    def _analyze_adaptive_weights_evolution(self) -> Dict:
        """分析自适应权重的演化过程"""
        try:
            step_details_dir = os.path.join(self.data_recorder.data_dir, 'step_details')
            
            weights_evolution = {
                'steps_with_adaptive_weights': 0,
                'weight_changes': [],
                'feature_weight_trends': {
                    'geometric': [],
                    'appearance': [],
                    'shape': [],
                    'spatial': []
                }
            }
            
            for step_num in range(1, self.tracking_count + 1):
                detail_file = os.path.join(step_details_dir, f'step_{step_num:02d}_detailed.json')
                
                if os.path.exists(detail_file):
                    try:
                        with open(detail_file, 'r', encoding='utf-8') as f:
                            step_data = json.load(f)
                        
                        adaptive_weights = step_data.get('adaptive_weights_used', {})
                        
                        if adaptive_weights:
                            weights_evolution['steps_with_adaptive_weights'] += 1
                            
                            # 记录各特征权重
                            for feature_type in weights_evolution['feature_weight_trends']:
                                if feature_type in adaptive_weights:
                                    weights_evolution['feature_weight_trends'][feature_type].append({
                                        'step': step_num,
                                        'weight': adaptive_weights[feature_type]
                                    })
                            
                            weights_evolution['weight_changes'].append({
                                'step': step_num,
                                'weights': adaptive_weights
                            })
                            
                    except Exception as e:
                        continue
            
            return weights_evolution
            
        except Exception as e:
            self.get_logger().error(f'分析权重演化 fail : {e}')
            return {'error': str(e)}

    # 12. 新增方法：分析候选模式
    def _analyze_candidates_patterns(self) -> Dict:
        """分析候选检测的模式和趋势"""
        try:
            step_details_dir = os.path.join(self.data_recorder.data_dir, 'step_details')
            
            patterns = {
                'total_candidates_analyzed': 0,
                'class_distribution': {},
                'similarity_score_distribution': {
                    'high': 0,      # > 0.8
                    'medium': 0,    # 0.5 - 0.8
                    'low': 0        # < 0.5
                },
                'best_match_evolution': [],
                'threshold_meeting_rate': 0
            }
            
            threshold_meeting_count = 0
            
            for step_num in range(1, self.tracking_count + 1):
                detail_file = os.path.join(step_details_dir, f'step_{step_num:02d}_detailed.json')
                
                if os.path.exists(detail_file):
                    try:
                        with open(detail_file, 'r', encoding='utf-8') as f:
                            step_data = json.load(f)
                        
                        candidates = step_data.get('all_candidates', [])
                        
                        for candidate in candidates:
                            patterns['total_candidates_analyzed'] += 1
                            
                            # 类别分布
                            class_name = candidate.get('detection_data', {}).get('class_name', 'unknown')
                            patterns['class_distribution'][class_name] = patterns['class_distribution'].get(class_name, 0) + 1
                            
                            # 相似度分布
                            similarity = candidate.get('similarity_to_target', {}).get('final_score', 0)
                            
                            if similarity > 0.8:
                                patterns['similarity_score_distribution']['high'] += 1
                            elif similarity > 0.5:
                                patterns['similarity_score_distribution']['medium'] += 1
                            else:
                                patterns['similarity_score_distribution']['low'] += 1
                            
                            # 阈值满足统计
                            if candidate.get('meets_threshold', False):
                                threshold_meeting_count += 1
                            
                            # 最佳匹配演化
                            if candidate.get('is_best_match', False):
                                patterns['best_match_evolution'].append({
                                    'step': step_num,
                                    'similarity': similarity,
                                    'class': class_name
                                })
                        
                    except Exception as e:
                        continue
            
            if patterns['total_candidates_analyzed'] > 0:
                patterns['threshold_meeting_rate'] = threshold_meeting_count / patterns['total_candidates_analyzed']
            
            return patterns
            
        except Exception as e:
            self.get_logger().error(f'分析候选模式 fail : {e}')
            return {'error': str(e)}
        
    def _collect_user_feedback(self):
        """收集User反馈 - 修复版，避免重复收集"""
        try:
            # 检查是否已经在real-time 过程中收集了所有反馈
            if hasattr(self, 'tracking_parameters_history') and self.tracking_parameters_history:
                steps_with_feedback = [step for step in self.tracking_parameters_history 
                                    if 'user_feedback' in step and step['user_feedback']]
                
                if len(steps_with_feedback) >= self.tracking_count:
                    self.get_logger().info(' 所有 step 已在real-time 过程中收集反馈，跳过额外反馈收集')
                    
                    # 保存学习结果
                    if hasattr(self, 'online_learner') and self.online_learner:
                        self.online_learner.save_learning_data()
                    if hasattr(self, 'data_recorder') and self.data_recorder:
                        self.data_recorder.save_session_data()
                    
                    self.get_logger().info(' 反馈收集完成，学习data已更新')
                    return
                else:
                    missing_count = self.tracking_count - len(steps_with_feedback)
                    self.get_logger().info(f' 还有 {missing_count} 步缺少反馈，启动补充收集')
            
            # 显示追踪结果可视化
            if hasattr(self, 'visualizer') and self.visualizer:
                self.visualizer.show_tracking_summary(
                    self.data_recorder.get_tracking_history(),
                    self.current_target_id
                )
            
            # 启动反馈收集线程
            feedback_thread = threading.Thread(
                target=self._feedback_collection_worker
            )
            feedback_thread.daemon = True
            feedback_thread.start()
            
        except Exception as e:
            self.get_logger().error(f'收集User反馈 fail : {e}')

    def _feedback_collection_worker(self):
        """反馈收集工作线程 - 修复重复问题"""
        try:
            #  检查是否已经有real-time 反馈
            if hasattr(self, 'tracking_parameters_history') and self.tracking_parameters_history:
                steps_with_feedback = [step for step in self.tracking_parameters_history if 'user_feedback' in step]
                if len(steps_with_feedback) == len(self.tracking_parameters_history):
                    self.get_logger().info(' 跳过重复反馈收集')
                    return
            
            tracking_history = self.data_recorder.get_tracking_history()
            
            for i, record in enumerate(tracking_history):
                #  检查是否已经有反馈
                if 'user_feedback' in record:
                    self.get_logger().info(f'  step  {i+1}: 已有反馈，跳过')
                    continue
                
                self.get_logger().info(f'  step  {i+1}: 请确认追踪结果是否正确')
                
                # 显示该 step 的image和结果
                if hasattr(self, 'visualizer') and self.visualizer:
                    self.visualizer.show_step_result(record, i+1)
                
                # 收集反馈
                feedback = self.feedback_collector.collect_feedback(
                    f" step  {i+1} 追踪是否正确？ (y/n): "
                )
                
                # 记录反馈
                record['human_feedback'] = 'correct' if feedback else 'incorrect'
                record['feedback_timestamp'] = datetime.now().isoformat()
                
                # 更新在线学习
                if hasattr(self, 'online_learner') and self.online_learner:
                    self.online_learner.update_with_feedback(
                        self.current_target_id,
                        record,
                        feedback
                    )
            
            # 保存学习结果
            if hasattr(self, 'online_learner') and self.online_learner:
                self.online_learner.save_learning_data()
            if hasattr(self, 'data_recorder') and self.data_recorder:
                self.data_recorder.save_session_data()
            
            self.get_logger().info(' 反馈收集完成，学习data已更新')
            
        except Exception as e:
            self.get_logger().error(f'反馈收集工作线程 fail : {e}')

    def _calculate_adaptive_movement_strategy(self, current_tcp_pose: dict, target_coord: dict, 
                                            object_info: dict) -> dict:
        """计算自适应移动策略 - 添加安全高度约束"""
        try:
            import math
            
            # 计算3D距离
            current_pos = current_tcp_pose['position']
            distance_2d = math.sqrt(
                (target_coord['x'] - current_pos['x'])**2 + 
                (target_coord['y'] - current_pos['y'])**2
            )
            z_compensation = math.sqrt(320**2 - 
               distance_2d**2
            )
            # 距离自适应策略
            if distance_2d > 200:
                xy_ratio = 0.40
            elif distance_2d > 100:
                xy_ratio = 0.40
            elif distance_2d > 50:
                xy_ratio = 0.5
            else:
                xy_ratio = 0.5
                
            #  安全移动高度计算
            background_z = object_info.get('background_z', 300)
            object_height = object_info.get('estimated_height', 30)
            safe_movement_z = max(350,background_z + object_height + z_compensation)
            
            return {
                'distance_3d': distance_2d,
                'xy_movement_ratio': xy_ratio,
                'z_compensation': z_compensation,
                'safe_movement_z': safe_movement_z,
                'strategy': 'adaptive_with_safety'
            }
            
        except Exception as e:
            self.get_logger().error(f'自适应移动策略计算 fail : {e}')
            return {
                'distance_3d': 100,
                'xy_movement_ratio': 0.25,
                'z_compensation': 200,
                'safe_movement_z': 300,
                'strategy': 'fallback'
            }

        
    def _evaluate_grasp_conditions(self, tracking_result: dict, movement_strategy: dict) -> dict:
        """评估grasp条件 - 修复Z距离判断逻辑"""
        try:
            # 获取当前和目标位置
            current_pos = self.current_tcp_pose['position']
            target_coord = tracking_result['grasp_coordinate']
            object_info = tracking_result['object_info']
            
            # 获取Background height和Object height
            background_z = object_info.get('background_z', 300)
            object_height = object_info.get('estimated_height', 30)
            
            # 计算期望的目标Z范围 (背景 + Object height ± 20mm)
            expected_target_z_min = background_z + object_height - 25
            expected_target_z_max = background_z + object_height + 25
            
            # 计算XY距离
            xy_distance = math.sqrt(
                (target_coord['x'] - current_pos['x'])**2 + 
                (target_coord['y'] - current_pos['y'])**2
            )
            
            #  修复后的grasp条件
            conditions = {
                'camera_min_height': current_pos['z'] >= 300,
                'z_in_expected_range': expected_target_z_min <= target_coord['z'] <= expected_target_z_max,
                'xy_aligned': xy_distance < 70.0,  # XY距离小于50mm
                'confidence_check': tracking_result['tracking_confidence'] > 0.7,
                'stability_check': self._check_tracking_stability()
            }
            
            # 综合判断
            grasp_ready = all(conditions.values())
            
            self.get_logger().info(f' grasp条件评估:')
            self.get_logger().info(f'   Background height: {background_z:.1f}mm')
            self.get_logger().info(f'   Object height: {object_height:.1f}mm')
            self.get_logger().info(f'   期望Z范围: [{expected_target_z_min:.1f}, {expected_target_z_max:.1f}]mm')
            self.get_logger().info(f'   目标Z坐标: {target_coord["z"]:.1f}mm (在范围内: {conditions["z_in_expected_range"]})')
            self.get_logger().info(f'   XY距离: {xy_distance:.1f}mm (< 50mm: {conditions["xy_aligned"]})')
            self.get_logger().info(f'   置信度: {tracking_result["tracking_confidence"]:.3f} (> 0.7: {conditions["confidence_check"]})')
            self.get_logger().info(f'   相机高度: {current_pos["z"]:.1f}mm (>= 300mm: {conditions["camera_min_height"]})')
            self.get_logger().info(f'   稳定性: {conditions["stability_check"]}')
            self.get_logger().info(f'    准备grasp: {grasp_ready}')
            
            return {
                'grasp_ready': grasp_ready,
                'conditions': conditions,
                'distances': {
                    'z_coordinate': target_coord['z'],
                    'expected_z_range': [expected_target_z_min, expected_target_z_max],
                    'xy_distance': xy_distance
                },
                'recommendation': 'grasp' if grasp_ready else 'continue_tracking'
            }
            
        except Exception as e:
            self.get_logger().error(f'grasp条件评估 fail : {e}')
            return {'grasp_ready': False, 'conditions': {}, 'recommendation': 'continue_tracking'}

    def _check_tracking_stability(self) -> bool:
        """检查追踪稳定性（最近3步位置变化）- 修复版"""
        try:
            # 检查参数histroy长度
            if not hasattr(self, 'tracking_parameters_history') or len(self.tracking_parameters_history) < 2:
                self.get_logger().debug(' 参数histroy不足，稳定性检查返回False')
                return False
            
            # 只考虑成功的 step 
            successful_steps = [step for step in self.tracking_parameters_history 
                            if step.get('user_feedback') in ['success', 'auto_success']]
            
            if len(successful_steps) < 2:
                self.get_logger().debug(' 成功 step 不足2步，稳定性检查返回False')
                return False
            
            # 取最近的2个成功 step 
            recent_positions = [step['target_coordinate'] for step in successful_steps[-2:]]
            
            max_deviation = 0
            for i in range(1, len(recent_positions)):
                deviation = math.sqrt(
                    (recent_positions[i]['x'] - recent_positions[i-1]['x'])**2 +
                    (recent_positions[i]['y'] - recent_positions[i-1]['y'])**2
                )
                max_deviation = max(max_deviation, deviation)
            
            is_stable = max_deviation < 30  # 30mm以内认为稳定
            
            self.get_logger().info(f' 稳定性检查: 最大偏差={max_deviation:.1f}mm, 稳定={is_stable}')
            
            return is_stable
            
        except Exception as e:
            self.get_logger().error(f'稳定性检查 fail : {e}')
            return False
    def _handle_detection_failure_with_retry(self):
        """处理Detection failure - 增强提示信息"""
        try:
            if not hasattr(self, 'consecutive_failures'):
                self.consecutive_failures = 0
            
            self.consecutive_failures += 1
            
            #  详细的 fail 提示
            self.get_logger().warn(f' Detection failure {self.consecutive_failures}/2')
            self.get_logger().info(f'Current tracking status:')
            self.get_logger().info(f'   Target ID: {self.current_target_id}')
            self.get_logger().info(f'   Current Location: {self._get_current_position_str()}')
            self.get_logger().info(f'   Tracking steps: {self.tracking_count}/{self.max_tracking_steps}')
            self.get_logger().info(f'   Tracking confidence histroy: {self._get_recent_confidence_str()}')
            
            if self.consecutive_failures < 2:
                self.get_logger().info('Retrying detection in 2 seconds...')
                self.get_logger().info('During retry, please ensure:')
                self.get_logger().info(' The target object is within the cameras field of view')
                self.get_logger().info(' Good lighting conditions')
                self.get_logger().info(' No obstructions')
                self.get_logger().info(' The camera is in focus')
                time.sleep(2)
                return False
            
            # 连续 fail 的详细提示
            self.get_logger().warn('Detection failed twice in a row!')
            self.get_logger().warn('Possible cause analysis:')
            self.get_logger().warn('1. The target object has moved out of the cameras field of view.')
            self.get_logger().warn('2. The lighting conditions have changed significantly.')
            self.get_logger().warn('3. The target object is obscured by other objects.')
            self.get_logger().warn('4. The camera angle is not suitable for the current detection.')
            self.get_logger().warn('5. The objects appearance has changed')

            self.get_logger().info('System suggestion:')
            self.get_logger().info(' Check if the camera field of view contains the target.')
            self.get_logger().info(' Adjust the ambient lighting conditions.')
            self.get_logger().info(' Remove possible obstructions')
                        
            user_confirms_failure = self._request_user_failure_confirmation()
            
            if user_confirms_failure:
                self.get_logger().info('User confirms Detection failure, preparing to roll back to the last successful location')
                self.get_logger().info('Searching for the most recent successful tracking location...')
                return self._rollback_to_last_successful_position()
            else:
                self.get_logger().info('User believes detection should be possible, resetting the fail count.')
                self.get_logger().info('The system will count detection attempts. Please ensure that environmental conditions are suitable.')
                self.consecutive_failures = 0
                return False
                
        except Exception as e:
            self.get_logger().error(f'Detection failure处理异常: {e}')
            return True
        
    def _get_current_position_str(self) -> str:
        """获取Current Location的字符串描述"""
        if self.current_tcp_pose:
            pos = self.current_tcp_pose['position']
            return f"[{pos['x']:.1f}, {pos['y']:.1f}, {pos['z']:.1f}]mm"
        return "位置未知"

    def _get_recent_confidence_str(self) -> str:
        """获取最近置信度histroy的字符串"""
        try:
            if hasattr(self, 'tracking_parameters_history') and self.tracking_parameters_history:
                recent_steps = self.tracking_parameters_history[-3:]  # 最近3步
                confidences = [f"{step.get('tracking_confidence', 0):.3f}" for step in recent_steps]
                return f"[{', '.join(confidences)}]"
            return "无histroydata"
        except:
            return "获取 fail "

    def _request_user_failure_confirmation(self) -> bool:
        """Request user confirmation for detection failure"""
        try:
            print(f"\n{'='*60}")
            print(f" Detection Failure Confirmation - Tracking Step {self.tracking_count + 1}")
            print(f"{'='*60}")
            print(f"Target ID: {self.current_target_id}")
            print(f"Current Location: {self._get_current_position_str()}")
            print(f"Consecutive Failures: {self.consecutive_failures}")
            print(f"\n Please confirm: Is the target object truly not visible at the current location?")
            print(f"   'y' - Confirm not visible (will backtrack to last successful position)")
            print(f"   'n' - Should be visible (continue detection attempts)")
            print(f"{'='*60}")
            
            while True:
                try:
                    user_input = input("Please enter your choice (y/n): ").strip().lower()
                    
                    if user_input in ['y', 'yes']:
                        print(" User confirmed: Target cannot be detected at current location")
                        return True
                    elif user_input in ['n', 'no', 'continue']:
                        print(" User confirmed: Target should be detectable, continue trying")
                        return False
                    else:
                        print(" Invalid input, please enter 'y' (confirm failure) or 'n' (continue trying)")
                        
                except (EOFError, KeyboardInterrupt):
                    print("\n User interrupted input, defaulting to confirm failure")
                    return True
                    
        except Exception as e:
            self.get_logger().error(f'Failed to request user confirmation: {e}')
            return True  # Default to confirm failure on exception
    def _rollback_to_last_successful_position(self) -> bool:
        """回退到上一个User确认成功的位置 - 增强版"""
        try:
            self.get_logger().info('Start fallback process analysis...')
            
            #find 最后一个User反馈为success的位置
            rollback_target = None
            for step in reversed(self.tracking_parameters_history):
                if step.get('user_feedback') == 'success' or step.get('user_feedback') == 'auto_success':
                    rollback_target = step
                    break
            
            if rollback_target is None:
                self.get_logger().error('No successful location to fallback to was found')
                return False
            
            # 显示回退信息
            self.get_logger().info(f' Find the fallback target: Steps {rollback_target["step"]}')
            self.get_logger().info(f' Fallback position: {rollback_target["target_coordinate"]}')
            self.get_logger().info(f' Tracking confidence of this location: {rollback_target.get("tracking_confidence", 0):.3f}')
            
            # 处理回退位置 - 修复字典访问问题
            target_coord = rollback_target['target_coordinate'].copy()  # 创建副本避免修改原data
            
            # 根据target_coordinate的data类型来处理
            if isinstance(target_coord, dict):
                # 如果是字典格式
                original_z = target_coord.get('z', 0)
                target_coord['z'] = 350.0  # 设置z坐标为350.0 (float类型)
            elif isinstance(target_coord, (list, tuple)):
                # 如果是列表或元组格式
                original_z = target_coord[2] if len(target_coord) > 2 else 0
                target_coord[2] = 350.0  # 使用float类型
            else:
                return False
            
            self.get_logger().info(f' 修改后的目标坐标: {target_coord}')
            
            # 发布回退位置
            self._publish_rollback_pose(target_coord)
            
            # 重置状态
            self.consecutive_failures = 0
            self.tracking_count = rollback_target['step']  # 重置追踪计数
            self.movement_complete = False  #  wait 移动完成
            
            self.get_logger().info(f' The retraction command has been issued, waiting for the robot arm to complete the movement')
            self.get_logger().info(f' The trace will start from step {self.tracking_count} continue')
            
            return True
            
        except Exception as e:
            self.get_logger().error(f'Fallback processing failed: {e}')
            import traceback
            self.get_logger().error(f'详细错误信息: {traceback.format_exc()}')
            return False

    def _publish_rollback_pose(self, target_coord: dict):
        """发布回退位置指令"""
        try:
            if isinstance(target_coord, dict):
                target_coord = {
                    'x': float(target_coord['x']),
                    'y': float(target_coord['y']),
                    'z': float(target_coord['z'])
                }
            pose_msg = PoseStamped()
            pose_msg.header.stamp = self.get_clock().now().to_msg()
            pose_msg.header.frame_id = "base_link"
            
            # 使用回退位置坐标
            pose_msg.pose.position.x = target_coord['x']
            pose_msg.pose.position.y = target_coord['y']  
            pose_msg.pose.position.z = target_coord['z']
            
            # 保持当前姿态
            if self.current_tcp_pose and 'orientation' in self.current_tcp_pose:
                orientation = self.current_tcp_pose['orientation']
                pose_msg.pose.orientation.x = orientation['x']
                pose_msg.pose.orientation.y = orientation['y']
                pose_msg.pose.orientation.z = orientation['z']
                pose_msg.pose.orientation.w = orientation['w']
            else:
                pose_msg.pose.orientation.x = 0.0
                pose_msg.pose.orientation.y = 1.0
                pose_msg.pose.orientation.z = 0.0
                pose_msg.pose.orientation.w = 0.0
            
            self.tracking_result_pub.publish(pose_msg)
            
            self.get_logger().info(f' Rollback instruction issued: ({target_coord["x"]:.1f}, {target_coord["y"]:.1f}, {target_coord["z"]:.1f}mm)')
            
        except Exception as e:
            self.get_logger().error(f'Publishing fallback location failed: {e}')     

    def _prepare_candidates_for_detailed_save(self, candidate_detections: List[Dict]) -> List[Dict]:
        try:
            candidates_backup = []
            
            for i, detection in enumerate(candidate_detections):
                candidate_backup = {
                    'original_index': i,
                    'detection_data': {
                        'bounding_box': detection.get('bounding_box', []).copy() if detection.get('bounding_box') else [],
                        'confidence': float(detection.get('confidence', 0.0)),
                        'class_id': detection.get('class_id', -1),
                        'class_name': str(detection.get('class_name', '')),
                        'mask_shape': detection.get('mask').shape if detection.get('mask') is not None else None,
                        'mask_area': int(detection.get('mask').sum()) if detection.get('mask') is not None else 0
                    },
                    'features': {
                        'geometric': detection.get('geometric_features', {}).copy() if detection.get('geometric_features') else {},
                        'appearance': detection.get('appearance_features', {}).copy() if detection.get('appearance_features') else {},
                        'shape': detection.get('shape_features', {}).copy() if detection.get('shape_features') else {},
                        'spatial': detection.get('spatial_features', {}).copy() if detection.get('spatial_features') else {}
                    },
                    'raw_detection': False  # 标记这是原始检测，非追踪结果
                }
                candidates_backup.append(candidate_backup)
            
            # self.get_logger().info(f' 准备了 {len(candidates_backup)} 个候选的备份data')
            return candidates_backup
            
        except Exception as e:
            self.get_logger().error(f'准备候选data fail : {e}')
            return []

    def _save_detailed_step_data(self, step_number: int, candidates_backup: List[Dict], 
                            tracking_result: Dict, waypoint_data: Dict):
        """保存详细 step data - 修复版，添加data验证"""
        try:
            # data验证
            if not tracking_result:
                self.get_logger().warn(f' step  {step_number}: tracking_result 为空，跳过详细data保存')
                return
            
            if not candidates_backup:
                self.get_logger().warn(f' step  {step_number}: candidates_backup 为空，使用空列表')
                candidates_backup = []
            
            # 计算移动策略和grasp条件
            movement_strategy = {}
            grasp_conditions = {}
            
            try:
                movement_strategy = self._calculate_adaptive_movement_strategy(
                    self.current_tcp_pose, 
                    tracking_result.get('grasp_coordinate', {}), 
                    tracking_result.get('object_info', {})
                )
            except Exception as e:
                self.get_logger().warn(f'计算移动策略 fail : {e}')
            
            try:
                grasp_conditions = self._evaluate_grasp_conditions(tracking_result, movement_strategy)
            except Exception as e:
                self.get_logger().warn(f'评估grasp条件 fail : {e}')
            
            # 合并原始候选和追踪分析结果
            all_candidates_data = self._merge_candidates_and_analysis(
                candidates_backup, 
                tracking_result.get('all_candidates_analysis', [])
            )
            
            # 获取当前使用的自适应权重
            adaptive_weights = tracking_result.get('adaptive_weights_used', {})
            
            # data验证：确保 current_tcp_pose 不为空
            tcp_pose_to_save = self.current_tcp_pose or {
                'position': {'x': 0, 'y': 0, 'z': 0},
                'orientation': {'x': 0, 'y': 0, 'z': 0, 'w': 1},
                'timestamp': None
            }
            
            # 调用data记录器保存详细data
            if self.data_recorder and hasattr(self.data_recorder, 'save_step_detailed_data'):
                self.data_recorder.save_step_detailed_data(
                    step_number=step_number,
                    target_id=self.current_target_id,
                    all_candidates_data=all_candidates_data,
                    tracking_result=tracking_result,
                    waypoint_data=waypoint_data,
                    current_tcp_pose=tcp_pose_to_save,
                    movement_strategy=movement_strategy,
                    grasp_conditions=grasp_conditions,
                    adaptive_weights=adaptive_weights
                )
                
                self.get_logger().info(f'  step  {step_number} 详细data已保存')
            else:
                if not self.data_recorder:
                    self.get_logger().warn(' data记录器未Initialized，跳过详细data保存')
                else:
                    self.get_logger().warn(' data记录器缺少 save_step_detailed_data 方法，跳过详细data保存')
                
        except Exception as e:
            self.get_logger().error(f'保存详细 step data fail : {e}')
            import traceback
            traceback.print_exc()

    def _merge_candidates_and_analysis(self, candidates_backup: List[Dict], 
                                    analysis_results: List[Dict]) -> List[Dict]:
        """合并原始候选data和追踪分析结果"""
        try:
            # 创建分析结果的索引映射
            analysis_map = {}
            for analysis in analysis_results:
                original_idx = analysis.get('candidate_index', -1)
                analysis_map[original_idx] = analysis
            
            merged_candidates = []
            
            for candidate in candidates_backup:
                original_idx = candidate['original_index']
                
                # 基础候选信息
                merged_candidate = candidate.copy()
                
                # 如果有对应的分析结果，添加分析信息
                if original_idx in analysis_map:
                    analysis = analysis_map[original_idx]
                    merged_candidate.update({
                        'similarity_to_target': analysis.get('similarity_to_target', {}),
                        'is_best_match': analysis.get('is_best_match', False),
                        'meets_threshold': analysis.get('meets_threshold', False),
                        'was_analyzed': True
                    })
                else:
                    # 未分析的候选（可能因为类别不匹配）
                    merged_candidate.update({
                        'similarity_to_target': {},
                        'is_best_match': False,
                        'meets_threshold': False,
                        'was_analyzed': False,
                        'skip_reason': 'class_mismatch_or_error'
                    })
                
                merged_candidates.append(merged_candidate)
            
            self.get_logger().info(f'🔗 合并了 {len(merged_candidates)} 个候选data，其中 {len(analysis_results)} 个有分析结果')
            return merged_candidates
            
        except Exception as e:
            self.get_logger().error(f'合并候选data fail : {e}')
            return candidates_backup


def main(args=None):
    rclpy.init(args=args)
    
    try:
        tracking_node = TrackingNode()
        tracking_node.get_logger().info(' Track节点运行中...')
        rclpy.spin(tracking_node)
    except KeyboardInterrupt:
        print(' Track节点被User中断')
    except Exception as e:
        print(f' Track节点运行错误: {e}')
    finally:
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()