import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String, Int32
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger
import cv2
import numpy as np
import json
import os
import time
import threading
from cv_bridge import CvBridge
from typing import Dict, List, Optional
import glob
import math

# 临时禁用新服务，强制使用topic模式
GRIPPER_SERVICES_AVAILABLE = False

class TrackingGraspSystem(Node):
    """追踪抓取系统 - 专门处理tracking触发的单个目标抓取"""
    
    def __init__(self):
        super().__init__('tracking_grasp_system')
        
        # 系统状态
        self.system_state = "WAITING_FOR_TRACKING"
        self.current_scan_dir = None
        self.current_target = None
        
        # 机械臂连接状态
        self.arm_connected = False
        self.arm_pose_received = False
        
        # 相机和可视化
        self.bridge = CvBridge()
        self.current_image = None
        
        # ROS2 Publishers
        self.target_pose_pub = self.create_publisher(PoseStamped, '/xarm/target_pose', 10)
        self.gripper_control_pub = self.create_publisher(Int32, '/xarm/gripper_target', 10)
        self.grasp_complete_pub = self.create_publisher(String, '/grasp_complete', 10)
        
        # ROS2 Subscribers
        self.grasp_trigger_sub = self.create_subscription(
            String, '/grasp_trigger', self.grasp_trigger_callback, 10)
        
        # 🆕 订阅机械臂位姿确认连接状态（参考tracking_node）
        self.tcp_pose_sub = self.create_subscription(
            PoseStamped, '/xarm/current_pose', self.tcp_pose_callback, 10
        )
        
        self.color_image_sub = self.create_subscription(
            Image, '/camera/color/image_raw', self.image_callback, 10)
        
        # ROS2 Service Clients
        self.gripper_open_client = self.create_client(Trigger, '/xarm/gripper_open')
        self.gripper_close_client = self.create_client(Trigger, '/xarm/gripper_close')
        self.go_home_client = self.create_client(Trigger, '/xarm/go_home')
        
        # 显示更新定时器
        self.display_timer = self.create_timer(0.1, self.update_display)
        
        # 启动检查
        self.startup_timer = self.create_timer(2.0, self.check_system_ready)
        
        self.get_logger().info('🎯 追踪抓取系统初始化完成')
        self.get_logger().info('⏳ 等待追踪触发信号和机械臂连接...')
    
    def tcp_pose_callback(self, msg):
        """TCP位姿回调 - 确认机械臂连接"""
        if not self.arm_pose_received:
            self.get_logger().info('🤖 检测到机械臂连接')
            self.arm_pose_received = True
    
    def check_system_ready(self):
        """检查系统就绪状态"""
        if not self.arm_connected and self.arm_pose_received:
            self.arm_connected = True
            self.get_logger().info('✅ 机械臂连接就绪')
            
            # 检查服务可用性
            services_ready = (
                self.gripper_open_client.wait_for_service(timeout_sec=1.0) and
                self.gripper_close_client.wait_for_service(timeout_sec=1.0) and
                self.go_home_client.wait_for_service(timeout_sec=1.0)
            )
            
            if services_ready:
                self.get_logger().info('✅ 机械臂服务就绪')
            else:
                self.get_logger().warn('⚠️ 部分机械臂服务不可用，但将尝试继续')
    
    def grasp_trigger_callback(self, msg):
        """处理追踪触发的抓取信号"""
        try:
            trigger_data = json.loads(msg.data)
            self.get_logger().info(f'🎯 收到追踪抓取触发信号')
            self.get_logger().info(f'   文件名: {trigger_data.get("filename", "未知")}')
            self.get_logger().info(f'   目标ID: {trigger_data.get("target_id", "未知")}')
            self.get_logger().info(f'   触发原因: {trigger_data.get("trigger_reason", "未知")}')
            
            # 检查系统状态
            if not self.arm_connected:
                self.get_logger().error('❌ 机械臂未连接，无法执行抓取')
                self.publish_grasp_complete(trigger_data.get('target_id', 'unknown'), False, 'arm_not_connected')
                return
            
            # 🆕 设置当前扫描目录（从trigger_data获取）
            scan_directory = trigger_data.get('scan_directory')
            if scan_directory:
                self.current_scan_dir = scan_directory
                self.get_logger().info(f'📁 设置扫描目录: {self.current_scan_dir}')
            else:
                self.get_logger().warn('⚠️ 未收到扫描目录信息，将尝试自动查找')
            
            # 启动抓取序列
            self.system_state = "LOADING_TARGET"
            threading.Thread(target=self.execute_tracking_grasp_sequence, 
                           args=(trigger_data,), daemon=True).start()
            
        except Exception as e:
            self.get_logger().error(f'处理追踪抓取触发失败: {e}')
    
    def execute_tracking_grasp_sequence(self, trigger_data):
        """执行追踪触发的抓取序列"""
        try:
            target_id = trigger_data.get('target_id', 'unknown')
            
            # 1. 查找并加载grasp_info.json文件
            self.get_logger().info('📁 步骤1: 查找grasp_info.json文件...')
            grasp_info = self.find_and_load_grasp_info()
            
            if not grasp_info:
                self.get_logger().error('❌ 未找到grasp_info.json文件')
                self.publish_grasp_complete(target_id, False, 'grasp_info_not_found')
                return
            
            # 2. 验证目标ID匹配
            if grasp_info.get('target_id') != target_id:
                self.get_logger().warn(f'⚠️ 目标ID不匹配: 期望 {target_id}, 实际 {grasp_info.get("target_id")}')
            
            # 3. 构建目标对象
            self.system_state = "PREPARING_GRASP"
            target = self.build_target_from_grasp_info(grasp_info)
            self.current_target = target
            
            self.get_logger().info('🎯 目标信息:')
            self.get_logger().info(f'   ID: {target["object_id"]}')
            self.get_logger().info(f'   坐标: {target["world_coordinates"]}')
            self.get_logger().info(f'   物体宽度: {target["features"]["spatial"]["gripper_width_info"]["real_width_mm"]:.1f}mm')
            self.get_logger().info(f'   物体高度: {target["features"]["depth_info"]["height_mm"]:.1f}mm')
            self.get_logger().info(f'   背景高度: {target["features"]["depth_info"]["background_world_z"]:.1f}mm')
            
            # 4. 执行抓取
            self.system_state = "EXECUTING_GRASP"
            self.get_logger().info('🤖 开始执行抓取序列...')
            
            success = self.execute_single_grasp_and_place(target)
            
            # 5. 发布结果
            if success:
                self.get_logger().info('✅ 追踪抓取序列执行成功!')
                self.publish_grasp_complete(target_id, True, 'success')
                self.system_state = "COMPLETED"
            else:
                self.get_logger().error('❌ 追踪抓取序列执行失败')
                self.publish_grasp_complete(target_id, False, 'execution_failed')
                self.system_state = "FAILED"
            
            # 6. 重置等待下次触发
            time.sleep(3)
            self.system_state = "WAITING_FOR_TRACKING"
            self.current_target = None
            self.get_logger().info('🔄 系统重置，等待下次追踪触发...')
            
        except Exception as e:
            self.get_logger().error(f'追踪抓取序列失败: {e}')
            import traceback
            traceback.print_exc()
            self.publish_grasp_complete(trigger_data.get('target_id', 'unknown'), False, 'exception')
            self.system_state = "ERROR"
    
    def find_and_load_grasp_info(self) -> Optional[Dict]:
        """查找并加载grasp_info.json文件 - 修复版，优先使用指定目录"""
        try:
            grasp_info_file = None
            
            # 🆕 方法1：如果有指定的扫描目录，直接使用
            if self.current_scan_dir:
                grasp_commands_dir = os.path.join(self.current_scan_dir, 'grasp_commands')
                candidate_file = os.path.join(grasp_commands_dir, 'grasp_info.json')
                
                if os.path.exists(candidate_file):
                    grasp_info_file = candidate_file
                    self.get_logger().info(f'✅ 使用指定扫描目录: {self.current_scan_dir}')
                else:
                    self.get_logger().warn(f'⚠️ 指定目录中未找到grasp_info.json: {candidate_file}')
            
            # 🆕 方法2：如果指定目录无效，在/home/qi/ros2_ws/下查找最新的scan_output目录
            if not grasp_info_file:
                self.get_logger().info('🔍 在/home/qi/ros2_ws/下查找最新scan_output目录...')
                base_path = "/home/qi/ros2_ws"
                scan_pattern = os.path.join(base_path, "scan_output_*")
                scan_dirs = glob.glob(scan_pattern)
                
                if not scan_dirs:
                    self.get_logger().error('❌ 未找到任何scan_output目录')
                    return None
                
                # 按时间排序，获取最新的
                scan_dirs.sort(key=lambda x: os.path.getctime(x), reverse=True)
                latest_scan_dir = scan_dirs[0]
                self.current_scan_dir = latest_scan_dir
                
                self.get_logger().info(f'📁 使用最新扫描目录: {latest_scan_dir}')
                
                # 查找grasp_info.json文件
                grasp_commands_dir = os.path.join(latest_scan_dir, 'grasp_commands')
                grasp_info_file = os.path.join(grasp_commands_dir, 'grasp_info.json')
            
            # 验证文件存在
            if not grasp_info_file or not os.path.exists(grasp_info_file):
                self.get_logger().error(f'❌ grasp_info.json不存在: {grasp_info_file}')
                return None
            
            # 加载文件
            with open(grasp_info_file, 'r', encoding='utf-8') as f:
                grasp_info = json.load(f)
            
            self.get_logger().info(f'✅ 成功加载grasp_info.json')
            self.get_logger().info(f'   文件路径: {grasp_info_file}')
            self.get_logger().info(f'   目标ID: {grasp_info.get("target_id", "未知")}')
            
            return grasp_info
            
        except Exception as e:
            self.get_logger().error(f'查找和加载grasp_info.json失败: {e}')
            return None
    
    def build_target_from_grasp_info(self, grasp_info: Dict) -> Dict:
        """从grasp_info构建target对象，兼容现有抓取逻辑"""
        try:
            # 🔧 计算背景深度（从世界坐标推算）
            background_world_z = grasp_info['background_height']
            object_z = grasp_info['object_coordinate']['z']
            
            # 估算背景深度（米）- 这是一个近似值
            estimated_background_depth_m = abs(background_world_z - object_z) / 1000.0
            if estimated_background_depth_m < 0.1:
                estimated_background_depth_m = 0.3  # 默认值
            
            target = {
                'object_id': grasp_info['target_id'],
                'class_name': grasp_info['target_id'].split('_')[0],  # 从ID提取类别
                'world_coordinates': [
                    grasp_info['object_coordinate']['x'],
                    grasp_info['object_coordinate']['y'],
                    grasp_info['object_coordinate']['z']
                ],
                'features': {
                    'depth_info': {
                        'background_world_z': background_world_z,
                        'height_mm': grasp_info['object_height'],
                        'background_depth_m': estimated_background_depth_m
                    },
                    'spatial': {
                        'gripper_width_info': {
                            'real_width_mm': grasp_info['object_width']
                        },
                        'scan_detail': [0, 0, 0]  # 默认扫描角度
                    }
                },
                'bounding_rect': grasp_info.get('bounding_rect', {}),
                'description': f"{grasp_info['target_id']} (追踪抓取)",
                'mask': None,  # 追踪抓取不需要mask
                'confidence': grasp_info.get('tracking_confidence', 0.8),
                'detection_confidence': grasp_info.get('detection_confidence', 0.8)
            }
            
            return target
            
        except Exception as e:
            self.get_logger().error(f'构建target对象失败: {e}')
            raise
    
    def publish_grasp_complete(self, target_id: str, success: bool, reason: str = ''):
        """发布抓取完成信号"""
        try:
            complete_data = {
                'target_id': target_id,
                'success': success,
                'reason': reason,
                'timestamp': time.time()
            }
            
            msg = String()
            msg.data = json.dumps(complete_data)
            self.grasp_complete_pub.publish(msg)
            
            status = "成功" if success else "失败"
            self.get_logger().info(f'📤 发布抓取完成信号: {target_id} - {status}')
            if reason:
                self.get_logger().info(f'   原因: {reason}')
                
        except Exception as e:
            self.get_logger().error(f'发布抓取完成信号失败: {e}')
    
    # 🔄 复用原有抓取逻辑 - 以下方法基本保持不变
    def execute_single_grasp_and_place(self, target: Dict) -> bool:
        """执行抓取和放置序列 - 复用原有逻辑"""
        try:
            self.get_logger().info(f'🤖 开始抓取序列: {target["description"]}')
            
            # 获取增强信息
            depth_info = target.get('features', {}).get('depth_info', {})
            spatial_features = target.get('features', {}).get('spatial', {})
            gripper_info = spatial_features.get('gripper_width_info', {})
            
            # 基础位置信息
            target_x, target_y, target_z = target['world_coordinates'][:3]
            
            # 使用正确的背景Z补偿公式
            background_world_z = depth_info.get('background_world_z', target_z)
            background_depth_m = depth_info.get('background_depth_m', 0.3)
            object_height_mm = depth_info.get('height_mm', 30.0)
            
            # 计算真实的背景补偿高度
            background_z_compensated = background_world_z - (background_depth_m * 1000)
            
            self.get_logger().info(f'📏 深度信息分析:')
            self.get_logger().info(f'   背景世界Z: {background_world_z:.1f}mm')
            self.get_logger().info(f'   背景深度: {background_depth_m:.3f}m ({background_depth_m*1000:.1f}mm)')
            self.get_logger().info(f'   补偿后背景Z: {background_z_compensated:.1f}mm')
            self.get_logger().info(f'   物体高度: {object_height_mm:.1f}mm')
            
            # 使用正确的抓夹宽度计算公式
            if gripper_info and 'real_width_mm' in gripper_info:
                real_width_mm = gripper_info['real_width_mm']
                
                # 抓夹宽度公式
                pre_grasp_width = int(real_width_mm * 10 + 50)
                final_grasp_width = int(real_width_mm * 10 - 150)
                
                # 确保在合理范围内
                pre_grasp_width = max(400, min(850, pre_grasp_width))
                final_grasp_width = max(50, min(800, final_grasp_width))
                
                self.get_logger().info(f'🤏 抓夹宽度计算:')
                self.get_logger().info(f'   实际宽度: {real_width_mm:.1f}mm')
                self.get_logger().info(f'   预抓取宽度: {real_width_mm:.1f} * 10 + 50 = {pre_grasp_width}')
                self.get_logger().info(f'   最终抓取宽度: {real_width_mm:.1f} * 10 - 150 = {final_grasp_width}')
                
            else:
                # 使用默认值
                pre_grasp_width = 600
                final_grasp_width = 300
                self.get_logger().warn('⚠️ 未找到real_width_mm，使用默认抓夹宽度')
            
            # 计算yaw角度（简化版，因为没有mask）
            yaw = 0.0  # 追踪抓取使用默认角度
            
            self.get_logger().info(f'🎯 抓取参数总结:')
            self.get_logger().info(f'   yaw角度: {yaw:.1f}°')
            self.get_logger().info(f'   预抓取宽度: {pre_grasp_width}')
            self.get_logger().info(f'   最终抓取宽度: {final_grasp_width}')
            
            # 基于补偿后的背景高度计算抓取策略
            strategy = self.plan_grasp_strategy_with_compensation(
                object_height_mm, background_z_compensated, yaw
            )
            
            # 计算实际抓取位置
            actual_grasp_z = background_z_compensated + strategy['z_offset_above_background']
            
            self.get_logger().info(f'🎯 抓取高度计算:')
            self.get_logger().info(f'   补偿背景Z: {background_z_compensated:.1f}mm')
            self.get_logger().info(f'   Z偏移: {strategy["z_offset_above_background"]:.1f}mm')
            self.get_logger().info(f'   最终抓取Z: {actual_grasp_z:.1f}mm')
            
            # 计算pitch补偿
            compensated_x, compensated_y, compensated_z = self.calculate_pitch_compensation(
                target_x, target_y, actual_grasp_z, 
                strategy['pitch'], yaw
            )
            
            self.get_logger().info(f'🔧 Pitch补偿后坐标: [{compensated_x:.1f}, {compensated_y:.1f}, {compensated_z:.1f}]')
            
            # 执行抓取序列
            success = self._execute_enhanced_grasp_sequence(
                compensated_x, compensated_y, compensated_z,
                strategy, yaw, pre_grasp_width, final_grasp_width, 
                background_z_compensated, target
            )
            
            return success
            
        except Exception as e:
            self.get_logger().error(f'单个抓取序列失败: {e}')
            import traceback
            traceback.print_exc()
            return False
    
    # 以下方法从原有系统复用，保持不变
    def plan_grasp_strategy_with_compensation(self, object_height_mm: float, background_z_compensated: float, yaw: float) -> Dict:
        """基于补偿后背景高度的抓取策略规划"""
        try:
            # 基础策略（基于物体高度）
            if object_height_mm < 20:
                base_pitch = 0.0
                height_offset = object_height_mm * 0.6
            elif object_height_mm < 40:
                base_pitch = -5.0
                height_offset = object_height_mm * 0.8
            elif object_height_mm < 80:
                base_pitch = -15.0
                height_offset = object_height_mm * 1
            elif object_height_mm < 120:
                base_pitch = -25.0
                height_offset = object_height_mm * 1
            else:
                base_pitch = -35.0
                height_offset = min(object_height_mm * 0.8, 100)
            
            # 根据补偿后的背景高度调整策略
            if background_z_compensated < 0:
                pitch_adjustment = 5.0
                height_adjustment = 10.0
                self.get_logger().info('⚠️ 背景低于基准面，使用保守策略')
            elif background_z_compensated > 200:
                pitch_adjustment = -5.0
                height_adjustment = -5.0
                self.get_logger().info('📈 背景较高，使用积极策略')
            else:
                pitch_adjustment = 0.0
                height_adjustment = 20.0
            
            final_pitch = base_pitch + pitch_adjustment
            
            # 添加安全余量
            if object_height_mm < 30:
                safety_margin = 110.0
            else:
                safety_margin = 120.0
            
            z_offset_above_background = height_offset + height_adjustment + safety_margin
            
            return {
                'pitch': final_pitch,
                'yaw': yaw,
                'z_offset_above_background': z_offset_above_background,
                'object_height': object_height_mm,
                'background_z_compensated': background_z_compensated,
                'safety_margin': safety_margin,
                'height_offset': height_offset
            }
            
        except Exception as e:
            self.get_logger().error(f'抓取策略规划失败: {e}')
            return {
                'pitch': 0.0,
                'yaw': yaw,
                'z_offset_above_background': 50.0,
                'object_height': 30.0,
                'background_z_compensated': background_z_compensated,
                'safety_margin': 15.0,
                'height_offset': 20.0
            }
    
    def calculate_pitch_compensation(self, target_x, target_y, target_z, pitch_deg, yaw_deg, gripper_length=120.0):
        """计算pitch角度补偿"""
        pitch_rad = math.radians(pitch_deg)
        yaw_rad = math.radians(yaw_deg)
        
        offset_forward = gripper_length * math.sin(-pitch_rad)
        offset_down = gripper_length * (1 - math.cos(-pitch_rad))
        
        offset_x = offset_forward * math.cos(yaw_rad)
        offset_y = offset_forward * math.sin(yaw_rad) * 0.8
        offset_z = offset_down
        
        compensated_x = target_x - offset_x
        compensated_y = target_y - offset_y
        compensated_z = target_z + offset_z
        
        return compensated_x, compensated_y, compensated_z
    
    def _execute_enhanced_grasp_sequence(self, x, y, z, strategy, yaw, pre_width, final_width, background_z_compensated, target):
        """执行增强的抓取序列"""
        try:
            target_desc = target.get("description", "unknown object")
            
            # 1. 设置预抓取宽度
            self.get_logger().info(f'📐 步骤1: 设置预抓取宽度: {pre_width}')
            if not self.call_gripper_close(pre_width):
                self.get_logger().warn('预抓取宽度设置可能失败，继续执行')
            time.sleep(2)
            
            # 2. 高安全位置
            safe_z_high = max(background_z_compensated + strategy['object_height'] + 200, 350)
            safe_comp_x, safe_comp_y, safe_comp_z = self.calculate_pitch_compensation(x, y, safe_z_high, 0, yaw)
            
            if yaw < -135 or -45 < y < 0:
                y = y - 20
                safe_comp_y = safe_comp_y - 20
                
            self.get_logger().info(f'🔝 步骤2: 移动到高安全位置: [{safe_comp_x:.1f}, {safe_comp_y:.1f}, {safe_comp_z:.1f}]')
            if not self.move_to_pose(safe_comp_x, safe_comp_y, safe_comp_z, 180, 0, yaw):
                return False
            time.sleep(1)
            
            # 3. 中等安全高度
            mid_safe_z = max(background_z_compensated + strategy['object_height'] + 100, z + 50)
            mid_pitch = strategy['pitch'] * 0.5
            mid_comp_x, mid_comp_y, mid_comp_z = self.calculate_pitch_compensation(x, y, mid_safe_z, mid_pitch, yaw)
            
            if yaw < -135 or -45 < y < 0:
                mid_comp_y = mid_comp_y - 20
                
            self.get_logger().info(f'🔄 步骤3: 移动到中等安全位置: [{mid_comp_x:.1f}, {mid_comp_y:.1f}, {mid_comp_z:.1f}]')
            if not self.move_to_pose(mid_comp_x, mid_comp_y, mid_comp_z, 180, mid_pitch, yaw):
                return False
            time.sleep(1)
            
            # 4. 最终抓取位置
            self.get_logger().info(f'⬇️ 步骤4: 下降到最终抓取位置: [{x:.1f}, {y:.1f}, {z:.1f}]')
            if not self.move_to_pose(x, y, z, 180, strategy['pitch'], yaw):
                return False
            time.sleep(10)
            
            # 5. 执行抓取
            self.get_logger().info(f'🤏 步骤5: 执行最终抓取，宽度: {final_width}')
            if not self.call_gripper_close(final_width):
                # 备用尝试
                backup_width = final_width + 30
                self.get_logger().warn(f'尝试备用宽度: {backup_width}')
                if not self.call_gripper_close(backup_width):
                    return False
            time.sleep(3)
            
            # 6-10. 提升和放置序列
            check_z = z + 100
            check_comp_x, check_comp_y, check_comp_z = self.calculate_pitch_compensation(x, y, check_z, strategy['pitch'], yaw)
            
            self.get_logger().info('🔍 步骤6: 轻微提升检查')
            if not self.move_to_pose(check_comp_x, check_comp_y, check_comp_z, 180, strategy['pitch'], yaw):
                return False
            time.sleep(2)
            
            # 提升到安全高度
            safe_com_z = max(check_comp_z + 50, 300)
            self.get_logger().info('⬆️ 继续提升到安全高度')
            if not self.move_to_pose(safe_comp_x, safe_comp_y, safe_com_z, 180, 0, yaw):
                return False
            time.sleep(1)
            
            # 放置序列
            placement_x, placement_y, placement_z_safe = 300, 0, 350
            
            self.get_logger().info(f'📦 步骤7: 移动到放置区域: [{placement_x}, {placement_y}, {placement_z_safe}]')
            if not self.move_to_pose(placement_x, placement_y, placement_z_safe, 180, 0, 0):
                return False
            time.sleep(1)
            
            placement_z_release = max(background_z_compensated + 40, 200)
            self.get_logger().info(f'⬇️ 步骤8: 下降到放置高度: {placement_z_release:.1f}mm')
            if not self.move_to_pose(placement_x, placement_y, placement_z_release, 180, 0, 0):
                return False
            time.sleep(1)
            
            # 9. 释放物体
            self.get_logger().info('🔓 步骤9: 释放物体')
            if not self.call_gripper_close(800):
                return False
            time.sleep(3)
            
            # 10. 回到初始位置
            self.get_logger().info('🏠 步骤10: 返回初始位置')
            if not self.call_service(self.go_home_client):
                return False
            
            self.get_logger().info(f'🎉 追踪抓取序列成功完成: {target_desc}')
            return True
            
        except Exception as e:
            self.get_logger().error(f'增强抓取序列执行失败: {e}')
            import traceback
            traceback.print_exc()
            return False
    
    # 辅助方法
    def call_gripper_close(self, position: int, timeout=5.0) -> bool:
        """设置夹爪位置"""
        try:
            self.get_logger().info(f'🤏 设置夹爪位置: {position}')
            position_msg = Int32()
            position_msg.data = max(0, min(850, position))
            self.gripper_control_pub.publish(position_msg)
            time.sleep(2)
            return True
        except Exception as e:
            self.get_logger().error(f'夹爪控制失败: {e}')
            return False
    
    def call_service(self, client, timeout=5.0) -> bool:
        """调用服务"""
        try:
            if not client.wait_for_service(timeout_sec=timeout):
                self.get_logger().error(f'服务不可用: {client.srv_name}')
                return False
            
            request = Trigger.Request()
            future = client.call_async(request)
            time.sleep(1)
            return True
            
        except Exception as e:
            self.get_logger().error(f'服务调用失败: {e}')
            return False
    
    def move_to_pose(self, x, y, z, roll, pitch, yaw, timeout=10.0) -> bool:
        """移动到指定位姿"""
        try:
            pose_msg = PoseStamped()
            pose_msg.header.stamp = self.get_clock().now().to_msg()
            pose_msg.header.frame_id = "link_base"
            
            pose_msg.pose.position.x = float(x)
            pose_msg.pose.position.y = float(y)
            pose_msg.pose.position.z = float(z)
            
            # 欧拉角转四元数
            roll_rad = np.deg2rad(roll)
            pitch_rad = np.deg2rad(pitch)
            yaw_rad = np.deg2rad(yaw)
            
            cy = np.cos(yaw_rad * 0.5)
            sy = np.sin(yaw_rad * 0.5)
            cp = np.cos(pitch_rad * 0.5)
            sp = np.sin(pitch_rad * 0.5)
            cr = np.cos(roll_rad * 0.5)
            sr = np.sin(roll_rad * 0.5)

            pose_msg.pose.orientation.w = cr * cp * cy + sr * sp * sy
            pose_msg.pose.orientation.x = sr * cp * cy - cr * sp * sy
            pose_msg.pose.orientation.y = cr * sp * cy + sr * cp * sy
            pose_msg.pose.orientation.z = cr * cp * sy - sr * sp * cy
            
            self.target_pose_pub.publish(pose_msg)
            time.sleep(3)
            return True
            
        except Exception as e:
            self.get_logger().error(f'移动到位姿失败: {e}')
            return False
    
    def image_callback(self, msg):
        """相机图像回调"""
        try:
            self.current_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f'图像转换失败: {e}')
    
    def update_display(self):
        """更新实时相机显示和系统状态"""
        if self.current_image is None:
            return
        
        display = self.current_image.copy()
        h, w = display.shape[:2]
        
        # 绘制系统状态
        status_color = (0, 255, 0) if self.arm_connected else (0, 0, 255)
        arm_status = "已连接" if self.arm_connected else "未连接"
        
        cv2.putText(display, f"追踪抓取系统", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(display, f"系统状态: {self.system_state}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(display, f"机械臂: {arm_status}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        
        # 根据不同状态显示不同信息
        if self.system_state == "WAITING_FOR_TRACKING":
            cv2.putText(display, "等待追踪触发信号...", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        elif self.system_state == "LOADING_TARGET":
            cv2.putText(display, "正在加载目标信息...", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        elif self.system_state == "PREPARING_GRASP":
            cv2.putText(display, "准备抓取序列...", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        elif self.system_state == "EXECUTING_GRASP":
            cv2.putText(display, "正在执行抓取...", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            if self.current_target:
                target_text = f"目标: {self.current_target['object_id']}"
                cv2.putText(display, target_text, (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                coords = self.current_target['world_coordinates']
                coord_text = f"坐标: [{coords[0]:.1f}, {coords[1]:.1f}, {coords[2]:.1f}]mm"
                cv2.putText(display, coord_text, (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        elif self.system_state == "COMPLETED":
            cv2.putText(display, "抓取完成!", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        elif self.system_state == "FAILED":
            cv2.putText(display, "抓取失败", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        elif self.system_state == "ERROR":
            cv2.putText(display, "系统错误", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # 显示扫描目录
        if self.current_scan_dir:
            scan_dir_name = os.path.basename(self.current_scan_dir)
            cv2.putText(display, f"扫描目录: {scan_dir_name}", (10, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        cv2.imshow('追踪抓取系统', display)
        cv2.waitKey(1)
    
    def destroy_node(self):
        """节点销毁时的清理"""
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    """主函数"""
    rclpy.init(args=args)
    
    try:
        node = TrackingGraspSystem()
        cv2.namedWindow('追踪抓取系统', cv2.WINDOW_AUTOSIZE)
        node.get_logger().info('🎯 追踪抓取系统启动完成')
        rclpy.spin(node)
    except KeyboardInterrupt:
        print('🛑 系统被用户中断')
    except Exception as e:
        print(f'❌ 系统错误: {e}')
    finally:
        if 'node' in locals():
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()