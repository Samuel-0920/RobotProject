#!/usr/bin/env python3
import json
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, Point, Quaternion, PoseStamped
from std_msgs.msg import String, Bool, Float64MultiArray, Int32, Float64
from sensor_msgs.msg import JointState
from xarm.wrapper import XArmAPI
import math
import time
import threading
import numpy as np
from rclpy.service import Service
from std_srvs.srv import Trigger, SetBool

# 🆕 导入自定义服务类型
try:
    from vision_ai_interfaces.srv import SetGripperPosition, SetGripperClose
except ImportError:
    class SetGripperPosition:
        class Request:
            def __init__(self):
                self.position = 0
        class Response:
            def __init__(self):
                self.success = False
                self.message = ""
    
    class SetGripperClose:
        class Request:
            def __init__(self):
                self.position = -1  
        class Response:
            def __init__(self):
                self.success = False
                self.message = ""
                self.actual_position = 0

class XArmControllerNode(Node):
    def __init__(self):
        super().__init__('xarm_controller')
        
        # 机械臂连接参数
        self.arm_ip = '192.168.1.117'
        self.safe_position = [0, 0, -0.15, 0, -0.1, 0]
        self.safe_pose = [200, 0, 250, 179, 0, 0]
        
        # 夹爪参数
        self.gripper_positions = {
            'fully_open': 850,
            'closed': 150,
            'partial_open': 70
        }
        
        # 避障参数
        self.workspace_limits = {
            'x': (-800, 800),
            'y': (-800, 800), 
            'z': (100, 800)
        }
        self.obstacle_center = (0, 0)
        self.obstacle_radius = 180
        self.safety_margin = 20  # mm
        
        # 连接机械臂
        try:
            self.arm = XArmAPI(self.arm_ip)
            self.get_logger().info(f'Connected to xArm at {self.arm_ip}')
            
            # 初始化机械臂
            self.initialize_arm()
            
            # 初始化夹爪
            self.initialize_gripper()
            
            # 移动到初始位置
            self.move_to_initial_position()
            
        except Exception as e:
            self.get_logger().error(f'Failed to connect to xArm: {e}')
            return
        
        # 创建发布者
        self.joint_state_pub = self.create_publisher(JointState, '/xarm/joint_states', 10)
        self.current_pose_pub = self.create_publisher(PoseStamped, '/xarm/current_pose', 10)
        self.arm_state_pub = self.create_publisher(String, '/xarm/state', 10)
        self.gripper_state_pub = self.create_publisher(Float64, '/xarm/gripper_position', 10)
        self.gripper_status_pub = self.create_publisher(String, '/xarm/gripper_status', 10)
        self.movement_complete_pub = self.create_publisher(
        String, '/xarm/movement_complete', 10
        )
        # 创建订阅者
        self.target_pose_sub = self.create_subscription(
            PoseStamped, '/xarm/target_pose', self.target_pose_callback, 10)
        self.joint_target_sub = self.create_subscription(
            Float64MultiArray, '/xarm/joint_target', self.joint_target_callback, 10)
        self.gripper_position_sub = self.create_subscription(
            Int32, '/xarm/gripper_target', self.gripper_position_callback, 10)
        
        # 创建服务
        self.enable_srv = self.create_service(SetBool, '/xarm/enable', self.enable_callback)
        self.go_home_srv = self.create_service(Trigger, '/xarm/go_home', self.go_home_callback)
        self.emergency_stop_srv = self.create_service(Trigger, '/xarm/emergency_stop', self.emergency_stop_callback)
        
        # 🆕 原有夹爪服务（保持兼容性）
        self.gripper_open_srv = self.create_service(Trigger, '/xarm/gripper_open', self.gripper_open_callback)
        self.gripper_close_srv = self.create_service(Trigger, '/xarm/gripper_close', self.gripper_close_callback)
        self.gripper_partial_srv = self.create_service(Trigger, '/xarm/gripper_partial_open', self.gripper_partial_callback)
        
        # 🆕 新增：带参数的夹爪服务
        self.gripper_set_position_srv = self.create_service(
            SetGripperPosition, '/xarm/set_gripper_position', self.set_gripper_position_callback)
        self.gripper_close_with_pos_srv = self.create_service(
            SetGripperClose, '/xarm/gripper_close_with_position', self.gripper_close_with_position_callback)
        
        # 状态监控定时器
        self.status_timer = self.create_timer(0.1, self.publish_status)
        
        # 运动控制锁
        self.motion_lock = threading.Lock()
        
        self.get_logger().info('xArm Controller Node initialized with enhanced gripper services')

    # 🆕 新的夹爪服务回调方法
    def set_gripper_position_callback(self, request, response):
        """设置夹爪位置服务回调"""
        try:
            position = request.position
            
            # 验证位置范围
            if not (0 <= position <= 850):
                response.success = False
                response.message = f"Position {position} out of range [0, 850]"
                self.get_logger().error(response.message)
                return response
            
            # 执行夹爪移动
            code = self.arm.set_gripper_position(position, wait=True)
            
            if code == 0:
                response.success = True
                response.message = f"Gripper moved to position {position}"
                self.get_logger().info(f" Gripper moved to position: {position}")
            else:
                response.success = False
                response.message = f"Failed to move gripper, error code: {code}"
                self.get_logger().error(response.message)
                
        except Exception as e:
            response.success = False
            response.message = f"Error: {str(e)}"
            self.get_logger().error(f'Error in set_gripper_position service: {e}')
            
        return response

    def gripper_close_with_position_callback(self, request, response):
        """带位置参数的夹爪关闭服务回调"""
        try:
            # 如果position为-1，使用默认关闭位置
            if request.position == -1:
                position = self.gripper_positions['closed']
            else:
                position = request.position
            
            # 验证位置范围
            if not (0 <= position <= 850):
                response.success = False
                response.message = f"Position {position} out of range [0, 850]"
                response.actual_position = -1
                self.get_logger().error(response.message)
                return response
            
            # 执行夹爪移动
            code = self.arm.set_gripper_position(position, wait=True)
            
            if code == 0:
                response.success = True
                response.message = f"Gripper closed to position {position}"
                response.actual_position = position
                self.get_logger().info(f"🔴 Gripper closed to position: {position}")
            else:
                response.success = False
                response.message = f"Failed to close gripper, error code: {code}"
                response.actual_position = -1
                self.get_logger().error(response.message)
                
        except Exception as e:
            response.success = False
            response.message = f"Error: {str(e)}"
            response.actual_position = -1
            self.get_logger().error(f'Error in gripper_close_with_position service: {e}')
            
        return response

    # ========== 保持所有原有方法不变 ==========
    def initialize_arm(self):
        """初始化机械臂"""
        try:
            code, state = self.arm.get_state()
            self.get_logger().info(f'Initial arm state: {state}')
            
            self.arm.set_mode(0)
            self.arm.set_state(0)
            self.arm.set_tcp_maxacc(500)
            self.arm.set_tcp_jerk(2000)
            
            self.get_logger().info(' xArm initialized')
            
            code, pos = self.arm.get_position()
            if code == 0:
                self.get_logger().info(f'📍 Current position: {pos}')
            
        except Exception as e:
            self.get_logger().error(f'Failed to initialize xArm: {e}')

    def initialize_gripper(self):
        """初始化夹爪"""
        try:
            # 设置初始TCP负载为0
            self.arm.set_tcp_load(0, [0, 0, 0])
            
            # 打开夹爪到初始位置
            code = self.arm.set_gripper_position(self.gripper_positions['fully_open'], wait=True)
            
            if code == 0:
                self.get_logger().info(' Gripper initialized and opened')
            else:
                self.get_logger().error(f' Gripper initialization failed, error code: {code}')
                
        except Exception as e:
            self.get_logger().error(f'Failed to initialize gripper: {e}')

    def move_to_initial_position(self):
        """移动到初始安全位置"""
        try:
            self.get_logger().info('移动到初始位置...')
            initial_pose = [380, 0, 380, 180, 0, 0]
            code = self.arm.set_position(*initial_pose, speed=80, wait=True)
            
            if code == 0:
                self.get_logger().info(' 已到达初始位置')
            else:
                self.get_logger().error(f' 初始化移动失败，错误码: {code}')
                
        except Exception as e:
            self.get_logger().error(f'初始化移动异常: {e}')

    def publish_status(self):
        """发布机械臂状态信息 - 修复角度转换"""
        try:
            current_time = self.get_clock().now().to_msg()
            
            # 获取关节角度
            code, angles = self.arm.get_servo_angle()
            if code == 0:
                joint_msg = JointState()
                joint_msg.header.stamp = current_time
                joint_msg.header.frame_id = "base_link"
                joint_msg.name = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
                joint_msg.position = [math.radians(angle) for angle in angles]
                self.joint_state_pub.publish(joint_msg)
            
            # 获取末端位置
            code, position = self.arm.get_position()
            if code == 0:
                pose_msg = PoseStamped()
                pose_msg.header.stamp = current_time
                pose_msg.header.frame_id = "base_link"
                
                pose_msg.pose.position.x = position[0] 
                pose_msg.pose.position.y = position[1] 
                pose_msg.pose.position.z = position[2]
                
                # 🔧 关键修复：角度转换
                roll_deg, pitch_deg, yaw_deg = position[3], position[4], position[5]
                
                # 🆕 添加调试信息
                self.get_logger().debug(f'xArm原始角度(度): roll={roll_deg:.2f}, pitch={pitch_deg:.2f}, yaw={yaw_deg:.2f}')
                
                # 🔧 修复：转换为弧度后再计算四元数
                roll = math.radians(roll_deg)
                pitch = math.radians(pitch_deg) 
                yaw = math.radians(yaw_deg)
                
                # 四元数计算（使用弧度）
                qx = math.sin(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) - math.cos(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
                qy = math.cos(roll/2) * math.sin(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.cos(pitch/2) * math.sin(yaw/2)
                qz = math.cos(roll/2) * math.cos(pitch/2) * math.sin(yaw/2) - math.sin(roll/2) * math.sin(pitch/2) * math.cos(yaw/2)
                qw = math.cos(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
                
                # 🆕 验证转换：将四元数转回欧拉角进行验证
                verify_roll, verify_pitch, verify_yaw = self.quat_to_euler(qx, qy, qz, qw)
                self.get_logger().debug(f'四元数验证角度(度): roll={verify_roll:.2f}, pitch={verify_pitch:.2f}, yaw={verify_yaw:.2f}')
                
                pose_msg.pose.orientation.x = qx
                pose_msg.pose.orientation.y = qy
                pose_msg.pose.orientation.z = qz
                pose_msg.pose.orientation.w = qw
                
                self.current_pose_pub.publish(pose_msg)
            
            # 机械臂状态
            code, state = self.arm.get_state()
            if code == 0:
                state_msg = String()
                state_descriptions = {
                    1: "in_motion",
                    2: "sleeping", 
                    3: "suspended",
                    4: "stopping"
                }
                state_msg.data = state_descriptions.get(state, f"unknown_state_{state}")
                self.arm_state_pub.publish(state_msg)
            
            # 发布夹爪状态
            self.publish_gripper_status()
                
        except Exception as e:
            self.get_logger().error(f'Error publishing status: {e}')

    def publish_gripper_status(self):
        """发布夹爪状态"""
        try:
            code, position = self.arm.get_gripper_position()
            
            if code == 0:
                # 发布夹爪位置
                position_msg = Float64()
                position_msg.data = float(position)
                self.gripper_state_pub.publish(position_msg)
                
                # 发布夹爪状态描述
                status_msg = String()
                if position >= 800:
                    status_msg.data = "fully_open"
                elif position <= 50:
                    status_msg.data = "closed"
                elif 50 < position <= 150:
                    status_msg.data = "partial_open"
                else:
                    status_msg.data = "intermediate"
                    
                self.gripper_status_pub.publish(status_msg)
                
        except Exception as e:
            self.get_logger().error(f'Error publishing gripper status: {e}')

    def gripper_position_callback(self, msg):
        """夹爪位置控制回调"""
        try:
            target_position = msg.data
            
            # 验证位置范围
            if not (0 <= target_position <= 850):
                self.get_logger().error(f'Gripper position {target_position} out of range [0, 850]')
                return
            
            code = self.arm.set_gripper_position(target_position, wait=True)
            
            if code == 0:
                self.get_logger().info(f' Gripper moved to position: {target_position}')
            else:
                self.get_logger().error(f' Gripper movement failed, error code: {code}')
                
        except Exception as e:
            self.get_logger().error(f'Error in gripper position callback: {e}')

    # 原有的夹爪服务回调方法（保持兼容性）
    def gripper_open_callback(self, request, response):
        """打开夹爪服务回调"""
        try:
            code = self.arm.set_gripper_position(self.gripper_positions['fully_open'], wait=True)
            
            if code == 0:
                response.success = True
                response.message = "Gripper opened successfully"
                self.get_logger().info("🟢 Gripper opened")
            else:
                response.success = False
                response.message = f"Failed to open gripper, error code: {code}"
                
        except Exception as e:
            response.success = False
            response.message = f"Error: {e}"
            self.get_logger().error(f'Error in gripper open service: {e}')
            
        return response

    def gripper_close_callback(self, request, response):
        """关闭夹爪服务回调"""
        try:
            code = self.arm.set_gripper_position(self.gripper_positions['closed'], wait=True)
            
            if code == 0:
                response.success = True
                response.message = "Gripper closed successfully"
                self.get_logger().info("🔴 Gripper closed")
            else:
                response.success = False
                response.message = f"Failed to close gripper, error code: {code}"
                
        except Exception as e:
            response.success = False
            response.message = f"Error: {e}"
            self.get_logger().error(f'Error in gripper close service: {e}')
            
        return response

    def gripper_partial_callback(self, request, response):
        """部分打开夹爪服务回调"""
        try:
            code = self.arm.set_gripper_position(self.gripper_positions['partial_open'], wait=True)
            
            if code == 0:
                response.success = True
                response.message = "Gripper partially opened successfully"
                self.get_logger().info("🟡 Gripper partially opened")
            else:
                response.success = False
                response.message = f"Failed to partially open gripper, error code: {code}"
                
        except Exception as e:
            response.success = False
            response.message = f"Error: {e}"
            self.get_logger().error(f'Error in gripper partial open service: {e}')
            
        return response

    # ========== 新增：夹爪服务回调方法 ==========
    def gripper_open_callback(self, request, response):
        """打开夹爪服务回调"""
        try:
            code = self.arm.set_gripper_position(self.gripper_positions['fully_open'], wait=True)
            
            if code == 0:
                response.success = True
                response.message = "Gripper opened successfully"
                self.get_logger().info("🟢 Gripper opened")
            else:
                response.success = False
                response.message = f"Failed to open gripper, error code: {code}"
                
        except Exception as e:
            response.success = False
            response.message = f"Error: {e}"
            self.get_logger().error(f'Error in gripper open service: {e}')
            
        return response

    def gripper_close_callback(self, request, response):
        """关闭夹爪服务回调"""
        try:
            code = self.arm.set_gripper_position(self.gripper_positions['closed'], wait=True)
            
            if code == 0:
                response.success = True
                response.message = "Gripper closed successfully"
                self.get_logger().info("🔴 Gripper closed")
            else:
                response.success = False
                response.message = f"Failed to close gripper, error code: {code}"
                
        except Exception as e:
            response.success = False
            response.message = f"Error: {e}"
            self.get_logger().error(f'Error in gripper close service: {e}')
            
        return response

    def gripper_partial_callback(self, request, response):
        """部分打开夹爪服务回调"""
        try:
            code = self.arm.set_gripper_position(self.gripper_positions['partial_open'], wait=True)
            
            if code == 0:
                response.success = True
                response.message = "Gripper partially opened successfully"
                self.get_logger().info("🟡 Gripper partially opened")
            else:
                response.success = False
                response.message = f"Failed to partially open gripper, error code: {code}"
                
        except Exception as e:
            response.success = False
            response.message = f"Error: {e}"
            self.get_logger().error(f'Error in gripper partial open service: {e}')
            
        return response

    # ========== 新增：TCP负载控制方法 ==========
    def set_tcp_load(self, weight, center_of_gravity):
        """设置TCP负载"""
        try:
            code = self.arm.set_tcp_load(weight, center_of_gravity)
            
            if code == 0:
                self.get_logger().info(f' TCP load set: weight={weight}kg, cog={center_of_gravity}')
                return True
            else:
                self.get_logger().error(f' Failed to set TCP load, error code: {code}')
                return False
                
        except Exception as e:
            self.get_logger().error(f'Error setting TCP load: {e}')
            return False

    # ========== 以下是原有方法，保持不变 ==========
    def target_pose_callback(self, msg):
        """目标位置回调 - 使用避障路径规划"""
        with self.motion_lock:
            try:
                target_x = msg.pose.position.x 
                target_y = msg.pose.position.y 
                target_z = msg.pose.position.z
                
                qx, qy, qz, qw = msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w
                roll, pitch, yaw = self.quat_to_euler(qx, qy, qz, qw)
                
                if not self.is_in_workspace(target_x, target_y, target_z):
                    return
                
                code, current_pos = self.arm.get_position()
                if code != 0:
                    self.get_logger().error('Cannot get current position')
                    return
                
                current_x, current_y, current_z = current_pos[:3]
                
                self.get_logger().info(f'🚀 规划路径: ({current_x:.1f},{current_y:.1f},{current_z:.1f}) → ({target_x:.1f},{target_y:.1f},{target_z:.1f})')
                
                # 使用避障路径规划
                safe_path = self.generate_safe_path(
                    [current_x, current_y, current_z],
                    [target_x, target_y, target_z]
                )
                roll = 179
                success = self.execute_path(safe_path, roll, pitch, yaw)
                
                if success:
                    self.get_logger().info(f' 安全到达目标位置')
                else:
                    self.get_logger().error(f' 路径执行失败')
                
            except Exception as e:
                self.get_logger().error(f'Error in pose callback: {e}')
                import traceback
                traceback.print_exc()

    def is_in_workspace(self, x, y, z):
        """检查位置是否在工作空间内"""
        if not (self.workspace_limits['x'][0] <= x <= self.workspace_limits['x'][1] and
                self.workspace_limits['y'][0] <= y <= self.workspace_limits['y'][1] and
                self.workspace_limits['z'][0] <= z <= self.workspace_limits['z'][1]):
            self.get_logger().error(f'Position ({x:.1f}, {y:.1f}, {z:.1f}) outside workspace')
            return False
        return True

    def generate_safe_path(self, start, end, step_size=20):
        """生成避障路径"""
        start_2d = np.array(start[:2])
        end_2d = np.array(end[:2])
        
        if not self.path_intersects_obstacle(start_2d, end_2d):
            return self.interpolate_path(start, end, step_size)
        
        waypoints = self.find_detour_path(start, end)
        
        full_path = []
        current = start
        
        for waypoint in waypoints + [end]:
            segment = self.interpolate_path(current, waypoint, step_size)
            if full_path:
                segment = segment[1:]
            full_path.extend(segment)
            current = waypoint
            
        return full_path

    def path_intersects_obstacle(self, start_2d, end_2d):
        """检查2D路径是否与圆形障碍物相交"""
        d = end_2d - start_2d
        f = start_2d - np.array(self.obstacle_center)
        
        a = np.dot(d, d)
        b = 2 * np.dot(f, d)
        c = np.dot(f, f) - (self.obstacle_radius + self.safety_margin) ** 2
        
        discriminant = b * b - 4 * a * c
        
        if discriminant < 0:
            return False
        
        sqrt_disc = math.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / (2 * a)
        t2 = (-b + sqrt_disc) / (2 * a)
        
        return (0 <= t1 <= 1) or (0 <= t2 <= 1) or (t1 < 0 and t2 > 1)

    def find_detour_path(self, start, end):
        """找到绕行路径"""
        start_2d = np.array(start[:2])
        end_2d = np.array(end[:2])
        center = np.array(self.obstacle_center)
        
        start_tangents = self.get_tangent_points(start_2d, center, self.obstacle_radius + self.safety_margin)
        end_tangents = self.get_tangent_points(end_2d, center, self.obstacle_radius + self.safety_margin)
        
        min_distance = float('inf')
        best_waypoint = None
        
        for start_tangent in start_tangents:
            for end_tangent in end_tangents:
                if not self.path_intersects_obstacle(start_tangent, end_tangent):
                    distance = (np.linalg.norm(start_tangent - start_2d) + 
                              np.linalg.norm(end_tangent - end_2d) +
                              np.linalg.norm(end_tangent - start_tangent))
                    
                    if distance < min_distance:
                        min_distance = distance
                        best_waypoint = start_tangent
        
        if best_waypoint is not None:
            waypoint_3d = [best_waypoint[0], best_waypoint[1], start[2]]
            return [waypoint_3d]
        else:
            safe_height = max(start[2], end[2]) + 155
            return [[start[0], start[1], safe_height], [end[0], end[1], safe_height]]

    def get_tangent_points(self, point, circle_center, circle_radius):
        """计算外部点到圆的切点"""
        d = point - circle_center
        distance = np.linalg.norm(d)
        
        if distance <= circle_radius:
            perp = np.array([-d[1], d[0]])
            if np.linalg.norm(perp) > 0:
                perp = perp / np.linalg.norm(perp) * circle_radius
            return [circle_center + perp, circle_center - perp]
        
        l = math.sqrt(distance**2 - circle_radius**2)
        angle = math.asin(circle_radius / distance)
        base_angle = math.atan2(d[1], d[0])
        
        angle1 = base_angle + angle
        angle2 = base_angle - angle
        
        t1 = circle_center + circle_radius * np.array([math.cos(angle1 + math.pi/2), 
                                                       math.sin(angle1 + math.pi/2)])
        t2 = circle_center + circle_radius * np.array([math.cos(angle2 - math.pi/2), 
                                                       math.sin(angle2 - math.pi/2)])
        
        return [t1, t2]

    def interpolate_path(self, start, end, step_size):
        """线性插值生成路径"""
        diff = np.array(end) - np.array(start)
        distance = np.linalg.norm(diff[:2])
        steps = max(int(distance / step_size), 1)
        
        path = []
        for i in range(steps + 1):
            t = i / steps
            interpolated = np.array(start) + diff * t
            path.append(interpolated.tolist())
        
        return path

    def execute_path(self, path, roll_deg, pitch_deg, yaw_deg):
        """执行路径点序列"""
        self.get_logger().info(f'Executing path with {len(path)} points')
        
        for i, point in enumerate(path):
            px, py, pz = point
            
            code = self.arm.set_position(px, py, pz, roll_deg, pitch_deg, yaw_deg, wait=True)
            
            if code == 0:
                self.get_logger().info(f' Waypoint {i+1}/{len(path)}: ({px:.1f}, {py:.1f}, {pz:.1f})')
            else:
                self.get_logger().error(f' Failed at waypoint {i+1}, error: {code}')
                return False
        
        # 🆕 路径执行完成后发布移动完成信号
        try:
            movement_complete_msg = String()
            movement_complete_msg.data = json.dumps({
                'status': 'movement_completed',
                'final_position': path[-1] if path else [0, 0, 0],
                'timestamp': time.time()
            })
            self.movement_complete_pub.publish(movement_complete_msg)
            self.get_logger().info('📡 移动完成信号已发布')
        except Exception as e:
            self.get_logger().error(f'发布移动完成信号失败: {e}')
        
        return True

    def quat_to_euler(self, qx, qy, qz, qw):
        """四元数转欧拉角"""
        # Roll
        sinr_cosp = 2 * (qw * qx + qy * qz)
        cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        
        # Pitch
        sinp = 2 * (qw * qy - qz * qx)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(sinp)
        
        # Yaw
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)

    def joint_target_callback(self, msg):
        """关节目标回调函数"""
        with self.motion_lock:
            try:
                if len(msg.data) != 6:
                    self.get_logger().error('Joint target must have 6 values')
                    return
                
                joint_angles = [math.degrees(angle) for angle in msg.data]
                code = self.arm.set_servo_angle(angle=joint_angles, wait=False)
                
                if code == 0:
                    self.get_logger().info(f'Moving to joint angles: {joint_angles}')
                else:
                    self.get_logger().error(f'Failed to move joints, error code: {code}')
                    
            except Exception as e:
                self.get_logger().error(f'Error in joint callback: {e}')

    def enable_callback(self, request, response):
        """使能/失能服务回调"""
        try:
            if request.data:
                self.arm.set_mode(0)
                self.arm.set_state(0)
                response.success = True
                response.message = "xArm enabled"
                self.get_logger().info("xArm enabled")
            else:
                self.arm.set_state(3)
                response.success = True
                response.message = "xArm disabled"
                self.get_logger().info("xArm disabled")
                
        except Exception as e:
            response.success = False
            response.message = f"Error: {e}"
            self.get_logger().error(f'Error in enable service: {e}')
            
        return response

    def go_home_callback(self, request, response):
        """回零服务回调"""
        try:
            with self.motion_lock:
                code = self.arm.set_position(*self.safe_pose, wait=True)
                
                if code == 0:
                    response.success = True
                    response.message = "Moved to home position"
                    self.get_logger().info("Moved to home position")
                else:
                    response.success = False
                    response.message = f"Failed to go home, error code: {code}"
                    
        except Exception as e:
            response.success = False
            response.message = f"Error: {e}"
            self.get_logger().error(f'Error in go_home service: {e}')
            
        return response

    def emergency_stop_callback(self, request, response):
        """急停服务回调"""
        try:
            self.arm.emergency_stop()
            response.success = True
            response.message = "Emergency stop activated"
            self.get_logger().warn("Emergency stop activated")
            
        except Exception as e:
            response.success = False
            response.message = f"Error: {e}"
            self.get_logger().error(f'Error in emergency stop: {e}')
            
        return response

    def destroy_node(self):
        """节点销毁时断开机械臂连接"""
        try:
            if hasattr(self, 'arm'):
                self.arm.disconnect()
                self.get_logger().info('Disconnected from xArm')
        except Exception as e:
            self.get_logger().error(f'Error disconnecting from xArm: {e}')
        
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    try:
        xarm_controller = XArmControllerNode()
        rclpy.spin(xarm_controller)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Error: {e}')
    finally:
        if 'xarm_controller' in locals():
            xarm_controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()