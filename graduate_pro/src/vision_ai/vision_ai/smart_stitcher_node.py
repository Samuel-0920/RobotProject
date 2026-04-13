#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from vision_ai_interfaces.srv import ProcessStitching
from vision_ai_interfaces.msg import StitchResult, ImageData
from vision_ai_interfaces.msg import GridPosition, RelativePosition
from sensor_msgs.msg import Image
import cv2
import numpy as np
import math
import os
from datetime import datetime
from cv_bridge import CvBridge
from std_msgs.msg import String
import pickle
import json

class SmartStitcherNode(Node):
    def __init__(self):
        super().__init__('smart_stitcher_node')
        
        # 创建服务
        self.stitch_service = self.create_service(
            ProcessStitching,
            'process_stitching', 
            self.stitch_callback
        )
        
        self.stitching_complete_pub = self.create_publisher(
            String,
            '/stitching_complete',
            10
        )
        
        self.reference_image_pub = self.create_publisher(
            Image,
            '/reference_image',
            10
        )

        self.bridge = CvBridge()
        
        # 相机参数
        self.camera_width = 1280
        self.camera_height = 720
        self.fx = 912.694580078125
        self.fy = 910.309814453125
        self.cx = 640
        self.cy = 360
        
        # FOV计算
        self.fov_h = 2 * math.atan(self.camera_width / (2 * self.fx))
        self.fov_v = 2 * math.atan(self.camera_height / (2 * self.fy))
        
        self.get_logger().info('智能拼接节点已启动 (支持重叠度优化)')
        self.get_logger().info(f'FOV: {math.degrees(self.fov_h):.1f}° × {math.degrees(self.fov_v):.1f}°')

    def stitch_callback(self, request, response):
        """处理拼接请求 - 🆕 支持重叠度信息"""
        try:
            self.get_logger().info(f'收到拼接请求: {len(request.image_data)} 张图像')
            
            # 保存输出目录
            self.current_output_dir = request.output_directory
            
            # 🆕 从 scan_plan 提取重叠度信息
            overlap_ratio = self._extract_overlap_ratio(request.scan_plan)
            self.get_logger().info(f'🎯 使用重叠度: {overlap_ratio:.1%}')
            
            # 转换输入数据 - 🆕 传递重叠度
            image_data = self._convert_input_data(request.image_data, overlap_ratio)
            scan_plan = request.scan_plan
            
            if not image_data:
                response.success = False
                response.message = "图像数据转换失败"
                return response
            
            # 根据策略选择处理模式
            if scan_plan.strategy == "single_point" or len(image_data) == 1:
                result_path, method = self._process_single_point(image_data[0], scan_plan, request.output_directory)
            else:
                # 🆕 传递重叠度到多点处理
                result_path, method = self._process_multi_point_with_overlap(
                    image_data, scan_plan, request.output_directory, overlap_ratio
                )
            
            if result_path:
                response.success = True
                response.message = f"拼接成功: {method}, 重叠度: {overlap_ratio:.1%}"
                response.result = StitchResult()
                response.result.method = method
                response.result.output_path = result_path
                response.result.input_images = len(request.image_data)
            else:
                response.success = False
                response.message = "拼接失败"
                
        except Exception as e:
            response.success = False
            response.message = f"拼接处理失败: {str(e)}"
            self.get_logger().error(f'拼接错误: {e}')
            import traceback
            traceback.print_exc()
        
        return response

    def _extract_overlap_ratio(self, scan_plan):
        """🆕 从 scan_plan 中提取重叠度信息"""
        try:
            # 假设接口已经扩展，包含 overlap_ratio 字段
            if hasattr(scan_plan, 'overlap_ratio'):
                overlap_ratio = float(scan_plan.overlap_ratio)
                self.get_logger().info(f'📐 从 scan_plan 获取重叠度: {overlap_ratio:.1%}')
                
                # 验证重叠度范围
                if 0.0 <= overlap_ratio <= 0.6:
                    return overlap_ratio
                else:
                    self.get_logger().warn(f'重叠度超出合理范围 [0, 0.6]: {overlap_ratio:.1%}，使用默认值')
            
            # 如果没有重叠度字段，尝试从其他信息推断
            return self._estimate_overlap_from_scan_plan(scan_plan)
            
        except Exception as e:
            self.get_logger().warn(f'提取重叠度失败: {e}，使用默认值')
            return 0.25  # 默认 25% 重叠度

    def _estimate_overlap_from_scan_plan(self, scan_plan):
        """🆕 从扫描计划推断合理的重叠度"""
        try:
            # 根据扫描策略和waypoint数量推断重叠度
            num_waypoints = len(scan_plan.waypoints)
            
            if scan_plan.strategy == "single_point":
                return 0.0  # 单点无需重叠
            elif num_waypoints <= 4:
                return 0.15  # 少量waypoint，较低重叠
            elif num_waypoints <= 9:
                return 0.25  # 中等waypoint，标准重叠
            else:
                return 0.35  # 大量waypoint，较高重叠
                
        except Exception as e:
            self.get_logger().warn(f'推断重叠度失败: {e}')
            return 0.25

    def _convert_input_data(self, ros_image_data, overlap_ratio):
        """转换ROS图像数据为内部格式 - 🆕 提取planning信息"""
        converted_data = []
        self.current_output_dir = getattr(self, 'current_output_dir', None)
        
        for img_data in ros_image_data:
            try:
                # 图像转换部分保持不变...
                self.get_logger().info(f'🎨 调试Waypoint {img_data.waypoint.waypoint_index} 图像转换...')
                
                # 安全的图像转换
                try:
                    ros_image = img_data.image
                    self.get_logger().info(f'原始ROS图像: {ros_image.width}x{ros_image.height}, 编码: {ros_image.encoding}')
                    
                    color_cv_raw = self.bridge.imgmsg_to_cv2(img_data.image, desired_encoding="passthrough")
                    self.get_logger().info(f'passthrough转换后: {color_cv_raw.shape}, dtype: {color_cv_raw.dtype}')
                    
                    if len(color_cv_raw.shape) == 3:
                        center_y, center_x = color_cv_raw.shape[0]//2, color_cv_raw.shape[1]//2
                        center_pixel = color_cv_raw[center_y, center_x]
                        self.get_logger().info(f'中心像素值(原始): {center_pixel}')
                    
                    if len(color_cv_raw.shape) == 3 and color_cv_raw.shape[2] == 3:
                        if ros_image.encoding == 'bgr8':
                            color_cv = cv2.cvtColor(color_cv_raw, cv2.COLOR_BGR2RGB)
                            self.get_logger().info('执行了BGR->RGB转换')
                        elif ros_image.encoding == 'rgb8':
                            color_cv = color_cv_raw.copy()
                            self.get_logger().info('图像已是RGB格式，直接使用')
                        else:
                            color_cv = cv2.cvtColor(color_cv_raw, cv2.COLOR_BGR2RGB)
                            self.get_logger().warn(f'未知编码 {ros_image.encoding}，尝试BGR->RGB转换')
                        
                        center_pixel_after = color_cv[center_y, center_x]
                        self.get_logger().info(f'中心像素值(转换后): {center_pixel_after}')
                        
                except Exception as cv_error:
                    self.get_logger().error(f'cv_bridge转换失败: {cv_error}')
                    # 备用方案保持不变...
                    try:
                        self.get_logger().info('尝试备用图像转换方案...')
                        ros_image = img_data.image
                        width = ros_image.width
                        height = ros_image.height
                        encoding = ros_image.encoding
                        step = ros_image.step
                        data = ros_image.data
                        
                        self.get_logger().info(f'ROS图像信息: {width}x{height}, 编码: {encoding}, step: {step}')
                        
                        if encoding == 'rgb8':
                            image_array = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 3))
                            color_cv = image_array.copy()
                        elif encoding == 'bgr8':
                            image_array = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 3))
                            color_cv = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
                        else:
                            self.get_logger().error(f'不支持的编码格式: {encoding}')
                            continue
                        
                        self.get_logger().info(f'备用方案转换成功: {color_cv.shape}')
                        
                    except Exception as backup_error:
                        self.get_logger().error(f'备用转换方案也失败: {backup_error}')
                        continue
                
                # 从waypoint提取位置和角度
                wp = img_data.waypoint
                world_x = wp.pose.position.x
                world_y = wp.pose.position.y
                world_z = wp.pose.position.z
                
                # 提取完整的四元数并转换为欧拉角
                qx = wp.pose.orientation.x
                qy = wp.pose.orientation.y
                qz = wp.pose.orientation.z
                qw = wp.pose.orientation.w
                
                from scipy.spatial.transform import Rotation as R
                rotation = R.from_quat([qx, qy, qz, qw])
                roll_rad, pitch_rad, yaw_rad = rotation.as_euler('xyz')
                
                roll_deg = abs(180-math.degrees(roll_rad))
                pitch_deg = math.degrees(pitch_rad)
                yaw_deg = math.degrees(yaw_rad)
                
                # 深度文件路径处理保持不变...
                color_filename = img_data.filename
                depth_filename = None
                
                if color_filename:
                    if 'color_waypoint_' in color_filename:
                        depth_filename = color_filename.replace('color_waypoint_', 'depth_raw_waypoint_').replace('.jpg', '.npy')
                    else:
                        waypoint_idx = wp.waypoint_index
                        depth_filename = f"depth_raw_waypoint_{waypoint_idx:03d}.npy"
                
                depth_file_exists = False
                if depth_filename and self.current_output_dir:
                    depth_full_path = os.path.join(self.current_output_dir, depth_filename)
                    depth_file_exists = os.path.exists(depth_full_path)
                    
                    self.get_logger().info(f'🔍 深度文件检查:')
                    self.get_logger().info(f'  - 文件名: {depth_filename}')
                    self.get_logger().info(f'  - 完整路径: {depth_full_path}')
                    self.get_logger().info(f'  - 文件存在: {depth_file_exists}')
                    
                    if not depth_file_exists:
                        alternative_patterns = [
                            f"depth_raw_waypoint_{wp.waypoint_index+1:03d}.npy",
                            f"depth_waypoint_{wp.waypoint_index:03d}.npy",
                            f"depth_waypoint_{wp.waypoint_index+1:03d}.npy"
                        ]
                        
                        for alt_name in alternative_patterns:
                            alt_path = os.path.join(self.current_output_dir, alt_name)
                            if os.path.exists(alt_path):
                                depth_filename = alt_name
                                depth_file_exists = True
                                self.get_logger().info(f'✅ 找到替代深度文件: {alt_name}')
                                break
                        
                        if not depth_file_exists:
                            self.get_logger().warn(f'⚠️ 未找到深度文件，尝试列出目录内容...')
                            try:
                                files_in_dir = os.listdir(self.current_output_dir)
                                depth_files = [f for f in files_in_dir if 'depth' in f and f.endswith('.npy')]
                                self.get_logger().info(f'📁 目录中的深度文件: {depth_files}')
                            except Exception as e:
                                self.get_logger().error(f'无法列出目录内容: {e}')

                # 🆕 提取planning信息
                grid_info = None
                relative_info = None
                overlap_info = None
                
                # 提取grid position信息
                if hasattr(wp, 'grid_position'):
                    grid_info = {
                        'grid_x': wp.grid_position.grid_x,
                        'grid_y': wp.grid_position.grid_y,
                        'total_x': wp.grid_position.total_x,
                        'total_y': wp.grid_position.total_y
                    }
                    self.get_logger().info(f'📊 Grid信息: ({grid_info["grid_x"]}, {grid_info["grid_y"]}) of {grid_info["total_x"]}×{grid_info["total_y"]}')
                
                # 提取relative position信息
                if hasattr(wp, 'relative_position'):
                    relative_info = {
                        'camera_x': wp.relative_position.camera_x,
                        'camera_y': wp.relative_position.camera_y,
                        'neighbors': list(wp.relative_position.neighbors)
                    }
                    self.get_logger().info(f'📍 相对位置: Camera({relative_info["camera_x"]:.1f}, {relative_info["camera_y"]:.1f})')
                    self.get_logger().info(f'🔗 邻居关系: {relative_info["neighbors"]}')
                
                # 提取overlap信息
                if hasattr(wp, 'overlap_x_ratio') and hasattr(wp, 'overlap_y_ratio'):
                    overlap_info = {
                        'overlap_x': wp.overlap_x_ratio,
                        'overlap_y': wp.overlap_y_ratio
                    }
                    self.get_logger().info(f'🔄 重叠信息: X={overlap_info["overlap_x"]:.1%}, Y={overlap_info["overlap_y"]:.1%}')

                converted_item = {
                    'color_image': color_cv,
                    'waypoint': wp,
                    'filename': color_filename,
                    'timestamp': img_data.timestamp,
                    'waypoint_index': wp.waypoint_index,
                    
                    # 完整的位姿信息
                    'world_pos': (world_x, world_y, world_z),
                    'roll': roll_deg,
                    'pitch': pitch_deg,  
                    'yaw': yaw_deg,
                    'quaternion': (qx, qy, qz, qw),
                    
                    # 文件路径信息
                    'color_filename': color_filename,
                    'depth_raw_filename': depth_filename,
                    'depth_file_exists': depth_file_exists,
                    
                    # 🆕 Planning信息
                    'grid_info': grid_info,
                    'relative_info': relative_info,
                    'overlap_info': overlap_info
                }
                
                self.get_logger().info(f'📋 Waypoint {wp.waypoint_index} 数据准备完成:')
                self.get_logger().info(f'  - 位置: ({world_x:.1f}, {world_y:.1f}, {world_z:.1f})')
                self.get_logger().info(f'  - 姿态: Roll={roll_deg:.1f}°, Pitch={pitch_deg:.1f}°, Yaw={yaw_deg:.1f}°')
                self.get_logger().info(f'  - 深度文件: {depth_filename} (存在: {depth_file_exists})')
                
                converted_data.append(converted_item)
                
            except Exception as e:
                self.get_logger().error(f'转换图像数据失败: {e}')
                import traceback
                traceback.print_exc()
                continue
        
        return converted_data

    def _rotate_image_for_stitching(self, image, rotate_180=True):
        """
        为拼接旋转图像以匹配世界坐标系
        
        Args:
            image: 输入图像 (RGB格式)
            rotate_180: 是否旋转180度
        
        Returns:
            rotated_image: 旋转后的图像
        """
        if not rotate_180:
            return image
        
        # 旋转180度：相当于同时水平和垂直翻转
        rotated_image = cv2.rotate(image, cv2.ROTATE_180)
        
        self.get_logger().info(f'🔄 图像已旋转180度: {image.shape} -> {rotated_image.shape}')
        
        return rotated_image

    def _rotate_final_result(self, stitched_image, rotate_180=True):
        """
        将拼接完成的图像旋转回正确方向
        
        Args:
            stitched_image: 拼接完成的图像
            rotate_180: 是否旋转180度
        
        Returns:
            final_image: 最终正确朝向的图像
        """
        if not rotate_180:
            return stitched_image
        
        # 将拼接结果旋转回来
        final_image = cv2.rotate(stitched_image, cv2.ROTATE_180)
        
        self.get_logger().info(f'🔄 最终结果已旋转回正确方向: {stitched_image.shape} -> {final_image.shape}')
        
        return final_image


    def _extract_planning_info(self, sorted_data):
        """🆕 直接从planning数据中提取waypoint关系，替代_analyze_waypoint_layout"""
        try:
            relationships = {}
            overlap_map = {}
            grid_layout = {}
            
            self.get_logger().info('🔍 提取planning信息...')
            
            # 提取grid布局信息
            if sorted_data and sorted_data[0]['grid_info']:
                first_grid = sorted_data[0]['grid_info']
                grid_layout = {
                    'total_x': first_grid['total_x'],
                    'total_y': first_grid['total_y'],
                    'is_regular_grid': True,
                    'num_waypoints': len(sorted_data)
                }
                self.get_logger().info(f'📊 Grid布局: {grid_layout["total_x"]}×{grid_layout["total_y"]} = {grid_layout["num_waypoints"]} waypoints')
            
            for img_data in sorted_data:
                wp_idx = img_data['waypoint_index']
                
                # 提取邻居关系
                if img_data['relative_info']:
                    rel_info = img_data['relative_info']
                    relationships[wp_idx] = {
                        'neighbors': rel_info['neighbors'],
                        'camera_pos': (rel_info['camera_x'], rel_info['camera_y'])
                    }
                    self.get_logger().info(f'📋 WP{wp_idx} 邻居关系: {rel_info["neighbors"]}')
                    self.get_logger().info(f'📍 WP{wp_idx} Camera坐标: ({rel_info["camera_x"]:.1f}, {rel_info["camera_y"]:.1f})')
                
                # 提取重叠度信息
                if img_data['overlap_info']:
                    overlap_info = img_data['overlap_info']
                    overlap_map[wp_idx] = {
                        'overlap_x': overlap_info['overlap_x'],
                        'overlap_y': overlap_info['overlap_y']
                    }
                    self.get_logger().info(f'🔄 WP{wp_idx} 重叠度: X={overlap_info["overlap_x"]:.1%}, Y={overlap_info["overlap_y"]:.1%}')
                
                # 提取grid位置
                if img_data['grid_info']:
                    grid_info = img_data['grid_info']
                    relationships[wp_idx]['grid_pos'] = (grid_info['grid_x'], grid_info['grid_y'])
                    self.get_logger().info(f'📊 WP{wp_idx} Grid位置: ({grid_info["grid_x"]}, {grid_info["grid_y"]})')
            
            layout_info = {
                'relationships': relationships,
                'overlap_map': overlap_map,
                'grid_layout': grid_layout,
                'source': 'planning_data',
                'scan_bounds': self._calculate_bounds_from_camera_positions(relationships)
            }
            
            self.get_logger().info(f'✅ Planning信息提取完成: {len(relationships)} waypoints')
            return layout_info
            
        except Exception as e:
            self.get_logger().error(f'从planning数据提取关系失败: {e}')
            import traceback
            traceback.print_exc()
            return None

    def _calculate_bounds_from_camera_positions(self, relationships):
        """🆕 从camera坐标计算边界"""
        try:
            camera_x_coords = []
            camera_y_coords = []
            
            for wp_idx, rel_info in relationships.items():
                camera_x, camera_y = rel_info['camera_pos']
                camera_x_coords.append(camera_x)
                camera_y_coords.append(camera_y)
            
            bounds = {
                'camera_x_min': min(camera_x_coords),
                'camera_x_max': max(camera_x_coords),
                'camera_y_min': min(camera_y_coords),
                'camera_y_max': max(camera_y_coords)
            }
            
            self.get_logger().info(f'📐 Camera坐标边界: X[{bounds["camera_x_min"]:.1f}, {bounds["camera_x_max"]:.1f}], Y[{bounds["camera_y_min"]:.1f}, {bounds["camera_y_max"]:.1f}]')
            
            return bounds
            
        except Exception as e:
            self.get_logger().error(f'计算camera边界失败: {e}')
            return {}
    def _process_single_point(self, image_data, scan_plan, output_dir):
        """单点模式：旋转矫正 + 裁剪 + 映射信息保存"""
        try:
            self.get_logger().info('处理单点扫描')
            
            image = image_data['color_image']
            yaw_deg = image_data['yaw']
            
            # 简化裁剪
            cropped_image = image
            
            # 🆕 为单点扫描创建映射信息
            self._create_single_point_mapping(image_data, cropped_image, output_dir)
            
            # 保存结果
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = os.path.join(output_dir, f"single_point_result_{timestamp}.jpg")
            
            result_bgr = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, result_bgr)
            
            self.get_logger().info(f'单点处理完成: {output_path}')
            
            # 🔧 修复: 添加缺失的参数
            self._publish_reference_data(cropped_image, None, output_dir)
            return output_path, "single_point_corrected"
            
        except Exception as e:
            self.get_logger().error(f'单点处理失败: {e}')
            return None, "failed"
    def _direct_grid_stitching(self, sorted_data, planning_info, overlap_ratio):
        """🆕 直接基于Grid位置拼接，无需Canvas尺寸计算"""
        try:
            relationships = planning_info['relationships']
            overlap_map = planning_info['overlap_map']
            
            self.get_logger().info('🔧 执行直接Grid拼接 (无Canvas)')
            
            # 分析Grid布局
            grid_layout = planning_info['grid_layout']
            total_x = grid_layout['total_x']
            total_y = grid_layout['total_y']
            
            self.get_logger().info(f'📏 Grid布局: {total_x}×{total_y}')
            
            # 获取图像尺寸（假设所有图像相同）
            sample_image = sorted_data[0]['color_image']
            img_h, img_w = sample_image.shape[:2]
            
            # 计算每个方向的有效步长（考虑重叠）
            effective_step_x = int(img_w * (1 - overlap_ratio))  # 90% of width
            effective_step_y = int(img_h * (1 - overlap_ratio))  # 90% of height
            
            # 计算最终拼接图像尺寸
            if total_x == 1:
                # 1×N 水平拼接
                final_width = img_w + (total_y - 1) * effective_step_x
                final_height = img_h
            elif total_y == 1:
                # N×1 垂直拼接
                final_width = img_w
                final_height = img_h + (total_x - 1) * effective_step_y
            else:
                # M×N 二维拼接
                final_width = img_w + (total_y - 1) * effective_step_x
                final_height = img_h + (total_x - 1) * effective_step_y
            
            self.get_logger().info(f'📐 最终尺寸: {final_width}×{final_height}')
            self.get_logger().info(f'📏 有效步长: X={effective_step_x}px, Y={effective_step_y}px')
            
            # 创建最终画布
            final_canvas = np.zeros((final_height, final_width, 3), dtype=np.float32)
            weight_map = np.zeros((final_height, final_width), dtype=np.float32)
            
            # 初始化映射记录
            self.fusion_mapping = {}
            self.waypoint_contributions = {}
            
            # 按Grid位置放置每个图像
            for img_data in sorted_data:
                # 旋转图像
                original_image = img_data['color_image'].astype(np.float32)
                rotated_image = self._rotate_image_for_stitching(original_image, rotate_180=True)
                
                waypoint_idx = img_data['waypoint_index']
                
                if waypoint_idx not in relationships:
                    self.get_logger().warn(f'⚠️ WP{waypoint_idx} 没有planning信息，跳过')
                    continue
                
                # 获取Grid位置
                rel_info = relationships[waypoint_idx]
                grid_x, grid_y = rel_info['grid_pos']
                neighbors = rel_info['neighbors']
                
                # 直接基于Grid位置计算放置坐标
                pos_x = grid_y * effective_step_x  # Grid Y -> Canvas X
                pos_y = grid_x * effective_step_y  # Grid X -> Canvas Y
                
                self.get_logger().info(f'🎯 WP{waypoint_idx} Grid({grid_x},{grid_y}) -> Canvas({pos_x},{pos_y})')
                
                # 获取重叠信息
                overlap_info = overlap_map.get(waypoint_idx, {'overlap_x': overlap_ratio, 'overlap_y': overlap_ratio})
                
                # 创建重叠mask
                mask = self._create_grid_overlap_mask(rotated_image.shape[:2], neighbors, overlap_info)
                
                # 直接放置到指定位置
                self._place_image_at_position(
                    final_canvas, weight_map, rotated_image, mask,
                    pos_x, pos_y, neighbors, overlap_info, waypoint_idx
                )
                
                # 记录映射信息
                self._record_grid_mapping(rotated_image, pos_x, pos_y, waypoint_idx, 
                                        final_width, final_height, img_data)
                
                self.get_logger().info(f'✅ WP{waypoint_idx} Grid拼接完成')
            
            # 归一化画布
            stitched_canvas = self._normalize_canvas(final_canvas, weight_map)
            
            # 旋转回正确方向
            final_result = self._rotate_final_result(stitched_canvas, rotate_180=True)
            
            # 🔧 返回结果和最终尺寸
            return final_result, (final_width, final_height)
            
        except Exception as e:
            self.get_logger().error(f'直接Grid拼接失败: {e}')
            import traceback
            traceback.print_exc()
            return None, (0, 0)
    def _create_grid_overlap_mask(self, image_shape, neighbors, overlap_info):
        """🆕 修复版本：创建更合理的重叠mask"""
        h, w = image_shape
        mask = np.ones((h, w), dtype=np.float32)
        
        overlap_x = overlap_info.get('overlap_x', 0.0)
        overlap_y = overlap_info.get('overlap_y', 0.0)
        
        # 🔧 修复：减小重叠区域的权重范围
        blend_size_x = max(1, int(w * overlap_x * 0.3)) if overlap_x > 0 else 0  # 降低到30%
        blend_size_y = max(1, int(h * overlap_y * 0.3)) if overlap_y > 0 else 0
        
        self.get_logger().info(f'🎭 修复版Grid重叠Mask:')
        self.get_logger().info(f'    重叠区域: X={blend_size_x}px({overlap_x:.1%}), Y={blend_size_y}px({overlap_y:.1%})')
        
        # 🆕 修复：使用更平滑的权重过渡
        if 'left' in neighbors and blend_size_x > 0:
            for i in range(blend_size_x):
                # 使用平方根函数创建更平滑的过渡
                weight = math.sqrt(i / blend_size_x)
                mask[:, i] = weight
        
        if 'right' in neighbors and blend_size_x > 0:
            for i in range(blend_size_x):
                weight = math.sqrt(i / blend_size_x)
                mask[:, -(i+1)] = weight
        
        if 'top' in neighbors and blend_size_y > 0:
            for i in range(blend_size_y):
                weight = math.sqrt(i / blend_size_y)
                mask[i, :] = weight
        
        if 'bottom' in neighbors and blend_size_y > 0:
            for i in range(blend_size_y):
                weight = math.sqrt(i / blend_size_y)
                mask[-(i+1), :] = weight
        
        # 🆕 修复：确保中心区域权重不会过高
        center_h_start, center_h_end = h//4, 3*h//4
        center_w_start, center_w_end = w//4, 3*w//4
        mask[center_h_start:center_h_end, center_w_start:center_w_end] = np.minimum(
            mask[center_h_start:center_h_end, center_w_start:center_w_end], 
            0.9  # 限制中心区域最大权重
        )
        
        return mask

    def _place_image_at_position(self, canvas, weight_map, image, mask, pos_x, pos_y, neighbors, overlap_info, waypoint_idx):
        """🆕 修复版本：正确处理重叠区域的像素融合"""
        try:
            img_h, img_w = image.shape[:2]
            canvas_h, canvas_w = canvas.shape[:2]
            
            # 计算放置区域
            end_x = pos_x + img_w
            end_y = pos_y + img_h
            
            # 边界检查
            if pos_x < 0 or pos_y < 0 or end_x > canvas_w or end_y > canvas_h:
                self.get_logger().error(f'❌ WP{waypoint_idx} 超出画布边界')
                return
            
            # 获取目标区域
            canvas_region = canvas[pos_y:end_y, pos_x:end_x]
            weight_region = weight_map[pos_y:end_y, pos_x:end_x]
            
            # 检测重叠区域
            existing_content = weight_region > 0
            
            self.get_logger().info(f'🔧 WP{waypoint_idx} 修复版融合:')
            self.get_logger().info(f'    位置: ({pos_x},{pos_y})')
            self.get_logger().info(f'    已有内容: {np.sum(existing_content)} 像素')
            
            # 🆕 修复：正确的融合算法
            for c in range(3):
                channel_data = image[:, :, c]
                canvas_channel = canvas_region[:, :, c]
                
                # 分别处理重叠和非重叠区域
                overlap_areas = existing_content & (mask > 0)
                new_areas = ~existing_content & (mask > 0)
                
                if np.any(overlap_areas):
                    # 重叠区域：加权平均融合
                    old_weight = weight_region[overlap_areas]
                    new_weight = mask[overlap_areas]
                    total_weight = old_weight + new_weight
                    
                    # 🔧 关键修复：正确的加权平均
                    canvas_channel[overlap_areas] = (
                        canvas_channel[overlap_areas] * old_weight +
                        channel_data[overlap_areas] * new_weight
                    ) / total_weight
                    
                    self.get_logger().info(f'    🔀 通道{c} 重叠融合: {np.sum(overlap_areas)} 像素')
                
                if np.any(new_areas):
                    # 新区域：直接赋值
                    canvas_channel[new_areas] = channel_data[new_areas]
                    self.get_logger().info(f'    🆕 通道{c} 新区域: {np.sum(new_areas)} 像素')
                
                # 更新画布
                canvas_region[:, :, c] = canvas_channel
            
            # 🆕 修复：正确更新权重图
            # 重叠区域保持原权重，新区域设置为mask值
            new_weight_map = weight_region.copy()
            new_areas_mask = ~existing_content & (mask > 0)
            overlap_areas_mask = existing_content & (mask > 0)
            
            new_weight_map[new_areas_mask] = mask[new_areas_mask]
            # 重叠区域权重保持不变，因为已经在像素值计算中处理了
            
            weight_map[pos_y:end_y, pos_x:end_x] = new_weight_map
            canvas[pos_y:end_y, pos_x:end_x] = canvas_region
            
        except Exception as e:
            self.get_logger().error(f'修复版放置图像失败: {e}')



    def _record_grid_mapping(self, image, pos_x, pos_y, waypoint_idx, canvas_width, canvas_height, img_data):
        """🆕 记录Grid拼接的映射信息"""
        try:
            img_h, img_w = image.shape[:2]
            
            # 记录waypoint贡献
            if waypoint_idx not in self.waypoint_contributions:
                self.waypoint_contributions[waypoint_idx] = {
                    'pixel_count': img_h * img_w,
                    'coverage_area': (pos_x, pos_y, pos_x + img_w, pos_y + img_h),
                    'waypoint_data': {
                        'waypoint_index': waypoint_idx,
                        'color_filename': img_data.get('color_filename', ''),
                        'depth_raw_filename': img_data.get('depth_raw_filename', ''),
                        'depth_file_exists': img_data.get('depth_file_exists', False),
                        'world_pos': img_data.get('world_pos', (0, 0, 0)),
                        'roll': img_data.get('roll', 0),
                        'pitch': img_data.get('pitch', 0),
                        'yaw': img_data.get('yaw', 0),
                        'quaternion': img_data.get('quaternion', (0,0,0,1)),
                        'grid_info': img_data.get('grid_info'),
                        'relative_info': img_data.get('relative_info'),
                        'overlap_info': img_data.get('overlap_info')
                    }
                }
            
            # 简化的像素级映射（稀疏采样）
            sample_step = 10  # 每10个像素采样一次
            for y in range(0, img_h, sample_step):
                for x in range(0, img_w, sample_step):
                    canvas_x = pos_x + x
                    canvas_y = pos_y + y
                    
                    if (0 <= canvas_x < canvas_width and 0 <= canvas_y < canvas_height):
                        self.fusion_mapping[(canvas_x, canvas_y)] = {
                            'source_waypoint': waypoint_idx,
                            'source_pixel': (x, y),
                            'waypoint_data': img_data
                        }
            
            self.get_logger().info(f'📊 WP{waypoint_idx} Grid映射: 位置({pos_x},{pos_y}), 尺寸{img_w}×{img_h}')
            
        except Exception as e:
            self.get_logger().error(f'记录Grid映射失败: {e}')


    def _process_multi_point_with_overlap(self, image_data_list, scan_plan, output_dir, overlap_ratio):
        """🔧 修改：使用直接Grid拼接"""
        try:
            self.get_logger().info(f'开始直接Grid拼接: {overlap_ratio:.1%}')
            
            # 排序waypoint
            sorted_data = sorted(image_data_list, key=lambda x: x['waypoint_index'])
            
            # 提取planning信息
            planning_info = self._extract_planning_info(sorted_data)
            if not planning_info:
                self.get_logger().error('无法提取planning信息，回退到传统方法')
                return self._fallback_traditional_stitching(sorted_data, scan_plan, output_dir, overlap_ratio)
            
            # 执行直接Grid拼接并获取最终尺寸
            stitched_image, final_size = self._direct_grid_stitching(sorted_data, planning_info, overlap_ratio)
            
            # 构建完整的stitch_params（包含实际使用的canvas_size）
            stitch_params = {
                'overlap_ratio': overlap_ratio,
                'canvas_size': final_size,
                'method': 'direct_grid_stitching'
            }
            
            if stitched_image is None:
                return None, "direct_grid_stitching_failed"
            
            # 创建映射信息
            self._create_planning_based_mapping(sorted_data, stitched_image, planning_info, stitch_params, output_dir)
            
            # 保存结果
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            final_path = os.path.join(output_dir, f"multi_point_{overlap_ratio:.0%}_{timestamp}.jpg")
            final_bgr = cv2.cvtColor(stitched_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(final_path, final_bgr)
            
            self.get_logger().info(f'直接Grid拼接完成: {final_path}')
            
            # 发布参考数据
            self._publish_reference_data(stitched_image, None, output_dir)
            
            return final_path, f"direct_grid_stitching_{overlap_ratio:.0%}"
            
        except Exception as e:
            self.get_logger().error(f'直接Grid拼接失败: {e}')
            import traceback
            traceback.print_exc()
            return None, "failed"


    def _create_planning_based_mapping(self, sorted_data, stitched_image, planning_info, stitch_params, output_dir):
        """🆕 创建基于planning的映射信息（保留深度映射）"""
        try:
            import pickle
            import json
            
            self.get_logger().info('🗂️ 创建planning映射信息...')
            
            # 保存完整映射信息（pickle格式）
            mapping_file = os.path.join(output_dir, "fusion_mapping.pkl")
            with open(mapping_file, 'wb') as f:
                pickle.dump({
                    'fusion_mapping': self.fusion_mapping,  # 🔍 保留完整的像素映射
                    'waypoint_contributions': self.waypoint_contributions,  # 🔍 包含深度文件信息
                    'fusion_params': {
                        'scan_type': 'planning_based_multi_point',
                        'total_waypoints': len(self.waypoint_contributions),
                        'canvas_size': stitch_params['canvas_size'],
                        'planning_info': planning_info,  # 🆕 保存planning信息
                        'stitch_params': stitch_params
                    }
                }, f)
            
            # 保存摘要信息（JSON格式）
            summary_file = os.path.join(output_dir, "fusion_mapping_summary.json")
            summary_data = {
                'scan_type': 'planning_based_multi_point',
                'total_mapped_pixels': len(self.fusion_mapping),
                'canvas_size': list(stitch_params['canvas_size']),
                'waypoint_contributions': {}
            }
            
            for wp_idx, info in self.waypoint_contributions.items():
                summary_data['waypoint_contributions'][str(wp_idx)] = {
                    'pixel_count': info['pixel_count'],
                    'coverage_percentage': info['pixel_count'] / len(self.fusion_mapping) * 100 if self.fusion_mapping else 100.0,
                    'waypoint_index': wp_idx,
                    'source_files': {
                        'color': info['waypoint_data'].get('color_filename', ''),
                        'depth': info['waypoint_data'].get('depth_raw_filename', ''),  # 🔍 深度文件
                        'depth_exists': info['waypoint_data'].get('depth_file_exists', False)  # 🔍 存在标志
                    },
                    'grid_position': info['waypoint_data'].get('grid_info'),  # 🆕 grid信息
                    'neighbors': info['waypoint_data'].get('relative_info', {}).get('neighbors', []),  # 🆕 邻居信息
                    'overlap_ratios': info['waypoint_data'].get('overlap_info', {})  # 🆕 重叠信息
                }
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, indent=2, ensure_ascii=False)
            
            self.get_logger().info(f'✅ Planning映射信息已保存:')
            self.get_logger().info(f'  - 完整映射: {mapping_file}')
            self.get_logger().info(f'  - 摘要信息: {summary_file}')
            self.get_logger().info(f'  - 深度文件映射: 已保留所有深度文件路径信息')
            
        except Exception as e:
            self.get_logger().error(f'保存planning映射信息失败: {e}')

    def _fallback_traditional_stitching(self, sorted_data, scan_plan, output_dir, overlap_ratio):
        """🆕 传统拼接方法的回退（当planning信息不可用时）"""
        try:
            self.get_logger().warn('⚠️ Planning信息不可用，使用传统拼接方法')
            
            # 调用原有的拼接方法
            layout_info = self._analyze_waypoint_layout(sorted_data, scan_plan)
            if not layout_info:
                return None, "fallback_layout_analysis_failed"
            
            base_params = self._calculate_overlap_aware_stitch_params(sorted_data, scan_plan, overlap_ratio)
            adaptive_params = self._calculate_adaptive_canvas_size(layout_info, base_params)
            stitched_image = self._improved_overlap_stitching(sorted_data, adaptive_params)
            
            if stitched_image is None:
                return None, "fallback_stitching_failed"
            
            self._create_multi_point_mapping(sorted_data, stitched_image, adaptive_params, output_dir)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            final_path = os.path.join(output_dir, f"fallback_traditional_{overlap_ratio:.0%}_{timestamp}.jpg")
            final_bgr = cv2.cvtColor(stitched_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(final_path, final_bgr)
            
            self.get_logger().info(f'传统方法拼接完成: {final_path}')
            self._publish_reference_data(stitched_image, None, output_dir)
            
            return final_path, f"fallback_traditional_{overlap_ratio:.0%}"
            
        except Exception as e:
            self.get_logger().error(f'传统回退方法失败: {e}')
            return None, "fallback_failed"
    def _calculate_overlap_aware_stitch_params(self, sorted_data, scan_plan, overlap_ratio):
        """简化版本的参数计算（用于回退）"""
        try:
            # 简化版本，只保留基本功能，用于传统拼接回退
            scan_region = scan_plan.scan_region
            region_x = [p.x for p in scan_region]
            region_y = [p.y for p in scan_region]
            
            scan_x_min, scan_x_max = min(region_x), max(region_x)
            scan_y_min, scan_y_max = min(region_y), max(region_y)
            scan_center_x = (scan_x_min + scan_x_max) / 2
            scan_center_y = (scan_y_min + scan_y_max) / 2
            
            scan_height = scan_plan.scan_height
            object_height = scan_plan.object_height
            effective_height = scan_height - object_height
            
            base_fov_width_mm = 2 * effective_height * math.tan(self.fov_h / 2)
            base_fov_height_mm = 2 * effective_height * math.tan(self.fov_v / 2)
            
            effective_step_width = base_fov_width_mm * (1 - overlap_ratio)
            effective_step_height = base_fov_height_mm * (1 - overlap_ratio)
            
            pixels_per_mm_x = self.camera_width / effective_step_width
            pixels_per_mm_y = self.camera_height / effective_step_height
            
            scan_width = scan_y_max - scan_y_min
            scan_height_dim = scan_x_max - scan_x_min
            
            safety_margin = 1.5
            canvas_width = int(scan_width * pixels_per_mm_x * safety_margin)
            canvas_height = int(scan_height_dim * pixels_per_mm_y * safety_margin)
            
            canvas_width = max(canvas_width, 2000)
            canvas_height = max(canvas_height, 1500)
            
            return {
                'scan_center': (scan_center_x, scan_center_y),
                'canvas_size': (canvas_width, canvas_height),
                'pixels_per_mm_x': pixels_per_mm_x,
                'pixels_per_mm_y': pixels_per_mm_y,
                'effective_height': effective_height,
                'overlap_ratio': overlap_ratio,
                'base_fov_width': base_fov_width_mm,
                'base_fov_height': base_fov_height_mm
            }
            
        except Exception as e:
            self.get_logger().error(f'计算基础参数失败: {e}')
            return None

    def _analyze_waypoint_layout(self, sorted_data, scan_plan):
        """🆕 分析waypoint的实际空间布局"""
        try:
            waypoints = []
            for data in sorted_data:
                world_x, world_y, world_z = data['world_pos']
                waypoints.append((world_x, world_y, data['waypoint_index']))
            
            # 提取所有坐标
            x_coords = [wp[0] for wp in waypoints]
            y_coords = [wp[1] for wp in waypoints]
            
            # 分析布局类型
            unique_x = sorted(set(round(x, 1) for x in x_coords))
            unique_y = sorted(set(round(y, 1) for y in y_coords))
            
            is_regular_grid = len(unique_x) * len(unique_y) == len(waypoints)
            
            # 计算waypoint之间的距离
            distances = []
            for i in range(len(waypoints)):
                for j in range(i+1, len(waypoints)):
                    dx = waypoints[i][0] - waypoints[j][0]
                    dy = waypoints[i][1] - waypoints[j][1]
                    dist = math.sqrt(dx*dx + dy*dy)
                    distances.append({
                        'distance': dist,
                        'wp1': waypoints[i][2],
                        'wp2': waypoints[j][2],
                        'delta': (dx, dy)
                    })
            
            distances.sort(key=lambda x: x['distance'])
            
            layout_info = {
                'waypoints': waypoints,
                'num_waypoints': len(waypoints),
                'is_regular_grid': is_regular_grid,
                'grid_dimensions': (len(unique_x), len(unique_y)),
                'x_positions': unique_x,
                'y_positions': unique_y,
                'distances': distances[:6],  # 最小的6个距离
                'scan_bounds': {
                    'x_min': min(x_coords), 'x_max': max(x_coords),
                    'y_min': min(y_coords), 'y_max': max(y_coords)
                }
            }
            
            self.get_logger().info(f'🏗️ Waypoint布局分析:')
            self.get_logger().info(f'  总数: {len(waypoints)} 个waypoint')
            self.get_logger().info(f'  规则网格: {is_regular_grid}')
            self.get_logger().info(f'  网格维度: {len(unique_x)}×{len(unique_y)}')
            self.get_logger().info(f'  最小间距: {distances[0]["distance"]:.1f}mm')
            
            return layout_info
            
        except Exception as e:
            self.get_logger().error(f'布局分析失败: {e}')
            return None

    def _calculate_adaptive_canvas_size(self, layout_info, stitch_params):
        """🆕 根据实际布局自适应计算Canvas大小"""
        try:
            bounds = layout_info['scan_bounds']
            overlap_ratio = stitch_params['overlap_ratio']
            
            # 计算实际扫描区域尺寸
            scan_width = bounds['y_max'] - bounds['y_min']   # Y方向
            scan_height = bounds['x_max'] - bounds['x_min']  # X方向
            
            # 🆕 基于实际waypoint布局调整像素密度
            if layout_info['is_regular_grid']:
                # 规则网格：基于相邻waypoint间距
                min_distance = layout_info['distances'][0]['distance']
                
                # 考虑重叠度的有效步长
                effective_step = min_distance * (1 - overlap_ratio)
                
                # 每个步长对应一个图像宽度，考虑重叠
                pixels_per_mm_x = self.camera_width / effective_step
                pixels_per_mm_y = self.camera_height / effective_step
            else:
                # 非规则布局：使用FOV参数
                base_fov_width = stitch_params['base_fov_width']
                base_fov_height = stitch_params['base_fov_height']
                
                effective_step_width = base_fov_width * (1 - overlap_ratio)
                effective_step_height = base_fov_height * (1 - overlap_ratio)
                
                pixels_per_mm_x = self.camera_width / effective_step_width
                pixels_per_mm_y = self.camera_height / effective_step_height
            
            # 计算Canvas尺寸，添加安全边距
            safety_margin = 1.5
            canvas_width = int(scan_width * pixels_per_mm_x * safety_margin)
            canvas_height = int(scan_height * pixels_per_mm_y * safety_margin)
            
            # 确保最小尺寸
            canvas_width = max(canvas_width, 1500)
            canvas_height = max(canvas_height, 1200)
            
            adaptive_params = stitch_params.copy()
            adaptive_params.update({
                'canvas_size': (canvas_width, canvas_height),
                'pixels_per_mm_x': pixels_per_mm_x,
                'pixels_per_mm_y': pixels_per_mm_y,
                'layout_info': layout_info
            })
            
            self.get_logger().info(f'🎯 自适应Canvas参数:')
            self.get_logger().info(f'  扫描尺寸: {scan_width:.1f}×{scan_height:.1f}mm')
            self.get_logger().info(f'  像素密度: {pixels_per_mm_x:.2f}×{pixels_per_mm_y:.2f} pix/mm')
            self.get_logger().info(f'  Canvas: {canvas_width}×{canvas_height}px')
            
            return adaptive_params
            
        except Exception as e:
            self.get_logger().error(f'自适应Canvas计算失败: {e}')
            return stitch_params

    def _improved_overlap_stitching(self, sorted_data, adaptive_params):
        """🆕 改进的重叠度拼接算法 - 添加图像旋转"""
        try:
            canvas_width, canvas_height = adaptive_params['canvas_size']
            layout_info = adaptive_params['layout_info']
            overlap_ratio = adaptive_params['overlap_ratio']
            
            # 计算扫描区域中心
            bounds = layout_info['scan_bounds']
            scan_center_x = (bounds['x_min'] + bounds['x_max']) / 2
            scan_center_y = (bounds['y_min'] + bounds['y_max']) / 2
            
            self.get_logger().info(f'🔄 改进拼接算法 (包含图像旋转): 重叠度={overlap_ratio:.1%}')
            
            # 创建画布
            canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.float32)
            weight_map = np.zeros((canvas_height, canvas_width), dtype=np.float32)
            
            canvas_center_x = canvas_width // 2
            canvas_center_y = canvas_height // 2
            
            # 初始化映射记录
            self.fusion_mapping = {}
            self.waypoint_contributions = {}
            
            # 获取共同的yaw角度
            common_yaw_rad = math.radians(sorted_data[0]['yaw'])
            
            # 计算每个waypoint的理论FOV覆盖区域
            effective_height = adaptive_params['effective_height']
            
            for img_data in sorted_data:
                # 🔧 修改: 先旋转图像
                original_image = img_data['color_image'].astype(np.float32)
                rotated_image = self._rotate_image_for_stitching(original_image, rotate_180=True)
                
                world_x, world_y, world_z = img_data['world_pos']
                waypoint_idx = img_data['waypoint_index']
                
                # 改进的坐标变换：考虑实际布局
                offset_x = world_x - scan_center_x
                offset_y = world_y - scan_center_y
                
                # 应用yaw旋转的逆变换
                cos_yaw = math.cos(-common_yaw_rad)
                sin_yaw = math.sin(-common_yaw_rad)
                
                rotated_x = offset_x * cos_yaw - offset_y * sin_yaw
                rotated_y = offset_x * sin_yaw + offset_y * cos_yaw
                
                # Canvas坐标映射
                canvas_pos_x = canvas_center_x + int(rotated_y * adaptive_params['pixels_per_mm_x'])
                canvas_pos_y = canvas_center_y + int(rotated_x * adaptive_params['pixels_per_mm_y'])
                
                self.get_logger().info(f'🎯 WP{waypoint_idx} Canvas坐标: ({canvas_pos_x}, {canvas_pos_y}) [已旋转]')
                
                # 边界检查和调整
                canvas_pos_x = max(rotated_image.shape[1]//2, min(canvas_width - rotated_image.shape[1]//2, canvas_pos_x))
                canvas_pos_y = max(rotated_image.shape[0]//2, min(canvas_height - rotated_image.shape[0]//2, canvas_pos_y))
                
                # 创建基于实际重叠度的混合蒙版
                mask = self._create_layout_aware_mask(rotated_image.shape[:2], overlap_ratio, layout_info, waypoint_idx)
                
                # 执行高级混合
                self._advanced_blend_with_layout_awareness(
                    canvas, weight_map, rotated_image, mask, 
                    canvas_pos_x, canvas_pos_y, 
                    overlap_ratio, layout_info, waypoint_idx
                )
                
                # 记录映射信息
                self._record_waypoint_mapping(rotated_image, canvas_pos_x, canvas_pos_y, waypoint_idx, 
                                            canvas_width, canvas_height, img_data)
                
                self.get_logger().info(f'✅ Waypoint {waypoint_idx} 旋转拼接完成')
            
            # 归一化画布
            stitched_canvas = self._normalize_canvas(canvas, weight_map)
            
            # 🔧 修改: 将最终结果旋转回正确方向
            final_result = self._rotate_final_result(stitched_canvas, rotate_180=True)
            
            return final_result
            
        except Exception as e:
            self.get_logger().error(f'旋转拼接算法失败: {e}')
            import traceback
            traceback.print_exc()
            return None

    def _create_layout_aware_mask(self, image_shape, overlap_ratio, layout_info, waypoint_idx):
        """🆕 创建布局感知的混合蒙版"""
        h, w = image_shape
        mask = np.ones((h, w), dtype=np.float32)
        
        # 根据布局类型调整混合策略
        if layout_info['is_regular_grid']:
            # 规则网格：使用标准边缘羽化
            blend_size = int(min(w, h) * overlap_ratio * 0.6)
            blend_size = max(15, min(blend_size, 80))
        else:
            # 非规则布局：更保守的混合
            blend_size = int(min(w, h) * overlap_ratio * 0.4)
            blend_size = max(10, min(blend_size, 50))
        
        # 渐变混合权重
        for i in range(blend_size):
            weight = 0.5 * (1 + math.cos(math.pi * i / blend_size))
            
            if i < h:
                mask[i, :] *= weight
                mask[-(i+1), :] *= weight
            
            if i < w:
                mask[:, i] *= weight
                mask[:, -(i+1)] *= weight
        
        self.get_logger().info(f'🎭 WP{waypoint_idx} 蒙版: 混合区域={blend_size}px')
        
        return mask

    def _advanced_blend_with_layout_awareness(self, canvas, weight_map, image, mask, 
                                            pos_x, pos_y, overlap_ratio, layout_info, waypoint_idx):
        """🆕 修复版本：布局感知的高级混合算法"""
        try:
            img_h, img_w = image.shape[:2]
            canvas_h, canvas_w = canvas.shape[:2]
            
            # 计算放置区域
            img_left = pos_x - img_w // 2
            img_right = pos_x + img_w // 2
            img_top = pos_y - img_h // 2
            img_bottom = pos_y + img_h // 2
            
            # Canvas有效区域
            canvas_left = max(0, img_left)
            canvas_right = min(canvas_w, img_right)
            canvas_top = max(0, img_top)
            canvas_bottom = min(canvas_h, img_bottom)
            
            # 图像对应区域
            img_crop_left = max(0, -img_left)
            img_crop_right = img_crop_left + (canvas_right - canvas_left)
            img_crop_top = max(0, -img_top)
            img_crop_bottom = img_crop_top + (canvas_bottom - canvas_top)
            
            if (canvas_right > canvas_left and canvas_bottom > canvas_top and
                img_crop_right > img_crop_left and img_crop_bottom > img_crop_top):
                
                # 提取区域
                img_region = image[img_crop_top:img_crop_bottom, img_crop_left:img_crop_right]
                mask_region = mask[img_crop_top:img_crop_bottom, img_crop_left:img_crop_right]
                
                if img_region.shape[:2] == mask_region.shape:
                    existing_weight = weight_map[canvas_top:canvas_bottom, canvas_left:canvas_right]
                    overlap_areas = existing_weight > 0
                    
                    self.get_logger().info(f'🔧 WP{waypoint_idx} 修复版高级融合')
                    
                    if np.any(overlap_areas):
                        # 🆕 修复：重叠区域的正确处理
                        for c in range(3):
                            canvas_channel = canvas[canvas_top:canvas_bottom, canvas_left:canvas_right, c]
                            
                            # 分离重叠和非重叠区域
                            overlap_mask = overlap_areas & (mask_region > 0)
                            new_mask = ~overlap_areas & (mask_region > 0)
                            
                            if np.any(overlap_mask):
                                # 重叠区域：加权平均
                                old_weights = existing_weight[overlap_mask]
                                new_weights = mask_region[overlap_mask]
                                total_weights = old_weights + new_weights
                                
                                # 🔧 关键修复：避免像素值累加
                                canvas_channel[overlap_mask] = (
                                    canvas_channel[overlap_mask] * old_weights +
                                    img_region[:, :, c][overlap_mask] * new_weights
                                ) / total_weights
                            
                            if np.any(new_mask):
                                # 新区域：直接赋值
                                canvas_channel[new_mask] = img_region[:, :, c][new_mask]
                            
                            canvas[canvas_top:canvas_bottom, canvas_left:canvas_right, c] = canvas_channel
                        
                        # 🆕 修复：权重图正确更新
                        weight_map[canvas_top:canvas_bottom, canvas_left:canvas_right] = np.where(
                            overlap_areas,
                            existing_weight,  # 重叠区域保持原权重
                            np.maximum(existing_weight, mask_region)  # 新区域使用mask权重
                        )
                        
                        overlap_pixels = np.sum(overlap_areas & (mask_region > 0))
                        self.get_logger().info(f'🔀 修复版重叠融合: {overlap_pixels} 像素')
                    else:
                        # 无重叠区域：正常混合
                        for c in range(3):
                            canvas[canvas_top:canvas_bottom, canvas_left:canvas_right, c] += img_region[:, :, c] * mask_region
                        
                        weight_map[canvas_top:canvas_bottom, canvas_left:canvas_right] += mask_region
                        self.get_logger().info(f'🆕 修复版新区域融合')
                        
        except Exception as e:
            self.get_logger().error(f'修复版布局感知融合失败: {e}')

    def _advanced_blend_to_canvas(self, canvas, weight_map, image, mask, pos_x, pos_y, overlap_ratio):
        """🆕 高级混合算法，考虑重叠度 - 修复坐标系统"""
        try:
            img_h, img_w = image.shape[:2]
            canvas_h, canvas_w = canvas.shape[:2]
            
            # 🆕 修复：正确的图像放置逻辑
            # pos_x 是canvas的列坐标（X坐标）
            # pos_y 是canvas的行坐标（Y坐标）
            
            # 计算图像在canvas上的放置区域
            img_left = pos_x - img_w // 2    # 列方向的左边界
            img_right = pos_x + img_w // 2   # 列方向的右边界
            img_top = pos_y - img_h // 2     # 行方向的上边界
            img_bottom = pos_y + img_h // 2  # 行方向的下边界
            
            # canvas有效区域（边界裁剪）
            canvas_left = max(0, img_left)
            canvas_right = min(canvas_w, img_right)
            canvas_top = max(0, img_top)
            canvas_bottom = min(canvas_h, img_bottom)
            
            # 对应的图像裁剪区域
            img_crop_left = max(0, -img_left)
            img_crop_right = img_crop_left + (canvas_right - canvas_left)
            img_crop_top = max(0, -img_top)
            img_crop_bottom = img_crop_top + (canvas_bottom - canvas_top)
            
            # 🆕 添加调试信息
            self.get_logger().info(f'📏 混合区域: canvas[{canvas_top}:{canvas_bottom}, {canvas_left}:{canvas_right}] <- img[{img_crop_top}:{img_crop_bottom}, {img_crop_left}:{img_crop_right}]')
            
            if (canvas_right > canvas_left and canvas_bottom > canvas_top and
                img_crop_right > img_crop_left and img_crop_bottom > img_crop_top):
                
                # 提取区域
                img_region = image[img_crop_top:img_crop_bottom, img_crop_left:img_crop_right]
                mask_region = mask[img_crop_top:img_crop_bottom, img_crop_left:img_crop_right]
                
                if img_region.shape[:2] == mask_region.shape:
                    # 🆕 重叠度感知的混合
                    existing_weight = weight_map[canvas_top:canvas_bottom, canvas_left:canvas_right]
                    
                    # 在重叠区域使用更智能的混合策略
                    overlap_areas = existing_weight > 0
                    
                    if np.any(overlap_areas):
                        # 重叠区域：使用加权平均
                        blend_weight = mask_region * (1.0 - overlap_ratio * 0.5)
                        
                        self.get_logger().info(f'🔀 检测到重叠区域: {np.sum(overlap_areas)} 像素')
                        
                        for c in range(3):
                            canvas_region = canvas[canvas_top:canvas_bottom, canvas_left:canvas_right, c]
                            new_contribution = img_region[:, :, c] * blend_weight
                            
                            # 在重叠区域进行智能混合
                            canvas_region[overlap_areas] = (
                                canvas_region[overlap_areas] * existing_weight[overlap_areas] + 
                                new_contribution[overlap_areas]
                            ) / (existing_weight[overlap_areas] + blend_weight[overlap_areas] + 1e-8)
                            
                            # 非重叠区域直接添加
                            canvas_region[~overlap_areas] += new_contribution[~overlap_areas]
                            
                            canvas[canvas_top:canvas_bottom, canvas_left:canvas_right, c] = canvas_region
                        
                        weight_map[canvas_top:canvas_bottom, canvas_left:canvas_right] = np.maximum(
                            existing_weight, blend_weight
                        )
                    else:
                        # 无重叠区域：正常混合
                        for c in range(3):
                            canvas[canvas_top:canvas_bottom, canvas_left:canvas_right, c] += img_region[:, :, c] * mask_region
                        
                        weight_map[canvas_top:canvas_bottom, canvas_left:canvas_right] += mask_region
                        
                        self.get_logger().info(f'🆕 新区域混合: {img_region.shape[0]}×{img_region.shape[1]} 像素')
                else:
                    self.get_logger().error(f'❌ 区域形状不匹配: img_region{img_region.shape} vs mask_region{mask_region.shape}')
            else:
                self.get_logger().warn(f'⚠️ 无效的混合区域')
                    
        except Exception as e:
            self.get_logger().error(f'高级画布混合失败: {e}')
            import traceback
            traceback.print_exc()

    def _create_simple_blend_mask(self, image_shape, blend_size=50):
        """创建简单的边缘羽化蒙版"""
        h, w = image_shape
        mask = np.ones((h, w), dtype=np.float32)
        
        # 边缘羽化
        for i in range(blend_size):
            weight = i / blend_size
            
            # 上下边缘
            if i < h:
                mask[i, :] *= weight
                mask[-(i+1), :] *= weight
            
            # 左右边缘
            if i < w:
                mask[:, i] *= weight
                mask[:, -(i+1)] *= weight
        
        return mask

    def _blend_to_canvas(self, canvas, weight_map, image, mask, pos_x, pos_y):
        """将图像融合到画布"""
        try:
            img_h, img_w = image.shape[:2]
            canvas_h, canvas_w = canvas.shape[:2]
            
            # 计算放置区域
            img_left = pos_x - img_w // 2
            img_top = pos_y - img_h // 2
            img_right = img_left + img_w
            img_bottom = img_top + img_h
            
            # 画布有效区域
            canvas_left = max(0, img_left)
            canvas_top = max(0, img_top)
            canvas_right = min(canvas_w, img_right)
            canvas_bottom = min(canvas_h, img_bottom)
            
            # 图像对应区域
            img_crop_left = max(0, -img_left)
            img_crop_top = max(0, -img_top)
            img_crop_right = img_crop_left + (canvas_right - canvas_left)
            img_crop_bottom = img_crop_top + (canvas_bottom - canvas_top)
            
            if (canvas_right > canvas_left and canvas_bottom > canvas_top and
                img_crop_right > img_crop_left and img_crop_bottom > img_crop_top):
                
                # 提取区域
                img_region = image[img_crop_top:img_crop_bottom, img_crop_left:img_crop_right]
                mask_region = mask[img_crop_top:img_crop_bottom, img_crop_left:img_crop_right]
                
                if img_region.shape[:2] == mask_region.shape:
                    # 融合
                    for c in range(3):
                        canvas[canvas_top:canvas_bottom, canvas_left:canvas_right, c] += img_region[:, :, c] * mask_region
                    
                    weight_map[canvas_top:canvas_bottom, canvas_left:canvas_right] += mask_region
                
        except Exception as e:
            self.get_logger().error(f'画布融合失败: {e}')

    def _normalize_canvas(self, canvas, weight_map):
        """🆕 修复版本：改进的画布归一化"""
        try:
            valid_mask = weight_map > 1e-6  # 避免除零
            result = np.zeros_like(canvas, dtype=np.uint8)
            
            self.get_logger().info(f'🎨 修复版归一化: 有效像素 {np.sum(valid_mask)} / {weight_map.size}')
            
            for c in range(3):
                channel = canvas[:, :, c].copy()
                
                # 🔧 修复：只对有效区域进行归一化
                if np.any(valid_mask):
                    # 检查异常值
                    valid_pixels = channel[valid_mask]
                    valid_weights = weight_map[valid_mask]
                    
                    # 归一化前的值检查
                    max_before = np.max(valid_pixels) if len(valid_pixels) > 0 else 0
                    self.get_logger().info(f'  通道{c} 归一化前最大值: {max_before:.1f}')
                    
                    # 正确归一化
                    normalized_values = valid_pixels / valid_weights
                    
                    # 检查归一化后的值
                    max_after = np.max(normalized_values) if len(normalized_values) > 0 else 0
                    self.get_logger().info(f'  通道{c} 归一化后最大值: {max_after:.1f}')
                    
                    # 安全的值范围限制
                    channel[valid_mask] = np.clip(normalized_values, 0, 255)
                
                result[:, :, c] = channel.astype(np.uint8)
            
            return result
            
        except Exception as e:
            self.get_logger().error(f'修复版归一化失败: {e}')
            # 返回安全的结果
            return np.clip(canvas, 0, 255).astype(np.uint8)


    def _record_waypoint_mapping(self, image, canvas_pos_x, canvas_pos_y, waypoint_idx, 
                                canvas_width, canvas_height, img_data):
        """记录waypoint映射信息"""
        try:
            img_h, img_w = image.shape[:2]
            
            # 计算覆盖区域
            img_left = canvas_pos_x - img_w // 2
            img_top = canvas_pos_y - img_h // 2
            img_right = img_left + img_w
            img_bottom = img_top + img_h
            
            canvas_left = max(0, img_left)
            canvas_top = max(0, img_top)
            canvas_right = min(canvas_width, img_right)
            canvas_bottom = min(canvas_height, img_bottom)
            
            # 记录waypoint贡献
            if waypoint_idx not in self.waypoint_contributions:
                self.waypoint_contributions[waypoint_idx] = {
                    'pixel_count': 0,
                    'coverage_area': (canvas_left, canvas_top, canvas_right, canvas_bottom),
                    'waypoint_data': {
                        'waypoint_index': waypoint_idx,
                        'color_filename': img_data.get('color_filename', ''),
                        'depth_raw_filename': img_data.get('depth_raw_filename', ''),
                        'world_pos': img_data.get('world_pos', (0, 0, 0)),
                        'roll': img_data.get('roll', 0),      # 🔧 添加roll
                        'pitch': img_data.get('pitch', 0),    # 🔧 添加pitch
                        'yaw': img_data.get('yaw', 0),
                        'quaternion': img_data.get('quaternion', (0,0,0,1))
                    }
                }
            
            contribution_pixels = (canvas_right - canvas_left) * (canvas_bottom - canvas_top)
            self.waypoint_contributions[waypoint_idx]['pixel_count'] += contribution_pixels
            
            # 简化像素级映射（只记录区域中心附近的点）
            center_x = (canvas_left + canvas_right) // 2
            center_y = (canvas_top + canvas_bottom) // 2
            
            # 只记录中心区域的映射
            sample_region = 100  # 100x100像素的采样区域
            for dy in range(-sample_region//2, sample_region//2, 5):  # 稀疏采样
                for dx in range(-sample_region//2, sample_region//2, 5):
                    canvas_x = center_x + dx
                    canvas_y = center_y + dy
                    
                    if (0 <= canvas_x < canvas_width and 0 <= canvas_y < canvas_height):
                        img_x = img_w // 2 + dx
                        img_y = img_h // 2 + dy
                        
                        if (0 <= img_x < img_w and 0 <= img_y < img_h):
                            self.fusion_mapping[(canvas_x, canvas_y)] = {
                                'source_waypoint': waypoint_idx,
                                'source_pixel': (img_x, img_y),
                                'waypoint_data': img_data
                            }
            
        except Exception as e:
            self.get_logger().error(f'记录映射失败: {e}')

    def _create_single_point_mapping(self, image_data, processed_image, output_dir):
        """为单点扫描创建映射信息"""
        try:
            import pickle
            import json
            
            # 单点扫描的映射相对简单：整个处理后的图像都来自同一个waypoint
            waypoint_idx = image_data['waypoint_index']
            h, w = processed_image.shape[:2]
            
            self.get_logger().info(f'🆕 创建单点映射信息: Waypoint {waypoint_idx}, 尺寸 {w}×{h}')
            
            # 创建简单的1:1映射（简化版本，不需要存储每个像素）
            fusion_mapping = {}
            # 对于单点，我们只存储关键信息，不存储每个像素的映射
            
            waypoint_contributions = {
                waypoint_idx: {
                    'pixel_count': w * h,
                    'coverage_area': (0, 0, w, h),
                    'waypoint_data': {
                        'waypoint_index': waypoint_idx,
                        'color_filename': image_data.get('color_filename', ''),
                        'depth_raw_filename': image_data.get('depth_raw_filename', ''),
                        'world_pos': image_data.get('world_pos', (0, 0, 0)),
                        'roll': image_data.get('roll', 0),
                        'pitch': image_data.get('pitch', 0),
                        'yaw': image_data.get('yaw', 0),
                        'quaternion': image_data.get('quaternion', (0,0,0,1))
                    }
                }
            }
            
            # 保存映射信息
            mapping_file = os.path.join(output_dir, "fusion_mapping.pkl")
            with open(mapping_file, 'wb') as f:
                pickle.dump({
                    'fusion_mapping': fusion_mapping,  # 单点时为空字典，表示使用简化模式
                    'waypoint_contributions': waypoint_contributions,
                    'fusion_params': {
                        'single_point': True,
                        'canvas_size': (w, h),
                        'yaw_correction': -image_data.get('yaw', 0),
                        'waypoint_index': waypoint_idx
                    }
                }, f)
            
            # 保存摘要信息
            summary_file = os.path.join(output_dir, "fusion_mapping_summary.json")
            summary_data = {
                'scan_type': 'single_point',
                'total_mapped_pixels': w * h,
                'canvas_size': [w, h],
                'waypoint_contributions': {
                    str(waypoint_idx): {
                        'pixel_count': w * h,
                        'coverage_percentage': 100.0,
                        'waypoint_index': waypoint_idx,
                        'source_files': {
                            'color': image_data.get('color_filename', ''),
                            'depth': image_data.get('depth_raw_filename', '')
                        }
                    }
                }
            }
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, indent=2, ensure_ascii=False)
            
            self.get_logger().info(f'✅ 单点映射信息已保存: waypoint_{waypoint_idx} -> {w}×{h} 像素')
            
        except Exception as e:
            self.get_logger().error(f'保存单点映射信息失败: {e}')

    def _create_multi_point_mapping(self, sorted_data, stitched_image, stitch_params, output_dir):
        """创建多点映射信息"""
        try:
            # 保存映射信息
            self._save_mapping_files(output_dir, self.fusion_mapping, self.waypoint_contributions, 'multi_point')
            
        except Exception as e:
            self.get_logger().error(f'创建多点映射失败: {e}')
    def _create_overlap_optimized_mask(self, image_shape, overlap_ratio, blend_size=None):
        """🆕 创建重叠度优化的混合蒙版"""
        h, w = image_shape
        mask = np.ones((h, w), dtype=np.float32)
        
        # 🆕 根据重叠度动态调整混合区域大小
        if blend_size is None:
            blend_size = int(min(w, h) * overlap_ratio * 0.5)  # 重叠度越高，混合区域越大
            blend_size = max(20, min(blend_size, 100))  # 限制在合理范围
        
        self.get_logger().info(f'📐 重叠度混合区域: {blend_size}px (重叠度: {overlap_ratio:.1%})')
        
        # 🆕 渐变混合权重
        for i in range(blend_size):
            # 使用平滑的余弦函数而不是线性
            weight = 0.5 * (1 + math.cos(math.pi * i / blend_size))
            
            # 上下边缘
            if i < h:
                mask[i, :] *= weight
                mask[-(i+1), :] *= weight
            
            # 左右边缘
            if i < w:
                mask[:, i] *= weight
                mask[:, -(i+1)] *= weight
        
        # 🆕 中心区域加权（重叠度高时降低中心权重以平衡重叠区域）
        center_weight = 1.0 - overlap_ratio * 0.3
        center_h, center_w = h//4, w//4
        mask[center_h:h-center_h, center_w:w-center_w] *= center_weight
        
        return mask

    def _advanced_blend_to_canvas(self, canvas, weight_map, image, mask, pos_x, pos_y, overlap_ratio):
        """🆕 高级混合算法，考虑重叠度"""
        try:
            img_h, img_w = image.shape[:2]
            canvas_h, canvas_w = canvas.shape[:2]
            
            # 计算放置区域
            img_left = pos_x - img_w // 2
            img_top = pos_y - img_h // 2
            img_right = img_left + img_w
            img_bottom = img_top + img_h
            
            # 画布有效区域
            canvas_left = max(0, img_left)
            canvas_top = max(0, img_top)
            canvas_right = min(canvas_w, img_right)
            canvas_bottom = min(canvas_h, img_bottom)
            
            # 图像对应区域
            img_crop_left = max(0, -img_left)
            img_crop_top = max(0, -img_top)
            img_crop_right = img_crop_left + (canvas_right - canvas_left)
            img_crop_bottom = img_crop_top + (canvas_bottom - canvas_top)
            
            if (canvas_right > canvas_left and canvas_bottom > canvas_top and
                img_crop_right > img_crop_left and img_crop_bottom > img_crop_top):
                
                # 提取区域
                img_region = image[img_crop_top:img_crop_bottom, img_crop_left:img_crop_right]
                mask_region = mask[img_crop_top:img_crop_bottom, img_crop_left:img_crop_right]
                
                if img_region.shape[:2] == mask_region.shape:
                    # 🆕 重叠度感知的混合
                    existing_weight = weight_map[canvas_top:canvas_bottom, canvas_left:canvas_right]
                    
                    # 在重叠区域使用更智能的混合策略
                    overlap_areas = existing_weight > 0
                    
                    if np.any(overlap_areas):
                        # 重叠区域：使用加权平均
                        blend_weight = mask_region * (1.0 - overlap_ratio * 0.5)
                        
                        for c in range(3):
                            canvas_region = canvas[canvas_top:canvas_bottom, canvas_left:canvas_right, c]
                            new_contribution = img_region[:, :, c] * blend_weight
                            
                            # 在重叠区域进行智能混合
                            canvas_region[overlap_areas] = (
                                canvas_region[overlap_areas] * existing_weight[overlap_areas] + 
                                new_contribution[overlap_areas]
                            ) / (existing_weight[overlap_areas] + blend_weight[overlap_areas] + 1e-8)
                            
                            # 非重叠区域直接添加
                            canvas_region[~overlap_areas] += new_contribution[~overlap_areas]
                            
                            canvas[canvas_top:canvas_bottom, canvas_left:canvas_right, c] = canvas_region
                        
                        weight_map[canvas_top:canvas_bottom, canvas_left:canvas_right] = np.maximum(
                            existing_weight, blend_weight
                        )
                    else:
                        # 无重叠区域：正常混合
                        for c in range(3):
                            canvas[canvas_top:canvas_bottom, canvas_left:canvas_right, c] += img_region[:, :, c] * mask_region
                        
                        weight_map[canvas_top:canvas_bottom, canvas_left:canvas_right] += mask_region
                
        except Exception as e:
            self.get_logger().error(f'高级画布混合失败: {e}')

    def _save_mapping_files(self, output_dir, fusion_mapping, waypoint_contributions, scan_type):
        """保存映射文件"""
        try:
            # 保存完整映射信息（pickle格式）
            mapping_file = os.path.join(output_dir, "fusion_mapping.pkl")
            with open(mapping_file, 'wb') as f:
                pickle.dump({
                    'fusion_mapping': fusion_mapping,
                    'waypoint_contributions': waypoint_contributions,
                    'fusion_params': {
                        'scan_type': scan_type,
                        'total_waypoints': len(waypoint_contributions)
                    }
                }, f)
            
            # 保存摘要信息（JSON格式）
            summary_file = os.path.join(output_dir, "fusion_mapping_summary.json")
            summary_data = {
                'scan_type': scan_type,
                'total_mapped_pixels': len(fusion_mapping),
                'waypoint_contributions': {
                    str(wp_idx): {
                        'pixel_count': info['pixel_count'],
                        'coverage_percentage': info['pixel_count'] / len(fusion_mapping) * 100 if fusion_mapping else 100.0,
                        'waypoint_index': wp_idx,
                        'source_files': {
                            'color': info['waypoint_data'].get('color_filename', ''),
                            'depth': info['waypoint_data'].get('depth_raw_filename', '')
                        }
                    }
                    for wp_idx, info in waypoint_contributions.items()
                }
            }
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, indent=2, ensure_ascii=False)
            
            self.get_logger().info(f'✅ 映射信息已保存:')
            self.get_logger().info(f'  - 完整映射: {mapping_file}')
            self.get_logger().info(f'  - 摘要信息: {summary_file}')
            
        except Exception as e:
            self.get_logger().error(f'保存映射文件失败: {e}')

    def _crop_single_point(self, image, scan_plan):
        """单点简化裁剪"""
        h, w = image.shape[:2]
        crop_ratio = 0.9
        margin_w = int(w * (1 - crop_ratio) / 2)
        margin_h = int(h * (1 - crop_ratio) / 2)
        return image[margin_h:h-margin_h, margin_w:w-margin_w]

    def _publish_reference_data(self, stitched_image, depth_data, output_dir):
        """发布参考图像数据供检测节点使用 - 修复cv_bridge问题"""
        try:
            # 1. 发布拼接完成信号
            complete_msg = String()
            complete_msg.data = output_dir
            self.stitching_complete_pub.publish(complete_msg)
            self.get_logger().info(f'📢 发布拼接完成信号: {output_dir}')
            
            # 2. 发布最终参考图像（修复cv_bridge）
            if stitched_image is not None:
                try:
                    # 确保图像是RGB格式
                    if len(stitched_image.shape) == 3:
                        # 🆕 修复：手动创建ROS消息，避免cv_bridge问题
                        reference_msg = Image()
                        reference_msg.header.stamp = self.get_clock().now().to_msg()
                        reference_msg.header.frame_id = "stitched_frame"
                        reference_msg.height = stitched_image.shape[0]
                        reference_msg.width = stitched_image.shape[1]
                        reference_msg.encoding = "rgb8"
                        reference_msg.is_bigendian = False
                        reference_msg.step = stitched_image.shape[1] * 3
                        reference_msg.data = stitched_image.tobytes()
                    else:
                        # 单通道图像
                        reference_msg = Image()
                        reference_msg.header.stamp = self.get_clock().now().to_msg()
                        reference_msg.header.frame_id = "stitched_frame"
                        reference_msg.height = stitched_image.shape[0]
                        reference_msg.width = stitched_image.shape[1]
                        reference_msg.encoding = "mono8"
                        reference_msg.is_bigendian = False
                        reference_msg.step = stitched_image.shape[1]
                        reference_msg.data = stitched_image.tobytes()
                    
                    self.reference_image_pub.publish(reference_msg)
                    self.get_logger().info('📸 发布参考图像（修复版）')
                    
                except Exception as img_error:
                    self.get_logger().error(f'参考图像发布失败: {img_error}')
            
            # 3. 发布深度信息（如果有）
            if depth_data is not None:
                try:
                    # 🆕 修复：手动创建深度消息
                    if depth_data.dtype != np.uint16:
                        depth_data = depth_data.astype(np.uint16)
                    
                    depth_msg = Image()
                    depth_msg.header.stamp = self.get_clock().now().to_msg()
                    depth_msg.header.frame_id = "stitched_frame"
                    depth_msg.height = depth_data.shape[0]
                    depth_msg.width = depth_data.shape[1]
                    depth_msg.encoding = "16UC1"
                    depth_msg.is_bigendian = False
                    depth_msg.step = depth_data.shape[1] * 2
                    depth_msg.data = depth_data.tobytes()
                    
                    self.reference_depth_pub.publish(depth_msg)
                    self.get_logger().info('📏 发布深度信息（修复版）')
                    
                except Exception as depth_error:
                    self.get_logger().error(f'深度信息发布失败: {depth_error}')
            else:
                self.get_logger().info('📏 深度数据不可用，跳过深度发布')
            
        except Exception as e:
            self.get_logger().error(f'发布参考数据失败: {e}')
            

def main(args=None):
    rclpy.init(args=args)
    node = SmartStitcherNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()