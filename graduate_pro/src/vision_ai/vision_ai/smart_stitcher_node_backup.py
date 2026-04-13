#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from vision_ai_interfaces.srv import ProcessStitching
from vision_ai_interfaces.msg import StitchResult, ImageData
from sensor_msgs.msg import Image
import cv2
import numpy as np
import math
import os
from datetime import datetime
from cv_bridge import CvBridge
from std_msgs.msg import String
import open3d as o3d
from scipy.spatial.transform import Rotation
import numpy as np
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
        
        # 🆕 发布者 - 发布最终参考图像
        self.reference_image_pub = self.create_publisher(
            Image,
            '/reference_image',
            10
        )
        
        # 🆕 发布者 - 发布参考深度信息
        self.reference_depth_pub = self.create_publisher(
            Image,
            '/reference_depth',
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
        self.use_3d_pointcloud = True  # 启用3D点云拼接
        self.voxel_size = 1.0  # mm
        self.normal_radius = 8.0  # mm
        self.get_logger().info('智能拼接节点已启动 (双轴优化版)')
        self.get_logger().info(f'FOV: {math.degrees(self.fov_h):.1f}° × {math.degrees(self.fov_v):.1f}°')

    def stitch_callback(self, request, response):
        """修改回调函数，保存输出目录信息"""
        try:
            self.get_logger().info(f'收到拼接请求: {len(request.image_data)} 张图像')
            
            # 🆕 保存输出目录信息供后续使用
            self.current_output_dir = request.output_directory
            self.get_logger().info(f'📁 输出目录: {self.current_output_dir}')
            
            # 转换输入数据
            image_data = self._convert_input_data(request.image_data)
            scan_plan = request.scan_plan
            
            if not image_data:
                response.success = False
                response.message = "图像数据转换失败"
                return response
            
            # 🆕 检查深度文件可用性
            depth_available_count = sum(1 for img in image_data if img.get('depth_file_exists', False))
            self.get_logger().info(f'🔍 深度文件可用性: {depth_available_count}/{len(image_data)}')
            
            # 根据策略选择处理模式
            if scan_plan.strategy == "single_point" or len(image_data) == 1:
                result_path, method = self._process_single_point(image_data[0], scan_plan, request.output_directory)
            else:
                result_path, method = self._process_multi_point(image_data, scan_plan, request.output_directory)
            
            if result_path:
                response.success = True
                response.message = f"拼接成功: {method}"
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

    def _process_multi_point_3d_pointcloud(self, image_data_list, scan_plan, output_dir):
        """多点模式：使用3D点云拼接"""
        try:
            self.get_logger().info('开始3D点云拼接...')
            
            # 检查是否有深度数据
            if not self._check_depth_data_availability(image_data_list, output_dir):
                self.get_logger().warn('深度数据不可用，回退到2D拼接')
                return self._process_multi_point(image_data_list, scan_plan, output_dir)
            
            # 执行3D点云拼接
            stitched_result = self._perform_3d_pointcloud_stitching(image_data_list, scan_plan, output_dir)
            
            if stitched_result is None:
                self.get_logger().warn('3D点云拼接失败，回退到2D拼接')
                return self._process_multi_point(image_data_list, scan_plan, output_dir)
            
            return stitched_result
            
        except Exception as e:
            self.get_logger().error(f'3D点云拼接异常: {e}')
            # 回退到原有2D拼接方法
            return self._process_multi_point(image_data_list, scan_plan, output_dir)
    
    def _check_depth_data_availability(self, image_data_list, output_dir):
        """检查深度数据是否可用"""
        try:
            available_count = 0
            for img_data in image_data_list:
                depth_filename = img_data.get('depth_raw_filename')
                if depth_filename:
                    depth_path = os.path.join(output_dir, depth_filename)
                    if os.path.exists(depth_path):
                        available_count += 1
            
            availability_ratio = available_count / len(image_data_list)
            self.get_logger().info(f'深度数据可用性: {available_count}/{len(image_data_list)} ({availability_ratio:.1%})')
            
            return availability_ratio >= 0.5  # 80%以上可用才使用3D拼接
            
        except Exception as e:
            self.get_logger().error(f'检查深度数据失败: {e}')
            return False
    
    def _perform_3d_pointcloud_stitching(self, image_data_list, scan_plan, output_dir):
        """执行3D点云拼接"""
        try:
            # 创建点云数据
            point_clouds = []
            valid_data = []
            
            for i, img_data in enumerate(image_data_list):
                pcd = self._create_pointcloud_from_rgbd_data(img_data, output_dir)
                if pcd is not None and len(pcd.points) > 1000:  # 至少1000个点
                    point_clouds.append(pcd)
                    valid_data.append(img_data)
                    self.get_logger().info(f'点云 {i+1}: {len(pcd.points)} 个点')
                else:
                    self.get_logger().warn(f'Waypoint {img_data["waypoint_index"]} 点云生成失败或点数不足')
            
            if len(point_clouds) < 2:
                self.get_logger().error('有效点云数量不足，无法进行拼接')
                return None
            
            # 执行点云配准和融合
            merged_pcd = self._register_and_merge_pointclouds(point_clouds, valid_data)
            
            if merged_pcd is None:
                return None
            
            # 生成俯视图作为拼接结果
            top_view_image = self._generate_topview_from_pointcloud(merged_pcd)
            
            if top_view_image is None:
                return None
            
            # 保存结果
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # 保存点云
            pcd_path = os.path.join(output_dir, f"merged_pointcloud_{timestamp}.ply")
            o3d.io.write_point_cloud(pcd_path, merged_pcd)
            
            # 保存俯视图
            final_path = os.path.join(output_dir, f"pointcloud_topview_{timestamp}.jpg")
            final_bgr = cv2.cvtColor(top_view_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(final_path, final_bgr)
            
            self.get_logger().info(f'3D点云拼接完成: {final_path}')
            self.get_logger().info(f'点云文件: {pcd_path}')
            
            # 发布参考数据
            self._publish_reference_data(top_view_image, None, output_dir)
            
            return final_path, "3d_pointcloud_stitching"
            
        except Exception as e:
            self.get_logger().error(f'3D点云拼接执行失败: {e}')
            import traceback
            traceback.print_exc()
            return None
    
    def _create_pointcloud_from_rgbd_data(self, img_data, output_dir):
        """从RGB-D数据创建点云"""
        try:
            # 获取RGB图像
            color_image = img_data['color_image']  # RGB格式
            
            # 加载深度数据
            depth_filename = img_data.get('depth_raw_filename')
            if not depth_filename:
                self.get_logger().error('缺少深度文件名')
                return None
            
            depth_path = os.path.join(output_dir, depth_filename)
            if not os.path.exists(depth_path):
                self.get_logger().error(f'深度文件不存在: {depth_path}')
                return None
            
            depth_raw = np.load(depth_path)
            
            # 验证数据
            if color_image.shape[:2] != depth_raw.shape[:2]:
                self.get_logger().error(f'RGB和深度图像尺寸不匹配: {color_image.shape} vs {depth_raw.shape}')
                return None
            
            # 获取位姿信息
            world_pos = img_data['world_pos']
            quaternion = img_data['quaternion']
            
            # 生成3D点云
            pcd = self._rgbd_to_pointcloud(color_image, depth_raw, world_pos, quaternion)
            
            return pcd
            
        except Exception as e:
            self.get_logger().error(f'点云生成失败: {e}')
            return None
    
    def _rgbd_to_pointcloud(self, color_image, depth_raw, world_pos, quaternion):
        """将RGB-D数据转换为世界坐标系点云"""
        try:
            height, width = depth_raw.shape
            
            # 生成像素坐标网格
            u, v = np.meshgrid(np.arange(width), np.arange(height))
            u = u.flatten()
            v = v.flatten()
            depth_flat = depth_raw.flatten()
            
            # 过滤有效深度值
            valid_mask = (depth_flat > 0) & (depth_flat < 2000)  # 0-2米有效范围
            if np.sum(valid_mask) == 0:
                self.get_logger().warning('没有有效的深度数据')
                return None
            
            u_valid = u[valid_mask]
            v_valid = v[valid_mask]
            depth_valid = depth_flat[valid_mask].astype(np.float32)
            
            # 像素坐标转相机坐标
            X = (u_valid - self.cx) * depth_valid / self.fx
            Y = (v_valid - self.cy) * depth_valid / self.fy
            Z = depth_valid
            
            # 相机坐标系下的3D点
            points_camera = np.column_stack([X, Y, Z])
            
            # 相机坐标转世界坐标
            points_world = self._transform_to_world_coordinates(points_camera, world_pos, quaternion)
            
            # 提取对应的颜色信息
            colors = color_image[v_valid, u_valid] / 255.0  # 归一化到[0,1]
            
            # 创建Open3D点云
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_world)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            
            # 点云预处理
            pcd = self._preprocess_pointcloud(pcd)
            
            return pcd
            
        except Exception as e:
            self.get_logger().error(f'RGB-D转点云失败: {e}')
            return None
    
    def _transform_to_world_coordinates(self, points_camera, world_pos, quaternion):
        """将相机坐标系点转换到世界坐标系"""
        # 提取位置和旋转
        position = np.array([world_pos[0], world_pos[1], world_pos[2]])
        quat = np.array([quaternion[0], quaternion[1], quaternion[2], quaternion[3]])  # [qx, qy, qz, qw]
        
        # 四元数转旋转矩阵
        rotation = Rotation.from_quat(quat)
        R = rotation.as_matrix()
        
        # 构建变换矩阵
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = position
        
        # 齐次坐标变换
        points_homo = np.column_stack([points_camera, np.ones(len(points_camera))])
        points_world_homo = (T @ points_homo.T).T
        
        return points_world_homo[:, :3]
    
    def _preprocess_pointcloud(self, pcd):
        """点云预处理"""
        # 统计滤波去除离群点
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        
        # 体素下采样
        pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)
        
        # 计算法线
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.normal_radius, max_nn=30
            )
        )
        
        return pcd
    
    def _register_and_merge_pointclouds(self, point_clouds, img_data_list):
        """配准和融合多个点云"""
        try:
            self.get_logger().info(f'开始配准融合 {len(point_clouds)} 个点云')
            
            # 检查是否为平行扫描
            yaw_angles = [data['yaw'] for data in img_data_list]
            yaw_std = np.std(yaw_angles)
            is_parallel = yaw_std < 5.0
            
            self.get_logger().info(f'扫描模式检测: 平行扫描={is_parallel}, yaw标准差={yaw_std:.2f}°')
            
            if is_parallel:
                # 平行扫描：直接基于机器人位姿融合
                return self._parallel_scan_fusion(point_clouds, img_data_list)
            else:
                # 非平行扫描：使用完整配准流程
                return self._general_pointcloud_registration(point_clouds, img_data_list)
                
        except Exception as e:
            self.get_logger().error(f'点云配准融合失败: {e}')
            return None
    
    def _parallel_scan_fusion(self, point_clouds, img_data_list):
        """平行扫描的简化融合"""
        try:
            self.get_logger().info('执行平行扫描优化融合...')
            
            # 选择参考点云（第一个）
            merged_pcd = point_clouds[0]
            
            # 逐个融合其他点云
            for i in range(1, len(point_clouds)):
                source_pcd = point_clouds[i]
                
                # 基于机器人位姿计算变换矩阵
                transform = self._compute_pose_transform(
                    img_data_list[0],  # 参考位姿
                    img_data_list[i]   # 当前位姿
                )
                
                # 应用变换
                source_transformed = source_pcd.transform(transform)
                
                # 可选：小幅ICP精调
                refined_transform = self._fine_tune_registration(source_transformed, merged_pcd)
                if refined_transform is not None:
                    source_transformed.transform(refined_transform)
                
                # 融合点云
                merged_pcd += source_transformed
                
                self.get_logger().info(f'点云 {i+1} 融合完成')
            
            # 后处理
            merged_pcd = self._postprocess_merged_pointcloud(merged_pcd)
            
            return merged_pcd
            
        except Exception as e:
            self.get_logger().error(f'平行扫描融合失败: {e}')
            return None
    
    def _compute_pose_transform(self, ref_data, curr_data):
        """计算基于机器人位姿的变换矩阵"""
        ref_pos = np.array(ref_data['world_pos'])
        curr_pos = np.array(curr_data['world_pos'])
        
        ref_quat = np.array(ref_data['quaternion'])
        curr_quat = np.array(curr_data['quaternion'])
        
        # 计算相对变换
        ref_rot = Rotation.from_quat(ref_quat)
        curr_rot = Rotation.from_quat(curr_quat)
        
        rel_rot = ref_rot.inv() * curr_rot
        rel_trans = ref_rot.inv().apply(curr_pos - ref_pos)
        
        # 构建变换矩阵
        T = np.eye(4)
        T[:3, :3] = rel_rot.as_matrix()
        T[:3, 3] = rel_trans
        
        return T
    
    def _fine_tune_registration(self, source, target, max_distance=3.0):
        """ICP精调"""
        try:
            result = o3d.pipelines.registration.registration_icp(
                source, target, max_distance, np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30)
            )
            
            if result.fitness > 0.8 and result.inlier_rmse < max_distance:
                self.get_logger().info(f'ICP精调成功 - fitness: {result.fitness:.3f}, RMSE: {result.inlier_rmse:.2f}mm')
                return result.transformation
            else:
                return None
                
        except Exception as e:
            self.get_logger().warning(f'ICP精调失败: {e}')
            return None
    
    def _postprocess_merged_pointcloud(self, merged_pcd):
        """融合后点云后处理"""
        # 去重和滤波
        merged_pcd = merged_pcd.voxel_down_sample(voxel_size=self.voxel_size * 0.5)
        merged_pcd, _ = merged_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        
        # 重新计算法线
        merged_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.normal_radius, max_nn=30
            )
        )
        
        return merged_pcd
    
    def _generate_topview_from_pointcloud(self, pcd):
        """从点云生成俯视图"""
        try:
            points = np.asarray(pcd.points)
            colors = np.asarray(pcd.colors)
            
            if len(points) == 0:
                return None
            
            # 投影到XY平面
            x_coords = points[:, 0]
            y_coords = points[:, 1]
            
            # 计算边界
            x_min, x_max = x_coords.min(), x_coords.max()
            y_min, y_max = y_coords.min(), y_coords.max()
            
            # 创建俯视图像
            resolution = 0.5  # 0.5mm/pixel
            width = int((y_max - y_min) / resolution) + 1
            height = int((x_max - x_min) / resolution) + 1
            
            # 限制图像大小
            max_size = 4000
            if width > max_size or height > max_size:
                scale = max_size / max(width, height)
                width = int(width * scale)
                height = int(height * scale)
                resolution = resolution / scale
            
            top_view = np.zeros((height, width, 3), dtype=np.uint8)
            
            # 将点投影到图像
            for i in range(len(points)):
                x_idx = int((x_coords[i] - x_min) / resolution)
                y_idx = int((y_coords[i] - y_min) / resolution)
                
                if 0 <= x_idx < height and 0 <= y_idx < width:
                    color = (colors[i] * 255).astype(np.uint8)
                    top_view[x_idx, y_idx] = color
            
            self.get_logger().info(f'俯视图生成完成: {width}×{height}, 分辨率={resolution:.2f}mm/px')
            
            return top_view
            
        except Exception as e:
            self.get_logger().error(f'俯视图生成失败: {e}')
            return None
    
    # 修改主处理流程
    def _process_multi_point(self, image_data_list, scan_plan, output_dir):
        """多点处理主入口 - 优先使用3D点云拼接"""
        try:
            # 优先尝试3D点云拼接
            if self.use_3d_pointcloud:
                result = self._process_multi_point_3d_pointcloud(image_data_list, scan_plan, output_dir)
                if result[0] is not None:  # 成功
                    return result
            
            # 回退到原有2D拼接算法
            self.get_logger().info('回退到2D拼接算法...')
            return self._process_multi_point_2d_fallback(image_data_list, scan_plan, output_dir)
            
        except Exception as e:
            self.get_logger().error(f'多点处理失败: {e}')
            return None, "failed"
        
    def _convert_input_data(self, ros_image_data):
        """转换ROS图像数据为内部格式 - 修复cv_bridge问题"""
        converted_data = []
        self.current_output_dir = getattr(self, 'current_output_dir', None)
        for img_data in ros_image_data:
            try:
                # 🆕 修复：安全的图像转换，绕过cv_bridge问题
                self.get_logger().info(f'🎨 调试Waypoint {img_data.waypoint.waypoint_index} 图像转换...')
                try:
                    # 尝试使用passthrough避免格式转换问题
                     # 获取原始ROS图像信息
                    ros_image = img_data.image
                    self.get_logger().info(f'原始ROS图像: {ros_image.width}x{ros_image.height}, 编码: {ros_image.encoding}')
                    
                    # 尝试passthrough模式
                    color_cv_raw = self.bridge.imgmsg_to_cv2(img_data.image, desired_encoding="passthrough")
                    self.get_logger().info(f'passthrough转换后: {color_cv_raw.shape}, dtype: {color_cv_raw.dtype}')
                    
                    # 🆕 检查原始像素值（用于调试颜色）
                    if len(color_cv_raw.shape) == 3:
                        # 取图像中心点的像素值作为调试样本
                        center_y, center_x = color_cv_raw.shape[0]//2, color_cv_raw.shape[1]//2
                        center_pixel = color_cv_raw[center_y, center_x]
                        self.get_logger().info(f'中心像素值(原始): {center_pixel}')
                    
                    # 检查并转换格式
                    if len(color_cv_raw.shape) == 3 and color_cv_raw.shape[2] == 3:
                        if ros_image.encoding == 'bgr8':
                            # BGR -> RGB
                            color_cv = cv2.cvtColor(color_cv_raw, cv2.COLOR_BGR2RGB)
                            self.get_logger().info('执行了BGR->RGB转换')
                        elif ros_image.encoding == 'rgb8':
                            # 已经是RGB，直接使用
                            color_cv = color_cv_raw.copy()
                            self.get_logger().info('图像已是RGB格式，直接使用')
                        else:
                            # 未知编码，尝试BGR->RGB转换
                            color_cv = cv2.cvtColor(color_cv_raw, cv2.COLOR_BGR2RGB)
                            self.get_logger().warn(f'未知编码 {ros_image.encoding}，尝试BGR->RGB转换')
                        
                        # 🆕 检查转换后的像素值
                        center_pixel_after = color_cv[center_y, center_x]
                        self.get_logger().info(f'中心像素值(转换后): {center_pixel_after}')
                        
                except Exception as cv_error:
                    self.get_logger().error(f'cv_bridge转换失败: {cv_error}')
                    
                    # 🆕 备用方案：直接从原始数据构造
                    try:
                        self.get_logger().info('尝试备用图像转换方案...')
                        
                        # 从ROS消息手动提取图像数据
                        ros_image = img_data.image
                        width = ros_image.width
                        height = ros_image.height
                        encoding = ros_image.encoding
                        step = ros_image.step
                        data = ros_image.data
                        
                        self.get_logger().info(f'ROS图像信息: {width}x{height}, 编码: {encoding}, step: {step}')
                        
                        # 转换为numpy数组
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
                
                # 🆕 改进的深度文件路径处理
                color_filename = img_data.filename  # 例如: "color_waypoint_001.jpg"
                depth_filename = None
                
                if color_filename:
                    # 🆕 根据颜色文件名推断深度文件名
                    if 'color_waypoint_' in color_filename:
                        # 从 "color_waypoint_001.jpg" 推断 "depth_raw_waypoint_001.npy"
                        depth_filename = color_filename.replace('color_waypoint_', 'depth_raw_waypoint_').replace('.jpg', '.npy')
                    else:
                        # 从waypoint_index生成标准深度文件名
                        waypoint_idx = wp.waypoint_index
                        depth_filename = f"depth_raw_waypoint_{waypoint_idx:03d}.npy"
                
                # 🆕 验证深度文件是否存在
                depth_file_exists = False
                if depth_filename and self.current_output_dir:
                    depth_full_path = os.path.join(self.current_output_dir, depth_filename)
                    depth_file_exists = os.path.exists(depth_full_path)
                    
                    self.get_logger().info(f'🔍 深度文件检查:')
                    self.get_logger().info(f'  - 文件名: {depth_filename}')
                    self.get_logger().info(f'  - 完整路径: {depth_full_path}')
                    self.get_logger().info(f'  - 文件存在: {depth_file_exists}')
                    
                    if not depth_file_exists:
                        # 🆕 尝试其他可能的文件名模式
                        alternative_patterns = [
                            f"depth_raw_waypoint_{wp.waypoint_index+1:03d}.npy",  # 索引+1
                            f"depth_waypoint_{wp.waypoint_index:03d}.npy",       # 不带raw
                            f"depth_waypoint_{wp.waypoint_index+1:03d}.npy"      # 不带raw，索引+1
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
                    
                    # 🆕 文件路径信息
                    'color_filename': color_filename,
                    'depth_raw_filename': depth_filename,
                    'depth_file_exists': depth_file_exists  # 🆕 添加存在标志
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

    def _process_multi_point_2d_fallback(self, image_data_list, scan_plan, output_dir):
        """多点模式：双轴优化融合"""
        try:
            self.get_logger().info('处理多点扫描 - 双轴优化融合')
            
            # 计算FOV覆盖参数
            fov_params = self._calculate_fov_parameters(scan_plan)
            
            # 计算理论参数
            theoretical_params = self._calculate_theoretical_parameters(image_data_list, scan_plan, fov_params)
            
            # 应用双轴优化参数（基于你的成功配置）
            dual_axis_params = {
                'margin_ratio_x': 0.01,     # 基于理论 + 微调
                'margin_ratio_y': 0.01,     # 基于理论 + 微调
                'blend_size_x': theoretical_params.get('blend_size_x', 60),
                'blend_size_y': theoretical_params.get('blend_size_y', 70),
                'scale_adjust_x': 1,    # 你的成功参数
                'scale_adjust_y': 1,    # 你的成功参数
                'offset_adjust_x': 0,
                'offset_adjust_y': 0       # 可根据需要调整
            }
            
            self.get_logger().info(f'双轴参数: X缩放={dual_axis_params["scale_adjust_x"]}, Y缩放={dual_axis_params["scale_adjust_y"]}')
            
            # 执行双轴融合
            stitched_image = self._dual_axis_fusion_stitching_grid_fix(image_data_list, scan_plan, fov_params, dual_axis_params)
            
            if stitched_image is None:
                return None, "fusion_failed"
            
            # 保存拼接结果
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            stitched_path = os.path.join(output_dir, f"dual_axis_stitched_{timestamp}.jpg")
            stitched_bgr = cv2.cvtColor(stitched_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(stitched_path, stitched_bgr)
            self.get_logger().info(f'双轴拼接完成: {stitched_path}')
            
            # 精确裁剪到scan area
            final_image = self._crop_to_scan_area_dual_axis(stitched_image, scan_plan, fov_params, dual_axis_params)
            
            # 保存最终结果
            final_path = os.path.join(output_dir, f"final_dual_axis_result_{timestamp}.jpg")
            final_bgr = cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(final_path, final_bgr)
            self._save_fusion_mapping(output_dir)
            self.get_logger().info(f'最终结果: {final_path}')
            self.get_logger().info(f'尺寸变化: {stitched_image.shape[1]}×{stitched_image.shape[0]} → {final_image.shape[1]}×{final_image.shape[0]}')
            # 第193行修改为：
            self._publish_reference_data(final_image, None, output_dir)
            return final_path, "dual_axis_optimized_fusion"
            
        except Exception as e:
            self.get_logger().error(f'多点处理失败: {e}')
            import traceback
            traceback.print_exc()
            return None, "failed"


    
    def _calculate_fov_parameters(self, scan_plan):
        """计算FOV相关参数"""
        scan_height = scan_plan.scan_height
        object_height = scan_plan.object_height
        effective_height = scan_height - object_height
        
        # FOV覆盖尺寸
        fov_width_mm = 2 * effective_height * math.tan(self.fov_h / 2)
        fov_height_mm = 2 * effective_height * math.tan(self.fov_v / 2)
        
        # 像素密度
        pixels_per_mm_x = self.camera_width / fov_width_mm
        pixels_per_mm_y = self.camera_height / fov_height_mm
        
        return {
            'effective_height': effective_height,
            'fov_width_mm': fov_width_mm,
            'fov_height_mm': fov_height_mm,
            'pixels_per_mm_x': pixels_per_mm_x,
            'pixels_per_mm_y': pixels_per_mm_y
        }

    def _calculate_theoretical_parameters(self, image_data_list, scan_plan, fov_params):
        """计算理论融合参数"""
        try:
            # 分析waypoint间距
            positions = [(img['world_pos'][0], img['world_pos'][1]) for img in image_data_list]
            
            distances_x = []
            distances_y = []
            
            for i in range(len(positions)):
                for j in range(i+1, len(positions)):
                    dx = abs(positions[i][0] - positions[j][0])
                    dy = abs(positions[i][1] - positions[j][1])
                    
                    if dx < 400 and dy < 400:  # 相邻waypoint阈值
                        if dx > 0:
                            distances_x.append(dx)
                        if dy > 0:
                            distances_y.append(dy)
            
            # 平均间距
            avg_distance_x = sum(distances_x) / len(distances_x) if distances_x else fov_params['fov_height_mm']
            avg_distance_y = sum(distances_y) / len(distances_y) if distances_y else fov_params['fov_width_mm']
            
            # 理论重叠率
            overlap_x = max(0, (fov_params['fov_height_mm'] - avg_distance_x)*1.5 / fov_params['fov_height_mm'])
            overlap_y = max(0, (fov_params['fov_width_mm'] - avg_distance_y)*1.5 / fov_params['fov_width_mm'])
            
            # 建议融合区域
            blend_size_x = max(40, int(self.camera_height * overlap_x * 1.5))
            blend_size_y = max(600, int(self.camera_width * overlap_y * 1.5))
            
            self.get_logger().info(f'理论参数: 重叠率X={overlap_x:.3f}, Y={overlap_y:.3f}')
            self.get_logger().info(f'建议融合区域: X={blend_size_x}px, Y={blend_size_y}px')
            
            return {
                'overlap_x': overlap_x,
                'overlap_y': overlap_y,
                'blend_size_x': blend_size_x,
                'blend_size_y': blend_size_y
            }
            
        except Exception as e:
            self.get_logger().error(f'理论参数计算失败: {e}')
            return {'blend_size_x': 60, 'blend_size_y': 70}

    def _dual_axis_fusion_stitchin(self, image_data_list, scan_plan, fov_params, dual_axis_params):
        """双轴融合拼接 - 调试版本，解决重影和拼接问题"""
        try:
            if not image_data_list:
                return None
            
            self.get_logger().info('🔧 开始调试版双轴融合拼接...')
            
            # 分析waypoint布局模式
            layout_type = self._analyze_waypoint_layout(image_data_list)
            
            # 获取yaw角度
            yaw_deg = image_data_list[0]['yaw']
            yaw_rad = math.radians(yaw_deg)
            
            # 计算scan area信息
            scan_region = scan_plan.scan_region
            if not scan_region or len(scan_region) < 4:
                self.get_logger().error('缺少scan region信息')
                return None
            
            region_x = [p.x for p in scan_region]
            region_y = [p.y for p in scan_region]
            scan_x_min, scan_x_max = min(region_x), max(region_x)
            scan_y_min, scan_y_max = min(region_y), max(region_y)
            scan_center_x = (scan_x_min + scan_x_max) / 2
            scan_center_y = (scan_y_min + scan_y_max) / 2
            
            self.get_logger().info(f'🎯 Scan区域: X[{scan_x_min:.1f}, {scan_x_max:.1f}], Y[{scan_y_min:.1f}, {scan_y_max:.1f}]')
            
            # 🆕 调试：检查重叠程度
            self._debug_waypoint_overlap(image_data_list, fov_params)
            
            # 排序waypoint
            waypoint_positions = []
            for img_data in image_data_list:
                world_x, world_y, world_z = img_data['world_pos']
                waypoint_idx = img_data['waypoint_index']
                waypoint_positions.append({
                    'index': waypoint_idx,
                    'world_pos': (world_x, world_y),
                    'data': img_data
                })
            
            # 按索引排序（保持扫描顺序）
            waypoint_positions.sort(key=lambda w: w['index'])
            
            # 计算画布尺寸
            scan_width = scan_y_max - scan_y_min
            scan_height = scan_x_max - scan_x_min
            
            cos_yaw = abs(math.cos(yaw_rad))
            sin_yaw = abs(math.sin(yaw_rad))
            rotated_width = scan_width * cos_yaw + scan_height * sin_yaw
            rotated_height = scan_height * cos_yaw + scan_width * sin_yaw
            
            # 🆕 调整画布尺寸策略 - 减少重叠
            canvas_width = int(rotated_width * fov_params['pixels_per_mm_x'] * 
                            (1 + dual_axis_params['margin_ratio_x'] * 0.5) * dual_axis_params['scale_adjust_x'])
            canvas_height = int(rotated_height * fov_params['pixels_per_mm_y'] * 
                            (1 + dual_axis_params['margin_ratio_y'] * 0.5) * dual_axis_params['scale_adjust_y'])
            
            self.get_logger().info(f'🎨 调试画布: {canvas_width}×{canvas_height}')
            
            # 初始化
            self.fusion_mapping = {}
            self.waypoint_contributions = {}
            
            canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.float32)
            weight_map = np.zeros((canvas_height, canvas_width), dtype=np.float32)
            
            canvas_center_x = canvas_width // 2
            canvas_center_y = canvas_height // 2
            
            # 🆕 改进的融合策略
            for i, wp_info in enumerate(waypoint_positions):
                img_data = wp_info['data']
                image = img_data['color_image'].astype(np.float32)
                world_x, world_y, world_z = img_data['world_pos']
                waypoint_idx = img_data['waypoint_index']
                
                # 世界坐标偏移
                offset_x = world_x - scan_center_x
                offset_y = world_y - scan_center_y
                
                # 旋转变换
                rotated_x = offset_x * math.cos(-yaw_rad) - offset_y * math.sin(-yaw_rad)
                rotated_y = offset_x * math.sin(-yaw_rad) + offset_y * math.cos(-yaw_rad)
                
                # 画布位置
                canvas_pos_x = (canvas_center_x + 
                            int(rotated_x * fov_params['pixels_per_mm_x'] * dual_axis_params['scale_adjust_x']))
                canvas_pos_y = (canvas_center_y + 
                            int(rotated_y * fov_params['pixels_per_mm_y'] * dual_axis_params['scale_adjust_y']))
                
                # 🆕 改进的融合掩码 - 更激进的边缘羽化
                improved_mask = self._create_improved_mask(
                    image.shape[:2], 
                    dual_axis_params['blend_size_x'] * 1.5,  # 增大融合区域
                    dual_axis_params['blend_size_y'] * 1.5,
                    waypoint_idx,
                    len(waypoint_positions)
                )
                
                # 🆕 检查重叠区域
                overlap_info = self._check_overlap_area(canvas, weight_map, image.shape[:2], 
                                                    canvas_pos_x, canvas_pos_y)
                
                self.get_logger().info(f'🔍 Waypoint {waypoint_idx}:')
                self.get_logger().info(f'  位置: ({canvas_pos_x}, {canvas_pos_y})')
                self.get_logger().info(f'  重叠像素: {overlap_info["overlap_pixels"]}')
                self.get_logger().info(f'  重叠率: {overlap_info["overlap_ratio"]:.1%}')
                
                # 融合到画布
                self._advanced_blend_to_canvas(canvas, weight_map, image, improved_mask, 
                                            canvas_pos_x, canvas_pos_y, waypoint_idx)
                
                # 🆕 保存调试图像
                if i < 2:  # 只保存前两个waypoint的调试信息
                    self._save_debug_image(image, improved_mask, waypoint_idx, 
                                        getattr(self, 'current_output_dir', '/tmp'))
            
            # 🆕 改进的归一化
            result = self._improved_normalize_canvas(canvas, weight_map)
            
            # 保存融合信息
            self._fusion_info = {
                'scan_center': (scan_center_x, scan_center_y),
                'canvas_center': (canvas_center_x, canvas_center_y),
                'yaw_rad': yaw_rad,
                'dual_axis_params': dual_axis_params
            }
            
            self.get_logger().info('✅ 调试版双轴融合完成')
            return result
            
        except Exception as e:
            self.get_logger().error(f'调试版融合失败: {e}')
            import traceback
            traceback.print_exc()
            return None

    def _debug_waypoint_overlap(self, image_data_list, fov_params):
        """调试waypoint重叠情况"""
        try:
            self.get_logger().info('🔍 分析Waypoint重叠情况:')
            
            positions = [(img['world_pos'][0], img['world_pos'][1], img['waypoint_index']) 
                        for img in image_data_list]
            
            # 计算理论FOV覆盖尺寸
            fov_width_mm = fov_params['fov_width_mm']
            fov_height_mm = fov_params['fov_height_mm']
            
            self.get_logger().info(f'  理论FOV尺寸: {fov_width_mm:.1f} × {fov_height_mm:.1f} mm')
            
            # 分析相邻waypoint距离
            for i in range(len(positions) - 1):
                curr_pos = positions[i]
                next_pos = positions[i + 1]
                
                dx = abs(next_pos[0] - curr_pos[0])
                dy = abs(next_pos[1] - curr_pos[1])
                distance = math.sqrt(dx*dx + dy*dy)
                
                # 计算重叠率
                if dx < fov_height_mm:  # X方向重叠
                    x_overlap = (fov_height_mm - dx) / fov_height_mm
                else:
                    x_overlap = 0
                    
                if dy < fov_width_mm:  # Y方向重叠
                    y_overlap = (fov_width_mm - dy) / fov_width_mm
                else:
                    y_overlap = 0
                
                self.get_logger().info(f'  Waypoint {curr_pos[2]} -> {next_pos[2]}:')
                self.get_logger().info(f'    距离: {distance:.1f}mm (dx={dx:.1f}, dy={dy:.1f})')
                self.get_logger().info(f'    重叠率: X={x_overlap:.1%}, Y={y_overlap:.1%}')
                
        except Exception as e:
            self.get_logger().error(f'重叠分析失败: {e}')

    def _create_improved_mask(self, image_shape, blend_size_x, blend_size_y, waypoint_idx, total_waypoints):
        """创建改进的融合掩码"""
        h, w = image_shape
        mask = np.ones((h, w), dtype=np.float32)
        
        # 🆕 根据waypoint位置调整掩码策略
        is_edge_waypoint = (waypoint_idx == 0 or waypoint_idx == total_waypoints - 1)
        
        # 更平滑的羽化函数
        def smooth_falloff(distance, blend_size):
            if distance >= blend_size:
                return 1.0
            # 使用更平滑的三次函数
            t = distance / blend_size
            return 3 * t * t - 2 * t * t * t
        
        # X轴方向融合（左右边缘）
        for x in range(w):
            dist_from_left = x
            dist_from_right = w - 1 - x
            min_dist = min(dist_from_left, dist_from_right)
            
            if min_dist < blend_size_x:
                weight = smooth_falloff(min_dist, blend_size_x)
                mask[:, x] *= weight
        
        # Y轴方向融合（上下边缘）
        for y in range(h):
            dist_from_top = y
            dist_from_bottom = h - 1 - y
            min_dist = min(dist_from_top, dist_from_bottom)
            
            if min_dist < blend_size_y:
                weight = smooth_falloff(min_dist, blend_size_y)
                mask[y, :] *= weight
        
        return mask
    
    
    def _analyze_waypoint_layout(self, image_data_list):
        """🆕 分析waypoint的空间布局，确定拼接策略"""
        try:
            positions = [(img['world_pos'][0], img['world_pos'][1], img['waypoint_index']) 
                        for img in image_data_list]
            
            if len(positions) < 2:
                return "single_point"
            
            # 按waypoint索引排序，分析路径模式
            positions_by_index = sorted(positions, key=lambda p: p[2])
            
            # 计算相对位置关系
            x_coords = [p[0] for p in positions]
            y_coords = [p[1] for p in positions]
            
            x_range = max(x_coords) - min(x_coords)
            y_range = max(y_coords) - min(y_coords)
            
            self.get_logger().info(f'📊 Waypoint分布分析:')
            self.get_logger().info(f'  X方向跨度: {x_range:.1f}mm')
            self.get_logger().info(f'  Y方向跨度: {y_range:.1f}mm')
            
            # 🆕 检测蛇形路径特征
            if len(positions) >= 4:
                # 分析相邻waypoint之间的方向变化
                direction_changes = 0
                prev_direction = None
                
                for i in range(1, len(positions_by_index)):
                    curr_pos = positions_by_index[i]
                    prev_pos = positions_by_index[i-1]
                    
                    dx = curr_pos[0] - prev_pos[0]
                    dy = curr_pos[1] - prev_pos[1]
                    
                    # 判断主要移动方向
                    if abs(dx) > abs(dy):
                        direction = "horizontal"
                    else:
                        direction = "vertical"
                    
                    if prev_direction and prev_direction != direction:
                        direction_changes += 1
                    prev_direction = direction
                
                # 如果有多次方向变化，可能是蛇形路径
                if direction_changes >= 1:
                    layout = "serpentine_pattern"
                    self.get_logger().info(f'  检测到 {direction_changes} 次方向变化，识别为蛇形路径')
                else:
                    layout = "grid_pattern"
            else:
                # 少量waypoint，判断主要排列方向
                if x_range > y_range * 1.5:
                    layout = "horizontal_line"
                elif y_range > x_range * 1.5:
                    layout = "vertical_line"
                else:
                    layout = "grid_pattern"
            
            self.get_logger().info(f'  布局模式: {layout}')
            
            # 输出每个waypoint的详细位置（按索引顺序）
            for x, y, idx in positions_by_index:
                self.get_logger().info(f'  Waypoint {idx}: ({x:.1f}, {y:.1f})')
            
            return layout
            
        except Exception as e:
            self.get_logger().error(f'分析waypoint布局失败: {e}')
            return "unknown"
    def _create_dual_axis_mask(self, image_shape, blend_size_x, blend_size_y):
        """创建双轴融合掩码"""
        h, w = image_shape
        mask = np.ones((h, w), dtype=np.float32)
        
        # X轴方向融合（水平边缘）
        for x in range(w):
            dist_from_x_edge = min(x, w - 1 - x)
            if dist_from_x_edge < blend_size_x:
                weight_x = dist_from_x_edge / blend_size_x
                mask[:, x] *= weight_x
        
        # Y轴方向融合（垂直边缘）
        for y in range(h):
            dist_from_y_edge = min(y, h - 1 - y)
            if dist_from_y_edge < blend_size_y:
                weight_y = dist_from_y_edge / blend_size_y
                mask[y, :] *= weight_y
        
        return mask

    def _blend_to_canvas(self, canvas, weight_map, image, mask, pos_x, pos_y):
        """融合图像到画布"""
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
                
                # 验证尺寸匹配
                if img_region.shape[:2] == mask_region.shape:
                    # 融合
                    for c in range(3):
                        canvas[canvas_top:canvas_bottom, canvas_left:canvas_right, c] += img_region[:, :, c] * mask_region
                    
                    weight_map[canvas_top:canvas_bottom, canvas_left:canvas_right] += mask_region
                
        except Exception as e:
            self.get_logger().error(f'画布融合失败: {e}')

    def _normalize_canvas(self, canvas, weight_map):
        """归一化画布"""
        valid_mask = weight_map > 0
        result = np.zeros_like(canvas, dtype=np.uint8)
        
        for c in range(3):
            channel = canvas[:, :, c].copy()
            channel[valid_mask] /= weight_map[valid_mask]
            result[:, :, c] = np.clip(channel, 0, 255).astype(np.uint8)
        
        return result

    def _process_single_point(self, image_data, scan_plan, output_dir):
        """单点模式：旋转矫正 + 裁剪 + 映射信息保存"""
        try:
            self.get_logger().info('处理单点扫描')
            
            image = image_data['color_image']
            yaw_deg = image_data['yaw']
            
            # 简化裁剪
            cropped_image = self._crop_single_point(image, scan_plan)
            
            # 🆕 为单点扫描创建映射信息
            self._create_single_point_mapping(image_data, cropped_image, output_dir)
            
            # 保存结果
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = os.path.join(output_dir, f"single_point_result_{timestamp}.jpg")
            
            result_bgr = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, result_bgr)
            
            self.get_logger().info(f'单点处理完成: {output_path}')
            
            # 发布参考数据
            self._publish_reference_data(cropped_image, None, output_dir)
            return output_path, "single_point_corrected"
            
        except Exception as e:
            self.get_logger().error(f'单点处理失败: {e}')
            return None, "failed"

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

    def _crop_to_scan_area_dual_axis(self, stitched_image, scan_plan, fov_params, dual_axis_params):
        """双轴优化的scan area裁剪"""
        try:
            if not hasattr(self, '_fusion_info'):
                self.get_logger().warn('缺少融合信息，使用简化裁剪')
                return self._crop_single_point(stitched_image, scan_plan)
            
            fusion_info = self._fusion_info
            
            # scan area边界
            scan_region = scan_plan.scan_region
            region_x = [p.x for p in scan_region]
            region_y = [p.y for p in scan_region]
            
            scan_x_min, scan_x_max = min(region_x), max(region_x)
            scan_y_min, scan_y_max = min(region_y), max(region_y)
            
            # 融合信息
            scan_center_x, scan_center_y = fusion_info['scan_center']
            canvas_center_x, canvas_center_y = fusion_info['canvas_center']
            yaw_rad = fusion_info['yaw_rad']
            
            # 计算scan area边界在画布中的位置
            corners_world = [
                (scan_x_min - scan_center_x, scan_y_min - scan_center_y),
                (scan_x_max - scan_center_x, scan_y_min - scan_center_y),
                (scan_x_max - scan_center_x, scan_y_max - scan_center_y),
                (scan_x_min - scan_center_x, scan_y_max - scan_center_y)
            ]
            
            # 旋转并转换为画布坐标
            corners_canvas = []
            for offset_x, offset_y in corners_world:
                rotated_x = offset_x * math.cos(-yaw_rad) - offset_y * math.sin(-yaw_rad)
                rotated_y = offset_x * math.sin(-yaw_rad) + offset_y * math.cos(-yaw_rad)
                
                canvas_x = canvas_center_x + int(rotated_y * fov_params['pixels_per_mm_x'] * dual_axis_params['scale_adjust_x'])
                canvas_y = canvas_center_y + int(rotated_x * fov_params['pixels_per_mm_y'] * dual_axis_params['scale_adjust_y'])
                
                corners_canvas.append((canvas_x, canvas_y))
            
            # 计算边界框
            canvas_xs = [p[0] for p in corners_canvas]
            canvas_ys = [p[1] for p in corners_canvas]
            
            crop_left = max(0, min(canvas_xs))
            crop_right = min(stitched_image.shape[1], max(canvas_xs))
            crop_top = max(0, min(canvas_ys))
            crop_bottom = min(stitched_image.shape[0], max(canvas_ys))
            
            # 执行裁剪
            if crop_right > crop_left and crop_bottom > crop_top:
                cropped = stitched_image[crop_top:crop_bottom, crop_left:crop_right]
                self.get_logger().info(f'双轴裁剪: {stitched_image.shape[1]}×{stitched_image.shape[0]} → {cropped.shape[1]}×{cropped.shape[0]}')
                return cropped
            else:
                self.get_logger().warn('裁剪区域无效，返回原图')
                return stitched_image
            
        except Exception as e:
            self.get_logger().error(f'双轴裁剪失败: {e}')
            return stitched_image

    def _crop_single_point(self, image, scan_plan):
        """单点简化裁剪"""
        h, w = image.shape[:2]
        crop_ratio = 0.8
        margin_w = int(w * (1 - crop_ratio) / 2)
        margin_h = int(h * (1 - crop_ratio) / 2)
        return image[margin_h:h-margin_h, margin_w:w-margin_w]

    def _rotate_image(self, image, angle_deg):
        """旋转图像"""
        if abs(angle_deg) < 1:
            return image
        
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
        
        cos_theta = abs(rotation_matrix[0, 0])
        sin_theta = abs(rotation_matrix[0, 1])
        new_w = int((h * sin_theta) + (w * cos_theta))
        new_h = int((h * cos_theta) + (w * sin_theta))
        
        rotation_matrix[0, 2] += (new_w / 2) - center[0]
        rotation_matrix[1, 2] += (new_h / 2) - center[1]
        
        return cv2.warpAffine(image, rotation_matrix, (new_w, new_h), 
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))


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
            
    def _record_waypoint_contribution(self, image, canvas_pos_x, canvas_pos_y, waypoint_idx, 
                                canvas_width, canvas_height, img_data):
        """记录waypoint对画布的贡献映射"""
        try:
            img_h, img_w = image.shape[:2]
            
            # 计算在画布上的实际放置区域
            img_left = canvas_pos_x - img_w // 2
            img_top = canvas_pos_y - img_h // 2
            img_right = img_left + img_w
            img_bottom = img_top + img_h
            
            # 画布有效区域
            canvas_left = max(0, img_left)
            canvas_top = max(0, img_top)
            canvas_right = min(canvas_width, img_right)
            canvas_bottom = min(canvas_height, img_bottom)
            
            # 图像对应区域
            img_crop_left = max(0, -img_left)
            img_crop_top = max(0, -img_top)
            
            # 为每个有效像素创建映射记录
            for canvas_y in range(canvas_top, canvas_bottom):
                for canvas_x in range(canvas_left, canvas_right):
                    # 计算在原图中的位置
                    img_x = img_crop_left + (canvas_x - canvas_left)
                    img_y = img_crop_top + (canvas_y - canvas_top)
                    
                    # 确保在原图范围内
                    if 0 <= img_x < img_w and 0 <= img_y < img_h:
                        # 🆕 增强的映射信息，包含完整位姿
                        mapping_info = {
                            'source_waypoint': waypoint_idx,
                            'source_pixel': (img_x, img_y),
                            'waypoint_data': {
                                'waypoint_index': waypoint_idx,
                                'world_pos': img_data.get('world_pos', (0, 0, 0)),  # 🆕 包含Z
                                'roll': img_data.get('roll', 0),                     # 🆕 Roll
                                'pitch': img_data.get('pitch', 0),                   # 🆕 Pitch
                                'yaw': img_data.get('yaw', 0),                       # 🆕 Yaw
                                'quaternion': img_data.get('quaternion', (0,0,0,1)), # 🆕 四元数
                                'color_filename': img_data.get('color_filename', ''),
                                'depth_raw_filename': img_data.get('depth_raw_filename', ''),
                                'full_data': img_data  # 保留完整数据
                            },
                            'transform_info': {
                                'canvas_pos': (canvas_pos_x, canvas_pos_y),
                                'world_pos': img_data.get('world_pos', (0, 0, 0)),
                                'full_pose': {
                                    'position': img_data.get('world_pos', (0, 0, 0)),
                                    'orientation': {
                                        'roll': img_data.get('roll', 0),
                                        'pitch': img_data.get('pitch', 0),
                                        'yaw': img_data.get('yaw', 0)
                                    }
                                }
                            }
                        }
                        
                        # 如果已有映射，选择权重更高的
                        if (canvas_x, canvas_y) in self.fusion_mapping:
                            # 可以根据距离图像中心的远近来判断权重
                            existing = self.fusion_mapping[(canvas_x, canvas_y)]
                            current_distance = math.sqrt((img_x - img_w//2)**2 + (img_y - img_h//2)**2)
                            existing_dist = existing.get('distance_from_center', float('inf'))
                            
                            if current_distance < existing_dist:
                                mapping_info['distance_from_center'] = current_distance
                                self.fusion_mapping[(canvas_x, canvas_y)] = mapping_info
                        else:
                            mapping_info['distance_from_center'] = math.sqrt((img_x - img_w//2)**2 + (img_y - img_h//2)**2)
                            self.fusion_mapping[(canvas_x, canvas_y)] = mapping_info
            
            if waypoint_idx not in self.waypoint_contributions:
                self.waypoint_contributions[waypoint_idx] = {
                    'pixel_count': 0,
                    'coverage_area': (canvas_left, canvas_top, canvas_right, canvas_bottom),
                    'waypoint_data': {
                        'waypoint_index': waypoint_idx,
                        'color_filename': img_data.get('color_filename', ''),
                        'depth_raw_filename': img_data.get('depth_raw_filename', ''),
                        'world_pos': img_data.get('world_pos', (0, 0, 0)),
                        'roll': img_data.get('roll', 0),
                        'pitch': img_data.get('pitch', 0),
                        'yaw': img_data.get('yaw', 0),
                        'quaternion': img_data.get('quaternion', (0,0,0,1))
                    }
                }
            
            contribution_pixels = (canvas_right - canvas_left) * (canvas_bottom - canvas_top)
            self.waypoint_contributions[waypoint_idx]['pixel_count'] += contribution_pixels
            
            self.get_logger().info(f'Waypoint {waypoint_idx} 贡献区域: {contribution_pixels} 像素')
            self.get_logger().info(f'  - 完整位姿: 位置{img_data.get("world_pos", (0,0,0))}, 姿态({img_data.get("roll", 0):.1f}°, {img_data.get("pitch", 0):.1f}°, {img_data.get("yaw", 0):.1f}°)')
            
        except Exception as e:
            self.get_logger().error(f'记录waypoint贡献失败: {e}')

    def _save_fusion_mapping(self, output_dir):
        """保存融合映射信息到文件"""
        try:
            import pickle
            import json
            
            # 保存完整映射信息（pickle格式，供程序读取）
            mapping_file = os.path.join(output_dir, "fusion_mapping.pkl")
            with open(mapping_file, 'wb') as f:
                pickle.dump({
                    'fusion_mapping': self.fusion_mapping,
                    'waypoint_contributions': self.waypoint_contributions,
                    'fusion_params': {
                        'dual_axis_params': getattr(self, '_fusion_info', {}).get('dual_axis_params', {}),
                        'canvas_size': (len(self.fusion_mapping) > 0 and 
                                    max([pos[0] for pos in self.fusion_mapping.keys()]) + 1,
                                    len(self.fusion_mapping) > 0 and 
                                    max([pos[1] for pos in self.fusion_mapping.keys()]) + 1) if self.fusion_mapping else (0, 0)
                    }
                }, f)
            
            # 保存可读的摘要信息（JSON格式）
            summary_file = os.path.join(output_dir, "fusion_mapping_summary.json")
            summary_data = {
                'total_mapped_pixels': len(self.fusion_mapping),
                'waypoint_contributions': {
                    str(wp_idx): {
                        'pixel_count': info['pixel_count'],
                        'coverage_percentage': info['pixel_count'] / len(self.fusion_mapping) * 100 if self.fusion_mapping else 0,
                        'waypoint_index': wp_idx,
                        'source_files': {
                            'color': info['waypoint_data'].get('color_filename', ''),
                            'depth': info['waypoint_data'].get('depth_raw_filename', '')
                        }
                    }
                    for wp_idx, info in self.waypoint_contributions.items()
                }
            }
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, indent=2, ensure_ascii=False)
            
            self.get_logger().info(f'融合映射信息已保存:')
            self.get_logger().info(f'  - 完整映射: {mapping_file}')
            self.get_logger().info(f'  - 摘要信息: {summary_file}')
            self.get_logger().info(f'  - 总映射像素: {len(self.fusion_mapping)}')
            
            # 输出waypoint贡献统计
            for wp_idx, info in self.waypoint_contributions.items():
                percentage = info['pixel_count'] / len(self.fusion_mapping) * 100 if self.fusion_mapping else 0
                self.get_logger().info(f'  - Waypoint {wp_idx}: {percentage:.1f}% ({info["pixel_count"]} 像素)')
            
        except Exception as e:
            self.get_logger().error(f'保存融合映射信息失败: {e}')
    def _dual_axis_fusion_stitching_grid_fix(self, image_data_list, scan_plan, fov_params, dual_axis_params):
        """专门针对2x2网格模式的拼接修复"""
        try:
            if not image_data_list:
                return None
            
            self.get_logger().info('🔧 开始2x2网格专用拼接...')
            
            # 获取yaw角度
            yaw_deg = image_data_list[0]['yaw']
            yaw_rad = math.radians(yaw_deg)
            
            # 🆕 重新解析waypoint为2x2网格
            grid_waypoints = self._parse_2x2_grid(image_data_list)
            
            # 计算scan area信息
            scan_region = scan_plan.scan_region
            region_x = [p.x for p in scan_region]
            region_y = [p.y for p in scan_region]
            scan_x_min, scan_x_max = min(region_x), max(region_x)
            scan_y_min, scan_y_max = min(region_y), max(region_y)
            scan_center_x = (scan_x_min + scan_x_max) / 2
            scan_center_y = (scan_y_min + scan_y_max) / 2
            
            self.get_logger().info(f'🎯 Scan中心: ({scan_center_x:.1f}, {scan_center_y:.1f})')
            
            # 🆕 计算实际的图像间距（基于重叠分析）
            actual_spacing_x, actual_spacing_y = self._calculate_actual_spacing(grid_waypoints, fov_params)
            
            # 🆕 创建优化的画布尺寸
            # 基于实际间距而不是理论值
            canvas_width = int(actual_spacing_x * 0.8 + self.camera_width)   # 1.2倍留余量
            canvas_height = int(actual_spacing_y * 0.8 + self.camera_height)
            
            self.get_logger().info(f'🎨 优化画布: {canvas_width}×{canvas_height}')
            self.get_logger().info(f'📏 实际间距: X={actual_spacing_x:.1f}px, Y={actual_spacing_y:.1f}px')
            
            # 创建画布
            canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.float32)
            weight_map = np.zeros((canvas_height, canvas_width), dtype=np.float32)
            
            canvas_center_x = canvas_width // 2
            canvas_center_y = canvas_height // 2
            
            # 🆕 按2x2网格模式精确放置
            for grid_pos, waypoint_data in grid_waypoints.items():
                if waypoint_data is None:
                    continue
                    
                image = waypoint_data['color_image'].astype(np.float32)
                waypoint_idx = waypoint_data['waypoint_index']
                
                # 🆕 网格位置到画布位置的直接映射
                grid_row, grid_col = grid_pos
                if not hasattr(self, 'fusion_mapping'):
                    self.fusion_mapping = {}
                if not hasattr(self, 'waypoint_contributions'):
                    self.waypoint_contributions = {}                
                # 计算画布上的精确位置
                canvas_pos_x = canvas_center_x - (grid_col - 0.5) * actual_spacing_x / 2
                canvas_pos_y = canvas_center_y + (grid_row - 0.5) * actual_spacing_y / 2
                
                canvas_pos_x = int(canvas_pos_x)
                canvas_pos_y = int(canvas_pos_y)
                
                # 🆕 创建网格专用掩码 - 减少重叠区域的权重
                grid_mask = self._create_grid_specific_mask(
                    image.shape[:2], 
                    grid_pos,
                    overlap_reduction_factor=0.7  # 在重叠区域降低权重
                )
                
                self.get_logger().info(f'🔍 网格位置({grid_row},{grid_col}) - Waypoint {waypoint_idx}:')
                self.get_logger().info(f'  画布位置: ({canvas_pos_x}, {canvas_pos_y})')
                
                # 融合到画布
                self._grid_blend_to_canvas(canvas, weight_map, image, grid_mask, 
                                        canvas_pos_x, canvas_pos_y, waypoint_idx)
            
            # 🆕 网格专用归一化
            result = self._grid_normalize_canvas(canvas, weight_map)
            
            self.get_logger().info('✅ 2x2网格拼接完成')
            return result
            
        except Exception as e:
            self.get_logger().error(f'2x2网格拼接失败: {e}')
            import traceback
            traceback.print_exc()
            return None


    def _parse_2x2_grid(self, image_data_list):
        """解析waypoint为2x2网格结构"""
        try:
            # 提取位置信息
            positions = []
            for img_data in image_data_list:
                world_x, world_y, world_z = img_data['world_pos']
                waypoint_idx = img_data['waypoint_index']
                positions.append({
                    'x': world_x, 'y': world_y, 'idx': waypoint_idx, 'data': img_data
                })
            
            # 按X和Y坐标排序以确定网格位置
            x_coords = sorted(set([p['x'] for p in positions]))
            y_coords = sorted(set([p['y'] for p in positions]))
            
            self.get_logger().info(f'🔍 检测到的网格结构:')
            self.get_logger().info(f'  X坐标: {[f"{x:.1f}" for x in x_coords]}')
            self.get_logger().info(f'  Y坐标: {[f"{y:.1f}" for y in y_coords]}')
            
            # 创建2x2网格映射
            grid_waypoints = {}
            
            for pos in positions:
                # 确定在网格中的位置
                grid_col = x_coords.index(pos['x']) if pos['x'] in x_coords else 0
                grid_row = y_coords.index(pos['y']) if pos['y'] in y_coords else 0
                
                # 调整为从0开始的索引
                if len(x_coords) > 1:
                    grid_col = 1 if pos['x'] > sum(x_coords)/len(x_coords) else 0
                if len(y_coords) > 1:
                    grid_row = 1 if pos['y'] > sum(y_coords)/len(y_coords) else 0
                
                grid_waypoints[(grid_row, grid_col)] = pos['data']
                
                self.get_logger().info(f'  Waypoint {pos["idx"]}: 网格位置({grid_row},{grid_col})')
            
            return grid_waypoints
            
        except Exception as e:
            self.get_logger().error(f'解析2x2网格失败: {e}')
            return {}


    def _calculate_actual_spacing(self, grid_waypoints, fov_params):
        """计算实际的图像间距"""
        try:
            # 理论FOV尺寸
            fov_width_mm = fov_params['fov_width_mm']
            fov_height_mm = fov_params['fov_height_mm']
            
            # 基于重叠率计算最优间距（大幅增加重叠）
            # 目标重叠率（大幅增加重叠以避免物体被分割）
            target_overlap_x = 0.50  # 50%重叠 - 确保物体不被分割
            target_overlap_y = 0.50  # 50%重叠 - 确保物体不被分割
            
            # 计算间距
            optimal_spacing_x = fov_width_mm * (1 - target_overlap_x)  # 注意坐标系对应
            optimal_spacing_y = fov_height_mm * (1 - target_overlap_y)
            
            # 转换为像素
            spacing_x_pixels = optimal_spacing_x * fov_params['pixels_per_mm_y']  # 注意对应关系
            spacing_y_pixels = optimal_spacing_y * fov_params['pixels_per_mm_x']
            
            self.get_logger().info(f'📐 最优间距计算:')
            self.get_logger().info(f'  物理间距: X={optimal_spacing_x:.1f}mm, Y={optimal_spacing_y:.1f}mm')
            self.get_logger().info(f'  像素间距: X={spacing_x_pixels:.1f}px, Y={spacing_y_pixels:.1f}px')
            
            return spacing_x_pixels, spacing_y_pixels
            
        except Exception as e:
            self.get_logger().error(f'计算实际间距失败: {e}')
            return 600, 400  # 默认值


    def _create_grid_specific_mask(self, image_shape, grid_pos, overlap_reduction_factor=0.3):
        """创建网格专用掩码，针对性处理边缘"""
        h, w = image_shape
        mask = np.ones((h, w), dtype=np.float32)
        
        grid_row, grid_col = grid_pos
        
        # 🆕 根据网格位置调整边缘处理策略
        # 边缘羽化尺寸 - 进一步增加羽化区域
        blend_size_x = w // 4  # 进一步增加羽化尺寸（从1/6改为1/4）
        blend_size_y = h // 4
        
        # 只对会重叠的边缘进行羽化
        # 左边缘（如果不是第一列）
        if grid_col > 0:
            for x in range(min(blend_size_x, w)):
                weight = (x / blend_size_x) * overlap_reduction_factor + (1 - overlap_reduction_factor)
                mask[:, x] *= weight
        
        # 右边缘（如果不是最后一列）
        if grid_col < 1:  # 2x2网格，最大列索引为1
            for x in range(max(0, w - blend_size_x), w):
                dist_from_edge = w - 1 - x
                weight = (dist_from_edge / blend_size_x) * overlap_reduction_factor + (1 - overlap_reduction_factor)
                mask[:, x] *= weight
        
        # 上边缘（如果不是第一行）
        if grid_row > 0:
            for y in range(min(blend_size_y, h)):
                weight = (y / blend_size_y) * overlap_reduction_factor + (1 - overlap_reduction_factor)
                mask[y, :] *= weight
        
        # 下边缘（如果不是最后一行）
        if grid_row < 1:  # 2x2网格，最大行索引为1
            for y in range(max(0, h - blend_size_y), h):
                dist_from_edge = h - 1 - y
                weight = (dist_from_edge / blend_size_y) * overlap_reduction_factor + (1 - overlap_reduction_factor)
                mask[y, :] *= weight
        
        return mask


    def _grid_blend_to_canvas(self, canvas, weight_map, image, mask, pos_x, pos_y, waypoint_idx):
        """网格专用融合函数"""
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
                    # 简化的融合：直接加权叠加
                    for c in range(3):
                        canvas[canvas_top:canvas_bottom, canvas_left:canvas_right, c] += (
                            img_region[:, :, c] * mask_region
                        )
                    
                    weight_map[canvas_top:canvas_bottom, canvas_left:canvas_right] += mask_region
                    
        except Exception as e:
            self.get_logger().error(f'网格融合失败: {e}')


    def _grid_normalize_canvas(self, canvas, weight_map):
        """网格专用归一化"""
        # 设置最小权重阈值
        min_weight = 0.1
        weight_map_safe = np.maximum(weight_map, min_weight)
        
        result = np.zeros_like(canvas, dtype=np.uint8)
        
        for c in range(3):
            normalized = canvas[:, :, c] / weight_map_safe
            # 轻微对比度调整
            normalized = np.clip(normalized * 1.02, 0, 255)
            result[:, :, c] = normalized.astype(np.uint8)
        
        return result

def main(args=None):
    rclpy.init(args=args)
    node = SmartStitcherNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()