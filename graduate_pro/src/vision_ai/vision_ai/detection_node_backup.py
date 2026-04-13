import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
import cv2
import numpy as np
import time
import os
import json
from datetime import datetime
from scipy.spatial.transform import Rotation as R
# 导入检测管道
from vision_ai.detection.detection_pipeline import DetectionPipeline
from .improved_coordinate_functions import HandEyeCalibratedCoordinateCalculator

class BypassCvBridgeDetectionNode(Node):
    def __init__(self):
        super().__init__('detection_node')
        
        # 检测管道
        self.detection_pipeline = None
        self._initialize_detection_pipeline()
        self.coordinate_calculator = HandEyeCalibratedCoordinateCalculator()
        # 状态管理
        self.processing_active = False
        self.latest_reference_image = None
        self.latest_depth_image = None
        self.current_scan_output_dir = None
        self.last_detection_result = None
        self.result_republish_timer = None
        # 订阅者 - 监听拼接完成消息
        self.stitching_complete_sub = self.create_subscription(
            String,
            '/stitching_complete',
            self.stitching_complete_callback,
            10
        )
        
        # 订阅者 - 接收最终参考图像
        self.reference_image_sub = self.create_subscription(
            Image,
            '/reference_image',
            self.reference_image_callback,
            10
        )
        
        # 订阅者 - 接收深度信息
        self.reference_depth_sub = self.create_subscription(
            Image,
            '/reference_depth',
            self.reference_depth_callback,
            10
        )
        
        # 发布者 - 发布检测结果
        self.detection_result_pub = self.create_publisher(
            String,
            '/detection_result',
            10
        )
        
        # 发布者 - 发布检测结果
        self.detection_complete_pub = self.create_publisher(
            String,
            '/detection_complete',
            10
        )
        # 发布者 - 发布可视化图像
        self.visualization_pub = self.create_publisher(
            Image,
            '/detection_visualization',
            10
        )
        
        self.get_logger().info('🔍 Detection node (bypass cv_bridge) started, waiting for stitching complete signal...')

    def _initialize_detection_pipeline(self):
        """初始化检测管道"""
        try:
            self.detection_pipeline = DetectionPipeline(config_file=None)
            self.get_logger().info('✅ 检测管道初始化成功')
        except Exception as e:
            self.get_logger().error(f'❌ 检测管道初始化失败: {e}')
            self.detection_pipeline = None

    def stitching_complete_callback(self, msg):
        """Stitching complete callback"""
        try:
            self.current_scan_output_dir = msg.data
            self.get_logger().info(f'🎯 Received stitching complete signal: {self.current_scan_output_dir}')
            
            # Load images directly, bypass ROS message transfer
            self._load_images_directly()
            
        except Exception as e:
            self.get_logger().error(f'Failed to process stitching complete signal: {e}')

    def reference_image_callback(self, msg):
        """接收最终参考图像（备用方法）"""
        self.latest_reference_image = msg
        self.get_logger().info('📸 收到参考图像（备用）')

    def reference_depth_callback(self, msg):
        """接收深度信息（备用方法）"""
        self.latest_depth_image = msg
        self.get_logger().info('📏 收到深度信息（备用）')

    def _load_images_directly(self):
        """直接从文件加载图像，绕过ROS消息"""
        try:
            if not os.path.exists(self.current_scan_output_dir):
                self.get_logger().error(f'扫描目录不存在: {self.current_scan_output_dir}')
                return
            
            # 🆕 加载和测试融合映射信息
            self.get_logger().info('🔄 开始加载融合映射信息...')
            self.fusion_mapping_data = self._load_fusion_mapping()
            
            if self.fusion_mapping_data:
                self.get_logger().info('✅ 映射信息加载和验证完成')
            else:
                self.get_logger().warn('⚠️ 映射信息加载失败，将使用简化模式')
            
            # 查找融合后的最终彩色图像
            color_files = []
            for file in os.listdir(self.current_scan_output_dir):
                if file.startswith('final_') and file.endswith('.jpg'):
                    color_files.insert(0, file)  # 优先使用融合结果
                elif file.startswith('color_waypoint_') and file.endswith('.jpg'):
                    color_files.append(file)
            
            if not color_files:
                self.get_logger().error('没有找到彩色图像文件')
                return
            
            # 加载最终融合图像
            color_file = os.path.join(self.current_scan_output_dir, color_files[0])
            image_bgr = cv2.imread(color_file)
            if image_bgr is None:
                self.get_logger().error(f'无法加载图像: {color_file}')
                return
            
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            self.get_logger().info(f'融合图像加载成功: {image_rgb.shape}')
            
            # 🆕 如果有映射信息，测试一下映射功能
            if self.fusion_mapping_data:
                self._test_detection_mapping_simulation(image_rgb)
            
            # 为深度映射创建虚拟深度数据（用于接口兼容）
            placeholder_depth = np.ones((image_rgb.shape[0], image_rgb.shape[1]), dtype=np.uint16) * 1000
            
            # 处理检测（暂时使用原有流程）
            self._process_detection_with_mapping(image_rgb, placeholder_depth)  # 这个方法有深度增强
            
        except Exception as e:
            self.get_logger().error(f'直接加载图像失败: {e}')
            import traceback
            traceback.print_exc()

    def _test_detection_mapping_simulation(self, image_rgb):
        """模拟检测结果的映射测试"""
        try:
            self.get_logger().info('🧪 开始模拟检测映射测试...')
            
            # 创建一个模拟的检测框（图像中心区域）
            h, w = image_rgb.shape[:2]
            test_bbox = [w//4, h//4, 3*w//4, 3*h//4]  # 中心区域的框
            
            self.get_logger().info(f'📦 测试检测框: {test_bbox}')
            
            # 分析这个区域的waypoint来源
            source_analysis = self._analyze_region_sources(test_bbox)
            
            if source_analysis:
                self.get_logger().info(f'✅ 模拟映射分析成功:')
                self.get_logger().info(f'  - 主要来源: Waypoint {source_analysis["dominant_waypoint"]}')
                self.get_logger().info(f'  - 覆盖率: {source_analysis["coverage_ratio"]:.1%}')
                self.get_logger().info(f'  - 阈值达成: {source_analysis["coverage_threshold_met"]}')
            else:
                self.get_logger().warn('⚠️ 模拟映射分析失败')
                
        except Exception as e:
            self.get_logger().error(f'模拟映射测试失败: {e}')

    def _analyze_region_sources(self, bbox):
        """分析指定区域的waypoint来源分布（模拟版本）"""
        try:
            if not self.fusion_mapping_data:
                return None
            
            x1, y1, x2, y2 = map(int, bbox)
            fusion_mapping = self.fusion_mapping_data['fusion_mapping']
            
            # 统计区域中每个waypoint的像素贡献
            waypoint_pixels = {}
            total_pixels = 0
            
            for y in range(y1, y2, 5):  # 每隔5个像素采样，减少计算量
                for x in range(x1, x2, 5):
                    total_pixels += 1
                    
                    # 查找像素来源
                    mapping_info = fusion_mapping.get((x, y))
                    if mapping_info:
                        wp_idx = mapping_info['source_waypoint']
                        waypoint_pixels[wp_idx] = waypoint_pixels.get(wp_idx, 0) + 1
            
            # 找到主要来源waypoint
            if waypoint_pixels:
                dominant_waypoint = max(waypoint_pixels.items(), key=lambda x: x[1])
                dominant_wp_idx = dominant_waypoint[0]
                dominant_pixel_count = dominant_waypoint[1]
                coverage_ratio = dominant_pixel_count / total_pixels if total_pixels > 0 else 0
            else:
                dominant_wp_idx = None
                coverage_ratio = 0
            
            return {
                'dominant_waypoint': dominant_wp_idx,
                'coverage_ratio': coverage_ratio,
                'waypoint_distribution': waypoint_pixels,
                'total_pixels': total_pixels,
                'coverage_threshold_met': coverage_ratio >= 0.7
            }
            
        except Exception as e:
            self.get_logger().error(f'分析区域来源失败: {e}')
            return None
    
    def _process_detection_with_mapping(self, image_rgb, placeholder_depth):
        """使用映射信息处理检测"""
        try:
            if not self.detection_pipeline:
                self.get_logger().error('检测管道未初始化')
                return
            
            self.processing_active = True
            self.get_logger().info('🚀 开始基于映射的检测处理...')
            
            # 更新检测管道的输出目录
            detection_output_dir = os.path.join(self.current_scan_output_dir, "detection_results")
            os.makedirs(detection_output_dir, exist_ok=True)
            self.detection_pipeline.output_dir = detection_output_dir
            
            # 执行检测（先获取基本检测结果）
            result = self.detection_pipeline.process_reference_image(
                image_rgb, 
                placeholder_depth,  # 传入占位符深度
                generate_visualization=True,
                auto_display=True
            )
            
            if result['success']:
                # 🆕 为每个检测结果增强深度信息
                enhanced_objects = []
                for i, obj in enumerate(result['objects']):
                    self.get_logger().info(f'🧬 开始增强对象 {i+1}/{len(result["objects"])}: {obj["object_id"]}')
                    enhanced_obj = self._enhance_object_with_depth_mapping(obj)
                    enhanced_objects.append(enhanced_obj)
                
                # 更新结果
                result['objects'] = enhanced_objects
                
                self.get_logger().info(f'✅ 检测完成，发现 {result["detection_count"]} 个目标')
                self.get_logger().info(f'🧬 深度信息增强完成')
                
                # 发布增强后的检测结果
                self._publish_detection_result_json(result)
                self._publish_visualization_from_file(detection_output_dir)
                self._display_detection_results(result)
                self._start_target_selection(result)
                
            else:
                self.get_logger().error(f'❌ 检测失败: {result.get("message", "未知错误")}')
            
        except Exception as e:
            self.get_logger().error(f'基于映射的检测处理失败: {e}')
            import traceback
            traceback.print_exc()
        finally:
            self.processing_active = False

    def _enhance_object_with_depth_mapping(self, detection_obj):
        """使用映射信息增强单个检测对象的深度信息"""
        try:
            if self.fusion_mapping_data is None:
                self.get_logger().warn('无融合映射数据，跳过深度增强')
                return detection_obj
            
            bbox = detection_obj['bounding_box']
            mask = detection_obj.get('mask')
            
            if mask is None:
                self.get_logger().warn(f'对象 {detection_obj["object_id"]} 无mask，跳过深度增强')
                return detection_obj
            
            # 检查是否是单点扫描
            fusion_params = self.fusion_mapping_data.get('fusion_params', {})
            is_single_point = fusion_params.get('single_point', False)
            
            if is_single_point:
                # 🆕 单点扫描：获取waypoint数据和深度数据
                waypoint_contributions = self.fusion_mapping_data['waypoint_contributions']
                main_waypoint_idx = list(waypoint_contributions.keys())[0]
                waypoint_data = waypoint_contributions[main_waypoint_idx]['waypoint_data']
                
                # 加载深度数据
                depth_file = waypoint_data.get('depth_raw_filename')
                if not depth_file:
                    self.get_logger().warn(f'对象 {detection_obj["object_id"]} 无深度文件')
                    return detection_obj
                
                depth_full_path = os.path.join(self.current_scan_output_dir, os.path.basename(depth_file))
                if not os.path.exists(depth_full_path):
                    self.get_logger().warn(f'深度文件不存在: {depth_full_path}')
                    return detection_obj
                
                depth_data = np.load(depth_full_path)
                
                # 🆕 使用增强的高度计算（传入正确的参数）
                depth_result = self._calculate_object_height_corrected_sin(
                    mask, depth_data, bbox, waypoint_data
                )
                
                # 计算3D空间信息（包含抓夹宽度）
                spatial_3d = self._calculate_3d_spatial_features_sin(
                    mask, depth_data, waypoint_data, bbox
                )
                
            else:
                # 多点扫描：分析对象的来源waypoint分布
                source_analysis = self._analyze_object_sources(bbox, mask)
                
                if not source_analysis['coverage_threshold_met']:
                    self.get_logger().warn(f'对象 {detection_obj["object_id"]} 跨越多个waypoint，使用主要来源')
                
                dominant_wp_idx = source_analysis['dominant_waypoint']
                waypoint_data = self.fusion_mapping_data['waypoint_contributions'][dominant_wp_idx]['waypoint_data']
                
                # 加载该waypoint的深度数据
                depth_file = waypoint_data.get('depth_raw_filename')
                if not depth_file:
                    self.get_logger().warn(f'Waypoint {dominant_wp_idx} 无深度文件')
                    return detection_obj
                
                depth_full_path = os.path.join(self.current_scan_output_dir, os.path.basename(depth_file))
                if not os.path.exists(depth_full_path):
                    self.get_logger().warn(f'Waypoint {dominant_wp_idx} 深度文件不存在: {depth_full_path}')
                    return detection_obj
                
                depth_data = np.load(depth_full_path)
                
                # 🆕 对于多点扫描，需要将融合图像中的检测结果映射回原始图像坐标
                # 这里需要进行坐标映射（简化版本，假设尺寸匹配）
                depth_result = self._calculate_object_height_corrected_sin(
                    mask, depth_data, bbox, waypoint_data
                )
                
                spatial_3d = self._calculate_3d_spatial_features_sin(
                    mask, depth_data, waypoint_data, bbox
                )
            
            # 🆕 应用增强信息到检测对象
            if depth_result:
                # 更新空间特征
                if 'spatial' not in detection_obj['features']:
                    detection_obj['features']['spatial'] = {}
                
                detection_obj['features']['spatial'].update(spatial_3d)
                
                # 🆕 更新深度信息，包含背景深度和抓夹宽度
                detection_obj['features']['depth_info'] = {
                    'height_mm': depth_result['height_mm'],
                    'background_world_z': depth_result['background_world_z'],
                    'background_depth_m': depth_result['background_depth_m'],
                    'depth_confidence': depth_result['confidence'],
                    'source_waypoint': main_waypoint_idx if is_single_point else dominant_wp_idx,
                    'coverage_ratio': 1.0 if is_single_point else source_analysis['coverage_ratio'],
                    'scan_type': 'single_point' if is_single_point else 'multi_point'
                }
                
                # 🆕 添加抓夹宽度信息
                gripper_info = spatial_3d.get('gripper_width_info')
                if gripper_info:
                    detection_obj['features']['grasp_info'] = {
                        'recommended_width': gripper_info['recommended_gripper_width'],
                        'real_width_mm': gripper_info['real_width_mm'],
                        'grasp_angle': gripper_info['angle'],
                        'pixel_width': gripper_info['pixel_width'],
                        'depth_used': gripper_info['depth_used']
                    }
                
                # 更新描述
                detection_obj['description'] = self._generate_enhanced_description(
                    detection_obj, {'depth_analysis': depth_result, 'spatial_features': spatial_3d}
                )
                
                self.get_logger().info(
                    f'✅ 对象 {detection_obj["object_id"]} 深度增强完成: '
                    f'高度={depth_result["height_mm"]:.1f}mm, '
                    f'背景Z={depth_result["background_world_z"]:.1f}mm'
                )
                
                # 如果有抓夹信息，也记录一下
                if gripper_info:
                    self.get_logger().info(
                        f'🤏 抓夹信息: 推荐宽度={gripper_info["recommended_gripper_width"]}mm, '
                        f'实际宽度={gripper_info["real_width_mm"]:.1f}mm'
                    )
            else:
                self.get_logger().warn(f'⚠️ 对象 {detection_obj["object_id"]} 深度增强失败')
            
            return detection_obj
            
        except Exception as e:
            self.get_logger().error(f'对象深度增强失败: {e}')
            import traceback
            traceback.print_exc()
            return detection_obj


    def _get_depth_info_single_point(self, detection_obj):
        """为单点扫描获取深度信息"""
        try:
            waypoint_contributions = self.fusion_mapping_data['waypoint_contributions']
            
            # 单点扫描只有一个waypoint
            main_waypoint_idx = list(waypoint_contributions.keys())[0]
            waypoint_data = waypoint_contributions[main_waypoint_idx]['waypoint_data']
            
            self.get_logger().info(f'🔍 单点扫描，使用Waypoint {main_waypoint_idx}')
            
            # 加载深度数据
            depth_file = waypoint_data.get('depth_raw_filename')
            if not depth_file:
                return None
            
            depth_full_path = os.path.join(self.current_scan_output_dir, os.path.basename(depth_file))
            if not os.path.exists(depth_full_path):
                self.get_logger().warn(f'深度文件不存在: {depth_full_path}')
                return None
            
            depth_data = np.load(depth_full_path)
            
            # 对于单点扫描，检测结果的坐标直接对应原始图像坐标
            bbox = detection_obj['bounding_box']
            mask = detection_obj['mask']
            
            # 🆕 使用你的三种高度计算方法
            height_mm = self._calculate_object_height_corrected_sin(
                mask, depth_data, bbox, waypoint_data
            )
            
            # 计算3D空间信息
            spatial_3d = self._calculate_3d_spatial_features_sin(
                mask, depth_data, waypoint_data, bbox
            )
            
            return {
                'depth_analysis': {
                    'height_mm': height_mm,
                    'source_waypoint': main_waypoint_idx,
                    'coverage_ratio': 1.0,  # 单点扫描100%覆盖
                    'depth_confidence': 1.0,
                    'scan_type': 'single_point'
                },
                'spatial_features': spatial_3d
            }
            
        except Exception as e:
            self.get_logger().error(f'单点深度信息获取失败: {e}')
            return None

    def _calculate_object_height_corrected(self, mask, depth_data, bbox, waypoint_data):
        """集成三种高度估计方法，使用融合映射回原图坐标"""
        try:
            fusion_mapping = self.fusion_mapping_data['fusion_mapping']
            x1, y1, x2, y2 = map(int, bbox)

            # 保证 bbox 有效范围
            h, w = mask.shape
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            self.get_logger().info(f'🔍 计算高度: bbox=[{x1},{y1},{x2},{y2}], center=[{center_x},{center_y}]')

            def _get_depth_from_mapping(x, y):
                mapping = fusion_mapping.get((x, y))
                if not mapping:
                    return None
                src_x, src_y = mapping['source_pixel']
                depth_file = mapping['waypoint_data'].get('depth_raw_filename')
                if not os.path.exists(depth_file):
                    return None
                depth_img = np.load(depth_file)
                if 0 <= src_x < depth_img.shape[1] and 0 <= src_y < depth_img.shape[0]:
                    depth_val = depth_img[src_y, src_x] / 1000.0  # mm → m
                    return depth_val if depth_val > 0.01 else None
                return None

            # 方法1: bbox对比
            method1_depths_mask, method1_depths_bg = [], []
            for y in range(y1, y2):
                for x in range(x1, x2):
                    if 0 <= y < h and 0 <= x < w:
                        depth_val = _get_depth_from_mapping(x, y)
                        if depth_val is None:
                            continue
                        if mask[y, x] > 0:
                            method1_depths_mask.append(depth_val)
                        else:
                            method1_depths_bg.append(depth_val)

            # 方法2: 扩展框外背景
            expand_px = 30
            method2_depths_bg = []
            for y in range(y1 - expand_px, y2 + expand_px):
                for x in range(x1 - expand_px, x2 + expand_px):
                    if not (x1 <= x < x2 and y1 <= y < y2):
                        depth_val = _get_depth_from_mapping(x, y)
                        if depth_val:
                            method2_depths_bg.append(depth_val)

            # 方法3: 中心点对比
            center_depth = _get_depth_from_mapping(center_x, center_y)
            center_bg_samples = []
            for dy in range(-50, 51, 5):
                for dx in range(-50, 51, 5):
                    sx, sy = center_x + dx, center_y + dy
                    if 0 <= sx < w and 0 <= sy < h and mask[sy, sx] == 0:
                        depth_val = _get_depth_from_mapping(sx, sy)
                        if depth_val:
                            center_bg_samples.append(depth_val)

            # 计算各自方法高度
            results = []
            if len(method1_depths_mask) >= 10 and len(method1_depths_bg) >= 5:
                h1 = abs(np.median(method1_depths_mask) - np.median(method1_depths_bg)) * 1000
                if 5 <= h1 <= 500:
                    results.append(("bbox内对比", h1))

            if len(method2_depths_bg) >= 20 and len(method1_depths_mask) >= 10:
                h2 = abs(np.median(method1_depths_mask) - np.median(method2_depths_bg)) * 1000
                if 5 <= h2 <= 500:
                    results.append(("扩展背景", h2))

            if center_depth and len(center_bg_samples) >= 10:
                h3 = abs(center_depth - np.median(center_bg_samples)) * 1000
                if 5 <= h3 <= 500:
                    results.append(("中心点对比", h3))

            self.get_logger().info(f'📊 高度计算结果: {len(results)} 个有效值')
            for name, val in results:
                self.get_logger().info(f'  - {name}: {val:.1f}mm')

            if not results:
                self.get_logger().warn('所有高度计算方法都失败，使用默认值')
                return 30.0

            heights = [r[1] for r in results]
            if len(heights) == 1 or max(heights) - min(heights) < 20:
                final = np.mean(heights)
                self.get_logger().info(f'✅ 使用平均值: {final:.1f}mm')
                return final
            else:
                final = np.median(heights)
                self.get_logger().info(f'✅ 使用中位数: {final:.1f}mm')
                return final

        except Exception as e:
            self.get_logger().error(f'高度计算失败: {e}')
            return 30.0

    def _calculate_3d_spatial_features(self, mask, depth_data, waypoint_data, bbox):
        try:
            # 找到 mask 中心点
            mask_indices = np.argwhere(mask > 0)
            if len(mask_indices) == 0:
                return {}

            center_y, center_x = np.median(mask_indices, axis=0).astype(int)

            # 使用融合映射回查原图坐标
            fusion_mapping = self.fusion_mapping_data['fusion_mapping']
            mapping = fusion_mapping.get((center_x, center_y))
            if not mapping:
                self.get_logger().warn(f'无法从融合映射中找到中心点 ({center_x}, {center_y})')
                return {}

            src_x, src_y = mapping['source_pixel']
            depth_path = mapping['waypoint_data']['depth_raw_filename']
            depth_array = np.load(depth_path)
            depth_val = depth_array[src_y, src_x] / 1000.0
            if depth_val <= 0.01:
                return {}

            # 相机内参（如果你有动态配置可替换）
            fx, fy = 912.7, 910.3
            cx, cy = 624.051574707031, 320.749542236328
            z = depth_val * 1000
            x = (src_x - cx) * z / fx * 1000
            y = (src_y - cy) * z / fy * 1000

            # 世界坐标转换
            cam_pos = np.array(waypoint_data['world_pos'])  # (x, y, z)
            rpy_deg = [abs(waypoint_data['roll'])-180, waypoint_data['pitch'], waypoint_data['yaw']]
            cam_to_obj = np.array([(y + 70), (x - 40), -(z)])  # 轴调换+偏移
            
            R_wc = R.from_euler('XYZ', rpy_deg, degrees=True).as_matrix()
            world_xyz = R_wc @ cam_to_obj + cam_pos
            self.get_logger().info(f'3D: {world_xyz}')
            # 面积等
            bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            mask_area = np.sum(mask > 0)
            coverage_ratio = mask_area / bbox_area if bbox_area > 0 else 0

            return {
                'centroid_3d_camera': (x, y, z),
                'world_coordinates': tuple(world_xyz),
                'distance_to_camera': z,
                'mask_area_pixels': int(mask_area),
                'bbox_area_pixels': int(bbox_area),
                'mask_coverage_ratio': float(coverage_ratio),
                'estimated_real_area_mm2': float(mask_area * (z * 1000) ** 2 / 1000000),
            }

        except Exception as e:
            self.get_logger().error(f'3D空间特征计算失败: {e}')
            return {}

    def _calculate_3d_spatial_features_sin(self, mask, depth_data, waypoint_data, bbox):
        """使用官方标定计算夹爪目标坐标"""
        try:
            # 使用新的校准计算器
            from .improved_coordinate_functions import calculate_3d_spatial_features_calibrated
            
            result = calculate_3d_spatial_features_calibrated(
                mask, depth_data, waypoint_data, bbox, self.coordinate_calculator
            )
            
            return result
            
        except Exception as e:
            print(f"[ERROR] 校准坐标计算异常: {e}")
        
    
    def _calculate_object_height_corrected_sin(self, mask, depth_data, bbox, waypoint_data):
        """集成高度计算方法并保存背景深度信息（使用校准）"""
        try:
            from .improved_coordinate_functions import calculate_object_height_with_background_depth
            
            result = calculate_object_height_with_background_depth(
                mask, depth_data, bbox, waypoint_data, self.coordinate_calculator
            )
            
            if result:
                return result
                
        except Exception as e:
            print(f"[ERROR] 校准高度计算异常: {e}")
            

    def _generate_enhanced_description(self, detection_obj, depth_info):
        """生成增强的描述（包含深度信息）"""
        try:
            # 基础信息
            class_name = detection_obj['class_name']
            confidence = detection_obj['confidence']
            
            # 深度信息
            depth_analysis = depth_info.get('depth_analysis', {})
            height_mm = depth_analysis.get('height_mm')
            spatial_features = depth_info.get('spatial_features', {})
            distance = spatial_features.get('distance_to_camera')
            
            # 构建增强描述
            description_parts = []
            
            # 颜色信息（如果有）
            color_info = detection_obj.get('features', {}).get('color', {})
            color_name = color_info.get('color_name', '')
            if color_name and color_name != 'unknown':
                description_parts.append(color_name.capitalize())
            
            # 基础对象名称
            description_parts.append(class_name)
            
            # 高度信息
            if height_mm is not None:
                if height_mm < 20:
                    description_parts.append("(flat)")
                elif height_mm < 80:
                    description_parts.append("(low)")
                elif height_mm < 120:
                    description_parts.append("(medium height)")
                else:
                    description_parts.append("(tall)")
            
            # 距离信息
            if distance is not None:
                if distance < 0.3:
                    description_parts.append("nearby")
                elif distance < 0.6:
                    description_parts.append("at medium distance")
                else:
                    description_parts.append("far")
            
            # 位置信息（原有的region_position）
            region_position = detection_obj.get('features', {}).get('spatial', {}).get('region_position')
            if region_position and region_position != 'unknown':
                region_desc = region_position.replace('-', ' ').replace('_', ' ')
                if region_desc == 'center':
                    region_desc = 'center area'
                elif 'center' not in region_desc:
                    region_desc = f"{region_desc} area"
                description_parts.append(f"in {region_desc}")
            
            # 置信度（如果较低）
            if confidence < 0.8:
                description_parts.append(f"(conf: {confidence:.2f})")
            
            enhanced_description = ' '.join(description_parts)
            
            # 添加详细的技术信息（用于调试）
            tech_info = []
            if height_mm is not None:
                tech_info.append(f"h={height_mm:.1f}mm")
            if distance is not None:
                tech_info.append(f"d={distance*1000:.0f}mm")
            
            if tech_info:
                enhanced_description += f" [{', '.join(tech_info)}]"
            
            return enhanced_description
            
        except Exception as e:
            self.get_logger().error(f'增强描述生成失败: {e}')
            return detection_obj.get('description', f"{detection_obj['class_name']}_{detection_obj.get('object_id', '')}")

    # 🆕 为多点扫描增加支持（如果需要的话）
    def _get_depth_info_from_waypoint(self, detection_obj, source_analysis):
        """从特定waypoint获取深度信息（多点扫描）"""
        try:
            if not source_analysis['coverage_threshold_met']:
                self.get_logger().warn(f'对象 {detection_obj["object_id"]} 跨越多个waypoint，使用主要来源')
            
            dominant_wp_idx = source_analysis['dominant_waypoint']
            waypoint_data = self.fusion_mapping_data['waypoint_contributions'][dominant_wp_idx]['waypoint_data']
            
            # 加载该waypoint的深度数据
            depth_file = waypoint_data.get('depth_raw_filename')
            if not depth_file:
                return None
            
            depth_full_path = os.path.join(self.current_scan_output_dir, os.path.basename(depth_file))
            if not os.path.exists(depth_full_path):
                self.get_logger().warn(f'Waypoint {dominant_wp_idx} 深度文件不存在: {depth_full_path}')
                return None
            
            depth_data = np.load(depth_full_path)
            
            # 对于多点扫描，需要将融合图像中的检测结果映射回原始图像坐标
            # 这里是简化版本，假设尺寸匹配
            bbox = detection_obj['bounding_box']
            mask = detection_obj['mask']
            
            # 🆕 使用三种高度计算方法
            height_mm = self._calculate_object_height_corrected(
                mask, depth_data, bbox, waypoint_data
            )
            
            # 计算3D空间信息
            spatial_3d = self._calculate_3d_spatial_features(
                mask, depth_data, waypoint_data, bbox
            )
            
            return {
                'depth_analysis': {
                    'height_mm': height_mm,
                    'source_waypoint': dominant_wp_idx,
                    'coverage_ratio': source_analysis['coverage_ratio'],
                    'depth_confidence': min(source_analysis['coverage_ratio'] * 2, 1.0),
                    'scan_type': 'multi_point'
                },
                'spatial_features': spatial_3d
            }
            
        except Exception as e:
            self.get_logger().error(f'获取waypoint深度信息失败: {e}')
            return None
        
    def _calculate_real_gripper_width(self, mask, depth_data, waypoint_data):
        """计算真实抓夹宽度"""
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
            center_depth = self._get_object_center_depth(mask, depth_data)
            if center_depth is None:
                return None
            
            # 相机内参（你可能需要根据实际情况调整）
            fx = 912.694580078125  # 根据你的相机内参
            
            # 像素到毫米转换
            # real_width_mm = pixel_width * (depth_m / focal_length) * 1000
            real_width_mm = shorter_side_pixels * (center_depth / fx) * 1000

            
            return {
                'real_width_mm': real_width_mm,
                'pixel_width': shorter_side_pixels,
                'depth_used': center_depth,
                'angle': angle
            }
            
        except Exception as e:
            self.get_logger().error(f'抓夹宽度计算失败: {e}')
            return None
                   
    def _get_object_center_depth(self, mask, depth_data):
        """获取物体中心的深度值"""
        try:
            # 找到mask中心
            mask_indices = np.where(mask > 0)
            if len(mask_indices[0]) == 0:
                return None
            
            center_y = int(np.median(mask_indices[0]))
            center_x = int(np.median(mask_indices[1]))
            
            # 获取中心点周围的深度值
            depth_samples = []
            for dy in range(-3, 4):
                for dx in range(-3, 4):
                    y, x = center_y + dy, center_x + dx
                    if 0 <= y < depth_data.shape[0] and 0 <= x < depth_data.shape[1]:
                        if mask[y, x] > 0:  # 确保在物体内
                            depth_val = depth_data[y, x] / 1000.0
                            if depth_val > 0.01:
                                depth_samples.append(depth_val)
            
            return np.median(depth_samples) if depth_samples else None
            
        except Exception:
            return None
    def _analyze_object_sources(self, bbox, mask):
        """分析对象的来源waypoint分布"""
        try:
            x1, y1, x2, y2 = map(int, bbox)
            fusion_mapping = self.fusion_mapping_data['fusion_mapping']
            
            # 统计bbox中每个waypoint的像素贡献
            waypoint_pixels = {}
            total_pixels = 0
            mask_pixels = 0
            
            for y in range(y1, y2):
                for x in range(x1, x2):
                    if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
                        total_pixels += 1
                        
                        # 检查是否在mask内
                        if mask[y, x] > 0:
                            mask_pixels += 1
                            
                            # 查找像素来源
                            mapping_info = fusion_mapping.get((x, y))
                            if mapping_info:
                                wp_idx = mapping_info['source_waypoint']
                                waypoint_pixels[wp_idx] = waypoint_pixels.get(wp_idx, 0) + 1
            
            # 找到主要来源waypoint
            if waypoint_pixels:
                dominant_waypoint = max(waypoint_pixels.items(), key=lambda x: x[1])
                dominant_wp_idx = dominant_waypoint[0]
                dominant_pixel_count = dominant_waypoint[1]
                coverage_ratio = dominant_pixel_count / mask_pixels if mask_pixels > 0 else 0
            else:
                dominant_wp_idx = None
                coverage_ratio = 0
            
            return {
                'dominant_waypoint': dominant_wp_idx,
                'coverage_ratio': coverage_ratio,
                'waypoint_distribution': waypoint_pixels,
                'total_mask_pixels': mask_pixels,
                'coverage_threshold_met': coverage_ratio >= 0.7  # 70%阈值
            }
            
        except Exception as e:
            self.get_logger().error(f'分析对象来源失败: {e}')
            return {'dominant_waypoint': None, 'coverage_ratio': 0}
        

    def _load_fusion_mapping(self):
        """加载融合映射信息"""
        try:
            import pickle
            
            mapping_file = os.path.join(self.current_scan_output_dir, "fusion_mapping.pkl")
            if not os.path.exists(mapping_file):
                self.get_logger().warn('未找到融合映射文件，将使用简化深度处理')
                return None
            
            with open(mapping_file, 'rb') as f:
                mapping_data = pickle.load(f)
            
            # 检查是否是单点扫描
            fusion_params = mapping_data.get('fusion_params', {})
            is_single_point = fusion_params.get('single_point', False)
            
            if is_single_point:
                self.get_logger().info(f'✅ 单点扫描映射信息加载成功:')
                self.get_logger().info(f'  - 扫描类型: 单点扫描')
                waypoint_idx = fusion_params.get('waypoint_index', 0)
                canvas_size = fusion_params.get('canvas_size', (0, 0))
                self.get_logger().info(f'  - 主waypoint: {waypoint_idx}')
                self.get_logger().info(f'  - 画布尺寸: {canvas_size}')
            else:
                # 多点扫描的处理逻辑（保持原有代码）
                fusion_mapping = mapping_data.get('fusion_mapping', {})
                self.get_logger().info(f'✅ 多点扫描映射信息加载成功:')
                self.get_logger().info(f'  - 映射像素数: {len(fusion_mapping)}')
            
            waypoint_contributions = mapping_data.get('waypoint_contributions', {})
            self.get_logger().info(f'  - 涉及waypoint数: {len(waypoint_contributions)}')
            
            # 验证每个waypoint的文件路径（保持原有代码）
            for wp_idx, contrib in waypoint_contributions.items():
                wp_data = contrib.get('waypoint_data', {})
                color_file = wp_data.get('color_filename', 'N/A')
                depth_file = wp_data.get('depth_raw_filename', 'N/A')
                
                self.get_logger().info(f'  - Waypoint {wp_idx}:')
                self.get_logger().info(f'    * 贡献像素: {contrib.get("pixel_count", 0)}')
                self.get_logger().info(f'    * 彩色文件: {color_file}')
                self.get_logger().info(f'    * 深度文件: {depth_file}')
                
                # 验证深度文件是否真实存在
                if depth_file and depth_file != 'N/A':
                    depth_full_path = os.path.join(self.current_scan_output_dir, os.path.basename(depth_file))
                    depth_exists = os.path.exists(depth_full_path)
                    self.get_logger().info(f'    * 深度文件存在: {depth_exists}')
                    
                    if depth_exists:
                        try:
                            depth_data = np.load(depth_full_path)
                            self.get_logger().info(f'    * 深度数据形状: {depth_data.shape}')
                            self.get_logger().info(f'    * 深度范围: {depth_data.min():.3f} - {depth_data.max():.3f}')
                        except Exception as e:
                            self.get_logger().error(f'    * 深度数据读取失败: {e}')
            
            return mapping_data
            
        except Exception as e:
            self.get_logger().error(f'加载融合映射失败: {e}')
            import traceback
            traceback.print_exc()
            return None
    
    def _test_mapping_queries(self, fusion_mapping):
        """测试映射查询功能"""
        try:
            if not fusion_mapping:
                return
            
            # 获取一些样本像素进行测试
            sample_pixels = list(fusion_mapping.keys())[:10]  # 取前10个像素
            
            self.get_logger().info(f'🧪 测试映射查询 (样本: {len(sample_pixels)} 个像素):')
            
            for pixel_pos in sample_pixels:
                mapping_info = fusion_mapping[pixel_pos]
                source_wp = mapping_info.get('source_waypoint', 'N/A')
                source_pixel = mapping_info.get('source_pixel', (0, 0))
                
                self.get_logger().info(f'  - 融合像素{pixel_pos} → Waypoint {source_wp}, 原像素{source_pixel}')
            
            # 统计waypoint分布
            waypoint_counts = {}
            for mapping_info in fusion_mapping.values():
                wp_idx = mapping_info.get('source_waypoint')
                waypoint_counts[wp_idx] = waypoint_counts.get(wp_idx, 0) + 1
            
            self.get_logger().info(f'🔍 映射分布统计:')
            for wp_idx, count in waypoint_counts.items():
                percentage = count / len(fusion_mapping) * 100
                self.get_logger().info(f'  - Waypoint {wp_idx}: {count} 像素 ({percentage:.1f}%)')
                
        except Exception as e:
            self.get_logger().error(f'映射查询测试失败: {e}')

    def _process_detection_directly(self, image_rgb, depth_data):
        """直接处理检测，绕过ROS消息转换"""
        try:
            if not self.detection_pipeline:
                self.get_logger().error('检测管道未初始化')
                return
            
            self.processing_active = True
            self.get_logger().info('🚀 开始直接检测处理...')
            
            # 更新检测管道的输出目录
            detection_output_dir = os.path.join(self.current_scan_output_dir, "detection_results")
            os.makedirs(detection_output_dir, exist_ok=True)
            self.detection_pipeline.output_dir = detection_output_dir
            
            self.get_logger().info(f'检测输出目录: {detection_output_dir}')
            self.get_logger().info(f'处理图像尺寸: RGB{image_rgb.shape}, 深度{depth_data.shape}')
            
            # 执行检测
            result = self.detection_pipeline.process_reference_image(
                image_rgb, 
                depth_data,
                generate_visualization=True,
                auto_display=True
            )
            
            if result['success']:
                self.get_logger().info(f'✅ 检测完成，发现 {result["detection_count"]} 个目标')
                
                # 发布检测结果
                self._publish_detection_result_json(result)
                
                # 发布可视化（直接从文件）
                self._publish_visualization_from_file(detection_output_dir)
                
                # 显示检测结果
                self._display_detection_results(result)
                
                # 启动目标选择流程
                self._start_target_selection(result)
                
            else:
                self.get_logger().error(f'❌ 检测失败: {result.get("message", "未知错误")}')
            
        except Exception as e:
            self.get_logger().error(f'直接检测处理失败: {e}')
            import traceback
            traceback.print_exc()
        finally:
            self.processing_active = False
        
    def _publish_detection_result_json(self, result):
        """发布检测结果 - 保存mask和3D坐标用于静态抓取"""
        try:
            result_data = {
                'detection_count': result['detection_count'],
                'processing_time': result.get('processing_time', 0.0),
                'output_directory': self.detection_pipeline.output_dir,
                'timestamp': datetime.now().isoformat(),
                'objects': []
            }
            
            for obj in result.get('objects', []):
                # 🆕 保存mask用于yaw计算
                mask_data = None
                if 'mask' in obj and obj['mask'] is not None:
                    mask = obj['mask']
                    if isinstance(mask, np.ndarray):
                        # 保存mask的轮廓点，用于重建和yaw计算
                        try:
                            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            if contours:
                                largest_contour = max(contours, key=cv2.contourArea)
                                # 简化轮廓，减少数据量
                                epsilon = 0.02 * cv2.arcLength(largest_contour, True)
                                simplified_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
                                contour_points = simplified_contour.reshape(-1, 2).tolist()
                                
                                # 保存mask信息
                                mask_data = {
                                    'contour_points': contour_points,
                                    'mask_shape': mask.shape,
                                    'mask_area': int(np.sum(mask > 0)),
                                    # 🆕 保存最小外接矩形信息，用于yaw计算
                                    'min_area_rect': {
                                        'center': cv2.minAreaRect(largest_contour)[0],
                                        'size': cv2.minAreaRect(largest_contour)[1], 
                                        'angle': cv2.minAreaRect(largest_contour)[2]
                                    }
                                }
                        except Exception as e:
                            self.get_logger().warn(f'处理mask失败: {e}')
                            mask_data = None
                
                # 确保3D坐标被正确保存
                spatial_features = obj.get('features', {}).get('spatial', {})
                world_coordinates = spatial_features.get('world_coordinates', [0, 0, 0])
                scan_detail = spatial_features.get('scan_detail', [0,0,0,0, 0, 0])
                # 构建对象数据
                obj_data = {
                    'object_id': obj['object_id'],
                    'class_id': int(obj['class_id']),
                    'class_name': obj['class_name'],
                    'confidence': float(obj['confidence']),
                    'description': obj['description'],
                    'bounding_box': obj.get('bounding_box', [0, 0, 0, 0]),
                    
                    # 🆕 保存mask信息用于yaw计算
                    'mask_info': mask_data,
                    
                    # 确保特征格式正确
                    'features': {
                        'color': {
                            'color_name': obj.get('features', {}).get('color', {}).get('color_name', 'unknown'),
                            'histogram': obj.get('features', {}).get('color', {}).get('histogram', []),
                            **obj.get('features', {}).get('color', {})
                        },
                        'shape': {
                            'hu_moments': obj.get('features', {}).get('shape', {}).get('hu_moments', []),
                            'contours': obj.get('features', {}).get('shape', {}).get('contours', []),
                            'area': obj.get('features', {}).get('shape', {}).get('area', 0),
                            'orientation': obj.get('features', {}).get('shape', {}).get('orientation', 0),
                            **obj.get('features', {}).get('shape', {})
                        },
                        'spatial': {
                            'centroid_2d': obj.get('features', {}).get('spatial', {}).get('centroid_2d', [0, 0]),
                            'region_position': obj.get('features', {}).get('spatial', {}).get('region_position', 'unknown'),
                            'scan_info':scan_detail,
                            # 🆕 确保3D坐标被保存
                            'world_coordinates': world_coordinates,
                            'centroid_3d_camera': obj.get('features', {}).get('spatial', {}).get('centroid_3d_camera', [0, 0, 0]),
                            **obj.get('features', {}).get('spatial', {})
                        },
                        'depth_info': obj.get('features', {}).get('depth_info', {})
                    }
                }
                
                result_data['objects'].append(obj_data)
            
            # 发布
            json_msg = String()
            json_msg.data = json.dumps(result_data, indent=2)
            self.detection_result_pub.publish(json_msg)
            
            # 🆕 同时保存mask信息到单独文件，供静态抓取使用
            self._save_masks_for_grasp(result_data)
            self._save_final_detection_results(result_data)
            self.get_logger().info('📤 Detection result published with mask and 3D coordinates')
            
        except Exception as e:
            self.get_logger().error(f'Failed to publish detection result: {e}')
            import traceback
            traceback.print_exc()

    def _save_final_detection_results(self, result_data):
        """
        保存最终的检测结果到 detection_results.json 文件，包含深度和世界坐标信息。
        它将覆盖 detection_pipeline 之前可能保存的同名文件。
        """
        try:
            # 确保输出目录与 detection_pipeline 使用的目录一致
            final_results_file = os.path.join(self.detection_pipeline.output_dir, "detection_results.json")
            
            # 在保存前，确保所有 mask 都是列表，并且 world_coordinates 也是可序列化的
            # 遍历对象，处理 mask 和 spatial features
            processed_objects = []
            for obj in result_data.get('objects', []):
                # 创建一个对象的副本，以避免直接修改原始对象，影响其他使用
                processed_obj = obj.copy()

                # 处理 mask (从 numpy array 到 list)
                if 'mask' in processed_obj and isinstance(processed_obj['mask'], np.ndarray):
                    processed_obj['mask'] = processed_obj['mask'].tolist()
                
                # 处理 world_coordinates (确保是 list 或 tuple)
                if 'features' in processed_obj and 'spatial' in processed_obj['features']:
                    if 'world_coordinates' in processed_obj['features']['spatial']:
                        wc = processed_obj['features']['spatial']['world_coordinates']
                        if isinstance(wc, np.ndarray):
                            processed_obj['features']['spatial']['world_coordinates'] = wc.tolist()
                        # 如果已经是 tuple (如你的日志所示)，则无需转换
                
                processed_objects.append(processed_obj)
            
            # 构建最终的 JSON 结构
            final_json_data = {
                'objects': processed_objects,
                'scan_info': result_data.get('scan_info', {}) # 包含扫描信息等元数据
                # 你可以根据需要添加其他顶级键
            }

            with open(final_results_file, 'w', encoding='utf-8') as f:
                json.dump(final_json_data, f, indent=4, ensure_ascii=False) # 使用 indent 4 使 JSON 更可读
            
            self.get_logger().info(f'💾 最终检测结果已保存并覆盖: {final_results_file}')
            
        except TypeError as e:
            self.get_logger().error(f"类型错误：无法将最终检测结果序列化到 JSON: {e}")
            self.get_logger().error(f"检查数据结构，特别是mask和world_coordinates是否已正确转换。")
        except Exception as e:
            self.get_logger().error(f'保存最终检测结果失败: {e}', exc_info=True)


    def _save_masks_for_grasp(self, result_data):
        """保存masks到单独文件，供静态抓取系统使用"""
        try:
            masks_file = os.path.join(self.detection_pipeline.output_dir, "object_masks.json")
            
            masks_data = {}
            for obj in result_data['objects']:
                if obj.get('mask_info'):
                    masks_data[obj['object_id']] = obj['mask_info']
            
            with open(masks_file, 'w') as f:
                json.dump(masks_data, f, indent=2)
            
            self.get_logger().info(f'💾 Masks saved to: {masks_file}')
            
        except Exception as e:
            self.get_logger().error(f'Failed to save masks: {e}')

    def _publish_visualization_from_file(self, output_dir):
        """从文件发布可视化图像"""
        try:
            vis_file = os.path.join(output_dir, "detection_visualization.jpg")
            if os.path.exists(vis_file):
                # 读取可视化图像
                vis_image = cv2.imread(vis_file)
                if vis_image is not None:
                    # 转换为RGB
                    vis_rgb = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)
                    
                    # 手动创建ROS图像消息
                    vis_msg = Image()
                    vis_msg.header.stamp = self.get_clock().now().to_msg()
                    vis_msg.header.frame_id = "detection_frame"
                    vis_msg.height = vis_rgb.shape[0]
                    vis_msg.width = vis_rgb.shape[1]
                    vis_msg.encoding = "rgb8"
                    vis_msg.is_bigendian = False
                    vis_msg.step = vis_rgb.shape[1] * 3
                    vis_msg.data = vis_rgb.tobytes()
                    
                    self.visualization_pub.publish(vis_msg)
                    self.get_logger().info('🖼️ 可视化图像已发布')
            else:
                self.get_logger().warn(f'可视化文件不存在: {vis_file}')
                
        except Exception as e:
            self.get_logger().error(f'发布可视化图像失败: {e}')
    def _collect_scan_info(self):
        """收集扫描信息用于tracking回退策略"""
        try:
            scan_info = {
                'center_pose': None,
                'bounds': None,
                'waypoint_poses': []
            }
            
            if self.fusion_mapping_data:
                waypoint_contributions = self.fusion_mapping_data.get('waypoint_contributions', {})
                
                # 收集所有waypoint位姿
                poses = []
                for wp_data in waypoint_contributions.values():
                    waypoint_info = wp_data.get('waypoint_data', {})
                    world_pos = waypoint_info.get('world_pos')
                    if world_pos:
                        poses.append({
                            'position': {'x': world_pos[0], 'y': world_pos[1], 'z': world_pos[2]},
                            'orientation': {
                                'roll': waypoint_info.get('roll', 0),
                                'pitch': waypoint_info.get('pitch', 0), 
                                'yaw': waypoint_info.get('yaw', 0)
                            }
                        })
                
                scan_info['waypoint_poses'] = poses
                
                # 计算扫描中心
                if poses:
                    center_x = sum(p['position']['x'] for p in poses) / len(poses)
                    center_y = sum(p['position']['y'] for p in poses) / len(poses)
                    center_z = 350  # 固定追踪高度
                    
                    scan_info['center_pose'] = {
                        'position': {'x': center_x, 'y': center_y, 'z': center_z},
                        'orientation': {'roll': 179, 'pitch': 0, 'yaw': waypoint_info.get('yaw', 0)}
                    }
            
            return scan_info
            
        except Exception as e:
            self.get_logger().error(f'Failed to collect scan info: {e}')
            return {'center_pose': None, 'bounds': None, 'waypoint_poses': []}
    
    def _display_detection_results(self, result):
        """显示检测结果"""
        objects = result.get('objects', [])
        
        self.get_logger().info(f"\n{'='*60}")
        self.get_logger().info("Detection Results Summary")
        self.get_logger().info(f"{'='*60}")
        
        if len(objects) == 0:
            self.get_logger().info("No objects detected")
            return
        
        self.get_logger().info(f"Detected {len(objects)} objects:")
        for i, obj in enumerate(objects, 1):
            self.get_logger().info(f"{i:2d}. {obj['description']} (ID: {obj['object_id']}, Conf: {obj['confidence']:.3f})")
        
        self.get_logger().info(f"{'='*60}")

    def republish_detection_results(self):
        """持续发布检测结果供tracking使用"""
        if self.last_detection_result:
            self._publish_detection_result_json(self.last_detection_result)

    def _start_target_selection(self, result):
        """启动目标选择流程"""
        try:
            self.get_logger().info('🎯 启动目标选择流程...')
            
            success = self.detection_pipeline.select_tracking_targets()
            
            if success:
                # 保存检测结果
                self.last_detection_result = result
                
                # 收集扫描信息用于tracking
                scan_info = self._collect_scan_info()
                
                # 发布完成信号（安全版）
                self._publish_detection_complete_signal(scan_info, result)
            else:
                self.get_logger().warn('目标选择失败或被跳过')
                
        except Exception as e:
            self.get_logger().error(f'目标选择流程失败: {e}')

    def _serialize_pose_data(self, pose_data):
        """安全地序列化位姿数据"""
        try:
            if not isinstance(pose_data, dict):
                return None
            
            serialized = {}
            
            # 处理位置信息
            position = pose_data.get('position', {})
            if isinstance(position, dict):
                serialized['position'] = {
                    'x': float(position.get('x', 0.0)),
                    'y': float(position.get('y', 0.0)),
                    'z': float(position.get('z', 0.0))
                }
            else:
                serialized['position'] = {'x': 0.0, 'y': 0.0, 'z': 0.0}
            
            # 处理姿态信息
            orientation = pose_data.get('orientation', {})
            if isinstance(orientation, dict):
                serialized['orientation'] = {
                    'roll': float(orientation.get('roll', 0.0)),
                    'pitch': float(orientation.get('pitch', 0.0)),
                    'yaw': float(orientation.get('yaw', 0.0))
                }
            else:
                serialized['orientation'] = {'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0}
            
            return serialized
            
        except Exception as e:
            self.get_logger().error(f'位姿数据序列化失败: {e}')
            return None

    def _serialize_bounds_data(self, bounds_data):
        """安全地序列化边界数据"""
        try:
            if isinstance(bounds_data, dict):
                return {
                    'min_x': float(bounds_data.get('min_x', 0.0)),
                    'max_x': float(bounds_data.get('max_x', 0.0)),
                    'min_y': float(bounds_data.get('min_y', 0.0)),
                    'max_y': float(bounds_data.get('max_y', 0.0)),
                    'min_z': float(bounds_data.get('min_z', 0.0)),
                    'max_z': float(bounds_data.get('max_z', 0.0))
                }
            elif isinstance(bounds_data, (list, tuple)) and len(bounds_data) >= 6:
                return {
                    'min_x': float(bounds_data[0]),
                    'max_x': float(bounds_data[1]),
                    'min_y': float(bounds_data[2]),
                    'max_y': float(bounds_data[3]),
                    'min_z': float(bounds_data[4]),
                    'max_z': float(bounds_data[5])
                }
            else:
                return None
                
        except Exception as e:
            self.get_logger().error(f'边界数据序列化失败: {e}')
            return None

    def _publish_detection_complete_signal(self, scan_info, result):
        """发布检测完成信号 - 添加参考特征数据库信息"""
        try:
            complete_data = {
                'scan_directory': str(self.current_scan_output_dir),
                'detection_count': len(result.get('objects', [])),
                'status': 'completed',
                'total_objects': len(result.get('objects', [])),
                'timestamp': datetime.now().isoformat(),
                
                # tracking 系统需要的扫描信息
                'scan_center_pose': self._serialize_pose_data(scan_info.get('center_pose')),
                'scan_bounds': self._serialize_bounds_data(scan_info.get('bounds')),
                'waypoint_poses': [
                    self._serialize_pose_data(pose) for pose in scan_info.get('waypoint_poses', [])
                    if isinstance(pose, dict)
                ],
                
                # 🆕 参考特征数据库信息
                'reference_features_path': self.detection_pipeline.output_dir,  # detection_pipeline 的输出目录
                'detection_results_file': os.path.join(self.detection_pipeline.output_dir, 'detection_results.json'),
                'selected_targets_file': os.path.join(self.detection_pipeline.output_dir, 'tracking_selection.txt')
            }
            
            # 发布信号
            try:
                json_str = json.dumps(complete_data, indent=2, ensure_ascii=False)
                complete_msg = String()
                complete_msg.data = json_str
                self.detection_complete_pub.publish(complete_msg)
                
                self.get_logger().info('✅ Detection complete signal published')
                
            except (TypeError, ValueError) as json_error:
                self.get_logger().error(f'JSON serialization failed: {json_error}')
                self._publish_minimal_complete_signal(result)
                
        except Exception as e:
            self.get_logger().error(f'Failed to publish detection complete signal: {e}')

    def _publish_minimal_complete_signal(self):
        """发布最小化完成信号"""
        try:
            minimal_data = {
                'status': 'completed',
                'scan_directory': str(self.current_scan_output_dir),
                'detection_count': 0
            }
            
            complete_msg = String()
            complete_msg.data = json.dumps(minimal_data)
            self.detection_complete_pub.publish(complete_msg)
            
            self.get_logger().warn('⚠️ 发布了最小化检测完成信号')
            
        except Exception as e:
            self.get_logger().error(f'最小化信号发布失败: {e}')


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = BypassCvBridgeDetectionNode()
        node.get_logger().info('🔍 绕过cv_bridge的检测节点运行中...')
        rclpy.spin(node)
    except KeyboardInterrupt:
        print('🛑 检测节点被用户中断')
    except Exception as e:
        print(f'❌ 检测节点运行错误: {e}')
    finally:
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()