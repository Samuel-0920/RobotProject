# detection_node.py - 重构版本，使用新的增强组件
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
from typing import Dict, List, Optional, Tuple
#  导入新的增强组件
from vision_ai.detection.enhanced_detection_pipeline import EnhancedDetectionPipeline
from vision_ai.detection.utils.coordinate_calculator import CoordinateCalculator, ObjectAnalyzer
from vision_ai.detection.utils.adaptive_learner import AdaptiveThresholdManager
from vision_ai.detection.utils.enhanced_config_manager import EnhancedConfigManager

class EnhancedDetectionNode(Node):
    def __init__(self):
        super().__init__('enhanced_detection_node')
        
        #  使用增强的配置管理器
        config_path = os.path.join(os.path.dirname(__file__), 'config', 'enhanced_detection_config.json')
        self.config_manager = EnhancedConfigManager(config_path)
        
        # 显示配置摘要
        self.get_logger().info(f"\n{self.config_manager.get_config_summary()}")
        
        #  初始化增强组件
        self._initialize_enhanced_components()
        
        # 状态管理
        self.processing_active = False
        self.latest_reference_image = None
        self.latest_depth_image = None
        self.current_scan_output_dir = None
        self.last_detection_result = None
        self.fusion_mapping_data = None
        
        # ROS订阅者和发布者
        self._setup_ros_interfaces()
        
        self.get_logger().info(' Enhanced detection node startup completed')
    
    def _initialize_enhanced_components(self):
        """初始化增强组件"""
        try:
            #  增强检测管道
            self.enhanced_pipeline = EnhancedDetectionPipeline(
                config_file=self.config_manager.config_file,
                output_dir=None  # 将在处理时动态设置
            )
            
            #  坐标计算器
            camera_config = self.config_manager.get_camera_config()
            calibration_data = {
                'camera_intrinsics': camera_config.get('intrinsics', {}),
                'hand_eye_translation': camera_config.get('hand_eye_calibration', {}).get('translation'),
                'hand_eye_quaternion': camera_config.get('hand_eye_calibration', {}).get('quaternion')
            }
            self.coordinate_calculator = CoordinateCalculator(calibration_data)
            
            #  物体分析器
            self.object_analyzer = ObjectAnalyzer(self.coordinate_calculator)
            
            #  自适应学习管理器
            adaptive_config = self.config_manager.get_adaptive_learning_config()
            if adaptive_config.get('enabled', True):
                learning_file = adaptive_config.get('learning_data_file', 'data/adaptive_learning.json')
                learning_file = os.path.join(os.path.dirname(__file__), learning_file)
                self.adaptive_manager = AdaptiveThresholdManager(learning_file)
                self.get_logger().info(' 自适应学习已启用')
            else:
                self.adaptive_manager = None
                self.get_logger().info(' 自适应学习已禁用')
            
            self.get_logger().info(' 增强组件初始化成功')
            
        except Exception as e:
            self.get_logger().error(f' 增强组件初始化失败: {e}')
            raise RuntimeError(f"组件初始化失败: {e}") from e
    
    def _setup_ros_interfaces(self):
        """设置ROS接口"""
        # 订阅者
        self.stitching_complete_sub = self.create_subscription(
            String, '/stitching_complete', self.stitching_complete_callback, 10
        )
        self.reference_image_sub = self.create_subscription(
            Image, '/reference_image', self.reference_image_callback, 10
        )
        self.reference_depth_sub = self.create_subscription(
            Image, '/reference_depth', self.reference_depth_callback, 10
        )
        
        # 发布者
        self.detection_result_pub = self.create_publisher(String, '/detection_result', 10)
        self.detection_complete_pub = self.create_publisher(String, '/detection_complete', 10)
        self.visualization_pub = self.create_publisher(Image, '/detection_visualization', 10)
    
    def stitching_complete_callback(self, msg):
        """拼接完成回调 - 使用增强管道处理"""
        try:
            self.current_scan_output_dir = msg.data
            self.get_logger().info(f'Receive stitching_complete message: {self.current_scan_output_dir}')
            self._load_and_process_enhanced()
            
        except Exception as e:
            self.get_logger().error(f'拼接完成处理失败: {e}')
            import traceback
            traceback.print_exc()
    
    def _load_and_process_enhanced(self):
        """加载并使用增强管道处理"""
        try:
            if not os.path.exists(self.current_scan_output_dir):
                self.get_logger().error(f'扫描目录不存在: {self.current_scan_output_dir}')
                return
            
            #  加载融合映射信息
            self.get_logger().info('🔄 开始加载融合映射信息...')
            self.fusion_mapping_data = self._load_fusion_mapping()
            
            if self.fusion_mapping_data:
                self.get_logger().info(' 映射信息加载成功')
            else:
                self.get_logger().warn(' 映射信息加载失败，将使用简化模式')
            
            #  查找并加载图像
            image_rgb, depth_data, waypoint_data = self._load_scan_data()
            
            if image_rgb is None:
                self.get_logger().error('无法加载扫描数据')
                return
            
            #  使用增强管道构建参考特征库
            self._process_with_enhanced_pipeline(image_rgb, depth_data, waypoint_data)
            
        except Exception as e:
            self.get_logger().error(f'增强处理失败: {e}')
            import traceback
            traceback.print_exc()
    
    def _load_scan_data(self):
        """加载扫描数据"""
        try:
            # 查找融合后的最终彩色图像
            color_files = []
            for file in os.listdir(self.current_scan_output_dir):
                if file.startswith('multi_point_') and file.endswith('.jpg'):
                    color_files.insert(0, file)
                elif file.startswith('single_point_') and file.endswith('.jpg'):
                    color_files.append(file)
            
            if not color_files:
                self.get_logger().error('没有找到彩色图像文件')
                return None, None, None
            
            # 加载图像
            color_file = os.path.join(self.current_scan_output_dir, color_files[0])
            image_bgr = cv2.imread(color_file)
            if image_bgr is None:
                self.get_logger().error(f'无法加载图像: {color_file}')
                return None, None, None
            
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            self.get_logger().info(f'图像加载成功: {image_rgb.shape}')
            
            #  获取深度数据和waypoint信息
            depth_data, waypoint_data = self._extract_depth_and_waypoint_data()
            
            return image_rgb, depth_data, waypoint_data
            
        except Exception as e:
            self.get_logger().error(f'扫描数据加载失败: {e}')
            return None, None, None
    
    def _extract_depth_and_waypoint_data(self):
        """提取深度数据和waypoint信息"""
        try:
            if not self.fusion_mapping_data:
                # 简化模式：创建占位符数据
                placeholder_depth = np.ones((480, 640), dtype=np.uint16) * 1000
                placeholder_waypoint = {
                    'world_pos': [0, 0, 350],
                    'roll': 179, 'pitch': 0, 'yaw': 0
                }
                return placeholder_depth, placeholder_waypoint
            
            #  从融合映射数据中提取
            fusion_params = self.fusion_mapping_data.get('fusion_params', {})
            is_single_point = fusion_params.get('single_point', False)
            
            if is_single_point:
                # 单点扫描
                waypoint_contributions = self.fusion_mapping_data['waypoint_contributions']
                main_waypoint_idx = list(waypoint_contributions.keys())[0]
                waypoint_data = waypoint_contributions[main_waypoint_idx]['waypoint_data']
                
                # 加载深度数据
                depth_file = waypoint_data.get('depth_raw_filename')
                if depth_file and os.path.exists(os.path.join(self.current_scan_output_dir, os.path.basename(depth_file))):
                    depth_full_path = os.path.join(self.current_scan_output_dir, os.path.basename(depth_file))
                    depth_data = np.load(depth_full_path)
                else:
                    depth_data = np.ones((480, 640), dtype=np.uint16) * 1000
                
            else:
                # 多点扫描：使用第一个waypoint的数据
                waypoint_contributions = self.fusion_mapping_data['waypoint_contributions']
                first_waypoint_idx = list(waypoint_contributions.keys())[0]
                waypoint_data = waypoint_contributions[first_waypoint_idx]['waypoint_data']
                
                depth_file = waypoint_data.get('depth_raw_filename')
                if depth_file and os.path.exists(os.path.join(self.current_scan_output_dir, os.path.basename(depth_file))):
                    depth_full_path = os.path.join(self.current_scan_output_dir, os.path.basename(depth_file))
                    depth_data = np.load(depth_full_path)
                else:
                    depth_data = np.ones((480, 640), dtype=np.uint16) * 1000
            
            return depth_data, waypoint_data
            
        except Exception as e:
            self.get_logger().error(f'深度和waypoint数据提取失败: {e}')
            # 返回占位符数据
            placeholder_depth = np.ones((480, 640), dtype=np.uint16) * 1000
            placeholder_waypoint = {
                'world_pos': [0, 0, 350],
                'roll': 179, 'pitch': 0, 'yaw': 0
            }
            return placeholder_depth, placeholder_waypoint
    
    def _process_with_enhanced_pipeline(self, image_rgb, depth_data, waypoint_data):
        """使用增强管道处理"""
        try:
            self.processing_active = True
            self.get_logger().info(' 开始增强管道处理...')
            
            # 设置增强管道的输出目录
            detection_output_dir = os.path.join(self.current_scan_output_dir, "enhanced_detection_results")
            os.makedirs(detection_output_dir, exist_ok=True)
            self.enhanced_pipeline.output_dir = detection_output_dir
            
            #  传递相机内参给管道（用于3D中心点计算）
            camera_config = self.config_manager.get_camera_config()
            intrinsics = camera_config.get('intrinsics', {})
            self.enhanced_pipeline.camera_fx = intrinsics.get('fx', 912.7)
            self.enhanced_pipeline.camera_fy = intrinsics.get('fy', 910.3)
            self.enhanced_pipeline.camera_cx = intrinsics.get('cx', 624.0)
            self.enhanced_pipeline.camera_cy = intrinsics.get('cy', 320.7)
            
            # 使用增强管道构建参考特征库
            result = self.enhanced_pipeline.build_reference_library(
                image_rgb=image_rgb,
                depth_image=depth_data,
                waypoint_data=waypoint_data,
                generate_visualization=True
            )
            
            if result['success']:
                self.get_logger().info(f' 增强检测完成，构建了 {result["detection_count"]} 个参考特征')
                
                # 转换为与原系统兼容的格式
                compatible_result = self._convert_to_compatible_format(result)
                
                # 发布结果
                self._publish_enhanced_detection_result(compatible_result, result)  # 添加result参数
                
                #  改进的可视化发布（现在包含弹窗显示）
                self._publish_visualization_from_enhanced_pipeline(detection_output_dir)
                
                self._display_enhanced_detection_results(compatible_result)
                self._start_enhanced_target_selection(compatible_result)
                
            else:
                self.get_logger().error(f' 增强检测失败: {result.get("message", "未知错误")}')
            
        except Exception as e:
            self.get_logger().error(f'增强管道处理失败: {e}')
            import traceback
            traceback.print_exc()
        finally:
            self.processing_active = False
    
    def _convert_to_compatible_format(self, enhanced_result):
        """将增强结果转换为与原系统兼容的格式"""
        try:
            reference_library = enhanced_result.get('reference_library', {})
            compatible_objects = []
            
            for obj_id, entry in reference_library.items():
                metadata = entry['metadata']
                features = entry['features']
                
                #  构建兼容的对象信息
                compatible_obj = {
                    'object_id': metadata['object_id'],
                    'class_id': metadata['class_id'],
                    'class_name': metadata['class_name'], 
                    'confidence': metadata['confidence'],
                    'bounding_box': metadata['bounding_box'],
                    'description': metadata['description'],
                    'features': features,
                    'quality_score': entry['quality_score']
                }
                
                #  添加增强的深度和抓夹信息
                if 'spatial' in features:
                    spatial_features = features['spatial']
                    
                    # 深度信息
                    compatible_obj['features']['depth_info'] = {
                        'height_mm': spatial_features.get('height_mm', 30.0),
                        'background_world_z': spatial_features.get('background_world_z', 300.0),
                        'background_depth_m': spatial_features.get('background_depth_m', 0.3),
                        'depth_confidence': spatial_features.get('confidence', 0.8)
                    }
                    
                    # 抓夹信息
                    if 'gripper_width_info' in spatial_features:
                        gripper_info = spatial_features['gripper_width_info']
                        compatible_obj['features']['grasp_info'] = {
                            'recommended_width': gripper_info['recommended_gripper_width'],
                            'real_width_mm': gripper_info['real_width_mm'],
                            'grasp_angle': gripper_info.get('angle', 0),
                            'pixel_width': gripper_info['pixel_width'],
                            'depth_used': gripper_info['depth_used']
                        }
                
                compatible_objects.append(compatible_obj)
            
            # 构建兼容的结果格式
            compatible_result = {
                'success': True,
                'objects': compatible_objects,
                'detection_count': len(compatible_objects),
                'processing_time': enhanced_result.get('processing_time', 0),
                'message': f'增强检测成功，构建了 {len(compatible_objects)} 个参考特征'
            }
            
            return compatible_result
            
        except Exception as e:
            self.get_logger().error(f'格式转换失败: {e}')
            return {'success': False, 'objects': [], 'detection_count': 0}
    
    def _publish_enhanced_detection_result(self, result, original_enhanced_result=None):
        """发布增强检测结果"""
        try:
            #  使用原有的发布逻辑，但添加了质量分数信息
            result_data = {
                'detection_count': result['detection_count'],
                'processing_time': result.get('processing_time', 0.0),
                'output_directory': self.enhanced_pipeline.output_dir,
                'timestamp': datetime.now().isoformat(),
                'enhanced_features': True,  #  标记使用了增强特征
                'objects': []
            }
            
            for obj in result.get('objects', []):
                # 构建对象数据（保持原有格式，添加质量信息）
                obj_data = {
                    'object_id': obj['object_id'],
                    'class_id': obj['class_id'],
                    'class_name': obj['class_name'],
                    'confidence': obj['confidence'],
                    'description': obj['description'],
                    'bounding_box': obj['bounding_box'],
                    'quality_score': obj.get('quality_score', 0.0),  #  特征质量分数
                    'features': self._sanitize_features_for_publishing(obj['features'])
                }
                
                result_data['objects'].append(obj_data)
            if original_enhanced_result:
                if 'detailed_3d_processing' in original_enhanced_result:
                    result_data['detailed_3d_processing'] = original_enhanced_result['detailed_3d_processing']
                
                if 'merge_decision_data' in original_enhanced_result:
                    result_data['merge_decision_data'] = original_enhanced_result['merge_decision_data']
            
            # 保存增强的检测结果
            self._save_enhanced_detection_results(result_data, original_enhanced_result)
            
            # 清理数据用于ROS消息发布
            sanitized_result_data = self._sanitize_features_for_publishing(result_data)

            # 发布ROS消息
            json_msg = String()
            json_msg.data = json.dumps(sanitized_result_data, indent=2)
            self.detection_result_pub.publish(json_msg)
            
            self.get_logger().info('📤 增强检测结果已发布')
            
        except Exception as e:
            self.get_logger().error(f'发布增强检测结果失败: {e}')
    
    def _save_enhanced_detection_results(self, result_data, original_enhanced_result=None):
        """保存增强检测结果"""
        try:
            # 深拷贝以避免修改原数据
            import copy
            sanitized_data = copy.deepcopy(result_data)
            # 添加原始增强结果中的统计信息
            if original_enhanced_result:
                # 添加后处理统计
                if 'post_processing_stats' in original_enhanced_result:
                    result_data['post_processing_stats'] = original_enhanced_result['post_processing_stats']
                
                # 添加各阶段时间统计
                if 'stage_times' in original_enhanced_result:
                    result_data['stage_times'] = original_enhanced_result['stage_times']
                
                # 添加性能数据
                if 'performance_data' in original_enhanced_result:
                    result_data['performance_data'] = original_enhanced_result['performance_data']

                if 'detailed_3d_processing' in original_enhanced_result:
                    result_data['detailed_3d_processing'] = original_enhanced_result['detailed_3d_processing']
                
                if 'merge_decision_data' in original_enhanced_result:
                    result_data['merge_decision_data'] = original_enhanced_result['merge_decision_data']
            result_data = self._sanitize_features_for_publishing(result_data)
            # 清理features数据用于JSON序列化
            if 'objects' in result_data:
                for obj in result_data['objects']:
                    if 'features' in obj:
                        obj['features'] = self._sanitize_features_for_publishing(obj['features'])
        
            # 测试JSON序列化
            try:
                json_str = json.dumps(result_data, indent=2)
                self.get_logger().info("JSON序列化测试通过")
            except Exception as json_error:
                self.get_logger().error(f"JSON序列化测试失败: {json_error}")
                # 继续寻找问题
                self._find_non_serializable_objects(result_data)
                return
            # 保存到增强管道的输出目录
            results_file = os.path.join(self.enhanced_pipeline.output_dir, "enhanced_detection_results.json")
            
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, indent=4, ensure_ascii=False)
            
            self.get_logger().info(f'💾 增强检测结果已保存: {results_file}')
            
        except Exception as e:
            self.get_logger().error(f'保存增强检测结果失败: {e}')

    def _find_non_serializable_objects(self, obj, path="root"):
        """找出不可序列化的对象"""
        import numpy as np
        
        if isinstance(obj, dict):
            for key, value in obj.items():
                self._find_non_serializable_objects(value, f"{path}.{key}")
        elif isinstance(obj, (list, tuple)):
            for i, item in enumerate(obj):
                self._find_non_serializable_objects(item, f"{path}[{i}]")
        elif isinstance(obj, (np.integer, np.floating, np.bool_)):
            self.get_logger().error(f"发现numpy类型在 {path}: {type(obj)} = {obj}")
        elif not isinstance(obj, (bool, int, float, str, type(None))):
            self.get_logger().error(f"发现不可序列化类型在 {path}: {type(obj)} = {obj}")

    def _sanitize_features_for_publishing(self, data, path="root") -> any:
        """递归清理所有数据用于JSON发布，带调试信息"""
        import numpy as np
        
        try:
            if isinstance(data, dict):
                sanitized = {}
                for key, value in data.items():
                    current_path = f"{path}.{key}"
                    try:
                        sanitized[key] = self._sanitize_features_for_publishing(value, current_path)
                    except Exception as e:
                        self.get_logger().error(f'清理失败在路径 {current_path}, 类型 {type(value)}: {e}')
                        sanitized[key] = str(value)
                return sanitized
                
            elif isinstance(data, (list, tuple)):
                sanitized_list = []
                for i, item in enumerate(data):
                    current_path = f"{path}[{i}]"
                    try:
                        sanitized_list.append(self._sanitize_features_for_publishing(item, current_path))
                    except Exception as e:
                        self.get_logger().error(f'清理失败在路径 {current_path}, 类型 {type(item)}: {e}')
                        sanitized_list.append(str(item))
                return sanitized_list
                
            elif isinstance(data, np.ndarray):
                return data.tolist()
                
            elif isinstance(data, np.integer):
                return int(data)
                
            elif isinstance(data, np.floating):
                return float(data)
                
            elif isinstance(data, np.bool_):
                return bool(data)
                
            elif hasattr(data, 'item') and callable(getattr(data, 'item')):
                return data.item()
                
            elif isinstance(data, (bool, int, float, str)) or data is None:
                return data
                
            else:
                # 检测可能的问题类型
                data_type_str = str(type(data))
                self.get_logger().warn(f'未知类型在路径 {path}: {data_type_str}, 值: {data}')
                return str(data)
                
        except Exception as e:
            self.get_logger().error(f'数据清理失败在路径 {path}, 类型 {type(data)}: {e}')
            return str(data)
        
    def _display_enhanced_detection_results(self, result):
        """显示增强检测结果"""
        objects = result.get('objects', [])
        
        self.get_logger().info(f"\n{'='*60}")
        self.get_logger().info(" 增强检测结果摘要")
        self.get_logger().info(f"{'='*60}")
        
        if len(objects) == 0:
            self.get_logger().info("未检测到任何目标")
            return
        
        self.get_logger().info(f"检测到 {len(objects)} 个目标（使用3D点云特征）:")
        for i, obj in enumerate(objects, 1):
            quality = obj.get('quality_score', 0.0)
            description = obj['description']
            confidence = obj['confidence']
            
            self.get_logger().info(
                f"{i:2d}. {description} "
                f"(ID: {obj['object_id']}, 置信度: {confidence:.3f}, 特征质量: {quality:.1f}%)"
            )
        
        self.get_logger().info(f"{'='*60}")
    
    def _start_enhanced_target_selection(self, result):
        """启动增强目标选择流程"""
        try:
            self.get_logger().info(' 启动增强目标选择流程...')
            
            #  使用增强管道的选择功能
            success = self.enhanced_pipeline.select_tracking_targets()
            
            if success:
                # 保存检测结果
                self.last_detection_result = result
                
                # 收集扫描信息
                scan_info = self._collect_enhanced_scan_info()
                
                # 发布完成信号
                self._publish_enhanced_detection_complete_signal(scan_info, result)
            else:
                self.get_logger().warn('目标选择失败或被跳过')
                
        except Exception as e:
            self.get_logger().error(f'增强目标选择流程失败: {e}')
    
    def _collect_enhanced_scan_info(self):
        """收集增强扫描信息"""
        try:
            scan_info = {
                'center_pose': None,
                'bounds': None,
                'waypoint_poses': [],
                'enhanced_features_available': True,  #  标记有增强特征
                'feature_quality_stats': self._get_feature_quality_stats()  #  特征质量统计
            }
            
            if self.fusion_mapping_data:
                waypoint_contributions = self.fusion_mapping_data.get('waypoint_contributions', {})
                
                # 收集waypoint位姿
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
                        'orientation': {'roll': 179, 'pitch': 0, 'yaw': poses[0]['orientation']['yaw']}
                    }
            
            return scan_info
            
        except Exception as e:
            self.get_logger().error(f'收集增强扫描信息失败: {e}')
            return {'center_pose': None, 'bounds': None, 'waypoint_poses': []}
    
    def _get_feature_quality_stats(self):
        """获取特征质量统计"""
        try:
            if not hasattr(self.enhanced_pipeline, 'reference_library'):
                return {}
            
            qualities = []
            for entry in self.enhanced_pipeline.reference_library.values():
                qualities.append(entry['quality_score'])
            
            if not qualities:
                return {}
            
            return {
                'mean_quality': float(np.mean(qualities)),
                'min_quality': float(np.min(qualities)),
                'max_quality': float(np.max(qualities)),
                'std_quality': float(np.std(qualities))
            }
            
        except Exception:
            return {}
    
    def _publish_enhanced_detection_complete_signal(self, scan_info, result):
        """发布增强检测完成信号"""
        try:
            complete_data = {
                'scan_directory': str(self.current_scan_output_dir),
                'detection_count': len(result.get('objects', [])),
                'status': 'completed',
                'total_objects': len(result.get('objects', [])),
                'timestamp': datetime.now().isoformat(),
                'enhanced_detection': True,  #  标记使用了增强检测
                
                # 扫描信息
                'scan_center_pose': self._serialize_pose_data(scan_info.get('center_pose')),
                'scan_bounds': self._serialize_bounds_data(scan_info.get('bounds')),
                'waypoint_poses': [
                    self._serialize_pose_data(pose) for pose in scan_info.get('waypoint_poses', [])
                    if isinstance(pose, dict)
                ],
                
                #  增强特征信息
                'reference_features_path': self.enhanced_pipeline.output_dir,
                'enhanced_detection_results_file': os.path.join(
                    self.enhanced_pipeline.output_dir, 'enhanced_detection_results.json'
                ),
                'reference_library_file': os.path.join(
                    self.enhanced_pipeline.output_dir, 'reference_library.json'
                ),
                'selected_targets_file': os.path.join(
                    self.enhanced_pipeline.output_dir, 'tracking_selection.txt'
                ),
                
                #  特征质量统计
                'feature_quality_stats': scan_info.get('feature_quality_stats', {}),
                
                #  自适应学习状态
                'adaptive_learning_enabled': self.adaptive_manager is not None,
                'adaptive_learning_stats': self._get_adaptive_learning_stats()
            }
            
            # 发布信号
            try:
                json_str = json.dumps(complete_data, indent=2, ensure_ascii=False)
                complete_msg = String()
                complete_msg.data = json_str
                self.detection_complete_pub.publish(complete_msg)
                
                self.get_logger().info(' 增强检测完成信号已发布')
                
            except (TypeError, ValueError) as json_error:
                self.get_logger().error(f'JSON序列化失败: {json_error}')
                self._publish_minimal_complete_signal()
                
        except Exception as e:
            self.get_logger().error(f'发布增强检测完成信号失败: {e}')
    
    def _get_adaptive_learning_stats(self):
        """获取自适应学习统计"""
        if not self.adaptive_manager:
            return {'enabled': False}
        
        try:
            report = self.adaptive_manager.get_performance_report()
            return {
                'enabled': True,
                'total_matches': report.get('overall_performance', {}).get('total_matches', 0),
                'success_rate': report.get('overall_performance', {}).get('success_rate', 0),
                'recent_matches': report.get('recent_performance', {}).get('matches_last_7_days', 0),
                'optimization_count': report.get('optimization_history', 0)
            }
        except Exception:
            return {'enabled': True, 'error': 'stats_unavailable'}
    
    def _load_fusion_mapping(self):
        """加载融合映射信息（保持原有逻辑）"""
        try:
            import pickle
            
            mapping_file = os.path.join(self.current_scan_output_dir, "fusion_mapping.pkl")
            if not os.path.exists(mapping_file):
                self.get_logger().warn('未找到融合映射文件，将使用简化深度处理')
                return None
            
            with open(mapping_file, 'rb') as f:
                mapping_data = pickle.load(f)
            
            fusion_params = mapping_data.get('fusion_params', {})
            is_single_point = fusion_params.get('single_point', False)
            
            if is_single_point:
                self.get_logger().info(' 单点扫描映射信息加载成功')
                waypoint_idx = fusion_params.get('waypoint_index', 0)
                canvas_size = fusion_params.get('canvas_size', (0, 0))
                self.get_logger().info(f'  - 主waypoint: {waypoint_idx}')
                self.get_logger().info(f'  - 画布尺寸: {canvas_size}')
            else:
                fusion_mapping = mapping_data.get('fusion_mapping', {})
                self.get_logger().info(f' 多点扫描映射信息加载成功')
                self.get_logger().info(f'  - 映射像素数: {len(fusion_mapping)}')
            
            waypoint_contributions = mapping_data.get('waypoint_contributions', {})
            self.get_logger().info(f'  - 涉及waypoint数: {len(waypoint_contributions)}')
            
            return mapping_data
            
        except Exception as e:
            self.get_logger().error(f'加载融合映射失败: {e}')
            return None
    
    def _publish_visualization_from_enhanced_pipeline(self, output_dir):
        """发布可视化 - 单一版本"""
        try:
            vis_file = os.path.join(output_dir, "detection_visualization.jpg")
            
            if os.path.exists(vis_file):
                vis_image_bgr = cv2.imread(vis_file)
                if vis_image_bgr is not None:
                    vis_rgb = cv2.cvtColor(vis_image_bgr, cv2.COLOR_BGR2RGB)
                    
                    # 显示弹窗
                    self._show_detection_popup(vis_rgb)
                    
                    # 发布ROS消息
                    self._publish_ros_visualization(vis_rgb)
                    
                    self.get_logger().info('Enhanced visualization displayed and published')
                else:
                    self.get_logger().error(f'Cannot read visualization: {vis_file}')
            else:
                self.get_logger().warn(f'Visualization file not found: {vis_file}')
                
        except Exception as e:
            self.get_logger().error(f'Failed to publish visualization: {e}')
        
    def _serialize_pose_data(self, pose_data):
        """安全地序列化位姿数据（保持原有逻辑）"""
        try:
            if not isinstance(pose_data, dict):
                return None
            
            serialized = {}
            
            position = pose_data.get('position', {})
            if isinstance(position, dict):
                serialized['position'] = {
                    'x': float(position.get('x', 0.0)),
                    'y': float(position.get('y', 0.0)),
                    'z': float(position.get('z', 0.0))
                }
            else:
                serialized['position'] = {'x': 0.0, 'y': 0.0, 'z': 0.0}
            
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
        """安全地序列化边界数据（保持原有逻辑）"""
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
    
    def _publish_minimal_complete_signal(self):
        """发布最小化完成信号（保持原有逻辑）"""
        try:
            minimal_data = {
                'status': 'completed',
                'scan_directory': str(self.current_scan_output_dir),
                'detection_count': 0,
                'enhanced_detection': False
            }
            
            complete_msg = String()
            complete_msg.data = json.dumps(minimal_data)
            self.detection_complete_pub.publish(complete_msg)
            
            self.get_logger().warn(' 发布了最小化检测完成信号')
            
        except Exception as e:
            self.get_logger().error(f'最小化信号发布失败: {e}')
    
    #  添加自适应学习反馈接口
    def update_tracking_feedback(self, object_id: str, similarity_scores: dict, 
                                is_correct_match: bool, context: dict = None):
        """
        更新追踪反馈，用于自适应学习
        
        Args:
            object_id: 对象ID
            similarity_scores: 各特征的相似度分数
            is_correct_match: 是否是正确匹配
            context: 上下文信息
        """
        if not self.adaptive_manager:
            return
        
        try:
            # 获取对象的特征质量
            feature_quality = self._get_object_feature_quality(object_id)
            
            # 为每个特征类型更新学习历史
            for feature_type, similarity in similarity_scores.items():
                if '.' in feature_type:  # 如果已经是 "type.subtype" 格式
                    main_type, sub_type = feature_type.split('.', 1)
                else:
                    main_type, sub_type = feature_type, 'overall'
                
                self.adaptive_manager.update_learning_history(
                    feature_type=main_type,
                    sub_feature=sub_type,
                    similarity=similarity,
                    feature_quality=feature_quality,
                    is_correct_match=is_correct_match,
                    context=context
                )
            
            self.get_logger().debug(f'📊 已更新自适应学习反馈: {object_id}')
            
        except Exception as e:
            self.get_logger().error(f'更新追踪反馈失败: {e}')
    
    def _get_object_feature_quality(self, object_id: str) -> float:
        """获取对象的特征质量分数"""
        try:
            if hasattr(self.enhanced_pipeline, 'reference_library'):
                entry = self.enhanced_pipeline.reference_library.get(object_id)
                if entry:
                    return entry.get('quality_score', 75.0)
            return 75.0  # 默认质量分数
        except Exception:
            return 75.0
    
    # 保持原有的备用回调方法
    def reference_image_callback(self, msg):
        """接收最终参考图像（备用方法）"""
        self.latest_reference_image = msg
        self.get_logger().info('📸 收到参考图像（备用）')

    def reference_depth_callback(self, msg):
        """接收深度信息（备用方法）"""
        self.latest_depth_image = msg
        self.get_logger().info('📏 收到深度信息（备用）')

    def _publish_visualization_from_enhanced_pipeline(self, output_dir):
        """从增强管道发布可视化图像 - 增强版本"""
        try:
            vis_file = os.path.join(output_dir, "detection_visualization.jpg")
            
            if os.path.exists(vis_file):
                # 读取可视化图像
                vis_image_bgr = cv2.imread(vis_file)
                
                if vis_image_bgr is not None:
                    vis_rgb = cv2.cvtColor(vis_image_bgr, cv2.COLOR_BGR2RGB)
                    
                    #  显示弹窗
                    self._show_detection_popup(vis_rgb)
                    
                    # 📡 发布ROS消息
                    self._publish_ros_visualization(vis_rgb)
                    
                    self.get_logger().info('🖼️ 增强可视化图像已显示并发布')
                else:
                    self.get_logger().error(f'无法读取可视化图像: {vis_file}')
            else:
                self.get_logger().warn(f'可视化文件不存在: {vis_file}')
                
        except Exception as e:
            self.get_logger().error(f'发布增强可视化图像失败: {e}')

    def _show_detection_popup(self, vis_image: np.ndarray):
        """显示单一检测弹窗"""
        try:
            import matplotlib
            matplotlib.use('TkAgg')
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(14, 10))
            plt.imshow(vis_image)
            plt.title('Enhanced Detection Results', fontsize=24, fontweight='bold')
            plt.axis('off')
            plt.tight_layout()
            plt.show(block=False)
            plt.pause(0.1)
            
            self.get_logger().info('Detection visualization displayed')
            
        except Exception as e:
            self.get_logger().error(f'Failed to show popup: {e}')

    def _show_opencv_popup(self, vis_image: np.ndarray):
        """使用OpenCV显示弹窗（fallback方案）"""
        try:
            # 转换为BGR格式
            vis_bgr = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
            
            # 创建窗口
            window_name = 'Enhanced Detection Results'
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 1200, 900)
            
            # 显示图像
            cv2.imshow(window_name, vis_bgr)
            cv2.waitKey(1)  # 非阻塞，让窗口响应
            
            self.get_logger().info(' OpenCV可视化窗口显示成功')
            
            # 设置窗口回调（可选）
            def on_key(key):
                if key == 27:  # ESC键
                    cv2.destroyWindow(window_name)
            
        except Exception as e:
            self.get_logger().error(f'OpenCV显示失败: {e}')

    def _publish_ros_visualization(self, vis_image: np.ndarray):
        """发布ROS可视化消息"""
        try:
            vis_msg = Image()
            vis_msg.header.stamp = self.get_clock().now().to_msg()
            vis_msg.header.frame_id = "detection_frame"
            vis_msg.height = vis_image.shape[0]
            vis_msg.width = vis_image.shape[1]
            vis_msg.encoding = "rgb8"
            vis_msg.is_bigendian = False
            vis_msg.step = vis_image.shape[1] * 3
            vis_msg.data = vis_image.tobytes()
            
            self.visualization_pub.publish(vis_msg)
            
        except Exception as e:
            self.get_logger().error(f'Failed to publish ROS visualization: {e}')

def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = EnhancedDetectionNode()
        node.get_logger().info(' 增强检测节点运行中...')
        rclpy.spin(node)
    except KeyboardInterrupt:
        print('🛑 增强检测节点被用户中断')
    except Exception as e:
        print(f' 增强检测节点运行错误: {e}')
    finally:
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()