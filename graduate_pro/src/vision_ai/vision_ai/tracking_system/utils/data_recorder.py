# utils/data_recorder.py
import os
import json
import cv2
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
from typing import Dict, List, Optional, Tuple
class DataRecorder:
    """追踪数据记录器 - 记录追踪过程和反馈数据"""
    
    def __init__(self, user_id: str, scan_dir: str):
        """
        初始化数据记录器
        
        Args:
            user_id: 用户ID
            scan_dir: 当前扫描目录
        """
        self.user_id = user_id
        self.scan_dir = scan_dir
        
        # 创建数据存储目录
        self.data_dir = os.path.join(scan_dir, 'tracking_data')
        self.images_dir = os.path.join(self.data_dir, 'images')
        self.session_file = os.path.join(self.data_dir, 'tracking_session.json')
        
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
        
        # 追踪历史记录
        self.tracking_history = []
        self.session_start_time = datetime.now()
        
        print(f"[DATA_RECORDER] 初始化完成，数据目录: {self.data_dir}")
    
    def record_tracking_step(self, target_id: str, tracking_result: Dict, 
                            rgb_image: np.ndarray, depth_image: np.ndarray):
        """
        记录单步追踪数据 - 修复版，确保反馈信息传递
        """
        try:
            step_num = len(self.tracking_history) + 1
            timestamp = datetime.now()
            
            # 保存图像（使用新的扩展布局）
            step_image_dir = os.path.join(self.images_dir, f'step_{step_num:02d}')
            os.makedirs(step_image_dir, exist_ok=True)
            
            rgb_file = os.path.join(step_image_dir, 'rgb_image.jpg')
            depth_file = os.path.join(step_image_dir, 'depth_image.png')
            depth_raw_file = os.path.join(step_image_dir, 'depth_raw.npy')
            
            # 保存带可视化的RGB图像
            annotated_rgb_file = os.path.join(step_image_dir, 'rgb_annotated.jpg')
            self._save_annotated_rgb_image(rgb_image, tracking_result, annotated_rgb_file, step_num)
            
            # 保存原始RGB图像
            rgb_bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(rgb_file, rgb_bgr)
            
            # 保存深度图像
            depth_colormap = self._create_enhanced_depth_colormap(depth_image)
            cv2.imwrite(depth_file, depth_colormap)
            np.save(depth_raw_file, depth_image)
            position_info = self._extract_position_info(tracking_result)
            # 🔧 修复：从tracking_result中提取用户反馈
            user_feedback = tracking_result.get('user_feedback')  # 可能是 'success', 'failure' 或 None
            feedback_timestamp = tracking_result.get('feedback_timestamp')
            
            # 构建记录数据
            record = {
                'step_number': step_num,
                'target_id': target_id,
                'timestamp': timestamp.isoformat(),
                'tracking_result': self._serialize_tracking_result(tracking_result),
                'position_info': position_info,
                'image_files': {
                    'rgb': os.path.relpath(rgb_file, self.data_dir),
                    'rgb_annotated': os.path.relpath(annotated_rgb_file, self.data_dir),
                    'depth_raw': os.path.relpath(depth_raw_file, self.data_dir)
                },
                'image_shape': rgb_image.shape,
                
                # 🔧 修复：确保反馈信息被记录
                'human_feedback': self._convert_feedback_format(user_feedback),  # 转换为 'correct'/'incorrect'
                'feedback_timestamp': feedback_timestamp
            }
            
            self.tracking_history.append(record)
            
            print(f"[DATA_RECORDER] 记录第 {step_num} 步追踪数据")
            if user_feedback:
                print(f"[DATA_RECORDER] 包含用户反馈: {user_feedback}")
            
        except Exception as e:
            print(f"[DATA_RECORDER] 记录追踪步骤失败: {e}")

    # 2. 新增辅助方法：转换反馈格式
    def _convert_feedback_format(self, feedback: str) -> str:
        """转换反馈格式以兼容现有系统"""
        if feedback == 'success':
            return 'correct'
        elif feedback == 'failure':
            return 'incorrect'
        else:
            return None
        
    def _extract_position_info(self, tracking_result: Dict) -> Dict:
        """从追踪结果中提取位置信息"""
        try:
            position_info = {}
            
            # 提取目标抓取坐标
            grasp_coordinate = tracking_result.get('grasp_coordinate', {})
            if grasp_coordinate:
                position_info['target_x'] = float(grasp_coordinate.get('x', 0))
                position_info['target_y'] = float(grasp_coordinate.get('y', 0))
                position_info['target_z'] = float(grasp_coordinate.get('z', 0))
            
            # 提取当前TCP位置（如果有的话）
            current_tcp_pose = tracking_result.get('current_tcp_pose')
            if current_tcp_pose:
                tcp_position = current_tcp_pose.get('position', {})
                if tcp_position:
                    position_info['current_tcp_x'] = float(tcp_position.get('x', 0))
                    position_info['current_tcp_y'] = float(tcp_position.get('y', 0))
                    position_info['current_tcp_z'] = float(tcp_position.get('z', 0))
            
            # 提取对象信息中的高度数据
            object_info = tracking_result.get('object_info', {})
            if object_info:
                position_info['object_height'] = float(object_info.get('estimated_height', 0))
                position_info['background_z'] = float(object_info.get('background_z', 0))
                position_info['estimated_width'] = float(object_info.get('estimated_width', 0))
            
            # 计算移动距离（如果有当前和目标位置）
            if ('current_tcp_x' in position_info and 'target_x' in position_info):
                import math
                distance_2d = math.sqrt(
                    (position_info['target_x'] - position_info['current_tcp_x'])**2 + 
                    (position_info['target_y'] - position_info['current_tcp_y'])**2
                )
                distance_3d = math.sqrt(
                    distance_2d**2 + 
                    (position_info['target_z'] - position_info['current_tcp_z'])**2
                )
                position_info['distance_to_target_2d'] = distance_2d
                position_info['distance_to_target_3d'] = distance_3d
            
            # 添加waypoint数据（如果tracking_result中包含）
            waypoint_data = tracking_result.get('waypoint_data', {})
            if waypoint_data:
                world_pos = waypoint_data.get('world_pos', [])
                if len(world_pos) >= 3:
                    position_info['waypoint_x'] = float(world_pos[0])
                    position_info['waypoint_y'] = float(world_pos[1])
                    position_info['waypoint_z'] = float(world_pos[2])
                
                position_info['waypoint_roll'] = float(waypoint_data.get('roll', 0))
                position_info['waypoint_pitch'] = float(waypoint_data.get('pitch', 0))
                position_info['waypoint_yaw'] = float(waypoint_data.get('yaw', 0))
            
            return position_info
            
        except Exception as e:
            print(f"[DATA_RECORDER] 提取位置信息失败: {e}")
            return {}
    
    def _create_depth_colormap(self, depth_image: np.ndarray) -> np.ndarray:
        """创建深度图像的彩色映射"""
        try:
            # 归一化深度值到0-255范围
            depth_normalized = cv2.normalize(
                depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
            )
            
            # 应用颜色映射
            depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
            
            return depth_colormap
            
        except Exception as e:
            print(f"[DATA_RECORDER] 创建深度颜色映射失败: {e}")
            # 返回灰度图像作为备用
            return cv2.cvtColor(depth_image.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        
        
    def _create_enhanced_depth_colormap(self, depth_image: np.ndarray, 
                                    min_depth: int = 150, max_depth: int = 500) -> np.ndarray:
        """
        创建针对近距离优化的深度图像彩色映射（150-500mm）
        """
        try:
            # 限制深度值到指定范围
            depth_clipped = np.clip(depth_image, min_depth, max_depth)
            
            # 创建mask，只处理有效深度区域
            valid_mask = (depth_image >= min_depth) & (depth_image <= max_depth)
            
            # 将深度值归一化到0-255范围
            depth_range = max_depth - min_depth
            depth_normalized = ((depth_clipped - min_depth) / depth_range * 255).astype(np.uint8)
            
            # 对无效区域设置为0（黑色）
            depth_normalized[~valid_mask] = 0
            
            # 使用更敏感的颜色映射
            depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_TURBO)
            
            # 对无效区域设置为灰色便于识别
            depth_colormap[~valid_mask] = [64, 64, 64]  # 深灰色
            
            return depth_colormap
            
        except Exception as e:
            print(f"[DATA_RECORDER] 创建增强深度颜色映射失败: {e}")
            # 备用方案：使用原始方法
            return self._create_depth_colormap_fallback(depth_image)

    def _create_depth_colormap_fallback(self, depth_image: np.ndarray) -> np.ndarray:
        """备用深度颜色映射方法"""
        try:
            depth_normalized = cv2.normalize(
                depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
            )
            depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
            return depth_colormap
        except Exception as e:
            print(f"[DATA_RECORDER] 备用深度映射也失败: {e}")
            return cv2.cvtColor(depth_image.astype(np.uint8), cv2.COLOR_GRAY2BGR)  
              
    def _serialize_tracking_result(self, tracking_result: Dict) -> Dict:
        """序列化追踪结果，移除不可JSON化的数据"""
        try:
            serialized = {}
            
            # 复制基本数据
            for key, value in tracking_result.items():
                if key in ['target_id', 'detection_confidence', 'tracking_confidence',
                          'match_confidence', 'grasp_coordinate', 'object_info',
                          'timestamp', 'frame_count']:
                    serialized[key] = value
            
            # 处理相似度分解
            if 'similarity_breakdown' in tracking_result:
                serialized['similarity_breakdown'] = tracking_result['similarity_breakdown']
            
            if 'valid_features' in tracking_result:
                serialized['valid_features'] = tracking_result['valid_features']
            
            return serialized
            
        except Exception as e:
            print(f"[DATA_RECORDER] 序列化追踪结果失败: {e}")
            return {'error': str(e)}
    
    def get_tracking_history(self) -> List[Dict]:
        """获取追踪历史记录"""
        return self.tracking_history.copy()
    
    def update_feedback(self, step_number: int, feedback: bool, timestamp: str = None):
        """
        更新指定步骤的反馈 - 修复版，确保更新传播
        
        Args:
            step_number: 步骤编号 (1-based)
            feedback: 反馈结果 (True=正确, False=错误)
            timestamp: 反馈时间戳
        """
        try:
            if 1 <= step_number <= len(self.tracking_history):
                step_index = step_number - 1
                feedback_str = 'correct' if feedback else 'incorrect'
                
                # 🔧 更新tracking_history中的反馈
                self.tracking_history[step_index]['human_feedback'] = feedback_str
                self.tracking_history[step_index]['feedback_timestamp'] = timestamp or datetime.now().isoformat()
                
                print(f"[DATA_RECORDER] 更新步骤 {step_number} session反馈: {feedback_str}")
                
                # 🆕 同时更新tracking_result中的反馈（如果存在）
                if 'tracking_result' in self.tracking_history[step_index]:
                    tr = self.tracking_history[step_index]['tracking_result']
                    tr['user_feedback'] = 'success' if feedback else 'failure'
                    tr['feedback_timestamp'] = timestamp or datetime.now().isoformat()
                
            else:
                print(f"[DATA_RECORDER] 无效的步骤编号: {step_number}")
                
        except Exception as e:
            print(f"[DATA_RECORDER] 更新反馈失败: {e}")

    def update_session_feedback_from_details(self):
        """从详细文件更新session中的反馈信息"""
        try:
            step_details_dir = os.path.join(self.data_dir, 'step_details')
            if not os.path.exists(step_details_dir):
                return
            
            updated_count = 0
            
            # 遍历所有步骤的详细文件
            for step_num in range(1, len(self.tracking_history) + 1):
                detail_file = os.path.join(step_details_dir, f'step_{step_num:02d}_detailed.json')
                
                if os.path.exists(detail_file):
                    try:
                        with open(detail_file, 'r', encoding='utf-8') as f:
                            detail_data = json.load(f)
                        
                        user_feedback = detail_data.get('user_feedback')
                        feedback_timestamp = detail_data.get('feedback_timestamp')
                        
                        if user_feedback:
                            # 更新session历史
                            step_index = step_num - 1
                            if step_index < len(self.tracking_history):
                                # 转换格式
                                session_feedback = self._convert_feedback_format(user_feedback)
                                
                                self.tracking_history[step_index]['human_feedback'] = session_feedback
                                self.tracking_history[step_index]['feedback_timestamp'] = feedback_timestamp
                                
                                updated_count += 1
                                
                    except Exception as file_error:
                        print(f"[DATA_RECORDER] 读取详细文件失败 {step_num}: {file_error}")
            
            print(f"[DATA_RECORDER] 从详细文件同步了 {updated_count} 个反馈到session")
            
        except Exception as e:
            print(f"[DATA_RECORDER] 批量更新session反馈失败: {e}")

    def save_session_data(self):
        """保存会话数据到文件 - 修复版，确保反馈同步"""
        try:
            # 🆕 保存前先从详细文件同步反馈
            self.update_session_feedback_from_details()
            
            session_data = {
                'session_info': {
                    'user_id': self.user_id,
                    'scan_directory': self.scan_dir,
                    'start_time': self.session_start_time.isoformat(),
                    'end_time': datetime.now().isoformat(),
                    'total_steps': len(self.tracking_history)
                },
                'tracking_history': self.tracking_history,
                'statistics': self._calculate_session_statistics()
            }
            
            # 保存到JSON文件
            with open(self.session_file, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False)
            
            print(f"[DATA_RECORDER] 会话数据已保存: {self.session_file}")
            
            # 创建汇总报告
            self._create_session_summary()
            
        except Exception as e:
            print(f"[DATA_RECORDER] 保存会话数据失败: {e}")
    
    def _calculate_session_statistics(self) -> Dict:
        """计算会话统计信息"""
        try:
            total_steps = len(self.tracking_history)
            if total_steps == 0:
                return {'total_steps': 0}
            
            # 成功/失败统计
            successful_steps = [r for r in self.tracking_history if 'tracking_result' in r]
            failed_steps = [r for r in self.tracking_history if r.get('status') == 'tracking_failed']
            
            # 反馈统计
            feedback_steps = [r for r in self.tracking_history if r.get('human_feedback') is not None]
            correct_feedback = [r for r in feedback_steps if r['human_feedback'] == 'correct']
            position_stats = self._calculate_position_statistics()
            # 置信度统计
            confidences = []
            for record in successful_steps:
                if 'tracking_result' in record and 'tracking_confidence' in record['tracking_result']:
                    confidences.append(record['tracking_result']['tracking_confidence'])
            
            statistics = {
                'total_steps': total_steps,
                'successful_steps': len(successful_steps),
                'failed_steps': len(failed_steps),
                'success_rate': len(successful_steps) / total_steps if total_steps > 0 else 0,
                'feedback_coverage': len(feedback_steps) / total_steps if total_steps > 0 else 0,
                'human_accuracy': len(correct_feedback) / len(feedback_steps) if feedback_steps else 0,
                'avg_confidence': np.mean(confidences) if confidences else 0,
                'min_confidence': np.min(confidences) if confidences else 0,
                'max_confidence': np.max(confidences) if confidences else 0,
                'position_statistics': position_stats
            }
            
            return statistics
            
        except Exception as e:
            print(f"[DATA_RECORDER] 计算统计信息失败: {e}")
            return {'error': str(e)}
    
    def _calculate_position_statistics(self) -> Dict:
        """计算位置相关统计"""
        try:
            position_stats = {
                'total_distance_moved': 0,
                'average_target_z': 0,
                'z_range': {'min': float('inf'), 'max': float('-inf')},
                'position_records': []
            }
            
            valid_positions = []
            total_distance = 0
            
            for record in self.tracking_history:
                pos_info = record.get('position_info', {})
                if pos_info and 'target_z' in pos_info:
                    target_z = pos_info['target_z']
                    valid_positions.append(target_z)
                    
                    # 更新Z值范围
                    position_stats['z_range']['min'] = min(position_stats['z_range']['min'], target_z)
                    position_stats['z_range']['max'] = max(position_stats['z_range']['max'], target_z)
                    
                    # 记录位置信息
                    position_record = {
                        'step': record['step_number'],
                        'target_z': target_z,
                        'target_x': pos_info.get('target_x', 0),
                        'target_y': pos_info.get('target_y', 0)
                    }
                    
                    if 'distance_to_target_3d' in pos_info:
                        total_distance += pos_info['distance_to_target_3d']
                        position_record['distance_moved'] = pos_info['distance_to_target_3d']
                    
                    position_stats['position_records'].append(position_record)
            
            if valid_positions:
                position_stats['average_target_z'] = sum(valid_positions) / len(valid_positions)
                position_stats['total_distance_moved'] = total_distance
                position_stats['valid_position_count'] = len(valid_positions)
            
            # 处理无效范围
            if position_stats['z_range']['min'] == float('inf'):
                position_stats['z_range'] = {'min': 0, 'max': 0}
            
            return position_stats
            
        except Exception as e:
            print(f"[DATA_RECORDER] 计算位置统计失败: {e}")
            return {}
        
    def _create_session_summary(self):
        """创建可读的会话摘要 - 修复版，包含反馈统计"""
        try:
            summary_file = os.path.join(self.data_dir, 'session_summary.txt')
            statistics = self._calculate_session_statistics()
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write("追踪会话摘要\n")
                f.write("=" * 50 + "\n\n")
                
                f.write(f"用户ID: {self.user_id}\n")
                f.write(f"扫描目录: {self.scan_dir}\n")
                f.write(f"开始时间: {self.session_start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                f.write("统计信息:\n")
                f.write("-" * 30 + "\n")
                f.write(f"总步数: {statistics.get('total_steps', 0)}\n")
                f.write(f"成功步数: {statistics.get('successful_steps', 0)}\n")
                f.write(f"失败步数: {statistics.get('failed_steps', 0)}\n")
                f.write(f"成功率: {statistics.get('success_rate', 0):.1%}\n")
                f.write(f"平均置信度: {statistics.get('avg_confidence', 0):.3f}\n")
                
                # 🆕 添加反馈统计
                f.write(f"反馈覆盖率: {statistics.get('feedback_coverage', 0):.1%}\n")
                f.write(f"人工准确率: {statistics.get('human_accuracy', 0):.1%}\n\n")
                
                f.write("详细步骤:\n")
                f.write("-" * 30 + "\n")
                
                for i, record in enumerate(self.tracking_history, 1):
                    f.write(f"步骤 {i}: ")
                    
                    if 'tracking_result' in record:
                        confidence = record['tracking_result'].get('tracking_confidence', 0)
                        f.write(f"成功 (置信度: {confidence:.3f})")
                    else:
                        f.write("失败")
                    # 🔧 添加位置信息
                    pos_info = record.get('position_info', {})
                    if pos_info and 'target_z' in pos_info:
                        target_z = pos_info['target_z']
                        f.write(f" - 目标Z: {target_z:.1f}mm")
                        
                        if 'distance_to_target_3d' in pos_info:
                            distance = pos_info['distance_to_target_3d']
                            f.write(f" - 移动: {distance:.1f}mm")
                        
                    # 🆕 添加反馈信息显示
                    feedback = record.get('human_feedback')
                    if feedback == 'correct':
                        f.write(" - 人工反馈: ✓ 正确")
                    elif feedback == 'incorrect':
                        f.write(" - 人工反馈: ✗ 错误")
                    elif feedback:
                        f.write(f" - 人工反馈: {feedback}")
                    else:
                        f.write(" - 人工反馈: 无")
                    
                    f.write("\n")
            
            print(f"[DATA_RECORDER] 会话摘要已保存: {summary_file}")
            
        except Exception as e:
            print(f"[DATA_RECORDER] 创建会话摘要失败: {e}")
    
    def load_session_data(self, session_file_path: str) -> Optional[Dict]:
        """
        加载历史会话数据
        
        Args:
            session_file_path: 会话文件路径
            
        Returns:
            Dict: 会话数据，失败返回None
        """
        try:
            if not os.path.exists(session_file_path):
                print(f"[DATA_RECORDER] 会话文件不存在: {session_file_path}")
                return None
            
            with open(session_file_path, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
            
            print(f"[DATA_RECORDER] 成功加载会话数据: {session_file_path}")
            return session_data
            
        except Exception as e:
            print(f"[DATA_RECORDER] 加载会话数据失败: {e}")
            return None

    def save_step_detailed_data(self, step_number: int, target_id: str, 
                            all_candidates_data: List[Dict], tracking_result: Dict,
                            waypoint_data: Dict, current_tcp_pose: Dict,
                            movement_strategy: Dict, grasp_conditions: Dict,
                            adaptive_weights: Dict):
        """
        保存每步的详细数据到单独文件 - 修复序列化问题，增强数据验证
        """
        try:
            # 创建step_details目录
            step_details_dir = os.path.join(self.data_dir, 'step_details')
            os.makedirs(step_details_dir, exist_ok=True)
            
            print(f"[DATA_RECORDER] 开始保存步骤 {step_number} 详细数据...")
            
            # 🔧 先进行数据清理和验证
            def clean_data(data):
                """清理数据中的问题类型"""
                if data is None:
                    return None
                elif isinstance(data, dict):
                    return {k: clean_data(v) for k, v in data.items()}
                elif isinstance(data, (list, tuple)):
                    return [clean_data(item) for item in data]
                else:
                    return data
            
            # 清理所有输入数据
            all_candidates_data = clean_data(all_candidates_data) or []
            tracking_result = clean_data(tracking_result) or {}
            waypoint_data = clean_data(waypoint_data) or {}
            current_tcp_pose = clean_data(current_tcp_pose) or {}
            movement_strategy = clean_data(movement_strategy) or {}
            grasp_conditions = clean_data(grasp_conditions) or {}
            adaptive_weights = clean_data(adaptive_weights) or {}
            
            # 🔧 序列化所有数据，处理ROS类型
            print(f"[DATA_RECORDER] 序列化数据...")
            serialized_tcp_pose = self._serialize_ros_data(current_tcp_pose)
            serialized_waypoint = self._serialize_ros_data(waypoint_data)
            serialized_tracking_result = self._serialize_ros_data(tracking_result)
            serialized_candidates = self._serialize_ros_data(all_candidates_data)
            serialized_movement = self._serialize_ros_data(movement_strategy)
            serialized_grasp = self._serialize_ros_data(grasp_conditions)
            serialized_weights = self._serialize_ros_data(adaptive_weights)
            
            # 构建详细数据
            detailed_data = {
                'step_number': step_number,
                'target_id': str(target_id) if target_id else 'unknown',
                'timestamp': datetime.now().isoformat(),
                'current_tcp_pose': serialized_tcp_pose,
                'waypoint_data': serialized_waypoint,
                'adaptive_weights_used': serialized_weights,
                
                # 所有候选数据
                'all_candidates': serialized_candidates,
                
                # 追踪结果（简化版）
                'tracking_result': {
                    'target_id': str(serialized_tracking_result.get('target_id', 'unknown')) if serialized_tracking_result else 'unknown',
                    'detection_confidence': float(serialized_tracking_result.get('detection_confidence', 0)) if serialized_tracking_result else 0.0,
                    'tracking_confidence': float(serialized_tracking_result.get('tracking_confidence', 0)) if serialized_tracking_result else 0.0,
                    'grasp_coordinate': serialized_tracking_result.get('grasp_coordinate', {}) if serialized_tracking_result else {},
                    'object_info': serialized_tracking_result.get('object_info', {}) if serialized_tracking_result else {},
                    'bounding_rect': serialized_tracking_result.get('bounding_rect', {}) if serialized_tracking_result else {}
                },
                
                'movement_strategy': serialized_movement,
                'grasp_conditions': serialized_grasp,
                'user_feedback': None,
                'feedback_timestamp': None
            }
            
            # 🔧 测试序列化（预先检查）
            try:
                test_json = json.dumps(detailed_data)
                print(f"[DATA_RECORDER] 序列化测试通过，数据大小: {len(test_json)} bytes")
            except Exception as test_error:
                print(f"[DATA_RECORDER] 序列化测试失败: {test_error}")
                print(f"[DATA_RECORDER] 尝试更激进的数据清理...")
                
                # 更激进的清理：移除可能有问题的字段
                detailed_data = {
                    'step_number': step_number,
                    'target_id': str(target_id) if target_id else 'unknown',
                    'timestamp': datetime.now().isoformat(),
                    'user_feedback': None,
                    'feedback_timestamp': None,
                    'error_recovery': True,
                    'original_error': str(test_error)
                }
            
            # 保存到文件
            detail_file = os.path.join(step_details_dir, f'step_{step_number:02d}_detailed.json')
            with open(detail_file, 'w', encoding='utf-8') as f:
                json.dump(detailed_data, f, indent=2, ensure_ascii=False)
            
            print(f"[DATA_RECORDER] 保存步骤 {step_number} 详细数据成功: {detail_file}")
            
        except Exception as e:
            print(f"[DATA_RECORDER] 保存详细数据失败: {e}")
            import traceback
            traceback.print_exc()
            
            # 🔧 最后的备用方案：保存基本信息
            try:
                step_details_dir = os.path.join(self.data_dir, 'step_details')
                os.makedirs(step_details_dir, exist_ok=True)
                
                minimal_data = {
                    'step_number': step_number,
                    'target_id': str(target_id) if target_id else 'unknown',
                    'timestamp': datetime.now().isoformat(),
                    'error': 'failed_to_save_detailed_data',
                    'error_message': str(e),
                    'user_feedback': None,
                    'feedback_timestamp': None
                }
                
                detail_file = os.path.join(step_details_dir, f'step_{step_number:02d}_detailed.json')
                with open(detail_file, 'w', encoding='utf-8') as f:
                    json.dump(minimal_data, f, indent=2, ensure_ascii=False)
                    
                print(f"[DATA_RECORDER] 保存最小数据作为备用: {detail_file}")
            except:
                print(f"[DATA_RECORDER] 连备用保存也失败了")
    def update_step_detailed_feedback(self, step_number: int, feedback: str, timestamp: str = None):
        """
        更新指定步骤详细文件中的反馈 - 修复版，同步到session
        """
        try:
            step_details_dir = os.path.join(self.data_dir, 'step_details')
            detail_file = os.path.join(step_details_dir, f'step_{step_number:02d}_detailed.json')
            
            if os.path.exists(detail_file):
                try:
                    # 读取现有数据
                    with open(detail_file, 'r', encoding='utf-8') as f:
                        detailed_data = json.load(f)
                    
                    # 创建备份
                    backup_file = detail_file + '.backup'
                    with open(backup_file, 'w', encoding='utf-8') as f:
                        json.dump(detailed_data, f, indent=2, ensure_ascii=False)
                    
                except json.JSONDecodeError as je:
                    print(f"[DATA_RECORDER] JSON解析失败: {je}")
                    detailed_data = {
                        'step_number': step_number,
                        'error_recovery': True,
                        'original_error': str(je)
                    }
                
                # 更新反馈
                detailed_data['user_feedback'] = feedback
                detailed_data['feedback_timestamp'] = timestamp or datetime.now().isoformat()
                
                # 序列化并保存详细文件
                serialized_data = self._serialize_ros_data(detailed_data)
                with open(detail_file, 'w', encoding='utf-8') as f:
                    json.dump(serialized_data, f, indent=2, ensure_ascii=False)
                
                print(f"[DATA_RECORDER] 更新步骤 {step_number} 详细文件反馈: {feedback}")
                
                # 🆕 同步更新session历史
                feedback_bool = (feedback == 'success')
                self.update_feedback(step_number, feedback_bool, timestamp)
                
                # 删除备份文件
                if os.path.exists(backup_file):
                    os.remove(backup_file)
                
                # 同时更新标注图像
                self.update_annotated_image_with_feedback(step_number, feedback)
                
                print(f"[DATA_RECORDER] 步骤 {step_number} 反馈已同步到所有组件")
                    
            else:
                print(f"[DATA_RECORDER] 详细文件不存在: {detail_file}")
                
        except Exception as e:
            print(f"[DATA_RECORDER] 更新详细反馈失败: {e}")

    def check_feedback_consistency(self):
        """检查反馈数据的一致性"""
        try:
            print(f"\n[DATA_RECORDER] 反馈一致性检查:")
            print(f"Session历史记录数: {len(self.tracking_history)}")
            
            session_feedback_count = 0
            detail_feedback_count = 0
            
            # 检查session中的反馈
            for i, record in enumerate(self.tracking_history, 1):
                session_feedback = record.get('human_feedback')
                if session_feedback:
                    session_feedback_count += 1
                    print(f"  步骤 {i} session反馈: {session_feedback}")
            
            # 检查详细文件中的反馈
            step_details_dir = os.path.join(self.data_dir, 'step_details')
            if os.path.exists(step_details_dir):
                for step_num in range(1, len(self.tracking_history) + 1):
                    detail_file = os.path.join(step_details_dir, f'step_{step_num:02d}_detailed.json')
                    if os.path.exists(detail_file):
                        try:
                            with open(detail_file, 'r', encoding='utf-8') as f:
                                detail_data = json.load(f)
                            
                            detail_feedback = detail_data.get('user_feedback')
                            if detail_feedback:
                                detail_feedback_count += 1
                                print(f"  步骤 {step_num} detail反馈: {detail_feedback}")
                                
                        except Exception as e:
                            print(f"  步骤 {step_num} detail文件读取失败: {e}")
            
            print(f"Session反馈数: {session_feedback_count}")
            print(f"Detail反馈数: {detail_feedback_count}")
            
            if session_feedback_count != detail_feedback_count:
                print(f"⚠️ 反馈数量不一致，建议运行同步")
            else:
                print(f"✅ 反馈数量一致")
                
        except Exception as e:
            print(f"[DATA_RECORDER] 反馈一致性检查失败: {e}")
    # 2. 修复 JSON 序列化问题 - 添加辅助方法
    def _serialize_ros_data(self, data):
        """
        递归序列化数据，处理ROS特定类型和所有不可序列化类型 - 完整修复版
        """
        if data is None:
            return None
        elif hasattr(data, 'sec') and hasattr(data, 'nanosec'):  # ROS Time 对象
            return {
                'sec': int(data.sec),
                'nanosec': int(data.nanosec),
                'timestamp_iso': datetime.fromtimestamp(data.sec + data.nanosec * 1e-9).isoformat()
            }
        elif isinstance(data, dict):
            return {key: self._serialize_ros_data(value) for key, value in data.items()}
        elif isinstance(data, (list, tuple)):
            return [self._serialize_ros_data(item) for item in data]
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(data)
        elif isinstance(data, (np.floating, np.float64, np.float32, np.float16)):
            return float(data)
        elif isinstance(data, (np.bool_, bool)):  # 🔧 修复：处理numpy bool和Python bool
            return bool(data)
        elif isinstance(data, (np.str_, str)):
            return str(data)
        elif hasattr(data, 'tolist'):  # 其他numpy类型
            return data.tolist()
        elif hasattr(data, '__dict__'):  # 自定义对象
            try:
                return self._serialize_ros_data(data.__dict__)
            except:
                return str(data)
        else:
            # 🔧 对于其他类型，尝试转换为基本类型
            try:
                if isinstance(data, (int, float, str, bool)):
                    return data
                elif hasattr(data, '__iter__') and not isinstance(data, (str, bytes)):
                    return [self._serialize_ros_data(item) for item in data]
                else:
                    return str(data)  # 最后的备用方案
            except:
                return str(data)
    def record_detection_retry_attempt(self, target_id: str, attempt_number: int, 
                                    rgb_image: np.ndarray = None, depth_image: np.ndarray = None):
        """
        记录检测重试尝试
        
        Args:
            target_id: 目标ID
            attempt_number: 尝试次数
            rgb_image: RGB图像（可选）
            depth_image: 深度图像（可选）
        """
        try:
            timestamp = datetime.now()
            
            retry_record = {
                'timestamp': timestamp.isoformat(),
                'target_id': target_id,
                'attempt_number': attempt_number,
                'event_type': 'detection_retry',
                'total_steps_so_far': len(self.tracking_history)
            }
            
            # 如果提供了图像，保存图像
            if rgb_image is not None and depth_image is not None:
                retry_dir = os.path.join(self.images_dir, f'retry_{attempt_number}_{int(timestamp.timestamp())}')
                os.makedirs(retry_dir, exist_ok=True)
                
                rgb_file = os.path.join(retry_dir, 'rgb_image.jpg')
                depth_file = os.path.join(retry_dir, 'depth_raw.npy')
                
                rgb_bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(rgb_file, rgb_bgr)
                np.save(depth_file, depth_image)
                
                retry_record['image_files'] = {
                    'rgb': os.path.relpath(rgb_file, self.data_dir),
                    'depth_raw': os.path.relpath(depth_file, self.data_dir)
                }
            
            # 保存重试记录到单独的文件
            retry_log_file = os.path.join(self.data_dir, 'retry_attempts.json')
            
            if os.path.exists(retry_log_file):
                with open(retry_log_file, 'r', encoding='utf-8') as f:
                    retry_log = json.load(f)
            else:
                retry_log = []
            
            retry_log.append(retry_record)
            
            with open(retry_log_file, 'w', encoding='utf-8') as f:
                json.dump(retry_log, f, indent=2, ensure_ascii=False)
            
            print(f"[DATA_RECORDER] 记录检测重试尝试 #{attempt_number}")
            
        except Exception as e:
            print(f"[DATA_RECORDER] 记录重试尝试失败: {e}")

    def set_tracking_context(self, consecutive_failures: int = None, similarity_threshold: float = None):
        """
        设置追踪上下文信息，用于失败记录
        
        Args:
            consecutive_failures: 连续失败次数
            similarity_threshold: 当前使用的相似度阈值
        """
        if consecutive_failures is not None:
            self._consecutive_failures = consecutive_failures
        if similarity_threshold is not None:
            self._last_similarity_threshold = similarity_threshold

    def get_failure_statistics(self) -> Dict:
        """
        获取失败统计信息
        
        Returns:
            Dict: 失败统计数据
        """
        try:
            failed_steps = [r for r in self.tracking_history if r.get('status') == 'tracking_failed']
            total_steps = len(self.tracking_history)
            
            failure_reasons = {}
            for step in failed_steps:
                reason = step.get('failure_reason', 'unknown')
                failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
            
            statistics = {
                'total_failures': len(failed_steps),
                'total_steps': total_steps,
                'failure_rate': len(failed_steps) / total_steps if total_steps > 0 else 0,
                'failure_reasons': failure_reasons,
                'failure_steps': [step['step_number'] for step in failed_steps]
            }
            
            return statistics
            
        except Exception as e:
            print(f"[DATA_RECORDER] 获取失败统计失败: {e}")
            return {'error': str(e)}
        
    def _save_annotated_rgb_image(self, rgb_image: np.ndarray, tracking_result: Dict, 
                                output_file: str, step_number: int):
        """
        保存带有bbox和置信度标注的RGB图像 - 修复版，支持反馈信息
        """
        try:
            # 🔧 调整区域大小 - 增加黑色标注区域，减少绿色原图区域
            header_height = 150  # 增加顶部标注区域（原150 -> 200）
            bottom_height = 80  # 增加底部标注区域（原100 -> 150）
            
            # 🆕 检查是否有反馈信息需要额外空间
            user_feedback = tracking_result.get('user_feedback')
            if user_feedback:
                header_height += 20  # 增加反馈区域（原60 -> 80）
            
            original_h, original_w = rgb_image.shape[:2]
            
            # 🔧 可选：缩放原始图像，进一步减少绿色区域
            scale_factor = 0.9  # 将原图缩放到80%
            scaled_h = int(original_h * scale_factor)
            scaled_w = int(original_w * scale_factor)
            
            # 缩放原始图像
            scaled_rgb = cv2.resize(rgb_image, (scaled_w, scaled_h))
            
            new_h = scaled_h + header_height + bottom_height
            new_w = max(original_w, scaled_w)  # 保持画布宽度
            
            # 创建扩展的画布（黑色背景）
            extended_image = np.zeros((new_h, new_w, 3), dtype=np.uint8)
            
            # 🔧 将缩放后的图像居中放置
            x_offset = (new_w - scaled_w) // 2
            extended_image[header_height:header_height + scaled_h, x_offset:x_offset + scaled_w] = scaled_rgb
            
            # 转换为BGR用于OpenCV
            annotated_bgr = cv2.cvtColor(extended_image, cv2.COLOR_RGB2BGR)
            
            # 提取追踪结果信息
            target_id = tracking_result.get('target_id', 'unknown')
            tracking_confidence = tracking_result.get('tracking_confidence', 0.0)
            detection_confidence = tracking_result.get('detection_confidence', 0.0)
            grasp_coordinate = tracking_result.get('grasp_coordinate', {})
            all_candidates = tracking_result.get('all_candidates_analysis', [])
            
            # 在扩展画布上绘制标注
            self._draw_all_candidates_bbox_extended(
            annotated_bgr, all_candidates, header_height, scale_factor, x_offset)
            self._draw_best_match_bbox_extended(
            annotated_bgr, all_candidates, tracking_result.get('tracking_confidence', 0),
            header_height, original_h, original_w, scale_factor, x_offset)
            
            # 绘制抓取点
            if grasp_coordinate and 'x' in grasp_coordinate:
                grasp_pixel_x, grasp_pixel_y = original_w // 2, header_height + original_h // 2
                cv2.drawMarker(annotated_bgr, (grasp_pixel_x, grasp_pixel_y), 
                            (0, 255, 255), cv2.MARKER_CROSS, 20, 3)
            
            # 添加标注到顶部和底部区域
            self._add_header_annotations_with_feedback(annotated_bgr, target_id, tracking_confidence, 
                                                    detection_confidence, step_number, grasp_coordinate, 
                                                    header_height, new_w, user_feedback)
            
            self._add_bottom_annotations(annotated_bgr, tracking_result, 
                                    header_height + original_h, bottom_height, new_w)
            
            # 保存图像
            cv2.imwrite(output_file, annotated_bgr)
            
            print(f"[DATA_RECORDER] 扩展布局标注图像已保存: {output_file}")
            if user_feedback:
                print(f"[DATA_RECORDER] 图像包含用户反馈: {user_feedback}")
            
        except Exception as e:
            print(f"[DATA_RECORDER] 保存标注图像失败: {e}")
            import traceback
            traceback.print_exc()

    def _add_header_annotations_with_feedback(self, image: np.ndarray, target_id: str, 
                                            tracking_confidence: float, detection_confidence: float,
                                            step_number: int, grasp_coordinate: Dict, 
                                            header_height: int, img_width: int, user_feedback: str = None):
        """添加带反馈信息的顶部标注"""
        try:
            # 处理坐标信息
            coord_text = "Grasp: (No coordinates)"
            if grasp_coordinate and isinstance(grasp_coordinate, dict):
                try:
                    x = float(grasp_coordinate.get('x', 0))
                    y = float(grasp_coordinate.get('y', 0))
                    z = float(grasp_coordinate.get('z', 0))
                    coord_text = f"Grasp: ({x:.1f}, {y:.1f}, {z:.1f})"
                except:
                    coord_text = "Grasp: (Invalid coordinates)"
            
            annotations = [
                f"Step: {step_number}",
                f"Target: {str(target_id)}",
                f"Track Conf: {float(tracking_confidence):.3f}",
                f"Detect Conf: {float(detection_confidence):.3f}",
                coord_text
            ]
            
            # 🆕 根据是否有反馈调整布局
            feedback_start_y = 90 if user_feedback else header_height
            
            # 绘制顶部背景
            cv2.rectangle(image, (0, 0), (img_width, feedback_start_y), (0, 0, 0), -1)
            cv2.rectangle(image, (0, 0), (img_width, feedback_start_y), (255, 255, 255), 2)
            
            # 添加标题
            cv2.putText(image, "TRACKING ANALYSIS", (10, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # 添加基础信息
            for i, text in enumerate(annotations):
                y_pos = 45 + i * 16
                
                # 颜色选择
                if i == 2:  # 追踪置信度
                    if tracking_confidence > 0.8:
                        color = (0, 255, 0)
                    elif tracking_confidence > 0.6:
                        color = (0, 255, 255)
                    else:
                        color = (0, 165, 255)
                else:
                    color = (255, 255, 255)
                
                cv2.putText(image, str(text), (10, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # 🆕 添加反馈信息区域
            if user_feedback:
                feedback_y_start = feedback_start_y
                feedback_height = header_height - feedback_start_y
                
                # 根据反馈类型选择颜色
                if user_feedback == 'success':
                    bg_color = (0, 120, 0)  # 深绿色
                    text_color = (255, 255, 255)
                    feedback_text = "USER CONFIRMED: SUCCESS"
                elif user_feedback == 'failure':
                    bg_color = (0, 0, 150)  # 深红色
                    text_color = (255, 255, 255)
                    feedback_text = "USER MARKED: FAILED"
                else:
                    bg_color = (80, 80, 80)  # 灰色
                    text_color = (255, 255, 255)
                    feedback_text = f"USER FEEDBACK: {user_feedback.upper()}"
                
                # 绘制反馈背景
                cv2.rectangle(image, (0, feedback_y_start), (img_width, header_height), bg_color, -1)
                cv2.rectangle(image, (0, feedback_y_start), (img_width, header_height), text_color, 2)
                
                # 添加反馈文字
                cv2.putText(image, feedback_text, (10, feedback_y_start + 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, text_color, 2)
                        
        except Exception as e:
            print(f"[DATA_RECORDER] 带反馈的顶部标注失败: {e}")

    # 3. 绘制所有候选的bbox
    def _draw_all_candidates_bbox_extended(self, image: np.ndarray, candidates: List[Dict], 
                                        y_offset: int, scale_factor: float = 1.0, x_offset: int = 0):
        """绘制所有候选的边界框 - 扩展画布版，支持缩放和偏移"""
        try:
            for i, candidate in enumerate(candidates):
                detection_data = candidate.get('detection_data', {})
                bbox = detection_data.get('bounding_box', [])
                class_name = detection_data.get('class_name', 'unknown')
                is_best_match = candidate.get('is_best_match', False)
                was_analyzed = candidate.get('was_analyzed', False)
                
                if len(bbox) >= 4:
                    x1, y1, x2, y2 = map(int, bbox[:4])
                    
                    # 🔧 应用缩放和偏移调整
                    x1 = int(x1 * scale_factor) + x_offset
                    y1 = int(y1 * scale_factor) + y_offset
                    x2 = int(x2 * scale_factor) + x_offset
                    y2 = int(y2 * scale_factor) + y_offset
                    
                    # 选择颜色
                    if not was_analyzed:
                        color = (128, 128, 128)
                        thickness = 1
                    elif is_best_match:
                        continue  # 最佳匹配单独绘制
                    else:
                        color = (200, 200, 200)
                        thickness = 1
                    
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
                    
                    if not is_best_match:
                        text = f"#{i} {class_name}"
                        # 🔧 文字位置也需要调整
                        text_y = max(y1 - 5, 10)  # 确保文字不超出边界
                        cv2.putText(image, text, (x1, text_y), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                        
        except Exception as e:
            print(f"[DATA_RECORDER] 绘制扩展候选bbox失败: {e}")

    # 3. 新增方法：绘制最佳匹配bbox（智能位置调整）
    def _draw_best_match_bbox_extended(self, image: np.ndarray, candidates: List[Dict], 
                                    tracking_confidence: float, y_offset: int, 
                                    original_h: int, original_w: int, 
                                    scale_factor: float = 1.0, x_offset: int = 0):
        """绘制最佳匹配的边界框 - 扩展画布版，支持缩放和偏移"""
        try:
            # 找到最佳匹配
            best_candidate = None
            for candidate in candidates:
                if candidate.get('is_best_match', False):
                    best_candidate = candidate
                    break
            
            if not best_candidate:
                return
            
            detection_data = best_candidate.get('detection_data', {})
            bbox = detection_data.get('bounding_box', [])
            confidence = detection_data.get('confidence', 0.0)
            class_name = detection_data.get('class_name', 'unknown')
            similarity_data = best_candidate.get('similarity_to_target', {})
            final_score = similarity_data.get('final_score', 0.0)
            
            if len(bbox) >= 4:
                x1, y1, x2, y2 = map(int, bbox[:4])
                
                # 🔧 应用缩放和偏移调整
                x1 = int(x1 * scale_factor) + x_offset
                y1 = int(y1 * scale_factor) + y_offset
                x2 = int(x2 * scale_factor) + x_offset
                y2 = int(y2 * scale_factor) + y_offset
                
                # 选择颜色
                if tracking_confidence > 0.8:
                    color = (0, 255, 0)
                elif tracking_confidence > 0.6:
                    color = (0, 255, 255)
                else:
                    color = (0, 165, 255)
                
                thickness = 3
                cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
                
                # 🔧 智能角标位置（考虑缩放后的坐标）
                scaled_original_h = int(original_h * scale_factor)
                scaled_original_w = int(original_w * scale_factor)
                
                marker_x, marker_y = self._get_smart_marker_position(
                    x1, y1, x2, y2, y_offset, scaled_original_h, scaled_original_w, x_offset
                )
                cv2.circle(image, (marker_x, marker_y), 8, color, -1)
                cv2.putText(image, "T", (marker_x - 3, marker_y + 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # 🔧 智能信息框位置
                info_text = [
                    f"Target: {str(class_name)}",
                    f"Track: {float(tracking_confidence):.3f}",
                    f"Detect: {float(confidence):.3f}",
                    f"Similar: {float(final_score):.3f}"
                ]
                
                info_width = 200
                info_height = len(info_text) * 20 + 10
                
                # 智能选择信息框位置
                info_x, info_y = self._get_smart_info_position(
                    x1, y1, x2, y2, info_width, info_height,
                    y_offset, scaled_original_h, scaled_original_w, 
                    image.shape[0], x_offset
                )
                
                # 绘制信息框
                cv2.rectangle(image, (info_x, info_y), (info_x + info_width, info_y + info_height), 
                            (0, 0, 0), -1)
                cv2.rectangle(image, (info_x, info_y), (info_x + info_width, info_y + info_height), 
                            color, 2)
                
                # 添加信息文字
                for i, text in enumerate(info_text):
                    try:
                        y_pos = info_y + 20 + i * 20
                        safe_text = str(text)
                        cv2.putText(image, safe_text, (info_x + 5, y_pos), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    except Exception as text_error:
                        print(f"[DEBUG] 信息框文字失败: {text_error}")
                            
        except Exception as e:
            print(f"[DATA_RECORDER] 绘制扩展最佳匹配失败: {e}")
    # 4. 新增方法：智能角标位置
    def _get_smart_marker_position(self, x1: int, y1: int, x2: int, y2: int,
                                y_offset: int, original_h: int, original_w: int, 
                                x_offset: int = 0) -> Tuple[int, int]:
        """智能选择角标位置，避免超出边界 - 支持偏移版本"""
        positions = [
            (x1, y1),           # 左上角
            (x2, y1),           # 右上角  
            (x1, y2),           # 左下角
            (x2, y2),           # 右下角
        ]
        
        # 考虑偏移和缩放后的边界
        min_x = x_offset + 10
        max_x = x_offset + original_w - 10
        min_y = y_offset + 10
        max_y = y_offset + original_h - 10
        
        for pos_x, pos_y in positions:
            if (min_x <= pos_x <= max_x and min_y <= pos_y <= max_y):
                return pos_x, pos_y
        
        # 如果都超出边界，固定在bbox中心
        return (x1 + x2) // 2, (y1 + y2) // 2

    def _get_smart_info_position(self, x1: int, y1: int, x2: int, y2: int,
                            info_width: int, info_height: int,
                            y_offset: int, original_h: int, original_w: int,
                            total_height: int, x_offset: int = 0) -> Tuple[int, int]:
        """智能选择信息框位置 - 支持偏移版本"""
        # 尝试的位置顺序：右下、右上、左下、左上、底部区域
        positions = [
            (x2 + 5, y2 + 5),                    # 右下
            (x2 + 5, y1 - info_height - 5),     # 右上
            (x1 - info_width - 5, y2 + 5),      # 左下  
            (x1 - info_width - 5, y1 - info_height - 5), # 左上
        ]
        
        # 考虑偏移后的边界
        min_x = x_offset
        max_x = x_offset + original_w - info_width
        min_y = y_offset
        max_y = y_offset + original_h - info_height
        
        # 检查每个位置是否在边界内
        for pos_x, pos_y in positions:
            if (min_x <= pos_x <= max_x and min_y <= pos_y <= max_y):
                return pos_x, pos_y
        
        # 🔧 如果都不合适，放在底部区域
        bottom_y = y_offset + original_h + 10
        bottom_x = max(x_offset + 5, min(x1, x_offset + original_w - info_width - 5))
        
        if bottom_y + info_height <= total_height - 10:
            return bottom_x, bottom_y
        
        # 最后备用：图像右下角
        return max(x_offset, x_offset + original_w - info_width - 5), \
            max(y_offset, y_offset + original_h - info_height - 5)

    # 6. 新增方法：顶部标注
    def _add_header_annotations(self, image: np.ndarray, target_id: str, 
                            tracking_confidence: float, detection_confidence: float,
                            step_number: int, grasp_coordinate: Dict, 
                            header_height: int, img_width: int):
        """添加顶部标注信息"""
        try:
            # 处理坐标信息
            coord_text = "Grasp: (No coordinates)"
            if grasp_coordinate and isinstance(grasp_coordinate, dict):
                try:
                    x = float(grasp_coordinate.get('x', 0))
                    y = float(grasp_coordinate.get('y', 0))
                    z = float(grasp_coordinate.get('z', 0))
                    coord_text = f"Grasp: ({x:.1f}, {y:.1f}, {z:.1f})"
                except:
                    coord_text = "Grasp: (Invalid coordinates)"
            
            annotations = [
                f"Step: {step_number}",
                f"Target: {str(target_id)}",
                f"Track Conf: {float(tracking_confidence):.3f}",
                f"Detect Conf: {float(detection_confidence):.3f}",
                coord_text
            ]
            
            # 绘制顶部背景
            cv2.rectangle(image, (0, 0), (img_width, header_height), (0, 0, 0), -1)
            cv2.rectangle(image, (0, 0), (img_width, header_height), (255, 255, 255), 2)
            
            # 添加标题
            cv2.putText(image, "TRACKING ANALYSIS", (10, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # 添加信息
            for i, text in enumerate(annotations):
                y_pos = 55 + i * 18
                
                # 颜色选择
                if i == 2:  # 追踪置信度
                    if tracking_confidence > 0.8:
                        color = (0, 255, 0)
                    elif tracking_confidence > 0.6:
                        color = (0, 255, 255)
                    else:
                        color = (0, 165, 255)
                else:
                    color = (255, 255, 255)
                
                cv2.putText(image, str(text), (10, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
                        
        except Exception as e:
            print(f"[DATA_RECORDER] 顶部标注失败: {e}")

    # 7. 新增方法：底部标注
    def _add_bottom_annotations(self, image: np.ndarray, tracking_result: Dict,
                            bottom_start_y: int, bottom_height: int, img_width: int):
        """添加底部标注信息"""
        try:
            # 绘制底部背景
            cv2.rectangle(image, (0, bottom_start_y), (img_width, bottom_start_y + bottom_height), 
                        (40, 40, 40), -1)
            cv2.rectangle(image, (0, bottom_start_y), (img_width, bottom_start_y + bottom_height), 
                        (255, 255, 255), 1)
            
            # 添加时间戳和其他信息
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(image, f"Generated: {timestamp}", (10, bottom_start_y + 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # 候选统计
            candidates = tracking_result.get('all_candidates_analysis', [])
            analyzed_count = len([c for c in candidates if c.get('was_analyzed', False)])
            cv2.putText(image, f"Candidates: {len(candidates)} total, {analyzed_count} analyzed", 
                    (10, bottom_start_y + 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # 权重信息
            weights = tracking_result.get('adaptive_weights_used', {})
            if weights:
                weight_text = " | ".join([f"{k}:{v:.2f}" for k, v in weights.items()])
                cv2.putText(image, f"Weights: {weight_text}", (10, bottom_start_y + 75), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
                        
        except Exception as e:
            print(f"[DATA_RECORDER] 底部标注失败: {e}")


    # 5. 添加文字标注
    def _add_text_annotations(self, image: np.ndarray, target_id: str, 
                            tracking_confidence: float, detection_confidence: float,
                            step_number: int, grasp_coordinate: Dict):
        """添加文字标注信息"""
        try:
            img_h, img_w = image.shape[:2]
            
            # 准备标注信息
            annotations = [
                f"Step: {step_number}",
                f"Target: {target_id}",
                f"Track Conf: {tracking_confidence:.3f}",
                f"Detect Conf: {detection_confidence:.3f}",
            ]
            
            # 添加坐标信息
            if grasp_coordinate and 'x' in grasp_coordinate:
                coord_text = f"Grasp: ({grasp_coordinate['x']:.1f}, {grasp_coordinate['y']:.1f}, {grasp_coordinate['z']:.1f})"
                annotations.append(coord_text)
            
            # 绘制顶部信息栏背景
            header_height = len(annotations) * 25 + 20
            cv2.rectangle(image, (0, 0), (img_w, header_height), (0, 0, 0), -1)  # 黑色背景
            cv2.rectangle(image, (0, 0), (img_w, header_height), (255, 255, 255), 2)  # 白色边框
            
            # 添加标注文字
            for i, text in enumerate(annotations):
                y_pos = 25 + i * 25
                
                # 根据置信度选择文字颜色
                if i == 2:  # 追踪置信度行
                    if tracking_confidence > 0.8:
                        text_color = (0, 255, 0)  # 绿色
                    elif tracking_confidence > 0.6:
                        text_color = (0, 255, 255)  # 黄色
                    else:
                        text_color = (0, 165, 255)  # 橙色
                else:
                    text_color = (255, 255, 255)  # 白色
                
                cv2.putText(image, text, (10, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
            
            # 🆕 添加时间戳
            timestamp_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(image, timestamp_text, (img_w - 200, img_h - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                    
        except Exception as e:
            print(f"[DATA_RECORDER] 添加文字标注失败: {e}")

    # 6. 同时修改失败情况的图像保存
    def record_tracking_failure(self, target_id: str, rgb_image: np.ndarray, 
                            depth_image: np.ndarray, failure_reason: str = None):
        """
        记录追踪失败数据 - 也添加可视化
        """
        try:
            step_num = len(self.tracking_history) + 1
            timestamp = datetime.now()
            
            # 保存失败时的图像
            failure_dir = os.path.join(self.images_dir, f'failure_{step_num:02d}')
            os.makedirs(failure_dir, exist_ok=True)
            
            rgb_file = os.path.join(failure_dir, 'rgb_image.jpg')
            rgb_annotated_file = os.path.join(failure_dir, 'rgb_annotated.jpg')
            depth_file = os.path.join(failure_dir, 'depth_image.png')
            depth_raw_file = os.path.join(failure_dir, 'depth_raw.npy')
            
            # 🆕 保存带失败标注的RGB图像
            self._save_failure_annotated_image(rgb_image, target_id, failure_reason, 
                                            rgb_annotated_file, step_num)
            
            # 保存原始RGB图像
            rgb_bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(rgb_file, rgb_bgr)
            
            # 保存深度图像
            depth_colormap = self._create_enhanced_depth_colormap(depth_image)
            cv2.imwrite(depth_file, depth_colormap)
            np.save(depth_raw_file, depth_image)
            
            # 记录失败数据
            failure_record = {
                'step_number': step_num,
                'target_id': target_id,
                'timestamp': timestamp.isoformat(),
                'status': 'tracking_failed',
                'failure_reason': failure_reason or 'no_suitable_match_found',
                'image_files': {
                    'rgb': os.path.relpath(rgb_file, self.data_dir),
                    'rgb_annotated': os.path.relpath(rgb_annotated_file, self.data_dir),
                    'depth_colormap': os.path.relpath(depth_file, self.data_dir),
                    'depth_raw': os.path.relpath(depth_raw_file, self.data_dir)
                },
                'image_shape': rgb_image.shape,
                'human_feedback': None,
                'feedback_timestamp': None,
                'failure_details': {
                    'consecutive_failures': getattr(self, '_consecutive_failures', 1),
                    'detection_attempted': True,
                    'similarity_threshold_used': getattr(self, '_last_similarity_threshold', 0.0)
                }
            }
            
            self.tracking_history.append(failure_record)
            
            print(f"[DATA_RECORDER] 记录第 {step_num} 步追踪失败数据")
            print(f"[DATA_RECORDER] 失败原因: {failure_reason}")
            
        except Exception as e:
            print(f"[DATA_RECORDER] 记录追踪失败失败: {e}")
            import traceback
            traceback.print_exc()

    # 7. 失败情况的标注图像
    def _save_failure_annotated_image(self, rgb_image: np.ndarray, target_id: str,
                                    failure_reason: str, output_file: str, step_number: int):
        """保存失败情况的标注图像"""
        try:
            annotated_image = rgb_image.copy()
            annotated_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
            
            img_h, img_w = annotated_bgr.shape[:2]
            
            # 添加失败标注
            failure_info = [
                f"Step: {step_number} - FAILED",
                f"Target: {target_id}",
                f"Reason: {failure_reason or 'Unknown'}",
                f"Time: {datetime.now().strftime('%H:%M:%S')}"
            ]
            
            # 绘制失败信息框（红色背景）
            header_height = len(failure_info) * 30 + 20
            cv2.rectangle(annotated_bgr, (0, 0), (img_w, header_height), (0, 0, 150), -1)  # 红色背景
            cv2.rectangle(annotated_bgr, (0, 0), (img_w, header_height), (255, 255, 255), 3)  # 白色边框
            
            # 添加失败信息文字
            for i, text in enumerate(failure_info):
                y_pos = 30 + i * 30
                cv2.putText(annotated_bgr, text, (15, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # 添加大型失败标记
            cv2.putText(annotated_bgr, "TRACKING FAILED", (img_w//4, img_h//2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
            
            cv2.imwrite(output_file, annotated_bgr)
            print(f"[DATA_RECORDER] 失败标注图像已保存: {output_file}")
            
        except Exception as e:
            print(f"[DATA_RECORDER] 保存失败标注图像失败: {e}")

    def update_annotated_image_with_feedback(self, step_number: int, feedback: str):
        """
        更新已保存的标注图像，添加用户反馈信息 - 简化版
        """
        try:
            # 由于图像保存时已经包含反馈信息，这里只需要记录日志
            step_image_dir = os.path.join(self.images_dir, f'step_{step_number:02d}')
            annotated_file = os.path.join(step_image_dir, 'rgb_annotated.jpg')
            
            if os.path.exists(annotated_file):
                print(f"[DATA_RECORDER] 标注图像已包含反馈信息: {feedback}")
                
                # 🆕 创建一个反馈文件标记
                feedback_marker_file = os.path.join(step_image_dir, f'feedback_{feedback}.txt')
                with open(feedback_marker_file, 'w') as f:
                    f.write(f"User feedback: {feedback}\n")
                    f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                
            else:
                print(f"[DATA_RECORDER] 标注图像不存在: {annotated_file}")
            
        except Exception as e:
            print(f"[DATA_RECORDER] 更新标注图像反馈失败: {e}")