#!/usr/bin/env python3
"""
Detection Node 信息监听测试工具
用于监听和显示 detection node 发布的所有信息
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
import json
import cv2
import numpy as np
from datetime import datetime
import os

class DetectionMonitor(Node):
    def __init__(self):
        super().__init__('detection_monitor')
        
        # 统计信息
        self.message_counts = {
            'detection_result': 0,
            'detection_complete': 0,
            'detection_visualization': 0
        }
        
        # 最新消息缓存
        self.latest_messages = {
            'detection_result': None,
            'detection_complete': None,
            'detection_visualization': None
        }
        
        # 创建订阅者
        self.create_subscribers()
        
        # 创建定时器显示状态
        self.status_timer = self.create_timer(5.0, self.print_status)
        
        # 保存可视化图像的目录
        self.save_dir = os.path.expanduser("~/detection_monitor_output")
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.get_logger().info('🔍 Detection Monitor 已启动')
        self.get_logger().info(f'📁 可视化图像保存到: {self.save_dir}')
        self.get_logger().info('='*80)
        
    def create_subscribers(self):
        """创建所有订阅者"""
        
        # 订阅检测结果
        self.detection_result_sub = self.create_subscription(
            String,
            '/detection_result',
            self.detection_result_callback,
            10
        )
        
        # 订阅检测完成信号
        self.detection_complete_sub = self.create_subscription(
            String,
            '/detection_complete',
            self.detection_complete_callback,
            10
        )
        
        # 订阅可视化图像
        self.visualization_sub = self.create_subscription(
            Image,
            '/detection_visualization',
            self.visualization_callback,
            10
        )
        
        self.get_logger().info('📡 所有订阅者已创建')
    
    def detection_result_callback(self, msg):
        """处理检测结果消息"""
        try:
            self.message_counts['detection_result'] += 1
            self.latest_messages['detection_result'] = msg
            
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.get_logger().info(f'\n🎯 [{timestamp}] 收到检测结果 (第 {self.message_counts["detection_result"]} 条)')
            
            # 解析JSON数据
            try:
                result_data = json.loads(msg.data)
                self.print_detection_result_details(result_data)
            except json.JSONDecodeError as e:
                self.get_logger().error(f'JSON解析失败: {e}')
                self.get_logger().info(f'原始数据: {msg.data[:200]}...')
                
        except Exception as e:
            self.get_logger().error(f'处理检测结果失败: {e}')
    
    def detection_complete_callback(self, msg):
        """处理检测完成信号"""
        try:
            self.message_counts['detection_complete'] += 1
            self.latest_messages['detection_complete'] = msg
            
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.get_logger().info(f'\n✅ [{timestamp}] 收到检测完成信号 (第 {self.message_counts["detection_complete"]} 条)')
            
            # 解析JSON数据
            try:
                complete_data = json.loads(msg.data)
                self.print_detection_complete_details(complete_data)
            except json.JSONDecodeError as e:
                self.get_logger().error(f'完成信号JSON解析失败: {e}')
                self.get_logger().info(f'原始数据: {msg.data[:200]}...')
                
        except Exception as e:
            self.get_logger().error(f'处理检测完成信号失败: {e}')
    
    def visualization_callback(self, msg):
        """处理可视化图像"""
        try:
            self.message_counts['detection_visualization'] += 1
            self.latest_messages['detection_visualization'] = msg
            
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.get_logger().info(f'\n🖼️ [{timestamp}] 收到可视化图像 (第 {self.message_counts["detection_visualization"]} 条)')
            
            # 显示图像信息
            self.get_logger().info(f'   图像尺寸: {msg.width}x{msg.height}')
            self.get_logger().info(f'   编码格式: {msg.encoding}')
            self.get_logger().info(f'   数据大小: {len(msg.data)} bytes')
            self.get_logger().info(f'   帧ID: {msg.header.frame_id}')
            
            # 保存图像
            self.save_visualization_image(msg)
            
        except Exception as e:
            self.get_logger().error(f'处理可视化图像失败: {e}')
    
    def print_detection_result_details(self, result_data):
        """详细打印检测结果"""
        try:
            self.get_logger().info('📊 检测结果详情:')
            self.get_logger().info(f'   检测数量: {result_data.get("detection_count", 0)}')
            self.get_logger().info(f'   处理时间: {result_data.get("processing_time", 0):.3f}s')
            self.get_logger().info(f'   输出目录: {result_data.get("output_directory", "N/A")}')
            
            objects = result_data.get('objects', [])
            if objects:
                self.get_logger().info(f'   检测对象列表:')
                for i, obj in enumerate(objects, 1):
                    self.get_logger().info(f'     {i}. {obj.get("description", "N/A")}')
                    self.get_logger().info(f'        ID: {obj.get("object_id", "N/A")}')
                    self.get_logger().info(f'        类别: {obj.get("class_name", "N/A")}')
                    self.get_logger().info(f'        置信度: {obj.get("confidence", 0):.3f}')
                    
                    # 显示特征信息
                    features = obj.get('features', {})
                    if features:
                        self.get_logger().info(f'        特征信息:')
                        
                        # 颜色特征
                        color_info = features.get('color', {})
                        if color_info:
                            self.get_logger().info(f'          颜色: {color_info.get("color_name", "N/A")}')
                        
                        # 空间特征
                        spatial_info = features.get('spatial', {})
                        if spatial_info:
                            self.get_logger().info(f'          区域位置: {spatial_info.get("region_position", "N/A")}')
                            world_coords = spatial_info.get('world_coordinates')
                            if world_coords:
                                self.get_logger().info(f'          世界坐标: ({world_coords[0]:.1f}, {world_coords[1]:.1f}, {world_coords[2]:.1f})')
                        
                        # 深度信息
                        depth_info = features.get('depth_info', {})
                        if depth_info:
                            height = depth_info.get('height_mm')
                            if height:
                                self.get_logger().info(f'          高度: {height:.1f}mm')
                            
                            source_wp = depth_info.get('source_waypoint')
                            if source_wp is not None:
                                self.get_logger().info(f'          来源waypoint: {source_wp}')
                    
                    self.get_logger().info('')  # 空行分隔
            else:
                self.get_logger().info('   无检测对象')
                
        except Exception as e:
            self.get_logger().error(f'打印检测结果详情失败: {e}')
    
    def print_detection_complete_details(self, complete_data):
        """详细打印检测完成信号"""
        try:
            self.get_logger().info('🏁 检测完成信号详情:')
            self.get_logger().info(f'   状态: {complete_data.get("status", "N/A")}')
            self.get_logger().info(f'   扫描目录: {complete_data.get("scan_directory", "N/A")}')
            self.get_logger().info(f'   检测数量: {complete_data.get("detection_count", 0)}')
            self.get_logger().info(f'   总对象数: {complete_data.get("total_objects", 0)}')
            
            # 扫描中心位姿
            center_pose = complete_data.get('scan_center_pose')
            if center_pose:
                self.get_logger().info('   扫描中心位姿:')
                pos = center_pose.get('position', {})
                ori = center_pose.get('orientation', {})
                self.get_logger().info(f'     位置: ({pos.get("x", 0):.1f}, {pos.get("y", 0):.1f}, {pos.get("z", 0):.1f})')
                self.get_logger().info(f'     姿态: (R:{ori.get("roll", 0):.1f}, P:{ori.get("pitch", 0):.1f}, Y:{ori.get("yaw", 0):.1f})')
            else:
                self.get_logger().info('   扫描中心位姿: 无')
            
            # 扫描边界
            bounds = complete_data.get('scan_bounds')
            if bounds:
                self.get_logger().info('   扫描边界:')
                self.get_logger().info(f'     X: {bounds.get("min_x", 0):.1f} ~ {bounds.get("max_x", 0):.1f}')
                self.get_logger().info(f'     Y: {bounds.get("min_y", 0):.1f} ~ {bounds.get("max_y", 0):.1f}')
                self.get_logger().info(f'     Z: {bounds.get("min_z", 0):.1f} ~ {bounds.get("max_z", 0):.1f}')
            else:
                self.get_logger().info('   扫描边界: 无')
            
            # Waypoint位姿
            waypoint_poses = complete_data.get('waypoint_poses', [])
            if waypoint_poses:
                self.get_logger().info(f'   Waypoint位姿 (共 {len(waypoint_poses)} 个):')
                for i, pose in enumerate(waypoint_poses[:5]):  # 只显示前5个
                    pos = pose.get('position', {})
                    ori = pose.get('orientation', {})
                    self.get_logger().info(f'     WP{i}: Pos({pos.get("x", 0):.1f}, {pos.get("y", 0):.1f}, {pos.get("z", 0):.1f}) '
                                         f'Ori(R:{ori.get("roll", 0):.1f}, P:{ori.get("pitch", 0):.1f}, Y:{ori.get("yaw", 0):.1f})')
                if len(waypoint_poses) > 5:
                    self.get_logger().info(f'     ... 还有 {len(waypoint_poses) - 5} 个waypoint')
            else:
                self.get_logger().info('   Waypoint位姿: 无')
                
        except Exception as e:
            self.get_logger().error(f'打印完成信号详情失败: {e}')
    
    def save_visualization_image(self, msg):
        """保存可视化图像"""
        try:
            # 转换图像数据
            if msg.encoding == 'rgb8':
                # RGB图像
                image_array = np.frombuffer(msg.data, dtype=np.uint8)
                image = image_array.reshape(msg.height, msg.width, 3)
                # 转换为BGR用于OpenCV保存
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            elif msg.encoding == 'bgr8':
                # BGR图像
                image_array = np.frombuffer(msg.data, dtype=np.uint8)
                image_bgr = image_array.reshape(msg.height, msg.width, 3)
            else:
                self.get_logger().warn(f'不支持的图像编码: {msg.encoding}')
                return
            
            # 生成文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"detection_viz_{timestamp}_{self.message_counts['detection_visualization']}.jpg"
            filepath = os.path.join(self.save_dir, filename)
            
            # 保存图像
            success = cv2.imwrite(filepath, image_bgr)
            if success:
                self.get_logger().info(f'   ✅ 图像已保存: {filename}')
            else:
                self.get_logger().error(f'   ❌ 图像保存失败: {filename}')
                
        except Exception as e:
            self.get_logger().error(f'保存可视化图像失败: {e}')
    
    def print_status(self):
        """定期打印状态信息"""
        self.get_logger().info('\n' + '='*80)
        self.get_logger().info('📊 Detection Monitor 状态报告')
        self.get_logger().info('='*80)
        self.get_logger().info(f'检测结果消息数: {self.message_counts["detection_result"]}')
        self.get_logger().info(f'检测完成消息数: {self.message_counts["detection_complete"]}')
        self.get_logger().info(f'可视化图像数: {self.message_counts["detection_visualization"]}')
        self.get_logger().info(f'时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        self.get_logger().info('='*80)
    
    def get_latest_result_summary(self):
        """获取最新结果摘要"""
        try:
            if self.latest_messages['detection_result']:
                result_data = json.loads(self.latest_messages['detection_result'].data)
                return {
                    'detection_count': result_data.get('detection_count', 0),
                    'objects': [obj.get('description', 'N/A') for obj in result_data.get('objects', [])],
                    'processing_time': result_data.get('processing_time', 0)
                }
            return None
        except:
            return None
    
    def export_message_log(self):
        """导出消息日志"""
        try:
            log_file = os.path.join(self.save_dir, f"detection_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            
            log_data = {
                'message_counts': self.message_counts,
                'latest_detection_result': self.latest_messages['detection_result'].data if self.latest_messages['detection_result'] else None,
                'latest_detection_complete': self.latest_messages['detection_complete'].data if self.latest_messages['detection_complete'] else None,
                'export_time': datetime.now().isoformat()
            }
            
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)
            
            self.get_logger().info(f'📝 消息日志已导出: {log_file}')
            
        except Exception as e:
            self.get_logger().error(f'导出消息日志失败: {e}')


def main(args=None):
    rclpy.init(args=args)
    
    try:
        monitor = DetectionMonitor()
        
        # 添加优雅关闭处理
        def signal_handler():
            monitor.get_logger().info('📝 正在导出最终日志...')
            monitor.export_message_log()
            monitor.get_logger().info('🛑 Detection Monitor 正在关闭...')
        
        # 注册信号处理
        import signal
        signal.signal(signal.SIGINT, lambda sig, frame: signal_handler())
        
        monitor.get_logger().info('🚀 Detection Monitor 开始监听...')
        monitor.get_logger().info('按 Ctrl+C 停止监听并导出日志')
        
        rclpy.spin(monitor)
        
    except KeyboardInterrupt:
        print('\n🛑 用户中断监听')
        if 'monitor' in locals():
            monitor.export_message_log()
    except Exception as e:
        print(f'❌ Monitor 运行错误: {e}')
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()