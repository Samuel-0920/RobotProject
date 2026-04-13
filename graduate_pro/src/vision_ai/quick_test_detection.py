#!/usr/bin/env python3
"""
minimal_detection_node.py - 最简化的检测节点
先验证ROS2功能，后续再添加检测管道
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import json
from datetime import datetime

class MinimalDetectionNode(Node):
    def __init__(self):
        super().__init__('detection_node')
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # 状态管理
        self.processing_active = False
        self.latest_reference_image = None
        self.latest_depth_image = None
        self.current_scan_output_dir = None
        
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
        
        # 发布者 - 发布可视化图像
        self.visualization_pub = self.create_publisher(
            Image,
            '/detection_visualization',
            10
        )
        
        self.get_logger().info('🔍 最简化检测节点已启动，等待拼接完成信号...')

    def stitching_complete_callback(self, msg):
        """拼接完成回调"""
        try:
            self.current_scan_output_dir = msg.data
            self.get_logger().info(f'🎯 收到拼接完成信号: {self.current_scan_output_dir}')
            self.get_logger().info('等待参考图像和深度数据...')
        except Exception as e:
            self.get_logger().error(f'处理拼接完成信号失败: {e}')

    def reference_image_callback(self, msg):
        """接收最终参考图像"""
        self.latest_reference_image = msg
        self.get_logger().info('📸 收到参考图像')
        self._check_and_process()

    def reference_depth_callback(self, msg):
        """接收深度信息"""
        self.latest_depth_image = msg
        self.get_logger().info('📏 收到深度信息')
        self._check_and_process()

    def _check_and_process(self):
        """检查是否可以开始处理"""
        if (not self.processing_active and 
            self.latest_reference_image is not None and 
            self.latest_depth_image is not None and
            self.current_scan_output_dir is not None):
            
            self.get_logger().info('🚀 所有数据就绪，开始模拟检测处理...')
            self._mock_detection_process()

    def _mock_detection_process(self):
        """模拟检测处理"""
        try:
            self.processing_active = True
            
            # 转换图像信息
            try:
                reference_rgb = self.bridge.imgmsg_to_cv2(self.latest_reference_image, "rgb8")
                depth_raw = self.bridge.imgmsg_to_cv2(self.latest_depth_image, "16UC1")
                self.get_logger().info(f'图像尺寸: {reference_rgb.shape}, 深度尺寸: {depth_raw.shape}')
            except Exception as e:
                self.get_logger().error(f'图像转换失败: {e}')
                return
            
            # 模拟检测结果
            mock_result = {
                'detection_count': 3,
                'processing_time': 0.5,
                'output_directory': f"{self.current_scan_output_dir}/detection_results",
                'objects': [
                    {
                        'object_id': 'strawberry_0',
                        'class_id': 6,
                        'class_name': 'strawberry',
                        'confidence': 0.95,
                        'description': 'Red strawberry in top-right corner',
                        'bounding_box': [800, 100, 900, 200],
                        'center_x': 850.0,
                        'center_y': 150.0
                    },
                    {
                        'object_id': 'corn_0',
                        'class_id': 3,
                        'class_name': 'corn',
                        'confidence': 0.88,
                        'description': 'Yellow corn in center area',
                        'bounding_box': [400, 300, 500, 450],
                        'center_x': 450.0,
                        'center_y': 375.0
                    },
                    {
                        'object_id': 'carrot_0',
                        'class_id': 2,
                        'class_name': 'carrot',
                        'confidence': 0.92,
                        'description': 'Orange carrot in bottom-left corner',
                        'bounding_box': [100, 500, 200, 650],
                        'center_x': 150.0,
                        'center_y': 575.0
                    }
                ]
            }
            
            self.get_logger().info(f'✅ 模拟检测完成，发现 {mock_result["detection_count"]} 个目标')
            
            # 发布模拟结果
            json_msg = String()
            json_msg.data = json.dumps(mock_result, indent=2)
            self.detection_result_pub.publish(json_msg)
            self.get_logger().info('📤 模拟检测结果已发布')
            
            # 显示结果
            self._display_mock_results(mock_result)
            
        except Exception as e:
            self.get_logger().error(f'模拟检测处理失败: {e}')
        finally:
            self.processing_active = False
            # 重置数据
            self.latest_reference_image = None
            self.latest_depth_image = None
            self.current_scan_output_dir = None

    def _display_mock_results(self, result):
        """显示模拟结果"""
        objects = result.get('objects', [])
        
        self.get_logger().info(f"\n{'='*60}")
        self.get_logger().info("Mock Detection Results Summary")
        self.get_logger().info(f"{'='*60}")
        
        self.get_logger().info(f"Detected {len(objects)} objects:")
        for i, obj in enumerate(objects, 1):
            self.get_logger().info(f"{i:2d}. {obj['description']} (Conf: {obj['confidence']:.3f})")
        
        self.get_logger().info(f"{'='*60}")
        self.get_logger().info("🎯 检测节点基础功能测试成功！")
        self.get_logger().info("下一步可以集成真实的检测管道")


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = MinimalDetectionNode()
        node.get_logger().info('🔍 最简化检测节点运行中...')
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