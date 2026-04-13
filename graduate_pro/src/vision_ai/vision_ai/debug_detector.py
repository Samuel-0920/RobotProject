#!/usr/bin/env python3
"""
检测调试工具 - 单独测试YOLO检测
"""

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import sys
import os

# 添加路径
sys.path.append('/home/qi/ros2_ws/src/vision_ai')

from vision_ai.tracking_system.utils.config import create_tracking_config
from vision_ai.tracking_system.detection.realtime_detector import RealtimeDetector

class DetectionDebugger(Node):
    def __init__(self):
        super().__init__('detection_debugger')
        
        # 初始化配置和检测器
        self.config = create_tracking_config()
        self.detector = RealtimeDetector(self.config, self.get_logger())
        
        # 订阅相机图像
        self.color_sub = self.create_subscription(
            Image, '/camera/color/image_raw',
            self.image_callback, 10
        )
        
        self.get_logger().info('🔧 检测调试器启动')
        self.get_logger().info('模型信息:')
        model_info = self.detector.get_model_info()
        for key, value in model_info.items():
            self.get_logger().info(f'  {key}: {value}')
    
    def image_callback(self, msg):
        """图像回调"""
        try:
            # 转换图像
            image_array = np.frombuffer(msg.data, dtype=np.uint8)
            bgr_image = image_array.reshape((msg.height, msg.width, 3))
            
            # 显示原始图像尺寸和格式信息
            self.get_logger().info(f'📷 图像尺寸: {bgr_image.shape}, 数据类型: {bgr_image.dtype}')
            
            # 检查图像是否有效
            if bgr_image.size == 0:
                self.get_logger().error('❌ 图像数据为空')
                return
            
            # 直接使用BGR图像进行YOLO检测（RealSense发布的就是BGR格式）
            self.get_logger().info('🔍 开始YOLO检测...')
            
            # 运行检测 - 直接传入BGR图像，让YOLO处理
            detection_results = self.detector.detect_and_segment(bgr_image, None)
            
            # 显示结果
            if detection_results:
                self.get_logger().info(f'✅ 检测到 {len(detection_results)} 个对象:')
                for result in detection_results:
                    self.get_logger().info(
                        f'  - {result.class_name} (class_id: {result.class_id}) '
                        f'置信度: {result.confidence:.3f}'
                    )
            else:
                self.get_logger().info('❌ 未检测到任何对象')
            
            # 可视化（直接使用BGR图像）
            vis_image = self.detector.visualize_detections(bgr_image, detection_results)
            
            # 显示图像
            cv2.imshow('Detection Debug', vis_image)
            cv2.waitKey(1)
            
        except Exception as e:
            self.get_logger().error(f'❌ 检测调试失败: {e}')
            import traceback
            self.get_logger().error(f'详细错误: {traceback.format_exc()}')

def main():
    rclpy.init()
    
    try:
        debugger = DetectionDebugger()
        
        print("🔧 检测调试器运行中...")
        print("按 Ctrl+C 退出")
        
        rclpy.spin(debugger)
        
        
    except KeyboardInterrupt:
        print('\n🔄 退出调试器')
    finally:
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()