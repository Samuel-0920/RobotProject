#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np


class CameraViewer(Node):
    def __init__(self):
        super().__init__('camera_viewer')
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # 退出标志
        self.should_exit = False
        
        # 订阅彩色和深度图像
        self.color_sub = self.create_subscription(
            Image, '/camera/color/image_raw', self.color_callback, 10
        )
        self.depth_sub = self.create_subscription(
            Image, '/camera/depth/image_raw', self.depth_callback, 10
        )
        
        # 存储最新的图像
        self.latest_color = None
        self.latest_depth = None
        
        # 创建窗口
        cv2.namedWindow('RealSense Color', cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow('RealSense Depth', cv2.WINDOW_AUTOSIZE)
        
        # 创建定时器用于显示图像
        self.display_timer = self.create_timer(0.03, self.display_images)  # 约30Hz
        
        self.get_logger().info('Camera Viewer started. Press ESC or Ctrl+C to exit.')

    def color_callback(self, msg):
        """彩色图像回调"""
        try:
            self.latest_color = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f'Error converting color image: {e}')

    def depth_callback(self, msg):
        """深度图像回调"""
        try:
            # 将深度图像转换为可视化格式
            depth_raw = self.bridge.imgmsg_to_cv2(msg, "16UC1")
            
            # 标准化深度图像用于显示 (0-255)
            depth_normalized = cv2.normalize(depth_raw, None, 0, 255, cv2.NORM_MINMAX)
            self.latest_depth = depth_normalized.astype(np.uint8)
            
        except Exception as e:
            self.get_logger().error(f'Error converting depth image: {e}')

    def display_images(self):
        """显示图像"""
        try:
            # 显示彩色图像
            if self.latest_color is not None:
                # 调整显示大小（如果图像太大）
                display_color = self.latest_color.copy()
                height, width = display_color.shape[:2]
                if width > 1280:
                    scale = 1280 / width
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    display_color = cv2.resize(display_color, (new_width, new_height))
                
                # 添加信息文本
                cv2.putText(display_color, f'Color: {display_color.shape[1]}x{display_color.shape[0]}', 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(display_color, 'Press ESC to exit', 
                           (10, display_color.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow('RealSense Color', display_color)
            
            # 显示深度图像
            if self.latest_depth is not None:
                # 调整显示大小
                display_depth = self.latest_depth.copy()
                height, width = display_depth.shape[:2]
                if width > 1280:
                    scale = 1280 / width
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    display_depth = cv2.resize(display_depth, (new_width, new_height))
                
                # 应用彩色映射使深度图更易观察
                depth_colormap = cv2.applyColorMap(display_depth, cv2.COLORMAP_JET)
                
                # 添加信息文本
                cv2.putText(depth_colormap, f'Depth: {display_depth.shape[1]}x{display_depth.shape[0]}', 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(depth_colormap, 'Press ESC to exit', 
                           (10, depth_colormap.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.imshow('RealSense Depth', depth_colormap)
            
            # 检查ESC键退出
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC键
                self.get_logger().info('ESC pressed, shutting down...')
                self.should_exit = True
                
        except Exception as e:
            self.get_logger().error(f'Error displaying images: {e}')

    def destroy_node(self):
        """清理资源"""
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    try:
        viewer = CameraViewer()
        
        # 使用自定义的spin循环
        while rclpy.ok() and not viewer.should_exit:
            rclpy.spin_once(viewer, timeout_sec=0.1)
            
        viewer.get_logger().info('Shutting down camera viewer...')
        
    except KeyboardInterrupt:
        print('\nKeyboard interrupt received, shutting down...')
    except Exception as e:
        print(f'Error: {e}')
    finally:
        cv2.destroyAllWindows()
        if 'viewer' in locals():
            viewer.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()