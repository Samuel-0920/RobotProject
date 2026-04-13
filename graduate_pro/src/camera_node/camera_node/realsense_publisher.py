#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TransformStamped
from cv_bridge import CvBridge
import pyrealsense2 as rs
import numpy as np
import cv2


class RealSensePublisher(Node):
    def __init__(self):
        super().__init__('realsense_publisher')
        
        # 创建发布者
        self.color_pub = self.create_publisher(Image, '/camera/color/image_raw', 10)
        self.depth_pub = self.create_publisher(Image, '/camera/depth/image_raw', 10)
        self.color_info_pub = self.create_publisher(CameraInfo, '/camera/color/camera_info', 10)
        self.depth_info_pub = self.create_publisher(CameraInfo, '/camera/depth/camera_info', 10)
        
        # CV Bridge用于图像转换
        self.bridge = CvBridge()
        
        # RealSense配置
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # 配置流
        self.config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        
        # 启动pipeline
        try:
            self.profile = self.pipeline.start(self.config)
            self.get_logger().info('RealSense camera started successfully')
            
            # 获取相机内参
            self.setup_camera_info()
            
        except Exception as e:
            self.get_logger().error(f'Failed to start RealSense camera: {e}')
            return
        
        # 创建对齐对象（将深度图对齐到彩色图）
        align_to = rs.stream.color
        self.align = rs.align(align_to)
        
        # 创建定时器，30Hz发布频率
        self.timer = self.create_timer(1.0/5.0, self.timer_callback)
        
        self.get_logger().info('RealSense Publisher Node initialized')

    def setup_camera_info(self):
        """设置相机内参信息"""
        # 获取深度和彩色流的内参
        depth_stream = self.profile.get_stream(rs.stream.depth)
        color_stream = self.profile.get_stream(rs.stream.color)
        
        depth_intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()
        color_intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
        
        # 创建深度相机信息
        self.depth_camera_info = CameraInfo()
        self.depth_camera_info.header.frame_id = "camera_depth_optical_frame"
        self.depth_camera_info.width = depth_intrinsics.width
        self.depth_camera_info.height = depth_intrinsics.height
        self.depth_camera_info.k = [
            depth_intrinsics.fx, 0.0, depth_intrinsics.ppx,
            0.0, depth_intrinsics.fy, depth_intrinsics.ppy,
            0.0, 0.0, 1.0
        ]
        
        # 创建彩色相机信息
        self.color_camera_info = CameraInfo()
        self.color_camera_info.header.frame_id = "camera_color_optical_frame"
        self.color_camera_info.width = color_intrinsics.width
        self.color_camera_info.height = color_intrinsics.height
        self.color_camera_info.k = [
            color_intrinsics.fx, 0.0, color_intrinsics.ppx,
            0.0, color_intrinsics.fy, color_intrinsics.ppy,
            0.0, 0.0, 1.0
        ]

    def timer_callback(self):
        """定时器回调函数，获取并发布图像"""
        try:
            # 等待新的帧
            frames = self.pipeline.wait_for_frames(timeout_ms=1000)
            
            # 对齐深度图到彩色图
            aligned_frames = self.align.process(frames)
            
            # 获取对齐后的帧
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if not aligned_depth_frame or not color_frame:
                self.get_logger().warn('Failed to get frames')
                return
            
            # 转换为numpy数组
            
            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            
            # 添加调试信息
            # self.get_logger().info(f"Original depth dtype: {depth_image.dtype}")
            # self.get_logger().info(f"Original depth shape: {depth_image.shape}")
            # self.get_logger().info(f"Original depth range: min={depth_image.min()}, max={depth_image.max()}")
            
            # 确保深度图像是uint16格式
            if depth_image.dtype != np.uint16:
                depth_image = depth_image.astype(np.uint16)

            color_image = np.asanyarray(color_frame.get_data())
            
            # 创建ROS图像消息
            current_time = self.get_clock().now().to_msg()
            
            # 发布彩色图像
            color_msg = self.bridge.cv2_to_imgmsg(color_image, "bgr8")
            color_msg.header.stamp = current_time
            color_msg.header.frame_id = "camera_color_optical_frame"
            self.color_pub.publish(color_msg)
            
            # 发布深度图像
            depth_msg = self.bridge.cv2_to_imgmsg(depth_image, encoding="passthrough")
            depth_msg.header.stamp = current_time
            depth_msg.header.frame_id = "camera_depth_optical_frame"
            self.depth_pub.publish(depth_msg)
            
            # 发布相机信息
            self.color_camera_info.header.stamp = current_time
            self.depth_camera_info.header.stamp = current_time
            self.color_info_pub.publish(self.color_camera_info)
            self.depth_info_pub.publish(self.depth_camera_info)
            
            # 记录发布信息（每秒一次，避免日志过多）
            if hasattr(self, 'frame_count'):
                self.frame_count += 1
            else:
                self.frame_count = 1
                
            if self.frame_count % 5 == 0:  # 每10帧（1秒）记录一次
                self.get_logger().info(f'Published frame #{self.frame_count}')
                
        except Exception as e:
            self.get_logger().error(f'Error in timer callback: {e}')

    def destroy_node(self):
        """节点销毁时清理资源"""
        try:
            if hasattr(self, 'pipeline'):
                self.pipeline.stop()
                self.get_logger().info('RealSense pipeline stopped')
        except Exception as e:
            self.get_logger().error(f'Error stopping pipeline: {e}')
        
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    try:
        realsense_publisher = RealSensePublisher()
        rclpy.spin(realsense_publisher)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Error: {e}')
    finally:
        if 'realsense_publisher' in locals():
            realsense_publisher.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()