#!/usr/bin/env python3
"""
手动触发追踪测试脚本
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json
import sys

class TrackingTrigger(Node):
    def __init__(self):
        super().__init__('tracking_trigger')
        
        # 发布者
        self.detection_complete_pub = self.create_publisher(
            String, '/detection_complete', 10
        )
        
        self.get_logger().info('🔧 追踪触发器启动')
    
    def trigger_tracking(self, scan_directory: str):
        """触发追踪"""
        try:
            # 构建检测完成信号
            detection_complete_data = {
                'scan_directory': scan_directory,
                'timestamp': self.get_clock().now().nanoseconds,
                'status': 'completed'
            }
            
            msg = String()
            msg.data = json.dumps(detection_complete_data)
            
            # 发布信号
            self.detection_complete_pub.publish(msg)
            self.get_logger().info(f'📡 发送检测完成信号: {scan_directory}')
            
        except Exception as e:
            self.get_logger().error(f'❌ 触发追踪失败: {e}')

def main():
    rclpy.init()
    
    # 检查命令行参数
    if len(sys.argv) < 2:
        print("用法: python manual_trigger.py <scan_directory>")
        print("示例: python manual_trigger.py /home/qi/ros2_ws/scan_output_20250718_192103")
        return
    
    scan_directory = sys.argv[1]
    
    try:
        trigger = TrackingTrigger()
        
        # 等待一秒让订阅者连接
        import time
        time.sleep(1)
        
        # 触发追踪
        trigger.trigger_tracking(scan_directory)
        
        # 保持运行一段时间
        import threading
        def shutdown_timer():
            time.sleep(3)
            rclpy.shutdown()
        
        timer_thread = threading.Thread(target=shutdown_timer)
        timer_thread.start()
        
        rclpy.spin(trigger)
        
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()