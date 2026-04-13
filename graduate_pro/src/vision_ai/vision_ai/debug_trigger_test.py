#!/usr/bin/env python3
# debug_trigger_test.py - 调试触发信号

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import time

class TriggerDebugNode(Node):
    def __init__(self):
        super().__init__('trigger_debug')
        
        # 监听触发信号
        self.trigger_sub = self.create_subscription(
            String,
            '/stitching_complete',
            self.trigger_callback,
            10
        )
        
        self.get_logger().info('🔍 开始监听 /stitching_complete 信号...')
        
    def trigger_callback(self, msg):
        """收到触发信号"""
        self.get_logger().info(f'📢 收到触发信号: {msg.data}')
        
        # 检查目录是否存在
        import os
        if os.path.exists(msg.data):
            self.get_logger().info(f'✅ 目录存在: {msg.data}')
            
            # 列出目录内容
            try:
                files = os.listdir(msg.data)
                self.get_logger().info(f'📁 目录内容 ({len(files)} 个文件):')
                for f in sorted(files):
                    self.get_logger().info(f'   - {f}')
            except Exception as e:
                self.get_logger().error(f'无法列出目录内容: {e}')
        else:
            self.get_logger().error(f'❌ 目录不存在: {msg.data}')

def main():
    rclpy.init()
    
    node = TriggerDebugNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print('🛑 调试节点被中断')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()