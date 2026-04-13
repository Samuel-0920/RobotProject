# 4. 创建一个新文件: vision_ai/visualization_trigger.py
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
import cv2
import numpy as np
import json
import tkinter as tk
from tkinter import ttk
from PIL import Image as PILImage, ImageTk
import threading
from datetime import datetime

class VisualizationTrigger(Node):
    def __init__(self):
        super().__init__('visualization_trigger')
        
        # 数据存储
        self.current_color_image = None
        self.current_depth_image = None
        self.current_detection_viz = None
        self.detection_results = []
        
        # ROS订阅者
        self.color_sub = self.create_subscription(
            Image, '/camera/color/image_raw', self.color_image_callback, 10
        )
        self.depth_sub = self.create_subscription(
            Image, '/camera/depth/image_raw', self.depth_image_callback, 10
        )
        self.detection_viz_sub = self.create_subscription(
            Image, '/detection_visualization', self.detection_viz_callback, 10
        )
        self.detection_complete_sub = self.create_subscription(
            String, '/detection_complete', self.detection_complete_callback, 10
        )
        
        # 启动GUI
        self.setup_gui()
        
        self.get_logger().info('Visualization Trigger started')
    
    def color_image_callback(self, msg):
        """彩色图像回调"""
        try:
            # 转换ROS图像为OpenCV格式
            image_data = np.frombuffer(msg.data, dtype=np.uint8)
            if msg.encoding == 'rgb8':
                image = image_data.reshape(msg.height, msg.width, 3)
            elif msg.encoding == 'bgr8':
                image = image_data.reshape(msg.height, msg.width, 3)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                return
            
            self.current_color_image = image
            self.update_gui_image('color', image)
            
        except Exception as e:
            self.get_logger().error(f'Color image callback failed: {e}')
    
    def depth_image_callback(self, msg):
        """深度图像回调 - 生成热度图"""
        try:
            # 转换深度图像
            if msg.encoding == 'mono16' or msg.encoding == '16UC1':
                depth_data = np.frombuffer(msg.data, dtype=np.uint16)
            else:
                depth_data = np.frombuffer(msg.data, dtype=np.uint8)
            
            depth_image = depth_data.reshape(msg.height, msg.width)
            
            # 🔥 生成热度图 - 更好的可视化效果
            # 过滤无效深度值
            valid_mask = (depth_image > 0) & (depth_image < 10000)  # 0-10m有效范围
            depth_filtered = depth_image.copy().astype(np.float32)
            depth_filtered[~valid_mask] = 0
            
            if np.any(valid_mask):
                # 只对有效区域进行归一化
                valid_depths = depth_filtered[valid_mask]
                min_depth, max_depth = np.percentile(valid_depths, [5, 95])  # 使用5-95百分位避免异常值
                
                # 归一化到0-255
                depth_normalized = np.zeros_like(depth_filtered)
                depth_normalized[valid_mask] = np.clip(
                    (depth_filtered[valid_mask] - min_depth) / (max_depth - min_depth + 1e-6) * 255, 
                    0, 255
                ).astype(np.uint8)
                
                # 应用颜色映射 - 使用TURBO colormap获得更好的热度图效果
                depth_colored = cv2.applyColorMap(depth_normalized.astype(np.uint8), cv2.COLORMAP_TURBO)
                
                # 将无效区域设为黑色
                depth_colored[~valid_mask] = [0, 0, 0]
                
            else:
                # 如果没有有效深度，显示全黑图像
                depth_colored = np.zeros((msg.height, msg.width, 3), dtype=np.uint8)
            
            # 转换为RGB
            depth_rgb = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)
            
            # 🆕 添加深度信息标注
            if np.any(valid_mask):
                valid_depths = depth_filtered[valid_mask]
                avg_depth = np.mean(valid_depths)
                min_depth_val = np.min(valid_depths)
                max_depth_val = np.max(valid_depths)
                
                # 在图像上添加深度信息
                font = cv2.FONT_HERSHEY_SIMPLEX
                info_text = [
                    f"Depth Range: {min_depth_val:.0f}-{max_depth_val:.0f}mm",
                    f"Avg Depth: {avg_depth:.0f}mm",
                    f"Valid Pixels: {np.sum(valid_mask)}"
                ]
                
                y_offset = 30
                for i, text in enumerate(info_text):
                    cv2.putText(depth_rgb, text, (10, y_offset + i * 25), 
                            font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(depth_rgb, text, (10, y_offset + i * 25), 
                            font, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
            
            self.current_depth_image = depth_rgb
            self.update_gui_image('depth', depth_rgb)
            
        except Exception as e:
            self.get_logger().error(f'Depth image callback failed: {e}')
    
    def detection_viz_callback(self, msg):
        """检测可视化回调"""
        try:
            # 转换检测可视化图像
            image_data = np.frombuffer(msg.data, dtype=np.uint8)
            image = image_data.reshape(msg.height, msg.width, 3)
            
            self.current_detection_viz = image
            self.update_gui_image('detection', image)
            
        except Exception as e:
            self.get_logger().error(f'Detection visualization callback failed: {e}')
    
    def detection_complete_callback(self, msg):
        """检测完成回调"""
        try:
            data = json.loads(msg.data)
            
            # 更新检测结果信息
            detection_info = {
                'timestamp': datetime.now().strftime('%H:%M:%S'),
                'count': data.get('detection_count', 0),
                'enhanced': data.get('enhanced_detection', False),
                'objects': data.get('total_objects', 0)
            }
            
            self.detection_results.append(detection_info)
            if len(self.detection_results) > 10:  # 保持最近10条记录
                self.detection_results = self.detection_results[-10:]
            
            self.update_gui_info()
            
        except Exception as e:
            self.get_logger().error(f'Detection complete callback failed: {e}')
    
    def setup_gui(self):
        """设置3窗口GUI界面 - 优化布局"""
        def run_gui():
            self.root = tk.Tk()
            self.root.title('Vision AI - 3-Window Visualization Dashboard')
            self.root.geometry('1600x900')
            self.root.configure(bg='#2b2b2b')
            
            # 创建主框架
            main_frame = ttk.Frame(self.root)
            main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # 🎨 创建样式
            style = ttk.Style()
            style.theme_use('clam')
            style.configure('Custom.TLabelframe', background='#3b3b3b', foreground='white')
            style.configure('Custom.TLabelframe.Label', background='#3b3b3b', foreground='white', font=('Arial', 10, 'bold'))
            
            # 🖼️ 图像显示区域 - 使用网格布局
            images_frame = ttk.Frame(main_frame)
            images_frame.pack(fill=tk.BOTH, expand=True)
            
            # 第一行：彩色图像 + 深度热度图
            top_frame = ttk.Frame(images_frame)
            top_frame.pack(fill=tk.BOTH, expand=True, pady=5)
            
            # 彩色图像
            color_frame = ttk.LabelFrame(top_frame, text='📷 Color Image Stream', style='Custom.TLabelframe')
            color_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
            self.color_label = ttk.Label(color_frame, background='black')
            self.color_label.pack(expand=True, pady=10)
            
            # 深度热度图
            depth_frame = ttk.LabelFrame(top_frame, text='🔥 Depth Heatmap', style='Custom.TLabelframe')
            depth_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
            self.depth_label = ttk.Label(depth_frame, background='black')
            self.depth_label.pack(expand=True, pady=10)
            
            # 第二行：检测结果（占据整行）
            bottom_frame = ttk.Frame(images_frame)
            bottom_frame.pack(fill=tk.BOTH, expand=True, pady=5)
            
            detection_frame = ttk.LabelFrame(bottom_frame, text='🎯 Enhanced Detection Results', style='Custom.TLabelframe')
            detection_frame.pack(fill=tk.BOTH, expand=True, padx=5)
            self.detection_label = ttk.Label(detection_frame, background='black')
            self.detection_label.pack(expand=True, pady=10)
            
            # 🆕 信息面板 - 紧凑设计
            info_frame = ttk.LabelFrame(main_frame, text='📊 Detection History & Status', style='Custom.TLabelframe')
            info_frame.pack(fill=tk.X, pady=10)
            
            # 创建信息显示表格
            info_container = ttk.Frame(info_frame)
            info_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
            
            # 表格
            columns = ('Time', 'Objects', 'Type', 'Status', 'Quality')
            self.info_tree = ttk.Treeview(info_container, columns=columns, show='headings', height=4)
            
            # 设置列宽
            column_widths = {'Time': 80, 'Objects': 60, 'Type': 80, 'Status': 100, 'Quality': 80}
            for col in columns:
                self.info_tree.heading(col, text=col)
                self.info_tree.column(col, width=column_widths.get(col, 100))
            
            # 滚动条
            scrollbar = ttk.Scrollbar(info_container, orient=tk.VERTICAL, command=self.info_tree.yview)
            self.info_tree.configure(yscrollcommand=scrollbar.set)
            
            self.info_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            # 🎮 控制按钮区域
            button_frame = ttk.Frame(main_frame)
            button_frame.pack(fill=tk.X, pady=5)
            
            # 左侧：保存按钮
            save_frame = ttk.Frame(button_frame)
            save_frame.pack(side=tk.LEFT)
            
            ttk.Button(save_frame, text='💾 Save Color', 
                    command=self.save_color_image).pack(side=tk.LEFT, padx=2)
            ttk.Button(save_frame, text='🔥 Save Depth', 
                    command=self.save_depth_image).pack(side=tk.LEFT, padx=2)
            ttk.Button(save_frame, text='🎯 Save Detection', 
                    command=self.save_detection_image).pack(side=tk.LEFT, padx=2)
            
            # 中间：系统控制
            control_frame = ttk.Frame(button_frame)
            control_frame.pack(side=tk.LEFT, padx=20)
            
            ttk.Button(control_frame, text='🔄 Refresh', 
                    command=self.refresh_display).pack(side=tk.LEFT, padx=2)
            ttk.Button(control_frame, text='🗑️ Clear History', 
                    command=self.clear_history).pack(side=tk.LEFT, padx=2)
            
            # 右侧：手动触发
            trigger_frame = ttk.Frame(button_frame)
            trigger_frame.pack(side=tk.RIGHT)
            
            ttk.Button(trigger_frame, text='🚀 Trigger Scan', 
                    command=self.trigger_manual_scan).pack(side=tk.LEFT, padx=2)
            
            # 📊 增强状态栏
            status_frame = ttk.Frame(main_frame)
            status_frame.pack(fill=tk.X, pady=2)
            
            self.status_var = tk.StringVar()
            self.status_var.set('🟡 Ready - Waiting for camera streams and detection results...')
            status_bar = ttk.Label(status_frame, textvariable=self.status_var, 
                                relief=tk.SUNKEN, background='#2b2b2b', foreground='white')
            status_bar.pack(fill=tk.X)
            
            # 🆕 实时统计显示
            stats_frame = ttk.Frame(status_frame)
            stats_frame.pack(fill=tk.X, pady=2)
            
            self.stats_vars = {
                'fps': tk.StringVar(value='FPS: --'),
                'objects': tk.StringVar(value='Objects: 0'),
                'quality': tk.StringVar(value='Avg Quality: --%')
            }
            
            for key, var in self.stats_vars.items():
                ttk.Label(stats_frame, textvariable=var, background='#3b3b3b', 
                        foreground='lightblue').pack(side=tk.LEFT, padx=10)
            
            self.root.mainloop()
        
        # 在单独线程中运行GUI
        self.gui_thread = threading.Thread(target=run_gui, daemon=True)
        self.gui_thread.start()
    
    def update_gui_image(self, image_type: str, image: np.ndarray):
        """更新GUI中的图像"""
        try:
            if not hasattr(self, 'root'):
                return
            
            # 调整图像大小
            height, width = image.shape[:2]
            max_size = 400
            if width > height:
                new_width = max_size
                new_height = int(height * max_size / width)
            else:
                new_height = max_size
                new_width = int(width * max_size / height)
            
            resized = cv2.resize(image, (new_width, new_height))
            
            # 转换为PIL格式
            pil_image = PILImage.fromarray(resized)
            photo = ImageTk.PhotoImage(pil_image)
            
            # 更新对应的标签
            def update_label():
                if image_type == 'color' and hasattr(self, 'color_label'):
                    self.color_label.configure(image=photo)
                    self.color_label.image = photo
                elif image_type == 'depth' and hasattr(self, 'depth_label'):
                    self.depth_label.configure(image=photo)
                    self.depth_label.image = photo
                elif image_type == 'detection' and hasattr(self, 'detection_label'):
                    self.detection_label.configure(image=photo)
                    self.detection_label.image = photo
                
                # 更新状态
                if hasattr(self, 'status_var'):
                    current_time = datetime.now().strftime('%H:%M:%S')
                    self.status_var.set(f'Last {image_type} update: {current_time}')
            
            # 在主线程中更新GUI
            self.root.after(0, update_label)
            
        except Exception as e:
            self.get_logger().error(f'GUI image update failed: {e}')
    
    def trigger_manual_scan(self):
        """手动触发扫描"""
        try:
            # 发布扫描触发消息
            trigger_pub = self.create_publisher(String, '/start_scan', 10)
            msg = String()
            msg.data = 'manual_trigger'
            trigger_pub.publish(msg)
            
            self.status_var.set('🚀 Manual scan triggered...')
            self.get_logger().info('Manual scan triggered')
            
        except Exception as e:
            self.get_logger().error(f'Failed to trigger scan: {e}')
            self.status_var.set('❌ Failed to trigger scan')

    def refresh_display(self):
        """刷新显示"""
        try:
            # 更新所有图像显示
            if self.current_color_image is not None:
                self.update_gui_image('color', self.current_color_image)
            if self.current_depth_image is not None:
                self.update_gui_image('depth', self.current_depth_image)
            if self.current_detection_viz is not None:
                self.update_gui_image('detection', self.current_detection_viz)
            
            self.status_var.set(f'🔄 Display refreshed at {datetime.now().strftime("%H:%M:%S")}')
            
        except Exception as e:
            self.get_logger().error(f'Failed to refresh display: {e}')

    def detection_complete_callback(self, msg):
        """检测完成回调 - 增强版本"""
        try:
            data = json.loads(msg.data)
            
            # 更新检测结果信息
            detection_info = {
                'timestamp': datetime.now().strftime('%H:%M:%S'),
                'count': data.get('detection_count', 0),
                'enhanced': data.get('enhanced_detection', False),
                'objects': data.get('total_objects', 0),
                'quality': data.get('feature_quality_stats', {}).get('mean_quality', 0)
            }
            
            self.detection_results.append(detection_info)
            if len(self.detection_results) > 20:  # 保持最近20条记录
                self.detection_results = self.detection_results[-20:]
            
            # 更新实时统计
            self.update_stats(detection_info)
            
            self.update_gui_info()
            
            # 更新状态栏
            status_msg = f"🟢 Detected {detection_info['objects']} objects"
            if detection_info['enhanced']:
                status_msg += f" (Enhanced, Quality: {detection_info['quality']:.1f}%)"
            self.status_var.set(status_msg)
            
        except Exception as e:
            self.get_logger().error(f'Detection complete callback failed: {e}')

    def update_stats(self, detection_info):
        """更新实时统计"""
        try:
            # 更新对象数量
            self.stats_vars['objects'].set(f'Objects: {detection_info["objects"]}')
            
            # 更新平均质量
            if detection_info['quality'] > 0:
                self.stats_vars['quality'].set(f'Avg Quality: {detection_info["quality"]:.1f}%')
            
            # 计算检测频率（简单估算）
            if len(self.detection_results) >= 2:
                time_diff = (datetime.now() - datetime.strptime(
                    self.detection_results[-2]['timestamp'], '%H:%M:%S'
                )).total_seconds()
                if time_diff > 0:
                    fps = 1.0 / time_diff
                    self.stats_vars['fps'].set(f'Detection Rate: {fps:.1f}/s')
            
        except Exception as e:
            self.get_logger().error(f'Failed to update stats: {e}')

    def update_gui_info(self):
        """更新GUI信息面板 - 增强版本"""
        try:
            if not hasattr(self, 'info_tree'):
                return
            
            def update_tree():
                # 清空现有内容
                for item in self.info_tree.get_children():
                    self.info_tree.delete(item)
                
                # 添加新内容
                for result in reversed(self.detection_results[-10:]):  # 显示最近10条
                    detection_type = '🚀 Enhanced' if result['enhanced'] else '📍 Standard'
                    status = f"{result['count']} detected"
                    quality = f"{result.get('quality', 0):.1f}%" if result.get('quality', 0) > 0 else "N/A"
                    
                    # 根据质量设置颜色标签
                    if result.get('quality', 0) > 80:
                        tag = 'high_quality'
                    elif result.get('quality', 0) > 50:
                        tag = 'medium_quality'
                    else:
                        tag = 'low_quality'
                    
                    self.info_tree.insert('', 0, values=(
                        result['timestamp'],
                        result['objects'],
                        detection_type,
                        status,
                        quality
                    ), tags=(tag,))
                
                # 设置标签颜色
                self.info_tree.tag_configure('high_quality', background='#2d5016', foreground='lightgreen')
                self.info_tree.tag_configure('medium_quality', background='#4d4d00', foreground='yellow')
                self.info_tree.tag_configure('low_quality', background='#4d1a1a', foreground='lightcoral')
            
            # 在主线程中更新
            self.root.after(0, update_tree)
            
        except Exception as e:
            self.get_logger().error(f'GUI info update failed: {e}')
    
    def save_color_image(self):
        """保存彩色图像"""
        if self.current_color_image is not None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'color_image_{timestamp}.jpg'
            cv2.imwrite(filename, cv2.cvtColor(self.current_color_image, cv2.COLOR_RGB2BGR))
            self.status_var.set(f'Color image saved: {filename}')
    
    def save_depth_image(self):
        """保存深度图像"""
        if self.current_depth_image is not None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'depth_image_{timestamp}.jpg'
            cv2.imwrite(filename, cv2.cvtColor(self.current_depth_image, cv2.COLOR_RGB2BGR))
            self.status_var.set(f'Depth image saved: {filename}')
    
    def save_detection_image(self):
        """保存检测图像"""
        if self.current_detection_viz is not None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'detection_result_{timestamp}.jpg'
            cv2.imwrite(filename, cv2.cvtColor(self.current_detection_viz, cv2.COLOR_RGB2BGR))
            self.status_var.set(f'Detection image saved: {filename}')
    
    def clear_history(self):
        """清空历史记录"""
        self.detection_results.clear()
        self.update_gui_info()
        self.status_var.set('History cleared')

def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = VisualizationTrigger()
        rclpy.spin(node)
    except KeyboardInterrupt:
        print('Visualization Trigger interrupted')
    finally:
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()