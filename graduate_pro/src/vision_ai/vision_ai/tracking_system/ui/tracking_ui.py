# tracking_system/ui/tracking_ui.py
import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import json
import os
import threading
import time
from typing import Dict, List, Optional, Callable
from enum import Enum

class UIState(Enum):
    """UI状态枚举"""
    WAITING_CONFIRMATION = "waiting_confirmation"
    CORRECT_ACTIONS = "correct_actions" 
    SELECT_CORRECT_ID = "select_correct_id"
    GRASP_CONFIRMATION = "grasp_confirmation"
    AUTO_MODE = "auto_mode"
    DETECTION_FAILURE = "detection_failure"

class TrackingUI:
    """简化的追踪UI - 状态机驱动"""
    
    def __init__(self, callback_handler: Callable):
        """
        初始化UI
        Args:
            callback_handler: 回调函数，处理UI事件
        """
        self.callback_handler = callback_handler
        
        # 加载配置
        self.config = self._load_config()
        
        # UI状态
        self.current_state = UIState.WAITING_CONFIRMATION
        self.current_image = None
        self.current_candidates = []
        self.current_tracking_result = None
        self.user_response = None
        self.waiting_for_response = False
        
        # UI组件
        self.root = None
        self.image_label = None
        self.status_label = None
        self.button_frame = None
        self.current_buttons = []
        
        # 线程控制
        self.ui_thread = None
        self.should_close = False
        
        print("[TRACKING_UI] UI初始化完成")
    
    def _load_config(self) -> Dict:
        """加载UI配置"""
        try:
            config_path = os.path.join(os.path.dirname(__file__), 'ui_config.json')
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"[TRACKING_UI] 配置加载失败: {e}, 使用默认配置")
            return {
                "ui": {"window_size": [800, 600], "image_display_size": [750, 400]},
                "buttons": {"font_size": 12, "button_height": 40},
                "visualization": {"bbox_thickness": 2, "font_scale": 0.7}
            }
    
    def start(self):
        """启动UI - 在独立线程中运行"""
        if self.ui_thread and self.ui_thread.is_alive():
            return
        
        self.ui_thread = threading.Thread(target=self._run_ui, daemon=True)
        self.ui_thread.start()
        print("[TRACKING_UI] UI线程已启动")
    
    def _run_ui(self):
        """运行UI主循环"""
        try:
            self.root = tk.Tk()
            self.root.title("Enhanced Tracking System")
            
            # 设置窗口大小和位置
            window_size = self.config["ui"]["window_size"]
            self.root.geometry(f"{window_size[0]}x{window_size[1]}+100+100")
            self.root.configure(bg='#f0f0f0')
            
            # 创建UI组件
            self._create_ui_components()
            
            # 设置关闭事件
            self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
            
            # 显示初始状态
            self._update_status("Waiting for tracking data...")
            self._setup_waiting_confirmation_buttons()
            
            # 启动主循环
            self.root.mainloop()
            
        except Exception as e:
            print(f"[TRACKING_UI] UI运行失败: {e}")
    
    def _create_ui_components(self):
        """创建UI组件"""
        # 图像显示区域
        image_size = self.config["ui"]["image_display_size"]
        self.image_label = tk.Label(
            self.root, 
            width=image_size[0], 
            height=image_size[1],
            bg='black',
            text="Waiting for camera data...",
            fg='white',
            font=('Arial', 14)
        )
        self.image_label.pack(pady=10)
        
        # 状态显示
        self.status_label = tk.Label(
            self.root,
            text="System Status: Initializing...",
            font=('Arial', 10),
            bg='#f0f0f0'
        )
        self.status_label.pack(pady=5)
        
        # 按钮框架
        self.button_frame = tk.Frame(self.root, bg='#f0f0f0')
        self.button_frame.pack(pady=10, fill='x', padx=20)
    
    def update_and_wait_for_input(self, image: np.ndarray, candidates: List[Dict], 
                                 tracking_result: Dict) -> str:
        """
        更新UI并等待用户输入
        
        Args:
            image: RGB图像
            candidates: 候选检测列表
            tracking_result: 追踪结果
            
        Returns:
            用户选择结果
        """
        try:
            # 存储当前数据
            self.current_image = image.copy()
            self.current_candidates = candidates.copy()
            self.current_tracking_result = tracking_result.copy()
            
            # 重置状态
            self.current_state = UIState.WAITING_CONFIRMATION
            self.user_response = None
            self.waiting_for_response = True
            
            # 更新UI显示
            self._update_ui_display()
            
            # 等待用户响应
            while self.waiting_for_response and not self.should_close:
                time.sleep(0.1)
            
            return self.user_response
            
        except Exception as e:
            print(f"[TRACKING_UI] 更新UI失败: {e}")
            return "error"
    
    def _update_ui_display(self):
        """更新UI显示"""
        if not self.root:
            return
        
        try:
            # 在UI线程中执行更新
            self.root.after(0, self._update_image_and_buttons)
        except Exception as e:
            print(f"[TRACKING_UI] UI显示更新失败: {e}")
    
    def _update_image_and_buttons(self):
        """更新图像和按钮显示"""
        try:
            # 更新图像
            if self.current_image is not None:
                processed_image = self._process_image_with_candidates()
                self._display_image(processed_image)
            
            # 更新状态信息
            if self.current_tracking_result:
                confidence = self.current_tracking_result.get('tracking_confidence', 0)
                target_id = self.current_tracking_result.get('target_id', 'unknown')
                step_info = f"Target: {target_id} | Confidence: {confidence:.3f}"
                self._update_status(step_info)
            
            # 更新按钮
            self._update_buttons_for_current_state()
            
        except Exception as e:
            print(f"[TRACKING_UI] 图像和按钮更新失败: {e}")
    
    def _process_image_with_candidates(self) -> np.ndarray:
        """处理图像，绘制候选框和ID - 修复绿色框显示"""
        try:
            img = self.current_image.copy()
            
            # 🔧 首先找到系统推荐的最佳候选
            best_candidate = self._find_best_candidate()
            
            # 为每个候选绘制边框和ID
            for i, candidate in enumerate(self.current_candidates):
                bbox = candidate.get('bounding_box', [])
                if len(bbox) >= 4:
                    x1, y1, x2, y2 = map(int, bbox)
                    candidate_id = i + 1
                    
                    # 🔧 判断是否是系统推荐的候选
                    is_best = (candidate == best_candidate)
                    
                    # 根据是否是最佳候选选择颜色
                    if is_best:
                        color = (0, 255, 0)  # 绿色 - 系统推荐
                        thickness = 4
                    else:
                        color = (255, 165, 0)  # 橙色 - 普通候选
                        thickness = self.config["visualization"]["bbox_thickness"]
                    
                    # 绘制边框
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
                    
                    # 绘制ID标签
                    label = str(candidate_id)
                    font_scale = self.config["visualization"]["font_scale"]
                    font_thickness = 2
                    
                    # 计算文字大小
                    (text_width, text_height), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
                    )
                    
                    # 绘制背景矩形 - 系统推荐用绿色背景
                    bg_color = (0, 200, 0) if is_best else (0, 0, 0)
                    cv2.rectangle(img, (x1, y1-text_height-10), 
                                (x1+text_width+10, y1), bg_color, -1)
                    
                    # 绘制文字
                    cv2.putText(img, label, (x1+5, y1-5), 
                              cv2.FONT_HERSHEY_SIMPLEX, font_scale, 
                              (255, 255, 255), font_thickness)
                    
                    # 🔧 如果是系统推荐，添加"SYSTEM"标记
                    if is_best:
                        system_label = "SYSTEM"
                        system_font_scale = 0.5
                        (sys_text_width, sys_text_height), _ = cv2.getTextSize(
                            system_label, cv2.FONT_HERSHEY_SIMPLEX, system_font_scale, 1
                        )
                        
                        # 在候选框右上角添加SYSTEM标记
                        sys_x = x2 - sys_text_width - 5
                        sys_y = y1 + sys_text_height + 5
                        
                        cv2.rectangle(img, (sys_x - 3, sys_y - sys_text_height - 3), 
                                    (sys_x + sys_text_width + 3, sys_y + 3), (0, 200, 0), -1)
                        cv2.putText(img, system_label, (sys_x, sys_y), 
                                  cv2.FONT_HERSHEY_SIMPLEX, system_font_scale, 
                                  (255, 255, 255), 1)
            
            return img
            
        except Exception as e:
            print(f"[TRACKING_UI] 图像处理失败: {e}")
            return self.current_image.copy()
    
    def _find_best_candidate(self) -> Optional[Dict]:
        """找到系统推荐的最佳候选 - 基于tracking_result"""
        try:
            if not self.current_tracking_result or not self.current_candidates:
                return None
            
            # 🔧 修复：基于追踪结果找到最佳候选
            # 从tracking_result中获取最佳匹配的坐标
            grasp_coord = self.current_tracking_result.get('grasp_coordinate', {})
            if not grasp_coord:
                # 备用方案：选择置信度最高的候选
                return max(self.current_candidates, key=lambda x: x.get('confidence', 0))
            
            target_x, target_y = grasp_coord.get('x', 0), grasp_coord.get('y', 0)
            
            # 找到与追踪结果坐标最接近的候选
            best_candidate = None
            min_distance = float('inf')
            
            for candidate in self.current_candidates:
                bbox = candidate.get('bounding_box', [])
                if len(bbox) >= 4:
                    x1, y1, x2, y2 = bbox
                    # 计算候选中心点
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    
                    # 计算距离（简化为2D距离）
                    distance = ((center_x - target_x) ** 2 + (center_y - target_y) ** 2) ** 0.5
                    
                    if distance < min_distance:
                        min_distance = distance
                        best_candidate = candidate
            
            return best_candidate or self.current_candidates[0]
            
        except Exception as e:
            print(f"[TRACKING_UI] 查找最佳候选失败: {e}")
            # 备用方案
            if self.current_candidates:
                return self.current_candidates[0]
            return None
    
    def _display_image(self, img: np.ndarray):
        """显示图像到UI"""
        try:
            # 调整图像大小
            display_size = self.config["ui"]["image_display_size"]
            img_resized = cv2.resize(img, tuple(display_size))
            
            # 转换为PIL图像
            img_pil = Image.fromarray(img_resized)
            img_tk = ImageTk.PhotoImage(img_pil)
            
            # 更新标签
            if self.image_label:
                self.image_label.configure(image=img_tk)
                self.image_label.image = img_tk  # 保持引用
                
        except Exception as e:
            print(f"[TRACKING_UI] 图像显示失败: {e}")
    
    def _update_status(self, status_text: str):
        """更新状态显示"""
        try:
            if self.status_label:
                self.status_label.configure(text=f"Status: {status_text}")
        except Exception as e:
            print(f"[TRACKING_UI] 状态更新失败: {e}")
    
    def _update_buttons_for_current_state(self):
        """根据当前状态更新按钮"""
        # 清除现有按钮
        self._clear_buttons()
        
        # 根据状态创建相应按钮
        if self.current_state == UIState.WAITING_CONFIRMATION:
            self._setup_waiting_confirmation_buttons()
        elif self.current_state == UIState.CORRECT_ACTIONS:
            self._setup_correct_actions_buttons()
        elif self.current_state == UIState.SELECT_CORRECT_ID:
            self._setup_select_id_buttons()
        elif self.current_state == UIState.GRASP_CONFIRMATION:
            self._setup_grasp_confirmation_buttons()
        elif self.current_state == UIState.AUTO_MODE:
            self._setup_auto_mode_buttons()
        elif self.current_state == UIState.DETECTION_FAILURE:
            self._setup_detection_failure_buttons()
    
    def _clear_buttons(self):
        """清除所有按钮"""
        for button in self.current_buttons:
            button.destroy()
        self.current_buttons.clear()
    
    def _create_button(self, text: str, command: Callable, color: str = "#e0e0e0") -> tk.Button:
        """创建按钮"""
        button = tk.Button(
            self.button_frame,
            text=text,
            command=command,
            font=('Arial', self.config["buttons"]["font_size"]),
            height=2,
            bg=color,
            relief='raised',
            bd=2
        )
        self.current_buttons.append(button)
        return button
    
    def _setup_waiting_confirmation_buttons(self):
        """设置等待确认状态的按钮"""
        buttons_config = [
            ("✓ Correct", self._on_correct, "#4CAF50"),
            ("✗ Wrong", self._on_wrong, "#F44336"),
            ("🔄 Retry", self._on_retry, "#FF9800"),
            ("❌ Quit", self._on_quit, "#757575")
        ]
        
        for i, (text, command, color) in enumerate(buttons_config):
            btn = self._create_button(text, command, color)
            btn.grid(row=0, column=i, padx=5, pady=5, sticky='ew')
        
        # 配置列权重
        for i in range(len(buttons_config)):
            self.button_frame.columnconfigure(i, weight=1)
    
    def _setup_correct_actions_buttons(self):
        """设置确认正确后的动作按钮"""
        buttons_config = [
            ("🚀 Quick Move", self._on_quick_move, "#2196F3"),
            ("📝 Move & Record", self._on_move_record, "#FF9800"),
            ("🤖 Auto Mode", self._on_auto_mode, "#9C27B0")
        ]
        
        for i, (text, command, color) in enumerate(buttons_config):
            btn = self._create_button(text, command, color)
            btn.grid(row=0, column=i, padx=5, pady=5, sticky='ew')
        
        for i in range(len(buttons_config)):
            self.button_frame.columnconfigure(i, weight=1)
    
    def _setup_select_id_buttons(self):
        """设置选择ID的按钮"""
        # 数字按钮 (1-9)
        num_candidates = min(len(self.current_candidates), 9)
        
        # 第一行：数字按钮
        for i in range(num_candidates):
            btn = self._create_button(
                str(i + 1), 
                lambda idx=i: self._on_select_id(idx),
                "#2196F3"
            )
            btn.grid(row=0, column=i, padx=2, pady=5, sticky='ew')
        
        # 第二行：控制按钮
        control_buttons = [
            ("❌ None Correct", self._on_none_correct, "#F44336"),
            ("⬅️ Back", self._on_back, "#757575")
        ]
        
        for i, (text, command, color) in enumerate(control_buttons):
            btn = self._create_button(text, command, color)
            btn.grid(row=1, column=i, padx=5, pady=5, sticky='ew')
        
        # 配置网格权重
        for i in range(max(num_candidates, len(control_buttons))):
            self.button_frame.columnconfigure(i, weight=1)
    
    def _setup_grasp_confirmation_buttons(self):
        """设置抓取确认按钮"""
        buttons_config = [
            ("🦾 Grasp", self._on_grasp, "#4CAF50"),
            ("⏭️ Continue", self._on_continue, "#2196F3"),
            ("❌ Cancel", self._on_cancel, "#757575")
        ]
        
        for i, (text, command, color) in enumerate(buttons_config):
            btn = self._create_button(text, command, color)
            btn.grid(row=0, column=i, padx=5, pady=5, sticky='ew')
        
        for i in range(len(buttons_config)):
            self.button_frame.columnconfigure(i, weight=1)
    
    def _setup_auto_mode_buttons(self):
        """设置自动模式按钮"""
        buttons_config = [
            ("⏸️ Pause Auto", self._on_pause_auto, "#FF9800"),
            ("❌ Stop Auto", self._on_stop_auto, "#F44336")
        ]
        
        for i, (text, command, color) in enumerate(buttons_config):
            btn = self._create_button(text, command, color)
            btn.grid(row=0, column=i, padx=5, pady=5, sticky='ew')
        
        for i in range(len(buttons_config)):
            self.button_frame.columnconfigure(i, weight=1)
    
    def _setup_detection_failure_buttons(self):
        """设置检测失败按钮"""
        buttons_config = [
            ("🔄 Retry Detection", self._on_retry, "#FF9800"),
            ("⬅️ Rollback", self._on_rollback, "#F44336"),
            ("❌ Quit", self._on_quit, "#757575")
        ]
        
        for i, (text, command, color) in enumerate(buttons_config):
            btn = self._create_button(text, command, color)
            btn.grid(row=0, column=i, padx=5, pady=5, sticky='ew')
        
        for i in range(len(buttons_config)):
            self.button_frame.columnconfigure(i, weight=1)
    
    # ======================== 按钮事件处理 ========================
    
    def _on_correct(self):
        """系统推荐正确"""
        self.current_state = UIState.CORRECT_ACTIONS
        self._update_buttons_for_current_state()
    
    def _on_wrong(self):
        """系统推荐错误"""
        self.current_state = UIState.SELECT_CORRECT_ID
        self._update_buttons_for_current_state()
    
    def _on_quick_move(self):
        """快速移动"""
        self._send_response("quick_move")
    
    def _on_move_record(self):
        """移动并记录"""
        self._send_response("move_record")
    
    def _on_auto_mode(self):
        """自动模式"""
        self.current_state = UIState.AUTO_MODE
        self._update_buttons_for_current_state()
        self._send_response("auto_mode")
    
    def _on_select_id(self, candidate_idx: int):
        """选择候选ID"""
        self._send_response(f"select_id_{candidate_idx}")
    
    def _on_none_correct(self):
        """没有正确选项"""
        self._send_response("none_correct")
    
    def _on_back(self):
        """返回上一步"""
        self.current_state = UIState.WAITING_CONFIRMATION
        self._update_buttons_for_current_state()
    
    def _on_grasp(self):
        """执行抓取"""
        self._send_response("grasp")
    
    def _on_continue(self):
        """继续追踪"""
        self._send_response("continue")
    
    def _on_cancel(self):
        """取消"""
        self._send_response("cancel")
    
    def _on_retry(self):
        """重试检测"""
        self._send_response("retry")
    
    def _on_rollback(self):
        """回退"""
        self._send_response("rollback")
    
    def _on_pause_auto(self):
        """暂停自动模式"""
        self.current_state = UIState.WAITING_CONFIRMATION
        self._update_buttons_for_current_state()
        self._send_response("pause_auto")
    
    def _on_stop_auto(self):
        """停止自动模式"""
        self.current_state = UIState.WAITING_CONFIRMATION
        self._update_buttons_for_current_state()
        self._send_response("stop_auto")
    
    def _on_quit(self):
        """退出"""
        self._send_response("quit")
    
    def _send_response(self, response: str):
        """发送用户响应"""
        self.user_response = response
        self.waiting_for_response = False
        print(f"[TRACKING_UI] 用户响应: {response}")
    
    def show_detection_failure(self):
        """显示检测失败状态"""
        self.current_state = UIState.DETECTION_FAILURE
        self._update_status("Detection Failed! Please check camera and environment.")
        if self.root:
            self.root.after(0, self._update_buttons_for_current_state)
    
    def show_grasp_ready(self):
        """显示抓取准备状态"""
        self.current_state = UIState.GRASP_CONFIRMATION
        self._update_status("Grasp conditions met! Ready to grasp.")
        if self.root:
            self.root.after(0, self._update_buttons_for_current_state)
    
    def _on_closing(self):
        """窗口关闭事件"""
        self.should_close = True
        self.waiting_for_response = False
        if self.root:
            self.root.quit()
            self.root.destroy()
    
    def close(self):
        """关闭UI"""
        self.should_close = True
        if self.root:
            self.root.after(0, self._on_closing)