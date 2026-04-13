# utils/tracking_visualizer.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict

class TrackingVisualizer:
    """追踪可视化器 - 显示追踪结果和反馈界面"""
    
    def __init__(self):
        self.window_name = "Tracking Results"
    
    def show_step_result(self, tracking_record: Dict, step_number: int):
        """显示单步追踪结果"""
        try:
            print(f"\n=== 步骤 {step_number} 追踪结果 ===")
            
            # 显示基本信息
            if 'tracking_result' in tracking_record:
                result = tracking_record['tracking_result']
                print(f"目标ID: {result.get('target_id', 'unknown')}")
                print(f"追踪置信度: {result.get('tracking_confidence', 0):.3f}")
                
                if 'grasp_coordinate' in result:
                    coord = result['grasp_coordinate']
                    print(f"抓取坐标: ({coord['x']:.1f}, {coord['y']:.1f}, {coord['z']:.1f})")
            
            # 加载并显示图像（如果有）
            if 'image_files' in tracking_record:
                rgb_path = tracking_record['image_files'].get('rgb')
                if rgb_path:
                    # 这里可以显示图像，简化版本只打印路径
                    print(f"图像文件: {rgb_path}")
            
            print("=" * 40)
            
        except Exception as e:
            print(f"显示步骤结果失败: {e}")
    
    def show_tracking_summary(self, tracking_history: List[Dict], target_id: str):
        """显示追踪会话摘要"""
        try:
            print(f"\n🎯 追踪摘要 - 目标: {target_id}")
            print("=" * 50)
            print(f"总步数: {len(tracking_history)}")
            
            # 统计成功/失败
            success_count = sum(1 for r in tracking_history if 'tracking_result' in r)
            print(f"成功步数: {success_count}")
            print(f"失败步数: {len(tracking_history) - success_count}")
            
            # 平均置信度
            confidences = []
            for record in tracking_history:
                if 'tracking_result' in record:
                    conf = record['tracking_result'].get('tracking_confidence', 0)
                    confidences.append(conf)
            
            if confidences:
                print(f"平均置信度: {np.mean(confidences):.3f}")
            
            print("=" * 50)
            
        except Exception as e:
            print(f"显示追踪摘要失败: {e}")