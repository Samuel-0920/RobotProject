# utils/feedback_collector.py
import sys
import threading
import time
from typing import Optional

class FeedbackCollector:
    """人工反馈收集器 - 简化的y/n输入"""
    
    def __init__(self, logger=None):
        self.logger = logger
        self.feedback_timeout = 30.0  # 30秒超时
    
    def collect_feedback(self, prompt: str) -> Optional[bool]:
        """
        收集用户反馈
        
        Args:
            prompt: 提示信息
            
        Returns:
            bool: True表示正确(y), False表示错误(n), None表示超时或错误
        """
        try:
            if self.logger:
                self.logger.info(f"⏳ {prompt}")
            
            print(f"\n{prompt}")
            print("请输入: y (正确) 或 n (错误)")
            
            # 创建输入线程，支持超时
            result = [None]  # 使用列表避免闭包问题
            input_thread = threading.Thread(
                target=self._input_worker, 
                args=(result,)
            )
            input_thread.daemon = True
            input_thread.start()
            
            # 等待输入或超时
            input_thread.join(timeout=self.feedback_timeout)
            
            if input_thread.is_alive():
                if self.logger:
                    self.logger.warn("⏰ 反馈输入超时，跳过该步骤")
                print("输入超时，跳过该步骤")
                return None
            
            return result[0]
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"收集反馈失败: {e}")
            return None
    
    def _input_worker(self, result_container):
        """输入工作线程"""
        try:
            while True:
                user_input = input(">> ").strip().lower()
                
                if user_input in ['y', 'yes', '是', '对', '正确']:
                    result_container[0] = True
                    print("✅ 已记录：正确")
                    break
                elif user_input in ['n', 'no', '否', '错', '错误']:
                    result_container[0] = False
                    print("❌ 已记录：错误")
                    break
                else:
                    print("请输入有效选项：y (正确) 或 n (错误)")
                    continue
                    
        except Exception as e:
            print(f"输入处理错误: {e}")
            result_container[0] = None
    
    def collect_batch_feedback(self, steps_data: list) -> list:
        """
        批量收集多个步骤的反馈
        
        Args:
            steps_data: 步骤数据列表
            
        Returns:
            list: 反馈结果列表
        """
        feedbacks = []
        
        print(f"\n📋 需要对 {len(steps_data)} 个追踪步骤进行确认")
        print("=" * 50)
        
        for i, step_data in enumerate(steps_data):
            print(f"\n--- 步骤 {i+1}/{len(steps_data)} ---")
            
            # 显示步骤信息
            if 'grasp_coordinate' in step_data:
                coord = step_data['grasp_coordinate']
                print(f"追踪到目标位置: ({coord['x']:.1f}, {coord['y']:.1f}, {coord['z']:.1f})")
            
            if 'tracking_confidence' in step_data:
                confidence = step_data['tracking_confidence']
                print(f"追踪置信度: {confidence:.3f}")
            
            # 收集该步骤的反馈
            feedback = self.collect_feedback(f"步骤 {i+1} 的追踪结果是否正确？")
            feedbacks.append(feedback)
            
            # 如果用户取消，询问是否继续
            if feedback is None:
                continue_feedback = input("是否继续剩余步骤的反馈？(y/n): ").strip().lower()
                if continue_feedback not in ['y', 'yes', '是']:
                    break
        
        return feedbacks
    
    def show_feedback_summary(self, feedbacks: list, target_id: str):
        """显示反馈汇总"""
        try:
            correct_count = sum(1 for f in feedbacks if f is True)
            incorrect_count = sum(1 for f in feedbacks if f is False)
            timeout_count = sum(1 for f in feedbacks if f is None)
            
            print(f"\n📊 反馈汇总 - 目标: {target_id}")
            print("=" * 40)
            print(f"✅ 正确: {correct_count} 步")
            print(f"❌ 错误: {incorrect_count} 步")
            print(f"⏰ 超时: {timeout_count} 步")
            print(f"📈 准确率: {correct_count/(len(feedbacks) - timeout_count):.1%}" if len(feedbacks) > timeout_count else "无有效反馈")
            print("=" * 40)
            
            if self.logger:
                self.logger.info(f"反馈汇总 - 正确:{correct_count}, 错误:{incorrect_count}, 超时:{timeout_count}")
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"显示反馈汇总失败: {e}")
    
    def collect_session_feedback(self, session_summary: dict) -> dict:
        """
        收集整体会话反馈
        
        Args:
            session_summary: 会话汇总信息
            
        Returns:
            dict: 会话反馈数据
        """
        try:
            print(f"\n🎯 追踪会话完成")
            print("=" * 50)
            print(f"目标: {session_summary.get('target_id', 'unknown')}")
            print(f"总步数: {session_summary.get('total_steps', 0)}")
            print(f"平均置信度: {session_summary.get('avg_confidence', 0):.3f}")
            
            # 整体满意度
            overall_feedback = self.collect_feedback("整体追踪效果是否满意？")
            
            # 收集额外意见（可选）
            print("\n💬 其他意见或建议（可选，按Enter跳过）:")
            comments = input(">> ").strip()
            
            feedback_data = {
                'overall_satisfaction': overall_feedback,
                'comments': comments if comments else None,
                'timestamp': time.time(),
                'session_summary': session_summary
            }
            
            return feedback_data
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"收集会话反馈失败: {e}")
            return {'overall_satisfaction': None, 'comments': None}