# learning_curve_visualizer.py
import os
import json
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import pandas as pd
from pathlib import Path

class LearningCurveVisualizer:
    """学习曲线可视化工具 - 专为论文设计"""
    
    def __init__(self, scan_directory: str):
        self.scan_dir = scan_directory
        self.data_dir = os.path.join(scan_directory, 'tracking_results')
        self.step_details_dir = os.path.join(self.data_dir, 'step_details')
        
        # 设置绘图样式
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
        
        # 论文级别的绘图配置
        self.fig_config = {
            'figsize': (12, 8),
            'dpi': 300,
            'font_size': 12,
            'title_size': 14,
            'legend_size': 10
        }
        
    def generate_comprehensive_learning_curves(self, save_path: Optional[str] = None):
        """生成综合学习曲线图 - 论文级别"""
        try:
            # 1. 加载所有学习数据
            learning_data = self._load_all_learning_data()
            
            if not learning_data:
                print("❌ 没有找到学习数据")
                return None
            
            # 2. 创建多子图布局
            fig = plt.figure(figsize=(16, 12))
            gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], width_ratios=[1, 1])
            
            # 3. 子图1: 置信度学习曲线
            ax1 = fig.add_subplot(gs[0, 0])
            self._plot_confidence_learning_curve(ax1, learning_data)
            
            # 4. 子图2: 混合相似度演化
            ax2 = fig.add_subplot(gs[0, 1])
            self._plot_hybrid_similarity_evolution(ax2, learning_data)
            
            # 5. 子图3: 自适应权重变化
            ax3 = fig.add_subplot(gs[1, 0])
            self._plot_adaptive_weights_evolution(ax3, learning_data)
            
            # 6. 子图4: 成功率统计
            ax4 = fig.add_subplot(gs[1, 1])
            self._plot_success_rate_analysis(ax4, learning_data)
            
            # 7. 子图5: 特征重要性热力图
            ax5 = fig.add_subplot(gs[2, :])
            self._plot_feature_importance_heatmap(ax5, learning_data)
            
            # 8. 总体标题和布局调整
            fig.suptitle('Hybrid Similarity Tracking - Learning Curves Analysis', 
                        fontsize=16, fontweight='bold', y=0.98)
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            # 9. 保存图像
            if save_path is None:
                save_path = os.path.join(self.data_dir, 'learning_curves_comprehensive.png')
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"✅ 综合学习曲线已保存: {save_path}")
            
            # 10. 生成论文数据统计
            self._generate_paper_statistics(learning_data)
            
            return save_path
            
        except Exception as e:
            print(f"❌ 生成学习曲线失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _load_all_learning_data(self) -> Dict:
        """加载所有学习相关数据"""
        try:
            learning_data = {
                'step_details': [],
                'parameters_history': [],
                'detection_history': {},
                'adaptive_learning': {},
                'session_summary': {}
            }
            
            # 1. 加载步骤详细数据
            if os.path.exists(self.step_details_dir):
                for file in os.listdir(self.step_details_dir):
                    if file.endswith('_detailed.json'):
                        file_path = os.path.join(self.step_details_dir, file)
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                step_data = json.load(f)
                            learning_data['step_details'].append(step_data)
                        except Exception as e:
                            print(f"⚠️ 加载 {file} 失败: {e}")
            
            # 2. 加载参数历史
            params_file = os.path.join(self.data_dir, 'tracking_parameters_history.json')
            if os.path.exists(params_file):
                with open(params_file, 'r', encoding='utf-8') as f:
                    learning_data['parameters_history'] = json.load(f)
            
            # 3. 加载检测历史
            history_dir = os.path.join(self.scan_dir, 'detection_history')
            history_file = os.path.join(history_dir, 'successful_detection_history.json')
            if os.path.exists(history_file):
                with open(history_file, 'r', encoding='utf-8') as f:
                    learning_data['detection_history'] = json.load(f)
            
            # 4. 加载自适应学习数据
            adaptive_file = os.path.join(self.scan_dir, 'adaptive_learning.json')
            if os.path.exists(adaptive_file):
                with open(adaptive_file, 'r', encoding='utf-8') as f:
                    learning_data['adaptive_learning'] = json.load(f)
            
            # 5. 加载会话摘要
            summary_file = os.path.join(self.step_details_dir, 'session_detailed_summary.json')
            if os.path.exists(summary_file):
                with open(summary_file, 'r', encoding='utf-8') as f:
                    learning_data['session_summary'] = json.load(f)
            
            print(f"✅ 学习数据加载完成:")
            print(f"  - 步骤详情: {len(learning_data['step_details'])} 个")
            print(f"  - 参数历史: {len(learning_data['parameters_history'])} 个")
            print(f"  - 检测历史: {len(learning_data['detection_history'])} 个目标")
            print(f"  - 自适应数据: {'已加载' if learning_data['adaptive_learning'] else '未找到'}")
            
            return learning_data
            
        except Exception as e:
            print(f"❌ 加载学习数据失败: {e}")
            return {}
    
    def _plot_confidence_learning_curve(self, ax, learning_data):
        """绘制置信度学习曲线"""
        try:
            # 从步骤详情中提取置信度数据
            steps = []
            confidences = []
            user_feedbacks = []
            
            for step_data in learning_data['step_details']:
                step_num = step_data.get('step_number', 0)
                tracking_result = step_data.get('tracking_result', {})
                confidence = tracking_result.get('tracking_confidence', 0)
                feedback = step_data.get('user_feedback', 'no_feedback')
                
                if step_num > 0 and confidence > 0:
                    steps.append(step_num)
                    confidences.append(confidence)
                    user_feedbacks.append(feedback)
            
            if not steps:
                ax.text(0.5, 0.5, 'No confidence data available', 
                       ha='center', va='center', transform=ax.transAxes)
                return
            
            # 绘制置信度曲线
            ax.plot(steps, confidences, 'b-', linewidth=2, alpha=0.7, label='Tracking Confidence')
            
            # 标记用户反馈
            success_steps = [s for s, f in zip(steps, user_feedbacks) if f == 'success']
            success_confs = [c for s, c, f in zip(steps, confidences, user_feedbacks) if f == 'success']
            failure_steps = [s for s, f in zip(steps, user_feedbacks) if f == 'failure']
            failure_confs = [c for s, c, f in zip(steps, confidences, user_feedbacks) if f == 'failure']
            
            if success_steps:
                ax.scatter(success_steps, success_confs, c='green', s=50, alpha=0.8, label='User Confirmed Success', zorder=5)
            if failure_steps:
                ax.scatter(failure_steps, failure_confs, c='red', s=50, alpha=0.8, label='User Marked Failure', zorder=5)
            
            # 添加趋势线
            if len(steps) > 2:
                z = np.polyfit(steps, confidences, 1)
                p = np.poly1d(z)
                ax.plot(steps, p(steps), "r--", alpha=0.6, linewidth=1, label='Trend Line')
            
            ax.set_xlabel('Tracking Step')
            ax.set_ylabel('Confidence Score')
            ax.set_title('Confidence Learning Curve')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)
            
        except Exception as e:
            print(f"❌ 绘制置信度学习曲线失败: {e}")
    
    def _plot_hybrid_similarity_evolution(self, ax, learning_data):
        """绘制混合相似度演化图"""
        try:
            steps = []
            reference_scores = []
            historical_scores = []
            final_scores = []
            history_counts = []
            
            for step_data in learning_data['step_details']:
                step_num = step_data.get('step_number', 0)
                tracking_result = step_data.get('tracking_result', {})
                hybrid_info = tracking_result.get('hybrid_similarity_info', {})
                
                if step_num > 0 and hybrid_info:
                    steps.append(step_num)
                    reference_scores.append(hybrid_info.get('reference_score', 0))
                    historical_scores.append(hybrid_info.get('historical_score', 0))
                    final_scores.append(tracking_result.get('tracking_confidence', 0))
                    history_counts.append(hybrid_info.get('history_count', 0))
            
            if not steps:
                ax.text(0.5, 0.5, 'No hybrid similarity data available', 
                       ha='center', va='center', transform=ax.transAxes)
                return
            
            # 绘制三条曲线
            ax.plot(steps, reference_scores, 'b-', linewidth=2, label='Reference Library Score', alpha=0.8)
            ax.plot(steps, historical_scores, 'g-', linewidth=2, label='Historical Score', alpha=0.8)
            ax.plot(steps, final_scores, 'r-', linewidth=2, label='Final Hybrid Score', alpha=0.8)
            
            # 添加历史计数的辅助图
            ax2 = ax.twinx()
            ax2.bar(steps, history_counts, alpha=0.3, width=0.8, color='orange', label='History Count')
            ax2.set_ylabel('History Count', color='orange')
            ax2.tick_params(axis='y', labelcolor='orange')
            
            ax.set_xlabel('Tracking Step')
            ax.set_ylabel('Similarity Score')
            ax.set_title('Hybrid Similarity Evolution')
            ax.legend(loc='upper left')
            ax2.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)
            
        except Exception as e:
            print(f"❌ 绘制混合相似度演化失败: {e}")
    
    def _plot_adaptive_weights_evolution(self, ax, learning_data):
        """绘制自适应权重变化"""
        try:
            # 从步骤详情中提取权重数据
            steps = []
            weights_data = {'geometric': [], 'appearance': [], 'shape': [], 'spatial': []}
            
            for step_data in learning_data['step_details']:
                step_num = step_data.get('step_number', 0)
                adaptive_weights = step_data.get('adaptive_weights_used', {})
                
                if step_num > 0 and adaptive_weights:
                    steps.append(step_num)
                    for feature_type in weights_data.keys():
                        weight = adaptive_weights.get(feature_type, 0)
                        weights_data[feature_type].append(weight)
            
            if not steps:
                ax.text(0.5, 0.5, 'No adaptive weights data available', 
                       ha='center', va='center', transform=ax.transAxes)
                return
            
            # 绘制各特征权重变化
            colors = ['blue', 'green', 'red', 'orange']
            for i, (feature_type, weights) in enumerate(weights_data.items()):
                if weights:  # 只绘制有数据的特征
                    ax.plot(steps, weights, color=colors[i], linewidth=2, 
                           marker='o', markersize=4, label=f'{feature_type.title()} Weight')
            
            ax.set_xlabel('Tracking Step')
            ax.set_ylabel('Feature Weight')
            ax.set_title('Adaptive Feature Weights Evolution')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)
            
        except Exception as e:
            print(f"❌ 绘制自适应权重变化失败: {e}")
    
    def _plot_success_rate_analysis(self, ax, learning_data):
        """绘制成功率分析"""
        try:
            # 统计用户反馈
            feedback_counts = {'success': 0, 'failure': 0, 'auto_success': 0, 'no_feedback': 0}
            total_steps = 0
            
            for step_data in learning_data['step_details']:
                feedback = step_data.get('user_feedback', 'no_feedback')
                feedback_counts[feedback] = feedback_counts.get(feedback, 0) + 1
                total_steps += 1
            
            if total_steps == 0:
                ax.text(0.5, 0.5, 'No feedback data available', 
                       ha='center', va='center', transform=ax.transAxes)
                return
            
            # 计算成功率
            successful_steps = feedback_counts['success'] + feedback_counts.get('auto_success', 0)
            success_rate = successful_steps / total_steps if total_steps > 0 else 0
            
            # 绘制饼图
            labels = []
            sizes = []
            colors = []
            
            color_map = {
                'success': 'lightgreen',
                'auto_success': 'darkgreen', 
                'failure': 'lightcoral',
                'no_feedback': 'lightgray'
            }
            
            for feedback_type, count in feedback_counts.items():
                if count > 0:
                    labels.append(f'{feedback_type.title()}\n({count})')
                    sizes.append(count)
                    colors.append(color_map.get(feedback_type, 'lightblue'))
            
            wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                            startangle=90, textprops={'fontsize': 9})
            
            ax.set_title(f'User Feedback Distribution\n(Success Rate: {success_rate:.1%})')
            
        except Exception as e:
            print(f"❌ 绘制成功率分析失败: {e}")
    
    def _plot_feature_importance_heatmap(self, ax, learning_data):
        """绘制特征重要性热力图"""
        try:
            # 收集特征权重数据
            feature_weights_over_time = []
            steps = []
            
            for step_data in learning_data['step_details']:
                step_num = step_data.get('step_number', 0)
                weights = step_data.get('adaptive_weights_used', {})
                
                if step_num > 0 and weights:
                    steps.append(f'Step {step_num}')
                    feature_weights_over_time.append(weights)
            
            if not feature_weights_over_time:
                ax.text(0.5, 0.5, 'No feature weights data available', 
                       ha='center', va='center', transform=ax.transAxes)
                return
            
            # 创建数据框
            df = pd.DataFrame(feature_weights_over_time, index=steps)
            df = df.fillna(0)  # 填充缺失值
            
            # 绘制热力图
            sns.heatmap(df.T, annot=True, fmt='.3f', cmap='YlOrRd', 
                       cbar_kws={'label': 'Feature Weight'}, ax=ax)
            
            ax.set_title('Feature Importance Heatmap Over Time')
            ax.set_xlabel('Tracking Steps')
            ax.set_ylabel('Feature Types')
            
        except Exception as e:
            print(f"❌ 绘制特征重要性热力图失败: {e}")
    
    def _generate_paper_statistics(self, learning_data):
        """生成论文用的统计数据"""
        try:
            stats = {
                'session_overview': {},
                'learning_performance': {},
                'adaptive_behavior': {},
                'user_interaction': {}
            }
            
            # 1. 会话概览
            total_steps = len(learning_data['step_details'])
            stats['session_overview'] = {
                'total_tracking_steps': total_steps,
                'successful_detections': len([s for s in learning_data['step_details'] 
                                            if s.get('user_feedback') == 'success']),
                'auto_successful_detections': len([s for s in learning_data['step_details'] 
                                                 if s.get('user_feedback') == 'auto_success']),
                'failed_detections': len([s for s in learning_data['step_details'] 
                                        if s.get('user_feedback') == 'failure'])
            }
            
            # 2. 学习性能分析
            confidences = []
            hybrid_scores = []
            
            for step_data in learning_data['step_details']:
                tracking_result = step_data.get('tracking_result', {})
                confidence = tracking_result.get('tracking_confidence', 0)
                if confidence > 0:
                    confidences.append(confidence)
                    hybrid_info = tracking_result.get('hybrid_similarity_info', {})
                    if hybrid_info:
                        hybrid_scores.append(hybrid_info.get('reference_score', 0))
            
            if confidences:
                stats['learning_performance'] = {
                    'mean_confidence': np.mean(confidences),
                    'std_confidence': np.std(confidences),
                    'confidence_improvement': confidences[-1] - confidences[0] if len(confidences) > 1 else 0,
                    'mean_reference_score': np.mean(hybrid_scores) if hybrid_scores else 0
                }
            
            # 3. 自适应行为分析
            adaptive_activations = len([s for s in learning_data['step_details'] 
                                      if s.get('adaptive_weights_used')])
            
            stats['adaptive_behavior'] = {
                'adaptive_weight_activations': adaptive_activations,
                'adaptation_rate': adaptive_activations / total_steps if total_steps > 0 else 0,
                'total_detection_history_entries': sum(len(hist) for hist in learning_data['detection_history'].values())
            }
            
            # 4. 用户交互分析
            manual_confirmations = len([s for s in learning_data['step_details'] 
                                      if s.get('user_feedback') in ['success', 'failure']])
            auto_decisions = len([s for s in learning_data['step_details'] 
                                if s.get('user_feedback') == 'auto_success'])
            
            stats['user_interaction'] = {
                'manual_confirmations': manual_confirmations,
                'automatic_decisions': auto_decisions,
                'automation_rate': auto_decisions / (manual_confirmations + auto_decisions) if (manual_confirmations + auto_decisions) > 0 else 0
            }
            
            # 保存统计数据
            stats_file = os.path.join(self.data_dir, 'paper_statistics.json')
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
            
            print(f"✅ 论文统计数据已保存: {stats_file}")
            
            # 打印关键统计信息
            print("\n📊 关键统计信息 (论文用):")
            print(f"  总追踪步数: {stats['session_overview']['total_tracking_steps']}")
            print(f"  成功率: {(stats['session_overview']['successful_detections'] + stats['session_overview']['auto_successful_detections']) / total_steps:.1%}")
            print(f"  自动化率: {stats['user_interaction']['automation_rate']:.1%}")
            print(f"  平均置信度: {stats['learning_performance'].get('mean_confidence', 0):.3f}")
            print(f"  自适应激活率: {stats['adaptive_behavior']['adaptation_rate']:.1%}")
            
            return stats
            
        except Exception as e:
            print(f"❌ 生成论文统计失败: {e}")
            return {}
    
    def export_learning_data_for_analysis(self, output_file: str = None):
        """导出学习数据用于进一步分析"""
        try:
            learning_data = self._load_all_learning_data()
            
            if output_file is None:
                output_file = os.path.join(self.data_dir, 'learning_data_export.csv')
            
            # 转换为CSV格式
            rows = []
            for step_data in learning_data['step_details']:
                tracking_result = step_data.get('tracking_result', {})
                hybrid_info = tracking_result.get('hybrid_similarity_info', {})
                
                row = {
                    'step': step_data.get('step_number', 0),
                    'tracking_confidence': tracking_result.get('tracking_confidence', 0),
                    'reference_score': hybrid_info.get('reference_score', 0),
                    'historical_score': hybrid_info.get('historical_score', 0),
                    'history_count': hybrid_info.get('history_count', 0),
                    'user_feedback': step_data.get('user_feedback', 'no_feedback'),
                    'grasp_ready': step_data.get('grasp_conditions', {}).get('grasp_ready', False),
                    'candidates_count': len(step_data.get('all_candidates', [])),
                    'adaptive_weights_used': bool(step_data.get('adaptive_weights_used'))
                }
                
                # 添加自适应权重
                adaptive_weights = step_data.get('adaptive_weights_used', {})
                for feature_type in ['geometric', 'appearance', 'shape', 'spatial']:
                    row[f'weight_{feature_type}'] = adaptive_weights.get(feature_type, 0)
                
                rows.append(row)
            
            # 保存为CSV
            df = pd.DataFrame(rows)
            df.to_csv(output_file, index=False)
            
            print(f"✅ 学习数据已导出为CSV: {output_file}")
            print(f"  包含 {len(rows)} 个追踪步骤的数据")
            print(f"  字段数: {len(df.columns)}")
            
            return output_file
            
        except Exception as e:
            print(f"❌ 导出学习数据失败: {e}")
            return None


def main():
    """主函数 - 演示如何使用"""
    import sys
    
    if len(sys.argv) != 2:
        print("用法: python learning_curve_visualizer.py <scan_directory>")
        print("例如: python learning_curve_visualizer.py /home/qi/ros2_ws/scan_output_20250728_214915")
        return
    
    scan_directory = sys.argv[1]
    
    if not os.path.exists(scan_directory):
        print(f"❌ 扫描目录不存在: {scan_directory}")
        return
    
    print(f"🚀 开始分析追踪学习数据: {scan_directory}")
    
    # 创建可视化工具
    visualizer = LearningCurveVisualizer(scan_directory)
    
    # 1. 生成综合学习曲线
    curve_path = visualizer.generate_comprehensive_learning_curves()
    
    # 2. 导出CSV数据用于进一步分析
    csv_path = visualizer.export_learning_data_for_analysis()
    
    if curve_path and csv_path:
        print(f"\n✅ 学习曲线分析完成!")
        print(f"📈 图像文件: {curve_path}")
        print(f"📊 数据文件: {csv_path}")
        print(f"📄 论文统计: {os.path.join(visualizer.data_dir, 'paper_statistics.json')}")
    else:
        print("❌ 学习曲线生成失败，请检查数据完整性")


if __name__ == '__main__':
    main()