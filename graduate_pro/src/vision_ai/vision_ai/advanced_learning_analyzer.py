# advanced_learning_analyzer.py
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import pandas as pd
import json
import os
from typing import Dict, List, Tuple, Optional

class AdvancedLearningAnalyzer:
    """高级学习性能分析器 - 专为论文深度分析设计"""
    
    def __init__(self, learning_data: Dict):
        self.data = learning_data
        self.processed_data = None
        
    def comprehensive_analysis(self, save_dir: str):
        """执行综合性学习分析"""
        try:
            print("🔬 开始高级学习性能分析...")
            
            # 1. 数据预处理
            self.processed_data = self._preprocess_learning_data()
            
            # 2. 执行各种分析
            analyses = {
                'convergence_analysis': self._analyze_convergence(),
                'stability_analysis': self._analyze_learning_stability(),
                'adaptation_effectiveness': self._analyze_adaptation_effectiveness(),
                'feature_learning_dynamics': self._analyze_feature_learning(),
                'user_interaction_impact': self._analyze_user_interaction_impact(),
                'generalization_analysis': self._analyze_generalization()
            }
            
            # 3. 生成论文级可视化
            self._generate_paper_visualizations(analyses, save_dir)
            
            # 4. 生成定量分析报告
            report = self._generate_quantitative_report(analyses)
            
            # 5. 保存分析结果
            self._save_analysis_results(analyses, report, save_dir)
            
            return analyses, report
            
        except Exception as e:
            print(f"❌ 高级分析失败: {e}")
            return None, None
    
    def _preprocess_learning_data(self) -> pd.DataFrame:
        """预处理学习数据为分析格式"""
        try:
            rows = []
            
            for step_data in self.data['step_details']:
                tracking_result = step_data.get('tracking_result', {})
                hybrid_info = tracking_result.get('hybrid_similarity_info', {})
                
                row = {
                    'step': step_data.get('step_number', 0),
                    'tracking_confidence': tracking_result.get('tracking_confidence', 0),
                    'reference_score': hybrid_info.get('reference_score', 0),
                    'historical_score': hybrid_info.get('historical_score', 0),
                    'final_hybrid_score': hybrid_info.get('final_score', tracking_result.get('tracking_confidence', 0)),
                    'history_count': hybrid_info.get('history_count', 0),
                    'reference_weight': hybrid_info.get('reference_weight', 1.0),
                    'historical_weight': hybrid_info.get('historical_weight', 0.0),
                    'user_feedback': step_data.get('user_feedback', 'no_feedback'),
                    'auto_success': tracking_result.get('auto_success', False),
                    'grasp_ready': step_data.get('grasp_conditions', {}).get('grasp_ready', False),
                    'candidates_count': len(step_data.get('all_candidates', [])),
                    'has_adaptive_weights': bool(step_data.get('adaptive_weights_used')),
                    'timestamp': step_data.get('timestamp', '')
                }
                
                # 添加自适应权重
                adaptive_weights = step_data.get('adaptive_weights_used', {})
                for feature_type in ['geometric', 'appearance', 'shape', 'spatial']:
                    row[f'weight_{feature_type}'] = adaptive_weights.get(feature_type, 0)
                
                # 计算衍生指标
                row['learning_progress'] = min(row['history_count'] / 5.0, 1.0)  # 归一化学习进度
                row['hybrid_effectiveness'] = row['final_hybrid_score'] - row['reference_score']
                row['is_successful'] = row['user_feedback'] in ['success', 'auto_success']
                
                rows.append(row)
            
            df = pd.DataFrame(rows)
            df = df.sort_values('step').reset_index(drop=True)
            
            print(f"✅ 预处理完成: {len(df)} 个数据点, {len(df.columns)} 个特征")
            return df
            
        except Exception as e:
            print(f"❌ 数据预处理失败: {e}")
            return pd.DataFrame()
    
    def _analyze_convergence(self) -> Dict:
        """分析学习收敛性"""
        try:
            df = self.processed_data
            
            # 1. 置信度收敛分析
            confidences = df['tracking_confidence'].values
            steps = df['step'].values
            
            # 计算移动平均
            window_size = min(5, len(confidences) // 2)
            if window_size > 1:
                moving_avg = np.convolve(confidences, np.ones(window_size)/window_size, mode='valid')
                moving_steps = steps[window_size-1:]
            else:
                moving_avg = confidences
                moving_steps = steps
            
            # 检测收敛点
            convergence_threshold = 0.05
            convergence_step = None
            
            if len(moving_avg) > 3:
                for i in range(3, len(moving_avg)):
                    recent_variance = np.var(moving_avg[i-3:i+1])
                    if recent_variance < convergence_threshold:
                        convergence_step = moving_steps[i]
                        break
            
            # 学习率分析
            if len(confidences) > 1:
                learning_rate = np.diff(confidences).mean()
                learning_acceleration = np.diff(np.diff(confidences)).mean() if len(confidences) > 2 else 0
            else:
                learning_rate = 0
                learning_acceleration = 0
            
            # 拟合学习曲线
            curve_fit_params = None
            if len(steps) > 3:
                try:
                    # 指数拟合: y = a * (1 - exp(-b * x)) + c
                    from scipy.optimize import curve_fit
                    
                    def exponential_learning(x, a, b, c):
                        return a * (1 - np.exp(-b * x)) + c
                    
                    popt, _ = curve_fit(exponential_learning, steps, confidences, 
                                       bounds=([0, 0, 0], [2, 1, 1]), maxfev=1000)
                    curve_fit_params = popt
                except:
                    curve_fit_params = None
            
            return {
                'convergence_step': convergence_step,
                'final_confidence': confidences[-1] if len(confidences) > 0 else 0,
                'initial_confidence': confidences[0] if len(confidences) > 0 else 0,
                'total_improvement': confidences[-1] - confidences[0] if len(confidences) > 0 else 0,
                'learning_rate': learning_rate,
                'learning_acceleration': learning_acceleration,
                'curve_fit_params': curve_fit_params,
                'convergence_quality': 'fast' if convergence_step and convergence_step < len(steps) * 0.5 else 'gradual'
            }
            
        except Exception as e:
            print(f"❌ 收敛分析失败: {e}")
            return {}
    
    def _analyze_learning_stability(self) -> Dict:
        """分析学习稳定性"""
        try:
            df = self.processed_data
            
            # 1. 置信度稳定性
            confidences = df['tracking_confidence'].values
            confidence_variance = np.var(confidences)
            confidence_cv = np.std(confidences) / np.mean(confidences) if np.mean(confidences) > 0 else float('inf')
            
            # 2. 特征权重稳定性
            weight_features = ['weight_geometric', 'weight_appearance', 'weight_shape', 'weight_spatial']
            weight_stability = {}
            
            for feature in weight_features:
                if feature in df.columns:
                    weights = df[feature].values
                    non_zero_weights = weights[weights > 0]
                    if len(non_zero_weights) > 1:
                        weight_stability[feature.replace('weight_', '')] = {
                            'variance': np.var(non_zero_weights),
                            'cv': np.std(non_zero_weights) / np.mean(non_zero_weights),
                            'stability_score': 1 / (1 + np.std(non_zero_weights))
                        }
            
            # 3. 性能退化检测
            if len(confidences) > 5:
                # 检查最近5步是否有显著下降
                recent_trend = np.polyfit(range(5), confidences[-5:], 1)[0]
                performance_degradation = recent_trend < -0.05
            else:
                performance_degradation = False
            
            # 4. 抗干扰性分析 (基于候选数量变化)
            candidates_counts = df['candidates_count'].values
            confidence_candidates_correlation = np.corrcoef(candidates_counts, confidences)[0, 1] if len(candidates_counts) > 1 else 0
            
            return {
                'confidence_variance': confidence_variance,
                'confidence_cv': confidence_cv,
                'weight_stability': weight_stability,
                'performance_degradation': performance_degradation,
                'robustness_score': abs(confidence_candidates_correlation),
                'stability_rating': self._calculate_stability_rating(confidence_cv, weight_stability)
            }
            
        except Exception as e:
            print(f"❌ 稳定性分析失败: {e}")
            return {}
    
    def _analyze_adaptation_effectiveness(self) -> Dict:
        """分析自适应机制有效性"""
        try:
            df = self.processed_data
            
            # 1. 自适应激活频率分析
            adaptive_steps = df[df['has_adaptive_weights'] == True]
            non_adaptive_steps = df[df['has_adaptive_weights'] == False]
            
            activation_rate = len(adaptive_steps) / len(df) if len(df) > 0 else 0
            
            # 2. 自适应前后性能比较
            if len(adaptive_steps) > 0 and len(non_adaptive_steps) > 0:
                adaptive_confidence = adaptive_steps['tracking_confidence'].mean()
                non_adaptive_confidence = non_adaptive_steps['tracking_confidence'].mean()
                adaptation_benefit = adaptive_confidence - non_adaptive_confidence
                
                # 统计显著性检验
                try:
                    _, p_value = stats.ttest_ind(adaptive_steps['tracking_confidence'], 
                                               non_adaptive_steps['tracking_confidence'])
                except:
                    p_value = 1.0
            else:
                adaptation_benefit = 0
                p_value = 1.0
            
            # 3. 权重变化模式分析
            weight_changes = []
            if len(df) > 1:
                for i in range(1, len(df)):
                    if df.iloc[i]['has_adaptive_weights'] and df.iloc[i-1]['has_adaptive_weights']:
                        curr_weights = [df.iloc[i][f'weight_{f}'] for f in ['geometric', 'appearance', 'shape', 'spatial']]
                        prev_weights = [df.iloc[i-1][f'weight_{f}'] for f in ['geometric', 'appearance', 'shape', 'spatial']]
                        
                        if any(w > 0 for w in curr_weights) and any(w > 0 for w in prev_weights):
                            change = np.linalg.norm(np.array(curr_weights) - np.array(prev_weights))
                            weight_changes.append(change)
            
            avg_weight_change = np.mean(weight_changes) if weight_changes else 0
            
            # 4. 触发条件分析
            trigger_conditions = {}
            if len(adaptive_steps) > 0:
                trigger_conditions = {
                    'low_confidence_triggers': len(adaptive_steps[adaptive_steps['tracking_confidence'] < 0.7]),
                    'multi_candidate_triggers': len(adaptive_steps[adaptive_steps['candidates_count'] > 1]),
                    'learning_progress_correlation': np.corrcoef(adaptive_steps['learning_progress'], 
                                                               adaptive_steps['tracking_confidence'])[0, 1] if len(adaptive_steps) > 1 else 0
                }
            
            return {
                'activation_rate': activation_rate,
                'adaptation_benefit': adaptation_benefit,
                'statistical_significance': p_value < 0.05,
                'p_value': p_value,
                'avg_weight_change': avg_weight_change,
                'trigger_conditions': trigger_conditions,
                'effectiveness_score': max(0, adaptation_benefit * (2 - p_value))  # 综合效果评分
            }
            
        except Exception as e:
            print(f"❌ 自适应效果分析失败: {e}")
            return {}
    
    def _analyze_feature_learning(self) -> Dict:
        """分析特征学习动态"""
        try:
            df = self.processed_data
            
            # 1. 特征权重演化分析
            feature_evolution = {}
            feature_names = ['geometric', 'appearance', 'shape', 'spatial']
            
            for feature in feature_names:
                weight_col = f'weight_{feature}'
                if weight_col in df.columns:
                    weights = df[weight_col].values
                    non_zero_weights = weights[weights > 0]
                    
                    if len(non_zero_weights) > 1:
                        # 计算趋势
                        steps_with_weights = df[df[weight_col] > 0]['step'].values
                        if len(steps_with_weights) > 1:
                            trend_slope = np.polyfit(steps_with_weights, non_zero_weights, 1)[0]
                        else:
                            trend_slope = 0
                        
                        feature_evolution[feature] = {
                            'usage_frequency': len(non_zero_weights) / len(df),
                            'average_weight': np.mean(non_zero_weights),
                            'weight_trend': trend_slope,
                            'weight_range': np.max(non_zero_weights) - np.min(non_zero_weights),
                            'importance_ranking': 0  # Will be filled later
                        }
            
            # 2. 特征重要性排序
            sorted_features = sorted(feature_evolution.items(), 
                                   key=lambda x: x[1]['average_weight'], reverse=True)
            
            for i, (feature, data) in enumerate(sorted_features):
                feature_evolution[feature]['importance_ranking'] = i + 1
            
            # 3. 特征协同效应分析
            feature_correlations = {}
            weight_columns = [f'weight_{f}' for f in feature_names if f'weight_{f}' in df.columns]
            
            if len(weight_columns) > 1:
                weight_data = df[weight_columns].values
                # 只考虑有权重的步骤
                non_zero_rows = np.any(weight_data > 0, axis=1)
                if np.sum(non_zero_rows) > 1:
                    corr_matrix = np.corrcoef(weight_data[non_zero_rows].T)
                    
                    for i, f1 in enumerate(feature_names[:len(weight_columns)]):
                        for j, f2 in enumerate(feature_names[:len(weight_columns)]):
                            if i < j:
                                correlation = corr_matrix[i, j] if not np.isnan(corr_matrix[i, j]) else 0
                                feature_correlations[f'{f1}_{f2}'] = correlation
            
            # 4. 学习效率分析
            learning_efficiency = {}
            if len(df) > 1:
                confidence_improvement = df['tracking_confidence'].iloc[-1] - df['tracking_confidence'].iloc[0]
                steps_taken = len(df)
                learning_efficiency = {
                    'confidence_per_step': confidence_improvement / steps_taken,
                    'steps_to_convergence': self._estimate_convergence_steps(df),
                    'learning_curve_smoothness': self._calculate_learning_smoothness(df['tracking_confidence'].values)
                }
            
            return {
                'feature_evolution': feature_evolution,
                'feature_correlations': feature_correlations,
                'learning_efficiency': learning_efficiency,
                'dominant_feature': sorted_features[0][0] if sorted_features else None
            }
            
        except Exception as e:
            print(f"❌ 特征学习分析失败: {e}")
            return {}
    
    def _analyze_user_interaction_impact(self) -> Dict:
        """分析用户交互对学习的影响"""
        try:
            df = self.processed_data
            
            # 1. 用户反馈类型分析
            feedback_types = df['user_feedback'].value_counts().to_dict()
            total_feedbacks = len(df[df['user_feedback'] != 'no_feedback'])
            
            # 2. 自动化进展分析
            auto_success_progression = []
            manual_success_progression = []
            
            for i, row in df.iterrows():
                if row['user_feedback'] == 'auto_success':
                    auto_success_progression.append(i)
                elif row['user_feedback'] == 'success':
                    manual_success_progression.append(i)
            
            # 计算自动化转折点
            automation_transition_point = None
            if len(auto_success_progression) > 0:
                automation_transition_point = min(auto_success_progression)
            
            # 3. 用户反馈对性能的即时影响
            feedback_impact = {}
            for feedback_type in ['success', 'failure']:
                feedback_steps = df[df['user_feedback'] == feedback_type]
                if len(feedback_steps) > 0:
                    # 分析反馈后的性能变化
                    post_feedback_improvements = []
                    for step_idx in feedback_steps.index:
                        if step_idx < len(df) - 1:
                            current_conf = df.iloc[step_idx]['tracking_confidence']
                            next_conf = df.iloc[step_idx + 1]['tracking_confidence']
                            improvement = next_conf - current_conf
                            post_feedback_improvements.append(improvement)
                    
                    feedback_impact[feedback_type] = {
                        'count': len(feedback_steps),
                        'avg_post_improvement': np.mean(post_feedback_improvements) if post_feedback_improvements else 0,
                        'avg_confidence': feedback_steps['tracking_confidence'].mean()
                    }
            
            # 4. 学习曲线的用户依赖性分析
            manual_phases = df[df['user_feedback'].isin(['success', 'failure'])]
            auto_phases = df[df['user_feedback'] == 'auto_success']
            
            user_dependency = {
                'manual_phase_performance': manual_phases['tracking_confidence'].mean() if len(manual_phases) > 0 else 0,
                'auto_phase_performance': auto_phases['tracking_confidence'].mean() if len(auto_phases) > 0 else 0,
                'independence_ratio': len(auto_phases) / len(df) if len(df) > 0 else 0
            }
            
            return {
                'feedback_distribution': feedback_types,
                'automation_transition_point': automation_transition_point,
                'feedback_impact': feedback_impact,
                'user_dependency': user_dependency,
                'total_user_interactions': total_feedbacks,
                'automation_success': user_dependency['independence_ratio'] > 0.5
            }
            
        except Exception as e:
            print(f"❌ 用户交互影响分析失败: {e}")
            return {}
    
    def _analyze_generalization(self) -> Dict:
        """分析泛化能力"""
        try:
            df = self.processed_data
            
            # 1. 历史特征利用效果分析
            historical_effectiveness = []
            
            for _, row in df.iterrows():
                if row['history_count'] > 0:
                    historical_benefit = row['historical_score'] - row['reference_score']
                    historical_effectiveness.append(historical_benefit)
            
            # 2. 跨步骤的特征一致性
            feature_consistency = {}
            if len(df) > 1:
                for feature in ['geometric', 'appearance', 'shape', 'spatial']:
                    weight_col = f'weight_{feature}'
                    if weight_col in df.columns:
                        weights = df[weight_col].values
                        # 计算权重的一致性（低方差表示高一致性）
                        consistency = 1 / (1 + np.var(weights[weights > 0])) if len(weights[weights > 0]) > 1 else 1
                        feature_consistency[feature] = consistency
            
            # 3. 适应新情况的能力
            adaptability_score = 0
            if len(df) > 5:
                # 分析后半段的性能是否保持或提升
                first_half = df.iloc[:len(df)//2]
                second_half = df.iloc[len(df)//2:]
                
                first_half_performance = first_half['tracking_confidence'].mean()
                second_half_performance = second_half['tracking_confidence'].mean()
                
                adaptability_score = max(0, second_half_performance - first_half_performance + 0.1)
            
            return {
                'historical_utilization_benefit': np.mean(historical_effectiveness) if historical_effectiveness else 0,
                'feature_consistency_scores': feature_consistency,
                'adaptability_score': adaptability_score,
                'generalization_quality': 'good' if adaptability_score > 0.05 else 'moderate'
            }
            
        except Exception as e:
            print(f"❌ 泛化分析失败: {e}")
            return {}
    
    def _calculate_stability_rating(self, confidence_cv: float, weight_stability: Dict) -> str:
        """计算稳定性评级"""
        try:
            # 置信度稳定性评分
            if confidence_cv < 0.1:
                confidence_score = 3
            elif confidence_cv < 0.2:
                confidence_score = 2
            else:
                confidence_score = 1
            
            # 权重稳定性评分
            if weight_stability:
                avg_stability = np.mean([data['stability_score'] for data in weight_stability.values()])
                if avg_stability > 0.8:
                    weight_score = 3
                elif avg_stability > 0.6:
                    weight_score = 2
                else:
                    weight_score = 1
            else:
                weight_score = 2  # 默认中等
            
            total_score = confidence_score + weight_score
            
            if total_score >= 5:
                return 'high'
            elif total_score >= 4:
                return 'moderate'
            else:
                return 'low'
                
        except:
            return 'unknown'
    
    def _estimate_convergence_steps(self, df: pd.DataFrame) -> int:
        """估算收敛所需步数"""
        try:
            confidences = df['tracking_confidence'].values
            if len(confidences) < 3:
                return len(confidences)
            
            # 寻找方差稳定的点
            window_size = max(3, len(confidences) // 4)
            for i in range(window_size, len(confidences)):
                window_var = np.var(confidences[i-window_size:i])
                if window_var < 0.01:
                    return i
            
            return len(confidences)
            
        except:
            return len(df) if len(df) > 0 else 1
    
    def _calculate_learning_smoothness(self, confidences: np.ndarray) -> float:
        """计算学习曲线平滑度"""
        try:
            if len(confidences) < 3:
                return 1.0
            
            # 计算二阶导数的方差（平滑度指标）
            second_derivative = np.diff(confidences, n=2)
            smoothness = 1 / (1 + np.var(second_derivative))
            
            return smoothness
            
        except:
            return 0.5
    
    def _generate_paper_visualizations(self, analyses: Dict, save_dir: str):
        """生成论文级可视化图表"""
        try:
            # 创建保存目录
            viz_dir = os.path.join(save_dir, 'paper_visualizations')
            os.makedirs(viz_dir, exist_ok=True)
            
            # 设置论文级绘图风格
            plt.rcParams.update({
                'font.size': 12,
                'axes.labelsize': 12,
                'axes.titlesize': 14,
                'legend.fontsize': 10,
                'xtick.labelsize': 10,
                'ytick.labelsize': 10,
                'figure.titlesize': 16
            })
            
            # 1. 学习收敛性分析图
            self._plot_convergence_analysis(analyses['convergence_analysis'], viz_dir)
            
            # 2. 自适应效果对比图
            self._plot_adaptation_effectiveness(analyses['adaptation_effectiveness'], viz_dir)
            
            # 3. 用户交互影响图
            self._plot_user_interaction_impact(analyses['user_interaction_impact'], viz_dir)
            
            # 4. 特征学习动态图
            self._plot_feature_learning_dynamics(analyses['feature_learning_dynamics'], viz_dir)
            
            print(f"✅ 论文可视化图表已保存到: {viz_dir}")
            
        except Exception as e:
            print(f"❌ 生成论文可视化失败: {e}")
    
    def _plot_convergence_analysis(self, analysis: Dict, save_dir: str):
        """Plot convergence analysis charts"""
        try:
            df = self.processed_data
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Left plot: Learning curve fitting
            steps = df['step'].values
            confidences = df['tracking_confidence'].values
            
            ax1.plot(steps, confidences, 'o-', color='blue', alpha=0.7, label='Actual Confidence')
            
            # Draw fitted curve if fitting parameters exist
            if analysis.get('curve_fit_params') is not None:
                params = analysis['curve_fit_params']
                
                def exponential_learning(x, a, b, c):
                    return a * (1 - np.exp(-b * x)) + c
                
                fitted_curve = exponential_learning(steps, *params)
                ax1.plot(steps, fitted_curve, '--', color='red', alpha=0.8, label='Exponential Fit Curve')
            
            # Mark convergence point
            if analysis.get('convergence_step'):
                ax1.axvline(analysis['convergence_step'], color='green', linestyle=':', 
                           label=f'Convergence Point: Step {analysis["convergence_step"]}')
            
            ax1.set_xlabel('Tracking Steps')
            ax1.set_ylabel('Confidence')
            ax1.set_title('Learning Convergence Analysis')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Right plot: Convergence quality metrics
            metrics = {
                'Total Improvement': analysis.get('total_improvement', 0),
                'Learning Rate': analysis.get('learning_rate', 0),
                'Acceleration': analysis.get('learning_acceleration', 0)
            }
            
            bars = ax2.bar(metrics.keys(), metrics.values(), 
                          color=['skyblue', 'lightgreen', 'salmon'])
            ax2.set_ylabel('Values')
            ax2.set_title('Learning Quality Metrics')
            ax2.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bar, value in zip(bars, metrics.values()):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(metrics.values())*0.01,
                        f'{value:.4f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'convergence_analysis.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"❌ Failed to plot convergence analysis: {e}")
    
    def _plot_adaptation_effectiveness(self, analysis: Dict, save_dir: str):
        """Plot adaptation effectiveness analysis charts"""
        try:
            df = self.processed_data
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # 1. Adaptive activation vs performance
            adaptive_steps = df[df['has_adaptive_weights'] == True]
            non_adaptive_steps = df[df['has_adaptive_weights'] == False]
            
            data_to_plot = []
            labels = []
            if len(adaptive_steps) > 0:
                data_to_plot.append(adaptive_steps['tracking_confidence'].values)
                labels.append(f'Adaptive Steps (n={len(adaptive_steps)})')
            if len(non_adaptive_steps) > 0:
                data_to_plot.append(non_adaptive_steps['tracking_confidence'].values)
                labels.append(f'Non-Adaptive Steps (n={len(non_adaptive_steps)})')
            
            if data_to_plot:
                bp = ax1.boxplot(data_to_plot, labels=labels, patch_artist=True)
                bp['boxes'][0].set_facecolor('lightblue') if len(bp['boxes']) > 0 else None
                if len(bp['boxes']) > 1:
                    bp['boxes'][1].set_facecolor('lightcoral')
                ax1.set_ylabel('Tracking Confidence')
                ax1.set_title('Adaptive vs Non-Adaptive Performance Comparison')
                ax1.grid(True, alpha=0.3, axis='y')
            
            # 2. Weight change trends
            weight_features = ['weight_geometric', 'weight_appearance', 'weight_shape', 'weight_spatial']
            colors = ['blue', 'green', 'red', 'orange']
            
            for i, feature in enumerate(weight_features):
                if feature in df.columns:
                    weights = df[feature].values
                    steps = df['step'].values
                    
                    # Only plot non-zero weights
                    non_zero_mask = weights > 0
                    if np.any(non_zero_mask):
                        ax2.plot(steps[non_zero_mask], weights[non_zero_mask], 
                               'o-', color=colors[i], alpha=0.7, 
                               label=feature.replace('weight_', '').title())
            
            ax2.set_xlabel('Tracking Steps')
            ax2.set_ylabel('Feature Weights')
            ax2.set_title('Adaptive Weight Evolution')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # 3. Trigger condition analysis
            trigger_data = analysis.get('trigger_conditions', {})
            if trigger_data:
                trigger_names = list(trigger_data.keys())[:3]  # Only show first 3
                trigger_values = [trigger_data[name] for name in trigger_names]
                
                bars = ax3.bar(range(len(trigger_names)), trigger_values, 
                              color=['skyblue', 'lightgreen', 'salmon'])
                ax3.set_xticks(range(len(trigger_names)))
                ax3.set_xticklabels([name.replace('_', ' ').title() for name in trigger_names], 
                                  rotation=45, ha='right')
                ax3.set_ylabel('Trigger Count')
                ax3.set_title('Adaptive Trigger Condition Analysis')
                ax3.grid(True, alpha=0.3, axis='y')
            
            # 4. Effectiveness statistics
            effectiveness_metrics = {
                'Activation Rate': analysis.get('activation_rate', 0) * 100,
                'Performance Improvement': analysis.get('adaptation_benefit', 0) * 100,
                'Effectiveness Score': analysis.get('effectiveness_score', 0) * 100
            }
            
            bars = ax4.bar(effectiveness_metrics.keys(), effectiveness_metrics.values(),
                          color=['lightblue', 'lightgreen', 'lightyellow'])
            ax4.set_ylabel('Percentage/Score')
            ax4.set_title('Adaptation Effectiveness Statistics')
            ax4.grid(True, alpha=0.3, axis='y')
            
            # Add significance annotation
            if analysis.get('statistical_significance'):
                ax4.text(0.5, max(effectiveness_metrics.values()) * 0.8, 
                        f'Statistical Significance: p < 0.05', ha='center', va='center',
                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'adaptation_effectiveness.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"❌ Failed to plot adaptation effectiveness analysis: {e}")
    
    def _plot_user_interaction_impact(self, analysis: Dict, save_dir: str):
        """Plot user interaction impact analysis charts"""
        try:
            df = self.processed_data
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # 1. Feedback type distribution
            feedback_dist = analysis.get('feedback_distribution', {})
            if feedback_dist:
                labels = [label.replace('_', ' ').title() for label in feedback_dist.keys()]
                sizes = list(feedback_dist.values())
                colors = ['lightgreen', 'lightcoral', 'darkgreen', 'lightgray']
                
                wedges, texts, autotexts = ax1.pie(sizes, labels=labels, colors=colors[:len(sizes)], 
                                                  autopct='%1.1f%%', startangle=90)
                ax1.set_title('User Feedback Type Distribution')
            
            # 2. Automation progress timeline
            manual_steps = df[df['user_feedback'] == 'success']['step'].values
            auto_steps = df[df['user_feedback'] == 'auto_success']['step'].values
            
            if len(manual_steps) > 0:
                ax2.scatter(manual_steps, [1] * len(manual_steps), 
                           color='orange', s=60, alpha=0.7, label='Manual Confirmation')
            if len(auto_steps) > 0:
                ax2.scatter(auto_steps, [2] * len(auto_steps), 
                           color='green', s=60, alpha=0.7, label='Auto Success')
            
            ax2.set_xlabel('Tracking Steps')
            ax2.set_yticks([1, 2])
            ax2.set_yticklabels(['Manual Confirmation', 'Auto Success'])
            ax2.set_title('Automation Progress Timeline')
            ax2.legend()
            ax2.grid(True, alpha=0.3, axis='x')
            
            # Mark transition point
            transition_point = analysis.get('automation_transition_point')
            if transition_point:
                ax2.axvline(transition_point, color='red', linestyle='--', 
                           label=f'Automation Start: Step {transition_point}')
                ax2.legend()
            
            # 3. Feedback impact on performance
            feedback_impact = analysis.get('feedback_impact', {})
            if feedback_impact:
                feedback_types = list(feedback_impact.keys())
                improvements = [feedback_impact[ft]['avg_post_improvement'] for ft in feedback_types]
                
                bars = ax3.bar(feedback_types, improvements, 
                              color=['lightgreen' if imp > 0 else 'lightcoral' for imp in improvements])
                ax3.set_ylabel('Subsequent Performance Improvement')
                ax3.set_title('Feedback Impact on Subsequent Performance')
                ax3.grid(True, alpha=0.3, axis='y')
                
                # Add baseline
                ax3.axhline(0, color='black', linestyle='-', alpha=0.5)
            
            # 4. User dependency analysis
            dependency_data = analysis.get('user_dependency', {})
            if dependency_data:
                phases = ['Manual Phase', 'Auto Phase']
                performances = [
                    dependency_data.get('manual_phase_performance', 0),
                    dependency_data.get('auto_phase_performance', 0)
                ]
                
                bars = ax4.bar(phases, performances, color=['orange', 'green'], alpha=0.7)
                ax4.set_ylabel('Average Confidence')
                ax4.set_title('Manual vs Auto Phase Performance')
                ax4.grid(True, alpha=0.3, axis='y')
                
                # Add independence ratio annotation
                independence_ratio = dependency_data.get('independence_ratio', 0)
                ax4.text(0.5, max(performances) * 0.8, 
                        f'Independence Ratio: {independence_ratio:.1%}',
                        ha='center', va='center', transform=ax4.transData,
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'user_interaction_impact.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"❌ Failed to plot user interaction impact: {e}")
    
    def _plot_feature_learning_dynamics(self, analysis: Dict, save_dir: str):
        """Plot feature learning dynamics analysis charts"""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # 1. Feature importance evolution
            feature_evolution = analysis.get('feature_evolution', {})
            if feature_evolution:
                features = list(feature_evolution.keys())
                avg_weights = [feature_evolution[f]['average_weight'] for f in features]
                usage_freq = [feature_evolution[f]['usage_frequency'] for f in features]
                
                ax1.scatter(usage_freq, avg_weights, s=100, alpha=0.7)
                
                for i, feature in enumerate(features):
                    ax1.annotate(feature.title(), (usage_freq[i], avg_weights[i]),
                               xytext=(5, 5), textcoords='offset points')
                
                ax1.set_xlabel('Usage Frequency')
                ax1.set_ylabel('Average Weight')
                ax1.set_title('Feature Importance Distribution')
                ax1.grid(True, alpha=0.3)
            
            # 2. Feature trend analysis
            df = self.processed_data
            weight_features = ['weight_geometric', 'weight_appearance', 'weight_shape', 'weight_spatial']
            colors = ['blue', 'green', 'red', 'orange']
            
            for i, feature in enumerate(weight_features):
                if feature in df.columns and feature_evolution:
                    feature_name = feature.replace('weight_', '')
                    if feature_name in feature_evolution:
                        trend = feature_evolution[feature_name]['weight_trend']
                        
                        # Plot trend line
                        steps = df['step'].values
                        weights = df[feature].values
                        
                        # Fit trend line
                        non_zero_mask = weights > 0
                        if np.any(non_zero_mask) and len(steps[non_zero_mask]) > 1:
                            z = np.polyfit(steps[non_zero_mask], weights[non_zero_mask], 1)
                            p = np.poly1d(z)
                            ax2.plot(steps, p(steps), '--', color=colors[i], alpha=0.7,
                                   label=f'{feature_name.title()} (Trend: {trend:.4f})')
            
            ax2.set_xlabel('Tracking Steps')
            ax2.set_ylabel('Weight Trend')
            ax2.set_title('Feature Weight Trend Analysis')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # 3. Feature synergy effects
            feature_correlations = analysis.get('feature_correlations', {})
            if feature_correlations:
                # Create correlation matrix heatmap
                feature_pairs = list(feature_correlations.keys())
                correlation_values = list(feature_correlations.values())
                
                # Build correlation matrix
                features = list(set([pair.split('_')[0] for pair in feature_pairs] + 
                                  [pair.split('_')[1] for pair in feature_pairs]))
                n_features = len(features)
                
                corr_matrix = np.zeros((n_features, n_features))
                np.fill_diagonal(corr_matrix, 1.0)
                
                for pair, corr in feature_correlations.items():
                    f1, f2 = pair.split('_')
                    if f1 in features and f2 in features:
                        i1, i2 = features.index(f1), features.index(f2)
                        corr_matrix[i1, i2] = corr
                        corr_matrix[i2, i1] = corr
                
                im = ax3.imshow(corr_matrix, cmap='RdYlBu', vmin=-1, vmax=1)
                ax3.set_xticks(range(n_features))
                ax3.set_yticks(range(n_features))
                ax3.set_xticklabels([f.title() for f in features], rotation=45)
                ax3.set_yticklabels([f.title() for f in features])
                ax3.set_title('Feature Synergy Effect Matrix')
                
                # Add value annotations
                for i in range(n_features):
                    for j in range(n_features):
                        text = ax3.text(j, i, f'{corr_matrix[i, j]:.2f}',
                                      ha="center", va="center", color="black")
                
                plt.colorbar(im, ax=ax3)
            
            # 4. Learning efficiency metrics
            learning_efficiency = analysis.get('learning_efficiency', {})
            if learning_efficiency:
                metrics = {
                    'Confidence per Step': learning_efficiency.get('confidence_per_step', 0),
                    'Steps to Convergence': learning_efficiency.get('steps_to_convergence', 0) / 20,  # Normalized
                    'Learning Smoothness': learning_efficiency.get('learning_curve_smoothness', 0)
                }
                
                # Radar chart
                angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)
                values = list(metrics.values())
                
                ax4 = plt.subplot(2, 2, 4, projection='polar')
                ax4.plot(angles, values, 'o-', linewidth=2, color='blue', alpha=0.7)
                ax4.fill(angles, values, alpha=0.25, color='blue')
                ax4.set_xticks(angles)
                ax4.set_xticklabels(metrics.keys())
                ax4.set_title('Learning Efficiency Radar Chart')
                ax4.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'feature_learning_dynamics.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"❌ Failed to plot feature learning dynamics: {e}")
    
    def _generate_quantitative_report(self, analyses: Dict) -> Dict:
        """生成定量分析报告"""
        try:
            df = self.processed_data
            
            report = {
                'executive_summary': {},
                'detailed_metrics': {},
                'statistical_tests': {},
                'recommendations': []
            }
            
            # 1. 执行摘要
            total_steps = len(df)
            successful_steps = len(df[df['is_successful'] == True])
            success_rate = successful_steps / total_steps if total_steps > 0 else 0
            
            convergence_analysis = analyses.get('convergence_analysis', {})
            adaptation_analysis = analyses.get('adaptation_effectiveness', {})
            user_analysis = analyses.get('user_interaction_impact', {})
            
            report['executive_summary'] = {
                'total_tracking_steps': total_steps,
                'overall_success_rate': success_rate,
                'final_confidence': convergence_analysis.get('final_confidence', 0),
                'learning_improvement': convergence_analysis.get('total_improvement', 0),
                'automation_rate': user_analysis.get('user_dependency', {}).get('independence_ratio', 0),
                'adaptation_effectiveness': adaptation_analysis.get('effectiveness_score', 0),
                'convergence_speed': convergence_analysis.get('convergence_quality', 'unknown')
            }
            
            # 2. 详细指标
            report['detailed_metrics'] = {
                'convergence': {
                    'convergence_step': convergence_analysis.get('convergence_step'),
                    'learning_rate': convergence_analysis.get('learning_rate', 0),
                    'learning_acceleration': convergence_analysis.get('learning_acceleration', 0)
                },
                'stability': analyses.get('stability_analysis', {}),
                'adaptation': {
                    'activation_rate': adaptation_analysis.get('activation_rate', 0),
                    'performance_benefit': adaptation_analysis.get('adaptation_benefit', 0),
                    'trigger_effectiveness': adaptation_analysis.get('effectiveness_score', 0)
                },
                'generalization': analyses.get('generalization_analysis', {})
            }
            
            # 3. 统计检验结果
            report['statistical_tests'] = {
                'adaptation_significance': {
                    'p_value': adaptation_analysis.get('p_value', 1.0),
                    'is_significant': adaptation_analysis.get('statistical_significance', False),
                    'effect_size': adaptation_analysis.get('adaptation_benefit', 0)
                }
            }
            
            # 4. 建议和结论
            recommendations = []
            
            if success_rate > 0.8:
                recommendations.append("系统表现优异，成功率超过80%")
            else:
                recommendations.append("建议进一步优化追踪算法以提高成功率")
            
            if adaptation_analysis.get('statistical_significance'):
                recommendations.append("自适应机制显著提升了性能，建议保持当前配置")
            else:
                recommendations.append("自适应机制效果不明显，建议调整触发条件")
            
            automation_rate = user_analysis.get('user_dependency', {}).get('independence_ratio', 0)
            if automation_rate > 0.5:
                recommendations.append("自动化程度良好，减少了用户干预需求")
            else:
                recommendations.append("建议提升自动判断的置信度阈值以增加自动化率")
            
            convergence_step = convergence_analysis.get('convergence_step')
            if convergence_step and convergence_step < total_steps * 0.5:
                recommendations.append("学习收敛速度快，系统快速适应能力强")
            else:
                recommendations.append("学习收敛较慢，建议优化学习算法参数")
            
            report['recommendations'] = recommendations
            
            return report
            
        except Exception as e:
            print(f"❌ 生成定量报告失败: {e}")
            return {}
    
    def _save_analysis_results(self, analyses: Dict, report: Dict, save_dir: str):
        """保存分析结果"""
        try:
            # 创建结果目录
            results_dir = os.path.join(save_dir, 'analysis_results')
            os.makedirs(results_dir, exist_ok=True)
            
            # 1. 保存完整分析结果
            with open(os.path.join(results_dir, 'complete_analysis.json'), 'w', encoding='utf-8') as f:
                json.dump(analyses, f, indent=2, ensure_ascii=False, default=str)
            
            # 2. 保存定量报告
            with open(os.path.join(results_dir, 'quantitative_report.json'), 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)
            
            # 3. 保存处理后的数据
            if self.processed_data is not None:
                self.processed_data.to_csv(os.path.join(results_dir, 'processed_learning_data.csv'), index=False)
            
            # 4. 生成简化的论文表格数据
            self._generate_paper_table_data(report, results_dir)
            
            print(f"✅ 分析结果已保存到: {results_dir}")
            
        except Exception as e:
            print(f"❌ 保存分析结果失败: {e}")
    
    def _generate_paper_table_data(self, report: Dict, save_dir: str):
        """生成论文表格数据"""
        try:
            # 关键性能指标表
            performance_table = {
                'Metric': [
                    'Overall Success Rate',
                    'Final Confidence Score',
                    'Learning Improvement',
                    'Automation Rate', 
                    'Adaptation Effectiveness',
                    'Convergence Speed',
                    'System Stability'
                ],
                'Value': [
                    f"{report['executive_summary'].get('overall_success_rate', 0):.1%}",
                    f"{report['executive_summary'].get('final_confidence', 0):.3f}",
                    f"{report['executive_summary'].get('learning_improvement', 0):.3f}",
                    f"{report['executive_summary'].get('automation_rate', 0):.1%}",
                    f"{report['executive_summary'].get('adaptation_effectiveness', 0):.3f}",
                    report['executive_summary'].get('convergence_speed', 'Unknown'),
                    report['detailed_metrics'].get('stability', {}).get('stability_rating', 'Unknown')
                ]
            }
            
            df_performance = pd.DataFrame(performance_table)
            df_performance.to_csv(os.path.join(save_dir, 'performance_metrics_table.csv'), index=False)
            
            # 统计显著性表
            if report.get('statistical_tests'):
                significance_table = {
                    'Test': ['Adaptation Effectiveness'],
                    'P-Value': [report['statistical_tests']['adaptation_significance'].get('p_value', 1.0)],
                    'Significant': [report['statistical_tests']['adaptation_significance'].get('is_significant', False)],
                    'Effect Size': [report['statistical_tests']['adaptation_significance'].get('effect_size', 0)]
                }
                
                df_significance = pd.DataFrame(significance_table)
                df_significance.to_csv(os.path.join(save_dir, 'statistical_tests_table.csv'), index=False)
            
            print("✅ 论文表格数据已生成")
            
        except Exception as e:
            print(f"❌ 生成论文表格数据失败: {e}")


# 使用示例和集成函数
def run_complete_learning_analysis(scan_directory: str):
    """运行完整的学习分析流程"""
    try:
        print("🚀 开始完整学习性能分析...")
        
        # 1. 基础学习曲线可视化
        from learning_curve_visualizer import LearningCurveVisualizer
        visualizer = LearningCurveVisualizer(scan_directory)
        
        # 生成基础学习曲线
        basic_curves = visualizer.generate_comprehensive_learning_curves()
        
        # 导出基础数据
        csv_data = visualizer.export_learning_data_for_analysis()
        
        # 2. 高级性能分析
        learning_data = visualizer._load_all_learning_data()
        
        if learning_data:
            analyzer = AdvancedLearningAnalyzer(learning_data)
            analyses, report = analyzer.comprehensive_analysis(visualizer.data_dir)
            
            if analyses and report:
                print("\n📊 学习性能分析完成!")
                print("=" * 60)
                
                # 输出关键结果
                exec_summary = report.get('executive_summary', {})
                print(f"📈 总体成功率: {exec_summary.get('overall_success_rate', 0):.1%}")
                print(f"🎯 最终置信度: {exec_summary.get('final_confidence', 0):.3f}")
                print(f"📚 学习改善度: {exec_summary.get('learning_improvement', 0):.3f}")
                print(f"🤖 自动化程度: {exec_summary.get('automation_rate', 0):.1%}")
                print(f"⚡ 适应有效性: {exec_summary.get('adaptation_effectiveness', 0):.3f}")
                print(f"🏃 收敛速度: {exec_summary.get('convergence_speed', 'Unknown')}")
                
                print(f"\n📁 分析结果保存路径:")
                print(f"  - 基础学习曲线: {basic_curves}")
                print(f"  - CSV数据: {csv_data}")
                print(f"  - 论文可视化: {os.path.join(visualizer.data_dir, 'paper_visualizations')}")
                print(f"  - 分析结果: {os.path.join(visualizer.data_dir, 'analysis_results')}")
                
                return True
            else:
                print("❌ 高级分析失败")
                return False
        else:
            print("❌ 无法加载学习数据")
            return False
            
    except Exception as e:
        print(f"❌ 完整分析失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) != 2:
        print("用法: python advanced_learning_analyzer.py <scan_directory>")
    
    scan_directory = sys.argv[1]
    success = run_complete_learning_analysis(scan_directory)
    
    if success:
        print("\n✅ 学习曲线分析完成! 可用于论文撰写 📝")
    else:
        print("\n❌ 分析失败，请检查数据完整性")