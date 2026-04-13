#!/usr/bin/env python3
# visualization_generator.py - 生成性能分析图表

import json
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List
import pandas as pd
from datetime import datetime

class VisualizationGenerator:
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.charts_dir = self.results_dir / 'charts'
        self.charts_dir.mkdir(exist_ok=True)
        
        # 设置matplotlib样式
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 加载数据 - 添加调试信息
        print(f"Loading data from: {self.results_dir}")
        self.test_results = self._load_test_results()
        self.performance_data = self._load_performance_data()
        self.user_annotations = self._load_user_annotations()
        
        print(f"Loaded {len(self.test_results)} test results")
        print(f"Loaded {len(self.performance_data)} performance records")
        print(f"Loaded {len(self.user_annotations)} user annotations")
        
        print(f"Visualization Generator initialized")
        print(f"Charts will be saved to: {self.charts_dir}")
    
    def _load_test_results(self) -> List[Dict]:
        """加载测试结果"""
        results_file = self.results_dir / 'final_test_results.json'
        print(f"Looking for test results at: {results_file}")
        if results_file.exists():
            try:
                with open(results_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    print(f"Successfully loaded {len(data)} test results")
                    return data
            except Exception as e:
                print(f"Error loading test results: {e}")
                return []
        else:
            print(f"Test results file not found: {results_file}")
            return []
    
    def _load_performance_data(self) -> List[Dict]:
        """加载性能数据"""
        perf_file = self.results_dir / 'performance_data.json'
        print(f"Looking for performance data at: {perf_file}")
        if perf_file.exists():
            try:
                with open(perf_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    print(f"Successfully loaded {len(data)} performance records")
                    return data
            except Exception as e:
                print(f"Error loading performance data: {e}")
                return []
        else:
            print(f"Performance data file not found: {perf_file}")
            return []
    
    def _load_user_annotations(self) -> List[Dict]:
        """加载用户标注"""
        ann_file = self.results_dir / 'user_annotations.json'
        print(f"Looking for user annotations at: {ann_file}")
        if ann_file.exists():
            try:
                with open(ann_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    print(f"Successfully loaded {len(data)} user annotations")
                    return data
            except Exception as e:
                print(f"Error loading user annotations: {e}")
                return []
        else:
            print(f"User annotations file not found: {ann_file}")
            return []
    
    def generate_sample_data(self):
        """生成示例数据用于测试"""
        print("Generating sample data for testing...")
        
        # 生成示例测试结果
        self.test_results = [
            {
                'scenario_id': f'test_{i}',
                'success': True,
                'detection_count': np.random.randint(1, 10),
                'test_metadata': {
                    'complexity_level': np.random.randint(1, 6),
                    'lighting_condition': np.random.choice(['bright', 'normal', 'dim', 'dark'])
                },
                'detection_results': {
                    'processing_time': np.random.uniform(0.5, 3.0),
                    'objects': [
                        {
                            'class_name': np.random.choice(['person', 'car', 'bicycle', 'dog']),
                            'quality_score': np.random.uniform(60, 95)
                        }
                    ]
                },
                'test_duration': np.random.uniform(0.5, 3.0)
            }
            for i in range(20)
        ]
        
        # 生成示例性能数据
        self.performance_data = [
            {
                'detection_stats': {
                    'original_detections': np.random.randint(5, 20),
                    'filtered_detections': np.random.randint(3, 15)
                },
                'timing': {
                    'stage_breakdown': {
                        'yolo_detection': np.random.uniform(0.1, 0.5),
                        'sam_segmentation': np.random.uniform(0.2, 0.8),
                        'feature_extraction': np.random.uniform(0.05, 0.3),
                        '3d_post_processing': np.random.uniform(0.1, 0.4),
                        'visualization': np.random.uniform(0.05, 0.2)
                    }
                }
            }
            for i in range(15)
        ]
        
        # 生成示例用户标注
        self.user_annotations = [
            {
                'scenario_id': f'test_{i}',
                'annotation': {
                    'detection_accuracy': np.random.randint(2, 6),
                    'feature_quality_accuracy': np.random.randint(2, 6),
                    'interrupted': False,
                    'error': False
                }
            }
            for i in range(10)
        ]
        
        print(f"Generated {len(self.test_results)} test results")
        print(f"Generated {len(self.performance_data)} performance records")
        print(f"Generated {len(self.user_annotations)} user annotations")
    
    def generate_all_charts(self):
        """Generate all visualization charts"""
        print("Generating visualization charts...")
        
        # 如果没有真实数据，生成示例数据
        if not self.test_results and not self.performance_data and not self.user_annotations:
            print("No real data found, generating sample data for demonstration...")
            self.generate_sample_data()
        
        chart_functions = [
            ('detection_accuracy_by_complexity', self.plot_detection_accuracy_by_complexity),
            ('lighting_condition_impact', self.plot_lighting_condition_impact),
            ('3d_post_processing_effectiveness', self.plot_3d_post_processing_effectiveness),
            ('feature_quality_correlation', self.plot_feature_quality_correlation),
            ('processing_time_breakdown', self.plot_processing_time_breakdown),
            ('error_analysis', self.plot_error_analysis),
            ('detection_consistency', self.plot_detection_consistency),
            ('performance_overview', self.plot_performance_overview)
        ]
        
        generated_charts = []
        for chart_name, chart_func in chart_functions:
            print(f"\nAttempting to generate: {chart_name}")
            try:
                chart_path = chart_func()
                if chart_path:
                    generated_charts.append((chart_name, chart_path))
                    print(f"  ✅ Generated: {chart_name}")
                    print(f"  📁 Saved to: {chart_path}")
                else:
                    print(f"  ⚠️ Skipped: {chart_name} (insufficient data or placeholder function)")
            except Exception as e:
                print(f"  ❌ Error generating {chart_name}: {e}")
                import traceback
                traceback.print_exc()
        
        # Generate chart index file
        self._generate_chart_index(generated_charts)
        
        print(f"\nChart generation completed. {len(generated_charts)} charts generated.")
        print(f"Charts saved to: {self.charts_dir}")
        return generated_charts
    
    def plot_detection_accuracy_by_complexity(self) -> str:
        """绘制检测准确性随场景复杂度变化的折线图"""
        if not self.test_results:
            print("No test results available for complexity analysis")
            return None
        
        print(f"Processing {len(self.test_results)} test results for complexity analysis...")
        
        # 准备数据
        complexity_data = {}
        for result in self.test_results:
            if not result.get('success', False):
                continue
                
            metadata = result.get('test_metadata', {})
            complexity = metadata.get('complexity_level', 3)
            detection_count = result.get('detection_count', 0)
            
            if complexity not in complexity_data:
                complexity_data[complexity] = {
                    'detection_counts': [],
                    'success_rate': [],
                    'avg_quality': []
                }
            
            complexity_data[complexity]['detection_counts'].append(detection_count)
            complexity_data[complexity]['success_rate'].append(1 if detection_count > 0 else 0)
            
            # 如果有用户标注，添加质量数据
            scenario_id = result.get('scenario_id', '')
            user_ann = self._get_user_annotation_for_scenario(scenario_id)
            if user_ann:
                quality = user_ann.get('annotation', {}).get('detection_accuracy', 3)
                complexity_data[complexity]['avg_quality'].append(quality)
        
        if not complexity_data:
            print("No valid complexity data found")
            return None
        
        print(f"Found complexity data for levels: {sorted(complexity_data.keys())}")
        
        # 计算统计值
        complexities = sorted(complexity_data.keys())
        avg_detections = []
        success_rates = []
        avg_qualities = []
        
        for complexity in complexities:
            data = complexity_data[complexity]
            avg_detections.append(np.mean(data['detection_counts']))
            success_rates.append(np.mean(data['success_rate']))
            if data['avg_quality']:
                avg_qualities.append(np.mean(data['avg_quality']))
            else:
                avg_qualities.append(None)
        
        # 绘制图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 左图：平均检测数量和成功率
        ax1_twin = ax1.twinx()
        
        line1 = ax1.plot(complexities, avg_detections, 'o-', linewidth=2, markersize=8, 
                        label='Avg Objects Detected', color='#2E8B57')
        line2 = ax1_twin.plot(complexities, success_rates, 's-', linewidth=2, markersize=8, 
                             label='Success Rate', color='#FF6347')
        
        ax1.set_xlabel('Scene Complexity Level', fontsize=12)
        ax1.set_ylabel('Average Objects Detected', fontsize=12, color='#2E8B57')
        ax1_twin.set_ylabel('Success Rate', fontsize=12, color='#FF6347')
        ax1.set_title('Detection Performance vs Scene Complexity', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # 合并图例
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper right')
        
        # 右图：用户评分质量（如果有数据）
        if any(q is not None for q in avg_qualities):
            valid_complexities = [c for c, q in zip(complexities, avg_qualities) if q is not None]
            valid_qualities = [q for q in avg_qualities if q is not None]
            
            ax2.plot(valid_complexities, valid_qualities, 'o-', linewidth=2, markersize=8, 
                    color='#4169E1', label='User Rating')
            ax2.fill_between(valid_complexities, valid_qualities, alpha=0.3, color='#4169E1')
            ax2.set_xlabel('Scene Complexity Level', fontsize=12)
            ax2.set_ylabel('User Rating (1-5)', fontsize=12)
            ax2.set_title('User-Rated Quality vs Complexity', fontsize=14, fontweight='bold')
            ax2.set_ylim(1, 5)
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No User Rating Data Available', 
                    transform=ax2.transAxes, ha='center', va='center', fontsize=12)
            ax2.set_title('User Rating Data Not Available', fontsize=14)
        
        plt.tight_layout()
        chart_path = self.charts_dir / 'detection_accuracy_by_complexity.png'
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved complexity analysis chart to: {chart_path}")
        return str(chart_path)
    
    def plot_lighting_condition_impact(self) -> str:
        """绘制光照条件对检测性能的影响"""
        if not self.test_results:
            return None
        
        print(f"Processing lighting condition impact for {len(self.test_results)} results...")
        
        # 准备数据
        lighting_data = {}
        for result in self.test_results:
            if not result.get('success', False):
                continue
                
            metadata = result.get('test_metadata', {})
            lighting = metadata.get('lighting_condition', 'unknown')
            detection_count = result.get('detection_count', 0)
            
            if lighting not in lighting_data:
                lighting_data[lighting] = {
                    'f1_scores': [],
                    'detection_counts': [],
                    'processing_times': []
                }
            
            # 计算F1分数（简化版本，基于检测数量和用户评分）
            scenario_id = result.get('scenario_id', '')
            user_ann = self._get_user_annotation_for_scenario(scenario_id)
            if user_ann:
                accuracy = user_ann.get('annotation', {}).get('detection_accuracy', 3)
                f1_score = (accuracy / 5.0) * (1.0 if detection_count > 0 else 0)
            else:
                f1_score = 0.8 if detection_count > 0 else 0.0  # 默认假设
            
            lighting_data[lighting]['f1_scores'].append(f1_score)
            lighting_data[lighting]['detection_counts'].append(detection_count)
            
            detection_results = result.get('detection_results', {})
            processing_time = detection_results.get('processing_time', result.get('test_duration', 0))
            lighting_data[lighting]['processing_times'].append(processing_time)
        
        if not lighting_data:
            return None
        
        print(f"Found lighting conditions: {list(lighting_data.keys())}")
        
        # 计算统计值
        lighting_conditions = list(lighting_data.keys())
        avg_f1_scores = []
        std_f1_scores = []
        avg_processing_times = []
        
        for condition in lighting_conditions:
            data = lighting_data[condition]
            f1_scores = data['f1_scores']
            processing_times = data['processing_times']
            
            avg_f1_scores.append(np.mean(f1_scores))
            std_f1_scores.append(np.std(f1_scores))
            avg_processing_times.append(np.mean(processing_times))
        
        # 绘制图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 左图：F1分数对比（带误差棒）
        x_pos = np.arange(len(lighting_conditions))
        bars = ax1.bar(x_pos, avg_f1_scores, yerr=std_f1_scores, capsize=5, 
                      color=sns.color_palette("husl", len(lighting_conditions)), alpha=0.8)
        
        ax1.set_xlabel('Lighting Condition', fontsize=12)
        ax1.set_ylabel('Average F1 Score', fontsize=12)
        ax1.set_title('Detection F1 Score by Lighting Condition', fontsize=14, fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(lighting_conditions, rotation=45)
        ax1.set_ylim(0, 1.0)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar, avg_val in zip(bars, avg_f1_scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{avg_val:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # 右图：处理时间对比
        bars2 = ax2.bar(x_pos, avg_processing_times, 
                       color=sns.color_palette("husl", len(lighting_conditions)), alpha=0.8)
        
        ax2.set_xlabel('Lighting Condition', fontsize=12)
        ax2.set_ylabel('Average Processing Time (s)', fontsize=12)
        ax2.set_title('Processing Time by Lighting Condition', fontsize=14, fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(lighting_conditions, rotation=45)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar, time_val in zip(bars2, avg_processing_times):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    f'{time_val:.1f}s', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        chart_path = self.charts_dir / 'lighting_condition_impact.png'
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(chart_path)
    
    def plot_3d_post_processing_effectiveness(self) -> str:
        """绘制3D后处理效果分析"""
        if not self.performance_data:
            return None
        
        print(f"Processing 3D post-processing data for {len(self.performance_data)} records...")
        
        # 准备数据
        removal_rates = []
        final_accuracies = []
        object_densities = []
        
        for perf in self.performance_data:
            detection_stats = perf.get('detection_stats', {})
            original = detection_stats.get('original_detections', 0)
            filtered = detection_stats.get('filtered_detections', 0)
            
            if original > 0:
                removal_rate = (original - filtered) / original
                removal_rates.append(removal_rate)
                
                # 估算密度（原始检测数量作为密度指标）
                object_densities.append(original)
                
                # 估算最终精度（基于去重效果）
                final_accuracy = 0.9 - removal_rate * 0.2  # 简化模型
                final_accuracies.append(final_accuracy)
        
        if not removal_rates:
            return None
        
        # 绘制双Y轴图
        fig, ax1 = plt.subplots(figsize=(12, 8))
        ax2 = ax1.twinx()
        
        # 按密度排序
        sorted_data = sorted(zip(object_densities, removal_rates, final_accuracies))
        densities, removals, accuracies = zip(*sorted_data)
        
        # 左Y轴：去重率
        line1 = ax1.plot(densities, removals, 'o-', linewidth=2, markersize=6, 
                        color='#FF6347', label='Duplicate Removal Rate')
        ax1.fill_between(densities, removals, alpha=0.3, color='#FF6347')
        
        # 右Y轴：最终精度
        line2 = ax2.plot(densities, accuracies, 's-', linewidth=2, markersize=6, 
                        color='#4169E1', label='Final Detection Accuracy')
        
        ax1.set_xlabel('Object Density (Original Detections)', fontsize=12)
        ax1.set_ylabel('Duplicate Removal Rate', fontsize=12, color='#FF6347')
        ax2.set_ylabel('Final Detection Accuracy', fontsize=12, color='#4169E1')
        
        ax1.set_title('3D Post-Processing Effectiveness Analysis', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # 合并图例
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='center right')
        
        plt.tight_layout()
        chart_path = self.charts_dir / '3d_post_processing_effectiveness.png'
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(chart_path)
    
    def plot_feature_quality_correlation(self) -> str:
        """绘制特征质量评估准确性散点图"""
        if not self.user_annotations:
            return None
        
        print(f"Processing feature quality correlation with {len(self.user_annotations)} annotations...")
        
        # 准备数据
        system_scores = []
        user_scores = []
        object_classes = []
        
        for ann_data in self.user_annotations:
            scenario_id = ann_data.get('scenario_id', '')
            annotation = ann_data.get('annotation', {})
            
            if annotation.get('interrupted', False) or annotation.get('error', False):
                continue
            
            # 查找对应的性能数据
            for result in self.test_results:
                if result.get('scenario_id') == scenario_id and result.get('success', False):
                    detection_results = result.get('detection_results', {})
                    objects = detection_results.get('objects', [])
                    
                    for obj in objects:
                        system_quality = obj.get('quality_score', 75)
                        user_quality = annotation.get('feature_quality_accuracy', 3) * 20  # 转换为0-100分
                        
                        system_scores.append(system_quality)
                        user_scores.append(user_quality)
                        object_classes.append(obj.get('class_name', 'unknown'))
                    break
        
        if not system_scores:
            return None
        
        # 绘制散点图
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 不同类别用不同颜色
        unique_classes = list(set(object_classes))
        colors = sns.color_palette("husl", len(unique_classes))
        
        for i, obj_class in enumerate(unique_classes):
            class_system = [s for s, c in zip(system_scores, object_classes) if c == obj_class]
            class_user = [u for u, c in zip(user_scores, object_classes) if c == obj_class]
            
            ax.scatter(class_system, class_user, c=[colors[i]], label=obj_class, 
                      alpha=0.7, s=60)
        
        # Add correlation line
        if len(system_scores) > 1:
            try:
                # 检查数据是否有足够的变异性
                if np.std(system_scores) > 0 and np.std(user_scores) > 0:
                    z = np.polyfit(system_scores, user_scores, 1)
                    p = np.poly1d(z)
                    x_line = np.linspace(min(system_scores), max(system_scores), 100)
                    ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, label='Correlation Line')
                    
                    # Calculate correlation coefficient
                    correlation = np.corrcoef(system_scores, user_scores)[0, 1]
                    if not np.isnan(correlation):
                        ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}', transform=ax.transAxes,
                               bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.5),
                               fontsize=12, verticalalignment='top')
                    else:
                        ax.text(0.05, 0.95, 'Correlation: N/A (insufficient variation)', transform=ax.transAxes,
                               bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.5),
                               fontsize=12, verticalalignment='top')
                else:
                    ax.text(0.05, 0.95, 'Correlation: N/A (no variation in data)', transform=ax.transAxes,
                           bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.5),
                           fontsize=12, verticalalignment='top')
            except (np.RankWarning, np.linalg.LinAlgError) as e:
                print(f"Warning: Could not compute correlation line: {e}")
                ax.text(0.05, 0.95, 'Correlation: N/A (computation error)', transform=ax.transAxes,
                       bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.5),
                       fontsize=12, verticalalignment='top')
        
        # Add diagonal reference line (perfect correlation)
        ax.plot([0, 100], [0, 100], 'k--', alpha=0.3, label='Perfect Correlation')
        
        ax.set_xlabel('System Predicted Quality Score', fontsize=12)
        ax.set_ylabel('User Rated Quality Score', fontsize=12)
        ax.set_title('Feature Quality Assessment Accuracy', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        chart_path = self.charts_dir / 'feature_quality_correlation.png'
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(chart_path)
    
    def plot_processing_time_breakdown(self) -> str:
        """Plot processing time breakdown bar chart"""
        if not self.performance_data:
            return None
        
        print(f"Processing time breakdown for {len(self.performance_data)} records...")
        
        # Prepare data
        stage_times = {
            'YOLO Detection': [],
            'SAM Segmentation': [],
            'Feature Extraction': [],
            '3D Post-processing': [],
            'Visualization': []
        }
        
        for perf in self.performance_data:
            timing = perf.get('timing', {})
            stage_breakdown = timing.get('stage_breakdown', {})
            
            stage_times['YOLO Detection'].append(stage_breakdown.get('yolo_detection', 0) * 1000)  # Convert to ms
            stage_times['SAM Segmentation'].append(stage_breakdown.get('sam_segmentation', 0) * 1000)
            stage_times['Feature Extraction'].append(stage_breakdown.get('feature_extraction', 0) * 1000)
            stage_times['3D Post-processing'].append(stage_breakdown.get('3d_post_processing', 0) * 1000)
            stage_times['Visualization'].append(stage_breakdown.get('visualization', 0) * 1000)
        
        if not any(stage_times.values()):
            return None
        
        # Calculate statistics
        stages = list(stage_times.keys())
        avg_times = [np.mean(times) if times else 0 for times in stage_times.values()]
        std_times = [np.std(times) if times else 0 for times in stage_times.values()]
        
        # Create bar chart
        fig, ax = plt.subplots(figsize=(14, 8))  # 增加宽度以容纳标签
        
        x_pos = np.arange(len(stages))
        bars = ax.bar(x_pos, avg_times, yerr=std_times, capsize=5,
                     color=sns.color_palette("viridis", len(stages)), alpha=0.8)
        
        ax.set_xlabel('Processing Stage', fontsize=12)
        ax.set_ylabel('Average Time (ms)', fontsize=12)
        ax.set_title('Processing Time Breakdown by Stage', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(stages, rotation=35, ha='right')  # 减少旋转角度
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, avg_val in zip(bars, avg_times):
            if avg_val > 0:  # 只显示有数值的标签
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(avg_times) * 0.01,
                       f'{avg_val:.1f}ms', ha='center', va='bottom', fontweight='bold')
        
        # 调整布局，增加底部边距
        plt.subplots_adjust(bottom=0.15)
        try:
            plt.tight_layout()
        except UserWarning:
            # 如果tight_layout失败，手动调整
            plt.subplots_adjust(bottom=0.2, top=0.9)
        chart_path = self.charts_dir / 'processing_time_breakdown.png'
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(chart_path)
    
    def _get_user_annotation_for_scenario(self, scenario_id: str) -> Dict:
        """Get user annotation for a specific scenario"""
        for ann_data in self.user_annotations:
            if ann_data.get('scenario_id') == scenario_id:
                return ann_data
        return {}
    
    def _generate_chart_index(self, generated_charts: List):
        """Generate an index file for all charts"""
        try:
            index_content = {
                'generation_timestamp': datetime.now().isoformat(),
                'total_charts': len(generated_charts),
                'charts': [
                    {
                        'name': name,
                        'file_path': str(path),
                        'file_name': Path(path).name
                    } for name, path in generated_charts
                ]
            }
            
            index_file = self.charts_dir / 'chart_index.json'
            with open(index_file, 'w', encoding='utf-8') as f:
                json.dump(index_content, f, indent=2, ensure_ascii=False)
            
            print(f"Generated chart index at: {index_file}")
            
        except Exception as e:
            print(f"Warning: Failed to generate chart index: {e}")

    def plot_error_analysis(self) -> str:
        """绘制错误分析图表"""
        if not self.test_results:
            print("No test results available for error analysis")
            return None
        
        print(f"Processing error analysis for {len(self.test_results)} results...")
        
        # 准备数据
        error_types = {'Success': 0, 'Detection Failed': 0, 'Processing Error': 0, 'Timeout': 0}
        complexity_errors = {}
        
        for result in self.test_results:
            success = result.get('success', False)
            detection_count = result.get('detection_count', 0)
            test_duration = result.get('test_duration', 0)
            complexity = result.get('test_metadata', {}).get('complexity_level', 3)
            
            # 分类错误类型
            if success and detection_count > 0:
                error_types['Success'] += 1
            elif success and detection_count == 0:
                error_types['Detection Failed'] += 1
            elif test_duration > 10:  # 假设超过10秒为超时
                error_types['Timeout'] += 1
            else:
                error_types['Processing Error'] += 1
            
            # 按复杂度分类错误
            if complexity not in complexity_errors:
                complexity_errors[complexity] = {'success': 0, 'failed': 0}
            
            if success and detection_count > 0:
                complexity_errors[complexity]['success'] += 1
            else:
                complexity_errors[complexity]['failed'] += 1
        
        # 创建双图布局
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 左图：错误类型饼图
        labels = list(error_types.keys())
        sizes = list(error_types.values())
        colors = ['#2E8B57', '#FF6347', '#FFD700', '#9370DB']
        
        wedges, texts, autotexts = ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                          startangle=90, explode=(0.1, 0, 0, 0))
        ax1.set_title('Test Result Distribution', fontsize=14, fontweight='bold')
        
        # 右图：复杂度vs失败率
        complexities = sorted(complexity_errors.keys())
        success_rates = []
        failure_rates = []
        
        for complexity in complexities:
            data = complexity_errors[complexity]
            total = data['success'] + data['failed']
            if total > 0:
                success_rates.append(data['success'] / total)
                failure_rates.append(data['failed'] / total)
            else:
                success_rates.append(0)
                failure_rates.append(0)
        
        x_pos = np.arange(len(complexities))
        width = 0.35
        
        bars1 = ax2.bar(x_pos - width/2, success_rates, width, label='Success Rate', color='#2E8B57', alpha=0.8)
        bars2 = ax2.bar(x_pos + width/2, failure_rates, width, label='Failure Rate', color='#FF6347', alpha=0.8)
        
        ax2.set_xlabel('Complexity Level', fontsize=12)
        ax2.set_ylabel('Rate', fontsize=12)
        ax2.set_title('Success/Failure Rate by Complexity', fontsize=14, fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(complexities)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'{height:.2f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        chart_path = self.charts_dir / 'error_analysis.png'
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(chart_path)
    
    def plot_detection_consistency(self) -> str:
        """Plot detection consistency analysis - Modified for single-run scenarios"""
        if not self.test_results or not self.user_annotations:
            print("Insufficient data for detection consistency analysis")
            return None
        
        print(f"Processing detection consistency for {len(self.test_results)} results...")
        
        # Since we have single runs per scenario, analyze consistency across similar scenarios
        # Group by scene characteristics instead of multiple runs of same scenario
        consistency_groups = {}
        
        for result in self.test_results:
            if not result.get('success', False):
                continue
            
            scenario_id = result.get('scenario_id', '')
            metadata = result.get('test_metadata', {})
            detection_count = result.get('detection_count', 0)
            
            # Group by complexity and lighting condition
            complexity = metadata.get('complexity_level', 3)
            lighting = metadata.get('lighting_condition', 'unknown')
            group_key = f"C{complexity}_{lighting}"
            
            if group_key not in consistency_groups:
                consistency_groups[group_key] = {
                    'detection_counts': [],
                    'user_ratings': [],
                    'processing_times': [],
                    'scenario_ids': []
                }
            
            consistency_groups[group_key]['detection_counts'].append(detection_count)
            consistency_groups[group_key]['scenario_ids'].append(scenario_id)
            
            # Get user annotation
            user_ann = self._get_user_annotation_for_scenario(scenario_id)
            if user_ann:
                user_rating = user_ann.get('annotation', {}).get('detection_accuracy', 3)
                consistency_groups[group_key]['user_ratings'].append(user_rating)
            
            processing_time = result.get('detection_results', {}).get('processing_time', 
                                    result.get('test_duration', 0))
            consistency_groups[group_key]['processing_times'].append(processing_time)
        
        # Filter groups with multiple data points
        valid_groups = {k: v for k, v in consistency_groups.items() 
                    if len(v['detection_counts']) > 1}
        
        if not valid_groups:
            print("No scenario groups with multiple data points found")
            return None
        
        # Calculate consistency metrics
        group_names = []
        detection_consistency = []
        rating_consistency = []
        time_consistency = []
        
        for group_key, data in valid_groups.items():
            # Use coefficient of variation (std/mean) as consistency metric
            det_counts = data['detection_counts']
            user_ratings = data['user_ratings']
            proc_times = data['processing_times']
            
            # Detection consistency
            if len(det_counts) > 1 and np.mean(det_counts) > 0:
                det_cv = np.std(det_counts) / np.mean(det_counts)
                detection_consistency.append(1 / (1 + det_cv))  # Convert to consistency score
            else:
                detection_consistency.append(1.0)
            
            # Rating consistency
            if len(user_ratings) > 1 and np.mean(user_ratings) > 0:
                rating_cv = np.std(user_ratings) / np.mean(user_ratings)
                rating_consistency.append(1 / (1 + rating_cv))
            else:
                rating_consistency.append(1.0)
            
            # Time consistency
            if len(proc_times) > 1 and np.mean(proc_times) > 0:
                time_cv = np.std(proc_times) / np.mean(proc_times)
                time_consistency.append(1 / (1 + time_cv))
            else:
                time_consistency.append(1.0)
            
            group_names.append(group_key)
        
        # Create chart
        fig, ax = plt.subplots(figsize=(12, 8))
        
        x_pos = np.arange(len(group_names))
        width = 0.25
        
        bars1 = ax.bar(x_pos - width, detection_consistency, width, 
                    label='Detection Consistency', color='#2E8B57', alpha=0.8)
        bars2 = ax.bar(x_pos, rating_consistency, width, 
                    label='Rating Consistency', color='#4169E1', alpha=0.8)
        bars3 = ax.bar(x_pos + width, time_consistency, width, 
                    label='Time Consistency', color='#FF6347', alpha=0.8)
        
        ax.set_xlabel('Scenario Groups (Complexity_Lighting)', fontsize=12)
        ax.set_ylabel('Consistency Score (0-1)', fontsize=12)
        ax.set_title('Detection Consistency Analysis by Scenario Groups', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(group_names, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        chart_path = self.charts_dir / 'detection_consistency.png'
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(chart_path)
    

    def plot_performance_overview(self) -> str:
        """Plot performance overview dashboard - Fixed bar labeling"""
        if not self.test_results or not self.performance_data:
            print("Insufficient data for performance overview")
            return None
        
        print(f"Creating performance overview dashboard...")
        
        # Create 2x2 subplot layout
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # Calculate key metrics
        total_tests = len(self.test_results)
        successful_tests = sum(1 for r in self.test_results if r.get('success', False))
        success_rate = successful_tests / total_tests if total_tests > 0 else 0
        
        detection_counts = [r.get('detection_count', 0) for r in self.test_results if r.get('success', False)]
        avg_detections = np.mean(detection_counts) if detection_counts else 0
        
        processing_times = []
        for perf in self.performance_data:
            timing = perf.get('timing', {})
            total_time = timing.get('total_time', 0)
            if total_time > 0:
                processing_times.append(total_time * 1000)  # Convert to ms
        
        avg_processing_time = np.mean(processing_times) if processing_times else 0
        
        # Subplot 1: Key Performance Indicators
        ax1 = fig.add_subplot(gs[0, 0])
        metrics = ['Success Rate (%)', 'Avg Detections', 'Avg Proc Time (ms)']
        values = [success_rate * 100, avg_detections, avg_processing_time]
        colors = ['#2E8B57', '#4169E1', '#FF6347']
        
        bars = ax1.barh(metrics, values, color=colors, alpha=0.8)
        ax1.set_xlabel('Value', fontsize=12)
        ax1.set_title('Key Performance Indicators', fontsize=14, fontweight='bold')
        
        # Add value labels - FIXED VERSION
        for i, (bar, value) in enumerate(zip(bars, values)):
            if 'Rate' in metrics[i]:
                label = f'{value:.1f}%'
            elif 'Time' in metrics[i]:
                label = f'{value:.1f}ms'
            else:
                label = f'{value:.1f}'
            
            ax1.text(bar.get_width() + max(values) * 0.01, bar.get_y() + bar.get_height()/2,
                    label, ha='left', va='center', fontweight='bold')
        
        # Subplot 2: Performance over time
        ax2 = fig.add_subplot(gs[0, 1])
        if len(self.test_results) > 1:
            test_indices = range(len(self.test_results))
            detection_counts_all = [r.get('detection_count', 0) for r in self.test_results]
            
            ax2.plot(test_indices, detection_counts_all, 'o-', linewidth=2, markersize=6, color='#4169E1')
            ax2.fill_between(test_indices, detection_counts_all, alpha=0.3, color='#4169E1')
            ax2.set_xlabel('Test Index', fontsize=12)
            ax2.set_ylabel('Detection Count', fontsize=12)
            ax2.set_title('Detection Performance Over Time', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'Insufficient Data\nfor Time Series', 
                    transform=ax2.transAxes, ha='center', va='center', fontsize=12)
        
        # Subplot 3: Complexity distribution
        ax3 = fig.add_subplot(gs[1, 0])
        complexity_counts = {}
        for result in self.test_results:
            complexity = result.get('test_metadata', {}).get('complexity_level', 'Unknown')
            complexity_counts[complexity] = complexity_counts.get(complexity, 0) + 1
        
        if complexity_counts:
            labels = [f"Level {k}" if isinstance(k, int) else str(k) for k in complexity_counts.keys()]
            sizes = list(complexity_counts.values())
            colors = sns.color_palette("husl", len(labels))
            
            wedges, texts, autotexts = ax3.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                            startangle=90)
            ax3.set_title('Test Complexity Distribution', fontsize=14, fontweight='bold')
        
        # Subplot 4: Processing stage times
        ax4 = fig.add_subplot(gs[1, 1])
        if self.performance_data:
            stage_names = ['YOLO', 'SAM', 'Feature', '3D Post', 'Viz']
            stage_keys = ['yolo_detection', 'sam_segmentation', 'feature_extraction', 
                        '3d_post_processing', 'visualization']
            
            stage_times = []
            for key in stage_keys:
                times = []
                for perf in self.performance_data:
                    timing = perf.get('timing', {}).get('stage_breakdown', {})
                    times.append(timing.get(key, 0) * 1000)  # Convert to ms
                stage_times.append(np.mean(times) if times else 0)
            
            bars = ax4.bar(stage_names, stage_times, 
                        color=sns.color_palette("viridis", len(stage_names)), alpha=0.8)
            ax4.set_ylabel('Time (ms)', fontsize=12)
            ax4.set_title('Average Processing Time by Stage', fontsize=14, fontweight='bold')
            ax4.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, time_val in zip(bars, stage_times):
                if time_val > 0:
                    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(stage_times) * 0.01,
                            f'{time_val:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        plt.suptitle('Performance Overview Dashboard', fontsize=16, fontweight='bold', y=0.98)
        
        chart_path = self.charts_dir / 'performance_overview.png'
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(chart_path)


# 主函数用于测试
def main():
    """Main function to test the visualization generator"""
    import sys
    import os
    
    # 设置结果目录
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    else:
        results_dir = './test_results'  # 默认目录
    
    print(f"Initializing VisualizationGenerator with directory: {results_dir}")
    
    # 确保目录存在
    os.makedirs(results_dir, exist_ok=True)
    
    try:
        # 创建可视化生成器
        viz_gen = VisualizationGenerator(results_dir)
        
        # 生成所有图表
        generated_charts = viz_gen.generate_all_charts()
        
        # 输出结果
        if generated_charts:
            print("\n" + "="*50)
            print("VISUALIZATION GENERATION COMPLETED")
            print("="*50)
            print(f"Total charts generated: {len(generated_charts)}")
            for name, path in generated_charts:
                print(f"  - {name}: {path}")
            print(f"\nAll charts saved to: {viz_gen.charts_dir}")
        else:
            print("\n" + "="*50)
            print("NO CHARTS WERE GENERATED")
            print("="*50)
            print("This could be due to:")
            print("  1. Missing or empty data files")
            print("  2. Data format issues")
            print("  3. Insufficient data for visualization")
            print("\nCheck the debug output above for more details.")
            
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())