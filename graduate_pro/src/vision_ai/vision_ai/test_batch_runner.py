#!/usr/bin/env python3
# system_performance_analyzer.py - 基于参考库数据的系统性能量化分析

import json
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import pandas as pd

class SystemPerformanceAnalyzer:
    def __init__(self, test_data_dir: str = "/home/qi/ros2_ws/test_data"):
        self.test_data_dir = Path(test_data_dir)
        self.reference_libraries = {}
        self.performance_data = {}
        self.metadata = {}
        
        print(f"System Performance Analyzer initialized")
        print(f"Test data directory: {self.test_data_dir}")
    
    def load_all_reference_libraries(self):
        """加载所有参考库文件"""
        scan_dirs = [d for d in self.test_data_dir.iterdir() 
                    if d.is_dir() and d.name.startswith('scan_output_')]
        
        print(f"Found {len(scan_dirs)} scan directories")
        
        for scan_dir in scan_dirs:
            scenario_id = scan_dir.name
            
            # 加载参考库
            ref_lib_file = scan_dir / 'enhanced_detection_results' / 'reference_library.json'
            if ref_lib_file.exists():
                try:
                    with open(ref_lib_file, 'r', encoding='utf-8') as f:
                        self.reference_libraries[scenario_id] = json.load(f)
                    print(f"✅ Loaded reference library: {scenario_id}")
                except Exception as e:
                    print(f"❌ Failed to load {scenario_id}: {e}")
                    continue
            else:
                print(f"❌ Reference library not found: {ref_lib_file}")
                continue
            
            # 加载元数据
            metadata_file = scan_dir / 'test_metadata.json'
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        self.metadata[scenario_id] = json.load(f)
                except:
                    self.metadata[scenario_id] = self._get_default_metadata()
            else:
                self.metadata[scenario_id] = self._get_default_metadata()
        
        print(f"Successfully loaded {len(self.reference_libraries)} reference libraries")
    
    def _get_default_metadata(self):
        return {
            'complexity_level': 3,
            'lighting_condition': 'unknown',
            'object_count': 0,
            'interference_objects': 0,
            'occlusion_level': 0.0,
            'scene_type': 'unknown'
        }
    
    def analyze_3d_post_processing_triggers(self) -> Dict:
        """分析3D后处理触发条件"""
        print("\n=== 3D Post-Processing Trigger Analysis ===")
        
        trigger_stats = {
            'by_lighting': {},
            'by_complexity': {},
            'by_object_count': {},
            'total_scenarios': len(self.reference_libraries),
            'triggered_scenarios': 0
        }
        
        for scenario_id, ref_lib in self.reference_libraries.items():
            metadata = self.metadata.get(scenario_id, {})
            
            # 检查是否触发了3D后处理（通过检测对象数量推断）
            detected_objects = len(ref_lib)
            expected_objects = metadata.get('object_count', 0)
            interference_objects = metadata.get('interference_objects', 0)
            
            # 如果检测到的对象数量显著少于预期，说明可能进行了去重
            likely_triggered = (expected_objects + interference_objects > detected_objects + 1)
            
            if likely_triggered:
                trigger_stats['triggered_scenarios'] += 1
                
                # 按光照条件统计
                lighting = metadata.get('lighting_condition', 'unknown')
                if lighting not in trigger_stats['by_lighting']:
                    trigger_stats['by_lighting'][lighting] = {'triggered': 0, 'total': 0}
                trigger_stats['by_lighting'][lighting]['triggered'] += 1
            
            # 统计总数
            lighting = metadata.get('lighting_condition', 'unknown')
            if lighting not in trigger_stats['by_lighting']:
                trigger_stats['by_lighting'][lighting] = {'triggered': 0, 'total': 0}
            trigger_stats['by_lighting'][lighting]['total'] += 1
            
            # 按复杂度统计
            complexity = metadata.get('complexity_level', 3)
            if complexity not in trigger_stats['by_complexity']:
                trigger_stats['by_complexity'][complexity] = {'triggered': 0, 'total': 0}
            if likely_triggered:
                trigger_stats['by_complexity'][complexity]['triggered'] += 1
            trigger_stats['by_complexity'][complexity]['total'] += 1
            
            # 按物体数量统计
            obj_count_range = self._get_object_count_range(expected_objects + interference_objects)
            if obj_count_range not in trigger_stats['by_object_count']:
                trigger_stats['by_object_count'][obj_count_range] = {'triggered': 0, 'total': 0}
            if likely_triggered:
                trigger_stats['by_object_count'][obj_count_range]['triggered'] += 1
            trigger_stats['by_object_count'][obj_count_range]['total'] += 1
        
        # 计算触发率
        total_trigger_rate = trigger_stats['triggered_scenarios'] / trigger_stats['total_scenarios']
        
        print(f"Overall Trigger Rate: {trigger_stats['triggered_scenarios']}/{trigger_stats['total_scenarios']} ({total_trigger_rate:.1%})")
        
        print("\nBy Lighting Condition:")
        for lighting, stats in trigger_stats['by_lighting'].items():
            rate = stats['triggered'] / stats['total'] if stats['total'] > 0 else 0
            print(f"  {lighting}: {stats['triggered']}/{stats['total']} ({rate:.1%})")
        
        return trigger_stats
    
    def _get_object_count_range(self, count):
        """将物体数量分组"""
        if count <= 1:
            return "1"
        elif count <= 2:
            return "2"
        elif count <= 4:
            return "3-4"
        else:
            return "5+"
    
    def analyze_depth_data_completeness(self) -> Dict:
        """分析深度数据完整性"""
        print("\n=== Depth Data Completeness Analysis ===")
        
        depth_stats = {
            'height_measurement_success': 0,
            'background_depth_success': 0,
            'gripper_width_success': 0,
            'spatial_features_complete': 0,
            'geometric_features_complete': 0,
            'total_objects': 0,
            'depth_confidence_scores': []
        }
        
        for scenario_id, ref_lib in self.reference_libraries.items():
            for obj_id, obj_data in ref_lib.items():
                depth_stats['total_objects'] += 1
                
                features = obj_data.get('features', {})
                spatial = features.get('spatial', {})
                geometric = features.get('geometric', {})
                
                # 高度测量成功率
                if spatial.get('height_mm', 0) > 0:
                    depth_stats['height_measurement_success'] += 1
                
                # 背景深度测量成功率
                if spatial.get('background_depth_m', 0) > 0:
                    depth_stats['background_depth_success'] += 1
                
                # 抓取宽度信息成功率
                if 'gripper_width_info' in spatial:
                    depth_stats['gripper_width_success'] += 1
                
                # 空间特征完整度
                spatial_features = ['centroid_3d_camera', 'world_coordinates', 'distance_to_camera']
                if all(key in spatial for key in spatial_features):
                    depth_stats['spatial_features_complete'] += 1
                
                # 3D几何特征完整度
                geometric_features = ['fpfh', 'pca_features', 'bbox_dimensions', 'density_stats']
                if all(key in geometric for key in geometric_features):
                    depth_stats['geometric_features_complete'] += 1
                
                # 收集置信度分数
                confidence = spatial.get('confidence', 0)
                if confidence > 0:
                    depth_stats['depth_confidence_scores'].append(confidence)
        
        # 计算成功率
        total = depth_stats['total_objects']
        if total > 0:
            print(f"Height Measurement Success Rate: {depth_stats['height_measurement_success']}/{total} ({depth_stats['height_measurement_success']/total:.1%})")
            print(f"Background Depth Success Rate: {depth_stats['background_depth_success']}/{total} ({depth_stats['background_depth_success']/total:.1%})")
            print(f"Gripper Width Estimation Success Rate: {depth_stats['gripper_width_success']}/{total} ({depth_stats['gripper_width_success']/total:.1%})")
            print(f"Spatial Features Completeness: {depth_stats['spatial_features_complete']}/{total} ({depth_stats['spatial_features_complete']/total:.1%})")
            print(f"3D Geometric Features Completeness: {depth_stats['geometric_features_complete']}/{total} ({depth_stats['geometric_features_complete']/total:.1%})")
        
        if depth_stats['depth_confidence_scores']:
            avg_confidence = np.mean(depth_stats['depth_confidence_scores'])
            print(f"Average Depth Confidence: {avg_confidence:.3f}")
        
        return depth_stats
    
    def analyze_feature_richness(self) -> Dict:
        """分析3D特征丰富度"""
        print("\n=== 3D Feature Richness Analysis ===")
        
        feature_stats = {
            'geometric_feature_counts': [],
            'fpfh_dimensions': [],
            'pca_feature_completeness': 0,
            'surface_normal_availability': 0,
            'density_analysis_availability': 0,
            'shape_context_3d_availability': 0,
            'total_objects': 0
        }
        
        for scenario_id, ref_lib in self.reference_libraries.items():
            for obj_id, obj_data in ref_lib.items():
                feature_stats['total_objects'] += 1
                
                geometric = obj_data.get('features', {}).get('geometric', {})
                
                # 计算几何特征数量
                geometric_feature_count = 0
                geometric_features = ['hu_moments', 'fpfh', 'pca_features', 'point_spread', 
                                    'shape_context_3d', 'density_stats', 'surface_normal_stats']
                
                for feature in geometric_features:
                    if feature in geometric:
                        geometric_feature_count += 1
                        
                        # 检查FPFH维度
                        if feature == 'fpfh' and isinstance(geometric[feature], list):
                            feature_stats['fpfh_dimensions'].append(len(geometric[feature]))
                        
                        # 检查PCA特征完整性
                        if feature == 'pca_features':
                            pca_features = ['eigenvalues', 'linearity', 'planarity', 'sphericity']
                            if all(key in geometric[feature] for key in pca_features):
                                feature_stats['pca_feature_completeness'] += 1
                        
                        # 表面法向量统计
                        if feature == 'surface_normal_stats':
                            feature_stats['surface_normal_availability'] += 1
                        
                        # 密度分析
                        if feature == 'density_stats':
                            feature_stats['density_analysis_availability'] += 1
                        
                        # 3D形状上下文
                        if feature == 'shape_context_3d':
                            feature_stats['shape_context_3d_availability'] += 1
                
                feature_stats['geometric_feature_counts'].append(geometric_feature_count)
        
        # 计算统计数据
        total = feature_stats['total_objects']
        if feature_stats['geometric_feature_counts']:
            avg_features = np.mean(feature_stats['geometric_feature_counts'])
            print(f"Average Geometric Features per Object: {avg_features:.1f}")
            print(f"Feature Count Distribution: min={np.min(feature_stats['geometric_feature_counts'])}, max={np.max(feature_stats['geometric_feature_counts'])}")
        
        if feature_stats['fpfh_dimensions']:
            avg_fpfh_dim = np.mean(feature_stats['fpfh_dimensions'])
            print(f"Average FPFH Dimensions: {avg_fpfh_dim:.0f}")
        
        if total > 0:
            print(f"PCA Feature Completeness: {feature_stats['pca_feature_completeness']}/{total} ({feature_stats['pca_feature_completeness']/total:.1%})")
            print(f"Surface Normal Analysis: {feature_stats['surface_normal_availability']}/{total} ({feature_stats['surface_normal_availability']/total:.1%})")
            print(f"Density Analysis: {feature_stats['density_analysis_availability']}/{total} ({feature_stats['density_analysis_availability']/total:.1%})")
            print(f"3D Shape Context: {feature_stats['shape_context_3d_availability']}/{total} ({feature_stats['shape_context_3d_availability']/total:.1%})")
        
        return feature_stats
    
    def analyze_processing_efficiency(self) -> Dict:
        """分析处理效率与物体复杂度关系"""
        print("\n=== Processing Efficiency Analysis ===")
        
        efficiency_stats = {
            'objects_per_scenario': [],
            'features_per_object': [],
            'complexity_vs_objects': {},
            'lighting_vs_processing': {}
        }
        
        for scenario_id, ref_lib in self.reference_libraries.items():
            metadata = self.metadata.get(scenario_id, {})
            object_count = len(ref_lib)
            complexity = metadata.get('complexity_level', 3)
            lighting = metadata.get('lighting_condition', 'unknown')
            
            efficiency_stats['objects_per_scenario'].append(object_count)
            
            # 复杂度vs物体数量
            if complexity not in efficiency_stats['complexity_vs_objects']:
                efficiency_stats['complexity_vs_objects'][complexity] = []
            efficiency_stats['complexity_vs_objects'][complexity].append(object_count)
            
            # 光照条件vs处理结果
            if lighting not in efficiency_stats['lighting_vs_processing']:
                efficiency_stats['lighting_vs_processing'][lighting] = []
            efficiency_stats['lighting_vs_processing'][lighting].append(object_count)
            
            # 计算每个物体的特征数量
            for obj_id, obj_data in ref_lib.items():
                features = obj_data.get('features', {})
                feature_count = sum(1 for feature_type in ['geometric', 'shape', 'appearance', 'spatial'] 
                                  if feature_type in features and features[feature_type])
                efficiency_stats['features_per_object'].append(feature_count)
        
        # 打印统计结果
        if efficiency_stats['objects_per_scenario']:
            avg_objects = np.mean(efficiency_stats['objects_per_scenario'])
            print(f"Average Objects per Scenario: {avg_objects:.1f}")
        
        if efficiency_stats['features_per_object']:
            avg_features = np.mean(efficiency_stats['features_per_object'])
            print(f"Average Feature Types per Object: {avg_features:.1f}")
        
        print("\nObjects Detected by Complexity Level:")
        for complexity in sorted(efficiency_stats['complexity_vs_objects'].keys()):
            objects = efficiency_stats['complexity_vs_objects'][complexity]
            avg_obj = np.mean(objects) if objects else 0
            print(f"  Level {complexity}: {avg_obj:.1f} avg objects ({len(objects)} scenarios)")
        
        print("\nObjects Detected by Lighting Condition:")
        for lighting, objects in efficiency_stats['lighting_vs_processing'].items():
            avg_obj = np.mean(objects) if objects else 0
            print(f"  {lighting}: {avg_obj:.1f} avg objects ({len(objects)} scenarios)")
        
        return efficiency_stats
    
    def analyze_descriptor_uniqueness(self) -> Dict:
        """分析描述子唯一性"""
        print("\n=== Descriptor Uniqueness Analysis ===")
        
        uniqueness_stats = {
            'fpfh_similarities': [],
            'color_diversities': [],
            'spatial_separations': [],
            'cross_object_comparisons': 0
        }
        
        all_objects = []
        for scenario_id, ref_lib in self.reference_libraries.items():
            for obj_id, obj_data in ref_lib.items():
                all_objects.append((scenario_id, obj_id, obj_data))
        
        # 比较不同物体间的特征相似度
        for i, (scenario1, obj_id1, obj_data1) in enumerate(all_objects):
            for j, (scenario2, obj_id2, obj_data2) in enumerate(all_objects[i+1:], i+1):
                if obj_data1.get('metadata', {}).get('class_name') != obj_data2.get('metadata', {}).get('class_name'):
                    # 只比较不同类别的物体
                    uniqueness_stats['cross_object_comparisons'] += 1
                    
                    # FPFH相似度
                    fpfh1 = obj_data1.get('features', {}).get('geometric', {}).get('fpfh', [])
                    fpfh2 = obj_data2.get('features', {}).get('geometric', {}).get('fpfh', [])
                    if fpfh1 and fpfh2 and len(fpfh1) == len(fpfh2):
                        similarity = np.dot(fpfh1, fpfh2) / (np.linalg.norm(fpfh1) * np.linalg.norm(fpfh2))
                        uniqueness_stats['fpfh_similarities'].append(similarity)
                    
                    # 颜色差异
                    color1 = obj_data1.get('features', {}).get('appearance', {}).get('mean_color', [])
                    color2 = obj_data2.get('features', {}).get('appearance', {}).get('mean_color', [])
                    if color1 and color2:
                        color_distance = np.linalg.norm(np.array(color1) - np.array(color2))
                        uniqueness_stats['color_diversities'].append(color_distance)
                    
                    # 空间分离
                    pos1 = obj_data1.get('features', {}).get('spatial', {}).get('world_coordinates', [])
                    pos2 = obj_data2.get('features', {}).get('spatial', {}).get('world_coordinates', [])
                    if pos1 and pos2:
                        spatial_distance = np.linalg.norm(np.array(pos1) - np.array(pos2))
                        uniqueness_stats['spatial_separations'].append(spatial_distance)
        
        # 打印结果
        print(f"Cross-object Comparisons: {uniqueness_stats['cross_object_comparisons']}")
        
        if uniqueness_stats['fpfh_similarities']:
            avg_sim = np.mean(uniqueness_stats['fpfh_similarities'])
            print(f"Average FPFH Similarity (different classes): {avg_sim:.3f} (lower is better)")
        
        if uniqueness_stats['color_diversities']:
            avg_color_div = np.mean(uniqueness_stats['color_diversities'])
            print(f"Average Color Diversity: {avg_color_div:.1f} (higher is better)")
        
        if uniqueness_stats['spatial_separations']:
            avg_spatial = np.mean(uniqueness_stats['spatial_separations'])
            print(f"Average Spatial Separation: {avg_spatial:.1f}mm")
        
        return uniqueness_stats
    
    def generate_performance_report(self, output_dir: str = None):
        """生成完整的性能分析报告"""
        if output_dir is None:
            output_dir = f"system_performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print(f"\n=== Generating Performance Report ===")
        print(f"Output directory: {output_path}")
        
        # 执行所有分析
        trigger_stats = self.analyze_3d_post_processing_triggers()
        depth_stats = self.analyze_depth_data_completeness()
        feature_stats = self.analyze_feature_richness()
        efficiency_stats = self.analyze_processing_efficiency()
        uniqueness_stats = self.analyze_descriptor_uniqueness()
        
        # 保存详细数据
        report_data = {
            'analysis_timestamp': datetime.now().isoformat(),
            'total_scenarios': len(self.reference_libraries),
            'total_objects': sum(len(lib) for lib in self.reference_libraries.values()),
            '3d_post_processing_triggers': trigger_stats,
            'depth_data_completeness': depth_stats,
            'feature_richness': feature_stats,
            'processing_efficiency': efficiency_stats,
            'descriptor_uniqueness': uniqueness_stats
        }
        
        # 保存JSON报告
        report_file = output_path / 'system_performance_report.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)
        
        # 生成可视化图表
        self._generate_performance_visualizations(report_data, output_path)
        
        print(f"✅ Performance report generated: {output_path}")
        
        return report_data
    
    def _generate_performance_visualizations(self, report_data: Dict, output_path: Path):
        """生成性能可视化图表"""
        
        # 图1: 3D后处理触发率
        trigger_stats = report_data['3d_post_processing_triggers']
        if trigger_stats['by_lighting']:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # 按光照条件的触发率
            lighting_conditions = list(trigger_stats['by_lighting'].keys())
            trigger_rates = [trigger_stats['by_lighting'][lc]['triggered']/trigger_stats['by_lighting'][lc]['total'] 
                           for lc in lighting_conditions]
            
            ax1.bar(lighting_conditions, trigger_rates, alpha=0.8, color=sns.color_palette("viridis", len(lighting_conditions)))
            ax1.set_ylabel('Trigger Rate')
            ax1.set_title('3D Post-Processing Trigger Rate by Lighting Condition')
            ax1.set_ylim(0, 1)
            
            # 添加数值标签
            for i, rate in enumerate(trigger_rates):
                ax1.text(i, rate + 0.02, f'{rate:.1%}', ha='center', fontweight='bold')
            
            # 按复杂度的触发率
            complexity_levels = sorted(trigger_stats['by_complexity'].keys())
            complexity_rates = [trigger_stats['by_complexity'][cl]['triggered']/trigger_stats['by_complexity'][cl]['total'] 
                              for cl in complexity_levels]
            
            ax2.bar([f'Level {cl}' for cl in complexity_levels], complexity_rates, 
                   alpha=0.8, color=sns.color_palette("plasma", len(complexity_levels)))
            ax2.set_ylabel('Trigger Rate')
            ax2.set_title('3D Post-Processing Trigger Rate by Complexity Level')
            ax2.set_ylim(0, 1)
            
            for i, rate in enumerate(complexity_rates):
                ax2.text(i, rate + 0.02, f'{rate:.1%}', ha='center', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(output_path / '3d_post_processing_triggers.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 图2: 深度数据完整性
        depth_stats = report_data['depth_data_completeness']
        if depth_stats['total_objects'] > 0:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            categories = ['Height\nMeasurement', 'Background\nDepth', 'Gripper Width\nEstimation', 
                         'Spatial Features\nComplete', '3D Geometric\nFeatures Complete']
            success_counts = [depth_stats['height_measurement_success'], depth_stats['background_depth_success'],
                            depth_stats['gripper_width_success'], depth_stats['spatial_features_complete'],
                            depth_stats['geometric_features_complete']]
            success_rates = [count/depth_stats['total_objects'] for count in success_counts]
            
            bars = ax.bar(categories, success_rates, alpha=0.8, color=sns.color_palette("Set2", len(categories)))
            ax.set_ylabel('Success Rate')
            ax.set_title('Depth-Enhanced Feature Extraction Completeness')
            ax.set_ylim(0, 1)
            
            # 添加数值标签
            for bar, rate in zip(bars, success_rates):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                       f'{rate:.1%}', ha='center', fontweight='bold')
            
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(output_path / 'depth_data_completeness.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 图3: 特征丰富度分布
        feature_stats = report_data['feature_richness']
        if feature_stats['geometric_feature_counts']:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # 几何特征数量分布
            ax1.hist(feature_stats['geometric_feature_counts'], bins=range(max(feature_stats['geometric_feature_counts'])+2), 
                    alpha=0.7, color='skyblue', edgecolor='black')
            ax1.set_xlabel('Number of Geometric Features')
            ax1.set_ylabel('Object Count')
            ax1.set_title('Distribution of Geometric Features per Object')
            
            # 特征可用性对比
            feature_types = ['PCA Features', 'Surface Normal\nAnalysis', 'Density Analysis', '3D Shape Context']
            availability_rates = [
                feature_stats['pca_feature_completeness']/feature_stats['total_objects'],
                feature_stats['surface_normal_availability']/feature_stats['total_objects'],
                feature_stats['density_analysis_availability']/feature_stats['total_objects'],
                feature_stats['shape_context_3d_availability']/feature_stats['total_objects']
            ]
            
            bars = ax2.bar(feature_types, availability_rates, alpha=0.8, color=sns.color_palette("viridis", len(feature_types)))
            ax2.set_ylabel('Availability Rate')
            ax2.set_title('Advanced 3D Feature Availability')
            ax2.set_ylim(0, 1)
            
            for bar, rate in zip(bars, availability_rates):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                        f'{rate:.1%}', ha='center', fontweight='bold')
            
            plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(output_path / 'feature_richness_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print("✅ Performance visualization charts generated")

def main():
    analyzer = SystemPerformanceAnalyzer()
    
    # 加载所有参考库
    analyzer.load_all_reference_libraries()
    
    if not analyzer.reference_libraries:
        print("❌ No reference libraries found!")
        return
    
    # 生成性能分析报告
    report = analyzer.generate_performance_report()
    
    print(f"\n{'='*60}")
    print("SYSTEM PERFORMANCE ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"Total Scenarios Analyzed: {report['total_scenarios']}")
    print(f"Total Objects in Reference Libraries: {report['total_objects']}")
    print(f"Analysis Completed: {report['analysis_timestamp']}")

if __name__ == '__main__':
    main()