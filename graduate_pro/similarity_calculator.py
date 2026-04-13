#!/usr/bin/env python3
"""
3D Post-Processing Statistical Analysis and Visualization Script
Analyzes detection results in test_data folder, calculates 3D post-processing trigger probability and reasons
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')
matplotlib.use('TkAgg')  # Use TkAgg backend
# Set font and style
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

class PostProcessingAnalyzer:
    def __init__(self, base_path='test_data'):
        self.base_path = base_path
        self.results = []
        self.total_yolo_detections = 0
        self.total_tests = 0
        self.pp_triggered_tests = 0
        self.merge_reasons = defaultdict(int)
        
    def load_all_data(self):
        """Load all detection result data"""
        print(f"Scanning directory: {self.base_path}")
        
        for root, dirs, files in os.walk(self.base_path):
            for file in files:
                if file == 'enhanced_detection_results.json':
                    file_path = os.path.join(root, file)
                    self._load_single_file(file_path)
        
        print(f"Loaded {len(self.results)} test cases")
        print(f"Total YOLO detections: {self.total_yolo_detections}")
        print(f"3D post-processing triggered: {self.pp_triggered_tests} times")
        
    def _load_single_file(self, file_path):
        """Load single detection result file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract basic information
            scan_dir = os.path.basename(os.path.dirname(file_path).replace('/enhanced_detection_results', ''))
            post_stats = data.get('post_processing_stats', {})
            merge_data = data.get('merge_decision_data', [])
            detailed_processing = data.get('detailed_3d_processing', {})
            
            original_detections = post_stats.get('original_detections', 0)
            filtered_detections = post_stats.get('filtered_detections', 0)
            duplicates_removed = post_stats.get('duplicates_removed', 0)
            
            # Update statistics
            self.total_yolo_detections += original_detections
            self.total_tests += 1
            
            # Check if 3D post-processing was triggered
            pp_triggered = duplicates_removed > 0
            if pp_triggered:
                self.pp_triggered_tests += 1
            
            # Analyze merge reasons
            for merge_info in merge_data:
                self._analyze_merge_reasons(merge_info)
            
            # Save results
            result_info = {
                'scan_dir': scan_dir,
                'file_path': file_path,
                'original_detections': original_detections,
                'filtered_detections': filtered_detections,
                'duplicates_removed': duplicates_removed,
                'pp_triggered': pp_triggered,
                'merge_data': merge_data,
                'detailed_processing': detailed_processing,
                'objects_count': len(data.get('objects', []))
            }
            
            self.results.append(result_info)
            
            if pp_triggered:
                print(f"✓ {scan_dir}: {original_detections}→{filtered_detections} (removed {duplicates_removed} duplicates)")
            
        except Exception as e:
            print(f"Failed to load file {file_path}: {e}")
    
    def _analyze_merge_reasons(self, merge_info):
        """Analyze reasons for merge decisions"""
        similarities = merge_info.get('individual_similarities', {})
        physical_evidence = merge_info.get('physical_evidence', {})
        
        # Determine main reasons based on similarity scores
        spatial_score = similarities.get('spatial', {}).get('similarity_score', 0)
        depth_score = similarities.get('depth', {}).get('similarity_score', 0)
        height_score = similarities.get('height', {}).get('similarity_score', 0)
        mask_score = similarities.get('mask_overlap', {}).get('similarity_score', 0)
        
        # Find strongest evidence
        if spatial_score > 0.8:
            self.merge_reasons['Too Close Spatially'] += 1
        if depth_score > 0.9:
            self.merge_reasons['Identical Depth'] += 1
        if height_score > 0.9:
            self.merge_reasons['Identical Height'] += 1
        if mask_score > 0.7:
            self.merge_reasons['High Mask Overlap'] += 1
        
        # Based on physical evidence
        if physical_evidence.get('identical_physical_properties', False):
            self.merge_reasons['Identical Physical Props'] += 1
        if physical_evidence.get('likely_yolo_duplicate', False):
            self.merge_reasons['YOLO Duplicate'] += 1
    
    def calculate_statistics(self):
        """Calculate statistical data"""
        if self.total_yolo_detections == 0:
            return {}
        
        pp_probability = (self.pp_triggered_tests / self.total_tests) * 100
        detection_based_prob = (sum(r['duplicates_removed'] for r in self.results) / self.total_yolo_detections) * 100
        
        stats = {
            'total_tests': self.total_tests,
            'total_yolo_detections': self.total_yolo_detections,
            'pp_triggered_tests': self.pp_triggered_tests,
            'pp_probability_by_test': pp_probability,
            'pp_probability_by_detection': detection_based_prob,
            'avg_detections_per_test': self.total_yolo_detections / self.total_tests,
            'total_duplicates_removed': sum(r['duplicates_removed'] for r in self.results)
        }
        
        return stats
    
    def create_individual_visualizations(self):
        """Create individual visualization charts and save them"""
        stats = self.calculate_statistics()
        
        # Create output directory for charts
        output_dir = os.path.join(self.base_path, 'post_processing_analysis')
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 3D Post-Processing Trigger Probability (by tests)
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        prob_data = [
            stats['pp_probability_by_test'],
            100 - stats['pp_probability_by_test']
        ]
        colors = ['#ff7f0e', '#1f77b4']
        wedges, texts, autotexts = ax1.pie(prob_data, 
                                          labels=['3D Post-Processing Triggered', 'No Post-Processing'],
                                          colors=colors,
                                          autopct='%1.1f%%',
                                          startangle=90)
        ax1.set_title(f'3D Post-Processing Trigger Probability\n(Based on {stats["total_tests"]} Tests)', 
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '1_pp_trigger_probability_by_test.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Duplicate Detection Probability (by detections)
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        detection_prob_data = [
            stats['pp_probability_by_detection'],
            100 - stats['pp_probability_by_detection']
        ]
        wedges2, texts2, autotexts2 = ax2.pie(detection_prob_data,
                                             labels=['Duplicate Detection', 'Independent Detection'],
                                             colors=['#d62728', '#2ca02c'],
                                             autopct='%1.2f%%',
                                             startangle=90)
        ax2.set_title(f'Duplicate Detection Probability\n(Based on {stats["total_yolo_detections"]} YOLO Detections)', 
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '2_duplicate_detection_probability.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Merge Reason Analysis
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        if self.merge_reasons:
            reasons = list(self.merge_reasons.keys())
            counts = list(self.merge_reasons.values())
            bars = ax3.barh(reasons, counts, color=['#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])
            ax3.set_title('3D Post-Processing Trigger Reason Distribution', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Number of Triggers')
            
            # Add value labels
            for bar in bars:
                width = bar.get_width()
                ax3.text(width + 0.1, bar.get_y() + bar.get_height()/2, f'{int(width)}', 
                        ha='left', va='center')
        else:
            ax3.text(0.5, 0.5, 'No merge reason data detected', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Merge Reason Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '3_merge_reason_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Detection Count Distribution
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        detection_counts = [r['original_detections'] for r in self.results]
        ax4.hist(detection_counts, bins=range(1, max(detection_counts)+2), alpha=0.7, color='skyblue', edgecolor='black')
        ax4.set_title('YOLO Detection Count Distribution per Test', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Number of Detections')
        ax4.set_ylabel('Number of Tests')
        ax4.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '4_detection_count_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Post-Processing Effect Comparison
        fig5, ax5 = plt.subplots(figsize=(12, 6))
        pp_results = [r for r in self.results if r['pp_triggered']]
        if pp_results:
            original_counts = [r['original_detections'] for r in pp_results]
            filtered_counts = [r['filtered_detections'] for r in pp_results]
            
            x = range(len(pp_results))
            width = 0.35
            ax5.bar([i - width/2 for i in x], original_counts, width, label='Original Detections', 
                   color='lightcoral', alpha=0.8)
            ax5.bar([i + width/2 for i in x], filtered_counts, width, label='After Filtering', 
                   color='lightblue', alpha=0.8)
            
            ax5.set_title('3D Post-Processing Effect Comparison', fontsize=14, fontweight='bold')
            ax5.set_xlabel('Test Cases with Post-Processing Triggered')
            ax5.set_ylabel('Number of Detections')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
            
            # Add case labels
            case_labels = [r['scan_dir'][-8:] for r in pp_results]  # Show last 8 chars
            ax5.set_xticks(x)
            ax5.set_xticklabels(case_labels, rotation=45)
        else:
            ax5.text(0.5, 0.5, 'No cases with 3D post-processing triggered', 
                    ha='center', va='center', transform=ax5.transAxes)
            ax5.set_title('3D Post-Processing Effect Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '5_post_processing_effect.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 6. Statistical Summary
        fig6, ax6 = plt.subplots(figsize=(10, 8))
        ax6.axis('off')
        summary_text = f"""3D Post-Processing Statistical Summary

Total Test Cases: {stats['total_tests']}
Total YOLO Detections: {stats['total_yolo_detections']}
3D Post-Processing Triggered: {stats['pp_triggered_tests']} times

Probability by Test: {stats['pp_probability_by_test']:.1f}%
Probability by Detection: {stats['pp_probability_by_detection']:.2f}%

Average Detections per Test: {stats['avg_detections_per_test']:.1f}
Total Duplicates Removed: {stats['total_duplicates_removed']}

YOLO Accuracy Rate: {100-stats['pp_probability_by_detection']:.2f}%
Duplicate Detection Rate: {stats['pp_probability_by_detection']:.2f}%

Analysis Interpretation:
- YOLO maintains high accuracy ({100-stats['pp_probability_by_detection']:.2f}%)
- 3D post-processing effectively handles rare duplicate cases
- Most detections ({100-stats['pp_probability_by_detection']:.2f}%) are correctly independent
- System demonstrates robust performance with minimal false positives"""

        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=12, 
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '6_statistical_summary.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nAll visualizations saved to: {output_dir}")
        print("Generated files:")
        for i in range(1, 7):
            filename = [
                "1_pp_trigger_probability_by_test.png",
                "2_duplicate_detection_probability.png", 
                "3_merge_reason_distribution.png",
                "4_detection_count_distribution.png",
                "5_post_processing_effect.png",
                "6_statistical_summary.png"
            ][i-1]
            print(f"  {filename}")
        
        return stats, output_dir
    
    def print_detailed_analysis(self):
        """Print detailed analysis report"""
        stats = self.calculate_statistics()
        
        print("\n" + "="*60)
        print("3D POST-PROCESSING DETAILED ANALYSIS REPORT")
        print("="*60)
        
        print(f"\n📊 BASIC STATISTICS:")
        print(f"  Total Test Cases: {stats['total_tests']}")
        print(f"  Total YOLO Detections: {stats['total_yolo_detections']}")
        print(f"  Average Detections per Test: {stats['avg_detections_per_test']:.1f}")
        
        print(f"\n🔄 3D POST-PROCESSING TRIGGER STATUS:")
        print(f"  Triggered Cases: {stats['pp_triggered_tests']}")
        print(f"  Trigger Probability (by test): {stats['pp_probability_by_test']:.1f}%")
        print(f"  Trigger Probability (by detection): {stats['pp_probability_by_detection']:.2f}%")
        print(f"  Total Duplicates Removed: {stats['total_duplicates_removed']}")
        
        print(f"\n🎯 YOLO PERFORMANCE EVALUATION:")
        print(f"  YOLO Accuracy Rate: {100-stats['pp_probability_by_detection']:.2f}%")
        print(f"  False Positive/Duplicate Rate: {stats['pp_probability_by_detection']:.2f}%")
        
        if self.merge_reasons:
            print(f"\n📋 MERGE REASON ANALYSIS:")
            for reason, count in sorted(self.merge_reasons.items(), key=lambda x: x[1], reverse=True):
                print(f"  {reason}: {count} times")
        
        # Show specific triggered cases
        pp_cases = [r for r in self.results if r['pp_triggered']]
        if pp_cases:
            print(f"\n📝 SPECIFIC TRIGGERED CASES:")
            for i, case in enumerate(pp_cases, 1):
                print(f"  {i}. {case['scan_dir']}: "
                      f"{case['original_detections']}→{case['filtered_detections']} "
                      f"(removed {case['duplicates_removed']})")

def main():
    """Main function"""
    # Create analyzer
    analyzer = PostProcessingAnalyzer('test_data')  # Modify to your actual path
    
    # Load data
    try:
        analyzer.load_all_data()
    except FileNotFoundError:
        print("❌ test_data directory not found, please ensure the path is correct")
        print("   Supported path formats:")
        print("   - test_data/")
        print("   - test_data/scan_output_*/enhanced_detection_results/enhanced_detection_results.json")
        return
    
    if len(analyzer.results) == 0:
        print("❌ No valid detection result files found")
        return
    
    # Generate report and visualizations
    analyzer.print_detailed_analysis()
    stats, output_dir = analyzer.create_individual_visualizations()
    
    return analyzer, stats, output_dir

if __name__ == "__main__":
    analyzer, stats, output_dir = main()