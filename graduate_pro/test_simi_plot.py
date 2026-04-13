import os
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

# 设置matplotlib后端
matplotlib.use('TKAgg')

# 从用户提供的文件中导入 FeatureSimilarityCalculator 类
from similarity_calculator import FeatureSimilarityCalculator

def load_json_data(file_path: str) -> Optional[Dict]:
    """安全地加载JSON文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"警告：文件未找到 - {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"警告：JSON文件解码失败 - {file_path}")
        return None

def group_objects_by_class(objects: List[Dict]) -> Dict[str, List[Dict]]:
    """将对象列表按类别分组"""
    grouped_objects = defaultdict(list)
    for obj in objects:
        class_name = obj.get('class_name')
        if class_name:
            grouped_objects[class_name].append(obj)
    return dict(grouped_objects)

def calculate_all_pairwise_similarities_by_class(
    ref_objects_by_class: Dict[str, List[Dict]], 
    current_objects_by_class: Dict[str, List[Dict]], 
    calculator: FeatureSimilarityCalculator
) -> Dict:
    """计算所有同类别物体之间的两两相似度"""
    
    all_similarities = defaultdict(list)
    
    # 遍历当前文件中的所有类别
    for class_name, current_objects in current_objects_by_class.items():
        # 仅当参考库中存在此类别时才进行比对
        if class_name in ref_objects_by_class:
            ref_objects = ref_objects_by_class[class_name]
            
            print(f"  正在比对类别 '{class_name}': {len(current_objects)} 个物体 vs {len(ref_objects)} 个物体")
            
            # 对同类别物体进行两两比对
            for current_obj in current_objects:
                for ref_obj in ref_objects:
                    result = calculator.calculate_overall_similarity(ref_obj['features'], current_obj['features'])
                    
                    # --- 调试代码开始 ---
                    print(f"    调试信息：正在处理 {class_name} 的一个案例。")
                    print(f"    完整返回结果：{json.dumps(result, indent=2)}")
                    # --- 调试代码结束 ---
                    
                    contributions = result.get('contributions', {})
                    
                    # 确保每个特征都存在，并从 individual_scores 字典中提取所有值
                    if 'geometric' in contributions and 'individual_scores' in contributions['geometric']:
                        all_similarities['geometric'].extend(contributions['geometric']['individual_scores'].values())
                    if 'appearance' in contributions and 'individual_scores' in contributions['appearance']:
                        all_similarities['appearance'].extend(contributions['appearance']['individual_scores'].values())
                    if 'shape' in contributions and 'individual_scores' in contributions['shape']:
                        all_similarities['shape'].extend(contributions['shape']['individual_scores'].values())
                    if 'spatial' in contributions and 'individual_scores' in contributions['spatial']:
                        all_similarities['spatial'].extend(contributions['spatial']['individual_scores'].values())

                    # 提取最终的组合分数
                    all_similarities['combined'].append(result.get('final_score', 0))
            
    return dict(all_similarities)

def plot_confidence_distributions(all_similarities: Dict, output_dir: str):
    """为每种特征绘制置信度分布直方图。"""
    os.makedirs(output_dir, exist_ok=True)
    
    titles = {
        'geometric': '3D Geometric Confidence Distribution',
        'appearance': 'Color (Appearance) Confidence Distribution',
        'shape': '2D Shape Confidence Distribution',
        'spatial': 'Spatial Confidence Distribution',
        'combined': 'Combined Matching Confidence Distribution'
    }
    
    colors = {
        'geometric': '#3498db',
        'appearance': '#f39c12',
        'shape': '#9b59b6',
        'spatial': '#2ecc71',
        'combined': '#1abc9c'
    }
    
    for key, values in all_similarities.items():
        if not values:
            print(f"警告: {key}特征没有数据，跳过绘制。")
            continue
        
        plt.figure(figsize=(8, 6))
        plt.hist(values, bins=20, alpha=0.7, color=colors[key], edgecolor='black')
        
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        plt.title(f'{titles[key]}\nMean: {mean_val:.3f}±{std_val:.3f}', 
                      fontsize=14, fontweight='bold')
        plt.xlabel('Similarity Score', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'chart_{key}_distribution.png'), dpi=300)
        plt.show(block=False)
        
def plot_confidence_box_plot(all_similarities: Dict, output_dir: str):
    """绘制置信度对比箱线图"""
    plt.figure(figsize=(10, 8))
    
    feature_keys = ['geometric', 'appearance', 'shape', 'spatial', 'combined']
    plot_data = [all_similarities[key] for key in feature_keys if all_similarities[key]]
    plot_labels = [key.capitalize() for key in feature_keys if all_similarities[key]]
    
    bp = plt.boxplot(plot_data, labels=plot_labels, patch_artist=True)
    colors = ['#3498db', '#f39c12', '#9b59b6', '#2ecc71', '#1abc9c']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    plt.title('Confidence Comparison (Box Plot)', fontsize=16, fontweight='bold')
    plt.ylabel('Confidence Value', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'chart_box_plot.png'), dpi=300)
    plt.show(block=False)

def plot_confidence_scatter(all_similarities: Dict, output_dir: str):
    """绘制置信度相关性散点图"""
    if not all_similarities.get('shape') or not all_similarities.get('appearance') or not all_similarities.get('combined'):
        print("警告: 缺少形状、颜色或组合相似度数据，无法绘制散点图。")
        return
        
    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(all_similarities['shape'], all_similarities['appearance'], 
                         c=all_similarities['combined'], cmap='viridis', alpha=0.6, s=50)
    
    plt.xlabel('2D Shape Similarity', fontsize=12)
    plt.ylabel('Color Similarity', fontsize=12)
    plt.title('Shape vs Color Confidence\n(Color = Combined Score)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    cbar = plt.colorbar(scatter)
    cbar.set_label('Combined Similarity', rotation=270, labelpad=15, fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'chart_scatter.png'), dpi=300)
    plt.show(block=True)

def main(reference_folder_name: Optional[str] = None):
    root_dir = "/home/qi/ros2_ws/test_data"
    output_dir = os.path.join(root_dir, "similarity_analysis_results")
    os.makedirs(output_dir, exist_ok=True)
    
    scan_dirs = sorted([d for d in os.listdir(root_dir) if d.startswith('scan_output_') and os.path.isdir(os.path.join(root_dir, d))])

    if not scan_dirs:
        print(f"在 {root_dir} 目录下未找到任何 'scan_output_' 文件夹。")
        return
    
    if reference_folder_name and reference_folder_name in scan_dirs:
        reference_dir = os.path.join(root_dir, reference_folder_name)
        print(f"使用指定的参考文件夹: {reference_folder_name}")
    else:
        reference_dir = os.path.join(root_dir, scan_dirs[0])
        print(f"未指定参考文件夹或指定文件夹不存在，默认使用第一个文件夹: {scan_dirs[0]}")
    
    reference_json_path = os.path.join(reference_dir, 'enhanced_detection_results', 'enhanced_detection_results.json')
    
    ref_data = load_json_data(reference_json_path)
    if not ref_data:
        print("无法加载参考库数据，程序终止。")
        return
        
    ref_objects_by_class = group_objects_by_class(ref_data['objects'])
    
    calculator = FeatureSimilarityCalculator()
    all_collected_similarities = {
        'geometric': [], 'appearance': [], 'shape': [], 'spatial': [], 'combined': []
    }
    
    current_scan_dirs = [d for d in scan_dirs if d != os.path.basename(reference_dir)]

    for i, scan_dir_name in enumerate(current_scan_dirs):
        current_dir = os.path.join(root_dir, scan_dir_name)
        current_json_path = os.path.join(current_dir, 'enhanced_detection_results', 'enhanced_detection_results.json')
        
        current_data = load_json_data(current_json_path)
        if not current_data:
            continue
            
        current_objects_by_class = group_objects_by_class(current_data['objects'])
        
        print(f"正在处理文件夹: {scan_dir_name}")
        current_similarities = calculate_all_pairwise_similarities_by_class(
            ref_objects_by_class, 
            current_objects_by_class, 
            calculator
        )
        
        for key, value_list in current_similarities.items():
            all_collected_similarities[key].extend(value_list)
            
    print("\n相似度计算完成，开始生成图表...")
    
    plot_confidence_distributions(all_collected_similarities, output_dir)
    plot_confidence_box_plot(all_collected_similarities, output_dir)
    plot_confidence_scatter(all_collected_similarities, output_dir)
    
    print(f"\n所有图表已保存至: {output_dir}")
    print("程序执行完毕。")

if __name__ == "__main__":
    main(reference_folder_name="scan_output_20250829_202323")