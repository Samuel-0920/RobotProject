#!/usr/bin/env python3
# test_enhanced_features.py - 测试增强检测功能的脚本

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import time
import sys
import os
import numpy as np
import cv2
import json
from pathlib import Path

# 添加项目路径到sys.path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def test_enhanced_components():
    """测试所有增强组件"""
    print("🧪 开始测试增强检测组件...")
    print("=" * 60)
    
    # 测试1: 配置管理器
    print("\n1️⃣ 测试配置管理器...")
    try:
        from detection.utils.enhanced_config_manager import EnhancedConfigManager
        
        config_path = "detection/config/enhanced_detection_config.json"
        config_manager = EnhancedConfigManager(config_path)
        
        print("✅ 配置管理器初始化成功")
        print(f"📋 配置摘要:\n{config_manager.get_config_summary()}")
        
        # 验证配置
        validation = config_manager.validate_config()
        if validation['errors']:
            print(f"❌ 配置错误: {validation['errors']}")
        else:
            print("✅ 配置验证通过")
            
    except Exception as e:
        print(f"❌ 配置管理器测试失败: {e}")
        return False
    
    # 测试2: 坐标计算器
    print("\n2️⃣ 测试坐标计算器...")
    try:
        from detection.utils.coordinate_calculator import CoordinateCalculator, ObjectAnalyzer
        
        # 获取相机配置
        camera_config = config_manager.get_camera_config()
        calibration_data = {
            'camera_intrinsics': camera_config.get('intrinsics', {}),
            'hand_eye_translation': camera_config.get('hand_eye_calibration', {}).get('translation'),
            'hand_eye_quaternion': camera_config.get('hand_eye_calibration', {}).get('quaternion')
        }
        
        coord_calc = CoordinateCalculator(calibration_data)
        obj_analyzer = ObjectAnalyzer(coord_calc)
        
        # 测试像素到相机坐标转换
        test_pixel_x, test_pixel_y, test_depth = 320, 240, 0.5  # 图像中心，0.5m深度
        camera_point = coord_calc.pixel_to_camera_coordinates(test_pixel_x, test_pixel_y, test_depth)
        print(f"✅ 像素坐标转换测试: ({test_pixel_x}, {test_pixel_y}, {test_depth}m) -> {camera_point}")
        
        # 测试相机到世界坐标转换
        tcp_pose = [100, 200, 350, 179, 0, 0]  # 示例TCP位姿
        world_point = coord_calc.camera_to_world_coordinates(camera_point, tcp_pose)
        print(f"✅ 世界坐标转换测试: {camera_point} -> {world_point}")
        
    except Exception as e:
        print(f"❌ 坐标计算器测试失败: {e}")
        return False
    
    # 测试3: 3D特征提取器
    print("\n3️⃣ 测试3D特征提取器...")
    try:
        from detection.features.shape_features import EnhancedShapeFeatureExtractor, FeatureQualityAssessor
        
        # 创建测试数据
        print("📊 创建测试数据...")
        test_mask = create_test_mask()
        test_depth = create_test_depth_data()
        test_waypoint_data = {
            'world_pos': [100, 200, 350],
            'roll': 179, 'pitch': 0, 'yaw': 0
        }
        
        # 初始化特征提取器
        camera_intrinsics = camera_config.get('intrinsics', {})
        shape_extractor = EnhancedShapeFeatureExtractor(camera_intrinsics)
        
        print("🔍 开始提取3D特征...")
        features = shape_extractor.extract_all_features(test_mask, test_depth, test_waypoint_data)
        
        print("✅ 3D特征提取成功!")
        print(f"📈 提取的特征类型: {list(features.keys())}")
        
        # 显示各特征的详细信息
        for feature_type, feature_data in features.items():
            if isinstance(feature_data, dict):
                print(f"  📊 {feature_type}: {len(feature_data)} 个子特征")
                for sub_feature, value in feature_data.items():
                    if isinstance(value, list):
                        print(f"    - {sub_feature}: {len(value)} 维向量")
                    elif isinstance(value, (int, float)):
                        print(f"    - {sub_feature}: {value:.3f}")
                    else:
                        print(f"    - {sub_feature}: {type(value).__name__}")
        
        # 测试特征质量评估
        quality_assessor = FeatureQualityAssessor()
        quality_score = quality_assessor.assess_feature_quality(features)
        print(f"🎯 特征质量评分: {quality_score:.1f}%")
        
    except Exception as e:
        print(f"❌ 3D特征提取器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 测试4: 相似度计算器
    print("\n4️⃣ 测试相似度计算器...")
    try:
        from detection.features.similarity_calculator import FeatureSimilarityCalculator
        
        # 获取相似度配置
        similarity_config = config_manager.get_similarity_config()
        feature_weights = similarity_config.get('feature_weights', {})
        
        similarity_calc = FeatureSimilarityCalculator(feature_weights)
        
        # 创建参考特征和候选特征（稍有不同）
        ref_features = features.copy()
        candidate_features = create_similar_features(features)
        
        print("🔍 计算特征相似度...")
        similarity_result = similarity_calc.calculate_overall_similarity(ref_features, candidate_features)
        
        print("✅ 相似度计算成功!")
        print(f"🎯 总体相似度: {similarity_result['overall_similarity']:.3f}")
        print(f"🎯 匹配置信度: {similarity_result['confidence']:.3f}")
        print(f"📊 各特征相似度:")
        for feature_type, similarity in similarity_result['feature_similarities'].items():
            print(f"  - {feature_type}: {similarity:.3f}")
        
        # 测试最佳匹配查找
        print("\n🔍 测试最佳匹配查找...")
        candidate_list = [
            {'features': candidate_features, 'metadata': {'id': 'candidate_1'}},
            {'features': create_different_features(features), 'metadata': {'id': 'candidate_2'}},
            {'features': features, 'metadata': {'id': 'candidate_3'}}  # 完全匹配
        ]
        
        best_match = similarity_calc.find_best_match(ref_features, candidate_list)
        if best_match:
            match_idx = best_match['candidate_index']
            match_sim = best_match['similarity_result']['overall_similarity']
            print(f"✅ 找到最佳匹配: 候选 {match_idx} (相似度: {match_sim:.3f})")
        else:
            print("❌ 未找到匹配")
        
    except Exception as e:
        print(f"❌ 相似度计算器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 测试5: 自适应学习管理器
    print("\n5️⃣ 测试自适应学习管理器...")
    try:
        from detection.utils.adaptive_learner import AdaptiveThresholdManager
        
        adaptive_config = config_manager.get_adaptive_learning_config()
        learning_file = adaptive_config.get('learning_data_file', 'detection/data/adaptive_learning.json')
        
        adaptive_manager = AdaptiveThresholdManager(learning_file)
        
        # 测试自适应阈值获取
        threshold = adaptive_manager.get_adaptive_threshold('geometric', 'fpfh', 85.0)
        print(f"✅ 自适应阈值计算: geometric.fpfh = {threshold:.3f} (质量85%)")
        
        # 模拟一些学习数据
        print("📚 模拟学习数据...")
        for i in range(10):
            adaptive_manager.update_learning_history(
                feature_type='geometric',
                sub_feature='fpfh',
                similarity=np.random.uniform(0.6, 0.9),
                feature_quality=np.random.uniform(70, 90),
                is_correct_match=np.random.choice([True, False], p=[0.8, 0.2])
            )
        
        # 获取性能报告
        performance_report = adaptive_manager.get_performance_report()
        print("✅ 性能报告生成成功")
        if 'overall_performance' in performance_report:
            total = performance_report['overall_performance']['total_matches']
            success_rate = performance_report['overall_performance']['success_rate']
            print(f"📊 模拟数据: {total} 次匹配, 成功率: {success_rate:.1%}")
        
    except Exception as e:
        print(f"❌ 自适应学习管理器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 测试6: 增强检测管道
    print("\n6️⃣ 测试增强检测管道...")
    try:
        from detection.enhanced_detection_pipeline import EnhancedDetectionPipeline
        
        # 创建输出目录
        test_output_dir = f"test_output_{int(time.time())}"
        os.makedirs(test_output_dir, exist_ok=True)
        
        enhanced_pipeline = EnhancedDetectionPipeline(
            config_file=config_path,
            output_dir=test_output_dir
        )
        
        print("✅ 增强检测管道初始化成功")
        print(f"📁 测试输出目录: {test_output_dir}")
        
        # 测试模拟检测（不需要真实的YOLO和SAM2）
        print("🎯 检测管道组件验证完成")
        
    except Exception as e:
        print(f"❌ 增强检测管道测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print(f"\n{'='*60}")
    print("🎉 所有增强组件测试完成!")
    print("✅ 系统已准备好进行实际检测")
    print("📋 下一步: 运行实际检测测试")
    print(f"{'='*60}")
    
    return True

def create_test_mask():
    """创建测试mask"""
    mask = np.zeros((480, 640), dtype=bool)
    # 创建一个椭圆形的mask
    center = (320, 240)
    axes = (80, 60)
    
    y, x = np.ogrid[:480, :640]
    mask_condition = ((x - center[0])**2 / axes[0]**2 + 
                     (y - center[1])**2 / axes[1]**2) <= 1
    mask[mask_condition] = True
    
    return mask

def create_test_depth_data():
    """创建测试深度数据"""
    depth = np.full((480, 640), 1000, dtype=np.uint16)  # 1米背景
    
    # 在中心区域创建一个凸起的物体
    center = (320, 240)
    for y in range(480):
        for x in range(640):
            dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)
            if dist < 80:  # 物体区域
                # 创建一个高度变化的物体
                height = 50 * np.exp(-(dist/40)**2)  # 高斯分布的高度
                depth[y, x] = max(800, int(1000 - height))  # 最高50mm
    
    return depth

def create_similar_features(original_features):
    """创建相似的特征（用于测试相似度计算）"""
    similar_features = {}
    
    for feature_type, feature_data in original_features.items():
        if isinstance(feature_data, dict):
            similar_features[feature_type] = {}
            for key, value in feature_data.items():
                if isinstance(value, list):
                    # 在原值基础上添加少量噪声
                    noise = np.random.normal(0, 0.1, len(value))
                    similar_features[feature_type][key] = (np.array(value) + noise).tolist()
                elif isinstance(value, (int, float)):
                    # 添加5%的变化
                    similar_features[feature_type][key] = value * (1 + np.random.uniform(-0.05, 0.05))
                else:
                    similar_features[feature_type][key] = value
        else:
            similar_features[feature_type] = feature_data
    
    return similar_features

def create_different_features(original_features):
    """创建不同的特征（用于测试对比）"""
    different_features = {}
    
    for feature_type, feature_data in original_features.items():
        if isinstance(feature_data, dict):
            different_features[feature_type] = {}
            for key, value in feature_data.items():
                if isinstance(value, list):
                    # 创建显著不同的值
                    different_features[feature_type][key] = np.random.uniform(0, 1, len(value)).tolist()
                elif isinstance(value, (int, float)):
                    # 创建显著不同的值
                    different_features[feature_type][key] = value * np.random.uniform(0.5, 2.0)
                else:
                    different_features[feature_type][key] = value
        else:
            different_features[feature_type] = feature_data
    
    return different_features

def trigger_enhanced_detection():
    """触发增强检测"""
    print("\n🚀 触发增强检测测试...")
    
    rclpy.init()
    
    node = Node('enhanced_detection_trigger')
    
    # 创建发布者
    trigger_pub = node.create_publisher(String, '/stitching_complete', 10)
    
    # 等待发布者就绪
    time.sleep(1.0)
    
    # 发送触发消息（使用您提供的扫描目录）
    scan_dir = '/home/qi/ros2_ws/scan_output_20250724_161725'
    
    msg = String()
    msg.data = scan_dir
    
    print(f'📁 使用扫描目录: {scan_dir}')
    
    if os.path.exists(scan_dir):
        print('✅ 扫描目录存在，发送触发消息...')
        trigger_pub.publish(msg)
        print('📤 增强检测触发消息已发送')
        
        # 保持运行，等待检测完成
        print('⏳ 等待检测完成...')
        time.sleep(5.0)
    else:
        print(f'❌ 扫描目录不存在: {scan_dir}')
        print('💡 请确认扫描目录路径是否正确')
    
    rclpy.shutdown()

if __name__ == '__main__':
    print("🔧 增强检测功能测试工具")
    print("=" * 60)
    
    # 首先测试所有组件
    if test_enhanced_components():
        print("\n🎯 组件测试成功！现在可以进行实际检测测试。")
        
        # 询问是否要触发实际检测
        response = input("\n🤔 是否要触发实际的增强检测测试？(y/n): ").strip().lower()
        
        if response in ['y', 'yes', '是']:
            trigger_enhanced_detection()
        else:
            print("✅ 测试完成。您可以稍后使用以下命令触发实际检测:")
            print("python3 trigger_detection.py")
    else:
        print("\n❌ 组件测试失败，请检查错误信息并修复问题。")
        sys.exit(1)