#!/bin/bash

echo "🎯 创建快速测试脚本..."

# 创建测试目录
mkdir -p ~/ros2_ws/src/vision_ai/vision_ai/tracking_system/tests

# 创建快速测试脚本
cat > ~/ros2_ws/src/vision_ai/vision_ai/tracking_system/tests/quick_test.py << 'EOF'
#!/usr/bin/env python3
"""
快速测试脚本 - 验证核心功能
"""

import os
import sys
import numpy as np
import time

# 添加路径
tracking_system_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, tracking_system_path)

print("🚀 Quick Test for Tracking System")
print("=" * 40)

# 测试1: 导入测试
print("\n1️⃣ Testing imports...")
try:
    print("   Importing config...")
    from utils.config import get_config, TrackingState, TrackingMode
    print("   ✅ Config imported")
    
    print("   Importing state machine...")
    from core.state_machine import TrackingStateMachine
    print("   ✅ State machine imported")
    
    print("   Importing data structures...")
    from utils.data_structures import DetectionResult, CameraFrame
    print("   ✅ Data structures imported")
    
    print("   Importing detector...")
    from detection.realtime_detector import RealtimeDetector
    print("   ✅ Detector imported")
    
    print("   Importing ID matcher...")
    from detection.id_matcher import IDMatcher
    print("   ✅ ID matcher imported")
    
    print("   🎉 All imports successful!")
    
except Exception as e:
    print(f"   ❌ Import failed: {e}")
    print("   💡 Check if all files are in place and __init__.py exists")
    sys.exit(1)

# 测试2: 配置测试
print("\n2️⃣ Testing configuration...")
try:
    config = get_config()
    print(f"   ✅ Proportional factor: {config.proportional_factor}")
    print(f"   ✅ Detection frequency: {config.detection_frequency}Hz")
    print(f"   ✅ Max lost frames: {config.max_lost_frames}")
    print(f"   ✅ Target tolerance: {config.target_tolerance_xy}mm")
    
    # 测试配置验证
    if config.validate_config():
        print("   ✅ Configuration validation passed")
    else:
        print("   ❌ Configuration validation failed")
        
except Exception as e:
    print(f"   ❌ Config test failed: {e}")

# 测试3: 状态机测试
print("\n3️⃣ Testing state machine...")
try:
    sm = TrackingStateMachine()
    
    # 测试状态转换
    print("   Testing state transitions...")
    success1 = sm.transition_to(TrackingState.INITIALIZING, "Test start")
    success2 = sm.transition_to(TrackingState.SEARCHING, "Init complete")
    success3 = sm.transition_to(TrackingState.TRACKING, "Target found")
    
    if success1 and success2 and success3:
        print("   ✅ State transitions successful")
        print(f"   ✅ Current state: {sm.current_state.value}")
        print(f"   ✅ State summary: {sm.get_state_summary()}")
    else:
        print("   ❌ State transitions failed")
        
except Exception as e:
    print(f"   ❌ State machine test failed: {e}")

# 测试4: 检测器测试
print("\n4️⃣ Testing detector...")
try:
    detector = RealtimeDetector()
    detector.initialize_models()
    
    # 创建测试图像
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    test_frame = CameraFrame(timestamp=time.time(), color_image=test_image)
    
    # 测试检测
    detections = detector.detect_objects(test_frame)
    print(f"   ✅ Full detection: {len(detections)} objects detected")
    
    # 测试轻量级检测
    lightweight_detections = detector.detect_objects_lightweight(test_frame, 4)
    print(f"   ✅ Lightweight detection: {len(lightweight_detections)} objects")
    
    # 验证检测结果
    for det in detections:
        if det.is_valid():
            print(f"   ✅ Valid detection: {det.object_id} ({det.class_name})")
        else:
            print(f"   ❌ Invalid detection: {det.object_id}")
            
except Exception as e:
    print(f"   ❌ Detector test failed: {e}")

# 测试5: ID匹配测试
print("\n5️⃣ Testing ID matcher...")
try:
    matcher = IDMatcher()
    
    # 设置参考特征
    matcher.reference_features = {
        'lemon_0': {
            'class_id': 4,
            'class_name': 'lemon',
            'features': {
                'shape': {
                    'hu_moments': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                    'area': 10000,
                    'aspect_ratio': 1.2,
                    'circularity': 0.8
                },
                'color': {
                    'histogram': [0.1] * 192,
                    'mean_color': [180, 150, 60]
                }
            }
        }
    }
    
    # 创建测试检测
    test_detection = DetectionResult(
        object_id="test_lemon",
        class_id=4,
        class_name="lemon",
        confidence=0.9,
        bounding_box=[100, 100, 200, 200],
        centroid_2d=(150, 150),
        features={
            'shape': {
                'hu_moments': [1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1],
                'area': 9500,
                'aspect_ratio': 1.15,
                'circularity': 0.82
            },
            'color': {
                'histogram': [0.12] * 192,
                'mean_color': [175, 145, 55]
            }
        },
        timestamp=time.time()
    )
    
    # 测试匹配
    matched_detections = matcher.match_detections([test_detection], ['lemon_0'])
    
    if matched_detections:
        match = matched_detections[0]
        print(f"   ✅ Matched ID: {match.object_id}")
        print(f"   ✅ Match confidence: {match.match_confidence:.3f}")
        print(f"   ✅ Match method: {match.match_method}")
        
        # 测试空间跟踪
        for i in range(6):
            matcher._update_spatial_tracker('lemon_0', (150 + i, 150 + i))
        
        can_lightweight = matcher.can_use_lightweight_tracking('lemon_0', 4, matched_detections)
        print(f"   ✅ Can use lightweight tracking: {can_lightweight}")
        
    else:
        print("   ❌ No matches found")
        
except Exception as e:
    print(f"   ❌ ID matcher test failed: {e}")

print("\n" + "=" * 40)
print("🎊 QUICK TEST COMPLETED!")
print("=" * 40)
print("\n🔥 Next steps:")
print("   1. If all tests passed, you're ready for the main tracking node")
print("   2. If any tests failed, check the error messages above")
print("   3. Run full test suite with: python3 run_all_tests.py")
print("\n💡 Debugging tips:")
print("   - Check file paths and __init__.py files")
print("   - Verify Python path includes tracking_system directory")
print("   - Look for typos in import statements")
EOF

chmod +x ~/ros2_ws/src/vision_ai/vision_ai/tracking_system/tests/quick_test.py

echo "✅ Quick test script created!"
echo ""
echo "🎯 现在可以运行测试了："
echo "   cd ~/ros2_ws"
echo "   python3 src/vision_ai/vision_ai/tracking_system/tests/quick_test.py"