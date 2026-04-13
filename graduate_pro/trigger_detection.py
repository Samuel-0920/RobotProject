#!/usr/bin/env python3
# enhanced_trigger_detection.py - 增强版本，添加检测完成信号订阅和测试功能

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import time
import os
import json
from pathlib import Path

class EnhancedDetectionTrigger(Node):
    def __init__(self):
        super().__init__('enhanced_detection_trigger')
        
        # 状态管理
        self.detection_triggered = False
        self.detection_completed = False
        self.current_scan_dir = None
        self.current_user_id = None
        self.detection_complete_data = None
        
        print("🔧 正在初始化ROS接口...")
        
        # 创建发布者
        self.trigger_pub = self.create_publisher(String, '/stitching_complete', 10)
        print("✅ 创建发布者: /stitching_complete")
        
        # 创建订阅者监听检测完成信号
        self.completion_sub = self.create_subscription(
            String, '/detection_complete', self.detection_complete_callback, 10
        )
        print("✅ 创建订阅者: /detection_complete")
        
        # 创建订阅者监听检测结果
        self.result_sub = self.create_subscription(
            String, '/detection_result', self.detection_result_callback, 10
        )
        print("✅ 创建订阅者: /detection_result")
        
        # 🆕 等待订阅者连接建立
        print("⏳ 等待订阅者连接建立...")
        time.sleep(2.0)
        
        # 🆕 检查初始连接状态
        self._check_initial_connections()
        
    def detection_complete_callback(self, msg):
        """检测完成回调 - 解析并测试增强追踪信号"""
        print(f"\n🔔 [DEBUG] 收到detection_complete信号! 消息长度: {len(msg.data)}字符")
        print(f"📝 [DEBUG] 原始消息前200字符: {msg.data[:200]}...")
        
        try:
            data = json.loads(msg.data)
            print(f"✅ [DEBUG] JSON解析成功")
            
            print(f"\n🎉 检测完成!")
            print(f"📊 检测状态: {data.get('status', 'unknown')}")
            print(f"🎯 检测到目标数: {data.get('detection_count', 0)}")
            print(f"📁 扫描目录: {data.get('scan_directory', 'unknown')}")
            
            # 🆕 新增：检查是否是增强检测结果
            enhanced = data.get('enhanced_detection', False)
            print(f"🔍 [DEBUG] enhanced_detection标志: {enhanced}")
            
            if enhanced:
                print("✨ 这是增强检测结果!")
                self._process_enhanced_detection_signal(data)
            else:
                print("⚠️ 这是传统检测结果")
            
            # 检查特征质量统计
            if 'feature_quality_stats' in data:
                stats = data['feature_quality_stats']
                print(f"📈 特征质量统计:")
                print(f"  - 平均质量: {stats.get('mean_quality', 0):.1f}%")
                print(f"  - 质量范围: {stats.get('min_quality', 0):.1f}% - {stats.get('max_quality', 0):.1f}%")
            
            # 检查自适应学习统计
            if 'adaptive_learning_stats' in data:
                adaptive = data['adaptive_learning_stats']
                if adaptive.get('enabled', False):
                    print(f"🧠 自适应学习:")
                    print(f"  - 总匹配次数: {adaptive.get('total_matches', 0)}")
                    print(f"  - 成功率: {adaptive.get('success_rate', 0):.1%}")
                    print(f"  - 最近匹配: {adaptive.get('recent_matches', 0)}")
            
            # 显示文件路径信息
            if 'reference_features_path' in data:
                print(f"📂 参考特征库路径: {data['reference_features_path']}")
            
            if 'enhanced_detection_results_file' in data:
                print(f"📄 增强检测结果文件: {data['enhanced_detection_results_file']}")
            
            if 'reference_library_file' in data:
                print(f"📚 参考库文件: {data['reference_library_file']}")
            
            if 'selected_targets_file' in data:
                print(f"🎯 选择目标文件: {data['selected_targets_file']}")
            
            # 保存数据供后续使用
            self.detection_complete_data = data
            self.detection_completed = True
            print(f"✅ [DEBUG] 设置detection_completed = True")
            
        except json.JSONDecodeError as json_error:
            print(f"❌ JSON解析失败: {json_error}")
            print(f"📝 问题消息: {msg.data}")
            # 即使JSON解析失败，也设置完成标志
            self.detection_completed = True
        except Exception as e:
            print(f"❌ 处理检测完成信号失败: {e}")
            import traceback
            traceback.print_exc()
            # 即使处理失败，也设置完成标志
            self.detection_completed = True
    
    def _process_enhanced_detection_signal(self, data):
        """🆕 处理增强检测信号 - 模拟追踪系统初始化"""
        try:
            print(f"\n🔧 开始处理增强检测信号...")
            
            # 提取扫描目录
            self.current_scan_dir = data.get('scan_directory')
            if not self.current_scan_dir:
                print('❌ 无法获取扫描目录')
                return False
            
            print(f'🔔 收到检测完成信号: {self.current_scan_dir}')
            
            # 提取用户ID
            self.current_user_id = self._extract_user_id(self.current_scan_dir)
            print(f'👤 提取用户ID: {self.current_user_id}')
            
            # 测试加载参考特征库
            success = self._test_load_reference_library(data)
            
            if success:
                print('✅ 增强检测信号处理成功')
                # 🆕 测试追踪目标选择
                self._test_tracking_target_selection(data)
            else:
                print('❌ 增强检测信号处理失败')
                
            return success
            
        except Exception as e:
            print(f'❌ 处理增强检测信号失败: {e}')
            import traceback
            traceback.print_exc()
            return False
    
    def _extract_user_id(self, scan_dir_path: str) -> str:
        """🆕 从扫描目录路径提取用户ID"""
        try:
            # /home/qi/ros2_ws/scan_output_20250728_214915 -> qi
            path_parts = scan_dir_path.split('/')
            for i, part in enumerate(path_parts):
                if part == 'home' and i + 1 < len(path_parts):
                    return path_parts[i + 1]
            
            # 如果提取失败，使用默认值
            return 'default_user'
            
        except Exception as e:
            print(f'❌ 提取用户ID失败: {e}')
            return 'default_user'
    
    def _test_load_reference_library(self, detection_data: dict) -> bool:
        """🆕 测试加载参考特征库"""
        try:
            print(f"\n📚 测试加载参考特征库...")
            
            # 检查参考特征库文件
            reference_library_file = detection_data.get('reference_library_file')
            if not reference_library_file:
                print('❌ 未找到参考特征库文件路径')
                return False
            
            if not os.path.exists(reference_library_file):
                print(f'❌ 参考特征库文件不存在: {reference_library_file}')
                return False
            
            print(f'✅ 参考特征库文件存在: {reference_library_file}')
            
            # 尝试加载并解析文件
            try:
                with open(reference_library_file, 'r', encoding='utf-8') as f:
                    library_data = json.load(f)
                
                print(f'📊 参考特征库统计:')
                print(f'  - 特征库条目数: {len(library_data)}')
                
                # 显示几个示例条目
                sample_count = min(3, len(library_data))
                for i, (obj_id, entry) in enumerate(list(library_data.items())[:sample_count]):
                    metadata = entry.get('metadata', {})
                    quality = entry.get('quality_score', 0)
                    print(f'  - 对象 {i+1}: {obj_id} ({metadata.get("class_name", "unknown")})')
                    print(f'    质量分数: {quality:.1f}%')
                    print(f'    描述: {metadata.get("description", "无描述")}')
                
                if len(library_data) > sample_count:
                    print(f'  - ... 还有 {len(library_data) - sample_count} 个对象')
                
                return True
                
            except json.JSONDecodeError as e:
                print(f'❌ JSON解析失败: {e}')
                return False
            except Exception as e:
                print(f'❌ 文件读取失败: {e}')
                return False
            
        except Exception as e:
            print(f'❌ 测试加载参考特征库失败: {e}')
            return False
    
    def _test_tracking_target_selection(self, detection_data: dict):
        """🆕 测试追踪目标选择"""
        try:
            print(f"\n🎯 测试追踪目标选择...")
            
            # 检查选择目标文件
            selected_targets_file = detection_data.get('selected_targets_file')
            if not selected_targets_file or not os.path.exists(selected_targets_file):
                print(f'⚠️ 选择目标文件不存在，可能用户未选择: {selected_targets_file}')
                return
            
            try:
                with open(selected_targets_file, 'r', encoding='utf-8') as f:
                    selection_content = f.read().strip()
                
                if selection_content:
                    print(f'✅ 找到用户选择的追踪目标: {selection_content}')
                    
                    # 验证目标是否在参考库中
                    reference_library_file = detection_data.get('reference_library_file')
                    if reference_library_file and os.path.exists(reference_library_file):
                        with open(reference_library_file, 'r', encoding='utf-8') as f:
                            library_data = json.load(f)
                        
                        if selection_content in library_data:
                            target_info = library_data[selection_content]
                            metadata = target_info.get('metadata', {})
                            quality = target_info.get('quality_score', 0)
                            
                            print(f'🔍 目标详细信息:')
                            print(f'  - 目标ID: {selection_content}')
                            print(f'  - 类别: {metadata.get("class_name", "unknown")}')
                            print(f'  - 描述: {metadata.get("description", "无描述")}')
                            print(f'  - 质量分数: {quality:.1f}%')
                            print(f'  - 检测置信度: {metadata.get("confidence", 0):.3f}')
                            
                            # 检查特征信息
                            features = target_info.get('features', {})
                            if 'spatial' in features:
                                spatial = features['spatial']
                                if 'world_coordinates' in spatial:
                                    coords = spatial['world_coordinates']
                                    print(f'  - 世界坐标: ({coords[0]:.1f}, {coords[1]:.1f}, {coords[2]:.1f})mm')
                                
                                if 'gripper_width_info' in spatial:
                                    gripper = spatial['gripper_width_info']
                                    print(f'  - 推荐抓夹宽度: {gripper.get("recommended_gripper_width", 0)}mm')
                            
                            print('✅ 追踪目标验证成功')
                        else:
                            print(f'❌ 选择的目标不在参考库中: {selection_content}')
                    else:
                        print('⚠️ 无法验证目标，参考库文件问题')
                else:
                    print('⚠️ 选择目标文件为空')
                    
            except Exception as e:
                print(f'❌ 读取选择目标文件失败: {e}')
                
        except Exception as e:
            print(f'❌ 测试追踪目标选择失败: {e}')
    
    def detection_result_callback(self, msg):
        """检测结果回调"""
        try:
            data = json.loads(msg.data)
            
            print(f"\n📋 检测结果详情:")
            print(f"🔍 处理时间: {data.get('processing_time', 0):.2f}秒")
            print(f"📊 检测到 {data.get('detection_count', 0)} 个对象:")
            
            for i, obj in enumerate(data.get('objects', []), 1):
                obj_id = obj.get('object_id', 'unknown')
                class_name = obj.get('class_name', 'unknown')
                confidence = obj.get('confidence', 0)
                quality = obj.get('quality_score', 0)
                
                print(f"  {i}. {obj_id} ({class_name})")
                print(f"     - 检测置信度: {confidence:.3f}")
                
                if quality > 0:
                    print(f"     - 特征质量: {quality:.1f}%")
                
                # 显示3D坐标信息
                if 'features' in obj and 'spatial' in obj['features']:
                    spatial = obj['features']['spatial']
                    if 'world_coordinates' in spatial:
                        coords = spatial['world_coordinates']
                        print(f"     - 世界坐标: ({coords[0]:.1f}, {coords[1]:.1f}, {coords[2]:.1f})mm")
                
                # 显示抓夹信息
                if 'features' in obj and 'grasp_info' in obj['features']:
                    grasp = obj['features']['grasp_info']
                    width = grasp.get('recommended_width', 0)
                    print(f"     - 推荐抓夹宽度: {width}mm")
            
            # 检查是否使用了增强特征
            if data.get('enhanced_features', False):
                print("✨ 使用了增强3D点云特征")
            
        except Exception as e:
            print(f"❌ 处理检测结果失败: {e}")
    
    def trigger_detection(self, scan_dir):
        """触发检测"""
        print(f"🚀 准备触发增强检测...")
        print(f"📁 扫描目录: {scan_dir}")
        
        # 检查扫描目录
        if not os.path.exists(scan_dir):
            print(f"❌ 扫描目录不存在: {scan_dir}")
            return False
        
        # 检查是否有必要的文件
        required_files = []
        for file in os.listdir(scan_dir):
            if file.startswith('final_') and file.endswith('.jpg'):
                required_files.append(f"融合图像: {file}")
            elif file.startswith('color_waypoint_') and file.endswith('.jpg'):
                required_files.append(f"waypoint图像: {file}")
        
        if required_files:
            print("✅ 找到扫描文件:")
            for file in required_files[:3]:  # 只显示前3个
                print(f"  - {file}")
            if len(required_files) > 3:
                print(f"  - ... 还有 {len(required_files)-3} 个文件")
        else:
            print("⚠️ 未找到预期的图像文件，但继续尝试...")
        
        # 🆕 再次检查连接状态
        print("🔍 触发前检查连接状态...")
        publishers_info = self.get_publishers_info_by_topic('/detection_complete')
        print(f"📊 /detection_complete 发布者数: {len(publishers_info)}")
        
        if len(publishers_info) == 0:
            print("⚠️ 警告: 没有发布者连接到 /detection_complete")
            print("💡 请确保 detection_node 正在运行")
        
        # 等待发布者就绪
        print("⏳ 等待ROS发布者就绪...")
        time.sleep(2.0)  # 增加等待时间
        
        # 🆕 重置完成标志
        self.detection_completed = False
        print(f"🔄 [DEBUG] 重置 detection_completed = {self.detection_completed}")
        
        # 发送触发消息
        msg = String()
        msg.data = scan_dir
        
        print("📤 发送增强检测触发消息...")
        self.trigger_pub.publish(msg)
        
        self.detection_triggered = True
        print("✅ 触发消息已发送")
        
        # 🆕 触发后短暂等待，让系统处理
        print("⏳ 等待系统处理触发消息...")
        time.sleep(1.0)
        
        return True
    
    def wait_for_completion(self, timeout=30):
        """等待检测完成"""
        print(f"⏳ 等待检测处理和完成 (超时: {timeout}秒)...")
        print(f"🔍 [DEBUG] 初始状态: detection_completed = {self.detection_completed}")
        print(f"💡 [INFO] 检测处理通常需要5-30秒，请耐心等待...")
        
        start_time = time.time()
        rate = self.create_rate(2)  # 降低到2Hz，减少CPU占用
        last_progress_time = 0
        detection_started = False
        
        while rclpy.ok() and not self.detection_completed:
            current_time = time.time()
            elapsed = current_time - start_time
            print(f"⏳ 检测处理中... (已等待 {elapsed:.0f}秒")
            if elapsed > timeout:
                print(f"⏰ 等待超时 ({timeout}秒)")
                print(f"🔍 [DEBUG] 最终状态: detection_completed = {self.detection_completed}")
                print(f"💡 建议: 检查detection_node是否正在处理，或增加超时时间")
                return False
            
            # 显示进度 (每10秒显示一次，给检测更多时间)
            remaining = timeout - elapsed
            current_10s = int(elapsed // 10)
            if current_10s != last_progress_time:
                if elapsed < 30:
                    print(f"⏳ 检测处理中... (已等待 {elapsed:.0f}秒, 剩余 {remaining:.0f}秒)")
                    if not detection_started and elapsed > 5:
                        print(f"🔍 检测可能正在进行特征提取和分析...")
                        detection_started = True
                else:
                    print(f"⏳ 长时间检测中... (已等待 {elapsed:.0f}秒, 剩余 {remaining:.0f}秒)")
                    
                last_progress_time = current_10s
            
            # 🆕 添加话题状态检查 (每30秒检查一次)
            if int(elapsed) == 30:
                print(f"🔍 [DEBUG] 30秒检查: 验证ROS话题状态...")
                self._check_topic_connections()
            
            # 🆕 让ROS处理消息，使用更长的超时
            rclpy.spin_once(self, timeout_sec=0.5)
            rate.sleep()
        
        print(f"✅ [DEBUG] 检测完成! detection_completed = {self.detection_completed}")
        return self.detection_completed
    
    def get_detection_summary(self):
        """🆕 获取检测摘要信息"""
        if not self.detection_complete_data:
            return "未收到检测完成信号"
        
        data = self.detection_complete_data
        summary = []
        
        summary.append("🎉 检测完成摘要")
        summary.append("=" * 50)
        summary.append(f"📁 扫描目录: {data.get('scan_directory', 'unknown')}")
        summary.append(f"👤 用户ID: {self.current_user_id or 'unknown'}")
        summary.append(f"🎯 检测目标数: {data.get('detection_count', 0)}")
        summary.append(f"✨ 增强检测: {'是' if data.get('enhanced_detection', False) else '否'}")
        
        if 'feature_quality_stats' in data:
            stats = data['feature_quality_stats']
            summary.append(f"📈 平均特征质量: {stats.get('mean_quality', 0):.1f}%")
        
        if 'adaptive_learning_stats' in data:
            adaptive = data['adaptive_learning_stats']
            if adaptive.get('enabled', False):
                summary.append(f"🧠 自适应学习: 启用 (成功率: {adaptive.get('success_rate', 0):.1%})")
            else:
                summary.append("🧠 自适应学习: 禁用")
        
        # 文件状态检查
        files_to_check = [
            ('参考特征库', 'reference_library_file'),
            ('检测结果', 'enhanced_detection_results_file'),
            ('选择目标', 'selected_targets_file')
        ]
        
        summary.append("\n📄 文件状态:")
        for file_name, key in files_to_check:
            file_path = data.get(key)
            if file_path and os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                summary.append(f"  ✅ {file_name}: {file_size} bytes")
            else:
                summary.append(f"  ❌ {file_name}: 不存在")
        
        return "\n".join(summary)
    
    def batch_test_with_metadata(self, test_scenarios):
        results = []
        for scenario in test_scenarios:
            result = self.run_single_test(scenario)
            results.append({**result, **scenario['metadata']})
        return self._generate_charts(results)

    def _check_topic_connections(self):
        """🆕 检查话题连接状态"""
        try:
            # 检查订阅者连接
            topic_names_and_types = self.get_topic_names_and_types()
            detection_complete_found = False
            
            for topic_name, topic_types in topic_names_and_types:
                if topic_name == '/detection_complete':
                    detection_complete_found = True
                    print(f"📡 [DEBUG] 找到话题 /detection_complete, 类型: {topic_types}")
                    break
            
            if not detection_complete_found:
                print(f"❌ [DEBUG] 未找到话题 /detection_complete")
            
            # 检查发布者数量
            publishers_info = self.get_publishers_info_by_topic('/detection_complete')
            subscriber_info = self.get_subscriptions_info_by_topic('/detection_complete')
            
            print(f"📊 [DEBUG] /detection_complete 话题状态:")
            print(f"  - 发布者数量: {len(publishers_info)}")
            print(f"  - 订阅者数量: {len(subscriber_info)}")
            
            if len(publishers_info) == 0:
                print(f"⚠️ [DEBUG] 没有发布者连接到 /detection_complete")
                print(f"💡 [DEBUG] 请检查 detection_node 是否正在运行")
            
        except Exception as e:
            print(f"❌ [DEBUG] 检查话题连接失败: {e}")
    
    def _check_initial_connections(self):
        """🆕 检查初始连接状态"""
        try:
            print("🔍 检查初始ROS连接状态...")
            
            # 检查话题是否存在
            topic_names_and_types = self.get_topic_names_and_types()
            topics = [name for name, _ in topic_names_and_types]
            
            important_topics = ['/detection_complete', '/detection_result', '/stitching_complete']
            for topic in important_topics:
                if topic in topics:
                    print(f"✅ 话题存在: {topic}")
                else:
                    print(f"❌ 话题不存在: {topic}")
            
            # 检查订阅者状态
            completion_subs = self.get_subscriptions_info_by_topic('/detection_complete')
            print(f"📊 /detection_complete 订阅者数: {len(completion_subs)}")
            
            result_subs = self.get_subscriptions_info_by_topic('/detection_result')
            print(f"📊 /detection_result 订阅者数: {len(result_subs)}")
            
        except Exception as e:
            print(f"❌ 检查初始连接失败: {e}")

def main():
    print("🔧 增强检测触发工具 v2.0")
    print("🆕 新增检测完成信号处理和测试功能")
    print("=" * 50)
    
    # 获取扫描目录
    default_scan_dir = '/home/qi/ros2_ws/scan_output_20250903_211959'
    
    print(f"📁 默认扫描目录: {default_scan_dir}")
    scan_dir = input(f"请输入扫描目录路径 (回车使用默认): ").strip()
    
    if not scan_dir:
        scan_dir = default_scan_dir
    
    # 初始化ROS
    rclpy.init()
    
    try:
        # 创建触发节点
        trigger_node = EnhancedDetectionTrigger()
        
        # 触发检测
        if trigger_node.trigger_detection(scan_dir):
            # 🆕 增加更长的等待时间，给检测充足的处理时间
            success = trigger_node.wait_for_completion(timeout=120)  # 从60秒增加到120秒
            
            if success:
                print("\n🎉 增强检测测试完成!")
                
                # 🆕 显示详细摘要
                print("\n" + trigger_node.get_detection_summary())
                
                print("\n📋 请检查生成的文件:")
                print("  - enhanced_detection_results.json")
                print("  - reference_library.json")
                print("  - detection_visualization.jpg")
                print("  - tracking_selection.txt (如果用户有选择)")
                
                # 🆕 询问是否进行追踪测试
                if trigger_node.detection_complete_data and trigger_node.detection_complete_data.get('enhanced_detection', False):
                    test_tracking = input("\n🎯 是否进行追踪目标测试? (y/n): ").strip().lower()
                    if test_tracking == 'y':
                        print("💡 追踪测试功能可以在这里实现...")
                        print("💡 当前已完成检测完成信号的解析和验证")
            else:
                print("\n⏰ 检测超时或失败")
                print("💡 请检查detection_node是否正在运行")
                print("💡 可以手动运行: ros2 run vision_ai detection_node")
        else:
            print("❌ 触发检测失败")
    
    except KeyboardInterrupt:
        print("\n🛑 用户中断")
    except Exception as e:
        print(f"\n❌ 运行出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()