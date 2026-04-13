#!/usr/bin/env python3
"""
使用历史数据测试拼接算法
"""

import cv2
import numpy as np
import json
import os
import glob
from datetime import datetime

class HistoricalDataTester:
    """使用历史数据测试拼接"""
    
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.camera_width = 1280
        self.camera_height = 720
        
    def load_historical_data(self):
        """加载历史数据"""
        print(f"📁 加载历史数据: {self.data_dir}")
        
        # 查找图像文件
        color_files = sorted(glob.glob(os.path.join(self.data_dir, "color_waypoint_*.jpg")))
        depth_files = sorted(glob.glob(os.path.join(self.data_dir, "depth_raw_waypoint_*.npy")))
        
        print(f"找到 {len(color_files)} 个颜色图像")
        print(f"找到 {len(depth_files)} 个深度文件")
        
        if not color_files:
            print("❌ 未找到颜色图像文件")
            return None
        
        # 尝试加载waypoint信息
        info_file = os.path.join(self.data_dir, "waypoint_info.json")
        waypoint_info = None
        
        if os.path.exists(info_file):
            with open(info_file, 'r') as f:
                waypoint_info = json.load(f)
            print(f"✅ 加载了waypoint信息: {len(waypoint_info.get('waypoints', []))} 个waypoint")
        else:
            print("⚠️ 未找到waypoint_info.json，将使用默认配置")
            waypoint_info = self.create_default_waypoint_info(len(color_files))
        
        # 加载图像数据
        loaded_data = []
        for i, color_file in enumerate(color_files):
            try:
                # 加载颜色图像
                image = cv2.imread(color_file)
                if image is None:
                    print(f"❌ 无法加载图像: {color_file}")
                    continue
                
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # 获取对应的waypoint信息
                wp_info = None
                if waypoint_info and i < len(waypoint_info.get('waypoints', [])):
                    wp_info = waypoint_info['waypoints'][i]
                else:
                    # 创建默认waypoint信息
                    wp_info = self.create_default_waypoint(i)
                
                data_item = {
                    'color_image': image_rgb,
                    'waypoint_index': i,
                    'filename': os.path.basename(color_file),
                    'world_pos': tuple(wp_info.get('world_pos', [0, 0, 430])),
                    'yaw': wp_info.get('yaw', 0),
                    'grid_info': wp_info.get('grid_info'),
                    'relative_info': wp_info.get('relative_info'),
                    'overlap_info': wp_info.get('overlap_info', {'overlap_x': 0.25, 'overlap_y': 0.25})
                }
                
                loaded_data.append(data_item)
                print(f"✅ 加载 WP{i}: {os.path.basename(color_file)}")
                
            except Exception as e:
                print(f"❌ 加载WP{i}失败: {e}")
                continue
        
        return loaded_data, waypoint_info
    
    def create_default_waypoint_info(self, num_waypoints):
        """创建默认的waypoint信息"""
        print(f"🔧 创建默认waypoint配置: {num_waypoints} 个waypoint")
        
        waypoints = []
        
        # 简单的网格布局
        if num_waypoints <= 4:
            grid_x, grid_y = 2, 2
        elif num_waypoints <= 6:
            grid_x, grid_y = 3, 2
        elif num_waypoints <= 9:
            grid_x, grid_y = 3, 3
        else:
            grid_x = int(np.ceil(np.sqrt(num_waypoints)))
            grid_y = int(np.ceil(num_waypoints / grid_x))
        
        for i in range(num_waypoints):
            # 计算网格位置
            gx = i % grid_x
            gy = i // grid_x
            
            # 估算世界坐标（简单布局）
            world_x = gx * 200 - (grid_x - 1) * 100
            world_y = gy * 200 - (grid_y - 1) * 100
            
            # 计算邻居
            neighbors = []
            if gx > 0: neighbors.append("left")
            if gx < grid_x - 1: neighbors.append("right")
            if gy > 0: neighbors.append("top")
            if gy < grid_y - 1: neighbors.append("bottom")
            
            waypoint = {
                "index": i,
                "world_pos": [world_x, world_y, 430.0],
                "yaw": 0.0,
                "grid_info": {"grid_x": gx, "grid_y": gy, "total_x": grid_x, "total_y": grid_y},
                "relative_info": {"camera_x": world_x, "camera_y": world_y, "neighbors": neighbors},
                "overlap_info": {"overlap_x": 0.25, "overlap_y": 0.25}
            }
            waypoints.append(waypoint)
        
        return {
            "waypoints": waypoints,
            "scan_plan": {
                "strategy": "multi_point",
                "scan_height": 430.0,
                "object_height": 50.0,
                "overlap_ratio": 0.25
            }
        }
    
    def create_default_waypoint(self, index):
        """创建单个默认waypoint"""
        return {
            "index": index,
            "world_pos": [0, 0, 430],
            "yaw": 0,
            "grid_info": {"grid_x": 0, "grid_y": 0, "total_x": 1, "total_y": 1},
            "relative_info": {"camera_x": 0, "camera_y": 0, "neighbors": []},
            "overlap_info": {"overlap_x": 0.25, "overlap_y": 0.25}
        }
    
    def test_with_historical_data(self, output_dir="historical_test_output"):
        """使用历史数据测试拼接"""
        print("\n=== 使用历史数据测试拼接 ===")
        
        # 加载数据
        loaded_data, waypoint_info = self.load_historical_data()
        if not loaded_data:
            print("❌ 无法加载历史数据")
            return None
        
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # 提取planning信息
            planning_info = self.extract_planning_from_data(loaded_data)
            
            if not planning_info:
                print("❌ 无法提取planning信息")
                return None
            
            # 计算拼接参数
            stitch_params = self.calculate_historical_stitch_params(planning_info)
            
            # 执行拼接
            result_image = self.stitch_historical_data(loaded_data, planning_info, stitch_params)
            
            if result_image is not None:
                # 保存结果
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                result_path = os.path.join(output_dir, f"historical_stitch_{timestamp}.jpg")
                result_bgr = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(result_path, result_bgr)
                
                # 保存测试报告
                report = {
                    'test_time': timestamp,
                    'data_source': self.data_dir,
                    'input_images': len(loaded_data),
                    'result_size': result_image.shape,
                    'waypoint_info': waypoint_info,
                    'planning_info': {
                        'grid_layout': planning_info['grid_layout'],
                        'relationships_count': len(planning_info['relationships'])
                    }
                }
                
                report_path = os.path.join(output_dir, f"test_report_{timestamp}.json")
                with open(report_path, 'w') as f:
                    json.dump(report, f, indent=2)
                
                print(f"✅ 历史数据拼接完成!")
                print(f"📸 结果图像: {result_path}")
                print(f"📊 测试报告: {report_path}")
                print(f"🔍 结果尺寸: {result_image.shape}")
                
                return result_path
            else:
                print("❌ 历史数据拼接失败")
                return None
                
        except Exception as e:
            print(f"❌ 历史数据测试异常: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def extract_planning_from_data(self, loaded_data):
        """从历史数据中提取planning信息"""
        try:
            relationships = {}
            overlap_map = {}
            grid_layout = {}
            
            print('📊 从历史数据提取planning信息...')
            
            # 提取grid布局信息
            if loaded_data and loaded_data[0]['grid_info']:
                first_grid = loaded_data[0]['grid_info']
                grid_layout = {
                    'total_x': first_grid['total_x'],
                    'total_y': first_grid['total_y'],
                    'is_regular_grid': True,
                    'num_waypoints': len(loaded_data)
                }
                print(f'📊 检测到Grid布局: {grid_layout["total_x"]}×{grid_layout["total_y"]} = {grid_layout["num_waypoints"]} waypoints')
            
            for img_data in loaded_data:
                wp_idx = img_data['waypoint_index']
                
                # 提取邻居关系
                if img_data['relative_info']:
                    rel_info = img_data['relative_info']
                    relationships[wp_idx] = {
                        'neighbors': rel_info['neighbors'],
                        'camera_pos': (rel_info['camera_x'], rel_info['camera_y'])
                    }
                    print(f'📋 WP{wp_idx} 邻居: {rel_info["neighbors"]}, Camera: ({rel_info["camera_x"]:.1f}, {rel_info["camera_y"]:.1f})')
                
                # 提取重叠度信息
                if img_data['overlap_info']:
                    overlap_info = img_data['overlap_info']
                    overlap_map[wp_idx] = {
                        'overlap_x': overlap_info['overlap_x'],
                        'overlap_y': overlap_info['overlap_y']
                    }
                
                # 提取grid位置
                if img_data['grid_info']:
                    grid_info = img_data['grid_info']
                    if wp_idx in relationships:
                        relationships[wp_idx]['grid_pos'] = (grid_info['grid_x'], grid_info['grid_y'])
            
            # 计算camera坐标边界
            camera_bounds = self.calculate_camera_bounds(relationships)
            
            planning_info = {
                'relationships': relationships,
                'overlap_map': overlap_map,
                'grid_layout': grid_layout,
                'source': 'historical_data',
                'scan_bounds': camera_bounds
            }
            
            print(f'✅ Planning信息提取完成: {len(relationships)} waypoints')
            return planning_info
            
        except Exception as e:
            print(f'❌ Planning信息提取失败: {e}')
            return None
    
    def calculate_camera_bounds(self, relationships):
        """计算camera坐标边界"""
        camera_x_coords = []
        camera_y_coords = []
        
        for wp_idx, rel_info in relationships.items():
            camera_x, camera_y = rel_info['camera_pos']
            camera_x_coords.append(camera_x)
            camera_y_coords.append(camera_y)
        
        bounds = {
            'camera_x_min': min(camera_x_coords),
            'camera_x_max': max(camera_x_coords),
            'camera_y_min': min(camera_y_coords),
            'camera_y_max': max(camera_y_coords)
        }
        
        print(f'📏 Camera边界: X[{bounds["camera_x_min"]:.1f}, {bounds["camera_x_max"]:.1f}], Y[{bounds["camera_y_min"]:.1f}, {bounds["camera_y_max"]:.1f}]')
        return bounds
    
    def calculate_historical_stitch_params(self, planning_info):
        """计算历史数据的拼接参数"""
        camera_bounds = planning_info['scan_bounds']
        
        # 计算canvas尺寸
        camera_range_x = camera_bounds['camera_x_max'] - camera_bounds['camera_x_min']
        camera_range_y = camera_bounds['camera_y_max'] - camera_bounds['camera_y_min']
        
        # 添加图像尺寸的缓冲
        total_width_needed = max(camera_range_y + self.camera_width, 2000)
        total_height_needed = max(camera_range_x + self.camera_height, 1500)
        
        # 添加安全边距
        safety_margin = 1.5
        canvas_width = int(total_width_needed * safety_margin)
        canvas_height = int(total_height_needed * safety_margin)
        
        return {
            'canvas_size': (canvas_width, canvas_height),
            'camera_bounds': camera_bounds,
            'overlap_ratio': 0.25,
            'scale_factor': 1.0  # 可调整的缩放因子
        }
    
    def stitch_historical_data(self, loaded_data, planning_info, stitch_params):
        """拼接历史数据"""
        try:
            canvas_width, canvas_height = stitch_params['canvas_size']
            relationships = planning_info['relationships']
            camera_bounds = stitch_params['camera_bounds']
            scale_factor = stitch_params.get('scale_factor', 1.0)
            
            print(f'🔧 开始拼接历史数据')
            print(f'📐 Canvas尺寸: {canvas_width}×{canvas_height}')
            
            # 创建画布
            canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.float32)
            weight_map = np.zeros((canvas_height, canvas_width), dtype=np.float32)
            
            canvas_center_x = canvas_width // 2
            canvas_center_y = canvas_height // 2
            
            # 计算camera坐标系的中心
            camera_center_x = (camera_bounds['camera_x_min'] + camera_bounds['camera_x_max']) / 2
            camera_center_y = (camera_bounds['camera_y_min'] + camera_bounds['camera_y_max']) / 2
            
            print(f'🎯 Canvas中心: ({canvas_center_x}, {canvas_center_y})')
            print(f'📍 Camera中心: ({camera_center_x:.1f}, {camera_center_y:.1f})')
            
            for img_data in loaded_data:
                image = img_data['color_image'].astype(np.float32)
                waypoint_idx = img_data['waypoint_index']
                
                if waypoint_idx not in relationships:
                    print(f'⚠️ WP{waypoint_idx} 没有关系信息，跳过')
                    continue
                
                rel_info = relationships[waypoint_idx]
                camera_x, camera_y = rel_info['camera_pos']
                neighbors = rel_info['neighbors']
                
                # 计算相对于camera中心的偏移
                offset_x = camera_x - camera_center_x
                offset_y = camera_y - camera_center_y
                
                # 映射到canvas坐标（根据实际效果调整映射方式）
                canvas_pos_x = canvas_center_x + int(offset_y * scale_factor)
                canvas_pos_y = canvas_center_y + int(offset_x * scale_factor)
                
                print(f'🎯 WP{waypoint_idx}: Camera({camera_x:.1f}, {camera_y:.1f}) -> Canvas({canvas_pos_x}, {canvas_pos_y})')
                
                # 边界检查
                canvas_pos_x = max(image.shape[1]//2, min(canvas_width - image.shape[1]//2, canvas_pos_x))
                canvas_pos_y = max(image.shape[0]//2, min(canvas_height - image.shape[0]//2, canvas_pos_y))
                
                # 创建蒙版
                mask = self.create_neighbor_mask(image.shape[:2], neighbors)
                
                # 执行混合
                self.blend_to_canvas_historical(canvas, weight_map, image, mask, canvas_pos_x, canvas_pos_y)
                
                print(f'✅ WP{waypoint_idx} 拼接完成')
            
            # 归一化并返回结果
            result = self.normalize_canvas_historical(canvas, weight_map)
            
            print(f'🎉 历史数据拼接完成! 结果尺寸: {result.shape}')
            return result
            
        except Exception as e:
            print(f'❌ 历史数据拼接失败: {e}')
            import traceback
            traceback.print_exc()
            return None
    
    def create_neighbor_mask(self, image_shape, neighbors):
        """根据邻居关系创建蒙版"""
        h, w = image_shape
        mask = np.ones((h, w), dtype=np.float32)
        
        blend_size = 60  # 混合区域大小
        
        # 根据邻居方向应用羽化
        if 'left' in neighbors:
            for i in range(blend_size):
                weight = i / blend_size
                if i < w:
                    mask[:, i] *= weight
        
        if 'right' in neighbors:
            for i in range(blend_size):
                weight = i / blend_size
                if i < w:
                    mask[:, -(i+1)] *= weight
        
        if 'top' in neighbors:
            for i in range(blend_size):
                weight = i / blend_size
                if i < h:
                    mask[i, :] *= weight
        
        if 'bottom' in neighbors:
            for i in range(blend_size):
                weight = i / blend_size
                if i < h:
                    mask[-(i+1), :] *= weight
        
        return mask
    
    def blend_to_canvas_historical(self, canvas, weight_map, image, mask, pos_x, pos_y):
        """历史数据的画布混合"""
        try:
            img_h, img_w = image.shape[:2]
            canvas_h, canvas_w = canvas.shape[:2]
            
            # 计算放置区域
            img_left = pos_x - img_w // 2
            img_right = pos_x + img_w // 2
            img_top = pos_y - img_h // 2
            img_bottom = pos_y + img_h // 2
            
            # Canvas有效区域
            canvas_left = max(0, img_left)
            canvas_right = min(canvas_w, img_right)
            canvas_top = max(0, img_top)
            canvas_bottom = min(canvas_h, img_bottom)
            
            # 图像对应区域
            img_crop_left = max(0, -img_left)
            img_crop_right = img_crop_left + (canvas_right - canvas_left)
            img_crop_top = max(0, -img_top)
            img_crop_bottom = img_crop_top + (canvas_bottom - canvas_top)
            
            if (canvas_right > canvas_left and canvas_bottom > canvas_top and
                img_crop_right > img_crop_left and img_crop_bottom > img_crop_top):
                
                # 提取区域
                img_region = image[img_crop_top:img_crop_bottom, img_crop_left:img_crop_right]
                mask_region = mask[img_crop_top:img_crop_bottom, img_crop_left:img_crop_right]
                
                if img_region.shape[:2] == mask_region.shape:
                    # 加权混合
                    for c in range(3):
                        canvas[canvas_top:canvas_bottom, canvas_left:canvas_right, c] += img_region[:, :, c] * mask_region
                    
                    weight_map[canvas_top:canvas_bottom, canvas_left:canvas_right] += mask_region
                        
        except Exception as e:
            print(f'❌ 画布混合失败: {e}')
    
    def normalize_canvas_historical(self, canvas, weight_map):
        """归一化canvas"""
        result = np.zeros_like(canvas, dtype=np.uint8)
        valid_mask = weight_map > 0
        
        for c in range(3):
            channel = canvas[:, :, c].copy()
            channel[valid_mask] /= weight_map[valid_mask]
            result[:, :, c] = np.clip(channel, 0, 255).astype(np.uint8)
        
        return result

def main():
    """主函数 - 使用历史数据测试"""
    print("📊 历史数据拼接测试")
    
    # 指定你的数据目录
    data_dir = input("请输入历史数据目录路径 (或按Enter使用当前目录): ").strip()
    if not data_dir:
        data_dir = "."
    
    if not os.path.exists(data_dir):
        print(f"❌ 目录不存在: {data_dir}")
        return
    
    # 创建测试器
    tester = HistoricalDataTester(data_dir)
    
    # 执行测试
    result_path = tester.test_with_historical_data()
    
    if result_path:
        print(f"\n🎉 测试完成!")
        print(f"📸 查看结果: {result_path}")
        print(f"\n💡 调试建议:")
        print(f"1. 检查waypoint排列是否正确")
        print(f"2. 观察重叠区域的混合效果")
        print(f"3. 如果位置不对，可以调整scale_factor参数")
        print(f"4. 如果没有waypoint_info.json，会使用默认配置")
    else:
        print(f"\n❌ 测试失败")
        print(f"💡 检查建议:")
        print(f"1. 确保目录中有color_waypoint_*.jpg文件")
        print(f"2. 检查图像文件是否能正常加载")
        print(f"3. 查看控制台错误信息")

if __name__ == "__main__":
    main()