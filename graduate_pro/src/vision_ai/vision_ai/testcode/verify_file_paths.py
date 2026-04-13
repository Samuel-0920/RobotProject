#!/usr/bin/env python3
# verify_file_paths.py - 验证扫描目录中的文件路径

import os
import sys
import glob

def verify_scan_files(scan_output_dir):
    """验证扫描目录中的文件"""
    print(f"🔍 验证扫描目录: {scan_output_dir}")
    
    if not os.path.exists(scan_output_dir):
        print(f"❌ 目录不存在: {scan_output_dir}")
        return False
    
    # 查找彩色图像文件
    color_files = glob.glob(os.path.join(scan_output_dir, "color_waypoint_*.jpg"))
    print(f"🎨 找到彩色文件 ({len(color_files)} 个):")
    for f in sorted(color_files):
        filename = os.path.basename(f)
        print(f"   - {filename}")
    
    # 查找深度文件
    depth_files = glob.glob(os.path.join(scan_output_dir, "depth_raw_waypoint_*.npy"))
    print(f"📏 找到深度文件 ({len(depth_files)} 个):")
    for f in sorted(depth_files):
        filename = os.path.basename(f)
        print(f"   - {filename}")
    
    # 验证配对
    print(f"\n🔗 验证文件配对:")
    waypoint_indices = set()
    
    for color_file in color_files:
        basename = os.path.basename(color_file)
        if 'color_waypoint_' in basename:
            # 提取waypoint编号
            try:
                wp_num = basename.split('color_waypoint_')[1].split('.')[0]
                waypoint_indices.add(wp_num)
                
                # 查找对应的深度文件
                depth_file = os.path.join(scan_output_dir, f"depth_raw_waypoint_{wp_num}.npy")
                has_depth = os.path.exists(depth_file)
                
                print(f"   - Waypoint {wp_num}: 彩色✅ 深度{'✅' if has_depth else '❌'}")
                
            except Exception as e:
                print(f"   - {basename}: 解析失败 - {e}")
    
    print(f"\n📊 统计:")
    print(f"   - 总waypoint数: {len(waypoint_indices)}")
    print(f"   - 彩色文件数: {len(color_files)}")
    print(f"   - 深度文件数: {len(depth_files)}")
    
    # 检查其他重要文件
    print(f"\n📁 其他重要文件:")
    important_files = [
        "fusion_mapping.pkl",
        "fusion_mapping_summary.json", 
        "mapping_test.json"
    ]
    
    for filename in important_files:
        filepath = os.path.join(scan_output_dir, filename)
        exists = os.path.exists(filepath)
        size = os.path.getsize(filepath) if exists else 0
        print(f"   - {filename}: {'✅' if exists else '❌'} ({size} bytes)")
    
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python3 verify_file_paths.py <scan_output_directory>")
        print("例如: python3 verify_file_paths.py scan_output_20250708_163607")
        sys.exit(1)
    
    scan_dir = sys.argv[1]
    verify_scan_files(scan_dir)