#!/usr/bin/env python3
"""
Enhanced Scan Planning Algorithm Tester
Tests geometric optimization algorithms with diverse shapes and rotations
Output: CSV performance table with optimization analysis
"""

import rclpy
from rclpy.node import Node
import time
import csv
import math
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.patches as patches

from vision_ai_interfaces.srv import PlanScan
from geometry_msgs.msg import Point

class EnhancedScanTester(Node):
    def __init__(self):
        super().__init__('enhanced_scan_tester')
        
        # Create service client for planning only
        self.plan_client = self.create_client(PlanScan, 'plan_scan')
        
        # Wait for planning service
        while not self.plan_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for scan planning service...')
        
        self.get_logger().info('Enhanced scan algorithm tester started')
        
        # Test results storage
        self.test_results = []

    def get_enhanced_test_cases(self):
        """Define 9 diverse test cases optimized for visualization"""
        return [
            # Row 1: Basic rectangles with different sizes
            {
                "name": "Small_Rectangle",
                "coordinates": [(150, -150), (150, 100), (350, 100), (350, -150)],
                "object_height": 50.0,
                "expected_waypoints": 1,
                "scan_height": 400.0
            },
            {
                "name": "Medium_Rectangle", 
                "coordinates": [(-250, -300), (-250, 200), (250, 200), (250, -300)],
                "object_height": 50.0,
                "expected_waypoints": 6,
                "scan_height": 430.0
            },
            {
                "name": "High_Object_Test",
                "coordinates": [(100, 100), (100, 300), (300, 300), (300, 100)],
                "object_height": 120.0,  # High object
                "expected_waypoints": 1,
                "scan_height": 500.0
            },
            
            # Row 2: Rotated rectangles  
            {
                "name": "Rotated_30deg",
                "coordinates": [(273, -87), (187, 87), (87, 37), (173, -137)],
                "object_height": 50.0,
                "expected_waypoints": 4,
                "scan_height": 430.0
            },
            {
                "name": "Rotated_45deg", 
                "coordinates": [(250, -141), (141, 0), (250, 141), (359, 0)],
                "object_height": 50.0,
                "expected_waypoints": 4,
                "scan_height": 430.0
            },
            {
                "name": "Rotated_60deg",
                "coordinates": [(300, -173), (127, -27), (200, 146), (373, 0)],
                "object_height": 50.0,
                "expected_waypoints": 4, 
                "scan_height": 430.0
            },
            
            # Row 3: Irregular shapes and extreme aspect ratios
            {
                "name": "Trapezoid",
                "coordinates": [(100, -150), (350, -100), (300, 200), (50, 150)],
                "object_height": 50.0,
                "expected_waypoints": 4,
                "scan_height": 430.0
            },
            {
                "name": "Elongated_Strip_8to1",
                "coordinates": [(50, -15), (650, -15), (650, 15), (50, 15)],
                "object_height": 50.0,
                "expected_waypoints": 10,
                "scan_height": 430.0
            },
            {
                "name": "Irregular_Parallelogram",
                "coordinates": [(150, -100), (450, 0), (400, 250), (100, 150)],
                "object_height": 50.0,
                "expected_waypoints": 6,
                "scan_height": 430.0
            }
        ]

    def generate_rotated_rectangles(self, angle_deg):
        """Generate rectangles at specified rotation angle"""
        rectangles = [
            {"name": f"Small_Rect_{angle_deg}deg", "width": 200, "height": 150, "center": (250, 0)},
            {"name": f"Medium_Rect_{angle_deg}deg", "width": 400, "height": 250, "center": (300, -200)},
            {"name": f"Large_Rect_{angle_deg}deg", "width": 500, "height": 400, "center": (200, 100)}
        ]
        
        test_cases = []
        angle_rad = math.radians(angle_deg)
        
        for rect in rectangles:
            # Calculate rectangle corners
            w, h = rect["width"] / 2, rect["height"] / 2
            cx, cy = rect["center"]
            
            # Corner points before rotation
            corners = [(-w, -h), (w, -h), (w, h), (-w, h)]
            
            # Apply rotation and translation
            rotated_corners = []
            for x, y in corners:
                rx = x * math.cos(angle_rad) - y * math.sin(angle_rad) + cx
                ry = x * math.sin(angle_rad) + y * math.cos(angle_rad) + cy
                rotated_corners.append((rx, ry))
            
            test_cases.append({
                "name": rect["name"],
                "coordinates": rotated_corners,
                "object_height": 50.0,
                "expected_rotation": angle_deg,
                "shape_type": "rotated_rectangle"
            })
        
        return test_cases

    def generate_irregular_shapes(self):
        """Generate irregular quadrilaterals"""
        return [
            {
                "name": "Trapezoid_1",
                "coordinates": [(100, -100), (300, -80), (250, 150), (80, 120)],
                "object_height": 50.0,
                "shape_type": "trapezoid"
            },
            {
                "name": "Parallelogram_1", 
                "coordinates": [(200, 0), (450, 50), (400, 300), (150, 250)],
                "object_height": 50.0,
                "shape_type": "parallelogram"
            },
            {
                "name": "Irregular_Quad_1",
                "coordinates": [(150, -200), (400, -150), (350, 100), (100, 50)],
                "object_height": 50.0,
                "shape_type": "irregular"
            },
            {
                "name": "L_Shape_Approx",
                "coordinates": [(100, 100), (300, 120), (280, 400), (120, 380)],
                "object_height": 50.0,
                "shape_type": "l_shape"
            }
        ]

    def generate_elongated_shapes(self):
        """Generate shapes with high aspect ratios"""
        return [
            {
                "name": "Thin_Strip_2to1",
                "coordinates": [(200, -50), (600, -50), (600, 50), (200, 50)],
                "object_height": 50.0,
                "shape_type": "elongated_2_1"
            },
            {
                "name": "Thin_Strip_4to1", 
                "coordinates": [(100, -25), (700, -25), (700, 25), (100, 25)],
                "object_height": 50.0,
                "shape_type": "elongated_4_1"
            },
            {
                "name": "Thin_Strip_6to1",
                "coordinates": [(50, -20), (750, -20), (750, 20), (50, 20)],
                "object_height": 50.0,
                "shape_type": "elongated_6_1"
            }
        ]

    def calculate_theoretical_bounding_box(self, coordinates):
        """Calculate minimum bounding box dimensions and rotation"""
        points = np.array(coordinates)
        
        best_area = float('inf')
        best_angle = 0
        best_dims = (0, 0)
        
        # Test different rotation angles
        angles = np.linspace(0, np.pi, 180)
        
        for angle in angles:
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            rotated_points = points @ rotation_matrix.T
            
            x_min, x_max = np.min(rotated_points[:, 0]), np.max(rotated_points[:, 0])
            y_min, y_max = np.min(rotated_points[:, 1]), np.max(rotated_points[:, 1])
            
            width = x_max - x_min
            height = y_max - y_min
            area = width * height
            
            if area < best_area:
                best_area = area
                best_angle = angle
                best_dims = (max(width, height), min(width, height))  # long, short
        
        return best_dims[0], best_dims[1], math.degrees(best_angle)

    def estimate_unoptimized_waypoints(self, length, width, object_height):
        """Estimate waypoints without rotation optimization"""
        # Camera FOV parameters (from scan_planner_node.py)
        fx, fy = 912.694580078125, 910.309814453125
        fov_h = 2 * math.atan(1280 / (2 * fx))
        fov_v = 2 * math.atan(720 / (2 * fy))
        
        # Assume fixed scan height of 430mm for multi-point
        scan_height = 430
        effective_height = scan_height - object_height
        
        # Coverage at scan height
        coverage_h = 2 * effective_height * math.tan(fov_h / 2)
        coverage_v = 2 * effective_height * math.tan(fov_v / 2)
        
        # Single point check
        if length <= coverage_h and width <= coverage_v:
            return 1
        
        # Multi-point without optimization (assume 25% overlap)
        overlap_ratio = 0.25
        step_h = coverage_h * (1 - overlap_ratio)
        step_v = coverage_v * (1 - overlap_ratio)
        
        grid_x = max(1, int(math.ceil(length / step_h)))
        grid_y = max(1, int(math.ceil(width / step_v)))
        
        return grid_x * grid_y

    def calculate_coverage_rate(self, scan_plan, theoretical_area):
        """Calculate theoretical coverage rate"""
        if not scan_plan.waypoints:
            return 0.0
        
        # Get scan parameters
        scan_height = scan_plan.scan_height
        object_height = scan_plan.object_height
        effective_height = scan_height - object_height
        
        # Camera FOV
        fx, fy = 912.694580078125, 910.309814453125
        fov_h = 2 * math.atan(1280 / (2 * fx))
        fov_v = 2 * math.atan(720 / (2 * fy))
        
        # FOV coverage at scan height
        fov_coverage_area = (2 * effective_height * math.tan(fov_h / 2)) * \
                           (2 * effective_height * math.tan(fov_v / 2))
        
        if len(scan_plan.waypoints) == 1:
            covered_area = fov_coverage_area
        else:
            # Multi-point with overlap consideration
            overlap_ratio = scan_plan.overlap_ratio
            effective_coverage_per_point = fov_coverage_area * (1 - overlap_ratio)
            covered_area = len(scan_plan.waypoints) * effective_coverage_per_point
        
        coverage_rate = min(covered_area / theoretical_area * 100, 100.0)
        return coverage_rate

    def run_enhanced_test_suite(self):
        """Run enhanced test suite with geometric analysis"""
        self.get_logger().info('Starting Enhanced Scan Algorithm Test Suite')
        self.get_logger().info('=' * 60)
        
        test_cases = self.get_enhanced_test_cases()
        results = []
        test_results_with_plans = []  # Store test cases with their scan plans for visualization
        
        for i, test_case in enumerate(test_cases):
            self.get_logger().info(f'Testing {i+1}/{len(test_cases)}: {test_case["name"]}')
            
            try:
                # Calculate theoretical bounding box
                length, width, rotation_deg = self.calculate_theoretical_bounding_box(test_case["coordinates"])
                theoretical_area = length * width
                
                # Estimate unoptimized waypoints
                waypoints_before = self.estimate_unoptimized_waypoints(length, width, test_case["object_height"])
                
                # Call actual planning service
                start_time = time.time()
                planning_result = self.call_planning_service(test_case)
                calculation_time = (time.time() - start_time) * 1000
                
                if planning_result["success"]:
                    scan_plan = planning_result["scan_plan"]
                    waypoints_after = len(scan_plan.waypoints)
                    coverage_rate = self.calculate_coverage_rate(scan_plan, theoretical_area)
                    
                    result = {
                        "Region_Name": test_case["name"],
                        "Length_mm": round(length, 1),
                        "Width_mm": round(width, 1),
                        "Rotation_deg": round(rotation_deg, 1),
                        "Waypoints_Before_Optimization": waypoints_before,
                        "Waypoints_After_Optimization": waypoints_after,
                        "Coverage_Rate_percent": round(coverage_rate, 1)
                    }
                    
                    results.append(result)
                    
                    # Store test case with actual scan plan for visualization
                    test_results_with_plans.append({
                        'test_case': test_case,
                        'scan_plan': scan_plan,
                        'result': result
                    })
                    
                    self.get_logger().info(f'  Success: {waypoints_before} -> {waypoints_after} waypoints')
                    self.get_logger().info(f'  Optimization: {((waypoints_before - waypoints_after) / waypoints_before * 100):.1f}% reduction')
                    self.get_logger().info(f'  Coverage: {coverage_rate:.1f}%')
                else:
                    self.get_logger().error(f'  Planning failed: {planning_result.get("error", "Unknown error")}')
                
            except Exception as e:
                self.get_logger().error(f'  Test failed: {str(e)}')
                continue
        
        return results, test_results_with_plans

    def call_planning_service(self, test_case):
        """Call the planning service for a test case"""
        try:
            request = PlanScan.Request()
            request.mode = "manual"
            request.object_height = test_case["object_height"]
            
            for x, y in test_case["coordinates"]:
                point = Point()
                point.x = float(x)
                point.y = float(y)
                point.z = 0.0
                request.points.append(point)
            
            future = self.plan_client.call_async(request)
            rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
            
            if future.result() is None:
                return {"success": False, "error": "Service call timeout"}
            
            response = future.result()
            if response.success:
                return {"success": True, "scan_plan": response.scan_plan}
            else:
                return {"success": False, "error": response.message}
                
        except Exception as e:
            return {"success": False, "error": str(e)}

    def save_results_to_csv(self, results):
        """Save results to CSV file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"enhanced_scan_results_{timestamp}.csv"
        
        fieldnames = [
            "Region_Name", "Length_mm", "Width_mm", "Rotation_deg",
            "Waypoints_Before_Optimization", "Waypoints_After_Optimization",
            "Coverage_Rate_percent"
        ]
        
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        
        self.get_logger().info(f'Results saved to: {filename}')
        return filename

    def generate_optimization_analysis(self, results):
        """Generate optimization analysis"""
        if not results:
            return
        
        total_cases = len(results)
        optimization_cases = sum(1 for r in results if r['Waypoints_Before_Optimization'] > r['Waypoints_After_Optimization'])
        avg_coverage = sum(r['Coverage_Rate_percent'] for r in results) / total_cases
        high_coverage_cases = sum(1 for r in results if r['Coverage_Rate_percent'] >= 95)
        
        # Calculate optimization efficiency
        total_savings = sum(max(0, r['Waypoints_Before_Optimization'] - r['Waypoints_After_Optimization']) for r in results)
        total_before = sum(r['Waypoints_Before_Optimization'] for r in results)
        overall_optimization = (total_savings / total_before * 100) if total_before > 0 else 0
        
        print("\nOptimization Analysis Summary:")
        print("=" * 50)
        print(f"Total test cases: {total_cases}")
        print(f"Cases with optimization: {optimization_cases} ({optimization_cases/total_cases*100:.1f}%)")
        print(f"Overall waypoint reduction: {overall_optimization:.1f}%")
        print(f"Average coverage rate: {avg_coverage:.1f}%")
        print(f"Cases with >95% coverage: {high_coverage_cases} ({high_coverage_cases/total_cases*100:.1f}%)")
        
        # Shape type analysis
        shape_analysis = {}
        for result in results:
            shape_type = result['Region_Name'].split('_')[0] if '_' in result['Region_Name'] else 'Unknown'
            if shape_type not in shape_analysis:
                shape_analysis[shape_type] = {'count': 0, 'avg_optimization': 0, 'avg_coverage': 0}
            
            shape_analysis[shape_type]['count'] += 1
            optimization_rate = max(0, (result['Waypoints_Before_Optimization'] - result['Waypoints_After_Optimization']) / result['Waypoints_Before_Optimization'] * 100)
            shape_analysis[shape_type]['avg_optimization'] += optimization_rate
            shape_analysis[shape_type]['avg_coverage'] += result['Coverage_Rate_percent']
        
        print(f"\nShape Type Analysis:")
        for shape_type, data in shape_analysis.items():
            if data['count'] > 0:
                avg_opt = data['avg_optimization'] / data['count']
                avg_cov = data['avg_coverage'] / data['count']
                print(f"  {shape_type}: {data['count']} cases, {avg_opt:.1f}% optimization, {avg_cov:.1f}% coverage")

    def generate_integrated_visualization(self, results, test_results_with_plans):
        """Generate integrated 3x3 visualization grids using actual scan plans"""
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle, Polygon
        
        # Select 9 representative cases
        selected_cases = test_results_with_plans[:9] if len(test_results_with_plans) >= 9 else test_results_with_plans
        
        # Generate shapes analysis grid
        self.plot_shapes_grid(selected_cases)
        
        # Generate coverage plans grid
        self.plot_coverage_grid(selected_cases)
        
        print(f"\nVisualization grids generated using {len(selected_cases)} test cases")

    def plot_shapes_grid(self, test_cases):
        """Plot 3x3 grid showing original shapes with bounding box analysis"""
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle, Polygon
        
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        fig.suptitle('Scan Region Shapes - Geometric Analysis', fontsize=26, fontweight='bold')
        
        for i, case_data in enumerate(test_cases[:9]):
            if i >= 9:
                break
                
            row, col = i // 3, i % 3
            ax = axes[row, col]
            
            test_case = case_data['test_case']
            coordinates = test_case["coordinates"]
            
            # Set up coordinate system (following GUI style)
            ax.set_xlim(-700, 700)
            ax.set_ylim(700, -700)  # Inverted Y
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3, which='major')
            ax.set_xticks(np.arange(-600, 700, 100))  # 添加具体的刻度
            ax.set_yticks(np.arange(-600, 700, 100))
            
            # Draw workspace elements
            ax.plot(0, 0, 'ko', markersize=4)
            
            # Obstacle zone
            obstacle = Circle((0, 0), 150, fill=False, color='red', linewidth=1, 
                             linestyle='--', alpha=0.6)
            ax.add_patch(obstacle)
            
            # Workspace boundary  
            workspace = Circle((0, 0), 700, fill=False, color='gray', linewidth=1,
                              linestyle='--', alpha=0.6)
            ax.add_patch(workspace)
            
            # Convert coordinates for plotting (x,y) -> (y,x) following GUI convention
            plot_points = [(p[1], p[0]) for p in coordinates]
            
            # Draw scanning region
            region_polygon = Polygon(plot_points, fill=True, alpha=0.3,
                                   color='lightblue', edgecolor='blue', linewidth=2)
            ax.add_patch(region_polygon)
            
            # Calculate and draw minimum bounding box
            length, width, rotation, bbox_corners = self.calculate_min_bounding_box(coordinates)
            
            if bbox_corners:
                bbox_plot_points = [(p[1], p[0]) for p in bbox_corners]
                bbox_polygon = Polygon(bbox_plot_points, fill=False,
                                     color='red', linewidth=1, linestyle='--', alpha=0.8)
                ax.add_patch(bbox_polygon)
            
            # Add shape info
            area = self.calculate_polygon_area(coordinates)
            info_text = f"{test_case['name']}\n{length:.0f}×{width:.0f}mm\nArea: {area/100:.1f}cm²"
            if test_case.get('object_height', 50.0) != 50.0:
                info_text += f"\nHeight: {test_case['object_height']:.0f}mm"
                
            ax.text(0.98, 0.98, info_text, transform=ax.transAxes,
                   verticalalignment='top', horizontalalignment='right',fontsize=18,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        
        plt.tight_layout()
        plt.savefig('scan_shapes_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Shape analysis grid saved as 'scan_shapes_analysis.png'")

    def plot_coverage_grid(self, test_cases):
        """Plot 3x3 grid showing actual scan coverage plans"""
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle, Polygon
        import numpy as np
        
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        fig.subplots_adjust(wspace=0, hspace=0)  # 添加这行
        fig.suptitle('Scan Coverage Plans - Waypoint Distribution and FOV Coverage', fontsize=26, fontweight='bold')
        
        for i, case_data in enumerate(test_cases[:9]):
            if i >= 9:
                break
                
            row, col = i // 3, i % 3
            ax = axes[row, col]
            
            test_case = case_data['test_case']
            scan_plan = case_data['scan_plan']
            coordinates = test_case["coordinates"]
            
            # Set up coordinate system
            ax.set_xlim(-700, 700)
            ax.set_ylim(700, -700)  # Inverted Y
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3, which='major')
            ax.set_xticks(np.arange(-600, 700, 100))  # 添加具体的刻度
            ax.set_yticks(np.arange(-600, 700, 100))
            ax.tick_params(axis='both', which='major', labelsize=8)
            # Draw workspace elements
            ax.plot(0, 0, 'ko', markersize=4)
            
            # Obstacle zone
            obstacle = Circle((0, 0), 150, fill=False, color='red', linewidth=1,
                             linestyle='--', alpha=0.6)
            ax.add_patch(obstacle)
            
            # Workspace boundary
            workspace = Circle((0, 0), 700, fill=False, color='gray', linewidth=1,
                              linestyle='--', alpha=0.6)
            ax.add_patch(workspace)
            
            # Draw original scanning region
            plot_points = [(p[1], p[0]) for p in coordinates]
            region_polygon = Polygon(plot_points, fill=True, alpha=0.2,
                                   color='lightblue', edgecolor='blue', linewidth=2)
            ax.add_patch(region_polygon)
            
            # Draw waypoints and FOV coverage using actual scan plan
            if scan_plan and scan_plan.waypoints:
                # Calculate FOV coverage size from actual scan plan
                effective_height = scan_plan.scan_height - scan_plan.object_height
                fov_h_deg = 70.1  # From GUI code
                fov_v_deg = 43.2
                fov_h_rad = math.radians(fov_h_deg / 2)
                fov_v_rad = math.radians(fov_v_deg / 2)
                
                fov_1280_size = 2 * effective_height * math.tan(fov_h_rad)
                fov_720_size = 2 * effective_height * math.tan(fov_v_rad)
                
                # Generate colors for waypoints
                colors = plt.cm.viridis(np.linspace(0, 1, len(scan_plan.waypoints)))
                
                for j, waypoint in enumerate(scan_plan.waypoints):
                    wp_x = waypoint.pose.position.x
                    wp_y = waypoint.pose.position.y
                    qz = waypoint.pose.orientation.z
                    qw = waypoint.pose.orientation.w
                    yaw_rad = -2 * math.atan2(qz, qw)
                    
                    # Convert coordinates for plotting
                    plot_x = wp_y
                    plot_y = wp_x
                    
                    # Plot waypoint
                    ax.plot(plot_x, plot_y, 'ro', markersize=8,
                           markeredgecolor='white', markeredgewidth=1)
                    
                    # Draw FOV coverage rectangle
                    half_1280 = fov_1280_size / 2
                    half_720 = fov_720_size / 2
                    camera_corners = np.array([
                        [-half_1280, -half_720],
                        [+half_1280, -half_720],
                        [+half_1280, +half_720],
                        [-half_1280, +half_720]
                    ])
                    
                    cos_yaw = math.cos(yaw_rad)
                    sin_yaw = math.sin(yaw_rad)
                    rotation_matrix = np.array([
                        [cos_yaw, -sin_yaw],
                        [sin_yaw, cos_yaw]
                    ])
                    rotated_corners = camera_corners @ rotation_matrix.T
                    world_corners = rotated_corners + np.array([plot_x, plot_y])
                    
                    fov_polygon = Polygon(world_corners, alpha=0.25,
                                        facecolor=colors[j], edgecolor='black', linewidth=0.5)
                    ax.add_patch(fov_polygon)
                    
                    # Label waypoint
                    ax.annotate(f'{j+1}', (plot_x, plot_y), xytext=(2, 2),
                               textcoords='offset points', fontsize=18, color='black',
                               weight='bold', ha='center', va='center')
                
                # Draw scan path for multi-point
                if len(scan_plan.waypoints) > 1:
                    path_x = [wp.pose.position.y for wp in scan_plan.waypoints]
                    path_y = [wp.pose.position.x for wp in scan_plan.waypoints]
                    ax.plot(path_x, path_y, 'r--', linewidth=1, alpha=0.7)
            
            # Coverage info using actual results
            strategy = scan_plan.strategy if scan_plan else "Unknown"
            waypoint_count = len(scan_plan.waypoints) if scan_plan and scan_plan.waypoints else 0
            scan_height = scan_plan.scan_height if scan_plan else 0
            
            info_text = f"{test_case['name']}\n{strategy.replace('_', ' ').title()}\n{waypoint_count} waypoints\nHeight: {scan_height:.0f}mm"
            
            ax.text(0.98, 0.98, info_text, transform=ax.transAxes,
                   verticalalignment='top', horizontalalignment='right',fontsize=18,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('scan_coverage_plans.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Coverage plans grid saved as 'scan_coverage_plans.png'")

    def calculate_min_bounding_box(self, coordinates):
        """Calculate minimum bounding box for a shape - FIXED VERSION"""
        points = np.array(coordinates)
        
        best_area = float('inf')
        best_angle = 0
        best_dims = (0, 0)
        best_corners = None
        
        # Test different rotation angles
        angles = np.linspace(0, np.pi, 180)  # More angles for better precision
        
        for angle in angles:
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            rotated_points = points @ rotation_matrix.T
            
            x_min, x_max = np.min(rotated_points[:, 0]), np.max(rotated_points[:, 0])
            y_min, y_max = np.min(rotated_points[:, 1]), np.max(rotated_points[:, 1])
            
            width = x_max - x_min
            height = y_max - y_min
            area = width * height
            
            if area < best_area:
                best_area = area
                best_angle = angle
                best_dims = (width, height)
                
                # Calculate bounding box corners correctly
                center = np.mean(points, axis=0)
                w, h = width/2, height/2
                
                # Create box corners in the rotated coordinate system
                box_corners_rotated = np.array([
                    [x_min, y_min],
                    [x_max, y_min], 
                    [x_max, y_max],
                    [x_min, y_max]
                ])
                
                # Transform back to original coordinate system using CORRECT inverse rotation
                inverse_rotation = np.array([[cos_a, sin_a], [-sin_a, cos_a]])
                box_corners = box_corners_rotated @ inverse_rotation.T
                
                best_corners = [(corner[0], corner[1]) for corner in box_corners]
        
        return best_dims[0], best_dims[1], math.degrees(best_angle), best_corners

    def calculate_polygon_area(self, coordinates):
        """Calculate polygon area using shoelace formula"""
        if len(coordinates) < 3:
            return 0
        
        x = [coord[0] for coord in coordinates]
        y = [coord[1] for coord in coordinates]
        
        area = 0.5 * abs(sum(x[i]*y[i+1] - x[i+1]*y[i] for i in range(-1, len(x)-1)))
        return area

    def visualize_test_case(self, test_case, scan_plan=None):
        """Visualize a single test case with optimization results"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Original shape and bounding box
        coords = test_case["coordinates"]
        length, width, rotation = self.calculate_theoretical_bounding_box(coords)
        
        # Original shape
        shape = Polygon(coords, fill=False, edgecolor='blue', linewidth=2, label='Original Shape')
        ax1.add_patch(shape)
        
        # Bounding box (approximate visualization)
        center = np.mean(coords, axis=0)
        angle_rad = math.radians(rotation)
        
        # Create bounding box corners
        w, h = length/2, width/2
        box_corners = [(-w, -h), (w, -h), (w, h), (-w, h)]
        rotated_box = []
        for x, y in box_corners:
            rx = x * math.cos(angle_rad) - y * math.sin(angle_rad) + center[0]
            ry = x * math.sin(angle_rad) + y * math.cos(angle_rad) + center[1]
            rotated_box.append((rx, ry))
        
        bbox = Polygon(rotated_box, fill=False, edgecolor='red', linewidth=2, linestyle='--', label='Min Bounding Box')
        ax1.add_patch(bbox)
        
        ax1.set_xlim(-200, 800)
        ax1.set_ylim(-400, 400)
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_title(f'{test_case["name"]}\nBounding Box: {length:.0f}×{width:.0f}mm, Rot: {rotation:.1f}°')
        
        # Plot 2: Waypoint distribution (if available)
        if scan_plan and scan_plan.waypoints:
            waypoint_x = [wp.pose.position.y for wp in scan_plan.waypoints]  # Note: swap for visualization
            waypoint_y = [wp.pose.position.x for wp in scan_plan.waypoints]
            
            ax2.scatter(waypoint_x, waypoint_y, c='red', s=100, marker='x', label='Waypoints')
            
            # Draw original shape
            shape2 = Polygon([(y, x) for x, y in coords], fill=False, edgecolor='blue', linewidth=2)
            ax2.add_patch(shape2)
            
            ax2.set_xlim(-400, 400)
            ax2.set_ylim(-200, 800)
            ax2.set_aspect('equal')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            ax2.set_title(f'Optimized Waypoints: {len(scan_plan.waypoints)}')
        else:
            ax2.text(0.5, 0.5, 'No waypoint data', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Waypoint Distribution')
        
        plt.tight_layout()
        plt.show()


def main():
    """Main execution function"""
    rclpy.init()
    
    try:
        tester = EnhancedScanTester()
        
        # Run enhanced test suite (now returns both results and test plans)
        results, test_results_with_plans = tester.run_enhanced_test_suite()
        
        if results:
            # Save to CSV
            csv_filename = tester.save_results_to_csv(results)
            
            # Generate analysis
            tester.generate_optimization_analysis(results)
            
            # Generate integrated visualization using actual scan plans
            tester.generate_integrated_visualization(results, test_results_with_plans)
            
            print(f"\nTesting completed successfully!")
            print(f"Results saved to: {csv_filename}")
            print("Visualization grids generated:")
            print("  - scan_shapes_analysis.png (geometric analysis)")
            print("  - scan_coverage_plans.png (actual waypoints and FOV coverage)")
            print("Algorithm validation: Geometric optimization and coverage analysis completed")
        else:
            print("No valid test results obtained")
            
    except KeyboardInterrupt:
        print("\nTesting interrupted by user")
    except Exception as e:
        print(f"Testing failed: {str(e)}")
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()