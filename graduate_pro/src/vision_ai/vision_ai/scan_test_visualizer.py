#!/usr/bin/env python3
"""
Scan System Nine-Grid Visualization Generator
Creates two 3x3 grid images showing:
1. Original scan regions with different shapes
2. Corresponding scan coverage plans with waypoints and FOV
"""
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle, Polygon
import numpy as np
import math

class ScanVisualizationGenerator:
    def __init__(self):
        # Camera parameters from GUI code
        self.fov_h_deg = 70.1  # Horizontal FOV 
        self.fov_v_deg = 43.2  # Vertical FOV
        self.fov_h_rad = math.radians(self.fov_h_deg / 2)
        self.fov_v_rad = math.radians(self.fov_v_deg / 2)
        
        # Define 9 test cases with diverse shapes
        self.test_cases = self.get_nine_test_cases()
        
    def get_nine_test_cases(self):
        """Define 9 diverse test cases for visualization"""
        return [
            {
                "name": "Small Rectangle",
                "coordinates": [(150, -150), (150, 100), (350, 100), (350, -150)],
                "object_height": 50.0,
                "expected_waypoints": 1,
                "scan_height": 400.0
            },
            {
                "name": "Rotated Rectangle (30°)",
                "coordinates": [(273, -87), (187, 87), (87, 37), (173, -137)],
                "object_height": 50.0,
                "expected_waypoints": 4,
                "scan_height": 430.0
            },
            {
                "name": "Large Rectangle",
                "coordinates": [(-250, -300), (-250, 200), (250, 200), (250, -300)],
                "object_height": 50.0,
                "expected_waypoints": 6,
                "scan_height": 430.0
            },
            {
                "name": "Trapezoid",
                "coordinates": [(100, -100), (300, -80), (250, 150), (80, 120)],
                "object_height": 50.0,
                "expected_waypoints": 4,
                "scan_height": 430.0
            },
            {
                "name": "High Object (100mm)",
                "coordinates": [(100, 100), (100, 300), (300, 300), (300, 100)],
                "object_height": 100.0,
                "expected_waypoints": 1,
                "scan_height": 500.0
            },
            {
                "name": "Elongated Strip (6:1)",
                "coordinates": [(50, -20), (650, -20), (650, 20), (50, 20)],
                "object_height": 50.0,
                "expected_waypoints": 8,
                "scan_height": 430.0
            },
            {
                "name": "Parallelogram",
                "coordinates": [(200, 0), (450, 50), (400, 300), (150, 250)],
                "object_height": 50.0,
                "expected_waypoints": 4,
                "scan_height": 430.0
            },
            {
                "name": "Irregular Quadrilateral",
                "coordinates": [(150, -200), (400, -150), (350, 100), (100, 50)],
                "object_height": 50.0,
                "expected_waypoints": 4,
                "scan_height": 430.0
            },
            {
                "name": "Rotated Rectangle (45°)",
                "coordinates": [(250, -141), (141, 0), (250, 141), (359, 0)],
                "object_height": 50.0,
                "expected_waypoints": 4,
                "scan_height": 430.0
            }
        ]
    
    def calculate_min_bounding_box(self, coordinates):
        """Calculate minimum bounding box for a shape"""
        points = np.array(coordinates)
        
        best_area = float('inf')
        best_angle = 0
        best_dims = (0, 0)
        best_corners = None
        
        # Test different rotation angles
        angles = np.linspace(0, np.pi, 90)
        
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
                
                # Calculate bounding box corners
                center = np.mean(points, axis=0)
                w, h = width/2, height/2
                box_corners_local = [(-w, -h), (w, -h), (w, h), (-w, h)]
                
                # Rotate back to original coordinate system
                inverse_matrix = np.array([[cos_a, sin_a], [-sin_a, cos_a]])
                box_corners = []
                for x, y in box_corners_local:
                    rotated = np.array([x, y]) @ inverse_matrix
                    box_corners.append((rotated[0] + center[0], rotated[1] + center[1]))
                
                best_corners = box_corners
        
        return best_dims[0], best_dims[1], math.degrees(best_angle), best_corners
    
    def generate_waypoints(self, test_case):
        """Generate waypoints for a test case using simplified logic"""
        coordinates = test_case["coordinates"]
        object_height = test_case["object_height"]
        scan_height = test_case["scan_height"]
        
        if test_case["expected_waypoints"] == 1:
            # Single point at centroid
            points = np.array(coordinates)
            centroid = np.mean(points, axis=0)
            return [(centroid[0], centroid[1], 0)]
        else:
            # Multi-point grid pattern
            points = np.array(coordinates)
            centroid = np.mean(points, axis=0)
            
            # Estimate grid dimensions
            length, width, rotation_deg, _ = self.calculate_min_bounding_box(coordinates)
            
            # Calculate effective FOV coverage
            effective_height = scan_height - object_height
            fov_coverage_h = 2 * effective_height * math.tan(self.fov_h_rad)
            fov_coverage_v = 2 * effective_height * math.tan(self.fov_v_rad)
            
            # Grid calculation with overlap
            overlap_ratio = 0.25
            step_h = fov_coverage_h * (1 - overlap_ratio)
            step_v = fov_coverage_v * (1 - overlap_ratio)
            
            grid_x = max(1, int(math.ceil(length / step_h)))
            grid_y = max(1, int(math.ceil(width / step_v)))
            
            # Generate grid waypoints
            waypoints = []
            x_start = -(grid_x - 1) * step_h / 2 if grid_x > 1 else 0
            y_start = -(grid_y - 1) * step_v / 2 if grid_y > 1 else 0
            
            for i in range(grid_x):
                for j in range(grid_y):
                    x_offset = x_start + i * step_h
                    y_offset = y_start + j * step_v
                    
                    # Add rotation if significant
                    yaw_rad = math.radians(rotation_deg)
                    world_x = centroid[0] + x_offset * math.cos(yaw_rad) - y_offset * math.sin(yaw_rad)
                    world_y = centroid[1] + x_offset * math.sin(yaw_rad) + y_offset * math.cos(yaw_rad)
                    
                    waypoints.append((world_x, world_y, rotation_deg))
            
            return waypoints
    
    def plot_single_region(self, ax, test_case, show_coverage=False):
        """Plot a single scanning region with optional coverage visualization"""
        # Set up coordinate system (following GUI style)
        ax.set_xlim(-700, 700)
        ax.set_ylim(700, -700)  # Inverted Y
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)
        
        # Draw workspace elements
        ax.plot(0, 0, 'ko', markersize=4)
        
        # Obstacle zone
        obstacle = Circle((0, 0), 150, fill=False, color='red', linewidth=1, 
                         linestyle='--', alpha=0.6)
        ax.add_patch(obstacle)
        
        # Workspace boundary
        workspace = Circle((0, 0), 700, fill=False, color='gray', linewidth=0.5, 
                          linestyle=':', alpha=0.4)
        ax.add_patch(workspace)
        
        coordinates = test_case["coordinates"]
        
        # Convert coordinates for plotting (x,y) -> (y,x) following GUI convention
        plot_points = [(p[1], p[0]) for p in coordinates]
        
        # Draw scanning region
        region_polygon = Polygon(plot_points, fill=True, alpha=0.3, 
                               color='lightblue', edgecolor='blue', linewidth=2)
        ax.add_patch(region_polygon)
        
        if not show_coverage:
            # Show original shape with bounding box
            length, width, rotation, bbox_corners = self.calculate_min_bounding_box(coordinates)
            
            # Draw minimum bounding box
            if bbox_corners:
                bbox_plot_points = [(p[1], p[0]) for p in bbox_corners]
                bbox_polygon = Polygon(bbox_plot_points, fill=False, 
                                     color='red', linewidth=1, linestyle='--', alpha=0.8)
                ax.add_patch(bbox_polygon)
            
            # Add shape info
            area = self.calculate_polygon_area(coordinates)
            info_text = f"{test_case['name']}\n{length:.0f}×{width:.0f}mm\nArea: {area/100:.1f}cm²"
            if test_case['object_height'] != 50.0:
                info_text += f"\nHeight: {test_case['object_height']:.0f}mm"
                
        else:
            # Show coverage with waypoints and FOV
            waypoints = self.generate_waypoints(test_case)
            
            # Calculate FOV coverage size
            effective_height = test_case["scan_height"] - test_case["object_height"]
            fov_1280_size = 2 * effective_height * math.tan(self.fov_h_rad)
            fov_720_size = 2 * effective_height * math.tan(self.fov_v_rad)
            
            # Draw waypoints and FOV coverage
            colors = plt.cm.viridis(np.linspace(0, 1, len(waypoints)))
            
            for i, (wp_x, wp_y, yaw) in enumerate(waypoints):
                # Plot waypoint (convert coordinates)
                plot_x, plot_y = wp_y, wp_x
                ax.plot(plot_x, plot_y, 'ro', markersize=8, 
                       markeredgecolor='white', markeredgewidth=1)
                
                # Draw FOV coverage rectangle
                yaw_rad = math.radians(yaw)
                half_1280 = fov_1280_size / 2
                half_720 = fov_720_size / 2
                
                # FOV corners in camera frame
                fov_corners = np.array([
                    [-half_1280, -half_720],
                    [+half_1280, -half_720], 
                    [+half_1280, +half_720],
                    [-half_1280, +half_720]
                ])
                
                # Rotate FOV coverage
                cos_yaw, sin_yaw = math.cos(yaw_rad), math.sin(yaw_rad)
                rotation_matrix = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
                rotated_corners = fov_corners @ rotation_matrix.T
                world_corners = rotated_corners + np.array([plot_x, plot_y])
                
                fov_polygon = Polygon(world_corners, alpha=0.25, 
                                    facecolor=colors[i], edgecolor='black', linewidth=0.5)
                ax.add_patch(fov_polygon)
                
                # Label waypoint
                ax.annotate(f'{i+1}', (plot_x, plot_y), xytext=(2, 2), 
                           textcoords='offset points', fontsize=18, color='white', 
                           weight='bold', ha='center', va='center')
            
            # Draw scan path for multi-point
            if len(waypoints) > 1:
                path_x = [wp[1] for wp in waypoints]  # Convert coordinates
                path_y = [wp[0] for wp in waypoints]
                ax.plot(path_x, path_y, 'r--', linewidth=1, alpha=0.7)
            
            # Coverage info
            strategy = "Single Point" if len(waypoints) == 1 else "Multi Point"
            info_text = f"{test_case['name']}\n{strategy}\n{len(waypoints)} waypoints\nHeight: {test_case['scan_height']:.0f}mm"
        
        ax.text(0.02, 0.02, info_text, transform=ax.transAxes, 
               verticalalignment='bottom', fontsize=20,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Remove axis labels for cleaner look
        ax.set_xticks([])
        ax.set_yticks([])
    
    def calculate_polygon_area(self, coordinates):
        """Calculate polygon area using shoelace formula"""
        if len(coordinates) < 3:
            return 0
        
        x = [coord[0] for coord in coordinates]
        y = [coord[1] for coord in coordinates]
        
        area = 0.5 * abs(sum(x[i]*y[i+1] - x[i+1]*y[i] for i in range(-1, len(x)-1)))
        return area
    
    def generate_shapes_grid(self):
        """Generate 3x3 grid showing original shapes"""
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        fig.suptitle('Scan Region Shapes - Geometric Analysis', fontsize=16, fontweight='bold')
        
        for i, test_case in enumerate(self.test_cases):
            row, col = i // 3, i % 3
            ax = axes[row, col]
            self.plot_single_region(ax, test_case, show_coverage=False)
        
        plt.tight_layout()
        plt.savefig('scan_shapes_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Shape analysis grid saved as 'scan_shapes_analysis.png'")
    
    def generate_coverage_grid(self):
        """Generate 3x3 grid showing scan coverage plans"""
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        fig.suptitle('Scan Coverage Plans - Waypoint Distribution and FOV Coverage', fontsize=16, fontweight='bold')
        
        for i, test_case in enumerate(self.test_cases):
            row, col = i // 3, i % 3
            ax = axes[row, col]
            self.plot_single_region(ax, test_case, show_coverage=True)
        
        plt.tight_layout()
        plt.savefig('scan_coverage_plans.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Coverage plans grid saved as 'scan_coverage_plans.png'")
    
    def generate_both_grids(self):
        """Generate both visualization grids"""
        print("Generating scan system visualization grids...")
        print("=" * 50)
        
        # Generate shapes analysis
        print("Creating shapes analysis grid...")
        self.generate_shapes_grid()
        
        # Generate coverage plans  
        print("Creating coverage plans grid...")
        self.generate_coverage_grid()
        
        print("=" * 50)
        print("Visualization generation completed!")
        print("Two PNG files have been created:")
        print("1. scan_shapes_analysis.png - Original regions with bounding box analysis")
        print("2. scan_coverage_plans.png - Scan coverage with waypoints and FOV")
        print("These images demonstrate the geometric optimization algorithms.")


def main():
    """Main execution function"""
    # Set up matplotlib for better rendering
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.linewidth'] = 0.5
    
    generator = ScanVisualizationGenerator()
    generator.generate_both_grids()


if __name__ == '__main__':
    main()