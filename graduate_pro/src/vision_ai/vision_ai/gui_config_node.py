#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from vision_ai_interfaces.srv import PlanScan
from geometry_msgs.msg import Point
import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
import threading
import time
import numpy as np
import math

class GuiConfigNode(Node):
    def __init__(self):
        super().__init__('gui_config_node')
        
        # Create the service client
        self.plan_client = self.create_client(PlanScan, 'plan_scan')
        
        # Wait for the service to become available
        while not self.plan_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for scan planning service...')
        
        self.get_logger().info('GUI config node started')
        
        # Store pending request
        self.pending_request = None
        self.pending_future = None
        
        # Create timer to check service responses
        self.timer = self.create_timer(0.1, self.check_service_response)
        
        # Launch GUI in a separate thread
        self.gui_thread = threading.Thread(target=self.start_gui)
        self.gui_thread.daemon = True
        self.gui_thread.start()

        self.planning_future = None
        self.planning_timer = None

    def check_service_response(self):
        """Check for completed service responses"""
        if self.pending_future and self.pending_future.done():
            try:
                response = self.pending_future.result()
                if response.success:
                    self.show_planning_result(response.scan_plan)
                else:
                    messagebox.showerror("Planning Failed", response.message)
            except Exception as e:
                messagebox.showerror("Service Error", f"Planning failed: {str(e)}")
            finally:
                self.pending_future = None

    def call_planning_service(self, mode, height, points):
        """Call scanning planning service"""
        request = PlanScan.Request()
        request.mode = mode
        request.object_height = height
        
        # Convert points
        for x, y in points:
            point = Point()
            point.x = float(x)
            point.y = float(y)
            point.z = 0.0
            request.points.append(point)
        
        self.get_logger().info(f'Sending planning request: {mode}, {height}mm, {len(points)} points')
        
        # Use async call to avoid spin conflict
        future = self.plan_client.call_async(request)
        
        # Check the result in a timer
        self.planning_future = future
        self.planning_timer = self.create_timer(0.1, self.check_planning_result)

    def check_planning_result(self):
        """Check planning service result"""
        if self.planning_future.done():
            self.planning_timer.destroy()  # Stop the timer
            
            try:
                response = self.planning_future.result()
                if response.success:
                    self.show_planning_result(response.scan_plan)
                else:
                    messagebox.showerror("Planning Failed", response.message)
            except Exception as e:
                self.get_logger().error(f'Service call failed: {e}')
                messagebox.showerror("Error", f"Planning service failed: {str(e)}")

    def start_gui(self):
        """Start the GUI"""
        self.show_mode_selection()

    def show_mode_selection(self):
        """Show mode selection"""
        mode_window = tk.Tk()
        mode_window.title("Scan Configuration")
        mode_window.geometry("300x150")
        
        tk.Label(mode_window, text="Choose Input Mode", font=("Arial", 22)).pack(pady=10)
        tk.Button(mode_window, text="Preset Regions", 
                 command=lambda: self.on_mode_selected("preset", mode_window)).pack(pady=5)
        tk.Button(mode_window, text="Manual Input", 
                 command=lambda: self.on_mode_selected("manual", mode_window)).pack(pady=5)

        mode_window.mainloop()

    def on_mode_selected(self, mode, window):
        """Handle mode selection"""
        window.destroy()
        if mode == "preset":
            self.show_preset_gui()
        elif mode == "manual":
            self.show_manual_gui()

    def show_preset_gui(self):
        """Display GUI for preset regions"""
        window = tk.Tk()
        window.title("Preset Regions")
        window.geometry("350x350")

        # Object height
        tk.Label(window, text="Object Height (mm):",font=("Arial", 18)).pack(pady=5)
        height_entry = tk.Entry(window, width=10)
        height_entry.insert(0, "50.00")
        height_entry.pack()

        # Preset regions
        regions = {
            'Small Square (200x200mm)': [(150, -150), (150, 100), (350, 100), (350, -150)],
            'Medium Rectangle (435x290mm)': [(-250, -390), (-250, -100), (235, -100), (235, -390)],
            'Large Rectangle (696x464mm)': [(778.55, -124.23),
(323.60, 83.20),
(95.43, -417.24),
(550.37, -624.67)]
        }

        selected_region = tk.StringVar(value=list(regions.keys())[0])
        for name in regions:
            tk.Radiobutton(window, text=name, variable=selected_region, value=name).pack(anchor=tk.W)

        tk.Button(window, text="Preview Region", 
                 command=lambda: self.preview_region(regions[selected_region.get()])).pack(pady=10)
        tk.Button(window, text="Confirm & Plan", 
                 command=lambda: self.confirm_and_plan("preset", regions[selected_region.get()], 
                                                     height_entry.get(), window)).pack(pady=10)
        tk.Button(window, text="Cancel", 
                 command=lambda: self.cancel(window)).pack(pady=5)

        window.mainloop()

    def show_manual_gui(self):
        """Display GUI for manual input"""
        window = tk.Tk()
        window.title("Manual Input")
        window.geometry("350x400") # Adjust window size for more space if needed

        # Object height
        tk.Label(window, text="Object Height (mm):",font=("Arial", 18)).pack(pady=5)
        height_entry = tk.Entry(window, width=10)
        height_entry.insert(0, "50")
        height_entry.pack()

        # Manual coordinate input
        entries = []
        point_names = ["Point 1 (X,Y)", "Point 2 (X,Y)", "Point 3 (X,Y)", "Point 4 (X,Y)"]
        
        for name in point_names:
            frame = tk.Frame(window)
            frame.pack(pady=5)
            tk.Label(frame, text=f"{name}: X:",font=("Arial", 18)).pack(side=tk.LEFT)
            x_entry = tk.Entry(frame, width=8)
            x_entry.pack(side=tk.LEFT)
            tk.Label(frame, text="Y:",font=("Arial", 18)).pack(side=tk.LEFT)
            y_entry = tk.Entry(frame, width=8)
            y_entry.pack(side=tk.LEFT)
            entries.append((x_entry, y_entry))
        
        # Add a note for coordinate range
        tk.Label(window, text="Note: X, Y coordinates\n must be between -900 and 900 mm.", 
                 font=("Arial", 15), fg="red").pack(pady=5)

        tk.Button(window, text="Preview Region", 
                 command=lambda: self.preview_region(self.get_manual_points(entries))).pack(pady=10)
        tk.Button(window, text="Confirm & Plan", 
                 command=lambda: self.confirm_manual_and_plan(entries, height_entry.get(), window)).pack(pady=10)
        tk.Button(window, text="Cancel", 
                 command=lambda: self.cancel(window)).pack(pady=5)

        window.mainloop()

    def get_manual_points(self, entries):
        """Get manually entered points"""
        try:
            points = []
            for x_entry, y_entry in entries:
                x_text = x_entry.get().strip()
                y_text = y_entry.get().strip()
                if x_text and y_text:
                    x = float(x_text)
                    y = float(y_text)
                    points.append((x, y))
            return points if len(points) == 4 else None
        except ValueError:
            return None

    def preview_region(self, points):
        """Preview scanning region with matplotlib"""
        if not points or len(points) != 4:
            messagebox.showwarning("Warning", "Please enter all 4 points first")
            return
        
        # Validate points for preview as well
        for x, y in points:
            if not (-900 <= x <= 900 and -900 <= y <= 900):
                messagebox.showerror("Error", "All X, Y coordinates must be within ±900mm for preview.")
                return
        
        self.get_logger().info(f'Preview region: {points}')
        
        # Create matplotlib figure
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Set up coordinate system (world coordinates)
        ax.set_xlim(-800, 800)
        ax.set_ylim(800, -800)  # Inverted Y for display
        ax.set_xlabel('Y (mm) - Horizontal: Left(-) → Right(+)')
        ax.set_ylabel('X (mm) - Vertical: Up(-) → Down(+)')
        ax.set_title('Scanning Region Preview (World Coordinates)')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Draw origin
        ax.plot(0, 0, 'ko', markersize=8, label='Origin (0,0)')
        
        # Draw obstacle zone (150mm radius circle at origin)
        obstacle = Circle((0, 0), 150, fill=False, color='red', linewidth=2, 
                         linestyle='--', label='Obstacle Zone (r=150mm)')
        ax.add_patch(obstacle)
        
        # Draw workspace boundary (800mm radius)
        workspace = Circle((0, 0), 800, fill=False, color='gray', linewidth=1, 
                          linestyle=':', alpha=0.5, label='Workspace Limit (r=800mm)')
        ax.add_patch(workspace)
        
        # Convert points to plot coordinates (world_x, world_y) -> (world_y, world_x)
        plot_points = [(p[1], p[0]) for p in points]
        
        # Draw scanning region
        region_polygon = Polygon(plot_points, fill=True, alpha=0.3, 
                               color='blue', label='Target Region')
        ax.add_patch(region_polygon)
        
        # Draw region outline and points
        region_x = [p[0] for p in plot_points] + [plot_points[0][0]]  # Close polygon
        region_y = [p[1] for p in plot_points] + [plot_points[0][1]]
        ax.plot(region_x, region_y, 'b-', linewidth=2)
        ax.plot([p[0] for p in plot_points], [p[1] for p in plot_points], 
               'bo', markersize=6)
        
        # Label points
        point_labels = ['P1', 'P2', 'P3', 'P4']
        for i, (px, py) in enumerate(plot_points):
            ax.annotate(f'{point_labels[i]}\n({points[i][0]:.0f},{points[i][1]:.0f})', 
                       (px, py), xytext=(10, 10), textcoords='offset points',
                       fontsize=18, ha='left')
        
        # Calculate and display region info
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        x_span = max(x_coords) - min(x_coords)
        y_span = max(y_coords) - min(y_coords)
        centroid_x = sum(x_coords) / 4
        centroid_y = sum(y_coords) / 4
        
        # Draw centroid
        ax.plot(centroid_y, centroid_x, 'r*', markersize=12, label='Centroid')
        
        # Calculate area using shoelace formula
        area = self.calculate_polygon_area(points)
        
        # Add info text
        info_text = f"""Region Info:
X span: {x_span:.1f} mm (vertical)
Y span: {y_span:.1f} mm (horizontal)  
Centroid: ({centroid_x:.1f}, {centroid_y:.1f})
Area: {area/1000:.1f} cm²"""
        
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
               verticalalignment='top', bbox=dict(boxstyle='round', 
               facecolor='wheat', alpha=0.8), fontsize=19)
        
        # Add coordinate system reminder
        coord_text = "Coordinate System:\nX: Up(-) → Down(+)\nY: Left(-) → Right(+)"
        ax.text(0.98, 0.02, coord_text, transform=ax.transAxes,
               horizontalalignment='right', verticalalignment='bottom',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
               fontsize=18)
        
        ax.legend(loc='upper right')
        plt.tight_layout()
        plt.show()

    def calculate_polygon_area(self, coordinates):
        """Calculate area of polygon using shoelace formula"""
        if len(coordinates) < 3:
            return 0
        
        x = [coord[0] for coord in coordinates]
        y = [coord[1] for coord in coordinates]
        
        area = 0.5 * abs(sum(x[i]*y[i+1] - x[i+1]*y[i] for i in range(-1, len(x)-1)))
        return area

    def confirm_and_plan(self, mode, points, height_str, window):
        """Confirm and call planning service"""
        self.get_logger().info(f'Confirming plan: mode={mode}, height_str="{height_str}"')
        
        try:
            height = float(height_str.strip())
            self.get_logger().info(f'Parsed height: {height}')
            
            if height <= 0 or height > 1000:
                self.get_logger().warn(f'Height {height} out of range')
                messagebox.showerror("Error", "Height must be between 0-1000mm")
                return
            
            self.get_logger().info('Height validation passed, destroying window')
            window.destroy()
            
            # Call ROS2 planning service
            self.get_logger().info('Calling planning service...')
            self.call_planning_service(mode, height, points)
            
        except ValueError as e:
            self.get_logger().error(f'ValueError: {e}')
            messagebox.showerror("Error", "Please enter a valid height")
        except Exception as e:
            self.get_logger().error(f'Unexpected error: {e}')

    def confirm_manual_and_plan(self, entries, height_str, window):
        """Confirm manual input and plan"""
        try:
            height = float(height_str.strip())
            if height <= 0 or height > 1000:
                messagebox.showerror("Error", "Height must be between 0-1000mm")
                return
                
            points = []
            for x_entry, y_entry in entries:
                x_str = x_entry.get().strip()
                y_str = y_entry.get().strip()

                if not x_str or not y_str:
                    messagebox.showerror("Error", "Please enter all 4 points.")
                    return

                x = float(x_str)
                y = float(y_str)

                # Validate X, Y coordinates
                if not (-900 <= x <= 900 and -900 <= y <= 900):
                    messagebox.showerror("Input Error", f"Coordinates ({x}, {y}) are out of range. X and Y must be between -900mm and 900mm. Please re-enter.")
                    return # Do not destroy window, allow user to correct
                points.append((x, y))
            
            if len(points) != 4:
                messagebox.showerror("Error", "Please enter all 4 points.")
                return
            
            # If all validations pass, then destroy window and call service
            window.destroy()  
            self.call_planning_service("manual", height, points)
            
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers for coordinates and height.")
            # Keep window open for user correction

    def cancel(self, window):
        """Cancel operation"""
        window.destroy()
        self.get_logger().info("User canceled scan")

    def show_planning_result(self, scan_plan):
        """Show planning result with visualization"""
        # Display basic info
        result_text = f"""Planning Complete!

    Strategy: {scan_plan.strategy}
    Scan Height: {scan_plan.scan_height:.1f}mm
    Required Height: {scan_plan.required_height:.1f}mm
    Waypoints: {len(scan_plan.waypoints)}

    View visualization?"""
        
        result = messagebox.askyesno("Planning Result", result_text)
        if result:
            self.visualize_scan_plan(scan_plan)
        
        # Ask if execution is desired
        execute_result = messagebox.askyesno("Execute Scan", "Execute scanning?")
        if execute_result:
            self.get_logger().info("User confirmed to execute scanning")
            # 🔧 Add actual execution service call
            self.call_execute_service(scan_plan)
        else:
            self.get_logger().info("User cancelled scanning")

    def call_execute_service(self, scan_plan):
        """Call scan execution service"""
        try:
            # Create execute service client if not already done
            if not hasattr(self, 'execute_client'):
                from vision_ai_interfaces.srv import ExecuteScan
                self.execute_client = self.create_client(ExecuteScan, 'execute_scan')
            
            # Wait for service to be available
            if not self.execute_client.wait_for_service(timeout_sec=5.0):
                messagebox.showerror("Error", "Execute service not available")
                return
            
            # Create request
            request = ExecuteScan.Request()
            request.scan_plan = scan_plan
            
            self.get_logger().info('Calling execute service...')
            
            # Async call
            future = self.execute_client.call_async(request)
            future.add_done_callback(self.execute_response_callback)
            
        except Exception as e:
            self.get_logger().error(f'Execute service call failed: {e}')
            messagebox.showerror("Error", f"Failed to start execution: {str(e)}")

    def execute_response_callback(self, future):
        """Execution service response callback"""
        try:
            response = future.result()
            if response.success:
                self.get_logger().info(f'Execution started: {response.message}')
                messagebox.showinfo("Success", f"Scanning started!\n{response.message}")
            else:
                self.get_logger().error(f'Execution failed: {response.message}')
                messagebox.showerror("Error", f"Execution failed: {response.message}")
        except Exception as e:
            self.get_logger().error(f'Execute response error: {e}')
            messagebox.showerror("Error", f"Execution error: {str(e)}")

    def visualize_scan_plan(self, scan_plan):
        """Visualize complete scan plan with waypoints, yaw angles, and rotated FOV coverage"""
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        
        # Coordinate system setup
        ax.set_xlim(-800, 800)
        ax.set_ylim(800, -800)
        ax.set_xlabel('Y (mm) - Horizontal: Left(-) → Right(+)')
        ax.set_ylabel('X (mm) - Vertical: Up(-) → Down(+)')
        ax.set_title(f'Scan Plan Visualization ({scan_plan.strategy} strategy)')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Draw origin and limits
        ax.plot(0, 0, 'ko', markersize=8, label='Origin (0,0)')
        obstacle = Circle((0, 0), 150, fill=False, color='red', linewidth=2, 
                        linestyle='--', label='Obstacle Zone (r=150mm)')
        ax.add_patch(obstacle)
        workspace = Circle((0, 0), 800, fill=False, color='gray', linewidth=1, 
                        linestyle=':', alpha=0.5, label='Workspace Limit')
        ax.add_patch(workspace)
        
        # Draw scan region
        if scan_plan.scan_region:
            region_points = [(p.y, p.x) for p in scan_plan.scan_region]
            region_polygon = Polygon(region_points, fill=True, alpha=0.2, 
                                     color='blue', label='Target Region')
            ax.add_patch(region_polygon)
            region_x = [p[0] for p in region_points] + [region_points[0][0]]
            region_y = [p[1] for p in region_points] + [region_points[0][1]]
            ax.plot(region_x, region_y, 'b-', linewidth=2)
        
        # Draw waypoints and FOV
        if scan_plan.waypoints:
            # FOV sizes at scan height
            fov_h_deg = 70.1  # Horizontal FOV (1280px)
            fov_v_deg = 43.2  # Vertical FOV (720px)
            fov_h_rad = math.radians(fov_h_deg / 2)
            fov_v_rad = math.radians(fov_v_deg / 2)
            effective_height = scan_plan.scan_height - scan_plan.object_height
            if scan_plan.waypoints:
                # 使用实际扫描参数计算FOV
                fov_h_deg = 70.1  # 从相机参数获取
                fov_v_deg = 43.2
                fov_h_rad = math.radians(fov_h_deg / 2)
                fov_v_rad = math.radians(fov_v_deg / 2)
                
                # 动态计算FOV覆盖范围
                effective_height = scan_plan.scan_height - scan_plan.object_height
                fov_1280_size = 2 * effective_height * math.tan(fov_h_rad)
                fov_720_size = 2 * effective_height * math.tan(fov_v_rad)
                
                self.get_logger().info(f'动态FOV计算: 有效高度={effective_height}mm, '
                                    f'FOV={fov_1280_size:.0f}×{fov_720_size:.0f}mm')

            waypoint_colors = plt.cm.viridis(np.linspace(0, 1, len(scan_plan.waypoints)))
            
            for i, waypoint in enumerate(scan_plan.waypoints):
                wp_x = waypoint.pose.position.x
                wp_y = waypoint.pose.position.y
                qz = waypoint.pose.orientation.z
                qw = waypoint.pose.orientation.w
                yaw_rad = -2 * math.atan2(qz, qw)
                yaw_deg = -math.degrees(yaw_rad)

                plot_x = wp_y
                plot_y = wp_x

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
                fov_polygon = Polygon(world_corners, alpha=0.3, 
                                    facecolor=waypoint_colors[i],
                                    edgecolor='black', linewidth=1)
                ax.add_patch(fov_polygon)

                ax.plot(plot_x, plot_y, 'ro', markersize=10, 
                        markeredgecolor='white', markeredgewidth=2)
                ax.annotate(f'WP{i+1}\n{yaw_deg:.0f}°', (plot_x, plot_y), 
                            xytext=(10, 10), textcoords='offset points',
                            fontsize=19, color='darkred', weight='bold',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            
            # Draw path
            if len(scan_plan.waypoints) > 1:
                path_x = [wp.pose.position.y for wp in scan_plan.waypoints]
                path_y = [wp.pose.position.x for wp in scan_plan.waypoints]
                for i in range(len(path_x) - 1):
                    dx = path_x[i+1] - path_x[i]
                    dy = path_y[i+1] - path_y[i]
                    if abs(dx) + abs(dy) > 0:
                        ax.arrow(path_x[i], path_y[i], dx*0.8, dy*0.8, 
                                 head_width=12, head_length=10, fc='red', ec='red', alpha=0.6)
                ax.plot(path_x, path_y, 'r--', linewidth=2, alpha=0.7, label='Scan Path')

        # Region & yaw info
        if scan_plan.scan_region and scan_plan.waypoints:
            region_area = self.calculate_polygon_area_from_points(scan_plan.scan_region)
            yaw_angles = []
            for wp in scan_plan.waypoints:
                qz = wp.pose.orientation.z
                qw = wp.pose.orientation.w
                yaw_rad = 2 * math.atan2(qz, qw)
                yaw_deg = math.degrees(yaw_rad)
                yaw_angles.append(yaw_deg)
            unique_yaws = list(set([round(y, 1) for y in yaw_angles]))
            yaw_info = f"{unique_yaws[0]:.0f}°" if len(unique_yaws) == 1 else f"{len(unique_yaws)} angles"
            info_text = f""" Scan Plan Info:
    Strategy: {scan_plan.strategy}
    Waypoints: {len(scan_plan.waypoints)}
    Yaw Angle(s): {yaw_info}
    Scan Height: {scan_plan.scan_height:.1f}mm
    Object Height: {scan_plan.object_height:.1f}mm

    FOV Coverage:
    1280px: {fov_1280_size:.0f}mm
    720px: {fov_720_size:.0f}mm
    Region Area: {region_area/1000:.1f} cm²"""
        else:
            info_text = f""" Scan Plan Info:
    Strategy: {scan_plan.strategy}
    Waypoints: {len(scan_plan.waypoints)}
    Scan Height: {scan_plan.scan_height:.1f}mm"""
        
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', 
                facecolor='wheat', alpha=0.9), fontsize=15)

        existing_handles, existing_labels = ax.get_legend_handles_labels()
        new_elements = [
            plt.Rectangle((0, 0), 1, 1, facecolor=plt.cm.viridis(0.5), alpha=0.3, label='FOV Coverage')
        ]
        all_handles = existing_handles + new_elements
        ax.legend(handles=all_handles, loc='upper right', fontsize=19)
        
        coord_text = """ Coordinate System:
    X: Up(-) → Down(+)
    Y: Left(-) → Right(+)
    Yaw: CCW from +Y axis"""
        ax.text(0.98, 0.02, coord_text, transform=ax.transAxes,
                horizontalalignment='right', verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                fontsize=19)
        
        plt.tight_layout()
        plt.show()

    def calculate_polygon_area_from_points(self, points):
        """Calculate area from ROS Point messages"""
        if len(points) < 3:
            return 0
        coords = [(p.x, p.y) for p in points]
        return self.calculate_polygon_area(coords)

def main(args=None):
    rclpy.init(args=args)
    node = GuiConfigNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()