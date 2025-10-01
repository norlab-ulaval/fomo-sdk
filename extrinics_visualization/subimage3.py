import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import sys

# Add paths for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import necessary functions
from transform_utils import get_se3_extrinsic, print_se3_info
from radar_lidar_calibration.radar_lidar_cali_utils import crop_lidar_by_range

try:
    from lidar_to_rgb_video import load_lidar
except ImportError:
    def load_lidar(bin_file_path):
        """Load lidar points from binary file - placeholder implementation"""
        print(f"Warning: Using placeholder load_lidar function for {bin_file_path}")
        return np.random.rand(1000, 4)  # 1000 points with x,y,z,intensity

def overlay_ls_rs_lidar(lslidar_bin_file_path, rslidar_bin_file_path, T_rs_ls, rs_xyz=None, rs_intensities=None, remove_ground=True, crop_range=True, r_max=100.0, yaw_start=None, yaw_end=None, rotation_degrees=90, zoom_factor=1.0):
    """
    Overlay two lidar point clouds (leishen and robosense) in the robosense frame
    Returns the image as subimage3
    """
    # Load both lidar point clouds
    lslidar_points = load_lidar(lslidar_bin_file_path)
    rslidar_points = load_lidar(rslidar_bin_file_path)
    
    # Extract XYZ and intensities
    ls_xyz = lslidar_points[:, :3]
    ls_intensities = lslidar_points[:, 3]
    rs_xyz = rslidar_points[:, :3]
    rs_intensities = rslidar_points[:, 3]
    
    # Transform leishen points to robosense frame
    ls_points_h = np.hstack([ls_xyz, np.ones((ls_xyz.shape[0], 1))])
    ls_xyz_in_rs = (T_rs_ls @ ls_points_h.T).T[:, :3]
    
    # Remove ground plane if requested (for both lidars)
    if remove_ground:
        import open3d as o3d
        
        # Robosense ground removal using RANSAC
        print("Applying RANSAC ground removal to Robosense points...")
        rs_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(rs_xyz))
        rs_plane, rs_inliers = rs_pc.segment_plane(0.05, 3, 2000)  # 5cm threshold
        rs_non_ground_mask = np.ones(len(rs_xyz), dtype=bool)
        rs_non_ground_mask[rs_inliers] = False
        rs_xyz = rs_xyz[rs_non_ground_mask]
        rs_intensities = rs_intensities[rs_non_ground_mask]
        
        # Leishen ground removal using RANSAC (in robosense frame)
        print("Applying RANSAC ground removal to Leishen points...")
        ls_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(ls_xyz_in_rs))
        ls_plane, ls_inliers = ls_pc.segment_plane(0.05, 3, 2000)  # 5cm threshold
        ls_non_ground_mask = np.ones(len(ls_xyz_in_rs), dtype=bool)
        ls_non_ground_mask[ls_inliers] = False
        ls_xyz_in_rs = ls_xyz_in_rs[ls_non_ground_mask]
        ls_intensities = ls_intensities[ls_non_ground_mask]
        
        print(f"Robosense RANSAC ground removal: removed {len(rs_inliers)} ground points, kept {len(rs_xyz)} points")
        print(f"Leishen RANSAC ground removal: removed {len(ls_inliers)} ground points, kept {len(ls_xyz_in_rs)} points")
        
        # Additional verification: check remaining Z ranges
        print(f"After RANSAC ground removal - Robosense Z range: [{np.min(rs_xyz[:, 2]):.2f}, {np.max(rs_xyz[:, 2]):.2f}]")
        print(f"After RANSAC ground removal - Leishen Z range: [{np.min(ls_xyz_in_rs[:, 2]):.2f}, {np.max(ls_xyz_in_rs[:, 2]):.2f}]")
    
    # Crop lidar range if requested (for both lidars)
    if crop_range:
        # Robosense range cropping
        rs_xyz, rs_range_mask = crop_lidar_by_range(rs_xyz, r_max=r_max, r_min=0.0, use_xy=True, return_mask=True)
        rs_intensities = rs_intensities[rs_range_mask]
        
        # Leishen range cropping
        ls_xyz_in_rs, ls_range_mask = crop_lidar_by_range(ls_xyz_in_rs, r_max=r_max, r_min=0.0, use_xy=True, return_mask=True)
        ls_intensities = ls_intensities[ls_range_mask]
        
        print(f"Robosense range cropping: kept {np.sum(rs_range_mask)} points within {r_max}m range")
        print(f"Leishen range cropping: kept {np.sum(ls_range_mask)} points within {r_max}m range")
    
    # FOV cropping: crop robosense points by yaw angle range
    if yaw_start is not None and yaw_end is not None:
        print(f"Applying FOV cropping: yaw range [{yaw_start}°, {yaw_end}°]...")
        
        # Convert to polar coordinates to filter by yaw angle
        rs_angles = np.arctan2(rs_xyz[:, 1], rs_xyz[:, 0]) * 180 / np.pi  # Convert to degrees
        rs_angles = (rs_angles + 360) % 360  # Convert to [0, 360] range
        
        # Handle angle wrapping (e.g., 0 to 360)
        if yaw_start > yaw_end:  # Crosses 0 degrees (e.g., 350° to 10°)
            fov_mask = (rs_angles >= yaw_start) | (rs_angles <= yaw_end)
        else:  # Normal range (e.g., 0° to 60°)
            fov_mask = (rs_angles >= yaw_start) & (rs_angles <= yaw_end)
        
        # Apply FOV filter
        rs_xyz = rs_xyz[fov_mask]
        rs_intensities = rs_intensities[fov_mask]
        
        print(f"FOV cropping: kept {np.sum(fov_mask)} robosense points in yaw range [{yaw_start}°, {yaw_end}°]")
        print(f"Removed {np.sum(~fov_mask)} robosense points outside FOV")
    
    # Rotate both point clouds clockwise by specified degrees
    # Convert degrees to radians
    rotation_rad = np.radians(rotation_degrees)
    
    # Create rotation matrix for clockwise rotation
    cos_angle = np.cos(rotation_rad)
    sin_angle = np.sin(rotation_rad)
    rotation_matrix = np.array([[cos_angle, sin_angle], [-sin_angle, cos_angle]])
    
    # Rotate robosense points
    rs_xy_rotated = rs_xyz[:, :2] @ rotation_matrix.T
    rs_xyz_rotated = np.column_stack([rs_xy_rotated, rs_xyz[:, 2]])
    
    # Rotate leishen points
    ls_xy_rotated = ls_xyz_in_rs[:, :2] @ rotation_matrix.T
    ls_xyz_rotated = np.column_stack([ls_xy_rotated, ls_xyz_in_rs[:, 2]])
    
    print(f"Applied {rotation_degrees}° clockwise rotation to both point clouds")
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(20, 20), dpi=150)
    
    # Debug: Print intensity statistics to compare with subimage1.py
    print(f"Robosense intensities in subimage3.py (after filtering):")
    print(f"  - Shape: {rs_intensities.shape}")
    print(f"  - Min: {np.min(rs_intensities):.2f}")
    print(f"  - Max: {np.max(rs_intensities):.2f}")
    print(f"  - Mean: {np.mean(rs_intensities):.2f}")
    print(f"  - First 10 values: {rs_intensities[0:10]}")
    
    print(f"Leishen intensity range: [{np.min(ls_intensities):.2f}, {np.max(ls_intensities):.2f}]")
    
    # Use same intensity range as subimage1.py and subimage2.py for consistent coloring
    min_intensity = 0
    max_intensity = 255
    
    # Plot robosense points (turbo colormap, same as subimage1.py and subimage2.py) - rotated
    ax.scatter(rs_xyz_rotated[:, 0], rs_xyz_rotated[:, 1], 
                         c=rs_intensities, cmap='turbo', s=1, alpha=0.3, 
                         vmin=min_intensity, vmax=max_intensity, label='Robosense')
    
    # Plot leishen points (viridis colormap for differentiation) - rotated
    ax.scatter(ls_xyz_rotated[:, 0], ls_xyz_rotated[:, 1], 
                         c=ls_intensities, cmap='viridis', s=8, alpha=0.9, 
                         vmin=min_intensity, vmax=max_intensity, label='Leishen')
    
    # Apply zoom to reduce dead space
    if zoom_factor != 1.0:
        # Get current data bounds
        all_x = np.concatenate([rs_xyz_rotated[:, 0], ls_xyz_rotated[:, 0]])
        all_y = np.concatenate([rs_xyz_rotated[:, 1], ls_xyz_rotated[:, 1]])
        
        # Calculate data range
        x_center = (np.min(all_x) + np.max(all_x)) / 2
        y_center = (np.min(all_y) + np.max(all_y)) / 2
        x_range = np.max(all_x) - np.min(all_x)
        y_range = np.max(all_y) - np.min(all_y)
        
        # Apply zoom factor (smaller factor = more zoomed in)
        x_half_range = (x_range * zoom_factor) / 2
        y_half_range = (y_range * zoom_factor) / 2
        
        # Set axis limits
        ax.set_xlim(x_center - x_half_range, x_center + x_half_range)
        ax.set_ylim(y_center - y_half_range, y_center + y_half_range)
        
        print(f"Applied zoom factor {zoom_factor}: X range [{x_center - x_half_range:.1f}, {x_center + x_half_range:.1f}], Y range [{y_center - y_half_range:.1f}, {y_center + y_half_range:.1f}]")
    
    # Set equal axis and remove labels
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title('')
    
    # Remove all axis labels and ticks
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    
    # Remove border/spine
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Add legend
    # ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=40, prop={'size': 40, 'family': 'Times New Roman'})
    
    plt.tight_layout()
    # Save as PDF
    plt.savefig("subimage3.pdf", dpi=300, bbox_inches='tight', facecolor='white', format='pdf')
    
    # Convert plot to image array
    fig.canvas.draw()
    subimage3 = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    subimage3 = subimage3.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    # Convert RGB to BGR for OpenCV
    subimage3 = cv2.cvtColor(subimage3, cv2.COLOR_RGB2BGR)
    
    plt.show()
    
    print(f"Lidar overlay visualization saved to: subimage3.pdf")
    return subimage3

if __name__ == "__main__":
    # Example usage
    target_asset_folder = "/home/samqiao/ASRL/fomo-public-sdk/raw_fomo_rosbags/red_2024-11-21-10-34/target_assets"
    
    lslidar_bin_file_path = os.path.join(target_asset_folder, "lslidar","1732203284099476.bin")
    rslidar_bin_file_path = os.path.join(target_asset_folder, "rslidar","1732203284100439.bin")
    
    # You would need to get the transformation matrix T_rs_ls from your transform.json
    # For now, using identity matrix as placeholder
    T_rs_ls = get_se3_extrinsic("leishen", "robosense")
    print_se3_info(T_rs_ls, "leishen", "robosense")
    
    subimage3 = overlay_ls_rs_lidar(lslidar_bin_file_path, rslidar_bin_file_path, T_rs_ls, 
                                   remove_ground=False, crop_range=False, r_max=100.0,
                                   yaw_start=120, yaw_end=240, rotation_degrees=0, zoom_factor=1.0)
    
    # Save as PNG
    cv2.imwrite("subimage3.png", subimage3, [cv2.IMWRITE_PNG_COMPRESSION, 1])
