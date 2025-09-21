import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import sys

# Add the parent directory to the path to import from vis.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import necessary functions from vis.py
from vis import load_lidar, crop_lidar_by_range, get_se3_extrinsic, print_se3_info

def overlay_ls_rs_lidar(lslidar_bin_file_path, rslidar_bin_file_path, T_rs_ls, rs_xyz=None, rs_intensities=None, remove_ground=True, crop_range=True, r_max=100.0):
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
        # Robosense ground removal
        rs_z_values = rs_xyz[:, 2]
        rs_ground_threshold = np.percentile(rs_z_values, 10)
        rs_non_ground_mask = rs_xyz[:, 2] > rs_ground_threshold
        rs_xyz = rs_xyz[rs_non_ground_mask]
        rs_intensities = rs_intensities[rs_non_ground_mask]
        
        # Leishen ground removal (in robosense frame)
        ls_z_values = ls_xyz_in_rs[:, 2]
        ls_ground_threshold = np.percentile(ls_z_values, 10)
        ls_non_ground_mask = ls_xyz_in_rs[:, 2] > ls_ground_threshold
        ls_xyz_in_rs = ls_xyz_in_rs[ls_non_ground_mask]
        ls_intensities = ls_intensities[ls_non_ground_mask]
        
        print(f"Robosense ground removal: Z range [{np.min(rs_z_values):.2f}, {np.max(rs_z_values):.2f}], threshold={rs_ground_threshold:.2f}")
        print(f"Removed {np.sum(~rs_non_ground_mask)} ground points, kept {np.sum(rs_non_ground_mask)} points")
        print(f"Leishen ground removal: Z range [{np.min(ls_z_values):.2f}, {np.max(ls_z_values):.2f}], threshold={ls_ground_threshold:.2f}")
        print(f"Removed {np.sum(~ls_non_ground_mask)} ground points, kept {np.sum(ls_non_ground_mask)} points")
    
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
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(20, 20), dpi=150)
    
    # Normalize intensities (0-255 to 0-1)
    ls_intensities_norm = ls_intensities / 255.0
    rs_intensities_norm = rs_intensities / 255.0
    
    # Plot robosense points (rainbow colors, smaller, more faded)
    scatter1 = ax.scatter(rs_xyz[:, 0], rs_xyz[:, 1], 
                         c=rs_intensities_norm, cmap='rainbow', s=1, alpha=0.3, 
                         edgecolors='navy', linewidths=0.2, label='Robosense')
    
    # Plot leishen points (warm colors, larger, more prominent)
    scatter2 = ax.scatter(ls_xyz_in_rs[:, 0], ls_xyz_in_rs[:, 1], 
                         c=ls_intensities_norm, cmap='hot', s=8, alpha=0.9, 
                         edgecolors='darkorange', linewidths=0.5, label='Leishen')
    
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
                                   remove_ground=True, crop_range=True, r_max=80.0)
    
    # Save as PNG
    cv2.imwrite("subimage3.png", subimage3, [cv2.IMWRITE_PNG_COMPRESSION, 1])
