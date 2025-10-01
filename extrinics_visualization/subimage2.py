import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import sys
import math

# Add paths for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import common utilities
from transform_utils import get_se3_extrinsic, print_se3_info
from radar_lidar_calibration.radar_lidar_cali_utils import KPeaks

# Import specific functions instead of everything to avoid triggering main()
try:
    from lidar_to_rgb_video import load_lidar
except ImportError:
    def load_lidar(bin_file_path):
        """Load lidar points from binary file - placeholder implementation"""
        print(f"Warning: Using placeholder load_lidar function for {bin_file_path}")
        # Return dummy data for testing
        return np.random.rand(1000, 4)  # 1000 points with x,y,z,intensity

def crop_lidar_by_range(lidar_xyz, r_max, r_min=0.0, use_xy=False,
                        origin=None, return_mask=False):
    """
    Keep LiDAR points within [r_min, r_max] of 'origin'.

    Args
    ----
    lidar_xyz : (N,3) float array  (points in LiDAR frame)
    r_max     : float, outer radius in meters
    r_min     : float, inner radius in meters (default 0)
    use_xy    : bool, if True use horizontal (XY) range; else full 3D range
    origin    : None or (3,) array, center of the ring (default [0,0,0])
    return_mask : bool, if True also return the boolean mask

    Returns
    -------
    cropped : (M,3) array of points inside the range (and mask if requested)
    """
    pts = np.asarray(lidar_xyz, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("lidar_xyz must be (N,3).")
    if r_max <= 0 or r_min < 0 or r_min > r_max:
        raise ValueError("Require 0 <= r_min <= r_max and r_max > 0.")

    o = np.zeros(3) if origin is None else np.asarray(origin, float).reshape(3)

    # Use squared distances (avoids sqrt for speed)
    if use_xy:
        v = pts[:, :2] - o[:2]
        d2 = v[:, 0]**2 + v[:, 1]**2
        rmin2, rmax2 = r_min**2, r_max**2
    else:
        v = pts - o
        d2 = (v * v).sum(axis=1)
        rmin2, rmax2 = r_min**2, r_max**2

    finite = np.isfinite(pts).all(axis=1)
    mask = finite & (d2 >= rmin2) & (d2 <= rmax2)
    cropped = pts[mask]

    return (cropped, mask) if return_mask else cropped


def project_radar_points_to_lidar(radar_image_path, rslidar_bin_file_path, T_lidar_radar):
    # CIR-304H radar parameters
    radar_resolution = 0.043809514
    encoder_size = 5600

    radar_image = cv2.imread(radar_image_path,cv2.IMREAD_GRAYSCALE)
    byte_array = np.array(radar_image)
    azimuths = byte_array[:, 8:10].view(np.uint16) / float(encoder_size) * 2 * np.pi
    polar = byte_array[:, 11:].astype(np.float32) / 255.0

    import torch
    import torchvision
    # preprocessing steps
    device = 'cpu'
    polar_intensity = torch.tensor(polar).to(device)
    polar_std = torch.std(polar_intensity, dim=1)
    polar_mean = torch.mean(polar_intensity, dim=1)
    polar_intensity -= (polar_mean.unsqueeze(1) + 2*polar_std.unsqueeze(1))
    polar_intensity[polar_intensity < 0] = 0
    polar_intensity = torchvision.transforms.functional.gaussian_blur(polar_intensity.unsqueeze(0), (9,1), 3).squeeze()
    polar_intensity /= torch.max(polar_intensity, dim=1, keepdim=True)[0]
    polar_intensity[torch.isnan(polar_intensity)] = 0

    import pyboreas as pb
    radar_cart_img = pb.utils.radar.radar_polar_to_cartesian(azimuths,polar_intensity,radar_resolution,cart_resolution=0.224,cart_pixel_width=3000)
    cv2.imwrite("radar_cart_img.png", radar_cart_img*255)

    cv2.imwrite("radar_polar_img.png", polar_intensity.numpy()*255)

    # # # lets viusalize the polar img in grey scale
    # plt.imshow(radar_cart_img, cmap='gray')
    # plt.colorbar()
    # plt.title(f'Radar Cart Image')
    # plt.xlabel('Range Bin')
    # plt.show()

    # # lets extract points from the polar img
    # KPEAKS = True # K-peaks is the best extractor
    # if KPEAKS:
    targets = KPeaks(polar_intensity.numpy(),minr=5,maxr=60,res=radar_resolution, K=10, static_threshold=0.27)

    radar_pts = []
    for target in targets:
        azimuth_idx = int(target[0])
        range_idx = int(target[1])

        x = range_idx * radar_resolution * math.cos(azimuths[azimuth_idx])
        y = range_idx * radar_resolution * math.sin(azimuths[azimuth_idx])
        z = 0  # Assume radar points are on ground plane
        intensity = polar_intensity[azimuth_idx, range_idx].item()

        radar_pts.append([x, y, z, intensity])

    radar_pts = np.array(radar_pts)
    radar_xy = radar_pts[:, 0:2]  # x, y
    # append a zeros as z
    radar_xyz = np.hstack([radar_xy, np.zeros((radar_xy.shape[0], 1))])
    radar_intensities = radar_pts[:, 3]*255  # intensity values ma

    # Load lidar points
    lidar_points = load_lidar(rslidar_bin_file_path)
    lidar_xyz = lidar_points[:, :3]
    lidar_intensities = lidar_points[:, 3]

    print("lidar intensities max:", np.max(lidar_intensities))
    print("lidar intensities min:", np.min(lidar_intensities))

    # Transform radar points to lidar frame
    radar_xyz_homogeneous = np.hstack([radar_xyz, np.ones((radar_xyz.shape[0], 1))])
    radar_xyz_in_lidar = (T_lidar_radar @ radar_xyz_homogeneous.T).T

    print(f'Lidar points shape: {lidar_xyz.shape}')
    print(f'Radar points shape: {radar_xyz_in_lidar.shape}')
    print(f'Lidar intensity range: {np.min(lidar_intensities):.2f} to {np.max(lidar_intensities):.2f}')
    print(f'Radar intensity range: {np.min(radar_intensities):.2f} to {np.max(radar_intensities):.2f}')

    return lidar_xyz, lidar_intensities, radar_xyz_in_lidar, radar_intensities

def visualize_lidar_radar_overlay_simple(lidar_xyz, lidar_intensities, radar_xyz, radar_intensities, save_path="subimage2.png", remove_ground=True, crop_range=True, r_max=45.0):
    """
    Visualization of radar points overlaid on lidar points with intensity colormaps
    Returns the image as subimage2
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    fig, ax = plt.subplots(figsize=(20, 20), dpi=150)
    fig.patch.set_facecolor('none')  # Make figure background transparent
    ax.set_facecolor('none')  # Make axes background transparent
    
    # Remove ground plane if requested
    if remove_ground:
        import open3d as o3d
        
        # RANSAC ground plane removal
        print("Applying RANSAC ground removal to lidar points...")
        lidar_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(lidar_xyz))
        plane, inliers = lidar_pc.segment_plane(0.05, 3, 2000)  # 5cm threshold
        non_ground_mask = np.ones(len(lidar_xyz), dtype=bool)
        non_ground_mask[inliers] = False
        lidar_xyz = lidar_xyz[non_ground_mask]
        lidar_intensities = lidar_intensities[non_ground_mask]
        
        print(f"RANSAC ground removal: removed {len(inliers)} ground points, kept {len(lidar_xyz)} points")
        print(f"After RANSAC ground removal - Lidar Z range: [{np.min(lidar_xyz[:, 2]):.2f}, {np.max(lidar_xyz[:, 2]):.2f}]")
    
    # Crop lidar range if requested
    if crop_range:
        lidar_xyz, range_mask = crop_lidar_by_range(lidar_xyz, r_max=r_max, r_min=0.0, use_xy=True, return_mask=True)
        lidar_intensities = lidar_intensities[range_mask]
        print(f"Range cropping: kept {np.sum(range_mask)} points within {r_max}m range")
    
    # Debug: Print intensity statistics to compare with subimage1.py
    print(f"Lidar intensities after filtering:")
    print(f"  - Shape: {lidar_intensities.shape}")
    print(f"  - Min: {np.min(lidar_intensities):.2f}")
    print(f"  - Max: {np.max(lidar_intensities):.2f}")
    print(f"  - Mean: {np.mean(lidar_intensities):.2f}")
    print(f"  - First 10 values: {lidar_intensities[0:10]}")
    
    # Use same intensity range as subimage1.py for consistent coloring
    min_intensity = 0  # Same as subimage1.py
    max_intensity = 255  # Same as subimage1.py
    print(f"Lidar intensity range: {min_intensity:.2f} to {max_intensity:.2f}")
    
    # Don't pre-normalize - let matplotlib handle it like subimage1.py does
    # This ensures identical color mapping
    
    # Radar intensities are already in 0-1 range
    
    # Rotate points clockwise by 90 degrees
    # 90-degree clockwise rotation matrix: [[0, 1], [-1, 0]]
    rotation_matrix = np.array([[0, 1], [-1, 0]])
    
    # Rotate lidar points
    lidar_xy_rotated = lidar_xyz[:, :2] @ rotation_matrix.T
    lidar_xyz_rotated = np.column_stack([lidar_xy_rotated, lidar_xyz[:, 2]])
    
    # Rotate radar points
    radar_xy_rotated = radar_xyz[:, :2] @ rotation_matrix.T
    radar_xyz_rotated = np.column_stack([radar_xy_rotated, radar_xyz[:, 2]])
    
    # Plot lidar points with intensity colormap (same as subimage1.py)
    # Use raw intensities and let matplotlib normalize with vmin/vmax like subimage1.py
    ax.scatter(lidar_xyz_rotated[:, 0], lidar_xyz_rotated[:, 1], 
                         c=lidar_intensities, cmap='turbo', s=1, alpha=0.3, 
                         vmin=min_intensity, vmax=max_intensity, label='Robosense')
    
    # Plot radar points with better styling
    ax.scatter(radar_xyz_rotated[:, 0], radar_xyz_rotated[:, 1], 
                         c=radar_intensities, cmap='YlOrRd', s=60, alpha=0.9, 
                         edgecolors='black', linewidths=1, label='Navtech')
    
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
    # Save as PDF with transparent background
    pdf_path = save_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight', facecolor='none', format='pdf', transparent=True)
    
    # Convert plot to image array with transparent background
    fig.canvas.draw()
    # Get RGBA buffer using buffer_rgba() method
    subimage2 = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    subimage2 = subimage2.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    
    # Convert RGBA to BGRA for OpenCV
    subimage2 = cv2.cvtColor(subimage2, cv2.COLOR_RGBA2BGRA)
    
    plt.show()
    
    print(f"Intensity overlay visualization saved to: {pdf_path}")
    return subimage2

if __name__ == "__main__":
    target_asset_folder = "/home/samqiao/ASRL/fomo-public-sdk/raw_fomo_rosbags/red_2024-11-21-10-34/target_assets"

    rslidar_bin_file_path = os.path.join(target_asset_folder, "rslidar","1732203284100439.bin")
    
    # Check if files exist
    if not os.path.exists(rslidar_bin_file_path):
        print(f"Error: Lidar file not found: {rslidar_bin_file_path}")
        exit(1)
    
    # okay first one is done next is the radar and lidar overlay
    radar_image_path = os.path.join(target_asset_folder, "navtech","1732203284153343.png")
    T_lidar_radar = get_se3_extrinsic("navtech", "robosense")
    print_se3_info(T_lidar_radar, "navtech", "robosense")
    
    # Project radar points to lidar frame and visualize overlay
    if T_lidar_radar is not None:
        lidar_xyz, lidar_intensities, radar_xyz_in_lidar, radar_intensities = project_radar_points_to_lidar(
            radar_image_path, rslidar_bin_file_path, T_lidar_radar
        )
        subimage2 = visualize_lidar_radar_overlay_simple(lidar_xyz, lidar_intensities, radar_xyz_in_lidar, radar_intensities, remove_ground=False, crop_range=True, r_max=60.0)
        # Save with PNG format (lossless) and high quality with transparency
        cv2.imwrite("subimage2.png", subimage2, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        # Also save as PDF using matplotlib with transparent background
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(20, 20), dpi=150)
        ax.imshow(cv2.cvtColor(subimage2, cv2.COLOR_BGRA2RGBA))
        ax.axis('off')
        plt.tight_layout()
        plt.savefig("subimage2.pdf", dpi=300, bbox_inches='tight', facecolor='none', format='pdf', transparent=True)
        plt.close()
    else:
        print("Failed to get radar to lidar transformation")
