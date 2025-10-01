import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import sys

# Add paths for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import common utilities
from transform_utils import get_se3_extrinsic, print_se3_info

try:
    from lidar_to_rgb_video import load_lidar
except ImportError:
    def load_lidar(bin_file_path):
        """Load lidar points from binary file - placeholder implementation"""
        print(f"Warning: Using placeholder load_lidar function for {bin_file_path}")
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

def return_image_with_point_cloud(
    image, image_points, intensities, global_min_i, global_max_i, colormap_name="turbo"
):
    import matplotlib
    norm = matplotlib.colors.Normalize(vmin=global_min_i, vmax=global_max_i)
    colormap = matplotlib.colormaps[colormap_name]
    colors = (colormap(norm(intensities))[:, :3] * 255).astype(np.uint8)

    out_img = image.copy()
    # initialize a blank image white with same shape as image
    out_img = np.ones_like(image) * 255
    for pt, color in zip(image_points.astype(int), colors):
        x, y = pt
        if 0 <= x < out_img.shape[1] and 0 <= y < out_img.shape[0]:
            # Convert RGB to BGR for OpenCV
            bgr_color = (int(color[2]), int(color[1]), int(color[0]))  # RGB -> BGR
            cv2.circle(out_img, (x, y), 1, bgr_color, -1)

    return out_img

def return_image_with_point_cloud_alpha(
    image, image_points, intensities, global_min_i, global_max_i, colormap_name="turbo", alpha=1.0
):
    import matplotlib
    norm = matplotlib.colors.Normalize(vmin=global_min_i, vmax=global_max_i)
    colormap = matplotlib.colormaps[colormap_name]
    colors = (colormap(norm(intensities))[:, :3] * 255).astype(np.uint8)

    out_img = image.copy()
    # initialize a blank image white with same shape as image
    out_img = np.ones_like(image) * 255
    out_img = out_img.astype(np.float32)
    for pt, color in zip(image_points.astype(int), colors):
        x, y = pt
        if 0 <= x < out_img.shape[1] and 0 <= y < out_img.shape[0]:
            # Convert RGB to BGR for OpenCV
            bgr_color = np.array([color[2], color[1], color[0]], dtype=np.float32)  # RGB -> BGR
            # Direct assignment for alpha=1.0, alpha blending for others
            if alpha == 1.0:
                out_img[y, x] = bgr_color
            else:
                out_img[y, x] = (1 - alpha) * out_img[y, x] + alpha * bgr_color

    return out_img.astype(np.uint8)

def return_image_with_point_cloud_combined(
    image, image_points, intensities, global_min_i, global_max_i, colormap_name="turbo", alpha=1.0
):
    """
    Special function for combined projection - handles alpha correctly
    """
    import matplotlib
    norm = matplotlib.colors.Normalize(vmin=global_min_i, vmax=global_max_i)
    colormap = matplotlib.colormaps[colormap_name]
    colors = (colormap(norm(intensities))[:, :3] * 255).astype(np.uint8)

    out_img = image.copy()
    # initialize a blank image white with same shape as image
    out_img = np.ones_like(image) * 255
    
    if alpha == 1.0:
        # For Robosense - use original method (no alpha blending)
        for pt, color in zip(image_points.astype(int), colors):
            x, y = pt
            if 0 <= x < out_img.shape[1] and 0 <= y < out_img.shape[0]:
                # Convert RGB to BGR for OpenCV
                bgr_color = (int(color[2]), int(color[1]), int(color[0]))  # RGB -> BGR
                cv2.circle(out_img, (x, y), 1, bgr_color, -1)
    else:
        # For Leishen - use alpha blending
        out_img = out_img.astype(np.float32)
        for pt, color in zip(image_points.astype(int), colors):
            x, y = pt
            if 0 <= x < out_img.shape[1] and 0 <= y < out_img.shape[0]:
                # Convert RGB to BGR for OpenCV
                bgr_color = np.array([color[2], color[1], color[0]], dtype=np.float32)  # RGB -> BGR
                # Alpha blending
                out_img[y, x] = (1 - alpha) * out_img[y, x] + alpha * bgr_color
        out_img = out_img.astype(np.uint8)

    return out_img

def project_lidar_points_to_image(lidar_xyz, lidar_intensities, camera_image_path, T_camera_lidar, K, dist, colormap_name="turbo"):
    """
    Project lidar points to camera image with intensity-based coloring
    
    Args:
        lidar_xyz: (N,3) array of lidar points
        lidar_intensities: (N,) array of intensity values
        camera_image_path: path to camera image
        T_camera_lidar: 4x4 transformation matrix from lidar to camera
        K: 3x3 camera intrinsic matrix
        dist: distortion coefficients
        colormap_name: matplotlib colormap name for coloring
    
    Returns:
        image_with_point_cloud: camera image with projected lidar points
    """
    points = lidar_xyz
    intensities = lidar_intensities

    points_h = np.hstack([points, np.ones((points.shape[0], 1))])
    points_cam = (T_camera_lidar @ points_h.T).T[:, :3]

    mask = points_cam[:, 2] > 0
    points_cam = points_cam[mask]
    intensities = intensities[mask]

    print(f"Projecting {len(points_cam)} points to camera image")
    print("points_cam shape:", points_cam.shape)
    print("intensities shape:", intensities.shape)

    image = cv2.imread(camera_image_path)

    # For rational polynomial distortion model, we need to provide distortion coefficients correctly
    # The dist array has 8 coefficients, but cv2.projectPoints expects them in a specific format
    dist_coeffs = np.array([dist[0], dist[1], dist[2], dist[3], dist[4], dist[5], dist[6], dist[7]], dtype=np.float64)
    
    image_points, _ = cv2.projectPoints(
        points_cam.astype(np.float64),
        rvec=np.zeros((3, 1), dtype=np.float64),
        tvec=np.zeros((3, 1), dtype=np.float64),
        cameraMatrix=K.astype(np.float64),
        distCoeffs=dist_coeffs,
    )
    image_points = image_points.squeeze()

    # Use same intensity range as other scripts for consistent coloring
    min_intensity = 0  # Same as subimage1.py, subimage2.py, subimage3.py
    max_intensity = 255  # Same as subimage1.py, subimage2.py, subimage3.py
    print(f"Intensity range: {min_intensity:.2f} to {max_intensity:.2f}")
    
    image_with_point_cloud = return_image_with_point_cloud(image, image_points, intensities, min_intensity, max_intensity, colormap_name)

    return image_with_point_cloud

def project_lidar_points_to_image_with_alpha(lidar_xyz, lidar_intensities, camera_image_path, T_camera_lidar, K, dist, colormap_name="turbo", alpha=1.0):
    """
    Project lidar points to camera image with intensity-based coloring and alpha support
    
    Args:
        lidar_xyz: (N,3) array of lidar points
        lidar_intensities: (N,) array of intensity values
        camera_image_path: path to camera image
        T_camera_lidar: 4x4 transformation matrix from lidar to camera
        K: 3x3 camera intrinsic matrix
        dist: distortion coefficients
        colormap_name: matplotlib colormap name for coloring
        alpha: transparency value (0.0 to 1.0)
    
    Returns:
        image_with_point_cloud: camera image with projected lidar points
    """
    points = lidar_xyz
    intensities = lidar_intensities

    points_h = np.hstack([points, np.ones((points.shape[0], 1))])
    points_cam = (T_camera_lidar @ points_h.T).T[:, :3]

    mask = points_cam[:, 2] > 0
    points_cam = points_cam[mask]
    intensities = intensities[mask]

    print(f"Projecting {len(points_cam)} points to camera image with alpha={alpha}")
    print("points_cam shape:", points_cam.shape)
    print("intensities shape:", intensities.shape)

    image = cv2.imread(camera_image_path)

    # For rational polynomial distortion model, we need to provide distortion coefficients correctly
    # The dist array has 8 coefficients, but cv2.projectPoints expects them in a specific format
    dist_coeffs = np.array([dist[0], dist[1], dist[2], dist[3], dist[4], dist[5], dist[6], dist[7]], dtype=np.float64)
    
    image_points, _ = cv2.projectPoints(
        points_cam.astype(np.float64),
        rvec=np.zeros((3, 1), dtype=np.float64),
        tvec=np.zeros((3, 1), dtype=np.float64),
        cameraMatrix=K.astype(np.float64),
        distCoeffs=dist_coeffs,
    )
    image_points = image_points.squeeze()

    # Use same intensity range as other scripts for consistent coloring
    min_intensity = 0  # Same as subimage1.py, subimage2.py, subimage3.py
    max_intensity = 255  # Same as subimage1.py, subimage2.py, subimage3.py
    print(f"Intensity range: {min_intensity:.2f} to {max_intensity:.2f}")
    
    image_with_point_cloud = return_image_with_point_cloud_alpha(image, image_points, intensities, min_intensity, max_intensity, colormap_name, alpha)

    return image_with_point_cloud

def overlay_ls_rs_lidar_with_camera_projection(lslidar_bin_file_path, rslidar_bin_file_path, camera_image_path,
                                             T_rs_ls, T_camera_rs, T_camera_ls, K, dist,
                                             remove_ground=True, crop_range=True, r_max=100.0, 
                                             rotation_degrees=90, ls_alpha=0.5):
    """
    Overlay two lidar point clouds (leishen and robosense) with camera projection
    Returns both the bird's eye view overlay and camera perspective projection
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
    
    # ===== CAMERA PROJECTION =====
    print("\n=== CAMERA PROJECTION ===")
    
    # Project robosense points to camera
    print("Projecting Robosense points to camera...")
    rs_camera_projection = project_lidar_points_to_image(
        rs_xyz, rs_intensities, camera_image_path, T_camera_rs, K, dist, "turbo"
    )
    
    # Project leishen points to camera (need to transform to robosense frame first, then to camera)
    print("Projecting Leishen points to camera...")
    # Use a custom function for Leishen with alpha=0.8
    ls_camera_projection = project_lidar_points_to_image_with_alpha(
        ls_xyz_in_rs, ls_intensities, camera_image_path, T_camera_rs, K, dist, "hot", alpha=0.8
    )
    
    # Create combined camera projection by literally overlaying the individual projections
    print("Creating combined camera projection...")
    
    # Start with the Robosense projection (which is perfect)
    combined_camera_projection = rs_camera_projection.copy()
    
    # DEBUG: Save the Robosense-only combined projection to see if it's correct
    cv2.imwrite("debug_combined_robosense_only.png", combined_camera_projection)
    print("DEBUG: Saved combined_robosense_only.png - this should look identical to individual Robosense")
    
    # Now overlay the Leishen points on top with alpha blending
    # We need to project Leishen points to get their image coordinates
    ls_points_h = np.hstack([ls_xyz_in_rs, np.ones((ls_xyz_in_rs.shape[0], 1))])
    ls_points_cam = (T_camera_rs @ ls_points_h.T).T[:, :3]
    ls_mask = ls_points_cam[:, 2] > 0
    ls_points_cam = ls_points_cam[ls_mask]
    ls_intensities_cam = ls_intensities[ls_mask]
    
    # Project to image coordinates
    dist_coeffs = np.array([dist[0], dist[1], dist[2], dist[3], dist[4], dist[5], dist[6], dist[7]], dtype=np.float64)
    ls_image_points, _ = cv2.projectPoints(
        ls_points_cam.astype(np.float64),
        rvec=np.zeros((3, 1), dtype=np.float64),
        tvec=np.zeros((3, 1), dtype=np.float64),
        cameraMatrix=K.astype(np.float64),
        distCoeffs=dist_coeffs,
    )
    ls_image_points = ls_image_points.squeeze()
    
    # Overlay Leishen points with alpha blending on top of the perfect Robosense projection
    # Use a custom approach that avoids overlapping with Robosense points
    import matplotlib
    norm = matplotlib.colors.Normalize(vmin=0, vmax=255)
    colormap = matplotlib.colormaps["hot"]
    colors = (colormap(norm(ls_intensities_cam))[:, :3] * 255).astype(np.uint8)

    # Create a mask to identify where Robosense points are (white background = 255, points = other values)
    robosense_mask = np.all(combined_camera_projection == 255, axis=2)  # True where background is white
    
    # Convert to float for alpha blending
    combined_camera_projection_float = combined_camera_projection.astype(np.float32)
    
    for pt, color in zip(ls_image_points.astype(int), colors):
        x, y = pt
        if 0 <= x < combined_camera_projection_float.shape[1] and 0 <= y < combined_camera_projection_float.shape[0]:
            # Only draw Leishen points where there are no Robosense points
            if robosense_mask[y, x]:  # Only if this pixel is background (white)
                # Convert RGB to BGR for OpenCV
                bgr_color = np.array([color[2], color[1], color[0]], dtype=np.float32)  # RGB -> BGR
                # Alpha blending - only affects this specific pixel
                combined_camera_projection_float[y, x] = (1 - 0.8) * combined_camera_projection_float[y, x] + 0.8 * bgr_color
    
    combined_camera_projection = combined_camera_projection_float.astype(np.uint8)
    
    # DEBUG: Save the final combined projection
    cv2.imwrite("debug_combined_final.png", combined_camera_projection)
    print("DEBUG: Saved combined_final.png - this is the final result")
    
    # ===== BIRD'S EYE VIEW OVERLAY =====
    print("\n=== BIRD'S EYE VIEW OVERLAY ===")
    
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
    
    # Create bird's eye view visualization
    fig, ax = plt.subplots(figsize=(20, 20), dpi=150)
    
    # Print intensity ranges for both lidars
    print(f"Robosense intensity range: [{np.min(rs_intensities):.2f}, {np.max(rs_intensities):.2f}]")
    print(f"Leishen intensity range: [{np.min(ls_intensities):.2f}, {np.max(ls_intensities):.2f}]")
    
    # Normalize intensities (0-255 to 0-1)
    ls_intensities_norm = ls_intensities / 255.0
    rs_intensities_norm = rs_intensities / 255.0
    
    # Plot robosense points (rainbow colors, smaller, more faded) - rotated
    ax.scatter(rs_xyz_rotated[:, 0], rs_xyz_rotated[:, 1], 
                         c=rs_intensities_norm, cmap='rainbow', s=1, alpha=0.3, 
                         edgecolors='navy', linewidths=0.2, label='Robosense')
    
    # Plot leishen points (hot colors, larger, more prominent) - rotated
    ax.scatter(ls_xyz_rotated[:, 0], ls_xyz_rotated[:, 1], 
                         c=ls_intensities_norm, cmap='hot', s=8, alpha=ls_alpha, 
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
    
    plt.tight_layout()
    # Save as PDF
    plt.savefig("subimage3_camera_perspective_birdseye.pdf", dpi=300, bbox_inches='tight', facecolor='white', format='pdf')
    
    # Convert plot to image array
    fig.canvas.draw()
    birdseye_overlay = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    birdseye_overlay = birdseye_overlay.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    # Convert RGB to BGR for OpenCV
    birdseye_overlay = cv2.cvtColor(birdseye_overlay, cv2.COLOR_RGB2BGR)
    
    plt.show()
    
    print(f"Bird's eye view overlay saved to: subimage3_camera_perspective_birdseye.pdf")
    
    return birdseye_overlay, combined_camera_projection, rs_camera_projection, ls_camera_projection

if __name__ == "__main__":
    # Example usage
    target_asset_folder = "/home/samqiao/ASRL/fomo-public-sdk/raw_fomo_rosbags/red_2024-11-21-10-34/target_assets"
    
    lslidar_bin_file_path = os.path.join(target_asset_folder, "lslidar","1732203284099476.bin")
    rslidar_bin_file_path = os.path.join(target_asset_folder, "rslidar","1732203284100439.bin")
    camera_image_path = os.path.join(target_asset_folder, "zedx_left","1732203284156539.png")
    
    # Check if files exist
    if not os.path.exists(lslidar_bin_file_path):
        print(f"Error: Leishen lidar file not found: {lslidar_bin_file_path}")
        exit(1)
    if not os.path.exists(rslidar_bin_file_path):
        print(f"Error: Robosense lidar file not found: {rslidar_bin_file_path}")
        exit(1)
    if not os.path.exists(camera_image_path):
        print(f"Error: Camera image file not found: {camera_image_path}")
        exit(1)
    
    # Get transformation matrices
    T_rs_ls = get_se3_extrinsic("leishen", "robosense")
    print_se3_info(T_rs_ls, "leishen", "robosense")
    
    T_camera_rs = get_se3_extrinsic("robosense", "zedx_left")
    print_se3_info(T_camera_rs, "robosense", "zedx_left")
    
    # Camera intrinsics
    K = np.array([
        [738.0576171875, 0, 936.5038452148438],
        [0, 738.0576171875, 594.3272705078125],
        [0, 0, 1]
    ])
    
    # Distortion coefficients
    dist = np.array([0, 0, 0, 0, 0, 0, 0, 0])
    
    if T_rs_ls is not None and T_camera_rs is not None:
        birdseye_overlay, combined_camera_projection, rs_camera_projection, ls_camera_projection = overlay_ls_rs_lidar_with_camera_projection(
            lslidar_bin_file_path, rslidar_bin_file_path, camera_image_path,
            T_rs_ls, T_camera_rs, None, K, dist,
            remove_ground=False, crop_range=False, r_max=80.0,
            rotation_degrees=0
        )
        
        # Save all outputs
        cv2.imwrite("subimage3_camera_perspective_birdseye.png", birdseye_overlay, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        cv2.imwrite("subimage3_camera_perspective_combined.png", combined_camera_projection, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        cv2.imwrite("subimage3_camera_perspective_robosense.png", rs_camera_projection, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        cv2.imwrite("subimage3_camera_perspective_leishen.png", ls_camera_projection, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        
        # Also save as PDF using matplotlib
        
        # Save combined camera projection as PDF
        fig, ax = plt.subplots(figsize=(20, 20), dpi=150)
        ax.imshow(cv2.cvtColor(combined_camera_projection, cv2.COLOR_BGR2RGB))
        ax.axis('off')
        plt.tight_layout()
        plt.savefig("subimage3_camera_perspective_combined.pdf", dpi=300, bbox_inches='tight', facecolor='white', format='pdf')
        plt.close()
        
        print("All visualizations saved successfully!")
        print("- Bird's eye view: subimage3_camera_perspective_birdseye.png/pdf")
        print("- Combined camera projection: subimage3_camera_perspective_combined.png/pdf")
        print("- Robosense camera projection: subimage3_camera_perspective_robosense.png")
        print("- Leishen camera projection: subimage3_camera_perspective_leishen.png")
        
    else:
        print("Failed to get transformation matrices")
