from pathlib import Path
import cv2
import os
import json
import numpy as np
from scipy.spatial.transform import Rotation as R
# Import specific functions instead of everything to avoid triggering main()
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from radar_lidar_calibration.radar_lidar_cali_utils import *

# Import specific functions instead of everything to avoid triggering main()
try:
    from lidar_to_rgb_video import load_lidar
except ImportError:
    def load_lidar(bin_file_path):
        """Load lidar points from binary file - placeholder implementation"""
        print(f"Warning: Using placeholder load_lidar function for {bin_file_path}")
        # Return dummy data for testing
        return np.random.rand(1000, 4)  # 1000 points with x,y,z,intensity

# so this script will generate 3  rows of images
# the first row
## lidar points projected on top of a camera image 
# the second row 
# # will be radar lidar point clouds overlay
# the third row
## the two lidar point clouds overlay
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

def load_transform_tree(json_file_path):
    """Load the transform tree from JSON file"""
    with open(json_file_path, 'r') as f:
        return json.load(f)

def quaternion_to_rotation_matrix(qx, qy, qz, qw):
    """Convert quaternion to rotation matrix"""
    # Create scipy rotation object from quaternion (x, y, z, w)
    rotation = R.from_quat([qx, qy, qz, qw])
    return rotation.as_matrix()

def position_to_translation(px, py, pz):
    """Convert position to translation vector"""
    return np.array([px, py, pz])

def create_se3_matrix(rotation_matrix, translation_vector):
    """Create SE3 transformation matrix from rotation matrix and translation vector"""
    se3 = np.eye(4)
    se3[:3, :3] = rotation_matrix
    se3[:3, 3] = translation_vector
    return se3

def find_transform_path(transform_tree, source_frame, target_frame):
    """Find the path from source to target frame using BFS"""
    # Build adjacency list
    graph = {}
    for transform in transform_tree:
        from_frame = transform['from']
        to_frame = transform['to']
        
        if from_frame not in graph:
            graph[from_frame] = []
        if to_frame not in graph:
            graph[to_frame] = []
            
        graph[from_frame].append((to_frame, transform, False))  # False indicates forward transform
        graph[to_frame].append((from_frame, transform, True))  # True indicates reverse transform
    
    # BFS to find path
    from collections import deque
    queue = deque([(source_frame, [])])
    visited = {source_frame}
    
    while queue:
        current_frame, path = queue.popleft()
        
        if current_frame == target_frame:
            return path
        
        for neighbor, transform_data, is_reverse in graph.get(current_frame, []):
            if neighbor not in visited:
                visited.add(neighbor)
                new_path = path + [(neighbor, transform_data, is_reverse)]
                queue.append((neighbor, new_path))
    
    return None  # No path found

def get_se3_extrinsic(source_frame, target_frame, transform_json_path="transform.json"):
    """
    Get SE3 extrinsic pose from source frame to target frame
    
    Args:
        source_frame (str): Source frame name
        target_frame (str): Target frame name  
        transform_json_path (str): Path to transform.json file
        
    Returns:
        numpy.ndarray: 4x4 SE3 transformation matrix from source to target
        None: If no path found between frames
    """
    # Load transform tree
    transform_tree = load_transform_tree(transform_json_path)
    
    # Find path from source to target
    path = find_transform_path(transform_tree, source_frame, target_frame)
    
    if path is None:
        print(f"No path found from {source_frame} to {target_frame}")
        return None
    
    # Start with identity matrix
    cumulative_transform = np.eye(4)
    
    # Apply each transformation in the path
    for frame_name, transform_data, is_reverse in path:
        # Extract position and orientation
        pos = transform_data['position']
        orient = transform_data['orientation']
        
        # Convert to rotation matrix and translation vector
        rotation_matrix = quaternion_to_rotation_matrix(
            orient['x'], orient['y'], orient['z'], orient['w']
        )
        translation_vector = position_to_translation(
            pos['x'], pos['y'], pos['z']
        )
        
        # Create SE3 matrix
        se3_matrix = create_se3_matrix(rotation_matrix, translation_vector)
        
        if is_reverse:
            # For reverse transforms, we need the inverse
            se3_matrix = np.linalg.inv(se3_matrix)
        
        # Compose with cumulative transform
        cumulative_transform = cumulative_transform @ se3_matrix
    
    return cumulative_transform

def print_se3_info(se3_matrix, source_frame, target_frame):
    """Print human-readable information about the SE3 transformation"""
    if se3_matrix is None:
        print(f"Failed to find transformation from {source_frame} to {target_frame}")
        return
    
    # Extract rotation and translation
    rotation_matrix = se3_matrix[:3, :3]
    translation_vector = se3_matrix[:3, 3]
    
    # Convert rotation matrix to euler angles (for readability)
    rotation_obj = R.from_matrix(rotation_matrix)
    euler_angles = rotation_obj.as_euler('xyz', degrees=True)
    
    print(f"\nTransformation from {source_frame} to {target_frame}:")
    print(f"Translation (x, y, z): [{translation_vector[0]:.6f}, {translation_vector[1]:.6f}, {translation_vector[2]:.6f}]")
    print(f"Rotation (roll, pitch, yaw) in degrees: [{euler_angles[0]:.6f}, {euler_angles[1]:.6f}, {euler_angles[2]:.6f}]")
    print(f"SE3 Matrix:")
    print(se3_matrix)



def return_image_with_point_cloud(
    image, image_points, intensities, global_min_i, global_max_i
):
    import matplotlib
    norm = matplotlib.colors.Normalize(vmin=global_min_i, vmax=global_max_i)
    colormap = matplotlib.colormaps["turbo"]
    colors = (colormap(norm(intensities))[:, :3] * 255).astype(np.uint8)

    out_img = image.copy()
    for pt, color in zip(image_points.astype(int), colors):
        x, y = pt
        if 0 <= x < out_img.shape[1] and 0 <= y < out_img.shape[0]:
            cv2.circle(out_img, (x, y), 1, tuple(int(c) for c in color), -1)

    return out_img

def project_lidar_points_to_image(lidar_bin_file_path, camera_image_path, T_camera_lidar, K, dist):
    # I will use the rslidar
    lidar_points = load_lidar(lidar_bin_file_path)

    points = lidar_points[:, :3]
    intensities = lidar_points[:, 3]

    points_h = np.hstack([points, np.ones((points.shape[0], 1))])
    points_cam = (T_camera_lidar @ points_h.T).T[:, :3]

    mask = points_cam[:, 2] > 0
    points_cam = points_cam[mask]
    intensities = intensities[mask]

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

    # Calculate better intensity range for more visible colors
    min_intensity = np.percentile(intensities, 5)  # Use 5th percentile instead of 0
    max_intensity = np.percentile(intensities, 95)  # Use 95th percentile instead of 255
    print(f"Intensity range: {min_intensity:.2f} to {max_intensity:.2f}")
    
    image_with_point_cloud = return_image_with_point_cloud(image, image_points, intensities, min_intensity, max_intensity)

    return image_with_point_cloud

def project_radar_points_to_lidar(radar_image_path, rslidar_bin_file_path, T_lidar_radar):
    # CIR-304H radar parameters
    radar_resolution = 0.043809514
    encoder_size = 5600
    num_of_range_bins = 6848
    max_range = 300

    radar_image = cv2.imread(radar_image_path,cv2.IMREAD_GRAYSCALE)
    byte_array = np.array(radar_image)
    timestamps = byte_array[:, :8].view(np.uint64) * 1e-3
    azimuths = byte_array[:, 8:10].view(np.uint16) / float(encoder_size) * 2 * np.pi
    polar = byte_array[:, 11:].astype(np.float32) / 255.0

    import torch
    import torchvision
    import matplotlib.pyplot as plt
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
    targets = KPeaks(polar_intensity.numpy(),minr=5,maxr=80,res=radar_resolution, K=10, static_threshold=0.30)

    radar_pts = []
    import math
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
    radar_intensities = radar_pts[:, 3]  # intensity values

    # Load lidar points
    lidar_points = load_lidar(rslidar_bin_file_path)
    lidar_xyz = lidar_points[:, :3]
    lidar_intensities = lidar_points[:, 3]

    # Transform radar points to lidar frame
    radar_xyz_homogeneous = np.hstack([radar_xyz, np.ones((radar_xyz.shape[0], 1))])
    radar_xyz_in_lidar = (T_lidar_radar @ radar_xyz_homogeneous.T).T

    print(f'Lidar points shape: {lidar_xyz.shape}')
    print(f'Radar points shape: {radar_xyz_in_lidar.shape}')
    print(f'Lidar intensity range: {np.min(lidar_intensities):.2f} to {np.max(lidar_intensities):.2f}')
    print(f'Radar intensity range: {np.min(radar_intensities):.2f} to {np.max(radar_intensities):.2f}')

    return lidar_xyz, lidar_intensities, radar_xyz_in_lidar, radar_intensities

def visualize_lidar_radar_overlay_simple(lidar_xyz, lidar_intensities, radar_xyz, radar_intensities, save_path="subimage2.png", remove_ground=True, crop_range=True, r_max=100.0):
    """
    Visualization of radar points overlaid on lidar points with intensity colormaps
    Returns the image as subimage2
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    fig, ax = plt.subplots(figsize=(20, 20), dpi=150)
    
    # Remove ground plane if requested
    if remove_ground:
        # Improved ground plane removal using percentile-based threshold
        z_values = lidar_xyz[:, 2]
        z_min = np.min(z_values)
        z_max = np.max(z_values)
        z_median = np.median(z_values)
        
        # Use 10th percentile as ground threshold (more aggressive)
        ground_threshold = np.percentile(z_values, 10)
        
        # Alternative: use median - 0.5m as threshold
        # ground_threshold = z_median - 0.5
        
        # Filter out ground points
        non_ground_mask = lidar_xyz[:, 2] > ground_threshold
        lidar_xyz = lidar_xyz[non_ground_mask]
        lidar_intensities = lidar_intensities[non_ground_mask]
        
        print(f"Ground plane removal: Z range [{z_min:.2f}, {z_max:.2f}], threshold={ground_threshold:.2f}")
        print(f"Removed {np.sum(~non_ground_mask)} ground points, kept {np.sum(non_ground_mask)} points")
    
    # Crop lidar range if requested
    if crop_range:
        lidar_xyz, range_mask = crop_lidar_by_range(lidar_xyz, r_max=r_max, r_min=0.0, use_xy=True, return_mask=True)
        lidar_intensities = lidar_intensities[range_mask]
        print(f"Range cropping: kept {np.sum(range_mask)} points within {r_max}m range")
    
    # Normalize lidar intensities (0-255 to 0-1)
    lidar_intensities_norm = lidar_intensities / 255.0
    
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
    
    # Plot lidar points with intensity colormap (more faded)
    scatter1 = ax.scatter(lidar_xyz_rotated[:, 0], lidar_xyz_rotated[:, 1], 
                         c=lidar_intensities_norm, cmap='rainbow', s=1, alpha=0.3, label='Robosense')
    
    # Plot radar points with better styling
    scatter2 = ax.scatter(radar_xyz_rotated[:, 0], radar_xyz_rotated[:, 1], 
                         c=radar_intensities, cmap='Reds', s=30, alpha=0.9, 
                         edgecolors='darkred', linewidths=0.5, label='Navtech')
    
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
    pdf_path = save_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight', facecolor='white', format='pdf')
    
    # Convert plot to image array
    fig.canvas.draw()
    subimage2 = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    subimage2 = subimage2.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    # Convert RGB to BGR for OpenCV
    subimage2 = cv2.cvtColor(subimage2, cv2.COLOR_RGB2BGR)
    
    plt.show()
    
    print(f"Intensity overlay visualization saved to: {pdf_path}")
    return subimage2



if __name__ == "__main__":
    target_asset_folder = "/home/samqiao/ASRL/fomo-public-sdk/raw_fomo_rosbags/red_2024-11-21-10-34/target_assets"

    rslidar_bin_file_path = os.path.join(target_asset_folder, "rslidar","1732203284100439.bin")
    zedx_left_image_path = os.path.join(target_asset_folder, "zedx_left","1732203284156539.png")
    
    # Check if files exist
    if not os.path.exists(rslidar_bin_file_path):
        print(f"Error: Lidar file not found: {rslidar_bin_file_path}")
        exit(1)
    if not os.path.exists(zedx_left_image_path):
        print(f"Error: Image file not found: {zedx_left_image_path}")
        exit(1)
    
    # lets do the lidar points projected to camera image first
    source_frame = "robosense"
    target_frame = "zedx_left"
    # intrincis of the zedx camera K and dist
    K = np.array([
    [738.0576171875, 0, 936.5038452148438],
    [0, 738.0576171875, 594.3272705078125],
    [0, 0, 1]
    ])

    # Distortion coefficients
    dist = np.array([0, 0, 0, 0, 0, 0, 0, 0]) 
    T_camera_lidar = get_se3_extrinsic(source_frame, target_frame)
    print_se3_info(T_camera_lidar, source_frame, target_frame)  
    
    if T_camera_lidar is not None:
        subimage1 = project_lidar_points_to_image(rslidar_bin_file_path, zedx_left_image_path, T_camera_lidar, K, dist)
        # cv2.imshow("subimage1", subimage1)
        cv2.imwrite("subimage1.png", subimage1)
        # Also save as PDF using matplotlib
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(20, 20), dpi=150)
        ax.imshow(cv2.cvtColor(subimage1, cv2.COLOR_BGR2RGB))
        ax.axis('off')
        plt.tight_layout()
        plt.savefig("subimage1.pdf", dpi=300, bbox_inches='tight', facecolor='white', format='pdf')
        plt.close()
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    else:
        print("Failed to get transformation matrix")

    # okay first one is done next is the radar and lidar overlay
    radar_image_path = os.path.join(target_asset_folder, "navtech","1732203284153343.png")
    T_lidar_radar = get_se3_extrinsic("navtech", "robosense")
    print_se3_info(T_lidar_radar, "navtech", "robosense")
    
    # Project radar points to lidar frame and visualize overlay
    if T_lidar_radar is not None:
        lidar_xyz, lidar_intensities, radar_xyz_in_lidar, radar_intensities = project_radar_points_to_lidar(
            radar_image_path, rslidar_bin_file_path, T_lidar_radar
        )
        subimage2 = visualize_lidar_radar_overlay_simple(lidar_xyz, lidar_intensities, radar_xyz_in_lidar, radar_intensities, remove_ground=True, crop_range=True, r_max=80.0)
        # Save with PNG format (lossless) and high quality
        cv2.imwrite("subimage2.png", subimage2, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        # Also save as PDF using matplotlib
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(20, 20), dpi=150)
        ax.imshow(cv2.cvtColor(subimage2, cv2.COLOR_BGR2RGB))
        ax.axis('off')
        plt.tight_layout()
        plt.savefig("subimage2.pdf", dpi=300, bbox_inches='tight', facecolor='white', format='pdf')
        plt.close()
    else:
        print("Failed to get radar to lidar transformation")

    #     # lastly I will visualize all two lidar point clouds overlay
    from subimage3 import overlay_ls_rs_lidar
    
    lslidar_bin_file_path = os.path.join(target_asset_folder, "lslidar","1732203284099476.bin")
    T_rs_ls = get_se3_extrinsic("leishen", "robosense")
    print_se3_info(T_rs_ls, "leishen", "robosense")
    subimage3 = overlay_ls_rs_lidar(lslidar_bin_file_path, rslidar_bin_file_path, T_rs_ls, remove_ground=True, crop_range=True, r_max=80.0)
    # Also save as PNG
    cv2.imwrite("subimage3.png", subimage3, [cv2.IMWRITE_PNG_COMPRESSION, 1])

    # with open("/home/samqiao/ASRL/fomo-public-sdk/extrinics_visualization/transform.json", "r") as f:
    #     transforms = json.load(f)

    # T_rslidar_zedxleft = get_T("zedx_left", "robosense",transforms)
    # T_zedxleft_rslidar = np.linalg.inv(T_rslidar_zedxleft)
    # print_se3_info(T_rslidar_zedxleft, "zedx_left", "robosense")
    # print_se3_info(T_zedxleft_rslidar, "robosense", "zedx_left")

