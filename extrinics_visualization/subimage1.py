import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import sys

# Add paths for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import common utilities
from transform_utils import get_se3_extrinsic, print_se3_info

# Import lidar loading function
try:
    from lidar_to_rgb_video import load_lidar
except ImportError:
    def load_lidar(bin_file_path):
        """Load lidar points from binary file - placeholder implementation"""
        print(f"Warning: Using placeholder load_lidar function for {bin_file_path}")
        return np.random.rand(1000, 4)  # 1000 points with x,y,z,intensity


def return_image_with_point_cloud(
    image, image_points, intensities, global_min_i, global_max_i
):
    import matplotlib
    norm = matplotlib.colors.Normalize(vmin=global_min_i, vmax=global_max_i)
    # print a few norm intensities

    colormap = matplotlib.colormaps["turbo"]
    colors = (colormap(norm(intensities))[:, :3] * 255).astype(np.uint8)

    out_img = image.copy()
    for pt, color in zip(image_points.astype(int), colors):
        x, y = pt
        if 0 <= x < out_img.shape[1] and 0 <= y < out_img.shape[0]:
            # Convert RGB to BGR for OpenCV
            bgr_color = (int(color[2]), int(color[1]), int(color[0]))  # RGB -> BGR
            cv2.circle(out_img, (x, y), 1, bgr_color, -1)

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
    
    # Debug: Print intensity statistics to compare with subimage2.py
    print(f"Lidar intensities in subimage1.py:")
    print(f"  - Shape: {intensities.shape}")
    print(f"  - Min: {np.min(intensities):.2f}")
    print(f"  - Max: {np.max(intensities):.2f}")
    print(f"  - Mean: {np.mean(intensities):.2f}")
    print(f"  - First 10 values: {intensities[0:10]}")

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
    min_intensity = 0 # np.percentile(intensities, 5)  # Use 5th percentile instead of 0
    max_intensity = 255 #np.percentile(intensities, 95)  # Use 95th percentile instead of 255
    # print(f"Intensity range: {min_intensity:.2f} to {max_intensity:.2f}")
    
    image_with_point_cloud = return_image_with_point_cloud(image, image_points, intensities, min_intensity, max_intensity)

    return image_with_point_cloud

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
