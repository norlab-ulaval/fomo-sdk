import os
import cv2
import numpy as np
import json
from pathlib import Path
from scipy.spatial.transform import Rotation
import matplotlib.colors
from concurrent.futures import ProcessPoolExecutor
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Project lidar points on images")
# parser.add_argument(
#     "--trajectory_path",
#     type=str,
#     required=True,
#     help="Path to folder containing subfolders rslidar128 and zedx_side",
# )
# parser.add_argument(
#     "--camera",
#     type=str,
#     required=False,
#     choices=["left", "right"],
#     default="left",
# )
# parser.add_argument(
#     "--lidar",
#     type=str,
#     required=False,
#     choices=["rs", "ls"],
#     default="rs",
# )
# args = parser.parse_args()

# img_dir = Path(args.trajectory_path) / f"zedx_{args.camera}"
# img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")])

# lidar_dir = Path(args.trajectory_path) / f"{args.lidar}lidar128"
# lidar_files = sorted([f for f in os.listdir(lidar_dir) if f.endswith(".bin")])
# lidar_timestamps = [int(Path(f).stem) for f in lidar_files]

# with open("data/calib/transforms.json", "r") as f:
#     transforms = json.load(f)


def get_T(from_node, to_node,transforms):
    """
    Returns a 4x4 transformation matrix from 'from_node' to 'to_node'
    using data loaded from the JSON file.
    """
    # Find the matching transform
    for tf in transforms:
        if tf["from"] == from_node and tf["to"] == to_node:
            t = np.array(
                [tf["position"]["x"], tf["position"]["y"], tf["position"]["z"]]
            )
            q = np.array(
                [
                    tf["orientation"]["x"],
                    tf["orientation"]["y"],
                    tf["orientation"]["z"],
                    tf["orientation"]["w"],
                ]
            )
            R = Rotation.from_quat(q).as_matrix()
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = t
            return T
    raise ValueError(f"No transform found from '{from_node}' to '{to_node}'")


def get_calibration_informations():
    with open(f"data/calib/zedx_{args.camera}.json", "r") as f:
        calib = json.load(f)

    K = np.array(calib["k"])
    K = K.reshape(3, 3)
    dist = np.array(calib["d"])

    T_rslidar_zedxleft = get_T("zedx_left", "robosense")

    T_zedxleft_rslidar = np.linalg.inv(T_rslidar_zedxleft)

    T_camera_lidar = T_zedxleft_rslidar
    if args.lidar == "ls":
        T_rslidar_lslidar = get_T("leishen", "robosense")
        T_zedxleft_lslidar = T_zedxleft_rslidar @ T_rslidar_lslidar
        T_camera_lidar = T_zedxleft_lslidar

    if args.camera == "right":
        T_zedxright_zedxleft = get_T("zedx_left", "zedx_right")
        T_camera_lidar = T_zedxright_zedxleft @ T_camera_lidar

    return K, dist, T_camera_lidar


def process_image(args):
    i, img_file, K, dist, T_lidar_camera_inv, global_min_i, global_max_i = args

    percent = 100 * i / len(img_files)
    print(f"\r{percent:.2f}% done", end="")

    img_timestamp = int(Path(img_file).stem)
    image = cv2.imread(os.path.join(img_dir, img_file))
    closest_idx = np.argmin([abs(ts - img_timestamp) for ts in lidar_timestamps])
    lidar_file = lidar_files[closest_idx]
    lidar_path = os.path.join(lidar_dir, lidar_file)

    lidar_points = load_lidar(lidar_path)
    points = lidar_points[:, :3]
    intensities = lidar_points[:, 3]

    points_h = np.hstack([points, np.ones((points.shape[0], 1))])
    points_cam = (T_lidar_camera_inv @ points_h.T).T[:, :3]

    mask = points_cam[:, 2] > 0
    points_cam = points_cam[mask]
    intensities = intensities[mask]

    image_points, _ = cv2.projectPoints(
        points_cam,
        rvec=np.zeros((3, 1)),
        tvec=np.zeros((3, 1)),
        cameraMatrix=K,
        distCoeffs=dist,
    )
    image_points = image_points.squeeze()

    save_image_with_point_cloud(
        image, image_points, intensities, img_timestamp, global_min_i, global_max_i
    )


def load_lidar(path):
    dtype = np.dtype(
        [
            ("x", np.float32),
            ("y", np.float32),
            ("z", np.float32),
            ("i", np.float32),
            ("r", np.uint16),
            ("t", np.uint64),
        ]
    )
    points = np.fromfile(path, dtype=dtype)
    arr = np.zeros((points.shape[0], 6), dtype=np.float64)
    arr[:, 0] = points["x"]
    arr[:, 1] = points["y"]
    arr[:, 2] = points["z"]
    arr[:, 3] = points["i"]
    arr[:, 4] = points["r"]
    arr[:, 5] = points["t"] * 1e-6
    return arr


def save_image_with_point_cloud(
    image, image_points, intensities, img_timestamp, global_min_i, global_max_i
):
    norm = matplotlib.colors.Normalize(vmin=global_min_i, vmax=global_max_i)
    colormap = matplotlib.colormaps["turbo"]
    colors = (colormap(norm(intensities))[:, :3] * 255).astype(np.uint8)

    out_img = image.copy()
    for pt, color in zip(image_points.astype(int), colors):
        x, y = pt
        if 0 <= x < out_img.shape[1] and 0 <= y < out_img.shape[0]:
            cv2.circle(out_img, (x, y), 1, tuple(int(c) for c in color), -1)

    out_path = Path(args.trajectory_path) / "lidar_to_camera_video/images/"
    os.makedirs(out_path, exist_ok=True)
    out_file = os.path.join(out_path, f"{img_timestamp}.png")
    cv2.imwrite(out_file, out_img)

    return out_path


def export_images_to_video(folder):
    image_folder = folder / "images"
    video_name = folder / "lidar_to_camara_video.mp4"

    images = sorted(
        [
            img
            for img in os.listdir(image_folder)
            if img.endswith((".png", ".jpg", ".jpeg"))
        ]
    )

    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(video_name, fourcc, 30, (width, height))

    for i in tqdm(range(0, len(images))):
        img = images[i]
        frame = cv2.imread(os.path.join(image_folder, img))
        video.write(frame)

    video.release()
    cv2.destroyAllWindows()


def get_global_intensity_range(lidar_dir, lidar_files):
    min_i, max_i = float("inf"), float("-inf")
    for f in tqdm(lidar_files, desc="Scanning intensities"):
        arr = load_lidar(os.path.join(lidar_dir, f))
        intensities = arr[:, 3]
        min_i = min(min_i, intensities.min())
        max_i = max(max_i, intensities.max())
    return min_i, max_i


# if __name__ == "__main__":
#     K, dist, T_lidar_camera_inv = get_calibration_informations()

#     global_min_i, global_max_i = get_global_intensity_range(lidar_dir, lidar_files)

#     print(f"Adding point clouds to {len(img_files)} images")
#     with ProcessPoolExecutor(max_workers=os.cpu_count() // 2) as executor:
#         executor.map(
#             process_image,
#             [
#                 (i, img_file, K, dist, T_lidar_camera_inv, global_min_i, global_max_i)
#                 for i, img_file in enumerate(img_files)
#             ],
#         )

#     print("\nExporting all images to mp4 video")
#     path = Path(args.trajectory_path) / "lidar_to_camera_video"
#     export_images_to_video(path)