import cv2
from rosbags.typesys.stores.latest import sensor_msgs__msg__Image as Image
from rosbags.image import message_to_cvimage
import numpy as np
import fomo_sdk.common.naming as naming
import pandas as pd
from pathlib import Path
import json
import matplotlib


def rgb_to_bayer_bggr8(img_rgb: np.ndarray) -> np.ndarray:
    h, w, _ = img_rgb.shape
    bayer = np.empty((h, w), np.uint8)

    (b, g, r) = cv2.split(img_rgb)

    bayer[0::2, 0::2] = b[0::2, 0::2]  # B
    bayer[0::2, 1::2] = g[0::2, 1::2]  # G
    bayer[1::2, 0::2] = g[1::2, 0::2]  # G
    bayer[1::2, 1::2] = r[1::2, 1::2]  # R
    return bayer


def lower_image_resolution(msg: Image, scale_factor=0.25, is_basler=False) -> Image:
    image = message_to_cvimage(msg)

    if is_basler:
        image = cv2.cvtColor(image, cv2.COLOR_BayerBG2RGB)

    resized_image = cv2.resize(
        image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA
    )
    if is_basler:
        resized_image = rgb_to_bayer_bggr8(resized_image)
    step = 0
    if msg.encoding == "bgra8":
        step = resized_image.shape[1] * 4
    elif msg.encoding == "bayer_bggr8":
        step = resized_image.shape[1] * 1
    elif msg.encoding == "bgr8":
        step = resized_image.shape[1] * 3
    elif msg.encoding == "rgb8":
        step = resized_image.shape[1] * 3
    elif msg.encoding == "mono8":
        step = resized_image.shape[1] * 1
    else:
        raise ValueError(f"Unsupported encoding: {msg.encoding}")
    new_msg = Image(
        header=msg.header,
        height=resized_image.shape[0],
        width=resized_image.shape[1],
        encoding=msg.encoding,
        is_bigendian=msg.is_bigendian,
        step=step,
        data=resized_image.flatten(),
    )
    return new_msg


def load_fomo_image(
    dataset_base_path: str, deployment: str, trajectory: str, num_of_files: int = None
) -> []:
    path = naming.construct_path(dataset_base_path, deployment, trajectory)
    if list(path.glob("*/.mcap")):
        raise NotImplementedError("Can't load image data from mcap files yet")

    has_zedx_left = len(list(path.glob("zedx_left/"))) > 0
    has_zedx_right = len(list(path.glob("zedx_right/"))) > 0

    if not has_zedx_left and not has_zedx_right:
        raise ValueError("No image data found in the dataset.")

    images_left = []
    images_right = []
    if has_zedx_left:
        for i, filename in enumerate(sorted(list(path.glob("zedx_left/*.png")))):
            if i >= num_of_files:
                break
            images_left.append(cv2.imread(str(filename), cv2.COLOR_BGR2RGB))

    if has_zedx_right:
        for i, filename in enumerate(sorted(list(path.glob("zedx_right/*.png")))):
            if i >= num_of_files:
                break
            images_right.append(cv2.imread(str(filename), cv2.COLOR_BGR2RGB))
    images = [
        (images_left[i], images_right[i])
        for i in range(min(len(images_left), len(images_right)))
    ]
    return images


def project_lidar_on_image(
    image: np.ndarray, lidar_data: pd.DataFrame, intrinsic_file: Path
) -> np.ndarray:
    points = lidar_data[["x", "y", "z"]].to_numpy()
    intensities = lidar_data["i"].to_numpy()

    mask = points[:, 2] > 0
    points = points[mask]
    intensities = intensities[mask]

    K, dist = get_intrinsics(intrinsic_file)
    image_points, _ = cv2.projectPoints(
        points,
        rvec=np.zeros((3, 1)),
        tvec=np.zeros((3, 1)),
        cameraMatrix=K,
        distCoeffs=dist,
    )
    image_points = image_points.squeeze()

    norm = matplotlib.colors.Normalize(vmin=intensities.min(), vmax=intensities.max())
    colormap = matplotlib.colormaps["turbo"]
    colors = (colormap(norm(intensities))[:, :3] * 255).astype(np.uint8)

    out_img = image.copy()
    for pt, color in zip(image_points.astype(int), colors):
        x, y = pt
        if 0 <= x < out_img.shape[1] and 0 <= y < out_img.shape[0]:
            cv2.circle(out_img, (x, y), 1, tuple(int(c) for c in color), -1)
    return out_img


def get_intrinsics(intrinsic_file: Path = "data/calib/zedx_left.json"):
    with open(intrinsic_file, "r") as f:
        calib = json.load(f)

    K = np.array(calib["k"])
    K = K.reshape(3, 3)
    dist = np.array(calib["d"])

    return K, dist


def create_no_data_image(height: int = 480, width: int = 640):
    """Create a black image with black border and 'N/A' text in center"""
    # Create black image
    img = 255 * np.ones((height, width, 3), dtype=np.uint8)

    cv2.rectangle(img, (0, 0), (width - 1, height - 1), (0, 0, 0), 4)

    # Add "N/A" text in the center
    text = "N/A"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    thickness = 3

    # Get text size to center it
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, font_scale, thickness
    )

    # Calculate position to center the text
    x = (width - text_width) // 2
    y = (height + text_height) // 2

    cv2.putText(img, text, (x, y), font, font_scale, (0, 0, 0), thickness)

    return img
