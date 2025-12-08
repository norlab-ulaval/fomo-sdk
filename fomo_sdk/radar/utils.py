import numpy as np
import cv2
from pathlib import Path
from rosbags.image import message_to_cvimage
import fomo_sdk.common.naming as naming

from rosbags.typesys.stores.latest import (
    std_msgs__msg__Header as Header,
    sensor_msgs__msg__Image as Image,
)

MIN_RANGE = 0.5
RADAR_RESOLUTION = 0.0438
RADAR_RANGE_BINS = 6848
ENDODER_SIZE = 5595


def polar_to_cartesian(
    fft_data,
    azimuths,
    cart_resolution=0.2384,
    cart_pixel_width=1024,
    interpolate_crossover=True,
    fix_wobble=False,
) -> np.ndarray:
    if (cart_pixel_width % 2) == 0:
        cart_min_range = (cart_pixel_width / 2 - 0.5) * cart_resolution
    else:
        cart_min_range = cart_pixel_width // 2 * cart_resolution

    coords = np.linspace(
        -cart_min_range, cart_min_range, cart_pixel_width, dtype=np.float32
    )
    Y, X = np.meshgrid(coords, -1 * coords)
    sample_range = np.sqrt(Y * Y + X * X)
    sample_angle = np.arctan2(Y, X)
    sample_angle += (sample_angle < 0).astype(np.float32) * 2.0 * np.pi

    azimuth_step = (azimuths[-1] - azimuths[0]) / (azimuths.shape[0] - 1)
    sample_u = (sample_range - RADAR_RESOLUTION / 2) / RADAR_RESOLUTION
    sample_v = (sample_angle - azimuths[0]) / azimuth_step

    if fix_wobble:
        M = azimuths.shape[0]
        c3 = np.searchsorted(azimuths.squeeze(), sample_angle.squeeze())
        c3[c3 == M] -= 1
        c2 = c3 - 1
        c2[c2 < 0] += 1
        a3 = azimuths[c3]
        delta = (
            (sample_angle.squeeze() - a3)
            * (sample_angle.squeeze() < 0)
            * (c3 > 0)
            / (a3 - azimuths[c2] + 1e-14)
        )
        sample_v = (c3 + delta).astype(np.float32)

    sample_u[sample_u < 0] = 0

    if interpolate_crossover:
        fft_data = np.concatenate((fft_data[-1:], fft_data, fft_data[:1]), 0)
        sample_v += 1

    polar_to_cart_warp = np.stack((sample_u, sample_v), -1)
    return cv2.remap(fft_data, polar_to_cart_warp, None, cv2.INTER_LINEAR)


def write_video(imgs, output_path, frame_size, frame_rate=60.0):
    codec = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_path, codec, frame_rate, frame_size)

    for img in imgs:
        img = cv2.resize(img, frame_size)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        video_writer.write(img)

    video_writer.release()


def cvimage_to_message(cv_image, header: Header):
    # Create a ROS 2 Image message
    radar_image_flat = cv_image.astype(np.uint8).flatten()
    height, width = cv_image.shape
    ros_image_msg = Image(
        header,
        height,
        width,
        "mono8",  # "mono8" for 8-bit grayscale
        False,
        width,
        np.array(radar_image_flat, dtype=np.uint8),
        # np.asarray(cv_image, dtype=np.uint8),
    )
    # ros_image_msg.data = np.array(cv_image).tobytes()
    return ros_image_msg


def extract_radar_info(cv_image):
    timestamps = cv_image[:, :8].view(np.int64).flatten()
    encoders = cv_image[:, 8:10].view(np.uint16).flatten()
    img = cv_image[:, 11:]
    return img, timestamps, encoders


def get_radar_img_msg(msg):
    polar_img = message_to_cvimage(msg.b_scan_img)
    azimuths = (
        msg.encoder_values / ENDODER_SIZE * 2 * np.pi
    )  # these are the end bin number for LAVAL radar
    radar_image = polar_to_cartesian(
        fft_data=polar_img,
        azimuths=azimuths,
        cart_resolution=0.2384,
        cart_pixel_width=1024,
    )

    return cvimage_to_message(radar_image, msg.b_scan_img.header)


def write_radar_image(
    writer, radar_image_connection, connection, rawdata, timestamp, typestore
):
    """
    Convert a radar scan into a ROS 2 image message
    and write it to a bag file using the provided writer.
    """
    msg = typestore.deserialize_cdr(rawdata, connection.msgtype)

    # write radar image
    radar_img_msg = get_radar_img_msg(msg)
    writer.write(
        radar_image_connection,
        timestamp,
        typestore.serialize_cdr(radar_img_msg, radar_img_msg.__msgtype__),
    )


def save(
    path: Path,
    timestamps: np.ndarray,
    azimuths: np.ndarray,
    fft_data: np.ndarray,
    is_cartesian=False,
):
    """
    Save a radar scan to a file.
    """
    if is_cartesian:
        cartesian_data = polar_to_cartesian(fft_data, azimuths)
        cv2.imwrite(str(path), cartesian_data)
    else:
        timestamp_bytes = np.frombuffer(timestamps.tobytes(), dtype=np.uint8).reshape(
            -1, 8
        )
        azimuth_bytes = np.frombuffer(
            (azimuths / 2 * np.pi * float(ENDODER_SIZE)).astype(np.uint16).tobytes(),
            dtype=np.uint8,
        ).reshape(-1, 2)

        final_data = np.zeros(
            (fft_data.shape[0], fft_data.shape[1] + 11), dtype=np.uint8
        )
        final_data[:, :8] = timestamp_bytes
        final_data[:, 8:10] = azimuth_bytes
        final_data[:, 11:] = fft_data

        cv2.imwrite(str(path), final_data)


def load_fomo_radar(dataset_base_path: str, deployment: str, trajectory: str):
    """
    Load FOMO radar data from the dataset. Currently only loads the first available radar scan.

    Args:
        dataset_base_path (str): Path to the dataset    .
        deployment (str): Deployment name.
        trajectory (str): Trajectory name.
    """
    path = naming.construct_path(dataset_base_path, deployment, trajectory)

    if list(path.glob("*/.mcap")):
        raise NotImplementedError("Can't load audio data from mcap files yet")

    has_navtech = len(list(path.glob("navtech/"))) > 0

    if has_navtech:
        first_filename = sorted(list(path.glob("navtech/*.png")))[0]
        return load(first_filename)
    else:
        raise ValueError("No radar data found in the dataset.")


def load(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    #
    # Copyright (c) 2017 University of Oxford
    # Authors:
    #  Dan Barnes (dbarnes@robots.ox.ac.uk)
    #
    # Decode a single Oxford Radar RobotCar Dataset radar example
    Args:
        example_path (AnyStr): Oxford Radar RobotCar Dataset Example png
    Returns:
        timestamps (np.ndarray): Timestamp for each azimuth in int64 (UNIX time)
        azimuths (np.ndarray): Rotation for each polar radar azimuth (radians)
        valid (np.ndarray) Mask of whether azimuth data is an original sensor reading or interpolated from adjacent
            azimuths
        fft_data (np.ndarray): Radar power readings along each azimuth
    """
    # Hard coded configuration to simplify parsing code
    raw_example_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    timestamps = raw_example_data[:, :8].copy().view(np.int64)
    azimuths = (
        raw_example_data[:, 8:10].copy().view(np.uint16)
        / float(ENDODER_SIZE)
        * 2
        * np.pi
    ).astype(np.float32)
    fft_data = raw_example_data[:, 11:].astype(np.float32)[:, :, np.newaxis]
    min_range = int(round(MIN_RANGE / RADAR_RESOLUTION))
    fft_data[:, :min_range] = 0
    fft_data = np.squeeze(fft_data)
    return timestamps, azimuths, fft_data
