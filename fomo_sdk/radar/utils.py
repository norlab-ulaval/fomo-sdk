from enum import Enum
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from rosbags.image import message_to_cvimage
from rosbags.typesys.stores.latest import (
    sensor_msgs__msg__Image as Image,
)
from rosbags.typesys.stores.latest import (
    std_msgs__msg__Header as Header,
)
from scipy.ndimage import gaussian_filter

import fomo_sdk.common.naming as naming

MIN_RANGE = 0.5
RADAR_RESOLUTION = 0.0438
RADAR_RANGE_BINS = 6848
ENDODER_SIZE = 5595


class RadarPointsExtractor(Enum):
    KPEAKS = 1
    MOD_CACFAR = 2


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


def load_fomo_radar(
    dataset_base_path: str,
    deployment: str,
    trajectory: str,
    number_of_scans: int = 1,
    timestamp_range: tuple[int, int] | None = None,
    timestamps: list[int] | None = None,
):
    """
    Load FOMO radar data from the dataset.

    Args:
        dataset_base_path (str): Path to the dataset.
        deployment (str): Deployment name.
        trajectory (str): Trajectory name.
        number_of_scans (int, optional): Number of scans to load. Defaults to 1.
        timestamp_range (tuple[int, int] | None, optional): Timestamp range to load data from. Defaults to None.
        timestamps (list[int] | None, optional):
    """
    path = naming.construct_path(dataset_base_path, deployment, trajectory)

    if list(path.glob("*/.mcap")):
        raise NotImplementedError("Can't load radar data from mcap files yet")

    has_navtech = (path / "navtech").exists()
    if not has_navtech:
        raise ValueError("No radar data found in the dataset.")

    filespaths = [f for f in (path / "navtech").iterdir() if f.suffix == ".png"]
    filespaths.sort()
    if timestamps:
        # for each timestamp, only keep the file with the closest timestamp
        closest_files = []
        for timestamp in timestamps:
            closest_file = min(filespaths, key=lambda f: abs(int(f.stem) - timestamp))

            closest_files.append(closest_file)
        filespaths = closest_files
    loaded_files = []
    i = 0
    for filename in filespaths:
        if timestamps is None:
            if number_of_scans > 0 and i >= number_of_scans:
                break
            elif (
                timestamp_range is not None
                and not timestamp_range[0] <= int(filename.stem) <= timestamp_range[1]
            ):
                continue
        loaded_files.append(load(filename))
        i += 1
    if len(loaded_files) == 1:
        return loaded_files[0]
    return loaded_files


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


def KPeaks(
    raw_scan: np.ndarray,
    minr: float = 2.0,
    maxr: float = 80.0,
    res: float = RADAR_RESOLUTION,
    K: int = 3,
    static_threshold: float = 0.25,
):
    """
    K-peaks radar extractor (Python)
    - raw_scan: 2D array [rows=azimuth, cols=range_bins] of float intensities
    - minr/maxr: meters
    - res: meters per bin (range resolution)
    - K: number of peaks to keep per row
    - static_threshold: intensity threshold to start a peak

    Returns:
      np.ndarray of shape [N, 2], entries (row_index, avg_col_index_float)
      Note: avg_col_index can be fractional due to averaging across a peak.
    """
    rows, cols = raw_scan.shape

    # convert meter limits to column limits, clamp to [0, cols]
    mincol = int(minr / res)
    if mincol > cols or mincol < 0:
        mincol = 0
    maxcol = int(maxr / res)
    if maxcol > cols or maxcol < 0:
        maxcol = cols

    targets_polar_pixels = []

    for i in range(rows):
        # 1) Collect (intensity, j) for bins above threshold in increasing j
        intens = []
        row_vals = raw_scan[i]
        for j in range(mincol, maxcol):
            v = row_vals[j]
            if v >= static_threshold:
                intens.append((v, j))

        if not intens:
            continue

        # 2) Group adjacent bins into peaks, tracking each peak’s max intensity
        peaks = []  # list of (peak_max_value, [bin_indices])
        current_bins = [intens[0][1]]
        current_max = intens[0][0]

        for val, j in intens[1:]:
            if j == current_bins[-1] + 1:
                # continue the current peak
                current_bins.append(j)
                if val > current_max:
                    current_max = val
            else:
                # finalize previous peak
                peaks.append((current_max, current_bins))
                # start new peak
                current_bins = [j]
                current_max = val

        # add the last peak
        peaks.append((current_max, current_bins))

        # 3) Sort peaks by max intensity (desc)
        peaks.sort(key=lambda x: x[0], reverse=True)

        # 4) Take top-K peaks; use averaged column index for each peak
        for p in range(min(K, len(peaks))):
            _, bins = peaks[p]
            avg_j = float(np.mean(bins))  # can be fractional
            # (i, avg_j) mirrors your KStrong (row, col) output convention
            targets_polar_pixels.append((i, avg_j))

    return np.asarray(targets_polar_pixels, dtype=np.float32)


def modifiedCACFAR(
    raw_scan: np.ndarray,
    minr=1.0,
    maxr=69.0,
    res=0.040308,
    width=137,
    guard=7,
    threshold=0.50,
    threshold2=0.0,
    threshold3=0.23,
    peak_summary_method="weighted_mean",
):
    # peak_summary_method: median, geometric_mean, max_intensity, weighted_mean
    rows = raw_scan.shape[0]
    cols = raw_scan.shape[1]
    if width % 2 == 0:
        width += 1
    w2 = int(np.floor(width / 2))
    mincol = int(minr / res + w2 + guard + 1)
    if mincol > cols or mincol < 0:
        mincol = 0
    maxcol = int(maxr / res - w2 - guard)
    if maxcol > cols or maxcol < 0:
        maxcol = cols
    targets_polar_pixels = []

    for i in range(rows):
        mean = np.mean(raw_scan[i])
        peak_points = []
        peak_point_intensities = []
        for j in range(mincol, maxcol):
            left = 0
            right = 0
            for k in range(-w2 - guard, -guard):
                left += raw_scan[i, j + k]
            for k in range(guard + 1, w2 + guard):
                right += raw_scan[i, j + k]
            # (statistic) estimate of clutter power
            stat = max(left, right) / w2  # GO-CFAR
            thres = threshold * stat + threshold2 * mean + threshold3
            if raw_scan[i, j] > thres:
                peak_points.append(j)
                peak_point_intensities.append(raw_scan[i, j])
            elif len(peak_points) > 0:
                if peak_summary_method == "median":
                    r = peak_points[len(peak_points) // 2]
                elif peak_summary_method == "geometric_mean":
                    r = np.mean(peak_points)
                elif peak_summary_method == "max_intensity":
                    r = peak_points[np.argmax(peak_point_intensities)]
                elif peak_summary_method == "weighted_mean":
                    r = np.sum(
                        np.array(peak_points)
                        * np.array(peak_point_intensities)
                        / np.sum(peak_point_intensities)
                    )
                else:
                    raise NotImplementedError(
                        "peak summary method: {} not supported".format(
                            peak_summary_method
                        )
                    )
                targets_polar_pixels.append((i, r))
                peak_points = []
                peak_point_intensities = []
    return np.asarray(targets_polar_pixels)


def extract_points(
    raw_scan: np.ndarray,
    azimuth: np.ndarray,
    timestamps: np.ndarray,
    extractor: RadarPointsExtractor,
) -> pd.DataFrame:
    polar_intensity = np.array(raw_scan)
    polar_std = np.std(polar_intensity, axis=1)
    polar_mean = np.mean(polar_intensity, axis=1)
    polar_intensity -= polar_mean[:, np.newaxis] + 2 * polar_std[:, np.newaxis]
    polar_intensity[polar_intensity < 0] = 0
    polar_intensity = gaussian_filter(polar_intensity, sigma=(3, 0), truncate=4.0)
    polar_intensity /= np.max(polar_intensity, axis=1, keepdims=True)
    polar_intensity[np.isnan(polar_intensity)] = 0

    if extractor == RadarPointsExtractor.KPEAKS:
        targets = KPeaks(
            polar_intensity,
            minr=3.0,
            maxr=100.0,
            res=RADAR_RESOLUTION,
            K=10,
            static_threshold=0.3,
        )
    elif extractor == RadarPointsExtractor.MOD_CACFAR:
        targets = modifiedCACFAR(
            polar_intensity,
            minr=5,
            maxr=100,
            res=RADAR_RESOLUTION,
            width=137,
            guard=7,
            threshold=0.50,
            threshold2=0.0,
            threshold3=0.30,
        )
    else:
        raise ValueError(f"Unknown extractor: {extractor}")

    x_coors = []
    y_coors = []
    intensities = []
    timestamps_list = []
    for target in targets:
        azimuth_idx = int(target[0])
        range_idx = int(target[1])

        x = range_idx * RADAR_RESOLUTION * np.cos(azimuth[azimuth_idx])
        y = range_idx * RADAR_RESOLUTION * np.sin(azimuth[azimuth_idx])
        intensity = polar_intensity[azimuth_idx, range_idx]
        x_coors.append(x)
        y_coors.append(y)
        intensities.append(intensity)
        timestamps_list.append(timestamps[azimuth_idx])

    return pd.DataFrame(
        {
            "x": np.array(x_coors).flatten(),
            "y": np.array(y_coors).flatten(),
            "z": np.zeros_like(x_coors).flatten(),
            "intensity": intensities,
            "timestamp": np.array(timestamps_list).flatten(),
        }
    )
