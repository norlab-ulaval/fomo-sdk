from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd
from rosbags.typesys.stores.latest import sensor_msgs__msg__PointCloud2 as PointCloud2

import fomo_sdk.common.naming as naming

DATA_TYPES = {
    1: np.int8,
    2: np.uint8,
    3: np.int16,
    4: np.uint16,
    5: np.int32,
    6: np.uint32,
    7: np.float32,
    8: np.float64,
}


def downsample_pointcloud(
    msg: PointCloud2, sampling_rate=0.1, seed=42, islslidar=False
) -> PointCloud2:
    # Step 1: Extract raw data as byte buffer
    data = np.frombuffer(msg.data, dtype=np.uint8).reshape(-1, msg.point_step)

    # Step 2: Build DataFrame for each field
    df = pd.DataFrame()
    for field in msg.fields:
        dtype = DATA_TYPES[field.datatype]
        n_bytes = np.dtype(dtype).itemsize
        col_data = data[:, field.offset : field.offset + n_bytes]
        df[field.name] = col_data.flatten().view(dtype)

    # Step 3: Drop any rows with NaNs (if any)
    df.dropna(inplace=True)

    # Step 4: Sample points
    df_sampled = df.sample(frac=sampling_rate, random_state=seed)
    number_of_points = len(df_sampled)

    # Step 5: Repack sampled data into binary
    buffer = bytearray()
    for _, row in df_sampled.iterrows():
        for field in msg.fields:
            dtype = DATA_TYPES[field.datatype]
            if islslidar:
                if field.name == "intensity" or field.name == "timestamp":
                    dtype = np.float64
                if field.name == "ring":
                    dtype = np.uint32
            buffer.extend(np.array([row[field.name]], dtype=dtype).tobytes())

    # Step 6: Convert to numpy array of uint8
    data_uint8 = np.frombuffer(bytes(buffer), dtype=np.uint8)

    # Step 7: Create downsampled PointCloud2
    new_msg = PointCloud2(
        header=msg.header,
        height=1,
        width=number_of_points,
        fields=msg.fields,
        is_bigendian=msg.is_bigendian,
        point_step=msg.point_step,
        row_step=msg.point_step * number_of_points,
        is_dense=msg.is_dense,
        data=data_uint8,
    )

    return new_msg


def load_fomo_lidar(
    dataset_base_path: str,
    deployment: str,
    trajectory: str,
    load_robosense: bool = True,
    number_of_scans: int = 1,
    timestamp_range: tuple[int, int] | None = None,
) -> list[pd.DataFrame] | pd.DataFrame:
    """
    Load number_of_scans FoMo lidar files from the dataset.

    Args:
        dataset_base_path (str): Path to the dataset.
        deployment (str): Deployment name.
        trajectory (str): Trajectory name.
        load_robosense (bool, optional): Whether to load robosense data. Defaults to True.
        number_of_scans (int, optional): Number of scans to load. Defaults to 1.
    """
    path = naming.construct_path(dataset_base_path, deployment, trajectory)

    if (path / ".mcap").exists():
        raise NotImplementedError("Can't load lidar data from mcap files yet")

    has_robosense = (path / "robosense").exists()
    has_leishen = (path / "leishen").exists()

    lidar_type = None
    if has_robosense and load_robosense:
        lidar_type = "robosense"
    elif has_leishen:
        lidar_type = "leishen"
    else:
        raise ValueError("No lidar data found in the dataset.")

    filespaths = [f for f in (path / lidar_type).iterdir() if f.suffix == '.bin']
    filespaths.sort()
    loaded_files = []
    for i, filename in enumerate(filespaths):
        print(timestamp_range, filename.name)
        if timestamp_range is None and i >= number_of_scans:
            break
        elif timestamp_range is not None and not timestamp_range[0] <= int(filename.stem) <= timestamp_range[1]:
            continue
        loaded_files.append(load(filename))
    if len(loaded_files) == 1:
        return loaded_files[0]
    return loaded_files


def load(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File {path} not found.")
    if path.suffix == ".bin":
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
        arr[:, 5] = points["t"]
        points = pd.DataFrame(arr, columns=["x", "y", "z", "i", "r", "t"])
    elif path.suffix == ".csv":
        points = pd.read_csv(path)
    else:
        raise ValueError(f"File {path} is not a .bin or .csv file.")
    return points


def save(path: Path, points: pd.DataFrame) -> None:
    if path.suffix == ".bin":
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
        points.to_records(index=False).astype(dtype).tofile(path)
    elif path.suffix == ".csv":
        points.to_csv(path, index=False)
    else:
        raise ValueError(f"File {path} is not a .bin or .csv file.")


def make_top_view(
    points: pd.DataFrame,
    res=0.1,
    side_range=(-50, 50),
    fwd_range=(-50, 50),
    global_min_i=0.0,
    global_max_i=255.0,
    colour_mode="intensity",
):
    x_points = points["x"].to_numpy()
    y_points = points["y"].to_numpy()
    z_points = points["z"].to_numpy()
    intensities = points["i"].to_numpy()

    f_filt = np.logical_and((x_points > fwd_range[0]), (x_points < fwd_range[1]))
    s_filt = np.logical_and((y_points > side_range[0]), (y_points < side_range[1]))
    filt = np.logical_and(f_filt, s_filt)

    x_points = x_points[filt]
    y_points = y_points[filt]
    z_points = z_points[filt]
    intensities = intensities[filt]

    x_img = (-y_points / res).astype(np.int32)
    y_img = (-x_points / res).astype(np.int32)

    x_img -= int(np.floor(side_range[0] / res))
    y_img -= int(np.floor(fwd_range[0] / res))

    H = int((fwd_range[1] - fwd_range[0]) / res)
    W = int((side_range[1] - side_range[0]) / res)

    img = np.zeros((H, W, 3), dtype=np.uint8)

    norm = matplotlib.colors.Normalize(vmin=global_min_i, vmax=global_max_i)
    colormap = matplotlib.colormaps["turbo"]

    if colour_mode == "intensity":
        colors = (colormap(norm(intensities))[:, :3] * 255).astype(np.uint8)
    else:
        colors = (colormap(norm(z_points))[:, :3] * 255).astype(np.uint8)

    x_img = np.clip(x_img, 0, W - 1)
    y_img = np.clip(y_img, 0, H - 1)

    for xi, yi, color in zip(x_img, y_img, colors):
        cv2.circle(img, (xi, yi), 0, tuple(int(c) for c in color), 1)

    return img


def transform_points(data: pd.DataFrame, tf: np.ndarray) -> pd.DataFrame:
    assert tf.shape == (4, 4), "Transform must be a 4x4 matrix"

    points = data[["x", "y", "z"]].to_numpy()
    points = np.hstack((points, np.ones((points.shape[0], 1)))).T
    points = tf @ points
    points = points.T

    data["x"] = points[:, 0]
    data["y"] = points[:, 1]
    data["z"] = points[:, 2]
    return data
