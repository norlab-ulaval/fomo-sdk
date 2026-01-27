import datetime

import numpy as np
import pandas as pd
import pyproj


def utc_to_epoch_ns(utc_str):
    date_str, time_str = utc_str.split(" ")
    dt = datetime.datetime.strptime(f"{date_str} {time_str}", "%Y/%m/%d %H:%M:%S.%f")
    return dt.replace(tzinfo=datetime.timezone.utc).timestamp() * 1e9


def open_tum_file(input_path: str):
    df = pd.read_csv(
        input_path, sep="\s+", names=["time", "px", "py", "pz", "qx", "qy", "qz", "qw"]
    )
    return df


def open_pos_file(input_path: str, start_time=0, end_time=float("inf")):
    df = pd.read_csv(
        input_path,
        comment="%",
        sep="\s+",
        header=None,
        names=[
            "date",
            "time",
            "latitude",
            "longitude",
            "altitude",
            "Q",
            "ns",
            "sdn",
            "sde",
            "sdu",
            "sdne",
            "sdeu",
            "sdun",
            "age",
            "ratio",
        ],
    )

    df["timestamp"] = df["date"] + " " + df["time"]
    df["timestamp"] = df["timestamp"].apply(utc_to_epoch_ns)

    # Filter by time range
    df = df[(df["timestamp"] >= start_time) & (df["timestamp"] <= end_time)]

    df_out = df[["timestamp", "latitude", "longitude", "altitude"]].copy()

    # Fix RTKlib std
    def signed_square(x):
        return (x**2) * (1 if x >= 0 else -1)

    cov_map = {
        "cov_xx": "sde",
        "cov_xy": "sdne",
        "cov_xz": "sdeu",
        "cov_yx": "sdne",
        "cov_yy": "sdn",
        "cov_yz": "sdun",
        "cov_zx": "sdeu",
        "cov_zy": "sdun",
        "cov_zz": "sdu",
    }

    for cov_col, src_col in cov_map.items():
        df_out[cov_col] = df[src_col].apply(signed_square)

    df_out["Q"] = df["Q"]
    df_out["ns"] = df["ns"]
    df_out["age"] = df["age"]
    df_out["ratio"] = df["ratio"]

    return df_out


def timestamp_to_utc_s(timestamp: float) -> str:
    nanoseconds_int = int(timestamp)
    seconds = nanoseconds_int // 1_000_000_000
    nanoseconds_remainder = nanoseconds_int % 1_000_000_000

    # Format as seconds with nanosecond precision
    return f"{seconds}.{nanoseconds_remainder:09d}"


def point_to_point_minimization(P, Q):
    """
    Computes the transformation matrix T that aligns triplets P (reference) to triplets Q (measurements).
    """
    # Compute centroids
    mu_p = np.mean(P[:3, :], axis=1, keepdims=True)
    mu_q = np.mean(Q[:3, :], axis=1, keepdims=True)
    # Center the points
    P_mu = P[:3, :] - mu_p
    Q_mu = Q[:3, :] - mu_q
    # Compute cross-covariance matrix
    H = P_mu @ Q_mu.T
    # SVD decomposition
    U, _, Vt = np.linalg.svd(H)
    # Compute rotation matrix
    M = np.eye(3)
    M[2, 2] = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ M @ U.T
    # Compute translation vector
    t = (mu_q - R @ mu_p).flatten()
    return t, R


def convert_LLH_to_ENU(df):
    """
    Converts latitude, longitude, and altitude in the DataFrame from LLH (WGS84) referential to ENU (MTM-7) (EPSG:2949).
    """
    transformer = pyproj.Transformer.from_crs(4326, 2949, always_xy=True)
    df["east"], df["north"], df["up"] = transformer.transform(
        df["longitude"], df["latitude"], df["altitude"]
    )
    return df[["timestamp", "east", "north", "up"]]
