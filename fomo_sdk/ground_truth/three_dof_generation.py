import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
import yaml
from point_to_gaussian import point_to_gaussian_df
from tqdm import tqdm

from fomo_sdk.ground_truth.utils import (
    open_pos_file,
    point_to_point_minimization,
    timestamp_to_utc_s,
)


def read_arguments():
    parser = argparse.ArgumentParser(description="Generate 6 DoF ground truth.")

    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="The emlid_processing input folder.",
        required=True,
    )
    parser.add_argument(
        "-d",
        "--debug",
        help="Output the intermediate Point-to-Point GT file.",
        required=False,
        action="store_true",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output folder for the 3 DoF Ground Truth trajectory.",
        required=False,
    )
    parser.add_argument(
        "--metadata",
        type=str,
        help="Rosbag metadata file used to read the start and end timestamps",
        required=False,
    )
    parser.add_argument(
        "-v", "--visualize", action="store_true", help="Visualize the 3 DoF trajectory."
    )
    parser.add_argument(
        "--singleband",
        action="store_true",
        help="Is the processed data singleband (RS+ receivers)",
        default=False,
    )
    return parser.parse_args()


def convert_LLH_to_ENU(df):
    """
    Converts latitude, longitude, and altitude in the DataFrame from LLH (WGS84) referential to ENU (MTM-7) (EPSG:2949).
    """
    transformer = pyproj.Transformer.from_crs(4326, 2949, always_xy=True)
    df["east"], df["north"], df["up"] = transformer.transform(
        df["longitude"], df["latitude"], df["altitude"]
    )
    return df[["timestamp", "east", "north", "up"]]


def extract_std_from_df(df):
    df = df.filter(
        regex="(timestamp|^cov_xx$|^cov_xy$|^cov_xz$|^cov_yx$|^cov_yy$|^cov_yz$|^cov_zx$|^cov_zy$|^cov_zz$)"
    )
    return df


def combine_dataframes(dataframes):
    """
    Combines DataFrames into one, aligning them by timestamp.
    """
    if len(dataframes) < 2:
        raise ValueError("At least two DataFrames are required for merging.")
    merged_df = pd.merge_asof(
        dataframes[0],
        dataframes[1],
        on="timestamp",
        tolerance=1000000000,
        suffixes=("_1", "_2"),
    )
    for i in range(2, len(dataframes)):
        merged_df = pd.merge_asof(
            merged_df, dataframes[i], on="timestamp", tolerance=1000000000
        )
        for col in dataframes[i].columns:
            if col != "timestamp":
                merged_df.rename(columns={col: f"{col}_{i + 1}"}, inplace=True)
    merged_df.dropna(inplace=True)
    merged_df.reset_index(drop=True, inplace=True)
    return merged_df


def read_tf_file(tf_file, nb_dataframes):
    """
    Reads the GNSS transforms (base_link -> GNSS) from the CSV file and returns the reference triplets P.
    """
    df = pd.read_csv(tf_file, skiprows=3)
    P = np.array(
        [
            df["x"].iloc[0:nb_dataframes],
            df["y"].iloc[0:nb_dataframes],
            df["z"].iloc[0:nb_dataframes],
        ]
    )
    return P


def variance_form(covariance, form):
    if form == "max":
        return np.max(covariance, axis=1)
    elif form == "mean":
        return np.mean(covariance, axis=1)
    elif form == "min":
        return np.min(covariance, axis=1)
    else:
        raise ValueError("Invalid form. Use 'max', 'mean', or 'min'.")


def point_to_point_df(
    gnss_reference,
    df_position_merged,
) -> pd.DataFrame:
    traj_list = []
    for idx, row in tqdm(
        df_position_merged.iterrows(),
        total=len(df_position_merged),
        desc="P2P Processing trajectory",
    ):
        geometry = gnss_reference
        Q = row[1:].to_numpy().reshape(3, 3).T

        point, _ = point_to_point_minimization(geometry, Q)
        traj_list.append([row["timestamp"], *point, 0, 0, 0, 1])
    return pd.DataFrame(
        traj_list,
        columns=["timestamp", "x", "y", "z", "qx", "qy", "qz", "qw"],
    )


def compute_trajectory(
    gnss_reference,
    df_position_merged,
    df_covariance_merged,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_traj_p2p = point_to_point_df(gnss_reference, df_position_merged)
    df_traj_gaus, df_covs = point_to_gaussian_df(
        gnss_reference,
        df_position_merged,
        df_covariance_merged,
        visualize=False,
        use_lpm=True,
        add_random_noise=False,
    )
    df_traj_gaus["timestamp"] = df_traj_gaus["timestamp"].apply(timestamp_to_utc_s)
    df_covs["timestamp"] = df_covs["timestamp"].apply(timestamp_to_utc_s)
    df_traj_p2p["timestamp"] = df_traj_p2p["timestamp"].apply(timestamp_to_utc_s)
    return df_traj_gaus, df_covs, df_traj_p2p


def save_trajectory(
    df_traj_p2g, df_traj_p2p, df_covariance, output_folder, debug: bool, overwrite: bool
):
    """
    Saves the trajectory DataFrame to a .txt file.
    """
    if df_traj_p2g.empty and df_covariance.empty:
        print(
            "\033[91mDataFrames df_traj_p2g, and df_covariance are empty. Skipping save.\033[0m"
        )
        return
    if debug and df_traj_p2p.empty:
        print("\033[91mDataFrames df_traj_p2p is empty. Skipping save.\033[0m")
        return

    os.makedirs(output_folder, exist_ok=True)
    files_data = [
        (os.path.join(output_folder, "gt.txt"), df_traj_p2g, " "),
        (os.path.join(output_folder, "gt_covariance.csv"), df_covariance, ","),
    ]
    if debug:
        files_data.append((os.path.join(output_folder, "gt_p2p.txt"), df_traj_p2p, " "))
    for file, data, sep in files_data:
        if os.path.exists(file):
            if not overwrite:
                print(
                    f"\033[91mFile {file} already exists. Not overwriting it, use --overwrite.\033[0m"
                )
                continue
            else:
                print(f"\033[33mFile {file} already exists. Overwriting it.\033[0m")
        print(f"Saving {file}...")
        data.to_csv(
            file,
            sep=sep,
            index=False,
            header=False,
            float_format="%.9f",
        )


def read_metadata(metadata_path):
    with open(metadata_path, "r") as f:
        metadata = yaml.safe_load(f)

    bag_info = metadata["rosbag2_bagfile_information"]
    start_time_ns = bag_info["starting_time"]["nanoseconds_since_epoch"]
    duration_ns = bag_info["duration"]["nanoseconds"]
    end_time_ns = start_time_ns + duration_ns

    return start_time_ns, end_time_ns


def main():
    """
    Reads GNSS data from CSV files, converts LLH to ENU and combines them, reads the GNSS TF from a YAML file and generate the 6dof ground truth.
    """
    args = read_arguments()

    if args.metadata:
        start_time, end_time = read_metadata(args.metadata)
    else:
        start_time = 0
        end_time = sys.maxsize

    dataframes_position = []
    dataframes_covariance = []

    input_path = args.input
    for location in ["front", "left", "right"]:
        df = open_pos_file(
            os.path.join(input_path, f"{location}.pos"), start_time, end_time
        )
        df_position = convert_LLH_to_ENU(df)
        df_covariance = extract_std_from_df(df)
        dataframes_position.append(df_position)
        dataframes_covariance.append(df_covariance)

    df_position_merged = combine_dataframes(dataframes_position)
    df_covariance_merged = combine_dataframes(dataframes_covariance)

    # See this figure for numbering of the TFs: ../resources/sensor_rack_gnss.png
    if args.singleband:
        tf_file = Path(__file__).parent / "emlid_rs+_tf.csv"
    else:
        tf_file = Path(__file__).parent / "emlid_tf.csv"
    gnss_reference = read_tf_file(tf_file, len(dataframes_position))
    df_traj_out_p2g, df_cov_out, df_traj_out_p2p = compute_trajectory(
        gnss_reference, df_position_merged, df_covariance_merged
    )

    if args.output:
        save_trajectory(
            df_traj_out_p2g,
            df_traj_out_p2p,
            df_cov_out,
            args.output,
            args.debug,
            overwrite=True,
        )

    if args.visualize:
        try:
            from fomo_sdk.ground_truth.visualize import visualize_trajectory_metrics

            visualize_trajectory_metrics(df_traj_out_p2g, df_cov_out, None)
            plt.show()
        except ImportError:
            print("Visualization module not found. Skipping visualization.")


if __name__ == "__main__":
    main()
