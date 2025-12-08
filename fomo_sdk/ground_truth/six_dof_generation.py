import argparse
import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
from paper_figures import (
    paper_plot_gnss_measurements,
    paper_plot_interdistance,
)
from scipy.spatial.transform import Rotation
from tqdm import tqdm
from visualize import (
    visualize_trajectory_metrics,
)


def read_arguments():
    parser = argparse.ArgumentParser(description="Generate 6 DoF ground truth.")

    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="Input folder of the <traj>_emlid_rosbag_extracted.",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output folder for the 6 DoF Ground Truth trajectory.",
        required=False,
    )
    parser.add_argument(
        "-s",
        "--save_figures",
        type=str,
        help="Output folder to save the 6 DoF figures.",
        required=False,
    )
    parser.add_argument(
        "-w",
        "--weighted",
        action="store_true",
        help="Use weighted point to point minimization for 6 DoF generation.",
    )
    parser.add_argument(
        "--old",
        action="store_true",
        help="Use old ReachRS+ GNSS device for 6 DoF generation.",
    )
    parser.add_argument(
        "-m",
        "--metric",
        action="store_true",
        help="Compute the interdistance between the GNSS reference and the trajectory.",
    )
    parser.add_argument(
        "-v", "--visualize", action="store_true", help="Visualize the 6 DoF trajectory."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file if it exists.",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Debug mode, plot the geometry of the GNSS reference points and the trajectory points.",
    )
    parser.add_argument(
        "--paper",
        action="store_true",
        help="Generate figures for the paper.",
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
    df = pd.read_csv(tf_file)
    P = np.array(
        [
            df["x"].iloc[0:nb_dataframes],
            df["y"].iloc[0:nb_dataframes],
            df["z"].iloc[0:nb_dataframes],
        ]
    )
    return P


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


def weighted_point_to_point_minimization(P, Q, weights):
    """
    Weighted version of the Point to Point minimization using the standard deviation as the weights.
    """
    # Normalize weights
    w = weights / np.sum(weights)
    # Compute weighted centroids
    mu_p = np.sum(P * w, axis=1, keepdims=True)
    mu_q = np.sum(Q * w, axis=1, keepdims=True)
    # Subtract centroids
    P_centered = P - mu_p
    Q_centered = Q - mu_q
    # Compute weighted cross-covariance matrix
    H = (P_centered * w) @ Q_centered.T
    # SVD
    U, _, Vt = np.linalg.svd(H)
    # Compute rotation matrix
    M = np.eye(3)
    M[2, 2] = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ M @ U.T
    # Compute translation vector
    t = (mu_q - R @ mu_p).flatten()
    return t, R


def variance_form(covariance, form):
    if form == "max":
        return np.max(covariance, axis=1)
    elif form == "mean":
        return np.mean(covariance, axis=1)
    elif form == "min":
        return np.min(covariance, axis=1)
    else:
        raise ValueError("Invalid form. Use 'max', 'mean', or 'min'.")


def compute_trajectory(
    gnss_reference,
    df_position_merged,
    df_covariance_merged,
    dataframes_position,
    weighted,
):
    """
    Perform the minimization between gnss_reference and df_merged and returns the 6 DoF trajectory.
    """
    traj = []
    for idx, row in tqdm(
        df_position_merged.iterrows(),
        total=len(df_position_merged),
        desc="Processing trajectory",
    ):
        P = gnss_reference
        Q = row[1:].to_numpy().reshape(3, len(dataframes_position)).T
        covariance = (
            df_covariance_merged.iloc[idx]
            .filter(regex="^(cov_xx|cov_yy|cov_zz)")
            .to_numpy()
            .reshape(-1, 3)
        )

        if weighted:
            variance = variance_form(covariance, "max")
            weights = 1 / variance
            t, R = weighted_point_to_point_minimization(P, Q, weights)
        else:
            t, R = point_to_point_minimization(P, Q)

        quat = Rotation.from_matrix(R).as_quat()
        traj.append([row["timestamp"], *t, *quat])
    df_traj = pd.DataFrame(
        traj, columns=["timestamp", "x", "y", "z", "qx", "qy", "qz", "qw"]
    )
    return df_traj


def save_trajectory(df_traj, df_covariance, weighted, output_folder, overwrite):
    """
    Saves the trajectory DataFrame to a CSV file.
    """
    os.makedirs(output_folder, exist_ok=True)
    if weighted:
        suffix = "_weighted"
    else:
        suffix = ""
    trajectory_output_file = os.path.join(output_folder, f"6dof_trajectory{suffix}.csv")
    covariance_output_file = os.path.join(output_folder, f"6dof_covariance{suffix}.csv")
    if not trajectory_output_file.endswith(".csv"):
        trajectory_output_file += ".csv"
    if not covariance_output_file.endswith(".csv"):
        covariance_output_file += ".csv"
    if os.path.exists(trajectory_output_file) and not overwrite:
        print(
            f"\033[91m...\nFile {trajectory_output_file} already exists. Not overwriting it, use --overwrite.\033[0m"
        )
        return
    if os.path.exists(trajectory_output_file) and overwrite:
        print(
            f"\033[33m...\nFile {trajectory_output_file} already exists. Overwriting it.\033[0m"
        )
        df_traj.to_csv(
            trajectory_output_file,
            sep=" ",
            index=False,
            header=False,
            float_format="%.9f",
        )
    if not os.path.exists(trajectory_output_file):
        df_traj.to_csv(
            trajectory_output_file,
            sep=" ",
            index=False,
            header=False,
            float_format="%.9f",
        )
        print(f"\033[92m...\nFile saved to {trajectory_output_file}.\033[0m")
    if os.path.exists(covariance_output_file) and not overwrite:
        print(
            f"\033[91m...\nFile {covariance_output_file} already exists. Not overwriting it, use --overwrite.\033[0m"
        )
        return
    if os.path.exists(covariance_output_file) and overwrite:
        print(
            f"\033[33m...\nFile {covariance_output_file} already exists. Overwriting it.\033[0m"
        )
        df_covariance.to_csv(
            covariance_output_file,
            index=False,
            float_format="%.9f",
        )
    if not os.path.exists(covariance_output_file):
        df_covariance.to_csv(
            covariance_output_file,
            index=False,
            float_format="%.9f",
        )
        print(f"\033[92m...\nFile saved to {covariance_output_file}.\033[0m")


def compute_metric_interdistance(df_merged, gnss_reference):
    """
    Computes the interdistance errors between GNSS reference points and trajectory points.

    Returns a DataFrame with errors for each pair at each timestamp.
    """

    def euclidean(p1, p2):
        return np.linalg.norm(p1 - p2)

    P_Dist = {
        (1, 2): euclidean(gnss_reference[:, 0], gnss_reference[:, 1]),
        (1, 3): euclidean(gnss_reference[:, 0], gnss_reference[:, 2]),
        (2, 3): euclidean(gnss_reference[:, 1], gnss_reference[:, 2]),
    }
    errors = []
    for _, row in df_merged.iterrows():
        Q_Dist = {
            (1, 2): euclidean(
                row[["east_1", "north_1", "up_1"]].values,
                row[["east_2", "north_2", "up_2"]].values,
            ),
            (1, 3): euclidean(
                row[["east_1", "north_1", "up_1"]].values,
                row[["east_3", "north_3", "up_3"]].values,
            ),
            (2, 3): euclidean(
                row[["east_2", "north_2", "up_2"]].values,
                row[["east_3", "north_3", "up_3"]].values,
            ),
        }
        errors.append(
            {
                "(1,2)": np.abs(P_Dist[(1, 2)] - Q_Dist[(1, 2)]),
                "(1,3)": np.abs(P_Dist[(1, 3)] - Q_Dist[(1, 3)]),
                "(2,3)": np.abs(P_Dist[(2, 3)] - Q_Dist[(2, 3)]),
            }
        )
    return pd.DataFrame(errors)


def main():
    """
    Reads GNSS data from CSV files, converts LLH to ENU and combines them, reads the GNSS TF from a YAML file and generate the 6dof ground truth.
    """
    args = read_arguments()

    dataframes_position = []
    dataframes_covariance = []
    dataframes_status = []

    for file in sorted(glob.glob(os.path.join(args.input, "gnss_*"))):
        df = pd.read_csv(os.path.join(file, f"{os.path.basename(file)}.csv"))
        df_position = convert_LLH_to_ENU(df)
        df_covariance = extract_std_from_df(df)
        dataframes_position.append(df_position)
        dataframes_covariance.append(df_covariance)
    df_position_merged = combine_dataframes(dataframes_position)
    df_covariance_merged = combine_dataframes(dataframes_covariance)

    for file in sorted(glob.glob(os.path.join(args.input, "status_*"))):
        df = pd.read_csv(os.path.join(file, f"{os.path.basename(file)}.csv"))
        file
        df.drop(columns=["ros_time"], inplace=True)
        dataframes_status.append(df)
    df_status_merged = combine_dataframes(dataframes_status)

    # See this figure for numbering of the TFs: ../resources/sensor_rack_gnss.png
    tf_file = Path(__file__).parent / (
        "gnss_tf.csv" if not args.old else "gnss_tf_old.csv"
    )
    gnss_reference = read_tf_file(tf_file, len(dataframes_position))
    df_trajectory = compute_trajectory(
        gnss_reference,
        df_position_merged,
        df_covariance_merged,
        dataframes_position,
        args.weighted,
    )
    interdistance = compute_metric_interdistance(df_position_merged, gnss_reference)

    # if args.metric: #TODO: save interdistance to a file

    if args.output:
        save_trajectory(
            df_trajectory,
            df_covariance_merged,
            args.weighted,
            args.output,
            args.overwrite,
        )
    if args.save_figures:
        visualize_trajectory_metrics(
            df_trajectory,
            # df_covariance_merged,
            df_status_merged,
            interdistance,
            args.save_figures,
        )

    # if args.debug:
    # plot_geometry(gnss_reference)
    # plot_gnss_measurements(
    #     df_position_merged
    # )  # TODO: add 3sigma around the measurements

    # To remove later
    if args.paper:
        paper_plot_gnss_measurements(
            df_position_merged,
            df_covariance_merged,
            start=0,
            end=len(df_position_merged),
        )
        paper_plot_interdistance(
            df_trajectory, interdistance, start_index=0, end_index=len(df_trajectory)
        )
        # paper_export_triplet(
        #     df_position_merged, df_covariance_merged, save_path=args.output
        # )
        plt.show()

    if args.visualize:
        visualize_trajectory_metrics(
            df_trajectory, df_status_merged, interdistance, None
        )
        plt.show()


if __name__ == "__main__":
    main()
