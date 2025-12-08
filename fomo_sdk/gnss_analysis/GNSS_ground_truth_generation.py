import os
import sys

import cartopy.crs as ccrs
import numpy as np
import pandas as pd
from pyproj import Proj, Transformer, transform
from tqdm import tqdm
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ----------------- GNSS Warthog's Coordinates ----------------------
GNSS_COORDINATES = {
    "reach_right": {
        "x": -1.06235,
        "y": -0.91605,
        "z": 0.43448,
        "roll": 0.0,
        "pitch": 0.0,
        "yaw": 1.5708,
    },
    "reach_front": {
        "x": -0.17789,
        "y": -0.67948,
        "z": 0.43448,
        "roll": 0.0,
        "pitch": 0.0,
        "yaw": 0.0,
    },
    "reach_left": {
        "x": -1.06239,
        "y": 0.17902,
        "z": 0.43448,
        "roll": 0.0,
        "pitch": 0.0,
        "yaw": 1.5708,
    },
}


# ----------------- Point to point Minimisation ----------------------
def minimisation(P, Q):
    errors_before = Q - P  # Errors at the beginning
    mu_p = np.mean(P[0:3, :], axis=1)  # Centroide of each pointcloud
    mu_q = np.mean(Q[0:3, :], axis=1)
    P_mu = np.ones((3, P.shape[1]))  # Centered each pointclouds
    Q_mu = np.ones((3, Q.shape[1]))
    for i in range(0, P_mu.shape[1]):
        P_mu[0:3, i] = P[0:3, i] - mu_p
    for i in range(0, Q_mu.shape[1]):
        Q_mu[0:3, i] = Q[0:3, i] - mu_q
    H = P_mu @ Q_mu.T  # Cross covariance matrix
    U, s, V = np.linalg.svd(H)  # Use SVD decomposition
    M = np.eye(3)  # Compute rotation
    M[2, 2] = np.linalg.det(V.T @ U.T)
    R = V.T @ M @ U.T
    t = mu_q - R @ mu_p  # Compute translation
    T = np.eye(4)  # Transformation matrix
    T[0:3, 0:3] = R
    T[0:3, 3] = t
    return T


# ----------------- Visualization ----------------------
def visualize_gnss_coordinates(coordinates, title, visualize=False):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    for key, value in coordinates.items():
        ax.scatter(value["x"], value["y"], value["z"], label=key)
        ax.text(value["x"], value["y"], value["z"], key)

    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_zlabel("Z Coordinate")
    ax.set_title(title)
    ax.legend()
    ax.axis("equal")
    if visualize:
        plt.show()


def visualize_trajectories(
    dfs,
    start=0,
    end=-1,
    type="3d",
    figname="Trajectories Visualization",
    visualize=False,
    save=False,
    color_timestamp=False,
):
    if visualize:
        fig = plt.figure()
        if type == "3d":
            ax = fig.add_subplot(111, projection="3d")

            for key, df in dfs.items():
                df_cut = df[start:end]
                ax.plot(
                    df_cut["East(m)"], df_cut["North(m)"], df_cut["Up(m)"], label=key
                )

            ax.set_xlabel("East Coordinate")
            ax.set_ylabel("North Coordinate")
            ax.set_zlabel("Up Coordinate")
            ax.set_title(figname)

        if type == "2d":
            for key, df in dfs.items():
                df_cut = df[start:end]
                if color_timestamp:
                    df_sampled = df_cut[::10]
                    scatter = plt.scatter(
                        df_sampled["East(m)"],
                        df_sampled["North(m)"],
                        c=df_sampled["Timestamp"],
                        cmap="viridis",
                        label="Data Points",
                    )
                    plt.colorbar(scatter, label="Timestamp")
                else:
                    plt.plot(df_cut["East(m)"], df_cut["North(m)"], label=key)

            plt.xlabel("East (m)")
            plt.ylabel("North (m)")
            plt.title(figname)

        plt.legend()
        plt.axis("equal")
    if save:
        plt.savefig(figname + ".png")
    if visualize:
        plt.show()


# ----------------- Coordinate Frame Function ----------------------
def LLH2ENU(df):
    """
    Convert LLH (Latitude, Longitude, Height) to ENU (East, North, Up) using ccrs.epsg(2949).

    Parameters:
        df (pandas DataFrame): DataFrame containing LLH columns: 'Latitude(deg)', 'Longitude(deg)', 'Height(m)'

    Returns:
        pandas DataFrame: DataFrame with ENU columns: 'East', 'North', 'Up', replacing the LLH columns.
    """

    transformer = Transformer.from_crs(
        4326, 2949, always_xy=True
    )  # Convert the Coordinate Frame from WGS84 to EPSG:2949

    east, north, up = [], [], []

    for _, row in tqdm(
        df.iterrows(),
        total=df.shape[0],
        desc="Converting LLH to ENU on left, right and front gnss",
    ):
        # Get the LLH values from the row (Series)
        lat, lon, h = row["Latitude(deg)"], row["Longitude(deg)"], row["Height(m)"]

        # Perform the transformation
        e, n, u = transformer.transform(lon, lat, h)

        # Append the results to lists
        east.append(e)
        north.append(n)
        up.append(u)

    # Add ENU columns to the dataframe
    df["East(m)"] = east
    df["North(m)"] = north
    df["Up(m)"] = up

    df.drop(columns=["Latitude(deg)", "Longitude(deg)", "Height(m)"], inplace=True)
    columns = ["East(m)", "North(m)", "Up(m)"] + [
        col for col in df.columns if col not in ["East(m)", "North(m)", "Up(m)"]
    ]
    df = df[columns]

    return df


# ----------------- Read Function ----------------------
def read_data(path, output_path, visualize=False, save=False):
    file_names = ["left.pos", "right.pos", "front.pos"]
    data_frames = {}  # To store individual DataFrames

    for file_name in file_names:
        file_path = os.path.join(path, file_name)
        if os.path.exists(file_path):
            df = pd.read_csv(
                file_path,
                sep=r"\s+",
                comment="%",
                header=None,
                names=[
                    "UTC",
                    "Time",
                    "Latitude(deg)",
                    "Longitude(deg)",
                    "Height(m)",
                    "Q",
                    "ns",
                    "sdn(m)",
                    "sde(m)",
                    "sdu(m)",
                    "sdne(m)",
                    "sdeu(m)",
                    "sdun(m)",
                    "age(s)",
                    "ratio",
                ],
            )
            # Combine "UTC" and "Time" columns into a single timestamp column
            df["Timestamp"] = pd.to_datetime(df["UTC"] + " " + df["Time"])
            df["Timestamp"] = df["Timestamp"].dt.round(freq="100ms")

            # Convert the "Timestamp" column to Unix timestamp in float (seconds + milliseconds)
            df["Unix_Timestamp"] = df["Timestamp"].astype("int64") / 1e9

            df.drop(columns=["UTC", "Time", "Timestamp"], inplace=True)

            # TODO: if convert: Convert LLH to ENU
            df = LLH2ENU(df)

            # Add the DataFrame to the dictionary
            key = file_name.split(".")[0]  # Use 'left', 'right', or 'front' as keys
            data_frames[key] = df
        else:
            print(f"File not found: {file_path}")
            data_frames[file_name.split(".")[0]] = pd.DataFrame()

    if visualize:
        visualize_gnss_coordinates(
            GNSS_COORDINATES, title="GNSS Warthog's Coordinates", visualize=True
        )
        visualize_trajectories(
            data_frames,
            start=0,
            end=-1,
            type="2d",
            visualize=True,
            figname="2D trajectory visualization",
        )
        plt.show()

    if save:
        for key, df in data_frames.items():
            df.to_csv(os.path.join(output_path, f"{key}.csv"), index=False)

    return data_frames


# def plot_error_trajectories(actual, reference):
#     """
#     Visualize errors between actual and reference trajectories with progress tracking.

#     Parameters:
#     actual (list of tuples): Actual trajectory points [(x1, y1), ...].
#     reference (list of tuples): Reference trajectory points [(x1, y1), ...].
#     """
#     # Ensure both trajectories have the same length
#     assert len(actual) == len(reference), "Actual and reference must have the same number of points."

#     # Convert to numpy arrays for easy computation
#     actual = np.array(actual)
#     reference = np.array(reference)

#     # Initialize error metrics list
#     error_metrics = []

#     # Compute errors with tqdm progress bar
#     for P, Q in tqdm(zip(actual, reference), total=len(actual), desc="Calculating Errors"):
#         error = np.linalg.norm(Q - P)
#         error_metrics.append(error)

#     # Convert error metrics to numpy array for plotting
#     errors = np.array(error_metrics)

#     # Plot trajectories
#     plt.figure(figsize=(10, 6))
#     plt.plot(actual[:, 0], actual[:, 1], label="Actual Trajectory", marker='o')
#     plt.plot(reference[:, 0], reference[:, 1], label="Reference Trajectory", marker='x')

#     # Plot error vectors
#     for (ax, ay), (rx, ry), err in zip(actual, reference, errors):
#         plt.arrow(rx, ry, ax - rx, ay - ry, color='red', alpha=0.5, head_width=0.1, length_includes_head=True)

#     # Plot error magnitude
#     plt.scatter(actual[:, 0], actual[:, 1], c=errors, cmap="viridis", label="Error Magnitude", s=50)
#     cbar = plt.colorbar()
#     cbar.set_label("Error Magnitude")

#     # Add plot details
#     plt.xlabel("X Coordinate")
#     plt.ylabel("Y Coordinate")
#     plt.title("Trajectory Error Visualization with tqdm")
#     plt.legend()
#     plt.grid()
#     plt.show()


# ----------------- Generate 6DOF Trajectory ----------------------
def generate_6dof_trajectory(
    left_df, right_df, front_df, gnss_coordinates, path, save=True, visualize=False
):
    """
    Generate a 6-DoF trajectory from synchronized GNSS data using the minimization function.

    Args:
        left_df (pd.DataFrame): Synchronized GNSS data for the left receiver.
        right_df (pd.DataFrame): Synchronized GNSS data for the right receiver.
        front_df (pd.DataFrame): Synchronized GNSS data for the front receiver.
        gnss_coordinates (dict): Dictionary of fixed GNSS positions relative to the platform.

    Returns:
        pd.DataFrame: 6-DoF trajectory with x, y, z, roll, pitch, yaw, and timestamps.
    """
    Q = np.array(
        [
            [
                gnss_coordinates["reach_left"]["x"],
                gnss_coordinates["reach_right"]["x"],
                gnss_coordinates["reach_front"]["x"],
            ],
            [
                gnss_coordinates["reach_left"]["y"],
                gnss_coordinates["reach_right"]["y"],
                gnss_coordinates["reach_front"]["y"],
            ],
            [
                gnss_coordinates["reach_left"]["z"],
                gnss_coordinates["reach_right"]["z"],
                gnss_coordinates["reach_front"]["z"],
            ],
        ]
    )

    for col in front_df.columns:
        if col != "Unix_Timestamp":
            front_df.rename(columns={col: col + "_front"}, inplace=True)

    front_timestamps = front_df[["Unix_Timestamp"]].rename(
        columns={"Unix_Timestamp": "Unix_Timestamp_front"}
    )
    left_timestamps = left_df[["Unix_Timestamp"]].rename(
        columns={"Unix_Timestamp": "Unix_Timestamp_left"}
    )
    right_timestamps = right_df[["Unix_Timestamp"]].rename(
        columns={"Unix_Timestamp": "Unix_Timestamp_right"}
    )

    # Combine the columns into a new DataFrame
    combined_timestamps = pd.concat(
        [front_timestamps, left_timestamps, right_timestamps], axis=1
    )

    # Save the combined DataFrame to a CSV file
    combined_timestamps.to_csv("combined_timestamps.csv", index=False)

    # Merge the dataframes on the closest timestamp
    merged_df = pd.merge(
        left_df,
        right_df,
        on="Unix_Timestamp",
        how="inner",
        suffixes=("_left", "_right"),
    )
    # print(merged_df.head().to_markdown())
    merged_df = pd.merge(front_df, merged_df, on="Unix_Timestamp", how="inner")

    # print(len(merged_df))
    # print(merged_df.head().to_markdown())
    trajectory = []

    left_df = merged_df[["East(m)_left", "North(m)_left", "Up(m)_left"]].rename(
        columns={
            "East(m)_left": "East(m)",
            "North(m)_left": "North(m)",
            "Up(m)_left": "Up(m)",
        }
    )
    right_df = merged_df[["East(m)_right", "North(m)_right", "Up(m)_right"]].rename(
        columns={
            "East(m)_right": "East(m)",
            "North(m)_right": "North(m)",
            "Up(m)_right": "Up(m)",
        }
    )
    front_df = merged_df[["East(m)_front", "North(m)_front", "Up(m)_front"]].rename(
        columns={
            "East(m)_front": "East(m)",
            "North(m)_front": "North(m)",
            "Up(m)_front": "Up(m)",
        }
    )

    if visualize:
        visualize_trajectories(
            {"left": left_df, "right": right_df, "front": front_df},
            start=0,
            end=-1,
            type="2d",
            figname="2D trajectory visualization after timestamp merge",
            visualize=True,
            save=True,
        )

    for i in tqdm(range(len(merged_df)), desc="Generating 6-DoF trajectory"):
        P = np.array(
            [
                [
                    merged_df.iloc[i]["East(m)_left"],
                    merged_df.iloc[i]["East(m)_right"],
                    merged_df.iloc[i]["East(m)_front"],
                ],
                [
                    merged_df.iloc[i]["North(m)_left"],
                    merged_df.iloc[i]["North(m)_right"],
                    merged_df.iloc[i]["North(m)_front"],
                ],
                [
                    merged_df.iloc[i]["Up(m)_left"],
                    merged_df.iloc[i]["Up(m)_right"],
                    merged_df.iloc[i]["Up(m)_front"],
                ],
            ]
        )

        # plot_error_trajectories(P, Q)

        T = minimisation(P, Q)
        T = np.linalg.inv(T)

        x, y, z = T[0:3, 3]
        C = T[0:3, 0:3]

        roll = np.arctan2(C[2, 1], C[2, 2])
        pitch = -np.arcsin(C[2, 0])
        yaw = np.arctan2(C[1, 0], C[0, 0])

        trajectory.append(
            {
                "Timestamp": merged_df.iloc[i]["Unix_Timestamp"],
                "East(m)": x,
                "North(m)": y,
                "Up(m)": z,
                "roll": roll,
                "pitch": pitch,
                "yaw": yaw,
                # 'error': error
            }
        )

    # Rename front dataframe columns back to normal
    for col in front_df.columns:
        front_df.rename(columns={col + "_front": col}, inplace=True)

    trajectory_df = pd.DataFrame(trajectory)

    if visualize:
        visualize_trajectories(
            {"6-DoF Trajectory": trajectory_df},
            type="2d",
            figname="6-DoF GNSS Trajectory",
            visualize=True,
            save=True,
            color_timestamp=True,
        )

    if save and not trajectory_df.empty:
        trajectory_df.to_csv(os.path.join(path, "6dof_trajectory.csv"), index=False)

    return trajectory_df, merged_df


def compute_interdistance(gnss_coordinates, merged_df, visualize=False):
    """
    Compute the interdistance between the GNSS coordinates and the dataframes,
    as well as the difference between relative distances within each triplet.

    Args:
        gnss_coordinates (dict): Dictionary of fixed GNSS positions relative to the platform.
        merged_df (DataFrame): Merged dataframe containing synchronized GNSS data.
        visualize (bool): Whether to print out the results.

    Returns:
        dict: Dictionary of interdistances between the GNSS coordinates and the measurements.
    """
    # Initialize absolute differences between measured positions and reference positions at 0
    absolute_interdistances = {gnss_key: [] for gnss_key in gnss_coordinates.keys()}

    for i in range(len(merged_df)):
        for gnss_key, gnss_value in gnss_coordinates.items():
            key = gnss_key.split("_")[1]  # Extract 'left', 'right', or 'front'
            absolute_interdistances[gnss_key].append(
                np.linalg.norm(
                    [
                        merged_df.iloc[i][f"East(m)_{key}"] - gnss_value["x"],
                        merged_df.iloc[i][f"North(m)_{key}"] - gnss_value["y"],
                        merged_df.iloc[i][f"Up(m)_{key}"] - gnss_value["z"],
                    ]
                )
            )

    # Compute interdistance between each pair of points in the reference configuration (Q)
    Q_left = np.array(
        [
            gnss_coordinates["reach_left"]["x"],
            gnss_coordinates["reach_left"]["y"],
            gnss_coordinates["reach_left"]["z"],
        ]
    )
    Q_right = np.array(
        [
            gnss_coordinates["reach_right"]["x"],
            gnss_coordinates["reach_right"]["y"],
            gnss_coordinates["reach_right"]["z"],
        ]
    )
    Q_front = np.array(
        [
            gnss_coordinates["reach_front"]["x"],
            gnss_coordinates["reach_front"]["y"],
            gnss_coordinates["reach_front"]["z"],
        ]
    )

    Q_left_right_dist = np.linalg.norm(Q_left - Q_right)
    Q_left_front_dist = np.linalg.norm(Q_left - Q_front)
    Q_right_front_dist = np.linalg.norm(Q_right - Q_front)

    # Calculate the same distances for each measured triplet (P)
    triplet_interdistances = {
        "left_right": [],
        "left_front": [],
        "right_front": [],
        "left_right_error": [],
        "left_front_error": [],
        "right_front_error": [],
    }

    for i in range(len(merged_df)):
        P_left = np.array(
            [
                merged_df.iloc[i]["East(m)_left"],
                merged_df.iloc[i]["North(m)_left"],
                merged_df.iloc[i]["Up(m)_left"],
            ]
        )
        P_right = np.array(
            [
                merged_df.iloc[i]["East(m)_right"],
                merged_df.iloc[i]["North(m)_right"],
                merged_df.iloc[i]["Up(m)_right"],
            ]
        )
        P_front = np.array(
            [
                merged_df.iloc[i]["East(m)_front"],
                merged_df.iloc[i]["North(m)_front"],
                merged_df.iloc[i]["Up(m)_front"],
            ]
        )

        # Calculate distances between each pair of points in the measured triplet
        P_left_right_dist = np.linalg.norm(P_left - P_right)
        P_left_front_dist = np.linalg.norm(P_left - P_front)
        P_right_front_dist = np.linalg.norm(P_right - P_front)

        # Store the distances
        triplet_interdistances["left_right"].append(P_left_right_dist)
        triplet_interdistances["left_front"].append(P_left_front_dist)
        triplet_interdistances["right_front"].append(P_right_front_dist)

        # Calculate the error between measured distances and reference distances
        triplet_interdistances["left_right_error"].append(
            P_left_right_dist - Q_left_right_dist
        )
        triplet_interdistances["left_front_error"].append(
            P_left_front_dist - Q_left_front_dist
        )
        triplet_interdistances["right_front_error"].append(
            P_right_front_dist - Q_right_front_dist
        )

    if visualize:
        # Print reference distances
        print("\nReference distances in the GNSS configuration:")
        print(f"Distance between left and right: {Q_left_right_dist:.4f} meters")
        print(f"Distance between left and front: {Q_left_front_dist:.4f} meters")
        print(f"Distance between right and front: {Q_right_front_dist:.4f} meters")

        # Print statistics for the measured distances
        print("\nStatistics for measured interdistances:")
        for key in ["left_right", "left_front", "right_front"]:
            distances = triplet_interdistances[key]
            print(f"\n{key.replace('_', '-')} distances:")
            print(f"  Mean: {np.mean(distances):.4f} meters")
            print(f"  Std dev: {np.std(distances):.4f} meters")
            print(f"  Min: {np.min(distances):.4f} meters")
            print(f"  Max: {np.max(distances):.4f} meters")

        # Print statistics for the errors
        print("\nStatistics for distance errors (measured - reference):")
        for key in ["left_right_error", "left_front_error", "right_front_error"]:
            errors = triplet_interdistances[key]
            print(f"\n{key.replace('_error', '').replace('_', '-')} errors:")
            print(f"  Mean error: {np.mean(errors):.4f} meters")
            print(f"  Std dev: {np.std(errors):.4f} meters")
            print(f"  Min error: {np.min(errors):.4f} meters")
            print(f"  Max error: {np.max(errors):.4f} meters")

        # Print absolute interdistance statistics
        print("\nStatistics for absolute position errors:")
        for gnss_key, interdist in absolute_interdistances.items():
            print(f"\n{gnss_key} absolute errors:")
            print(f"  Mean error: {np.mean(interdist):.4f} meters")
            print(f"  Std dev: {np.std(interdist):.4f} meters")
            print(f"  Min error: {np.min(interdist):.4f} meters")
            print(f"  Max error: {np.max(interdist):.4f} meters")

        # Optionally plot the errors over time
        if len(merged_df) > 1:
            timestamps = merged_df["Unix_Timestamp"]
            plt.figure(figsize=(12, 8))

            plt.subplot(2, 1, 1)
            plt.plot(
                timestamps,
                triplet_interdistances["left_right_error"],
                label="Left-Right Error",
            )
            plt.plot(
                timestamps,
                triplet_interdistances["left_front_error"],
                label="Left-Front Error",
            )
            plt.plot(
                timestamps,
                triplet_interdistances["right_front_error"],
                label="Right-Front Error",
            )
            plt.xlabel("Time")
            plt.ylabel("Distance Error (m)")
            plt.title("Errors in Inter-Receiver Distances")
            plt.legend()
            plt.grid(True)

            plt.subplot(2, 1, 2)
            for gnss_key in absolute_interdistances:
                plt.plot(
                    timestamps,
                    absolute_interdistances[gnss_key],
                    label=f"{gnss_key} Error",
                )
            plt.xlabel("Time")
            plt.ylabel("Absolute Error (m)")
            plt.title("Absolute Position Errors")
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            plt.savefig("interdistance_errors.png")
            plt.show()

            # Create a new figure for boxplot visualization
            plt.figure(figsize=(14, 10))

            # Subplot for triplet interdistance errors boxplot
            plt.subplot(2, 1, 1)
            error_data = [
                triplet_interdistances["left_right_error"],
                triplet_interdistances["left_front_error"],
                triplet_interdistances["right_front_error"],
            ]
            labels = ["Left-Right", "Left-Front", "Right-Front"]
            bp1 = plt.boxplot(error_data, labels=labels, patch_artist=True)

            # Set colors for the boxplots
            colors = ["lightblue", "lightgreen", "lightpink"]
            for patch, color in zip(bp1["boxes"], colors):
                patch.set_facecolor(color)

            plt.ylabel("Distance Error (m)")
            plt.title("Boxplot of Inter-Receiver Distance Errors")
            plt.grid(True, linestyle="--", alpha=0.7)

            # Subplot for absolute errors boxplot
            plt.subplot(2, 1, 2)
            abs_error_data = [
                absolute_interdistances[key] for key in gnss_coordinates.keys()
            ]
            abs_labels = list(gnss_coordinates.keys())
            bp2 = plt.boxplot(abs_error_data, labels=abs_labels, patch_artist=True)

            # Set colors for the boxplots
            colors = ["lightblue", "lightgreen", "lightpink"]
            for patch, color in zip(bp2["boxes"], colors):
                patch.set_facecolor(color)

            plt.ylabel("Absolute Error (m)")
            plt.title("Boxplot of Absolute Position Errors")
            plt.grid(True, linestyle="--", alpha=0.7)

            plt.tight_layout()
            plt.savefig("interdistance_boxplots.png")
            plt.show()

    # Combine both types of results
    results = {
        "absolute_interdistances": absolute_interdistances,
        "triplet_interdistances": triplet_interdistances,
        "reference_distances": {
            "left_right": Q_left_right_dist,
            "left_front": Q_left_front_dist,
            "right_front": Q_right_front_dist,
        },
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="Process GNSS data.")

    # Adding command-line options
    parser.add_argument(
        "-v", "--visualize", action="store_true", help="Visualize the data"
    )
    parser.add_argument("-d", "--debug", action="store_true", help="Debug the data")
    parser.add_argument(
        "-i", "--input", type=str, help="Path to the dataset", required=True
    )
    parser.add_argument(
        "-o", "--output", type=str, help="Path to the output directory", required=True
    )

    args = parser.parse_args()

    # Default paths
    data_output_path = os.path.join(args.output, "gnss_processed")
    trajectory_output_path = os.path.join(args.output, "6_dof_gnss_trajectory")

    # Create the output directories if they don't exist
    os.makedirs(data_output_path, exist_ok=True)
    os.makedirs(trajectory_output_path, exist_ok=True)

    # Read data
    data_frames = read_data(
        args.input, data_output_path, visualize=args.visualize, save=False
    )

    # Generate 6DOF trajectory
    six_dof_trajectory, merged_df = generate_6dof_trajectory(
        data_frames["left"],
        data_frames["right"],
        data_frames["front"],
        GNSS_COORDINATES,
        trajectory_output_path,
        save=True,
        visualize=args.visualize,
    )

    # Print GNSS coordinates
    # print(GNSS_COORDINATES)

    # Compute interdistance
    compute_interdistance(GNSS_COORDINATES, merged_df, visualize=args.visualize)

    # Debugging option
    if args.debug:
        print("Debugging mode is on.")
        # Add any additional debugging code here
        #

    # print(merged_df.head().to_markdown())
    print(GNSS_COORDINATES)
    compute_interdistance(GNSS_COORDINATES, merged_df, visualize=True)
    # if debug: visualize=True
    # print(data_frames["left"].head(), data_frames["right"].head(), data_frames["front"].head())
    # print(len(data_frames["left"]), len(data_frames["right"]), len(data_frames["front"]))
    # pd.set_option('display.float_format', '{:.9f}'.format)
    # print(six_dof_trajectory.head())


if __name__ == "__main__":
    main()
