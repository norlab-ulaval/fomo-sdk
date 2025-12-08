import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


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
        "-o",
        "--output",
        type=str,
        help="The output folder.",
        required=True,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = read_arguments()
    # Load the ground truth data
    # Read the CSV files
    p2g = pd.read_csv(
        os.path.join(args.input, "3dof_trajectory_p2g.csv"),
        sep=" ",
        names=["timestamp", "x", "y", "z", "qx", "qy", "qz", "qw"],
    )
    p2p = pd.read_csv(
        os.path.join(args.input, "3dof_trajectory_p2p.csv"),
        sep=" ",
        names=["timestamp", "x", "y", "z", "qx", "qy", "qz", "qw"],
    )

    # Merge dataframes on timestamp to align data
    merged = pd.merge(p2g, p2p, on="timestamp", suffixes=("_p2g", "_p2p"))

    # Get the initial timestamp and initial positions from merged data
    t0 = merged["timestamp"].iloc[0]
    x0_p2g = merged["x_p2g"].iloc[0]
    y0_p2g = merged["y_p2g"].iloc[0]
    z0_p2g = merged["z_p2g"].iloc[0]

    # Create a figure and axis
    fig, ax = plt.subplots(3, 2, figsize=(10, 10))

    for row_idx, coord in enumerate(["x", "y", "z"]):
        # Left column: Plot both trajectories relative to initial position
        ax[row_idx, 0].plot(
            merged["timestamp"] - t0,
            merged[f"{coord}_p2g"] - merged[f"{coord}_p2g"].iloc[0],
            label="P2G",
            linestyle="--",
        )
        ax[row_idx, 0].plot(
            merged["timestamp"] - t0,
            merged[f"{coord}_p2p"] - merged[f"{coord}_p2g"].iloc[0],
            label="P2P",
        )
        ax[row_idx, 0].set_xlabel("Time (s)")
        ax[row_idx, 0].set_ylabel(f"{coord} Position (m)")
        ax[row_idx, 0].legend()

        # Right column: Plot difference between P2G and P2P
        ax[row_idx, 1].plot(
            merged["timestamp"] - t0,
            merged[f"{coord}_p2g"] - merged[f"{coord}_p2p"],
            linestyle="--",
        )
        ax[row_idx, 1].set_xlabel("Time (s)")
        ax[row_idx, 1].set_ylabel(f"{coord} P2G-P2P (m)")

    # plt.show()
    plt.tight_layout()
    plt.savefig(os.path.join(args.output, "3dof_trajectory_comparison.png"))
