import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress


def compute_std_norm(df_cov):
    std_x = np.sqrt((df_cov["cov_xx_1"] + df_cov["cov_xx_2"] + df_cov["cov_xx_3"]) / 3)
    std_y = np.sqrt((df_cov["cov_yy_1"] + df_cov["cov_yy_2"] + df_cov["cov_yy_3"]) / 3)
    std_z = np.sqrt((df_cov["cov_zz_1"] + df_cov["cov_zz_2"] + df_cov["cov_zz_3"]) / 3)
    return np.sqrt(std_x**2 + std_y**2 + std_z**2)


def paper_visualize_boxplot_std_vs_interdistance(df_traj, df_cov, interdistance):
    std_norm = compute_std_norm(df_cov)

    # Matplotlib config
    plt.rc("font", family="serif", serif="Times")
    plt.rc("text", usetex=True)
    plt.rcParams.update(
        {"xtick.labelsize": 20, "ytick.labelsize": 20, "axes.labelsize": 15}
    )
    width = 3.487
    height = width / 1.618
    fig, ax = plt.subplots(figsize=(width, height))

    boxprops = dict(linewidth=1.5, edgecolor="#333333")
    whiskerprops = capprops = dict(linewidth=1.5, color="#333333")
    colors = ["#FFB347", "#779ECB"]

    bp = ax.boxplot(
        [std_norm, interdistance.mean(axis=1)],
        showfliers=False,
        notch=False,
        patch_artist=True,
        labels=[r"\textbf{GNSS Standard Deviation}", r"\textbf{Interdistance Metric}"],
        boxprops=boxprops,
        whiskerprops=whiskerprops,
        capprops=capprops,
    )

    for median in bp["medians"]:
        median.set_visible(False)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)

    ax.set_ylabel("Error [m]", fontsize=15)
    plt.tight_layout()


def paper_plot_time_series(df_traj, df_cov, interdistance):
    std_norm = compute_std_norm(df_cov)
    plt.figure(figsize=(10, 6))
    plt.plot(
        df_traj["timestamp"],
        interdistance.mean(axis=1),
        label="Interdistance Metric",
        color="#779ECB",
    )
    plt.plot(df_traj["timestamp"], std_norm**2, label="Variance", color="#FFB347")
    plt.xlabel("Timestamp")
    plt.ylabel("Error")
    plt.legend()


def paper_std_vs_interdistance(df_traj, df_cov, interdistance):
    std_norm = compute_std_norm(df_cov)
    x = std_norm**2
    y = interdistance.mean(axis=1)
    slope, intercept, *_ = linregress(x, y)

    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, color="teal", alpha=0.7, label="Data points")
    plt.plot(x, slope * x + intercept, color="red", linewidth=2)
    plt.xlabel("Variance [m²]")
    plt.ylabel("Interdistance Metric [m]")


def paper_plot_interdistance(df_traj, interdistance, start_index, end_index):
    plt.figure(figsize=(10, 6))
    t = df_traj["timestamp"][start_index:end_index]
    t = t - t.iloc[0]
    t = t / 1e9
    for key, color in zip(["(1,2)", "(1,3)", "(2,3)"], ["blue", "orange", "green"]):
        plt.plot(
            t,
            interdistance[key][start_index:end_index],
            label=f"Interdistance {key}",
            color=color,
        )

    plt.xlabel("Time [s]")
    plt.ylabel("Interdistance Error (m)")
    plt.title("Interdistance Errors Over Time", fontsize=14)
    plt.yscale("log")
    plt.legend(loc="lower right")


def _plot_position_with_cov(ax, timestamps, positions, covariances, color, label):
    delta = 3 * np.sqrt(covariances)
    ax.plot(timestamps, positions, c=color, label=label)
    ax.plot(
        timestamps,
        positions + delta,
        linestyle="--",
        c=color,
        alpha=0.5,
    )
    ax.plot(
        timestamps,
        positions - delta,
        linestyle="--",
        c=color,
        alpha=0.5,
    )
    ax.fill_between(
        timestamps,
        positions - delta,
        positions + delta,
        color=color,
        alpha=0.1,
    )


def paper_plot_gnss_measurements(df_pos, df_cov, start=3110, end=3150):
    fig, axs = plt.subplots(3, 1, figsize=(12, 16), sharex=True)
    t = df_pos["timestamp"][start:end]

    receivers = [("1", "red"), ("2", "green"), ("3", "blue")]
    axes = (
        [("east", "cov_xx"), "East [m]"],
        [("north", "cov_yy"), "North [m]"],
        [("up", "cov_zz"), "Up [m]"],
    )

    for ax, (axis_labels, ylabel) in zip(axs, axes):
        axis_key, cov_key = axis_labels
        for receiver, color in receivers:
            pos = df_pos[f"{axis_key}_{receiver}"][start:end]
            cov = df_cov[f"{cov_key}_{receiver}"][start:end]
            _plot_position_with_cov(ax, t, pos, cov, color, f"Receiver {receiver}")
        ax.set_ylabel(ylabel)
        ax.legend()

    axs[2].set_xlabel("Timestamp [s]")
    plt.tight_layout()


def paper_plot_gnss_z_measurements(df_pos, df_cov, start=3110, end=3150):
    fig, ax = plt.subplots(figsize=(12, 4))
    t = df_pos["timestamp"][start:end]
    t = t - t.iloc[0]  # Make timestamp start at 0

    receivers = [("1", "red"), ("2", "green"), ("3", "blue")]
    for receiver, color in receivers:
        pos = df_pos[f"up_{receiver}"][start:end]
        cov = df_cov[f"cov_zz_{receiver}"][start:end]
        _plot_position_with_cov(ax, t, pos, cov, color, f"Receiver {receiver}")

    ax.set_ylabel("Up [m]")
    ax.set_xlabel("Timestamp [s]")
    ax.legend()
    plt.tight_layout()


def paper_export_triplet(df_position_merged, df_covariance_merged, save_path):
    measurements = [
        "east_1",
        "north_1",
        "up_1",
        "east_2",
        "north_2",
        "up_2",
        "east_3",
        "north_3",
        "up_3",
    ]

    cov_names = [
        ["cov_xx_1", "cov_xy_1", "cov_xz_1", "cov_yy_1", "cov_yz_1", "cov_zz_1"],
        ["cov_xx_2", "cov_xy_2", "cov_xz_2", "cov_yy_2", "cov_yz_2", "cov_zz_2"],
        ["cov_xx_3", "cov_xy_3", "cov_xz_3", "cov_yy_3", "cov_yz_3", "cov_zz_3"],
    ]

    def build_cov_matrix(fields, idx):
        cov_xx, cov_xy, cov_xz, cov_yy, cov_yz, cov_zz = [
            df_covariance_merged[f].iloc[idx] for f in fields
        ]
        return np.array(
            [
                [cov_xx, cov_xy, cov_xz],
                [cov_xy, cov_yy, cov_yz],
                [cov_xz, cov_yz, cov_zz],
            ]
        )

    n = len(df_position_merged)
    pos_all = []
    covs_all = [[], [], []]  # One list for each receiver

    for idx in range(n):
        pos = [df_position_merged[m].iloc[idx] for m in measurements]
        pos_all.append(pos)
        for i, fields in enumerate(cov_names):
            covs_all[i].append(build_cov_matrix(fields, idx).flatten())

    # Save all positions in one file (shape: n x 9)
    pos_df = pd.DataFrame(pos_all)
    pos_df.to_csv(
        os.path.join(save_path, "measured_points_all.csv"),
        index=False,
        header=False,
    )

    # Save all covariances in one file per receiver (shape: n x 9)
    for i, cov_list in enumerate(covs_all, 1):
        cov_df = pd.DataFrame(cov_list)
        cov_df.to_csv(
            os.path.join(save_path, f"cov_{i}_all.csv"),
            index=False,
            header=False,
        )

    print(
        f"Saved measured_points_all.csv and cov_1_all.csv, cov_2_all.csv, cov_3_all.csv to {save_path}"
    )
    return np.array(pos_all), [np.array(c) for c in covs_all]
