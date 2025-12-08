import matplotlib.pyplot as plt
import numpy as np


def plot_evaluation_matrix(
    matrix, add_marker_matrix, labels_maps, labels_locs, title, ax, cmap="Reds"
):
    """
    Plot evaluation matrix with values and colors.
    """
    # Create a masked array to handle NaN values
    masked_matrix = np.ma.masked_where(np.isnan(matrix), matrix)
    masked_matrix_display = masked_matrix.copy()
    masked_matrix_display[add_marker_matrix] = np.nanmax(matrix)
    masked_matrix_display[np.isnan(matrix)] = np.nanmax(matrix)
    # Create heatmap
    im = ax.imshow(
        masked_matrix_display,
        cmap=cmap,
        aspect="equal",
        vmin=np.nanmin(matrix),
        vmax=np.nanmax(matrix),
    )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.59)
    cbar.set_label("Mean translation drift [\%]", rotation=90, labelpad=5)

    # Set ticks and labels
    ax.set_xticks(range(len(labels_locs)))
    ax.set_yticks(range(len(labels_maps)))
    ax.set_xticklabels(labels_locs)
    ax.set_yticklabels(labels_maps)

    # Add text annotations
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if not np.isnan(matrix[i, j]):
                # Choose text color based on background intensity
                text_color = (
                    "white"
                    if masked_matrix[i, j] > (np.nanmax(matrix) * 0.6)
                    or add_marker_matrix[i, j]
                    else "black"
                )
                marker = "*" if add_marker_matrix[i, j] else ""
                ax.text(
                    j,
                    i,
                    f"{matrix[i, j]:.1f}{marker}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontweight="bold",
                )
            else:
                ax.text(j, i, "N/A", ha="center", va="center", color="white")

    # Labels and title
    ax.set_xlabel("Localization Deployment", fontweight="bold")
    ax.set_ylabel("Mapping Deployment", fontweight="bold")
    # ax.set_title(title, fontweight="bold")

    # Add grid
    ax.set_xticks(np.arange(-0.5, len(labels_locs), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels_maps), 1), minor=True)
    ax.tick_params(axis="both", which="minor", length=0)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1)


def plot_trajectory_xy(
    ax,
    gt,
    est,
    method_name,
    show_xlabel=True,
    show_ylabel=True,
    move_to_origin=True,
):
    """
    Plot XY trajectory comparison.

    Args:
        ax: Matplotlib axis
        gt: Ground truth trajectory
        est: Estimated trajectory
        deployment_label: Label for the deployment (e.g., "Nov21")
        method_name: Name of the method
        show_legend: Whether to show legend
    """
    gt_xyz = gt.positions_xyz
    est_xyz = est.positions_xyz
    # Zero trajectories to start at origin
    if move_to_origin:
        gt_xy = gt_xyz[:, :2] - gt_xyz[0, :2]
        est_xy = est_xyz[:, :2] - est_xyz[0, :2]
    else:
        gt_xy = gt_xyz[:, :2]
        est_xy = est_xyz[:, :2]

    # Plot trajectories
    ax.plot(
        gt_xy[:, 0],
        gt_xy[:, 1],
        label="Ground Truth",
        linestyle="-",
        linewidth=1.0,
        alpha=0.8,
    )

    ax.plot(
        est_xy[:, 0],
        est_xy[:, 1],
        label="Estimated",
        linestyle="--",
        linewidth=1.0,
        alpha=1.0,
        zorder=10,
    )

    # Mark start and end points
    ax.scatter(
        gt_xy[0, 0],
        gt_xy[0, 1],
        color="black",
        marker="o",
        s=20,
        zorder=10,
        facecolors="none",
        label="Start",
    )

    ax.scatter(
        gt_xy[-1, 0],
        gt_xy[-1, 1],
        color="black",
        marker="+",
        s=30,
        zorder=10,
        label="End",
    )

    # Formatting
    if show_xlabel:
        ax.set_xlabel("X [m]", fontweight="bold")
    if show_ylabel:
        ax.set_ylabel("Y [m]", fontweight="bold")
    ax.grid(True, alpha=0.1)
    ax.set_aspect("equal", adjustable="datalim")
    ax.text(
        0.5,
        1.1,
        f"{method_name}",
        transform=ax.transAxes,
        fontsize=7,
        fontweight="bold",
        verticalalignment="top",
        horizontalalignment="center",
    )


def plot_trajectory_timestamp(ax, traj_ref, traj_est, coord: str):
    """
    Plot the given coordinates (reference and estimated) on the given axis
    as a function of time.
    """
    index = 0
    if coord.lower() == "x":
        index = 0
    elif coord.lower() == "y":
        index = 1
    elif coord.lower() == "z":
        index = 2
    error = np.abs(traj_ref.positions_xyz[:, index] - traj_est.positions_xyz[:, index])
    time = traj_ref.timestamps - traj_ref.timestamps[0]
    ax.plot(
        time,
        error,
        label=f"Error ({coord.capitalize()})",
        linestyle="-",
        marker="o",
        markersize=2,
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"{coord.capitalize()} Error (m)")
    ax.set_title(f"{coord.capitalize()} Trajectory Error Plot")
    ax.grid()
    ax.set_aspect("equal", adjustable="datalim")


def set_equal_aspect_3d(ax, positions):
    """
    Set equal aspect ratio for a 3D plot based on trajectory positions.
    """
    x_limits = [np.min(positions[:, 0]), np.max(positions[:, 0])]
    y_limits = [np.min(positions[:, 1]), np.max(positions[:, 1])]
    z_limits = [np.min(positions[:, 2]), np.max(positions[:, 2])]
    max_range = max(np.ptp(x_limits), np.ptp(y_limits), np.ptp(z_limits))
    mid_x, mid_y, mid_z = np.mean(x_limits), np.mean(y_limits), np.mean(z_limits)
    ax.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
    ax.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
    ax.set_zlim(mid_z - max_range / 2, mid_z + max_range / 2)


def plot_trajectory_3d(ax, traj_ref, traj_est):
    """
    Plot the 3D trajectories on the given axis.
    """
    ax.plot(
        traj_ref.positions_xyz[:, 0],
        traj_ref.positions_xyz[:, 1],
        traj_ref.positions_xyz[:, 2],
        label="Ground Truth (GNSS)",
    )
    ax.plot(
        traj_est.positions_xyz[:, 0],
        traj_est.positions_xyz[:, 1],
        traj_est.positions_xyz[:, 2],
        label="Lidar SLAM",
    )
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.set_zlabel("Z Position (m)")
    ax.set_title("3D Trajectory Plot")
    ax.legend()
    ax.grid()
    set_equal_aspect_3d(ax, traj_ref.positions_xyz)


def plot_summary_table(
    ax, avg_relative_rpe, ape_rmse, mapping_date: str, localization_date: str, slam: str
):
    """
    Plot a summary table of the computed metrics.
    """
    ax.axis("tight")
    ax.axis("off")
    ax.set_title(
        f"SUMMARY METRICS\n({mapping_date} to {localization_date})\nMethod: {slam}",
        fontsize=12,
        fontweight="bold",
    )
    table_data = [[f"{ape_rmse:.3f} m", f"{avg_relative_rpe:.2f} %"]]
    col_labels = ["APE RMSE (m)", "AVG RMSE RPE (%)"]
    table = ax.table(
        cellText=table_data, colLabels=col_labels, loc="center", cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 2.4)


def plot_rpe_details_table(ax, rpe_table):
    """
    Plot a table showing detailed RPE results.
    """
    ax.axis("off")
    table = ax.table(
        cellText=rpe_table,
        colLabels=["Delta", "Relative RPE", "Absolute RPE [m]"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 4.0)


def create_evaluation_figure(
    traj_ref,
    traj_est,
    rpe_table,
    avg_relative_rpe,
    ape_rmse,
    save_path,
    mapping_date: str,
    localization_date: str,
    slam: str,
    is_zeroed: bool,
):
    """
    Create and save a figure with:
    - XY trajectory plot
    - Summary metrics table
    - 3D trajectory plot
    - RPE details table
    """
    fig, axs = plt.subplots(3, 2, figsize=(12, 16))

    # Summary Table
    plot_summary_table(
        axs[0, 0], avg_relative_rpe, ape_rmse, mapping_date, localization_date, slam
    )

    # RPE Details Table
    plot_rpe_details_table(axs[0, 1], rpe_table)

    plot_trajectory_timestamp(axs[1, 0], traj_ref, traj_est, "x")
    plot_trajectory_timestamp(axs[1, 1], traj_ref, traj_est, "y")
    plot_trajectory_timestamp(axs[2, 0], traj_ref, traj_est, "z")

    # XY Trajectory Plot
    plot_trajectory_xy(axs[2, 1], traj_ref, traj_est, is_zeroed)

    # 3D Trajectory Plot (added as subplot 3)
    # ax_3d = fig.add_subplot(4, 2, 7, projection='3d')
    # plot_trajectory_3d(ax_3d, traj_ref, traj_est)

    plt.tight_layout()
    plt.savefig(f"{save_path}.jpg", format="jpg", dpi=300)
    plt.close()
