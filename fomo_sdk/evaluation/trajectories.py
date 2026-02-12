import copy
from pathlib import Path

import numpy as np
from evo.core import lie_algebra as lie
from evo.core import metrics, sync
from evo.core.filters import FilterException
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface
from scipy import interpolate
from scipy.signal import savgol_filter

from fomo_sdk.common.naming import Slam
from fomo_sdk.evaluation.io import export_results_to_yaml
from fomo_sdk.evaluation.utils import (
    EVALUATION_DELTAS,
    LocalDriftMetric,
    Metric,
    kabsch_algorithm,
)
from fomo_sdk.evaluation.visualization import create_evaluation_figure


def move_trajectories_to_origin(traj_ref: PoseTrajectory3D, traj_est: PoseTrajectory3D):
    traj_ref.transform(lie.se3(t=-traj_ref.positions_xyz[0]))
    traj_est.transform(lie.se3(t=-traj_est.positions_xyz[0]))


def load_trajectories(
    gt_file: Path, est_file: Path, slam: Slam
) -> tuple[PoseTrajectory3D, PoseTrajectory3D]:
    traj_ref = file_interface.read_tum_trajectory_file(gt_file)
    traj_est = file_interface.read_tum_trajectory_file(est_file)

    if slam == Slam.DROIDSLAM:
        print("Fixing droidslam scale")
        # Scale the estimate trajectory to compensate for the stereo baseline difference
        scale = 0.119702 / 0.1
        traj_est.scale(s=scale)
    return traj_ref, traj_est


def synchronize_trajectories(
    traj_ref: PoseTrajectory3D, traj_est: PoseTrajectory3D, max_diff=0.05
):
    return sync.associate_trajectories(traj_ref, traj_est, max_diff)


def align_trajectories(
    traj_ref_sync: PoseTrajectory3D, traj_est_sync: PoseTrajectory3D, alignment: str
) -> tuple[int, PoseTrajectory3D, PoseTrajectory3D]:
    traj_ref_aligned = copy.deepcopy(traj_ref_sync)
    traj_est_aligned = copy.deepcopy(traj_est_sync)

    if alignment == "start":
        traj_est_aligned.align(
            traj_ref_aligned, correct_scale=False, correct_only_scale=False, n=1000
        )
        num_used_poses = 1000
    elif alignment == "full":
        traj_est_aligned.align(
            traj_ref_aligned, correct_scale=False, correct_only_scale=False, n=-1
        )
        num_used_poses = traj_ref_aligned.num_poses
    elif alignment == "kabsch":
        num_used_poses, r_a, t_a = kabsch_algorithm(
            traj_ref_aligned.positions_xyz, traj_est_aligned.positions_xyz
        )
        traj_est_aligned.transform(lie.se3(r_a, t_a))

    else:
        raise ValueError("Invalid alignment type")

    return num_used_poses, traj_ref_aligned, traj_est_aligned


def set_identity_orientations(traj: PoseTrajectory3D):
    num_poses = len(traj.positions_xyz)
    identity_quats = np.zeros((num_poses, 4))
    identity_quats[:, 0] = 1
    return PoseTrajectory3D(
        positions_xyz=traj.positions_xyz,
        orientations_quat_wxyz=identity_quats,
        timestamps=traj.timestamps,
    )


# =============================================================================
# Metric Computation: APE and RPE
# =============================================================================
def compute_ape(traj_pair):
    """
    Compute Absolute Pose Error (APE) using the translation part (equal to point distance).
    """
    pose_relation = metrics.PoseRelation.translation_part
    ape_metric = metrics.APE(pose_relation)
    ape_metric.process_data(traj_pair)
    return ape_metric.get_statistic(
        metrics.StatisticsType.rmse
    ), ape_metric.get_all_statistics()


def compute_rpe_for_delta(traj_pair, delta_meters, metric: Metric, debug=False):
    """
    Compute Relative Pose Error (RPE) for a given delta (in meters).
    Returns the computed statistics or None if processing fails.
    """
    delta_unit = metrics.Unit.meters
    if metric == Metric.POINT_DISTANCE_METRIC:
        pose_relation = metrics.PoseRelation.point_distance
        metric_class = metrics.RPE(
            pose_relation,
            delta_meters,
            delta_unit,
            all_pairs=True,
            pairs_from_reference=True,
        )
    elif metric == Metric.RPE_METRIC:
        pose_relation = metrics.PoseRelation.translation_part
        metric_class = metrics.RPE(
            pose_relation,
            delta_meters,
            delta_unit,
            all_pairs=True,
            pairs_from_reference=True,
        )
    elif metric == Metric.LOCAL_DRIFT_METRIC:
        pose_relation = metrics.PoseRelation.translation_part
        metric_class = LocalDriftMetric(
            pose_relation,
            delta_meters,
            delta_unit,
            all_pairs=False,
            pairs_from_reference=True,
            alignment_frac=0.5,
            debug=debug,
        )
    else:
        return None

    try:
        metric_class.process_data(traj_pair)
        return metric_class.get_all_statistics()
    except FilterException as e:
        print(f"Error processing '{metric.name.lower()}' for delta {delta_meters}: {e}")
        return None
    except Exception as e:
        raise e


def compute_rpe_set(traj_pair, delta_list, debug=False):
    """
    Compute RPE for a list of delta values.
    Returns a dictionary mapping delta to its statistics.
    """
    results = {}
    for metric in [
        Metric.POINT_DISTANCE_METRIC,
        Metric.RPE_METRIC,
        Metric.LOCAL_DRIFT_METRIC,
    ]:
        results[metric] = {}
        for delta in delta_list:
            stats = compute_rpe_for_delta(traj_pair, delta, metric, debug)
            if stats is not None:
                results[metric][delta] = stats
            else:
                print(f"Skipping delta {delta} due to processing error.")
                break
    return results


def remove_duplicates(traj: PoseTrajectory3D):
    _, indices, cts = np.unique(traj.timestamps, return_index=True, return_counts=True)
    if (cts > 1).any():
        print("Trajectory contains duplicate timestamps. Removing duplicates.")
        unique_timestamps = traj.timestamps[indices]
        unique_positions = traj.positions_xyz[indices]
        unique_quats = traj.orientations_quat_wxyz[indices]
        return PoseTrajectory3D(unique_positions, unique_quats, unique_timestamps)
    return traj


def process_trajectories(
    gt_file: Path,
    est_file: Path,
    alignment: str,
    slam: Slam,
    plot: bool = False,
) -> tuple[PoseTrajectory3D, PoseTrajectory3D, dict[str, int | float]]:
    if not gt_file.exists():
        print(f"File {gt_file} does not exist (gt_file)")
        raise FileNotFoundError(f"File {gt_file} does not exist")
    if not est_file.exists():
        print(f"File {est_file} does not exist (est_file)")
        raise FileNotFoundError(f"File {est_file} does not exist")

    try:
        traj_ref, traj_est = load_trajectories(gt_file, est_file, slam)
    except Exception as e:
        print(f"Error loading trajectories: {e}")
        print(f"file paths: {gt_file} {est_file}")
        raise e

    alignement_dict = {
        "ref": {
            "length": traj_ref.path_length,
            "size": traj_ref.num_poses,
            "start": float(traj_ref.timestamps[0]),
            "end": float(traj_ref.timestamps[-1]),
        },
        "est": {
            "length": traj_est.path_length,
            "size": traj_est.num_poses,
            "start": float(traj_est.timestamps[0]),
            "end": float(traj_est.timestamps[-1]),
        },
        "length_diff": traj_ref.path_length - traj_est.path_length,
        "shortened": False,
    }

    traj_ref, traj_est = synchronize_trajectories(traj_ref, traj_est)

    traj_ref = remove_duplicates(traj_ref)
    traj_est = remove_duplicates(traj_est)

    speeds_ref = traj_ref.speeds
    speeds_est = traj_est.speeds

    smoothing_window_size = 40

    # Smooth speeds
    smoothed_speeds_ref = savgol_filter(
        speeds_ref, smoothing_window_size, polyorder=2, mode="nearest"
    )
    smoothed_speeds_est = savgol_filter(
        speeds_est, smoothing_window_size, polyorder=2, mode="nearest"
    )

    # Align timestamps with speeds arrays (speeds are typically one element shorter)
    times_ref_smooth = traj_ref.timestamps[: len(smoothed_speeds_ref)]
    times_est_smooth = traj_est.timestamps[: len(smoothed_speeds_est)]

    # Find first index where speed > 0.5 m/s
    idx_ref = np.argmax(smoothed_speeds_ref > 0.5)
    idx_est = np.argmax(smoothed_speeds_est > 0.5)

    time_interval = 100
    time_ref_start = times_ref_smooth[idx_ref]
    time_est_start = times_est_smooth[idx_est]
    # Get indices for 30-second windows
    mask_ref = (times_ref_smooth >= time_ref_start - 1) & (
        times_ref_smooth < time_ref_start + time_interval
    )
    mask_est = (times_est_smooth >= time_est_start - 1) & (
        times_est_smooth < time_est_start + time_interval
    )

    signal_ref = smoothed_speeds_ref[mask_ref]
    signal_ref -= np.mean(signal_ref)
    signal_est = smoothed_speeds_est[mask_est]
    signal_est -= np.mean(signal_est)
    times_ref_window = times_ref_smooth[mask_ref]
    times_est_window = times_est_smooth[mask_est]

    from matplotlib import pyplot as plt

    plt.figure()
    plt.plot(times_ref_window, signal_ref, label="Reference")
    plt.plot(times_est_window, signal_est, label="Estimated")
    plt.xlabel("Time (s)")
    plt.ylabel("Speed (m/s)")
    plt.legend()
    plt.show()

    # Resample est signal to ref timestamps
    # This puts both signals on the same time grid
    interp_func = interpolate.interp1d(
        times_est_window, signal_est, kind="linear", bounds_error=False, fill_value=0
    )

    # Create common time grid based on ref signal
    common_times = times_ref_window
    signal_est_resampled = interp_func(common_times)

    # Now correlation works correctly - both on same time grid
    corr = np.correlate(signal_ref, signal_est_resampled, "full")
    idx_max = np.argmax(corr)

    # Calculate lag in samples (now samples are at ref's sampling rate)
    lag = idx_max - (len(signal_est_resampled) - 1)

    # Convert to time using ref's timestamps
    dt_ref = np.median(np.diff(times_ref_window))
    total_time_shift = lag * dt_ref

    print(f"Lag: {lag} samples (at ref sampling rate)")
    print(f"Time shift: {total_time_shift:.3f} seconds")
    print(
        f"Est signal leads ref by {total_time_shift:.3f}s"
        if total_time_shift < 0
        else f"Ref signal leads est by {total_time_shift:.3f}s"
    )
    alignement_dict["time_sync"] = {
        "lag": float(lag),
        "total_time_shift": float(total_time_shift),
        "smoothing_window_size": smoothing_window_size,
        "est_corr_length": len(signal_est),
        "ref_corr_length": len(signal_ref),
    }
    if plot:
        from matplotlib import pyplot as plt

        fig, ax = plt.subplots(3, 1, figsize=(12, 10))

        # Plot 1: Original signals with correlation window highlighted
        ax[0].vlines(
            [
                time_ref_start,
                time_est_start,
            ],
            0,
            2,
            colors="gray",
            linestyles="dashed",
            label="Detected speed",
        )
        ax[0].plot(
            times_ref_smooth,
            smoothed_speeds_ref,
            label="Reference",
            color="blue",
            linewidth=1.5,
        )
        ax[0].plot(
            times_est_smooth,
            smoothed_speeds_est,
            label="Estimate (original)",
            color="red",
            linewidth=1.5,
        )
        ax[0].plot(
            times_est_smooth + total_time_shift,
            smoothed_speeds_est,
            label="Estimate (time-corrected)",
            color="green",
            linewidth=1.5,
            linestyle="--",
        )
        # Highlight correlation windows
        ax[0].axvspan(
            times_ref_window[0],
            times_ref_window[-1],
            facecolor="blue",
            alpha=0.15,
            label="Ref correlation window",
        )
        ax[0].axvspan(
            times_est_window[0],
            times_est_window[-1],
            facecolor="red",
            alpha=0.15,
            label="Est correlation window",
        )
        ax[0].set_xlabel("Time (s)")
        ax[0].set_ylabel("Speed (m/s)")
        ax[0].set_title("Full Trajectory - Original and Time-Corrected Signals")
        ax[0].legend(loc="best")
        ax[0].grid(True, alpha=0.3)

        # Plot 2: Zoomed view of correlation windows
        ax[1].plot(
            times_ref_window,
            signal_ref,
            label="Reference",
            color="blue",
            linewidth=2,
            marker="o",
            markersize=3,
        )
        ax[1].plot(
            times_est_window,
            signal_est,
            label="Estimate (original)",
            color="red",
            linewidth=2,
            marker="s",
            markersize=3,
        )
        ax[1].plot(
            times_est_window + total_time_shift,
            signal_est,
            label="Estimate (time-corrected)",
            color="green",
            linewidth=2,
            marker="^",
            markersize=3,
            linestyle="--",
        )
        ax[1].set_xlabel("Time (s)")
        ax[1].set_ylabel("Speed (m/s)")
        ax[1].set_title(
            f"Correlation Window (Time shift: {total_time_shift:.3f}s, Lag: {lag} samples)"
        )
        ax[1].legend(loc="best")
        ax[1].grid(True, alpha=0.3)

        # Plot 3: Correlation function
        # Create time axis for correlation (centered at zero lag)
        corr_time_axis = (
            np.arange(len(corr)) - (len(signal_est_resampled) - 1)
        ) * dt_ref
        ax[2].plot(
            corr_time_axis,
            corr / np.max(corr),
            label="Normalized Correlation",
            color="black",
            linewidth=2,
        )
        ax[2].axvline(
            total_time_shift,
            color="green",
            linestyle="--",
            linewidth=2,
            label=f"Max correlation at {total_time_shift:.3f}s",
        )
        ax[2].axvline(0, color="gray", linestyle=":", linewidth=1, label="Zero lag")
        ax[2].set_xlabel("Time lag (s)")
        ax[2].set_ylabel("Normalized Correlation")
        ax[2].set_title("Cross-Correlation Function")
        ax[2].legend(loc="best")
        ax[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
        exit()

    traj_est.timestamps += total_time_shift
    traj_ref_sync, traj_est_sync = synchronize_trajectories(traj_ref, traj_est)

    if traj_ref_sync.path_length < traj_ref.path_length * 0.9:
        print(
            "Reference trajectory got shorten in the synchronization process. Labeling estimate."
        )
        alignement_dict["est"] = {
            "length": traj_est_sync.path_length,
            "size": traj_est_sync.num_poses,
            "start": float(traj_est_sync.timestamps[0]),
            "end": float(traj_est_sync.timestamps[-1]),
        }
        alignement_dict["length_diff"] = (
            traj_ref.path_length - traj_est_sync.path_length
        )
        alignement_dict["shortened"] = True
    move_trajectories_to_origin(traj_ref_sync, traj_est_sync)
    num_used_poses, traj_ref_aligned, traj_est_aligned = align_trajectories(
        traj_ref_sync, traj_est_sync, alignment
    )

    traj_ref_final = set_identity_orientations(traj_ref_aligned)
    traj_est_final = set_identity_orientations(traj_est_aligned)

    return traj_ref_final, traj_est_final, alignement_dict


def create_rpe_table(rpe_results):
    """
    Create a table (list of lists) summarizing the RPE results.
    Also computes the average relative RPE.
    """
    table_data = []
    relative_rpe_values = []
    for delta, stats in rpe_results[Metric.RPE_METRIC].items():
        rel_rpe = (stats["rmse"] / delta) * 100  # percentage
        relative_rpe_values.append(rel_rpe)
        table_data.append(
            [
                f"{delta}m",
                f"RMSE: {rel_rpe:.2f}%\nSTD: {(stats['std'] / delta) * 100:.2f}%\n"
                f"MIN: {(stats['min'] / delta) * 100:.2f}%\nMAX: {(stats['max'] / delta) * 100:.2f}%",
                f"RMSE: {stats['rmse']:.3f} m\nSTD: {stats['std']:.3f} m\n"
                f"MIN: {stats['min']:.3f} m\nMAX: {stats['max']:.3f} m",
            ]
        )
    return table_data


def evaluate(
    output: Path,
    gt: Path,
    est: Path,
    alignment: str,
    mapping_date: str,
    localization_date: str,
    slam: Slam,
    move_to_origin: bool,
    export_yaml: bool,
    export_figure: bool,
    plot_figure: bool = False,
    debug: bool = False,
):
    """
    Compute evaluation metrics for the given trajectories.
    Outputs a YAML file and a figure.
    """
    output.mkdir(parents=True, exist_ok=True)

    traj_gt, traj_est, alignement_dict = process_trajectories(
        gt, est, alignment, slam, plot=debug
    )
    traj_pair = (traj_gt, traj_est)

    ape_rmse, ape_stats = compute_ape(traj_pair)

    rpe_results = compute_rpe_set(traj_pair, EVALUATION_DELTAS, debug)

    if len(rpe_results) == 0:
        raise ValueError(
            "\033[91mToo big deltas! Try turning on test mode with --test\033[0m"
        )

    rpe_table = create_rpe_table(rpe_results)
    if export_yaml:
        yaml_filename = output / f"{mapping_date}_{localization_date}.yaml"
        export_results_to_yaml(yaml_filename, ape_rmse, rpe_results, alignement_dict)

    analysis_filename = output / f"{mapping_date}_{localization_date}"
    create_evaluation_figure(
        traj_pair[0],
        traj_pair[1],
        rpe_table,
        ape_rmse,
        analysis_filename,
        mapping_date,
        localization_date,
        slam.value,
        move_to_origin,
        export_figure=export_figure,
        plot_figure=plot_figure,
    )
