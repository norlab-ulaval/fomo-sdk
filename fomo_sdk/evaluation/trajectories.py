import copy
from pathlib import Path

import numpy as np
from evo.core import lie_algebra as lie
from evo.core import metrics, sync
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface

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
    gt_file: Path, est_file: Path
) -> tuple[PoseTrajectory3D, PoseTrajectory3D]:
    traj_ref = file_interface.read_tum_trajectory_file(gt_file)
    traj_est = file_interface.read_tum_trajectory_file(est_file)
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


def compute_rpe_for_delta(traj_pair, delta_meters, metric: Metric):
    """
    Compute Relative Pose Error (RPE) for a given delta (in meters).
    Returns the computed statistics or None if processing fails.
    """
    if metric == Metric.POINT_DISTANCE_METRIC:
        pose_relation = metrics.PoseRelation.point_distance
        delta_unit = metrics.Unit.meters
        point_distance_metric = metrics.RPE(
            pose_relation,
            delta_meters,
            delta_unit,
            all_pairs=False,
            pairs_from_reference=True,
        )
        try:
            point_distance_metric.process_data(traj_pair)
            return point_distance_metric.get_all_statistics()
        except Exception as e:
            print(
                f"Error processing '{Metric.POINT_DISTANCE_METRIC.name.lower()}' for delta {delta_meters}: {e}"
            )
            return None
    elif metric == Metric.RPE_METRIC:
        pose_relation = metrics.PoseRelation.translation_part
        delta_unit = metrics.Unit.meters
        rpe_metric = metrics.RPE(
            pose_relation,
            delta_meters,
            delta_unit,
            all_pairs=False,
            pairs_from_reference=True,
        )
        try:
            rpe_metric.process_data(traj_pair)
            return rpe_metric.get_all_statistics()
        except Exception as e:
            print(
                f"Error processing '{Metric.POINT_DISTANCE_METRIC.name.lower()}' for delta {delta_meters}: {e}"
            )
            return None
    elif metric == Metric.LOCAL_DRIFT_METRIC:
        pose_relation = metrics.PoseRelation.translation_part
        delta_unit = metrics.Unit.meters
        local_drift_metric = LocalDriftMetric(
            pose_relation,
            delta_meters,
            delta_unit,
            all_pairs=False,
            pairs_from_reference=True,
            alignment_frac=0.1,
        )
        try:
            local_drift_metric.process_data(traj_pair)
            return local_drift_metric.get_all_statistics()
        except Exception as e:
            print(
                f"Error processing '{Metric.POINT_DISTANCE_METRIC.name.lower()}' for delta {delta_meters}: {e}"
            )
            return None
    else:
        return None


def compute_rpe_set(traj_pair, delta_list):
    """
    Compute RPE for a list of delta values.
    Returns a dictionary mapping delta to its statistics.
    """
    results = {}
    metrics = [
        Metric.POINT_DISTANCE_METRIC,
        Metric.RPE_METRIC,
        Metric.LOCAL_DRIFT_METRIC,
    ]
    for metric in metrics:
        results[metric] = {}
        for delta in delta_list:
            stats = compute_rpe_for_delta(traj_pair, delta, metric)
            if stats is not None:
                results[metric][delta] = stats
            else:
                print(f"Skipping delta {delta} due to processing error.")
                break
    return results


def compute_ate_rmse(rpe_results):
    """
    Compute an aggregated ATE RMSE value from RPE results.
    """
    rmse_values = [stats["rmse"] for stats in rpe_results.values()]
    return float(np.sqrt(np.mean(np.square(rmse_values))))


def process_trajectories(
    gt_file: Path, est_file: Path, alignment: str
) -> tuple[PoseTrajectory3D, PoseTrajectory3D, dict[str, int | float]]:
    if not gt_file.exists():
        print(f"File {gt_file} does not exist (gt_file)")
        raise FileNotFoundError(f"File {gt_file} does not exist")
    if not est_file.exists():
        print(f"File {est_file} does not exist (est_file)")
        raise FileNotFoundError(f"File {est_file} does not exist")

    try:
        traj_ref, traj_est = load_trajectories(gt_file, est_file)
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

    traj_ref_sync, traj_est_sync = synchronize_trajectories(traj_ref, traj_est)

    speeds_ref = traj_ref_sync.speeds
    speeds_est = traj_est_sync.speeds

    smoothing_window_size = 5
    print("Smoothing trajectory speeds over", smoothing_window_size, "points")

    smoothed_speeds_ref = np.convolve(
        speeds_ref, np.ones(smoothing_window_size) / smoothing_window_size, mode="valid"
    )
    smoothed_speeds_est = np.convolve(
        speeds_est, np.ones(smoothing_window_size) / smoothing_window_size, mode="valid"
    )

    # find first index where speed is > 0.5 m/s
    idx_ref = np.argmax(smoothed_speeds_ref > 0.5)
    idx_est = np.argmax(smoothed_speeds_est > 0.5)

    # define a duration window to be used for speed alignment
    for window_duration in [100, 20]:
        try:
            GNSS_RATE = 10
            window_length = window_duration * GNSS_RATE

            # align trajectories by speed
            idx_ref_start = 0 if idx_ref < 10 else idx_ref - 10
            idx_est_start = 0 if idx_est < 10 else idx_est - 10
            signal_ref = smoothed_speeds_ref[idx_ref_start : idx_ref + window_length]
            signal_est = smoothed_speeds_est[idx_est_start : idx_est + window_length]
            break
        except IndexError:
            print("IndexError: Window duration is too long for the given trajectory.")
            continue

    corr = np.correlate(signal_ref, signal_est, "full")
    idx_max = np.argmax(corr) + 1  # +1 to account for 'full'
    # Calculate actual lag (subtract the zero-lag position)
    lag = idx_max - (len(signal_est) - 1)
    total_time_shift = (lag + (idx_ref_start - idx_est_start)) / GNSS_RATE
    print(f"Correcting time offset of {total_time_shift} seconds.")
    alignement_dict["time_sync"] = {
        "lag": float(lag),
        "total_time_shift": float(total_time_shift),
        "smoothing_window_size": smoothing_window_size,
        "est_corr_length": len(signal_est),
        "ref_corr_length": len(signal_ref),
    }

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
    slam: str,
    move_to_origin: bool,
    export_yaml: bool,
    export_figure: bool,
    plot_figure: bool = False,
):
    """
    Compute evaluation metrics for the given trajectories.
    Outputs a YAML file and a figure.
    """
    output.mkdir(parents=True, exist_ok=True)

    traj_gt, traj_est, alignement_dict = process_trajectories(gt, est, alignment)
    traj_pair = (traj_gt, traj_est)

    ape_rmse, ape_stats = compute_ape(traj_pair)

    rpe_results = compute_rpe_set(traj_pair, EVALUATION_DELTAS)

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
        slam,
        move_to_origin,
        export_figure=export_figure,
        plot_figure=plot_figure,
    )
