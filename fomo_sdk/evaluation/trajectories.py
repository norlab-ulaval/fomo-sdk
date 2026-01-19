import copy
from pathlib import Path

import numpy as np
from evo.core import lie_algebra as lie
from evo.core import metrics, sync
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface

from fomo_sdk.evaluation.io import export_results_to_yaml
from fomo_sdk.evaluation.utils import EVALUATION_DELTAS
from fomo_sdk.evaluation.visualization import create_evaluation_figure


def kabsch_algorithm(traj1, traj2) -> tuple[int, np.ndarray, np.ndarray]:
    """
    Modified Kabsch algorithm that fixes first points and finds optimal rotation.

    Parameters:
    traj1, traj2: numpy arrays of shape (n_points, 3) representing 3D trajectories

    Returns:
    r_a: rotation matrix (3x3) for evo transform
    t_a: translation vector (3,) for evo transform
    """

    traj1 = np.array(traj1)
    traj2 = np.array(traj2)

    # Translation to align first points
    t_a = traj1[0] - traj2[0]

    # Center trajectories at first point
    traj1_centered = traj1 - traj1[0]
    traj2_centered = traj2 - traj2[0]

    alignment_frac = 0.25
    target_len = int(alignment_frac * traj1_centered.shape[0])
    print(f"Using first {target_len} points for alignment ({alignment_frac * 100}%)")

    P = traj1_centered[0:target_len].T  # 3 x (n-1) - target points
    Q = traj2_centered[0:target_len].T  # 3 x (n-1) - points to rotate

    # Compute cross-covariance matrix
    H = Q @ P.T

    # SVD of cross-covariance matrix
    U, S, Vt = np.linalg.svd(H)

    # Compute rotation matrix
    r_a = Vt.T @ U.T

    # Ensure proper rotation (det(R) = 1)
    if np.linalg.det(r_a) < 0:
        Vt[-1, :] *= -1
        r_a = Vt.T @ U.T

    return target_len, r_a, t_a


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


def compute_rpe_for_delta(traj_pair, delta_meters):
    """
    Compute Relative Pose Error (RPE) for a given delta (in meters).
    Returns the computed statistics or None if processing fails.
    """
    pose_relation = metrics.PoseRelation.point_distance
    delta_unit = metrics.Unit.meters
    rpe_metric = metrics.RPE(pose_relation, delta_meters, delta_unit, all_pairs=True)
    try:
        rpe_metric.process_data(traj_pair)
    except Exception as e:
        print(f"Error processing RPE for delta {delta_meters}: {e}")
        return None

    pose_relation = metrics.PoseRelation.translation_part
    delta_unit = metrics.Unit.meters
    rpe_metric = metrics.RPE(pose_relation, delta_meters, delta_unit, all_pairs=True)
    try:
        rpe_metric.process_data(traj_pair)
    except Exception as e:
        print(f"Error processing RPE for delta {delta_meters}: {e}")
        return None

    return rpe_metric.get_all_statistics()


def compute_rpe_set(traj_pair, delta_list):
    """
    Compute RPE for a list of delta values.
    Returns a dictionary mapping delta to its statistics.
    """
    results = {}
    for delta in delta_list:
        stats = compute_rpe_for_delta(traj_pair, delta)
        if stats is not None:
            results[delta] = stats
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
) -> tuple[PoseTrajectory3D, PoseTrajectory3D, dict[str, float]]:
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

    trajectories = {
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

    if traj_ref_sync.path_length < traj_ref.path_length * 0.9:
        print(
            "Reference trajectory got shorten in the synchronization process. Labeling estimate."
        )
        trajectories["est"] = {
            "length": traj_est_sync.path_length,
            "size": traj_est_sync.num_poses,
            "start": float(traj_est_sync.timestamps[0]),
            "end": float(traj_est_sync.timestamps[-1]),
        }
        trajectories["length_diff"] = traj_ref.path_length - traj_est_sync.path_length
        trajectories["shortened"] = True
    move_trajectories_to_origin(traj_ref_sync, traj_est_sync)
    num_used_poses, traj_ref_aligned, traj_est_aligned = align_trajectories(
        traj_ref_sync, traj_est_sync, alignment
    )

    traj_ref_final = set_identity_orientations(traj_ref_aligned)
    traj_est_final = set_identity_orientations(traj_est_aligned)

    return traj_ref_final, traj_est_final, trajectories


def create_rpe_table(rpe_results):
    """
    Create a table (list of lists) summarizing the RPE results.
    Also computes the average relative RPE.
    """
    table_data = []
    relative_rpe_values = []
    for delta, stats in rpe_results.items():
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
    avg_relative_rpe = float(np.mean(relative_rpe_values))
    return table_data, avg_relative_rpe


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

    traj_gt, traj_est, trajectories = process_trajectories(gt, est, alignment)
    traj_pair = (traj_gt, traj_est)

    ape_rmse, ape_stats = compute_ape(traj_pair)

    rpe_results = compute_rpe_set(traj_pair, EVALUATION_DELTAS)

    if len(rpe_results) == 0:
        raise ValueError(
            "\033[91mToo big deltas! Try turning on test mode with --test\033[0m"
        )

    rpe_table, avg_relative_rpe = create_rpe_table(rpe_results)
    if export_yaml:
        yaml_filename = output / f"{mapping_date}_{localization_date}.yaml"
        export_results_to_yaml(
            yaml_filename, avg_relative_rpe, ape_rmse, rpe_results, trajectories
        )

    if export_figure:
        _ate_rmse = compute_ate_rmse(rpe_results)
        analysis_filename = output / f"{mapping_date}_{localization_date}"
        create_evaluation_figure(
            traj_pair[0],
            traj_pair[1],
            rpe_table,
            avg_relative_rpe,
            ape_rmse,
            analysis_filename,
            mapping_date,
            localization_date,
            slam,
            move_to_origin,
            export_figure=True,
            plot_figure=plot_figure,
        )
