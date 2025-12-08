import sys
import numpy as np
from evo.core import sync
from evo.core.trajectory import PoseTrajectory3D
from evo.core.metrics import PoseRelation
from evo.tools import file_interface
import evo.main_ape as main_ape
from evo.core.metrics import RPE


def load_tum_trajectory(file_path, convert_timestamp=False):
    """
    Load trajectory from TUM format file.

    Args:
        file_path: Path to the TUM format file
        convert_timestamp: If True, divide timestamps by 1e9 to convert to seconds

    Returns:
        PoseTrajectory3D object
    """
    try:
        traj = file_interface.read_tum_trajectory_file(file_path)

        if convert_timestamp:
            # Convert timestamps from nanoseconds to seconds
            timestamps = np.array(traj.timestamps) / 1e9
            traj = PoseTrajectory3D(
                positions_xyz=traj.positions_xyz,
                orientations_quat_wxyz=traj.orientations_quat_wxyz,
                timestamps=timestamps,
            )

        return traj
    except Exception as e:
        print(f"Error loading trajectory from {file_path}: {e}")
        sys.exit(1)


def align_trajectories(traj_ref, traj_est):
    """
    Synchronize and align two trajectories.

    Args:
        traj_ref: Reference trajectory
        traj_est: Estimated trajectory

    Returns:
        Tuple of (aligned_reference, aligned_estimated)
    """
    print("Synchronizing trajectories...")

    # Synchronize trajectories based on timestamps
    traj_ref_sync, traj_est_sync = sync.associate_trajectories(
        traj_ref, traj_est, max_diff=0.01
    )

    print(
        f"Synchronized trajectory lengths: ref={len(traj_ref_sync.timestamps)}, est={len(traj_est_sync.timestamps)}"
    )

    # Align trajectories using Umeyama alignment
    print("Aligning trajectories using Umeyama alignment...")

    T = np.eye(4)
    T[:3, 3] -= traj_ref_sync.positions_xyz[0, :]
    traj_ref_sync.transform(T)

    traj_est_sync.align(traj_ref_sync, correct_scale=True)

    return traj_ref_sync, traj_est_sync


def compute_ape(traj_ref, traj_est):
    """
    Compute Absolute Pose Error (APE).

    Args:
        traj_ref: Reference trajectory
        traj_est: Estimated trajectory

    Returns:
        APE result object
    """
    print("\nComputing APE (Absolute Pose Error)...")

    # Compute APE for translation
    ape_metric = main_ape.ape(
        traj_ref=traj_ref,
        traj_est=traj_est,
        pose_relation=PoseRelation.translation_part,
        align=False,  # Already aligned
        correct_scale=False,  # Already corrected during alignment
    )

    return ape_metric


def compute_rpe(traj_ref, traj_est, delta=1.0, delta_unit="s"):
    """
    Compute Relative Pose Error (RPE).

    Args:
        traj_ref: Reference trajectory
        traj_est: Estimated trajectory
        delta: Delta for RPE computation
        delta_unit: Unit for delta ('s' for seconds, 'm' for meters, 'f' for frames)

    Returns:
        RPE result object
    """
    print(f"\nComputing RPE (Relative Pose Error) with delta={delta}{delta_unit}...")

    # Create RPE metric object directly
    rpe_metric = RPE(pose_relation=PoseRelation.translation_part)

    # Process the data
    rpe_metric.process_data((traj_ref, traj_est))

    return rpe_metric


def print_statistics(metric, metric_name) -> float:
    """Print statistics for a metric."""
    # Check if it's a Result object or a metric object
    if hasattr(metric, "stats"):
        stats = metric.stats
    elif hasattr(metric, "error"):
        # For metric objects, compute statistics from the error array
        import numpy as np

        errors = np.array(metric.error)
        stats = {
            "rmse": np.sqrt(np.mean(errors**2)),
            "mean": np.mean(errors),
            "median": np.median(errors),
            "std": np.std(errors),
            "min": np.min(errors),
            "max": np.max(errors),
        }
    else:
        print(f"Cannot extract statistics from {metric_name}")
        return

    print(f"\n{metric_name} Statistics:")
    print(f"  RMSE:   {stats['rmse']:.6f} m")
    print(f"  Mean:   {stats['mean']:.6f} m")
    print(f"  Median: {stats['median']:.6f} m")
    print(f"  Std:    {stats['std']:.6f} m")
    print(f"  Min:    {stats['min']:.6f} m")
    print(f"  Max:    {stats['max']:.6f} m")

    return stats["rmse"]


def plot(ref, est, output_path):
    from matplotlib import pyplot as plt

    # Plot the trajectories
    plt.figure(figsize=(8, 6))
    plt.plot(
        ref.positions_xyz[:, 0],
        ref.positions_xyz[:, 1],
        label="Reference",
    )
    plt.plot(
        est.positions_xyz[:, 0],
        est.positions_xyz[:, 1],
        label="Estimated",
    )
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.title("Trajectory Comparison")
    plt.legend()

    plt.axis("equal")

    plt.tight_layout()
    plt.savefig(output_path)


def main():
    # parser = argparse.ArgumentParser(
    #     description="Evaluate trajectory using EVO package with APE and RPE metrics"
    # )
    # parser.add_argument(
    #     "reference_traj", help="Path to reference trajectory file (TUM format)"
    # )
    # parser.add_argument(
    #     "estimated_traj", help="Path to estimated trajectory file (TUM format)"
    # )
    # parser.add_argument(
    #     "--convert-timestamps",
    #     action="store_true",
    #     default=True,
    #     help="Convert timestamps of estimated trajectory from nanoseconds to seconds (divide by 1e9)",
    # )
    # parser.add_argument(
    #     "--rpe-delta",
    #     type=float,
    #     default=1.0,
    #     help="Delta for RPE computation (default: 1.0)",
    # )
    # parser.add_argument(
    #     "--rpe-unit",
    #     choices=["s", "m", "f"],
    #     default="m",
    #     help="Unit for RPE delta: s=seconds, m=meters, f=frames (default: s)",
    # )

    # args = parser.parse_args()

    # ref = args.reference_traj
    # est = args.estimated_traj
    # convert_timestmaps = args.convert_timestamps

    base_path = "/Users/mbo/Desktop/FoMo-SDK/data/lidar-evaluation"
    traj_dict = {
        "blue-2024-11-21-10-44": ("2024-11-21", "2024-11-21/blue-2024-11-21-10-44"),
        "blue-2025-01-29-10-08": ("2025-01-29", "2025-01-29/blue-2025-01-29-10-08"),
        "blue-2025-03-10-16-59": ("2025-03-10", "2025-03-10/blue-2025-03-10-16-59"),
        "blue-2025-06-26-10-35": ("2025-06-26", "2025-06-26/blue-2025-06-26-10-35"),
    }
    rpe_delta = 1.0
    rpe_unit = "m"
    convert_timestmaps = True

    matrix_rpe = np.zeros((len(traj_dict), len(traj_dict)))
    matrix_ape = np.zeros((len(traj_dict), len(traj_dict)))

    for i, ref_key in enumerate(traj_dict.keys()):
        ref = base_path + "/" + f"{traj_dict[ref_key][0]}/gt.tum"
        for j, est_key in enumerate(traj_dict.keys()):
            est = base_path + "/" + f"{traj_dict[ref_key][1]}/{est_key}_trajectory.tum"

            print("Loading trajectories:\n{}\n{}".format(ref, est))

            # Load reference trajectory (no timestamp conversion)
            traj_ref = load_tum_trajectory(ref, convert_timestamp=False)
            print(f"Reference trajectory loaded: {len(traj_ref.timestamps)} poses")

            # Load estimated trajectory (with optional timestamp conversion)
            traj_est = load_tum_trajectory(est, convert_timestamp=convert_timestmaps)
            input(f"Estimated trajectory loaded: {len(traj_est.timestamps)} poses")

            # Align trajectories
            traj_ref_aligned, traj_est_aligned = align_trajectories(traj_ref, traj_est)

            # Compute APE
            ape_result = compute_ape(traj_ref_aligned, traj_est_aligned)
            ape = print_statistics(ape_result, "APE")

            # Compute RPE
            rpe_result = compute_rpe(
                traj_ref_aligned,
                traj_est_aligned,
                delta=rpe_delta,
                delta_unit=rpe_unit,
            )
            rpe = print_statistics(rpe_result, "RPE")

            matrix_rpe[i, j] = rpe
            matrix_ape[i, j] = ape

            plot(
                traj_ref_aligned,
                traj_est_aligned,
                output_path=base_path
                + "/"
                + f"{traj_dict[ref_key][1]}/{est_key}_alignement.png",
            )
    np.savetxt(
        base_path + "/results_rpe.csv",
        matrix_rpe,
    )
    np.savetxt(
        base_path + "/results_ape.csv",
        matrix_ape,
    )


if __name__ == "__main__":
    main()
