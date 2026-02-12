from enum import Enum, auto
from pathlib import Path

import numpy as np
import yaml
from evo.core import lie_algebra as lie
from evo.core.metrics import (
    RPE,
    MetricsException,
    PathPair,
    PoseRelation,
    Unit,
    id_pairs_from_delta,
)
from evo.core.trajectory import (
    se3_poses_to_xyz_quat_wxyz,
    xyz_quat_wxyz_to_se3_poses,
)
from tqdm import tqdm

from fomo_sdk.common.naming import DEPLOYMENT_DATE_LABEL

EVALUATION_DELTAS = [100, 200, 300, 400, 500, 600, 700, 800]


def kabsch_algorithm(
    traj1, traj2, alignment_frac: float = 0.25
) -> tuple[int, np.ndarray, np.ndarray]:
    """
    Modified Kabsch algorithm that fixes first points and finds optimal rotation.

    Parameters:
    traj1, traj2: numpy arrays of shape (n_points, 3) representing 3D trajectories
    alignment_frac: fraction of points to use for alignment

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

    target_len = int(alignment_frac * traj1_centered.shape[0])
    # print(f"Using first {target_len} points for alignment ({alignment_frac * 100}%)")

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


class LocalDriftMetric(RPE):
    def __init__(
        self,
        pose_relation: PoseRelation = PoseRelation.translation_part,
        delta: float = 1.0,
        delta_unit: Unit = Unit.frames,
        rel_delta_tol: float = 0.1,
        all_pairs: bool = False,
        pairs_from_reference: bool = False,
        alignment_frac: float = 0.25,
        debug: bool = False,
    ):
        super().__init__(
            pose_relation,
            delta,
            delta_unit,
            rel_delta_tol,
            all_pairs,
            pairs_from_reference,
        )
        self.alignment_frac = alignment_frac
        self.debug = debug

    def compute_aligned_rpe(self, data: PathPair, id_pairs: list[tuple[int, int]]):
        min_alignment_window_size = 5  # in % of the previous window length
        if self.debug:
            from matplotlib import pyplot as plt

            plt.figure(figsize=(12, 6))
            plt.plot([], [], color="b", label="Reference")
            plt.plot([], [], color="r", label="Estimate")
            plt.plot([], [], color="black", label="Aligned Estimate")
            plt.plot(
                [],
                [],
                alpha=0.8,
                color="gray",
                zorder=-1,
                linewidth=5,
                label="Alignement window",
            )
        E = []
        last_end_index = -1
        for i, j in id_pairs:
            index_start = int(i - (j - i) * self.alignment_frac)
            if not self.all_pairs and i < last_end_index:
                continue
            last_end_index = j

            ref = data[0].poses_se3[i:j]
            ref_positions_xyz, _ = se3_poses_to_xyz_quat_wxyz(ref)

            # the start of each trajectory is already aligned
            if index_start < 0:
                est_aligned = data[1].poses_se3[i:j]
                est_aligned_positions_xyz, est_aligned_quat = (
                    se3_poses_to_xyz_quat_wxyz(est_aligned)
                )
                est_orig_positions_xyz = est_aligned_positions_xyz
                arr_align_ref = ref_positions_xyz[0:1, :]
            elif i - index_start < min_alignment_window_size:
                # print(f"Not enough poses for alignment {index_start} {i} {j - i}")
                continue
            else:
                arr_align_ref = data[0].positions_xyz[index_start:i]
                arr_align_est = data[1].positions_xyz[index_start:i]
                if (
                    len(arr_align_est) < min_alignment_window_size
                    or len(arr_align_ref) < min_alignment_window_size
                ):
                    # print("Not enough poses for alignment. Too short")
                    continue
                num_used_poses, r_a, t_a = kabsch_algorithm(
                    arr_align_ref,
                    arr_align_est,
                    alignment_frac=1.0,
                )
                T = lie.se3(r_a, t_a)

                est_orig = data[1].poses_se3[i:j]
                est_orig_positions_xyz, _ = se3_poses_to_xyz_quat_wxyz(est_orig)
                est_orig_positions_xyz = (
                    ref_positions_xyz[0, :]
                    + est_orig_positions_xyz
                    - est_orig_positions_xyz[0, :]
                )

                est_aligned = [np.dot(T, p) for p in data[1].poses_se3[i:j]]
                est_aligned_positions_xyz, est_aligned_quat = (
                    se3_poses_to_xyz_quat_wxyz(est_aligned)
                )

                est_aligned_positions_xyz = (
                    ref_positions_xyz[0, :]
                    + est_aligned_positions_xyz
                    - est_aligned_positions_xyz[0, :]
                )

                num_poses = len(est_aligned_positions_xyz)
                identity_quats = np.zeros((num_poses, 4))
                identity_quats[:, 0] = 1
                est_aligned = xyz_quat_wxyz_to_se3_poses(
                    est_aligned_positions_xyz, identity_quats
                )

            E.append(
                super().rpe_base(
                    ref[0],
                    ref[-1],
                    est_aligned[0],
                    est_aligned[-1],
                )
            )
            if self.debug:
                plt.plot(
                    ref_positions_xyz[:, 0],
                    ref_positions_xyz[:, 1],
                    color="b",
                )
                plt.plot(
                    est_orig_positions_xyz[:, 0],
                    est_orig_positions_xyz[:, 1],
                    color="r",
                )
                plt.plot(
                    est_aligned_positions_xyz[:, 0],
                    est_aligned_positions_xyz[:, 1],
                    color="black",
                )
                plt.plot(
                    arr_align_ref[:, 0],
                    arr_align_ref[:, 1],
                    alpha=0.8,
                    color="gray",
                    zorder=-1,
                    linewidth=5,
                )
        if self.debug:
            plt.legend()
            plt.title(f"Delta: {self.delta}m")
            plt.show()
        return E

    def process_data(self, data: PathPair) -> None:
        """
        Calculates the RPE on a batch of SE(3) poses from trajectories.
        :param data: tuple (traj_ref, traj_est) with:
        traj_ref: reference evo.trajectory.PosePath or derived
        traj_est: estimated evo.trajectory.PosePath or derived
        """
        if len(data) != 2:
            raise MetricsException("please provide data tuple as: (traj_ref, traj_est)")
        traj_ref, traj_est = data
        if traj_ref.num_poses != traj_est.num_poses:
            raise MetricsException("trajectories must have same number of poses")

        id_pairs = id_pairs_from_delta(
            (traj_ref.poses_se3 if self.pairs_from_reference else traj_est.poses_se3),
            self.delta,
            self.delta_unit,
            self.rel_delta_tol,
            all_pairs=True,
        )

        # Store flat id list e.g. for plotting.
        self.delta_ids = [j for i, j in id_pairs]

        self.E = self.compute_aligned_rpe(data, id_pairs)
        self.error = np.array([np.linalg.norm(E_i[:3, 3]) for E_i in self.E])


class Metric(Enum):
    RPE_METRIC = "RPE [%]"
    POINT_DISTANCE_METRIC = "Point Distance [m]"
    LOCAL_DRIFT_METRIC = "Local Relative Drift [%]"
    APE = "APE [m]"


def parse_evaluation_file_name(file_name: Path | str):
    """
    Parse the evaluation file name to extract the map and location names.
    """
    if isinstance(file_name, Path):
        file_name = file_name.name
    if file_name.count("_") != 3:
        raise ValueError(
            f"Invalid file name: {file_name}. The evaluation file name should be of the form: <row-recording>_<col-recording>.txt, where <recording> is <color>_<datetime>"
        )
    return file_name.split("_")[1], file_name.split("_")[3].split(".")[0]


def compute_rte(data: dict, max_delta: int) -> tuple[float, float]:
    """
    Compute the relative translation drift for a given evaluation dictionary.
    """
    rpe = []
    std = []
    # Calculate mean RPE across different deltas
    for delta in EVALUATION_DELTAS:
        relative_drift = 100 * data[f"{delta}m"]["rmse_meters"] / delta
        relative_std = 100 * data[f"{delta}m"]["std_meters"] / delta
        rpe.append(relative_drift)
        std.append(relative_std)
        if delta == max_delta:
            break
    rpe = np.mean(rpe)
    std = np.mean(std)
    return rpe, std


def construct_matrix(
    path: str,
    metric: Metric,
    max_delta: int = EVALUATION_DELTAS[-1],
):
    # get the number of yaml files in the directory
    yaml_files = [f.name for f in Path(path).glob("*.yaml")]
    yaml_files.sort()
    unique_map_names = []
    unique_loc_names = []
    for f in yaml_files:
        map_traj, loc_traj = parse_evaluation_file_name(f)
        unique_map_names.append(map_traj)
        unique_loc_names.append(loc_traj)
    unique_map_names = sorted(list(set(unique_map_names)))
    unique_loc_names = sorted(list(set(unique_loc_names)))

    number_of_deployments_map = len(unique_map_names)
    number_of_deployments_loc = len(unique_loc_names)

    matrix = np.full((number_of_deployments_map, number_of_deployments_loc), np.nan)
    add_marker_matrix = np.full(
        (number_of_deployments_map, number_of_deployments_loc), False
    )

    unique_map_name_index_map = {name: i for i, name in enumerate(unique_map_names)}
    unique_loc_name_index_map = {name: i for i, name in enumerate(unique_loc_names)}

    labels_maps = []
    labels_locs = []
    deployment_mapping = DEPLOYMENT_DATE_LABEL

    for f in yaml_files:
        map_traj, loc_traj = parse_evaluation_file_name(f)
        with open(Path(path) / f, "r") as file:
            data = yaml.safe_load(file)
            add_marker = False
            value = np.nan
            if metric == Metric.APE:
                try:
                    value = data["ape"]["rmse_meters"]
                except Exception as e:
                    print(f"Error processing file {f}: {e}")
            else:
                try:
                    data = data[metric.name.lower()]
                    value, _std = compute_rte(data, max_delta)
                    try:
                        add_marker = data["trajectories"]["shortened"]
                    except KeyError:
                        pass
                except Exception as e:
                    print(f"Error processing file {f}: {e}")
            map_idx = unique_map_name_index_map[map_traj]
            loc_idx = unique_loc_name_index_map[loc_traj]
            # Update the matrices
            matrix[map_idx, loc_idx] = value
            add_marker_matrix[map_idx, loc_idx] = add_marker

            if len(labels_maps) < number_of_deployments_map:
                for key in deployment_mapping.keys():
                    if key in f:
                        label = deployment_mapping[key]
                        if label not in labels_maps:
                            labels_maps.append(label)

            if len(labels_locs) < number_of_deployments_loc:
                for key in deployment_mapping.keys():
                    if key in f:
                        label = deployment_mapping[key]
                        if label not in labels_locs:
                            labels_locs.append(label)

    return matrix, add_marker_matrix, labels_maps, labels_locs
