from pathlib import Path
import yaml
import numpy as np

from fomo_sdk.common.naming import DEPLOYMENT_DATE_LABEL


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


def compute_rpe(data: dict) -> tuple[float, float]:
    """
    Compute the relative translation drift for a given evaluation dictionary.
    """
    rpe = []
    std = []
    # Calculate mean RPE across different deltas (100m to 800m)
    for delta in range(100, 801, 100):
        relative_drift = 100 * data["rpe_details"][f"{delta}m"]["rmse_meters"] / delta
        relative_std = 100 * data["rpe_details"][f"{delta}m"]["std_meters"] / delta
        rpe.append(relative_drift)
        std.append(relative_std)
    rpe = np.mean(rpe)
    std = np.mean(std)
    return rpe, std


def construct_matrix(path: str):
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

    ape_matrix = np.full((number_of_deployments_map, number_of_deployments_loc), np.nan)
    rpe_matrix = np.full((number_of_deployments_map, number_of_deployments_loc), np.nan)
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
            try:
                ape = data["results"]["ape_rmse_meters"]
                rpe, _std = compute_rpe(data)
                try:
                    add_marker = data["trajectories"]["shortened"]
                except KeyError:
                    pass
            except KeyError:
                ape = np.nan
                rpe = np.nan
            except Exception as e:
                print(f"Error processing file {f}: {e}")
                ape = np.nan
                rpe = np.nan
                continue
            map_idx = unique_map_name_index_map[map_traj]
            loc_idx = unique_loc_name_index_map[loc_traj]
            # Update the matrices
            ape_matrix[map_idx, loc_idx] = ape
            rpe_matrix[map_idx, loc_idx] = rpe
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

    return ape_matrix, rpe_matrix, add_marker_matrix, labels_maps, labels_locs
