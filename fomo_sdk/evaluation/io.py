import yaml


def export_results_to_yaml(
    filename, avg_relative_rpe, ape_rmse, rpe_results, trajectories
):
    """
    Save the computed metrics (average RPE and ATE RMSE along with detailed RPE stats) to a YAML file.
    """
    rpe_details = {
        f"{delta}m": {
            "rmse_meters": float(stats["rmse"]),
            "std_meters": float(stats["std"]),
            "min_meters": float(stats["min"]),
            "max_meters": float(stats["max"]),
        }
        for delta, stats in rpe_results.items()
    }
    data = {
        "results": {
            "ape_rmse_meters": ape_rmse,
            "rpe_avg_rmse_percentage": avg_relative_rpe,
        },
        "rpe_details": rpe_details,
        "trajectories": trajectories,
    }
    with open(filename, "w") as file:
        yaml.dump(data, file)
