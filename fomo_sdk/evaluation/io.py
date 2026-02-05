import yaml

from fomo_sdk.evaluation.utils import Metric


def export_results_to_yaml(filename, ape_rmse, rpe_results, trajectories):
    """
    Save the computed metrics (average RPE and ATE RMSE along with detailed RPE stats) to a YAML file.
    """
    data = {
        Metric.APE.name.lower(): {
            "rmse_meters": ape_rmse,
        },
        "trajectories": trajectories,
    }
    for metric, stats in rpe_results.items():
        data[metric.name.lower()] = {
            f"{delta}m": {
                "rmse_meters": float(stats["rmse"]),
                "std_meters": float(stats["std"]),
                "min_meters": float(stats["min"]),
                "max_meters": float(stats["max"]),
            }
            for delta, stats in stats.items()
        }
    with open(filename, "w") as file:
        yaml.dump(data, file)
