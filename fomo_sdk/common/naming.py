from pathlib import Path
from enum import Enum
import re

DEPLOYMENT_DATE_LABEL = {
    "2024-11-21": "Nov21",
    "2024-11-28": "Nov28",
    "2025-01-10": "Jan10",
    "2025-01-29": "Jan29",
    "2025-01-30": "Jan29",
    "2025-03-10": "Mar10",
    "2025-03-14": "Mar10",
    "2025-04-16": "Apr16",
    "2025-05-28": "May28",
    "2025-06-26": "Jun26",
    "2025-08-20": "Aug20",
    "2025-09-24": "Sep24",
    "2025-10-14": "Oct14",
    "2025-11-03": "Nov03",
}
DEPLOYMENT_LABEL_DATE = {v: k for k, v in DEPLOYMENT_DATE_LABEL.items()}


def construct_path(
    dataset_base_path: str | Path, deployment: str, trajectory: str
) -> Path:
    if deployment in DEPLOYMENT_DATE_LABEL.keys():
        deployment_date = deployment
    else:
        deployment_date = DEPLOYMENT_LABEL_DATE[deployment]

    path = Path(dataset_base_path) / deployment_date
    trajectory_full = None
    for dir in [d.name for d in path.iterdir() if d.is_dir()]:
        if trajectory in dir:
            trajectory_full = dir
            break
    if trajectory_full is None:
        raise ValueError(f"Trajectory {trajectory} not found in {path}")

    return path / trajectory_full


def parse_proprioceptive_file_name(file_name: Path | str):
    if isinstance(file_name, Path):
        file_name = file_name.name
    if file_name.count("_") != 3:
        raise ValueError(f"Invalid file name: {file_name}")
    odom_file = file_name.replace(".txt", "").split("_")[2:]
    odom_file = (
        odom_file[0]
        + "_"
        + odom_file[1]
        + "_"
        + odom_file[0]
        + "_"
        + odom_file[1]
        + ".txt"
    )
    return odom_file


#### SLAM implementations


class Slam(Enum):
    PROPRIOCEPTIVE = "proprioceptive"
    LIDAR = "lidar"
    RTR = "rtr"
    VSLAM = "vslam"


def get_slam_title(slam: Slam):
    if slam == Slam.PROPRIOCEPTIVE.value:
        return "Proprioceptive"
    elif slam == Slam.LIDAR.value:
        return "Lidar-Inertial Odometry"
    elif slam == Slam.RTR.value:
        return "Radar-Gyro Teach and Repeat"
    elif slam == Slam.VSLAM.value:
        return "Stereo-Inertial Visual SLAM"
    else:
        raise ValueError(f"Unknown SLAM: {slam}")


def check_recording_name(recording_name: str):
    """
    Check if the recording name is valid.
    Expected format is:
    <name>_<YYYY-MM-DD-HH-MM>
    """
    if recording_name.count("_") != 1:
        raise ValueError(f"Invalid recording name: {recording_name}")
    name, timestamp = recording_name.split("_")
    if not re.match(r"\d{4}-\d{2}-\d{2}-\d{2}-\d{2}", timestamp):
        raise ValueError(f"Invalid recording name: {recording_name}")
    return True
