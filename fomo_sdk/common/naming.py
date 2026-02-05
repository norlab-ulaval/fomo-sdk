import re
from enum import Enum
from pathlib import Path

DEPLOYMENT_DATE_LABEL = {
    "2024-11-21": "Nov21",
    "2024-11-28": "Nov28",
    "2025-01-10": "Jan10",
    "2025-01-29": "Jan29",
    "2025-01-30": "Jan29",
    "2025-03-10": "Mar10",
    "2025-03-14": "Mar10",
    "2025-04-15": "Apr15",
    "2025-05-28": "May28",
    "2025-06-26": "Jun26",
    "2025-08-20": "Aug20",
    "2025-09-24": "Sep24",
    "2025-10-14": "Oct14",
    "2025-11-03": "Nov03",
}
DEPLOYMENT_LABEL_DATE = {v: k for k, v in DEPLOYMENT_DATE_LABEL.items()}


def construct_path_from_filename(filename: str | Path) -> Path:
    if isinstance(filename, Path):
        filename = filename.name
    localization_date = construct_localization_recording(filename)
    deployment = "-".join(localization_date.split("_")[1].split("-")[:3])
    return Path(deployment) / Path(localization_date)


def construct_path(
    dataset_base_path: str | Path, deployment: str, trajectory: str
) -> Path:
    if deployment in DEPLOYMENT_DATE_LABEL.keys():
        deployment_date = deployment
    else:
        deployment_date = DEPLOYMENT_LABEL_DATE[deployment]

    if not Trajectory.is_valid(trajectory):
        raise ValueError(
            f"Trajectory {trajectory} is invalid. Valid trajectories are: {Trajectory.__members__}"
        )

    path = Path(dataset_base_path) / deployment_date
    trajectory_full = None
    for dir in [d.name for d in path.iterdir() if d.is_dir()]:
        if trajectory in dir:
            trajectory_full = dir
            break
    if trajectory_full is None:
        raise ValueError(f"Trajectory {trajectory} not found in {path}")

    return path / trajectory_full


def construct_evaluation_file_name(
    recording_mapping: str, recording_localization: str, suffix: str = ".txt"
) -> str:
    return f"{recording_mapping}_{recording_localization}{suffix}"


def construct_mapping_recording(filename: str) -> str:
    return filename.split("_")[0] + "_" + filename.split("_")[1]


def construct_localization_recording(filename: str) -> str:
    return filename.split("_")[2] + "_" + filename.split("_")[3].split(".")[0]


def construct_deployment(recording: str) -> str:
    deployment = "-".join(recording.split("_")[1].split("-")[:3])
    if deployment == "2025-01-30":
        return "2025-01-29"
    if deployment == "2025-03-14":
        return "2025-03-10"
    return deployment


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


class Trajectory(Enum):
    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))

    @classmethod
    def is_valid(cls, trajectory: str):
        return trajectory in cls.list()

    RED = "red"
    BLUE = "blue"
    GREEN = "green"
    ORANGE = "orange"
    YELLOW = "yellow"
    MAGENTA = "magenta"


#### SLAM implementations
class Slam(Enum):
    PROPRIOCEPTIVE = "proprioceptive"
    NORLAB_ICP_MAPPER = "norlab_icp_mapper"
    RTR = "rtr"
    ORB_SLAM3 = "orb_slam3"
    KISS = "kiss"
    NAVTECH_RADAR_SLAM = "navtech_radar_slam"


def get_slam_title(slam: Slam):
    if slam == Slam.PROPRIOCEPTIVE.value:
        return "Proprioceptive"
    elif slam == Slam.NORLAB_ICP_MAPPER.value:
        return "Norlab ICP Mapper"
    elif slam == Slam.RTR.value:
        return "Radar-Gyro Teach and Repeat"
    elif slam == Slam.ORB_SLAM3.value:
        return "ORB-SLAM3"
    elif slam == Slam.KISS.value:
        return "KISS-ICP/SLAM"
    elif slam == Slam.NAVTECH_RADAR_SLAM.value:
        return "Kaist Radar SLAM"
    else:
        return slam


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
