import re
import os
import cv2
from fomo_sdk.image.utils import create_no_data_image
import numpy as np
from pathlib import Path
from fomo_sdk.common.naming import check_recording_name, DEPLOYMENT_DATE_LABEL

loop_matcher = re.compile(r"/(.+)_202")


def walk_with_max_depth(top_dir, max_depth):
    top_dir = os.path.abspath(top_dir)  # Ensure absolute path for accurate comparison
    initial_depth = top_dir.count(os.sep)

    for root, dirs, files in os.walk(top_dir, topdown=True):
        current_depth = root.count(os.sep)

        # Yield the current root, dirs, and files
        yield root, dirs, files

        # Check if we are at or beyond the maximum desired depth
        if current_depth - initial_depth >= max_depth:
            # Clear the 'dirs' list to prevent further recursion into subdirectories
            del dirs[:]


def find_gt_files(root_dir) -> list[Path]:
    gt_files = []
    for dirpath, _, filenames in walk_with_max_depth(root_dir, 2):
        gt_found = False
        for filename in filenames:
            if filename == "gt.txt":
                gt_files.append(Path(dirpath, filename).relative_to(root_dir))
                gt_found = True
                break

        if not gt_found and loop_matcher.search(dirpath):
            print(f"Folder {dirpath} contains no ground truth")

    return gt_files


def find_closest_file(timestamp: float, dir: Path, ext: str) -> Path | None:
    file_path = dir / f"{int(timestamp * 1e6)}.{ext}"
    # Find the file >= path
    for file in sorted(os.listdir(dir)):
        if file >= file_path.name:
            return dir / file
    raise FileNotFoundError(f"No file found in {dir} for timestamp {timestamp}")


def find_closest_image(timestamp: float, image_dir: Path) -> np.ndarray:
    image_path = find_closest_file(timestamp, image_dir, "png")

    img = cv2.imread(image_path)
    if np.mean(img) > 254:
        print(f"Image {image_path} appears to be empty")
    if img is not None:
        return img
    # If image not found, add a blank image
    print(f"Image {image_path} is invalid!")
    return create_no_data_image()


def verify_fomo_recording(recording_path: Path) -> tuple[str, str]:
    """
    Verify that the recording is a valid FoMo recording.
    A valid FoMo recording must be in this format:
    1) Path:
        <deployment>/<recording>, where <deployment> is in the <YYYY-MM-DD> format and <recording> is in the <trajectory_name>_<YYYY-MM-DD-HH-MM> format.
    2) Content:
        Must contain a calib folder
        Contains at least one of the following folders:
            audio_left
            audio_right
            basler
            leishen
            navtech
            robosense
            vectornav
            xsens
            zedx_left
            zedx_right
    """
    if not recording_path.exists():
        raise ValueError(f"Recording {recording_path} does not exist")
    if not recording_path.is_dir():
        raise ValueError(f"Recording {recording_path} is not a directory")
    if not check_recording_name(recording_path.name):
        raise ValueError(f"Recording {recording_path} does not have a valid name")
    if recording_path.parent.name not in DEPLOYMENT_DATE_LABEL.keys():
        raise ValueError(
            f"Recording {recording_path} does not have a valid deployment date"
        )
    if not (recording_path / "calib").exists():
        raise ValueError(f"Recording {recording_path} does not contain a calib folder")
    if not any(
        (recording_path / folder).exists()
        for folder in [
            "audio_left",
            "audio_right",
            "basler",
            "leishen",
            "navtech",
            "robosense",
            "vectornav",
            "xsens",
            "zedx_left",
            "zedx_right",
        ]
    ):
        raise ValueError(
            f"Recording {recording_path} does not contain any of the required folders"
        )
    return recording_path.parent.name, recording_path.name
