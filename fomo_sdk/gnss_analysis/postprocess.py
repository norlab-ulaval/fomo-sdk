import glob
import os
import re
import shutil
import subprocess
import time
from datetime import datetime
from ftplib import FTP

from emlid_gnss_to_rosbag import main as emlid_gnss_to_rosbag


def exctract_coordinates(path):
    # Open the file and read the last line
    with open(path, "r") as file:
        lines = file.readlines()

    # Extract the last line (coordinates are on the last line)
    last_line = lines[-1]

    # Split the last line into columns
    columns = last_line.split()

    print(columns)

    # Extract the latitude, longitude, and height
    return (float(columns[2]), float(columns[3]), float(columns[4]))


def find_file(extension, path, exclude=[], log=True):
    pattern = f"{path}/**/*{extension}"
    files = glob.glob(pattern, recursive=True)
    if len(exclude) > 0:
        files = [f for f in files if not any([e in f for e in exclude])]
    if not files or len(files) > 1:
        if log:
            print(f"Error: No file ending with {extension} found in {path}")
        return None
    elif len(files) > 1:
        if log:
            print(f"Error: Multiple files ending with {extension} found in {path}")
        return None
    else:
        return files[0]


def get_session_letter(time_str):
    # Parse hour from the time string
    hour = int(time_str[:2])

    # Mapping of hours to letters for GMT
    hour_to_letter_gmt = {
        0: "A",
        1: "B",
        2: "C",
        3: "D",
        4: "E",
        5: "F",
        6: "G",
        7: "H",
        8: "I",
        9: "J",
        10: "K",
        11: "L",
        12: "M",
        13: "N",
        14: "O",
        15: "P",
        16: "Q",
        17: "R",
        18: "S",
        19: "T",
        20: "U",
        21: "V",
        22: "W",
        23: "X",
    }

    # Get the letter based on the hour for GMT
    letter = hour_to_letter_gmt.get(hour, "Invalid hour").lower()

    return letter


def get_path_autocomplete(msg=""):
    return (
        subprocess.check_output(
            f'read -e -p "{msg}" var ; echo $var',
            shell=True,
        )
        .rstrip()
        .decode("utf-8")
    )


def copy_logs_to_tmp(path, path_out):
    if path.endswith(".zip"):
        # unzip
        os.system(f"unzip -u {path} -d {path_out}")
    else:
        os.system(f"cp -r {path} {path_out}")


def wait_for_pos_file(path):
    print(f"looking for files in {path}")
    path_pos = find_file("pos", path, exclude=["event"], log=False)
    print("Waiting for processing to finish...")
    while not path_pos or not os.path.exists(path_pos):
        path_pos = find_file("pos", path, exclude=["event"], log=False)
        time.sleep(1)
    return path_pos


if not os.path.exists("/tmp/emlid_static"):
    static_path = get_path_autocomplete(
        "Specify Emlid Static antenna .zip logs folder:\n"
    )
    copy_logs_to_tmp(static_path, "/tmp/emlid_static")
static_path = "/tmp/emlid_static"

print(f"Debug: static copied to {static_path}")

# List all files in the directory ending with "O"
static_O_file = find_file("O", static_path)

# Print the file (assuming there's only one match)
if not static_O_file:
    exit(1)
with open(static_O_file, "r") as file:
    file.readline()  # Skip the first line
    second_line = file.readline().strip()  # Read the second line

    # Regular expression to match the datetime format
    match = re.search(r"\d{8} \d{6}", second_line)
    if match:
        datetime_str = match.group()
        date_str, time_str = datetime_str.split(" ")
        print("Extracted datetime:", date_str, time_str)
        date = datetime.strptime(date_str, "%Y%m%d")
        # Start of the year
        start_of_year = datetime(date.year, 1, 1)

        # Calculate the difference in days
        days_since_start = (
            date - start_of_year
        ).days + 1  # +1 to include the current day

        print("Days since the year's start:", days_since_start)
        cros_basename = "ATR2" + str(days_since_start).zfill(3)

        cros_foldername = cros_basename + str(date.year)[-1]
        session_letter = get_session_letter(time_str)
        cros_filename = cros_basename + session_letter + ".zip"
        cros_full_path = os.path.join(
            "/Public/GPS/Quebec",
            cros_foldername,
            cros_filename,
        )

    else:
        print("No datetime found on the second line.")
        exit(1)


cros_path = "/tmp/emlid_cros"
if not os.path.exists(cros_path):
    print(f"Downloading CROS file from: {cros_full_path}")
    with FTP("ftp.mrn.gouv.qc.ca", encoding="latin-1") as ftp:
        ftp.login()  # Anonymous login
        # Download the remote file
        static_O_file = ftp.nlst(os.path.join("/Public/GPS/Quebec", cros_foldername))
        ftp_command = "RETR " + cros_full_path
        local_cros_archive = "/tmp/emlid_cros.zip"
        ftp.retrbinary(
            ftp_command,
            open(local_cros_archive, "wb").write,
        )

    print(f"Unzipping to {cros_path}")
    os.system(f"unzip -u {local_cros_archive} -d {cros_path}")

static_path_O = find_file("O", static_path)
cros_path_O = find_file("O", cros_path)
static_path_P = find_file("P", static_path)
static_antenna_height = f"{4.5} TODO"

print("\n--- Emlid Studio Setup Instructions ---")
print("1. Open Emlid Studio and select 'Static'.")
print(f"2. In the Static field, use the file: {static_path_O}")
print(f"3. Set the antenna height to: {static_antenna_height} meters")
print(f"4. In the Base field, use the file: {cros_path_O}")
print(f"5. In the Navigation field, use the file: {static_path_P}")
print("6. Click 'Process' to begin the operation.\n")

static_path_pos = wait_for_pos_file(static_path)
static_coords = exctract_coordinates(static_path_pos)
print(f"New processed static antenna coordinates detected! File: {static_path_pos}")
print(f"Postprocessed coordinates: {static_coords}")

rover_pose_files = {}
processing_path = "/tmp/emlid_processing"
if not os.path.exists(processing_path):
    os.makedirs(processing_path)
print("\n=== Kinematics processing Instructions ===")
for receiver in ["front", "left", "right"]:
    if not os.path.exists(f"/tmp/emlid_{receiver}"):
        rover_path = get_path_autocomplete(
            f"\nSpecify Emlid {receiver} .zip logs folder:\n"
        )
        copy_logs_to_tmp(rover_path, f"/tmp/emlid_{receiver}")
    rover_path = f"/tmp/emlid_{receiver}"
    rover_path_O = find_file("O", rover_path)
    rover_path_P = find_file("O", rover_path)
    rover_antenna_height = "1.05 TODO"

    print("\n--- Emlid Studio Kinematics Setup Instructions ---")
    print("1. Open Emlid Studio and select 'Kinematics'.")
    print(f"2. In the Rover field, use the file: {rover_path_O}")
    print(f"3. Set the antenna height to: {rover_antenna_height} meters")
    print(f"4. In the Base field, use the file: {static_path_O}")
    print(
        f"5. Drag and drop this file: {static_path_pos} over the Latitude and Longitude field in the Base section."
    )
    print(f"6. In the Navigation field, use the file: {rover_path_P}")
    print("7. Click 'Process' to begin the operation.\n")

    rover_path_pos = wait_for_pos_file(rover_path)
    print(f"New processed {receiver} .pos detected! File: {rover_path_pos}")
    rover_pose_files[receiver] = os.path.join(processing_path, receiver + ".pos")
    print(f"Copying the file to {rover_pose_files[receiver]}...")
    shutil.copy(rover_path_pos, rover_pose_files[receiver])

print("Kinematics processing completed")
print("\n=== Converting the .pos files into a rosbag file ===\n")

anwer = input(
    "Do you want to filter the GNSS poses automatically using a rosbag? (y/n): "
)

timestamp_start = None
timestamp_end = None
rosbag_path = None
if anwer.lower() == "y":
    rosbag_path = get_path_autocomplete("\nSpecify the rosbag folder path:\n")
else:
    timestamp_start = input(
        "Specify the start timestamp in epoch time [seconds] (or hit enter to ignore):\n"
    )
    timestamp_end = input(
        "Specify the end timestamp in epoch time [seconds] (or hit enter to ignore):\n"
    )
    if timestamp_start == "":
        timestamp_start = None
    else:
        timestamp_start = float(timestamp_start)
    if timestamp_end == "":
        timestamp_end = None
    else:
        timestamp_end = float(timestamp_end)
    if timestamp_start is not None and timestamp_end is None:
        print(f"Keeping the GNSS poses after {timestamp_start}...")
    elif timestamp_start is None and timestamp_end is not None:
        print(f"Keeping the GNSS poses using before {timestamp_end}...")
    elif timestamp_start is not None and timestamp_end is not None:
        print(
            f"Keeping the GNSS poses using between {timestamp_start} and {timestamp_end}..."
        )
print("Converting GNSS poses into a ros2 bag file...")
topic_namespace = input("Set base topic namespace (e.g. /emlid): ")
if topic_namespace[0] != "/":
    topic_namespace = "/" + topic_namespace
if topic_namespace[-1] == "/":
    topic_namespace = topic_namespace[:-1]

output_rosbag_path = "/tmp/emlid_rosbag"
emlid_gnss_to_rosbag(
    processing_path,
    output_rosbag_path,
    timestamp_start,
    timestamp_end,
    rosbag_path,
    overwrite=True,
    topic_namespace=topic_namespace,
)
