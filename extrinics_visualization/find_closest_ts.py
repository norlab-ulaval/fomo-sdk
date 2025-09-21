from pathlib import Path
import cv2
import os
# so this script will generate 3  rows of images
# the first row
## lidar points projected on top of a camera image 
# the second row 
# # will be radar lidar point clouds overlay
# the third row
## the two lidar point clouds overlay


ijrr_folder = "/home/samqiao/ASRL/fomo-public-sdk/raw_fomo_rosbags/red_2024-11-21-10-34"

# we need to get the lidar points and camera image and radar image in the ijrr_folder

target_frame_timestamp = 1732203284093580.607 # micro secs

# radar folder 
radar_folder = os.path.join(ijrr_folder, "navtech")

# find the radar image that is closest to the target frame timestamp
# what is in the folder file name is like 1732203284.png

files = sorted([f for f in Path(radar_folder).iterdir() if f.is_file() and f.name.endswith(".png")])


# find the radar image that is closest to the target frame timestamp
closest_file = min(files, key=lambda x: abs(float(x.stem) - target_frame_timestamp))
print("the closest radar file is: ", closest_file)

# how many files are there?
print(len(files))

# load the radar image
radar_image = cv2.imread(closest_file)


# similarly I need to find the rslidar point cloud that is closest to the target frame timestamp
rslidar_folder = os.path.join(ijrr_folder, "robosense")

files = sorted([f for f in Path(rslidar_folder).iterdir() if f.is_file() and f.name.endswith(".bin")])


# find the rslidar point cloud that is closest to the target frame timestamp
closest_file = min(files, key=lambda x: abs(float(x.stem) - target_frame_timestamp))
print("the closest rslidar file is: ", closest_file)

## also the lslidar point cloud that is closest to the target frame timestamp
lslidar_folder = os.path.join(ijrr_folder, "leishen")
files = sorted([f for f in Path(lslidar_folder).iterdir() if f.is_file() and f.name.endswith(".bin")])


# find the lslidar point cloud that is closest to the target frame timestamp
closest_file = min(files, key=lambda x: abs(float(x.stem) - target_frame_timestamp))
print("the closest lslidar file is: ", closest_file)

# and the zedx camera
zedx_folder = os.path.join(ijrr_folder, "zedx_left")
files = sorted([f for f in Path(zedx_folder).iterdir() if f.is_file() and f.name.endswith(".png")])


# find the zedx camera image that is closest to the target frame timestamp
closest_file = min(files, key=lambda x: abs(float(x.stem) - target_frame_timestamp))
print("the closest zedx_left file is: ", closest_file)

# load the zedx camera image
zedx_image = cv2.imread(closest_file)

# there is also a zedx_right camera image
zedx_right_folder = os.path.join(ijrr_folder, "zedx_right")
files = sorted([f for f in Path(zedx_right_folder).iterdir() if f.is_file() and f.name.endswith(".png")])


# find the zedx right camera image that is closest to the target frame timestamp
closest_file = min(files, key=lambda x: abs(float(x.stem) - target_frame_timestamp))
print("the closest zedx_right file is: ", closest_file)

# ok last one is the basler camera image
basler_folder = os.path.join(ijrr_folder, "basler")
files = sorted([f for f in Path(basler_folder).iterdir() if f.is_file() and f.name.endswith(".png")])


# find the basler camera image that is closest to the target frame timestamp
closest_file = min(files, key=lambda x: abs(float(x.stem) - target_frame_timestamp))
print("the closest basler file is: ", closest_file)

