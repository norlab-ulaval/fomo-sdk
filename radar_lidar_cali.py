import os
from pathlib import Path
import numpy as np
import math
import cv2
import argparse
from rosbags.rosbag2 import Reader
from rosbags.typesys import get_typestore, Stores, get_types_from_msg
from rosbags.serde import deserialize_cdr
import torch
import tqdm
from utils import *

import open3d as o3d

from matplotlib import pyplot as plt

from radar_lidar_cali_utils import *

import pyboreas as pb

# Define custom message types
RADAR_SCAN_MSG = """
# A ROS message carrying a B Scan and its associated metadata (e.g. timestamps, encoder IDs)
sensor_msgs/Image b_scan_img
uint16[] encoder_values
uint64[] timestamps
"""

DATA_TYPES = {
    1: np.int8,
    2: np.uint8,
    3: np.int16,
    4: np.uint16,
    5: np.int32,
    6: np.uint32,
    7: np.float32,
    8: np.float64,
}

def get_fomo_typestore():
    typestore = get_typestore(Stores.ROS2_HUMBLE)

    typestore.register(
        get_types_from_msg(RADAR_SCAN_MSG, "navtech_msgs/msg/RadarBScanMsg")
    )

    return typestore

def extract_radar_from_bag(bag_path, radar_msg):
    typestore = get_fomo_typestore()

    polar_img = []
    azimuths = []
    azimuth_timestamps = []
    timestamps = []

    with Reader(bag_path) as reader:
        reader.typestore = typestore
        connections = list(reader.connections)

        # loop through all the connections
        for connection, timestamp, rawdata in tqdm.tqdm(reader.messages(), total=reader.message_count, desc="Processing data"):
            try:
                topic_name = connection.topic
                
                # radar navtech topic
                if topic_name.startswith('/radar/b_scan_msg'):
                    msg = deserialize_cdr(rawdata, connection.msgtype, typestore=typestore)

                    img_msg = msg.b_scan_img
                    timestamp_row = np.array(msg.timestamps, dtype=np.uint64) # possibly in nano-secs

                    # put it in microsecs
                    timestamp_row = np.floor(timestamp_row / 1000).astype(np.uint64)  # convert to microseconds
                    # print("Timestamp row:", timestamp_row)
                    encoder_values = np.array(msg.encoder_values, dtype=np.uint16)

                    radar_data = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(img_msg.height, img_msg.width)

                    timestamp = img_msg.header.stamp
                    nano_sec = img_msg.header.stamp.nanosec
                    stamp_in_micro = timestamp.sec * 1_000_000 + (nano_sec // 1_000) # use microsecs

                    polar_img.append(radar_data)
                    azimuths.append(encoder_values)
                    azimuth_timestamps.append(timestamp_row)
                    timestamps.append(stamp_in_micro)


            except Exception as e:
                    print(f'Error processing message: {str(e)}')

        return np.array(polar_img), np.array(azimuths), np.array(azimuth_timestamps), np.array(timestamps).reshape(-1, 1)
    

def extract_lidar_from_bag(bag_path):
    typestore = get_fomo_typestore()

    lidar_pts = []
    timestamps = []
    dict = {}

    with Reader(bag_path) as reader:
        reader.typestore = typestore
        connections = list(reader.connections)

        # loop through all the connections
        for connection, timestamp, rawdata in tqdm.tqdm(reader.messages(), total=reader.message_count, desc="Processing data"):
            try:
                topic_name = connection.topic

                if topic_name.startswith('/rslidar128/points'):
                    msg = typestore.deserialize_cdr(rawdata, connection.msgtype)

                    array_dtype =  np.dtype([
                    ('x',         np.float32),
                    ('y',         np.float32),
                    ('z',         np.float32),
                    ('intensity', np.float32),
                    ('ring',      np.uint16),
                    ('timestamp', np.uint64), # floast64 for rslidar was ('timestamp', np.float64)
                    ])
                    # 2) Compute number of points
                    point_step   = msg.point_step              # bytes per point
        
                    total_bytes  = len(msg.data)

                    num_points   = total_bytes // point_step

                    arr = np.zeros(num_points, dtype=array_dtype)

                    timestamp = msg.header.stamp
                    micro_timestamp = timestamp.sec * 1_000_000 + (timestamp.nanosec // 1_000) # for rslidar this refers to the end of the scan 

                    # 3) View the buffer as that structured array
                    data = np.frombuffer(msg.data, dtype=np.uint8).reshape(-1,msg.point_step)


                    for field in msg.fields:
                        # print(field.datatype, field.name, field.offset, field.count)
                        name = field.name
                        offset = field.offset
                        type = DATA_TYPES[field.datatype] # this is the data type of the field
                        num_bytes = np.dtype(type).itemsize

                        raw = data[:, offset:offset + num_bytes].ravel()

                        col = np.frombuffer(raw, dtype=type)

                        if name == 'timestamp':
                            arr[name] = ((col * 1_000_000_000) // 1_000).astype(np.uint64) # in microseconds # (here matej does 1e9 as int) then / 1_000
                        else:
                            arr[name] = col

                    # get rid of invalid points (nans)
                    if not msg.is_dense:
                        mask = ~np.isnan(arr['x']) & ~np.isnan(arr['y']) & ~np.isnan(arr['z'])
                        arr = arr[mask]

                # print("10 data:", arr[0:50])
                    export_timestamp = np.min(arr['timestamp']) # microseconds # first point timestamp

                    # timestamps.append(export_timestamp)
                    # lidar_pts.append(arr)
                    dict[export_timestamp] = np.array([arr['x'], arr['y'], arr['z']]).T # only keep the xyz points

            except Exception as e:
                    print(f'Error processing message: {str(e)}')

        return dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate radar and lidar data.")
    parser.add_argument("--bag_dir", type=Path, default= "/home/samqiao/ASRL/fomo-public-sdk/raw_fomo_rosbags/static", help="Path to the radar ROS bag file.")
    args = parser.parse_args()

    bag_dir = args.bag_dir

    # I want to see the subfolders within the parent folder
    subfolders = [f for f in bag_dir.iterdir() if f.is_dir()]
    print("Subfolders:")
    for subfolder in subfolders:
        print(f" - {subfolder.name}")

    T_final = [] # stores the final result

    DISPLAY = True
    VISUALIZE = True
    # from navtech to rslidar
    T_lidar_radar_initial = np.array([[1,0,0,0.891],
                                [0,-1,0,0],
                                [0,0,-1,0.121],
                                [0,0,0,1]])
    # CIR-304H radar parameters
    radar_resolution = 0.043809514
    encoder_size = 5600
    num_of_range_bins = 6848
    max_range = 300

    for subfolder in subfolders:
        subfolder_path = os.path.join(bag_dir, subfolder.name)
        print(f"----------------------------------Processing subfolder path: {subfolder_path}")
        # check if the subfolder name contains "static"
        if not "static" in subfolder.name:
            continue
        
        # extract radar info
        polar, azimuths, timestamps , msg_timestamp = extract_radar_from_bag(subfolder_path, RADAR_SCAN_MSG)
        # verify the shape
        print(f'Polar shape: {polar.shape}, Azimuths shape: {azimuths.shape}, Timestamps shape: {timestamps.shape}, Msg timestamp: {msg_timestamp.shape}')


        # extract lidar info
        lidar_timestamp2pts = extract_lidar_from_bag(subfolder_path)
        # the length of the dictionary is
        print(f'Lidar timestamp to points dictionary length: {len(lidar_timestamp2pts)}')


        keys_time_stamps = list(lidar_timestamp2pts.keys())

        # verify the shape
        print(f'Example lidar points shape: {lidar_timestamp2pts[keys_time_stamps[0]].shape}')

        # lets do one frame 
        nb_radar_msgs = polar.shape[0]
        print(f'Number of radar messages: {nb_radar_msgs}')

        T_icp = [] # stores each folder result 40 of them

        for radar_idx in range(0, nb_radar_msgs):
            print("processing radar_idx:", radar_idx)
            polar_img = polar[radar_idx] / 255.0 
            azimuth = azimuths[radar_idx]
            timestamp = timestamps[radar_idx]
            radar_msg_timestamp = msg_timestamp[radar_idx]

            print("shape info")
            print("polar shape:", polar_img.shape)
            print("azimuth shape:", azimuth.shape)
            print("timestamp shape:", timestamp.shape)
            print("radar_msg_timestamp shape:", radar_msg_timestamp.shape)

            import torch
            import torchvision
            # preprocessing steps
            device = 'cpu'
            polar_intensity = torch.tensor(polar_img).to(device)
            polar_std = torch.std(polar_intensity, dim=1)
            polar_mean = torch.mean(polar_intensity, dim=1)
            polar_intensity -= (polar_mean.unsqueeze(1) + 2*polar_std.unsqueeze(1))
            polar_intensity[polar_intensity < 0] = 0
            polar_intensity = torchvision.transforms.functional.gaussian_blur(polar_intensity.unsqueeze(0), (9,1), 3).squeeze()
            polar_intensity /= torch.max(polar_intensity, dim=1, keepdim=True)[0]
            polar_intensity[torch.isnan(polar_intensity)] = 0


            # if DISPLAY:
            #     # lets viusalize the polar img in grey scale
            #     plt.imshow(polar_img, cmap='gray')
            #     plt.colorbar()
            #     plt.title(f'Polar Image at Timestamp: {timestamp[radar_idx]}')
            #     plt.xlabel('Range Bin')
            #     plt.show()

            # lets extract points from the polar img
            KPEAKS = True # K-peaks is the best extractor
            if KPEAKS:
                targets = KPeaks(polar_intensity.numpy(),minr=5,maxr=300,res=radar_resolution, K=10, static_threshold=0.25)
            else:
                targets = modifiedCACFAR(polar_intensity.numpy(),minr=5,maxr=300,res=radar_resolution, width=137, guard=7, threshold=0.50, threshold2=0.0, threshold3=0.30)

            radar_pts = []

            for target in targets:
                azimuth_idx = int(target[0])
                range_idx = int(target[1])

                x = range_idx * radar_resolution * math.cos(2 * math.pi * azimuth[azimuth_idx] / encoder_size)
                y = range_idx * radar_resolution * math.sin(2 * math.pi * azimuth[azimuth_idx] / encoder_size)
                intensity = polar_intensity[azimuth_idx, range_idx]

                radar_pts.append([x,y,intensity])

            radar_pts = np.array(radar_pts)
            radar_pts = radar_pts[:,0:2]


            radar_cart_img = pb.utils.radar.radar_polar_to_cartesian(azimuth/encoder_size*2*np.pi,polar_intensity,radar_resolution,cart_resolution=0.224,cart_pixel_width=3000)

            print(f'Radar points shape: {radar_pts.shape}')

            # # # # to visualize the radar_pts in 2d and its correponding cart image on the side
            # if DISPLAY:
            #     plt.subplot(1, 2, 1)
            #     plt.scatter(radar_pts[:, 0], radar_pts[:, 1], cmap='gray',s=1)
            #     plt.xlabel('X')
            #     plt.ylabel('Y')
            #     plt.axis('equal')

            #     plt.subplot(1, 2, 2)
            #     plt.imshow(radar_cart_img, cmap='gray')
            #     plt.show()


            # print(f'K-strong targets shape: {k_strong_targets.shape}')


            closest_lidar_timestamp = min(lidar_timestamp2pts.keys(), key=lambda x: abs(x - radar_msg_timestamp))
            lidar_pts = lidar_timestamp2pts[closest_lidar_timestamp]

            # lets remove ground in the lidar pts
            pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(lidar_pts))
            plane, inliers = pc.segment_plane(0.05, 3, 2000)  # 5cm threshold
            lidar_pts_no_ground = np.asarray(pc.select_by_index(inliers, invert=True).points)

            print(f'Lidar points shape: {lidar_pts.shape}')
            print(f'Lidar points no ground shape: {lidar_pts_no_ground.shape}')

            # visualize lidar pts in 3d using matplot lib
            # if DISPLAY:
            #     fig = plt.figure()
            #     ax = fig.add_subplot(111, projection='3d')
            #     ax.scatter(lidar_pts[:,0], lidar_pts[:,1], lidar_pts[:,2], c='gray')
            #     ax.set_xlabel('X')
            #     ax.set_ylabel('Y')
            #     ax.set_zlabel('Z')
            #     plt.show()


            T_ref, fitness, rmse = icp_multistage(radar_pts, lidar_pts_no_ground,T_init=T_lidar_radar_initial) # I update this initial guess later on


            # print(f'ICP Result - T_ref: {T_ref}, Fitness: {fitness}, RMSE: {rmse}')


            # do a check here
            def invert_se3(T):
                R, t = T[:3,:3], T[:3,3]
                Ti = np.eye(4); Ti[:3,:3] = R.T; Ti[:3,3] = -R.T @ t
                return Ti
            
            T_lidar_radar_initial = np.array([[ 1., 0., 0., 0.891],
                            [ 0.,-1., 0., 0.   ],
                            [ 0., 0.,-1., 0.121],
                            [ 0., 0., 0., 1.   ]])

            # Delta transform: what ICP changed relative to CAD
            Delta = T_ref @ invert_se3(T_lidar_radar_initial)
            dR, dt = Delta[:3,:3], Delta[:3,3]
            angle = np.degrees(np.arccos(np.clip((np.trace(dR)-1)/2, -1, 1)))
            print("Δt (m):", dt, "  ‖Δt‖ =", np.linalg.norm(dt), "m")
            print("ΔR (deg):", angle)

            # reject outliners
            if angle > 10 or np.linalg.norm(dt) > 1.0:
                print("Rejected due to large change from initial guess.")
                continue

            T_icp.append(T_ref)

            # T_lidar_radar_initial = T_ref # update initial guess

            if radar_idx == 10:
                T_avg = se3_mean(T_icp)
                print("Averaged T_icp:\n", T_avg)
        
                # # check alignment
                if VISUALIZE:
                    visualize_xy_overlay(radar_pts, lidar_pts_no_ground, T_avg)

                    # visualize_alignment_3d(radar_pts, lidar_pts_no_ground, T_avg,voxel_lidar=0.05, voxel_radar=0.03, point_size=2.0)

                break

        T_final.append(T_avg)
    

    

    T_final_avg = se3_mean(T_final)

    print(f'Final average transformation matrix:\n{T_final_avg}')
