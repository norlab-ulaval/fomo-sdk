#!/usr/bin/env python3
# The script reads a QR code from an image file, trying multiple strategies to decode it.
import sys
import os
# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cv2
import argparse
from rosbags.rosbag2 import Reader
from rosbags.typesys import get_typestore, Stores
import tqdm
import numpy as np
from utils import *


def try_decode(detector, img):
    data, pts, _ = detector.detectAndDecode(img)
    if data:
        return data
    # try multi
    retval, datas, points, _ = detector.detectAndDecodeMulti(img)
    if datas and len(datas) > 0 and datas[0]:
        return datas[0]
    return None

def process_img(img):
    detector = cv2.QRCodeDetector()

    # Whole image
    data = try_decode(detector, img)
    if data:
        print("Decoded QR payload (full image):", data)
        return data

    # Heuristic ROI: right side where we placed it (adjust margins if needed)
    H, W = img.shape[:2]
    roi_x0 = max(0, int(W - 40 - 320 - 20))  # match writer placement (x0 ≈ W - QR_SIZE - 40)
    roi_y0 = max(0, (H - 320)//2 - 20)
    roi_x1 = min(W, roi_x0 + 320 + 40)
    roi_y1 = min(H, roi_y0 + 320 + 40)
    roi = img[roi_y0:roi_y1, roi_x0:roi_x1]

    data = try_decode(detector, roi)
    if data:
        print("Decoded QR payload (ROI):", data)
        return data

    # Preprocess ROI for clarity
    # Check if image is already grayscale
    if len(roi.shape) == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi
    # Upscale to make modules bigger for detector
    gray_big = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_NEAREST)
    # Binarize (strong contrast)
    _, bw = cv2.threshold(gray_big, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    data = try_decode(detector, bw)
    if data:
        print("Decoded QR payload (ROI preprocessed):", data)
        return data

    print("No QR code detected. Tips:")
    print("- Ensure USE_QR=True in the writer.")
    print("- Keep QR_SIZE >= 300 and QR_BORDER >= 4.")
    print("- Use PNG (lossless); avoid JPEG compression.")
    print("- Try better lighting/contrast if capturing from a photo or screen.")

def main():
    DEBUG = True
    print("Starting the script")
    parser = argparse.ArgumentParser(description='Convert ROS2 bag to sensor data files.')
    parser.add_argument('--input', type=str, help='Path to input bag file')
    # parser.add_argument('--output', type=str,help='Output directory')
    # parser.add_argument('--overwrite', action='store_true', help='Overwrite existing output directory')
    args = parser.parse_args()

    # cwd
    print(os.getcwd())

    ros_bag_path = args.input
    if not os.path.exists(ros_bag_path):
        print(f"Error: input bag file {ros_bag_path} does not exist.")
        sys.exit(1)

    print(f"Reading image from {ros_bag_path}")

    micro_ros_timstamps = []
    micro_qr_timestamps = []
    delays = []

    # offset csv path
    offset_csv_path = os.path.join(ros_bag_path, "qr_code_execution_delay_look_up.csv")
    if not os.path.exists(offset_csv_path):
        print(f"Error: offset csv file {offset_csv_path} does not exist.")
        sys.exit(1)
    # read the offset csv file
    import pandas as pd

    # Or if your CSV has headers:
    df = pd.read_csv(offset_csv_path)
    offset_dict = dict(zip(df['qr_encoded_value'], df['code_execution_time_in_ns']))
    print(f"dictionary shape: {len(offset_dict)}")

    success_count = 0
    failure_count = 0

    with Reader(ros_bag_path) as reader:
            # Create typestore for deserialization
            typestore = get_typestore(Stores.ROS2_FOXY)
            connections = list(reader.connections)
            if DEBUG:
                print(f"Found {len(connections)} connections in the bag file.")

            # loop through all the connections
            for connection, timestamp, rawdata in tqdm.tqdm(reader.messages(), total=reader.message_count, desc="Processing data"):
                try:
                    topic_name = connection.topic
                    if topic_name == "/zed_node/left_gray/image_rect_gray":
                        msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
                        timestamp = msg.header.stamp
                        nano_sec = msg.header.stamp.nanosec
                        stamp_in_micro = timestamp.sec * 1_000_000 + (nano_sec // 1_000) # use microsecs

                        print(f"Image timestamp (microsecs): {stamp_in_micro}")
                        img = image_to_numpy(msg)

                        # print(f"Image shape: {img.shape}, dtype: {img.dtype}")

                        # # # show this image as greyscale
                        # if DEBUG:
                        #     cv2.imshow("Image", img)
                        #     cv2.waitKey(1)
                        qr_timestamp = process_img(img)
                        print(f"QR timestamp: {qr_timestamp}")
                        if qr_timestamp is not None:
                            qr_timestamp = int(qr_timestamp)

                            # find the offset in the offset_dict, note that the offset is in nanoseconds
                            offset = offset_dict[qr_timestamp]
                            offset = offset // 1_000 # convert to microseconds
                            print(f"Offset: {offset}")

                            micro_qr_timestamps.append(qr_timestamp // 1_000) # convert to microsecs
                            micro_ros_timstamps.append(stamp_in_micro)


                            delay = stamp_in_micro - (qr_timestamp//1_000 + offset) # delay is in microseconds
                            delays.append(delay)

                            success_count += 1
                        else:
                            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!No QR code detected!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                            failure_count += 1
                except Exception as e:
                    print(f'Error processing message: {str(e)}')
    
    #   code_execution_time = offset_dict[qr_timestamp]
    micro_ros_timstamps = np.array(micro_ros_timstamps)
    micro_qr_timestamps = np.array(micro_qr_timestamps)
    delays = np.array(delays)

    print(f"Total images processed: {len(micro_ros_timstamps)}")
    print(f"Total QR codes decoded: {len(micro_qr_timestamps)}")

    # the success rate
    success_rate = success_count / (success_count + failure_count)
    print(f"Success rate: {success_rate}")

    # I like to get the mean and std of the delays
    mean_delay = np.mean(delays)
    std_delay = np.std(delays)
    print(f"Mean delay: {mean_delay}")
    print(f"Std delay: {std_delay}")

    # plot the mean and std of the delays
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(delays, marker='o', linestyle='-', markersize=3)
    plt.axhline(mean_delay, color='r', linestyle='--', label=f'Mean Delay: {mean_delay:.1f} µs')
    plt.fill_between(range(len(delays)), mean_delay - std_delay, mean_delay + std_delay, color='gray', alpha=0.5, label='±1 Std Dev')
    plt.title('Delays between ROS Image Timestamps and QR Code Timestamps')
    plt.xlabel('Image Index')
    plt.ylabel('Delay (microseconds)')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    # plot the offset
    plt.figure(figsize=(10, 6))
    plt.plot(offset_dict.keys(), offset_dict.values(), marker='o', linestyle='-', markersize=3)
    plt.title('Offset between QR Code Timestamps and Code Execution Time')
    plt.xlabel('QR Code Timestamp')
    plt.ylabel('Offset (nanoseconds)')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

#     differences = micro_ros_timstamps - micro_qr_timestamps
#     mean_diff = np.mean(differences)
#     std_diff = np.std(differences)
#     print(f"Mean difference (us): {mean_diff}")
#     print(f"Std deviation of difference (us): {std_diff}")

#     # do a plot here
#     import matplotlib.pyplot as plt
#     plt.figure(figsize=(10, 6))
#     plt.plot(differences, marker='o', linestyle='-', markersize=3)
#     plt.axhline(mean_diff, color='r', linestyle='--', label=f'Mean Difference : {mean_diff:.1f} µs')
#     plt.fill_between(range(len(differences)), mean_diff - std_diff, mean_diff + std_diff, color='gray', alpha=0.5, label='±1 Std Dev')
#     plt.title('Differences between ROS Image Timestamps and QR Code Timestamps')
#     plt.xlabel('Image Index')
#     plt.ylabel('Time Difference (microseconds)')
#     plt.legend()
#     plt.grid()
#     plt.tight_layout()
#     # plt.savefig('timestamp_differences.png')
#     plt.show()

    
    


if __name__ == '__main__':
    main()