import os

from attrs import field
import rclpy
from rclpy.node import Node
import numpy as np
import cv2
import argparse
from pathlib import Path

from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr
from rosbags.typesys import get_typestore, Stores,get_types_from_msg, register_types

from sensor_msgs_py import point_cloud2
from sensor_msgs.msg import *
from rosbags.typesys.types import sensor_msgs__msg__PointCloud2 as PointCloud2Msg


import tqdm

import audio_utils as autils


# Define custom message types
RADAR_SCAN_MSG = """
# A ROS message carrying a B Scan and its associated metadata (e.g. timestamps, encoder IDs)
sensor_msgs/Image b_scan_img
uint16[] encoder_values
uint64[] timestamps
"""

RADAR_FFT_MSG = """
# A ROS message based on an FFT data message from a radar
std_msgs/Header header
uint8[] angle
uint8[] azimuth
uint8[] sweep_counter
uint8[] ntp_seconds
uint8[] ntp_split_seconds
uint8[] data
uint8[] data_length
"""
RADAR_CONFIG_MSG = """
# A ROS message based on a configuration data message from a radar, with corrected types

# add a header message to hold message timestamp
std_msgs/Header header

# azimuth_samples (uint16)
uint16 azimuth_samples

# encoder_size (uint16)
uint16 encoder_size

# user-provided azimuth offset (uint16)
uint16 azimuth_offset

# bin_size (float64)
float32 bin_size

# range_in_bins (uint16)
uint16 range_in_bins

# expected_rotation_rate (uint16)
uint16 expected_rotation_rate

# range_gain (float32)
float32 range_gain

# range_offset (float32)
float32 range_offset
"""
SYSNERGIE_MSG = """
# A ROS message for Sysnergie batteries
std_msgs/Header header

float32 pack_voltage
float32 pack_current

int8 state
int8 main_fault_reg
int8 soc
int64 remaining_capacity

float32 min_temp
float32 max_temp
float32 min_cell_volt
float32 max_cell_volt

float32 brick_volt_avg
float32 cell_volt_avg
float32 pack_temp_avg
int64 pack_imbalance
"""
CLEARPATH_POWER_MSG = """
# A ROS message for Clearpath power
# Robot Power readings

std_msgs/Header header

# AC Power
int8 NOT_APPLICABLE=-1

int8 shore_power_connected  # Indicates if AC power is connected.
int8 battery_connected      # Indicates if battery is connected.
int8 power_12v_user_nominal # Indicates if the 12V user power is good.
int8 charger_connected      # Indicates if a charger is connected.
int8 charging_complete      # Indicates if charging is complete.

# Voltage rails, in volts
# Averaged over the message period

# Jackal (J100)
uint8 J100_MEASURED_BATTERY=0
uint8 J100_MEASURED_5V=1
uint8 J100_MEASURED_12V=2

# Dingo 1.0 (D100)
uint8 D100_MEASURED_BATTERY=0
uint8 D100_MEASURED_5V=1
uint8 D100_MEASURED_12V=2

# Dingo 1.5 (D150)
uint8 D150_MEASURED_BATTERY=0
uint8 D150_MEASURED_5V=1
uint8 D150_MEASURED_12V=2

# Warthog (W200)
uint8 W200_MEASURED_BATTERY=0
uint8 W200_MEASURED_12V=1
uint8 W200_MEASURED_24V=2
uint8 W200_MEASURED_48V=3

# Ridgeback (R100)
uint8 R100_MEASURED_BATTERY=0
uint8 R100_MEASURED_5V=1
uint8 R100_MEASURED_12V=2
uint8 R100_MEASURED_INVERTER=3
uint8 R100_MEASURED_FRONT_AXLE=4
uint8 R100_MEASURED_REAR_AXLE=5
uint8 R100_MEASURED_LIGHT=6

# Husky (A200)
uint8 A200_BATTERY_VOLTAGE=0
uint8 A200_LEFT_DRIVER_VOLTAGE=1
uint8 A200_RIGHT_DRIVER_VOLTAGE=2
uint8 A200_VOLTAGES_SIZE=3

float32[] measured_voltages

# Current senses available on platform, in amps.
# Averaged over the message period

# Jackal (J100)
uint8 J100_TOTAL_CURRENT=0
uint8 J100_COMPUTER_CURRENT=1
uint8 J100_DRIVE_CURRENT=2
uint8 J100_USER_CURRENT=3

# Dingo 1.0 (D100)
uint8 D100_TOTAL_CURRENT=0
uint8 D100_COMPUTER_CURRENT=1

# Dingo 1.5 (D150)
uint8 D150_TOTAL_CURRENT=0
uint8 D150_COMPUTER_CURRENT=1

# Warthog (W200)
uint8 W200_TOTAL_CURRENT=0
uint8 W200_COMPUTER_CURRENT=1
uint8 W200_12V_CURRENT=2
uint8 W200_24V_CURRENT=3

# Ridgeback (R100)
uint8 R100_TOTAL_CURRENT=0

# Husky (A200)
uint8 A200_MCU_AND_USER_PORT_CURRENT=0
uint8 A200_LEFT_DRIVER_CURRENT=1
uint8 A200_RIGHT_DRIVER_CURRENT=2
uint8 A200_CURRENTS_SIZE=3

float32[] measured_currents
"""
CLEARPATH_STATUS_MSG = """
# This message represents lower-frequency status updates
# Default publish frequency is 1Hz.

std_msgs/Header header

# Robot Hardware ID
string hardware_id

# Firmware version
string firmware_version

# Times since MCU power-on.
builtin_interfaces/Duration mcu_uptime
builtin_interfaces/Duration connection_uptime

# Temperature of MCU's PCB in Celsius.
float32 pcb_temperature
float32 mcu_temperature
"""
CLEARPATH_LIGHTS_MSG = """
# Represents a command for the pairs of RGB body lights on a CPR robot.

# Dingo 1.0 (D100)
uint8 D100_LIGHTS_REAR_LEFT=0
uint8 D100_LIGHTS_FRONT_LEFT=1
uint8 D100_LIGHTS_FRONT_RIGHT=2
uint8 D100_LIGHTS_REAR_RIGHT=3

# Dingo 1.5 (D150)
uint8 D150_LIGHTS_REAR_LEFT=0
uint8 D150_LIGHTS_FRONT_LEFT=1
uint8 D150_LIGHTS_FRONT_RIGHT=2
uint8 D150_LIGHTS_REAR_RIGHT=3

# Ridgeback (R100)
uint8 R100_LIGHTS_FRONT_PORT_UPPER=0
uint8 R100_LIGHTS_FRONT_PORT_LOWER=1
uint8 R100_LIGHTS_FRONT_STARBOARD_UPPER=2
uint8 R100_LIGHTS_FRONT_STARBOARD_LOWER=3
uint8 R100_LIGHTS_REAR_PORT_UPPER=4
uint8 R100_LIGHTS_REAR_PORT_LOWER=5
uint8 R100_LIGHTS_REAR_STARBOARD_UPPER=6
uint8 R100_LIGHTS_REAR_STARBOARD_LOWER=7

# Warthog (W200)
uint8 W200_LIGHTS_FRONT_LEFT=0
uint8 W200_LIGHTS_FRONT_RIGHT=1
uint8 W200_LIGHTS_REAR_LEFT=2
uint8 W200_LIGHTS_REAR_RIGHT=3

RGB[] lights
"""
CLEARPATH_RGB_MSG = """
# Represents the intensity of a single RGB LED, either reported or commanded.
# Each channel is limited to a range of [0, 255]

uint8 red
uint8 green
uint8 blue
"""

BASLER_CURRENT_PARAMS_MSG = """
uint32 offset_x # -20000 = Error
uint32 offset_y # -20000 = Error
bool reverse_x
bool reverse_y

int32 black_level # -10000 = error/not available

int32 pgi_mode # -3 = Unknown, -2 = Error, -1 = Not available, 0 = Off, 1 = On
int32 demosaicing_mode # -3 = Unknown, -2 = Error, -1 = Not available, 0 = Simple, 1 = BaslerPGI
float32 noise_reduction # -20000.0 = Error, -10000.0 = Not available
float32 sharpness_enhancement # -20000.0 = Error, -10000.0 = Not available
int32 light_source_preset # -3 = Unknown, -2 = Error, -1 = Not available, 0 = Off, 1 = Daylight5000K, 2 = Daylight6500K, 3 = Tungsten2800K
int32 balance_white_auto # -3 = Unknown, -2 = Error, -1 = Not available, 0 = Off, 1 = Once, 2 = Continuous

int32 sensor_readout_mode # -3 = Unknown, -2 = Error, -1 = Not available, 0 = Normal, 1 = Fast
int32 acquisition_frame_count # -20000 = Error, -10000 = Not available
int32 trigger_selector # -3 = Unknown, -2 = Error, -1 = Not available, 0 = FrameStart, 1 = FrameBurstStart(USB)/AcquisitionStart(GigE)
int32 trigger_mode # -3 = Unknown, -2 = Error, -1 = Not available, 0 = Off, 1 = On
int32 trigger_source # -3 = Unknown, -2 = Error, -1 = Not available, 0 = Software, 1 = Line1, 2 = Line3, 3 = Line4, 4 = Action1(Selected Gige)
int32 trigger_activation # -3 = Unknown, -2 = Error, -1 = Not available, 0 = RisingEdge, 1 = FallingEdge
float32 trigger_delay # -20000.0 = Error, -10000.0 = Not available

int32 user_set_selector # -3 = Unknown, -2 = Error, -1 = Not available, 0 = Default, 1 = UserSet1, 2 = UserSet2, 3 = UserSet3, 4 = HighGain, 5 = AutoFunctions, 6 = ColorRaw
int32 user_set_default_selector # -3 = Unknown, -2 = Error, -1 = Not available, 0 = Default, 1 = UserSet1, 2 = UserSet2, 3 = UserSet3, 4 = HighGain, 5 = AutoFunctions, 6 = ColorRaw

bool is_sleeping
float32 brightness
float32 exposure
float32 gain
float32 gamma
uint32 binning_x
uint32 binning_y
float32 temperature # Shows the camera temperature. If not available, then 0.0. USB uses DeviceTemperature and GigE TemperatureAbs parameters.
int32 max_num_buffer		# -2 = Error, -1 = Not available
sensor_msgs/RegionOfInterest roi

string[] available_image_encoding
string current_image_encoding
string current_image_ros_encoding

bool success
string message

string ptp_status           # latched state of the PTP clock, see https://ja.docs.baslerweb.com/pylonapi/net/T_Basler_Pylon_PLCamera_PtpStatusEnum
string ptp_servo_status     # latched state of the clock servo, see https://docs.baslerweb.com/pylonapi/net/T_Basler_Pylon_PLCamera_PtpServoStatusEnum
int64 ptp_offset    # ptp offset from master in ticks [ns]
"""
RTF_MSG = """
std_msgs/Header header

float64 c0
float64 c1
float64 c00
float64 c01
float64 c10
float64 c11
float64 c20
float64 c21
float64 c30

float64 raw_pressure
float64 raw_temperature
float64 scale_pressure
float64 scale_temperature
float64 pressure
float64 temperature
"""

AUDIO_INFO_MSG = """
# This message contains the audio meta data

# Number of channels
uint8 channels
# Sampling rate [Hz]
uint32 sample_rate
# Audio format (e.g. S16LE)
string sample_format
# Amount of audio data per second [bits/s]
uint32 bitrate
# Audio coding format (e.g. WAVE, MP3)
string coding_format
"""

AUDIO_DATA_MSG = """
uint8[] data
"""

AUDIO_DATA_STAMPED_MSG = """
std_msgs/Header header
audio_common_msgs/AudioData audio
"""
POINT_CLOUD2_MSG= """
std_msgs/Header header
uint32 height
uint32 width
sensor_msgs/PointField[] fields
bool is_bigendian
uint32 point_step
uint32 row_step
uint8[] data
bool is_dense
"""


def get_fomo_typestore():
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    typestore.register(
        get_types_from_msg(RADAR_FFT_MSG, "nav_messages/msg/RadarFftDataMsg")
    )
    typestore.register(
        get_types_from_msg(RADAR_SCAN_MSG, "navtech_msgs/msg/RadarBScanMsg")
    )
    typestore.register(
        get_types_from_msg(RADAR_CONFIG_MSG, "navtech_msgs/msg/RadarConfigurationMsg")
    )
    typestore.register(
        get_types_from_msg(CLEARPATH_POWER_MSG, "clearpath_platform_msgs/msg/Power")
    )
    typestore.register(
        get_types_from_msg(SYSNERGIE_MSG, "sysnergie_msgs/msg/BatteryLog")
    )
    typestore.register(
        get_types_from_msg(CLEARPATH_STATUS_MSG, "clearpath_platform_msgs/msg/Status")
    )
    typestore.register(
        get_types_from_msg(CLEARPATH_LIGHTS_MSG, "clearpath_platform_msgs/msg/Lights")
    )
    typestore.register(
        get_types_from_msg(CLEARPATH_RGB_MSG, "clearpath_platform_msgs/msg/RGB")
    )
    typestore.register(
        get_types_from_msg(
            BASLER_CURRENT_PARAMS_MSG, "pylon_ros2_camera_interfaces/msg/CurrentParams"
        )
    )
    typestore.register(
        get_types_from_msg(RTF_MSG, "rtf_sensors_msgs/msg/CustomPressureTemperature")
    )

    typestore.register(
        get_types_from_msg(AUDIO_INFO_MSG, "audio_common_msgs/msg/AudioInfo")
    )
    typestore.register(
        get_types_from_msg(AUDIO_DATA_MSG, "audio_common_msgs/msg/AudioData")
    )
    typestore.register(
        get_types_from_msg(
            AUDIO_DATA_STAMPED_MSG, "audio_common_msgs/msg/AudioDataStamped"
        )
    )


    return typestore

def to_ros_pointcloud2(msg):
    from sensor_msgs.msg import PointCloud2, PointField
    from std_msgs.msg import Header

    ros_msg = PointCloud2()
    ros_msg.header = Header()
    ros_msg.header.stamp.sec = msg.header.stamp.sec
    ros_msg.header.stamp.nanosec = msg.header.stamp.nanosec
    ros_msg.header.frame_id = msg.header.frame_id

    ros_msg.height = msg.height
    ros_msg.width = msg.width
    ros_msg.is_bigendian = msg.is_bigendian
    ros_msg.point_step = msg.point_step
    ros_msg.row_step = msg.row_step
    ros_msg.is_dense = msg.is_dense
    ros_msg.data = list(bytearray(msg.data))  # ✅ FIXED: convert bytes to list[int]

    ros_msg.fields = []
    for f in msg.fields:
        pf = PointField()
        pf.name = f.name
        pf.offset = f.offset
        pf.datatype = f.datatype
        pf.count = f.count
        ros_msg.fields.append(pf)

    return ros_msg

class BagToDir():
    def __init__(self, bag_file, output_dir):
        self.bag_file = bag_file
        # radar
        self.radar_image_dir = os.path.join(output_dir, 'radar')
        os.makedirs(self.radar_image_dir, exist_ok=True)

        # lidar
        self.ls_lidar_bin_dir = os.path.join(output_dir, 'lslidar')
        os.makedirs(self.ls_lidar_bin_dir, exist_ok=True)

        self.rs_lidar_bin_dir = os.path.join(output_dir, 'rslidar')
        os.makedirs(self.rs_lidar_bin_dir, exist_ok=True)
        
        # imu
        self.vn100_imu_file = open(os.path.join(output_dir, 'vn100.csv'), 'w')
        self.vn100_imu_file.write("timestamp,ang_vel_x,ang_vel_y,ang_vel_z,lin_acc_x,lin_acc_y,lin_acc_z\n")

        self.mti30_imu_file = open(os.path.join(output_dir, 'mti30.csv'), 'w')
        self.mti30_imu_file.write("timestamp,ang_vel_x,ang_vel_y,ang_vel_z,lin_acc_x,lin_acc_y,lin_acc_z\n")
        
        # camera

        # audio
        self.first_msg = True
        self.read_bag()

    def read_bag(self):
        bag_path = Path(self.bag_file)
        typestore = get_fomo_typestore()


        with Reader(bag_path) as reader:
            reader.open()

            for connection, timestamp, rawdata in reader.messages():
                try:
                    msg = deserialize_cdr(rawdata, connection.msgtype, typestore=typestore)
                    topic_name = connection.topic

                    # print("connection.msgtype:", connection.msgtype)


                    if topic_name.startswith('/radar/b_scan_msg'):
                        self.save_radar_image(msg)

                    if topic_name.startswith('/lslidar128/points'):
                        print("Saving LS Lidar bins")
        
                        self.save_lidar_bins(msg, self.ls_lidar_bin_dir)
                    elif topic_name.startswith('/rslidar128/points'):
                        print("Saving RS Lidar bins")
                        self.save_lidar_bins(msg, self.rs_lidar_bin_dir)

              
                    if topic_name.startswith('/vn100/data_raw'):
                        ts = float(msg.header.stamp.sec) + msg.header.stamp.nanosec * 1e-9
                        ang_vel = msg.angular_velocity
                        lin_acc = msg.linear_acceleration
                        self.vn100_imu_file.write(f"{ts},{ang_vel.x},{ang_vel.y},{ang_vel.z},{lin_acc.x},{lin_acc.y},{lin_acc.z}\n")
                    elif topic_name.startswith('/mti30/data_raw'):
                        ts = float(msg.header.stamp.sec) + msg.header.stamp.nanosec * 1e-9
                        ang_vel = msg.angular_velocity
                        lin_acc = msg.linear_acceleration
                        self.mti30_imu_file.write(f"{ts},{ang_vel.x},{ang_vel.y},{ang_vel.z},{lin_acc.x},{lin_acc.y},{lin_acc.z}\n")

                except Exception as e:
                    print(f'Error processing message: {str(e)}')

        self.vn100_imu_file.close()
        self.mti30_imu_file.close()

    def save_radar_image(self, msg):
        try:
            img_msg = msg.b_scan_img
            timestamp_row = np.array(msg.timestamps, dtype=np.uint64)
            encoder_values = np.array(msg.encoder_values, dtype=np.uint16)

            radar_data = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(img_msg.height, img_msg.width)
            timestamp = img_msg.header.stamp
            image_filename = os.path.join(self.radar_image_dir, f'{timestamp.sec}{int(timestamp.nanosec/1e3):06d}.png')

            timestamp_bytes = np.frombuffer(timestamp_row.tobytes(), dtype=np.uint8).reshape(-1, 8)
            encoder_bytes = np.frombuffer(encoder_values.tobytes(), dtype=np.uint8).reshape(-1, 2)

            final_data = np.zeros((radar_data.shape[0], radar_data.shape[1] + 11), dtype=np.uint8)
            final_data[:, :8] = timestamp_bytes
            final_data[:, 8:10] = encoder_bytes
            final_data[:, 11:] = radar_data

            cv2.imwrite(image_filename, final_data)
            print(f'Saved image: {image_filename}')

        except Exception as e:
            print(f'Error saving radar image: {str(e)}')

    def save_lidar_bins(self, msg, lidar_bin_dir):
        from sensor_msgs.msg import PointCloud2, PointField
        from std_msgs.msg import Header
        from builtin_interfaces.msg import Time

        def convert_to_ros2_pointcloud2(msg):
            # Reconstruct ROS2 Time object manually
            stamp = Time()
            stamp.sec = msg.header.stamp.sec
            stamp.nanosec = msg.header.stamp.nanosec

            header = Header()
            header.stamp = stamp
            header.frame_id = msg.header.frame_id

            fields = []
            for f in msg.fields:
                pf = PointField()
                pf.name = f.name
                pf.offset = f.offset
                pf.datatype = f.datatype
                pf.count = f.count
                fields.append(pf)

            return PointCloud2(
                header=header,
                height=msg.height,
                width=msg.width,
                fields=fields,
                is_bigendian=msg.is_bigendian,
                point_step=msg.point_step,
                row_step=msg.row_step,
                data=bytes(msg.data),
                is_dense=msg.is_dense,
            )
        try:
            timestamp = msg.header.stamp.sec * 1e9 + msg.header.stamp.nanosec

            print("timestamp:", timestamp/1e9)

            # Convert rosbags dynamic msg to true PointCloud2
            pc2_msg = convert_to_ros2_pointcloud2(msg)  
      
            cloud_points = list(point_cloud2.read_points(
                pc2_msg,
                field_names=('x', 'y', 'z', 'intensity', 'ring', 'timestamp'),
                skip_nans=True))

            points = np.array(cloud_points, dtype=[
                ('x', np.float32),
                ('y', np.float32),
                ('z', np.float32),
                ('intensity', np.float32),
                ('ring', np.uint16),
                ('timestamp', np.uint64),
            ])

            # print one of the points
            if len(points) > 0:
                print(f'Point: {points[0:50]}')

            timestamp = int(timestamp / 1000)
            points.tofile(os.path.join(lidar_bin_dir, f'{timestamp}.bin'))
            print(f'Saved bin: {lidar_bin_dir}/{timestamp}.bin')

        except Exception as e:
            print(f'Error saving lidar bins: {str(e)}')
    

    def save_rectified_camera_imgs(self, msg, camera_dir):
        # camera info topic 
        # /zed_node/right_raw/camera_info | Type: sensor_msgs/msg/CameraInfo | Count: 3662 | Serialization Format: cdr
        # /zed_node/left_raw/camera_info | Type: sensor_msgs/msg/CameraInfo | Count: 3662 | Serialization Format: cdr
        # stereo camera ZED X
        # /zed_node/right_raw/image_raw_color | Type: sensor_msgs/msg/Image | Count: 3662 | Serialization Format: cdr
        # /zed_node/left_raw/image_raw_color | Type: sensor_msgs/msg/Image | Count: 3662 | Serialization Format: cdr

        # so I need two folders one for the left and one for the right camera

        # I also wanna save a video in the end

        # np_arr = np.fromstring(msg.data, np.uint8)
        #         img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        #         img = undistort(img, K, dist, roi, P[:3, :3])
        #         cv2.imwrite(root + "camera/" + str(timestamp) + ".png", img)
        pass



    # def get_audio_data(input: str) -> autils.Stereo:
    #     typestore = get_fomo_typestore()
    #     audio_stereo = autils.Stereo()

    #     with Reader(input) as reader:
    #         total_messages = reader.message_count
    #         for connection, timestamp, rawdata in tqdm(
    #             reader.messages(connections=reader.connections),
    #             total=total_messages,
    #             desc=f"Exporting audio data from {input}",
    #         ):
    #             if audio_stereo.is_mic_topic(connection.topic):
    #                 audio_stereo.add_message(connection, timestamp, rawdata, typestore)

    #     audio_stereo.postprocess_audio_data()
    #     return audio_stereo


    def undistort(self, img, K, dist, roi=None, P=None):
        dst = cv2.undistort(img, K, dist, None, P)
        if roi is not None and P is not None:
            h, w, _ = img.shape
            x, y, w2, h2 = roi
            dst = dst[y:y+h2, x:x+w2]
            dst = cv2.resize(dst, (w, h))
        return dst


def main(args=None):
    parser = argparse.ArgumentParser(description='Convert MCAP ROS2 bag to radar/lidar images and data.')
    parser.add_argument('--input', type=str, default='/home/samqiao/ASRL/fomo-public-sdk/raw_fomo_rosbags/deployment4/red/', help='Path to the MCAP or DB3 bag file')
    parser.add_argument('--output', type=str,  default='/home/samqiao/ASRL/fomo-public-sdk/output',help='Directory to store the output files')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite the output directory if it exists')
    args = parser.parse_args()

    if os.path.exists(args.output) and args.overwrite:
        import shutil
        shutil.rmtree(args.output)
    os.makedirs(args.output, exist_ok=True)

    try:
        converter = BagToDir(args.input, args.output)
        converter.read_bag()
    except Exception as e:
        print(f'Error: {str(e)}')
    finally:

        print("Conversion Done!")

if __name__ == '__main__':
    main()
