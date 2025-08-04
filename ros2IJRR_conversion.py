import os
from pathlib import Path
import numpy as np
import cv2
import argparse
import rclpy
from rosbags.rosbag2 import Reader
from rosbags.typesys import get_typestore, Stores, get_types_from_msg
from rosbags.serde import deserialize_cdr
# from sensor_msgs_py import point_cloud2
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


class BagToDir():
    def __init__(self, bag_file, output_dir):
        self.bag_file = bag_file
        self.output_dir = output_dir
        
        # Create output directories
        self.radar_image_dir = os.path.join(output_dir, 'radar')
        self.ls_lidar_bin_dir = os.path.join(output_dir, 'lslidar')
        self.rs_lidar_bin_dir = os.path.join(output_dir, 'rslidar')
        self.zed_node_right_dir = os.path.join(output_dir, 'zednode_right')
        self.zed_node_left_dir = os.path.join(output_dir, 'zednode_left')
        
        os.makedirs(self.radar_image_dir, exist_ok=True)
        os.makedirs(self.ls_lidar_bin_dir, exist_ok=True)
        os.makedirs(self.rs_lidar_bin_dir, exist_ok=True)
        os.makedirs(self.zed_node_right_dir, exist_ok=True)
        os.makedirs(self.zed_node_left_dir, exist_ok=True)
        
        # Initialize camera parameters
        self.K_left = self.D_left = self.R_left = self.P_left = None
        self.K_right = self.D_right = self.R_right = self.P_right = None
        
        # Initialize rectification maps
        self.map1_left = self.map2_left = None
        self.map1_right = self.map2_right = None
        
        # Open IMU files
        self.vn100_imu_file = open(os.path.join(output_dir, 'vn100.csv'), 'w')
        self.vn100_imu_file.write("timestamp,ang_vel_x,ang_vel_y,ang_vel_z,lin_acc_x,lin_acc_y,lin_acc_z\n")
        self.mti30_imu_file = open(os.path.join(output_dir, 'mti30.csv'), 'w')
        self.mti30_imu_file.write("timestamp,ang_vel_x,ang_vel_y,ang_vel_z,lin_acc_x,lin_acc_y,lin_acc_z\n")

        # Image map constructed
        self.zed_map_processed = False

    def read_bag(self):
        bag_path = Path(self.bag_file)
        typestore = get_fomo_typestore()

        with Reader(bag_path) as reader:
            print("haha")
            reader.typestore = typestore
            connections = list(reader.connections)
            
            # First pass: Process camera info to set rectification maps
            for connection, timestamp, rawdata in tqdm.tqdm(reader.messages(), total=reader.message_count, desc="Processing camera info"):
                if connection.topic in ['/zed_node/right_raw/camera_info', '/zed_node/left_raw/camera_info']:
                    self.process_camera_info(connection, rawdata, typestore)

                    if self.zed_map_processed:
                        # If both camera infos are processed, break the loop
                        break

            print("Camera info processed, rectification maps set.")
                    
            # Second pass: Process all data
            for connection, timestamp, rawdata in tqdm.tqdm(reader.messages(), total=reader.message_count, desc="Processing data"):
                try:
                    topic_name = connection.topic
                    
                    if topic_name.startswith('/radar/b_scan_msg'):
                        self.save_radar_image(connection, rawdata, typestore)
                    elif topic_name.startswith('/lslidar128/points'):
                        self.save_lslidar_bins(connection, rawdata, typestore, self.ls_lidar_bin_dir)
                    elif topic_name.startswith('/rslidar128/points'):
                        self.save_rslidar_bins(connection, rawdata, typestore, self.rs_lidar_bin_dir)
                    elif topic_name.startswith('/vn100/data_raw'):
                        self.save_imu_data(connection, rawdata, typestore, self.vn100_imu_file)
                    elif topic_name.startswith('/mti30/data_raw'):
                        self.save_imu_data(connection, rawdata, typestore, self.mti30_imu_file)
                    elif topic_name in ['/zed_node/right_raw/image_raw_color', '/zed_node/left_raw/image_raw_color']:
                        self.save_camera_image(connection, rawdata, typestore)
                    elif topic_name in ["/audio/left_mic", "/audio/right_mic"]:
                        self.save_audio_data(connection, rawdata, typestore)

                except Exception as e:
                    print(f'Error processing message: {str(e)}')

        # Close files
        self.vn100_imu_file.close()
        self.mti30_imu_file.close()

    def process_camera_info(self, connection, rawdata, typestore):
        msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
        
        if connection.topic == '/zed_node/left_raw/camera_info':
            self.K_left = np.array(msg.k).reshape(3, 3)
            self.D_left = np.array(msg.d)
            self.R_left = np.array(msg.r).reshape(3, 3)
            self.P_left = np.array(msg.p).reshape(3, 4)
            print("Processed left camera info")
            
        elif connection.topic == '/zed_node/right_raw/camera_info':
            self.K_right = np.array(msg.k).reshape(3, 3)
            self.D_right = np.array(msg.d)
            self.R_right = np.array(msg.r).reshape(3, 3)
            self.P_right = np.array(msg.p).reshape(3, 4)
            print("Processed right camera info")
            
        # Compute rectification maps if both camera infos are available
        if self.K_left is not None and self.K_right is not None and self.map1_left is None:
            self.compute_rectification_maps()
            
            self.zed_map_processed = True

    def compute_rectification_maps(self):
        """Compute rectification maps once for both cameras"""
        # Use the image size from left camera (assuming both are same)
        image_size = (1920,1200) # width, height

        print("image size is:", image_size)
        
        R = np.eye(3)
        t = np.array([0.12, 0, 0], dtype=np.float64)  # Small translation along x-axis read from tf tree
        
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            self.K_left, self.D_left, self.K_right, self.D_right,
            image_size, R, t, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
        )
        
        self.map1_left, self.map2_left = cv2.initUndistortRectifyMap(
            self.K_left, self.D_left, R1, P1, image_size, cv2.CV_32FC1
        )
        
        self.map1_right, self.map2_right = cv2.initUndistortRectifyMap(
            self.K_right, self.D_right, R2, P2, image_size, cv2.CV_32FC1
        )
        
        print("Computed rectification maps")


    def save_radar_image(self, msg, rawdata, typestore):
        msg = deserialize_cdr(rawdata, msg.msgtype, typestore=typestore)
        try:
            img_msg = msg.b_scan_img
            timestamp_row = np.array(msg.timestamps, dtype=np.uint64)
            encoder_values = np.array(msg.encoder_values, dtype=np.uint16)

            radar_data = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(img_msg.height, img_msg.width)
            timestamp = img_msg.header.stamp
            nano_sec = img_msg.header.stamp.nanosec
            stamp = timestamp.sec + nano_sec/1e9
            image_filename = os.path.join(self.radar_image_dir, f'{str(stamp*1000)}.png')

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

    def save_lslidar_bins(self, connection, rawdata, typestore, output_dir):

        msg = typestore.deserialize_cdr(rawdata, connection.msgtype)

        try:
            array_dtype =  np.dtype([
            ('x',         np.float32),
            ('y',         np.float32),
            ('z',         np.float32),
            ('intensity', np.float32),
            ('ring',      np.uint16),
            ('timestamp', np.float32), # float32 for lslidar
            ])
            # 2) Compute number of points
            point_step   = msg.point_step              # bytes per point

            total_bytes  = len(msg.data)

            num_points   = total_bytes // point_step

            arr = np.zeros(num_points, dtype=array_dtype)

            timestamp = msg.header.stamp
            ns_timestamp = timestamp.sec*1e9 + timestamp.nanosec

               # 3) View the buffer as that structured array
            data = np.frombuffer(msg.data, dtype=np.uint8).reshape(-1,msg.point_step)
            # print("sam: the shape of data is:", data.shape)


            for field in msg.fields:
                print(field.datatype, field.name, field.offset, field.count)
                name = field.name
                offset = field.offset
                type = DATA_TYPES[field.datatype] # this is the data type of the field
                num_bytes = np.dtype(type).itemsize

                raw = data[:, offset:offset + num_bytes].ravel()

                col = np.frombuffer(raw, dtype=type)

                if name == 'timestamp':
                    arr[name] = col*1e9 + ns_timestamp  # Convert to nanoseconds
                else:
                    arr[name] = col

             # get rid of invalid points (nan)
            if not msg.is_dense:
                mask = ~np.isnan(arr['x']) & ~np.isnan(arr['y']) & ~np.isnan(arr['z'])
                arr = arr[mask]
            # print("10 data:", arr[0:10])
            # Save to binary file
            bin_filename = os.path.join(output_dir, f'{ns_timestamp/1e3}.bin') # save in microseconds
            arr.tofile(bin_filename)
            print(f'Saved lidar bin: {bin_filename}')

        except Exception as e:
            print(f'Error saving lidar bins: {str(e)}')
    
    


    def save_rslidar_bins(self, connection, rawdata, typestore, output_dir):

        msg = typestore.deserialize_cdr(rawdata, connection.msgtype)

        try:
            array_dtype =  np.dtype([
            ('x',         np.float32),
            ('y',         np.float32),
            ('z',         np.float32),
            ('intensity', np.float32),
            ('ring',      np.uint16),
            ('timestamp', np.float64), # floast64 for rslidar
            ])
            # 2) Compute number of points
            point_step   = msg.point_step              # bytes per point
 
            total_bytes  = len(msg.data)

            num_points   = total_bytes // point_step

            arr = np.zeros(num_points, dtype=array_dtype)

            timestamp = msg.header.stamp
            ns_timestamp = timestamp.sec + timestamp.nanosec/1e9

               # 3) View the buffer as that structured array
            data = np.frombuffer(msg.data, dtype=np.uint8).reshape(-1,msg.point_step)


            for field in msg.fields:
                print(field.datatype, field.name, field.offset, field.count)
                name = field.name
                offset = field.offset
                type = DATA_TYPES[field.datatype] # this is the data type of the field
                num_bytes = np.dtype(type).itemsize

                raw = data[:, offset:offset + num_bytes].ravel()

                col = np.frombuffer(raw, dtype=type)

                if name == 'timestamp':
                    arr[name] = col*1e9 # in nanoseconds
                else:
                    arr[name] = col

            # get rid of invalid points (nans)
            if not msg.is_dense:
                mask = ~np.isnan(arr['x']) & ~np.isnan(arr['y']) & ~np.isnan(arr['z'])
                arr = arr[mask]

            # print("10 data:", arr[0:10])
            export_timestamp = arr[0]['timestamp']/1e3 # microseconds
            
            # Save to binary file
            bin_filename = os.path.join(output_dir, f'{export_timestamp}.bin')
            arr.tofile(bin_filename)
            print(f'Saved lidar bin: {bin_filename}')

        except Exception as e:
            print(f'Error saving lidar bins: {str(e)}')

    def save_imu_data(self, connection, rawdata, typestore, imu_file):
        msg = deserialize_cdr(rawdata, connection.msgtype, typestore=typestore)
        
        try:
            ts = msg.header.stamp
            timestamp = ts.sec + ts.nanosec * 1e-9
            ang_vel = msg.angular_velocity
            lin_acc = msg.linear_acceleration
            
            imu_file.write(
                f"{timestamp},{ang_vel.x},{ang_vel.y},{ang_vel.z},"
                f"{lin_acc.x},{lin_acc.y},{lin_acc.z}\n"
            )
            
        except Exception as e:
            print(f'Error saving IMU data: {str(e)}')

    def save_camera_image(self, connection, rawdata, typestore):
        msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
        
        try:
            ts = msg.header.stamp
            timestamp = ts.sec + ts.nanosec/1e9
            
            # Decode image
            from cv_bridge import CvBridge
            bridge = CvBridge()
            img = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                
            # Rectify image
            if connection.topic == '/zed_node/left_raw/image_raw_color' and self.map1_left is not None:
                img = cv2.remap(img, self.map1_left, self.map2_left, cv2.INTER_LINEAR)
                output_dir = self.zed_node_left_dir
            elif connection.topic == '/zed_node/right_raw/image_raw_color' and self.map1_right is not None:
                img = cv2.remap(img, self.map1_right, self.map2_right, cv2.INTER_LINEAR)
                output_dir = self.zed_node_right_dir
            else:
                print("Rectification maps not available, saving raw image")
                output_dir = self.zed_node_left_dir if 'left' in connection.topic else self.zed_node_right_dir
                
            # Save image
            img_filename = os.path.join(output_dir, f'{timestamp*1000}.png')
            cv2.imwrite(img_filename, img)
            print(f'Saved camera image: {img_filename}')
            
        except Exception as e:
            print(f'Error saving camera image: {str(e)}')

    def save_audio_data(self, connection, rawdata, typestore):
        msg = typestore.deserialize_cdr(rawdata, connection.msgtype)

        audio_stereo = autils.Stereo()

        output_dir = os.path.join(self.output_dir, 'audio')
        os.makedirs(output_dir, exist_ok=True)

        try:
            ts = msg.header.stamp
            timestamp = ts.sec + ts.nanosec/1e9

            if audio_stereo.is_mic_topic(connection.topic):
                print(f"Processing audio data from topic: {connection.topic}")
                audio_stereo.add_message(connection, timestamp, rawdata, typestore)
            

            # audio_stereo.postprocess_audio_data()
            audio_stereo.save_audio(f"{output_dir}/{timestamp}.wav", True)


        except Exception as e:
            print(f'Error saving audio data: {str(e)}')

def main():
    parser = argparse.ArgumentParser(description='Convert ROS2 bag to sensor data files.')
    parser.add_argument('--input', type=str, default='/home/samqiao/ASRL/fomo-public-sdk/raw_fomo_rosbags/deployment4/red/', help='Path to input bag file')
    parser.add_argument('--output', type=str, default='/home/samqiao/ASRL/fomo-public-sdk/output', help='Output directory')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing output directory')
    args = parser.parse_args()

    if os.path.exists(args.output):
        if args.overwrite:
            import shutil
            shutil.rmtree(args.output)
        else:
            raise FileExistsError(f"Output directory {args.output} already exists. Use --overwrite to replace.")
    
    os.makedirs(args.output, exist_ok=True)
    
    converter = BagToDir(args.input, args.output)
    converter.read_bag()
    print("Conversion completed successfully!")

if __name__ == '__main__':
    main()