import os
import rclpy
from rclpy.node import Node
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from navtech_msgs.msg import RadarBScanMsg
import numpy as np
import cv2
from sensor_msgs.msg import Imu
from sensor_msgs_py import point_cloud2
import argparse


class BagToDir(Node):
    def __init__(self, bag_file, output_dir):
        super().__init__('bag_to_dir')
        self.bag_file = bag_file
        self.radar_image_dir = os.path.join(output_dir, 'radar')
        os.makedirs(self.radar_image_dir, exist_ok=True)

        self.ls_lidar_bin_dir = os.path.join(output_dir, 'ls_lidar') # one for ls_lidar
        os.makedirs(self.ls_lidar_bin_dir, exist_ok=True)

        self.rs_lidar_bin_dir = os.path.join(output_dir, 'rs_lidar') # one for rs_lidar
        os.makedirs(self.rs_lidar_bin_dir, exist_ok=True)

        # I actually have two imus as well 
        self.vn100_imu_file = open(os.path.join(output_dir, 'vn100_imu.csv'), 'w')
        self.vn100_imu_file.write("timestamp,ang_vel_x,ang_vel_y,ang_vel_z,lin_acc_x,lin_acc_y,lin_acc_z\n")

        self.mti30_imu_file = open(os.path.join(output_dir, 'mti30_imu.csv'), 'w')
        self.mti30_imu_file.write("timestamp,ang_vel_x,ang_vel_y,ang_vel_z,lin_acc_x,lin_acc_y,lin_acc_z\n")

        # also two cameras one is the stereo camera ZED X and the other one is the camera basler ace2 + fisheye lens TODO


        # I need to do the audio file which is a lossless wav file TODO

        self.first_msg = True
        self.read_bag()

    def read_bag(self):
        storage_options = StorageOptions(uri=self.bag_file, storage_id='mcap')
        converter_options = ConverterOptions(
            input_serialization_format='cdr',
            output_serialization_format='cdr'
        )
        
        reader = SequentialReader()
        reader.open(storage_options, converter_options)
        
        topic_types = reader.get_all_topics_and_types()
        print(f"Found {len(topic_types)} topics in the bag file.")
        type_map = {topic.name: topic.type for topic in topic_types}
        msg_type_map = {}
        
        for topic_name, topic_type in type_map.items():
           try:
                msg_type_map[topic_name] = get_message(topic_type)
           except (ModuleNotFoundError, AttributeError) as e:
                self.get_logger().warn(f"Skipping topic '{topic_name}' with unknown type '{topic_type}': {e}")
                continue

        while reader.has_next():
            topic_name, data, t = reader.read_next()
            if topic_name in msg_type_map:
                try:
                    msg_type = msg_type_map[topic_name]
                    msg = deserialize_message(data, msg_type)
                    
                    if isinstance(msg, RadarBScanMsg):
                        self.save_radar_image(msg)

                    if isinstance(msg, point_cloud2.PointCloud2): # but there are two lidar topics
                        if topic_name.startswith('/lslidar128/points'):
                            self.save_lidar_bins(msg,self.ls_lidar_bin_dir)
                        elif topic_name.startswith('/rslidar128/points'):
                            self.save_lidar_bins(msg,self.rs_lidar_bin_dir)

                    if isinstance(msg, Imu):
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
                    self.get_logger().error(f'Error processing message: {str(e)}')
        # Close all files after processing            
        self.mti30_imu_file.close()
        self.vn100_imu_file.close()
                

    def save_radar_image(self, msg):
        try:
            # Extract image data from b_scan_img
            img_msg = msg.b_scan_img
            timestamp_row = np.array(msg.timestamps, dtype=np.uint64)
            encoder_values = np.array(msg.encoder_values, dtype=np.uint16)
            
            # Convert image data to numpy array
            radar_data = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(
                img_msg.height, img_msg.width)
            
            # Create filename with timestamp from the header
            timestamp = img_msg.header.stamp
            image_filename = os.path.join(
                self.radar_image_dir, 
                f'{timestamp.sec}{int(timestamp.nanosec/1e3):06d}.png'
            )

            #encode in the first 8 bytes of each row the timestamp and in the second ones the encoder values
            timestamp_bytes = np.frombuffer(timestamp_row.tobytes(), dtype=np.uint8).reshape(-1, 8)
            encoder_bytes = np.frombuffer(encoder_values.tobytes(), dtype=np.uint8).reshape(-1, 2)
            # Shift radar data to the right to make space for timestamp and encoder values
            final_data = np.zeros((radar_data.shape[0], radar_data.shape[1] + 11), dtype=np.uint8)

            final_data[:, :8] = timestamp_bytes
            final_data[:, 8:10] = encoder_bytes
            final_data[:, 11:] = radar_data
            
            # Save image
            cv2.imwrite(image_filename, final_data)
            self.get_logger().info(f'Saved image: {image_filename}')
            
        except Exception as e:
            self.get_logger().error(f'Error saving radar image: {str(e)}')

    def save_lidar_bins(self, msg, lidar_bin_dir):
        try:
            timestamp = msg.header.stamp.sec * 1e9 + msg.header.stamp.nanosec

            cloud_points = list(point_cloud2.read_points(
                msg, field_names=('x', 'y', 'z', 'intensity','ring','timestamp'), skip_nans=True))
            points = np.array(cloud_points, dtype=[
                                                    ('x', np.float32),
                                                    ('y', np.float32),
                                                    ('z', np.float32),
                                                    ('intensity', np.float32),
                                                    ('ring', np.uint16),
                                                    ('timestamp', np.uint64) # changed to be uint64
                                             ])
            timestamp = int(timestamp / 1000)
            points.tofile(lidar_bin_dir + '/{}.bin'.format(timestamp))
            self.get_logger().info(f'Saved bin: {lidar_bin_dir + "/{}.bin".format(timestamp)}')
        except Exception as e:
            self.get_logger().error(f'Error saving lidar bins: {str(e)}')

    
    def save_rectified_camera_imgs(self, msg, camera_dir):
        # camera info topic 
        # /zed_node/right_raw/camera_info | Type: sensor_msgs/msg/CameraInfo | Count: 3662 | Serialization Format: cdr
        # /zed_node/left_raw/camera_info | Type: sensor_msgs/msg/CameraInfo | Count: 3662 | Serialization Format: cdr
        # stereo camera ZED X
        # /zed_node/right_raw/image_raw_color | Type: sensor_msgs/msg/Image | Count: 3662 | Serialization Format: cdr
        # /zed_node/left_raw/image_raw_color | Type: sensor_msgs/msg/Image | Count: 3662 | Serialization Format: cdr
        pass

    # input: distortion parameters are in the camera_info message but only for the z???? (start with the zed x camera)
    # for the basler ace2 camera I need to use the fisheye model, we need to do the calibration ourselves



    # def save_microphone_audio(self, msg, audio_dir):
    #     pass 
    # prefer split in secs
    # split in msgs or in secs???


def main(args=None):
      # Argument parser setup
    parser = argparse.ArgumentParser(description='Convert MCAP ROS2 bag to radar/lidar images and data.')
    parser.add_argument(
        '--input',
        type=str,
        default='/home/samqiao/ASRL/fomo-public-sdk/raw_fomo_rosbags/deployment4/red/red.mcap', # will remove default 
        help='Path to the MCAP or DB3 bag file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='/home/samqiao/ASRL/fomo-public-sdk/output', # will remove default
        help='Directory to store the output files'
    )

    parser.add_argument(
    '--overwrite',
    action='store_true',
    help='Overwrite the output directory if it exists'
)

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)

    # Start ROS node
    rclpy.init()
    
    try:
        MCAP2IJRR = BagToDir(args.input, args.output)
        rclpy.spin(MCAP2IJRR)
    except Exception as e:
        print(f'Error: {str(e)}')
    finally:
        rclpy.shutdown()
    print("Conversion Done!")

if __name__ == '__main__':
    main()