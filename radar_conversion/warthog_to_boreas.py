import os
import rclpy
from rclpy.node import Node
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from navtech_msgs.msg import RadarBScanMsg
import numpy as np
import math
import cv2
from sensor_msgs.msg import NavSatFix, Imu

def gnss_to_cartesian(lat, lon, alt=0):
    """
    Convert geodetic coordinates (latitude, longitude, altitude) 
    to ECEF (Earth-Centered, Earth-Fixed) Cartesian coordinates.
    
    :param lat: Latitude in decimal degrees
    :param lon: Longitude in decimal degrees
    :param alt: Altitude in meters (default 0)
    :return: x, y, z coordinates in meters
    """
    # WGS84 ellipsoid constants
    a = 6378137.0  # semi-major axis in meters
    f = 1 / 298.257223563  # flattening
    e2 = 2 * f - f * f  # square of first eccentricity
    
    # Convert latitude and longitude to radians
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    
    # Calculate prime vertical radius of curvature
    N = a / math.sqrt(1 - e2 * math.sin(lat_rad)**2)
    
    # Calculate ECEF coordinates
    x = (N + alt) * math.cos(lat_rad) * math.cos(lon_rad)
    y = (N + alt) * math.cos(lat_rad) * math.sin(lon_rad)
    z = ((1 - e2) * N + alt) * math.sin(lat_rad)
    
    return x, y, z

class BagToDir(Node):
    def __init__(self, bag_file, output_dir):
        super().__init__('bag_to_dir')
        self.bag_file = bag_file
        self.radar_image_dir = os.path.join(output_dir, 'radar')
        os.makedirs(self.radar_image_dir, exist_ok=True)
        self.gt_file = os.path.join(output_dir, 'gps_cartesian.txt')
        self.imu_file = open(os.path.join(output_dir, 'ouster_imu.csv'), 'w')
        self.outfile = open(self.gt_file, 'w')
        self.outfile.write("timestamp,latitude,longitude,altitude,x,y,z\n")
        self.imu_file.write("timestamp,ang_vel_z,ang_vel_y,ang_vel_x,lin_acc_z,lin_acc_y,lin_acc_x\n")
        self.init_x = 0
        self.init_y = 0
        self.init_z = 0
        self.first_msg = True
        self.read_bag()

    def read_bag(self):
        storage_options = StorageOptions(uri=self.bag_file, storage_id='sqlite3')
        converter_options = ConverterOptions(
            input_serialization_format='cdr',
            output_serialization_format='cdr'
        )
        
        reader = SequentialReader()
        reader.open(storage_options, converter_options)
        
        topic_types = reader.get_all_topics_and_types()
        type_map = {topic.name: topic.type for topic in topic_types}
        msg_type_map = {}
        
        for topic_name, topic_type in type_map.items():
            msg_type_map[topic_name] = get_message(topic_type)

        while reader.has_next():
            topic_name, data, t = reader.read_next()
            if topic_name in msg_type_map:
                try:
                    msg_type = msg_type_map[topic_name]
                    msg = deserialize_message(data, msg_type)
                    
                    if isinstance(msg, RadarBScanMsg):
                        self.save_radar_image(msg)

                    if isinstance(msg, NavSatFix):
                        if msg.status.status >= msg.status.STATUS_NO_FIX:
                            x, y, z = gnss_to_cartesian(
                                msg.latitude, 
                                msg.longitude, 
                                msg.altitude
                            )
                            if self.first_msg:
                                self.init_x = x
                                self.init_y = y
                                self.init_z = z
                                self.first_msg = False
                            x = x - self.init_x
                            y = y - self.init_y
                            z = z - self.init_z
                            self.outfile.write(
                                f"{t},{msg.latitude},{msg.longitude},{msg.altitude},{x},{y},{z}\n"
                            )
                            print(f"{t},{msg.latitude},{msg.longitude},{msg.altitude},{x},{y},{z}")

                    if isinstance(msg, Imu):
                        ts = float(msg.header.stamp.sec) + msg.header.stamp.nanosec * 1e-9
                        ang_vel = msg.angular_velocity
                        lin_acc = msg.linear_acceleration
                        self.imu_file.write(f"{ts},{ang_vel.z},{ang_vel.y},{ang_vel.x},{lin_acc.z},{lin_acc.y},{lin_acc.x}\n")

                        
                except Exception as e:
                    self.get_logger().error(f'Error processing message: {str(e)}')

        self.outfile.close()

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

def main(args=None):
    rclpy.init(args=args)
    
    bag_file = 'mars_t1_0-001.db3'
    output_dir = 'mars_t1_0-001'
    
    try:
        radar_to_image = BagToDir(bag_file, output_dir)
        rclpy.spin(radar_to_image)
    except Exception as e:
        print(f'Error: {str(e)}')
    finally:
        rclpy.shutdown()
    print("Done!")

if __name__ == '__main__':
    main()