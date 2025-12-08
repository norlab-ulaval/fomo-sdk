use super::basic::{Quaternion, Vector3};
use super::common::CsvSaveable;
use super::header::{self};
use super::timestamp::{Timestamp, TimestampPrecision};
use super::utils::{HasHeader, ToRosMsg};
use byteorder::{LittleEndian, WriteBytesExt};
use cdr_encoding::{from_bytes, to_vec};
use serde::{Deserialize, Serialize};
use std::io::{Cursor, Write};

pub const SCHEMA_DEF: &str = "std_msgs/Header header\ngeometry_msgs/Quaternion orientation\nfloat64[9] orientation_covariance\ngeometry_msgs/Vector3 angular_velocity\nfloat64[9] angular_velocity_covariance\ngeometry_msgs/Vector3 linear_acceleration\nfloat64[9] linear_acceleration_covariance\n================================================================================\nMSG: std_msgs/Header\nbuiltin_interfaces/Time stamp\nstring frame_id\n================================================================================\nMSG: builtin_interfaces/Time\nint32 sec\nuint32 nanosec\n================================================================================\nMSG: geometry_msgs/Quaternion\nfloat64 x\nfloat64 y\nfloat64 z\nfloat64 w\n================================================================================\nMSG: geometry_msgs/Vector3\nfloat64 x\nfloat64 y\nfloat64 z\n";

pub const VECTORANV_TOPIC: &str = "/vectornav/data_raw";
pub const VECTORNAV_FRAME_ID: &str = "vectornav";
pub const XSENS_TOPIC: &str = "/xsens/data_raw";
pub const XSENS_FRAME_ID: &str = "xsens";

#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub struct Imu {
    pub header: header::Header,
    pub orientation: Quaternion,
    pub orientation_covariance: [f64; 9],
    pub angular_velocity: Vector3,
    pub angular_velocity_covariance: [f64; 9],
    pub linear_acceleration: Vector3,
    pub linear_acceleration_covariance: [f64; 9],
}

impl Imu {
    pub fn get_tuple(
        &self,
        prec: &TimestampPrecision,
    ) -> (Timestamp, f64, f64, f64, f64, f64, f64) {
        (
            self.header.get_timestamp(prec),
            self.angular_velocity.x,
            self.angular_velocity.y,
            self.angular_velocity.z,
            self.linear_acceleration.x,
            self.linear_acceleration.y,
            self.linear_acceleration.z,
        )
    }

    pub fn from_csv_line(
        line: &String,
        frame_id: String,
        prec: &TimestampPrecision,
    ) -> Result<Imu, Box<dyn std::error::Error>> {
        let values = line.split(",").collect::<Vec<&str>>();
        let timestamp = values[0].parse::<u64>()?;
        let ax = values[1].parse::<f64>()?;
        let ay = values[2].parse::<f64>()?;
        let az = values[3].parse::<f64>()?;
        let lx = values[4].parse::<f64>()?;
        let ly = values[5].parse::<f64>()?;
        let lz = values[6].parse::<f64>()?;

        let (stamp_sec, stamp_nsec) = header::get_sec_nsec(timestamp, prec);

        let header = header::Header {
            stamp_sec,
            stamp_nsec,
            frame_id,
        };
        let orientation = Quaternion {
            x: 0.0,
            y: 0.0,
            z: -0.0,
            w: 1.0,
        };
        let angular_velocity = Vector3 {
            x: ax,
            y: ay,
            z: az,
        };
        let linear_acceleration = Vector3 {
            x: lx,
            y: ly,
            z: lz,
        };

        let imu = Imu {
            header,
            orientation,
            orientation_covariance: [0f64; 9],
            angular_velocity,
            angular_velocity_covariance: [0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01],
            linear_acceleration,
            linear_acceleration_covariance: [0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01],
        };
        Ok(imu)
    }
}

impl ToRosMsg<String> for Imu {
    fn construct_msg(imu: Imu, buffer: &mut Vec<u8>) -> Result<(), Box<dyn std::error::Error>> {
        let mut cursor = Cursor::new(buffer);

        // write endian
        cursor.write_u32::<LittleEndian>(256)?;

        let serialized = to_vec::<Imu, LittleEndian>(&imu).unwrap();
        cursor.write(&serialized)?;
        Ok(())
    }

    fn get_schema_name() -> &'static str {
        "sensor_msgs/msg/Imu"
    }
    fn get_schema_def() -> &'static [u8] {
        SCHEMA_DEF.as_bytes()
    }

    fn from_item(
        item: &String,
        frame_id: String,
        prec: &TimestampPrecision,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Imu::from_csv_line(item, frame_id, prec)
    }
}

impl HasHeader for Imu {
    fn get_header(&self) -> header::Header {
        self.header.clone()
    }
}

impl CsvSaveable for Imu {
    fn get_csv_headers() -> &'static str {
        "t,ax,ay,az,lx,ly,lz"
    }

    fn to_csv_row(&self, prec: &TimestampPrecision) -> String {
        let timestamp = self.header.get_timestamp(prec).timestamp;
        let la = self.linear_acceleration;
        let av = self.angular_velocity;
        format!(
            "{},{},{},{},{},{},{}",
            timestamp, av.x, av.y, av.z, la.x, la.y, la.z
        )
    }
}

pub fn parse_msg(input: &[u8]) -> Result<Imu, Box<dyn std::error::Error>> {
    let (deserialized_message, _consumed_byte_count) =
        from_bytes::<Imu, LittleEndian>(&input[4..])?; // first 4 bytes are endian, always Little in our case
    Ok(deserialized_message)
}

#[cfg(test)]
mod tests {
    use super::super::utils;

    use super::*;

    fn compare_byte_arrays(read_buf: &Vec<u8>, write_buf: &Vec<u8>) {
        for i in 0..write_buf.len() {
            let constructed_b = write_buf[i];
            let b = read_buf[i];
            assert_eq!(
                b, constructed_b,
                "Failed on byte #{}: {:02X} x {:02X}",
                i, b, constructed_b
            );
        }
        assert_eq!(
            utils::hash_bytes(&read_buf),
            utils::hash_bytes(&write_buf),
            "Point cloud data mismatch"
        );
    }

    fn check_imu_message(data: &[u8], expected_imu: &Imu) {
        let parsed_data = parse_msg(data).unwrap();

        assert_eq!(parsed_data.header, expected_imu.header);
        assert_eq!(
            parsed_data
                .header
                .get_timestamp(&TimestampPrecision::NanoSecond),
            expected_imu
                .header
                .get_timestamp(&TimestampPrecision::NanoSecond)
        );
        assert_eq!(parsed_data.orientation, expected_imu.orientation);
        assert_eq!(parsed_data.angular_velocity, expected_imu.angular_velocity);
        assert_eq!(
            parsed_data.linear_acceleration,
            expected_imu.linear_acceleration
        );

        let mut buffer: Vec<u8> = Vec::new();
        Imu::construct_msg(parsed_data, &mut buffer).unwrap();
        compare_byte_arrays(&Vec::from(data), &buffer);
    }

    #[test]
    fn process_vn100_message() {
        let data = [
            0, 1, 0, 0, 187, 33, 55, 104, 24, 133, 221, 57, 11, 0, 0, 0, 118, 110, 49, 48, 48, 95,
            108, 105, 110, 107, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 128, 0, 0, 0, 0, 0, 0, 240, 63, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 32, 226, 105, 84, 63, 0, 0, 0, 0, 172, 43, 116, 63, 0, 0, 0, 224, 130, 13, 143,
            191, 123, 20, 174, 71, 225, 122, 132, 63, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 123, 20, 174, 71, 225, 122, 132, 63, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 123, 20, 174, 71, 225, 122, 132, 63, 0,
            0, 0, 64, 118, 141, 210, 63, 0, 0, 0, 32, 51, 71, 210, 191, 0, 0, 0, 160, 216, 100, 35,
            64, 123, 20, 174, 71, 225, 122, 132, 63, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 123, 20, 174, 71, 225, 122, 132, 63, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 123, 20, 174, 71, 225, 122, 132, 63,
        ];

        let header = header::Header {
            stamp_sec: 1748443579,
            stamp_nsec: 970818840,
            frame_id: "vn100_link".to_string(),
        };
        let orientation = Quaternion {
            x: 0.0,
            y: 0.0,
            z: -0.0,
            w: 1.0,
        };
        let angular_velocity = Vector3 {
            x: 0.001245947671122849,
            y: 0.004924461245536804,
            z: -0.015162489376962185,
        };
        let linear_acceleration = Vector3 {
            x: 0.28988415002822876,
            y: -0.28559568524360657,
            z: 9.696965217590332,
        };

        let imu = Imu {
            header,
            orientation,
            orientation_covariance: [0f64; 9],
            angular_velocity,
            angular_velocity_covariance: [0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01],
            linear_acceleration,
            linear_acceleration_covariance: [0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01],
        };

        check_imu_message(&data, &imu);
    }

    #[test]
    fn process_mti30_message() {
        let data = [
            0, 1, 0, 0, 187, 33, 55, 104, 86, 50, 29, 58, 11, 0, 0, 0, 77, 84, 105, 51, 48, 95,
            108, 105, 110, 107, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 240, 63, 0, 0, 0, 0, 0, 0, 240, 191, 0, 0, 0, 0, 0, 0,
            240, 191, 0, 0, 0, 0, 0, 0, 240, 191, 0, 0, 0, 0, 0, 0, 240, 191, 0, 0, 0, 0, 0, 0,
            240, 191, 0, 0, 0, 0, 0, 0, 240, 191, 0, 0, 0, 0, 0, 0, 240, 191, 0, 0, 0, 0, 0, 0,
            240, 191, 0, 0, 0, 0, 0, 0, 240, 191, 0, 0, 0, 192, 157, 29, 98, 191, 0, 0, 0, 224, 90,
            38, 128, 191, 0, 0, 0, 0, 155, 24, 113, 191, 45, 67, 28, 235, 226, 54, 58, 63, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 45, 67, 28, 235, 226,
            54, 58, 63, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 45,
            67, 28, 235, 226, 54, 58, 63, 0, 0, 0, 64, 192, 66, 210, 191, 0, 0, 0, 224, 31, 144,
            211, 191, 0, 0, 0, 128, 49, 140, 35, 64, 45, 67, 28, 235, 226, 54, 58, 63, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 45, 67, 28, 235, 226, 54,
            58, 63, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 45, 67,
            28, 235, 226, 54, 58, 63,
        ];

        let header = header::Header {
            stamp_sec: 1748443579,
            stamp_nsec: 974991958,
            frame_id: "MTi30_link".to_string(),
        };
        let orientation = Quaternion {
            x: 0.0,
            y: 0.0,
            z: -0.0,
            w: 1.0,
        };
        let angular_velocity = Vector3 {
            x: -0.002211387734860182,
            y: -0.00788565631955862,
            z: -0.004173856228590012,
        };
        let linear_acceleration = Vector3 {
            x: -0.2853241562843323,
            y: -0.30567166209220886,
            z: 9.773815155029297,
        };
        let imu = Imu {
            header,
            orientation,
            orientation_covariance: [0f64; 9],
            angular_velocity,
            angular_velocity_covariance: [0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01],
            linear_acceleration,
            linear_acceleration_covariance: [0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01],
        };

        check_imu_message(&data, &imu);
    }
}
