use super::basic::{Quaternion, Vector3};
use super::header;
use super::timestamp::TimestampPrecision;
use byteorder::{LittleEndian, WriteBytesExt};
use cdr_encoding::to_vec;
use serde::{Deserialize, Serialize};
use serde_json;
use std::fs;
use std::io::{Cursor, Write};

pub const SCHEMA_DEF: &str = "geometry_msgs/TransformStamped[] transforms\n================================================================================\nMSG: geometry_msgs/TransformStamped\nstd_msgs/Header header\nstring child_frame_id\ngeometry_msgs/Transform transform\n================================================================================\nMSG: std_msgs/Header\nbuiltin_interfaces/Time stamp\nstring frame_id\n================================================================================\nMSG: builtin_interfaces/Time\nint32 sec\nuint32 nanosec\n================================================================================\nMSG: geometry_msgs/Transform\ngeometry_msgs/Vector3 translation\ngeometry_msgs/Quaternion rotation\n================================================================================\nMSG: geometry_msgs/Vector3\nfloat64 x\nfloat64 y\nfloat64 z\n================================================================================\nMSG: geometry_msgs/Quaternion\nfloat64 x\nfloat64 y\nfloat64 z\nfloat64 w\n";

#[derive(Debug, Serialize, Deserialize)]
pub struct Transform {
    position: Vector3,
    orientation: Quaternion,
}

impl Transform {
    pub fn from_line(line: String) -> Result<Transform, Box<dyn std::error::Error>> {
        let values = line.trim().split(",").collect::<Vec<&str>>();
        let x = values[0].parse::<f64>()?;
        let y = values[1].parse::<f64>()?;
        let z = values[2].parse::<f64>()?;
        let qx = values[3].parse::<f64>()?;
        let qy = values[4].parse::<f64>()?;
        let qz = values[5].parse::<f64>()?;
        let qw = values[6].parse::<f64>()?;
        let position = Vector3 { x, y, z };
        let orientation = Quaternion {
            x: qx,
            y: qy,
            z: qz,
            w: qw,
        };
        Ok(Transform {
            position,
            orientation,
        })
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TransformStamped {
    header: header::Header,
    child_frame_id: String,
    transform: Transform,
}

#[derive(Debug, Deserialize, Clone)]
struct TransformIJRR {
    from: String,
    to: String,
    position: Vector3,
    orientation: Quaternion,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TFMessage {
    transforms: Vec<TransformStamped>,
}

impl TFMessage {
    pub fn new() -> TFMessage {
        TFMessage {
            transforms: Vec::new(),
        }
    }

    pub fn add(&mut self, tf_stamped: TransformStamped) {
        self.transforms.push(tf_stamped);
    }

    pub fn from_file<P: AsRef<std::path::Path>>(
        path: P,
        timestamp: u64,
        prec: &TimestampPrecision,
    ) -> Result<TFMessage, Box<dyn std::error::Error>> {
        if path.as_ref().extension().unwrap() != "json" {
            return Err(format!(
                "The tf path must point to a .json file: {:?}",
                path.as_ref()
            )
            .into());
        }

        let transforms_ijrr: Vec<TransformIJRR> =
            serde_json::from_str(&fs::read_to_string(&path)?)?;

        let (stamp_sec, stamp_nsec) = header::get_sec_nsec(timestamp, prec);

        let mut tf_message = TFMessage::new();
        for tf in &transforms_ijrr {
            let tf_ros = Transform {
                position: tf.position,
                orientation: tf.orientation,
            };
            let header = header::Header {
                stamp_sec,
                stamp_nsec,
                frame_id: tf.from.clone(),
            };
            let tf_stamped = TransformStamped {
                header,
                child_frame_id: tf.to.clone(),
                transform: tf_ros,
            };
            tf_message.add(tf_stamped);
        }

        Ok(tf_message)
    }
}

pub fn construct_msg(
    tf_message: &TFMessage,
    buffer: &mut Vec<u8>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut cursor = Cursor::new(buffer);
    // Write endian value
    cursor.write_u32::<LittleEndian>(256)?;

    let serialized = to_vec::<TFMessage, LittleEndian>(&tf_message)?;
    cursor.write_all(&serialized)?;

    Ok(())
}
