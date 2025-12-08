use clap::ValueEnum;
use num_quaternion;
use std::{
    cmp::Ordering,
    io::{Cursor, Write},
};

use byteorder::{LittleEndian, WriteBytesExt};
use cdr_encoding::to_vec;
use serde::{Deserialize, Serialize};

use super::{
    basic::{Point, Pose, PoseWithCovariance, Quaternion, Twist, TwistWithCovariance, Vector3},
    common::CsvSaveable,
    header::{self, Header},
    timestamp::TimestampPrecision,
    utils::{HasHeader, ToRosMsg},
};

pub const SCHEMA_DEF: &str = "std_msgs/Header header\nstring child_frame_id\ngeometry_msgs/PoseWithCovariance pose\ngeometry_msgs/TwistWithCovariance twist\n================================================================================\nMSG: std_msgs/Header\nbuiltin_interfaces/Time stamp\nstring frame_id\n================================================================================\nMSG: builtin_interfaces/Time\nint32 sec\nuint32 nanosec\n================================================================================\nMSG: geometry_msgs/PoseWithCovariance\ngeometry_msgs/Pose pose\nfloat64[36] covariance\n================================================================================\nMSG: geometry_msgs/Pose\ngeometry_msgs/Point position\ngeometry_msgs/Quaternion orientation\n================================================================================\nMSG: geometry_msgs/Point\nfloat64 x\nfloat64 y\nfloat64 z\n================================================================================\nMSG: geometry_msgs/Quaternion\nfloat64 x\nfloat64 y\nfloat64 z\nfloat64 w\n================================================================================\nMSG: geometry_msgs/TwistWithCovariance\ngeometry_msgs/Twist twist\nfloat64[36] covariance\n================================================================================\nMSG: geometry_msgs/Twist\ngeometry_msgs/Vector3 linear\ngeometry_msgs/Vector3 angular\n================================================================================\nMSG: geometry_msgs/Vector3\nfloat64 x\nfloat64 y\nfloat64 z\n";

pub const ODOM_TOPIC: &str = "/warthog/platform/odom";
pub const ODOM_FRAME_ID: &str = "odom";
pub const ODOM_CHILD_FRAME_ID: &str = "base_link";
pub const DIFF_DRIVE_FREQUENCY: f64 = 10.0;

pub struct DiffDrive {
    wheel_base: f64,
    wheel_radius: f64,
    velocity_covariance: f64,
    update_frequence: f64,
    x: f64,
    y: f64,
    theta: f64,
    vels: Vec<MotorVelocity>,
}

impl DiffDrive {
    pub(crate) fn new(drivetrain: &Drivetrain, update_frequence: f64) -> DiffDrive {
        let velocity_covariance = 0.001;
        let wheel_base = 1.5;
        let wheel_radius = match drivetrain {
            Drivetrain::Wheels => 0.3,
            Drivetrain::Tracks => 0.18,
        };
        DiffDrive {
            wheel_base,
            wheel_radius,
            velocity_covariance,
            update_frequence,
            x: 0.0,
            y: 0.0,
            theta: 0.0,
            vels: vec![],
        }
    }

    pub(crate) fn process_velocities(&mut self) -> Result<Vec<Odom>, Box<dyn std::error::Error>> {
        if self.vels.is_empty() {
            return Err(format!("Vector of velocities is empty!").into());
        }
        self.vels.sort_by_key(|mv| mv.timestamp);

        let period = 1.0 / self.update_frequence; // in secs

        let mut left_vel_m = 0.0;
        let mut right_vel_m = 0.0;
        let mut time_last = self.vels[0].timestamp as f64 / 1e9;

        let mut odoms: Vec<Odom> = Vec::new();

        for vel in self.vels.clone() {
            let time = vel.timestamp as f64 / 1e9;
            if time - time_last == 0.0 {
                continue;
            }
            let mut dt = (time - time_last) as f64;
            if dt > period {
                // we need to add more messages using the last computed velocity.
                let num_iters = (dt / period).floor();
                for i in 0..num_iters as u64 {
                    odoms.push(self.compute_update(
                        time_last + ((i + 1) as f64) * period,
                        right_vel_m,
                        left_vel_m,
                        period,
                    ));
                }
                dt = dt - num_iters * period;
            }
            match vel.side {
                WheelSide::Left => left_vel_m = vel.velocity * self.wheel_radius,
                WheelSide::Right => right_vel_m = vel.velocity * self.wheel_radius,
            };

            odoms.push(self.compute_update(time, right_vel_m, left_vel_m, dt));
            time_last = time;
        }
        Ok(odoms)
    }

    fn compute_update(&mut self, time: f64, right_vel_m: f64, left_vel_m: f64, dt: f64) -> Odom {
        let v = (right_vel_m + left_vel_m) / 2.0;
        let omega = (right_vel_m - left_vel_m) / self.wheel_base;

        self.x += v * dt * self.theta.cos();
        self.y += v * dt * self.theta.sin();
        self.theta += omega * dt;

        let (stamp_sec, stamp_nsec) =
            header::get_sec_nsec((time * 1e9) as u64, &TimestampPrecision::NanoSecond);
        let header = Header {
            stamp_sec,
            stamp_nsec,
            frame_id: ODOM_FRAME_ID.to_string(),
        };
        let mut pose = PoseWithCovariance::new();
        let mut twist = TwistWithCovariance::new();

        let uq = num_quaternion::UnitQuaternion::from_euler_angles(0.0, 0.0, self.theta);
        let uq = uq.into_quaternion();

        pose.pose.position.x = self.x;
        pose.pose.position.y = self.y;
        pose.pose.position.z = 0.0;
        pose.pose.orientation.x = uq.x;
        pose.pose.orientation.y = uq.y;
        pose.pose.orientation.z = uq.z;
        pose.pose.orientation.w = uq.w;

        twist.twist.linear.x = v;
        twist.twist.angular.z = omega;

        // Set covariance
        twist.covariance[0] = self.velocity_covariance;
        twist.covariance[7] = self.velocity_covariance;
        twist.covariance[14] = self.velocity_covariance;
        twist.covariance[21] = self.velocity_covariance;
        twist.covariance[28] = self.velocity_covariance;
        twist.covariance[35] = self.velocity_covariance;

        Odom {
            header,
            child_frame_id: ODOM_CHILD_FRAME_ID.to_string(),
            pose,
            twist,
        }
    }

    pub(crate) fn add_vel(&mut self, vel: MotorVelocity) {
        self.vels.push(vel);
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum Drivetrain {
    Wheels,
    Tracks,
}

impl Drivetrain {
    const fn as_str(&self) -> &'static str {
        match self {
            Self::Wheels => "wheels",
            Self::Tracks => "tracks",
        }
    }
}

impl ValueEnum for Drivetrain {
    fn value_variants<'a>() -> &'a [Self] {
        &[Self::Wheels, Self::Tracks]
    }
    fn to_possible_value(&self) -> Option<clap::builder::PossibleValue> {
        Some(clap::builder::PossibleValue::new(self.as_str()))
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub(crate) enum WheelSide {
    Left,
    Right,
}

#[derive(Debug, PartialEq, Clone)]
pub(crate) struct MotorVelocity {
    pub(crate) velocity: f64,  // radians per second
    pub(crate) timestamp: u64, // in nanoseconds
    pub(crate) side: WheelSide,
}

impl PartialOrd for MotorVelocity {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.timestamp.partial_cmp(&other.timestamp)
    }
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub struct Odom {
    pub header: Header,
    pub child_frame_id: String,
    pub pose: PoseWithCovariance,
    pub twist: TwistWithCovariance,
}

impl Odom {
    pub fn from_csv_line(
        line: &String,
        prec: &TimestampPrecision,
    ) -> Result<Odom, Box<dyn std::error::Error>> {
        let values = line.split(",").collect::<Vec<&str>>();
        let timestamp = values[0].parse::<u64>()?;
        let point_x = values[1].parse::<f64>()?;
        let point_y = values[2].parse::<f64>()?;
        let point_z = values[3].parse::<f64>()?;
        let orientation_x = values[4].parse::<f64>()?;
        let orientation_y = values[5].parse::<f64>()?;
        let orientation_z = values[6].parse::<f64>()?;
        let orientation_w = values[7].parse::<f64>()?;
        let vx = values[8].parse::<f64>()?;
        let vy = values[9].parse::<f64>()?;
        let vz = values[10].parse::<f64>()?;
        let ax = values[11].parse::<f64>()?;
        let ay = values[12].parse::<f64>()?;
        let az = values[13].parse::<f64>()?;

        let pose = PoseWithCovariance {
            pose: Pose {
                position: Point {
                    x: point_x,
                    y: point_y,
                    z: point_z,
                },
                orientation: Quaternion {
                    x: orientation_x,
                    y: orientation_y,
                    z: orientation_z,
                    w: orientation_w,
                },
            },
            covariance: [
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0,
            ],
        };

        let twist = TwistWithCovariance {
            twist: Twist {
                linear: Vector3 {
                    x: vx,
                    y: vy,
                    z: vz,
                },
                angular: Vector3 {
                    x: ax,
                    y: ay,
                    z: az,
                },
            },
            covariance: [
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0,
            ],
        };

        let (stamp_sec, stamp_nsec) = header::get_sec_nsec(timestamp, prec);

        let header = header::Header {
            stamp_sec,
            stamp_nsec,
            frame_id: ODOM_FRAME_ID.to_string(),
        };
        let odom = Odom {
            header,
            child_frame_id: ODOM_CHILD_FRAME_ID.to_string(),
            pose,
            twist,
        };
        Ok(odom)
    }
}

impl ToRosMsg<String> for Odom {
    fn get_schema_def() -> &'static [u8] {
        SCHEMA_DEF.as_bytes()
    }

    fn get_schema_name() -> &'static str {
        "nav_msgs/msg/Odometry"
    }

    fn from_item(
        item: &String,
        frame_id: String,
        prec: &TimestampPrecision,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Odom::from_csv_line(item, prec)
    }

    fn construct_msg(odom: Odom, buffer: &mut Vec<u8>) -> Result<(), Box<dyn std::error::Error>> {
        let mut cursor = Cursor::new(buffer);
        cursor.write_u32::<LittleEndian>(256)?;

        let serialized = to_vec::<Odom, LittleEndian>(&odom).unwrap();
        cursor.write(&serialized)?;

        Ok(())
    }
}

impl HasHeader for Odom {
    fn get_header(&self) -> Header {
        self.header.clone()
    }
}

impl CsvSaveable for Odom {
    fn get_csv_headers() -> &'static str {
        "t,px,py,pz,qx,qy,qz,qw,tlx,tly,tlz,tax,tay,taz,tax,tay,taz"
    }

    fn to_csv_row(&self, prec: &TimestampPrecision) -> String {
        let timestamp = self.header.get_timestamp(prec).timestamp;
        let point = &self.pose.pose.position;
        let orie = &self.pose.pose.orientation;
        let twist_ang = &self.twist.twist.angular;
        let twist_lin = &self.twist.twist.linear;
        format!(
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            timestamp,
            point.x,
            point.y,
            point.z,
            orie.x,
            orie.y,
            orie.z,
            orie.w,
            twist_lin.x,
            twist_lin.y,
            twist_lin.z,
            twist_ang.x,
            twist_ang.y,
            twist_ang.z,
        )
    }
}
