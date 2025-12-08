use serde::{Deserialize, Serialize};
use serde_big_array::BigArray;

#[derive(Debug, Deserialize)]
pub struct RegionOfInterest {
    // Define ROI fields as needed
    pub x_offset: u32,
    pub y_offset: u32,
    pub height: u32,
    pub width: u32,
    pub do_rectify: bool,
}

#[derive(Debug, PartialEq, Serialize, Deserialize, Clone, Copy)]
pub(crate) struct Vector3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}
impl Vector3 {
    pub(crate) fn new() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub(crate) struct Point {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}
impl Point {
    pub(crate) fn new() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }
}

#[derive(Debug, PartialEq, Serialize, Deserialize, Clone, Copy)]
pub(crate) struct Quaternion {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub w: f64,
}
impl Quaternion {
    pub(crate) fn new() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            w: 1.0,
        }
    }
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub struct Pose {
    pub position: Point,
    pub orientation: Quaternion,
}
impl Pose {
    fn new() -> Self {
        Self {
            position: Point::new(),
            orientation: Quaternion::new(),
        }
    }
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub struct PoseWithCovariance {
    pub pose: Pose,
    #[serde(with = "BigArray")]
    pub covariance: [f64; 36],
}
impl PoseWithCovariance {
    pub(crate) fn new() -> Self {
        Self {
            pose: Pose::new(),
            covariance: [0.0; 36],
        }
    }
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub struct Twist {
    pub linear: Vector3,
    pub angular: Vector3,
}
impl Twist {
    fn new() -> Self {
        Self {
            linear: Vector3::new(),
            angular: Vector3::new(),
        }
    }
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub struct TwistWithCovariance {
    pub twist: Twist,
    #[serde(with = "BigArray")]
    pub covariance: [f64; 36],
}
impl TwistWithCovariance {
    pub(crate) fn new() -> Self {
        Self {
            twist: Twist::new(),
            covariance: [0.0; 36],
        }
    }
}
