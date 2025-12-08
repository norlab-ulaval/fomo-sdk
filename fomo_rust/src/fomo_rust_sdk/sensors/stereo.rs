use super::image::Image;
use super::timestamp::TimestampPrecision;
use super::utils;
use crate::fomo_rust_sdk::calib;
use camino::{Utf8Path, Utf8PathBuf};
use std::collections::HashMap;
use std::fs::create_dir;

pub const ZEDXLEFT_NAMESPACE: &str = "/zedx/left";
pub const ZEDXLEFT_FRAME_ID: &str = "zedx_left";
pub const ZEDXRIGHT_NAMESPACE: &str = "/zedx/right";
pub const ZEDXRIGHT_FRAME_ID: &str = "zedx_right";

pub enum StereoSide {
    Left,
    Right,
}

pub struct FomoStereoImages<'a> {
    pub left_topic_name: &'a str,
    pub right_topic_name: &'a str,
    pub left_output_path: Utf8PathBuf,
    pub right_output_path: Utf8PathBuf,
    pub images_left: HashMap<u64, Image>,
    pub images_right: HashMap<u64, Image>,
    pub timestamps: Vec<u64>,
}

impl FomoStereoImages<'_> {
    pub fn new<'a, P: AsRef<Utf8Path>>(
        left_topic_name: &'a str,
        right_topic_name: &'a str,
        left_path: P,
        right_path: P,
    ) -> FomoStereoImages<'a> {
        if !left_path.as_ref().exists() {
            create_dir(left_path.as_ref()).unwrap();
        }
        if !right_path.as_ref().exists() {
            create_dir(right_path.as_ref()).unwrap();
        }
        FomoStereoImages {
            left_topic_name,
            right_topic_name,
            left_output_path: left_path.as_ref().to_path_buf(),
            right_output_path: right_path.as_ref().to_path_buf(),
            images_left: HashMap::new(),
            images_right: HashMap::new(),
            timestamps: Vec::new(),
        }
    }

    pub fn add(&mut self, image: Image, side: StereoSide, prec: &TimestampPrecision) {
        let timestamp = image.header.get_timestamp(prec);
        if !self.timestamps.contains(&timestamp.timestamp) {
            self.timestamps.push(timestamp.timestamp);
        }
        match side {
            StereoSide::Left => {
                self.images_left.insert(timestamp.timestamp, image);
            }
            StereoSide::Right => {
                self.images_right.insert(timestamp.timestamp, image);
            }
        }
    }

    /// Input /tmp/YYYY-MM-DD-HH-mm/traj-name/
    pub fn maybe_save_images(&mut self) -> Result<(), std::io::Error> {
        let mut unprocessed_timestamps = Vec::new();
        for timestamp in &self.timestamps {
            if self.images_left.contains_key(timestamp) && self.images_right.contains_key(timestamp)
            {
                let image_left = self.images_left.remove(timestamp);
                let image_right = self.images_right.remove(timestamp);
                if let (Some(left), Some(right)) = (image_left, image_right) {
                    let filename = format!("{}.png", timestamp);
                    // Use calib module to rectify the stereo images
                    if let Ok((rectified_left, rectified_right)) =
                        calib::stereo_calib(&left.image, &right.image)
                    {
                        utils::save_png(&rectified_left, self.left_output_path.join(&filename))
                            .unwrap();
                        utils::save_png(&rectified_right, self.right_output_path.join(&filename))
                            .unwrap();
                    }
                }
            } else {
                unprocessed_timestamps.push(timestamp.clone());
            }
        }
        self.timestamps = unprocessed_timestamps;
        Ok(())
    }
}
