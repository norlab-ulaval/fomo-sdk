use super::header::{self, Header};
use super::timestamp::TimestampPrecision;
use super::utils::{self, HasHeader, ToRosMsg, ToRosMsgWithInfo};
use crate::fomo_rust_sdk::calib::{mono_calib, CameraCalibration};
use byteorder::{LittleEndian, WriteBytesExt};
use camino::{Utf8Path, Utf8PathBuf};
use cdr_encoding::to_vec;
use serde::{Deserialize, Serialize};
use serde_json;
use std::fs::{self, create_dir};
use std::io::{Cursor, Write};

use super::utils::ImageData;

pub const DATA_SCHEMA_DEF: &str = "std_msgs/Header header\nuint32 height\nuint32 width\nstring encoding\nuint8 is_bigendian\nuint32 step\nuint8[] data\n================================================================================\nMSG: std_msgs/Header\nbuiltin_interfaces/Time stamp\nstring frame_id\n================================================================================\nMSG: builtin_interfaces/Time\nint32 sec\nuint32 nanosec\n";
pub const INFO_SCHEMA_DEF: &str = "std_msgs/Header header\nuint32 height\nuint32 width\nstring distortion_model\nfloat64[] d\nfloat64[9] k\nfloat64[9] r\nfloat64[12] p\nuint32 binning_x\nuint32 binning_y\nsensor_msgs/RegionOfInterest roi\n================================================================================\nMSG: std_msgs/Header\nbuiltin_interfaces/Time stamp\nstring frame_id\n================================================================================\nMSG: builtin_interfaces/Time\nint32 sec\nuint32 nanosec\n================================================================================\nMSG: sensor_msgs/RegionOfInterest\nuint32 x_offset\nuint32 y_offset\nuint32 height\nuint32 width\nbool do_rectify\n";

pub const BASLER_NAMESPACE: &str = "/basler";
pub const BASLER_FRAME_ID: &str = "basler";

pub enum CameraType {
    Basler,
    ZedxLeft,
    ZedxRight,
}

pub struct BaslerImages<'a> {
    pub topic_name: &'a str,
    pub output_path: Utf8PathBuf,
    pub images: Vec<Image>,
}

impl BaslerImages<'_> {
    pub fn new<P: AsRef<Utf8Path>>(topic_name: &str, output_path: P) -> BaslerImages {
        if !output_path.as_ref().exists() {
            create_dir(output_path.as_ref()).unwrap();
        }
        BaslerImages {
            topic_name,
            output_path: output_path.as_ref().to_path_buf(),
            images: Vec::new(),
        }
    }

    pub fn add(&mut self, image: Image) {
        self.images.push(image);
    }

    /// Input /tmp/YYYY-MM-DD-HH-mm/traj-name/basler
    pub fn save<P: AsRef<std::path::Path>>(&self, path: P) -> Result<(), std::io::Error> {
        if !path.as_ref().exists() {
            create_dir(path.as_ref())?;
        }
        for image in &self.images {
            let filename = format!(
                "{}.png",
                image
                    .header
                    .get_timestamp(&TimestampPrecision::MicroSecond)
                    .timestamp
            );
            image.save_msg(path.as_ref().join(filename), true).unwrap();
        }
        Ok(())
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub struct RosImage {
    pub header: header::Header,
    pub height: u32,
    pub width: u32,
    pub encoding: String,
    pub is_bigendian: u8,
    pub step: u32,
    pub image: Vec<u8>,
}
impl RosImage {
    pub(crate) fn from_image(image: Image) -> Result<RosImage, Box<dyn std::error::Error>> {
        let multiplier = match image.image {
            ImageData::BGRA(_) => 4,
            ImageData::RGBFromBayer(_) => 3,
            ImageData::Gray(_) => 1,
        };
        Ok(RosImage {
            header: image.header,
            height: image.height,
            width: image.width,
            encoding: image.encoding,
            is_bigendian: 0,
            step: multiplier * image.step,
            image: utils::bytes_from_image(&image.image)?,
        })
    }
}
#[derive(Debug)]
pub struct Image {
    pub header: header::Header,
    pub height: u32,
    pub width: u32,
    pub encoding: String,
    pub is_bigendian: u8,
    pub step: u32,
    pub image: utils::ImageData,
}

impl Image {
    pub fn save_msg<P: AsRef<std::path::Path>>(
        &self,
        path: P,
        rectify: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if rectify {
            let rectified = mono_calib(&self.image)?;
            utils::save_png(&rectified, path).unwrap();
        } else {
            utils::save_png(&self.image, path).unwrap();
        }
        Ok(())
    }

    pub fn from_file<P: AsRef<std::path::Path>>(
        path: P,
        frame_id: &str,
        prec: &TimestampPrecision,
    ) -> Result<Image, Box<dyn std::error::Error>> {
        let timestamp = path
            .as_ref()
            .file_stem()
            .unwrap()
            .to_str()
            .unwrap()
            .parse::<u64>()?;

        let (stamp_sec, stamp_nsec) = header::get_sec_nsec(timestamp, prec);
        let header = header::Header {
            stamp_sec,
            stamp_nsec,
            frame_id: frame_id.to_string(),
        };

        // open the input .png file
        let image_data = utils::load_png(path, true)?;

        match image_data {
            utils::ImageData::BGRA(ref image_arr) => {
                let (height, width, _) = image_arr.dim();
                let height = height as u32;
                let width = width as u32;

                Ok(Image {
                    header,
                    height,
                    width,
                    encoding: "bgra8".to_string(),
                    is_bigendian: 0,
                    step: width,
                    image: image_data,
                })
            }
            _ => Err("Failed to process image data".into()),
        }
    }

    pub(crate) fn from_ros_image(ros_image: RosImage) -> Result<Image, Box<dyn std::error::Error>> {
        let width = ros_image.width;
        let height = ros_image.height;
        let encoding = ros_image.encoding;
        let image =
            utils::image_from_bytes(&ros_image.image, width as usize, height as usize, &encoding)?;

        Ok(Image {
            header: ros_image.header,
            height,
            width,
            encoding,
            step: ros_image.step,
            is_bigendian: ros_image.is_bigendian,
            image,
        })
    }
}

impl ToRosMsg<Utf8PathBuf> for Image {
    fn get_schema_def() -> &'static [u8] {
        DATA_SCHEMA_DEF.as_bytes()
    }

    fn get_schema_name() -> &'static str {
        "sensor_msgs/msg/Image"
    }

    fn from_item(
        item: &Utf8PathBuf,
        frame_id: String,
        prec: &TimestampPrecision,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Image::from_file(item, &frame_id, prec)
    }

    fn construct_msg(image: Image, buffer: &mut Vec<u8>) -> Result<(), Box<dyn std::error::Error>> {
        let ros_image = RosImage::from_image(image)?;

        let mut cursor = Cursor::new(buffer);

        // Write endian value
        cursor.write_u32::<LittleEndian>(256)?;
        let serialized = to_vec::<RosImage, LittleEndian>(&ros_image)?;
        cursor.write(&serialized)?;
        Ok(())
    }
}

impl ToRosMsgWithInfo<Utf8PathBuf> for Image {
    fn get_info_schema_name() -> &'static str {
        "sensor_msgs/msg/CameraInfo"
    }

    fn get_info_schema_def() -> &'static [u8] {
        INFO_SCHEMA_DEF.as_bytes()
    }

    fn construct_info_msg<P: AsRef<Utf8Path>>(
        image: &Image,
        buffer: &mut Vec<u8>,
        calib_path: P,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut cursor = Cursor::new(buffer);

        // Write endian value
        cursor.write_u32::<LittleEndian>(256)?;

        let camera_file_path = format!("{}/{}.json", calib_path.as_ref(), image.header.frame_id);
        let data = fs::read_to_string(&camera_file_path).map_err(|e| {
            format!(
                "Failed to read calibration file '{}': {}",
                &camera_file_path, e
            )
        })?;
        let camera_calibration: CameraCalibration = serde_json::from_str(&data)?;

        let camera_info = CameraInfo {
            header: image.header.clone(),
            height: image.height,
            width: image.width,
            camera_calibration,
        };
        let serialized = to_vec::<CameraInfo, LittleEndian>(&camera_info).unwrap();
        cursor.write(&serialized)?;
        Ok(())
    }

    fn get_info_topic() -> &'static str {
        "camera_info"
    }

    fn get_data_topic() -> &'static str {
        "image_rect"
    }
}

impl HasHeader for Image {
    fn get_header(&self) -> Header {
        self.header.clone()
    }
}

#[derive(Debug, Serialize)]
struct CameraInfo {
    header: Header,
    height: u32,
    width: u32,
    camera_calibration: CameraCalibration,
}

#[cfg(test)]
mod tests {
    use cdr_encoding::from_bytes;

    use super::*;
    use std::{
        fs::{read, File},
        io::Read,
    };

    fn assert_images_match(
        parsed: &Image,
        ref_path: &str,
        save_path: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        parsed.save_msg(save_path, false)?;
        let ref_bytes = read(ref_path)?;
        let saved_bytes = read(save_path)?;
        assert_eq!(
            utils::hash_bytes(&ref_bytes),
            utils::hash_bytes(&saved_bytes),
            "Image mismatch"
        );
        Ok(())
    }

    fn load_and_parse(path: &str) -> Result<Image, Box<dyn std::error::Error>> {
        let mut buf = Vec::new();
        File::open(path)?.read_to_end(&mut buf)?;
        let (ros_image, _consumed_byte_count) = from_bytes::<RosImage, LittleEndian>(&buf[4..])?;
        Image::from_ros_image(ros_image)
    }

    fn run_test(
        bin_path: &str,
        expected_header: header::Header,
        encoding: &str,
        width: u32,
        height: u32,
        step: u32,
        ref_png_path: &str,
        tmp_png_path: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let parsed = load_and_parse(bin_path)?;
        assert_eq!(parsed.header, expected_header);
        assert_eq!(
            parsed.header.get_timestamp(&TimestampPrecision::NanoSecond),
            expected_header.get_timestamp(&TimestampPrecision::NanoSecond)
        );
        assert_eq!(parsed.width, width);
        assert_eq!(parsed.height, height);
        assert_eq!(parsed.encoding, encoding);
        assert_eq!(parsed.step, step);
        assert_eq!(parsed.image.len(), (height * step) as usize);

        assert_images_match(&parsed, ref_png_path, tmp_png_path)?;

        Ok(())
    }

    #[test]
    fn process_message_zed_left() -> Result<(), Box<dyn std::error::Error>> {
        run_test(
            "tests/image_zed_left.bin",
            header::Header {
                stamp_sec: 1748443579,
                stamp_nsec: 859716000,
                frame_id: "zedx_left_camera_optical_frame".into(),
            },
            "bgra8",
            480,
            300,
            1920,
            "tests/output/zed_left_1748443579859716000.png",
            "/tmp/test_zed_left.png",
        )
    }

    #[test]
    fn process_message_zed_right() -> Result<(), Box<dyn std::error::Error>> {
        run_test(
            "tests/image_zed_right.bin",
            header::Header {
                stamp_sec: 1748443579,
                stamp_nsec: 792976000,
                frame_id: "zedx_right_camera_optical_frame".into(),
            },
            "bgra8",
            480,
            300,
            1920,
            "tests/output/zed_right_1748443579792976000.png",
            "/tmp/test_zed_right.png",
        )
    }

    #[test]
    fn process_message_basler() -> Result<(), Box<dyn std::error::Error>> {
        run_test(
            "tests/image_basler.bin",
            header::Header {
                stamp_sec: 1748443579,
                stamp_nsec: 942855430,
                frame_id: "basler_link".into(),
            },
            "bayer_bggr8",
            480,
            300,
            480,
            "tests/output/basler_1748443579942855430.png",
            "/tmp/test_basler.png",
        )
    }

    #[test]
    fn read_write() -> Result<(), Box<dyn std::error::Error>> {
        let ref_path = "tests/1732203252090223000.png";
        let save_path = "/tmp/test_image.png";
        let image_obj = Image::from_file(ref_path, "zed", &TimestampPrecision::NanoSecond)?;
        image_obj.save_msg(save_path, false)?;

        let ref_bytes = read(ref_path)?;
        let saved_bytes = read(save_path)?;
        assert_eq!(
            utils::hash_bytes(&ref_bytes),
            utils::hash_bytes(&saved_bytes),
            "Image png file mismatch"
        );
        Ok(())
    }
}
