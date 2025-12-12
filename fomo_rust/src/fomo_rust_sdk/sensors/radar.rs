use super::header::{self};
use super::image::RosImage;
use super::timestamp::TimestampPrecision;
use super::utils::{self, HasHeader, ToRosMsg, ToRosMsgWithInfo};
use byteorder::{LittleEndian, WriteBytesExt};
use camino::{Utf8Path, Utf8PathBuf};
use cdr_encoding::to_vec;
use ndarray::concatenate;
use ndarray::{Array2, Axis};
use serde::{Deserialize, Serialize};
use std::fs::create_dir;
use std::io::{Cursor, Write};

use super::header::Header;
use super::utils::ImageData;

pub const DATA_SCHEMA_DEF: &str = "sensor_msgs/Image b_scan_img\nuint16[] encoder_values\nuint64[] timestamps\n================================================================================\nMSG: sensor_msgs/Image\nstd_msgs/Header header\nuint32 height\nuint32 width\nstring encoding\nuint8 is_bigendian\nuint32 step\nuint8[] data\n================================================================================\nMSG: std_msgs/Header\nbuiltin_interfaces/Time stamp\nstring frame_id\n================================================================================\nMSG: builtin_interfaces/Time\nint32 sec\nuint32 nanosec\n";
pub const INFO_SCHEMA_DEF: &str = "std_msgs/Header header\nuint16 azimuth_samples\nuint16 encoder_size\nuint16 azimuth_offset\nfloat32 bin_size\nuint16 range_in_bins\nuint16 expected_rotation_rate\nfloat32 range_gain\nfloat32 range_offset\n================================================================================\nMSG: std_msgs/Header\nbuiltin_interfaces/Time stamp\nstring frame_id\n================================================================================\nMSG: builtin_interfaces/Time\nint32 sec\nuint32 nanosec\n";
pub const NAVTECH_NAMESPACE: &str = "/navtech";
pub const NAVTECH_FRAME_ID: &str = "navtech";

pub struct FomoRadars<'a> {
    pub topic_name: &'a str,
    pub output_path: Utf8PathBuf,
    pub radar_scans: Vec<RadarScan>,
}

impl FomoRadars<'_> {
    pub fn new<P: AsRef<Utf8Path>>(topic_name: &str, output_path: P) -> FomoRadars {
        if !output_path.as_ref().exists() {
            create_dir(output_path.as_ref()).unwrap();
        }
        FomoRadars {
            topic_name,
            output_path: output_path.as_ref().to_path_buf(),
            radar_scans: Vec::new(),
        }
    }

    pub fn add(&mut self, point_cloud: RadarScan) {
        self.radar_scans.push(point_cloud);
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RadarScan {
    pub b_scan_image: RosImage,
    pub encoder_values: Vec<u16>,
    pub timestamps: Vec<u64>,
}

impl RadarScan {
    pub fn save_msg<P: AsRef<std::path::Path>>(
        &self,
        path: P,
        prec: &TimestampPrecision,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let height = self.b_scan_image.height;
        let width = self.b_scan_image.width;
        // construct the output 2D image array
        let encoder_part = Array2::from_shape_vec(
            (height as usize, 2),
            self.encoder_values
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<u8>>(),
        )
        .expect("Encoder values shape mismatch");

        let prec_conversion = |v: &u64| match prec {
            TimestampPrecision::NanoSecond => *v,
            TimestampPrecision::MicroSecond => v / 1_000,
            TimestampPrecision::MiliSecond => v / 1_000_000,
        };

        let timestamps_part = Array2::from_shape_vec(
            (height as usize, 8),
            self.timestamps
                .iter()
                .map(prec_conversion)
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<u8>>(),
        )
        .expect("Timestamps values shape mismatch");

        let zero_column = Array2::<u8>::zeros((height as usize, 1));

        let data_image = utils::image_from_bytes(
            &self.b_scan_image.image,
            width as usize,
            height as usize,
            &self.b_scan_image.encoding,
        )
        .unwrap();

        if let ImageData::Gray(arr, _, _) = data_image {
            let arr_as_array2 = Array2::from_shape_vec((height as usize, width as usize), arr)
                .expect("Failed to recreate Array2 from Vec<u8>");

            let arrays: Vec<Array2<u8>> =
                vec![timestamps_part, encoder_part, zero_column, arr_as_array2];
            let combined: Array2<u8> = concatenate(
                Axis(1),
                &arrays.iter().map(|f| f.view()).collect::<Vec<_>>(),
            )
            .expect("Combined values shape mismatch");

            let (h, w) = combined.dim();
            let (combined_vec, _) = combined.into_raw_vec_and_offset();

            let image = ImageData::Gray(combined_vec, w as u32, h as u32);
            utils::save_png(&image, path)?;
            return Ok(());
        };

        return Err("Failed to process radar data".into());
    }

    pub fn from_file<P: AsRef<std::path::Path>>(
        path: P,
        prec: &TimestampPrecision,
    ) -> Result<RadarScan, Box<dyn std::error::Error>> {
        let timestamp = path
            .as_ref()
            .file_stem()
            .unwrap()
            .to_str()
            .unwrap()
            .parse::<u64>()?;

        let (stamp_sec, stamp_nsec) = header::get_sec_nsec(timestamp, prec);
        let header = Header {
            stamp_sec,
            stamp_nsec,
            frame_id: NAVTECH_FRAME_ID.to_string(),
        };

        // open the input .png file
        let image_data_obj = utils::load_png(path, false)?;

        match image_data_obj {
            ImageData::Gray(ref image_raw_data, width, height) => {
                let width = width as usize;
                let height = height as usize;

                let mut encoder_values = Vec::<u16>::with_capacity(height);
                for row in 0..height {
                    let idx1 = row * width + 8;
                    let idx2 = row * width + 9;

                    let byte1 = image_raw_data[idx1];
                    let byte2 = image_raw_data[idx2];

                    // Reconstruct the u16 value from little-endian bytes
                    let value = u16::from_le_bytes([byte1, byte2]);
                    encoder_values.push(value);
                }

                let timestamps: Vec<u64> = (0..height)
                    .map(|row| {
                        let base_idx = row * width;
                        let bytes = [
                            image_raw_data[base_idx + 0],
                            image_raw_data[base_idx + 1],
                            image_raw_data[base_idx + 2],
                            image_raw_data[base_idx + 3],
                            image_raw_data[base_idx + 4],
                            image_raw_data[base_idx + 5],
                            image_raw_data[base_idx + 6],
                            image_raw_data[base_idx + 7],
                        ];
                        u64::from_le_bytes(bytes)
                    })
                    .collect();

                let ros_image_data = image_raw_data.clone();
                let b_scan_image = RosImage {
                    header,
                    height: height as u32,
                    width: width as u32,
                    encoding: "8UC1".to_string(),
                    is_bigendian: 0,
                    step: width as u32,
                    image: ros_image_data,
                };
                Ok(RadarScan {
                    b_scan_image,
                    encoder_values,
                    timestamps,
                })
            }
            _ => Err("Failed to process radar data".into()),
        }
    }

    ///
    /// Iterate over all timestamps and change the header timestamp value to the minimum value found
    pub(crate) fn fix_header_time(&mut self) {
        let min_timestamp = *self.timestamps.iter().min().unwrap();
        if self
            .b_scan_image
            .header
            .get_timestamp(&TimestampPrecision::NanoSecond)
            .timestamp
            > min_timestamp
        {
            let (sec, nsec) = header::get_sec_nsec(min_timestamp, &TimestampPrecision::NanoSecond);
            self.b_scan_image.header.stamp_sec = sec;
            self.b_scan_image.header.stamp_nsec = nsec;
        }
    }
}

impl ToRosMsg<Utf8PathBuf> for RadarScan {
    fn get_schema_def() -> &'static [u8] {
        DATA_SCHEMA_DEF.as_bytes()
    }

    fn get_schema_name() -> &'static str {
        "navtech_msgs/msg/RadarBScanMsg"
    }

    fn from_item(
        item: &Utf8PathBuf,
        _frame_id: String,
        prec: &TimestampPrecision,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        RadarScan::from_file(item, prec)
    }

    fn construct_msg(
        radar_scan: RadarScan,
        buffer: &mut Vec<u8>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut cursor = Cursor::new(buffer);

        // Write endian value
        cursor.write_u32::<LittleEndian>(256)?;
        let serialized = to_vec::<RadarScan, LittleEndian>(&radar_scan)?;
        cursor.write(&serialized)?;
        Ok(())
    }
}

impl ToRosMsgWithInfo<Utf8PathBuf> for RadarScan {
    fn get_info_schema_name() -> &'static str {
        "navtech_msgs/msg/RadarConfigurationMsg"
    }

    fn get_info_schema_def() -> &'static [u8] {
        INFO_SCHEMA_DEF.as_bytes()
    }

    fn construct_info_msg<P: AsRef<Utf8Path>>(
        radar_scan: &RadarScan,
        buffer: &mut Vec<u8>,
        _calib_path: P,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut cursor = Cursor::new(buffer);

        // Write endian value
        cursor.write_u32::<LittleEndian>(256)?;
        let radar_info = RadarConfig::default(radar_scan.b_scan_image.header.clone());
        let serialized = to_vec::<RadarConfig, LittleEndian>(&radar_info)?;
        cursor.write_all(&serialized)?;

        Ok(())
    }

    fn get_info_topic() -> &'static str {
        "config_data"
    }

    fn get_data_topic() -> &'static str {
        "b_scan_msg"
    }
}

impl HasHeader for RadarScan {
    fn get_header(&self) -> Header {
        self.b_scan_image.header.clone()
    }
}

#[derive(Debug, Serialize)]
struct RadarConfig {
    header: Header,
    azimuth_samples: u16,
    encoder_size: u16,
    azimuth_offset: u16,
    bin_size: f32,
    range_bins: u16,
    rotation_rate: u16,
    range_gain: f32,
    range_offset: u32,
}

impl RadarConfig {
    fn default(header: Header) -> Self {
        RadarConfig {
            header,
            azimuth_samples: 400,
            encoder_size: 5600,
            azimuth_offset: 0,
            bin_size: 438.0,
            range_bins: 6848,
            rotation_rate: 4000,
            range_gain: 1.0,
            range_offset: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use cdr_encoding::from_bytes;

    use super::*;
    use std::{fs::File, io::Read};

    #[test]
    fn parse_message() -> Result<(), Box<dyn std::error::Error>> {
        let mut file = File::open("tests/radar.bin")?;
        let mut buf = Vec::new();
        file.read_to_end(&mut buf)?;

        let (mut parsed_data, _consumed_byte_count) =
            from_bytes::<RadarScan, LittleEndian>(&buf[4..])?;
        let header = header::Header {
            stamp_sec: 1748443580,
            stamp_nsec: 215335336,
            frame_id: "b_scan_extended".to_string(),
        };
        assert_eq!(parsed_data.b_scan_image.header, header);
        assert_eq!(
            parsed_data
                .b_scan_image
                .header
                .get_timestamp(&TimestampPrecision::NanoSecond),
            header.get_timestamp(&TimestampPrecision::NanoSecond)
        );

        parsed_data.fix_header_time();
        assert_ne!(parsed_data.b_scan_image.header, header);
        assert_ne!(
            parsed_data
                .b_scan_image
                .header
                .get_timestamp(&TimestampPrecision::NanoSecond),
            header.get_timestamp(&TimestampPrecision::NanoSecond)
        );

        assert_eq!(parsed_data.b_scan_image.height, 400);
        assert_eq!(parsed_data.b_scan_image.width, 6848);
        assert_eq!(parsed_data.b_scan_image.encoding, "8UC1");
        assert_eq!(parsed_data.b_scan_image.step, 6848);
        assert_eq!(parsed_data.b_scan_image.image.len(), 400 * 6848);

        assert_eq!(parsed_data.encoder_values.len(), 400);
        assert_eq!(parsed_data.encoder_values.len(), 400);
        Ok(())
    }

    #[test]
    fn construct_message() -> Result<(), Box<dyn std::error::Error>> {
        let mut file = File::open("tests/radar.bin")?;
        let mut read_buf = Vec::new();
        file.read_to_end(&mut read_buf)?;
        let (parsed_data, _consumed_byte_count) =
            from_bytes::<RadarScan, LittleEndian>(&read_buf[4..])?;
        let mut write_buf: Vec<u8> = Vec::new();
        RadarScan::construct_msg(parsed_data, &mut write_buf).unwrap();
        assert_eq!(
            utils::hash_bytes(&read_buf),
            utils::hash_bytes(&write_buf),
            "Radar data mismatch"
        );
        Ok(())
    }
}
