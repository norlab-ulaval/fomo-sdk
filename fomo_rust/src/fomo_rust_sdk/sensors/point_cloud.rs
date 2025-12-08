use super::common::CsvSaveable;
use super::header::{get_sec_nsec, Header};
use super::timestamp::{convert_timestamp, Timestamp, TimestampPrecision};
use super::utils::{self, HasHeader, ToRosMsg};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use camino::{Utf8Path, Utf8PathBuf};
use cdr_encoding::{from_bytes, to_vec};
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fs::create_dir;
use std::fs::File;
use std::io::Cursor;
use std::io::Read;
use std::io::{BufWriter, Write};
use std::{fmt, io};

pub const SCHEMA_DEF: &str = "std_msgs/Header header\nuint32 height\nuint32 width\nsensor_msgs/PointField[] fields\nbool is_bigendian\nuint32 point_step\nuint32 row_step\nuint8[] data\nbool is_dense\n================================================================================\nMSG: std_msgs/Header\nbuiltin_interfaces/Time stamp\nstring frame_id\n================================================================================\nMSG: builtin_interfaces/Time\nint32 sec\nuint32 nanosec\n================================================================================\nMSG: sensor_msgs/PointField\nuint8 INT8=1\nuint8 UINT8=2\nuint8 INT16=3\nuint8 UINT16=4\nuint8 INT32=5\nuint8 UINT32=6\nuint8 FLOAT32=7\nuint8 FLOAT64=8\nstring name\nuint32 offset\nuint8 datatype\nuint32 count\n";

pub const ROBOSENSE_TOPIC: &str = "/robosense/points";
pub const ROBOSENSE_FRAME_ID: &str = "robosense";
pub const LEISHEN_TOPIC: &str = "/leishen/points";
pub const LEISHEN_FRAME_ID: &str = "leishen";

pub struct FomoPointClouds<'a> {
    pub topic_name: &'a str,
    pub output_path: Utf8PathBuf,
    pub point_clouds: Vec<PointCloud>,
}

impl FomoPointClouds<'_> {
    pub fn new<P: AsRef<Utf8Path>>(topic_name: &str, output_path: P) -> FomoPointClouds {
        if !output_path.as_ref().exists() {
            create_dir(output_path.as_ref()).unwrap();
        }
        FomoPointClouds {
            topic_name,
            output_path: output_path.as_ref().to_path_buf(),
            point_clouds: Vec::new(),
        }
    }

    pub fn add(&mut self, point_cloud: PointCloud) {
        self.point_clouds.push(point_cloud);
    }

    /// Input /tmp/YYYY-MM-DD/traj-name/rslidar
    pub fn save<P: AsRef<std::path::Path>>(
        // TODO remove this function
        &self,
        path: P,
        extension: String,
        prec: &TimestampPrecision,
    ) -> Result<(), std::io::Error> {
        if !path.as_ref().exists() {
            create_dir(path.as_ref())?;
        }
        for point_cloud in &self.point_clouds {
            // Create full file name
            let timestamp = point_cloud
                .header
                .get_timestamp(&TimestampPrecision::MicroSecond)
                .timestamp;
            let filename = format!("{}.{}", timestamp, extension);
            point_cloud
                .save(&path.as_ref().join(filename), prec)
                .unwrap();
        }
        Ok(())
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct PointField {
    name: String,
    offset: u32,
    datatype: u32,
    count: u32,
}

impl PointField {
    pub(crate) fn construct_pointfield(
        field: &PointField,
        cursor: &mut Cursor<&mut Vec<u8>>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        utils::construct_string(&field.name, cursor)?;
        cursor.write_u32::<LittleEndian>(field.offset)?;
        cursor.write_u32::<LittleEndian>(field.datatype)?;
        cursor.write_u32::<LittleEndian>(field.count)?;
        Ok(())
    }
}

#[derive(Debug, PartialEq)]
pub struct Point {
    x: f32,
    y: f32,
    z: f32,
    i: f32,
    r: u16,
    t: u64,
}

impl CsvSaveable for Point {
    fn get_csv_headers() -> &'static str {
        "x,y,z,i,r,t"
    }

    fn to_csv_row(&self, prec: &TimestampPrecision) -> String {
        let point_timestamp = convert_timestamp(
            &Timestamp {
                prec: TimestampPrecision::NanoSecond,
                timestamp: self.t,
            },
            prec,
        );
        let point_t = point_timestamp.timestamp;
        format!(
            "{},{},{},{},{},{}",
            self.x, self.y, self.z, self.i, self.r, point_t
        )
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct PointCloudRos {
    pub header: Header,
    height: u32,
    width: u32,
    fields: Vec<PointField>,
    is_bigendian: bool,
    point_step: u32,
    row_step: u32,
    data: Vec<u8>,
    is_dense: bool,
}
impl PointCloudRos {
    fn from_point_cloud(pc: PointCloud) -> Result<PointCloudRos, Box<dyn std::error::Error>> {
        let mut data: Vec<u8> = Vec::new();
        let mut cursor = Cursor::new(&mut data);
        for point in &pc.data {
            cursor.write_f32::<LittleEndian>(point.x)?;
            cursor.write_f32::<LittleEndian>(point.y)?;
            cursor.write_f32::<LittleEndian>(point.z)?;
            cursor.write_f32::<LittleEndian>(point.i)?;
            cursor.write_u16::<LittleEndian>(point.r)?;
            cursor.write_f64::<LittleEndian>(point.t as f64)?;
        }

        Ok(PointCloudRos {
            header: pc.header,
            height: pc.height,
            width: pc.width,
            fields: pc.fields,
            is_bigendian: pc.is_bigendian,
            point_step: pc.point_step,
            row_step: pc.row_step,
            data,
            is_dense: pc.is_dense,
        })
    }
}

#[derive(Debug)]
pub struct PointCloud {
    pub header: Header,
    height: u32,
    width: u32,
    fields: Vec<PointField>,
    is_bigendian: bool,
    point_step: u32,
    row_step: u32,
    pub data: Vec<Point>,
    is_dense: bool,
}

impl PointCloud {
    pub fn new(timestamp: u64, frame_id: String, prec: &TimestampPrecision) -> PointCloud {
        let (stamp_sec, stamp_nsec) = get_sec_nsec(timestamp, prec);
        let header = Header {
            stamp_sec,
            stamp_nsec,
            frame_id,
        };
        PointCloud {
            header,
            height: 0,
            width: 0,
            fields: Vec::new(),
            is_bigendian: false,
            point_step: 0,
            row_step: 0,
            data: Vec::new(),
            is_dense: true,
        }
    }

    pub fn save<P: AsRef<std::path::Path>>(
        &self,
        path: P,
        prec: &TimestampPrecision,
    ) -> Result<(), std::io::Error> {
        match path.as_ref().extension().and_then(|s| s.to_str()) {
            Some("bin") => {
                let file = File::create(path)?;
                let mut writer = BufWriter::new(file);
                for point in &self.data {
                    let point_timestamp = convert_timestamp(
                        &Timestamp {
                            prec: TimestampPrecision::NanoSecond,
                            timestamp: point.t,
                        },
                        prec,
                    );
                    let point_t = point_timestamp.timestamp;
                    writer.write_all(&point.x.to_le_bytes())?;
                    writer.write_all(&point.y.to_le_bytes())?;
                    writer.write_all(&point.z.to_le_bytes())?;
                    writer.write_all(&point.i.to_le_bytes())?;
                    writer.write_all(&point.r.to_le_bytes())?;
                    writer.write_all(&point_t.to_le_bytes())?;
                }
            }
            Some("csv") => {
                let file = File::create(path)?;
                let mut writer = BufWriter::new(file);
                writeln!(writer, "{}", Point::get_csv_headers())?;

                for point in &self.data {
                    writeln!(writer, "{}", point.to_csv_row(prec))?;
                }
            }
            Some(ext) => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("Invalid extension value: {}", ext),
                ))
            }
            None => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("Path is missing an extension: {:?}", path.as_ref()),
                ))
            }
        }
        Ok(())
    }

    pub fn load_points<P: AsRef<std::path::Path>>(path: P) -> Result<Vec<Point>, std::io::Error> {
        let mut file = File::open(path)?;
        let mut input = Vec::new();
        file.read_to_end(&mut input)?;
        let number_of_points = input.len() / (4 + 4 + 4 + 4 + 2 + 8); // TODO replace this by the struct size
        let mut cursor = Cursor::new(input);
        let mut points = Vec::new();

        for _ in 0..number_of_points {
            let x = cursor.read_f32::<LittleEndian>()?;
            let y = cursor.read_f32::<LittleEndian>()?;
            let z = cursor.read_f32::<LittleEndian>()?;
            let i = cursor.read_f32::<LittleEndian>()?;
            let r = cursor.read_u16::<LittleEndian>()?;
            let t = cursor.read_u64::<LittleEndian>()?;
            points.push(Point { x, y, z, t, i, r });
        }

        Ok(points)
    }

    pub fn from_file<P: AsRef<std::path::Path>>(
        path: P,
        frame_id: &str,
        prec: &TimestampPrecision,
    ) -> Result<PointCloud, Box<dyn std::error::Error>> {
        let timestamp = path
            .as_ref()
            .file_stem()
            .unwrap()
            .to_str()
            .unwrap()
            .parse::<u64>()?;

        let (stamp_sec, stamp_nsec) = get_sec_nsec(timestamp, prec);
        let header = Header {
            stamp_sec,
            stamp_nsec,
            frame_id: frame_id.to_string(),
        };

        let mut fields = Vec::with_capacity(6);
        fields.push(PointField {
            name: "x".to_string(),
            offset: 0,
            datatype: 7,
            count: 1,
        });
        fields.push(PointField {
            name: "y".to_string(),
            offset: 4,
            datatype: 7,
            count: 1,
        });
        fields.push(PointField {
            name: "z".to_string(),
            offset: 8,
            datatype: 7,
            count: 1,
        });
        fields.push(PointField {
            name: "intensity".to_string(),
            offset: 12,
            datatype: 7,
            count: 1,
        });
        fields.push(PointField {
            name: "ring".to_string(),
            offset: 16,
            datatype: 4,
            count: 1,
        });
        fields.push(PointField {
            name: "timestamp".to_string(),
            offset: 18,
            datatype: 8,
            count: 1,
        });

        let data = PointCloud::load_points(path)?;
        let width = data.len() as u32;
        let point_step = 26; // TODO make this a constant
        let row_step = point_step * width;
        Ok(PointCloud {
            header,
            height: 1,
            width,
            fields,
            is_bigendian: false,
            point_step,
            row_step,
            is_dense: true,
            data,
        })
    }

    fn from_point_cloud_ros(
        is_lslidar: bool,
        mut pcr: PointCloudRos,
    ) -> Result<PointCloud, Box<dyn std::error::Error>> {
        let mut data: Vec<Point> = Vec::new();
        let mut cursor = Cursor::new(pcr.data.as_slice());
        let timestamp = pcr
            .header
            .get_timestamp(&TimestampPrecision::NanoSecond)
            .timestamp;
        let mut min_timestamp = timestamp;
        for _ in 0..(pcr.data.len() / pcr.point_step as usize) {
            let start = cursor.position();
            match parse_point(&mut cursor, &pcr.fields, is_lslidar, timestamp) {
                Ok(point) => {
                    if point.t < min_timestamp {
                        min_timestamp = point.t;
                    };
                    data.push(point);
                }
                Err(ParsePointError::NanPoint(_)) => (), // Skip points with NaN values
                Err(other_error) => return Err(other_error.into()), // Return other errors
            };
            cursor.set_position(start + pcr.point_step as u64);
        }

        if timestamp - min_timestamp != 0 {
            (pcr.header.stamp_sec, pcr.header.stamp_nsec) =
                get_sec_nsec(min_timestamp, &TimestampPrecision::NanoSecond)
        }

        Ok(PointCloud {
            header: pcr.header,
            height: pcr.height,
            width: pcr.width,
            fields: pcr.fields,
            is_bigendian: pcr.is_bigendian,
            point_step: pcr.point_step,
            row_step: pcr.row_step,
            is_dense: pcr.is_dense,
            data,
        })
    }
}

impl ToRosMsg<Utf8PathBuf> for PointCloud {
    fn get_schema_def() -> &'static [u8] {
        SCHEMA_DEF.as_bytes()
    }

    fn get_schema_name() -> &'static str {
        "sensor_msgs/msg/PointCloud2"
    }

    fn from_item(
        item: &Utf8PathBuf,
        frame_id: String,
        prec: &TimestampPrecision,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        PointCloud::from_file(item, &frame_id, prec)
    }

    fn construct_msg(
        pc: PointCloud,
        buffer: &mut Vec<u8>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut cursor = Cursor::new(buffer);

        // Write endian value
        cursor.write_u32::<LittleEndian>(256)?;

        let pcr = PointCloudRos::from_point_cloud(pc)?;
        let serialized = to_vec::<PointCloudRos, LittleEndian>(&pcr)?;
        cursor.write(&serialized)?;
        Ok(())
    }
}

impl HasHeader for PointCloud {
    fn get_header(&self) -> Header {
        self.header.clone()
    }
}

#[derive(Debug)]
pub enum ParsePointError {
    NanPoint(String),
    IoError(std::io::Error),
    Other(String),
}

impl fmt::Display for ParsePointError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ParsePointError::NanPoint(msg) => write!(f, "NaN point: {}", msg),
            ParsePointError::IoError(err) => write!(f, "IO error: {}", err),
            ParsePointError::Other(msg) => write!(f, "Parse error: {}", msg),
        }
    }
}

impl Error for ParsePointError {}

impl From<std::io::Error> for ParsePointError {
    fn from(err: std::io::Error) -> Self {
        ParsePointError::IoError(err)
    }
}

fn parse_point(
    cursor: &mut Cursor<&[u8]>,
    fields: &Vec<PointField>,
    is_lslidar: bool,
    timestamp: u64, // in nanoseconds
) -> Result<Point, ParsePointError> {
    let mut x = 0.;
    let mut y = 0.;
    let mut z = 0.;
    let mut i = 0.;
    let mut r = 0;
    let mut t = 0;
    let start = cursor.position();
    for field in fields {
        cursor.set_position(start + field.offset as u64);
        match field.name.as_str() {
            "x" => {
                x = cursor.read_f32::<LittleEndian>()?;
                if x.is_nan() {
                    return Err(ParsePointError::NanPoint(format!(
                        "Point contains NaN values: x={}",
                        x
                    )));
                }
            }
            "y" => y = cursor.read_f32::<LittleEndian>()?,
            "z" => z = cursor.read_f32::<LittleEndian>()?,
            "intensity" => i = cursor.read_f32::<LittleEndian>()?,
            "ring" => r = cursor.read_u16::<LittleEndian>()?,
            "timestamp" => {
                let temp = cursor.read_f64::<LittleEndian>()?;
                t = match is_lslidar {
                    true => (temp * 1e9).floor() as u64 + timestamp,

                    false => (temp * 1e9).floor() as u64,
                }
            }
            _ => {}
        }
    }
    Ok(Point { x, y, z, t, i, r })
}

pub fn parse_msg(input: &[u8], is_lslidar: bool) -> Result<PointCloud, Box<dyn std::error::Error>> {
    let mut cursor = Cursor::new(input);

    // Read endian value
    let endian = cursor.read_u32::<LittleEndian>()?;
    utils::print_bytes(endian, "endian");

    let (point_cloud_ros, _consumed_byte_count) =
        from_bytes::<PointCloudRos, LittleEndian>(&input[4..])?;
    PointCloud::from_point_cloud_ros(is_lslidar, point_cloud_ros)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::read;
    use std::fs::File;
    use std::io::Read;

    fn get_first_point_timestamp(cloud: &PointCloud) -> u64 {
        let mut min_timestamp = u64::MAX;
        for point in &cloud.data {
            if point.t < min_timestamp {
                min_timestamp = point.t;
            }
        }
        return min_timestamp;
    }

    fn compare_byte_arrays(read_buf: &Vec<u8>, write_buf: &Vec<u8>) {
        for i in 0..write_buf.len() {
            let constructed_b = write_buf[i];
            let b = read_buf[i];
            assert_eq!(b, constructed_b, "Failed on byte #{}", i);
        }
        assert_eq!(
            utils::hash_bytes(&read_buf),
            utils::hash_bytes(&write_buf),
            "Point cloud data mismatch"
        );
    }

    #[test]
    fn process_field_message() -> Result<(), Box<dyn std::error::Error>> {
        let data: Vec<u8> = vec![
            2, 0, 0, 0, 120, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 121, 0, 0, 0,
            4, 0, 0, 0, 7, 0, 0, 0, 1, 0, 0, 0,
        ];
        let (field, consumed_byte_count) = from_bytes::<PointField, LittleEndian>(&data)?;
        assert_eq!(field.name, "x");
        assert_eq!(field.offset, 0);
        assert_eq!(field.datatype, 7);
        assert_eq!(field.count, 1);

        let (field, _consumed_byte_count) =
            from_bytes::<PointField, LittleEndian>(&data[consumed_byte_count..])?;
        assert_eq!(field.name, "y");
        assert_eq!(field.offset, 4);
        assert_eq!(field.datatype, 7);
        assert_eq!(field.count, 1);
        Ok(())
    }
    #[test]
    fn process_point_nan() -> Result<(), ParsePointError> {
        let data = vec![
            0, 0, 192, 127, 0, 0, 192, 127, 0, 0, 192, 127, 0, 0, 32, 65, 68, 0, 224, 148, 251,
            110, 200, 13, 218, 65,
        ]; // rslidar

        let mut cursor = Cursor::new(data.as_slice());
        let mut fields = Vec::with_capacity(6);
        fields.push(PointField {
            name: "x".to_string(),
            offset: 0,
            datatype: 7,
            count: 1,
        });
        fields.push(PointField {
            name: "y".to_string(),
            offset: 4,
            datatype: 7,
            count: 1,
        });
        fields.push(PointField {
            name: "z".to_string(),
            offset: 8,
            datatype: 7,
            count: 1,
        });
        fields.push(PointField {
            name: "intensity".to_string(),
            offset: 12,
            datatype: 7,
            count: 1,
        });
        fields.push(PointField {
            name: "ring".to_string(),
            offset: 16,
            datatype: 4,
            count: 1,
        });
        fields.push(PointField {
            name: "timestamp".to_string(),
            offset: 18,
            datatype: 8,
            count: 1,
        });
        let result = parse_point(&mut cursor, &fields, false, 1748443580000503296);
        match result {
            Err(ParsePointError::NanPoint(_)) => (), // This is what we expect
            _ => panic!("Expected NanPoint error"),
        }
        Ok(())
    }

    #[test]
    fn process_point_rslidar() -> Result<(), ParsePointError> {
        let data = vec![
            131, 217, 154, 192, 109, 4, 24, 193, 239, 39, 41, 190, 0, 0, 32, 65, 68, 0, 224, 148,
            251, 110, 200, 13, 218, 65,
        ]; // rslidar

        let mut cursor = Cursor::new(data.as_slice());
        let mut fields = Vec::with_capacity(6);
        fields.push(PointField {
            name: "x".to_string(),
            offset: 0,
            datatype: 7,
            count: 1,
        });
        fields.push(PointField {
            name: "y".to_string(),
            offset: 4,
            datatype: 7,
            count: 1,
        });
        fields.push(PointField {
            name: "z".to_string(),
            offset: 8,
            datatype: 7,
            count: 1,
        });
        fields.push(PointField {
            name: "intensity".to_string(),
            offset: 12,
            datatype: 7,
            count: 1,
        });
        fields.push(PointField {
            name: "ring".to_string(),
            offset: 16,
            datatype: 4,
            count: 1,
        });
        fields.push(PointField {
            name: "timestamp".to_string(),
            offset: 18,
            datatype: 8,
            count: 1,
        });
        let point = parse_point(&mut cursor, &fields, false, 1748443580000503296)?;

        assert_eq!(point.x, -4.8390517);
        assert_eq!(point.y, -9.5010805);
        assert_eq!(point.z, -0.1651914);
        assert_eq!(point.i, 10.0);
        assert_eq!(point.r, 68);
        assert_eq!(point.t, 1748443579930961664);
        Ok(())
    }

    #[test]
    fn process_point_lslidar() -> Result<(), ParsePointError> {
        let data = vec![
            119, 109, 85, 191, 17, 1, 177, 64, 130, 94, 18, 191, 0, 0, 0, 0, 0, 128, 65, 64, 2, 0,
            0, 0, 0, 0, 0, 0, 232, 228, 178, 63,
        ]; // lslidar 1748443580168068608.csv
        let mut cursor = Cursor::new(data.as_slice());
        let mut fields = Vec::with_capacity(6);
        fields.push(PointField {
            name: "x".to_string(),
            offset: 0,
            datatype: 7,
            count: 1,
        });
        fields.push(PointField {
            name: "y".to_string(),
            offset: 4,
            datatype: 7,
            count: 1,
        });
        fields.push(PointField {
            name: "z".to_string(),
            offset: 8,
            datatype: 7,
            count: 1,
        });
        fields.push(PointField {
            name: "intensity".to_string(),
            offset: 16,
            datatype: 7,
            count: 1,
        });
        fields.push(PointField {
            name: "ring".to_string(),
            offset: 20,
            datatype: 4,
            count: 1,
        });
        fields.push(PointField {
            name: "timestamp".to_string(),
            offset: 24,
            datatype: 7,
            count: 1,
        });
        let point = parse_point(&mut cursor, &fields, true, 1748443580168068608)?;

        assert_eq!(point.x, -0.83370155);
        assert_eq!(point.y, 5.53138);
        assert_eq!(point.z, -0.5717546);
        assert_eq!(point.i, 3.0234375);
        assert_eq!(point.r, 2);
        assert_eq!(point.t, 1748443580241873940); //FIXME use point with non-zero offset and reference it wrt to the start of the point cloud
        Ok(())
    }

    #[test]
    fn read_write() -> Result<(), Box<dyn std::error::Error>> {
        let ref_path = "tests/output/1732203252200401920.bin";
        let save_path = "/tmp/test_point_cloud.bin";
        let cloud_obj =
            PointCloud::from_file(ref_path, "RS128_link", &TimestampPrecision::NanoSecond)?;
        cloud_obj.save(save_path, &TimestampPrecision::NanoSecond)?;

        let ref_bytes = read(ref_path)?;
        let saved_bytes = read(save_path)?;
        compare_byte_arrays(&ref_bytes, &saved_bytes);
        assert_eq!(
            utils::hash_bytes(&ref_bytes),
            utils::hash_bytes(&saved_bytes),
            "Point cloud file mismatch"
        );
        Ok(())
    }

    #[test]
    fn process_message_rslidar() -> Result<(), Box<dyn std::error::Error>> {
        let mut file = File::open("tests/point_cloud_rs.bin")?;
        let mut read_buf = Vec::new();
        file.read_to_end(&mut read_buf)?;
        let cloud = parse_msg(&read_buf, false)?;
        assert!(
            cloud
                .header
                .get_timestamp(&TimestampPrecision::NanoSecond)
                .timestamp
                <= get_first_point_timestamp(&cloud)
        );

        let test_save_path = "/tmp/test_point_cloud_rs.bin";
        cloud.save(test_save_path, &TimestampPrecision::NanoSecond)?;

        let mut cloud2 = PointCloud::new(
            cloud
                .header
                .get_timestamp(&TimestampPrecision::NanoSecond)
                .timestamp,
            cloud.header.frame_id,
            &TimestampPrecision::NanoSecond,
        );
        cloud2.data = PointCloud::load_points(test_save_path)?;
        assert_eq!(cloud.data.len(), cloud2.data.len());
        for i in 0..cloud.data.len() {
            assert_eq!(cloud.data[i], cloud2.data[i]);
        }

        std::fs::remove_file(test_save_path)?;
        Ok(())
    }
    #[test]
    fn process_message_lslidar() -> Result<(), Box<dyn std::error::Error>> {
        let mut file = File::open("tests/point_cloud_ls.bin")?;
        let mut buf = Vec::new();
        file.read_to_end(&mut buf)?;
        let cloud = parse_msg(&buf, true)?;
        assert!(
            cloud
                .header
                .get_timestamp(&TimestampPrecision::NanoSecond)
                .timestamp
                <= get_first_point_timestamp(&cloud)
        );

        let test_save_path = "/tmp/test_point_cloud_ls.bin";
        cloud.save(test_save_path, &TimestampPrecision::NanoSecond)?;
        let mut cloud2 = PointCloud::new(
            cloud
                .header
                .get_timestamp(&TimestampPrecision::NanoSecond)
                .timestamp,
            cloud.header.frame_id,
            &TimestampPrecision::NanoSecond,
        );
        cloud2.data = PointCloud::load_points(test_save_path)?;

        assert_eq!(cloud.data.len(), cloud2.data.len());
        for i in 0..cloud.data.len() {
            assert_eq!(cloud.data[i], cloud2.data[i]);
        }

        std::fs::remove_file(test_save_path)?;
        Ok(())
    }
}
