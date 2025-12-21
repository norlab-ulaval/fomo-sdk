use anyhow::Result;
use bayer::run_demosaic;
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use camino::Utf8Path;
use image::{GrayImage, Luma, Rgb, RgbImage, Rgba, RgbaImage};
use mcap::Schema;
use ndarray::{s, Array2, Array3};

use num_traits::ToBytes;
use std::io::{Cursor, Read, Write};
use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
};

use super::header::Header;
use super::timestamp::TimestampPrecision;

pub(crate) trait HasHeader {
    fn get_header(&self) -> Header;
}

pub(crate) trait ToRosMsg<Item>: HasHeader {
    fn get_schema_name() -> &'static str;
    fn get_schema_def() -> &'static [u8];
    fn from_item(
        item: &Item,
        frame_id: String,
        prec: &TimestampPrecision,
    ) -> Result<Self, Box<dyn std::error::Error>>
    where
        Self: Sized;
    fn construct_msg(data: Self, buffer: &mut Vec<u8>) -> Result<(), Box<dyn std::error::Error>>;
}

pub(crate) trait ToRosMsgWithInfo<Item>: ToRosMsg<Item> {
    fn get_info_schema_name() -> &'static str;
    fn get_info_schema_def() -> &'static [u8];

    fn get_info_topic() -> &'static str;
    fn get_data_topic() -> &'static str;

    fn construct_info_msg<P: AsRef<Utf8Path>>(
        data: &Self,
        buffer: &mut Vec<u8>,
        calib_path: P,
    ) -> Result<(), Box<dyn std::error::Error>>;
}
pub fn print_schema(schema: &Schema) {
    println!("{:?}", std::str::from_utf8(schema.data.as_ref()));
    // for line in std::str::from_utf8(schema.data.as_ref())
    //     .unwrap()
    //     .split('\n')
    // {
    //     println!("{:?}", line);
    // }
}

pub fn print_message(message: &mcap::Message, print_bytes: bool) {
    if print_bytes {
        println!(
            "{:?}\n{:?}",
            message.channel.topic,
            message
                .data
                .iter()
                .map(|b| format!("{:02X}", b))
                .collect::<Vec<_>>()
                .join(" ")
        );
    } else {
        println!("{:?}\n{:?}", message.channel.topic, message.data);
    }
}

pub fn print_bytes<T: ToBytes + std::fmt::Display>(value: T, name: &str) {
    let bytes = value.to_le_bytes();
    let byte_slice = bytes.as_ref();
    // print!("{}: {} | ", name, value);
    // for b in byte_slice {
    //     print!("{:02X} ", b);
    // }
    // println!();
}

pub fn read_string(
    cursor: &mut Cursor<&[u8]>,
    result: &mut String,
    name: &str,
) -> Result<u32, std::io::Error> {
    let len = cursor.read_u32::<LittleEndian>()?;
    let name_len = String::from(name);
    print_bytes(len, &(name_len + " len"));

    let mut buf = vec![0u8; len as usize];
    cursor.read_exact(&mut buf)?;

    *result = String::from_utf8_lossy(&buf)
        .trim_end_matches('\0')
        .to_string();
    // print!("{}: {} | ", name, result);
    // for b in buf {
    //     print!("{:02X} ", b);
    // }
    // println!();
    // align to 4 bytes
    let padding = (4 - (len % 4) % 4) % 4;
    // print!(
    //     "Skipping {} padding bytes. Position before: {}",
    //     padding,
    //     cursor.position()
    // );
    cursor.set_position(cursor.position() + padding as u64);
    // println!(" | Position after {}", cursor.position());
    Ok(padding)
}

pub fn construct_string(
    text: &str,
    cursor: &mut Cursor<&mut Vec<u8>>,
) -> Result<(), std::io::Error> {
    let len = text.len() + 1;
    cursor.write_u32::<LittleEndian>(len as u32)?;
    let buf = text.as_bytes();
    cursor.write_all(&buf)?;
    cursor.write_u8(0)?;

    let padding = (4 - (len % 4) % 4) % 4;
    let padding_bytes = vec![0u8; padding];
    cursor.write_all(&padding_bytes)?;
    Ok(())
}

#[derive(Debug, PartialEq)]
pub enum ImageData {
    Gray(Vec<u8>, u32, u32),
    RGBA(Vec<u8>, u32, u32),
    BGRA(Vec<u8>, u32, u32),
    RGBFromBayer(Vec<u8>, u32, u32),
}

impl ImageData {
    pub fn len(&self) -> usize {
        match self {
            ImageData::Gray(data, _, _) => data.len(),
            ImageData::RGBA(data, _, _) => data.len(),
            ImageData::BGRA(data, _, _) => data.len(),
            ImageData::RGBFromBayer(data, _, _) => data.len(),
        }
    }
}

pub fn image_from_bytes(
    data: &[u8],
    width: usize,
    height: usize,
    encoding: &str,
) -> Result<ImageData, Box<dyn std::error::Error>> {
    match encoding {
        "8UC1" | "mono8" => {
            if data.len() != width * height {
                return Err("Invalid data size".into());
            }
            Ok(ImageData::Gray(data.to_vec(), width as u32, height as u32))
        }
        "bayer_bggr8" => {
            if data.len() != width * height {
                return Err("Invalid data size".into());
            }

            let mut buf = vec![0u8; width * height * 3];
            let mut dst =
                bayer::RasterMut::new(width, height, bayer::RasterDepth::Depth8, &mut buf);

            let mut reader = Cursor::new(data);
            run_demosaic(
                &mut reader,
                bayer::BayerDepth::Depth8,
                bayer::CFA::BGGR,
                bayer::Demosaic::Linear,
                &mut dst,
            )?;

            Ok(ImageData::RGBFromBayer(buf, width as u32, height as u32))
        }
        "rgba8" => {
            if data.len() != width * height * 4 {
                return Err("Invalid data size".into());
            }
            Ok(ImageData::RGBA(data.to_vec(), width as u32, height as u32))
        }
        "bgra8" => {
            if data.len() != width * height * 4 {
                return Err("Invalid data size".into());
            }
            Ok(ImageData::BGRA(data.to_vec(), width as u32, height as u32))
        }
        encoding => Err(format!("Unknown encoding {}", encoding).into()),
    }
}

pub fn bytes_from_image(image_data: &ImageData) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    match image_data {
        ImageData::Gray(data, _, _) => Ok(data.clone()),
        ImageData::RGBA(data, _, _) => Ok(data.clone()),
        ImageData::BGRA(data, _, _) => Ok(data.clone()),
        ImageData::RGBFromBayer(data, _, _) => Ok(data.clone()),
    }
}

pub fn save_png<P: AsRef<std::path::Path>>(
    img: &ImageData,
    path: P,
) -> Result<(), image::ImageError> {
    match img {
        ImageData::Gray(data, w, h) => {
            let gray = GrayImage::from_raw(*w, *h, data.clone()).unwrap();
            gray.save(path)
        }
        ImageData::RGBA(data, w, h) => {
            let out = RgbaImage::from_raw(*w, *h, data.clone()).unwrap();
            out.save(path)
        }
        ImageData::BGRA(data, w, h) => {
            let mut rgba_buffer = vec![0u8; (w * h * 4) as usize];
            for (src, dst) in data.chunks_exact(4).zip(rgba_buffer.chunks_exact_mut(4)) {
                let [b, g, r, a] = src.try_into().unwrap();
                dst.copy_from_slice(&[r, g, b, a]);
            }
            let out = RgbaImage::from_raw(*w, *h, rgba_buffer.clone()).unwrap();
            // for y in 0..*h {
            //     for x in 0..*w {
            //         let px = data.slice(s![y, x, ..]);
            //         out.put_pixel(x as u32, y as u32, Rgba([px[2], px[1], px[0], px[3]]));
            //     }
            // }
            out.save(path)
        }
        ImageData::RGBFromBayer(data, w, h) => {
            let out = RgbImage::from_raw(*w, *h, data.clone()).unwrap();
            out.save(path)
        }
    }
}

pub fn load_png<P: AsRef<std::path::Path>>(
    path: P,
    is_color: bool,
) -> Result<ImageData, image::ImageError> {
    let file = std::fs::File::open(path)?;
    let reader = std::io::BufReader::new(file);
    let img = image::load(reader, image::ImageFormat::Png)?;

    match is_color {
        true => {
            let rgba_img = img.to_rgba8();
            let (w, h) = rgba_img.dimensions();
            let data = rgba_img.into_vec();
            Ok(ImageData::RGBA(data, w, h))
        }
        false => {
            let gray_img = img.to_luma8();
            let (w, h) = gray_img.dimensions();
            let data = gray_img.into_vec();
            Ok(ImageData::Gray(data, w, h))
        }
    }
}

pub(crate) fn hash_bytes(bytes: &[u8]) -> u64 {
    let mut hasher = DefaultHasher::new();
    bytes.hash(&mut hasher);
    hasher.finish()
}

/*
 * Not sure how this works, it appers that the padding is necessary for camera_info
 * f64 arrays, but only if the cursor position is alligned to 8 bytes, not 4.
 * Example: Basler with position 56 requires padding
 * Example: ZedX with position 84 doesn't require padding
 */
pub(crate) fn add_padding_before_f64(
    cursor: &mut Cursor<&mut Vec<u8>>,
) -> Result<(), Box<dyn std::error::Error>> {
    if cursor.position() % 8 == 0 {
        cursor.write_u32::<LittleEndian>(0)?;
    }
    Ok(())
}

// Alternative implementation using to_le_bytes() and from_le_bytes() (more idiomatic)
pub(crate) fn sign_warthog_current(current_measurement: f64) -> f64 {
    let scaling = 1.0 / 16.0;

    let uint_value = (current_measurement / scaling) as u16;
    let uint_bytes = uint_value.to_le_bytes();
    let int_value = i16::from_le_bytes(uint_bytes);
    let float_value = int_value as f64 * scaling;

    float_value
}
