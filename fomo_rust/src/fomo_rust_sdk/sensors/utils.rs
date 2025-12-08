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
    Gray(Array2<u8>),
    BGRA(Array3<u8>),
    RGBFromBayer(Array3<u8>),
}

impl ImageData {
    pub fn len(&self) -> usize {
        match self {
            ImageData::Gray(array_base) => {
                let dim = array_base.dim();
                dim.0 * dim.1 - (11 * dim.0) // image dimensions with the first 3 columns removed (encoders, timestamps, zero column)
            }
            ImageData::BGRA(array_base) => {
                let dim = array_base.dim();
                dim.0 * dim.1 * dim.2
            }
            ImageData::RGBFromBayer(array_base) => {
                let dim = array_base.dim();
                dim.0 * dim.1
            }
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
            // Convert flat Vec<u8> to Array2<u8>
            let array = Array2::from_shape_vec((height, width), data.to_vec())
                .map_err(|_| "Shape mismatch")?;
            Ok(ImageData::Gray(array))
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

            let array = Array3::from_shape_vec((height, width, 3), buf.to_vec())
                .map_err(|_| "Shape mismatch")?;
            Ok(ImageData::RGBFromBayer(array))
        }
        "bgra8" => {
            if data.len() != width * height * 4 {
                return Err("Invalid data size".into());
            }
            let array = Array3::from_shape_vec((height, width, 4), data.to_vec())
                .map_err(|_| "Shape mismatch")?;
            Ok(ImageData::BGRA(array))
        }
        encoding => Err(format!("Unknown encoding {}", encoding).into()),
    }
}

pub fn bytes_from_image(image_data: &ImageData) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    match image_data {
        ImageData::Gray(array_base) => {
            let byte_arr = array_base
                .slice(s![.., 11..]) // Skip first 11 columns, keep all rows
                .iter()
                .copied()
                .collect();
            Ok(byte_arr)
        }
        ImageData::BGRA(array_base) => Ok(array_base.iter().copied().collect()),
        ImageData::RGBFromBayer(_) => todo!(),
    }
}

pub fn save_png<P: AsRef<std::path::Path>>(
    img: &ImageData,
    path: P,
) -> Result<(), image::ImageError> {
    match img {
        ImageData::Gray(arr) => {
            let (h, w) = arr.dim();
            let mut gray = GrayImage::new(w as u32, h as u32);
            for (y, row) in arr.outer_iter().enumerate() {
                for (x, &val) in row.iter().enumerate() {
                    gray.put_pixel(x as u32, y as u32, Luma([val]));
                }
            }
            gray.save(path)
        }
        ImageData::BGRA(arr) => {
            let (h, w, _) = arr.dim();
            let mut out = RgbaImage::new(w as u32, h as u32);
            for y in 0..h {
                for x in 0..w {
                    let px = arr.slice(s![y, x, ..]);
                    out.put_pixel(x as u32, y as u32, Rgba([px[2], px[1], px[0], px[3]]));
                }
            }
            out.save(path)
        }
        ImageData::RGBFromBayer(arr) => {
            let (h, w, _) = arr.dim();
            let mut out = RgbImage::new(w as u32, h as u32);
            for y in 0..h {
                for x in 0..w {
                    let px = arr.slice(s![y, x, ..]);
                    out.put_pixel(x as u32, y as u32, Rgb([px[0], px[1], px[2]]));
                }
            }
            out.save(path)
        }
    }
}

pub fn load_png<P: AsRef<std::path::Path>>(
    path: P,
    is_color: bool,
) -> Result<ImageData, image::ImageError> {
    let img = image::open(path)?;

    match is_color {
        true => {
            let rgba_img = img.to_rgb8();
            let (w, h) = rgba_img.dimensions();
            let mut arr = Array3::<u8>::zeros((h as usize, w as usize, 4));

            for y in 0..h {
                for x in 0..w {
                    let pixel = rgba_img.get_pixel(x, y);
                    arr[[y as usize, x as usize, 0]] = pixel[2]; // B (blue from RGB)
                    arr[[y as usize, x as usize, 1]] = pixel[1]; // G (green)
                    arr[[y as usize, x as usize, 2]] = pixel[0]; // R (red from RGB)
                    arr[[y as usize, x as usize, 3]] = 255; // A (alpha)
                }
            }
            Ok(ImageData::BGRA(arr))
        }
        false => {
            let gray_img = img.to_luma8();

            let (w, h) = gray_img.dimensions();

            let mut arr = Array2::<u8>::zeros((h as usize, w as usize));

            for y in 0..h {
                for x in 0..w {
                    let pixel = gray_img.get_pixel(x, y);
                    arr[[y as usize, x as usize]] = pixel[0]
                }
            }

            Ok(ImageData::Gray(arr))
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
