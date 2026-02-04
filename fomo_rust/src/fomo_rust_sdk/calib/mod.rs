use crate::fomo_rust_sdk::sensors::utils::ImageData;
use configparser::ini::Ini;
use opencv::{calib3d, core::Mat, imgproc, prelude::*, Result};
use std::fs;

use serde::{Deserialize, Serialize};
use serde_json;

const CALIBRATION_PATH: &str = "../data/calib_to_ijrr";

#[derive(Debug, Deserialize, Serialize)]
pub(crate) struct CameraCalibration {
    pub(crate) distortion_model: String,
    pub(crate) d: Vec<f64>,
    pub(crate) k: [f64; 9],
    pub(crate) r: [f64; 9],
    pub(crate) p: [f64; 12],
    pub(crate) binning_x: u32,
    pub(crate) binning_y: u32,
    pub(crate) x_offset: u32,
    pub(crate) y_offset: u32,
    pub(crate) height_offset: u32,
    pub(crate) width_offset: u32,
    pub(crate) do_rectify: bool,
}

// Helper function to convert ImageData to OpenCV Mat
fn imagedata_to_mat(image_data: &ImageData) -> Result<Mat> {
    match image_data {
        ImageData::BGRA(arr) => {
            let (height, width, channels) = arr.dim();
            let data: Vec<u8> = arr.iter().copied().collect();
            let rows: Vec<&[u8]> = data.chunks(width * channels).collect();
            let mat = Mat::from_slice_2d(&rows)?;
            let reshaped = mat.reshape(channels as i32, height as i32)?;
            reshaped.try_clone()
        }
        ImageData::Gray(arr) => {
            let (_, width) = arr.dim();
            let data: Vec<u8> = arr.iter().copied().collect();
            let rows: Vec<&[u8]> = data.chunks(width).collect();
            Mat::from_slice_2d(&rows)
        }
        ImageData::RGBFromBayer(arr) => {
            let (height, width, channels) = arr.dim();
            let data: Vec<u8> = arr.iter().copied().collect();
            let rows: Vec<&[u8]> = data.chunks(width * channels).collect();
            let mat = Mat::from_slice_2d(&rows)?;
            let reshaped = mat.reshape(channels as i32, height as i32)?;
            reshaped.try_clone()
        }
    }
}

// Helper function to convert OpenCV Mat to ImageData
fn mat_to_imagedata(mat: &Mat) -> Result<ImageData> {
    let height = mat.rows() as usize;
    let width = mat.cols() as usize;
    let channels = mat.channels() as usize;

    let data_bytes = mat.data_bytes()?;
    let data = data_bytes.to_vec();

    match channels {
        1 => {
            let arr = ndarray::Array2::from_shape_vec((height, width), data)
                .map_err(|_| opencv::Error::new(opencv::core::StsError, "Shape mismatch"))?;
            Ok(ImageData::Gray(arr))
        }
        3 => {
            let arr = ndarray::Array3::from_shape_vec((height, width, channels), data)
                .map_err(|_| opencv::Error::new(opencv::core::StsError, "Shape mismatch"))?;
            Ok(ImageData::RGBFromBayer(arr))
        }
        4 => {
            let arr = ndarray::Array3::from_shape_vec((height, width, channels), data)
                .map_err(|_| opencv::Error::new(opencv::core::StsError, "Shape mismatch"))?;
            Ok(ImageData::BGRA(arr))
        }
        _ => Err(opencv::Error::new(
            opencv::core::StsError,
            "Unsupported channel count",
        )),
    }
}

pub fn get_calib_params(
    filepath: &str,
) -> Result<(Mat, Mat, Mat, Mat, Mat, Mat), Box<dyn std::error::Error>> {
    let mut conf = Ini::new();
    conf.load(filepath).map_err(|e| {
        Box::new(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("Failed to load calibration config: {}", e),
        )) as Box<dyn std::error::Error>
    })?;

    // Extract left camera parameters
    // Extract left camera parameters
    let fx_left = conf
        .getfloat("LEFT_CAM_FHD1200", "fx")?
        .ok_or("Missing fx_left parameter")?;
    let fy_left = conf
        .getfloat("LEFT_CAM_FHD1200", "fy")?
        .ok_or("Missing fy_left parameter")?;
    let cx_left = conf
        .getfloat("LEFT_CAM_FHD1200", "cx")?
        .ok_or("Missing cx_left parameter")?;
    let cy_left = conf
        .getfloat("LEFT_CAM_FHD1200", "cy")?
        .ok_or("Missing cy_left parameter")?;
    let k1_left = conf
        .getfloat("LEFT_CAM_FHD1200", "k1")?
        .ok_or("Missing k1_left parameter")?;
    let k2_left = conf
        .getfloat("LEFT_CAM_FHD1200", "k2")?
        .ok_or("Missing k2_left parameter")?;
    let p1_left = conf
        .getfloat("LEFT_CAM_FHD1200", "p1")?
        .ok_or("Missing p1_left parameter")?;
    let p2_left = conf
        .getfloat("LEFT_CAM_FHD1200", "p2")?
        .ok_or("Missing p2_left parameter")?;
    let k3_left = conf
        .getfloat("LEFT_CAM_FHD1200", "k3")?
        .ok_or("Missing k3_left parameter")?;

    // Extract right camera parameters
    let fx_right = conf
        .getfloat("RIGHT_CAM_FHD1200", "fx")?
        .ok_or("Missing fx_right parameter")?;
    let fy_right = conf
        .getfloat("RIGHT_CAM_FHD1200", "fy")?
        .ok_or("Missing fy_right parameter")?;
    let cx_right = conf
        .getfloat("RIGHT_CAM_FHD1200", "cx")?
        .ok_or("Missing cx_right parameter")?;
    let cy_right = conf
        .getfloat("RIGHT_CAM_FHD1200", "cy")?
        .ok_or("Missing cy_right parameter")?;
    let k1_right = conf
        .getfloat("RIGHT_CAM_FHD1200", "k1")?
        .ok_or("Missing k1_right parameter")?;
    let k2_right = conf
        .getfloat("RIGHT_CAM_FHD1200", "k2")?
        .ok_or("Missing k2_right parameter")?;
    let p1_right = conf
        .getfloat("RIGHT_CAM_FHD1200", "p1")?
        .ok_or("Missing p1_right parameter")?;
    let p2_right = conf
        .getfloat("RIGHT_CAM_FHD1200", "p2")?
        .ok_or("Missing p2_right parameter")?;
    let k3_right = conf
        .getfloat("RIGHT_CAM_FHD1200", "k3")?
        .ok_or("Missing k3_right parameter")?;

    // Extract stereo parameters
    let baseline = conf
        .getfloat("STEREO", "Baseline")?
        .ok_or("Missing baseline parameter")?;
    let ty = conf
        .getfloat("STEREO", "TY")?
        .ok_or("Missing TY parameter")?;
    let tz = conf
        .getfloat("STEREO", "TZ")?
        .ok_or("Missing TZ parameter")?;
    let rx = conf
        .getfloat("STEREO", "RX_FHD1200")?
        .ok_or("Missing RX parameter")?;
    let ry = conf
        .getfloat("STEREO", "CV_FHD1200")?
        .ok_or("Missing RY parameter")?;
    let rz = conf
        .getfloat("STEREO", "RZ_FHD1200")?
        .ok_or("Missing RZ parameter")?;

    // Camera matrices
    let k_left = Mat::from_slice_2d(&[
        &[fx_left, 0.0, cx_left],
        &[0.0, fy_left, cy_left],
        &[0.0, 0.0, 1.0],
    ])?;
    let k_right = Mat::from_slice_2d(&[
        &[fx_right, 0.0, cx_right],
        &[0.0, fy_right, cy_right],
        &[0.0, 0.0, 1.0],
    ])?;

    // Distortion coefficients
    let binding = [k1_left, k2_left, p1_left, p2_left, k3_left];
    let d_left = Mat::from_slice(&binding)?;
    let d_left = d_left.reshape(1, 5)?.try_clone()?;
    let binding = [k1_right, k2_right, p1_right, p2_right, k3_right];
    let d_right = Mat::from_slice(&binding)?;
    let d_right = d_right.reshape(1, 5)?.try_clone()?;

    // Stereo extrinsics
    let binding = [baseline / 1000.0, ty / 1000.0, tz / 1000.0];
    let t = Mat::from_slice(&binding)?;
    let t = t.reshape(1, 3)?.try_clone()?;

    let binding = [rx, ry, rz];
    let rvec = Mat::from_slice(&binding)?;
    let rvec = rvec.reshape(1, 3)?.try_clone()?;
    Ok((k_left, k_right, d_left, d_right, t, rvec))
}

pub fn stereo_calib_mat(
    left_img: &Mat,
    right_img: &Mat,
) -> Result<(Mat, Mat), Box<dyn std::error::Error>> {
    // Path to .conf file
    let config_path = format!("{}/zedx_SN41705768.conf", CALIBRATION_PATH);
    let (k_left, k_right, d_left, d_right, t, rvec) = get_calib_params(&config_path)?;

    let mut r = Mat::default();
    calib3d::rodrigues_def(&rvec, &mut r)?;

    let mut r1 = Mat::default();
    let mut r2 = Mat::default();
    let mut p1 = Mat::default();
    let mut p2 = Mat::default();
    let mut q = Mat::default();

    let size = left_img.size()?;

    calib3d::stereo_rectify(
        &k_left,
        &d_left,
        &k_right,
        &d_right,
        size,
        &r,
        &t,
        &mut r1,
        &mut r2,
        &mut p1,
        &mut p2,
        &mut q,
        calib3d::CALIB_ZERO_DISPARITY,
        -0.0,
        size,
        &mut opencv::core::Rect::default(),
        &mut opencv::core::Rect::default(),
    )?;

    // Create rectification maps
    let mut map1_left = Mat::default();
    let mut map2_left = Mat::default();
    let mut map1_right = Mat::default();
    let mut map2_right = Mat::default();

    calib3d::init_undistort_rectify_map(
        &k_left,
        &d_left,
        &r1,
        &p1,
        left_img.size()?,
        opencv::core::CV_32FC1,
        &mut map1_left,
        &mut map2_left,
    )?;

    calib3d::init_undistort_rectify_map(
        &k_right,
        &d_right,
        &r2,
        &p2,
        right_img.size()?,
        opencv::core::CV_32FC1,
        &mut map1_right,
        &mut map2_right,
    )?;

    // Apply rectification to images
    let mut left_rectified = Mat::default();
    let mut right_rectified = Mat::default();

    imgproc::remap(
        &left_img,
        &mut left_rectified,
        &map1_left,
        &map2_left,
        imgproc::INTER_LINEAR,
        opencv::core::BORDER_CONSTANT,
        opencv::core::Scalar::default(),
    )?;

    imgproc::remap(
        &right_img,
        &mut right_rectified,
        &map1_right,
        &map2_right,
        imgproc::INTER_LINEAR,
        opencv::core::BORDER_CONSTANT,
        opencv::core::Scalar::default(),
    )?;

    Ok((left_rectified, right_rectified))
}

pub fn stereo_calib(
    left_image: &ImageData,
    right_image: &ImageData,
) -> Result<(ImageData, ImageData), Box<dyn std::error::Error>> {
    // Convert ImageData to OpenCV Mat
    let left_img = imagedata_to_mat(left_image)?;
    let right_img = imagedata_to_mat(right_image)?;

    let (left_rectified, right_rectified) = stereo_calib_mat(&left_img, &right_img)?;

    // Convert rectified OpenCV Mats back to ImageData
    let left_rectified_data = mat_to_imagedata(&left_rectified)?;
    let right_rectified_data = mat_to_imagedata(&right_rectified)?;
    Ok((left_rectified_data, right_rectified_data))
}

pub fn mono_calib_mat(img: &Mat) -> Result<Mat, Box<dyn std::error::Error>> {
    let camera_file_path = format!("{}/basler.json", CALIBRATION_PATH);
    let data = fs::read_to_string(&camera_file_path).map_err(|e| {
        format!(
            "Failed to read calibration file '{}': {}",
            &camera_file_path, e
        )
    })?;
    let calib: CameraCalibration = serde_json::from_str(&data)?;
    let k = Mat::from_slice(&calib.k)?;
    let k = k.reshape(1, 3)?;

    let d = Mat::from_slice(&calib.d)?;
    let d = d.reshape(1, 5)?;

    let mut new_k = Mat::default();
    let mut rectified = Mat::default();

    calib3d::undistort(&img, &mut rectified, &k, &d, &mut new_k)?;
    Ok(rectified)
}

pub fn mono_calib(img: &ImageData) -> Result<ImageData, Box<dyn std::error::Error>> {
    let mat_img = imagedata_to_mat(img)?;
    let rectified = mono_calib_mat(&mat_img)?;
    let output = mat_to_imagedata(&rectified)?;
    Ok(output)
}
