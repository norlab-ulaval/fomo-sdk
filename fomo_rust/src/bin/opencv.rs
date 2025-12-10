use fomo_rust_sdk::fomo_rust_sdk::calib::{mono_calib_mat, stereo_calib_mat};
use opencv::{
    core::{hconcat, vconcat, Mat, Point, Scalar, Vector},
    highgui, imgcodecs, imgproc,
    prelude::*,
    Result,
};

fn create_stacked_visualization(
    left_img: &Mat,
    right_img: &Mat,
    left_rectified: &Mat,
    right_rectified: &Mat,
) -> Result<()> {
    let input_mats: Vector<Mat> = Vector::from_iter([left_img.clone(), right_img.clone()]);
    let mut raw_stacked = Mat::default();
    hconcat(&input_mats, &mut raw_stacked)?;

    let input_mats: Vector<Mat> =
        Vector::from_iter([left_rectified.clone(), right_rectified.clone()]);
    let mut rectified_stacked = Mat::default();
    hconcat(&input_mats, &mut rectified_stacked)?;

    // Draw horizontal lines on raw stacked image (red lines)
    let mut raw_with_lines = raw_stacked.clone();
    draw_horizontal_lines(&mut raw_with_lines, Scalar::new(0.0, 0.0, 255.0, 0.0))?; // Red in BGR

    // Draw horizontal lines on rectified stacked image (green lines)
    let mut rectified_with_lines = rectified_stacked.clone();
    draw_horizontal_lines(&mut rectified_with_lines, Scalar::new(0.0, 255.0, 0.0, 0.0))?; // Green in BGR

    let input_mats: Vector<Mat> =
        Vector::from_iter([raw_with_lines.clone(), rectified_with_lines.clone()]);
    let mut all_stacked = Mat::default();
    vconcat(&input_mats, &mut all_stacked)?;
    // highgui::imshow("Original", &raw_with_lines)?;
    highgui::imshow("All", &all_stacked)?;

    // imgcodecs::imwrite("zedx.jpg", &all_stacked, &Vector::new())?;

    highgui::wait_key(0)?;
    highgui::destroy_all_windows()?;
    Ok(())
}

fn draw_horizontal_lines(img: &mut Mat, color: Scalar) -> Result<()> {
    let height = img.rows();
    let width = img.cols();

    // Draw horizontal lines every 100 pixels
    let line_spacing = 100;
    let thickness = 2;

    for y in (line_spacing..height).step_by(line_spacing as usize) {
        let start_point = Point::new(0, y);
        let end_point = Point::new(width - 1, y);

        imgproc::line(
            img,
            start_point,
            end_point,
            color,
            thickness,
            imgproc::LINE_8,
            0,
        )?;
    }

    Ok(())
}

fn stereo_calib() -> Result<(), Box<dyn std::error::Error>> {
    // Load raw stereo images
    let left_img_path =
        "/Users/mbo/Desktop/zed_node_left_raw_image_raw_color-1758735719-139372000.png".to_string();
    let right_img_path = left_img_path.replace("left", "right");
    let left_img = imgcodecs::imread(&left_img_path, imgcodecs::IMREAD_COLOR)?;
    let right_img = imgcodecs::imread(&right_img_path, imgcodecs::IMREAD_COLOR)?;

    let (left_rectified, right_rectified) = stereo_calib_mat(&left_img, &right_img)?;
    // Create stacked visualization with horizontal lines
    create_stacked_visualization(&left_img, &right_img, &left_rectified, &right_rectified)?;

    Ok(())
}

fn basler_calib() -> Result<(), Box<dyn std::error::Error>> {
    // Load raw stereo images
    let img_path = "../data/mono/basler_1750948540-353413885.png".to_string();
    let img = imgcodecs::imread(&img_path, imgcodecs::IMREAD_COLOR)?;
    let rectified = mono_calib_mat(&img)?;
    highgui::imshow("Original", &img)?;
    highgui::imshow("Rectified", &rectified)?;

    highgui::wait_key(0)?;
    highgui::destroy_all_windows()?;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    stereo_calib()?;
    // basler_calib()?;
    Ok(())
}
