import cv2
import numpy as np
import os
import glob


def load_tiff_images(folder_path):
    """Load all .tiff images from the specified folder."""
    tiff_files = glob.glob(os.path.join(folder_path, "*.tiff")) + glob.glob(
        os.path.join(folder_path, "*.tif")
    )
    tiff_files.sort()  # Sort for consistent ordering
    print(f"Found {len(tiff_files)} TIFF images in {folder_path}")
    return tiff_files


def detect_checkerboard(image_path, checkerboard_size, roi=None):
    """
    Detect checkerboard corners in an image.

    Args:
        image_path: Path to the image file
        checkerboard_size: Tuple (width, height) of internal corners
        roi: Region of interest as (x1, y1, x2, y2)

    Returns:
        corners: Detected corners or None if not found
        gray: Grayscale image for visualization
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not load image: {image_path}")
        return None, None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply ROI if specified
    if roi is not None:
        x1, y1, x2, y2 = roi
        # Create a mask for the ROI
        mask = np.zeros(gray.shape, dtype=np.uint8)
        mask[y1:y2, x1:x2] = 255
        gray_roi = cv2.bitwise_and(gray, mask)
    else:
        gray_roi = gray

    # Find checkerboard corners
    ret, corners = cv2.findChessboardCorners(gray_roi, checkerboard_size, None)

    if ret:
        # Refine corners for better accuracy
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # Filter corners to be within ROI if specified
        if roi is not None:
            x1, y1, x2, y2 = roi
            valid_corners = []
            for corner in corners:
                x, y = corner[0]
                if x1 <= x <= x2 and y1 <= y <= y2:
                    valid_corners.append(corner)

            if len(valid_corners) == len(corners):
                return corners, gray
            else:
                return None, gray

        return corners, gray

    return None, gray


def display_detection_results(image_files, checkerboard_size, roi=None):
    """
    Display detection results and let user select which images to use.

    Returns:
        selected_images: List of tuples (image_path, corners)
    """
    detected_images = []

    print("\nDetecting checkerboards in images...")
    for i, img_path in enumerate(image_files):
        corners, gray = detect_checkerboard(img_path, checkerboard_size, roi)

        if corners is not None:
            detected_images.append((img_path, corners, gray))
            print(f"✓ Detected checkerboard in {os.path.basename(img_path)}")
        else:
            print(f"✗ No checkerboard found in {os.path.basename(img_path)}")

    print(
        f"\nFound checkerboards in {len(detected_images)} out of {len(image_files)} images"
    )

    if len(detected_images) == 0:
        print("No checkerboards detected! Exiting...")
        return []

    # Display images and let user select
    selected_images = []

    print("\nPress 'y' to select image, 'n' to skip, 'q' to finish selection")

    for i, (img_path, corners, gray) in enumerate(detected_images):
        # Create visualization
        img_display = cv2.imread(img_path)
        cv2.drawChessboardCorners(img_display, checkerboard_size, corners, True)

        # Draw ROI if specified
        if roi is not None:
            x1, y1, x2, y2 = roi
            cv2.rectangle(img_display, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Resize for display
        height, width = img_display.shape[:2]
        if height > 800:
            scale = 800 / height
            new_width = int(width * scale)
            new_height = int(height * scale)
            img_display = cv2.resize(img_display, (new_width, new_height))

        window_name = (
            f"Image {i + 1}/{len(detected_images)}: {os.path.basename(img_path)}"
        )
        cv2.imshow(window_name, img_display)

        print(
            f"\nShowing image {i + 1}/{len(detected_images)}: {os.path.basename(img_path)}"
        )
        print(f"Currently selected: {len(selected_images)} images")

        while True:
            key = cv2.waitKey(0) & 0xFF

            if key == ord("y"):
                selected_images.append((img_path, corners))
                print("✓ Image selected")
                break
            elif key == ord("n"):
                print("✗ Image skipped")
                break
            elif key == ord("q"):
                cv2.destroyAllWindows()
                print(f"Selection finished with {len(selected_images)} images")
                return selected_images
            else:
                print("Press 'y' to select, 'n' to skip, 'q' to finish")

        cv2.destroyWindow(window_name)

    cv2.destroyAllWindows()
    return selected_images


def calibrate_camera(selected_images, checkerboard_size, square_size):
    """
    Perform camera calibration using selected images.

    Args:
        selected_images: List of tuples (image_path, corners)
        checkerboard_size: Tuple (width, height) of internal corners
        square_size: Size of checkerboard square in meters

    Returns:
        Camera matrix, distortion coefficients, and reprojection error
    """
    # Prepare object points
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[
        0 : checkerboard_size[0], 0 : checkerboard_size[1]
    ].T.reshape(-1, 2)
    objp *= square_size

    # Arrays to store object points and image points
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane

    # Get image size
    sample_img = cv2.imread(selected_images[0][0])
    img_shape = sample_img.shape[:2][::-1]  # (width, height)

    for img_path, corners in selected_images:
        objpoints.append(objp)
        imgpoints.append(corners)

    print(f"\nPerforming camera calibration with {len(selected_images)} images...")

    # Perform calibration
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_shape, None, None
    )

    if not ret:
        print("Camera calibration failed!")
        return None, None, None

    # Calculate reprojection error
    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        total_error += error

    mean_error = total_error / len(objpoints)

    print("Camera calibration completed!")
    print(f"Reprojection error: {mean_error:.4f} pixels")

    return mtx, dist, mean_error


def main():
    # Configuration
    folder_path = (
        "/Users/mbo/Desktop/FoMo-SDK/data/calibration/pylon_big_calibration_outside_2"
    )
    CHECKERBOARD = (8, 11)
    square_size = 0.06  # 6 cm in meters
    ROI = (85, 112, 1815, 1015)  # (x1, y1, x2, y2)

    print("Camera Calibration Script")
    print("=" * 40)
    print(f"Folder: {folder_path}")
    print(f"Checkerboard size: {CHECKERBOARD}")
    print(f"Square size: {square_size} m")
    print(f"ROI: {ROI}")

    # Load images
    image_files = load_tiff_images(folder_path)

    if len(image_files) == 0:
        print("No TIFF images found in the specified folder!")
        return

    # Detect checkerboards and let user select images
    selected_images = display_detection_results(image_files, CHECKERBOARD, ROI)

    if len(selected_images) < 20:
        print(
            f"\nError: Need at least 20 images for calibration, but only {len(selected_images)} were selected."
        )
        print("Please run the script again and select more images.")
        return

    print(f"\nUsing {len(selected_images)} images for calibration")

    # Perform calibration
    camera_matrix, dist_coeffs, reprojection_error = calibrate_camera(
        selected_images, CHECKERBOARD, square_size
    )

    if camera_matrix is not None:
        print("\n" + "=" * 40)
        print("CALIBRATION RESULTS")
        print("=" * 40)
        print("Camera Matrix:")
        print(camera_matrix)
        print("\nDistortion Coefficients:")
        print(dist_coeffs)
        print(f"\nReprojection Error: {reprojection_error:.4f} pixels")

        # Save results
        output_file = "camera_calibration_results.npz"
        np.savez(
            output_file,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            reprojection_error=reprojection_error,
        )
        print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
