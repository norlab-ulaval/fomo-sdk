import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import os
import math
import yaml
import argparse


def compute_reprojection_error(objpoints, imgpoints, rvecs, tvecs, mtx, dist):
    """Compute and return reprojection error statistics"""
    total_error = 0
    residuals = []

    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        total_error += error

        diff = imgpoints[i] - imgpoints2
        residuals.extend(np.linalg.norm(diff.reshape(-1, 2), axis=1))

    mean_error = total_error / len(objpoints)
    return mean_error, residuals


def save_yaml_format(mtx, dist, image_width, image_height, filename):
    """Save calibration data in YAML format"""

    # Create rectification matrix (identity for monocular)
    rectification = np.eye(3)

    # Create projection matrix (for monocular, it's [K|0])
    projection = np.zeros((3, 4))
    projection[:3, :3] = mtx

    # Prepare calibration data
    calibration_data = {
        "image_width": int(image_width),
        "image_height": int(image_height),
        "camera_name": "narrow_stereo",
        "camera_matrix": {"rows": 3, "cols": 3, "data": mtx.flatten().tolist()},
        "distortion_model": "plumb_bob",
        "distortion_coefficients": {"rows": 1, "cols": 5, "data": dist[0].tolist()},
        "rectification_matrix": {
            "rows": 3,
            "cols": 3,
            "data": rectification.flatten().tolist(),
        },
        "projection_matrix": {
            "rows": 3,
            "cols": 4,
            "data": projection.flatten().tolist(),
        },
    }

    # Create directory if it doesn't exist
    dirname = os.path.dirname(filename)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    with open(filename, "w") as f:
        yaml.dump(calibration_data, f, default_flow_style=False, sort_keys=False)


def load_calibration_from_yaml(yaml_file):
    """Load calibration parameters from YAML file"""
    with open(yaml_file, "r") as f:
        calibration = yaml.safe_load(f)

    # Extract camera matrix
    camera_matrix = np.array(calibration["camera_matrix"]["data"]).reshape(3, 3)

    # Extract distortion coefficients
    distortion_coeffs = np.array(calibration["distortion_coefficients"]["data"])

    return camera_matrix, distortion_coeffs, calibration


def compute_reprojection_error_from_config(
    yaml_file, image_paths, show_results=True, method_name=""
):
    def compute_reprojection_error_from_config(
        yaml_file, image_paths, show_results=True, method_name=""
    ):
        """
        Compute reprojection error using calibration from config file

        Args:
            yaml_file: Path to YAML calibration file
            image_paths: List of paths to calibration images
            show_results: Whether to show visualization
            method_name: Name for this validation test

        Returns:
            Dictionary with error statistics
        """

        # Load calibration parameters
        camera_matrix, dist_coeffs, config = load_calibration_from_yaml(yaml_file)

        print(f"\n=== {method_name} ===")
        print(f"Loaded calibration from: {os.path.basename(yaml_file)}")
        print(f"Camera matrix:\n{camera_matrix}")
        print(f"Distortion coefficients: {dist_coeffs}")

        # Load and process images to detect mixed board types
        images = []
        for path in image_paths:
            images += glob.glob(path)

        print(f"Processing {len(images)} images...")
        print("Auto-detecting board types from image paths...")

        # Check if we have mixed board types
        has_big_board = any("big" in img_path for img_path in images)
        has_small_board = any("small" in img_path for img_path in images)
        is_mixed_boards = has_big_board and has_small_board

        if is_mixed_boards:
            print("Testing on: MIXED board types (both SMALL and BIG)")
        elif has_big_board:
            print("Testing on: BIG checkerboard images")
        else:
            print("Testing on: SMALL checkerboard images")

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)

    objpoints = []
    imgpoints = []
    successful_images = 0
    small_board_count = 0
    big_board_count = 0

    for i, fname in enumerate(sorted(images)):
        img = cv2.imread(fname)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Determine board type for this specific image
        is_big_board_img = "big" in fname

        if is_big_board_img:
            CHECKERBOARD = (8, 11)
            square_size = 0.06
            big_board_count += 1
        else:
            CHECKERBOARD = (6, 9)
            square_size = 0.023
            small_board_count += 1

        # Prepare 3D points for this board type
        objp = np.zeros((CHECKERBOARD[1] * CHECKERBOARD[0], 3), np.float32)
        objp[:, :2] = np.mgrid[0 : CHECKERBOARD[0], 0 : CHECKERBOARD[1]].T.reshape(
            -1, 2
        )
        objp *= square_size

        # Find checkerboard corners
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

        if ret:
            # Refine corners
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            objpoints.append(objp)
            imgpoints.append(corners2)
            successful_images += 1

    if len(objpoints) == 0:
        print("Error: No valid images found!")
        return None

    print(f"Successfully processed {successful_images}/{len(images)} images")
    if is_mixed_boards:
        print(f"  - Small board images: {small_board_count}")
        print(f"  - Big board images: {big_board_count}")
        print(f"  - Successfully processed: {successful_images} total")

    # Compute reprojection errors
    total_error = 0
    residuals = []
    image_errors = []

    for i in range(len(objpoints)):
        # Estimate pose for this image
        ret, rvec, tvec = cv2.solvePnP(
            objpoints[i], imgpoints[i], camera_matrix, dist_coeffs
        )

        # Project points back
        imgpoints_proj, _ = cv2.projectPoints(
            objpoints[i], rvec, tvec, camera_matrix, dist_coeffs
        )

        # Compute error for this image
        error = cv2.norm(imgpoints[i], imgpoints_proj, cv2.NORM_L2) / len(
            imgpoints_proj
        )
        total_error += error
        image_errors.append(error)

        # Store individual residuals
        diff = imgpoints[i] - imgpoints_proj
        image_residuals = np.linalg.norm(diff.reshape(-1, 2), axis=1)
        residuals.extend(image_residuals)

    mean_error = total_error / len(objpoints)

    # Determine board type for results
    if is_mixed_boards:
        board_type_str = "mixed"
    elif has_big_board:
        board_type_str = "big"
    else:
        board_type_str = "small"

    results = {
        "method_name": method_name,
        "yaml_file": yaml_file,
        "mean_reprojection_error": mean_error,
        "max_error": max(image_errors),
        "min_error": min(image_errors),
        "std_error": np.std(image_errors),
        "residuals": residuals,
        "image_errors": image_errors,
        "num_images": len(objpoints),
        "camera_matrix": camera_matrix,
        "distortion_coeffs": dist_coeffs,
        "board_type": board_type_str,
        "small_board_count": small_board_count,
        "big_board_count": big_board_count,
        "is_mixed_boards": is_mixed_boards,
    }

    print(f"Mean reprojection error: {mean_error:.4f} pixels")
    print(f"Max image error: {max(image_errors):.4f} pixels")
    print(f"Min image error: {min(image_errors):.4f} pixels")

    if show_results:
        # Create visualization for this specific validation
        try:
            output_dir = OUTPUT_DIR
        except NameError:
            output_dir = "data/calibration/basler"
        os.makedirs(output_dir, exist_ok=True)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Residuals histogram
        ax1.hist(residuals, bins=30, alpha=0.7, edgecolor="black", color="skyblue")
        ax1.set_xlabel("Residual (pixels)")
        ax1.set_ylabel("Frequency")
        ax1.set_title("Reprojection Error Residuals")
        ax1.grid(True, alpha=0.3)
        ax1.axvline(
            mean_error, color="red", linestyle="--", label=f"Mean: {mean_error:.3f}"
        )
        ax1.legend()

        # Per-image errors
        ax2.plot(range(len(image_errors)), image_errors, "b-o", markersize=4)
        ax2.set_xlabel("Image Index")
        ax2.set_ylabel("Reprojection Error (pixels)")
        ax2.set_title("Per-Image Reprojection Errors")
        ax2.grid(True, alpha=0.3)
        ax2.axhline(
            mean_error, color="red", linestyle="--", label=f"Mean: {mean_error:.3f}"
        )
        ax2.legend()

        # Error statistics
        ax3.boxplot([residuals], labels=["All Residuals"])
        ax3.set_ylabel("Residual (pixels)")
        ax3.set_title("Residual Distribution")
        ax3.grid(True, alpha=0.3)

        # Summary text
        ax4.axis("off")
        if is_mixed_boards:
            board_info = f"MIXED (Small: {small_board_count}, Big: {big_board_count})"
        elif has_big_board:
            board_info = "BIG"
        else:
            board_info = "SMALL"

        summary_text = f"""
VALIDATION SUMMARY

Method: {method_name}
Board Type: {board_info}
Images: {len(objpoints)} total

REPROJECTION ERRORS:
Mean error: {mean_error:.4f} px
Max error: {max(image_errors):.4f} px
Min error: {min(image_errors):.4f} px
Std deviation: {np.std(image_errors):.4f} px

QUALITY ASSESSMENT:
{"✓ Excellent" if mean_error < 0.5 else "✓ Good" if mean_error < 1.0 else "⚠ Acceptable" if mean_error < 2.0 else "✗ Poor"}
(Mean: {mean_error:.3f}px)

CAMERA PARAMETERS:
fx, fy: {camera_matrix[0, 0]:.1f}, {camera_matrix[1, 1]:.1f}
cx, cy: {camera_matrix[0, 2]:.1f}, {camera_matrix[1, 2]:.1f}
k1, k2: {dist_coeffs[0]:.6f}, {dist_coeffs[1]:.6f}
        """

        ax4.text(
            0.05,
            0.95,
            summary_text,
            transform=ax4.transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
        )

        plt.suptitle(f"{method_name}", fontsize=14, fontweight="bold")
        plt.tight_layout()

        # Save figure
        safe_filename = method_name.replace(" → ", "_to_").replace(" ", "_").lower()
        figure_path = os.path.join(output_dir, f"validation_{safe_filename}.png")
        plt.savefig(figure_path, dpi=300, bbox_inches="tight")
        plt.show()

        print(f"Saved validation plot: {figure_path}")

    return results


def calibrate_camera(paths, method_name, show_images=True, image_delay=500):
    """Calibrate camera and return results

    Args:
        paths: List of image paths
        method_name: Name of calibration method
        show_images: Whether to show images during processing
        image_delay: Delay in milliseconds between images
    """
    is_big_board = "big" in paths[0]

    # Settings
    if is_big_board:
        CHECKERBOARD = (8, 11)  # number of inner corners (cols, rows)
        square_size = 0.06  # size of a square in meters
    else:
        CHECKERBOARD = (6, 9)  # number of inner corners (cols, rows)
        square_size = 0.023  # size of a square in meters

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)

    # Prepare 3D points of the checkerboard corners
    objp = np.zeros((CHECKERBOARD[1] * CHECKERBOARD[0], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : CHECKERBOARD[0], 0 : CHECKERBOARD[1]].T.reshape(-1, 2)
    objp *= square_size

    objpoints = []  # 3D points in world
    imgpoints = []  # 2D points in images

    # Load images
    images = []
    for path in paths:
        images += glob.glob(path)

    print(f"\n=== {method_name} ===")
    print(f"Found {len(images)} images")

    ctr = 0
    image_shape = None
    processed_images = []  # Store processed images for summary visualization
    fast_mode = False
    current_delay = image_delay

    for i, fname in enumerate(sorted(images)):
        img = cv2.imread(fname)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_shape is None:
            image_shape = gray.shape

        # Find checkerboard corners
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

        if ret:
            ctr += 1
            objpoints.append(objp)

            # Refine corner locations
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Visualize the detected corners
            img_with_corners = img.copy()
            cv2.drawChessboardCorners(img_with_corners, CHECKERBOARD, corners2, ret)

            # Add text overlay with image info
            cv2.putText(
                img_with_corners,
                f"{method_name} - Image {ctr}/{len(images)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                img_with_corners,
                f"File: {os.path.basename(fname)}",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                img_with_corners,
                f"Corners found: {len(corners2)}",
                (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            # Store for summary visualization
            processed_images.append(
                {
                    "image": img.copy(),
                    "corners": corners2,
                    "success": True,
                    "filename": os.path.basename(fname),
                }
            )

            if show_images:
                # Resize image for display if too large
                height, width = img_with_corners.shape[:2]
                if width > 1200 or height > 800:
                    scale = min(1200 / width, 800 / height)
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    img_with_corners = cv2.resize(
                        img_with_corners, (new_width, new_height)
                    )

                cv2.imshow(f"Calibration - {method_name}", img_with_corners)

                # Use current delay (might be modified by fast mode)
                wait_time = 0 if fast_mode else current_delay
                key = cv2.waitKey(wait_time)

                # Handle key presses
                if key & 0xFF == ord("q"):
                    cv2.destroyAllWindows()
                    return None
                elif key & 0xFF == ord("s"):
                    cv2.destroyAllWindows()
                    break
                elif key & 0xFF == ord("f"):
                    fast_mode = not fast_mode
                    print(f"Fast mode: {'ON' if fast_mode else 'OFF'}")
        else:
            # Store failed detection for summary
            processed_images.append(
                {
                    "image": img.copy(),
                    "corners": None,
                    "success": False,
                    "filename": os.path.basename(fname),
                }
            )

            if show_images:
                # Show failed detection
                img_failed = img.copy()
                cv2.putText(
                    img_failed,
                    f"{method_name} - FAILED Detection",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )
                cv2.putText(
                    img_failed,
                    f"File: {os.path.basename(fname)}",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

                # Resize if needed
                height, width = img_failed.shape[:2]
                if width > 1200 or height > 800:
                    scale = min(1200 / width, 800 / height)
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    img_failed = cv2.resize(img_failed, (new_width, new_height))

                cv2.imshow(f"Calibration - {method_name}", img_failed)

                # Show failed images briefly or wait in fast mode
                wait_time = 0 if fast_mode else min(200, current_delay)
                key = cv2.waitKey(wait_time)

                if key & 0xFF == ord("q"):
                    cv2.destroyAllWindows()
                    return None
                elif key & 0xFF == ord("f"):
                    fast_mode = not fast_mode
                    print(f"Fast mode: {'ON' if fast_mode else 'OFF'}")

    if show_images:
        cv2.destroyAllWindows()

    # Show summary visualization of processed images
    if show_images:
        create_summary_visualization(processed_images, method_name)

    print(f"Successfully processed {ctr} images out of {len(images)}")

    if ctr < 10:
        print(
            "Warning: Less than 10 images used for calibration. Results may be unreliable."
        )
        return None

    # Calibrate Camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_shape[::-1], None, None
    )

    print("Camera Matrix:")
    print(mtx)
    print("Distortion Coefficients:")
    print(dist)

    # Compute reprojection error
    mean_error, residuals = compute_reprojection_error(
        objpoints, imgpoints, rvecs, tvecs, mtx, dist
    )
    print(f"Mean Reprojection Error: {mean_error:.4f} pixels")

    return {
        "method_name": method_name,
        "camera_matrix": mtx,
        "distortion": dist,
        "mean_error": mean_error,
        "residuals": residuals,
        "image_shape": image_shape,
        "num_images": ctr,
    }


def create_summary_visualization(processed_images, method_name):
    """Create a summary visualization of all processed calibration images"""
    if not processed_images:
        return

    # Limit to first 16 images for visualization
    images_to_show = processed_images[:16]

    # Calculate grid size
    n_images = len(images_to_show)
    cols = min(4, n_images)
    rows = math.ceil(n_images / cols)

    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=(15, 12))
    if rows == 1:
        axes = [axes] if cols == 1 else axes
    else:
        axes = axes.flatten() if rows > 1 else axes

    fig.suptitle(f"{method_name} - Calibration Images Summary", fontsize=16)

    for i, img_data in enumerate(images_to_show):
        if i >= len(axes):
            break

        ax = axes[i]
        img = img_data["image"]

        # Convert BGR to RGB for matplotlib
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Draw corners if successful
        if img_data["success"] and img_data["corners"] is not None:
            # Draw checkerboard corners
            for corner in img_data["corners"]:
                cv2.circle(img_rgb, tuple(corner[0].astype(int)), 5, (0, 255, 0), -1)
            ax.set_title(f"✓ {img_data['filename'][:15]}...", color="green", fontsize=8)
        else:
            ax.set_title(f"✗ {img_data['filename'][:15]}...", color="red", fontsize=8)

        ax.imshow(img_rgb)
        ax.axis("off")

    # Hide unused subplots
    for i in range(n_images, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()

    # Save to output directory
    try:
        output_dir = OUTPUT_DIR
    except NameError:
        output_dir = "data/calibration/basler"
    os.makedirs(output_dir, exist_ok=True)

    filename = f"{method_name.lower().replace(' ', '_')}_summary.png"
    filepath = os.path.join(output_dir, filename)

    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved summary image: {filepath}")


def compare_calibrations(show_images=True, image_delay=500):
    """Compare calibration results between two methods

    Args:
        show_images (bool): Whether to show images during processing
        image_delay (int): Delay in milliseconds between images (0 = wait for keypress)
    """

    print("=== CALIBRATION COMPARISON ===")
    print("Controls during image processing:")
    print("  'q' - Quit current calibration")
    print("  's' - Skip remaining images")
    print("  'f' - Toggle fast mode (no delay)")
    print("  Any other key - Continue")
    print()

    # Define calibration methods
    methods = [
        {
            "name": "Small Board Calibration",
            "paths": [
                "/Volumes/SSD_Matej/calibration/calibration_small_board_ros_tool_outside/*.png",
                "/Volumes/SSD_Matej/calibration/pylon_small_calibration_outside/*.tiff",
            ],
        },
        {
            "name": "Big Board Calibration",
            "paths": [
                "/Volumes/SSD_Matej/calibration/calibration_big_board_ros_tool_outside/*.png",
                "/Volumes/SSD_Matej/calibration/pylon_big_calibration_outside/*.tiff",
            ],
        },
    ]

    results = []

    # Run calibrations
    for method in methods:
        result = calibrate_camera(
            method["paths"], method["name"], show_images, image_delay
        )
        if result is not None:
            results.append(result)

    if len(results) < 2:
        print("Error: Could not complete both calibrations")
        return

    # Create comparison plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Plot residuals histograms
    ax1.hist(
        results[0]["residuals"], bins=30, alpha=0.7, label=results[0]["method_name"]
    )
    ax1.set_xlabel("Residual (pixels)")
    ax1.set_ylabel("Frequency")
    ax1.set_title(f"{results[0]['method_name']} - Residuals")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.hist(
        results[1]["residuals"],
        bins=30,
        alpha=0.7,
        label=results[1]["method_name"],
        color="orange",
    )
    ax2.set_xlabel("Residual (pixels)")
    ax2.set_ylabel("Frequency")
    ax2.set_title(f"{results[1]['method_name']} - Residuals")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Combined comparison
    ax3.hist(
        results[0]["residuals"],
        bins=30,
        alpha=0.5,
        label=f"{results[0]['method_name']} (μ={results[0]['mean_error']:.3f})",
    )
    ax3.hist(
        results[1]["residuals"],
        bins=30,
        alpha=0.5,
        label=f"{results[1]['method_name']} (μ={results[1]['mean_error']:.3f})",
    )
    ax3.set_xlabel("Residual (pixels)")
    ax3.set_ylabel("Frequency")
    ax3.set_title("Calibration Methods Comparison")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Summary statistics
    ax4.axis("off")
    summary_text = "CALIBRATION COMPARISON SUMMARY\n\n"

    for i, result in enumerate(results):
        summary_text += f"{result['method_name']}:\n"
        summary_text += f"  Images used: {result['num_images']}\n"
        summary_text += f"  Mean reprojection error: {result['mean_error']:.4f} px\n"
        summary_text += f"  Max residual: {max(result['residuals']):.3f} px\n"
        summary_text += f"  Focal length (fx, fy): ({result['camera_matrix'][0, 0]:.2f}, {result['camera_matrix'][1, 1]:.2f})\n"
        summary_text += f"  Principal point: ({result['camera_matrix'][0, 2]:.2f}, {result['camera_matrix'][1, 2]:.2f})\n"
        summary_text += f"  Distortion (k1, k2): ({result['distortion'][0][0]:.6f}, {result['distortion'][0][1]:.6f})\n\n"

    # Determine better method
    better_method = (
        results[0]
        if results[0]["mean_error"] < results[1]["mean_error"]
        else results[1]
    )
    summary_text += (
        f"RECOMMENDATION: {better_method['method_name']} shows lower reprojection error"
    )

    ax4.text(
        0.05,
        0.95,
        summary_text,
        transform=ax4.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
    )

    plt.tight_layout()

    # Save to output directory
    try:
        output_dir = OUTPUT_DIR
    except NameError:
        output_dir = "data/calibration/basler"
    os.makedirs(output_dir, exist_ok=True)

    comparison_path = os.path.join(output_dir, "calibration_comparison.png")
    plt.savefig(comparison_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved comparison plot: {comparison_path}")

    # Save calibration files in YAML format
    for result in results:
        filename = f"calibration_{result['method_name'].lower().replace(' ', '_')}.yaml"
        filepath = os.path.join(output_dir, filename)
        height, width = result["image_shape"]
        save_yaml_format(
            result["camera_matrix"], result["distortion"], width, height, filepath
        )
        print(f"Saved {result['method_name']} calibration to {filepath}")

    # Save the better calibration as default
    best_result = min(results, key=lambda x: x["mean_error"])
    height, width = best_result["image_shape"]
    best_path = os.path.join(output_dir, "best_calibration.yaml")
    save_yaml_format(
        best_result["camera_matrix"],
        best_result["distortion"],
        width,
        height,
        best_path,
    )
    print(f"Saved best calibration ({best_result['method_name']}) to {best_path}")


def cross_validate_calibrations():
    """
    Cross-validate calibrations by testing:
    1. All calibrations on ALL available images (to detect overfitting)
    2. Separate analysis by board type for detailed comparison
    """

    print("=== CROSS-VALIDATION ANALYSIS ===")
    print("Testing ALL calibrations on ALL available images to detect overfitting")
    print()

    # Use global paths if available, otherwise use defaults
    try:
        small_board_paths = SMALL_BOARD_PATHS
        big_board_paths = BIG_BOARD_PATHS
        existing_small_cal = EXISTING_SMALL_CAL
        existing_big_cal = EXISTING_BIG_CAL
    except NameError:
        # Fallback to defaults if globals not set
        small_board_paths = [
            "/Volumes/SSD_Matej/calibration/calibration_small_board_ros_tool_outside/*.png",
            "/Volumes/SSD_Matej/calibration/pylon_small_calibration_outside/*.tiff",
        ]
        big_board_paths = [
            "/Volumes/SSD_Matej/calibration/calibration_big_board_ros_tool_outside/*.png",
            "/Volumes/SSD_Matej/calibration/pylon_big_calibration_outside/*.tiff",
        ]
        existing_small_cal = "/Volumes/SSD_Matej/calibration/calibration_small_board_ros_tool_outside/ost.yaml"
        existing_big_cal = "/Volumes/SSD_Matej/calibration/calibration_big_board_ros_tool_outside/ost.yaml"

    # Combine ALL image paths for comprehensive testing
    all_image_paths = small_board_paths + big_board_paths

    # All calibration files to test
    calibrations_to_test = {
        "Existing Small Board Calibration": existing_small_cal,
        "Existing Big Board Calibration": existing_big_cal,
    }

    # Add new calibrations if they exist
    new_calibrations = {
        "New Small Board Calibration": "calibration_small_board_calibration.yaml",
        "New Big Board Calibration": "calibration_big_board_calibration.yaml",
    }

    for cal_name, cal_file in new_calibrations.items():
        if os.path.exists(cal_file):
            calibrations_to_test[cal_name] = cal_file

    results = []

    # Test each calibration on ALL available images
    for cal_name, cal_file in calibrations_to_test.items():
        if os.path.exists(cal_file):
            print(f"\nTesting {cal_name} on ALL images...")

            # Test on ALL images (comprehensive evaluation)
            result = compute_reprojection_error_from_config(
                cal_file,
                all_image_paths,
                show_results=False,
                method_name=f"{cal_name} → ALL Images",
            )
            if result:
                results.append(result)

            # Also test on individual board types for detailed analysis
            result_small = compute_reprojection_error_from_config(
                cal_file,
                small_board_paths,
                show_results=False,
                method_name=f"{cal_name} → Small Board Images",
            )
            if result_small:
                results.append(result_small)

            result_big = compute_reprojection_error_from_config(
                cal_file,
                big_board_paths,
                show_results=False,
                method_name=f"{cal_name} → Big Board Images",
            )
            if result_big:
                results.append(result_big)
        else:
            print(f"Warning: Calibration file not found: {cal_file}")

    if not results:
        print("No calibration files found for cross-validation!")
        return

    # Create separate figures for each validation result
    create_individual_validation_plots(results)

    # Create comprehensive comparison visualization
    create_cross_validation_plots(results)

    # Print summary table
    print("\n" + "=" * 80)
    print("CROSS-VALIDATION SUMMARY")
    print("=" * 80)
    print(f"{'Method':<50} {'Board Type':<12} {'Mean Error':<12} {'Max Error':<12}")
    print("-" * 80)

    for result in results:
        print(
            f"{result['method_name']:<50} {result['board_type'].upper():<12} "
            f"{result['mean_reprojection_error']:<12.4f} {result['max_error']:<12.4f}"
        )

    # Find best performing calibration
    best_result = min(results, key=lambda x: x["mean_reprojection_error"])
    print(f"\nBEST PERFORMING: {best_result['method_name']}")
    print(f"Mean Error: {best_result['mean_reprojection_error']:.4f} pixels")


def create_individual_validation_plots(results):
    """Create separate figures for each cross-validation result"""

    try:
        output_dir = OUTPUT_DIR
    except NameError:
        output_dir = "data/calibration/basler"
    os.makedirs(output_dir, exist_ok=True)

    for i, result in enumerate(results):
        # Create individual figure for each validation result
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Plot 1: Residuals histogram
        ax1.hist(
            result["residuals"], bins=30, alpha=0.7, edgecolor="black", color="skyblue"
        )
        ax1.set_xlabel("Residual (pixels)")
        ax1.set_ylabel("Frequency")
        ax1.set_title("Reprojection Error Residuals")
        ax1.grid(True, alpha=0.3)
        ax1.axvline(
            result["mean_reprojection_error"],
            color="red",
            linestyle="--",
            label=f"Mean: {result['mean_reprojection_error']:.3f}px",
        )
        ax1.legend()

        # Plot 2: Per-image errors (if we have them)
        if "image_errors" in result and result["image_errors"]:
            ax2.plot(
                range(len(result["image_errors"])),
                result["image_errors"],
                "b-o",
                markersize=4,
            )
            ax2.set_xlabel("Image Index")
            ax2.set_ylabel("Reprojection Error (pixels)")
            ax2.set_title("Per-Image Reprojection Errors")
            ax2.grid(True, alpha=0.3)
            ax2.axhline(
                result["mean_reprojection_error"],
                color="red",
                linestyle="--",
                label=f"Mean: {result['mean_reprojection_error']:.3f}px",
            )
            ax2.legend()
        else:
            ax2.axis("off")
            ax2.text(
                0.5,
                0.5,
                "Per-image errors\nnot available",
                ha="center",
                va="center",
                transform=ax2.transAxes,
                fontsize=12,
            )

        # Plot 3: Error statistics box plot
        ax3.boxplot([result["residuals"]], labels=["All Residuals"])
        ax3.set_ylabel("Residual (pixels)")
        ax3.set_title("Residual Distribution")
        ax3.grid(True, alpha=0.3)

        # Plot 4: Summary text
        ax4.axis("off")
        # Format board type info
        if result.get("is_mixed_boards", False):
            board_info = f"MIXED (S:{result.get('small_board_count', 0)}, B:{result.get('big_board_count', 0)})"
        else:
            board_info = result["board_type"].upper()

        summary_text = f"""
VALIDATION SUMMARY

Method: {result["method_name"]}
Board Type: {board_info}
Images: {result["num_images"]}

REPROJECTION ERRORS:
Mean error: {result["mean_reprojection_error"]:.4f} px
Max error: {result["max_error"]:.4f} px
Min error: {result["min_error"]:.4f} px
Std deviation: {result["std_error"]:.4f} px

QUALITY ASSESSMENT:
{
            "✓ Excellent"
            if result["mean_reprojection_error"] < 0.5
            else "✓ Good"
            if result["mean_reprojection_error"] < 1.0
            else "⚠ Acceptable"
            if result["mean_reprojection_error"] < 2.0
            else "✗ Poor"
        } (Mean: {result["mean_reprojection_error"]:.3f}px)

CAMERA PARAMETERS:
fx, fy: {result["camera_matrix"][0, 0]:.1f}, {result["camera_matrix"][1, 1]:.1f}
cx, cy: {result["camera_matrix"][0, 2]:.1f}, {result["camera_matrix"][1, 2]:.1f}
k1, k2: {result["distortion_coeffs"][0]:.6f}, {result["distortion_coeffs"][1]:.6f}
        """

        ax4.text(
            0.05,
            0.95,
            summary_text,
            transform=ax4.transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
        )

        # Set main title
        plt.suptitle(f"{result['method_name']}", fontsize=14, fontweight="bold")
        plt.tight_layout()

        # Save individual figure
        safe_filename = (
            result["method_name"].replace(" → ", "_to_").replace(" ", "_").lower()
        )
        figure_path = os.path.join(output_dir, f"validation_{safe_filename}.png")
        plt.savefig(figure_path, dpi=300, bbox_inches="tight")
        plt.show()

        print(f"Saved individual validation plot: {figure_path}")


def create_cross_validation_plots(results):
    """Create comprehensive plots for cross-validation results"""

    if not results:
        return

    n_results = len(results)
    fig = plt.figure(figsize=(20, 12))

    # Create grid layout
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

    # Plot 1: Mean errors comparison
    ax1 = fig.add_subplot(gs[0, :2])
    methods = [r["method_name"] for r in results]
    errors = [r["mean_reprojection_error"] for r in results]
    colors = [
        "blue" if "small" in r["board_type"].lower() else "orange" for r in results
    ]

    bars = ax1.bar(range(len(methods)), errors, color=colors, alpha=0.7)
    ax1.set_xlabel("Calibration Method")
    ax1.set_ylabel("Mean Reprojection Error (pixels)")
    ax1.set_title("Cross-Validation: Mean Reprojection Errors")
    ax1.set_xticks(range(len(methods)))
    ax1.set_xticklabels(
        [m.replace(" → ", "\n→\n") for m in methods], rotation=45, ha="right"
    )
    ax1.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, error in zip(bars, errors):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{error:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # Plot 2: Error distribution boxplot
    ax2 = fig.add_subplot(gs[0, 2:])
    residuals_data = [r["residuals"] for r in results]
    ax2.boxplot(
        residuals_data,
        labels=[m.split(" → ")[0].replace("Calibration", "Cal") for m in methods],
    )
    ax2.set_ylabel("Residual (pixels)")
    ax2.set_title("Error Distribution Comparison")
    ax2.tick_params(axis="x", rotation=45)
    ax2.grid(True, alpha=0.3)

    # Plot 3-6: Individual histograms for each result
    for i, result in enumerate(results[:4]):  # Show first 4 results
        if i >= 4:
            break
        row = 1 + i // 2
        col = (i % 2) * 2
        ax = fig.add_subplot(gs[row, col : col + 2])

        ax.hist(result["residuals"], bins=30, alpha=0.7, edgecolor="black")
        ax.set_xlabel("Residual (pixels)")
        ax.set_ylabel("Frequency")
        ax.set_title(
            f"{result['method_name']}\nMean: {result['mean_reprojection_error']:.3f}px"
        )
        ax.grid(True, alpha=0.3)
        ax.axvline(
            result["mean_reprojection_error"], color="red", linestyle="--", alpha=0.8
        )

    # Summary statistics table
    if len(results) > 4:
        ax_table = fig.add_subplot(gs[2, 2:])
        ax_table.axis("off")

        table_data = []
        headers = ["Method", "Board", "Mean (px)", "Max (px)", "Std (px)"]

        for result in results:
            table_data.append(
                [
                    result["method_name"].split(" → ")[0].replace("Calibration", "Cal"),
                    result["board_type"].upper(),
                    f"{result['mean_reprojection_error']:.3f}",
                    f"{result['max_error']:.3f}",
                    f"{result['std_error']:.3f}",
                ]
            )

        table = ax_table.table(
            cellText=table_data, colLabels=headers, cellLoc="center", loc="center"
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.2, 1.5)
        ax_table.set_title("Detailed Statistics", pad=20)

    plt.suptitle("Cross-Validation Analysis: Calibration Performance", fontsize=16)

    # Save to output directory
    try:
        output_dir = OUTPUT_DIR
    except NameError:
        output_dir = "data/calibration/basler"
    os.makedirs(output_dir, exist_ok=True)

    summary_path = os.path.join(output_dir, "cross_validation_summary.png")
    plt.savefig(summary_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Saved cross-validation summary: {summary_path}")


def analyze_existing_calibrations():
    """Analyze and compare existing calibration files"""

    print("=== EXISTING CALIBRATION ANALYSIS ===")

    existing_files = {
        "Small Board (Existing)": "/Volumes/SSD_Matej/calibration/calibration_small_board_ros_tool_outside/ost.yaml",
        "Big Board (Existing)": "/Volumes/SSD_Matej/calibration/calibration_big_board_ros_tool_outside/ost.yaml",
    }

    results = []

    for name, file_path in existing_files.items():
        if os.path.exists(file_path):
            try:
                camera_matrix, dist_coeffs, config = load_calibration_from_yaml(
                    file_path
                )

                print(f"\n{name}:")
                print(f"  File: {file_path}")
                print(f"  Camera Matrix:\n{camera_matrix}")
                print(f"  Distortion: {dist_coeffs}")
                print(
                    f"  Focal Length (fx, fy): {camera_matrix[0, 0]:.2f}, {camera_matrix[1, 1]:.2f}"
                )
                print(
                    f"  Principal Point (cx, cy): {camera_matrix[0, 2]:.2f}, {camera_matrix[1, 2]:.2f}"
                )

                results.append(
                    {
                        "name": name,
                        "camera_matrix": camera_matrix,
                        "distortion": dist_coeffs,
                        "file_path": file_path,
                    }
                )

            except Exception as e:
                print(f"Error reading {file_path}: {e}")
        else:
            print(f"File not found: {file_path}")

    if len(results) >= 2:
        print("\n=== COMPARISON ===")
        r1, r2 = results[0], results[1]

        # Compare focal lengths
        fx_diff = abs(r1["camera_matrix"][0, 0] - r2["camera_matrix"][0, 0])
        fy_diff = abs(r1["camera_matrix"][1, 1] - r2["camera_matrix"][1, 1])

        print(f"Focal length difference (fx): {fx_diff:.2f} pixels")
        print(f"Focal length difference (fy): {fy_diff:.2f} pixels")

        # Compare distortion
        dist_diff = np.abs(r1["distortion"] - r2["distortion"])
        print(f"Max distortion coefficient difference: {np.max(dist_diff):.6f}")


def print_saved_files_summary():
    """Print summary of all files saved to output directory"""

    try:
        output_dir = OUTPUT_DIR
    except NameError:
        output_dir = "data/calibration/basler"
    print(f"\n{'=' * 60}")
    print("FILES SAVED SUMMARY")
    print(f"{'=' * 60}")
    print(f"All results saved to: {os.path.abspath(output_dir)}")
    print()

    if os.path.exists(output_dir):
        files = sorted(os.listdir(output_dir))
        if files:
            print("Generated files:")
            for file in files:
                file_path = os.path.join(output_dir, file)
                file_size = os.path.getsize(file_path)
                if file_size > 1024:
                    size_str = f"{file_size / 1024:.1f} KB"
                else:
                    size_str = f"{file_size} bytes"
                print(f"  📁 {file} ({size_str})")
        else:
            print("No files found in output directory.")
    else:
        print("Output directory not found.")

    print(f"{'=' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Camera calibration analysis and comparison tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Run normal calibration comparison (default)
  %(prog)s --mode compare --no-show-images   # Compare calibrations without visualization
  %(prog)s --mode cross-validate             # Cross-validate existing calibrations
  %(prog)s --mode analyze                    # Analyze existing calibration files
  %(prog)s --mode all --delay 100            # Run all analyses with faster image processing
  %(prog)s --small-paths path1/* path2/*     # Custom small board image paths
  %(prog)s --big-paths path1/* path2/*       # Custom big board image paths
  %(prog)s --existing-small /path/to/cal.yaml --existing-big /path/to/cal.yaml
        """,
    )

    parser.add_argument(
        "--mode",
        choices=["compare", "cross-validate", "analyze", "all"],
        default="compare",
        help="Analysis mode to run (default: compare)",
    )

    parser.add_argument(
        "--show-images",
        action="store_true",
        default=True,
        help="Show images during processing (default: True)",
    )

    parser.add_argument(
        "--no-show-images",
        action="store_true",
        help="Disable image visualization during processing",
    )

    parser.add_argument(
        "--delay",
        type=int,
        default=500,
        help="Delay in milliseconds between images (default: 500, 0=wait for keypress)",
    )

    parser.add_argument(
        "--output-dir",
        default="data/calibration/basler",
        help="Output directory for results (default: data/calibration/basler)",
    )

    parser.add_argument(
        "--existing-small",
        default="/Volumes/SSD_Matej/calibration/calibration_small_board_ros_tool_outside/ost.yaml",
        help="Path to existing small board calibration file",
    )

    parser.add_argument(
        "--existing-big",
        default="/Volumes/SSD_Matej/calibration/calibration_big_board_ros_tool_outside/ost.yaml",
        help="Path to existing big board calibration file",
    )

    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")

    args = parser.parse_args()

    # Handle show-images logic
    show_images = args.show_images and not args.no_show_images

    # Update global paths for cross-validation functions
    global \
        SMALL_BOARD_PATHS, \
        BIG_BOARD_PATHS, \
        EXISTING_SMALL_CAL, \
        EXISTING_BIG_CAL, \
        OUTPUT_DIR
    SMALL_BOARD_PATHS = args.small_paths
    BIG_BOARD_PATHS = args.big_paths
    EXISTING_SMALL_CAL = args.existing_small
    EXISTING_BIG_CAL = args.existing_big
    OUTPUT_DIR = args.output_dir

    if not args.quiet:
        print(f"Running mode: {args.mode}")
        print(f"Show images: {show_images}")
        print(f"Image delay: {args.delay}ms")
        print(f"Output directory: {args.output_dir}")
        print()

    if args.mode == "compare":
        compare_calibrations(show_images=show_images, image_delay=args.delay)
        print_saved_files_summary()
    elif args.mode == "cross-validate":
        cross_validate_calibrations()
        print_saved_files_summary()
    elif args.mode == "analyze":
        analyze_existing_calibrations()
    elif args.mode == "all":
        if not args.quiet:
            print("Running complete analysis...")
        compare_calibrations(show_images=show_images, image_delay=args.delay)
        analyze_existing_calibrations()
        cross_validate_calibrations()
        print_saved_files_summary()
