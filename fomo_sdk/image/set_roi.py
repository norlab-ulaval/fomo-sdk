import cv2
import numpy as np
import os
import glob
from pathlib import Path


class ROIVisualizer:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.roi = None
        self.drawing = False
        self.start_point = None

    def load_tiff_images(self):
        """Load all TIFF images from the specified folder"""
        tiff_patterns = ["*.tiff", "*.tif", "*.TIFF", "*.TIF"]
        image_paths = []

        for pattern in tiff_patterns:
            image_paths.extend(glob.glob(os.path.join(self.folder_path, pattern)))

        if not image_paths:
            print(f"No TIFF images found in {self.folder_path}")
            return []

        print(f"Found {len(image_paths)} TIFF images")
        return sorted(image_paths)

    def mouse_callback(self, event, x, y, flags, param):
        """Mouse callback function for ROI selection"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                # Create a copy of the image to draw temporary rectangle
                temp_img = param["img"].copy()
                cv2.rectangle(temp_img, self.start_point, (x, y), (0, 255, 0), 2)
                cv2.imshow("Select ROI", temp_img)

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            end_point = (x, y)

            # Calculate ROI coordinates (top-left and bottom-right)
            x1 = min(self.start_point[0], end_point[0])
            y1 = min(self.start_point[1], end_point[1])
            x2 = max(self.start_point[0], end_point[0])
            y2 = max(self.start_point[1], end_point[1])

            self.roi = (x1, y1, x2, y2)
            print(f"ROI selected: ({x1}, {y1}) to ({x2}, {y2})")

            # Draw final rectangle
            cv2.rectangle(param["img"], (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imshow("Select ROI", param["img"])

    def select_roi(self, image_path):
        """Allow user to select ROI on the first image"""
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Could not load image: {image_path}")
            return False

        # Convert to 8-bit if necessary for display
        if img.dtype != np.uint8:
            if img.max() > 255:
                img_display = (img / img.max() * 255).astype(np.uint8)
            else:
                img_display = img.astype(np.uint8)
        else:
            img_display = img.copy()

        # If image has multiple channels, convert to BGR for display
        if len(img_display.shape) == 3:
            if img_display.shape[2] == 1:
                img_display = cv2.cvtColor(img_display, cv2.COLOR_GRAY2BGR)
        else:
            img_display = cv2.cvtColor(img_display, cv2.COLOR_GRAY2BGR)

        print("Click and drag to select ROI. Press 'Enter' when done, 'ESC' to exit.")

        cv2.namedWindow("Select ROI", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Select ROI", 800, 600)

        # Set mouse callback
        callback_params = {"img": img_display}
        cv2.setMouseCallback("Select ROI", self.mouse_callback, callback_params)

        cv2.imshow("Select ROI", img_display)

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 13:  # Enter key
                if self.roi is not None:
                    cv2.destroyAllWindows()
                    return True
                else:
                    print("Please select an ROI first")
            elif key == 27:  # ESC key
                cv2.destroyAllWindows()
                return False

    def visualize_roi_on_images(self, image_paths):
        """Visualize ROI on all images one by one"""
        if self.roi is None:
            print("No ROI selected!")
            return

        x1, y1, x2, y2 = self.roi

        for i, image_path in enumerate(image_paths):
            img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"Could not load image: {image_path}")
                continue

            # Convert to 8-bit if necessary for display
            if img.dtype != np.uint8:
                if img.max() > 255:
                    img_display = (img / img.max() * 255).astype(np.uint8)
                else:
                    img_display = img.astype(np.uint8)
            else:
                img_display = img.copy()

            # If image has multiple channels, convert to BGR for display
            if len(img_display.shape) == 3:
                if img_display.shape[2] == 1:
                    img_display = cv2.cvtColor(img_display, cv2.COLOR_GRAY2BGR)
            else:
                img_display = cv2.cvtColor(img_display, cv2.COLOR_GRAY2BGR)

            # Draw ROI rectangle
            cv2.rectangle(img_display, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Add text with image info
            filename = os.path.basename(image_path)
            cv2.putText(
                img_display,
                f"{i + 1}/{len(image_paths)}: {filename}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            cv2.namedWindow("ROI Visualization", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("ROI Visualization", 800, 600)
            cv2.imshow("ROI Visualization", img_display)

            print(f"Showing image {i + 1}/{len(image_paths)}: {filename}")
            print("Press 'n' for next, 'p' for previous, 'q' to quit")

            while True:
                key = cv2.waitKey(0) & 0xFF
                if key == ord("n") or key == 13:  # Next image (n or Enter)
                    break
                elif key == ord("p") and i > 0:  # Previous image
                    i -= 2  # Will be incremented by 1 in the loop
                    break
                elif key == ord("q") or key == 27:  # Quit (q or ESC)
                    cv2.destroyAllWindows()
                    return

        cv2.destroyAllWindows()
        print("Finished visualizing all images!")


def main():
    folder_path = (
        "/Users/mbo/Desktop/FoMo-SDK/data/calibration/pylon_big_calibration_outside_2"
    )

    # Check if folder exists
    if not os.path.exists(folder_path):
        print(f"Folder does not exist: {folder_path}")
        return

    visualizer = ROIVisualizer(folder_path)

    # Load TIFF images
    image_paths = visualizer.load_tiff_images()
    if not image_paths:
        return

    # Select ROI on the first image
    print(f"Using first image for ROI selection: {os.path.basename(image_paths[0])}")
    if visualizer.select_roi(image_paths[0]):
        # Visualize ROI on all images
        visualizer.visualize_roi_on_images(image_paths)
    else:
        print("ROI selection cancelled.")


if __name__ == "__main__":
    main()
