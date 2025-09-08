#!/usr/bin/env python3
# The script reads a QR code from an image file, trying multiple strategies to decode it.

import sys
import cv2
import numpy as np

def try_decode(detector, img):
    data, pts, _ = detector.detectAndDecode(img)
    if data:
        return data
    # try multi
    datas, points, _ = detector.detectAndDecodeMulti(img)
    if datas and len(datas) > 0 and datas[0]:
        return datas[0]
    return None

def main(path):
    img = cv2.imread(path)
    if img is None:
        print("Error: could not read", path)
        return

    detector = cv2.QRCodeDetector()

    # Whole image
    data = try_decode(detector, img)
    if data:
        print("Decoded QR payload (full image):", data)
        return

    # Heuristic ROI: right side where we placed it (adjust margins if needed)
    H, W = img.shape[:2]
    roi_x0 = max(0, int(W - 40 - 320 - 20))  # match writer placement (x0 ≈ W - QR_SIZE - 40)
    roi_y0 = max(0, (H - 320)//2 - 20)
    roi_x1 = min(W, roi_x0 + 320 + 40)
    roi_y1 = min(H, roi_y0 + 320 + 40)
    roi = img[roi_y0:roi_y1, roi_x0:roi_x1]

    data = try_decode(detector, roi)
    if data:
        print("Decoded QR payload (ROI):", data)
        return

    # Preprocess ROI for clarity
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # Upscale to make modules bigger for detector
    gray_big = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_NEAREST)
    # Binarize (strong contrast)
    _, bw = cv2.threshold(gray_big, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    data = try_decode(detector, bw)
    if data:
        print("Decoded QR payload (ROI preprocessed):", data)
        return

    print("No QR code detected. Tips:")
    print("- Ensure USE_QR=True in the writer.")
    print("- Keep QR_SIZE >= 300 and QR_BORDER >= 4.")
    print("- Use PNG (lossless); avoid JPEG compression.")
    print("- Try better lighting/contrast if capturing from a photo or screen.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python read_overlay_qr.py <overlay_image>")
    else:
        main(sys.argv[1])
