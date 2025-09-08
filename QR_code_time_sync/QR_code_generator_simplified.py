#!/usr/bin/env python3
# QR-only overlay of UTC epoch time (ns or us), ~120 FPS target.
# Optional FPS text with --show-fps. Window shows ONLY the QR (plus optional FPS).

import time
import argparse
import cv2
import numpy as np
import segno  # pip install segno

# Window defaults 
WINDOW_NAME = "Epoch QR"
DEFAULT_W, DEFAULT_H = 1280, 720
DEFAULT_FPS = 120.0
BG_COLOR = (18, 18, 18)
FONT = cv2.FONT_HERSHEY_SIMPLEX

# QR defaults (tune as needed) 
DEFAULT_ECL = "H"      # L/M/Q/H
DEFAULT_BORDER = 4     # quiet zone in modules
DEFAULT_SCALE = 10     # module pixel scale (only used if --auto-scale is off)

# UTC epoch (ns) anchored to a monotonic clock to avoid mid-run jumps 
_BASE_EPOCH_NS = time.time_ns()
_BASE_PERF_NS  = time.perf_counter_ns()

def now_epoch_ns() -> int:
    return _BASE_EPOCH_NS + (time.perf_counter_ns() - _BASE_PERF_NS)

# Fast QR: render segno matrix directly to NumPy (no PNG round-trip)
def qr_numpy(payload: str, ecl: str, border: int, scale: int,
             auto_scale_dims=None) -> np.ndarray:
    """
    Build a crisp QR (uint8 grayscale, 0=black, 255=white).
    If auto_scale_dims=(W,H), choose the largest integer scale that fits in the window.
    """
    q = segno.make(payload, error=ecl, micro=False, boost_error=False)
    # True=dark; convert to 0/255
    mat = np.array([[0 if cell else 255 for cell in row] for row in q.matrix], dtype=np.uint8)

    # Determine scale
    if auto_scale_dims is not None:
        W, H = auto_scale_dims
        # Size in modules (without border)
        h_mod, w_mod = mat.shape
        # Try to fit QR + quiet zone inside window; compute max integer scale
        # total modules per side = core + 2*border
        tot_w_mod = w_mod + 2 * border
        tot_h_mod = h_mod + 2 * border
        scale_w = max(1, (W // tot_w_mod))
        scale_h = max(1, (H // tot_h_mod))
        scale = max(1, min(scale_w, scale_h))

    # Add quiet zone (white)
    if border:
        mat = np.pad(mat, border, mode='constant', constant_values=255)

    # Scale up with nearest-neighbor to keep modules crisp
    if scale > 1:
        mat = cv2.resize(mat, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

    return mat

def main():
    ap = argparse.ArgumentParser(description="QR-only UTC epoch overlay")
    ap.add_argument("--unit", choices=["ns", "us"], default="ns",
                    help="encode epoch in nanoseconds or microseconds (default: ns)")
    ap.add_argument("--show-fps", action="store_true", help="draw FPS in bottom-left")
    ap.add_argument("--width", type=int, default=DEFAULT_W)
    ap.add_argument("--height", type=int, default=DEFAULT_H)
    ap.add_argument("--fps", type=float, default=DEFAULT_FPS, help="target FPS (default 120)")
    ap.add_argument("--ecl", choices=list("LMQH"), default=DEFAULT_ECL, help="QR error correction level")
    ap.add_argument("--border", type=int, default=DEFAULT_BORDER, help="quiet zone in modules")
    ap.add_argument("--scale", type=int, default=DEFAULT_SCALE, help="module pixel scale (ignored if --auto-scale)")
    ap.add_argument("--auto-scale", action="store_true",
                    help="auto-fit QR to window using integer scaling (recommended)")
    args = ap.parse_args()

    width, height = args.width, args.height
    target_fps = max(1.0, args.fps)
    frame_ns = int(1e9 / target_fps)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, width, height)

    img = np.full((height, width, 3), BG_COLOR, dtype=np.uint8)

    # FPS tracking (only if requested)
    last_tick = time.perf_counter_ns()
    frames = 0
    fps_display = 0.0
    fps_update_interval = 0.25
    fps_accum_time = 0.0

    while True:
        loop_start = time.perf_counter_ns()

        # UTC epoch value (ns or us)
        epoch_ns = now_epoch_ns()
        if args.unit == "us":
            value = epoch_ns // 1_000  # integer microseconds
        else:
            value = epoch_ns          # nanoseconds

        # Clear background (no realloc)
        img[:] = BG_COLOR

        # Generate QR and paste centered
        if args.auto_scale:
            qr_gray = qr_numpy(str(value), args.ecl, args.border, args.scale,
                               auto_scale_dims=(width, height))
        else:
            qr_gray = qr_numpy(str(value), args.ecl, args.border, args.scale)

        qh, qw = qr_gray.shape
        x0 = (width  - qw) // 2
        y0 = (height - qh) // 2
        roi = img[y0:y0+qh, x0:x0+qw]
        roi[:, :, 0] = qr_gray
        roi[:, :, 1] = qr_gray
        roi[:, :, 2] = qr_gray

        # Optional FPS overlay
        if args.show_fps:
            now_tick = time.perf_counter_ns()
            dt_sec = (now_tick - last_tick) / 1e9
            last_tick = now_tick
            frames += 1
            fps_accum_time += dt_sec
            if fps_accum_time >= fps_update_interval:
                fps_display = frames / fps_accum_time
                frames = 0
                fps_accum_time = 0.0
            cv2.putText(img, f"{fps_display:5.1f} FPS",
                        (20, height - 30), FONT, 0.9, (180, 255, 180), 2, cv2.LINE_AA)

        cv2.imshow(WINDOW_NAME, img)

        # Pace to target FPS
        elapsed = time.perf_counter_ns() - loop_start
        remaining = frame_ns - elapsed
        if remaining > 0:
            time.sleep(remaining / 1e9)

        k = cv2.waitKey(1) & 0xFF
        if k in (ord('q'), 27):  # q or Esc
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
