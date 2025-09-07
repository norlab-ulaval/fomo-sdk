#!/usr/bin/env python3
# This version optimizes QR code generation to avoid PNG encoding/decoding
# and uses direct NumPy array manipulation for faster performance.
# Full-screen overlay showing UTC epoch time in nanoseconds at ~120 FPS (ideally)
import time
import datetime as dt
import cv2
import numpy as np
import segno  # pip install segno

WINDOW_NAME = "Epoch Time @ 120 FPS"
WIDTH, HEIGHT = 1280, 720
BG_COLOR = (18, 18, 18)
FG_COLOR = (240, 240, 240)
ACCENT = (140, 200, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX

TARGET_FPS = 120.0
FRAME_NS = int(1e9 / TARGET_FPS)

# QR settings: tune for your camera; "L" is faster/smaller than "M" and "H"
QR_ECL = "M"  # error correction level
QR_BORDER = 4       # quiet zone in modules
QR_SCALE  = 8       # module pixel scale

#  UTC epoch anchored to monotonic clock (no mid-run jumps) 
_BASE_EPOCH_NS = time.time_ns()          # UTC epoch ns at start
_BASE_PERF_NS  = time.perf_counter_ns()  # monotonic reference ns

def now_epoch_ns() -> int:
    return _BASE_EPOCH_NS + (time.perf_counter_ns() - _BASE_PERF_NS)

def iso8601_from_ns(ns: int) -> str:
    sec, rem = divmod(ns, 1_000_000_000)
    t = dt.datetime.fromtimestamp(sec, tz=dt.timezone.utc)
    return f"{t.strftime('%Y-%m-%dT%H:%M:%S')}.{rem:09d}Z"

def qr_numpy(payload: str, ecl: str = QR_ECL, border: int = QR_BORDER, scale: int = QR_SCALE) -> np.ndarray:
    """Fast QR render to grayscale NumPy (0=black, 255=white), no PNG encode/decode."""
    q = segno.make(payload, error=ecl, micro=False, boost_error=False)
    mat = np.array([[0 if cell else 255 for cell in row] for row in q.matrix], dtype=np.uint8)
    if border > 0:
        mat = np.pad(mat, pad_width=border, mode='constant', constant_values=255)
    if scale > 1:
        mat = cv2.resize(mat, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    return mat

def main():
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, WIDTH, HEIGHT)

    # FPS tracking
    last_tick = time.perf_counter_ns()
    frames = 0
    fps_display = 0.0
    fps_update_interval = 0.25
    fps_accum_time = 0.0

    # Optional ISO throttle
    last_iso_sec = None
    iso_text = ""

    saved_sample = False

    while True:
        loop_start = time.perf_counter_ns()

        epoch_ns = now_epoch_ns()
        sec = epoch_ns // 1_000_000_000
        if sec != last_iso_sec:
            last_iso_sec = sec
            iso_text = iso8601_from_ns(epoch_ns)

        img = np.full((HEIGHT, WIDTH, 3), BG_COLOR, dtype=np.uint8)

        cv2.putText(img, "UTC Epoch (ns):", (40, 120), FONT, 1.2, ACCENT, 2, cv2.LINE_AA)
        cv2.putText(img, str(epoch_ns), (40, 210), FONT, 2.2, FG_COLOR, 3, cv2.LINE_AA)

        cv2.putText(img, "ISO-8601 (UTC):", (40, 300), FONT, 1.2, ACCENT, 2, cv2.LINE_AA)
        cv2.putText(img, iso_text, (40, 370), FONT, 1.6, FG_COLOR, 2, cv2.LINE_AA)

        # FAST QR each frame 
        qr_gray = qr_numpy(str(epoch_ns))  # fresh code per frame
        qh, qw = qr_gray.shape
        x0 = WIDTH - qw - 60
        y0 = (HEIGHT - qh) // 2
        # Paste grayscale into BGR without allocating temp
        roi = img[y0:y0+qh, x0:x0+qw]
        roi[:, :, 0] = qr_gray
        roi[:, :, 1] = qr_gray
        roi[:, :, 2] = qr_gray
        cv2.rectangle(img, (x0-2, y0-2), (x0+qw+2, y0+qh+2), (0, 255, 255), 2)
        cv2.putText(img, "QR: epoch_ns (UTC)", (x0, y0 - 14), FONT, 0.9, ACCENT, 2, cv2.LINE_AA)

        # FPS display
        now_tick = time.perf_counter_ns()
        dt_sec = (now_tick - last_tick) / 1e9
        last_tick = now_tick
        frames += 1
        fps_accum_time += dt_sec
        if fps_accum_time >= fps_update_interval:
            fps_display = frames / fps_accum_time
            frames = 0
            fps_accum_time = 0.0

        cv2.putText(img, f"{fps_display:5.1f} FPS (target {TARGET_FPS:.0f})  |  press 'q' to quit",
                    (40, HEIGHT - 40), FONT, 1.0, (180, 255, 180), 2, cv2.LINE_AA)

        cv2.imshow(WINDOW_NAME, img)

        if not saved_sample:
            cv2.imwrite("overlay_sample.png", img)
            saved_sample = True

        # Pace for 120 FPS
        elapsed = time.perf_counter_ns() - loop_start
        rem = FRAME_NS - elapsed
        if rem > 0:
            time.sleep(rem / 1e9)

        if (cv2.waitKey(1) & 0xFF) in (ord('q'), 27):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
