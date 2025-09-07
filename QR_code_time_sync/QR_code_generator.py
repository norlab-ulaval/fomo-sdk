#!/usr/bin/env python3
# Full-screen overlay showing UTC epoch time in nanoseconds at ~120 FPS
# with an ISO-8601 (UTC) string and a QR code encoding the epoch_ns.

import time
import datetime as dt
import cv2
import numpy as np
import io
import segno  # pip install segno

WINDOW_NAME = "Epoch Time @ 120 FPS"
WIDTH, HEIGHT = 1280, 720  # change window size as needed but it will affect fps
BG_COLOR = (18, 18, 18)
FG_COLOR = (240, 240, 240)
ACCENT = (140, 200, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX

TARGET_FPS = 120.0
FRAME_NS = int(1e9 / TARGET_FPS)

# QR code settings
QR_SIZE = 320
QR_BORDER = 4
QR_ECL = "H"

# NOTE: UTC epoch (ns) anchored to a monotonic clock to avoid mid-run jumps 
_BASE_EPOCH_NS = time.time_ns()          # UTC epoch (nanoseconds) at start
_BASE_PERF_NS  = time.perf_counter_ns()  # monotonic reference (nanoseconds)

def now_epoch_ns() -> int:
    """UTC epoch time (ns) that won't jump if system clock adjusts mid-run."""
    return _BASE_EPOCH_NS + (time.perf_counter_ns() - _BASE_PERF_NS)

def iso8601_from_ns(ns: int) -> str:
    """ISO-8601 UTC string with 9-digit fractional seconds (ns)."""
    sec = ns // 1_000_000_000
    rem_ns = ns % 1_000_000_000
    t = dt.datetime.fromtimestamp(sec, tz=dt.timezone.utc)
    return f"{t.strftime('%Y-%m-%dT%H:%M:%S')}.{rem_ns:09d}Z"

def make_qr_image(payload: str, size_px: int = QR_SIZE, border: int = QR_BORDER, ecl: str = QR_ECL):
    """Generate a crisp QR image as grayscale uint8 using segno."""
    qrobj = segno.make(payload, error=ecl)
    buf = io.BytesIO()
    scale = max(1, size_px // 60)
    qrobj.save(buf, kind="png", scale=scale, border=border, dark="black", light="white")
    buf.seek(0)
    data = np.frombuffer(buf.read(), dtype=np.uint8)
    qr = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    qr = cv2.resize(qr, (size_px, size_px), interpolation=cv2.INTER_NEAREST)
    return qr

def main():
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, WIDTH, HEIGHT)

    # I need to include FPS tracking (smoothed, ~4x/sec)
    last_tick = time.perf_counter_ns()
    frames = 0
    fps_display = 0.0
    fps_update_interval = 0.25  # seconds
    fps_accum_time = 0.0

    saved_sample = False

    while True:
        loop_start = time.perf_counter_ns()

        # UTC epoch (ns), stable within the session
        epoch_ns = now_epoch_ns()
        iso = iso8601_from_ns(epoch_ns)

        img = np.full((HEIGHT, WIDTH, 3), BG_COLOR, dtype=np.uint8)

        # Labels and values
        cv2.putText(img, "UTC Epoch (ns):", (30, 80), FONT, 1.0, ACCENT, 2, cv2.LINE_AA)
        cv2.putText(img, str(epoch_ns), (30, 160), FONT, 2.0, FG_COLOR, 3, cv2.LINE_AA)

        cv2.putText(img, "ISO-8601 (UTC):", (30, 230), FONT, 1.0, ACCENT, 2, cv2.LINE_AA)
        cv2.putText(img, iso, (30, 300), FONT, 1.4, FG_COLOR, 2, cv2.LINE_AA)

        # QR overlay on right (encodes epoch_ns as string) 
        qr = make_qr_image(str(epoch_ns))
        qr_bgr = cv2.cvtColor(qr, cv2.COLOR_GRAY2BGR)
        x0 = WIDTH - QR_SIZE - 40
        y0 = (HEIGHT - QR_SIZE) // 2
        img[y0:y0+QR_SIZE, x0:x0+QR_SIZE] = qr_bgr
        cv2.rectangle(img, (x0-2, y0-2), (x0+QR_SIZE+2, y0+QR_SIZE+2), (0, 255, 255), 2)
        cv2.putText(img, "QR: epoch_ns (UTC)", (x0, y0 - 14), FONT, 0.8, ACCENT, 2, cv2.LINE_AA)

        # FPS compute & draw 
        now_tick = time.perf_counter_ns()
        dt_sec = (now_tick - last_tick) / 1e9
        last_tick = now_tick
        frames += 1
        fps_accum_time += dt_sec
        if fps_accum_time >= fps_update_interval:
            fps_display = frames / fps_accum_time
            frames = 0
            fps_accum_time = 0.0

        cv2.putText(
            img,
            f"{fps_display:5.1f} FPS (target {TARGET_FPS:.0f})  |  press 'q' to quit",
            (30, HEIGHT - 40),
            FONT, 0.9, (180, 255, 180), 2, cv2.LINE_AA
        )

        cv2.imshow(WINDOW_NAME, img)

        # Save a single sample frame (PNG) for testing your reader
        if not saved_sample:
            cv2.imwrite("overlay_sample.png", img)
            print("Saved overlay_sample.png (with QR).")
            saved_sample = True

        # --- pace toward target frame time ---
        elapsed = time.perf_counter_ns() - loop_start
        remaining_ns = FRAME_NS - elapsed
        if remaining_ns > 0:
            time.sleep(remaining_ns / 1e9)

        if (cv2.waitKey(1) & 0xFF) in (ord('q'), 27):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
