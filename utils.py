from datetime import datetime
import numpy as np
import cv2
import sys

ENCODINGS = {
    "rgb8": (np.uint8, 3),
    "rgba8": (np.uint8, 4),
    "rgb16": (np.uint16, 3),
    "rgba16": (np.uint16, 4),
    "bgr8": (np.uint8, 3),
    "bgra8": (np.uint8, 4),
    "bgr16": (np.uint16, 3),
    "bgra16": (np.uint16, 4),
    "mono8": (np.uint8, 1),
    "mono16": (np.uint16, 1),
    "bayer_rggb8": (np.uint8, 1),
    "bayer_bggr8": (np.uint8, 1),
    "bayer_gbrg8": (np.uint8, 1),
    "bayer_grbg8": (np.uint8, 1),
    "bayer_rggb16": (np.uint16, 1),
    "bayer_bggr16": (np.uint16, 1),
    "bayer_gbrg16": (np.uint16, 1),
    "bayer_grbg16": (np.uint16, 1),
}


changeovertime = 1627387200 * 1e9
def get_num_times(bag, topics):
    times = [t for topic, msg, t in bag.read_messages(topics)]
    return len(times)

def get_start_week(rostime, gpstime):
    start_epoch = rostime * 1e-9
    dt = datetime.fromtimestamp(start_epoch)
    weekday = dt.isoweekday()
    if weekday == 7:
        weekday = 0  # Sunday
    g2 = weekday * 24 * 3600 + dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond * 1e-6
    start_week = round(start_epoch - g2)
    hour_offset = round((gpstime - g2) / 3600)
    time_zone_offset = hour_offset * 3600.0        # Toronto time is GMT-4 or GMT-5 depending on time of year
    print('START WEEK: {} TIME ZONE OFFSET: {}'.format(start_week, time_zone_offset))
    return start_week, time_zone_offset


def rosimg_to_cv2(msg, desired="bgr"):
    h, w = msg.height, msg.width
    enc = msg.encoding.lower()
    big_endian = bool(msg.is_bigendian)
    step = int(msg.step)

    if enc.endswith("16"):
        dtype = np.dtype(">u2" if big_endian else "<u2")
    else:
        dtype = np.dtype(np.uint8)   

    buf = np.frombuffer(msg.data, dtype=dtype)

    if dtype.byteorder in (">", "<"):
        need_swap = ((dtype.byteorder == ">" and sys.byteorder == "little") or
                     (dtype.byteorder == "<" and sys.byteorder == "big"))
        if need_swap:
            buf = buf.byteswap().newbyteorder()

    row_elems = step // dtype.itemsize  
    img2d = buf.reshape(h, row_elems)[:, :w]

    if enc in ("mono8", "8uc1"):
        img = img2d
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if desired == "bgr" else img
    if enc in ("mono16", "16uc1"):
        img = img2d
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if desired == "bgr" else img

    if enc == "bgr8":
        img = img2d.reshape(h, w, 3)
        return img if desired != "rgb" else img[:, :, ::-1]
    if enc == "rgb8":
        img = img2d.reshape(h, w, 3)
        return img[:, :, ::-1] if desired == "bgr" else img

    bayer2cv = {
        "bayer_bggr8":  cv2.COLOR_BAYER_BG2BGR_EA,
        "bayer_gbrg8":  cv2.COLOR_BAYER_GB2BGR_EA,
        "bayer_grbg8":  cv2.COLOR_BAYER_GR2BGR_EA,
        "bayer_rggb8":  cv2.COLOR_BAYER_RG2BGR_EA,
        "bayer_bggr16": cv2.COLOR_BAYER_BG2BGR_EA,
        "bayer_gbrg16": cv2.COLOR_BAYER_GB2BGR_EA,
        "bayer_grbg16": cv2.COLOR_BAYER_GR2BGR_EA,
        "bayer_rggb16": cv2.COLOR_BAYER_RG2BGR_EA,
    }
    if enc in bayer2cv:
        img = cv2.cvtColor(img2d, bayer2cv[enc])
        return img if desired != "rgb" else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    raise ValueError(f"Unsupported encoding: {msg.encoding}")

# fomo-sdk version of image_to_numpy
def image_to_numpy(msg):
    # Taken from https://github.com/eric-wieser/ros_numpy

    if not msg.encoding in ENCODINGS:
        raise TypeError("Unrecognized encoding {}".format(msg.encoding))

    dtype_class, channels = ENCODINGS[msg.encoding]
    dtype = np.dtype(dtype_class)
    dtype = dtype.newbyteorder(">" if msg.is_bigendian else "<")
    shape = (msg.height, msg.width, channels)

    data = np.frombuffer(msg.data, dtype=dtype).reshape(shape)
    data.strides = (msg.step, dtype.itemsize * channels, dtype.itemsize)

    if "rgb" in msg.encoding:
        data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
    if channels == 1:
        data = data[..., 0]
    return data
