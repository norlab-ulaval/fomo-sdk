import argparse
from functools import lru_cache
from typing import List

import cv2
import numpy as np
import torch
import torchvision
from moviepy.editor import ImageSequenceClip
from moviepy.video.io.VideoFileClip import VideoFileClip
from pathlib import Path

IMG_EXTS = [".jpg", ".jpeg", ".png"]


@lru_cache
def get_device() -> str:
    """
    Return the device type
    """
    return (
        "cpu"
        if not torch.cuda.is_available()
        else f"cuda:{torch.cuda.current_device()}"
    )


def load_models(face_model_path, lp_model_path):
    device = get_device()

    if face_model_path is not None:
        face_detector = torch.jit.load(face_model_path, map_location="cpu").to(device)
        face_detector.eval()
    else:
        face_detector = None

    if lp_model_path is not None:
        lp_detector = torch.jit.load(lp_model_path, map_location="cpu").to(device)
        lp_detector.eval()
    else:
        lp_detector = None

    return face_detector, lp_detector


def read_image(image_path: str) -> np.ndarray:
    """
    parameter image_path: absolute path to an image
    Return an image in BGR format
    """
    bgr_image = cv2.imread(image_path)
    if len(bgr_image.shape) == 2:
        bgr_image = cv2.cvtColor(bgr_image, cv2.COLOR_GRAY2BGR)
    return bgr_image


def write_image(image: np.ndarray, image_path: str) -> None:
    """
    parameter image: np.ndarray in BGR format
    parameter image_path: absolute path where we want to save the visualized image
    """
    cv2.imwrite(image_path, image)


def get_image_tensor(bgr_image: np.ndarray) -> torch.Tensor:
    """
    parameter bgr_image: image on which we want to make detections

    Return the image tensor
    """
    bgr_image_transposed = np.transpose(bgr_image, (2, 0, 1))
    image_tensor = torch.from_numpy(bgr_image_transposed).to(get_device())

    return image_tensor


def get_detections(
    detector: torch.jit._script.RecursiveScriptModule,
    image_tensor: torch.Tensor,
    model_score_threshold: float,
    nms_iou_threshold: float,
) -> List[List[float]]:
    """
    parameter detector: Torchscript module to perform detections
    parameter image_tensor: image tensor on which we want to make detections
    parameter model_score_threshold: model score threshold to filter out low confidence detection
    parameter nms_iou_threshold: NMS iou threshold to filter out low confidence overlapping boxes

    Returns the list of detections
    """
    with torch.no_grad():
        detections = detector(image_tensor)

    boxes, _, scores, _ = detections  # returns boxes, labels, scores, dims

    nms_keep_idx = torchvision.ops.nms(boxes, scores, nms_iou_threshold)
    boxes = boxes[nms_keep_idx]
    scores = scores[nms_keep_idx]

    boxes = boxes.cpu().numpy()
    scores = scores.cpu().numpy()

    score_keep_idx = np.where(scores > model_score_threshold)[0]
    boxes = boxes[score_keep_idx]
    return boxes.tolist()


def scale_box(
    box: List[List[float]], max_width: int, max_height: int, scale: float
) -> List[List[float]]:
    """
    parameter box: detection box in format (x1, y1, x2, y2)
    parameter scale: scaling factor

    Returns a scaled bbox as (x1, y1, x2, y2)
    """
    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
    w = x2 - x1
    h = y2 - y1

    xc = x1 + w / 2
    yc = y1 + h / 2

    w = scale * w
    h = scale * h

    x1 = max(xc - w / 2, 0)
    y1 = max(yc - h / 2, 0)

    x2 = min(xc + w / 2, max_width)
    y2 = min(yc + h / 2, max_height)

    return [x1, y1, x2, y2]


def visualize(
    image: np.ndarray,
    detections: List[List[float]],
    scale_factor_detections: float,
) -> np.ndarray:
    """
    parameter image: image on which we want to make detections
    parameter detections: list of bounding boxes in format [x1, y1, x2, y2]
    parameter scale_factor_detections: scale detections by the given factor to allow blurring more area, 1.15 would mean 15% scaling

    Visualize the input image with the detections and save the output image at the given path
    """
    image_fg = image.copy()
    mask_shape = (image.shape[0], image.shape[1], 1)
    mask = np.full(mask_shape, 0, dtype=np.uint8)

    for box in detections:
        if scale_factor_detections != 1.0:
            box = scale_box(
                box, image.shape[1], image.shape[0], scale_factor_detections
            )
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        w = x2 - x1
        h = y2 - y1

        ksize = (image.shape[0] // 2, image.shape[1] // 2)
        image_fg[y1:y2, x1:x2] = cv2.blur(image_fg[y1:y2, x1:x2], ksize)
        cv2.ellipse(mask, (((x1 + x2) // 2, (y1 + y2) // 2), (w, h), 0), 255, -1)

    inverse_mask = cv2.bitwise_not(mask)
    image_bg = cv2.bitwise_and(image, image, mask=inverse_mask)
    image_fg = cv2.bitwise_and(image_fg, image_fg, mask=mask)
    image = cv2.add(image_bg, image_fg)

    return image


def visualize_image(
    input_image_path: str,
    face_detector: torch.jit._script.RecursiveScriptModule,
    lp_detector: torch.jit._script.RecursiveScriptModule,
    face_model_score_threshold: float,
    lp_model_score_threshold: float,
    nms_iou_threshold: float,
    scale_factor_detections: float,
    ouput_path,
):
    """
    parameter input_image_path: absolute path to the input image
    parameter face_detector: face detector model to perform face detections
    parameter lp_detector: face detector model to perform face detections
    parameter face_model_score_threshold: face model score threshold to filter out low confidence detection
    parameter lp_model_score_threshold: license plate model score threshold to filter out low confidence detection
    parameter nms_iou_threshold: NMS iou threshold
    parameter output_image_path: absolute path where the visualized image will be saved
    parameter scale_factor_detections: scale detections by the given factor to allow blurring more area

    Perform detections on the input image and save the output image at the given path.
    """
    bgr_image = read_image(input_image_path)
    image = bgr_image.copy()

    image_tensor = get_image_tensor(bgr_image)
    image_tensor_copy = image_tensor.clone()
    detections = []
    # get face detections
    if face_detector is not None:
        detections.extend(
            get_detections(
                face_detector,
                image_tensor,
                face_model_score_threshold,
                nms_iou_threshold,
            )
        )

    # get license plate detections
    if lp_detector is not None:
        detections.extend(
            get_detections(
                lp_detector,
                image_tensor_copy,
                lp_model_score_threshold,
                nms_iou_threshold,
            )
        )

    if len(detections) != 0:
        image = visualize(
            image,
            detections,
            scale_factor_detections,
        )
        write_image(image, ouput_path)


def visualize_video(
    input_video_path: str,
    face_detector: torch.jit._script.RecursiveScriptModule,
    lp_detector: torch.jit._script.RecursiveScriptModule,
    face_model_score_threshold: float,
    lp_model_score_threshold: float,
    nms_iou_threshold: float,
    output_video_path: str,
    scale_factor_detections: float,
    output_video_fps: int,
):
    """
    parameter input_video_path: absolute path to the input video
    parameter face_detector: face detector model to perform face detections
    parameter lp_detector: face detector model to perform face detections
    parameter face_model_score_threshold: face model score threshold to filter out low confidence detection
    parameter lp_model_score_threshold: license plate model score threshold to filter out low confidence detection
    parameter nms_iou_threshold: NMS iou threshold
    parameter output_video_path: absolute path where the visualized video will be saved
    parameter scale_factor_detections: scale detections by the given factor to allow blurring more area
    parameter output_video_fps: fps of the visualized video

    Perform detections on the input video and save the output video at the given path.
    """
    visualized_images = []
    video_reader_clip = VideoFileClip(input_video_path)
    for frame in video_reader_clip.iter_frames():
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        image = frame.copy()
        bgr_image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        image_tensor = get_image_tensor(bgr_image)
        image_tensor_copy = image_tensor.clone()
        detections = []
        # get face detections
        if face_detector is not None:
            detections.extend(
                get_detections(
                    face_detector,
                    image_tensor,
                    face_model_score_threshold,
                    nms_iou_threshold,
                )
            )
        # get license plate detections
        if lp_detector is not None:
            detections.extend(
                get_detections(
                    lp_detector,
                    image_tensor_copy,
                    lp_model_score_threshold,
                    nms_iou_threshold,
                )
            )
        visualized_images.append(
            visualize(
                image,
                detections,
                scale_factor_detections,
            )
        )

    video_reader_clip.close()

    if visualized_images:
        video_writer_clip = ImageSequenceClip(visualized_images, fps=output_video_fps)
        video_writer_clip.write_videofile(output_video_path)
        video_writer_clip.close()


def process_file(face_detector, lp_detector, input_path: Path, output_path: Path):
    if output_path.exists():
        print(f"Skipping {input_path} because output already exists: {output_path}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    input_path_str = str(input_path)

    if input_path.suffix.lower() in IMG_EXTS and (
        "zedx" in input_path_str
        or "basler" in input_path_str
        or "media" in input_path_str
    ):
        print(f"Processing {input_path} -> {output_path}")
        visualize_image(
            str(input_path),
            face_detector,
            lp_detector,
            0.8,
            0.8,
            0.3,
            1,
            str(output_path),
        )

    elif input_path.suffix.lower() == ".mp4" and (
        "zedx" in input_path_str
        or "basler" in input_path_str
        or "media" in input_path_str
    ):
        print(f"Processing {input_path} -> {output_path}")
        cap = cv2.VideoCapture(str(input_path))
        fps = cap.get(cv2.CAP_PROP_FPS)

        visualize_video(
            str(input_path),
            face_detector,
            lp_detector,
            0.8,
            0.8,
            0.3,
            str(output_path),
            1,
            fps,
        )

    else:
        print(f"Skipping unsupported file: {input_path}")


def main(args):
    face_detector, lp_detector = load_models(args.face_model_path, args.lp_model_path)

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    for in_file in input_dir.rglob("*"):
        if in_file.suffix.lower() in IMG_EXTS + [".mp4"]:
            rel_path = in_file.relative_to(input_dir)
            out_file = output_dir / rel_path
            process_file(face_detector, lp_detector, in_file, out_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Recursive EgoBlur on images and videos."
    )
    parser.add_argument("--input", required=True, help="Input folder")
    parser.add_argument("--output", required=True, help="Output folder")
    parser.add_argument(
        "--face_model_path", default=None, help="Path to TorchScript face model"
    )
    parser.add_argument(
        "--lp_model_path", default=None, help="Path to TorchScript license plate model"
    )
    args = parser.parse_args()
    main(args)
