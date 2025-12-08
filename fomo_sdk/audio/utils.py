import logging
import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from rosbags.typesys.stores.latest import (
    builtin_interfaces__msg__Time as Time,
)
from rosbags.typesys.stores.latest import (
    sensor_msgs__msg__Image as Image,
)
from rosbags.typesys.stores.latest import (
    std_msgs__msg__Header as Header,
)
from scipy.io import wavfile
from typing_extensions import OrderedDict

import fomo_sdk.common.naming as naming
import fomo_sdk.common.utils as utils

MIC_LEFT_TOPIC = "/audio/left_mic"
MIC_RIGHT_TOPIC = "/audio/right_mic"
SAMPLE_RATE = 44100


class Stereo:
    def __init__(self) -> None:
        self.left = []
        self.right = []
        self.left_timestamps = []
        self.right_timestamps = []
        self.first_timestamp = 0
        self.first_message = {"right": True, "left": True}

        self.sample_rate = SAMPLE_RATE
        self.stereo = np.array([[], []])

        self.logger = logging.getLogger(__name__)
        logging.basicConfig(filename="fomo_audio.log", level=logging.INFO)

    def is_mic_topic(self, topic: str):
        return topic == MIC_RIGHT_TOPIC or topic == MIC_LEFT_TOPIC

    def add_message(self, connection, timestamp, rawdata, typestore):
        if connection.topic == MIC_LEFT_TOPIC or connection.topic == MIC_RIGHT_TOPIC:
            msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
            timestamp = utils.ros_timestamp_to_seconds(msg.header.stamp)
            if self.first_timestamp == 0:
                self.first_timestamp = timestamp
            timestamp -= self.first_timestamp
            audio_data = np.frombuffer(msg.audio.data, dtype=np.int16)
            timestamps = timestamp + np.linspace(
                0, len(audio_data) / self.sample_rate, len(audio_data)
            )
            if connection.topic == MIC_LEFT_TOPIC:
                self.left.append(audio_data)
                self.left_timestamps.append(timestamps)
                self.first_message["left"] = False
            elif connection.topic == MIC_RIGHT_TOPIC:
                self.right.append(audio_data)
                self.right_timestamps.append(timestamps)
                self.first_message["right"] = False

    def postprocess_audio_data(self):
        if type(self.left_timestamps) is list:
            # Combine and align timestamps and audio data
            self.left_timestamps, self.right_timestamps = map(
                lambda x: np.concatenate(x, axis=0),
                [self.left_timestamps, self.right_timestamps],
            )
            self.left, self.right = map(
                lambda x: np.concatenate(x, axis=0), [self.left, self.right]
            )

        # Crop the beginning and the end to match lengths
        min_timestamp = max(self.left_timestamps[0], self.right_timestamps[0])
        max_timestamp = min(self.left_timestamps[-1], self.right_timestamps[-1])

        def crop_to_range(data, timestamps, min_ts, max_ts):
            mask = (timestamps >= min_ts) & (timestamps <= max_ts)
            return data[mask], timestamps[mask]

        self.left, self.left_timestamps = crop_to_range(
            self.left, self.left_timestamps, min_timestamp, max_timestamp
        )
        self.right, self.right_timestamps = crop_to_range(
            self.right, self.right_timestamps, min_timestamp, max_timestamp
        )
        if len(self.left) != len(self.right):
            print(
                f"Warning: Left and right audio data have different lengths. Left - right = {len(self.left) - len(self.right)} samples"
            )

            if len(self.left) < len(self.right):
                self.right_timestamps = self.right_timestamps[: len(self.left)]
                self.right = self.right[: len(self.left)]
            else:
                self.left_timestamps = self.left_timestamps[: len(self.right)]
                self.left = self.left[: len(self.right)]
        self.stereo = np.vstack((self.left, self.right))
        self.stereo = self.stereo.transpose()

    def save_spectrogram(self):
        self.postprocess_audio_data()

        def save_figure(samples: np.ndarray, fig_name: str):
            plt.figure()
            plt.specgram(
                samples, Fs=self.sample_rate, NFFT=1024, noverlap=512, cmap="viridis"
            )
            plt.ylabel("Frequency [Hz]")
            plt.xlabel("Time [sec]")
            plt.title("Right channel")
            plt.savefig(fig_name, bbox_inches="tight")

        save_figure(self.left, "/tmp/left_channel.png")
        save_figure(self.right, "/tmp/right_channel.png")

    def save_audio(self, output: str, overwrite: bool):
        if self.left is None or self.right is None or self.stereo is None:
            print("No audio data to save.")
            return

        if os.path.exists(output) and overwrite:
            shutil.rmtree(output)

        try:
            if not output.endswith(".wav"):
                print("Only wav format is supported")
                return
            # Write the audio data to a WAV file
            filename = Path(output).stem
            wavfile.write(
                output.replace(filename, f"{filename}_left"),
                self.sample_rate,
                self.left,
            )
            wavfile.write(
                output.replace(filename, f"{filename}_right"),
                self.sample_rate,
                self.right,
            )
            wavfile.write(
                output.replace(filename, f"{filename}_stereo"),
                self.sample_rate,
                self.stereo,
            )
        except Exception as e:
            print(f"Error saving WAV file: {e}")
            return

        print(f"WAV file saved to {output}")

    def load_channel(self, path: Path):
        timestamps_samples = {}
        for wav in list(path.glob("*.wav")):
            sample_rate, samples = wavfile.read(wav)
            assert sample_rate == len(samples)
            timestamps = np.linspace(
                int(wav.stem),
                int(wav.stem) + 1_000_000 * len(samples) / self.sample_rate,
                len(samples),
            )  # in microseconds
            timestamps_samples[int(wav.stem)] = (timestamps, samples)
        timestamps_samples = OrderedDict(sorted(timestamps_samples.items()))
        all_samples, all_timestamps = [], []
        for timestamps, samples in timestamps_samples.values():
            all_samples.append(samples)
            all_timestamps.append(timestamps)
        all_samples = np.array(all_samples).flatten()
        all_timestamps = np.array(all_timestamps).flatten()
        return all_samples, all_timestamps

    def load_fomo_audio(self, dataset_base_path: str, deployment: str, trajectory: str):
        path = naming.construct_path(dataset_base_path, deployment, trajectory)

        if list(path.glob("*/.mcap")):
            raise NotImplementedError("Can't load audio data from mcap files yet")

        has_left_channel = len(list(path.glob("audio_left/"))) > 0
        has_right_channel = len(list(path.glob("audio_right/"))) > 0

        if has_left_channel:
            self.logger.info("Loading left channel")
            self.left, self.left_timestamps = self.load_channel(path / "audio_left")
        if has_right_channel:
            self.logger.info("Loading left channel")
            self.right, self.right_timestamps = self.load_channel(path / "audio_right")

        self.postprocess_audio_data()

    def load_audio(self, input: str):
        self.sample_rate, samples = wavfile.read("test_stereo.wav")
        if samples.shape[1] != 2:
            print("Only stereo audio is supported")
            return
        self.left = samples[:, 0]
        self.right = samples[:, 1]

        self.right_timestamps = np.linspace(
            0, len(self.right) / self.sample_rate, len(self.right)
        )
        self.left_timestamps = np.linspace(
            0, len(self.left) / self.sample_rate, len(self.left)
        )
        self.postprocess_audio_data()

    def create_spectogram(
        self, output="/tmp/spectogram.png", show: bool = False, save=False
    ) -> np.ndarray:
        fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
        ax[0].specgram(
            self.left, Fs=self.sample_rate, NFFT=1024, noverlap=512, cmap="viridis"
        )
        ax[0].set_ylabel("Frequency [Hz]")
        ax[0].set_xlabel("Time [sec]")
        ax[0].set_title("Left channel")

        ax[1].specgram(
            self.right, Fs=self.sample_rate, NFFT=1024, noverlap=512, cmap="viridis"
        )
        ax[1].set_xlabel("Time [sec]")
        ax[1].set_title("Right channel")
        if save:
            plt.savefig(output, bbox_inches="tight")
        if show:
            plt.show()

        canvas = FigureCanvas(fig)
        canvas.draw()  # Render the figure
        image = np.frombuffer(canvas.tostring_rgb(), dtype="uint8")
        image = image.reshape(
            canvas.get_width_height()[::-1] + (3,)
        )  # Reshape to (height, width, RGB)

        # Close the figure to release memory
        plt.close(fig)
        return image

    def write_spectogram_image(self, writer, timestamp: int, typestore):
        image = self.create_spectogram()
        image_flat = image.astype(np.uint8).flatten()
        height, width, colors = image.shape
        sec, nsec = utils.seconds_to_ros_timestamp(timestamp / 1e9)
        header = Header(Time(sec, nsec), "audio")
        image_msg = Image(
            header,
            height,
            width,
            "rgb8",  # "mono8" for 8-bit grayscale
            is_bigendian=False,
            step=3 * width,
            data=np.array(image_flat, dtype=np.uint8),
        )

        spectogram_image_connection = writer.add_connection(
            "/audio/spectogram",
            Image.__msgtype__,
            typestore=typestore,
        )

        writer.write(
            spectogram_image_connection,
            timestamp,
            typestore.serialize_cdr(image_msg, image_msg.__msgtype__),
        )
