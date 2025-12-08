"""
Implements a wrapper around mcap.Writer that behaves like rosbags.Writer.
Code inspired and taken from mcap_ros2: https://mcap.dev/docs/python/mcap-ros2-apidoc/
and rosbags: https://gitlab.com/ternaris/rosbags
"""

import time
from typing import Any, Dict, Optional
from rosbags.rosbag2.metadata import dump_qos_v8, dump_qos_v9, parse_qos
from rosbags.interfaces import Qos
from rosbags.typesys.store import Typestore
from collections.abc import Sequence
import yaml
from enum import Enum

import mcap
from mcap.exceptions import McapError
from mcap.records import Channel, Schema
from mcap.well_known import SchemaEncoding
from mcap.writer import CompressionType as McapCompressionType
from mcap.writer import Writer as McapWriter

from mcap_ros2 import __version__
from mcap_ros2._dynamic import EncoderFunction, serialize_dynamic

from pathlib import Path
import sys

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self


class McapROS2WriteError(McapError):
    """Raised if a ROS2 message cannot be encoded to CDR with a given schema."""

    pass


def _library_identifier():
    mcap_version = getattr(mcap, "__version__", "<=0.0.10")
    return f"mcap-ros2-support {__version__}; mcap {mcap_version}"


class Writer:
    class CompressionFormat(Enum):
        NONE = McapCompressionType.NONE
        LZ4 = McapCompressionType.LZ4
        ZSTD = McapCompressionType.ZSTD

    class CompressionMode(Enum):
        MESSAGE = "message"
        NONE = ""

    def __init__(
        self,
        output: str | Path,
        version: int = 8,
        chunk_size: int = 1024 * 1024,
        compression: CompressionFormat = CompressionFormat.NONE,
        enable_crcs: bool = True,
    ):
        self.path = Path(output)
        self.chunk_size = chunk_size
        self.enable_crcs = enable_crcs
        self.metadata_path = self.path / "metadata.yaml"
        self.mcap_path = self.path / f"{self.path.name}.mcap"
        self.version = version
        self._encoders: Dict[int, EncoderFunction] = {}
        self._channel_ids: Dict[str, int] = {}
        self._finished = False
        self.qos = {}
        self.hashes = {}
        self.is_open = False
        if compression == self.CompressionFormat.NONE:
            compression_mode = self.CompressionMode.NONE
        else:
            compression_mode = self.CompressionMode.MESSAGE
        self.set_compression(compression_mode, compression)

    def set_compression(
        self, mode: CompressionMode, compression_format: CompressionFormat
    ):
        """Set the compression mode and type for the MCAP stream.

        Args:
            mode: Either 'file' or 'message'.
            compression: The compression type to use.

        Raises:
            ValueError: If the mode or compression type is invalid.
        """
        if self.is_open:
            raise ValueError(
                "Cannot set compression mode after opening the MCAP stream."
            )
        if mode not in self.CompressionMode:
            raise ValueError(
                f"Invalid compression mode: {mode}. Must be one of {self.CompressionMode}"
            )
        if compression_format not in self.CompressionFormat:
            raise ValueError(f"Invalid compression type: {compression_format}")
        self.compression_mode = mode
        self.compression_type = compression_format

    def open(self):
        """Open the MCAP stream for writing."""
        try:
            self.path.mkdir(mode=0o755, parents=True)
        except FileExistsError:
            raise FileNotFoundError(f"{self.path} exists already, not overwriting.")
        compression = McapCompressionType.NONE
        if self.compression_type == self.CompressionFormat.ZSTD:
            compression = McapCompressionType.ZSTD
        elif self.compression_type == self.CompressionFormat.LZ4:
            compression = McapCompressionType.LZ4
        self._writer = McapWriter(
            output=str(self.mcap_path),
            chunk_size=self.chunk_size,
            compression=compression,
            enable_crcs=self.enable_crcs,
        )
        self._writer.start(profile="ros2", library=_library_identifier())
        self.is_open = True

    def close(self):
        """Finishes writing to the MCAP stream. This must be called before the stream is closed."""
        if self._finished:
            return
        self._writer.finish()
        self._finished = True

        statistics = self._writer.__statistics
        start = statistics.message_start_time
        duration = statistics.message_end_time - start
        count = statistics.message_count

        dump_qos = dump_qos_v9 if self.version >= 9 else dump_qos_v8

        metadata = {
            "rosbag2_bagfile_information": {
                "version": self.version,
                "storage_identifier": "mcap",
                "relative_file_paths": [self.mcap_path.name],
                "duration": {"nanoseconds": duration},
                "starting_time": {"nanoseconds_since_epoch": start},
                "message_count": count,
                "topics_with_message_count": [
                    {
                        "topic_metadata": {
                            "name": x.topic,
                            "type": self._writer.__schemas[x.schema_id].name,
                            "serialization_format": x.message_encoding,
                            "offered_qos_profiles": dump_qos(self.qos[x.id]),
                            "type_description_hash": self.hashes[x.id],
                        },
                        "message_count": statistics.channel_message_counts[x.id],
                    }
                    for x in self._writer.__channels.values()
                ],
                "compression_format": "",
                "compression_mode": "",
                "files": [
                    {
                        "path": self.mcap_path.name,
                        "starting_time": {"nanoseconds_since_epoch": start},
                        "duration": {"nanoseconds": duration},
                        "message_count": count,
                    },
                ],
                # 'custom_data': self.custom_data, # FIXME custom data might be a nice place to store additional info about fomo trajectories
                "ros_distro": "fomo_sdk",
            },
        }

        with open(self.metadata_path, "w") as file:
            yaml.dump(metadata, file, default_flow_style=False, sort_keys=False)

    def add_connection(
        self,
        topic: str,
        msgtype: str,
        *,
        typestore: Typestore,
        msgdef: str | None = None,
        rihs01: str | None = None,
        serialization_format: str = "cdr",
        offered_qos_profiles: Sequence[Qos] | str = (),
    ) -> Channel:
        """Add a connection.

        This function can only be called after opening a bag.

        Args:
            topic: Topic name.
            msgtype: Message type.
            typestore: Typestore.
            msgdef: Message definiton.
            rihs01: Message hash.
            serialization_format: Serialization format.
            offered_qos_profiles: QOS Profile.

        Returns:
            Connection object.

        Raises:
            WriterError: Bag not open or topic previously registered.

        """
        if msgdef is None:
            msgdef, _ = typestore.generate_msgdef(msgtype, ros_version=2)
        assert msgdef is not None

        if isinstance(offered_qos_profiles, str):
            qos_profiles = parse_qos(offered_qos_profiles)
        else:
            qos_profiles = list(offered_qos_profiles)

        # Write a Schema record for a ROS2 message definition.
        msgdef_data = msgdef.encode()
        schema_id = self._writer.register_schema(
            msgtype, SchemaEncoding.ROS2, msgdef_data
        )

        channel_id = self._writer.register_channel(
            topic=topic,
            message_encoding="cdr",
            schema_id=schema_id,
        )
        self.qos[channel_id] = qos_profiles
        self.hashes[channel_id] = typestore.hash_rihs01(msgtype)
        self._channel_ids[topic] = channel_id
        return self._writer.__channels[channel_id]

    def write(self, channel: Channel, timestamp: int, data: bytes | memoryview):
        """ """
        topic = self._writer.__channels[channel.id].topic
        schema = self._writer.__schemas[channel.schema_id]
        self._write_message(
            topic, schema, data, log_time=timestamp, publish_time=timestamp
        )

    def _write_message(
        self,
        topic: str,
        schema: Schema,
        message: Any,
        log_time: Optional[int] = None,
        publish_time: Optional[int] = None,
        sequence: int = 0,
        serialize=True,
    ):
        """
        Write a ROS2 Message record, automatically registering a channel as needed.

        :param topic: The topic of the message.
        :param message: The message to write.
        :param log_time: The time at which the message was logged as a nanosecond UNIX timestamp.
            Will default to the current time if not specified.
        :param publish_time: The time at which the message was published as a nanosecond UNIX
            timestamp. Will default to ``log_time`` if not specified.
        :param sequence: An optional sequence number.
        """
        encoder = self._encoders.get(schema.id)
        if encoder is None:
            if schema.encoding != SchemaEncoding.ROS2:
                raise McapROS2WriteError(
                    f'can\'t parse schema with encoding "{schema.encoding}"'
                )
            type_dict = serialize_dynamic(  # type: ignore
                schema.name, schema.data.decode()
            )
            # Check if schema.name is in type_dict
            if schema.name not in type_dict:
                raise McapROS2WriteError(f'schema parsing failed for "{schema.name}"')
            encoder = type_dict[schema.name]
            self._encoders[schema.id] = encoder

        if topic not in self._channel_ids:
            raise ValueError("Unknown channel id (topic). Register it first")
        channel_id = self._channel_ids[topic]

        if type(message) is not bytes and type(message) is not memoryview:
            data = encoder(message)
        else:
            data = message

        if log_time is None:
            log_time = time.time_ns()
        if publish_time is None:
            publish_time = log_time
        self._writer.add_message(
            channel_id=channel_id,
            log_time=log_time,
            publish_time=publish_time,
            sequence=sequence,
            data=data,
        )

    def __enter__(self) -> Self:
        """Context manager support."""
        self.open()
        return self

    def __exit__(self, exc_: Any, exc_type_: Any, tb_: Any):
        """Call finish() on exit."""
        self.close()
