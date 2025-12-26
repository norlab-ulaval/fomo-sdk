use crate::fomo_rust_sdk::qos::create_tf_qos_metadata_string;

use super::rosbag::{TopicMetadata, TopicWithMessageCount, TopicWithMessageCountWithTimestamps};
use super::sensors::utils::{HasHeader, ToRosMsg, ToRosMsgWithInfo};
use super::utils::create_schema_channel;
use std::marker::PhantomData;

use anyhow::Result;
use camino::{Utf8Path, Utf8PathBuf};

use super::sensors::tf::{self, TFMessage};
use super::sensors::timestamp::{Timestamp, TimestampPrecision};
use std::fmt::Debug;

use super::qos::{
    create_sensor_qos_metadata, create_sensor_qos_metadata_string, create_tf_qos_metadata,
};
use crossbeam::channel::Sender;
use mcap;
use mcap::Writer;
use std::io::BufReader;
use std::{
    cmp,
    fs::{self, File},
    io::BufRead,
};
use std::{io::BufWriter, u64};

pub(super) struct McapLogMessage {
    pub channel_id: u16,
    pub sequence: u32,
    pub log_time: u64,
    pub publish_time: u64,
    pub data: Vec<u8>,
}

pub(super) trait DataLoader: Iterator {
    fn new<P: AsRef<Utf8Path>>(
        directory_path: P,
        extension: &str,
    ) -> Result<Self, Box<dyn std::error::Error>>
    where
        Self: Sized;
}

pub(super) struct DirectoryLoader {
    files: Vec<Utf8PathBuf>,
    pub(super) current_index: usize,
}

impl DataLoader for DirectoryLoader {
    fn new<P: AsRef<Utf8Path>>(
        directory_path: P,
        extension: &str,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let mut files = Vec::new();

        for entry in fs::read_dir(directory_path.as_ref())? {
            let entry = entry?;
            let path = entry.path();
            if path.is_file()
                && path
                    .extension()
                    .expect(
                        format!(
                            "Expected an extension `{}` but `{:?}` has none",
                            extension, path
                        )
                        .as_str(),
                    )
                    .eq(extension)
            {
                let utf8_buf = Utf8PathBuf::from_path_buf(path).unwrap();
                files.push(utf8_buf);
            }
        }

        files.sort();

        Ok(DirectoryLoader {
            files,
            current_index: 0,
        })
    }
}

impl Iterator for DirectoryLoader {
    type Item = Utf8PathBuf;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_index < self.files.len() {
            let file_path = &self.files[self.current_index];
            self.current_index += 1;

            Some(file_path.to_path_buf())
        } else {
            None
        }
    }
}

pub(super) struct CsvLoader {
    reader: BufReader<File>,
}

impl DataLoader for CsvLoader {
    fn new<P: AsRef<Utf8Path>>(
        path: P,
        extension: &str,
    ) -> Result<CsvLoader, Box<dyn std::error::Error>> {
        if path.as_ref().extension().unwrap().ne(extension) {
            return Err(format!(
                "The file path {} doesn't match the extension {}",
                path.as_ref(),
                extension
            )
            .into());
        }

        let file = File::open(path.as_ref())?;
        let mut reader = std::io::BufReader::new(file);

        // skip the header
        let mut buf = String::new();
        reader.read_line(&mut buf)?;

        Ok(Self { reader })
    }
}

impl Iterator for CsvLoader {
    type Item = String;

    fn next(&mut self) -> Option<Self::Item> {
        let mut line = String::new();

        match self.reader.read_line(&mut line) {
            Ok(0) => None, // EOF
            Ok(_) => {
                // Remove trailing newline
                if line.ends_with('\n') {
                    line.pop();
                    if line.ends_with('\r') {
                        line.pop();
                    }
                }
                Some(line)
            }
            Err(_) => todo!(),
        }
    }
}

pub(super) trait SensorMcapWriter: Iterator {
    type DataType: HasHeader;

    fn create_channels(
        &self,
        mcap_writer: &mut Writer<BufWriter<File>>,
    ) -> Result<Vec<u16>, Box<dyn std::error::Error>>;

    fn get_topic_metadatas(&self) -> Result<Vec<TopicMetadata>, Box<dyn std::error::Error>>;

    fn process_item(
        &self,
        item: &Self::Item,
        prec: &TimestampPrecision,
    ) -> Result<Self::DataType, Box<dyn std::error::Error>>;

    fn serialize_message(
        &self,
        data: Self::DataType,
        channels: &Vec<u16>,
        sequence: u32,
        timestamp: &Timestamp,
    ) -> Result<Vec<McapLogMessage>, Box<dyn std::error::Error>>;
    fn get_topic(&self) -> &str;
}

pub(super) fn write_tf_data<P: AsRef<Utf8Path>>(
    path: P,
    timestamp: u64,
    mcap_writer: &mut Writer<BufWriter<File>>,
    prec: &TimestampPrecision,
) -> Result<TopicWithMessageCount, Box<dyn std::error::Error>> {
    let filepath = path.as_ref().join("calib").join("transforms.json");
    let topic_name = "/tf_static";
    let schema_name = "tf2_msgs/msg/TFMessage";
    let message_count = 1;
    println!("Processing tf data. Saving to timestamp: {}", timestamp);
    let tf_message = TFMessage::from_file(&filepath, timestamp, prec)
        .map_err(|e| format!("Failed to read TF file '{}': {}", &filepath, e))?;

    let (_, channel_id) = create_schema_channel(
        mcap_writer,
        schema_name,
        tf::SCHEMA_DEF.as_bytes(),
        topic_name,
        &create_tf_qos_metadata(),
    )?;

    let sequence = 0;
    let msg_header = mcap::records::MessageHeader {
        channel_id,
        sequence,
        log_time: timestamp,
        publish_time: timestamp,
    };
    let mut buffer: Vec<u8> = Vec::new();
    tf::construct_msg(&tf_message, &mut buffer)?;
    mcap_writer
        .write_to_known_channel(&msg_header, &buffer)
        .unwrap();
    let topic_metadata = TopicMetadata::new(
        topic_name.to_string(),
        schema_name.to_string(),
        create_tf_qos_metadata_string(),
    );
    Ok(TopicWithMessageCount {
        topic_metadata,
        message_count,
    })
}

pub(super) struct MsgMcapWriter<T, L> {
    loader: L,
    topic: String,
    frame_id: String,
    _phantom: PhantomData<T>,
}

impl<T, L: DataLoader> MsgMcapWriter<T, L> {
    pub(super) fn new<P: AsRef<Utf8Path>>(
        file_path: P,
        topic: String,
        frame_id: String,
        extension: &str,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        // Todo check that path is extension
        let loader = L::new(file_path.as_ref().to_path_buf(), extension)?;
        Ok(Self {
            loader,
            topic,
            frame_id,
            _phantom: PhantomData,
        })
    }
}

impl<T: ToRosMsg<L::Item>, L: DataLoader> SensorMcapWriter for MsgMcapWriter<T, L> {
    type DataType = T;

    fn create_channels(
        &self,
        mcap_writer: &mut Writer<BufWriter<File>>,
    ) -> Result<Vec<u16>, Box<dyn std::error::Error>> {
        let (_, channel_id) = create_schema_channel(
            mcap_writer,
            Self::DataType::get_schema_name(),
            Self::DataType::get_schema_def(),
            &self.topic,
            &create_sensor_qos_metadata(),
        )?;
        Ok(vec![channel_id])
    }

    fn get_topic_metadatas(&self) -> Result<Vec<TopicMetadata>, Box<dyn std::error::Error>> {
        Ok(vec![TopicMetadata::new(
            self.topic.clone(),
            Self::DataType::get_schema_name().to_string(),
            create_sensor_qos_metadata_string(),
        )])
    }

    fn process_item(
        &self,
        item: &Self::Item,
        prec: &TimestampPrecision,
    ) -> Result<Self::DataType, Box<dyn std::error::Error>> {
        Self::DataType::from_item(item, self.frame_id.clone(), prec)
    }

    fn serialize_message(
        &self,
        data: Self::DataType,
        channels: &Vec<u16>,
        sequence: u32,
        timestamp: &Timestamp,
    ) -> Result<Vec<McapLogMessage>, Box<dyn std::error::Error>> {
        let mut buffer: Vec<u8> = Vec::new();
        Self::DataType::construct_msg(data, &mut buffer).unwrap();
        Ok(vec![McapLogMessage {
            channel_id: channels[0],
            sequence,
            log_time: timestamp.timestamp,
            publish_time: timestamp.timestamp,
            data: buffer,
        }])
    }

    fn get_topic(&self) -> &str {
        &self.topic
    }
}

impl<T, L: DataLoader> Iterator for MsgMcapWriter<T, L> {
    type Item = <L as Iterator>::Item;
    fn next(&mut self) -> Option<Self::Item> {
        self.loader.next()
    }
}

pub(super) struct MsgWithInfoMcapWriter<T, L> {
    loader: L,
    topic: String,
    frame_id: String,
    calib_path: Utf8PathBuf,
    _phantom: PhantomData<T>,
}

impl<T, L: DataLoader> MsgWithInfoMcapWriter<T, L> {
    pub(super) fn new<P: AsRef<Utf8Path>>(
        file_path: P,
        calib_path: P,
        topic: String,
        frame_id: String,
        extension: &str,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        // Todo check that path is extension
        let loader = L::new(file_path.as_ref().to_path_buf(), extension)?;
        Ok(Self {
            loader,
            topic,
            frame_id,
            calib_path: calib_path.as_ref().to_path_buf(),
            _phantom: PhantomData,
        })
    }
}

impl<T: ToRosMsgWithInfo<L::Item>, L: DataLoader> SensorMcapWriter for MsgWithInfoMcapWriter<T, L> {
    type DataType = T;

    fn create_channels(
        &self,
        mcap_writer: &mut Writer<BufWriter<File>>,
    ) -> Result<Vec<u16>, Box<dyn std::error::Error>> {
        let (_, channel_id_data) = create_schema_channel(
            mcap_writer,
            Self::DataType::get_schema_name(),
            Self::DataType::get_schema_def(),
            &format!("{}/{}", self.topic, Self::DataType::get_data_topic()),
            &create_sensor_qos_metadata(),
        )?;

        let (_, channel_id_info) = create_schema_channel(
            mcap_writer,
            Self::DataType::get_info_schema_name(),
            Self::DataType::get_info_schema_def(),
            &format!("{}/{}", self.topic, Self::DataType::get_info_topic()),
            &create_sensor_qos_metadata(),
        )?;
        Ok(vec![channel_id_data, channel_id_info])
    }

    fn get_topic_metadatas(&self) -> Result<Vec<TopicMetadata>, Box<dyn std::error::Error>> {
        let data_topic_metadata = TopicMetadata::new(
            format!("{}/{}", self.topic, Self::DataType::get_data_topic()),
            Self::DataType::get_schema_name().to_string(),
            create_sensor_qos_metadata_string(),
        );
        let info_topic_metadata = TopicMetadata::new(
            format!("{}/{}", self.topic, Self::DataType::get_info_topic()),
            Self::DataType::get_info_schema_name().to_string(),
            create_sensor_qos_metadata_string(),
        );
        Ok(vec![data_topic_metadata, info_topic_metadata])
    }

    fn process_item(
        &self,
        item: &Self::Item,
        prec: &TimestampPrecision,
    ) -> Result<Self::DataType, Box<dyn std::error::Error>> {
        Self::DataType::from_item(item, self.frame_id.clone(), prec)
    }

    fn serialize_message(
        &self,
        data: Self::DataType,
        channels: &Vec<u16>,
        sequence: u32,
        timestamp: &Timestamp,
    ) -> Result<Vec<McapLogMessage>, Box<dyn std::error::Error>> {
        let mut messages = Vec::new();
        let mut info_buffer: Vec<u8> = Vec::new();
        Self::DataType::construct_info_msg(&data, &mut info_buffer, self.calib_path.as_path())
            .unwrap();
        messages.push(McapLogMessage {
            channel_id: channels[1],
            sequence,
            log_time: timestamp.timestamp,
            publish_time: timestamp.timestamp,
            data: info_buffer,
        });

        let mut data_buffer: Vec<u8> = Vec::new();
        Self::DataType::construct_msg(data, &mut data_buffer).unwrap();
        messages.push(McapLogMessage {
            channel_id: channels[0],
            sequence,
            log_time: timestamp.timestamp,
            publish_time: timestamp.timestamp,
            data: data_buffer,
        });

        Ok(messages)
    }

    fn get_topic(&self) -> &str {
        &self.topic
    }
}

impl<T, L: DataLoader> Iterator for MsgWithInfoMcapWriter<T, L> {
    type Item = <L as Iterator>::Item;
    fn next(&mut self) -> Option<Self::Item> {
        self.loader.next()
    }
}

use indicatif::ProgressBar;
use rayon::prelude::*;

pub(super) fn produce_sensor_data<W: SensorMcapWriter + Sync>(
    sensor_writer: &mut W,
    channels: Vec<u16>,
    prec: &TimestampPrecision,
    tx: Sender<McapLogMessage>,
    pb: ProgressBar,
) -> Result<Vec<TopicWithMessageCountWithTimestamps>, Box<dyn std::error::Error>>
where
    <W as Iterator>::Item: Debug + Send + Sync,
{
    // Collect all items first (sequential load of file paths/lines)
    let items: Vec<_> = sensor_writer.collect();
    let message_count = items.len();

    let writer_ref = &*sensor_writer;

    pb.set_length(message_count as u64);
    pb.set_message(format!("Processing {}", writer_ref.get_topic()));

    let (start_time, end_time) = items
        .into_par_iter()
        .enumerate()
        .map(|(seq, item)| {
            let data = writer_ref.process_item(&item, &prec).map_err(|e| {
                format!(
                    "{}: Failed to process sensor data with seq {}: {:?}",
                    e, seq, item
                )
            })?;

            let timestamp = data
                .get_header()
                .get_timestamp(&TimestampPrecision::NanoSecond);

            let messages = writer_ref
                .serialize_message(data, &channels, seq as u32, &timestamp)
                .map_err(|e| format!("Serialization failed: {}", e))?;

            for msg in messages {
                tx.send(msg).expect("Failed to send message to channel");
            }

            pb.inc(1);
            Ok((timestamp.timestamp, timestamp.timestamp))
        })
        .reduce(
            || -> Result<(u64, u64), String> { Ok((u64::MAX, u64::MIN)) },
            |acc, res| match (acc, res) {
                (Ok(a), Ok(b)) => Ok((cmp::min(a.0, b.0), cmp::max(a.1, b.1))),
                (Err(e), _) => Err(e),
                (_, Err(e)) => Err(e),
            },
        )?;

    pb.finish_with_message("Done");

    // We don't have easy access to topic_metadata here anymore unless we call writer (which is immutable).
    // We can get topic metadata from writer_ref.
    let topic_metadatas = writer_ref.get_topic_metadatas()?;

    let mut topics_with_timestamps: Vec<TopicWithMessageCountWithTimestamps> = vec![];
    for topic_metadata in topic_metadatas {
        let topic_with_msg_count = TopicWithMessageCount {
            topic_metadata,
            message_count: message_count as u64,
        };
        topics_with_timestamps.push(TopicWithMessageCountWithTimestamps {
            topic_with_msg_count,
            start_time,
            end_time,
        });
    }

    Ok(topics_with_timestamps)
}
