use std::fs;

use camino::{Utf8Path, Utf8PathBuf};
use mcap::Compression;
use memmap::Mmap;
use serde::{Deserialize, Serialize};
use serde_yaml;
use std::collections::BTreeMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StartingTime {
    nanoseconds_since_epoch: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Duration {
    nanoseconds: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct File {
    path: String,
    starting_time: StartingTime,
    duration: Duration,
    message_count: u64,
}

#[derive(Debug, Clone)]
pub(crate) struct TopicWithMessageCountWithTimestamps {
    pub(crate) topic_with_msg_count: TopicWithMessageCount,
    pub(crate) start_time: u64,
    pub(crate) end_time: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct TopicWithMessageCount {
    pub(crate) topic_metadata: TopicMetadata,
    pub(crate) message_count: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct TopicMetadata {
    name: String,
    #[serde(rename = "type")]
    type_: String,
    serialization_format: String,
    offered_qos_profiles: String,
}

impl TopicMetadata {
    pub fn new(name: String, type_: String, offered_qos_profiles: String) -> TopicMetadata {
        TopicMetadata {
            name,
            type_,
            serialization_format: "cdr".to_string(),
            offered_qos_profiles,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct RosbagInfo {
    version: u8,
    storage_identifier: String,
    relative_file_paths: Vec<String>,
    duration: Duration,
    starting_time: StartingTime,
    message_count: u64,
    compression_format: String,
    compression_mode: String,
    ros_distro: String,
    files: Vec<File>,
    topics_with_message_count: Vec<TopicWithMessageCount>,
}

impl RosbagInfo {
    pub fn new(
        filename: String,
        start_time: u64,
        end_time: u64,
        message_count: u64,
        compression: Option<Compression>,
        topics_with_message_count: Vec<TopicWithMessageCount>,
    ) -> RosbagInfo {
        let relative_file_paths: Vec<String> = vec![filename.clone()];
        let duration = Duration {
            nanoseconds: end_time - start_time,
        };
        let starting_time = StartingTime {
            nanoseconds_since_epoch: start_time,
        };

        let file = File {
            path: filename.clone(),
            starting_time: starting_time.clone(),
            duration: duration.clone(),
            message_count,
        };
        let files = vec![file];

        let (compression_format, compression_mode) = ("".to_string(), "".to_string());
        // let (compression_format, compression_mode) = match compression {
        //     None => ("".to_string(), "".to_string()),
        //     Some(comp) => match comp {
        //         Compression::Lz4 => ("lz4".to_string(), "chunk".to_string()),
        //         Compression::Zstd => ("zstd".to_string(), "chunk".to_string()),
        //     },
        // };

        RosbagInfo {
            version: 5,
            storage_identifier: "mcap".to_string(),
            relative_file_paths,
            duration,
            starting_time,
            message_count,
            compression_format,
            compression_mode,
            ros_distro: "fomo_rust_sdk".to_string(),
            files,
            topics_with_message_count,
        }
    }

    pub fn save_metadata<P: AsRef<Utf8Path>>(
        &self,
        path: P,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut wrapper = BTreeMap::new();
        wrapper.insert("rosbag2_bagfile_information", self);
        let yaml_string = serde_yaml::to_string(&wrapper)?;
        fs::write(path.as_ref(), yaml_string)?;
        Ok(())
    }
}

pub fn save_metadata_file(
    mapped: &Mmap,
    output_path: &Utf8PathBuf,
) -> Result<Utf8PathBuf, anyhow::Error> {
    // Get the header
    let reader = mcap::read::LinearReader::new(mapped)?;
    let mut compression = None;
    for record in reader {
        match record? {
            mcap::records::Record::ChunkIndex(chunk_idx) => match chunk_idx.compression.as_str() {
                "zstd" => {
                    compression = Some(Compression::Zstd);
                    break;
                }
                "lz4" => {
                    compression = Some(Compression::Lz4);
                    break;
                }
                _ => {}
            },
            _ => {}
        }
    }

    let summary = mcap::Summary::read(mapped)?.unwrap();
    let stats = summary.stats.unwrap();

    let channels = summary.channels.values();

    let mut topics_with_msg_counts: Vec<TopicWithMessageCount> = vec![];

    for channel in channels {
        let schema = channel.schema.clone().unwrap();
        let qos = channel.metadata.get("offered_qos_profiles").unwrap();
        let msg_count = stats.channel_message_counts.get(&channel.id);
        match msg_count {
            Some(msg_count) => {
                let metadata =
                    TopicMetadata::new(channel.topic.clone(), schema.name.clone(), qos.clone());
                let topic_with_msg_count = TopicWithMessageCount {
                    topic_metadata: metadata,
                    message_count: msg_count.clone(),
                };
                topics_with_msg_counts.push(topic_with_msg_count);
            }
            None => eprintln!("No messages for channel {}", channel.topic),
        }
    }

    let filename = output_path.as_path().file_name().unwrap();
    let info = RosbagInfo::new(
        filename.to_string(),
        stats.message_start_time,
        stats.message_end_time,
        stats.message_count,
        compression,
        topics_with_msg_counts,
    );
    let metadata_path = output_path
        .as_path()
        .parent()
        .unwrap()
        .join("metadata.yaml");
    info.save_metadata(metadata_path.as_path()).unwrap();
    Ok(metadata_path)
}
