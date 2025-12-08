use super::header::{self, Header};
use super::timestamp::TimestampPrecision;
use super::utils::{HasHeader, ToRosMsg};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use camino::{Utf8Path, Utf8PathBuf};
use cdr_encoding::{from_bytes, to_vec};
use hound;
use serde::{Deserialize, Serialize};
use std::{
    fs::create_dir,
    io::{Cursor, Write},
};

pub const DATA_SCHEMA_DEF: &str = "std_msgs/Header header\naudio_common_msgs/AudioData audio\n================================================================================\nMSG: std_msgs/Header\nbuiltin_interfaces/Time stamp\nstring frame_id\n================================================================================\nMSG: builtin_interfaces/Time\nint32 sec\nuint32 nanosec\n================================================================================\nMSG: audio_common_msgs/AudioData\nuint8[] data\n";
pub const INFO_SCHEMA_DEF: &str = "uint8 channels\nuint32 sample_rate\nstring sample_format\nuint32 bitrate\nstring coding_format\n";

pub const FOXGLOVE_SCHEMA_DEF: &str = "builtin_interfaces/Time timestamp\nuint8[] data\nstring format\nuint32 sample_rate\nuint32 number_of_channels\n================================================================================\nMSG: builtin_interfaces/Time\nint32 sec\nuint32 nanosec\n";

const SAMPLE_RATE: u32 = 44100;
pub const AUDIOLEFT_TOPIC: &str = "/audio/left";
pub const AUDIOLEFT_FRAME_ID: &str = "audio_left";
pub const AUDIORIGHT_TOPIC: &str = "/audio/right";
pub const AUDIORIGHT_FRAME_ID: &str = "audio_right";

pub struct FomoAudios<'a> {
    pub topic_name: &'a str,
    pub output_path: Utf8PathBuf,
    pub audios: Vec<Audio>,
}

impl FomoAudios<'_> {
    pub fn new<P: AsRef<Utf8Path>>(topic_name: &str, output_path: P) -> FomoAudios {
        if !output_path.as_ref().exists() {
            create_dir(output_path.as_ref()).unwrap();
        }
        FomoAudios {
            topic_name,
            output_path: output_path.as_ref().to_path_buf(),
            audios: Vec::new(),
        }
    }

    pub fn add(&mut self, audio: Audio) {
        self.audios.push(audio);
    }

    /// Input /tmp/YYYY-MM-DD-HH-mm/traj-name/audio_left
    pub fn save<P: AsRef<std::path::Path>>(
        &self,
        path: P,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if !path.as_ref().exists() {
            create_dir(path.as_ref())?;
        }
        for audio in &self.audios {
            let filename = format!(
                "{}.wav",
                audio
                    .header
                    .get_timestamp(&TimestampPrecision::MicroSecond)
                    .timestamp
            );
            audio.save_msg(path.as_ref().join(filename))?;
        }
        Ok(())
    }
}

#[derive(Debug, Deserialize, PartialEq)]
pub struct Audio {
    pub header: Header,
    pub data: Vec<u8>,
}
#[derive(Debug, Serialize, PartialEq, Deserialize)]
pub(crate) struct SplitTimestamp {
    pub(crate) sec: u32,
    pub(crate) nsec: u32,
}

#[derive(Debug, Serialize, PartialEq)]
pub struct FoxgloveAudio {
    pub(crate) timestamp: SplitTimestamp,
    pub(crate) data: Vec<u8>,
    pub(crate) format: String,
    pub(crate) sample_rate: u32,
    pub(crate) number_of_channels: u32,
}

impl Audio {
    pub fn from_file<P: AsRef<std::path::Path>>(
        path: P,
        frame_id: &str,
        prec: &TimestampPrecision,
    ) -> Result<Audio, Box<dyn std::error::Error>> {
        let file_stem = path.as_ref().file_stem().unwrap().to_str().unwrap();
        let timestamp = file_stem.parse::<u64>()?;
        let (stamp_sec, stamp_nsec) = header::get_sec_nsec(timestamp, prec);
        let header = Header {
            stamp_sec,
            stamp_nsec,
            frame_id: frame_id.to_string(),
        };

        let mut reader = hound::WavReader::open(path)?;
        let mut data = Vec::new();
        let mut cursor = Cursor::new(&mut data);
        for sample in reader.samples::<i16>() {
            let sample = sample?;
            cursor.write_i16::<LittleEndian>(sample)?;
        }
        Ok(Audio { header, data })
    }

    pub fn save_msg<P: AsRef<std::path::Path>>(
        &self,
        path: P,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: SAMPLE_RATE,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };

        let mut writer = hound::WavWriter::create(path.as_ref(), spec)?;
        let mut cursor = Cursor::new(&self.data);
        while let Ok(sample) = cursor.read_i16::<LittleEndian>() {
            writer.write_sample(sample)?;
        }
        writer.finalize().unwrap();
        Ok(())
    }
}

impl ToRosMsg<Utf8PathBuf> for Audio {
    fn get_schema_def() -> &'static [u8] {
        FOXGLOVE_SCHEMA_DEF.as_bytes()
    }

    fn get_schema_name() -> &'static str {
        "foxglove_msgs/msg/RawAudio"
    }

    fn from_item(
        item: &Utf8PathBuf,
        frame_id: String,
        prec: &TimestampPrecision,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Audio::from_file(item, &frame_id, prec)
    }

    fn construct_msg(audio: Audio, buffer: &mut Vec<u8>) -> Result<(), Box<dyn std::error::Error>> {
        let mut cursor = Cursor::new(buffer);
        cursor.write_u32::<LittleEndian>(256)?;

        let foxglove_audio = FoxgloveAudio {
            timestamp: SplitTimestamp {
                sec: audio.header.stamp_sec as u32,
                nsec: audio.header.stamp_nsec,
            },
            data: audio.data.clone(),
            format: "pcm-s16".to_string(),
            sample_rate: SAMPLE_RATE,
            number_of_channels: 1,
        };

        let serialized = to_vec::<FoxgloveAudio, LittleEndian>(&foxglove_audio)?;
        cursor.write(&serialized)?;

        Ok(())
    }
}

impl HasHeader for Audio {
    fn get_header(&self) -> Header {
        self.header.clone()
    }
}

pub fn parse_msg(input: &[u8]) -> Result<Audio, Box<dyn std::error::Error>> {
    let (deserialized_message, _consumed_byte_count) =
        from_bytes::<Audio, LittleEndian>(&input[4..])?; // first 4 bytes are endian, always Little in our case

    Ok(deserialized_message)
}

#[cfg(test)]
mod tests {
    use super::super::utils;
    use super::*;
    use std::fs::read;
    use std::fs::File;
    use std::io::Read;

    #[test]
    fn process_message() -> Result<(), Box<dyn std::error::Error>> {
        let mut file = File::open("tests/audio.bin")?;
        let mut buf = Vec::new();
        file.read_to_end(&mut buf)?;
        let parsed_data = parse_msg(&buf)?;
        let header = Header {
            stamp_sec: 1748443580,
            stamp_nsec: 201237201,
            frame_id: "microphone_left".to_string(),
        };
        assert_eq!(parsed_data.header, header);
        assert_eq!(
            parsed_data
                .header
                .get_timestamp(&TimestampPrecision::NanoSecond),
            header.get_timestamp(&TimestampPrecision::NanoSecond)
        );
        Ok(())
    }

    #[test]
    fn read_write() -> Result<(), Box<dyn std::error::Error>> {
        let ref_path = "tests/output/1748443589201619148.wav";
        let save_path = "/tmp/test_audio_left.wav";
        let audio_obj =
            Audio::from_file(ref_path, "microphone_left", &TimestampPrecision::NanoSecond)?;
        audio_obj.save_msg(save_path)?;

        let ref_bytes = read(ref_path)?;
        let saved_bytes = read(save_path)?;
        assert_eq!(
            utils::hash_bytes(&ref_bytes),
            utils::hash_bytes(&saved_bytes),
            "Audio wav file mismatch"
        );
        Ok(())
    }
}
