pub mod calib;
pub mod cli_common;
pub mod data_writer;
pub mod io;
pub mod metadata;
pub mod qos;
pub mod rosbag;
pub mod sensors;
pub mod utils;

use byteorder::LittleEndian;
use opencv::core::Vector;
use rosbag::{Duration, RosbagInfo, StartingTime, TopicWithMessageCountWithTimestamps};
use std::fmt::Debug;

use cdr_encoding::from_bytes;
use clap::ValueEnum;

use anyhow::{anyhow, Result};
use camino::{Utf8Path, Utf8PathBuf};
use metadata::{
    BaslerParams, BaslerParamsTimestamped, BatteryLog, F64Data, TemperaturePressure,
    TwistTimestamp, DPS310,
};
use metadata::{
    BASLER_PARAMS_FILE_NAME, BATTERY_LOGS_FILE_NAME, CMD_VEL_FILE_NAME, DPS310_DATA_FILE_NAME,
    LEFT_CMD_VELOCITY_FILE_NAME, LEFT_CURRENT_FILE_NAME, LEFT_VELOCITY_FILE_NAME,
    LEFT_VOLTAGE_FILE_NAME, RIGHT_CMD_VELOCITY_FILE_NAME, RIGHT_CURRENT_FILE_NAME,
    RIGHT_VELOCITY_FILE_NAME, RIGHT_VOLTAGE_FILE_NAME, VN100_PRESSURE_FILE_NAME,
    VN100_TEMPERATURE_FILE_NAME,
};
use sensors::audio::{AUDIOLEFT_FRAME_ID, AUDIOLEFT_TOPIC, AUDIORIGHT_FRAME_ID, AUDIORIGHT_TOPIC};
use sensors::basic::Twist;
use sensors::common::DataVector;
use sensors::image::{RosImage, BASLER_FRAME_ID, BASLER_NAMESPACE};
use sensors::odom::{Drivetrain, MotorVelocity, WheelSide, ODOM_FRAME_ID, ODOM_TOPIC};
use sensors::point_cloud::{LEISHEN_FRAME_ID, LEISHEN_TOPIC};
use sensors::point_cloud::{ROBOSENSE_FRAME_ID, ROBOSENSE_TOPIC};
use sensors::radar::{RadarScan, NAVTECH_FRAME_ID, NAVTECH_NAMESPACE};
use sensors::stereo::{
    ZEDXLEFT_FRAME_ID, ZEDXLEFT_NAMESPACE, ZEDXRIGHT_FRAME_ID, ZEDXRIGHT_NAMESPACE,
};
use sensors::{
    imu::{VECTORANV_TOPIC, VECTORNAV_FRAME_ID, XSENS_FRAME_ID, XSENS_TOPIC},
    odom::DiffDrive,
};
use tqdm::tqdm;

use mcap::{read, Compression};

use data_writer::{
    write_sensor_data, write_tf_data, CsvLoader, DirectoryLoader, MsgMcapWriter,
    MsgWithInfoMcapWriter,
};
use mcap;
use sensors::audio;
use sensors::image;
use sensors::imu::{self};
use sensors::odom;
use sensors::point_cloud;
use sensors::radar;
use sensors::stereo;
use sensors::timestamp::{Timestamp, TimestampPrecision};
use std::time::Instant;
use std::{
    cmp,
    fs::{self},
};
use std::{io::BufWriter, u64};

#[derive(Debug, Clone, PartialEq)]
pub enum SensorType {
    Navtech,
    ZedXLeft,
    ZedXRight,
    Basler,
    RoboSense,
    Leishen,
    Audio,
}
impl SensorType {
    const fn as_str(&self) -> &'static str {
        match self {
            Self::Navtech => NAVTECH_FRAME_ID,
            Self::ZedXLeft => ZEDXLEFT_FRAME_ID,
            Self::ZedXRight => ZEDXRIGHT_FRAME_ID,
            Self::Basler => BASLER_FRAME_ID,
            Self::RoboSense => ROBOSENSE_FRAME_ID,
            Self::Leishen => LEISHEN_FRAME_ID,
            Self::Audio => "audio",
        }
    }

    fn get_folder(&self) -> Option<String> {
        Some(self.as_str().to_string())
    }

    pub fn get_all() -> Vec<SensorType> {
        vec![
            SensorType::Navtech,
            SensorType::ZedXLeft,
            SensorType::ZedXRight,
            SensorType::Basler,
            SensorType::RoboSense,
            SensorType::Leishen,
            SensorType::Audio,
        ]
    }
}

impl ValueEnum for SensorType {
    fn value_variants<'a>() -> &'a [Self] {
        &[
            Self::Navtech,
            Self::ZedXLeft,
            Self::ZedXRight,
            Self::Basler,
            Self::RoboSense,
            Self::Leishen,
            Self::Audio,
        ]
    }

    fn to_possible_value(&self) -> Option<clap::builder::PossibleValue> {
        Some(clap::builder::PossibleValue::new(self.as_str()))
    }
}

pub fn process_rosbag<P: AsRef<Utf8Path>>(
    input: P,
    output: P,
    sensors: &Vec<SensorType>,
    prec: &TimestampPrecision,
    drivetrain: &Drivetrain,
) -> Result<()> {
    // check that the input is either .mcap or folder/name.mcap
    let input_path = io::check_mcap_input_path(input)?;
    io::check_ijrr_output_path(output.as_ref())?;

    if sensors.contains(&SensorType::ZedXLeft) ^ sensors.contains(&SensorType::ZedXRight) {
        return Err(anyhow!("Must export both left and right zedx cameras"));
    }

    let mapped = io::map_mcap(input_path)?;

    let summary = mcap::Summary::read(&mapped)?.unwrap();
    let stats = summary.stats.unwrap();
    let msg_count: usize = stats.message_count.try_into().unwrap();
    // if the recording took place before Wednesday, June 25, 2025 1:00:00 AM,
    // fix warthog motor currents
    let fix_current = stats.message_start_time < 1750813200 * 1_000_000_000;

    let mut mti_vector = DataVector::new("/mti30/data_raw".to_string());
    let mut vn100_vector = DataVector::new("/vn100/data_raw".to_string());
    let mut odom_vector = DataVector::new("/warthog/platform/odom".to_string());

    let left_audio =
        audio::FomoAudios::new("/audio/left_mic", output.as_ref().join(AUDIOLEFT_FRAME_ID));
    let right_audio = audio::FomoAudios::new(
        "/audio/right_mic",
        output.as_ref().join(AUDIORIGHT_FRAME_ID),
    );

    // TODO don't create a new folder if not in `sensors`
    let lslidar = point_cloud::FomoPointClouds::new(
        "/lslidar128/points",
        output.as_ref().join(LEISHEN_FRAME_ID),
    );
    let rslidar = point_cloud::FomoPointClouds::new(
        "/rslidar128/points",
        output.as_ref().join(ROBOSENSE_FRAME_ID),
    );

    let radar = radar::FomoRadars::new("/radar/b_scan_msg", output.as_ref().join(NAVTECH_FRAME_ID));

    let basler_image = image::BaslerImages::new(
        "/basler/driver/image_raw",
        output.as_ref().join(BASLER_FRAME_ID),
    );
    let mut zed_images = stereo::FomoStereoImages::new(
        "/zed_node/left_raw/image_raw_color",
        "/zed_node/right_raw/image_raw_color",
        output.as_ref().join(ZEDXLEFT_FRAME_ID),
        output.as_ref().join(ZEDXRIGHT_FRAME_ID),
    );
    let mut basler_params_vec = DataVector::new("/basler/driver/current_params".to_string());
    let mut battery_logs = DataVector::new("/battery_logs".to_string());
    let mut dps310_vec = DataVector::new("/dps310_warthog/data".to_string());

    let mut vn100_temperatures = DataVector::new("/warthog/platform/temperature".to_string());
    let mut vn100_pressures = DataVector::new("/warthog/platform/pressure".to_string());

    let mut cmd_vel_vec = DataVector::new("/warthog/platform/cmd_vel".to_string());

    let mut left_voltage =
        DataVector::new("/warthog/platform/motor/left/status/voltage_battery".to_string());
    let mut right_voltage =
        DataVector::new("/warthog/platform/motor/right/status/voltage_battery".to_string());

    let mut left_current =
        DataVector::new("/warthog/platform/motor/left/status/current_battery".to_string());
    let mut right_current =
        DataVector::new("/warthog/platform/motor/right/status/current_battery".to_string());

    let mut left_velocity =
        DataVector::new("/warthog/platform/motor/left/status/velocity".to_string());
    let mut right_velocity =
        DataVector::new("/warthog/platform/motor/right/status/velocity".to_string());

    let mut left_cmd_velocity =
        DataVector::new("/warthog/platform/motor/left/cmd_velocity".to_string());
    let mut right_cmd_velocity =
        DataVector::new("/warthog/platform/motor/right/cmd_velocity".to_string());

    // predefine this to prevent num_message * O(n) cost
    let export_lslidar = sensors.contains(&SensorType::Leishen);
    let export_rslidar = sensors.contains(&SensorType::RoboSense);
    let export_basler = sensors.contains(&SensorType::Basler);
    let export_zedx =
        sensors.contains(&SensorType::ZedXLeft) && sensors.contains(&SensorType::ZedXRight);
    let export_radar = sensors.contains(&SensorType::Navtech);
    let mut motor_vel: MotorVelocity;
    let mut diff_drive = DiffDrive::new(drivetrain, odom::DIFF_DRIVE_FREQUENCY);

    let is_tty = atty::is(atty::Stream::Stdout);
    let message_stream = read::MessageStream::new(&mapped)?;
    let iter: Box<dyn Iterator<Item = _>> = if is_tty {
        Box::new(
            tqdm(message_stream)
                .desc(Some("Processing input data"))
                .total(Some(msg_count)),
        )
    } else {
        Box::new(message_stream)
    };

    let mut last_perc = 0.0;

    for (ctr, message) in iter.enumerate() {
        let perc = (10.0 * (ctr as f64 / msg_count as f64)).floor();
        if !is_tty && perc != last_perc {
            println!("Reached {}0 %", perc);
            last_perc = perc;
        }

        let message = message?;

        let schema = message
            .channel
            .schema
            .as_ref()
            .ok_or(Err::<mcap::Schema, anyhow::Error>(anyhow!(
                "Message schema must exist."
            )))
            .unwrap();

        match schema.name.as_str() {
            "pylon_ros2_camera_interfaces/msg/CurrentParams" => {
                // utils::print_message(&message, true);
                // println!("{:?}", &message.data[4..]);
                let t = Timestamp::new(message.publish_time, &TimestampPrecision::NanoSecond);
                let (data, _consumed_byte_count) =
                    from_bytes::<BaslerParams, LittleEndian>(&message.data[4..])?;
                basler_params_vec.add(BaslerParamsTimestamped::new(t, data));
            }
            "sysnergie_msgs/msg/BatteryLog" => {
                let (data, _consumed_byte_count) =
                    from_bytes::<BatteryLog, LittleEndian>(&message.data[4..])?;
                battery_logs.add(data);
            }
            "rtf_sensors_msgs/msg/CustomPressureTemperature" => {
                let (data, _consumed_byte_count) =
                    from_bytes::<DPS310, LittleEndian>(&message.data[4..])?;
                dps310_vec.add(data);
            }
            "sensor_msgs/msg/Temperature" | "sensor_msgs/msg/FluidPressure" => {
                let (data, _consumed_byte_count) =
                    from_bytes::<TemperaturePressure, LittleEndian>(&message.data[4..])?;
                match schema.name.as_str() {
                    "sensor_msgs/msg/Temperature" => {
                        let temperature = data;
                        vn100_temperatures.add(temperature);
                    }
                    "sensor_msgs/msg/FluidPressure" => {
                        let pressure = data;
                        vn100_pressures.add(pressure);
                    }
                    _ => continue,
                }
            }
            "geometry_msgs/msg/Twist" => {
                let topic = message.channel.topic.as_str();
                if topic != cmd_vel_vec.get_topic() {
                    continue;
                }
                let (twist, _consumed_byte_count) =
                    from_bytes::<Twist, LittleEndian>(&message.data[4..])?;
                cmd_vel_vec.add(TwistTimestamp::new(
                    Timestamp::new(message.publish_time, &TimestampPrecision::NanoSecond),
                    twist,
                ));
            }
            "std_msgs/msg/Float64" => {
                let t = Timestamp::new(message.publish_time, &TimestampPrecision::NanoSecond);
                let topic = message.channel.topic.as_str();
                match topic {
                    topic_str
                        if topic_str == left_velocity.get_topic()
                            || topic_str == right_velocity.get_topic() =>
                    {
                        let side = match topic.contains("left") {
                            true => WheelSide::Left,
                            false => WheelSide::Right,
                        };
                        let (velocity, _consumed_byte_count) =
                            from_bytes::<f64, LittleEndian>(&message.data[4..])?;
                        match side {
                            WheelSide::Left => left_velocity.add(F64Data::new(t, velocity)),
                            WheelSide::Right => right_velocity.add(F64Data::new(t, velocity)),
                        };
                        motor_vel = MotorVelocity {
                            velocity,
                            timestamp: message.publish_time,
                            side,
                        };
                        diff_drive.add_vel(motor_vel);
                    }
                    topic_str
                        if topic_str == left_voltage.get_topic()
                            || topic_str == right_voltage.get_topic() =>
                    {
                        let (value, _consumed_byte_count) =
                            from_bytes::<f64, LittleEndian>(&message.data[4..])?;
                        match topic.contains("left") {
                            true => left_voltage.add(F64Data::new(t, value)),
                            false => right_voltage.add(F64Data::new(t, value)),
                        };
                    }
                    topic_str
                        if topic_str == left_current.get_topic()
                            || topic_str == right_current.get_topic() =>
                    {
                        let (mut value, _consumed_byte_count) =
                            from_bytes::<f64, LittleEndian>(&message.data[4..])?;
                        if fix_current {
                            value = sensors::utils::sign_warthog_current(value);
                        }
                        match topic.contains("left") {
                            true => left_current.add(F64Data::new(t, value)),
                            false => right_current.add(F64Data::new(t, value)),
                        };
                    }
                    topic_str
                        if topic_str == left_cmd_velocity.get_topic()
                            || topic_str == right_cmd_velocity.get_topic() =>
                    {
                        let (value, _consumed_byte_count) =
                            from_bytes::<f64, LittleEndian>(&message.data[4..])?;
                        match topic.contains("left") {
                            true => left_cmd_velocity.add(F64Data::new(t, value)),
                            false => right_cmd_velocity.add(F64Data::new(t, value)),
                        };
                    }
                    _ => continue,
                };
            }
            "sensor_msgs/msg/Imu" => {
                let parsed_imu_data = imu::parse_msg(&message.data).unwrap();
                if message.channel.topic == mti_vector.get_topic() {
                    mti_vector.add(parsed_imu_data);
                } else if message.channel.topic == vn100_vector.get_topic() {
                    vn100_vector.add(parsed_imu_data);
                }
            }
            "audio_common_msgs/msg/AudioDataStamped" => {
                let parsed_audio_data = audio::parse_msg(&message.data).unwrap();
                let filename = format!(
                    "{}.wav",
                    parsed_audio_data.header.get_timestamp(prec).timestamp
                );
                if message.channel.topic == left_audio.topic_name {
                    parsed_audio_data
                        .save_msg(left_audio.output_path.join(filename))
                        .unwrap();
                } else if message.channel.topic == right_audio.topic_name {
                    parsed_audio_data
                        .save_msg(right_audio.output_path.join(filename))
                        .unwrap();
                }
            }
            "sensor_msgs/msg/PointCloud2" => {
                let is_lslidar = lslidar.topic_name == message.channel.topic;

                if is_lslidar && !export_lslidar {
                    continue;
                }
                if !is_lslidar && !export_rslidar {
                    continue;
                }

                let parsed_point_cloud = point_cloud::parse_msg(&message.data, is_lslidar).unwrap();
                let filename = format!(
                    "{}.bin",
                    parsed_point_cloud.header.get_timestamp(prec).timestamp
                );
                if message.channel.topic == lslidar.topic_name {
                    parsed_point_cloud
                        .save(lslidar.output_path.join(filename), prec)
                        .unwrap();
                } else if message.channel.topic == rslidar.topic_name {
                    parsed_point_cloud
                        .save(rslidar.output_path.join(filename), prec)
                        .unwrap();
                }
            }
            "navtech_msgs/msg/RadarBScanMsg" => {
                if !export_radar {
                    continue;
                }
                let (mut parsed_msg, _consumed_byte_count) =
                    from_bytes::<RadarScan, LittleEndian>(&message.data[4..])?;
                parsed_msg.fix_header_time();
                let filename = format!(
                    "{}.png",
                    parsed_msg.b_scan_image.header.get_timestamp(prec).timestamp
                );
                parsed_msg
                    .save_msg(radar.output_path.join(filename), prec)
                    .unwrap();
            }
            "sensor_msgs/msg/Image" => {
                if message.channel.topic == "/audio/spectogram" {
                    continue;
                }
                let is_basler = message.channel.topic == basler_image.topic_name;
                if is_basler && !export_basler {
                    continue;
                }
                if !is_basler && !export_zedx {
                    continue;
                }
                let (ros_image, _consumed_byte_count) =
                    from_bytes::<RosImage, LittleEndian>(&message.data[4..])?;
                let image = image::Image::from_ros_image(ros_image).unwrap();
                let filename = format!("{}.png", image.header.get_timestamp(prec).timestamp);
                if is_basler {
                    image
                        .save_msg(basler_image.output_path.join(filename), true)
                        .unwrap();
                } else if message.channel.topic == zed_images.left_topic_name {
                    zed_images.add(image, stereo::StereoSide::Left, prec);
                    zed_images.maybe_save_images()?;
                } else if message.channel.topic == zed_images.right_topic_name {
                    zed_images.add(image, stereo::StereoSide::Right, prec);
                    zed_images.maybe_save_images()?;
                }
            }
            _ => {
                continue;
            }
        }
    }

    // Save all data
    println!("Saving mti data");
    mti_vector
        .save(
            output.as_ref().join(format!("{}.csv", XSENS_FRAME_ID)),
            prec,
        )
        .unwrap();
    println!("Saving vn100 data");
    vn100_vector
        .save(
            output.as_ref().join(format!("{}.csv", VECTORNAV_FRAME_ID)),
            prec,
        )
        .unwrap();

    fs::create_dir_all(output.as_ref().join("metadata"))?;
    println!("Saving basler params");
    basler_params_vec
        .save(
            output
                .as_ref()
                .join("metadata")
                .join(format!("{}.csv", BASLER_PARAMS_FILE_NAME)),
            prec,
        )
        .unwrap();
    println!("Saving battery logs");
    battery_logs
        .save(
            output
                .as_ref()
                .join("metadata")
                .join(format!("{}.csv", BATTERY_LOGS_FILE_NAME)),
            prec,
        )
        .unwrap();
    println!("Saving dps310 data");
    dps310_vec
        .save(
            output
                .as_ref()
                .join("metadata")
                .join(format!("{}.csv", DPS310_DATA_FILE_NAME)),
            prec,
        )
        .unwrap();
    println!("Saving vn100 pressure data");
    vn100_pressures
        .save(
            output
                .as_ref()
                .join("metadata")
                .join(format!("{}.csv", VN100_PRESSURE_FILE_NAME)),
            prec,
        )
        .unwrap();
    println!("Saving vn100 temperature data");
    vn100_temperatures
        .save(
            output
                .as_ref()
                .join("metadata")
                .join(format!("{}.csv", VN100_TEMPERATURE_FILE_NAME)),
            prec,
        )
        .unwrap();
    println!("Saving cmd vel data");
    cmd_vel_vec
        .save(
            output
                .as_ref()
                .join("metadata")
                .join(format!("{}.csv", CMD_VEL_FILE_NAME)),
            prec,
        )
        .unwrap();
    println!("Saving left voltage data");
    left_voltage
        .save(
            output
                .as_ref()
                .join("metadata")
                .join(format!("{}.csv", LEFT_VOLTAGE_FILE_NAME)),
            prec,
        )
        .unwrap();
    println!("Saving right voltage data");
    right_voltage
        .save(
            output
                .as_ref()
                .join("metadata")
                .join(format!("{}.csv", RIGHT_VOLTAGE_FILE_NAME)),
            prec,
        )
        .unwrap();
    println!("Saving left velocity data");
    left_velocity
        .save(
            output
                .as_ref()
                .join("metadata")
                .join(format!("{}.csv", LEFT_VELOCITY_FILE_NAME)),
            prec,
        )
        .unwrap();
    println!("Saving right velocity data");
    right_velocity
        .save(
            output
                .as_ref()
                .join("metadata")
                .join(format!("{}.csv", RIGHT_VELOCITY_FILE_NAME)),
            prec,
        )
        .unwrap();
    println!("Saving left current data");
    left_current
        .save(
            output
                .as_ref()
                .join("metadata")
                .join(format!("{}.csv", LEFT_CURRENT_FILE_NAME)),
            prec,
        )
        .unwrap();
    println!("Saving right current data");
    right_current
        .save(
            output
                .as_ref()
                .join("metadata")
                .join(format!("{}.csv", RIGHT_CURRENT_FILE_NAME)),
            prec,
        )
        .unwrap();
    println!("Saving left cmd velocity data");
    left_cmd_velocity
        .save(
            output
                .as_ref()
                .join("metadata")
                .join(format!("{}.csv", LEFT_CMD_VELOCITY_FILE_NAME)),
            prec,
        )
        .unwrap();
    println!("Saving right cmd velocity data");
    right_cmd_velocity
        .save(
            output
                .as_ref()
                .join("metadata")
                .join(format!("{}.csv", RIGHT_CMD_VELOCITY_FILE_NAME)),
            prec,
        )
        .unwrap();
    println!("Saving odom vector");
    let odoms = diff_drive.process_velocities();
    match odoms {
        Ok(odoms) => {
            odom_vector.add_data_vec(odoms);
            odom_vector
                .save(output.as_ref().join(format!("{}.csv", ODOM_FRAME_ID)), prec)
                .unwrap();
        }
        Err(e) => eprintln!("{}", e),
    }

    Ok(())
}

pub fn process_folder<P: AsRef<Utf8Path>>(
    input: P,
    output: P,
    sensors: &Vec<SensorType>,
    compress: bool,
    prec: &TimestampPrecision,
) -> Result<(), anyhow::Error> {
    io::check_ijrr_input_path(input.as_ref())?;
    let output_path = io::check_mcap_output_path(output.as_ref())?;

    let calib_path = input.as_ref().join("calib");
    let mut compression = None;
    if compress == true {
        compression = Some(Compression::Zstd);
    }
    let write_options = mcap::WriteOptions::new()
        .compression(compression)
        .profile("ros2")
        .library(&format!(
            "fomo-sdk/memory-leak-fix/{}",
            env!("CARGO_PKG_VERSION")
        ))
        .use_chunks(true)
        .disable_seeking(false)
        .emit_summary_records(true)
        .emit_summary_offsets(true)
        .emit_message_indexes(true);

    let mut mcap_writer = write_options
        .create(BufWriter::new(fs::File::create(&output_path).map_err(
            |e| anyhow!("Failed to create output file {}: {}", &output_path, e),
        )?))
        .map_err(|e| anyhow!("Failed to create write_options: {}", e))?;

    let start = Instant::now();

    // all output mcaps contain the IMU, odom and TF data
    let mut imu_mcap_writer: MsgMcapWriter<imu::Imu, CsvLoader> = MsgMcapWriter::new(
        input.as_ref().join(format!("{}.csv", VECTORNAV_FRAME_ID)),
        VECTORANV_TOPIC.to_string(),
        VECTORNAV_FRAME_ID.to_string(),
        "csv",
    )
    .unwrap();
    let mut topics_with_timestamps: Vec<TopicWithMessageCountWithTimestamps> = vec![];

    topics_with_timestamps.append(
        &mut write_sensor_data(&mut mcap_writer, &mut imu_mcap_writer, prec)
            .inspect_err(|e| eprintln!("Failed to process {}.csv: {}", VECTORNAV_FRAME_ID, e))
            .unwrap(),
    );
    let mut imu_mcap_writer: MsgMcapWriter<imu::Imu, CsvLoader> = MsgMcapWriter::new(
        input.as_ref().join(format!("{}.csv", XSENS_FRAME_ID)),
        XSENS_TOPIC.to_string(),
        XSENS_FRAME_ID.to_string(),
        "csv",
    )
    .unwrap();
    topics_with_timestamps.append(
        &mut write_sensor_data(&mut mcap_writer, &mut imu_mcap_writer, prec)
            .inspect_err(|e| eprintln!("Failed to process {}.csv: {}", XSENS_FRAME_ID, e))
            .unwrap(),
    );
    let mut odom_mcap_writer: MsgMcapWriter<odom::Odom, CsvLoader> = MsgMcapWriter::new(
        input.as_ref().join(format!("{}.csv", ODOM_FRAME_ID)),
        ODOM_TOPIC.to_string(),
        ODOM_FRAME_ID.to_string(),
        "csv",
    )
    .unwrap();
    topics_with_timestamps.append(
        &mut write_sensor_data(&mut mcap_writer, &mut odom_mcap_writer, prec)
            .inspect_err(|e| eprintln!("Failed to process {}.csv: {}", ODOM_FRAME_ID, e))
            .unwrap(),
    );
    for sensor_type in sensors {
        let path = input.as_ref().join(sensor_type.get_folder().unwrap());
        match sensor_type {
            SensorType::Navtech => {
                let mut navtech_mcap_writer: MsgWithInfoMcapWriter<
                    radar::RadarScan,
                    DirectoryLoader,
                > = MsgWithInfoMcapWriter::new(
                    path,
                    calib_path.clone(),
                    NAVTECH_NAMESPACE.to_string(),
                    NAVTECH_FRAME_ID.to_string(),
                    "png",
                )
                .unwrap();
                topics_with_timestamps.append(
                    &mut write_sensor_data(&mut mcap_writer, &mut navtech_mcap_writer, prec)
                        .inspect_err(|e| eprintln!("Failed to process audio data: {}", e))
                        .unwrap(),
                );
            }
            SensorType::ZedXLeft => {
                let mut navtech_mcap_writer: MsgWithInfoMcapWriter<image::Image, DirectoryLoader> =
                    MsgWithInfoMcapWriter::new(
                        path,
                        calib_path.clone(),
                        ZEDXLEFT_NAMESPACE.to_string(),
                        ZEDXLEFT_FRAME_ID.to_string(),
                        "png",
                    )
                    .unwrap();
                topics_with_timestamps.append(
                    &mut write_sensor_data(&mut mcap_writer, &mut navtech_mcap_writer, prec)
                        .inspect_err(|e| eprintln!("Failed to process zedx_left data: {}", e))
                        .unwrap(),
                );
            }
            SensorType::ZedXRight => {
                let mut navtech_mcap_writer: MsgWithInfoMcapWriter<image::Image, DirectoryLoader> =
                    MsgWithInfoMcapWriter::new(
                        path,
                        calib_path.clone(),
                        ZEDXRIGHT_NAMESPACE.to_string(),
                        ZEDXRIGHT_FRAME_ID.to_string(),
                        "png",
                    )
                    .unwrap();
                topics_with_timestamps.append(
                    &mut write_sensor_data(&mut mcap_writer, &mut navtech_mcap_writer, prec)
                        .inspect_err(|e| eprintln!("Failed to process zedx_right data: {}", e))
                        .unwrap(),
                );
            }
            SensorType::Basler => {
                let mut basler_mcap_writer: MsgWithInfoMcapWriter<image::Image, DirectoryLoader> =
                    MsgWithInfoMcapWriter::new(
                        path,
                        calib_path.clone(),
                        BASLER_NAMESPACE.to_string(),
                        BASLER_FRAME_ID.to_string(),
                        "png",
                    )
                    .unwrap();
                topics_with_timestamps.append(
                    &mut write_sensor_data(&mut mcap_writer, &mut basler_mcap_writer, prec)
                        .inspect_err(|e| eprintln!("Failed to process audio data: {}", e))
                        .unwrap(),
                );
            }
            SensorType::RoboSense => {
                let mut lidar_mcap_writer: MsgMcapWriter<point_cloud::PointCloud, DirectoryLoader> =
                    MsgMcapWriter::new(
                        path,
                        ROBOSENSE_TOPIC.to_string(),
                        ROBOSENSE_FRAME_ID.to_string(),
                        "bin",
                    )
                    .unwrap();
                topics_with_timestamps.append(
                    &mut write_sensor_data(&mut mcap_writer, &mut lidar_mcap_writer, prec)
                        .inspect_err(|e| eprintln!("Failed to process lidar data: {}", e))
                        .unwrap(),
                );
            }
            SensorType::Leishen => {
                let mut lidar_mcap_writer: MsgMcapWriter<point_cloud::PointCloud, DirectoryLoader> =
                    MsgMcapWriter::new(
                        path,
                        LEISHEN_TOPIC.to_string(),
                        LEISHEN_FRAME_ID.to_string(),
                        "bin",
                    )
                    .unwrap();
                topics_with_timestamps.append(
                    &mut write_sensor_data(&mut mcap_writer, &mut lidar_mcap_writer, prec)
                        .inspect_err(|e| eprintln!("Failed to process audio data: {}", e))
                        .unwrap(),
                );
            }
            SensorType::Audio => {
                let path_left = Utf8PathBuf::from(format!("{}_left", path));
                let mut audio_mcap_writer: MsgMcapWriter<audio::Audio, DirectoryLoader> =
                    MsgMcapWriter::new(
                        path_left,
                        AUDIOLEFT_TOPIC.to_string(),
                        AUDIOLEFT_FRAME_ID.to_string(),
                        "wav",
                    )
                    .unwrap();
                topics_with_timestamps.append(
                    &mut write_sensor_data(&mut mcap_writer, &mut audio_mcap_writer, prec)
                        .inspect_err(|e| eprintln!("Failed to process audio data: {}", e))
                        .unwrap(),
                );
                let path_right = Utf8PathBuf::from(format!("{}_right", path));
                let mut audio_mcap_writer: MsgMcapWriter<audio::Audio, DirectoryLoader> =
                    MsgMcapWriter::new(
                        path_right,
                        AUDIORIGHT_TOPIC.to_string(),
                        AUDIORIGHT_FRAME_ID.to_string(),
                        "wav",
                    )
                    .unwrap();
                topics_with_timestamps.append(
                    &mut write_sensor_data(&mut mcap_writer, &mut audio_mcap_writer, prec)
                        .inspect_err(|e| eprintln!("Failed to process audio data: {}", e))
                        .unwrap(),
                );
            }
        }
    }

    let (start_time, end_time, mut message_count, mut topics_with_msg_count) =
        topics_with_timestamps.iter().fold(
            (u64::MAX, u64::MIN, 0u64, Vec::new()),
            |(min_start, max_end, count, mut topics), v| {
                topics.push(v.topic_with_msg_count.clone());
                (
                    min_start.min(v.start_time),
                    max_end.max(v.end_time),
                    count + v.topic_with_msg_count.message_count,
                    topics,
                )
            },
        );

    let tf_with_msg_count = write_tf_data(input, start_time, &mut mcap_writer, prec).unwrap();
    message_count += 1;
    topics_with_msg_count.push(tf_with_msg_count);

    mcap_writer.finish()?;

    let filename = output_path.as_path().file_name().unwrap();
    let info = RosbagInfo::new(
        filename.to_string(),
        start_time,
        end_time,
        message_count,
        compression,
        topics_with_msg_count,
    );
    let metadata_path = output_path
        .as_path()
        .parent()
        .unwrap()
        .join("metadata.yaml");
    info.save_metadata(metadata_path.as_path())
        .expect("Failed to save metadata.yaml");

    let duration = start.elapsed();
    println!("Export finished in {:?}", duration);

    Ok(())
}

pub fn fix_wheel_odom_from_rosbag<P: AsRef<Utf8Path>>(
    input: P,
    output: P,
    prec: &TimestampPrecision,
    drivetrain: &Drivetrain,
) -> Result<()> {
    // check that the input is either .mcap or folder/name.mcap
    let input_path = io::check_mcap_input_path(input)?;
    io::check_ijrr_output_path(output.as_ref())?;

    let mapped = io::map_mcap(input_path)?;

    let summary = mcap::Summary::read(&mapped)?.unwrap();
    let stats = summary.stats.unwrap();
    let msg_count: usize = stats.message_count.try_into().unwrap();

    let mut motor_vel: MotorVelocity;
    let mut diff_drive = DiffDrive::new(drivetrain, odom::DIFF_DRIVE_FREQUENCY);
    for message in tqdm(read::MessageStream::new(&mapped)?)
        .desc(Some("Processing input data"))
        .total(Some(msg_count))
    {
        let message = message?;

        let schema = message
            .channel
            .schema
            .as_ref()
            .ok_or(Err::<mcap::Schema, anyhow::Error>(anyhow!(
                "Message schema must exist."
            )))
            .unwrap();

        match schema.name.as_str() {
            "std_msgs/msg/Float64" => {
                let side = match message.channel.topic.as_str() {
                    "/warthog/platform/motor/right/status/velocity" => WheelSide::Right,
                    "/warthog/platform/motor/left/status/velocity" => WheelSide::Left,
                    _ => continue,
                };
                let (velocity, _consumed_byte_count) =
                    from_bytes::<f64, LittleEndian>(&message.data[4..])?;
                motor_vel = MotorVelocity {
                    velocity,
                    timestamp: message.publish_time,
                    side,
                };
                diff_drive.add_vel(motor_vel);
            }

            _ => {
                continue;
            }
        }
    }

    let mut odom_vector = DataVector::new("/warthog/platform/odom_fixed".to_string());
    let odoms = diff_drive.process_velocities();
    match odoms {
        Ok(odoms) => {
            odom_vector.add_data_vec(odoms);
            odom_vector
                .save(output.as_ref().join(format!("{}.csv", ODOM_FRAME_ID)), prec)
                .unwrap();
        }
        Err(e) => eprintln!("{}", e),
    }

    Ok(())
}
