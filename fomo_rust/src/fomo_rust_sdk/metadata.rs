use serde::Deserialize;

use super::sensors::basic::{RegionOfInterest, Twist};
use super::sensors::common::CsvSaveable;
use super::sensors::header::Header;
use super::sensors::timestamp::{convert_timestamp, Timestamp, TimestampPrecision};

pub(crate) const LEFT_VOLTAGE_FILE_NAME: &str = "voltage_left";
pub(crate) const RIGHT_VOLTAGE_FILE_NAME: &str = "voltage_right";
pub(crate) const LEFT_CURRENT_FILE_NAME: &str = "current_left";
pub(crate) const RIGHT_CURRENT_FILE_NAME: &str = "current_right";
pub(crate) const LEFT_VELOCITY_FILE_NAME: &str = "velocity_left";
pub(crate) const RIGHT_VELOCITY_FILE_NAME: &str = "velocity_right";
pub(crate) const LEFT_CMD_VELOCITY_FILE_NAME: &str = "cmd_velocity_left";
pub(crate) const RIGHT_CMD_VELOCITY_FILE_NAME: &str = "cmd_velocity_right";
pub(crate) const CMD_VEL_FILE_NAME: &str = "cmd_velocity";
pub(crate) const VN100_TEMPERATURE_FILE_NAME: &str = "vn100_temperature";
pub(crate) const VN100_PRESSURE_FILE_NAME: &str = "vn100_pressure";
pub(crate) const DPS310_DATA_FILE_NAME: &str = "dps310";
pub(crate) const BATTERY_LOGS_FILE_NAME: &str = "battery_logs";
pub(crate) const BASLER_PARAMS_FILE_NAME: &str = "basler_metadata";

impl CsvSaveable for TwistTimestamp {
    fn get_csv_headers() -> &'static str {
        "timestamp,lx,ly,lz,ax,ay,az"
    }

    fn to_csv_row(&self, prec: &TimestampPrecision) -> String {
        let timestamp = convert_timestamp(&self.t, prec).timestamp;
        let twist = &self.value;
        format!(
            "{},{},{},{},{},{},{}",
            timestamp,
            twist.linear.x,
            twist.linear.y,
            twist.linear.z,
            twist.angular.x,
            twist.angular.y,
            twist.angular.z
        )
    }
}

pub(crate) struct TwistTimestamp {
    t: Timestamp,
    value: Twist,
}

impl TwistTimestamp {
    pub fn new(t: Timestamp, value: Twist) -> Self {
        Self { t, value }
    }
}

pub(crate) struct F64Data {
    t: Timestamp,
    value: f64,
}

impl F64Data {
    pub fn new(t: Timestamp, value: f64) -> Self {
        Self { t, value }
    }
}

impl CsvSaveable for F64Data {
    fn get_csv_headers() -> &'static str {
        "timestamp,value"
    }

    fn to_csv_row(&self, prec: &TimestampPrecision) -> String {
        let timestamp = convert_timestamp(&self.t, prec).timestamp;
        format!("{},{}", timestamp, self.value)
    }
}

#[derive(Debug, Deserialize)]
pub(crate) struct TemperaturePressure {
    header: Header,
    data: f64,
    variance: f64,
}

impl CsvSaveable for TemperaturePressure {
    fn get_csv_headers() -> &'static str {
        "timestamp,value"
    }

    fn to_csv_row(&self, prec: &TimestampPrecision) -> String {
        let timestamp = self.header.get_timestamp(prec).timestamp;
        format!("{},{}", timestamp, self.data)
    }
}

#[derive(Debug, Deserialize)]
pub(crate) struct DPS310 {
    header: Header,
    c0: f64,
    c1: f64,
    c00: f64,
    c01: f64,
    c10: f64,
    c11: f64,
    c20: f64,
    c21: f64,
    c30: f64,
    raw_pressure: f64,
    raw_temperature: f64,
    scale_pressure: f64,
    scale_temperature: f64,
    pressure: f64,
    temperature: f64,
}

impl CsvSaveable for DPS310 {
    fn get_csv_headers() -> &'static str {
        "timestamp,c0,c1,c00,c01,c10,c11,c20,c21,c30,raw_pressure,raw_temperature,scale_pressure,scale_temperature,pressure,temperature"
    }

    fn to_csv_row(&self, prec: &TimestampPrecision) -> String {
        let timestamp = self.header.get_timestamp(prec).timestamp;
        format!(
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            timestamp,
            self.c0,
            self.c1,
            self.c00,
            self.c01,
            self.c10,
            self.c11,
            self.c20,
            self.c21,
            self.c30,
            self.raw_pressure,
            self.raw_temperature,
            self.scale_pressure,
            self.scale_temperature,
            self.pressure,
            self.temperature,
        )
    }
}

#[derive(Debug, Deserialize)]
pub(crate) struct BatteryLog {
    header: Header,
    pack_voltage: f32,
    pack_current: f32,

    state: i8,
    main_fault_reg: i8,
    soc: i8,
    remaining_capacity: i64,

    min_temp: f32,
    max_temp: f32,
    min_cell_volt: f32,
    max_cell_volt: f32,

    brick_volt_avg: f32,
    cell_volt_avg: f32,
    pack_temp_avg: f32,
    pack_imbalance: i64,
}

impl CsvSaveable for BatteryLog {
    fn get_csv_headers() -> &'static str {
        "timestamp,pack_voltage,pack_current,state,main_fault_reg,soc,remaining_capacity,min_temp,max_temp,min_cell_volt,max_cell_volt,brick_volt_avg,cell_volt_avg,pack_temp_avg,pack_imbalance"
    }

    fn to_csv_row(&self, prec: &TimestampPrecision) -> String {
        let timestamp = self.header.get_timestamp(prec).timestamp;
        format!(
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            timestamp,
            self.pack_voltage,
            self.pack_current,
            self.state,
            self.main_fault_reg,
            self.soc,
            self.remaining_capacity,
            self.min_temp,
            self.max_temp,
            self.min_cell_volt,
            self.max_cell_volt,
            self.brick_volt_avg,
            self.cell_volt_avg,
            self.pack_temp_avg,
            self.pack_imbalance
        )
    }
}

#[derive(Debug, Deserialize)]
pub(crate) struct BaslerParams {
    // Offset parameters
    offset_x: u32, // -20000 = Error
    offset_y: u32, // -20000 = Error
    reverse_x: bool,
    reverse_y: bool,

    // Image processing parameters
    black_level: i32,           // -10000 = error/not available
    pgi_mode: i32,              // -3 = Unknown, -2 = Error, -1 = Not available, 0 = Off, 1 = On
    demosaicing_mode: i32, // -3 = Unknown, -2 = Error, -1 = Not available, 0 = Simple, 1 = BaslerPGI
    noise_reduction: f32,  // -20000.0 = Error, -10000.0 = Not available
    sharpness_enhancement: f32, // -20000.0 = Error, -10000.0 = Not available

    // White balance parameters
    light_source_preset: i32, // -3 = Unknown, -2 = Error, -1 = Not available, 0 = Off, 1 = Daylight5000K, 2 = Daylight6500K, 3 = Tungsten2800K
    balance_white_auto: i32, // -3 = Unknown, -2 = Error, -1 = Not available, 0 = Off, 1 = Once, 2 = Continuous

    // Sensor parameters
    sensor_readout_mode: i32, // -3 = Unknown, -2 = Error, -1 = Not available, 0 = Normal, 1 = Fast
    acquisition_frame_count: i32, // -20000 = Error, -10000 = Not available

    // Trigger parameters
    trigger_selector: i32, // -3 = Unknown, -2 = Error, -1 = Not available, 0 = FrameStart, 1 = FrameBurstStart(USB)/AcquisitionStart(GigE)
    trigger_mode: i32,     // -3 = Unknown, -2 = Error, -1 = Not available, 0 = Off, 1 = On
    trigger_source: i32, // -3 = Unknown, -2 = Error, -1 = Not available, 0 = Software, 1 = Line1, 2 = Line3, 3 = Line4, 4 = Action1(Selected Gige)
    trigger_activation: i32, // -3 = Unknown, -2 = Error, -1 = Not available, 0 = RisingEdge, 1 = FallingEdge
    trigger_delay: f32,      // -20000.0 = Error, -10000.0 = Not available

    // User set parameters
    user_set_selector: i32, // -3 = Unknown, -2 = Error, -1 = Not available, 0 = Default, 1 = UserSet1, 2 = UserSet2, 3 = UserSet3, 4 = HighGain, 5 = AutoFunctions, 6 = ColorRaw
    user_set_default_selector: i32, // -3 = Unknown, -2 = Error, -1 = Not available, 0 = Default, 1 = UserSet1, 2 = UserSet2, 3 = UserSet3, 4 = HighGain, 5 = AutoFunctions, 6 = ColorRaw

    // Camera state parameters
    is_sleeping: bool,
    brightness: f32,
    exposure: f32,
    gain: f32,
    gamma: f32,
    binning_x: u32,
    binning_y: u32,
    temperature: f32, // Shows the camera temperature. If not available, then 0.0. USB uses DeviceTemperature and GigE TemperatureAbs parameters.
    max_num_buffer: i32, // -2 = Error, -1 = Not available

    // Region of Interest
    roi: RegionOfInterest,

    // Image encoding parameters
    available_image_encoding: Vec<String>,
    current_image_encoding: String,
    current_image_ros_encoding: String,

    // Status parameters
    success: bool,
    message: String,

    // PTP (Precision Time Protocol) parameters
    ptp_status: String,       // latched state of the PTP clock
    ptp_servo_status: String, // latched state of the clock servo
    ptp_offset: i64,          // ptp offset from master in ticks [ns]
}
#[derive(Debug)]
pub(crate) struct BaslerParamsTimestamped {
    timestamp: Timestamp,
    basler_params: BaslerParams,
}

impl BaslerParamsTimestamped {
    pub fn new(timestamp: Timestamp, basler_params: BaslerParams) -> Self {
        BaslerParamsTimestamped {
            timestamp,
            basler_params,
        }
    }
}

impl CsvSaveable for BaslerParamsTimestamped {
    fn get_csv_headers() -> &'static str {
        "timestamp,brightness,exposure,temperature,ptp_status,ptp_servo_status,ptp_offset"
    }

    fn to_csv_row(&self, prec: &TimestampPrecision) -> String {
        let timestamp = convert_timestamp(&self.timestamp, prec).timestamp;
        let entry = &self.basler_params;
        format!(
            "{},{},{},{},{},{},{}",
            timestamp,
            entry.brightness,
            entry.exposure,
            entry.temperature,
            entry.ptp_status,
            entry.ptp_servo_status,
            entry.ptp_offset
        )
    }
}
