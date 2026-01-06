use crate::fomo_rust_sdk::SensorType;
use anyhow::Result;
use camino::Utf8PathBuf;
use clap::{Args, ValueEnum};
use std::sync::LazyLock;

use super::sensors::odom::Drivetrain;

static SENSOR_HELP: LazyLock<String> = LazyLock::new(|| {
    // same logic as above
    let mut valid_options: Vec<String> = SensorType::value_variants()
        .iter()
        .filter_map(|v| {
            let value = v.to_possible_value()?;
            Some(value.get_name().to_string())
        })
        .collect();
    valid_options.push("all".to_string());
    valid_options.sort();

    format!(
        "Sensor types to process. Valid options: {}",
        valid_options.join(", ")
    )
});

pub static DRIVETRAIN_HELP: LazyLock<String> = LazyLock::new(|| {
    // same logic as above
    let valid_options: Vec<String> = Drivetrain::value_variants()
        .iter()
        .filter_map(|v| {
            let value = v.to_possible_value()?;
            Some(value.get_name().to_string())
        })
        .collect();

    format!(
        "Sensor types to process. Valid options: {}",
        valid_options.join(", ")
    )
});

pub fn parse_drivetrain(s: &str) -> Result<Drivetrain, String> {
    let a = Drivetrain::from_str(s, true).map_err(|_| {
        let valid_options: Vec<String> = Drivetrain::value_variants()
            .iter()
            .filter_map(|v| {
                let value = v.to_possible_value()?;
                let name = value.get_name();
                Some(name.to_string())
            })
            .collect();
        format!(
            "Invalid drive train type: {}. Valid options are: {:?}",
            s, valid_options
        )
    });
    return a;
}

pub fn parse_sensors(s: &str) -> Result<Vec<SensorType>, String> {
    if s.to_lowercase() == "all" {
        Ok(vec![
            SensorType::Navtech,
            SensorType::ZedXLeft,
            SensorType::ZedXRight,
            SensorType::Basler,
            SensorType::RoboSense,
            SensorType::Leishen,
            SensorType::Audio,
        ])
    } else {
        SensorType::from_str(s, true)
            .map(|sensor| vec![sensor])
            .map_err(|_| {
                let mut valid_options: Vec<String> = SensorType::value_variants()
                    .iter()
                    .filter_map(|v| {
                        let value = v.to_possible_value()?;
                        let name = value.get_name();
                        if name.to_lowercase() != "all" {
                            Some(name.to_string())
                        } else {
                            None
                        }
                    })
                    .collect();
                valid_options.push("all".to_string());
                valid_options.sort();
                format!(
                    "Invalid sensor type: {}. Valid options are: {:?}",
                    s, valid_options
                )
            })
    }
}

#[derive(Args, Debug)]
pub struct CommonArgs {
    #[arg(short, long)]
    pub input: Utf8PathBuf,

    #[arg(short, long)]
    pub output: Utf8PathBuf,

    #[arg(short, long,
        value_parser = parse_sensors,
        help = SENSOR_HELP.as_str()
    )]
    pub sensors: Vec<Vec<SensorType>>,
}

impl CommonArgs {
    pub fn flatten_sensors(self) -> (Utf8PathBuf, Utf8PathBuf, Vec<SensorType>) {
        let sensors: Vec<SensorType> = self
            .sensors
            .into_iter()
            .flatten()
            .collect::<std::collections::HashSet<SensorType>>()
            .into_iter()
            .collect();
        (self.input, self.output, sensors)
    }
}
