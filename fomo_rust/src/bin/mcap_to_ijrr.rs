use anyhow::Result;
use clap::Parser;
use fomo_rust_sdk::fomo_rust_sdk::cli_common::{parse_drivetrain, CommonArgs, DRIVETRAIN_HELP};
use fomo_rust_sdk::fomo_rust_sdk::sensors::odom::Drivetrain;
use fomo_rust_sdk::fomo_rust_sdk::{process_rosbag, sensors::timestamp::TimestampPrecision};

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[command(flatten)]
    common: CommonArgs,

    #[arg(short, long, value_parser = parse_drivetrain, help = DRIVETRAIN_HELP.as_str())]
    drivetrain: Drivetrain,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let (input, output, sensors) = args.common.flatten_sensors();
    process_rosbag(
        &input,
        &output,
        &sensors,
        &TimestampPrecision::MicroSecond,
        &args.drivetrain,
    )?;
    Ok(())
}
