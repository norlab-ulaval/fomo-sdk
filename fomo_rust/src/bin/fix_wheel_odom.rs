use anyhow::Result;
use camino::Utf8PathBuf;
use clap::Parser;
use fomo_rust_sdk::fomo_rust_sdk::cli_common::parse_drivetrain;
use fomo_rust_sdk::fomo_rust_sdk::cli_common::DRIVETRAIN_HELP;
use fomo_rust_sdk::fomo_rust_sdk::sensors::odom::Drivetrain;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    input: Utf8PathBuf,

    #[arg(short, long)]
    output: Utf8PathBuf,

    #[arg(short, long, value_parser = parse_drivetrain, help = DRIVETRAIN_HELP.as_str())]
    drivetrain: Drivetrain,
}

fn main() -> Result<()> {
    let args = Args::parse();
    fomo_rust_sdk::fomo_rust_sdk::fix_wheel_odom_from_rosbag(
        args.input,
        args.output,
        &fomo_rust_sdk::fomo_rust_sdk::sensors::timestamp::TimestampPrecision::MicroSecond,
        &args.drivetrain,
    )?;
    Ok(())
}
