use anyhow::Result;
use clap::Parser;
use fomo_rust_sdk::fomo_rust_sdk::{
    cli_common::CommonArgs,
    process_folder,
    sensors::timestamp::{Timestamp, TimestampPrecision},
};

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[command(flatten)]
    common: CommonArgs,

    /// Use Zstd compression. Default is false
    #[arg(short, long, action = clap::ArgAction::SetTrue)]
    pub compress: bool,
    /// Timestamp to start data extraction (in microseconds). Defaults to u64::MIN
    #[arg(short, long, default_value_t=u64::MIN)]
    pub start: u64,
    /// Timestamp to end data extraction (in microseconds). Defaults to u64::MAX
    #[arg(short, long, default_value_t=u64::MAX)]
    pub end: u64,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let compress = args.compress;
    let prec = &TimestampPrecision::MicroSecond;
    let sequence_start = Timestamp::new(args.start, prec);
    let sequence_end = Timestamp::new(args.end, prec);
    let (input, output, overwrite, sensors) = args.common.flatten_sensors();
    process_folder(
        &input,
        &output,
        &sensors,
        compress,
        overwrite,
        prec,
        sequence_start,
        sequence_end,
    )?;
    Ok(())
}
