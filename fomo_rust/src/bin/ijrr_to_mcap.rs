use anyhow::Result;
use clap::Parser;
use fomo_rust_sdk::fomo_rust_sdk::{
    cli_common::CommonArgs, process_folder, sensors::timestamp::TimestampPrecision,
};

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[command(flatten)]
    common: CommonArgs,

    /// Use Zstd compression. Default is true
    #[arg(short, long, action = clap::ArgAction::SetTrue)]
    pub compress: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let compress = args.compress;
    let (input, output, sensors) = args.common.flatten_sensors();
    process_folder(
        &input,
        &output,
        &sensors,
        compress,
        &TimestampPrecision::MicroSecond,
    )?;
    Ok(())
}
