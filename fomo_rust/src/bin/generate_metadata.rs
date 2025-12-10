use anyhow::Result;
use camino::{Utf8Path, Utf8PathBuf};
use clap::Parser;
use fomo_rust_sdk::fomo_rust_sdk::io;
use fomo_rust_sdk::fomo_rust_sdk::rosbag;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    input: Utf8PathBuf,
}

fn main() -> Result<()> {
    let args = Args::parse();
    generate_metadata(args.input)
}

fn generate_metadata<P: AsRef<Utf8Path>>(input: P) -> Result<()> {
    let input_path = io::check_mcap_input_path(input.as_ref())?;
    let mapped = io::map_mcap(input_path.to_path_buf())?;

    let metadata_path = rosbag::save_metadata_file(&mapped, &input_path)?;

    println!("Output save to {:?}", metadata_path);

    Ok(())
}
