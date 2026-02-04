use anyhow::{anyhow, Result};
use camino::{Utf8Path, Utf8PathBuf};
use clap::Parser;
use fomo_rust_sdk::fomo_rust_sdk::sensors::{
    point_cloud::PointCloud, timestamp::TimestampPrecision,
};
use std::fs;
use tqdm::tqdm;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    input: Utf8PathBuf,

    #[arg(short, long)]
    output: Utf8PathBuf,
}

fn main() -> Result<()> {
    let args = Args::parse();
    convert_point_clouds(&args.input, &args.output, &TimestampPrecision::MicroSecond)?;
    Ok(())
}

fn convert_point_clouds<P: AsRef<Utf8Path>>(
    input: P,
    output: P,
    prec: &TimestampPrecision,
) -> Result<()> {
    let mut contains_csv = false;
    let mut contains_bin = false;
    let mut extension = "";
    if input.as_ref().is_dir() {
        for subfile in fs::read_dir(input.as_ref()).unwrap() {
            match subfile.unwrap().path().extension().and_then(|s| s.to_str()) {
                Some("bin") => contains_bin = true,
                Some("csv") => contains_csv = true,
                _ => {}
            }
        }

        match (contains_bin, contains_csv) {
            (true, true) => {
                return Err(anyhow!(
                    "Input path cannot contain both .csv and .bin files."
                ))
            }
            (true, false) => extension = "csv",
            (false, true) => extension = "bin",
            (false, false) => {
                return Err(anyhow!(
                    "Input path must contain either .csv or .bin files to convert."
                ))
            }
        }
    }

    if output.as_ref().exists() && !output.as_ref().is_dir() {
        return Err(anyhow!(
            "The output path {} is not a directory",
            output.as_ref()
        ));
    }

    if !output.as_ref().exists() {
        fs::create_dir_all(output.as_ref())?;
    }

    for subfile in tqdm(fs::read_dir(input.as_ref())?) {
        let filepath = subfile.as_ref().unwrap().path();
        match filepath.extension().and_then(|s| s.to_str()) {
            Some("bin") | Some("csv") => {
                let timestamp = filepath
                    .file_stem()
                    .unwrap()
                    .to_str()
                    .unwrap()
                    .parse::<u64>()?;
                let mut pc = PointCloud::new(timestamp, "lslidar".to_string(), prec); // TODO fix frame_id
                pc.data = PointCloud::load_points(filepath)?;
                let filename = format!("{}.{}", pc.header.get_timestamp(prec).timestamp, extension);
                pc.save(output.as_ref().join(filename), prec)?;
            }
            _ => {}
        }
    }

    // check if the input directory
    Ok(())
}
