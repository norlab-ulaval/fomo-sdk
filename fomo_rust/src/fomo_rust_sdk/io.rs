use std::fs;

use anyhow::anyhow;
use camino::{Utf8Path, Utf8PathBuf};
use memmap::Mmap;

const CALIB_FOLDER_NAME: &str = "calib";

pub fn check_mcap_input_path<P: AsRef<Utf8Path>>(input: P) -> Result<Utf8PathBuf, anyhow::Error> {
    let mut input_path: Utf8PathBuf = input.as_ref().to_path_buf();
    if input.as_ref().is_file() && input.as_ref().extension().unwrap() != "mcap" {
        println!("{:?}", input.as_ref());
        return Err(anyhow!(
            "Input path must point to an .mcap file or a folder containing one."
        ));
    };
    if input.as_ref().is_dir() {
        let mut contains_mcap = false;
        for subfile in fs::read_dir(input.as_ref()).unwrap() {
            let path = subfile.unwrap().path();
            if path.extension().and_then(|s| s.to_str()) == Some("mcap") {
                contains_mcap = true;
                input_path = Utf8PathBuf::from_path_buf(path).expect("Path contains invalid UTF-8");
            }
        }
        if !contains_mcap {
            return Err(anyhow!(
                "Input path must point to an .mcap file or a folder containing one."
            ));
        }
    }

    Ok(input_path)
}

pub(crate) fn check_ijrr_output_path<P: AsRef<Utf8Path>>(output: P) -> Result<(), anyhow::Error> {
    if !output.as_ref().exists() {
        fs::create_dir_all(output.as_ref())?;
    }
    if !output.as_ref().is_dir() {
        return Err(anyhow!(
            "The output path {} is not a directory",
            output.as_ref()
        ));
    }
    Ok(())
}

pub(crate) fn check_ijrr_input_path<P: AsRef<Utf8Path>>(input: P) -> Result<(), anyhow::Error> {
    if !input.as_ref().exists() {
        return Err(anyhow!("Input path doesn't exists: {}", input.as_ref()));
    }
    if !input.as_ref().is_dir() {
        return Err(anyhow!(
            "Input path must be a directory: {}",
            input.as_ref()
        ));
    }

    let calib_path = input.as_ref().join(CALIB_FOLDER_NAME);
    if !calib_path.exists() {
        return Err(anyhow!(
            "Input path {} doesn't contain the calibration folder {}",
            input.as_ref(),
            CALIB_FOLDER_NAME
        ));
    }

    Ok(())
}

pub(crate) fn check_mcap_output_path<P: AsRef<Utf8Path>>(
    output: P,
) -> Result<Utf8PathBuf, anyhow::Error> {
    let output_path: Utf8PathBuf = output.as_ref().to_path_buf();
    let extension = output.as_ref().extension();
    if extension != Some("mcap") && extension != None {
        return Err(anyhow!(
            "Output path must either be an .mcap file, or a directory"
        ));
    }

    let directory_path = output.as_ref().parent().unwrap();
    let directory_name = output.as_ref().file_stem().unwrap();

    let output_path = match extension {
        Some("mcap") => {
            let mcap_dir = directory_path.join(directory_name);
            if !mcap_dir.exists() {
                fs::create_dir_all(mcap_dir)?;
            }
            directory_path
                .join(directory_name)
                .join(format!("{}.mcap", directory_name))
        }
        None => {
            if !output_path.exists() {
                fs::create_dir_all(output_path)?;
            }
            output.as_ref().join(format!("{}.mcap", directory_name))
        }
        _ => {
            return Err(anyhow!(
                "Output path must either be an .mcap file, or a directory"
            ))
        }
    };
    if output_path.exists() {
        return Err(anyhow!("Output path {} already exists", output_path));
    }

    Ok(output_path)
}

pub fn map_mcap<P: AsRef<Utf8Path>>(p: P) -> Result<Mmap, anyhow::Error> {
    let fd =
        fs::File::open(p.as_ref()).or(Err(anyhow!("Couldn't open MCAP file {}", p.as_ref())))?;
    unsafe { Mmap::map(&fd) }.or(Err(anyhow!("Couldn't map MCAP file")))
}
