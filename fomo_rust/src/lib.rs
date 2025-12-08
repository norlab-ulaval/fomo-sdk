use include_dir::{include_dir, Dir};

pub static DATA_DIR: Dir = include_dir!("$CARGO_MANIFEST_DIR/../fomo_sdk/data");

pub mod fomo_rust_sdk;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_dir_contains_calib() {
        assert!(DATA_DIR.get_dir("calib").is_some());
        assert!(DATA_DIR.get_dir("calib_to_ijrr").is_some());
    }
}
