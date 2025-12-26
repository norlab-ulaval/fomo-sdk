extern crate fomo_rust_sdk;
use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use fomo_rust_sdk::fomo_rust_sdk::{
    process_folder, sensors::timestamp::TimestampPrecision, SensorType,
};
use std::fs;
use std::time::Duration;

static NUMBER_OF_ITERS: usize = 10;
static NUMBER_OF_SECS: u64 = 500;

fn criterion_benchmark(c: &mut Criterion) {
    let input = "/tmp/benchmark";
    let output = "/tmp/benchmark.mcap";
    let output_file = "/tmp/benchmark/benchmark.mcap";

    // copy calib directory
    copy_dir_all("../data/calib", format!("{}/calib", input))
        .expect("Failed to copy calibration data");
    let sensors = SensorType::get_all();

    let mut group_read = c.benchmark_group("Write mcap format");
    group_read.sample_size(NUMBER_OF_ITERS);
    group_read.measurement_time(Duration::new(NUMBER_OF_SECS, 0));
    group_read.bench_function("write", |b| {
        b.iter_batched(
            || fs::remove_file(output_file), // Setup function (runs before each iteration)
            |_| {
                process_folder(
                    input,
                    output,
                    &sensors,
                    false,
                    &TimestampPrecision::MicroSecond,
                )
            },
            BatchSize::SmallInput,
        )
    });
    group_read.finish();
}

// Helper function to recursively copy directories
fn copy_dir_all(
    src: impl AsRef<std::path::Path>,
    dst: impl AsRef<std::path::Path>,
) -> std::io::Result<()> {
    fs::create_dir_all(&dst)?;
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let ty = entry.file_type()?;
        if ty.is_dir() {
            copy_dir_all(entry.path(), dst.as_ref().join(entry.file_name()))?;
        } else {
            fs::copy(entry.path(), dst.as_ref().join(entry.file_name()))?;
        }
    }
    Ok(())
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
