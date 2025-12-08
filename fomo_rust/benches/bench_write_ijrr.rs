extern crate fomo_rust_sdk;
use criterion::{criterion_group, criterion_main, Criterion};
use fomo_rust_sdk::fomo_rust_sdk::{
    process_rosbag,
    sensors::{odom::Drivetrain, timestamp::TimestampPrecision},
    SensorType,
};
use std::time::Duration;

static NUMBER_OF_ITERS: usize = 10;
static NUMBER_OF_SECS: u64 = 500;

fn criterion_benchmark(c: &mut Criterion) {
    let input = "../data/benchmark/benchmark.mcap";
    let output = "/tmp/benchmark";
    let sensors = SensorType::get_all();

    let mut group_read = c.benchmark_group("Write IJRR format");
    group_read.sample_size(NUMBER_OF_ITERS);
    group_read.measurement_time(Duration::new(NUMBER_OF_SECS, 0));
    group_read.bench_function("write", |b| {
        b.iter(|| {
            process_rosbag(
                input,
                output,
                &sensors,
                &TimestampPrecision::MicroSecond,
                &Drivetrain::Tracks,
            )
        })
    });
    group_read.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
