use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use super::audio::SplitTimestamp;

#[derive(Debug, Serialize, Deserialize)]
struct QosProfile {
    history: u8,
    depth: i32,
    reliability: u8,
    durability: u8,
    liveliness: u8,
    avoid_ros_namespace_conventions: bool,
    deadline: SplitTimestamp,
    lifespan: SplitTimestamp,
    liveliness_lease_duration: SplitTimestamp,
}

pub(crate) fn create_sensor_qos_metadata_string() -> String {
    let qos_profile = QosProfile {
        history: 1, // keep_last
        depth: 10,
        reliability: 1, // reliable
        durability: 2,  // volatile
        liveliness: 1,  // automatic
        avoid_ros_namespace_conventions: false,
        deadline: SplitTimestamp {
            sec: 854775807,
            nsec: 92233726,
        },
        lifespan: SplitTimestamp {
            sec: 854775807,
            nsec: 92233726,
        },
        liveliness_lease_duration: SplitTimestamp {
            sec: 854775807,
            nsec: 92233726,
        },
    };

    serde_yaml::to_string(&vec![qos_profile])
        .expect("Serde should convert QoS profile to yaml string")
}

pub(crate) fn create_sensor_qos_metadata() -> BTreeMap<String, String> {
    let mut qos = BTreeMap::new();
    qos.insert(
        "offered_qos_profiles".to_string(),
        create_sensor_qos_metadata_string(),
    );
    qos
}

pub(crate) fn create_tf_qos_metadata_string() -> String {
    let qos_profile = QosProfile {
        history: 1, // keep_last
        depth: 10,
        reliability: 1, // reliable
        durability: 1,  // transient_local
        liveliness: 1,  // automatic
        avoid_ros_namespace_conventions: false,
        deadline: SplitTimestamp {
            sec: 854775807,
            nsec: 92233726,
        },
        lifespan: SplitTimestamp {
            sec: 854775807,
            nsec: 92233726,
        },
        liveliness_lease_duration: SplitTimestamp {
            sec: 854775807,
            nsec: 92233726,
        },
    };

    serde_yaml::to_string(&vec![qos_profile])
        .expect("Serde should convert QoS profile to yaml string")
}

pub(crate) fn create_tf_qos_metadata() -> BTreeMap<String, String> {
    let mut qos = BTreeMap::new();
    qos.insert(
        "offered_qos_profiles".to_string(),
        create_tf_qos_metadata_string(),
    );
    qos
}
