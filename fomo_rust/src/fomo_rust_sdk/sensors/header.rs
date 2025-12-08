use super::timestamp::{convert_timestamp, Timestamp, TimestampPrecision};
use serde::{Deserialize, Serialize};

#[derive(Debug, PartialEq, Serialize, Deserialize, Clone)]
pub struct Header {
    pub stamp_sec: i32,
    pub stamp_nsec: u32,
    pub frame_id: String,
}

impl Header {
    /// Returns the timestamp of this [`Header`].
    #[inline]
    pub fn get_timestamp(&self, prec: &TimestampPrecision) -> Timestamp {
        let timestamp = Timestamp {
            timestamp: (self.stamp_sec as u64) * 1_000_000_000 + (self.stamp_nsec as u64),
            prec: TimestampPrecision::NanoSecond,
        };
        convert_timestamp(&timestamp, prec)
    }
}

/// prec: input timestamp precision.
#[inline]
pub fn get_sec_nsec(timestamp: u64, prec: &TimestampPrecision) -> (i32, u32) {
    let t = Timestamp {
        timestamp,
        prec: prec.clone(),
    };
    t.get_sec_nsec()
}
