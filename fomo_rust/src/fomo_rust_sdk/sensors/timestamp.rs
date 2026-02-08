#[derive(Clone, Debug, PartialEq, Copy)]
pub enum TimestampPrecision {
    NanoSecond,
    MicroSecond,
    MiliSecond,
}

#[derive(Clone, Debug, Copy)]
pub struct Timestamp {
    pub prec: TimestampPrecision,
    pub timestamp: u64,
}

impl Timestamp {
    /// Is self before other
    #[inline]
    pub fn is_before(&self, other: &Self) -> bool {
        assert_eq!(self.prec, other.prec);

        return self.timestamp < other.timestamp;
    }

    /// prec: input timestamp precision.
    #[inline]
    pub fn get_sec_nsec(&self) -> (i32, u32) {
        match self.prec {
            TimestampPrecision::NanoSecond => {
                let sec = self.timestamp / 1_000_000_000;
                let remainder = self.timestamp - sec * 1_000_000_000;
                (sec as i32, remainder as u32)
            }
            TimestampPrecision::MicroSecond => {
                let sec = self.timestamp / 1_000_000;
                let remainder = self.timestamp - sec * 1_000_000;
                let nsec = remainder * 1_000;
                (sec as i32, nsec as u32)
            }
            TimestampPrecision::MiliSecond => {
                let sec = self.timestamp / 1_000;
                let remainder = self.timestamp - sec * 1_000;
                let nsec = remainder * 1_000_000;
                (sec as i32, nsec as u32)
            }
        }
    }

    pub(crate) fn new(timestamp: u64, prec: &TimestampPrecision) -> Self {
        Self {
            timestamp,
            prec: prec.clone(),
        }
    }
}

impl PartialEq for Timestamp {
    fn eq(&self, other: &Self) -> bool {
        match (&self.prec, &other.prec) {
            (TimestampPrecision::NanoSecond, TimestampPrecision::NanoSecond)
            | (TimestampPrecision::MicroSecond, TimestampPrecision::MicroSecond)
            | (TimestampPrecision::MiliSecond, TimestampPrecision::MiliSecond) => {
                self.timestamp == other.timestamp
            }

            (TimestampPrecision::NanoSecond, TimestampPrecision::MicroSecond) => {
                self.timestamp / 1_000 == other.timestamp
            }

            (TimestampPrecision::NanoSecond, TimestampPrecision::MiliSecond) => {
                self.timestamp / 1_000_000 == other.timestamp
            }
            (TimestampPrecision::MicroSecond, TimestampPrecision::NanoSecond) => {
                self.timestamp == other.timestamp / 1_000
            }
            (TimestampPrecision::MicroSecond, TimestampPrecision::MiliSecond) => {
                self.timestamp / 1_000 == other.timestamp
            }
            (TimestampPrecision::MiliSecond, TimestampPrecision::NanoSecond) => {
                self.timestamp == other.timestamp / 1_000_000
            }
            (TimestampPrecision::MiliSecond, TimestampPrecision::MicroSecond) => {
                self.timestamp == other.timestamp / 1_000
            }
        }
    }
}

pub fn convert_timestamp(input: &Timestamp, output_prec: &TimestampPrecision) -> Timestamp {
    match (&input.prec, output_prec) {
        (TimestampPrecision::NanoSecond, TimestampPrecision::NanoSecond)
        | (TimestampPrecision::MicroSecond, TimestampPrecision::MicroSecond)
        | (TimestampPrecision::MiliSecond, TimestampPrecision::MiliSecond) => input.clone(),
        (TimestampPrecision::NanoSecond, TimestampPrecision::MicroSecond) => Timestamp {
            prec: output_prec.clone(),
            timestamp: input.timestamp / 1_000,
        },
        (TimestampPrecision::NanoSecond, TimestampPrecision::MiliSecond) => Timestamp {
            prec: output_prec.clone(),
            timestamp: input.timestamp / 1_000_000,
        },
        (TimestampPrecision::MicroSecond, TimestampPrecision::NanoSecond) => Timestamp {
            prec: output_prec.clone(),
            timestamp: input.timestamp * 1_000,
        },
        (TimestampPrecision::MicroSecond, TimestampPrecision::MiliSecond) => Timestamp {
            prec: output_prec.clone(),
            timestamp: input.timestamp / 1_000,
        },
        (TimestampPrecision::MiliSecond, TimestampPrecision::NanoSecond) => Timestamp {
            prec: output_prec.clone(),
            timestamp: input.timestamp * 1_000_000,
        },
        (TimestampPrecision::MiliSecond, TimestampPrecision::MicroSecond) => Timestamp {
            prec: output_prec.clone(),
            timestamp: input.timestamp * 1_000,
        },
    }
}

#[test]
fn partial_eq() {
    let input_nsec = Timestamp {
        prec: TimestampPrecision::NanoSecond,
        timestamp: 1748443589084976896,
    };
    let input_usec = Timestamp {
        prec: TimestampPrecision::MicroSecond,
        timestamp: 1748443589084976,
    };
    let input_msec = Timestamp {
        prec: TimestampPrecision::MiliSecond,
        timestamp: 1748443589084,
    };

    assert_eq!(input_nsec, input_nsec);
    assert_eq!(input_nsec, input_usec);
    assert_eq!(input_nsec, input_msec);

    assert_eq!(input_usec, input_usec);
    assert_eq!(input_usec, input_nsec);
    assert_eq!(input_usec, input_msec);

    assert_eq!(input_msec, input_msec);
    assert_eq!(input_msec, input_usec);
    assert_eq!(input_msec, input_nsec);
}

#[test]
fn convert_timestamps_test_nsec() {
    let input_nsec = Timestamp {
        prec: TimestampPrecision::NanoSecond,
        timestamp: 1748443589084976896,
    };
    let input_usec = Timestamp {
        prec: TimestampPrecision::MicroSecond,
        timestamp: 1748443589084976,
    };
    let input_msec = Timestamp {
        prec: TimestampPrecision::MiliSecond,
        timestamp: 1748443589084,
    };

    assert_eq!(
        convert_timestamp(&input_nsec, &TimestampPrecision::NanoSecond),
        input_nsec
    );

    assert_eq!(
        convert_timestamp(&input_nsec, &TimestampPrecision::MicroSecond),
        input_usec
    );

    assert_eq!(
        convert_timestamp(&input_usec, &TimestampPrecision::MiliSecond),
        input_msec
    );
}

#[test]
fn convert_timestamps_test_usec() {
    let input_nsec = Timestamp {
        prec: TimestampPrecision::NanoSecond,
        timestamp: 1748443589084976000,
    };
    let input_usec = Timestamp {
        prec: TimestampPrecision::MicroSecond,
        timestamp: 1748443589084976,
    };
    let input_msec = Timestamp {
        prec: TimestampPrecision::MiliSecond,
        timestamp: 1748443589084,
    };
    assert_eq!(
        convert_timestamp(&input_usec, &TimestampPrecision::MicroSecond),
        input_usec
    );

    assert_eq!(
        convert_timestamp(&input_usec, &TimestampPrecision::NanoSecond),
        input_nsec
    );

    assert_eq!(
        convert_timestamp(&input_usec, &TimestampPrecision::MiliSecond),
        input_msec
    );
}

#[test]
fn convert_timestamps_test_msec() {
    let input_nsec = Timestamp {
        prec: TimestampPrecision::NanoSecond,
        timestamp: 1748443589084000000,
    };
    let input_usec = Timestamp {
        prec: TimestampPrecision::MicroSecond,
        timestamp: 1748443589084000,
    };
    let input_msec = Timestamp {
        prec: TimestampPrecision::MiliSecond,
        timestamp: 1748443589084,
    };

    assert_eq!(
        convert_timestamp(&input_msec, &TimestampPrecision::MiliSecond),
        input_msec
    );

    assert_eq!(
        convert_timestamp(&input_msec, &TimestampPrecision::NanoSecond),
        input_nsec
    );

    assert_eq!(
        convert_timestamp(&input_msec, &TimestampPrecision::MicroSecond),
        input_usec
    );
}
