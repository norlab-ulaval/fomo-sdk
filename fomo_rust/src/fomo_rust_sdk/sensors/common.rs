use super::timestamp::TimestampPrecision;
use std::io::Write;
use std::{fs, io::BufWriter};

pub(crate) trait CsvSaveable {
    fn get_csv_headers() -> &'static str;
    fn to_csv_row(&self, prec: &TimestampPrecision) -> String;
}

pub(crate) struct DataVector<T> {
    entries: Vec<T>,
    pub(crate) topic: String,
}

impl<T> DataVector<T>
where
    T: CsvSaveable,
{
    pub fn new(topic: String) -> Self {
        Self {
            entries: Vec::new(),
            topic,
        }
    }

    pub fn add(&mut self, entry: T) {
        self.entries.push(entry);
    }

    pub fn add_data_vec(&mut self, data_vec: Vec<T>) {
        self.entries.extend(data_vec);
    }

    pub fn get_topic(&self) -> &str {
        &self.topic
    }

    pub fn save<P: AsRef<std::path::Path>>(
        &self,
        path: P,
        prec: &TimestampPrecision,
    ) -> Result<(), Box<(dyn std::error::Error + 'static)>> {
        if self.entries.is_empty() {
            eprintln!("No data to save to {}", path.as_ref().display());
            return Ok(());
        }
        let file = fs::File::create(path)?;
        let mut writer = BufWriter::new(file);
        let first_line = T::get_csv_headers();
        writeln!(writer, "{}", first_line)?;
        for entry in &self.entries {
            let row = entry.to_csv_row(prec);
            writeln!(writer, "{}", row)?;
        }
        Ok(())
    }
}
