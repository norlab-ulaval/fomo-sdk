use mcap;
use mcap::Writer;
use std::fs::File;
use std::{collections::BTreeMap, io::BufWriter};

pub(crate) fn create_schema_channel(
    mcap_writer: &mut Writer<BufWriter<File>>,
    schema_name: &str,
    schema_def: &[u8],
    topic_name: &str,
    metadata: &BTreeMap<String, String>,
) -> Result<(u16, u16), Box<dyn std::error::Error>> {
    let schema_id = mcap_writer.add_schema(schema_name, "ros2msg", schema_def)?;
    Ok((
        schema_id,
        mcap_writer.add_channel(schema_id, topic_name, "cdr", metadata)?,
    ))
}
