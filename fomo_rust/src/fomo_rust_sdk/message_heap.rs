use std::cmp::Ordering;

use super::data_writer::McapLogMessage;

#[derive(Debug)]
pub(crate) struct HeapItem {
    pub(crate) timestamp: u64,
    pub(crate) receiver_index: usize,
    pub(crate) message: McapLogMessage,
}

impl PartialEq for HeapItem {
    fn eq(&self, other: &Self) -> bool {
        self.timestamp == other.timestamp
    }
}
impl Eq for HeapItem {}

// Reverse ordering to make BinaryHeap act as a Min-Heap
impl PartialOrd for HeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for HeapItem {
    fn cmp(&self, other: &Self) -> Ordering {
        other.timestamp.cmp(&self.timestamp)
    }
}
