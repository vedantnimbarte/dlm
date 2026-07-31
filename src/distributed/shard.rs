//! Pipeline layer partitioning (`specs.md` §3.3 Pipeline Parallelism).

/// A contiguous half-open range of transformer layers `[start, end)` assigned to
/// one node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerShard {
    pub start: usize,
    pub end: usize,
}

impl LayerShard {
    /// Number of layers in the shard.
    pub fn len(&self) -> usize {
        self.end - self.start
    }

    /// Whether the shard is empty.
    pub fn is_empty(&self) -> bool {
        self.start >= self.end
    }
}

/// Partition `num_layers` into `num_shards` balanced contiguous ranges (earlier
/// shards absorb the remainder). Empty shards are dropped.
pub fn partition_layers(num_layers: usize, num_shards: usize) -> Vec<LayerShard> {
    let shards = num_shards.max(1);
    let base = num_layers / shards;
    let extra = num_layers % shards;

    let mut result = Vec::new();
    let mut start = 0usize;
    for s in 0..shards {
        let count = base + if s < extra { 1 } else { 0 };
        if count == 0 {
            continue;
        }
        result.push(LayerShard {
            start,
            end: start + count,
        });
        start += count;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partitions_evenly() {
        let s = partition_layers(8, 2);
        assert_eq!(
            s,
            vec![
                LayerShard { start: 0, end: 4 },
                LayerShard { start: 4, end: 8 }
            ]
        );
    }

    #[test]
    fn distributes_remainder_to_earlier_shards() {
        let s = partition_layers(10, 3);
        assert_eq!(
            s,
            vec![
                LayerShard { start: 0, end: 4 },
                LayerShard { start: 4, end: 7 },
                LayerShard { start: 7, end: 10 },
            ]
        );
        assert_eq!(s.iter().map(|x| x.len()).sum::<usize>(), 10);
    }

    #[test]
    fn covers_all_layers_without_gaps() {
        let s = partition_layers(80, 7);
        let mut next = 0;
        for shard in &s {
            assert_eq!(shard.start, next);
            next = shard.end;
        }
        assert_eq!(next, 80);
    }

    #[test]
    fn drops_empty_shards_when_more_shards_than_layers() {
        let s = partition_layers(2, 5);
        assert_eq!(s.len(), 2);
    }
}
