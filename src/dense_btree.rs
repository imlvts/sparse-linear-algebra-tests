const KEYS_PER_NODE: usize = 8;
const BTREE_BRANCHES: usize = KEYS_PER_NODE + 1;

pub struct DenseBTree<T: Ord> {
    data: Vec<T>,
    internal: Vec<T>,
    height: usize,
    level_offsets: Vec<usize>,
}

impl<T: Ord + Clone> DenseBTree<T> {
    pub fn new(values: &mut [T]) -> Self {
        values.sort();
        Self::from_sorted(values)
    }

    pub fn from_sorted(values: &[T]) -> Self {
        let data: Vec<T> = values.to_vec();
        let n = data.len();

        if n == 0 {
            return DenseBTree { data, internal: Vec::new(), height: 0, level_offsets: Vec::new() };
        }

        let leaf_count = (n + KEYS_PER_NODE - 1) / KEYS_PER_NODE;
        if leaf_count <= 1 {
            return DenseBTree { data, internal: Vec::new(), height: 0, level_offsets: Vec::new() };
        }

        // height = smallest h such that 9^h >= leaf_count
        let mut height = 0;
        let mut span = 1usize;
        while span < leaf_count {
            span = span.saturating_mul(BTREE_BRANCHES);
            height += 1;
        }

        // Level sizes (top-down: level 0 = root) and byte offsets in node-count
        let mut level_offsets = Vec::with_capacity(height);
        let mut offset = 0usize;
        let mut s = span;
        for _ in 0..height {
            let level_size = (leaf_count + s - 1) / s;
            level_offsets.push(offset);
            offset += level_size;
            s /= BTREE_BRANCHES;
        }
        let num_internal_nodes = offset;

        // Pre-fill with max value (padding)
        let mut internal = vec![data[n - 1].clone(); num_internal_nodes * KEYS_PER_NODE];

        // Fill separator keys: key[k] = max element in child k's leaf range
        let mut child_stride = span / BTREE_BRANCHES;
        s = span;
        for level in 0..height {
            let level_off = level_offsets[level];
            let level_size = if level + 1 < height {
                level_offsets[level + 1] - level_off
            } else {
                num_internal_nodes - level_off
            };

            for i in 0..level_size {
                let base = (level_off + i) * KEYS_PER_NODE;
                let node_first_chunk = i * s;

                for k in 0..KEYS_PER_NODE {
                    let child_first = node_first_chunk + k * child_stride;
                    if child_first >= leaf_count {
                        break;
                    }
                    let child_end = (child_first + child_stride).min(leaf_count);
                    let last_data_idx = (child_end * KEYS_PER_NODE).min(n) - 1;
                    internal[base + k] = data[last_data_idx].clone();
                }
            }

            s = child_stride;
            child_stride /= BTREE_BRANCHES;
        }

        DenseBTree { data, internal, height, level_offsets }
    }

    pub fn index(&self, value: &T) -> Result<usize, usize> {
        let n = self.data.len();
        if n == 0 { return Err(0); }
        if *value > self.data[n - 1] { return Err(n); }

        // Route through internal levels
        let mut idx = 0usize;
        for level in 0..self.height {
            let base = (self.level_offsets[level] + idx) * KEYS_PER_NODE;
            let keys = &self.internal[base..base + KEYS_PER_NODE];
            let mut k = 0;
            while k < KEYS_PER_NODE && *value > keys[k] {
                k += 1;
            }
            idx = idx * BTREE_BRANCHES + k;
        }

        // idx = leaf chunk index
        let chunk_start = idx * KEYS_PER_NODE;
        let chunk_end = (chunk_start + KEYS_PER_NODE).min(n);

        match self.data[chunk_start..chunk_end].binary_search(value) {
            Ok(i) => Ok(chunk_start + i),
            Err(i) => Err(chunk_start + i),
        }
    }

    pub fn len(&self) -> usize { self.data.len() }
    pub fn is_empty(&self) -> bool { self.data.is_empty() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty() {
        let tree = DenseBTree::<u32>::from_sorted(&[]);
        assert_eq!(tree.len(), 0);
        assert_eq!(tree.index(&0), Err(0));
    }

    #[test]
    fn test_single() {
        let tree = DenseBTree::from_sorted(&[42]);
        assert_eq!(tree.index(&42), Ok(0));
        assert_eq!(tree.index(&0), Err(0));
        assert_eq!(tree.index(&100), Err(1));
    }

    #[test]
    fn test_one_full_node() {
        let vals: Vec<u32> = (0..8).collect();
        let tree = DenseBTree::from_sorted(&vals);
        for i in 0..8 {
            assert_eq!(tree.index(&i), Ok(i as usize), "searching for {i}");
        }
        assert_eq!(tree.index(&100), Err(8));
    }

    #[test]
    fn test_two_levels() {
        let vals: Vec<u32> = (0..20).collect();
        let tree = DenseBTree::from_sorted(&vals);
        for i in 0..20 {
            assert_eq!(tree.index(&i), Ok(i as usize), "searching for {i}");
        }
    }

    #[test]
    fn test_matches_binary_search() {
        for n in 0..=200 {
            let vals: Vec<u32> = (0..n).map(|x| x * 3).collect();
            let tree = DenseBTree::from_sorted(&vals);
            assert_eq!(tree.len(), n as usize);

            for query in 0..=(n * 3 + 5) {
                let expected = vals.binary_search(&query);
                let got = tree.index(&query);
                assert_eq!(
                    got, expected,
                    "n={n}, query={query}: expected {expected:?}, got {got:?}"
                );
            }
        }
    }

    #[test]
    fn test_new_sorts() {
        let mut vals = vec![5u32, 3, 8, 1, 9, 2, 7, 4, 6, 0];
        let tree = DenseBTree::new(&mut vals);
        for i in 0..10 {
            assert_eq!(tree.index(&i), Ok(i as usize));
        }
    }

    #[test]
    fn test_large() {
        let vals: Vec<u32> = (0..10000).collect();
        let tree = DenseBTree::from_sorted(&vals);
        for i in 0..10000 {
            assert_eq!(tree.index(&i), Ok(i as usize));
        }
        assert_eq!(tree.index(&10000), Err(10000));
    }

    #[test]
    fn test_exact_node_boundaries() {
        for &n in &[8u32, 9, 16, 17, 72, 73, 80, 81, 100, 128, 255, 256] {
            let vals: Vec<u32> = (0..n).collect();
            let tree = DenseBTree::from_sorted(&vals);
            for i in 0..n {
                assert_eq!(tree.index(&i), Ok(i as usize), "n={n}, i={i}");
            }
            assert_eq!(tree.index(&n), Err(n as usize), "n={n}, past end");
        }
    }
}
