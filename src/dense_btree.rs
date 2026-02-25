const KEYS_PER_NODE: usize = 8;

pub struct DenseBTree<T: Ord> {
    data: Vec<T>,
    internal: Vec<T>,
    height: usize,
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
            return DenseBTree { data, internal: Vec::new(), height: 0 };
        }

        let leaf_count = (n + KEYS_PER_NODE - 1) / KEYS_PER_NODE;
        if leaf_count <= 1 {
            return DenseBTree { data, internal: Vec::new(), height: 0 };
        }

        // height = smallest h such that 9^h >= leaf_count
        let mut height = 0;
        let mut span = 1usize;
        let mut bottom_span = 1usize;
        while span < leaf_count {
            span *= KEYS_PER_NODE;
            bottom_span += span;
            height += 1;
        }
        bottom_span -= span;

        // Pre-fill with max value (padding)
        let mut internal = vec![data[n - 1].clone(); bottom_span * KEYS_PER_NODE];
        // Fill internal nodes with max key of each child.
        // Each data[ii] writes to its bottom-level parent key, then propagates
        // upward only when it's the last key in a node (key_idx == 7), since
        // that means it's the max of the entire node's subtree.
        for ii in 0..data.len() {
            let datum = &data[ii];
            let mut idx = bottom_span - 1 + ii / KEYS_PER_NODE;
            loop {
                internal[idx] = datum.clone();
                if idx % KEYS_PER_NODE != KEYS_PER_NODE - 1 {
                    break;
                }
                match (idx / KEYS_PER_NODE).checked_sub(1) {
                    Some(parent) => idx = parent,
                    None => break,
                }
            }
        }
        DenseBTree { data, internal, height }
    }

    pub fn index(&self, value: &T) -> Result<usize, usize> {
        let n = self.data.len();
        if n == 0 { return Err(0); }
        if *value > self.data[n - 1] { return Err(n); }

        // Route through internal levels
        let mut base = 0usize;
        for level in 0..self.height {
            let keys = &self.internal[base..base + KEYS_PER_NODE];
            let mut k = 0;
            while k < KEYS_PER_NODE && *value > keys[k] {
                k += 1;
            }
            let child_idx = (base + k + 1) * KEYS_PER_NODE;
            base = child_idx;
        }

        base -= self.internal.len();  // convert to leaf index

        // idx = leaf chunk index
        let chunk_start = base;
        let chunk_end = (chunk_start + KEYS_PER_NODE).min(n);
        
        for i in chunk_start..chunk_end {
            match self.data[i].cmp(value) {
            std::cmp::Ordering::Equal => return Ok(i),
            std::cmp::Ordering::Greater => return Err(i),
            std::cmp::Ordering::Less => {}
            }
        }
        Err(chunk_end)
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
        for &n in &[8u32, 9, 16, 17, 64, 65, 72, 73, 80, 81, 100, 128, 255, 256] {
            let vals: Vec<u32> = (0..n).collect();
            let tree = DenseBTree::from_sorted(&vals);
            for i in 0..n {
                assert_eq!(tree.index(&i), Ok(i as usize), "n={n}, i={i}");
            }
            assert_eq!(tree.index(&n), Err(n as usize), "n={n}, past end");
        }
    }
}
