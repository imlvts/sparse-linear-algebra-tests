use std::marker::PhantomData;
const KEYS_PER_NODE: usize = 16;
const MAX_BTREE_HEIGHT: usize = 8;

/// Compute compact level structure for n data elements.
/// Returns (height, level_sizes[0..height], level_starts[0..height]).
/// Level 0 = root, level height-1 = bottom internal level.
/// Each level has only ceil(children_below / KPN) nodes.
fn compute_levels(n: usize) -> (usize, [usize; MAX_BTREE_HEIGHT], [usize; MAX_BTREE_HEIGHT]) {
    let kpn = KEYS_PER_NODE;
    let leaf_count = (n + kpn - 1) / kpn;

    let mut sizes = [0usize; MAX_BTREE_HEIGHT];
    let mut starts = [0usize; MAX_BTREE_HEIGHT];

    if leaf_count <= 1 {
        return (0, sizes, starts);
    }

    // Build level sizes bottom-up, then reverse to top-down
    let mut stack = [0usize; MAX_BTREE_HEIGHT];
    let mut h = 0;
    let mut count = leaf_count;
    while count > 1 {
        count = (count + kpn - 1) / kpn;
        stack[h] = count;
        h += 1;
    }
    let height = h;

    for i in 0..height {
        sizes[i] = stack[height - 1 - i];
    }

    let mut start = 0;
    for i in 0..height {
        starts[i] = start;
        start += sizes[i] * kpn;
    }

    (height, sizes, starts)
}

/// A compact, cache-friendly B-tree for sorted data with O(log_K n) lookup.
///
/// Data is stored in a single flat array: `[internal nodes... | data...]`.
/// Internal nodes form a compact K-ary tree (K = [`KEYS_PER_NODE`]) laid out
/// level-by-level (BFS order), where each node stores K separator keys.
/// Only the nodes actually needed are allocated — no wasted padding for
/// incomplete subtrees. Overhead is ~6-7% for typical sizes.
///
/// The `index` method is a drop-in replacement for [`slice::binary_search`],
/// returning `Ok(pos)` on exact match or `Err(insert_pos)` on miss.
///
/// Generic over storage `V`: owns data as `Vec<T>`, or borrows as `&[T]`
/// (used internally by [`DenseBTreeList`] for zero-copy views).
///
/// # Examples
///
/// ```
/// use sparse_linear_algebra_tests::dense_btree::DenseBTree;
///
/// // Build from sorted data
/// let tree = DenseBTree::from_sorted(&[10, 20, 30, 40, 50]);
/// assert_eq!(tree.index(&30), Ok(2));
/// assert_eq!(tree.index(&25), Err(2)); // insert position
/// assert_eq!(tree.data(), &[10, 20, 30, 40, 50]);
///
/// // Build from unsorted data (sorts in place)
/// let mut vals = vec![50, 10, 40, 20, 30];
/// let tree = DenseBTree::new(&mut vals);
/// assert_eq!(tree.index(&40), Ok(3));
/// assert_eq!(tree.len(), 5);
/// ```
#[derive(Clone, Debug)]
pub struct DenseBTree<T: Ord, V=Vec<T>>
    where V: AsRef<[T]>
{
    nodes: V,
    internal_len: usize,
    height: usize,
    level_starts: [usize; MAX_BTREE_HEIGHT],
    _marker: PhantomData<T>,
}

impl<'a, T: Ord> DenseBTree<T, &'a [T]> {
    fn from_slice(slice: &'a [T], internal_len: usize,
                  height: usize, level_starts: [usize; MAX_BTREE_HEIGHT]) -> Self {
        DenseBTree { nodes: slice, internal_len, height, level_starts, _marker: PhantomData }
    }
}

impl<T: Ord + Clone> DenseBTree<T, Vec<T>> {
    /// Build a tree from unsorted data. Sorts `values` in place.
    pub fn new(values: &mut [T]) -> Self {
        values.sort();
        Self::from_sorted(values)
    }

    /// Build a tree from data that is already sorted in ascending order.
    ///
    /// # Panics
    ///
    /// Does not panic on unsorted input, but `index` results will be incorrect.
    pub fn from_sorted(values: &[T]) -> Self {
        let mut nodes = Vec::new();
        let internal_len = Self::extend_from_sorted(&mut nodes, values);
        let (height, _sizes, level_starts) = compute_levels(values.len());
        DenseBTree { nodes, internal_len, height, level_starts, _marker: PhantomData }
    }
}

impl <T: Ord + Clone, V> DenseBTree<T, V>
    where V: AsRef<[T]>
{
    fn extend_from_sorted(target: &mut Vec<T>, values: &[T]) -> usize {
        let n = values.len();

        let kpn = KEYS_PER_NODE;
        let leaf_count = (n + kpn - 1) / kpn;
        if leaf_count <= 1 {
            target.extend_from_slice(values);
            return 0;
        }

        let (height, sizes, starts) = compute_levels(n);
        let internal_len = starts[height - 1] + sizes[height - 1] * kpn;

        let max_val = values[n - 1].clone();
        let mut nodes = vec![max_val; internal_len];

        // Step 1: Fill bottom internal level from leaf chunks
        let bottom = height - 1;
        for j in 0..sizes[bottom] {
            let base = starts[bottom] + j * kpn;
            for k in 0..kpn {
                let leaf_chunk = j * kpn + k;
                let end = ((leaf_chunk + 1) * kpn).min(n);
                // For chunks beyond leaf_count, end = n → values[n-1] = max (padding)
                nodes[base + k] = values[end - 1].clone();
            }
        }

        // Step 2: Fill upper levels bottom-to-top
        for level in (0..bottom).rev() {
            for j in 0..sizes[level] {
                let p_base = starts[level] + j * kpn;
                for k in 0..kpn {
                    let child_node = j * kpn + k;
                    if child_node < sizes[level + 1] {
                        let c_base = starts[level + 1] + child_node * kpn;
                        nodes[p_base + k] = nodes[c_base + kpn - 1].clone();
                    }
                    // else: stays as max_val (padding)
                }
            }
        }

        target.extend_from_slice(&nodes);
        target.extend_from_slice(values);
        internal_len
    }

    /// Search for `value` in the tree.
    ///
    /// Returns `Ok(i)` if `data()[i] == value`, or `Err(i)` where `i` is the
    /// sorted insertion point — identical semantics to [`slice::binary_search`].
    ///
    /// # Examples
    ///
    /// ```
    /// use sparse_linear_algebra_tests::dense_btree::DenseBTree;
    ///
    /// let tree = DenseBTree::from_sorted(&[2, 4, 6, 8]);
    /// assert_eq!(tree.index(&4), Ok(1));
    /// assert_eq!(tree.index(&5), Err(2));
    /// assert_eq!(tree.index(&0), Err(0));
    /// assert_eq!(tree.index(&9), Err(4));
    /// ```
    pub fn index(&self, value: &T) -> Result<usize, usize> {
        let nodes = self.nodes.as_ref();
        let n = nodes.len() - self.internal_len;
        if n == 0 { return Err(0); }
        if *value > nodes[nodes.len() - 1] { return Err(n); }

        // Route through internal levels using precomputed level starts
        let mut node_idx = 0usize;
        for level in 0..self.height {
            let base = self.level_starts[level] + node_idx * KEYS_PER_NODE;
            let keys = &nodes[base..base + KEYS_PER_NODE];
            // Scalar search for the child index
            let k = (0..KEYS_PER_NODE).filter(|&k| *value > keys[k]).count();
            node_idx = node_idx * KEYS_PER_NODE + k;
        }

        // node_idx is now the leaf chunk index
        let chunk_start = self.internal_len + node_idx * KEYS_PER_NODE;
        let chunk_end = (chunk_start + KEYS_PER_NODE).min(nodes.len());

        for i in chunk_start..chunk_end {
            match nodes[i].cmp(value) {
            std::cmp::Ordering::Equal => return Ok(i - self.internal_len),
            std::cmp::Ordering::Greater => return Err(i - self.internal_len),
            std::cmp::Ordering::Less => {}
            }
        }
        Err(chunk_end - self.internal_len)
    }

    /// Number of data elements (excludes internal nodes).
    pub fn len(&self) -> usize { self.nodes.as_ref().len() - self.internal_len }

    /// True if the tree contains no data elements.
    pub fn is_empty(&self) -> bool { self.nodes.as_ref().len() == self.internal_len }

    /// The sorted data slice (excludes internal nodes).
    pub fn data(&self) -> &[T] { &self.nodes.as_ref()[self.internal_len..] }

    /// The internal separator nodes (for debugging/analysis).
    pub fn internal_nodes(&self) -> &[T] { &self.nodes.as_ref()[..self.internal_len] }
}

#[derive(Clone, Debug)]
struct NodeEntry {
    offset: usize,
    internal_len: usize,
    total_len: usize,
    /// Cumulative count of data elements before this row.
    data_start: usize,
    height: usize,
    level_starts: [usize; MAX_BTREE_HEIGHT],
}

/// A collection of [`DenseBTree`]s packed into a single flat allocation.
///
/// Each row is an independent B-tree, but all internal nodes and data share
/// one contiguous `Vec<T>`. This avoids per-row heap allocations and improves
/// cache locality when iterating across rows (e.g., sparse matrix row scans).
///
/// Tracks cumulative data offsets so a parallel values array can be indexed
/// by `data_start(row)` without a separate row-pointer array.
///
/// # Examples
///
/// ```
/// use sparse_linear_algebra_tests::dense_btree::DenseBTreeList;
///
/// let mut list = DenseBTreeList::new();
/// list.add_from_sorted(&[10, 20, 30]);
/// list.add_from_sorted(&[5, 15]);
/// list.add_from_sorted(&[]);
///
/// assert_eq!(list.row_count(), 3);
/// assert_eq!(list.data(0), &[10, 20, 30]);
/// assert_eq!(list.data(1), &[5, 15]);
/// assert_eq!(list.data(2), &[]);
///
/// // Search within a specific row
/// assert_eq!(list.index(0, &20), Ok(1));
/// assert_eq!(list.index(1, &10), Err(1));
///
/// // Cumulative data offsets for parallel value indexing
/// assert_eq!(list.data_start(0), 0);
/// assert_eq!(list.data_start(1), 3);
/// assert_eq!(list.data_start(2), 5);
/// assert_eq!(list.total_data_len(), 5);
/// ```
#[derive(Clone, Debug)]
pub struct DenseBTreeList<T: Ord + Clone> {
    index: Vec<NodeEntry>,
    nodes: Vec<T>,
}

impl<T: Ord + Clone> DenseBTreeList<T> {
    pub fn new() -> Self {
        DenseBTreeList { index: Vec::new(), nodes: Vec::new() }
    }

    /// Append a new row from unsorted data. Sorts `values` in place.
    pub fn add_from(&mut self, values: &mut [T]) {
        values.sort();
        self.add_from_sorted(values);
    }

    /// Append a new row from data that is already sorted in ascending order.
    pub fn add_from_sorted(&mut self, values: &[T]) {
        let offset = self.nodes.len();
        let data_start = self.total_data_len();
        let internal_len = DenseBTree::<T, &[T]>::extend_from_sorted(&mut self.nodes, values);
        let total_len = self.nodes.len() - offset;
        let (height, _sizes, level_starts) = compute_levels(values.len());
        self.index.push(NodeEntry { offset, internal_len, total_len, data_start, height, level_starts });
    }

    fn tree(&self, row_idx: usize) -> DenseBTree<T, &[T]> {
        let entry = &self.index[row_idx];
        DenseBTree::from_slice(
            &self.nodes[entry.offset..entry.offset + entry.total_len],
            entry.internal_len,
            entry.height,
            entry.level_starts,
        )
    }

    /// Search for `value` within row `row_idx`. See [`DenseBTree::index`].
    pub fn index(&self, row_idx: usize, value: &T) -> Result<usize, usize> {
        self.tree(row_idx).index(value)
    }

    /// The sorted data slice for row `row_idx`.
    pub fn data(&self, row_idx: usize) -> &[T] {
        let entry = &self.index[row_idx];
        &self.nodes[entry.offset + entry.internal_len..entry.offset + entry.total_len]
    }

    /// Number of rows in the list.
    pub fn row_count(&self) -> usize { self.index.len() }

    /// Cumulative data offset for row `row_idx` (for indexing into a parallel values array).
    pub fn data_start(&self, row_idx: usize) -> usize {
        self.index[row_idx].data_start
    }

    /// Total number of data elements across all rows.
    pub fn total_data_len(&self) -> usize {
        match self.index.last() {
            Some(e) => e.data_start + e.total_len - e.internal_len,
            None => 0,
        }
    }
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
    fn print_overhead_csv() {
        println!("n,internal_len,total_len");
        for n in 1..=10000u32 {
            let vals: Vec<u32> = (0..n).collect();
            let tree = DenseBTree::from_sorted(&vals);
            println!("{},{},{}", n, tree.internal_nodes().len(), tree.nodes.len());
        }
    }

    #[test]
    #[cfg(feature = "long-tests")]
    fn bench_btree_vs_binary_search() {
        use std::time::Instant;
        use rand::prelude::*;
        use rand::SeedableRng;

        let mut rng = StdRng::from_seed([7; 32]);
        let sizes: &[usize] = &[16, 32, 64, 128, 256, 512, 1024, 4096, 16384, 65536];
        let queries_per_run = 1_000_000;

        println!();
        println!("{:<8} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12}   {:>7} {:>7}",
            "n", "bs_unif_ns", "bt_unif_ns", "bs_skew_ns", "bt_skew_ns",
            "bs_miss_ns", "bt_miss_ns", "unif×", "skew×");
        println!("{}", "-".repeat(110));

        for &n in sizes {
            // Build sorted data with gaps (values 0, 3, 6, ... to allow misses)
            let vals: Vec<u32> = (0..n as u32).map(|x| x * 3).collect();
            let tree = DenseBTree::from_sorted(&vals);
            let max_val = vals[n - 1];

            // Uniform random queries (mix of hits and misses)
            let uniform_queries: Vec<u32> = (0..queries_per_run)
                .map(|_| rng.random_range(0..=max_val))
                .collect();

            // Skewed queries: 80% from the first 10% of the range
            let hot_end = max_val / 10;
            let skewed_queries: Vec<u32> = (0..queries_per_run)
                .map(|_| {
                    if rng.random_range(0.0..1.0) < 0.8 {
                        rng.random_range(0..=hot_end)
                    } else {
                        rng.random_range(0..=max_val)
                    }
                })
                .collect();

            // Miss-only queries (odd numbers, never in data)
            let miss_queries: Vec<u32> = (0..queries_per_run)
                .map(|_| rng.random_range(0..max_val) | 1) // ensure odd
                .collect();

            // Warmup
            let mut sink = 0usize;
            for q in uniform_queries.iter().take(1000) {
                sink ^= vals.binary_search(q).unwrap_or(0);
                sink ^= tree.index(q).unwrap_or(0);
            }

            // Bench: binary search uniform
            let t0 = Instant::now();
            for q in &uniform_queries {
                sink ^= vals.binary_search(q).unwrap_or(0);
            }
            let bs_unif = t0.elapsed().as_nanos() as f64 / queries_per_run as f64;

            // Bench: btree uniform
            let t0 = Instant::now();
            for q in &uniform_queries {
                sink ^= tree.index(q).unwrap_or(0);
            }
            let bt_unif = t0.elapsed().as_nanos() as f64 / queries_per_run as f64;

            // Bench: binary search skewed
            let t0 = Instant::now();
            for q in &skewed_queries {
                sink ^= vals.binary_search(q).unwrap_or(0);
            }
            let bs_skew = t0.elapsed().as_nanos() as f64 / queries_per_run as f64;

            // Bench: btree skewed
            let t0 = Instant::now();
            for q in &skewed_queries {
                sink ^= tree.index(q).unwrap_or(0);
            }
            let bt_skew = t0.elapsed().as_nanos() as f64 / queries_per_run as f64;

            // Bench: binary search miss-only
            let t0 = Instant::now();
            for q in &miss_queries {
                sink ^= vals.binary_search(q).unwrap_or(0);
            }
            let bs_miss = t0.elapsed().as_nanos() as f64 / queries_per_run as f64;

            // Bench: btree miss-only
            let t0 = Instant::now();
            for q in &miss_queries {
                sink ^= tree.index(q).unwrap_or(0);
            }
            let bt_miss = t0.elapsed().as_nanos() as f64 / queries_per_run as f64;

            // Prevent dead-code elimination
            assert!(sink < usize::MAX);

            let ratio_unif = bs_unif / bt_unif;
            let ratio_skew = bs_skew / bt_skew;

            println!("{:<8} {:>12.1} {:>12.1} {:>12.1} {:>12.1} {:>12.1} {:>12.1}   {:>7.3} {:>7.3}",
                n, bs_unif, bt_unif, bs_skew, bt_skew, bs_miss, bt_miss,
                ratio_unif, ratio_skew);
        }
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
