use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Instant;
use rand::Rng;
use rayon::prelude::*;

use std::num::Saturating;

/// Set to `true` to make `matmul_par` print row-progress to stderr.
pub static MATMUL_PROGRESS: AtomicBool = AtomicBool::new(false);

/// Node index type — u32 suffices for < 4 billion nodes and halves col_idx memory.
pub type NodeId = u32;

/// Wrapper to send a raw pointer across threads.
/// SAFETY: caller must guarantee disjoint access from each thread.
#[derive(Clone, Copy)]
struct SendPtr<T>(*mut T);
unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}
impl<T> SendPtr<T> {
    fn ptr(self) -> *mut T { self.0 }
}

#[inline(always)]
fn sadd(a: u64, b: u64) -> u64 {
    (Saturating(a) + Saturating(b)).0
}

#[inline(always)]
fn smul(a: u64, b: u64) -> u64 {
    (Saturating(a) * Saturating(b)).0
}

/// Sparse integer matrix in Compressed Sparse Row (CSR) format.
/// Immutable after construction — use builder methods (`from_edges`, `from_coo`, etc.).
#[derive(Clone, Debug)]
pub struct CsrMatrix {
    /// Number of nodes (matrix is n x n)
    pub n: NodeId,
    /// Row pointers: row i spans col_idx[row_ptr[i]..row_ptr[i+1]]
    pub row_ptr: Vec<usize>,
    /// Column indices (sorted within each row)
    pub col_idx: Vec<NodeId>,
    /// Corresponding non-zero values
    pub values: Vec<u64>,
}

impl CsrMatrix {
    /// Empty n×n matrix.
    pub fn new(n: NodeId) -> Self {
        Self {
            n,
            row_ptr: vec![0; n as usize + 1],
            col_idx: Vec::new(),
            values: Vec::new(),
        }
    }

    /// Identity matrix.
    pub fn identity(n: NodeId) -> Self {
        let nu = n as usize;
        let mut row_ptr = Vec::with_capacity(nu + 1);
        let mut col_idx = Vec::with_capacity(nu);
        let mut values = Vec::with_capacity(nu);
        for i in 0..n {
            row_ptr.push(i as usize);
            col_idx.push(i);
            values.push(1);
        }
        row_ptr.push(nu);
        Self { n, row_ptr, col_idx, values }
    }

    /// Build CSR from COO triplets. Sorts by (row, col), merges duplicates by summing.
    pub fn from_coo(n: NodeId, triplets: &mut Vec<(NodeId, NodeId, u64)>) -> Self {
        let nu = n as usize;
        triplets.sort_unstable_by_key(|&(r, c, _)| (r, c));

        // Deduplicate and merge
        let mut prev_row = NodeId::MAX;
        let mut prev_col = NodeId::MAX;
        let mut deduped: Vec<(NodeId, NodeId, u64)> = Vec::with_capacity(triplets.len());
        for &(r, c, v) in triplets.iter() {
            if r == prev_row && c == prev_col {
                deduped.last_mut().unwrap().2 += v;
            } else {
                deduped.push((r, c, v));
                prev_row = r;
                prev_col = c;
            }
        }

        // Build final arrays, filtering zeros
        let mut final_col: Vec<NodeId> = Vec::with_capacity(deduped.len());
        let mut final_val = Vec::with_capacity(deduped.len());
        let mut final_row_ptr = vec![0usize; nu + 1];
        let mut cur_row = 0usize;

        for &(r, c, v) in &deduped {
            if v == 0 { continue; }
            let ru = r as usize;
            while cur_row <= ru {
                final_row_ptr[cur_row] = final_col.len();
                cur_row += 1;
            }
            final_col.push(c);
            final_val.push(v);
        }
        while cur_row <= nu {
            final_row_ptr[cur_row] = final_col.len();
            cur_row += 1;
        }

        Self {
            n,
            row_ptr: final_row_ptr,
            col_idx: final_col,
            values: final_val,
        }
    }

    /// Build from directed edge list.
    pub fn from_edges(n: NodeId, edges: &[(NodeId, NodeId)]) -> Self {
        let mut triplets: Vec<(NodeId, NodeId, u64)> = edges.iter().map(|&(r, c)| (r, c, 1)).collect();
        Self::from_coo(n, &mut triplets)
    }

    /// Build from edge list, making it undirected (symmetric).
    pub fn from_edges_undirected(n: NodeId, edges: &[(NodeId, NodeId)]) -> Self {
        let mut triplets: Vec<(NodeId, NodeId, u64)> = Vec::with_capacity(edges.len() * 2);
        for &(r, c) in edges {
            triplets.push((r, c, 1));
            if r != c {
                triplets.push((c, r, 1));
            }
        }
        Self::from_coo(n, &mut triplets)
    }

    /// Build from named edge pairs. Returns the matrix and name→index mapping.
    pub fn from_adjacency<'a>(it: impl Iterator<Item = (&'a str, &'a str)>) -> (Self, BTreeMap<String, NodeId>) {
        let mut names: BTreeMap<String, NodeId> = BTreeMap::new();
        let mut edges = Vec::new();
        let mut next_id = 0 as NodeId;
        for (a, b) in it {
            let ai = *names.entry(a.to_string()).or_insert_with(|| { let id = next_id; next_id += 1; id });
            let bi = *names.entry(b.to_string()).or_insert_with(|| { let id = next_id; next_id += 1; id });
            edges.push((ai, bi));
        }
        (Self::from_edges(next_id, &edges), names)
    }

    /// Random directed graph with n nodes and m edges (no self-loops).
    pub fn random(rng: &mut impl Rng, n: NodeId, m: usize) -> Self {
        let nu = n as usize;
        assert!(nu >= 2, "need at least 2 nodes to avoid self-loops");
        let mut triplets: Vec<(NodeId, NodeId, u64)> = Vec::with_capacity(m);
        for _ in 0..m {
            let r = rng.random_range(0..nu);
            let c = rng.random_range(0..nu - 1);
            let c = if c >= r { c + 1 } else { c };
            triplets.push((r as NodeId, c as NodeId, 1));
        }
        Self::from_coo(n, &mut triplets)
    }

    /// N-dimensional Moore neighborhood lattice (see graph.rs for full docs).
    pub fn lattice(dims: &[usize], torus: bool) -> Self {
        let ndim = dims.len();
        let total: usize = dims.iter().product();

        let mut strides = vec![1usize; ndim];
        for i in (0..ndim.saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * dims[i + 1];
        }

        let mut triplets = Vec::new();
        let mut coord = vec![0usize; ndim];

        for node in 0..total {
            let n_neighbors = 3usize.pow(ndim as u32);
            for off_idx in 0..n_neighbors {
                let mut tmp = off_idx;
                let mut all_zero = true;
                let mut neighbor = 0usize;
                let mut valid = true;
                for d in 0..ndim {
                    let delta = (tmp % 3) as isize - 1;
                    tmp /= 3;
                    if delta != 0 { all_zero = false; }
                    let c = coord[d] as isize + delta;
                    let c = if torus {
                        c.rem_euclid(dims[d] as isize) as usize
                    } else if c < 0 || c >= dims[d] as isize {
                        valid = false;
                        break;
                    } else {
                        c as usize
                    };
                    neighbor += c * strides[d];
                }
                if all_zero || !valid { continue; }
                triplets.push((node as NodeId, neighbor as NodeId, 1u64));
            }

            for d in (0..ndim).rev() {
                coord[d] += 1;
                if coord[d] < dims[d] { break; }
                coord[d] = 0;
            }
        }
        Self::from_coo(total as NodeId, &mut triplets)
    }

    /// Randomly keep a fraction of edges, preserving symmetry.
    pub fn thin(&self, rng: &mut impl Rng, density: f64) -> Self {
        let mut triplets = Vec::new();
        // To preserve symmetry: decide per unordered pair {r,c} with r <= c
        for r in 0..self.n {
            let ru = r as usize;
            let start = self.row_ptr[ru];
            let end = self.row_ptr[ru + 1];
            for idx in start..end {
                let c = self.col_idx[idx];
                let v = self.values[idx];
                if r <= c && rng.random_range(0.0..1.0) < density {
                    triplets.push((r, c, v));
                    if r != c {
                        let rev = self.get(c, r);
                        if rev > 0 {
                            triplets.push((c, r, rev));
                        }
                    }
                }
            }
        }
        Self::from_coo(self.n, &mut triplets)
    }

    /// Lookup value at (r, c) via binary search.
    pub fn get(&self, r: NodeId, c: NodeId) -> u64 {
        let start = self.row_ptr[r as usize];
        let end = self.row_ptr[r as usize + 1];
        match self.col_idx[start..end].binary_search(&c) {
            Ok(i) => self.values[start + i],
            Err(_) => 0,
        }
    }

    /// Number of non-zero entries.
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Iterate over non-zero entries in row r: yields (col, value).
    pub fn row_iter(&self, r: NodeId) -> impl Iterator<Item = (NodeId, u64)> + '_ {
        let start = self.row_ptr[r as usize];
        let end = self.row_ptr[r as usize + 1];
        self.col_idx[start..end].iter().zip(&self.values[start..end])
            .map(|(&c, &v)| (c, v))
    }

    /// Matrix multiply using a BTreeMap accumulator per row.
    /// Slower than `matmul` (O(flops * log n)) but uses less scratch memory.
    pub fn matmul_btree(&self, other: &Self) -> Self {
        assert_eq!(self.n, other.n);
        let n = self.n;
        let nu = n as usize;

        let mut row_ptr = Vec::with_capacity(nu + 1);
        let mut col_idx = Vec::new();
        let mut values = Vec::new();
        row_ptr.push(0);

        for i in 0..n {
            let mut acc: BTreeMap<NodeId, u64> = BTreeMap::new();
            for (k, a_ik) in self.row_iter(i) {
                for (j, b_kj) in other.row_iter(k) {
                    let e = acc.entry(j).or_insert(0);
                    *e = sadd(*e, smul(a_ik, b_kj));
                }
            }
            for (j, v) in acc {
                if v != 0 {
                    col_idx.push(j);
                    values.push(v);
                }
            }
            row_ptr.push(col_idx.len());
        }

        Self { n, row_ptr, col_idx, values }
    }

    /// Matrix multiply: self × other, using a dense Vec<u64> accumulator per row.
    /// O(flops + n * active_rows) time, O(n) scratch space.
    pub fn matmul(&self, other: &Self) -> Self {
        assert_eq!(self.n, other.n);
        let n = self.n;
        let nu = n as usize;

        let mut row_ptr = Vec::with_capacity(nu + 1);
        let mut col_idx = Vec::new();
        let mut values = Vec::new();
        row_ptr.push(0);

        let mut acc = vec![0u64; nu];
        let mut nz_cols: Vec<NodeId> = Vec::new();

        for i in 0..n {
            // Scatter phase
            for (k, a_ik) in self.row_iter(i) {
                for (j, b_kj) in other.row_iter(k) {
                    if acc[j as usize] == 0 {
                        nz_cols.push(j);
                    }
                    acc[j as usize] = sadd(acc[j as usize], smul(a_ik, b_kj));
                }
            }

            // Gather phase — sort columns, emit, clear
            nz_cols.sort_unstable();
            for &j in &nz_cols {
                let v = acc[j as usize];
                if v != 0 {
                    col_idx.push(j);
                    values.push(v);
                }
                acc[j as usize] = 0;
            }
            nz_cols.clear();

            row_ptr.push(col_idx.len());
        }

        Self { n, row_ptr, col_idx, values }
    }

    /// Parallel matrix multiply using rayon. Two-pass symbolic+numeric:
    /// pass 1 counts nnz per row, pass 2 fills exact-sized output arrays.
    pub fn matmul_par(&self, other: &Self) -> Self {
        assert_eq!(self.n, other.n);
        let n = self.n;
        let nu = n as usize;

        let progress = MATMUL_PROGRESS.load(Ordering::Relaxed);
        let rows_done = AtomicUsize::new(0);
        let rows_done_ref = &rows_done;
        let print_interval = (nu / 200).max(1); // ~0.5% granularity

        // Pass 1: symbolic — count nnz per output row
        let pass_start = Instant::now();
        let mut nnz_per_row = vec![0usize; nu];
        nnz_per_row.par_iter_mut().enumerate().for_each_init(
            || vec![false; nu],
            |mask, (i, nnz_out)| {
                let mut count = 0usize;
                let a_start = self.row_ptr[i];
                let a_end = self.row_ptr[i + 1];
                for idx in a_start..a_end {
                    let k = self.col_idx[idx] as usize;
                    let b_start = other.row_ptr[k];
                    let b_end = other.row_ptr[k + 1];
                    for jdx in b_start..b_end {
                        let j = other.col_idx[jdx] as usize;
                        if !mask[j] {
                            mask[j] = true;
                            count += 1;
                        }
                    }
                }
                *nnz_out = count;
                // Clear mask
                for idx in a_start..a_end {
                    let k = self.col_idx[idx] as usize;
                    let b_start = other.row_ptr[k];
                    let b_end = other.row_ptr[k + 1];
                    for jdx in b_start..b_end {
                        mask[other.col_idx[jdx] as usize] = false;
                    }
                }

                if progress {
                    let done = rows_done_ref.fetch_add(1, Ordering::Relaxed) + 1;
                    if done % print_interval == 0 || done == nu {
                        let elapsed = pass_start.elapsed().as_secs_f64();
                        let rps = done as f64 / elapsed;
                        let eta = (nu - done) as f64 / rps;
                        eprint!("\r  symbolic: {done}/{nu} ({:.1}%)  {rps:.0} rows/s  ETA {eta:.1}s   ",
                            done as f64 / nu as f64 * 100.0);
                    }
                }
            },
        );
        if progress {
            let elapsed = pass_start.elapsed().as_secs_f64();
            eprintln!("\r  symbolic: done in {elapsed:.1}s ({:.0} rows/s)                    ",
                nu as f64 / elapsed);
            rows_done.store(0, Ordering::Relaxed);
        }

        // Build row_ptr from nnz counts
        let mut row_ptr = Vec::with_capacity(nu + 1);
        row_ptr.push(0);
        for &c in &nnz_per_row {
            row_ptr.push(row_ptr.last().unwrap() + c);
        }
        let total_nnz = *row_ptr.last().unwrap();

        // Allocate output arrays exactly
        let mut col_idx = vec![0 as NodeId; total_nnz];
        let mut values = vec![0u64; total_nnz];

        // Pass 2: numeric — fill col_idx and values in parallel
        // SAFETY: each row writes to a disjoint slice [row_ptr[i]..row_ptr[i+1]]
        let pass_start = Instant::now();
        let col_ptr = &row_ptr;
        let out_c = SendPtr(col_idx.as_mut_ptr());
        let out_v = SendPtr(values.as_mut_ptr());

        (0..nu).into_par_iter().for_each_init(
            || (vec![0u64; nu], Vec::<NodeId>::new()),
            |(acc, nz_cols), i| {
                let a_start = self.row_ptr[i];
                let a_end = self.row_ptr[i + 1];
                for idx in a_start..a_end {
                    let k = self.col_idx[idx] as usize;
                    let a_ik = self.values[idx];
                    let b_start = other.row_ptr[k];
                    let b_end = other.row_ptr[k + 1];
                    for jdx in b_start..b_end {
                        let j = other.col_idx[jdx];
                        if acc[j as usize] == 0 {
                            nz_cols.push(j);
                        }
                        acc[j as usize] = sadd(acc[j as usize], smul(a_ik, other.values[jdx]));
                    }
                }

                nz_cols.sort_unstable();
                let out_start = col_ptr[i];
                let mut pos = out_start;
                for &j in nz_cols.iter() {
                    let v = acc[j as usize];
                    if v != 0 {
                        unsafe {
                            *out_c.ptr().add(pos) = j;
                            *out_v.ptr().add(pos) = v;
                        }
                        pos += 1;
                    }
                    acc[j as usize] = 0;
                }
                nz_cols.clear();

                if progress {
                    let done = rows_done_ref.fetch_add(1, Ordering::Relaxed) + 1;
                    if done % print_interval == 0 || done == nu {
                        let elapsed = pass_start.elapsed().as_secs_f64();
                        let rps = done as f64 / elapsed;
                        let eta = (nu - done) as f64 / rps;
                        eprint!("\r  numeric:  {done}/{nu} ({:.1}%)  {rps:.0} rows/s  ETA {eta:.1}s   ",
                            done as f64 / nu as f64 * 100.0);
                    }
                }
            },
        );
        if progress {
            let elapsed = pass_start.elapsed().as_secs_f64();
            eprintln!("\r  numeric:  done in {elapsed:.1}s ({:.0} rows/s)                    ",
                nu as f64 / elapsed);
        }

        Self { n, row_ptr, col_idx, values }
    }

    /// Element-wise addition via sorted merge of each row.
    pub fn add(&self, other: &Self) -> Self {
        assert_eq!(self.n, other.n);
        let n = self.n;
        let nu = n as usize;

        let mut row_ptr = Vec::with_capacity(nu + 1);
        let mut col_idx = Vec::new();
        let mut values = Vec::new();
        row_ptr.push(0);

        for r in 0..nu {
            let a_start = self.row_ptr[r];
            let a_end = self.row_ptr[r + 1];
            let b_start = other.row_ptr[r];
            let b_end = other.row_ptr[r + 1];

            let mut ai = a_start;
            let mut bi = b_start;

            while ai < a_end && bi < b_end {
                let ac = self.col_idx[ai];
                let bc = other.col_idx[bi];
                if ac < bc {
                    col_idx.push(ac);
                    values.push(self.values[ai]);
                    ai += 1;
                } else if ac > bc {
                    col_idx.push(bc);
                    values.push(other.values[bi]);
                    bi += 1;
                } else {
                    let v = sadd(self.values[ai], other.values[bi]);
                    if v != 0 {
                        col_idx.push(ac);
                        values.push(v);
                    }
                    ai += 1;
                    bi += 1;
                }
            }
            while ai < a_end {
                col_idx.push(self.col_idx[ai]);
                values.push(self.values[ai]);
                ai += 1;
            }
            while bi < b_end {
                col_idx.push(other.col_idx[bi]);
                values.push(other.values[bi]);
                bi += 1;
            }

            row_ptr.push(col_idx.len());
        }

        Self { n, row_ptr, col_idx, values }
    }

    /// Compute A + A^2 + ... until sparsity pattern stabilizes.
    pub fn reachability_sum(&self) -> (Self, usize) {
        let mut power = self.clone();
        let mut sum = self.clone();
        let mut k = 1usize;
        loop {
            power = power.matmul(self);
            k += 1;
            let new_sum = sum.add(&power);
            if new_sum.nnz() == sum.nnz() {
                return (new_sum, k);
            }
            sum = new_sum;
        }
    }

    /// Repeated squaring until sparsity pattern stabilizes.
    pub fn power_until_stable(&self) -> (Self, usize) {
        let mut current = self.clone();
        let mut k = 0usize;
        loop {
            let next = current.matmul(&current);
            k += 1;
            if next.nnz() == current.nnz()
                && next.col_idx == current.col_idx
                && next.row_ptr == current.row_ptr
            {
                return (next, k);
            }
            current = next;
        }
    }

    /// Connected components via transitive closure.
    pub fn connected_components(&self) -> Vec<usize> {
        let with_id = self.add(&Self::identity(self.n));
        let (closure, _) = with_id.power_until_stable();

        let nu = self.n as usize;
        let mut component = vec![usize::MAX; nu];
        let mut next_id = 0;

        for i in 0..nu {
            if component[i] != usize::MAX {
                continue;
            }
            let id = next_id;
            next_id += 1;
            component[i] = id;
            for j in (i + 1)..nu {
                if closure.get(i as NodeId, j as NodeId) > 0 && closure.get(j as NodeId, i as NodeId) > 0 {
                    component[j] = id;
                }
            }
        }
        component
    }

    /// Connected components via union-find. O(nnz * α(n)), essentially linear.
    /// Returns a Vec where result[i] = component id for node i.
    /// Treats the graph as undirected (edge in either direction connects nodes).
    pub fn connected_components_uf(&self) -> Vec<usize> {
        let nu = self.n as usize;
        let mut parent: Vec<usize> = (0..nu).collect();
        let mut rank = vec![0u8; nu];

        fn find(parent: &mut [usize], mut x: usize) -> usize {
            while parent[x] != x {
                parent[x] = parent[parent[x]]; // path halving
                x = parent[x];
            }
            x
        }

        fn union(parent: &mut [usize], rank: &mut [u8], a: usize, b: usize) {
            let ra = find(parent, a);
            let rb = find(parent, b);
            if ra == rb { return; }
            if rank[ra] < rank[rb] {
                parent[ra] = rb;
            } else if rank[ra] > rank[rb] {
                parent[rb] = ra;
            } else {
                parent[rb] = ra;
                rank[ra] += 1;
            }
        }

        for r in 0..self.n {
            for (c, _) in self.row_iter(r) {
                union(&mut parent, &mut rank, r as usize, c as usize);
            }
        }

        // Canonicalize: map roots to sequential ids
        let mut id_map = vec![usize::MAX; nu];
        let mut result = vec![0usize; nu];
        let mut next_id = 0;
        for i in 0..nu {
            let root = find(&mut parent, i);
            if id_map[root] == usize::MAX {
                id_map[root] = next_id;
                next_id += 1;
            }
            result[i] = id_map[root];
        }
        result
    }

    /// Number of connected components (via union-find).
    pub fn num_components(&self) -> usize {
        let comp = self.connected_components_uf();
        comp.iter().copied().max().map_or(0, |m| m + 1)
    }

    /// Print (for small matrices).
    pub fn print(&self) {
        for r in 0..self.n {
            for c in 0..self.n {
                let v = self.get(r, c);
                if v == 0 {
                    print!(". ");
                } else {
                    print!("{v} ");
                }
            }
            println!();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn test_identity_matmul() {
        let m = CsrMatrix::from_edges(3, &[(0, 1), (1, 2)]);
        let id = CsrMatrix::identity(3);
        let result = m.matmul(&id);
        assert_eq!(result.get(0, 1), 1);
        assert_eq!(result.get(1, 2), 1);
        assert_eq!(result.get(0, 2), 0);
        assert_eq!(result.nnz(), 2);
    }

    #[test]
    fn test_path_counting_triangle() {
        let m = CsrMatrix::from_edges(3, &[(0, 1), (1, 2), (2, 0)]);
        let m2 = m.matmul(&m);
        assert_eq!(m2.get(0, 2), 1);
        assert_eq!(m2.get(1, 0), 1);
        assert_eq!(m2.get(2, 1), 1);
        let m3 = m2.matmul(&m);
        assert_eq!(m3.get(0, 0), 1);
        assert_eq!(m3.get(1, 1), 1);
        assert_eq!(m3.get(2, 2), 1);
    }

    #[test]
    fn test_path_counting_parallel_paths() {
        let m = CsrMatrix::from_edges(2, &[(0, 1), (0, 1)]);
        assert_eq!(m.get(0, 1), 2);
    }

    #[test]
    fn test_path_counting_diamond() {
        let m = CsrMatrix::from_edges(4, &[
            (0, 1), (0, 2), (1, 3), (2, 3),
        ]);
        let m2 = m.matmul(&m);
        assert_eq!(m2.get(0, 3), 2);
    }

    #[test]
    fn test_reachability_chain() {
        let m = CsrMatrix::from_edges(4, &[(0, 1), (1, 2), (2, 3)]);
        let (sum, _k) = m.reachability_sum();
        assert!(sum.get(0, 1) > 0);
        assert!(sum.get(0, 2) > 0);
        assert!(sum.get(0, 3) > 0);
        assert!(sum.get(1, 2) > 0);
        assert!(sum.get(1, 3) > 0);
        assert!(sum.get(2, 3) > 0);
        assert_eq!(sum.get(3, 0), 0);
        assert_eq!(sum.get(2, 0), 0);
    }

    #[test]
    fn test_power_until_stable_chain() {
        let n: NodeId = 64;
        let edges: Vec<(NodeId, NodeId)> = (0..n - 1).map(|i| (i, i + 1)).collect();
        let m = CsrMatrix::from_edges(n, &edges);
        let with_id = m.add(&CsrMatrix::identity(n));
        let (_stable, iters) = with_id.power_until_stable();
        assert!(iters <= 8, "took {iters} iterations for chain of {n}");
    }

    #[test]
    fn test_connected_components_two_triangles() {
        let m = CsrMatrix::from_edges_undirected(6, &[
            (0, 1), (1, 2), (2, 0),
            (3, 4), (4, 5), (5, 3),
        ]);
        let comp = m.connected_components();
        assert_eq!(comp[0], comp[1]);
        assert_eq!(comp[1], comp[2]);
        assert_eq!(comp[3], comp[4]);
        assert_eq!(comp[4], comp[5]);
        assert_ne!(comp[0], comp[3]);
    }

    #[test]
    fn test_connected_components_isolated() {
        let m = CsrMatrix::new(5);
        let comp = m.connected_components();
        let unique: HashSet<usize> = comp.into_iter().collect();
        assert_eq!(unique.len(), 5);
    }

    #[test]
    fn test_from_adjacency_basic() {
        let edges = vec![("a", "b"), ("b", "c"), ("c", "a")];
        let (m, names) = CsrMatrix::from_adjacency(edges.into_iter());
        assert_eq!(names.len(), 3);
        assert_eq!(m.n, 3);
        assert_eq!(m.nnz(), 3);
        let a = names["a"];
        let b = names["b"];
        let c = names["c"];
        assert_eq!(m.get(a, b), 1);
        assert_eq!(m.get(b, c), 1);
        assert_eq!(m.get(c, a), 1);
        assert_eq!(m.get(a, c), 0);
    }

    #[test]
    fn test_from_adjacency_duplicate_edges() {
        let edges = vec![("x", "y"), ("x", "y"), ("y", "x")];
        let (m, names) = CsrMatrix::from_adjacency(edges.into_iter());
        let x = names["x"];
        let y = names["y"];
        assert_eq!(m.get(x, y), 2);
        assert_eq!(m.get(y, x), 1);
    }

    #[test]
    fn test_from_adjacency_self_loop() {
        let edges = vec![("a", "a"), ("a", "b")];
        let (m, names) = CsrMatrix::from_adjacency(edges.into_iter());
        let a = names["a"];
        let b = names["b"];
        assert_eq!(m.get(a, a), 1);
        assert_eq!(m.get(a, b), 1);
        assert_eq!(m.n, 2);
    }

    #[test]
    fn test_from_adjacency_components() {
        let edges = vec![("a", "b"), ("b", "a"), ("c", "d"), ("d", "c")];
        let (m, names) = CsrMatrix::from_adjacency(edges.into_iter());
        let comp = m.connected_components();
        assert_eq!(comp[names["a"] as usize], comp[names["b"] as usize]);
        assert_eq!(comp[names["c"] as usize], comp[names["d"] as usize]);
        assert_ne!(comp[names["a"] as usize], comp[names["c"] as usize]);
    }

    #[test]
    fn test_lattice_1d_no_torus() {
        let m = CsrMatrix::lattice(&[5], false);
        assert_eq!(m.n, 5);
        assert_eq!(m.get(0, 1), 1);
        assert_eq!(m.get(1, 0), 1);
        assert_eq!(m.get(0, 0), 0);
        assert_eq!(m.get(4, 3), 1);
        assert_eq!(m.get(4, 0), 0);
        assert_eq!(m.nnz(), 8);
    }

    #[test]
    fn test_lattice_1d_torus() {
        let m = CsrMatrix::lattice(&[5], true);
        assert_eq!(m.get(0, 4), 1);
        assert_eq!(m.get(4, 0), 1);
        assert_eq!(m.nnz(), 10);
    }

    #[test]
    fn test_lattice_2d_corner() {
        let m = CsrMatrix::lattice(&[3, 3], false);
        assert_eq!(m.n, 9);
        assert_eq!(m.get(0, 1), 1);
        assert_eq!(m.get(0, 3), 1);
        assert_eq!(m.get(0, 4), 1);
        assert_eq!(m.nnz() % 2, 0);
    }

    #[test]
    fn test_lattice_2d_center() {
        let m = CsrMatrix::lattice(&[3, 3], false);
        let mut count = 0;
        for j in 0..9u32 {
            if m.get(4, j) > 0 { count += 1; }
        }
        assert_eq!(count, 8);
    }

    #[test]
    fn test_lattice_2d_torus() {
        let m = CsrMatrix::lattice(&[3, 3], true);
        assert_eq!(m.n, 9);
        for i in 0..9u32 {
            let mut count = 0;
            for j in 0..9u32 {
                if m.get(i, j) > 0 { count += 1; }
            }
            assert_eq!(count, 8, "node {i} should have 8 neighbors on torus");
        }
        assert_eq!(m.nnz(), 9 * 8);
    }

    #[test]
    fn test_lattice_2d_single_component() {
        let m = CsrMatrix::lattice(&[4, 4], false);
        let comp = m.connected_components();
        for i in 1..16 {
            assert_eq!(comp[0], comp[i]);
        }
    }

    #[test]
    fn test_lattice_3d() {
        let m = CsrMatrix::lattice(&[2, 2, 2], false);
        assert_eq!(m.n, 8);
        let mut count = 0;
        for j in 0..8u32 {
            if m.get(0, j) > 0 { count += 1; }
        }
        assert_eq!(count, 7);
        assert_eq!(m.nnz(), 8 * 7);
    }

    #[test]
    fn test_lattice_symmetry() {
        let m = CsrMatrix::lattice(&[4, 3], false);
        for r in 0..m.n {
            for (c, v) in m.row_iter(r) {
                assert_eq!(m.get(c, r), v, "asymmetry at ({r},{c})");
            }
        }
    }

    #[test]
    fn test_connected_components_single_component() {
        let m = CsrMatrix::from_edges_undirected(4, &[
            (0, 1), (1, 2), (2, 3),
        ]);
        let comp = m.connected_components();
        assert_eq!(comp[0], comp[1]);
        assert_eq!(comp[1], comp[2]);
        assert_eq!(comp[2], comp[3]);
    }

    #[test]
    #[cfg(feature = "long-tests")]
    fn bench_matmul_btree_vs_csr() {
        use crate::graph::SparseCountMatrix;
        use std::time::Instant;
        use rand::prelude::StdRng;
        use rand::SeedableRng;

        let mut rng = StdRng::from_seed([42; 32]);
        let grid_sizes: &[usize] = &[5, 10, 20, 30];
        let edges_per_node: &[f64] = &[2.0, 3.0, 4.0, 8.0, 26.0];

        println!();
        println!("side,nodes,e_per_n,nnz,components,orig_btree_us,csr_us,csr_par_us,ratio_csr,ratio_par");
        for &s in grid_sizes {
            let full_btree = SparseCountMatrix::lattice(&[s, s, s], true);
            let full_csr = CsrMatrix::lattice(&[s, s, s], true);
            let n = full_btree.n;
            let full_epn = full_btree.nnz() as f64 / n as f64;

            for &epn in edges_per_node {
                let density = epn / full_epn;

                let a_bt = if density >= 1.0 { full_btree.clone() } else { full_btree.thin(&mut rng, density) };
                let b_bt = a_bt.clone();

                // Build CSR from the same edges
                let a_csr = if density >= 1.0 {
                    full_csr.clone()
                } else {
                    let mut triplets: Vec<(NodeId, NodeId, u64)> = a_bt.entries.iter()
                        .map(|(&(r, c), &v)| (r as NodeId, c as NodeId, v)).collect();
                    CsrMatrix::from_coo(n as NodeId, &mut triplets)
                };
                let b_csr = a_csr.clone();

                let t0 = Instant::now();
                let _r_bt = a_bt.matmul(&b_bt);
                let t_bt = t0.elapsed().as_micros();

                let t0 = Instant::now();
                let _r_csr = a_csr.matmul(&b_csr);
                let t_csr = t0.elapsed().as_micros();

                let t0 = Instant::now();
                let _r_par = a_csr.matmul_par(&b_csr);
                let t_par = t0.elapsed().as_micros();

                let components = a_csr.num_components();

                let ratio_csr = if t_bt > 0 { t_csr as f64 / t_bt as f64 } else { 0.0 };
                let ratio_par = if t_bt > 0 { t_par as f64 / t_bt as f64 } else { 0.0 };

                println!("{s},{n},{epn:.0},{},{components},{t_bt},{t_csr},{t_par},{ratio_csr:.6},{ratio_par:.6}",
                    a_bt.nnz());
            }
        }
    }

    /// Load directed edges from a file with "<int> <int>" per line.
    /// Returns (n, edges) where n = max node id + 1.
    #[cfg(feature = "long-tests")]
    fn load_edges(path: &str) -> (NodeId, Vec<(NodeId, NodeId)>) {
        let contents = std::fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("cannot read {path}: {e}"));
        let mut edges = Vec::new();
        let mut max_id = 0u32;
        for line in contents.lines() {
            let line = line.trim();
            if line.is_empty() { continue; }
            let mut parts = line.split_whitespace();
            let a: NodeId = parts.next().unwrap().parse().unwrap();
            let b: NodeId = parts.next().unwrap().parse().unwrap();
            max_id = max_id.max(a).max(b);
            edges.push((a, b));
        }
        (max_id + 1, edges)
    }

    #[test]
    #[cfg(feature = "long-tests")]
    fn bench_diameter() {
        use std::time::Instant;

        let graphs = &[
            ("cora", "gen-graphs/cora.edges"),
            // ("nell", "gen-graphs/nell.edges"),
            ("ogbn_arxiv", "gen-graphs/ogbn_arxiv.edges"),
        ];

        println!();
        println!("=== Diameter via repeated squaring ===");

        for &(name, path) in graphs {
            let (n, edges) = load_edges(path);
            // Undirected + self-loops: R[i][j] > 0 ⟺ dist(i,j) ≤ 1
            let a_sym = CsrMatrix::from_edges_undirected(n, &edges);
            let r0 = a_sym.add(&CsrMatrix::identity(n));

            println!();
            println!("[{name}] n={n}, edges={} (undirected nnz={})", edges.len(), a_sym.nnz());

            // Phase 1: repeated squaring to find upper bound
            let mut current = r0.clone();
            let mut reach = 1usize; // current covers distances ≤ reach
            let mut prev_saved = r0.clone(); // last power before stabilisation
            let mut prev_reach = 0usize;
            let mut squarings = 0;

            loop {
                // Enable progress for ogbn_arxiv squaring #3
                let want_progress = name == "ogbn_arxiv" && squarings == 1;
                MATMUL_PROGRESS.store(want_progress, Ordering::Relaxed);

                let t0 = Instant::now();
                let mut next = current.matmul_par(&current);
                let t_ms = t0.elapsed().as_millis();
                squarings += 1;
                let new_reach = reach * 2;

                println!("  squaring {squarings}: reach ≤{new_reach}, nnz={}, {t_ms} ms", next.nnz());

                if next.nnz() == current.nnz()
                    && next.col_idx == current.col_idx
                    && next.row_ptr == current.row_ptr
                {
                    // Stabilised: diameter ∈ (prev_reach, new_reach]
                    // But we need the boundary between reach and new_reach
                    // prev_saved covers ≤ reach (the pre-squaring matrix)
                    println!("  stabilised: diameter ∈ ({prev_reach}, {new_reach}]");
                    MATMUL_PROGRESS.store(false, Ordering::Relaxed);
                    break;
                }

                prev_saved = current;
                prev_reach = reach;
                current = next;
                reach = new_reach;
            }

            // Phase 2: linear refinement from prev_saved (covers ≤ prev_reach)
            // Multiply by r0 one step at a time until stable
            if prev_reach == 0 {
                // Stabilised on the very first squaring: diameter ≤ 2
                // Refine: check if A_sym + I itself was already the closure
                if r0.nnz() == current.nnz() {
                    println!("  diameter = 1 (or 0 if isolated nodes)");
                } else {
                    println!("  diameter = 2");
                }
            } else {
                let mut refine = prev_saved;
                let mut d = prev_reach;
                loop {
                    let t0 = Instant::now();
                    let next = refine.matmul_par(&r0);
                    let t_ms = t0.elapsed().as_millis();
                    d += 1;

                    println!("  refine d={d}: nnz={}, {t_ms} ms", next.nnz());

                    if next.nnz() == refine.nnz()
                        && next.col_idx == refine.col_idx
                        && next.row_ptr == refine.row_ptr
                    {
                        println!("  diameter = {}", d - 1);
                        break;
                    }
                    refine = next;
                }
            }
        }
    }

    #[test]
    #[cfg(feature = "long-tests")]
    fn bench_real_graphs() {
        use std::time::Instant;

        let graphs = &[
            ("cora", "gen-graphs/cora.edges"),
            ("nell", "gen-graphs/nell.edges"),
            ("ogbn_arxiv", "gen-graphs/ogbn_arxiv.edges"),
        ];

        const ITERS: u32 = 1; // !!
        const MAX_POWER: usize = 14;

        println!();
        println!("graph,nodes,edges,step,nnz_out,csr_par_us");

        // Each nnz costs 12 bytes (col_idx u32 + value u64). matmul_par holds prev + A + result
        // + per-thread scratch, so budget ~3× output size. 2B nnz ≈ 24 GB × 3 ≈ 72 GB.
        const MAX_NNZ: usize = 4_400_000_000; // !!

        for &(name, path) in graphs {
            let (n, edges) = load_edges(path);
            let a = CsrMatrix::from_edges(n, &edges);
            let mut prev = a.clone();

            for step in 2..=MAX_POWER {
                let result = prev.matmul_par(&a);
                let nnz_out = result.nnz();

                let t0 = Instant::now();
                for _ in 0..ITERS { let _ = prev.matmul_par(&a); }
                let t_us = t0.elapsed().as_micros() / ITERS as u128;

                println!("{name},{n},{},{step},{nnz_out},{t_us}", edges.len());

                prev = result;

                if nnz_out > MAX_NNZ {
                    break;
                }
            }
        }
    }
}
