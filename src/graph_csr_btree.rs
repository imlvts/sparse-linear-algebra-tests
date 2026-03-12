use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Instant;
use rand::Rng;
use rayon::prelude::*;

use std::num::Saturating;

use crate::dense_btree::DenseBTreeList;
use einsum_dyn::NDIndex;

/// Node index type — u32 suffices for < 4 billion nodes and halves col_idx memory.
pub type NodeId = u32;

/// Value type for matrix entries. u32 halves memory vs u64; change to u64 if counts overflow.
pub type Val = u32;

/// Set to `true` to make `matmul_par` print row-progress to stderr.
pub static MATMUL_PROGRESS: AtomicBool = AtomicBool::new(false);

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
fn sadd(a: Val, b: Val) -> Val {
    (Saturating(a) + Saturating(b)).0
}

#[inline(always)]
fn smul(a: Val, b: Val) -> Val {
    (Saturating(a) * Saturating(b)).0
}

/// Sparse integer matrix in CSR format with DenseBTreeList-indexed columns.
/// Immutable after construction — use builder methods (`from_edges`, `from_coo`, etc.).
#[derive(Clone, Debug)]
pub struct CsrBTreeMatrix {
    /// Number of nodes (matrix is n x n)
    pub n: NodeId,
    /// All per-row column-index B-trees packed into a single flat vec
    pub col_trees: DenseBTreeList<NodeId>,
    /// Corresponding non-zero values (flat, indexed by col_trees.data_start)
    pub values: Vec<Val>,
    /// Permutation mapping: perm[new_index] = old_index. Set by `rcm()` / `permute()`.
    pub perm: Option<Vec<NodeId>>,
}

impl CsrBTreeMatrix {
    /// Build from pre-computed flat arrays, converting col_idx into DenseBTreeList.
    fn from_flat(n: NodeId, row_ptr: &[usize], col_idx: &[NodeId], values: Vec<Val>) -> Self {
        let mut col_trees = DenseBTreeList::new();
        for r in 0..n as usize {
            col_trees.add_from_sorted(&col_idx[row_ptr[r]..row_ptr[r + 1]]);
        }
        Self { n, col_trees, values, perm: None }
    }

    /// Empty n×n matrix.
    pub fn new(n: NodeId) -> Self {
        let mut col_trees = DenseBTreeList::new();
        for _ in 0..n as usize {
            col_trees.add_from_sorted(&[]);
        }
        Self { n, col_trees, values: Vec::new(), perm: None }
    }

    /// Identity matrix.
    pub fn identity(n: NodeId) -> Self {
        let nu = n as usize;
        let row_ptr: Vec<usize> = (0..=nu).collect();
        let col_idx: Vec<NodeId> = (0..n).collect();
        let values = vec![1 as Val; nu];
        Self::from_flat(n, &row_ptr, &col_idx, values)
    }

    /// Build CSR from COO triplets. Sorts by (row, col), merges duplicates by summing.
    pub fn from_coo(n: NodeId, triplets: &mut Vec<(NodeId, NodeId, Val)>) -> Self {
        let nu = n as usize;
        triplets.sort_unstable_by_key(|&(r, c, _)| (r, c));

        // Deduplicate and merge
        let mut prev_row = NodeId::MAX;
        let mut prev_col = NodeId::MAX;
        let mut deduped: Vec<(NodeId, NodeId, Val)> = Vec::with_capacity(triplets.len());
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

        Self::from_flat(n, &final_row_ptr, &final_col, final_val)
    }

    /// Build from directed edge list.
    pub fn from_edges(n: NodeId, edges: &[(NodeId, NodeId)]) -> Self {
        let mut triplets: Vec<(NodeId, NodeId, Val)> = edges.iter().map(|&(r, c)| (r, c, 1)).collect();
        Self::from_coo(n, &mut triplets)
    }

    /// Build from edge list, making it undirected (symmetric).
    pub fn from_edges_undirected(n: NodeId, edges: &[(NodeId, NodeId)]) -> Self {
        let mut triplets: Vec<(NodeId, NodeId, Val)> = Vec::with_capacity(edges.len() * 2);
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
        let mut triplets: Vec<(NodeId, NodeId, Val)> = Vec::with_capacity(m);
        for _ in 0..m {
            let r = rng.random_range(0..nu);
            let c = rng.random_range(0..nu - 1);
            let c = if c >= r { c + 1 } else { c };
            triplets.push((r as NodeId, c as NodeId, 1));
        }
        Self::from_coo(n, &mut triplets)
    }

    /// N-dimensional Moore neighborhood lattice.
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
                triplets.push((node as NodeId, neighbor as NodeId, 1 as Val));
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
        for r in 0..self.n {
            let ru = r as usize;
            let cols = self.col_trees.data(ru);
            let start = self.col_trees.data_start(ru);
            for (local, &c) in cols.iter().enumerate() {
                let v = self.values[start + local];
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

    /// Lookup value at (r, c) via DenseBTree search.
    pub fn get(&self, r: NodeId, c: NodeId) -> Val {
        let start = self.col_trees.data_start(r as usize);
        match self.col_trees.index(r as usize, &c) {
            Ok(i) => self.values[start + i],
            Err(_) => 0,
        }
    }

    /// Number of non-zero entries.
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Iterate over non-zero entries in row r: yields (col, value).
    pub fn row_iter(&self, r: NodeId) -> impl Iterator<Item = (NodeId, Val)> + '_ {
        let ru = r as usize;
        let start = self.col_trees.data_start(ru);
        let cols = self.col_trees.data(ru);
        cols.iter().enumerate().map(move |(i, &c)| (c, self.values[start + i]))
    }

    /// Compare column structure with another matrix.
    fn col_eq(&self, other: &Self) -> bool {
        if self.n != other.n { return false; }
        for r in 0..self.n as usize {
            if self.col_trees.data(r) != other.col_trees.data(r) {
                return false;
            }
        }
        true
    }

    /// Matrix multiply using a BTreeMap accumulator per row.
    pub fn matmul_btree(&self, other: &Self) -> Self {
        assert_eq!(self.n, other.n);
        let n = self.n;
        let nu = n as usize;

        let mut row_ptr = Vec::with_capacity(nu + 1);
        let mut col_idx = Vec::new();
        let mut values = Vec::new();
        row_ptr.push(0);

        for i in 0..n {
            let mut acc: BTreeMap<NodeId, Val> = BTreeMap::new();
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

        Self::from_flat(n, &row_ptr, &col_idx, values)
    }

    /// Matrix multiply: self × other, using a dense Vec<Val> accumulator per row.
    pub fn matmul(&self, other: &Self) -> Self {
        assert_eq!(self.n, other.n);
        let n = self.n;
        let nu = n as usize;

        let mut row_ptr = Vec::with_capacity(nu + 1);
        let mut col_idx = Vec::new();
        let mut values = Vec::new();
        row_ptr.push(0);

        let mut acc = vec![0 as Val; nu];
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

        Self::from_flat(n, &row_ptr, &col_idx, values)
    }

    /// Parallel matrix multiply using rayon.
    pub fn matmul_par(&self, other: &Self) -> Self {
        assert_eq!(self.n, other.n);
        let n = self.n;
        let nu = n as usize;

        let progress = MATMUL_PROGRESS.load(Ordering::Relaxed);
        let rows_done = AtomicUsize::new(0);
        let rows_done_ref = &rows_done;
        let print_interval = (nu / 200).max(1);

        // Pass 1: symbolic — count nnz per output row
        let pass_start = Instant::now();
        let mut nnz_per_row = vec![0usize; nu];
        nnz_per_row.par_iter_mut().enumerate().for_each_init(
            || vec![false; nu],
            |mask, (i, nnz_out)| {
                let mut count = 0usize;
                let a_cols = self.col_trees.data(i);
                for &k_node in a_cols {
                    let k = k_node as usize;
                    let b_cols = other.col_trees.data(k);
                    for &j_node in b_cols {
                        let j = j_node as usize;
                        if !mask[j] {
                            mask[j] = true;
                            count += 1;
                        }
                    }
                }
                *nnz_out = count;
                // Clear mask
                for &k_node in a_cols {
                    let k = k_node as usize;
                    for &j_node in other.col_trees.data(k) {
                        mask[j_node as usize] = false;
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
        let mut values = vec![0 as Val; total_nnz];

        // Pass 2: numeric — fill col_idx and values in parallel
        let pass_start = Instant::now();
        let col_ptr = &row_ptr;
        let out_c = SendPtr(col_idx.as_mut_ptr());
        let out_v = SendPtr(values.as_mut_ptr());

        (0..nu).into_par_iter().for_each_init(
            || (vec![0 as Val; nu], Vec::<NodeId>::new()),
            |(acc, nz_cols), i| {
                let a_cols = self.col_trees.data(i);
                let a_start = self.col_trees.data_start(i);
                for (local, &k_node) in a_cols.iter().enumerate() {
                    let a_ik = self.values[a_start + local];
                    let k = k_node as usize;
                    let b_cols = other.col_trees.data(k);
                    let b_start = other.col_trees.data_start(k);
                    for (blocal, &j_node) in b_cols.iter().enumerate() {
                        let j = j_node as usize;
                        if acc[j] == 0 {
                            nz_cols.push(j_node);
                        }
                        acc[j] = sadd(acc[j], smul(a_ik, other.values[b_start + blocal]));
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

        Self::from_flat(n, &row_ptr, &col_idx, values)
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
            let a_cols = self.col_trees.data(r);
            let a_start = self.col_trees.data_start(r);
            let b_cols = other.col_trees.data(r);
            let b_start = other.col_trees.data_start(r);

            let mut ai = 0;
            let mut bi = 0;

            while ai < a_cols.len() && bi < b_cols.len() {
                let ac = a_cols[ai];
                let bc = b_cols[bi];
                if ac < bc {
                    col_idx.push(ac);
                    values.push(self.values[a_start + ai]);
                    ai += 1;
                } else if ac > bc {
                    col_idx.push(bc);
                    values.push(other.values[b_start + bi]);
                    bi += 1;
                } else {
                    let v = sadd(self.values[a_start + ai], other.values[b_start + bi]);
                    if v != 0 {
                        col_idx.push(ac);
                        values.push(v);
                    }
                    ai += 1;
                    bi += 1;
                }
            }
            while ai < a_cols.len() {
                col_idx.push(a_cols[ai]);
                values.push(self.values[a_start + ai]);
                ai += 1;
            }
            while bi < b_cols.len() {
                col_idx.push(b_cols[bi]);
                values.push(other.values[b_start + bi]);
                bi += 1;
            }

            row_ptr.push(col_idx.len());
        }

        Self::from_flat(n, &row_ptr, &col_idx, values)
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
                && next.col_eq(&current)
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
    pub fn connected_components_uf(&self) -> Vec<usize> {
        let nu = self.n as usize;
        let mut parent: Vec<usize> = (0..nu).collect();
        let mut rank = vec![0u8; nu];

        fn find(parent: &mut [usize], mut x: usize) -> usize {
            while parent[x] != x {
                parent[x] = parent[parent[x]];
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

    /// Reverse Cuthill-McKee reordering in place. Reduces bandwidth.
    pub fn rcm(&mut self) {
        let nu = self.n as usize;
        let mut visited = vec![false; nu];
        let mut order: Vec<NodeId> = Vec::with_capacity(nu);

        let deg = |node: usize| self.col_trees.data(node).len();

        for seed in 0..nu {
            if visited[seed] { continue; }

            let start = {
                let mut queue = std::collections::VecDeque::new();
                queue.push_back(seed);
                let mut last = seed;
                let mut vis2 = vec![false; nu];
                vis2[seed] = true;
                while let Some(u) = queue.pop_front() {
                    last = u;
                    for &v_node in self.col_trees.data(u) {
                        let v = v_node as usize;
                        if !vis2[v] {
                            vis2[v] = true;
                            queue.push_back(v);
                        }
                    }
                }
                last
            };

            let mut queue = std::collections::VecDeque::new();
            queue.push_back(start);
            visited[start] = true;
            while let Some(u) = queue.pop_front() {
                order.push(u as NodeId);
                let mut nbrs: Vec<usize> = Vec::new();
                for &v_node in self.col_trees.data(u) {
                    let v = v_node as usize;
                    if !visited[v] {
                        nbrs.push(v);
                    }
                }
                nbrs.sort_unstable_by_key(|&v| deg(v));
                for v in nbrs {
                    if !visited[v] {
                        visited[v] = true;
                        queue.push_back(v);
                    }
                }
            }
        }

        order.reverse();
        self.permute(&order);
    }

    /// Reorder rows and columns in place by a permutation. `perm[new] = old`.
    pub fn permute(&mut self, perm: &[NodeId]) {
        let nu = self.n as usize;
        let nnz = self.nnz();
        assert_eq!(perm.len(), nu);

        // Build inverse: inv[old] = new
        let mut inv = vec![0 as NodeId; nu];
        for (new_idx, &old) in perm.iter().enumerate() {
            inv[old as usize] = new_idx as NodeId;
        }

        // Count entries per new row
        let mut new_row_ptr = vec![0usize; nu + 1];
        for old_r in 0..nu {
            let count = self.col_trees.data(old_r).len();
            new_row_ptr[inv[old_r] as usize + 1] = count;
        }
        for i in 1..=nu {
            new_row_ptr[i] += new_row_ptr[i - 1];
        }

        // Scatter into new arrays
        let mut new_col = vec![0 as NodeId; nnz];
        let mut new_val = vec![0 as Val; nnz];
        let mut cursor = new_row_ptr[..nu].to_vec();
        for old_r in 0..nu {
            let new_r = inv[old_r] as usize;
            let cols = self.col_trees.data(old_r);
            let start = self.col_trees.data_start(old_r);
            for (local, &c) in cols.iter().enumerate() {
                let pos = cursor[new_r];
                new_col[pos] = inv[c as usize];
                new_val[pos] = self.values[start + local];
                cursor[new_r] += 1;
            }
        }

        // Sort columns within each new row
        let mut pairs: Vec<(NodeId, Val)> = Vec::new();
        for r in 0..nu {
            let s = new_row_ptr[r];
            let e = new_row_ptr[r + 1];
            if e - s <= 1 { continue; }
            pairs.clear();
            pairs.extend(new_col[s..e].iter().copied()
                .zip(new_val[s..e].iter().copied()));
            pairs.sort_unstable_by_key(|&(c, _)| c);
            for (i, &(c, v)) in pairs.iter().enumerate() {
                new_col[s + i] = c;
                new_val[s + i] = v;
            }
        }

        let mut col_trees = DenseBTreeList::new();
        for r in 0..nu {
            col_trees.add_from_sorted(&new_col[new_row_ptr[r]..new_row_ptr[r + 1]]);
        }
        self.values = new_val;
        self.col_trees = col_trees;
        self.perm = Some(perm.to_vec());
    }

    /// Undo the stored permutation in place, restoring original index order.
    pub fn unpermute(&mut self) {
        let Some(perm) = self.perm.take() else { return; };
        let nu = self.n as usize;

        let mut inv = vec![0 as NodeId; nu];
        for (new_idx, &old) in perm.iter().enumerate() {
            inv[old as usize] = new_idx as NodeId;
        }
        self.permute(&inv);
        self.perm = None;
    }

    /// Bandwidth stats: returns (max |r-c|, avg |r-c|) over all nonzeros.
    pub fn bandwidth_stats(&self) -> (usize, f64) {
        let mut max_bw: usize = 0;
        let mut sum_bw: u64 = 0;
        let mut count: u64 = 0;
        for r in 0..self.n as usize {
            for &c_node in self.col_trees.data(r) {
                let c = c_node as usize;
                let d = if r > c { r - c } else { c - r };
                max_bw = max_bw.max(d);
                sum_bw += d as u64;
                count += 1;
            }
        }
        (max_bw, sum_bw as f64 / count.max(1) as f64)
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

impl NDIndex<Val> for CsrBTreeMatrix {
    fn ndim(&self) -> usize { 2 }
    fn dim(&self, _axis: usize) -> usize { self.n as usize }
    fn get(&self, ix: &[usize]) -> Val { CsrBTreeMatrix::get(self, ix[0] as NodeId, ix[1] as NodeId) }
    fn set(&mut self, _ix: &[usize], _v: Val) { panic!("CsrBTreeMatrix is immutable after construction") }
    fn get_opt(&self, ix: &[usize]) -> Option<Val> {
        let r = ix[0] as NodeId;
        let c = ix[1] as NodeId;
        let start = self.col_trees.data_start(r as usize);
        match self.col_trees.index(r as usize, &c) {
            Ok(i) => Some(self.values[start + i]),
            Err(_) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn test_identity_matmul() {
        let m = CsrBTreeMatrix::from_edges(3, &[(0, 1), (1, 2)]);
        let id = CsrBTreeMatrix::identity(3);
        let result = m.matmul(&id);
        assert_eq!(result.get(0, 1), 1);
        assert_eq!(result.get(1, 2), 1);
        assert_eq!(result.get(0, 2), 0);
        assert_eq!(result.nnz(), 2);
    }

    #[test]
    fn test_path_counting_triangle() {
        let m = CsrBTreeMatrix::from_edges(3, &[(0, 1), (1, 2), (2, 0)]);
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
        let m = CsrBTreeMatrix::from_edges(2, &[(0, 1), (0, 1)]);
        assert_eq!(m.get(0, 1), 2);
    }

    #[test]
    fn test_path_counting_diamond() {
        let m = CsrBTreeMatrix::from_edges(4, &[
            (0, 1), (0, 2), (1, 3), (2, 3),
        ]);
        let m2 = m.matmul(&m);
        assert_eq!(m2.get(0, 3), 2);
    }

    #[test]
    fn test_reachability_chain() {
        let m = CsrBTreeMatrix::from_edges(4, &[(0, 1), (1, 2), (2, 3)]);
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
        let m = CsrBTreeMatrix::from_edges(n, &edges);
        let with_id = m.add(&CsrBTreeMatrix::identity(n));
        let (_stable, iters) = with_id.power_until_stable();
        assert!(iters <= 8, "took {iters} iterations for chain of {n}");
    }

    #[test]
    fn test_connected_components_two_triangles() {
        let m = CsrBTreeMatrix::from_edges_undirected(6, &[
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
        let m = CsrBTreeMatrix::new(5);
        let comp = m.connected_components();
        let unique: HashSet<usize> = comp.into_iter().collect();
        assert_eq!(unique.len(), 5);
    }

    #[test]
    fn test_lattice_2d_torus() {
        let m = CsrBTreeMatrix::lattice(&[3, 3], true);
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
    fn test_lattice_symmetry() {
        let m = CsrBTreeMatrix::lattice(&[4, 3], false);
        for r in 0..m.n {
            for (c, v) in m.row_iter(r) {
                assert_eq!(m.get(c, r), v, "asymmetry at ({r},{c})");
            }
        }
    }

    #[test]
    fn test_rcm_unpermute_roundtrip() {
        let orig = CsrBTreeMatrix::from_edges_undirected(6, &[
            (0, 3), (1, 4), (2, 5), (0, 1), (3, 4),
        ]);
        let mut m = orig.clone();
        m.rcm();
        assert!(m.perm.is_some());
        m.unpermute();
        assert!(m.perm.is_none());
        assert_eq!(m.values, orig.values);
        for r in 0..orig.n as usize {
            assert_eq!(m.col_trees.data(r), orig.col_trees.data(r));
        }
    }

    #[test]
    #[cfg(feature = "long-tests")]
    fn bench_csr_vs_csr_btree() {
        use crate::graph_csr::CsrMatrix;
        use std::time::Instant;
        use rand::prelude::StdRng;
        use rand::SeedableRng;

        let mut rng = StdRng::from_seed([42; 32]);
        let grid_sizes: &[usize] = &[10, 20, 30];
        let edges_per_node: &[f64] = &[2.0, 4.0, 8.0, 26.0];

        println!();
        println!("{:<6} {:>7} {:>6} {:>8}   {:>10} {:>10} {:>10} {:>10}   {:>7} {:>7}",
            "side", "nodes", "e/n", "nnz",
            "csr_us", "csr_par", "bt_us", "bt_par",
            "1T×", "MT×");
        println!("{}", "-".repeat(105));

        for &s in grid_sizes {
            let full_csr = CsrMatrix::lattice(&[s, s, s], true);
            let full_bt = CsrBTreeMatrix::lattice(&[s, s, s], true);
            let n = full_csr.n as usize;
            let full_epn = full_csr.nnz() as f64 / n as f64;

            for &epn in edges_per_node {
                let density = epn / full_epn;

                let a_csr = if density >= 1.0 {
                    full_csr.clone()
                } else {
                    full_csr.thin(&mut rng, density)
                };
                let a_bt = if density >= 1.0 {
                    full_bt.clone()
                } else {
                    // Build btree matrix from same edges as csr
                    let mut triplets: Vec<(NodeId, NodeId, Val)> = Vec::new();
                    for r in 0..a_csr.n {
                        for (c, v) in a_csr.row_iter(r) {
                            triplets.push((r, c, v));
                        }
                    }
                    CsrBTreeMatrix::from_coo(a_csr.n, &mut triplets)
                };

                let actual_epn = a_csr.nnz() as f64 / n as f64;
                let nnz = a_csr.nnz();

                // Warmup
                let _ = a_csr.matmul(&a_csr);
                let _ = a_bt.matmul(&a_bt);

                // Single-threaded
                let t0 = Instant::now();
                let _ = a_csr.matmul(&a_csr);
                let t_csr = t0.elapsed().as_micros();

                let t0 = Instant::now();
                let _ = a_bt.matmul(&a_bt);
                let t_bt = t0.elapsed().as_micros();

                // Parallel
                let t0 = Instant::now();
                let _ = a_csr.matmul_par(&a_csr);
                let t_csr_par = t0.elapsed().as_micros();

                let t0 = Instant::now();
                let _ = a_bt.matmul_par(&a_bt);
                let t_bt_par = t0.elapsed().as_micros();

                let ratio_1t = if t_bt > 0 { t_csr as f64 / t_bt as f64 } else { 0.0 };
                let ratio_mt = if t_bt_par > 0 { t_csr_par as f64 / t_bt_par as f64 } else { 0.0 };

                println!("{:<6} {:>7} {:>6.1} {:>8}   {:>10} {:>10} {:>10} {:>10}   {:>7.3} {:>7.3}",
                    s, n, actual_epn, nnz,
                    t_csr, t_csr_par, t_bt, t_bt_par,
                    ratio_1t, ratio_mt);
            }
            println!();
        }
    }

    #[test]
    #[cfg(feature = "long-tests")]
    fn bench_csr_vs_bt_power4() {
        use crate::graph_csr::CsrMatrix;
        use std::time::Instant;
        use rand::prelude::StdRng;
        use rand::SeedableRng;

        let mut rng = StdRng::from_seed([42; 32]);

        // 32×32 lattice, thin to 60%, add self-connections, raise to power 4
        let full_csr = CsrMatrix::lattice(&[32, 32], false);
        let a_csr = full_csr.thin(&mut rng, 0.6).add(&CsrMatrix::identity(full_csr.n));

        let mut triplets: Vec<(NodeId, NodeId, Val)> = Vec::new();
        for r in 0..a_csr.n {
            for (c, v) in a_csr.row_iter(r) {
                triplets.push((r, c, v));
            }
        }
        let a_bt = CsrBTreeMatrix::from_coo(a_csr.n, &mut triplets);

        let n = a_csr.n as usize;
        println!();
        println!("32x32 lattice, 60% edges + identity, n={n}, nnz={}", a_csr.nnz());
        println!();
        println!("{:<6} {:>10} {:>10} {:>10} {:>10}   {:>7} {:>7}",
            "power", "csr_us", "csr_par", "bt_us", "bt_par", "1T×", "MT×");
        println!("{}", "-".repeat(75));

        let mut prev_csr = a_csr.clone();
        let mut prev_bt = a_bt.clone();

        for step in 2..=8 {
            let t0 = Instant::now();
            let next_csr = prev_csr.matmul(&a_csr);
            let t_csr = t0.elapsed().as_micros();

            let t0 = Instant::now();
            let next_bt = prev_bt.matmul(&a_bt);
            let t_bt = t0.elapsed().as_micros();

            let t0 = Instant::now();
            let next_csr_par = prev_csr.matmul_par(&a_csr);
            let t_csr_par = t0.elapsed().as_micros();

            let t0 = Instant::now();
            let next_bt_par = prev_bt.matmul_par(&a_bt);
            let t_bt_par = t0.elapsed().as_micros();

            let ratio_1t = if t_bt > 0 { t_csr as f64 / t_bt as f64 } else { 0.0 };
            let ratio_mt = if t_bt_par > 0 { t_csr_par as f64 / t_bt_par as f64 } else { 0.0 };

            println!("A^{:<4} {:>10} {:>10} {:>10} {:>10}   {:>7.3} {:>7.3}  nnz={}",
                step, t_csr, t_csr_par, t_bt, t_bt_par,
                ratio_1t, ratio_mt, next_csr.nnz());

            prev_csr = next_csr;
            prev_bt = next_bt;
        }
    }
}
