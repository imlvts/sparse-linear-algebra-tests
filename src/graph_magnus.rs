use std::collections::BTreeMap;
use rand::Rng;

use magnus::{SparseMatrixCSR, magnus_spgemm, magnus_spgemm_parallel, MagnusConfig};

use crate::graph_sprs::Sat64;

/// Sparse integer matrix backed by MAGNUS SpGEMM library (ICS'25 row-categorization algorithm).
#[derive(Clone, Debug)]
pub struct MagnusMatrix {
    pub n: usize,
    pub mat: SparseMatrixCSR<Sat64>,
}

impl MagnusMatrix {
    /// Empty n×n matrix.
    pub fn new(n: usize) -> Self {
        Self {
            n,
            mat: SparseMatrixCSR::zeros(n, n),
        }
    }

    /// Identity matrix.
    pub fn identity(n: usize) -> Self {
        Self {
            n,
            mat: SparseMatrixCSR::identity(n),
        }
    }

    /// Build from COO triplets. Sorts by (row, col), merges duplicates by summing.
    fn from_coo(n: usize, triplets: &mut Vec<(usize, usize, u64)>) -> Self {
        triplets.sort_unstable_by_key(|&(r, c, _)| (r, c));

        // Dedup-sum
        let mut deduped: Vec<(usize, usize, u64)> = Vec::with_capacity(triplets.len());
        let mut prev_row = usize::MAX;
        let mut prev_col = usize::MAX;
        for &(r, c, v) in triplets.iter() {
            if r == prev_row && c == prev_col {
                deduped.last_mut().unwrap().2 += v;
            } else {
                deduped.push((r, c, v));
                prev_row = r;
                prev_col = c;
            }
        }

        // Build CSR arrays
        let mut row_ptr = Vec::with_capacity(n + 1);
        let mut col_idx = Vec::new();
        let mut values = Vec::new();
        let mut cur_row = 0;
        row_ptr.push(0);

        for &(r, c, v) in &deduped {
            if v == 0 { continue; }
            while cur_row < r {
                row_ptr.push(col_idx.len());
                cur_row += 1;
            }
            col_idx.push(c);
            values.push(Sat64(v));
        }
        while cur_row < n {
            row_ptr.push(col_idx.len());
            cur_row += 1;
        }

        Self {
            n,
            mat: SparseMatrixCSR::new(n, n, row_ptr, col_idx, values),
        }
    }

    /// Build from directed edge list.
    pub fn from_edges(n: usize, edges: &[(usize, usize)]) -> Self {
        let mut triplets: Vec<_> = edges.iter().map(|&(r, c)| (r, c, 1u64)).collect();
        Self::from_coo(n, &mut triplets)
    }

    /// Build from edge list, making it undirected (symmetric).
    pub fn from_edges_undirected(n: usize, edges: &[(usize, usize)]) -> Self {
        let mut triplets = Vec::with_capacity(edges.len() * 2);
        for &(r, c) in edges {
            triplets.push((r, c, 1u64));
            if r != c {
                triplets.push((c, r, 1u64));
            }
        }
        Self::from_coo(n, &mut triplets)
    }

    /// Build from named edge pairs. Returns the matrix and name->index mapping.
    pub fn from_adjacency<'a>(
        it: impl Iterator<Item = (&'a str, &'a str)>,
    ) -> (Self, BTreeMap<String, usize>) {
        let mut names: BTreeMap<String, usize> = BTreeMap::new();
        let mut edges = Vec::new();
        let mut next_id = 0usize;
        for (a, b) in it {
            let ai = *names.entry(a.to_string()).or_insert_with(|| {
                let id = next_id;
                next_id += 1;
                id
            });
            let bi = *names.entry(b.to_string()).or_insert_with(|| {
                let id = next_id;
                next_id += 1;
                id
            });
            edges.push((ai, bi));
        }
        (Self::from_edges(next_id, &edges), names)
    }

    /// Random directed graph with n nodes and m edges (no self-loops).
    pub fn random(rng: &mut impl Rng, n: usize, m: usize) -> Self {
        assert!(n >= 2, "need at least 2 nodes to avoid self-loops");
        let mut triplets = Vec::with_capacity(m);
        for _ in 0..m {
            let r = rng.random_range(0..n);
            let c = rng.random_range(0..n - 1);
            let c = if c >= r { c + 1 } else { c };
            triplets.push((r, c, 1u64));
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
                    if delta != 0 {
                        all_zero = false;
                    }
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
                if all_zero || !valid {
                    continue;
                }
                triplets.push((node, neighbor, 1u64));
            }

            for d in (0..ndim).rev() {
                coord[d] += 1;
                if coord[d] < dims[d] {
                    break;
                }
                coord[d] = 0;
            }
        }
        Self::from_coo(total, &mut triplets)
    }

    /// Randomly keep a fraction of edges, preserving symmetry.
    pub fn thin(&self, rng: &mut impl Rng, density: f64) -> Self {
        let mut triplets = Vec::new();
        for r in 0..self.n {
            let start = self.mat.row_ptr[r];
            let end = self.mat.row_ptr[r + 1];
            for idx in start..end {
                let c = self.mat.col_idx[idx];
                let v = self.mat.values[idx].0;
                if r <= c && rng.random_range(0.0f64..1.0) < density {
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
    pub fn get(&self, r: usize, c: usize) -> u64 {
        let start = self.mat.row_ptr[r];
        let end = self.mat.row_ptr[r + 1];
        match self.mat.col_idx[start..end].binary_search(&c) {
            Ok(i) => self.mat.values[start + i].0,
            Err(_) => 0,
        }
    }

    /// Number of non-zero entries.
    pub fn nnz(&self) -> usize {
        self.mat.nnz()
    }

    /// Matrix multiply via MAGNUS parallel SpGEMM.
    pub fn matmul(&self, other: &Self) -> Self {
        assert_eq!(self.n, other.n);
        let config = MagnusConfig::default();
        Self {
            n: self.n,
            mat: magnus_spgemm_parallel(&self.mat, &other.mat, &config),
        }
    }

    /// Matrix multiply via MAGNUS sequential SpGEMM.
    pub fn matmul_seq(&self, other: &Self) -> Self {
        assert_eq!(self.n, other.n);
        let config = MagnusConfig::default();
        Self {
            n: self.n,
            mat: magnus_spgemm(&self.mat, &other.mat, &config),
        }
    }

    /// Element-wise addition via sorted merge of each row.
    pub fn add(&self, other: &Self) -> Self {
        assert_eq!(self.n, other.n);
        let n = self.n;

        let mut row_ptr = Vec::with_capacity(n + 1);
        let mut col_idx = Vec::new();
        let mut values = Vec::new();
        row_ptr.push(0);

        for r in 0..n {
            let a_start = self.mat.row_ptr[r];
            let a_end = self.mat.row_ptr[r + 1];
            let b_start = other.mat.row_ptr[r];
            let b_end = other.mat.row_ptr[r + 1];

            let mut ai = a_start;
            let mut bi = b_start;

            while ai < a_end && bi < b_end {
                let ac = self.mat.col_idx[ai];
                let bc = other.mat.col_idx[bi];
                if ac < bc {
                    col_idx.push(ac);
                    values.push(self.mat.values[ai]);
                    ai += 1;
                } else if ac > bc {
                    col_idx.push(bc);
                    values.push(other.mat.values[bi]);
                    bi += 1;
                } else {
                    let v = self.mat.values[ai] + other.mat.values[bi];
                    if v.0 != 0 {
                        col_idx.push(ac);
                        values.push(v);
                    }
                    ai += 1;
                    bi += 1;
                }
            }
            while ai < a_end {
                col_idx.push(self.mat.col_idx[ai]);
                values.push(self.mat.values[ai]);
                ai += 1;
            }
            while bi < b_end {
                col_idx.push(other.mat.col_idx[bi]);
                values.push(other.mat.values[bi]);
                bi += 1;
            }

            row_ptr.push(col_idx.len());
        }

        Self {
            n,
            mat: SparseMatrixCSR::new(n, n, row_ptr, col_idx, values),
        }
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
                && next.mat.row_ptr == current.mat.row_ptr
                && next.mat.col_idx == current.mat.col_idx
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

        let mut component = vec![usize::MAX; self.n];
        let mut next_id = 0;

        for i in 0..self.n {
            if component[i] != usize::MAX {
                continue;
            }
            let id = next_id;
            next_id += 1;
            component[i] = id;
            for j in (i + 1)..self.n {
                if closure.get(i, j) > 0 && closure.get(j, i) > 0 {
                    component[j] = id;
                }
            }
        }
        component
    }

    /// Connected components via union-find. O(nnz * alpha(n)).
    pub fn connected_components_uf(&self) -> Vec<usize> {
        let mut parent: Vec<usize> = (0..self.n).collect();
        let mut rank = vec![0u8; self.n];

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
            if ra == rb {
                return;
            }
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
            let start = self.mat.row_ptr[r];
            let end = self.mat.row_ptr[r + 1];
            for idx in start..end {
                union(&mut parent, &mut rank, r, self.mat.col_idx[idx]);
            }
        }

        let mut id_map = vec![usize::MAX; self.n];
        let mut result = vec![0usize; self.n];
        let mut next_id = 0;
        for i in 0..self.n {
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
        let m = MagnusMatrix::from_edges(3, &[(0, 1), (1, 2)]);
        let id = MagnusMatrix::identity(3);
        let result = m.matmul(&id);
        assert_eq!(result.get(0, 1), 1);
        assert_eq!(result.get(1, 2), 1);
        assert_eq!(result.get(0, 2), 0);
        assert_eq!(result.nnz(), 2);
    }

    #[test]
    fn test_path_counting_triangle() {
        let m = MagnusMatrix::from_edges(3, &[(0, 1), (1, 2), (2, 0)]);
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
        let m = MagnusMatrix::from_edges(2, &[(0, 1), (0, 1)]);
        assert_eq!(m.get(0, 1), 2);
    }

    #[test]
    fn test_path_counting_diamond() {
        let m = MagnusMatrix::from_edges(4, &[(0, 1), (0, 2), (1, 3), (2, 3)]);
        let m2 = m.matmul(&m);
        assert_eq!(m2.get(0, 3), 2);
    }

    #[test]
    fn test_reachability_chain() {
        let m = MagnusMatrix::from_edges(4, &[(0, 1), (1, 2), (2, 3)]);
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
        let n = 64;
        let edges: Vec<(usize, usize)> = (0..n - 1).map(|i| (i, i + 1)).collect();
        let m = MagnusMatrix::from_edges(n, &edges);
        let with_id = m.add(&MagnusMatrix::identity(n));
        let (_stable, iters) = with_id.power_until_stable();
        assert!(iters <= 8, "took {iters} iterations for chain of {n}");
    }

    #[test]
    fn test_connected_components_two_triangles() {
        let m = MagnusMatrix::from_edges_undirected(6, &[
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
        let m = MagnusMatrix::new(5);
        let comp = m.connected_components();
        let unique: HashSet<usize> = comp.into_iter().collect();
        assert_eq!(unique.len(), 5);
    }

    #[test]
    fn test_from_adjacency_basic() {
        let edges = vec![("a", "b"), ("b", "c"), ("c", "a")];
        let (m, names) = MagnusMatrix::from_adjacency(edges.into_iter());
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
        let (m, names) = MagnusMatrix::from_adjacency(edges.into_iter());
        let x = names["x"];
        let y = names["y"];
        assert_eq!(m.get(x, y), 2);
        assert_eq!(m.get(y, x), 1);
    }

    #[test]
    fn test_from_adjacency_self_loop() {
        let edges = vec![("a", "a"), ("a", "b")];
        let (m, names) = MagnusMatrix::from_adjacency(edges.into_iter());
        let a = names["a"];
        let b = names["b"];
        assert_eq!(m.get(a, a), 1);
        assert_eq!(m.get(a, b), 1);
        assert_eq!(m.n, 2);
    }

    #[test]
    fn test_from_adjacency_components() {
        let edges = vec![("a", "b"), ("b", "a"), ("c", "d"), ("d", "c")];
        let (m, names) = MagnusMatrix::from_adjacency(edges.into_iter());
        let comp = m.connected_components();
        assert_eq!(comp[names["a"]], comp[names["b"]]);
        assert_eq!(comp[names["c"]], comp[names["d"]]);
        assert_ne!(comp[names["a"]], comp[names["c"]]);
    }

    #[test]
    fn test_lattice_1d_no_torus() {
        let m = MagnusMatrix::lattice(&[5], false);
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
        let m = MagnusMatrix::lattice(&[5], true);
        assert_eq!(m.get(0, 4), 1);
        assert_eq!(m.get(4, 0), 1);
        assert_eq!(m.nnz(), 10);
    }

    #[test]
    fn test_lattice_2d_corner() {
        let m = MagnusMatrix::lattice(&[3, 3], false);
        assert_eq!(m.n, 9);
        assert_eq!(m.get(0, 1), 1);
        assert_eq!(m.get(0, 3), 1);
        assert_eq!(m.get(0, 4), 1);
        assert_eq!(m.nnz() % 2, 0);
    }

    #[test]
    fn test_lattice_2d_center() {
        let m = MagnusMatrix::lattice(&[3, 3], false);
        let mut count = 0;
        for j in 0..9 {
            if m.get(4, j) > 0 {
                count += 1;
            }
        }
        assert_eq!(count, 8);
    }

    #[test]
    fn test_lattice_2d_torus() {
        let m = MagnusMatrix::lattice(&[3, 3], true);
        assert_eq!(m.n, 9);
        for i in 0..9 {
            let mut count = 0;
            for j in 0..9 {
                if m.get(i, j) > 0 {
                    count += 1;
                }
            }
            assert_eq!(count, 8, "node {i} should have 8 neighbors on torus");
        }
        assert_eq!(m.nnz(), 9 * 8);
    }

    #[test]
    fn test_lattice_2d_single_component() {
        let m = MagnusMatrix::lattice(&[4, 4], false);
        let comp = m.connected_components();
        for i in 1..16 {
            assert_eq!(comp[0], comp[i]);
        }
    }

    #[test]
    fn test_lattice_3d() {
        let m = MagnusMatrix::lattice(&[2, 2, 2], false);
        assert_eq!(m.n, 8);
        let mut count = 0;
        for j in 0..8 {
            if m.get(0, j) > 0 {
                count += 1;
            }
        }
        assert_eq!(count, 7);
        assert_eq!(m.nnz(), 8 * 7);
    }

    #[test]
    fn test_lattice_symmetry() {
        let m = MagnusMatrix::lattice(&[4, 3], false);
        for r in 0..m.n {
            let start = m.mat.row_ptr[r];
            let end = m.mat.row_ptr[r + 1];
            for idx in start..end {
                let c = m.mat.col_idx[idx];
                let v = m.mat.values[idx].0;
                assert_eq!(m.get(c, r), v, "asymmetry at ({r},{c})");
            }
        }
    }

    #[test]
    fn test_connected_components_single_component() {
        let m = MagnusMatrix::from_edges_undirected(4, &[(0, 1), (1, 2), (2, 3)]);
        let comp = m.connected_components();
        assert_eq!(comp[0], comp[1]);
        assert_eq!(comp[1], comp[2]);
        assert_eq!(comp[2], comp[3]);
    }

    #[test]
    fn test_matmul_seq_vs_par() {
        let m = MagnusMatrix::from_edges(4, &[(0, 1), (0, 2), (1, 3), (2, 3)]);
        let seq = m.matmul_seq(&m);
        let par = m.matmul(&m);
        assert_eq!(seq.nnz(), par.nnz());
        assert_eq!(seq.get(0, 3), par.get(0, 3));
        assert_eq!(seq.get(0, 3), 2);
    }

    #[test]
    #[cfg(any())]
    #[cfg(feature = "long-tests")]
    fn bench_repeated_exponentiation() {
        use crate::graph_csr::CsrMatrix;
        use rand::prelude::StdRng;
        use rand::SeedableRng;
        use std::time::Instant;

        let mut rng = StdRng::from_seed([42; 32]);
        let s = 30;
        let target_epn = 3.0;
        let max_steps = 7;
        const ITERS: u32 = 3;

        let full_csr = CsrMatrix::lattice(&[s, s, s], true);
        let full_magnus = MagnusMatrix::lattice(&[s, s, s], true);
        let n = full_csr.n;
        let full_epn = full_csr.nnz() as f64 / n as f64;
        let density = target_epn / full_epn;

        let a_csr = full_csr.thin(&mut rng, density);
        let mut triplets: Vec<(usize, usize, u64)> = Vec::new();
        for r in 0..n as usize {
            let start = a_csr.row_ptr[r];
            let end = a_csr.row_ptr[r + 1];
            for idx in start..end {
                triplets.push((r, a_csr.col_idx[idx] as usize, a_csr.values[idx]));
            }
        }
        let mut t2 = triplets.clone();
        let a_magnus = MagnusMatrix::from_coo(n as usize, &mut t2);

        let actual_epn = a_csr.nnz() as f64 / n as f64;
        println!();
        println!("Repeated exponentiation: {s}x{s}x{s} torus, {n} nodes, {:.1} e/n, {} nnz", actual_epn, a_csr.nnz());
        println!("step,nnz,csr_us,csr_par_us,magnus_seq_us,magnus_par_us,x_csr_par,x_magnus_seq,x_magnus_par");

        // Step 1 = A itself (no multiply)
        let mut powers_csr = vec![a_csr.clone()];
        let mut powers_magnus = vec![a_magnus.clone()];

        for step in 2..=max_steps {
            // Compute A^step = A^(step-1) * A
            let prev_csr = &powers_csr[step - 2];
            let prev_magnus = &powers_magnus[step - 2];

            // Warmup + correctness
            let r_csr = prev_csr.matmul(&a_csr);
            let r_par = prev_csr.matmul_par(&a_csr);
            let r_mseq = prev_magnus.matmul_seq(&a_magnus);
            let r_mpar = prev_magnus.matmul(&a_magnus);

            assert_eq!(r_csr.nnz(), r_par.nnz(), "step {step}: csr vs csr_par nnz");
            assert_eq!(r_csr.nnz(), r_mseq.nnz(), "step {step}: csr vs magnus_seq nnz");
            assert_eq!(r_csr.nnz(), r_mpar.nnz(), "step {step}: csr vs magnus_par nnz");

            let nnz = r_csr.nnz();

            // Timed runs
            let t0 = Instant::now();
            for _ in 0..ITERS { let _ = prev_csr.matmul(&a_csr); }
            let t_csr = t0.elapsed().as_micros() / ITERS as u128;

            let t0 = Instant::now();
            for _ in 0..ITERS { let _ = prev_csr.matmul_par(&a_csr); }
            let t_par = t0.elapsed().as_micros() / ITERS as u128;

            let t0 = Instant::now();
            for _ in 0..ITERS { let _ = prev_magnus.matmul_seq(&a_magnus); }
            let t_mseq = t0.elapsed().as_micros() / ITERS as u128;

            let t0 = Instant::now();
            for _ in 0..ITERS { let _ = prev_magnus.matmul(&a_magnus); }
            let t_mpar = t0.elapsed().as_micros() / ITERS as u128;

            let x = |base: u128, t: u128| -> String {
                if t > 0 { format!("{:.4}", base as f64 / t as f64) } else { "inf".to_string() }
            };

            println!(
                "{step},{nnz},{t_csr},{t_par},{t_mseq},{t_mpar},{},{},{}",
                x(t_csr, t_par),
                x(t_csr, t_mseq),
                x(t_csr, t_mpar),
            );

            powers_csr.push(r_csr);
            powers_magnus.push(r_mseq);
        }
    }

    #[test]
    #[cfg(any())]
    #[cfg(feature = "long-tests")]
    fn bench_matmul_magnus() {
        use crate::graph::SparseCountMatrix;
        use crate::graph_csr::CsrMatrix;
        use crate::graph_sprs::SprsMatrix;
        use rand::prelude::StdRng;
        use rand::SeedableRng;
        use std::time::Instant;

        let mut rng = StdRng::from_seed([42; 32]);
        let grid_sizes: &[usize] = &[5, 10, 20, 30];
        let edges_per_node: &[f64] = &[2.0, 3.0, 4.0, 8.0, 26.0];

        println!();
        println!("side,nodes,e_per_n,nnz,components,orig_btree_us,csr_us,csr_par_us,sprs_us,magnus_seq_us,magnus_par_us,x_csr,x_csr_par,x_sprs,x_magnus_seq,x_magnus_par");
        for &s in grid_sizes {
            let full_btree = SparseCountMatrix::lattice(&[s, s, s], true);
            let full_csr = CsrMatrix::lattice(&[s, s, s], true);
            let full_sprs = SprsMatrix::lattice(&[s, s, s], true);
            let full_magnus = MagnusMatrix::lattice(&[s, s, s], true);
            let n = full_btree.n;
            let full_epn = full_btree.nnz() as f64 / n as f64;

            for &epn in edges_per_node {
                let density = epn / full_epn;

                let a_bt = if density >= 1.0 {
                    full_btree.clone()
                } else {
                    full_btree.thin(&mut rng, density)
                };

                // Build other formats from same edges
                let triplets_u64: Vec<(usize, usize, u64)> = a_bt
                    .entries
                    .iter()
                    .map(|(&(r, c), &v)| (r, c, v))
                    .collect();

                let a_csr = if density >= 1.0 {
                    full_csr.clone()
                } else {
                    let mut t: Vec<(u32, u32, u64)> = triplets_u64.iter()
                        .map(|&(r, c, v)| (r as u32, c as u32, v)).collect();
                    CsrMatrix::from_coo(n as u32, &mut t)
                };

                let a_sprs = if density >= 1.0 {
                    full_sprs.clone()
                } else {
                    SprsMatrix::from_coo(n, &triplets_u64)
                };

                let a_magnus = if density >= 1.0 {
                    full_magnus.clone()
                } else {
                    let mut t = triplets_u64.clone();
                    MagnusMatrix::from_coo(n, &mut t)
                };

                let b_bt = a_bt.clone();
                let b_csr = a_csr.clone();
                let b_sprs = a_sprs.clone();
                let b_magnus = a_magnus.clone();

                const ITERS: u32 = 10;

                // Warmup + verify on first iteration
                let r_bt = a_bt.matmul(&b_bt);
                let r_csr = a_csr.matmul(&b_csr);
                let r_par = a_csr.matmul_par(&b_csr);
                let r_sprs = a_sprs.matmul(&b_sprs);
                let r_magnus_seq = a_magnus.matmul_seq(&b_magnus);
                let r_magnus_par = a_magnus.matmul(&b_magnus);

                assert_eq!(r_bt.nnz(), r_csr.nnz(), "nnz mismatch: btree vs csr (s={s}, epn={epn})");
                assert_eq!(r_csr.nnz(), r_par.nnz(), "nnz mismatch: csr vs csr_par (s={s}, epn={epn})");
                assert_eq!(r_csr.nnz(), r_sprs.nnz(), "nnz mismatch: csr vs sprs (s={s}, epn={epn})");
                assert_eq!(r_csr.nnz(), r_magnus_seq.nnz(), "nnz mismatch: csr vs magnus_seq (s={s}, epn={epn})");
                assert_eq!(r_csr.nnz(), r_magnus_par.nnz(), "nnz mismatch: csr vs magnus_par (s={s}, epn={epn})");
                for i in 0..n.min(10) {
                    for j in 0..n.min(10) {
                        let v_csr = r_csr.get(i as u32, j as u32);
                        assert_eq!(r_bt.get(i, j), v_csr, "btree vs csr at ({i},{j})");
                        assert_eq!(r_par.get(i as u32, j as u32), v_csr, "csr_par vs csr at ({i},{j})");
                        assert_eq!(r_sprs.get(i, j), v_csr, "sprs vs csr at ({i},{j})");
                        assert_eq!(r_magnus_seq.get(i, j), v_csr, "magnus_seq vs csr at ({i},{j})");
                        assert_eq!(r_magnus_par.get(i, j), v_csr, "magnus_par vs csr at ({i},{j})");
                    }
                }
                drop((r_bt, r_csr, r_par, r_sprs, r_magnus_seq, r_magnus_par));

                // Timed runs (10x average)
                let t0 = Instant::now();
                for _ in 0..ITERS { let _ = a_bt.matmul(&b_bt); }
                let t_bt = t0.elapsed().as_micros() / ITERS as u128;

                let t0 = Instant::now();
                for _ in 0..ITERS { let _ = a_csr.matmul(&b_csr); }
                let t_csr = t0.elapsed().as_micros() / ITERS as u128;

                let t0 = Instant::now();
                for _ in 0..ITERS { let _ = a_csr.matmul_par(&b_csr); }
                let t_par = t0.elapsed().as_micros() / ITERS as u128;

                let t0 = Instant::now();
                for _ in 0..ITERS { let _ = a_sprs.matmul(&b_sprs); }
                let t_sprs = t0.elapsed().as_micros() / ITERS as u128;

                let t0 = Instant::now();
                for _ in 0..ITERS { let _ = a_magnus.matmul_seq(&b_magnus); }
                let t_magnus_seq = t0.elapsed().as_micros() / ITERS as u128;

                let t0 = Instant::now();
                for _ in 0..ITERS { let _ = a_magnus.matmul(&b_magnus); }
                let t_magnus_par = t0.elapsed().as_micros() / ITERS as u128;

                let components = a_csr.num_components();

                let x = |t: u128| -> String {
                    if t_bt > 0 {
                        format!("{:.4}", t_bt as f64 / t as f64)
                    } else {
                        "inf".to_string()
                    }
                };

                println!(
                    "{s},{n},{epn:.0},{},{components},{t_bt},{t_csr},{t_par},{t_sprs},{t_magnus_seq},{t_magnus_par},{},{},{},{},{}",
                    a_bt.nnz(),
                    x(t_csr),
                    x(t_par),
                    x(t_sprs),
                    x(t_magnus_seq),
                    x(t_magnus_par),
                );
            }
        }
    }
}
