// use std::collections::HashMap as Map;
use std::collections::BTreeMap as Map;
use rand::Rng;

/// Sparse integer matrix represented as a Map of (row, col) -> count.
/// Only stores non-zero entries. Used for counting paths in graphs:
/// if A is the adjacency matrix, then A^k[i,j] = number of paths of length k from i to j.
#[derive(Clone, Debug)]
pub struct SparseCountMatrix {
    /// Number of nodes (matrix is n x n)
    pub n: usize,
    /// Non-zero entries: (row, col) -> value
    pub entries: Map<(usize, usize), u64>,
}

impl SparseCountMatrix {
    pub fn new(n: usize) -> Self {
        Self { n, entries: Map::new() }
    }

    /// Create an identity matrix
    pub fn identity(n: usize) -> Self {
        let mut m = Self::new(n);
        for i in 0..n {
            m.entries.insert((i, i), 1);
        }
        m
    }

    /// Build from an edge list (directed), each edge contributes 1
    pub fn from_edges(n: usize, edges: &[(usize, usize)]) -> Self {
        let mut m = Self::new(n);
        for &(r, c) in edges {
            *m.entries.entry((r, c)).or_insert(0) += 1;
        }
        m
    }

    /// Build from an edge list, making it undirected (symmetric)
    pub fn from_edges_undirected(n: usize, edges: &[(usize, usize)]) -> Self {
        let mut m = Self::new(n);
        for &(r, c) in edges {
            *m.entries.entry((r, c)).or_insert(0) += 1;
            if r != c {
                *m.entries.entry((c, r)).or_insert(0) += 1;
            }
        }
        m
    }

    /// Build from named edge pairs. Assigns indices to names in order of first appearance.
    /// Returns the matrix and the name -> index mapping.
    pub fn from_adjacency<'a>(it: impl Iterator<Item = (&'a str, &'a str)>) -> (Self, Map<String, usize>) {
        let mut names: Map<String, usize> = Map::new();
        let mut edges = Vec::new();
        let mut next_id = 0usize;
        for (a, b) in it {
            let ai = *names.entry(a.to_string()).or_insert_with(|| { let id = next_id; next_id += 1; id });
            let bi = *names.entry(b.to_string()).or_insert_with(|| { let id = next_id; next_id += 1; id });
            edges.push((ai, bi));
        }
        (Self::from_edges(next_id, &edges), names)
    }

    /// Generate a random directed graph with n nodes and exactly m edges (no self-loops).
    /// Duplicate edges between the same pair increment the count.
    pub fn random(rng: &mut impl Rng, n: usize, m: usize) -> Self {
        assert!(n >= 2, "need at least 2 nodes to avoid self-loops");
        let mut mat = Self::new(n);
        for _ in 0..m {
            let r = rng.random_range(0..n);
            let c = rng.random_range(0..n - 1);
            let c = if c >= r { c + 1 } else { c };
            *mat.entries.entry((r, c)).or_insert(0) += 1;
        }
        mat
    }

    /// Generate an N-dimensional lattice graph with Moore neighborhood connectivity.
    /// Each node connects to all neighbors differing by at most 1 in each coordinate
    /// (up to 3^N - 1 neighbors per node, excluding self-loops).
    ///
    /// `dims`: size along each dimension, e.g. &[10, 10] for a 10x10 grid.
    /// `torus`: if true, coordinates wrap around (periodic boundary).
    ///
    /// Edges are undirected (symmetric). Node index is row-major:
    /// for dims [d0, d1, ..., dk], node (i0, i1, ..., ik) has index
    /// i0*d1*d2*...*dk + i1*d2*...*dk + ... + ik.
    pub fn lattice(dims: &[usize], torus: bool) -> Self {
        let ndim = dims.len();
        let total: usize = dims.iter().product();
        let mut mat = Self::new(total);

        // Strides for row-major indexing
        let mut strides = vec![1usize; ndim];
        for i in (0..ndim.saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * dims[i + 1];
        }

        // Iterate over all nodes
        let mut coord = vec![0usize; ndim];
        for node in 0..total {
            // Iterate over all 3^ndim offset combos via counting in base 3
            let n_neighbors = 3usize.pow(ndim as u32);
            for off_idx in 0..n_neighbors {
                // Decode off_idx into offsets (-1, 0, +1) per dimension
                let mut tmp = off_idx;
                let mut all_zero = true;
                let mut neighbor = 0usize;
                let mut valid = true;
                for d in 0..ndim {
                    let delta = (tmp % 3) as isize - 1; // -1, 0, or 1
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
                mat.entries.insert((node, neighbor), 1);
            }

            // Advance coordinate
            for d in (0..ndim).rev() {
                coord[d] += 1;
                if coord[d] < dims[d] { break; }
                coord[d] = 0;
            }
        }
        mat
    }

    /// Randomly keep a fraction of edges. `density` in 0.0..=1.0.
    /// Keeps the matrix symmetric if it was symmetric.
    pub fn thin(&self, rng: &mut impl Rng, density: f64) -> Self {
        let mut mat = Self::new(self.n);
        for (&(r, c), &v) in &self.entries {
            if r <= c && rng.random_range(0.0..1.0) < density {
                mat.entries.insert((r, c), v);
                if r != c && self.get(c, r) > 0 {
                    mat.entries.insert((c, r), self.get(c, r));
                }
            }
        }
        mat
    }

    pub fn get(&self, r: usize, c: usize) -> u64 {
        self.entries.get(&(r, c)).copied().unwrap_or(0)
    }

    pub fn set(&mut self, r: usize, c: usize, v: u64) {
        if v == 0 {
            self.entries.remove(&(r, c));
        } else {
            self.entries.insert((r, c), v);
        }
    }

    pub fn nnz(&self) -> usize {
        self.entries.len()
    }

    /// Integer matrix multiply using intermediate row maps.
    pub fn matmul(&self, other: &Self) -> Self {
        self.matmul_maps(other)
    }

    /// matmul via intermediate row maps (original)
    pub fn matmul_maps(&self, other: &Self) -> Self {
        assert_eq!(self.n, other.n);
        let n = self.n;

        // Build row -> [(col, val)] for self
        let mut a_rows: Map<usize, Vec<(usize, u64)>> = Map::new();
        for (&(r, c), &v) in &self.entries {
            a_rows.entry(r).or_default().push((c, v));
        }

        // Build row -> [(col, val)] for other
        let mut b_rows: Map<usize, Vec<(usize, u64)>> = Map::new();
        for (&(r, c), &v) in &other.entries {
            b_rows.entry(r).or_default().push((c, v));
        }

        let mut result = Self::new(n);
        for (&i, a_cols) in &a_rows {
            for &(k, a_ik) in a_cols {
                if let Some(b_cols) = b_rows.get(&k) {
                    for &(j, b_kj) in b_cols {
                        let e = result.entries.entry((i, j)).or_insert(0);
                        *e += a_ik * b_kj;
                    }
                }
            }
        }
        result
    }

    /// matmul via BTreeMap range queries (no intermediate allocation)
    pub fn matmul_range(&self, other: &Self) -> Self {
        assert_eq!(self.n, other.n);
        let mut result = Self::new(self.n);

        for (&(i, k), &a_ik) in &self.entries {
            for (&(_, j), &b_kj) in other.entries.range((k, 0)..=(k, usize::MAX)) {
                *result.entries.entry((i, j)).or_insert(0) += a_ik * b_kj;
            }
        }
        result
    }

    /// Element-wise addition
    pub fn add(&self, other: &Self) -> Self {
        assert_eq!(self.n, other.n);
        let mut result = self.clone();
        for (&k, &v) in &other.entries {
            *result.entries.entry(k).or_insert(0) += v;
        }
        result
    }

    /// Compute A + A^2 + A^3 + ... until the sparsity pattern stabilizes
    /// (i.e. no new reachable pairs appear). The values give cumulative path counts.
    /// Returns (cumulative_matrix, power_at_convergence).
    pub fn reachability_sum(&self) -> (Self, usize) {
        // We track both the running sum S = A + A^2 + ... + A^k
        // and the current power P = A^k.
        // The sparsity pattern stabilizes when P doesn't introduce new non-zero positions in S.
        let mut power = self.clone();      // A^1
        let mut sum = self.clone();        // S = A
        let mut k = 1usize;
        loop {
            power = power.matmul(self);    // A^(k+1)
            k += 1;
            let new_sum = sum.add(&power);
            // Check if sparsity pattern changed
            if new_sum.nnz() == sum.nnz() {
                return (new_sum, k);
            }
            sum = new_sum;
        }
    }

    /// Repeated squaring: compute A^(2^k) until sparsity pattern stabilizes.
    /// Returns (A^(2^k), k) where no new non-zero positions appear.
    /// This finds the connectivity pattern faster than linear iteration for long chains.
    pub fn power_until_stable(&self) -> (Self, usize) {
        let mut current = self.clone();
        let mut k = 0usize;
        loop {
            let next = current.matmul(&current);
            k += 1;
            // Check if sparsity pattern changed (ignore values, just positions)
            if next.nnz() == current.nnz()
                && next.entries.keys().all(|k| current.entries.contains_key(k))
            {
                return (next, k);
            }
            current = next;
        }
    }

    /// Extract connected components using the sparsity pattern of A^* (transitive closure).
    /// Two nodes i,j are in the same (strongly connected) component if both
    /// A*[i,j] > 0 and A*[j,i] > 0. For undirected graphs this is the same as
    /// weak connectivity.
    /// Returns a Vec where result[i] = component id for node i.
    pub fn connected_components(&self) -> Vec<usize> {
        // Add identity so self-loops exist, then square until stable
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

    /// Print the matrix (for debugging, only practical for small n)
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
        let m = SparseCountMatrix::from_edges(3, &[(0, 1), (1, 2)]);
        let id = SparseCountMatrix::identity(3);
        let result = m.matmul(&id);
        assert_eq!(result.get(0, 1), 1);
        assert_eq!(result.get(1, 2), 1);
        assert_eq!(result.get(0, 2), 0);
        assert_eq!(result.nnz(), 2);
    }

    #[test]
    fn test_path_counting_triangle() {
        // Complete directed triangle: 0->1, 1->2, 2->0
        let m = SparseCountMatrix::from_edges(3, &[(0, 1), (1, 2), (2, 0)]);
        // A^2: paths of length 2
        let m2 = m.matmul(&m);
        assert_eq!(m2.get(0, 2), 1); // 0->1->2
        assert_eq!(m2.get(1, 0), 1); // 1->2->0
        assert_eq!(m2.get(2, 1), 1); // 2->0->1
        // A^3: paths of length 3 (full cycle back to self)
        let m3 = m2.matmul(&m);
        assert_eq!(m3.get(0, 0), 1); // 0->1->2->0
        assert_eq!(m3.get(1, 1), 1); // 1->2->0->1
        assert_eq!(m3.get(2, 2), 1); // 2->0->1->2
    }

    #[test]
    fn test_path_counting_parallel_paths() {
        // Two parallel edges from 0 to 1 (multigraph)
        let m = SparseCountMatrix::from_edges(2, &[(0, 1), (0, 1)]);
        assert_eq!(m.get(0, 1), 2);
    }

    #[test]
    fn test_path_counting_diamond() {
        // Diamond: 0->1, 0->2, 1->3, 2->3
        let m = SparseCountMatrix::from_edges(4, &[
            (0, 1), (0, 2), (1, 3), (2, 3),
        ]);
        let m2 = m.matmul(&m);
        // Paths of length 2 from 0 to 3: 0->1->3 and 0->2->3
        assert_eq!(m2.get(0, 3), 2);
    }

    #[test]
    fn test_reachability_chain() {
        // Chain: 0 -> 1 -> 2 -> 3
        let m = SparseCountMatrix::from_edges(4, &[(0, 1), (1, 2), (2, 3)]);
        let (sum, _k) = m.reachability_sum();
        // All forward pairs should be reachable
        assert!(sum.get(0, 1) > 0);
        assert!(sum.get(0, 2) > 0);
        assert!(sum.get(0, 3) > 0);
        assert!(sum.get(1, 2) > 0);
        assert!(sum.get(1, 3) > 0);
        assert!(sum.get(2, 3) > 0);
        // No backward reachability
        assert_eq!(sum.get(3, 0), 0);
        assert_eq!(sum.get(2, 0), 0);
    }

    #[test]
    fn test_power_until_stable_chain() {
        // Chain of 64 nodes
        let n = 64;
        let edges: Vec<(usize, usize)> = (0..n - 1).map(|i| (i, i + 1)).collect();
        let m = SparseCountMatrix::from_edges(n, &edges);
        let with_id = m.add(&SparseCountMatrix::identity(n));
        let (_stable, iters) = with_id.power_until_stable();
        // Should converge in O(log n) squarings
        assert!(iters <= 8, "took {iters} iterations for chain of {n}");
    }

    #[test]
    fn test_connected_components_two_triangles() {
        // Two separate triangles: {0,1,2} and {3,4,5}
        let m = SparseCountMatrix::from_edges_undirected(6, &[
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
        let m = SparseCountMatrix::new(5);
        let comp = m.connected_components();
        let unique: HashSet<usize> = comp.into_iter().collect();
        assert_eq!(unique.len(), 5);
    }

    #[test]
    fn test_from_adjacency_basic() {
        let edges = vec![("a", "b"), ("b", "c"), ("c", "a")];
        let (m, names) = SparseCountMatrix::from_adjacency(edges.into_iter());
        assert_eq!(names.len(), 3);
        assert_eq!(m.n, 3);
        // Triangle: each directed edge present once
        assert_eq!(m.nnz(), 3);
        let a = names["a"];
        let b = names["b"];
        let c = names["c"];
        assert_eq!(m.get(a, b), 1);
        assert_eq!(m.get(b, c), 1);
        assert_eq!(m.get(c, a), 1);
        assert_eq!(m.get(a, c), 0); // no reverse edge
    }

    #[test]
    fn test_from_adjacency_duplicate_edges() {
        let edges = vec![("x", "y"), ("x", "y"), ("y", "x")];
        let (m, names) = SparseCountMatrix::from_adjacency(edges.into_iter());
        let x = names["x"];
        let y = names["y"];
        assert_eq!(m.get(x, y), 2); // two parallel edges
        assert_eq!(m.get(y, x), 1);
    }

    #[test]
    fn test_from_adjacency_self_loop() {
        let edges = vec![("a", "a"), ("a", "b")];
        let (m, names) = SparseCountMatrix::from_adjacency(edges.into_iter());
        let a = names["a"];
        let b = names["b"];
        assert_eq!(m.get(a, a), 1);
        assert_eq!(m.get(a, b), 1);
        assert_eq!(m.n, 2);
    }

    #[test]
    fn test_from_adjacency_components() {
        // Two disconnected pairs: {a,b} and {c,d}
        let edges = vec![("a", "b"), ("b", "a"), ("c", "d"), ("d", "c")];
        let (m, names) = SparseCountMatrix::from_adjacency(edges.into_iter());
        let comp = m.connected_components();
        assert_eq!(comp[names["a"]], comp[names["b"]]);
        assert_eq!(comp[names["c"]], comp[names["d"]]);
        assert_ne!(comp[names["a"]], comp[names["c"]]);
    }

    #[test]
    fn test_lattice_1d_no_torus() {
        // 1D chain of 5 nodes: 0-1-2-3-4
        let m = SparseCountMatrix::lattice(&[5], false);
        assert_eq!(m.n, 5);
        // Endpoints have 1 neighbor, interior nodes have 2
        assert_eq!(m.get(0, 1), 1);
        assert_eq!(m.get(1, 0), 1);
        assert_eq!(m.get(0, 0), 0); // no self-loops
        assert_eq!(m.get(4, 3), 1);
        assert_eq!(m.get(4, 0), 0); // no wrap
        // Total edges: 2 endpoints * 1 + 3 interior * 2 = 8
        assert_eq!(m.nnz(), 8);
    }

    #[test]
    fn test_lattice_1d_torus() {
        let m = SparseCountMatrix::lattice(&[5], true);
        // Now 0 and 4 are also neighbors
        assert_eq!(m.get(0, 4), 1);
        assert_eq!(m.get(4, 0), 1);
        // Every node has exactly 2 neighbors
        assert_eq!(m.nnz(), 10);
    }

    #[test]
    fn test_lattice_2d_corner() {
        // 3x3 grid, no torus
        let m = SparseCountMatrix::lattice(&[3, 3], false);
        assert_eq!(m.n, 9);
        // Node (0,0) = index 0 has 3 neighbors: (0,1)=1, (1,0)=3, (1,1)=4
        assert_eq!(m.get(0, 1), 1);
        assert_eq!(m.get(0, 3), 1);
        assert_eq!(m.get(0, 4), 1); // diagonal
        assert_eq!(m.nnz() % 2, 0); // symmetric, so even edge count
    }

    #[test]
    fn test_lattice_2d_center() {
        // 3x3 grid, center node (1,1) = index 4 has 8 neighbors
        let m = SparseCountMatrix::lattice(&[3, 3], false);
        let mut count = 0;
        for j in 0..9 {
            if m.get(4, j) > 0 { count += 1; }
        }
        assert_eq!(count, 8);
    }

    #[test]
    fn test_lattice_2d_torus() {
        // 3x3 torus: every node has exactly 8 neighbors
        let m = SparseCountMatrix::lattice(&[3, 3], true);
        assert_eq!(m.n, 9);
        for i in 0..9 {
            let mut count = 0;
            for j in 0..9 {
                if m.get(i, j) > 0 { count += 1; }
            }
            assert_eq!(count, 8, "node {i} should have 8 neighbors on torus");
        }
        assert_eq!(m.nnz(), 9 * 8); // 9 nodes * 8 neighbors each
    }

    #[test]
    fn test_lattice_2d_single_component() {
        let m = SparseCountMatrix::lattice(&[4, 4], false);
        let comp = m.connected_components();
        // All nodes in one component
        for i in 1..16 {
            assert_eq!(comp[0], comp[i]);
        }
    }

    #[test]
    fn test_lattice_3d() {
        // 2x2x2 cube, no torus
        let m = SparseCountMatrix::lattice(&[2, 2, 2], false);
        assert_eq!(m.n, 8);
        // Corner node 0 = (0,0,0) connects to all other 7 nodes
        // (each differs by at most 1 in each coord)
        let mut count = 0;
        for j in 0..8 {
            if m.get(0, j) > 0 { count += 1; }
        }
        assert_eq!(count, 7); // 2^3 - 1 = 7
        // Symmetric, single component
        assert_eq!(m.nnz(), 8 * 7); // every node sees all others
    }

    #[test]
    fn test_lattice_symmetry() {
        // Lattice should always be symmetric
        let m = SparseCountMatrix::lattice(&[4, 3], false);
        for (&(r, c), &v) in m.entries.iter() {
            assert_eq!(m.get(c, r), v, "asymmetry at ({r},{c})");
        }
    }

    #[test]
    fn test_connected_components_single_component() {
        let m = SparseCountMatrix::from_edges_undirected(4, &[
            (0, 1), (1, 2), (2, 3),
        ]);
        let comp = m.connected_components();
        assert_eq!(comp[0], comp[1]);
        assert_eq!(comp[1], comp[2]);
        assert_eq!(comp[2], comp[3]);
    }

    #[test]
    #[cfg(feature = "long-tests")]
    fn bench_matmul_maps_vs_range() {
        use std::time::Instant;
        use rand::prelude::StdRng;
        use rand::SeedableRng;

        let mut rng = StdRng::from_seed([42; 32]);
        let grid_sizes: &[usize] = &[5, 10, 20, 30];
        let edges_per_node: &[f64] = &[2.0, 8.0, 26.0];

        println!();
        println!("side,nodes,e_per_n,nnz_a,nnz_b,maps_us,range_us,ratio,match");
        for &s in grid_sizes {
            let full = SparseCountMatrix::lattice(&[s, s, s], true);
            let n = full.n;
            let full_epn = full.nnz() as f64 / n as f64;

            for &epn in edges_per_node {
                let density = epn / full_epn;
                let a = if density >= 1.0 { full.clone() } else { full.thin(&mut rng, density) };
                let b = if density >= 1.0 { full.clone() } else { full.thin(&mut rng, density) };

                let t0 = Instant::now();
                let r_maps = a.matmul_maps(&b);
                let t_maps = t0.elapsed().as_micros();

                let t0 = Instant::now();
                let r_range = a.matmul_range(&b);
                let t_range = t0.elapsed().as_micros();

                let ok = r_maps.entries == r_range.entries;
                let ratio = if t_maps > 0 { t_range as f64 / t_maps as f64 } else { 0.0 };

                println!("{s},{n},{epn:.0},{},{},{t_maps},{t_range},{ratio:.2},{ok}",
                    a.nnz(), b.nnz());
            }
        }
    }

    #[test]
    #[cfg(feature = "long-tests")]
    fn bench_lattice_3d_power_until_stable() {
        use std::time::Instant;
        use rand::prelude::StdRng;
        use rand::SeedableRng;

        let mut rng = StdRng::from_seed([42; 32]);
        // side lengths: 5^3=125, 10^3=1000, 20^3=8000, 30^3=27000, 46^3=97336
        let grid_sizes: &[usize] = &[5, 10, 20, 30, 46];
        // edges per node: from sparse to full Moore neighborhood (max 26 for 3D)
        let edges_per_node: &[f64] = &[1.0, 2.0, 4.0, 8.0, 13.0, 26.0];

        println!();
        println!("side,nodes,e_per_n,edges,components,iters,time_us");
        for &s in grid_sizes {
            let full = SparseCountMatrix::lattice(&[s, s, s], true);
            let n = full.n;
            let full_epn = full.nnz() as f64 / n as f64; // 26 for torus

            for &epn in edges_per_node {
                let density = epn / full_epn;
                let m = if density >= 1.0 { full.clone() } else { full.thin(&mut rng, density) };
                let nnz = m.nnz();
                let actual_epn = nnz as f64 / n as f64;
                let with_id = m.add(&SparseCountMatrix::identity(n));

                let t0 = Instant::now();
                let (_stable, iters) = with_id.power_until_stable();
                let elapsed = t0.elapsed().as_micros();

                let comp = m.connected_components();
                let n_comp = comp.iter().collect::<HashSet<_>>().len();

                println!("{s},{n},{actual_epn:.1},{nnz},{n_comp},{iters},{elapsed}");
            }
        }
    }
}
