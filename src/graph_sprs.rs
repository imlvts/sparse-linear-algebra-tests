use std::collections::BTreeMap;
use std::fmt;
use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Rem, Sub};

use num_traits::{Num, One, Zero};
use rand::Rng;
use sprs::{CsMat, TriMat};

use crate::graph_csr::GRAPH_USE_SATURATING_ARITH;

// ---------------------------------------------------------------------------
// Sat64 — newtype over u64 with saturating arithmetic
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Sat64(pub u64);

impl fmt::Display for Sat64 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Add for Sat64 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        if GRAPH_USE_SATURATING_ARITH {
            Sat64(self.0.saturating_add(rhs.0))
        } else {
            Sat64(self.0 + rhs.0)
        }
    }
}

impl<'a, 'b> Add<&'b Sat64> for &'a Sat64 {
    type Output = Sat64;
    #[inline]
    fn add(self, rhs: &'b Sat64) -> Sat64 {
        *self + *rhs
    }
}

impl AddAssign for Sat64 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Sub for Sat64 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Sat64(self.0.saturating_sub(rhs.0))
    }
}

impl Mul for Sat64 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        if GRAPH_USE_SATURATING_ARITH {
            Sat64(self.0.saturating_mul(rhs.0))
        } else {
            Sat64(self.0 * rhs.0)
        }
    }
}

impl<'a, 'b> Mul<&'b Sat64> for &'a Sat64 {
    type Output = Sat64;
    #[inline]
    fn mul(self, rhs: &'b Sat64) -> Sat64 {
        *self * *rhs
    }
}

impl MulAssign for Sat64 {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl Div for Sat64 {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self {
        Sat64(self.0 / rhs.0)
    }
}

impl Rem for Sat64 {
    type Output = Self;
    #[inline]
    fn rem(self, rhs: Self) -> Self {
        Sat64(self.0 % rhs.0)
    }
}

impl Zero for Sat64 {
    #[inline]
    fn zero() -> Self {
        Sat64(0)
    }
    #[inline]
    fn is_zero(&self) -> bool {
        self.0 == 0
    }
}

impl One for Sat64 {
    #[inline]
    fn one() -> Self {
        Sat64(1)
    }
}

impl Num for Sat64 {
    type FromStrRadixErr = <u64 as Num>::FromStrRadixErr;
    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        u64::from_str_radix(str, radix).map(Sat64)
    }
}

// ---------------------------------------------------------------------------
// SprsMatrix — wrapper around sprs::CsMat<Sat64>
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct SprsMatrix {
    pub n: usize,
    pub mat: CsMat<Sat64>,
}

impl SprsMatrix {
    /// Empty n×n matrix.
    pub fn new(n: usize) -> Self {
        Self {
            n,
            mat: CsMat::zero((n, n)),
        }
    }

    /// Identity matrix.
    pub fn identity(n: usize) -> Self {
        Self {
            n,
            mat: CsMat::eye(n),
        }
    }

    /// Build from COO triplets via TriMat (duplicates are summed).
    fn from_coo(n: usize, triplets: &[(usize, usize, u64)]) -> Self {
        let mut tri = TriMat::with_capacity((n, n), triplets.len());
        for &(r, c, v) in triplets {
            tri.add_triplet(r, c, Sat64(v));
        }
        Self {
            n,
            mat: tri.to_csr(),
        }
    }

    /// Build from directed edge list.
    pub fn from_edges(n: usize, edges: &[(usize, usize)]) -> Self {
        let triplets: Vec<_> = edges.iter().map(|&(r, c)| (r, c, 1u64)).collect();
        Self::from_coo(n, &triplets)
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
        Self::from_coo(n, &triplets)
    }

    /// Build from named edge pairs. Returns the matrix and name→index mapping.
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
        Self::from_coo(n, &triplets)
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
        Self::from_coo(total, &triplets)
    }

    /// Randomly keep a fraction of edges, preserving symmetry.
    pub fn thin(&self, rng: &mut impl Rng, density: f64) -> Self {
        let mut triplets = Vec::new();
        for r in 0..self.n {
            if let Some(row) = self.mat.outer_view(r) {
                for (c, &v) in row.iter() {
                    if r <= c && rng.random_range(0.0f64..1.0) < density {
                        triplets.push((r, c, v.0));
                        if r != c {
                            let rev = self.get(c, r);
                            if rev > 0 {
                                triplets.push((c, r, rev));
                            }
                        }
                    }
                }
            }
        }
        Self::from_coo(self.n, &triplets)
    }

    /// Lookup value at (r, c).
    pub fn get(&self, r: usize, c: usize) -> u64 {
        self.mat.get(r, c).map_or(0, |v| v.0)
    }

    /// Number of non-zero entries.
    pub fn nnz(&self) -> usize {
        self.mat.nnz()
    }

    /// Matrix multiply via sprs SpGEMM (auto-threaded with multi_thread feature).
    pub fn matmul(&self, other: &Self) -> Self {
        assert_eq!(self.n, other.n);
        Self {
            n: self.n,
            mat: &self.mat * &other.mat,
        }
    }

    /// Element-wise addition.
    pub fn add(&self, other: &Self) -> Self {
        assert_eq!(self.n, other.n);
        Self {
            n: self.n,
            mat: &self.mat + &other.mat,
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
            if next.nnz() == current.nnz() && same_structure(&next.mat, &current.mat) {
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

    /// Connected components via union-find. O(nnz * α(n)).
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
            if let Some(row) = self.mat.outer_view(r) {
                for (c, _) in row.iter() {
                    union(&mut parent, &mut rank, r, c);
                }
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

/// Check if two CsMat have the same sparsity structure (same indptr + indices).
fn same_structure(a: &CsMat<Sat64>, b: &CsMat<Sat64>) -> bool {
    a.indptr().raw_storage() == b.indptr().raw_storage()
        && a.indices() == b.indices()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn test_identity_matmul() {
        let m = SprsMatrix::from_edges(3, &[(0, 1), (1, 2)]);
        let id = SprsMatrix::identity(3);
        let result = m.matmul(&id);
        assert_eq!(result.get(0, 1), 1);
        assert_eq!(result.get(1, 2), 1);
        assert_eq!(result.get(0, 2), 0);
        assert_eq!(result.nnz(), 2);
    }

    #[test]
    fn test_path_counting_triangle() {
        let m = SprsMatrix::from_edges(3, &[(0, 1), (1, 2), (2, 0)]);
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
        let m = SprsMatrix::from_edges(2, &[(0, 1), (0, 1)]);
        assert_eq!(m.get(0, 1), 2);
    }

    #[test]
    fn test_path_counting_diamond() {
        let m = SprsMatrix::from_edges(4, &[(0, 1), (0, 2), (1, 3), (2, 3)]);
        let m2 = m.matmul(&m);
        assert_eq!(m2.get(0, 3), 2);
    }

    #[test]
    fn test_reachability_chain() {
        let m = SprsMatrix::from_edges(4, &[(0, 1), (1, 2), (2, 3)]);
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
        let m = SprsMatrix::from_edges(n, &edges);
        let with_id = m.add(&SprsMatrix::identity(n));
        let (_stable, iters) = with_id.power_until_stable();
        assert!(iters <= 8, "took {iters} iterations for chain of {n}");
    }

    #[test]
    fn test_connected_components_two_triangles() {
        let m = SprsMatrix::from_edges_undirected(6, &[
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
        let m = SprsMatrix::new(5);
        let comp = m.connected_components();
        let unique: HashSet<usize> = comp.into_iter().collect();
        assert_eq!(unique.len(), 5);
    }

    #[test]
    fn test_from_adjacency_basic() {
        let edges = vec![("a", "b"), ("b", "c"), ("c", "a")];
        let (m, names) = SprsMatrix::from_adjacency(edges.into_iter());
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
        let (m, names) = SprsMatrix::from_adjacency(edges.into_iter());
        let x = names["x"];
        let y = names["y"];
        assert_eq!(m.get(x, y), 2);
        assert_eq!(m.get(y, x), 1);
    }

    #[test]
    fn test_from_adjacency_self_loop() {
        let edges = vec![("a", "a"), ("a", "b")];
        let (m, names) = SprsMatrix::from_adjacency(edges.into_iter());
        let a = names["a"];
        let b = names["b"];
        assert_eq!(m.get(a, a), 1);
        assert_eq!(m.get(a, b), 1);
        assert_eq!(m.n, 2);
    }

    #[test]
    fn test_from_adjacency_components() {
        let edges = vec![("a", "b"), ("b", "a"), ("c", "d"), ("d", "c")];
        let (m, names) = SprsMatrix::from_adjacency(edges.into_iter());
        let comp = m.connected_components();
        assert_eq!(comp[names["a"]], comp[names["b"]]);
        assert_eq!(comp[names["c"]], comp[names["d"]]);
        assert_ne!(comp[names["a"]], comp[names["c"]]);
    }

    #[test]
    fn test_lattice_1d_no_torus() {
        let m = SprsMatrix::lattice(&[5], false);
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
        let m = SprsMatrix::lattice(&[5], true);
        assert_eq!(m.get(0, 4), 1);
        assert_eq!(m.get(4, 0), 1);
        assert_eq!(m.nnz(), 10);
    }

    #[test]
    fn test_lattice_2d_corner() {
        let m = SprsMatrix::lattice(&[3, 3], false);
        assert_eq!(m.n, 9);
        assert_eq!(m.get(0, 1), 1);
        assert_eq!(m.get(0, 3), 1);
        assert_eq!(m.get(0, 4), 1);
        assert_eq!(m.nnz() % 2, 0);
    }

    #[test]
    fn test_lattice_2d_center() {
        let m = SprsMatrix::lattice(&[3, 3], false);
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
        let m = SprsMatrix::lattice(&[3, 3], true);
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
        let m = SprsMatrix::lattice(&[4, 4], false);
        let comp = m.connected_components();
        for i in 1..16 {
            assert_eq!(comp[0], comp[i]);
        }
    }

    #[test]
    fn test_lattice_3d() {
        let m = SprsMatrix::lattice(&[2, 2, 2], false);
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
        let m = SprsMatrix::lattice(&[4, 3], false);
        for r in 0..m.n {
            if let Some(row) = m.mat.outer_view(r) {
                for (c, v) in row.iter() {
                    assert_eq!(m.get(c, r), v.0, "asymmetry at ({r},{c})");
                }
            }
        }
    }

    #[test]
    fn test_connected_components_single_component() {
        let m = SprsMatrix::from_edges_undirected(4, &[(0, 1), (1, 2), (2, 3)]);
        let comp = m.connected_components();
        assert_eq!(comp[0], comp[1]);
        assert_eq!(comp[1], comp[2]);
        assert_eq!(comp[2], comp[3]);
    }

    #[test]
    fn bench_matmul_sprs_vs_csr() {
        use crate::graph::SparseCountMatrix;
        use crate::graph_csr::CsrMatrix;
        use rand::prelude::StdRng;
        use rand::SeedableRng;
        use std::time::Instant;

        let mut rng = StdRng::from_seed([42; 32]);
        let grid_sizes: &[usize] = &[5, 10, 20, 30];
        let edges_per_node: &[f64] = &[2.0, 3.0, 4.0, 8.0, 26.0];

        println!();
        println!("side,nodes,e_per_n,nnz,components,orig_btree_us,csr_us,csr_par_us,sprs_us,ratio_csr,ratio_par,ratio_sprs");
        for &s in grid_sizes {
            let full_btree = SparseCountMatrix::lattice(&[s, s, s], true);
            let full_csr = CsrMatrix::lattice(&[s, s, s], true);
            let full_sprs = SprsMatrix::lattice(&[s, s, s], true);
            let n = full_btree.n;
            let full_epn = full_btree.nnz() as f64 / n as f64;

            for &epn in edges_per_node {
                let density = epn / full_epn;

                let a_bt = if density >= 1.0 {
                    full_btree.clone()
                } else {
                    full_btree.thin(&mut rng, density)
                };
                let b_bt = a_bt.clone();

                // Build CSR from the same edges
                let a_csr = if density >= 1.0 {
                    full_csr.clone()
                } else {
                    let mut triplets: Vec<(usize, usize, u64)> = a_bt
                        .entries
                        .iter()
                        .map(|(&(r, c), &v)| (r, c, v))
                        .collect();
                    CsrMatrix::from_coo(n, &mut triplets)
                };
                let b_csr = a_csr.clone();

                // Build sprs from same edges
                let a_sprs = if density >= 1.0 {
                    full_sprs.clone()
                } else {
                    let triplets: Vec<(usize, usize, u64)> = a_bt
                        .entries
                        .iter()
                        .map(|(&(r, c), &v)| (r, c, v))
                        .collect();
                    SprsMatrix::from_coo(n, &triplets)
                };
                let b_sprs = a_sprs.clone();

                let t0 = Instant::now();
                let _r_bt = a_bt.matmul(&b_bt);
                let t_bt = t0.elapsed().as_micros();

                let t0 = Instant::now();
                let _r_csr = a_csr.matmul(&b_csr);
                let t_csr = t0.elapsed().as_micros();

                let t0 = Instant::now();
                let _r_par = a_csr.matmul_par(&b_csr);
                let t_par = t0.elapsed().as_micros();

                let t0 = Instant::now();
                let _r_sprs = a_sprs.matmul(&b_sprs);
                let t_sprs = t0.elapsed().as_micros();

                let components = a_csr.num_components();

                let ratio_csr = if t_bt > 0 {
                    t_csr as f64 / t_bt as f64
                } else {
                    0.0
                };
                let ratio_par = if t_bt > 0 {
                    t_par as f64 / t_bt as f64
                } else {
                    0.0
                };
                let ratio_sprs = if t_bt > 0 {
                    t_sprs as f64 / t_bt as f64
                } else {
                    0.0
                };

                println!(
                    "{s},{n},{epn:.0},{},{components},{t_bt},{t_csr},{t_par},{t_sprs},{ratio_csr:.6},{ratio_par:.6},{ratio_sprs:.6}",
                    a_bt.nnz()
                );
            }
        }
    }
}
