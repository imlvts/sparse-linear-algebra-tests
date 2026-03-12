use einsum_dyn::NDIndex;
use crate::graph_csr::{CsrMatrix, NodeId, Val};

/// CSR matrix builder that accepts `set` calls in row-major (lexicographic) order.
///
/// Implements `NDIndex` so it can be used directly as an einsum output.
/// Einsum guarantees that `set` is called in lexicographic order of the output
/// indices, which is exactly row-major for a 2D matrix — so this builds valid
/// CSR arrays incrementally with no post-processing.
///
/// Zero values are skipped (not stored), producing a sparse output automatically.
pub struct CsrBuilder {
    n: NodeId,
    row_ptr: Vec<usize>,
    col_idx: Vec<NodeId>,
    values: Vec<Val>,
    /// The row currently being written, or `n` if no `set` has been called yet.
    last_row: usize,
}

impl CsrBuilder {
    /// Create a builder for an `n × n` matrix.
    pub fn new(n: NodeId) -> Self {
        Self {
            n,
            row_ptr: Vec::with_capacity(n as usize + 1),
            col_idx: Vec::new(),
            values: Vec::new(),
            last_row: n as usize, // sentinel: no row started yet
        }
    }

    /// Finish building and return the completed `CsrMatrix`.
    ///
    /// Closes any remaining open rows so that `row_ptr` has exactly `n + 1` entries.
    pub fn finish(mut self) -> CsrMatrix {
        let n = self.n as usize;
        let nnz = self.col_idx.len();
        // Pad row_ptr for any trailing empty rows + final sentinel
        while self.row_ptr.len() <= n {
            self.row_ptr.push(nnz);
        }
        CsrMatrix {
            n: self.n,
            row_ptr: self.row_ptr,
            col_idx: self.col_idx,
            values: self.values,
            perm: None,
        }
    }
}

impl NDIndex<Val> for CsrBuilder {
    fn ndim(&self) -> usize { 2 }

    fn dim(&self, _axis: usize) -> usize { self.n as usize }

    fn get(&self, _ix: &[usize]) -> Val {
        panic!("CsrBuilder is write-only; use CsrMatrix for reads")
    }

    fn set(&mut self, ix: &[usize], v: Val) {
        let r = ix[0];
        let c = ix[1] as NodeId;

        // If we've moved to a new row (or this is the first call),
        // emit row_ptr entries for all rows up to and including r.
        if r != self.last_row {
            let nnz = self.col_idx.len();
            // On first call, fill row_ptr[0..=r]. On subsequent calls,
            // fill row_ptr for any skipped empty rows between last_row+1 and r.
            let start = self.row_ptr.len();
            let end = r + 1; // we need row_ptr[0..=r] to exist
            for _ in start..end {
                self.row_ptr.push(nnz);
            }
            self.last_row = r;
        }

        if v != 0 {
            self.col_idx.push(c);
            self.values.push(v);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use einsum_dyn::{einsum_binary, einsum_unary};

    /// Simple dense 2D matrix for test inputs.
    struct Dense {
        n: usize,
        data: Vec<Val>,
    }

    impl Dense {
        fn new(n: usize) -> Self {
            Self { n, data: vec![0; n * n] }
        }
        fn from_data(n: usize, data: Vec<Val>) -> Self {
            assert_eq!(data.len(), n * n);
            Self { n, data }
        }
    }

    impl NDIndex<Val> for Dense {
        fn ndim(&self) -> usize { 2 }
        fn dim(&self, _axis: usize) -> usize { self.n }
        fn get(&self, ix: &[usize]) -> Val { self.data[ix[0] * self.n + ix[1]] }
        fn set(&mut self, ix: &[usize], v: Val) { self.data[ix[0] * self.n + ix[1]] = v; }
    }

    #[test]
    fn test_matmul_identity() {
        let id = Dense::from_data(3, vec![
            1, 0, 0,
            0, 1, 0,
            0, 0, 1,
        ]);
        let a = Dense::from_data(3, vec![
            1, 2, 0,
            0, 0, 3,
            4, 0, 5,
        ]);
        let mut out = CsrBuilder::new(3);
        einsum_binary("ab,bc->ac", &a, &id, &mut out).unwrap();
        let csr = out.finish();

        assert_eq!(csr.get(0, 0), 1);
        assert_eq!(csr.get(0, 1), 2);
        assert_eq!(csr.get(1, 2), 3);
        assert_eq!(csr.get(2, 0), 4);
        assert_eq!(csr.get(2, 2), 5);
        assert_eq!(csr.get(0, 2), 0);
        assert_eq!(csr.nnz(), 5);
    }

    #[test]
    fn test_matmul_sparse_result() {
        // A × B where the result has zeros
        let a = Dense::from_data(2, vec![
            1, 0,
            0, 1,
        ]);
        let b = Dense::from_data(2, vec![
            0, 3,
            7, 0,
        ]);
        let mut out = CsrBuilder::new(2);
        einsum_binary("ab,bc->ac", &a, &b, &mut out).unwrap();
        let csr = out.finish();

        assert_eq!(csr.get(0, 0), 0);
        assert_eq!(csr.get(0, 1), 3);
        assert_eq!(csr.get(1, 0), 7);
        assert_eq!(csr.get(1, 1), 0);
        assert_eq!(csr.nnz(), 2);
    }

    #[test]
    fn test_matmul_dense_result() {
        let a = Dense::from_data(2, vec![1, 2, 3, 4]);
        let b = Dense::from_data(2, vec![5, 6, 7, 8]);
        let mut out = CsrBuilder::new(2);
        einsum_binary("ab,bc->ac", &a, &b, &mut out).unwrap();
        let csr = out.finish();

        assert_eq!(csr.get(0, 0), 19);
        assert_eq!(csr.get(0, 1), 22);
        assert_eq!(csr.get(1, 0), 43);
        assert_eq!(csr.get(1, 1), 50);
        assert_eq!(csr.nnz(), 4);
    }

    #[test]
    fn test_transpose() {
        let a = Dense::from_data(3, vec![
            1, 0, 2,
            0, 0, 0,
            3, 0, 4,
        ]);
        let mut out = CsrBuilder::new(3);
        einsum_unary("ab->ba", &a, &mut out).unwrap();
        let csr = out.finish();

        assert_eq!(csr.get(0, 0), 1);
        assert_eq!(csr.get(0, 2), 3);
        assert_eq!(csr.get(2, 0), 2);
        assert_eq!(csr.get(2, 2), 4);
        assert_eq!(csr.nnz(), 4);
    }

    #[test]
    fn test_empty_rows() {
        // Result has an entirely empty middle row
        let a = Dense::from_data(3, vec![
            1, 0, 0,
            0, 0, 0,
            0, 0, 1,
        ]);
        let mut out = CsrBuilder::new(3);
        einsum_binary("ab,bc->ac", &a, &a, &mut out).unwrap();
        let csr = out.finish();

        assert_eq!(csr.get(0, 0), 1);
        assert_eq!(csr.get(1, 0), 0);
        assert_eq!(csr.get(1, 1), 0);
        assert_eq!(csr.get(1, 2), 0);
        assert_eq!(csr.get(2, 2), 1);
        assert_eq!(csr.nnz(), 2);
        // row_ptr should have n+1 = 4 entries
        assert_eq!(csr.row_ptr.len(), 4);
    }

    #[test]
    fn test_all_zero_result() {
        let a = Dense::from_data(2, vec![1, 0, 0, 0]);
        let b = Dense::from_data(2, vec![0, 0, 0, 1]);
        let mut out = CsrBuilder::new(2);
        einsum_binary("ab,bc->ac", &a, &b, &mut out).unwrap();
        let csr = out.finish();

        assert_eq!(csr.nnz(), 0);
        assert_eq!(csr.row_ptr.len(), 3);
    }

    #[test]
    fn test_agrees_with_csr_matmul() {
        // Build from edges, multiply via CsrMatrix, compare with einsum + CsrBuilder
        let edges: &[(usize, usize)] = &[
            (0, 1), (0, 2), (1, 3), (2, 3), (3, 0), (2, 1),
        ];
        let n = 4u32;

        let mut triplets: Vec<(u32, u32, Val)> = edges.iter()
            .map(|&(r, c)| (r as u32, c as u32, 1 as Val))
            .collect();
        let csr_a = CsrMatrix::from_coo(n, &mut triplets);
        let csr_expected = csr_a.matmul(&csr_a);

        // Build dense input for einsum
        let mut dense = Dense::new(n as usize);
        for &(r, c) in edges {
            dense.data[r * n as usize + c] = 1;
        }

        let mut out = CsrBuilder::new(n);
        einsum_binary("ab,bc->ac", &dense, &dense, &mut out).unwrap();
        let csr_built = out.finish();

        for r in 0..n {
            for c in 0..n {
                assert_eq!(
                    csr_built.get(r, c), csr_expected.get(r, c),
                    "mismatch at ({r},{c})"
                );
            }
        }
        assert_eq!(csr_built.nnz(), csr_expected.nnz());
    }
}
