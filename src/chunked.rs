use crate::traits::{Attention, SparseMatrix, FromRng};

// ── AVX2+FMA SIMD kernels ──────────────────────────────────────────────

#[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
use std::arch::x86_64::*;

/// 8×8 C += A @ B^T using AVX2+FMA.
/// Each row is one ymm register (8 f32).
#[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
#[inline(always)]
unsafe fn avx2_matmul_acc_8x8(a: *const f32, b: *const f32, c: *mut f32) {
    unsafe {
        // Transpose B into stack buffer
        let mut bt = [0.0f32; 64];
        for row in 0..8 {
            for col in 0..8 {
                *bt.get_unchecked_mut(col * 8 + row) = *b.add(row * 8 + col);
            }
        }

        // Load 8 result rows from C
        let mut res = [_mm256_setzero_ps(); 8];
        for i in 0..8 {
            res[i] = _mm256_loadu_ps(c.add(i * 8));
        }

        // Accumulate: for each k, load transposed-B row, broadcast A[i,k], FMA
        for k in 0..8 {
            let bt_row = _mm256_loadu_ps(bt.as_ptr().add(k * 8));
            for i in 0..8 {
                let a_ik = _mm256_set1_ps(*a.add(i * 8 + k));
                res[i] = _mm256_fmadd_ps(a_ik, bt_row, res[i]);
            }
        }

        // Store results back
        for i in 0..8 {
            _mm256_storeu_ps(c.add(i * 8), res[i]);
        }
    }
}

/// 16×16 C += A @ B^T using AVX2+FMA, tiled 4 rows at a time.
/// Each row = 2 ymm (lo 0..8, hi 8..16). 4 tiles of 4 rows each.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
#[inline(always)]
unsafe fn avx2_matmul_acc_16x16(a: *const f32, b: *const f32, c: *mut f32) {
    unsafe {
        // Transpose B into stack buffer
        let mut bt = [0.0f32; 256];
        for row in 0..16 {
            for col in 0..16 {
                *bt.get_unchecked_mut(col * 16 + row) = *b.add(row * 16 + col);
            }
        }

        // Process 4 output rows per tile
        for tile in 0..4 {
            let base_i = tile * 4;

            // Load 4 result rows (each split into lo + hi halves)
            let mut res_lo = [_mm256_setzero_ps(); 4];
            let mut res_hi = [_mm256_setzero_ps(); 4];
            for r in 0..4 {
                res_lo[r] = _mm256_loadu_ps(c.add((base_i + r) * 16));
                res_hi[r] = _mm256_loadu_ps(c.add((base_i + r) * 16 + 8));
            }

            // Accumulate over k dimension
            for k in 0..16 {
                let bt_lo = _mm256_loadu_ps(bt.as_ptr().add(k * 16));
                let bt_hi = _mm256_loadu_ps(bt.as_ptr().add(k * 16 + 8));
                for r in 0..4 {
                    let a_ik = _mm256_set1_ps(*a.add((base_i + r) * 16 + k));
                    res_lo[r] = _mm256_fmadd_ps(a_ik, bt_lo, res_lo[r]);
                    res_hi[r] = _mm256_fmadd_ps(a_ik, bt_hi, res_hi[r]);
                }
            }

            // Store results back
            for r in 0..4 {
                _mm256_storeu_ps(c.add((base_i + r) * 16), res_lo[r]);
                _mm256_storeu_ps(c.add((base_i + r) * 16 + 8), res_hi[r]);
            }
        }
    }
}

// ── Dispatch wrapper ────────────────────────────────────────────────────

/// Inline N×N multiply-accumulate: C += A @ B^T
/// Dispatches to AVX2+FMA SIMD kernels for N=8 and N=16 when available,
/// falls back to scalar otherwise.
#[inline(always)]
fn block_matmul_acc<const N: usize, const SQ: usize>(
    a: &[f32; SQ], b: &[f32; SQ], c: &mut [f32; SQ],
) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
    {
        match N {
            8  => { unsafe { avx2_matmul_acc_8x8(a.as_ptr(), b.as_ptr(), c.as_mut_ptr()) }; return }
            16 => { unsafe { avx2_matmul_acc_16x16(a.as_ptr(), b.as_ptr(), c.as_mut_ptr()) }; return }
            _  => {}
        }
    }
    block_matmul_acc_scalar::<N, SQ>(a, b, c);
}

/// Scalar fallback: N×N multiply-accumulate C += A @ B^T
/// Transposes B into a stack-local buffer so the inner vectorizable loop
/// streams contiguously over columns.
#[inline(always)]
fn block_matmul_acc_scalar<const N: usize, const SQ: usize>(
    a: &[f32; SQ], b: &[f32; SQ], c: &mut [f32; SQ],
) {
    let mut bt = [0.0f32; SQ];
    for row in 0..N {
        for col in 0..N {
            bt[col * N + row] = b[row * N + col];
        }
    }
    for i in 0..N {
        for k in 0..N {
            let a_ik = a[i * N + k];
            for j in 0..N {
                c[i * N + j] = a_ik.mul_add(bt[k * N + j], c[i * N + j]);
            }
        }
    }
}

fn ceil_div(a: usize, b: usize) -> usize {
    (a + b - 1) / b
}

/// Block-sparse tensor using N×N chunks (N=CHUNK, SQ=CHUNK*CHUNK).
///
/// The last two dimensions are tiled into CHUNK × CHUNK blocks.
/// Each block is either present (Some) or absent (None, treated as all-zero).
/// Outer dimensions (all dims except the last two) index into separate block grids.
pub struct ChunkedTensor<const CHUNK: usize, const SQ: usize> {
    d: Vec<usize>,
    n_block_rows: usize,
    n_block_cols: usize,
    outer_size: usize,
    blocks: Vec<Option<Box<[f32; SQ]>>>,
}

pub type Chunked8 = ChunkedTensor<8, 64>;
pub type Chunked16 = ChunkedTensor<16, 256>;

impl<const CHUNK: usize, const SQ: usize> ChunkedTensor<CHUNK, SQ> {
    pub fn with_shape(d: Vec<usize>) -> Self {
        assert!(d.len() >= 2, "need at least 2 dimensions");
        let n = d.len();
        let n_block_rows = ceil_div(d[n - 2], CHUNK);
        let n_block_cols = ceil_div(d[n - 1], CHUNK);
        let outer_size: usize = if n > 2 { d[..n - 2].iter().product() } else { 1 };
        let total_blocks = outer_size * n_block_rows * n_block_cols;
        let mut blocks = Vec::with_capacity(total_blocks);
        blocks.resize_with(total_blocks, || None);
        Self { d, n_block_rows, n_block_cols, outer_size, blocks }
    }

    pub fn estimate_memory_usage(&self) -> usize {
        let non_empty = self.blocks.iter().filter(|b| b.is_some()).count();
        self.blocks.len() * std::mem::size_of::<Option<Box<[f32; SQ]>>>()
            + non_empty * std::mem::size_of::<[f32; SQ]>()
    }

    fn outer_linear_index(&self, ix: &[usize]) -> usize {
        let n = self.d.len();
        if n <= 2 { return 0; }
        let mut idx = 0;
        let mut stride = 1;
        for i in (0..n - 2).rev() {
            idx += ix[i] * stride;
            stride *= self.d[i];
        }
        idx
    }

    fn block_coords(&self, ix: &[usize]) -> (usize, usize) {
        let n = self.d.len();
        let outer = self.outer_linear_index(ix);
        let block_row = ix[n - 2] / CHUNK;
        let block_col = ix[n - 1] / CHUNK;
        let block_idx = outer * (self.n_block_rows * self.n_block_cols)
            + block_row * self.n_block_cols + block_col;
        let local_row = ix[n - 2] % CHUNK;
        let local_col = ix[n - 1] % CHUNK;
        let local_idx = local_row * CHUNK + local_col;
        (block_idx, local_idx)
    }
}

impl<const CHUNK: usize, const SQ: usize> SparseMatrix for ChunkedTensor<CHUNK, SQ> {
    type Value = f32;

    fn new(dimensions: usize) -> Self {
        Self {
            d: vec![0; dimensions],
            n_block_rows: 0,
            n_block_cols: 0,
            outer_size: 0,
            blocks: vec![],
        }
    }

    fn ndim(&self) -> usize { self.d.len() }

    fn get(&self, ix: &[usize]) -> Option<f32> {
        if self.blocks.is_empty() { return None; }
        let (block_idx, local_idx) = self.block_coords(ix);
        self.blocks[block_idx].as_ref().and_then(|b| {
            let v = b[local_idx];
            if v != 0.0 { Some(v) } else { None }
        })
    }

    fn set(&mut self, ix: &[usize], v: f32) {
        let (block_idx, local_idx) = self.block_coords(ix);
        let block = self.blocks[block_idx].get_or_insert_with(|| Box::new([0.0; SQ]));
        block[local_idx] = v;
    }

    fn remove(&mut self, ix: &[usize]) -> Option<f32> {
        if self.blocks.is_empty() { return None; }
        let (block_idx, local_idx) = self.block_coords(ix);
        if let Some(block) = &mut self.blocks[block_idx] {
            let old = block[local_idx];
            if old == 0.0 { return None; }
            block[local_idx] = 0.0;
            if block.iter().all(|&v| v == 0.0) {
                self.blocks[block_idx] = None;
            }
            Some(old)
        } else {
            None
        }
    }

    fn nnz(&self) -> usize {
        self.blocks.iter()
            .filter_map(|b| b.as_ref())
            .map(|b| b.iter().filter(|&&v| v != 0.0).count())
            .sum()
    }

    fn add(&self, other: &Self) -> Self {
        assert_eq!(self.d, other.d);
        let blocks = self.blocks.iter().zip(other.blocks.iter()).map(|(a, b)| {
            match (a, b) {
                (Some(a), Some(b)) => {
                    let mut c = Box::new([0.0f32; SQ]);
                    for i in 0..SQ { c[i] = a[i] + b[i]; }
                    Some(c)
                }
                (Some(x), None) | (None, Some(x)) => Some(x.clone()),
                (None, None) => None,
            }
        }).collect();
        Self {
            d: self.d.clone(),
            n_block_rows: self.n_block_rows,
            n_block_cols: self.n_block_cols,
            outer_size: self.outer_size,
            blocks,
        }
    }

    fn mul(&self, other: &Self) -> Self {
        assert_eq!(self.d, other.d);
        let blocks = self.blocks.iter().zip(other.blocks.iter()).map(|(a, b)| {
            match (a, b) {
                (Some(a), Some(b)) => {
                    let mut c = Box::new([0.0f32; SQ]);
                    for i in 0..SQ { c[i] = a[i] * b[i]; }
                    if c.iter().any(|&v| v != 0.0) { Some(c) } else { None }
                }
                _ => None,
            }
        }).collect();
        Self {
            d: self.d.clone(),
            n_block_rows: self.n_block_rows,
            n_block_cols: self.n_block_cols,
            outer_size: self.outer_size,
            blocks,
        }
    }

    fn rel_diff(&self, other: &Self) -> f32 {
        assert_eq!(self.d, other.d);
        let mut max_diff = 0.0f32;
        for (a, b) in self.blocks.iter().zip(other.blocks.iter()) {
            match (a, b) {
                (Some(a), Some(b)) => {
                    for i in 0..SQ {
                        let denom = a[i].abs().max(b[i].abs()).max(1e-8);
                        max_diff = max_diff.max((a[i] - b[i]).abs() / denom);
                    }
                }
                (Some(x), None) | (None, Some(x)) => {
                    if x.iter().any(|&v| v != 0.0) { max_diff = 1.0; }
                }
                (None, None) => {}
            }
        }
        max_diff
    }
}

impl<const CHUNK: usize, const SQ: usize> Attention for ChunkedTensor<CHUNK, SQ> {
    /// bhqd,bhkd->bhqk using block-sparse inline N×N matmul.
    #[allow(non_snake_case)]
    fn attention(&self, K: &Self, out: &mut Self) -> usize {
        let q_brows = self.n_block_rows;
        let d_blocks = self.n_block_cols;
        let k_brows = K.n_block_rows;

        assert_eq!(d_blocks, K.n_block_cols, "inner dimension (d) block count mismatch");
        assert_eq!(out.n_block_rows, q_brows, "output q-block count mismatch");
        assert_eq!(out.n_block_cols, k_brows, "output k-block count mismatch");
        assert_eq!(self.outer_size, K.outer_size, "outer dimension mismatch");
        assert_eq!(self.outer_size, out.outer_size, "outer dimension mismatch");

        let q_per_outer = q_brows * d_blocks;
        let k_per_outer = k_brows * K.n_block_cols;
        let o_per_outer = out.n_block_rows * out.n_block_cols;

        let mut count = 0usize;

        for outer in 0..self.outer_size {
            let q_base = outer * q_per_outer;
            let k_base = outer * k_per_outer;
            let o_base = outer * o_per_outer;

            for qi in 0..q_brows {
                for ki in 0..k_brows {
                    let mut acc = [0.0f32; SQ];
                    let mut hits = 0usize;

                    for di in 0..d_blocks {
                        let q_idx = q_base + qi * d_blocks + di;
                        let k_idx = k_base + ki * K.n_block_cols + di;

                        if let (Some(q_blk), Some(k_blk)) =
                            (&self.blocks[q_idx], &K.blocks[k_idx])
                        {
                            block_matmul_acc::<CHUNK, SQ>(q_blk, k_blk, &mut acc);
                            hits += 1;
                        }
                    }

                    if hits > 0 {
                        let o_idx = o_base + qi * out.n_block_cols + ki;
                        out.blocks[o_idx] = Some(Box::new(acc));
                        count += hits * CHUNK * CHUNK * CHUNK;
                    }
                }
            }
        }

        count
    }
}

impl<const CHUNK: usize, const SQ: usize> FromRng for ChunkedTensor<CHUNK, SQ> {
    fn with_density(rng: &mut impl rand::Rng, dimensions: &[usize], mut density: f32) -> Self {
        density = density.clamp(0.0, 1.0);
        let mut t = Self::with_shape(dimensions.to_vec());
        let n: usize = dimensions.iter().product();
        let mut to_fill = (density * n as f32) as usize;
        let ndim = dimensions.len();
        let mut dim = vec![0usize; ndim];

        if density < 0.5 {
            while to_fill > 0 {
                let i = rng.random_range(0..n);
                let mut rem = i;
                for (d, &s) in dim.iter_mut().zip(dimensions.iter()) {
                    *d = rem % s;
                    rem /= s;
                }
                let (block_idx, local_idx) = t.block_coords(&dim);
                if let Some(blk) = &t.blocks[block_idx] {
                    if blk[local_idx] != 0.0 { continue; }
                }
                let block = t.blocks[block_idx].get_or_insert_with(|| Box::new([0.0; SQ]));
                block[local_idx] = rng.random();
                to_fill -= 1;
            }
        } else {
            let mut visit_idx = (0..n).collect::<Vec<usize>>();
            while to_fill > 0 {
                let ii = rng.random_range(0..visit_idx.len());
                let i = visit_idx.swap_remove(ii);
                let mut rem = i;
                for (d, &s) in dim.iter_mut().zip(dimensions.iter()) {
                    *d = rem % s;
                    rem /= s;
                }
                let (block_idx, local_idx) = t.block_coords(&dim);
                let block = t.blocks[block_idx].get_or_insert_with(|| Box::new([0.0; SQ]));
                block[local_idx] = rng.random();
                to_fill -= 1;
            }
        }
        t
    }
}
