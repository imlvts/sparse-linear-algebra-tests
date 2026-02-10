use crate::traits::{Attention, FromRng};

pub trait AttentionStrategy {}
#[derive(Default)]
pub struct Naive;
impl AttentionStrategy for Naive {}
#[derive(Default)]
pub struct Blas;
impl AttentionStrategy for Blas {}

pub type DenseTensorF = DenseTensorFRef<Naive>;
pub type DenseTensorFBlas = DenseTensorFRef<Blas>;

pub struct DenseTensorFRef<S: AttentionStrategy=Naive> {
    pub m: Vec<f32>,
    pub d: Vec<usize>,
    _strategy: S,
}

impl Attention for DenseTensorFRef<Naive> {
    /// bhqd,bhkd->bhqk  (auto generated, do not touch)
    /// Example shapes:
    ///   query:       [batch, heads, q_len, dim]
    ///   key:         [batch, heads, k_len, dim]
    ///   attn_scores: [batch, heads, q_len, k_len]
    #[allow(non_snake_case)]
    fn attention(&self, K: &Self, out: &mut Self) -> usize {
        let mut rcount = 0;
        assert_eq!(self.d[0], K.d[0]);
        for b in 0..self.d[0] {
            assert_eq!(self.d[1], K.d[1]);
            for h in 0..self.d[1] {
                for q in 0..self.d[2] {
                    for k in 0..K.d[2] {
                        let mut acc = 0f32;
                        assert_eq!(self.d[3], K.d[3]);
                        for d in 0..self.d[3] {
                            let qv = self.get(&[b, h, q, d]);
                            let kv = K.get(&[b, h, k, d]);
                            acc += qv*kv;
                            rcount += 1;
                            // unsafe { RCOUNT += 1; }
                        }
                        out.set(&[b, h, q, k], acc);
                    }
                }
            }
        }
        rcount
    }
}

/*
/// bhqd,bhkd->bhqk using BLAS
#[cfg(any())]
#[allow(non_snake_case)]
fn blas_attention(Q: &DenseTensorF, K: &DenseTensorF, out: &mut DenseTensorF) {
    let (batch, seq, n_heads, dim) = (Q.d[0], Q.d[1], Q.d[2], Q.d[3]);
    
    // For each batch and sequence position
    for b in 0..batch {
        for h in 0..seq {
            // Q[b,h,:,:] starts at linear index: (b*seq + h) * n_heads * dim
            // Shape: n_heads × dim (row-major)
            let q_offset = ((b * seq + h) * n_heads) * dim;
            let q_slice = &Q.m[q_offset..q_offset + n_heads * dim];
            
            // K[b,h,:,:] same layout
            let k_offset = ((b * seq + h) * n_heads) * dim;
            let k_slice = &K.m[k_offset..k_offset + n_heads * dim];
            
            // out[b,h,:,:] shape: n_heads × n_heads
            let out_offset = ((b * seq + h) * n_heads) * n_heads;
            let out_slice = &mut out.m[out_offset..out_offset + n_heads * n_heads];
            
            // We want: C = Q * K^T (row-major)
            // BLAS sees our row-major data as column-major transposed:
            //   Q_blas = Q^T (dim × n_heads)
            //   K_blas = K^T (dim × n_heads)
            // BLAS output interpreted as row-major = BLAS output^T
            // So we need BLAS to compute: (Q * K^T)^T = K * Q^T
            // 
            // op(A) = K = (K^T)^T → transA = 'T' on K_blas
            // op(B) = Q^T         → transB = 'N' on Q_blas
            unsafe {
                blas::sgemm(
                    b'T',           // transA: transpose K^T to get K
                    b'N',           // transB: keep Q^T as is
                    n_heads as i32, // m: rows of result
                    n_heads as i32, // n: cols of result  
                    dim as i32,     // k: inner dimension
                    1.0,            // alpha
                    k_slice, dim as i32,       // A = K^T (dim × n_heads), lda = dim
                    q_slice, dim as i32,       // B = Q^T (dim × n_heads), ldb = dim
                    0.0,            // beta
                    out_slice, n_heads as i32, // C (n_heads × n_heads), ldc = n_heads
                );
            }
        }
    }
}
*/

unsafe extern "C" {
    /// CBLAS batched strided GEMM
    /// Available in OpenBLAS 0.3+, Intel MKL, etc.
    fn cblas_sgemm_batch_strided(
        order: i32,      // CblasRowMajor=101, CblasColMajor=102
        trans_a: i32,    // CblasNoTrans=111, CblasTrans=112
        trans_b: i32,
        m: i32, n: i32, k: i32,
        alpha: f32,
        a: *const f32, lda: i32, stride_a: i64,
        b: *const f32, ldb: i32, stride_b: i64,
        beta: f32,
        c: *mut f32, ldc: i32, stride_c: i64,
        batch_count: i64,
    );
}


impl Attention for DenseTensorFRef<Blas> {
    /// bhqd,bhkd->bhqk using BLAS
    #[allow(non_snake_case)]
    fn attention(&self, K: &Self, out: &mut Self) -> usize {
        const CBLAS_ROW_MAJOR: i32 = 101;
        // const CBLAS_COL_MAJOR: i32 = 102;
        const CBLAS_NO_TRANS: i32 = 111;
        const CBLAS_TRANS: i32 = 112;

        let (batch, seq, n_heads, dim) = (self.d[0], self.d[1], self.d[2], self.d[3]);
        let batch_count = (batch * seq) as i64;
        
        // Each "batch" is an (n_heads × dim) matrix
        let stride_qk = (n_heads * dim) as i64;
        // Output is (n_heads × n_heads) per batch
        let stride_out = (n_heads * n_heads) as i64;
        
        unsafe {
            // Row-major: C = A * B^T becomes computing Q @ K^T for each batch
            // But CBLAS row-major with trans flags handles this directly
            cblas_sgemm_batch_strided(
                CBLAS_ROW_MAJOR,
                CBLAS_NO_TRANS,    // Q: n_heads × dim
                CBLAS_TRANS,       // K^T: dim × n_heads → result n_heads × n_heads
                n_heads as i32,    // m
                n_heads as i32,    // n
                dim as i32,        // k
                1.0,               // alpha
                self.m.as_ptr(), dim as i32, stride_qk,
                K.m.as_ptr(), dim as i32, stride_qk,
                0.0,               // beta
                out.m.as_mut_ptr(), n_heads as i32, stride_out,
                batch_count,
            );
        }
        (batch * seq * n_heads * n_heads) as usize
    }
}

impl<S: AttentionStrategy + Default> DenseTensorFRef<S> {
    pub fn new(d: Vec<usize>) -> Self {
        let n: usize = d.iter().product();
        Self { m: vec![0.0; n], d, _strategy: S::default() }
    }
    pub fn size(&self) -> usize {
        self.m.len()
    }
    pub fn estimate_memory_usage(&self) -> usize {
        self.m.len() * core::mem::size_of::<f32>()
    }
    pub fn to_strategy<S2: AttentionStrategy>(self, strategy: S2) -> DenseTensorFRef<S2> {
        DenseTensorFRef { m: self.m, d: self.d, _strategy: strategy }
    }

    pub fn linear_index(&self, ix: &[usize]) -> usize {
        assert_eq!(ix.len(), self.d.len(), "rank mismatch");

        let mut idx: usize = 0;
        let mut stride: usize = 1;

        for (&k, &dim) in ix.iter().rev().zip(self.d.iter().rev()) {
            assert!(k < dim, "index out of bounds");
            idx += k * stride;
            stride *= dim;
        }

        idx
    }

    pub fn get(&self, ix: &[usize]) -> f32 {
        let i = self.linear_index(ix);
        self.m[i]
    }

    pub fn set(&mut self, ix: &[usize], v: f32) {
        let i = self.linear_index(ix);
        self.m[i] = v;
    }

    pub fn foreach(d: &[usize], mut f: impl FnMut(&[usize])) {
        let n = d.len();
        if n == 0 || d.iter().any(|&x| x == 0) { return; }

        let mut ix = vec![0usize; n];
        loop {
            // Visit current index
            f(&ix[..]);
            // Increment like an odometer: carry from the last dimension backwards.
            let mut k = n;
            while k > 0 {
                k -= 1;
                ix[k] += 1;
                // no carry needed
                if ix[k] < d[k] { break; }
                ix[k] = 0;
                // overflowed the most-significant digit => done
                if k == 0 { return; }
            }
        }
    }

    pub fn add(&self, other: &Self) -> Self {
        assert_eq!(self.d, other.d, "shape mismatch");
        assert_eq!(self.m.len(), other.m.len(), "storage mismatch");

        let m = self
            .m
            .iter()
            .zip(other.m.iter())
            .map(|(&a, &b)| a + b)
            .collect();

        Self { m, d: self.d.clone(), _strategy: S::default() }
    }
    /// Relative difference between two matrices, defined as the max relative error over all entries
    fn rel_diff(&self, other: &Self) -> f32 {
        assert_eq!(self.d, other.d, "shape mismatch");
        assert_eq!(self.m.len(), other.m.len(), "storage mismatch");

        self.m
            .iter()
            .zip(other.m.iter())
            .map(|(&a, &b)| {
                let denom = a.abs().max(b.abs()).max(1e-8); // avoid division by zero
                ((a - b).abs()) / denom
            })
            .fold(0.0, |acc, x| acc.max(x))
    }
}

impl<S: AttentionStrategy + Default> FromRng for DenseTensorFRef<S> {
    fn with_density(rng: &mut impl rand::Rng, dimensions: &[usize], mut density: f32) -> Self {
        density = density.clamp(0.0, 1.0);
        let mut t = Self::new(dimensions.to_vec());
        let mut visit_idx = (0..t.m.len()).collect::<Vec<usize>>();
        let mut to_fill = (density * (t.m.len() as f32)) as usize;
        while to_fill > 0 {
            let ii = rng.random_range(0..visit_idx.len());
            let i = visit_idx[ii];
            visit_idx.swap_remove(ii);
            t.m[i] = rng.random();
            to_fill -= 1;
        }
        t
    }
}
