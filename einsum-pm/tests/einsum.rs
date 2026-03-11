use einsum_pm::einsum_fn;

/// Minimal dense tensor for testing — mirrors the DenseTensorFRef API.
struct Tensor {
    m: Vec<f32>,
    d: Vec<usize>,
}

impl Tensor {
    fn new(d: Vec<usize>) -> Self {
        let n: usize = d.iter().product();
        Self { m: vec![0.0; n], d }
    }

    fn linear_index(&self, ix: &[usize]) -> usize {
        let mut idx = 0usize;
        let mut stride = 1usize;
        for (&k, &dim) in ix.iter().rev().zip(self.d.iter().rev()) {
            idx += k * stride;
            stride *= dim;
        }
        idx
    }

    fn get(&self, ix: &[usize]) -> f32 {
        self.m[self.linear_index(ix)]
    }

    fn set(&mut self, ix: &[usize], v: f32) {
        let i = self.linear_index(ix);
        self.m[i] = v;
    }
}

// matmul: ab,bc->ac
einsum_fn!(matmul(a: Tensor/ab, b: Tensor/bc) -> Tensor/ac);

// batched matmul: abc,acd->abd
einsum_fn!(batched_matmul(a: Tensor/abc, b: Tensor/acd) -> Tensor/abd);

// attention: bhqd,bhkd->bhqk
einsum_fn!(attention(q: Tensor/bhqd, k: Tensor/bhkd) -> Tensor/bhqk);

// transpose: ab->ba
einsum_fn!(transpose(a: Tensor/ab) -> Tensor/ba);

// outer product: a,b->ab (no contraction)
einsum_fn!(outer(a: Tensor/a, b: Tensor/b) -> Tensor/ab);

// vector-matrix: a,ab->b
einsum_fn!(vecmat(v: Tensor/a, m: Tensor/ab) -> Tensor/b);

// dot product: a,a-> scalar
einsum_fn!(dot(a: Tensor/i, b: Tensor/i) -> f32);

// trace: aa-> scalar
einsum_fn!(trace(m: Tensor/aa) -> f32);

// frobenius norm squared: ab,ab-> scalar
einsum_fn!(frobenius2(a: Tensor/ab, b: Tensor/ab) -> f32);

#[test]
fn test_matmul() {
    let mut a = Tensor::new(vec![2, 3]);
    let mut b = Tensor::new(vec![3, 2]);

    // a = [[1,2,3],[4,5,6]]
    a.set(&[0, 0], 1.0); a.set(&[0, 1], 2.0); a.set(&[0, 2], 3.0);
    a.set(&[1, 0], 4.0); a.set(&[1, 1], 5.0); a.set(&[1, 2], 6.0);

    // b = [[7,8],[9,10],[11,12]]
    b.set(&[0, 0], 7.0); b.set(&[0, 1], 8.0);
    b.set(&[1, 0], 9.0); b.set(&[1, 1], 10.0);
    b.set(&[2, 0], 11.0); b.set(&[2, 1], 12.0);

    let c = matmul(&a, &b);
    assert_eq!(c.d, vec![2, 2]);
    assert_eq!(c.get(&[0, 0]), 58.0);
    assert_eq!(c.get(&[0, 1]), 64.0);
    assert_eq!(c.get(&[1, 0]), 139.0);
    assert_eq!(c.get(&[1, 1]), 154.0);
}

#[test]
fn test_transpose() {
    let mut a = Tensor::new(vec![2, 3]);
    a.set(&[0, 0], 1.0); a.set(&[0, 1], 2.0); a.set(&[0, 2], 3.0);
    a.set(&[1, 0], 4.0); a.set(&[1, 1], 5.0); a.set(&[1, 2], 6.0);

    let t = transpose(&a);
    assert_eq!(t.d, vec![3, 2]);
    assert_eq!(t.get(&[0, 0]), 1.0);
    assert_eq!(t.get(&[0, 1]), 4.0);
    assert_eq!(t.get(&[1, 0]), 2.0);
    assert_eq!(t.get(&[1, 1]), 5.0);
    assert_eq!(t.get(&[2, 0]), 3.0);
    assert_eq!(t.get(&[2, 1]), 6.0);
}

#[test]
fn test_outer_product() {
    let mut a = Tensor::new(vec![3]);
    a.set(&[0], 1.0); a.set(&[1], 2.0); a.set(&[2], 3.0);

    let mut b = Tensor::new(vec![2]);
    b.set(&[0], 4.0); b.set(&[1], 5.0);

    let c = outer(&a, &b);
    assert_eq!(c.d, vec![3, 2]);
    assert_eq!(c.get(&[0, 0]), 4.0);
    assert_eq!(c.get(&[0, 1]), 5.0);
    assert_eq!(c.get(&[1, 0]), 8.0);
    assert_eq!(c.get(&[1, 1]), 10.0);
    assert_eq!(c.get(&[2, 0]), 12.0);
    assert_eq!(c.get(&[2, 1]), 15.0);
}

#[test]
fn test_vecmat() {
    let mut v = Tensor::new(vec![2]);
    v.set(&[0], 1.0); v.set(&[1], 2.0);

    let mut m = Tensor::new(vec![2, 2]);
    m.set(&[0, 0], 3.0); m.set(&[0, 1], 4.0);
    m.set(&[1, 0], 5.0); m.set(&[1, 1], 6.0);

    let r = vecmat(&v, &m);
    assert_eq!(r.d, vec![2]);
    assert_eq!(r.get(&[0]), 13.0);
    assert_eq!(r.get(&[1]), 16.0);
}

#[test]
fn test_attention() {
    let mut q = Tensor::new(vec![1, 1, 2, 3]);
    let mut k = Tensor::new(vec![1, 1, 2, 3]);

    q.set(&[0, 0, 0, 0], 1.0);
    q.set(&[0, 0, 1, 1], 1.0);

    k.set(&[0, 0, 0, 0], 1.0);
    k.set(&[0, 0, 1, 1], 1.0);

    let out = attention(&q, &k);
    assert_eq!(out.d, vec![1, 1, 2, 2]);
    assert_eq!(out.get(&[0, 0, 0, 0]), 1.0);
    assert_eq!(out.get(&[0, 0, 0, 1]), 0.0);
    assert_eq!(out.get(&[0, 0, 1, 0]), 0.0);
    assert_eq!(out.get(&[0, 0, 1, 1]), 1.0);
}

#[test]
fn test_batched_attention() {
    // batch=2, heads=2, q_len=3, k_len=2, dim=4
    let (b, h, q_len, k_len, dim) = (2, 2, 3, 2, 4);
    let mut q = Tensor::new(vec![b, h, q_len, dim]);
    let mut k = Tensor::new(vec![b, h, k_len, dim]);

    // Fill with deterministic values: q[b,h,q,d] = (b+1)*(h+1)*(q+1) + d
    //                                  k[b,h,k,d] = (b+1)*(h+1)*(k+1) * (d+1)
    for bi in 0..b {
        for hi in 0..h {
            for qi in 0..q_len {
                for di in 0..dim {
                    let v = (bi + 1) as f32 * (hi + 1) as f32 * (qi + 1) as f32 + di as f32;
                    q.set(&[bi, hi, qi, di], v);
                }
            }
            for ki in 0..k_len {
                for di in 0..dim {
                    let v = (bi + 1) as f32 * (hi + 1) as f32 * (ki + 1) as f32 * (di + 1) as f32;
                    k.set(&[bi, hi, ki, di], v);
                }
            }
        }
    }

    let out = attention(&q, &k);
    assert_eq!(out.d, vec![b, h, q_len, k_len]);

    // Verify every element against naive computation
    for bi in 0..b {
        for hi in 0..h {
            for qi in 0..q_len {
                for ki in 0..k_len {
                    let mut expected = 0.0f32;
                    for di in 0..dim {
                        expected += q.get(&[bi, hi, qi, di]) * k.get(&[bi, hi, ki, di]);
                    }
                    let actual = out.get(&[bi, hi, qi, ki]);
                    assert!(
                        (actual - expected).abs() < 1e-3,
                        "mismatch at [{bi},{hi},{qi},{ki}]: got {actual}, expected {expected}"
                    );
                }
            }
        }
    }
}

#[test]
fn test_dot() {
    let mut a = Tensor::new(vec![4]);
    let mut b = Tensor::new(vec![4]);
    a.set(&[0], 1.0); a.set(&[1], 2.0); a.set(&[2], 3.0); a.set(&[3], 4.0);
    b.set(&[0], 5.0); b.set(&[1], 6.0); b.set(&[2], 7.0); b.set(&[3], 8.0);
    // 1*5 + 2*6 + 3*7 + 4*8 = 5+12+21+32 = 70
    assert_eq!(dot(&a, &b), 70.0);
}

#[test]
fn test_trace() {
    let mut m = Tensor::new(vec![3, 3]);
    m.set(&[0, 0], 1.0); m.set(&[0, 1], 2.0); m.set(&[0, 2], 3.0);
    m.set(&[1, 0], 4.0); m.set(&[1, 1], 5.0); m.set(&[1, 2], 6.0);
    m.set(&[2, 0], 7.0); m.set(&[2, 1], 8.0); m.set(&[2, 2], 9.0);
    // trace = 1+5+9 = 15
    assert_eq!(trace(&m), 15.0);
}

#[test]
fn test_frobenius2() {
    // frobenius2(A, A) = sum of squares of all elements
    let mut a = Tensor::new(vec![2, 2]);
    a.set(&[0, 0], 1.0); a.set(&[0, 1], 2.0);
    a.set(&[1, 0], 3.0); a.set(&[1, 1], 4.0);
    // 1+4+9+16 = 30
    assert_eq!(frobenius2(&a, &a), 30.0);
}
