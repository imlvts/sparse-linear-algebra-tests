use einsum_dyn::{NDIndex, einsum_binary, einsum_binary_scalar, einsum_unary, einsum_unary_scalar};
use einsum_pm::einsum_fn;
use std::time::Instant;

/// Dense tensor shared by both implementations.
struct T {
    m: Vec<f64>,
    d: Vec<usize>,
}

impl T {
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

    fn get(&self, ix: &[usize]) -> f64 {
        self.m[self.linear_index(ix)]
    }

    fn set(&mut self, ix: &[usize], v: f64) {
        let i = self.linear_index(ix);
        self.m[i] = v;
    }
}

impl NDIndex<f64> for T {
    fn ndim(&self) -> usize { self.d.len() }
    fn dim(&self, axis: usize) -> usize { self.d[axis] }
    fn get(&self, ix: &[usize]) -> f64 { self.m[self.linear_index(ix)] }
    fn set(&mut self, ix: &[usize], v: f64) {
        let i = self.linear_index(ix);
        self.m[i] = v;
    }
}

// Proc-macro generated functions
einsum_fn!(pm_matmul(a: T/ab, b: T/bc) -> T/ac);
einsum_fn!(pm_attention(q: T/bhqd, k: T/bhkd) -> T/bhqk);
einsum_fn!(pm_transpose(a: T/ab) -> T/ba);
einsum_fn!(pm_dot(a: T/i, b: T/i) -> f64);
einsum_fn!(pm_trace(m: T/aa) -> f64);

fn fill_rand(t: &mut T, seed: u64) {
    let mut x = seed;
    for v in t.m.iter_mut() {
        // Simple xorshift
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *v = (x as f64) / (u64::MAX as f64);
    }
}

fn bench<F: FnMut()>(name: &str, iters: u32, mut f: F) -> f64 {
    // Warmup
    for _ in 0..3 {
        f();
    }
    let start = Instant::now();
    for _ in 0..iters {
        f();
    }
    let elapsed = start.elapsed();
    let per_iter_ns = elapsed.as_nanos() as f64 / iters as f64;
    let per_iter_us = per_iter_ns / 1000.0;
    println!("  {name:30} {per_iter_us:10.2} µs/iter  ({iters} iters)");
    per_iter_us
}

fn main() {
    println!("=== einsum benchmark: proc-macro vs dynamic ===\n");

    // --- Matmul NxN ---
    for &n in &[16, 64, 128, 256] {
        println!("--- matmul {n}×{n} ---");
        let mut a = T::new(vec![n, n]);
        let mut b = T::new(vec![n, n]);
        fill_rand(&mut a, 42);
        fill_rand(&mut b, 123);

        let iters = if n <= 64 { 1000 } else if n <= 128 { 100 } else { 10 };

        let pm = bench("proc-macro", iters, || {
            let _ = std::hint::black_box(pm_matmul(&a, &b));
        });
        let dyn_t = bench("dynamic", iters, || {
            let mut c = T::new(vec![n, n]);
            einsum_binary("ab,bc->ac", &a, &b, &mut c).unwrap();
            std::hint::black_box(&c);
        });
        println!("  ratio (dyn/pm): {:.2}×\n", dyn_t / pm);
    }

    // --- Attention bhqd,bhkd->bhqk ---
    {
        let (b, h, q, k, d) = (2, 8, 32, 32, 64);
        println!("--- attention b={b} h={h} q={q} k={k} d={d} ---");
        let mut qm = T::new(vec![b, h, q, d]);
        let mut km = T::new(vec![b, h, k, d]);
        fill_rand(&mut qm, 42);
        fill_rand(&mut km, 99);

        let iters = 20;
        let pm = bench("proc-macro", iters, || {
            let _ = std::hint::black_box(pm_attention(&qm, &km));
        });
        let dyn_t = bench("dynamic", iters, || {
            let mut out = T::new(vec![b, h, q, k]);
            einsum_binary("bhqd,bhkd->bhqk", &qm, &km, &mut out).unwrap();
            std::hint::black_box(&out);
        });
        println!("  ratio (dyn/pm): {:.2}×\n", dyn_t / pm);
    }

    // --- Transpose ---
    {
        let n = 256;
        println!("--- transpose {n}×{n} ---");
        let mut a = T::new(vec![n, n]);
        fill_rand(&mut a, 42);

        let iters = 200;
        let pm = bench("proc-macro", iters, || {
            let _ = std::hint::black_box(pm_transpose(&a));
        });
        let dyn_t = bench("dynamic", iters, || {
            let mut out = T::new(vec![n, n]);
            einsum_unary("ab->ba", &a, &mut out).unwrap();
            std::hint::black_box(&out);
        });
        println!("  ratio (dyn/pm): {:.2}×\n", dyn_t / pm);
    }

    // --- Dot product ---
    {
        let n = 10000;
        println!("--- dot product n={n} ---");
        let mut a = T::new(vec![n]);
        let mut b = T::new(vec![n]);
        fill_rand(&mut a, 42);
        fill_rand(&mut b, 99);

        let iters = 2000;
        let pm = bench("proc-macro", iters, || {
            let _ = std::hint::black_box(pm_dot(&a, &b));
        });
        let dyn_t = bench("dynamic", iters, || {
            let r: f64 = einsum_binary_scalar("i,i->", &a, &b).unwrap();
            std::hint::black_box(r);
        });
        println!("  ratio (dyn/pm): {:.2}×\n", dyn_t / pm);
    }

    // --- Trace ---
    {
        let n = 500;
        println!("--- trace {n}×{n} ---");
        let mut m = T::new(vec![n, n]);
        fill_rand(&mut m, 42);

        let iters = 2000;
        let pm = bench("proc-macro", iters, || {
            let _ = std::hint::black_box(pm_trace(&m));
        });
        let dyn_t = bench("dynamic", iters, || {
            let r: f64 = einsum_unary_scalar("aa->", &m).unwrap();
            std::hint::black_box(r);
        });
        println!("  ratio (dyn/pm): {:.2}×\n", dyn_t / pm);
    }
}
