//! Dense matmul via einsum_vm_oneshot — apples-to-apples reference for
//! the new linalg crate's VM. Same shapes, same dtype (f32) as
//! linalg/benches/perf.rs section 1.

use einsum_dyn::{NDIndex, sparse::einsum_vm_oneshot};
use std::time::Instant;

struct T {
    m: Vec<f32>,
    d: Vec<usize>,
}

impl T {
    fn new(d: Vec<usize>) -> Self {
        let n: usize = d.iter().product::<usize>().max(1);
        Self { m: vec![0.0; n], d }
    }
    fn linear_index(&self, ix: &[usize]) -> usize {
        let mut idx = 0;
        let mut stride = 1;
        for (&k, &dim) in ix.iter().rev().zip(self.d.iter().rev()) {
            idx += k * stride;
            stride *= dim;
        }
        idx
    }
}

impl NDIndex<f32> for T {
    fn ndim(&self) -> usize { self.d.len() }
    fn dim(&self, axis: usize) -> usize { self.d[axis] }
    fn get(&self, ix: &[usize]) -> f32 { self.m[self.linear_index(ix)] }
    fn set(&mut self, ix: &[usize], v: f32) {
        let i = self.linear_index(ix);
        self.m[i] = v;
    }
}

fn xorshift(seed: u64) -> impl FnMut() -> u64 {
    let mut x = seed.max(1);
    move || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    }
}

fn fill_rand(t: &mut T, seed: u64) {
    let mut rng = xorshift(seed);
    for v in t.m.iter_mut() {
        *v = (rng() as f64 / u64::MAX as f64) as f32;
    }
}

fn bench<F: FnMut()>(name: &str, iters: u32, mut f: F) -> f64 {
    for _ in 0..3 { f(); }
    let start = Instant::now();
    for _ in 0..iters { f(); }
    let per_iter_us = start.elapsed().as_nanos() as f64 / iters as f64 / 1000.0;
    println!("  {name:42} {per_iter_us:12.2} µs/iter  ({iters} iters)");
    per_iter_us
}

fn main() {
    println!("=== einsum-dyn VM bench (f32, dense matmul) ===");
    for &n in &[16usize, 64, 128, 256] {
        println!("\n--- {n}×{n} ---");
        let mut a = T::new(vec![n, n]);
        let mut b = T::new(vec![n, n]);
        fill_rand(&mut a, 42);
        fill_rand(&mut b, 99);
        let iters: u32 = match n {
            16 => 5000, 64 => 500, 128 => 100, 256 => 20, _ => 5
        };
        bench("einsum_vm_oneshot", iters, || {
            let mut c = T::new(vec![n, n]);
            einsum_vm_oneshot("ab,bc->ac", &[&a, &b], &mut [&mut c]).unwrap();
            std::hint::black_box(&c);
        });
    }
}
