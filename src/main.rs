extern crate blas_src;  // Force linker to include BLAS implementation

use std::time::Instant;
use rand::prelude::StdRng;
use rand::{SeedableRng};
use pathmap::morphisms::Catamorphism;

mod traits;
mod dense;
mod sparse;

use traits::{Attention, FromRng};
use dense::{DenseTensorF, DenseTensorFBlas};
use sparse::{SparseTensorFBOB, SparseTensorFWeave};

/// Q[10,11,00,00]
/// path = [00001100, 00000100]
/// bhqd,bhkd->bhqk  ((to be) auto generated, do not touch)
static mut COUNT: usize = 0;

#[cfg(feature="viz")]
fn sparse_dimensionwise() {
    let mut t0 = SparseTensorFBOB::new(4);
    t0.set(&[3, 1, 6, 6], 0.5);
    t0.set(&[3, 2, 6, 6], 1.0);
    let mut t1 = SparseTensorFBOB::new(4);
    t1.set(&[3, 1, 6, 6], 0.2);
    t1.set(&[3, 1, 6, 7], 0.2);
    t1.set(&[5, 0, 0, 1], 10.0);
    t1.set(&[5, 0, 0, 2], 20.0);
    t1.set(&[5, 0, 0, 3], 30.0);
    let t1p2 = t0.add(&t1);
    let t1m2 = t0.mul(&t1);

    use pathmap::viz::{viz_maps, DrawConfig};
    let mut v = vec![];
    let dc = DrawConfig{ mode: pathmap::viz::VizMode::Ascii, ascii_path: false, hide_value_paths: false, minimize_values: false, logical: true, color: false };
    viz_maps(&&[t0, t1, t1p2, t1m2].into_iter().map(|t| t.vf().clone()).collect::<Vec<_>>()[..], &dc, &mut v).unwrap();
    println!("{}", str::from_utf8(&v[..]).unwrap());
}


static mut RCOUNT: usize = 0;

fn tipover_attention_bob() {
    let mut rng = StdRng::from_seed([0; 32]);
    // let (batch_size, sequence_length, n_heads, embedding_dim) = (2, 3, 4, 8); // toy
    // let (batch_size, sequence_length, n_heads, embedding_dim) = (32, 5, 12, 384); // shakespeare-char
    let (batch_size, sequence_length, n_heads, embedding_dim) = (8, 512, 25, 1600); // GPT-2 xl
    let mut rtq = DenseTensorF::new(vec![batch_size, sequence_length, n_heads, embedding_dim/n_heads]);
    let mut rtk = DenseTensorF::new(vec![batch_size, sequence_length, n_heads, embedding_dim/n_heads]);
    let mut rtr = DenseTensorF::new(vec![batch_size, sequence_length, n_heads, n_heads]);
    let mut c = 0f32;
    for b in 0..batch_size {
        for h in 0..sequence_length {
            for k in 0..n_heads {
                for d in 0..embedding_dim/n_heads {
                    c += 1f32;
                    rtq.set(&[b, h, k, d], c);
                    rtk.set(&[b, h, k, d], -c);
                }
            }
        }
    }
    let n_weights = rtq.m.len();
    let t0 = Instant::now();
    unsafe {
        std::env::set_var("OPENBLAS_NUM_THREADS", "1");  // single-threaded
        std::env::set_var("OMP_NUM_THREADS", "1");       // OpenMP fallback
    }

    let rc = rtq.attention(&rtk, &mut rtr);
    unsafe { RCOUNT += rc; }
    // reference_attention(&rtq, &rtk, &mut rtrb);
    println!("ref  {} µs ({n_weights} weights)", t0.elapsed().as_micros());
    println!("rcount {}", unsafe{ RCOUNT });

    let t0 = Instant::now();
    let rtq = rtq.to_strategy(dense::Blas);
    let rtk = rtk.to_strategy(dense::Blas);
    let mut rtrb = DenseTensorFBlas::new(vec![batch_size, sequence_length, n_heads, n_heads]);
    rtq.attention(&rtk, &mut rtrb);
    println!("blas {} µs ({n_weights} weights)", t0.elapsed().as_micros());
    println!("estimated memory usage: {} / {} / {} bytes",
        rtq.estimate_memory_usage(),
        rtk.estimate_memory_usage(),
        rtr.estimate_memory_usage());
    // verify correctness of BLAS against reference
    for b in 0..batch_size {
        for h in 0..sequence_length {
            for k in 0..n_heads {
                for q in 0..n_heads {
                    const EPS: f32 = 1e-4;
                    let refe = rtr.get(&[b, h, k, q]);
                    let blas = rtrb.get(&[b, h, k, q]);
                    let max_abs = refe.abs().max(blas.abs()).max(1.0);
                    let diff = ((refe - blas) / max_abs).abs();
                    assert!(diff < EPS,
                        "mismatch at [{b}, {h}, {k}, {q}] {refe} <-> {blas}");
                }
            }
        }
    }

    let mut rtr_ = SparseTensorFBOB::new(4);
    for b in 0..batch_size {
        for h in 0..sequence_length {
            for k in 0..n_heads {
                for q in 0..n_heads {
                    rtr_.set(&[b, h, k, q], rtr.get(&[b, h, k, q]));
                }
            }
        }
    }

    let mut rtq = SparseTensorFBOB::new(4);
    let mut rtk = SparseTensorFBOB::new(4);
    let mut rtr = SparseTensorFBOB::new(4);
    let mut c = 0f32;
    for b in 0..batch_size {
        for h in 0..sequence_length {
            for k in 0..n_heads {
                for d in 0..embedding_dim/n_heads {
                    c += 1f32;
                    rtq.set(&[b, h, k, d], c);
                    rtk.set(&[b, h, k, d], -c);
                }
            }
        }
    }
    let q_nz = rtq.m.val_count();
    let k_nz = rtk.m.val_count();
    // rtq.vF_mut().merkleize();
    // rtk.vF_mut().merkleize();
    let t0 = Instant::now();
    let count = rtq.attention(&rtk, &mut rtr);
    unsafe { COUNT += count; }
    // bob_attention(&mut rtq.m.read_zipper(), &mut rtk.m.read_zipper(), &mut rtr.m.write_zipper(), 0);
    println!("bob {} µs ({n_weights} weights, {q_nz} Q nz, {k_nz} K nz)", t0.elapsed().as_micros());
    println!(" count {}", unsafe{ COUNT });
    unsafe{ COUNT = 0 };

    assert_eq!(rtr.vf().clone().hash(), rtr_.vf().clone().hash());

    let shape = [batch_size, sequence_length, n_heads, embedding_dim/n_heads];
    // in completely unstructured sparsity, at 2% PathMap outperforms the naive dense implementation
    let rtq = SparseTensorFBOB::with_density(&mut rng, &shape, 0.02);
    let rtk = SparseTensorFBOB::with_density(&mut rng, &shape, 0.02);
    let mut rtr = SparseTensorFBOB::new(4);
    let q_nz = rtq.m.val_count();
    let k_nz = rtk.m.val_count();
    // rtq.vF_mut().merkleize();
    // rtk.vF_mut().merkleize();
    let t0 = Instant::now();
    let count = rtq.attention(&rtk, &mut rtr);
    unsafe { COUNT += count; }
    // bob_attention(&mut rtq.m.read_zipper(), &mut rtk.m.read_zipper(), &mut rtr.m.write_zipper(), 0);
    println!("bob {} µs ({n_weights} weights, {q_nz} Q nz, {k_nz} K nz)", t0.elapsed().as_micros());
    println!("count {}", unsafe{ COUNT });
    println!("estimated memory usage: {} / {} / {} bytes",
        rtq.estimate_memory_usage(),
        rtk.estimate_memory_usage(),
        rtr.estimate_memory_usage());
}

fn tipover_attention_weave(density: f64) {
    // Constants
    let mut rng = StdRng::from_seed([0; 32]);
    // let (batch_size, sequence_length, n_heads, embedding_dim) = (32, 512, 12, 384); // shakespeare-char
    let (batch_size, sequence_length, n_heads, embedding_dim) = (8, 512, 25, 1600); // GPT-2 xl
    let args_d = vec![batch_size, sequence_length, n_heads, embedding_dim/n_heads];
    let result_d = vec![batch_size, sequence_length, n_heads, n_heads];

    // Reference/dense
    let mut rtq = DenseTensorF::new(args_d.clone());
    let mut rtk = DenseTensorF::new(args_d.clone());
    let mut rtr = DenseTensorF::new(result_d.clone());
    let mut c = 0f32;
    DenseTensorF::foreach(&args_d[..], |ix| {
        c += 1f32;
        rtq.set(ix, c);
        rtk.set(ix, -c);
    });
    let n_weights = rtq.m.len();
    let t0 = Instant::now();
    let rc = rtq.attention(&rtk, &mut rtr);
    unsafe { RCOUNT += rc; }
    // reference_attention(&rtq, &rtk, &mut rtr);
    println!("ref {} µs ({n_weights} weights)", t0.elapsed().as_micros());
    println!("rcount {}", unsafe{ RCOUNT });
    println!("estimated memory usage: {} / {} / {} bytes",
        rtq.estimate_memory_usage(),
        rtk.estimate_memory_usage(),
        rtr.estimate_memory_usage());

    // Copy dense solution into sparse
    let mut rtr_ = SparseTensorFWeave::new(4);
    DenseTensorF::foreach(&result_d[..], |ix| rtr_.set(ix, rtr.get(ix)));

    // Weave/sparse
    let mut rtk = SparseTensorFWeave::new(4);
    let mut rtq = SparseTensorFWeave::new(4);
    let mut rtr = SparseTensorFWeave::new(4);
    let mut c = 0f32;
    DenseTensorF::foreach(&args_d[..], |ix| {
        c += 1f32;
        rtq.set(ix, c);
        rtk.set(ix, -c);
    });
    let q_nz = rtq.m.val_count();
    let k_nz = rtk.m.val_count();
    let t0 = Instant::now();
    let count = rtq.attention(&rtk, &mut rtr);
    unsafe { COUNT += count; }
    // _short_weave_attention(&mut rtq.m.read_zipper(), &mut rtk.m.read_zipper(), &mut rtr.m.write_zipper());
    println!("weave {} µs ({n_weights} weights, {q_nz} Q nz, {k_nz} K nz)", t0.elapsed().as_micros());
    println!("count {}", unsafe{ COUNT });
    assert_eq!(rtr.vf().clone().hash(), rtr_.vf().clone().hash());
    unsafe{ COUNT = 0 };

    // Weave/sparse
    let shape = &args_d[..];
    let rtq = SparseTensorFWeave::with_density(&mut rng, shape, density as f32);
    let rtk = SparseTensorFWeave::with_density(&mut rng, shape, density as f32);
    let mut rtr = SparseTensorFWeave::new(4);
    let q_nz = rtq.m.val_count();
    let k_nz = rtk.m.val_count();
    println!("weave density {}%: {q_nz} Q nz, {k_nz} K nz", density*100.0);
    // rtq.vF_mut().merkleize();
    // rtk.vF_mut().merkleize();
    let t0 = Instant::now();
    let count = rtq.attention(&rtk, &mut rtr);
    unsafe { COUNT += count; }
    // _short_weave_attention(&mut rtq.m.read_zipper(), &mut rtk.m.read_zipper(), &mut rtr.m.write_zipper());
    println!("weave {} µs ({n_weights} weights, {q_nz} Q nz, {k_nz} K nz)", t0.elapsed().as_micros());
    println!("count {}", unsafe{ COUNT });
    println!("estimated memory usage: {} / {} / {} bytes",
        rtq.estimate_memory_usage(),
        rtk.estimate_memory_usage(),
        rtr.estimate_memory_usage());
}



fn main() {
    // Observation 1: worst case is best with Bob
    // Observation 2: bob and weave don't differ that much with the unoptimized impl
    // Observation 3: real-world matrix distributions are often not homogenously sparse (and benefit from dense submatrix)

    // Note: what do attention matrices actually look like?
    // !!! Sparse Auto-Encoders (sparsification fine-tuning) !!!
    // via some regularizer

    // Scientific application where the matrix simply does not fit on the GPU?

    // show sharing between pointwise operations:
    // sparse_dimensionwise();

    tipover_attention_bob();
    tipover_attention_weave(0.02);
}
