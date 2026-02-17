extern crate blas_src;  // Force linker to include BLAS implementation

use std::time::Instant;
use rand::prelude::StdRng;
use rand::{SeedableRng};
use pathmap::morphisms::Catamorphism;

mod traits;
mod dense;
mod sparse;
mod chunked;

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
// (batch_size, sequence_length, n_heads, embedding_dim)
const GPT_CONFIGS: [(usize, usize, usize, usize); 5] = [
    (32, 512, 12, 384),    // shakespeare-char
    (8, 1024, 12, 768),    // GPT-2 117M (Small)
    (8, 1024, 16, 1024),   // GPT-2 345M (Medium)
    (8, 1024, 20, 1280),   // GPT-2 762M (Large)
    (8, 1024, 25, 1600),   // GPT-2 1542M (XL)
];

fn tipover_attention_bob() {
    let mut rng = StdRng::from_seed([0; 32]);
    // let (batch_size, sequence_length, n_heads, embedding_dim) = (2, 3, 4, 8); // toy
    // let (batch_size, sequence_length, n_heads, embedding_dim) = (32, 5, 12, 384); // shakespeare-char
    for cc in 0..5 {
    let mut output = String::new();
    let (batch_size, sequence_length, n_heads, embedding_dim) = GPT_CONFIGS[cc];
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

    let rc = rtq.attention(&rtk, &mut rtr);
    unsafe { RCOUNT += rc; }
    // reference_attention(&rtq, &rtk, &mut rtrb);
    println!("rcount {}", unsafe{ RCOUNT });
    let ref_time = t0.elapsed().as_micros();
    println!("ref {ref_time} µs ({n_weights} weights)");
    let t0 = Instant::now();
    let rtq = rtq.to_strategy(dense::Blas);
    let rtk = rtk.to_strategy(dense::Blas);
    let mut rtrb = DenseTensorFBlas::new(vec![batch_size, sequence_length, n_heads, n_heads]);
    rtq.attention(&rtk, &mut rtrb);
    let blas_time = t0.elapsed().as_micros();
    println!("blas {blas_time} µs ({n_weights} weights)");
    let total_mem = rtq.estimate_memory_usage() + rtk.estimate_memory_usage() + rtrb.estimate_memory_usage();
    output.push_str(&format!("ref_time={ref_time} µs blas_time={blas_time} µs n_weights={n_weights} total_mem={total_mem}\n"));
    println!("estimated memory usage: {} / {} / {} bytes (total {})",
        rtq.estimate_memory_usage(),
        rtk.estimate_memory_usage(),
        rtr.estimate_memory_usage(),
        total_mem);
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
    println!("bob,density,q_nz,k_nz,v_nz,mem_k,mem_q,mem_v,attn_time,gen_time,attn_dry");
    output.push_str("bob,density,q_nz,k_nz,v_nz,mem_k,mem_q,mem_v,attn_time,gen_time,attn_dry\n");
    for ii in 0..=16 {
    // log scale: 4 steps per decade, from 1/10000 to 1
    let density = 0.0001 * 10.0_f32.powf(ii as f32 / 4.0);
    let shape = [batch_size, sequence_length, n_heads, embedding_dim/n_heads];
    // in completely unstructured sparsity, at 2% PathMap outperforms the naive dense implementation
    let t0 = Instant::now();
    let rtq = SparseTensorFBOB::with_density(&mut rng, &shape, density);
    let rtk = SparseTensorFBOB::with_density(&mut rng, &shape, density);
    let gen_time = t0.elapsed().as_micros();
    let mut rtr = SparseTensorFBOB::new(4);
    let q_nz = rtq.m.val_count();
    let k_nz = rtk.m.val_count();
    // rtq.vF_mut().merkleize();
    // rtk.vF_mut().merkleize();
    let _count = rtq.attention_dry(&rtk, &mut rtr);
    let t0 = Instant::now();
    let _count = rtq.attention_dry(&rtk, &mut rtr);
    let attn_dry = t0.elapsed().as_micros();
    // let attn_dry = 0;
    let t0 = Instant::now();
    let count = rtq.attention(&rtk, &mut rtr);
    let attn_time = t0.elapsed().as_micros();
    unsafe { COUNT += count; }
    let mem_k = rtq.estimate_memory_usage();
    let mem_q = rtk.estimate_memory_usage();
    let mem_v = rtr.estimate_memory_usage();
    let v_nz = rtr.m.val_count();
    // let nz = q_nz + k_nz + v_nz;
    println!("bob,{density:.4},{q_nz},{k_nz},{v_nz},{mem_k},{mem_q},{mem_v},{attn_time},{gen_time},{attn_dry}");
    output.push_str(&format!("bob,{density:.4},{q_nz},{k_nz},{v_nz},{mem_k},{mem_q},{mem_v},{attn_time},{gen_time},{attn_dry}\n"));
    // bob_attention(&mut rtq.m.read_zipper(), &mut rtk.m.read_zipper(), &mut rtr.m.write_zipper(), 0);
    // println!("bob {} µs ({n_weights} weights, {q_nz} Q nz, {k_nz} K nz)", t0.elapsed().as_micros());
    // println!("count {}", unsafe{ COUNT });
    // println!("estimated memory usage: {} / {} / {} bytes",
    //     rtq.estimate_memory_usage(),
    //     rtk.estimate_memory_usage(),
    //     rtr.estimate_memory_usage());
    std::fs::write(format!("bob_results_{cc}.csv"), &output).unwrap();
    }
    }
}

fn tipover_attention_weave() {
    // Constants
    let mut rng = StdRng::from_seed([0; 32]);

    let (batch_size, sequence_length, n_heads, embedding_dim) = GPT_CONFIGS[4]; // GPT-2 XL
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
    println!("weave,density,q_nz,k_nz,v_nz,mem_k,mem_q,mem_v,attn_time,gen_time");
    for ii in 0..90 {
    let density = ((ii as f32)).max(0.1) / 100.0;
    // Weave/sparse
    let shape = &args_d[..];
    let t0 = Instant::now();
    let rtq = SparseTensorFWeave::with_density(&mut rng, shape, density as f32);
    let rtk = SparseTensorFWeave::with_density(&mut rng, shape, density as f32);
    let mut rtr = SparseTensorFWeave::new(4);
    let gen_time = t0.elapsed().as_micros();
    let q_nz = rtq.m.val_count();
    let k_nz = rtk.m.val_count();
    println!("weave density {}%: {q_nz} Q nz, {k_nz} K nz", density*100.0);
    // rtq.vF_mut().merkleize();
    // rtk.vF_mut().merkleize();
    let t0 = Instant::now();
    let count = rtq.attention(&rtk, &mut rtr);
    let attn_time = t0.elapsed().as_micros();
    unsafe { COUNT += count; }
    let v_nz = rtr.m.val_count();
    let mem_k = rtq.estimate_memory_usage();
    let mem_q = rtk.estimate_memory_usage();
    let mem_v = rtr.estimate_memory_usage();
    println!("weave,{density:.3},{q_nz},{k_nz},{v_nz},{mem_k},{mem_q},{mem_v},{attn_time},{gen_time}");
    // _short_weave_attention(&mut rtq.m.read_zipper(), &mut rtk.m.read_zipper(), &mut rtr.m.write_zipper());
    // println!("weave {} µs ({n_weights} weights, {q_nz} Q nz, {k_nz} K nz)", t0.elapsed().as_micros());
    // println!("count {}", unsafe{ COUNT });
    // println!("estimated memory usage: {} / {} / {} bytes",
    //     rtq.estimate_memory_usage(),
    //     rtk.estimate_memory_usage(),
    //     rtr.estimate_memory_usage());
    }
}



fn main() {
    unsafe {
        std::env::set_var("OPENBLAS_NUM_THREADS", "1");  // single-threaded
        std::env::set_var("OMP_NUM_THREADS", "1");       // OpenMP fallback
    }

    // Observation 1: worst case is best with Bob
    // Observation 2: bob and weave don't differ that much with the unoptimized impl
    // Observation 3: real-world matrix distributions are often not homogenously sparse (and benefit from dense submatrix)

    // Note: what do attention matrices actually look like?
    // !!! Sparse Auto-Encoders (sparsification fine-tuning) !!!
    // via some regularizer

    // Scientific application where the matrix simply does not fit on the GPU?

    // show sharing between pointwise operations:
    // sparse_dimensionwise();

    // tipover_attention_bob();
    // tipover_attention_weave();
    bench_chunked_vs_blas();
}

fn bench_chunked_vs_blas() {
    use chunked::{Chunked8, Chunked16};
    use traits::SparseMatrix;

    fn bench_one_config<const CHUNK: usize, const SQ: usize>(
        tag: &str,
        batch: usize, seq: usize, n_heads: usize, dim: usize,
    ) {
        use chunked::ChunkedTensor;
        const NRUNS: u128 = 10;
        let shape_qk = vec![batch, seq, n_heads, dim];
        let shape_out_dense = vec![batch, seq, n_heads, n_heads];

        // --- 100% density: correctness + baseline ---
        {
            let mut dq = dense::DenseTensorFRef::<dense::Blas>::new(shape_qk.clone());
            let mut dk = dense::DenseTensorFRef::<dense::Blas>::new(shape_qk.clone());
            let mut dr = dense::DenseTensorFRef::<dense::Blas>::new(shape_out_dense.clone());

            let mut cq = ChunkedTensor::<CHUNK, SQ>::with_shape(shape_qk.clone());
            let mut ck = ChunkedTensor::<CHUNK, SQ>::with_shape(shape_qk.clone());
            let mut cr = ChunkedTensor::<CHUNK, SQ>::with_shape(vec![batch, seq, n_heads, n_heads]);

            let mut c = 0f32;
            for b in 0..batch {
                for h in 0..seq {
                    for q in 0..n_heads {
                        for d in 0..dim {
                            c += 0.01;
                            dq.set(&[b, h, q, d], c);
                            dk.set(&[b, h, q, d], -c);
                            cq.set(&[b, h, q, d], c);
                            ck.set(&[b, h, q, d], -c);
                        }
                    }
                }
            }

            // Warmup + measure BLAS
            dq.attention(&dk, &mut dr);
            let t0 = Instant::now();
            for _ in 0..NRUNS {
                dr = dense::DenseTensorFRef::<dense::Blas>::new(shape_out_dense.clone());
                dq.attention(&dk, &mut dr);
            }
            let blas_us = t0.elapsed().as_micros() / NRUNS;

            // Warmup + measure chunked
            cq.attention(&ck, &mut cr);
            let t0 = Instant::now();
            for _ in 0..NRUNS {
                cr = ChunkedTensor::<CHUNK, SQ>::with_shape(vec![batch, seq, n_heads, n_heads]);
                cq.attention(&ck, &mut cr);
            }
            let chunk_us = t0.elapsed().as_micros() / NRUNS;

            let mut max_rel = 0.0f32;
            for b in 0..batch {
                for h in 0..seq {
                    for q in 0..n_heads {
                        for k in 0..n_heads {
                            let dv = dr.get(&[b, h, q, k]);
                            let cv = <ChunkedTensor<CHUNK, SQ> as SparseMatrix>::get(&cr, &[b, h, q, k]).unwrap_or(0.0);
                            let denom = dv.abs().max(cv.abs()).max(1.0);
                            max_rel = max_rel.max((dv - cv).abs() / denom);
                        }
                    }
                }
            }
            assert!(max_rel < 1e-3, "{tag} mismatch at 100% density: {max_rel}");

            let mem_blas = dq.estimate_memory_usage() + dk.estimate_memory_usage() + dr.estimate_memory_usage();
            let mem_chunk = cq.estimate_memory_usage() + ck.estimate_memory_usage() + cr.estimate_memory_usage();
            let speedup = blas_us as f64 / chunk_us.max(1) as f64;
            println!("{tag}_b{batch}_s{seq}_h{n_heads}_d{dim},1.0000,{blas_us},{chunk_us},{speedup:.3},{mem_blas},{mem_chunk},{max_rel:.2e}");
        }

        // --- sweep densities ---
        let mut rng = StdRng::from_seed([42; 32]);
        for ii in 0..=16 {
            let density = 0.0001 * 10.0_f32.powf(ii as f32 / 4.0);
            if density > 0.95 { continue; }
            let shape = &shape_qk[..];

            let cq = ChunkedTensor::<CHUNK, SQ>::with_density(&mut rng, shape, density);
            let ck = ChunkedTensor::<CHUNK, SQ>::with_density(&mut rng, shape, density);
            let mut cr = ChunkedTensor::<CHUNK, SQ>::with_shape(vec![batch, seq, n_heads, n_heads]);

            // Build matching dense BLAS tensors
            let mut dq = dense::DenseTensorFRef::<dense::Blas>::new(shape_qk.clone());
            let mut dk = dense::DenseTensorFRef::<dense::Blas>::new(shape_qk.clone());
            let mut dr = dense::DenseTensorFRef::<dense::Blas>::new(shape_out_dense.clone());
            for b in 0..batch {
                for h in 0..seq {
                    for q in 0..n_heads {
                        for d in 0..dim {
                            let ix = [b, h, q, d];
                            if let Some(v) = <ChunkedTensor<CHUNK, SQ> as SparseMatrix>::get(&cq, &ix) {
                                dq.set(&ix, v);
                            }
                            if let Some(v) = <ChunkedTensor<CHUNK, SQ> as SparseMatrix>::get(&ck, &ix) {
                                dk.set(&ix, v);
                            }
                        }
                    }
                }
            }

            // Warmup + measure BLAS (10x average)
            dq.attention(&dk, &mut dr);
            let t0 = Instant::now();
            for _ in 0..NRUNS {
                dr = dense::DenseTensorFRef::<dense::Blas>::new(shape_out_dense.clone());
                dq.attention(&dk, &mut dr);
            }
            let blas_us = t0.elapsed().as_micros() / NRUNS;

            // Warmup + measure chunked (10x average)
            cq.attention(&ck, &mut cr);
            let t0 = Instant::now();
            for _ in 0..NRUNS {
                cr = ChunkedTensor::<CHUNK, SQ>::with_shape(vec![batch, seq, n_heads, n_heads]);
                cq.attention(&ck, &mut cr);
            }
            let chunk_us = t0.elapsed().as_micros() / NRUNS;

            let mem_blas = dq.estimate_memory_usage() + dk.estimate_memory_usage() + dr.estimate_memory_usage();
            let mem_chunk = cq.estimate_memory_usage() + ck.estimate_memory_usage() + cr.estimate_memory_usage();
            let speedup = blas_us as f64 / chunk_us.max(1) as f64;
            println!("{tag}_b{batch}_s{seq}_h{n_heads}_d{dim},{density:.4},{blas_us},{chunk_us},{speedup:.3},{mem_blas},{mem_chunk},");
        }
    }

    println!("config,density,blas_us,chunked_us,speedup,mem_blas,mem_chunked,max_rel_err");

    for &(batch, seq, n_heads, edim) in &GPT_CONFIGS {
        let dim = edim / n_heads;
        bench_one_config::<8, 64>("c8", batch, seq, n_heads, dim);
        bench_one_config::<16, 256>("c16", batch, seq, n_heads, dim);
    }
}
