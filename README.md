# Sparse Linear Algebra Tests

Benchmarking SpGEMM (sparse general matrix multiplication) across multiple implementations:

| Implementation | Module | Description |
|---|---|---|
| BTreeMap | `graph.rs` | Naive `BTreeMap<(usize,usize), u64>` baseline |
| CSR (seq) | `graph_csr.rs` | Hand-rolled CSR, dense accumulator |
| CSR (par) | `graph_csr.rs` | Two-pass symbolic+numeric with rayon |
| sprs | `graph_sprs.rs` | Wrapper around the `sprs` crate |
| MAGNUS (seq) | `graph_magnus.rs` | ICS'25 row-categorization SpGEMM |
| MAGNUS (par) | `graph_magnus.rs` | MAGNUS with rayon parallelism |

## Running benchmarks

```bash
cargo test --lib --release graph_magnus::tests::bench -- --nocapture
```

This prints CSV to stdout with columns:
`side,nodes,e_per_n,nnz,components,orig_btree_us,csr_us,csr_par_us,sprs_us,magnus_seq_us,magnus_par_us,x_csr,x_csr_par,x_sprs,x_magnus_seq,x_magnus_par`

Save the output for plotting:

```bash
cargo test --lib --release graph_magnus::tests::bench -- --nocapture 2>&1 | tee bench_output.txt
```

## Repeated exponentiation benchmark

Measures A, A², A³, … A⁷ on a 30×30×30 3D Moore torus (27k nodes, ~3 edges/node), comparing CSR seq, CSR par, MAGNUS seq, and MAGNUS par. Each step computes A^k = A^(k-1) × A.

```bash
cargo test --lib --release bench_repeated_exponentiation -- --nocapture
```

Sample results (release, 3-iteration average):

| Step | nnz | CSR seq | CSR par | MAGNUS seq | MAGNUS par | ×CSR par | ×MAGNUS seq | ×MAGNUS par |
|------|-----|---------|---------|------------|------------|----------|-------------|-------------|
| A² | 252k | 4.0ms | 4.9ms | 19.8ms | 8.3ms | 0.82× | 0.20× | 0.49× |
| A³ | 655k | 14.8ms | 5.8ms | 47.4ms | 14.4ms | 2.54× | 0.31× | 1.03× |
| A⁴ | 1.57M | 43.9ms | 9.0ms | 97.4ms | 23.5ms | 4.89× | 0.45× | 1.86× |
| A⁵ | 3.38M | 101ms | 17.1ms | 169ms | 28.3ms | 5.88× | 0.59× | 3.56× |
| A⁶ | 6.59M | 192ms | 24.4ms | 290ms | 80.4ms | 7.84× | 0.66× | 2.38× |
| A⁷ | 11.7M | 358ms | 40.5ms | 457ms | 129ms | 8.83× | 0.78× | 2.77× |

Speedups are relative to CSR sequential. CSR par scales from 0.82× (overhead at low nnz) to 8.83× at step 7. MAGNUS sequential is consistently slower than CSR sequential but the gap narrows with density. MAGNUS parallel sits between the two CSR variants.

## Generating plots

Requires Python 3 with `numpy` and `matplotlib`.

3D surface plot (log-log-log) comparing CSR vs MAGNUS, sequential and parallel:

```bash
python3 plot_surface.py bench_output.txt
```

This saves `surface_csr_vs_magnus.png`.

You can also pipe directly:

```bash
cargo test --lib --release graph_magnus::tests::bench -- --nocapture 2>&1 | python3 plot_surface.py
```
