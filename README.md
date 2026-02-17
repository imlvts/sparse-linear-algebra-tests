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
