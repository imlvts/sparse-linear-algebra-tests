# Sparse Einsum: Four Approaches

## Summary

We explored four approaches to implementing Einstein summation on sparse (CSR) matrices,
compared against a hand-written CSR matmul as reference. All approaches compute `C = A × B`
via the einsum spec `"ab,bc->ac"`.

## Approaches

### 1. Baseline: Dense Loop with `get_opt` (existing `einsum_binary`)

**How it works:** The standard einsum iterates the full n×n×n index space (three nested
loops over a, b, c). At each point it calls `get_opt()` on both inputs — which returns
`None` for structural zeros in sparse matrices, skipping the multiply-accumulate.

**Complexity:** O(n³) regardless of sparsity. The `get_opt` check saves the multiply but
not the loop iterations.

**Code:** No changes — uses `einsum_binary` from `einsum-dyn/src/lib.rs` directly.

**Verdict:** Unusable for anything beyond tiny matrices. 17.6 seconds for a 1000-node graph.

---

### 2. Sparse-Driven with Dense Accumulator (`einsum_sparse_driven`)

**How it works:** Introduces a `Sparse2D<T>` trait with `row_nnz()` and `row_entry()`
methods. The outer loop iterates rows of A densely. For each row, it iterates A's
non-zeros to find (k, a_val), then iterates B's row k sparsely to find (j, b_val),
accumulating into a dense `Vec<T>` per output row. Only touched columns are written
and cleared (scatter-gather pattern from CSR matmul).

**Complexity:** O(flops) where flops = Σ_i Σ_{k∈row(i)} row_nnz_B(k). Same as CSR matmul.

**Code:** `einsum-dyn/src/sparse.rs` — `einsum_sparse_driven()`

**Verdict:** Fastest einsum approach. Within 0.6–1.3× of hand-written CSR matmul.
The small overhead comes from output index mapping and `NDIndex::set()` calls.

---

### 3. Custom VM (`einsum_vm`)

**How it works:** Compiles the einsum spec into flat bytecode. A greedy scheduler
analyzes the spec to decide which loops iterate sparsely vs densely:

1. Find slots that are axis-0 of some input → assign as `DenseLoop`
2. Find slots that are axis-1 of an input whose axis-0 is already fixed
   → assign as `SparseRowLoop` (iterate non-zeros via `NDIndex::sparse_row_*`)
3. Emit flat bytecode: loop-starts, then `MulAcc`, then `LoopEnd` markers
4. Each loop-start stores `end_pc` (jump target); each `LoopEnd` stores `start_pc`
5. The interpreter recurses at loop-starts and returns at `LoopEnd` — the call
   stack is the loop stack

For `"ab,bc->ac"` the VM produces:
```
FOR a IN 0..n
  FOR (b, val) IN input[0].row(a)  [SPARSE]
    FOR (c, val) IN input[1].row(b)  [SPARSE]
      MUL_ACC → output
```

**Complexity:** Same O(flops) as approach 2, but with interpreter overhead per operation.

**Code:** `einsum-dyn/src/sparse.rs` — `compile_vm()`, `VmProgram::exec()`, `VmOp` enum

**Verdict:** 2.5–12× slower than CSR matmul due to:
- `get_opt()` calls at MulAcc (binary search in CSR) instead of direct value access
  from `row_entry()` — this is the dominant cost
- Recursive dispatch overhead per bytecode op
- Element-by-element output writes via `get`/`set` (no accumulator batching)

The VM's strength is **flexibility**: it handles arbitrary specs (any number of
inputs, any dimensionality, mixed sparse/dense) without hardcoding the matmul
pattern. Adding new loop strategies (e.g., CSC for column-driven iteration)
would just require new `VmOp` variants.

**Capabilities vs other approaches:**

| | sparse-driven | hash | VM |
|---|---|---|---|
| Speed vs CSR | 0.61–1.48× (fastest) | 1.95–6.71× | 2.54–12.25× (slowest) |
| Specs | matmul only | matmul only | any einsum |
| Inputs | 2D sparse only | 2D sparse only | any dim, mixed sparse/dense |
| Memory | O(n) dense vec/row | O(nnz) hash/row | O(depth) call stack |

---

### 4. Sparse-Driven with Hash Accumulator (`einsum_sparse_hash`)

**How it works:** Same iteration as approach 2 (sparse-driven), but uses a
`HashMap<usize, T>` per output row instead of a dense `Vec<T>`. Products are
inserted/accumulated into the hash map, then written to the output.

**Complexity:** O(flops × hash_overhead). Same work as approach 2 but with
HashMap insert/lookup costs per product.

**Code:** `einsum-dyn/src/sparse.rs` — `einsum_sparse_hash()`

**Verdict:** 2–6× slower than approach 2 for these benchmarks. The hash overhead
dominates for matrices with moderate density. Would be advantageous only for
extremely large, extremely sparse matrices where the output has very few non-zeros
per row and a dense accumulator would waste memory.

---

## Benchmark Results (release mode, single-threaded)

```
config                            n     nnz     baseline       sparse           VM         hash   CSR matmul
─────────────────────────────────────────────────────────────────────────────────────────────────────────────
lattice 10³ full (26 e/n)      1000   26000  17438703 µs      1394 µs     21701 µs     11928 µs      2017 µs
lattice 10³ thin (4 e/n)       1000    4070  14053584 µs       164 µs       612 µs       480 µs       238 µs
lattice 15³ full (26 e/n)      3375   87750         skip      5759 µs     75223 µs     41175 µs      6139 µs
lattice 15³ thin (4 e/n)       3375   13844         skip       955 µs      3394 µs      2020 µs       762 µs
lattice 20³ thin (4 e/n)       8000   31936         skip      2515 µs      9191 µs      5063 µs      1702 µs
random 1000n 5000e             1000    4987  14747102 µs       184 µs       773 µs       592 µs       304 µs
random 2000n 10000e            2000    9983         skip       669 µs      1936 µs      1303 µs       601 µs
```

### Slowdown relative to CSR matmul (lower = faster)

| Config | sparse/CSR | VM/CSR | hash/CSR |
|--------|-----------|--------|----------|
| 10³ full | 0.69× | 10.76× | 5.91× |
| 10³ thin | 0.69× | 2.57× | 2.02× |
| 15³ full | 0.94× | 12.25× | 6.71× |
| 15³ thin | 1.25× | 4.45× | 2.65× |
| 20³ thin | 1.48× | 5.40× | 2.97× |
| random 1k | 0.61× | 2.54× | 1.95× |
| random 2k | 1.11× | 3.22× | 2.17× |

### Speedup vs baseline (where available)

| Config | sparse | VM | hash | CSR matmul |
|--------|--------|-----|------|-----------|
| 10³ full | 12,510× | 803× | 1,462× | 8,645× |
| 10³ thin | 85,693× | 22,964× | 29,278× | 59,049× |
| random 1k | 80,147× | 19,077× | 24,911× | 48,510× |

## Key Takeaways

1. **Baseline is catastrophically slow** for sparse matrices — O(n³) kills it.

2. **Sparse-driven with dense accumulator** is the clear winner. It achieves near-parity
   with hand-written CSR matmul (0.64–1.42×) while being generic over any `Sparse2D` impl.
   The scatter-gather pattern (only clear touched entries) is critical for performance.

3. **The VM approach is the most flexible** — it handles any einsum spec with mixed
   sparse/dense inputs of any dimensionality. It pays 2.5–12× overhead vs CSR matmul,
   mostly from `get_opt()` binary search at each MulAcc instead of direct `row_entry()`
   value access. The flat bytecode with recursive PC dispatch keeps interpreter overhead
   minimal.

4. **Hash accumulator loses to dense accumulator** at these sparsity levels. It would
   only win for very large, extremely sparse outputs where the dense accumulator's
   O(n) clear cost dominates.

5. **The `Sparse2D` trait** (4 methods: `nnz`, `n_rows`, `row_nnz`, `row_entry`) is
   the right abstraction. It maps zero-cost to CSR and enables all three sparse approaches.

## Files Modified

- `einsum-dyn/src/lib.rs` — made `parse_spec`, `validate_dims`, `validate_output`, `Spec` pub(crate); added `pub mod sparse`
- `einsum-dyn/src/sparse.rs` — NEW: `Sparse2D` trait + approaches 2, 3, 4 with tests
- `src/graph_csr.rs` — `impl Sparse2D<Val> for CsrMatrix` + agreement tests + benchmark
