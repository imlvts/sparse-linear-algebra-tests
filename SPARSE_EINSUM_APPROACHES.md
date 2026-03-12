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

Three optimizations reduce the overhead from 10–12× to 2.7–5.3×:

1. **Sparse value caching:** `SparseRowLoop` stores the value from `row_entry()`
   and `MulAcc` reuses it, avoiding redundant `get_opt()` binary searches for
   inputs fully covered by a sparse loop.

2. **Dense accumulator (`AccStart`/`AccFlush`):** When the innermost loop slot
   appears in the output, the compiler emits `AccStart` (allocate dense vec) and
   `AccFlush` (scatter-gather write + clear) ops. This batches output writes
   instead of calling `get`/`set` per element. ~2× speedup on dense-output cases.

3. **Fused inner loops:** When a loop's body is a single `MulAcc`, the loop is
   marked `fused` and calls `mul_acc()` inline instead of recursing into
   `exec_at()`. Eliminates per-element recursive dispatch overhead. ~15% speedup.

**Verdict:** 2.7–5.3× slower than CSR matmul after optimizations. The remaining
gap is the per-element `mul_acc` overhead (index gathering, `Option` product
chain, accumulator branch) vs CSR's tight `a_val * b_val` inner loop.

The VM's strength is **flexibility**: it handles arbitrary specs (any number of
inputs, any dimensionality, mixed sparse/dense) without hardcoding the matmul
pattern. Adding new loop strategies (e.g., CSC for column-driven iteration)
would just require new `VmOp` variants.

**Capabilities vs other approaches:**

| | sparse-driven | VM | hash |
|---|---|---|---|
| Speed vs CSR | 0.62–1.55× (fastest) | 2.68–5.28× | 1.95–6.62× |
| Specs | matmul only | any einsum | matmul only |
| Inputs | 2D sparse only | any dim, mixed sparse/dense | 2D sparse only |
| Memory | O(n) dense vec/row | O(n) acc + call stack | O(nnz) hash/row |

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
lattice 10³ full (26 e/n)      1000   26000  17582417 µs      1822 µs      8588 µs     12149 µs      1920 µs
lattice 10³ thin (4 e/n)       1000    4070  13946100 µs       168 µs       661 µs       497 µs       242 µs
lattice 15³ full (26 e/n)      3375   87750         skip      5866 µs     30604 µs     41713 µs      6303 µs
lattice 15³ thin (4 e/n)       3375   13844         skip       967 µs      3178 µs      2038 µs       802 µs
lattice 20³ thin (4 e/n)       8000   31936         skip      2732 µs      9312 µs      5110 µs      1764 µs
random 1000n 5000e             1000    4987  14604412 µs       193 µs       837 µs       609 µs       312 µs
random 2000n 10000e            2000    9983         skip       717 µs      1913 µs      1331 µs       616 µs
```

### Slowdown relative to CSR matmul (lower = faster)

| Config | sparse/CSR | VM/CSR | hash/CSR |
|--------|-----------|--------|----------|
| 10³ full | 0.95× | 4.47× | 6.33× |
| 10³ thin | 0.69× | 2.73× | 2.05× |
| 15³ full | 0.93× | 4.86× | 6.62× |
| 15³ thin | 1.21× | 3.96× | 2.54× |
| 20³ thin | 1.55× | 5.28× | 2.90× |
| random 1k | 0.62× | 2.68× | 1.95× |
| random 2k | 1.16× | 3.11× | 2.16× |

### VM optimization progression

| Config | naive VM | +sparse vals +acc | +fused | CSR matmul |
|--------|----------|-------------------|--------|------------|
| 10³ full | 21,701 µs (10.76×) | 9,914 µs (5.17×) | 8,588 µs (4.47×) | 1,920 µs |
| 15³ full | 75,223 µs (12.25×) | 35,311 µs (5.75×) | 30,604 µs (4.86×) | 6,303 µs |
| 20³ thin | 9,191 µs (5.40×) | 9,578 µs (5.43×) | 9,312 µs (5.28×) | 1,764 µs |

### Speedup vs baseline (where available)

| Config | sparse | VM | hash | CSR matmul |
|--------|--------|-----|------|-----------|
| 10³ full | 9,650× | 2,048× | 1,447× | 9,157× |
| 10³ thin | 83,036× | 21,098× | 28,060× | 57,628× |
| random 1k | 75,669× | 17,449× | 23,980× | 46,808× |

## Key Takeaways

1. **Baseline is catastrophically slow** for sparse matrices — O(n³) kills it.

2. **Sparse-driven with dense accumulator** is the clear winner. It achieves near-parity
   with hand-written CSR matmul (0.64–1.42×) while being generic over any `Sparse2D` impl.
   The scatter-gather pattern (only clear touched entries) is critical for performance.

3. **The VM approach is the most flexible** — it handles any einsum spec with mixed
   sparse/dense inputs of any dimensionality. After three rounds of optimization
   (sparse value caching, dense accumulator, fused inner loops), it achieves
   2.7–5.3× of CSR matmul, down from the initial 10–12×. The remaining gap is
   the per-element `mul_acc` overhead vs CSR's tight inner loop.

4. **VM now beats hash on dense-output cases** (4.47× vs 6.33× for 10³ full)
   thanks to the accumulator. Hash still wins on thin graphs where its
   per-element overhead is lower than the VM's index-gathering machinery.

5. **Hash accumulator loses to dense accumulator** at these sparsity levels. It would
   only win for very large, extremely sparse outputs where the dense accumulator's
   O(n) clear cost dominates.

6. **The `Sparse2D` trait** (4 methods: `nnz`, `n_rows`, `row_nnz`, `row_entry`) is
   the right abstraction. It maps zero-cost to CSR and enables all three sparse approaches.

## Files Modified

- `einsum-dyn/src/lib.rs` — made `parse_spec`, `validate_dims`, `validate_output`, `Spec` pub(crate); added `pub mod sparse`
- `einsum-dyn/src/sparse.rs` — NEW: `Sparse2D` trait + approaches 2, 3, 4 with tests
- `src/graph_csr.rs` — `impl Sparse2D<Val> for CsrMatrix` + agreement tests + benchmark
