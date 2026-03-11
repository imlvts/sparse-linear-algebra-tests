use std::collections::BTreeMap;
use std::fmt;
use std::ops::{AddAssign, Mul};

/// Trait for N-dimensional array access.
pub trait NDIndex<T> {
    fn ndim(&self) -> usize;
    fn dim(&self, axis: usize) -> usize;
    fn get(&self, indices: &[usize]) -> T;
    fn set(&mut self, indices: &[usize], val: T);
}

/// Error returned when an einsum spec string is invalid.
#[derive(Debug, Clone)]
pub struct InvalidSpec(String);

impl fmt::Display for InvalidSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid einsum spec: {}", self.0)
    }
}

impl std::error::Error for InvalidSpec {}

/// Parsed einsum specification.
struct Spec {
    inputs: Vec<Vec<char>>,
    output: Vec<char>,
}

fn parse_spec(spec: &str, expected_inputs: usize) -> Result<Spec, InvalidSpec> {
    let spec = spec.replace(' ', "");

    let (lhs, rhs) = spec
        .split_once("->")
        .ok_or_else(|| InvalidSpec("missing '->'".into()))?;

    let inputs: Vec<Vec<char>> = lhs.split(',').map(|s| s.chars().collect()).collect();

    if inputs.len() != expected_inputs {
        return Err(InvalidSpec(format!(
            "expected {} input(s), got {}",
            expected_inputs,
            inputs.len()
        )));
    }

    for (i, inp) in inputs.iter().enumerate() {
        if inp.is_empty() {
            return Err(InvalidSpec(format!("input {} has no indices", i)));
        }
        for &ch in inp {
            if !ch.is_ascii_lowercase() {
                return Err(InvalidSpec(format!(
                    "index '{}' is not a lowercase letter",
                    ch
                )));
            }
        }
    }

    let output: Vec<char> = rhs.chars().collect();
    for &ch in &output {
        if !ch.is_ascii_lowercase() {
            return Err(InvalidSpec(format!(
                "output index '{}' is not a lowercase letter",
                ch
            )));
        }
    }

    // Build index locations map
    let mut index_locs: BTreeMap<char, Vec<(usize, usize)>> = BTreeMap::new();
    for (pi, inp) in inputs.iter().enumerate() {
        for (pos, &ch) in inp.iter().enumerate() {
            index_locs.entry(ch).or_default().push((pi, pos));
        }
    }

    // Validate: every output index must appear in at least one input
    for &ch in &output {
        if !index_locs.contains_key(&ch) {
            return Err(InvalidSpec(format!(
                "output index '{}' does not appear in any input",
                ch
            )));
        }
    }

    Ok(Spec { inputs, output })
}

/// Validates that array dimensions match the spec.
fn validate_dims<T, Arr: NDIndex<T>>(
    spec: &Spec,
    arrays: &[&Arr],
) -> Result<BTreeMap<char, usize>, InvalidSpec> {
    // Check ndim of each input
    for (i, (inp, arr)) in spec.inputs.iter().zip(arrays.iter()).enumerate() {
        if arr.ndim() != inp.len() {
            return Err(InvalidSpec(format!(
                "input {} has {} dimensions but spec has {} indices",
                i,
                arr.ndim(),
                inp.len()
            )));
        }
    }

    // Build dimension map, checking consistency
    let mut dims: BTreeMap<char, usize> = BTreeMap::new();
    for (pi, inp) in spec.inputs.iter().enumerate() {
        for (pos, &ch) in inp.iter().enumerate() {
            let d = arrays[pi].dim(pos);
            if let Some(&existing) = dims.get(&ch) {
                if existing != d {
                    return Err(InvalidSpec(format!(
                        "dimension mismatch for index '{}': {} vs {}",
                        ch, existing, d
                    )));
                }
            } else {
                dims.insert(ch, d);
            }
        }
    }

    Ok(dims)
}

/// Collects all unique indices in order of first appearance across inputs.
fn all_indices_ordered(spec: &Spec) -> Vec<char> {
    let mut seen = Vec::new();
    for inp in &spec.inputs {
        for &ch in inp {
            if !seen.contains(&ch) {
                seen.push(ch);
            }
        }
    }
    seen
}

/// Executes the einsum loop nest, calling `emit` for each combination of index values.
///
/// `loop_indices` is the ordered list of index chars to iterate over.
/// `dims` maps each index char to its size.
/// `emit` receives the current index values map for each innermost iteration.
fn loop_nest(
    loop_indices: &[char],
    dims: &BTreeMap<char, usize>,
    idx_vals: &mut BTreeMap<char, usize>,
    emit: &mut impl FnMut(&BTreeMap<char, usize>),
) {
    if loop_indices.is_empty() {
        emit(idx_vals);
        return;
    }
    let ch = loop_indices[0];
    let rest = &loop_indices[1..];
    let n = dims[&ch];
    for v in 0..n {
        idx_vals.insert(ch, v);
        loop_nest(rest, dims, idx_vals, emit);
    }
}

/// Builds an index slice from index chars and current values.
fn gather_indices(chars: &[char], vals: &BTreeMap<char, usize>, buf: &mut Vec<usize>) {
    buf.clear();
    for &ch in chars {
        buf.push(vals[&ch]);
    }
}

/// `einsum_binary(spec, a, b, out)` — binary einsum with tensor output.
///
/// Spec format: `"ab,bc->ac"` (numpy-style).
/// All indices in the output must appear in at least one input.
/// Indices present in inputs but absent from the output are contracted (summed over).
/// The output array must already have the correct shape.
pub fn einsum_binary<T, Arr>(spec: &str, a: &Arr, b: &Arr, out: &mut Arr) -> Result<(), InvalidSpec>
where
    T: Default + Copy + AddAssign + Mul<Output = T>,
    Arr: NDIndex<T>,
{
    let spec = parse_spec(spec, 2)?;
    let dims = validate_dims(&spec, &[a, b])?;

    // Validate output dimensions
    if out.ndim() != spec.output.len() {
        return Err(InvalidSpec(format!(
            "output has {} dimensions but spec has {} output indices",
            out.ndim(),
            spec.output.len()
        )));
    }
    for (pos, &ch) in spec.output.iter().enumerate() {
        if out.dim(pos) != dims[&ch] {
            return Err(InvalidSpec(format!(
                "output dimension {} is {} but expected {}",
                pos,
                out.dim(pos),
                dims[&ch]
            )));
        }
    }

    let free_set: Vec<char> = spec.output.clone();
    let all = all_indices_ordered(&spec);
    let contracted: Vec<char> = all
        .iter()
        .filter(|ch| !free_set.contains(ch))
        .copied()
        .collect();

    // Loop order: free indices first, then contracted
    let mut loop_order = free_set.clone();
    loop_order.extend(&contracted);

    let mut idx_vals = BTreeMap::new();
    let mut buf_a = Vec::new();
    let mut buf_b = Vec::new();
    let mut buf_out = Vec::new();

    let inp_a = &spec.inputs[0];
    let inp_b = &spec.inputs[1];
    let out_indices = &spec.output;

    if contracted.is_empty() {
        // No contraction — direct assignment
        loop_nest(&loop_order, &dims, &mut idx_vals, &mut |vals| {
            gather_indices(inp_a, vals, &mut buf_a);
            gather_indices(inp_b, vals, &mut buf_b);
            gather_indices(out_indices, vals, &mut buf_out);
            let v = a.get(&buf_a) * b.get(&buf_b);
            out.set(&buf_out, v);
        });
    } else {
        // With contraction — need to accumulate
        // We iterate over free indices, and for each, sum over contracted
        let mut idx_vals_outer = BTreeMap::new();
        loop_nest(&free_set, &dims, &mut idx_vals_outer, &mut |free_vals| {
            let mut acc: T = Default::default();
            let mut idx_vals_inner = free_vals.clone();
            loop_nest(
                &contracted,
                &dims,
                &mut idx_vals_inner,
                &mut |vals| {
                    gather_indices(inp_a, vals, &mut buf_a);
                    gather_indices(inp_b, vals, &mut buf_b);
                    acc += a.get(&buf_a) * b.get(&buf_b);
                },
            );
            gather_indices(out_indices, &idx_vals_inner, &mut buf_out);
            out.set(&buf_out, acc);
        });
    }

    Ok(())
}

/// `einsum_unary(spec, a, out)` — unary einsum with tensor output.
///
/// Spec format: `"ab->ba"` (numpy-style).
pub fn einsum_unary<T, Arr>(spec: &str, a: &Arr, out: &mut Arr) -> Result<(), InvalidSpec>
where
    T: Default + Copy + AddAssign + Mul<Output = T>,
    Arr: NDIndex<T>,
{
    let spec = parse_spec(spec, 1)?;
    let dims = validate_dims(&spec, &[a])?;

    if out.ndim() != spec.output.len() {
        return Err(InvalidSpec(format!(
            "output has {} dimensions but spec has {} output indices",
            out.ndim(),
            spec.output.len()
        )));
    }
    for (pos, &ch) in spec.output.iter().enumerate() {
        if out.dim(pos) != dims[&ch] {
            return Err(InvalidSpec(format!(
                "output dimension {} is {} but expected {}",
                pos,
                out.dim(pos),
                dims[&ch]
            )));
        }
    }

    let free_set: Vec<char> = spec.output.clone();
    let all = all_indices_ordered(&spec);
    let contracted: Vec<char> = all
        .iter()
        .filter(|ch| !free_set.contains(ch))
        .copied()
        .collect();

    let inp_a = &spec.inputs[0];
    let out_indices = &spec.output;
    let mut buf_a = Vec::new();
    let mut buf_out = Vec::new();

    if contracted.is_empty() {
        let loop_order = free_set.clone();
        let mut idx_vals = BTreeMap::new();
        loop_nest(&loop_order, &dims, &mut idx_vals, &mut |vals| {
            gather_indices(inp_a, vals, &mut buf_a);
            gather_indices(out_indices, vals, &mut buf_out);
            out.set(&buf_out, a.get(&buf_a));
        });
    } else {
        let mut idx_vals_outer = BTreeMap::new();
        loop_nest(&free_set, &dims, &mut idx_vals_outer, &mut |free_vals| {
            let mut acc: T = Default::default();
            let mut idx_vals_inner = free_vals.clone();
            loop_nest(
                &contracted,
                &dims,
                &mut idx_vals_inner,
                &mut |vals| {
                    gather_indices(inp_a, vals, &mut buf_a);
                    acc += a.get(&buf_a);
                },
            );
            gather_indices(out_indices, &idx_vals_inner, &mut buf_out);
            out.set(&buf_out, acc);
        });
    }

    Ok(())
}

/// `einsum_binary_scalar(spec, a, b)` — binary einsum with scalar output.
///
/// Spec format: `"ab,ab->"` (empty output = scalar).
pub fn einsum_binary_scalar<T, Arr>(spec: &str, a: &Arr, b: &Arr) -> Result<T, InvalidSpec>
where
    T: Default + Copy + AddAssign + Mul<Output = T>,
    Arr: NDIndex<T>,
{
    let spec = parse_spec(spec, 2)?;
    let dims = validate_dims(&spec, &[a, b])?;

    if !spec.output.is_empty() {
        return Err(InvalidSpec(
            "scalar output requires empty output indices (use 'ab,ab->')".into(),
        ));
    }

    let all = all_indices_ordered(&spec);
    let inp_a = &spec.inputs[0];
    let inp_b = &spec.inputs[1];
    let mut buf_a = Vec::new();
    let mut buf_b = Vec::new();
    let mut acc: T = Default::default();

    let mut idx_vals = BTreeMap::new();
    loop_nest(&all, &dims, &mut idx_vals, &mut |vals| {
        gather_indices(inp_a, vals, &mut buf_a);
        gather_indices(inp_b, vals, &mut buf_b);
        acc += a.get(&buf_a) * b.get(&buf_b);
    });

    Ok(acc)
}

/// `einsum_unary_scalar(spec, a)` — unary einsum with scalar output.
///
/// Spec format: `"aa->"` (empty output = scalar).
pub fn einsum_unary_scalar<T, Arr>(spec: &str, a: &Arr) -> Result<T, InvalidSpec>
where
    T: Default + Copy + AddAssign + Mul<Output = T>,
    Arr: NDIndex<T>,
{
    let spec = parse_spec(spec, 1)?;
    let dims = validate_dims(&spec, &[a])?;

    if !spec.output.is_empty() {
        return Err(InvalidSpec(
            "scalar output requires empty output indices (use 'aa->')".into(),
        ));
    }

    let all = all_indices_ordered(&spec);
    let inp_a = &spec.inputs[0];
    let mut buf_a = Vec::new();
    let mut acc: T = Default::default();

    let mut idx_vals = BTreeMap::new();
    loop_nest(&all, &dims, &mut idx_vals, &mut |vals| {
        gather_indices(inp_a, vals, &mut buf_a);
        acc += a.get(&buf_a);
    });

    Ok(acc)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal dense tensor for testing.
    struct T {
        m: Vec<f32>,
        d: Vec<usize>,
    }

    impl T {
        fn new(d: Vec<usize>) -> Self {
            let n: usize = d.iter().product();
            Self {
                m: vec![0.0; n],
                d,
            }
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
    }

    impl NDIndex<f32> for T {
        fn ndim(&self) -> usize {
            self.d.len()
        }

        fn dim(&self, axis: usize) -> usize {
            self.d[axis]
        }

        fn get(&self, ix: &[usize]) -> f32 {
            self.m[self.linear_index(ix)]
        }

        fn set(&mut self, ix: &[usize], v: f32) {
            let i = self.linear_index(ix);
            self.m[i] = v;
        }
    }

    fn set_matrix(t: &mut T, vals: &[f32]) {
        for (i, &v) in vals.iter().enumerate() {
            t.m[i] = v;
        }
    }

    #[test]
    fn test_matmul() {
        let mut a = T::new(vec![2, 3]);
        let mut b = T::new(vec![3, 2]);
        set_matrix(&mut a, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        set_matrix(&mut b, &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);

        let mut c = T::new(vec![2, 2]);
        einsum_binary("ab,bc->ac", &a, &b, &mut c).unwrap();

        assert_eq!(c.get(&[0, 0]), 58.0);
        assert_eq!(c.get(&[0, 1]), 64.0);
        assert_eq!(c.get(&[1, 0]), 139.0);
        assert_eq!(c.get(&[1, 1]), 154.0);
    }

    #[test]
    fn test_transpose() {
        let mut a = T::new(vec![2, 3]);
        set_matrix(&mut a, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let mut t = T::new(vec![3, 2]);
        einsum_unary("ab->ba", &a, &mut t).unwrap();

        assert_eq!(t.get(&[0, 0]), 1.0);
        assert_eq!(t.get(&[0, 1]), 4.0);
        assert_eq!(t.get(&[1, 0]), 2.0);
        assert_eq!(t.get(&[1, 1]), 5.0);
        assert_eq!(t.get(&[2, 0]), 3.0);
        assert_eq!(t.get(&[2, 1]), 6.0);
    }

    #[test]
    fn test_outer_product() {
        let mut a = T::new(vec![3]);
        set_matrix(&mut a, &[1.0, 2.0, 3.0]);
        let mut b = T::new(vec![2]);
        set_matrix(&mut b, &[4.0, 5.0]);

        let mut c = T::new(vec![3, 2]);
        einsum_binary("a,b->ab", &a, &b, &mut c).unwrap();

        assert_eq!(c.get(&[0, 0]), 4.0);
        assert_eq!(c.get(&[0, 1]), 5.0);
        assert_eq!(c.get(&[1, 0]), 8.0);
        assert_eq!(c.get(&[1, 1]), 10.0);
        assert_eq!(c.get(&[2, 0]), 12.0);
        assert_eq!(c.get(&[2, 1]), 15.0);
    }

    #[test]
    fn test_vecmat() {
        let mut v = T::new(vec![2]);
        set_matrix(&mut v, &[1.0, 2.0]);
        let mut m = T::new(vec![2, 2]);
        set_matrix(&mut m, &[3.0, 4.0, 5.0, 6.0]);

        let mut r = T::new(vec![2]);
        einsum_binary("a,ab->b", &v, &m, &mut r).unwrap();

        assert_eq!(r.get(&[0]), 13.0);
        assert_eq!(r.get(&[1]), 16.0);
    }

    #[test]
    fn test_dot() {
        let mut a = T::new(vec![4]);
        let mut b = T::new(vec![4]);
        set_matrix(&mut a, &[1.0, 2.0, 3.0, 4.0]);
        set_matrix(&mut b, &[5.0, 6.0, 7.0, 8.0]);

        let result: f32 = einsum_binary_scalar("i,i->", &a, &b).unwrap();
        assert_eq!(result, 70.0);
    }

    #[test]
    fn test_trace() {
        let mut m = T::new(vec![3, 3]);
        set_matrix(
            &mut m,
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        );

        let result: f32 = einsum_unary_scalar("aa->", &m).unwrap();
        assert_eq!(result, 15.0);
    }

    #[test]
    fn test_frobenius2() {
        let mut a = T::new(vec![2, 2]);
        set_matrix(&mut a, &[1.0, 2.0, 3.0, 4.0]);

        let result: f32 = einsum_binary_scalar("ab,ab->", &a, &a).unwrap();
        assert_eq!(result, 30.0);
    }

    #[test]
    fn test_attention() {
        let (b, h, q_len, k_len, dim) = (2, 2, 3, 2, 4);
        let mut q = T::new(vec![b, h, q_len, dim]);
        let mut k = T::new(vec![b, h, k_len, dim]);

        for bi in 0..b {
            for hi in 0..h {
                for qi in 0..q_len {
                    for di in 0..dim {
                        let v =
                            (bi + 1) as f32 * (hi + 1) as f32 * (qi + 1) as f32 + di as f32;
                        q.set(&[bi, hi, qi, di], v);
                    }
                }
                for ki in 0..k_len {
                    for di in 0..dim {
                        let v =
                            (bi + 1) as f32 * (hi + 1) as f32 * (ki + 1) as f32 * (di + 1) as f32;
                        k.set(&[bi, hi, ki, di], v);
                    }
                }
            }
        }

        let mut out = T::new(vec![b, h, q_len, k_len]);
        einsum_binary("bhqd,bhkd->bhqk", &q, &k, &mut out).unwrap();

        // Verify against naive computation
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
    fn test_invalid_spec() {
        let a = T::new(vec![2, 3]);
        let b = T::new(vec![3, 2]);
        let mut c = T::new(vec![2, 2]);

        // Missing arrow
        assert!(einsum_binary("ab,bc", &a, &b, &mut c).is_err());

        // Wrong number of inputs
        assert!(einsum_binary("ab->ab", &a, &b, &mut c).is_err());

        // Output index not in any input
        assert!(einsum_binary("ab,bc->az", &a, &b, &mut c).is_err());

        // Dimension mismatch (a is 2x3, b is 3x2, spec says both first dims match)
        let mut d = T::new(vec![2, 2]);
        assert!(einsum_binary("ab,ac->bc", &a, &b, &mut d).is_err());
    }

    #[test]
    fn test_unary_row_sum() {
        // Sum over columns: ab->a means sum over b
        let mut a = T::new(vec![2, 3]);
        set_matrix(&mut a, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let mut out = T::new(vec![2]);
        einsum_unary("ab->a", &a, &mut out).unwrap();

        assert_eq!(out.get(&[0]), 6.0); // 1+2+3
        assert_eq!(out.get(&[1]), 15.0); // 4+5+6
    }
}
