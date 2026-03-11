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

/// Slot index for a char: `ch as u8 - b'a'`, so 'a'=0, 'b'=1, ..., 'z'=25.
#[inline(always)]
fn slot(ch: char) -> u8 {
    ch as u8 - b'a'
}

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

    // Validate: every output index must appear in at least one input
    let mut seen = [false; 26];
    for inp in &inputs {
        for &ch in inp {
            seen[slot(ch) as usize] = true;
        }
    }
    for &ch in &output {
        if !seen[slot(ch) as usize] {
            return Err(InvalidSpec(format!(
                "output index '{}' does not appear in any input",
                ch
            )));
        }
    }

    Ok(Spec { inputs, output })
}

/// Validates that array dimensions match the spec. Returns dims as a `[usize; 26]`
/// array (indexed by slot). Unused slots are 0.
fn validate_dims<T, Arr: NDIndex<T>>(
    spec: &Spec,
    arrays: &[&Arr],
) -> Result<[usize; 26], InvalidSpec> {
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

    let mut dims = [0usize; 26];
    let mut set = [false; 26];
    for (pi, inp) in spec.inputs.iter().enumerate() {
        for (pos, &ch) in inp.iter().enumerate() {
            let s = slot(ch) as usize;
            let d = arrays[pi].dim(pos);
            if set[s] {
                if dims[s] != d {
                    return Err(InvalidSpec(format!(
                        "dimension mismatch for index '{}': {} vs {}",
                        ch, dims[s], d
                    )));
                }
            } else {
                dims[s] = d;
                set[s] = true;
            }
        }
    }

    Ok(dims)
}

/// Collects all unique indices in order of first appearance, as a SlotList.
fn all_slots_ordered(spec: &Spec) -> SlotList {
    let mut seen = [false; 26];
    let mut slots = [0u8; 26];
    let mut len = 0u8;
    for inp in &spec.inputs {
        for &ch in inp {
            let s = slot(ch);
            if !seen[s as usize] {
                seen[s as usize] = true;
                slots[len as usize] = s;
                len += 1;
            }
        }
    }
    SlotList { slots, len }
}

/// Stack-only index buffer: fixed array + length, no heap.
struct Idx {
    data: [usize; 26],
    len: u8,
}

impl Idx {
    #[inline(always)]
    fn as_slice(&self) -> &[usize] {
        &self.data[..self.len as usize]
    }
}

/// Precomputed gather pattern: slot indices stored on the stack.
struct Pattern {
    slots: [u8; 26],
    len: u8,
}

impl Pattern {
    fn from_chars(chars: &[char]) -> Self {
        let mut slots = [0u8; 26];
        for (i, &ch) in chars.iter().enumerate() {
            slots[i] = slot(ch);
        }
        Pattern {
            slots,
            len: chars.len() as u8,
        }
    }

    /// Gather index values from `vals` into `out` according to this pattern.
    #[inline(always)]
    fn gather(&self, vals: &[usize; 26], out: &mut Idx) {
        out.len = self.len;
        for i in 0..self.len as usize {
            out.data[i] = vals[self.slots[i] as usize];
        }
    }
}

/// Precomputed loop-slot list stored on the stack.
struct SlotList {
    slots: [u8; 26],
    len: u8,
}

impl SlotList {
    fn from_pattern(p: &Pattern) -> Self {
        SlotList {
            slots: p.slots,
            len: p.len,
        }
    }

    fn from_chars(chars: &[char]) -> Self {
        let p = Pattern::from_chars(chars);
        Self::from_pattern(&p)
    }

    fn as_slice(&self) -> &[u8] {
        &self.slots[..self.len as usize]
    }

    fn contains(&self, s: u8) -> bool {
        self.as_slice().contains(&s)
    }

    fn filtered_complement(all: &[u8], free: &SlotList) -> Self {
        let mut slots = [0u8; 26];
        let mut len = 0u8;
        for &s in all {
            if !free.contains(s) {
                slots[len as usize] = s;
                len += 1;
            }
        }
        SlotList { slots, len }
    }
}

/// Recursive loop nest over slots. `loop_slots[i]` is a slot index,
/// `dims` and `vals` are flat [usize; 26] arrays.
#[inline(always)]
fn loop_nest(
    loop_slots: &[u8],
    dims: &[usize; 26],
    vals: &mut [usize; 26],
    emit: &mut impl FnMut(&[usize; 26]),
) {
    if loop_slots.is_empty() {
        emit(vals);
        return;
    }
    let s = loop_slots[0] as usize;
    let rest = &loop_slots[1..];
    let n = dims[s];
    for v in 0..n {
        vals[s] = v;
        loop_nest(rest, dims, vals, emit);
    }
}

// Iterative variant using an explicit counter stack.
// Benchmarked ~same as recursive.
#[cfg(any())]
#[inline(always)]
fn loop_nest_iterative(
    loop_slots: &[u8],
    dims: &[usize; 26],
    vals: &mut [usize; 26],
    emit: &mut impl FnMut(&[usize; 26]),
) {
    let depth = loop_slots.len();
    if depth == 0 {
        emit(vals);
        return;
    }
    let mut counters = [0usize; 26];
    for i in 0..depth {
        vals[loop_slots[i] as usize] = 0;
    }
    loop {
        emit(vals);
        let mut level = depth - 1;
        loop {
            let s = loop_slots[level] as usize;
            counters[level] += 1;
            if counters[level] < dims[s] {
                vals[s] = counters[level];
                break;
            }
            counters[level] = 0;
            vals[s] = 0;
            if level == 0 {
                return;
            }
            level -= 1;
        }
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

    if out.ndim() != spec.output.len() {
        return Err(InvalidSpec(format!(
            "output has {} dimensions but spec has {} output indices",
            out.ndim(),
            spec.output.len()
        )));
    }
    for (pos, &ch) in spec.output.iter().enumerate() {
        if out.dim(pos) != dims[slot(ch) as usize] {
            return Err(InvalidSpec(format!(
                "output dimension {} is {} but expected {}",
                pos,
                out.dim(pos),
                dims[slot(ch) as usize]
            )));
        }
    }

    let free_slots = SlotList::from_chars(&spec.output);
    let all = all_slots_ordered(&spec);
    let contracted_slots = SlotList::filtered_complement(all.as_slice(), &free_slots);

    let pat_a = Pattern::from_chars(&spec.inputs[0]);
    let pat_b = Pattern::from_chars(&spec.inputs[1]);
    let pat_out = Pattern::from_chars(&spec.output);

    let mut vals = [0usize; 26];
    let mut buf_a = Idx { data: [0; 26], len: 0 };
    let mut buf_b = Idx { data: [0; 26], len: 0 };
    let mut buf_out = Idx { data: [0; 26], len: 0 };

    if contracted_slots.len == 0 {
        // No contraction — direct assignment
        loop_nest(free_slots.as_slice(), &dims, &mut vals, &mut |vals| {
            pat_a.gather(vals, &mut buf_a);
            pat_b.gather(vals, &mut buf_b);
            pat_out.gather(vals, &mut buf_out);
            out.set(buf_out.as_slice(), a.get(buf_a.as_slice()) * b.get(buf_b.as_slice()));
        });
    } else {
        // With contraction — accumulate per output element
        loop_nest(free_slots.as_slice(), &dims, &mut vals, &mut |free_vals| {
            let mut acc: T = Default::default();
            let mut inner_vals = *free_vals;
            loop_nest(
                contracted_slots.as_slice(),
                &dims,
                &mut inner_vals,
                &mut |vals| {
                    pat_a.gather(vals, &mut buf_a);
                    pat_b.gather(vals, &mut buf_b);
                    acc += a.get(buf_a.as_slice()) * b.get(buf_b.as_slice());
                },
            );
            pat_out.gather(free_vals, &mut buf_out);
            out.set(buf_out.as_slice(), acc);
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
        if out.dim(pos) != dims[slot(ch) as usize] {
            return Err(InvalidSpec(format!(
                "output dimension {} is {} but expected {}",
                pos,
                out.dim(pos),
                dims[slot(ch) as usize]
            )));
        }
    }

    let free_slots = SlotList::from_chars(&spec.output);
    let all = all_slots_ordered(&spec);
    let contracted_slots = SlotList::filtered_complement(all.as_slice(), &free_slots);

    let pat_a = Pattern::from_chars(&spec.inputs[0]);
    let pat_out = Pattern::from_chars(&spec.output);
    let mut vals = [0usize; 26];
    let mut buf_a = Idx { data: [0; 26], len: 0 };
    let mut buf_out = Idx { data: [0; 26], len: 0 };

    if contracted_slots.len == 0 {
        loop_nest(free_slots.as_slice(), &dims, &mut vals, &mut |vals| {
            pat_a.gather(vals, &mut buf_a);
            pat_out.gather(vals, &mut buf_out);
            out.set(buf_out.as_slice(), a.get(buf_a.as_slice()));
        });
    } else {
        loop_nest(free_slots.as_slice(), &dims, &mut vals, &mut |free_vals| {
            let mut acc: T = Default::default();
            let mut inner_vals = *free_vals;
            loop_nest(
                contracted_slots.as_slice(),
                &dims,
                &mut inner_vals,
                &mut |vals| {
                    pat_a.gather(vals, &mut buf_a);
                    acc += a.get(buf_a.as_slice());
                },
            );
            pat_out.gather(free_vals, &mut buf_out);
            out.set(buf_out.as_slice(), acc);
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

    let all = all_slots_ordered(&spec);
    let pat_a = Pattern::from_chars(&spec.inputs[0]);
    let pat_b = Pattern::from_chars(&spec.inputs[1]);
    let mut vals = [0usize; 26];
    let mut buf_a = Idx { data: [0; 26], len: 0 };
    let mut buf_b = Idx { data: [0; 26], len: 0 };
    let mut acc: T = Default::default();

    loop_nest(all.as_slice(), &dims, &mut vals, &mut |vals| {
        pat_a.gather(vals, &mut buf_a);
        pat_b.gather(vals, &mut buf_b);
        acc += a.get(buf_a.as_slice()) * b.get(buf_b.as_slice());
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

    let all = all_slots_ordered(&spec);
    let pat_a = Pattern::from_chars(&spec.inputs[0]);
    let mut vals = [0usize; 26];
    let mut buf_a = Idx { data: [0; 26], len: 0 };
    let mut acc: T = Default::default();

    loop_nest(all.as_slice(), &dims, &mut vals, &mut |vals| {
        pat_a.gather(vals, &mut buf_a);
        acc += a.get(buf_a.as_slice());
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

        assert!(einsum_binary("ab,bc", &a, &b, &mut c).is_err());
        assert!(einsum_binary("ab->ab", &a, &b, &mut c).is_err());
        assert!(einsum_binary("ab,bc->az", &a, &b, &mut c).is_err());

        let mut d = T::new(vec![2, 2]);
        assert!(einsum_binary("ab,ac->bc", &a, &b, &mut d).is_err());
    }

    #[test]
    fn test_unary_row_sum() {
        let mut a = T::new(vec![2, 3]);
        set_matrix(&mut a, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let mut out = T::new(vec![2]);
        einsum_unary("ab->a", &a, &mut out).unwrap();

        assert_eq!(out.get(&[0]), 6.0);
        assert_eq!(out.get(&[1]), 15.0);
    }
}
