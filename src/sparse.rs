use std::hash::Hasher;
use num_traits::Zero;
use pathmap::*;
use pathmap::ring::{AlgebraicResult, Lattice};
use pathmap::utils::ints::{indices_to_weave, indices_to_bob};
use pathmap::utils::{BitMask, ByteMask};
use pathmap::zipper::{ReadZipperUntracked, WriteZipperUntracked, ZipperWriting, ZipperMoving, Zipper, ZipperValues};

use crate::traits::{Attention, SparseMatrix, FromRng};

#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct FAddMul(pub f32);

impl std::ops::Deref for FAddMul { 
    type Target = f32; 
    fn deref(&self) -> &Self::Target { &self.0 } 
}

impl std::hash::Hash for FAddMul { 
    fn hash<H: Hasher>(&self, state: &mut H) { 
        self.0.to_bits().hash(state); 
    } 
}

// Note FAddMul is *not* a valid lattice under pjoin, but until we have bitraversal policies, this will have to do
impl Lattice for FAddMul {
    fn pjoin(&self, other: &Self) -> AlgebraicResult<Self> where Self: Sized {
        if self.0.is_zero() { return AlgebraicResult::Identity(1) }
        if other.0.is_zero() { return AlgebraicResult::Identity(2) }
        let s = self.0 + other.0;
        // make sparse if the dense sides had opposite signs and nearly cancelled out
        if self.0 * other.0 < 0f32 && s.abs() < 1e-9 { return AlgebraicResult::None }
        AlgebraicResult::Element(FAddMul(s))
    }

    fn pmeet(&self, other: &Self) -> AlgebraicResult<Self> where Self: Sized {
        let s = self.0*other.0;
        if s.abs() < 1e-9 { return AlgebraicResult::None }
        AlgebraicResult::Element(FAddMul(s))
    }
}

// ============================================================================
// SparseTensorFBOB - Big-endian Ordered Binary encoding
// ============================================================================

pub struct SparseTensorFBOB {
    pub m: PathMap<f32>,
    pub d: usize,
    p: Vec<u8>
}

impl SparseTensorFBOB {
    pub fn set(&mut self, ix: &[usize], v: f32) {
        self.p.clear();
        let len = indices_to_bob(ix, &mut vec![]);
        self.p.extend(std::iter::repeat_n(0u8, 64 - len));
        indices_to_bob(ix, &mut self.p);
        self.m.insert(&self.p[..], v);
    }

    pub fn add(&self, other: &Self) -> Self { 
        Self::vf32(self.vf().join(other.vf()), self.d) 
    }

    pub fn mul(&self, other: &Self) -> Self { 
        Self::vf32(self.vf().meet(other.vf()), self.d) 
    }

    // Safety: F has the same layout as f32 (but exposes a different set of traits)
    pub fn vf(&self) -> &PathMap<FAddMul> { 
        unsafe { (&self.m as *const PathMap<f32> as *const PathMap<FAddMul>).as_ref().unwrap_unchecked() } 
    }

    // pub fn vf_mut(&mut self) -> &mut PathMap<FAddMul> { 
    //     unsafe { (&mut self.m as *mut PathMap<f32> as *mut PathMap<FAddMul>).as_mut().unwrap_unchecked() } 
    // }

    fn vf32(m: PathMap<FAddMul>, d: usize) -> Self { 
        unsafe { Self { m: std::mem::transmute::<PathMap::<FAddMul>, PathMap::<f32>>(m), d, p: Vec::new() } } 
    }

    pub fn new(dimensions: usize) -> Self { 
        Self { m: PathMap::new(), d: dimensions, p: Vec::new() } 
    }

    pub fn estimate_memory_usage(&self) -> usize {
        0 // TODO: restore when PathMap supports estimate_memory_usage
    }

    fn index_to_path(&self, ix: &[usize]) -> Vec<u8> {
        let mut p = Vec::new();
        let len = indices_to_bob(ix, &mut vec![]);
        p.extend(std::iter::repeat_n(0u8, 64 - len));
        indices_to_bob(ix, &mut p);
        p
    }
    
    pub fn get(&self, ix: &[usize]) -> Option<f32> {
        let path = self.index_to_path(ix);
        self.m.get(&path[..]).copied()
    }
    
    pub fn remove(&mut self, ix: &[usize]) -> Option<f32> {
        let path = self.index_to_path(ix);
        self.m.remove(&path[..])
    }
    pub fn attention_dry(&self, k: &Self, out: &mut Self) -> usize {
        bob_attention::<true>(&mut self.m.read_zipper(), &mut k.m.read_zipper(), &mut out.m.write_zipper(), 0)
    }
}

impl SparseMatrix for SparseTensorFBOB {
    type Value = f32;
    
    fn new(dimensions: usize) -> Self { 
        Self { m: PathMap::new(), d: dimensions, p: Vec::new() } 
    }

    fn ndim(&self) -> usize { self.d }
    
    fn get(&self, ix: &[usize]) -> Option<f32> { 
        SparseTensorFBOB::get(self, ix) 
    }

    fn set(&mut self, ix: &[usize], v: f32) { 
        SparseTensorFBOB::set(self, ix, v) 
    }

    fn remove(&mut self, ix: &[usize]) -> Option<f32> { 
        SparseTensorFBOB::remove(self, ix) 
    }

    fn nnz(&self) -> usize { 
        self.m.val_count() 
    }

    fn add(&self, other: &Self) -> Self { 
        SparseTensorFBOB::add(self, other) 
    }

    fn mul(&self, other: &Self) -> Self { 
        SparseTensorFBOB::mul(self, other) 
    }

    /// Relative difference between two matrices, defined as the max relative error over all entries
    fn rel_diff(&self, _other: &Self) -> f32 {
        todo!()
    }
}

// [b0,b1,h0,h1,k0,k1,d0,d1]  <- this is the best case if everything is in the right order
// hqdb,hkdb->bhqk
#[allow(non_snake_case)]
fn bob_attention<const DRY_RUN: bool>(Q: &mut ReadZipperUntracked<f32>, K: &mut ReadZipperUntracked<f32>, out: &mut WriteZipperUntracked<f32>, depth: usize) -> usize {
    let QF = 0b00001011u8; let QB = 0b00000111u8;
    let KF = 0b00001011u8; let KB = 0b00000100u8;
    let qm = Q.child_mask();
    let km = K.child_mask();
    let mut count = 0;
    for i in qm.iter() {
        // GFNI
        let rkm: ByteMask = km; // k_must_on | k_must_off;
        let Q_proj_out: u8 = QB & i; // permute (hardcoded)
        for j in rkm.iter() {
            let K_proj_out: u8 = (KB & j) << 1; // permute (hardcoded)
            let out_b: u8 = Q_proj_out | K_proj_out;
            if QF & i != KF & j { continue }

            Q.descend_to_byte(i);
            K.descend_to_byte(j);
            if !DRY_RUN { out.descend_to_byte(out_b); }
            if depth == 63 {
                let total = out.get_val_or_set_mut(0f32);
                *total += unsafe { *Q.val().unwrap_unchecked() * *K.val().unwrap_unchecked() };
                count += 1;
                // unsafe { COUNT += 1; }
            } else {
                count += bob_attention::<DRY_RUN>(Q, K, out, depth + 1);
            }
            Q.ascend_byte();
            K.ascend_byte();
            if !DRY_RUN { out.ascend_byte(); }
        }
    }
    count
}

impl Attention for SparseTensorFBOB {
    /// Q[10,11,00,00]
    /// path = [00001100, 00000100]
    /// bhqd,bhkd->bhqk  ((to be) auto generated, do not touch)
    fn attention(&self, k: &Self, out: &mut Self) -> usize {
        bob_attention::<false>(&mut self.m.read_zipper(), &mut k.m.read_zipper(), &mut out.m.write_zipper(), 0)
    }
}
// ============================================================================
// SparseTensorFWeave - Bit-interleaved (Z-order/Morton) encoding
// ============================================================================

pub struct SparseTensorFWeave {
    pub m: PathMap<f32>,
    pub d: usize,
    p: Vec<u8>
}

impl SparseTensorFWeave {
    pub fn set(&mut self, ix: &[usize], v: f32) {
        self.p.clear();
        indices_to_weave::<2, u16>(ix, &mut self.p);
        assert_eq!(self.p.len(), 8);
        self.m.insert(&self.p[..], v);
    }
    pub fn add(&self, other: &Self) -> Self { 
        Self::vf32(self.vf().join(other.vf()), self.d) 
    }

    pub fn mul(&self, other: &Self) -> Self { 
        Self::vf32(self.vf().meet(other.vf()), self.d) 
    }

    // Safety: F has the same layout as f32 (but exposes a different set of traits)
    pub fn vf(&self) -> &PathMap<FAddMul> { 
        unsafe { (&self.m as *const PathMap<f32> as *const PathMap<FAddMul>).as_ref().unwrap_unchecked() } 
    }

    // pub fn vf_mut(&mut self) -> &mut PathMap<FAddMul> { 
    //     unsafe { (&mut self.m as *mut PathMap<f32> as *mut PathMap<FAddMul>).as_mut().unwrap_unchecked() } 
    // }

    fn vf32(m: PathMap<FAddMul>, d: usize) -> Self { 
        unsafe { Self { m: std::mem::transmute::<PathMap::<FAddMul>, PathMap::<f32>>(m), d, p: Vec::new() } } 
    }

    pub fn new(dimensions: usize) -> Self { 
        Self { m: PathMap::new(), d: dimensions, p: Vec::new() } 
    }
    pub fn estimate_memory_usage(&self) -> usize {
        0 // TODO: restore when PathMap supports estimate_memory_usage
    }

    fn index_to_path(&self, ix: &[usize]) -> Vec<u8> {
        let mut p = Vec::new();
        indices_to_weave::<2, u16>(ix, &mut p);
        p
    }
    
    pub fn get(&self, ix: &[usize]) -> Option<f32> {
        let path = self.index_to_path(ix);
        self.m.get(&path[..]).copied()
    }
    
    pub fn remove(&mut self, ix: &[usize]) -> Option<f32> {
        let path = self.index_to_path(ix);
        self.m.remove(&path[..])
    }

}

impl SparseMatrix for SparseTensorFWeave {
    type Value = f32;
    
    fn new(dimensions: usize) -> Self { 
        Self { m: PathMap::new(), d: dimensions, p: Vec::new() } 
    }

    fn ndim(&self) -> usize { self.d }
    
    fn get(&self, ix: &[usize]) -> Option<f32> { 
        SparseTensorFWeave::get(self, ix) 
    }

    fn set(&mut self, ix: &[usize], v: f32) { 
        SparseTensorFWeave::set(self, ix, v) 
    }

    fn remove(&mut self, ix: &[usize]) -> Option<f32> { 
        SparseTensorFWeave::remove(self, ix) 
    }

    fn nnz(&self) -> usize { 
        self.m.val_count() 
    }

    fn add(&self, other: &Self) -> Self { 
        SparseTensorFWeave::add(self, other) 
    }

    fn mul(&self, other: &Self) -> Self { 
        SparseTensorFWeave::mul(self, other) 
    }
    
    /// Relative difference between two matrices, defined as the max relative error over all entries
    fn rel_diff(&self, _other: &Self) -> f32 {
        todo!()
    }
}

/*
#[allow(unused)]
#[allow(non_snake_case)]
fn weave_attention(Q: &mut ReadZipperUntracked<f32>, K: &mut ReadZipperUntracked<f32>, out: &mut WriteZipperUntracked<f32>) -> usize {
    let mut count = 0;
    query_byte_weave_attention(Q, K, out, 2, |Q, K, out| {
        let mut acc = 0f32;
        let acc_ptr = &mut acc as *mut f32;
        aggregate_byte_weave_attention(Q, K, 2, |Q, K| unsafe {
            *acc_ptr +=  *Q.val().unwrap() * *K.val().unwrap();
            count += 1;
        });
        out.set_val(acc);
    });
    count
}

#[allow(unused)]
#[allow(non_snake_case)]
fn query_byte_weave_attention<F : FnMut(&mut ReadZipperUntracked<f32>, &mut ReadZipperUntracked<f32>, &mut WriteZipperUntracked<f32>) + Copy>(
    Q: &mut ReadZipperUntracked<f32>, K: &mut ReadZipperUntracked<f32>, out: &mut WriteZipperUntracked<f32>, depth: usize, mut cb: F) {
    let bm = Q.child_mask().and(&K.child_mask());
    for b in bm.iter() {
        Q.descend_to_byte(b);
        K.descend_to_byte(b);
        out.descend_to_byte(b);
        let hm = Q.child_mask().and(&K.child_mask());
        for h in hm.iter() {
            Q.descend_to_byte(h);
            K.descend_to_byte(h);
            out.descend_to_byte(h);
            let qm = Q.child_mask();
            for q in qm.iter() {
                Q.descend_to_byte(q);
                out.descend_to_byte(q);
                let km = K.child_mask();
                for k in km.iter() {
                    K.descend_to_byte(k);
                    out.descend_to_byte(k);

                    if depth == 1 { cb(Q, K, out); }
                    else { query_byte_weave_attention(Q, K, out, depth - 1, cb); }

                    K.ascend_byte();
                    out.ascend_byte();
                }
                Q.ascend_byte();
                out.ascend_byte();
            }
            Q.ascend_byte();
            K.ascend_byte();
            out.ascend_byte();
        }
        Q.ascend_byte();
        K.ascend_byte();
        out.ascend_byte();
    }
}

#[allow(unused)]
#[allow(non_snake_case)]
fn aggregate_byte_weave_attention<F : FnMut(&mut ReadZipperUntracked<f32>, &mut ReadZipperUntracked<f32>) + Copy>(
    Q: &mut ReadZipperUntracked<f32>, K: &mut ReadZipperUntracked<f32>, depth: usize, mut cb: F) {
    let dm = Q.child_mask().and(&K.child_mask());
    for d in dm.iter() {
        Q.descend_to_byte(d);
        K.descend_to_byte(d);

        if depth == 1 { cb(Q, K) }
        else { aggregate_byte_weave_attention(Q, K, depth - 1, cb) }

        Q.ascend_byte();
        K.ascend_byte();
    }
}
*/

/*

// [b,h,q,d]
/// bhqd,bhkd->bhqk  (auto generated, do not touch)
#[allow(non_snake_case, unused)]
fn _byte_weave_attention(Q: &mut ReadZipperUntracked<f32>, K: &mut ReadZipperUntracked<f32>, out: &mut WriteZipperUntracked<f32>) {
    let bm = Q.child_mask().and(&K.child_mask());
    for b in bm.iter() {
        Q.descend_to_byte(b);
        K.descend_to_byte(b);
        out.descend_to_byte(b);
        let hm = Q.child_mask().and(&K.child_mask());
        for h in hm.iter() {
            Q.descend_to_byte(h);
            K.descend_to_byte(h);
            out.descend_to_byte(h);
            let qm = Q.child_mask();
            for q in qm.iter() {
                Q.descend_to_byte(q);
                out.descend_to_byte(q);
                let km = K.child_mask();
                for k in km.iter() {
                    K.descend_to_byte(k);
                    out.descend_to_byte(k);
                    let mut acc = 0f32;
                    let mut ran = 0;
                    let dm = Q.child_mask().and(&K.child_mask());
                    for d in dm.iter() {
                        Q.descend_to_byte(d);
                        K.descend_to_byte(d);
                        acc += unsafe { *Q.val().unwrap() * *K.val().unwrap() };
                        unsafe { COUNT += 1 };
                        ran += 1;
                        Q.ascend_byte();
                        K.ascend_byte();
                    }
                    out.set_val(acc);
                    K.ascend_byte();
                    out.ascend_byte();
                }
                Q.ascend_byte();
                out.ascend_byte();
            }
            Q.ascend_byte();
            K.ascend_byte();
            out.ascend_byte();
        }
        Q.ascend_byte();
        K.ascend_byte();
        out.ascend_byte();
    }
}
*/

// [b0,h0,k0,d0,b1,h1,k1,d1]
#[allow(non_snake_case, unused)]
fn short_weave_attention<const DRY_RUN: bool>(Q: &mut ReadZipperUntracked<f32>, K: &mut ReadZipperUntracked<f32>, out: &mut WriteZipperUntracked<f32>) -> usize {
    let bm = Q.child_mask().and(&K.child_mask());
    let mut count = 0;
    for b in bm.iter() {
        Q.descend_to_byte(b);
        K.descend_to_byte(b);
        if !DRY_RUN { out.descend_to_byte(b); }
        let hm = Q.child_mask().and(&K.child_mask());
        for h in hm.iter() {
            Q.descend_to_byte(h);
            K.descend_to_byte(h);
            if !DRY_RUN { out.descend_to_byte(h); }
            let qm = Q.child_mask();
            for q in qm.iter() {
                Q.descend_to_byte(q);
                if !DRY_RUN { out.descend_to_byte(q); }
                let km = K.child_mask();
                for k in km.iter() {
                    K.descend_to_byte(k);
                    if !DRY_RUN { out.descend_to_byte(k); }


                    let dm = Q.child_mask().and(&K.child_mask());
                    for d in dm.iter() {
                        Q.descend_to_byte(d);
                        K.descend_to_byte(d);

                    let bm = Q.child_mask().and(&K.child_mask());
                    for b in bm.iter() {
                        Q.descend_to_byte(b);
                        K.descend_to_byte(b);
                        if !DRY_RUN { out.descend_to_byte(b); }
                        let hm = Q.child_mask().and(&K.child_mask());
                        for h in hm.iter() {
                            Q.descend_to_byte(h);
                            K.descend_to_byte(h);
                            if !DRY_RUN { out.descend_to_byte(h); }
                            let qm = Q.child_mask();
                            for q in qm.iter() {
                                Q.descend_to_byte(q);
                                if !DRY_RUN { out.descend_to_byte(q); }
                                let km = K.child_mask();
                                for k in km.iter() {
                                    K.descend_to_byte(k);
                                    if !DRY_RUN { out.descend_to_byte(k); }


                                        let mut acc = 0f32;


                                        let dm = Q.child_mask().and(&K.child_mask());
                                        for d in dm.iter() {
                                            Q.descend_to_byte(d);
                                            K.descend_to_byte(d);

                                            acc += unsafe { *Q.val().unwrap() * *K.val().unwrap() };
                                            // unsafe { COUNT += 1 };
                                            count += 1;

                                            Q.ascend_byte();
                                            K.ascend_byte();
                                        }

                                    if !DRY_RUN {
                                        let total = out.get_val_or_set_mut(0f32);
                                        *total += acc;
                                    }


                                    K.ascend_byte();
                                    if !DRY_RUN { out.ascend_byte(); }
                                }
                                Q.ascend_byte();
                                if !DRY_RUN { out.ascend_byte(); }
                            }
                            Q.ascend_byte();
                            K.ascend_byte();
                            if !DRY_RUN { out.ascend_byte(); }
                        }
                        Q.ascend_byte();
                        K.ascend_byte();
                        if !DRY_RUN { out.ascend_byte(); }
                    }


                        Q.ascend_byte();
                        K.ascend_byte();
                    }


                    K.ascend_byte();
                    if !DRY_RUN { out.ascend_byte(); }
                }
                Q.ascend_byte();
                if !DRY_RUN { out.ascend_byte(); }
            }
            Q.ascend_byte();
            K.ascend_byte();
            if !DRY_RUN { out.ascend_byte(); }
        }
        Q.ascend_byte();
        K.ascend_byte();
        if !DRY_RUN { out.ascend_byte(); }
    }
    count
}


impl Attention for SparseTensorFWeave {
    /// bhqd,bhkd->bhqk  ((to be) auto generated, do not touch)
    fn attention(&self, k: &Self, out: &mut Self) -> usize {
        short_weave_attention::<false>(&mut self.m.read_zipper(), &mut k.m.read_zipper(), &mut out.m.write_zipper())
    }
}

macro_rules! impl_from_rng_for_sparse {
    ($t:ty) => {
impl FromRng for $t {
    fn with_density(rng: &mut impl rand::Rng, dimensions: &[usize], mut density: f32) -> Self {
        density = density.clamp(0.0, 1.0);
        let n = dimensions.iter().product();
        let mut t = Self::new(dimensions.len());
        let mut to_fill = (density * (n as f32)) as usize;
        let mut dim = vec![0usize; dimensions.len()];
        if density < 0.5 {
            // If density is low, it's more efficient to randomly sample indices to fill.
            while to_fill > 0 {
                let i = rng.random_range(0..n);

                // convert linear index i to multi-dimensional index dim
                let mut rem = i;
                for (d, &s) in dim.iter_mut().zip(dimensions.iter()) {
                    *d = rem % s;
                    rem /= s;
                }
                if t.get(&dim).is_some() {
                    continue;
                }
                t.set(&dim, rng.random());
                to_fill -= 1;
            }
        } else {
            // If density is high, it's more efficient to visit all indices and randomly skip some
            let mut visit_idx = (0..n).collect::<Vec<usize>>();
            while to_fill > 0 {
                let ii = rng.random_range(0..visit_idx.len());
                let i = visit_idx.swap_remove(ii);

                // convert linear index i to multi-dimensional index dim
                let mut rem = i;
                for (d, &s) in dim.iter_mut().zip(dimensions.iter()) {
                    *d = rem % s;
                    rem /= s;
                }
                t.set(&dim, rng.random());
                to_fill -= 1;
            }
        }
        t
    }
}

    };
}

impl_from_rng_for_sparse!(SparseTensorFBOB);
impl_from_rng_for_sparse!(SparseTensorFWeave);
