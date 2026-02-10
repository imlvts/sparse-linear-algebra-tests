#![allow(unused)]
/// Trait for sparse matrix/tensor operations with arbitrary indexing
pub trait SparseMatrix: Sized {
    type Value: Copy;
    
    /// Create a new sparse matrix with the given number of dimensions
    fn new(dimensions: usize) -> Self;
    
    /// Get the number of dimensions
    fn ndim(&self) -> usize;
    
    /// Get a value at the given index, returning None if not present
    fn get(&self, ix: &[usize]) -> Option<Self::Value>;
    
    /// Set a value at the given index
    fn set(&mut self, ix: &[usize], v: Self::Value);
    
    /// Remove a value at the given index, returning the old value if present
    fn remove(&mut self, ix: &[usize]) -> Option<Self::Value>;
    
    /// Get the number of non-zero values
    fn nnz(&self) -> usize;
    
    /// Element-wise addition (join in lattice terms)
    fn add(&self, other: &Self) -> Self;
    
    /// Element-wise multiplication (meet in lattice terms)  
    fn mul(&self, other: &Self) -> Self;
    
    /// Relative difference between two matrices, defined as the max relative error over all entries
    fn rel_diff(&self, other: &Self) -> Self::Value;
}

/// Trait for attention-like operations on tensors
pub trait Attention {
    /// Compute attention output given query and key tensors, writing into the output tensor
    fn attention(&self, k: &Self, out: &mut Self) -> usize;
}

pub trait FromRng {
    fn with_density(rng: &mut impl rand::Rng, dimensions: &[usize], density: f32) -> Self;
}