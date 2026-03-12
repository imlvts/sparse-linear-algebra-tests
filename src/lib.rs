#![feature(portable_simd)]

extern crate blas_src;

pub mod traits;
pub mod dense;
pub mod sparse;
pub mod graph;
pub mod graph_csr;
pub mod graph_csr_btree;
pub mod graph_sprs;
pub mod graph_magnus;
pub mod dense_btree;
pub mod graph_csr_builder;

pub use traits::*;
pub use dense::*;
pub use sparse::*;
