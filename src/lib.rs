#![cfg_attr(not(feature = "std"), no_std)]
// Enabling SIMD feature means including portable_simd and AVX512(BW), this enables the mask register access for smaller sizes
#![cfg_attr(feature = "simd", feature(portable_simd, avx512_target_feature))]
// Unfortunately, we only need a subset of generic_const_exprs which the minimal implementation should suffice
#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

pub mod concise_bitvec;
pub mod mask;
pub(crate) mod multiversion;
pub mod signature;
pub mod utils;

pub use mask::Mask;
pub use signature::Signature;
