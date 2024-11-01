use crate::pad_zeroes_slice_unchecked;
use bitvec::prelude::*;
use core::ops::{BitOr, Shl};
use num_traits::{One, Zero};
pub trait Bits {
    const BITS: usize;
}

impl Bits for u8 {
    const BITS: usize = u8::BITS as usize;
}

impl Bits for u16 {
    const BITS: usize = u16::BITS as usize;
}

impl Bits for u32 {
    const BITS: usize = u32::BITS as usize;
}

impl Bits for u64 {
    const BITS: usize = u64::BITS as usize;
}

impl Bits for usize {
    const BITS: usize = usize::BITS as usize;
}

#[inline(always)]
pub fn iterate_haystack_pattern_mask_aligned_simd<'a, T, const N: usize>(
    chunk: &'a [u8; N],
    pattern: &'a [u8; N],
    mask: &'a BitSlice<u8>,
) -> impl Iterator<Item = ([u8; T::BITS], [u8; T::BITS], T)> + 'a
where
    T: Bits + One + Zero + Shl<usize, Output = T> + BitOr<Output = T>,
{
    let bits = T::BITS;

    let haystack_chunks_aligned = chunk
        .chunks(bits)
        .map(|x| unsafe { pad_zeroes_slice_unchecked(x) });

    let pattern_chunks_aligned = pattern
        .chunks(bits)
        .map(|x| unsafe { pad_zeroes_slice_unchecked(x) });

    haystack_chunks_aligned
        .zip(pattern_chunks_aligned)
        .zip(mask.chunks(bits))
        .map(|((haystack, pattern), mask)| {
            (
                haystack,
                pattern,
                mask.iter_ones()
                    .fold(T::zero(), |acc, x| acc | (T::one() << x)),
            )
        })
}
