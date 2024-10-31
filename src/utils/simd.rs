use crate::pad_zeroes_slice_unchecked;
use bitvec::prelude::*;
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
    mask: &'a BitArray<[u8; N.div_ceil(u8::BITS as usize)]>,
) -> impl Iterator<Item = ([u8; T::BITS], [u8; T::BITS], &'a BitSlice<u8>)>
where
    T: Bits,
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
        .map(|((haystack, pattern), mask)| (haystack, pattern, mask))
}
