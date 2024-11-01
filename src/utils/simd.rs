use crate::pad_zeroes_slice_unchecked;
use bitvec::prelude::*;
use core::ops::{BitOr, Shl};

// Unfortunately, where LaneCount<{ Self::LANES }>: SupportedLaneCount does not work, we do the best we can
pub trait SimdBits: Shl<usize, Output = Self> + BitOr<Output = Self> + Sized {
    const LANES: usize;
    const ONE: Self;
    const ZERO: Self;

    fn to_u64(self) -> u64;
}

impl SimdBits for u8 {
    const LANES: usize = u8::BITS as usize;
    const ONE: Self = 1;
    const ZERO: Self = 0;

    fn to_u64(self) -> u64 {
        self as _
    }
}

impl SimdBits for u16 {
    const LANES: usize = u16::BITS as usize;
    const ONE: Self = 1;
    const ZERO: Self = 0;
    fn to_u64(self) -> u64 {
        self as _
    }
}

impl SimdBits for u32 {
    const LANES: usize = u32::BITS as usize;
    const ONE: Self = 1;
    const ZERO: Self = 0;
    fn to_u64(self) -> u64 {
        self as _
    }
}

impl SimdBits for u64 {
    const LANES: usize = u64::BITS as usize;
    const ONE: Self = 1;
    const ZERO: Self = 0;
    fn to_u64(self) -> u64 {
        self as _
    }
}

impl SimdBits for usize {
    const LANES: usize = usize::BITS as usize;
    const ONE: Self = 1;
    const ZERO: Self = 0;
    fn to_u64(self) -> u64 {
        self as _
    }
}

#[inline(always)]
pub fn iterate_haystack_pattern_mask_aligned_simd<'a, T: SimdBits>(
    chunk: &'a [u8],
    pattern: &'a [u8],
    mask: &'a BitSlice<u8>,
) -> impl Iterator<Item = ([u8; T::LANES], [u8; T::LANES], T)> + 'a {
    let lanes = T::LANES;

    let haystack_chunks_aligned = chunk
        .chunks(lanes)
        .map(|x| unsafe { pad_zeroes_slice_unchecked(x) });

    let pattern_chunks_aligned = pattern
        .chunks(lanes)
        .map(|x| unsafe { pad_zeroes_slice_unchecked(x) });

    haystack_chunks_aligned
        .zip(pattern_chunks_aligned)
        .zip(mask.chunks(lanes))
        .map(|((haystack, pattern), mask)| {
            (
                haystack,
                pattern,
                mask.iter_ones().fold(T::ZERO, |acc, x| acc | (T::ONE << x)),
            )
        })
}
