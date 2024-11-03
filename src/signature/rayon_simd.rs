use super::Signature;
use crate::{
    multiversion::simd::{equal_then_find_second_position_simd, match_simd_core},
    utils::{pad_zeroes_slice_unchecked, simd::SimdBits},
};
use arrayvec::ArrayVec;
use core::{
    mem::transmute_copy,
    simd::{LaneCount, SupportedLaneCount},
};
use derive_more::derive::{Deref, From, Into};
use rayon::{
    iter::{IndexedParallelIterator, IntoParallelIterator},
    prelude::*,
};

#[derive(Debug, Copy, Clone, From, Into, Deref)]
#[repr(C, align(1))]
pub struct SignatureWithRayonAndSimd<const N: usize>(pub Signature<N>)
where
    [(); N.div_ceil(u8::BITS as usize)]:;

impl<const N: usize> SignatureWithRayonAndSimd<N>
where
    [(); N.div_ceil(u8::BITS as usize)]:,
{
    #[inline(always)]
    fn match_chunk<T: SimdBits>(
        &self,
        chunk: &[u8; N],
        f: impl Fn(&[u8; T::LANES], &[u8; T::LANES], u64) -> Result<(), usize> + Sync,
    ) -> bool
    where
        [(); N.div_ceil(T::LANES)]:,
    {
        let chunks: ArrayVec<&[u8], { N.div_ceil(T::LANES) }> = chunk.chunks(T::LANES).collect();
        let patterns: ArrayVec<&[u8], { N.div_ceil(T::LANES) }> =
            self.pattern.chunks(T::LANES).collect();
        let masks = self
            .mask
            .chunks(T::LANES)
            .map(|mask| mask.iter_ones().fold(T::ZERO, |acc, x| acc | (T::ONE << x)))
            .collect::<ArrayVec<T, { N.div_ceil(T::LANES) }>>();

        chunks
            .into_par_iter()
            .zip(patterns.into_par_iter())
            .zip(masks.into_par_iter())
            .all(|((&haystack, &pattern), mask)| {
                let haystack: &[u8; T::LANES] = unsafe {
                    if haystack.len() < T::LANES {
                        &pad_zeroes_slice_unchecked::<{ T::LANES }>(haystack)
                    } else {
                        transmute_copy(&haystack)
                    }
                };

                let pattern: &[u8; T::LANES] = unsafe {
                    if pattern.len() < T::LANES {
                        &pad_zeroes_slice_unchecked::<{ T::LANES }>(pattern)
                    } else {
                        transmute_copy(&pattern)
                    }
                };
                f(haystack, pattern, mask.to_u64()).is_ok()
            })
    }

    pub fn scan<'a, T: SimdBits>(&self, mut haystack: &'a [u8]) -> Option<&'a [u8]>
    where
        LaneCount<{ T::LANES }>: SupportedLaneCount,
        [(); N.div_ceil(T::LANES)]:,
    {
        let exact_match = self.mask.is_exact();
        while !haystack.is_empty() {
            let haystack_smaller_than_n = haystack.len() < N;

            let window: &[u8; N] = unsafe {
                if haystack_smaller_than_n {
                    &pad_zeroes_slice_unchecked::<N>(haystack)
                } else {
                    transmute_copy(&haystack)
                }
            };

            if self.match_chunk(window, match_simd_core) {
                return Some(haystack);
            } else if exact_match && haystack_smaller_than_n {
                // If we are having the mask to match for all, and the chunk is actually smaller than N, we are cooked anyway
                return None;
            }

            let move_position = if unsafe { self.mask.get_unchecked(0) } && !haystack_smaller_than_n
            {
                equal_then_find_second_position_simd(unsafe { self.get_unchecked(0) }, window)
                    .unwrap_or(N)
            } else {
                1
            };
            haystack = unsafe {
                core::slice::from_raw_parts(
                    haystack.as_ptr().add(move_position),
                    haystack.len() - move_position,
                )
            };
        }
        None
    }
}
