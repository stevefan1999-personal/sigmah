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
    [(); N.div_ceil(u64::LANES)]:,
    [(); N.div_ceil(u32::LANES)]:,
    [(); N.div_ceil(u16::LANES)]:,
    [(); N.div_ceil(u8::LANES)]:,
{
    #[inline(always)]
    pub fn match_chunk<T: SimdBits>(
        &self,
        chunk: &[u8; N],
        f: impl Fn(&[u8; T::LANES], &[u8; T::LANES], u64) -> Result<(), usize> + Sync,
    ) -> bool
    where
        [(); N.div_ceil(T::LANES)]:,
        LaneCount<{ T::LANES }>: SupportedLaneCount,
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

    #[inline]
    pub fn scan<'a, T: SimdBits>(&self, mut haystack: &'a [u8]) -> Option<&'a [u8]>
    where
        [(); N.div_ceil(T::LANES)]:,
        LaneCount<{ T::LANES }>: SupportedLaneCount,
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

            // Since we are using a sliding window approach, we are safe to determine that we can either:
            //   1. Skip to the first position of c for all c in window[1..] where c == window[0]
            //   2. Skip this entire window
            // The optimization is derived from the Z-Algorithm which constructs an array Z,
            // where Z[i] represents the length of the longest substring starting from i which is also a prefix of the string.
            // More formally, given first Z[0] is tombstone, then for i in 1..N:
            //   Z[i] is the length of the longest substring starting at i that matches the prefix of S (i.e. memcmp(S[0:], S[i:])).
            // Then we further simplify that to find the first position where Z[i] != 0, it to use the fact that if Z[i] > 0, it has to be a prefix of our pattern,
            // so it is a potential search point. If all that is in the Z box are 0, then we are safe to assume all patterns are unique and need one-by-one brute force.
            // Technically speaking, if we repeat this process to each shift of the window with respect to its mask position, we can obtain the Z-box algorithm as well.
            // It is speculated that we can redefine the special window[0] prefix to a value of "w" and index "i" for any c for all i, c in window[1..] where i == first(for i, m in mask[1..] where m == true),
            // and then do skip to the "i"th position of c for all c in window[1..] where c == w. For now I'm too lazy to investigate whether the proof is correct.
            //
            // If in SIMD manner, we can first take the first character, splat it to vector width and match it with the haystack window after first element,
            // then do find-first-set and add 1 to cover for the real next position. It is always assumed the scanner will always go at least 1 byte ahead
            let move_position = if unsafe { self.mask.get_unchecked(0) } && !haystack_smaller_than_n
            {
                self.equal_then_find_second_position(unsafe { self.get_unchecked(0) }, window)
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

    #[inline(always)]
    pub(crate) fn equal_then_find_second_position(
        &self,
        first: u8,
        window: &[u8; N],
    ) -> Option<usize> {
        if N >= 64 {
            equal_then_find_second_position_simd::<u64, N>(first, window)
        } else if N >= 32 {
            equal_then_find_second_position_simd::<u32, N>(first, window)
        } else if N >= 16 {
            equal_then_find_second_position_simd::<u16, N>(first, window)
        } else {
            equal_then_find_second_position_simd::<u8, N>(first, window)
        }
    }
}
