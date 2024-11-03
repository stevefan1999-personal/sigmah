use crate::{
    concise_bitvec::ConciseBitArray,
    mask::Mask,
    multiversion::{equal_then_find_second_position_naive_const, match_naive_const},
    utils::{const_get_unchecked, const_set_unchecked, pad_zeroes_slice_unchecked, simd::SimdBits},
};
use core::mem::{transmute_copy, MaybeUninit};
use derive_more::derive::{From, Into};

#[derive(Debug, Copy, Clone, From, Into)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[repr(C, align(1))]
pub struct Signature<const N: usize>
where
    [(); N.div_ceil(u8::BITS as usize)]:,
{
    #[cfg_attr(feature = "serde", serde(with = "serde_big_array::BigArray"))]
    pub pattern: [u8; N],
    pub mask: Mask<N>,
}

impl<const N: usize> Signature<N>
where
    [(); N.div_ceil(u8::BITS as usize)]:,
{
    #[inline(always)]
    pub const fn get(&self, idx: usize) -> Option<u8> {
        if idx < N {
            Some(unsafe { self.get_unchecked(idx) })
        } else {
            None
        }
    }

    #[inline(always)]
    pub const unsafe fn get_unchecked(&self, idx: usize) -> u8 {
        const_get_unchecked(&self.pattern, idx)
    }

    #[inline(always)]
    pub const fn longest_prefix_suffix_array(&self) -> [usize; N] {
        let mask = self.mask.to_bool_array();
        let mut lps: [usize; N] = [0; N];

        // Length of the previous longest prefix suffix
        let mut len = 0;
        let mut i = 1;
        while i < N {
            // const violation: unnecessary bound check
            let (new_lps, new_len, advance) = {
                let pat = unsafe {
                    match (
                        const_get_unchecked(&mask, len),
                        const_get_unchecked(&mask, i),
                    ) {
                        (false, false) => true,
                        (false, true) => true,
                        (true, false) => true,
                        (true, true) => self.get_unchecked(i) == self.get_unchecked(len),
                    }
                };

                if pat {
                    let forward = Some(len + 1);
                    (forward, forward, true)
                } else if len != 0 {
                    (
                        None,
                        Some(unsafe { const_get_unchecked(&lps, len - 1) }),
                        false,
                    )
                } else {
                    (Some(0), None, true)
                }
            };

            if let Some(new_lps) = new_lps {
                unsafe {
                    const_set_unchecked(&mut lps, i, new_lps);
                }
            }
            if let Some(new_len) = new_len {
                len = new_len;
            }
            i += advance as usize;
        }

        lps
    }

    #[inline(always)]
    pub const fn from_pattern_mask(pattern: &[u8; N], mask: &[u8; N]) -> Self {
        Self {
            pattern: *pattern,
            mask: Mask::from_byte_slice_or_panic(mask),
        }
    }

    // Notice we cannot use From<([u8; N], [u8; N])> because it will break const guarantee
    #[inline(always)]
    pub const fn from_pattern_mask_tuple((pattern, mask): (&[u8; N], &[u8; N])) -> Self {
        Self::from_pattern_mask(pattern, mask)
    }

    #[inline(always)]
    pub const fn from_option_array(needle: [Option<u8>; N]) -> Self {
        Self::from_option_slice(&needle)
    }

    #[inline(always)]
    pub const fn from_option_slice(needle: &[Option<u8>; N]) -> Self {
        unsafe { Self::from_option_slice_unchecked(needle) }
    }

    #[inline(always)]
    pub const fn from_array_with_exact_match_mask(pattern: [u8; N]) -> Self {
        Self {
            pattern,
            mask: Mask::MAX,
        }
    }

    #[inline(always)]
    pub const fn from_slice_with_exact_match_mask(pattern: &[u8; N]) -> Self {
        Self::from_array_with_exact_match_mask(*pattern)
    }

    #[inline(always)]
    pub const fn from_option_slice_zero_padded_if_not_aligned(needle: &[Option<u8>]) -> Self {
        Self::from_option_slice_inner(needle, [0; N])
    }

    #[inline(always)]
    pub const unsafe fn from_option_slice_unchecked(needle: &[Option<u8>]) -> Self {
        Self::from_option_slice_inner(needle, transmute_copy(&[MaybeUninit::<u8>::uninit(); N]))
    }

    #[inline(always)]
    const fn from_option_slice_inner(needle: &[Option<u8>], mut pattern: [u8; N]) -> Self {
        let mut mask: [bool; N] = [false; N];
        let mut i = 0;
        while i < needle.len() {
            if let Some(x) = unsafe { *needle.as_ptr().add(i) } {
                unsafe {
                    const_set_unchecked(&mut pattern, i, x);
                    const_set_unchecked(&mut mask, i, true);
                }
            }
            i += 1;
        }
        Self {
            pattern,
            mask: Mask::from_bit_array(ConciseBitArray::from_bool_array(mask)),
        }
    }

    #[inline(always)]
    pub fn naive(self) -> naive::SignatureWithNaive<N> {
        self.into()
    }

    #[inline(always)]
    #[cfg(feature = "simd")]
    pub fn simd(self) -> simd::SignatureWithSimd<N> {
        self.into()
    }

    #[inline(always)]
    #[cfg(feature = "rayon")]
    pub fn rayon_naive(self) -> rayon::SignatureWithRayonNaive<N> {
        self.into()
    }

    #[inline(always)]
    #[cfg(all(feature = "rayon", feature = "simd"))]
    pub fn rayon_simd(self) -> rayon_simd::SignatureWithRayonAndSimd<N> {
        self.into()
    }
}

impl<const N: usize> Signature<N>
where
    [(); N.div_ceil(u8::BITS as usize)]:,
    [(); N.div_ceil(u64::LANES)]:,
    [(); N.div_ceil(u32::LANES)]:,
    [(); N.div_ceil(u16::LANES)]:,
    [(); N.div_ceil(u8::LANES)]:,
{
    pub fn scan<'a>(&self, haystack: &'a [u8]) -> Option<&'a [u8]> {
        self.scan_inner(haystack, |chunk| self.match_best_effort(chunk))
    }

    pub fn scan_naive<'a>(&self, haystack: &'a [u8]) -> Option<&'a [u8]> {
        self.scan_naive_const(haystack)
    }

    // scan_inner is the main function (due to impl Fn being a trait and not const friendly) and please synchronize in time
    pub const fn scan_naive_const<'a>(&self, mut haystack: &'a [u8]) -> Option<&'a [u8]> {
        let exact_match = self.mask.0.is_exact();
        while !haystack.is_empty() {
            let haystack_smaller_than_n = haystack.len() < N;

            let window: &[u8; N] = unsafe {
                if haystack_smaller_than_n {
                    &pad_zeroes_slice_unchecked::<N>(haystack)
                } else {
                    transmute_copy(&haystack)
                }
            };

            if match_naive_const(window, &self.pattern, &self.mask.0).is_ok() {
                return Some(haystack);
            } else if exact_match && haystack_smaller_than_n {
                // If we are having the mask to match for all, and the chunk is actually smaller than N, we are cooked anyway
                return None;
            }

            let move_position =
                if unsafe { self.mask.0.get_unchecked(0) } && !haystack_smaller_than_n {
                    if let Some(pos) = equal_then_find_second_position_naive_const(
                        unsafe { self.get_unchecked(0) },
                        window,
                    ) {
                        pos
                    } else {
                        N
                    }
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

    #[inline]
    fn scan_inner<'a>(
        &self,
        mut haystack: &'a [u8],
        f: impl Fn(&[u8; N]) -> bool,
    ) -> Option<&'a [u8]> {
        let exact_match = self.mask.0.is_exact();
        while !haystack.is_empty() {
            let haystack_smaller_than_n = haystack.len() < N;

            let window: &[u8; N] = unsafe {
                if haystack_smaller_than_n {
                    &pad_zeroes_slice_unchecked::<N>(haystack)
                } else {
                    transmute_copy(&haystack)
                }
            };

            if f(window) {
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
            let move_position =
                if unsafe { self.mask.0.get_unchecked(0) } && !haystack_smaller_than_n {
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
}

impl<const N: usize> Signature<N>
where
    [(); N.div_ceil(u8::BITS as usize)]:,
    [(); N.div_ceil(u64::LANES)]:,
    [(); N.div_ceil(u32::LANES)]:,
    [(); N.div_ceil(u16::LANES)]:,
    [(); N.div_ceil(u8::LANES)]:,
{
    #[inline(always)]
    pub fn match_best_effort(&self, chunk: &[u8; N]) -> bool {
        #[cfg(feature = "simd")]
        {
            if N >= 64 {
                self.simd().match_chunk::<u64>(chunk)
            } else if N >= 32 {
                self.simd().match_chunk::<u32>(chunk)
            } else if N >= 16 {
                self.simd().match_chunk::<u16>(chunk)
            } else if N >= 8 {
                self.simd().match_chunk::<u8>(chunk)
            } else {
                // for the lulz
                self.match_naive(chunk)
            }
        }

        #[cfg(not(feature = "simd"))]
        {
            self.match_naive(chunk)
        }
    }

    #[inline(always)]
    pub fn match_naive(&self, chunk: &[u8; N]) -> bool {
        match_naive_const(chunk, &self.pattern, &self.mask.0).is_ok()
    }
}

impl<const N: usize> Signature<N>
where
    [(); N.div_ceil(u8::BITS as usize)]:,
    [(); N.div_ceil(u64::LANES)]:,
    [(); N.div_ceil(u32::LANES)]:,
    [(); N.div_ceil(u16::LANES)]:,
    [(); N.div_ceil(u8::LANES)]:,
{
    #[inline(always)]
    fn equal_then_find_second_position(&self, first: u8, window: &[u8; N]) -> Option<usize> {
        #[cfg(feature = "simd")]
        {
            self.simd().equal_then_find_second_position(first, window)
        }

        #[cfg(not(feature = "simd"))]
        {
            self.equal_then_find_second_position_naive(first, window)
        }
    }
}

#[cfg(feature = "simd")]
mod simd;

#[cfg(feature = "rayon")]
mod rayon;

#[cfg(all(feature = "rayon", feature = "simd"))]
mod rayon_simd;

mod naive;
