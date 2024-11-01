#![cfg_attr(not(feature = "std"), no_std)]
// Enabling SIMD feature means including portable_simd and AVX512(BW), this enables the mask register access for smaller sizes
#![cfg_attr(feature = "simd", feature(portable_simd, avx512_target_feature))]
// Unfortunately, we only need a subset of generic_const_exprs which the minimal implementation should suffice
#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use crate::multiversion::{equal_then_find_first_position_naive, match_naive_directly};
use bitvec::prelude::*;

#[cfg(feature = "simd")]
use {
    crate::{
        multiversion::simd::{
            equal_then_find_first_position_simd, match_simd_core, match_simd_select_core,
        },
        utils::simd::{iterate_haystack_pattern_mask_aligned_simd, Bits},
    },
    core::{
        ops::{BitOr, Shl},
        simd::{LaneCount, SupportedLaneCount},
    },
    num_traits::{One, Zero},
};

use crate::utils::pad_zeroes_slice_unchecked;

#[derive(Debug, Copy, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[repr(transparent)]
pub struct SignatureMask<const N: usize>(BitArray<[u8; N.div_ceil(u8::BITS as usize)]>)
where
    [(); N.div_ceil(u8::BITS as usize)]:;

impl<const N: usize> SignatureMask<N>
where
    [(); N.div_ceil(u8::BITS as usize)]:,
{
    pub const MAX: Self = Self(BitArray {
        data: [u8::MAX; N.div_ceil(u8::BITS as usize)],
        ..BitArray::ZERO
    });

    pub const fn all(&self) -> bool {
        let mut i = 0;
        while i < N {
            const BITS: usize = u8::BITS as usize;
            let bit = 1 << (i % BITS);
            if (self.0.data[i / BITS] & bit) != bit {
                return false;
            }
            i += 1;
        }
        true
    }
}

impl<const N: usize> SignatureMask<N>
where
    [(); N.div_ceil(u8::BITS as usize)]:,
{
    #[inline(always)]
    pub const fn from_bool_array(pattern: [bool; N]) -> Self {
        Self::from_bool_slice(&pattern)
    }

    #[inline(always)]
    pub const fn from_bool_slice(pattern: &[bool; N]) -> Self {
        Self(Self::from_bool_slice_to_bitarr(pattern))
    }

    #[inline(always)]
    pub const fn from_bool_array_to_bitarr(
        pattern: [bool; N],
    ) -> BitArray<[u8; N.div_ceil(u8::BITS as usize)]> {
        Self::from_bool_slice_to_bitarr(&pattern)
    }

    #[inline(always)]
    pub const fn from_bool_slice_to_bitarr(
        pattern: &[bool; N],
    ) -> BitArray<[u8; N.div_ceil(u8::BITS as usize)]> {
        let mut arr: BitArray<[u8; N.div_ceil(u8::BITS as usize)]> = BitArray::ZERO;
        let mut i = 0;
        while i < pattern.len() {
            if pattern[i] {
                const BITS: usize = u8::BITS as usize;
                arr.data[i / BITS] |= 1 << (i % BITS);
            }
            i += 1;
        }
        arr
    }

    #[inline(always)]
    pub const fn to_bool_array(&self) -> [bool; N] {
        let mut arr: [bool; N] = [false; N];
        let mut i = 0;
        while i < N {
            const BITS: usize = u8::BITS as usize;
            let bit = 1 << (i % BITS);
            arr[i] = (self.0.data[i / BITS] & bit) == bit;
            i += 1;
        }
        arr
    }
}

impl<const N: usize> SignatureMask<N>
where
    [(); N.div_ceil(u8::BITS as usize)]:,
{
    #[inline(always)]
    pub const fn from_byte_array(pattern: [u8; N]) -> Self {
        Self::from_byte_slice(&pattern)
    }

    #[inline(always)]
    pub const fn from_byte_slice(pattern: &[u8; N]) -> Self {
        match Self::try_from_byte_slice_to_bitarr(pattern) {
            Ok(x) => Self(x),
            Err(e) => panic!("{}", e),
        }
    }

    #[inline(always)]
    pub const fn try_from_byte_array_to_bitarr(
        pattern: [u8; N],
    ) -> Result<BitArray<[u8; N.div_ceil(u8::BITS as usize)]>, &'static str> {
        Self::try_from_byte_slice_to_bitarr(&pattern)
    }

    #[inline(always)]
    pub const fn try_from_byte_slice_to_bitarr(
        pattern: &[u8; N],
    ) -> Result<BitArray<[u8; N.div_ceil(u8::BITS as usize)]>, &'static str> {
        let mut pattern_bool: [bool; N] = [false; N];
        let mut i = 0;
        while i < pattern.len() {
            pattern_bool[i] = match pattern[i] {
                b'x' => true,
                b'?' => false,
                _ => return Err("unknown character in pattern"),
            };
            i += 1;
        }
        Ok(Self::from_bool_slice_to_bitarr(&pattern_bool))
    }

    #[inline(always)]
    pub const fn to_byte_array(&self) -> [u8; N] {
        let mut arr: [u8; N] = [b'?'; N];
        let mut i = 0;
        while i < N {
            const BITS: usize = u8::BITS as usize;
            let bit = 1 << (i % BITS);
            arr[i] = if (self.0.data[i / BITS] & bit) == bit {
                b'x'
            } else {
                b'?'
            };
            i += 1;
        }
        arr
    }
}

#[derive(Debug, Copy, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[repr(C, align(1))]
pub struct Signature<const N: usize>
where
    [(); N.div_ceil(u8::BITS as usize)]:,
{
    #[cfg_attr(feature = "serde", serde(with = "serde_big_array::BigArray"))]
    pub pattern: [u8; N],
    pub mask: SignatureMask<N>,
}

impl<const N: usize> Signature<N>
where
    [(); N.div_ceil(u8::BITS as usize)]:,
{
    #[inline(always)]
    pub const fn from_pattern_mask(pattern: [u8; N], mask: [u8; N]) -> Self {
        Self {
            pattern,
            mask: SignatureMask::from_byte_array(mask),
        }
    }

    // Notice we cannot use From<([u8; N], [u8; N])> because it will break const guarantee

    #[inline(always)]
    pub const fn from_pattern_mask_tuple((pattern, mask): ([u8; N], [u8; N])) -> Self {
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
            mask: SignatureMask::MAX,
        }
    }

    #[inline(always)]
    pub const fn from_slice_with_exact_match_mask(pattern: &[u8; N]) -> Self {
        Self::from_array_with_exact_match_mask(*pattern)
    }

    #[inline(always)]
    pub const unsafe fn from_option_slice_unchecked(needle: &[Option<u8>]) -> Self {
        let mut pattern: [u8; N] = [0; N];
        let mut mask: [bool; N] = [false; N];
        let mut i = 0;
        while i < needle.len() {
            if let Some(x) = needle[i] {
                pattern[i] = x;
                mask[i] = true;
            }
            i += 1;
        }
        Self {
            pattern,
            mask: SignatureMask::from_bool_array(mask),
        }
    }
}

impl<const N: usize> Signature<N>
where
    [(); N.div_ceil(u8::BITS as usize)]:,
    [(); N - 1]:,
{
    #[inline]
    pub fn scan<'a>(&self, haystack: &'a [u8]) -> Option<&'a [u8]> {
        self.scan_inner(haystack, |chunk| self.match_best_effort(chunk))
    }

    #[inline]
    pub fn scan_naive<'a>(&self, haystack: &'a [u8]) -> Option<&'a [u8]> {
        self.scan_inner(haystack, |chunk| {
            match_naive_directly(chunk, self.pattern, &self.mask.0)
        })
    }

    #[inline(always)]
    fn scan_inner<'a>(
        &self,
        mut haystack: &'a [u8],
        f: impl Fn([u8; N]) -> bool,
    ) -> Option<&'a [u8]> {
        let exact_match = self.mask.all();

        if haystack.len() < N {
            if exact_match {
                None
            } else {
                f(unsafe { pad_zeroes_slice_unchecked::<N>(haystack) }).then_some(haystack)
            }
        } else {
            while !haystack.is_empty() {
                let haystack_smaller_than_n = haystack.len() < N;
                // If we are having the mask to match for all, and the chunk is actually smaller than N, we are cooked anyway
                if exact_match && haystack_smaller_than_n {
                    return None;
                }

                let window = unsafe { pad_zeroes_slice_unchecked::<N>(haystack) };

                if f(window) {
                    return Some(haystack);
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
                let potential_position_after_first = if self.mask.0[0] && !haystack_smaller_than_n {
                    self.equal_then_find_first_position(self.pattern[0], unsafe {
                        pad_zeroes_slice_unchecked::<{ N - 1 }>(&window[1..])
                    })
                    .unwrap_or(N - 1)
                } else {
                    0
                };

                haystack = &haystack[1 + potential_position_after_first..];
            }
            None
        }
    }
}

impl<const N: usize> Signature<N>
where
    [(); N.div_ceil(u8::BITS as usize)]:,
{
    #[inline]
    pub fn match_best_effort(&self, chunk: [u8; N]) -> bool {
        #[cfg(feature = "simd")]
        {
            if N >= 64 {
                self.match_simd::<u64>(chunk)
            } else if N >= 32 {
                self.match_simd::<u32>(chunk)
            } else if N >= 16 {
                self.match_simd::<u16>(chunk)
            } else if N >= 8 {
                self.match_simd::<u8>(chunk)
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

    #[inline]
    fn equal_then_find_first_position(&self, first: u8, window: [u8; N - 1]) -> Option<usize> {
        #[cfg(feature = "simd")]
        {
            if N >= 64 {
                equal_then_find_first_position_simd::<u64, { N - 1 }>(first, window)
            } else if N >= 32 {
                equal_then_find_first_position_simd::<u32, { N - 1 }>(first, window)
            } else if N >= 16 {
                equal_then_find_first_position_simd::<u16, { N - 1 }>(first, window)
            } else if N >= 8 {
                equal_then_find_first_position_simd::<u8, { N - 1 }>(first, window)
            } else {
                // for the lulz
                equal_then_find_first_position_naive(first, window)
            }
        }

        #[cfg(not(feature = "simd"))]
        {
            equal_then_find_first_position_naive(first, window)
        }
    }

    #[inline]
    pub fn match_naive(&self, chunk: [u8; N]) -> bool {
        match_naive_directly(chunk, self.pattern, &self.mask.0)
    }
}

#[cfg(feature = "simd")]
impl<const N: usize> Signature<N>
where
    [(); N.div_ceil(u8::BITS as usize)]:,
{
    #[inline]
    pub fn scan_simd<'a, T>(&self, haystack: &'a [u8]) -> Option<&'a [u8]>
    where
        T: Bits + One + Zero + Shl<usize, Output = T> + BitOr<Output = T>,
        LaneCount<{ T::BITS }>: SupportedLaneCount,
        u64: From<T>,
        [(); N - 1]:,
    {
        self.scan_inner(haystack, |chunk| self.match_simd(chunk))
    }

    #[inline]
    pub fn scan_simd_select<'a, T>(&self, haystack: &'a [u8]) -> Option<&'a [u8]>
    where
        T: Bits + One + Zero + Shl<usize, Output = T> + BitOr<Output = T>,
        LaneCount<{ T::BITS }>: SupportedLaneCount,
        u64: From<T>,
        [(); N - 1]:,
    {
        self.scan_inner(haystack, |chunk| self.match_simd_select(chunk))
    }

    #[inline]
    pub fn match_simd<T>(&self, chunk: [u8; N]) -> bool
    where
        T: Bits + One + Zero + Shl<usize, Output = T> + BitOr<Output = T>,
        LaneCount<{ T::BITS }>: SupportedLaneCount,
        u64: From<T>,
    {
        self.match_simd_inner(chunk, match_simd_core)
    }

    #[inline]
    pub fn match_simd_select<T>(&self, chunk: [u8; N]) -> bool
    where
        T: Bits + One + Zero + Shl<usize, Output = T> + BitOr<Output = T>,
        LaneCount<{ T::BITS }>: SupportedLaneCount,
        u64: From<T>,
    {
        self.match_simd_inner(chunk, match_simd_select_core)
    }

    #[inline]
    pub fn match_simd_inner<T>(
        &self,
        chunk: [u8; N],
        f: impl Fn([u8; T::BITS], [u8; T::BITS], T) -> bool,
    ) -> bool
    where
        T: Bits + One + Zero + Shl<usize, Output = T> + BitOr<Output = T>,
        LaneCount<{ T::BITS }>: SupportedLaneCount,
        u64: From<T>,
    {
        iterate_haystack_pattern_mask_aligned_simd(&chunk, &self.pattern, &self.mask.0)
            .all(|(haystack, pattern, mask)| f(pattern, haystack, mask))
    }
}

pub(crate) mod multiversion;
pub(crate) mod utils;
