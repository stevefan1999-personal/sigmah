#![cfg_attr(not(feature = "std"), no_std)]
// Enabling SIMD feature means including portable_simd and AVX512(BW), this enables the mask register access for smaller sizes
#![cfg_attr(feature = "simd", feature(portable_simd, avx512_target_feature))]
// Unfortunately, we only need a subset of generic_const_exprs which the minimal implementation should suffice
#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use bitvec::prelude::*;
use multiversion::multiversion;

#[cfg(feature = "simd")]
use {
    crate::{
        simd::{simd_match, simd_match_select},
        utils::Bits,
    },
    core::simd::{LaneCount, SupportedLaneCount},
    num_traits::PrimInt,
};

use crate::utils::pad_zeroes_slice_unchecked;

#[derive(Debug, Copy, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[repr(C, align(1))]
pub struct Signature<const N: usize>
where
    [(); N.div_ceil(u8::BITS as usize)]:,
{
    #[cfg_attr(feature = "serde", serde(with = "serde_big_array::BigArray"))]
    pub pattern: [u8; N],
    pub mask: BitArray<[u8; N.div_ceil(u8::BITS as usize)]>,
}

impl<const N: usize> Signature<N>
where
    [(); N.div_ceil(u8::BITS as usize)]:,
{
    #[inline(always)]
    pub const fn with_mask(pattern: [u8; N], mask: [u8; N]) -> Self {
        Self {
            pattern,
            mask: Self::bstring_mask_array_to_bitarr(mask),
        }
    }

    #[inline(always)]
    pub fn scan<'a>(&self, haystack: &'a [u8]) -> &'a [u8] {
        self.scan_inner(haystack, Self::match_best_effort)
    }

    #[inline(always)]
    pub fn scan_naive<'a>(&self, haystack: &'a [u8]) -> &'a [u8] {
        self.scan_inner(haystack, Self::match_naive)
    }

    #[cfg(feature = "simd")]
    #[inline(always)]
    pub fn scan_simd<'a, T>(&self, haystack: &'a [u8]) -> &'a [u8]
    where
        T: Bits + PrimInt,
        [(); T::BITS as usize]:,
        LaneCount<{ T::BITS as usize }>: SupportedLaneCount,
        u64: From<T>,
    {
        self.scan_inner(haystack, Self::match_simd)
    }

    #[cfg(feature = "simd")]
    #[inline(always)]
    pub fn scan_simd_select<'a, T>(&self, haystack: &'a [u8]) -> &'a [u8]
    where
        T: Bits + PrimInt,
        [(); T::BITS as usize]:,
        LaneCount<{ T::BITS as usize }>: SupportedLaneCount,
        u64: From<T>,
    {
        self.scan_inner(haystack, Self::match_simd_select)
    }

    #[inline(always)]
    fn scan_inner<'a>(
        &self,
        mut haystack: &'a [u8],
        f: impl Fn(&Self, &'a [u8]) -> bool,
    ) -> &'a [u8] {
        while !haystack.is_empty() {
            if f(self, haystack) {
                break;
            }
            haystack = &haystack[1..];
        }
        haystack
    }
}

impl<const N: usize> Signature<N>
where
    [(); N.div_ceil(u8::BITS as usize)]:,
{
    #[inline(always)]
    pub fn match_best_effort<'a>(&self, haystack: &'a [u8]) -> bool {
        #[cfg(feature = "simd")]
        {
            if N > 32 {
                self.match_simd::<u64>(haystack)
            } else if N > 16 {
                self.match_simd::<u32>(haystack)
            } else if N > 8 {
                self.match_simd::<u16>(haystack)
            } else {
                self.match_simd::<u8>(haystack)
            }
        }

        #[cfg(not(feature = "simd"))]
        {
            self.match_naive(haystack)
        }
    }

    #[inline(always)]
    pub fn match_naive<'a>(&self, haystack: &'a [u8]) -> bool {
        #[inline(always)]
        #[multiversion(targets(
            "x86_64+avx2",
            "x86_64+avx",
            "x86_64+sse2",
            "x86_64+sse",
            "x86+avx2",
            "x86+avx",
            "x86+sse2",
            "x86+sse",
            "aarch64+sve2",
            "aarch64+sve",
            "aarch64+neon",
            "arm+neon",
            "arm+vfp4",
            "arm+vfp3",
            "arm+vfp2",
        ))]
        fn match_naive_inner<'a, const N: usize>(this: &Signature<N>, haystack: &'a [u8]) -> bool
        where
            [(); N.div_ceil(u8::BITS as usize)]:,
        {
            unsafe { pad_zeroes_slice_unchecked::<N>(haystack) }
                .into_iter()
                .zip(this.pattern)
                .zip(this.mask)
                .all(|((haystack, pattern), mask)| !mask || haystack == pattern)
        }
        match_naive_inner::<N>(self, haystack)
    }

    #[cfg(feature = "simd")]
    #[inline(always)]
    pub fn match_simd<'a, T>(&self, haystack: &'a [u8]) -> bool
    where
        T: Bits + PrimInt,
        LaneCount<{ T::BITS as usize }>: SupportedLaneCount,
        u64: From<T>,
    {
        self.iterate_simd(haystack)
            .all(|(haystack, pattern, mask)| {
                simd_match(
                    pattern,
                    mask.iter_ones()
                        .fold(T::zero(), |acc, x| acc | (T::one() << x)),
                    haystack,
                )
            })
    }

    #[cfg(feature = "simd")]
    #[inline(always)]
    pub fn match_simd_select<'a, T>(&self, haystack: &'a [u8]) -> bool
    where
        T: Bits + PrimInt,
        LaneCount<{ T::BITS as usize }>: SupportedLaneCount,
        u64: From<T>,
    {
        self.iterate_simd(haystack)
            .all(|(haystack, pattern, mask)| {
                simd_match_select(
                    pattern,
                    mask.iter_ones()
                        .fold(T::zero(), |acc, x| acc | (T::one() << x)),
                    haystack,
                )
            })
    }

    #[cfg(feature = "simd")]
    #[inline(always)]
    pub fn iterate_simd<'a, T>(
        &'a self,
        haystack: &'a [u8],
    ) -> impl Iterator<
        Item = (
            [u8; T::BITS as usize],
            [u8; T::BITS as usize],
            &'a BitSlice<u8>,
        ),
    >
    where
        T: Bits + PrimInt,
    {
        let bits: usize = T::BITS as usize;

        haystack
            .chunks(bits)
            .map(|x| unsafe { pad_zeroes_slice_unchecked(x) })
            .zip(
                self.pattern
                    .chunks(bits)
                    .map(|x| unsafe { pad_zeroes_slice_unchecked(x) }),
            )
            .zip(self.mask.chunks(bits))
            .map(|((a, b), c)| (a, b, c))
    }
}

impl<const N: usize> Signature<N>
where
    [(); N.div_ceil(u8::BITS as usize)]:,
{
    #[inline(always)]
    pub const fn from_option_array(needle: [Option<u8>; N]) -> Self {
        unsafe { Self::from_option_slice_unchecked(&needle) }
    }

    #[inline(always)]
    pub const fn bstring_mask_array_to_bitarr(
        pattern: [u8; N],
    ) -> BitArray<[u8; N.div_ceil(u8::BITS as usize)]> {
        unsafe { Self::bstring_mask_slice_to_bitarr_unchecked(&pattern) }
    }

    #[inline(always)]
    pub const fn boolean_mask_array_to_bitarr(
        pattern: [bool; N],
    ) -> BitArray<[u8; N.div_ceil(u8::BITS as usize)]> {
        unsafe { Self::boolean_mask_slice_to_bitarr_unchecked(&pattern) }
    }
}

impl<const N: usize> Signature<N>
where
    [(); N.div_ceil(u8::BITS as usize)]:,
{
    #[inline(always)]
    pub const unsafe fn from_option_slice_unchecked(needle: &[Option<u8>]) -> Self {
        let mut needle_: [u8; N] = [0; N];
        let mut pattern: [bool; N] = [false; N];
        let mut i = 0;
        while i < needle.len() {
            if let Some(x) = needle[i] {
                needle_[i] = x;
                pattern[i] = true;
            } else {
                pattern[i] = false;
            }
            i += 1;
        }
        Self {
            pattern: needle_,
            mask: Self::boolean_mask_slice_to_bitarr_unchecked(&pattern),
        }
    }

    #[inline(always)]
    pub const unsafe fn bstring_mask_slice_to_bitarr_unchecked(
        pattern: &[u8],
    ) -> BitArray<[u8; N.div_ceil(u8::BITS as usize)]> {
        let mut pattern_bool: [bool; N] = [false; N];
        let mut i = 0;
        while i < pattern.len() {
            pattern_bool[i] = match pattern[i] {
                b'x' => true,
                b'?' => false,
                _ => panic!("unknown character in pattern"),
            };
            i += 1;
        }
        Self::boolean_mask_slice_to_bitarr_unchecked(&pattern_bool)
    }

    #[inline(always)]
    pub const unsafe fn boolean_mask_slice_to_bitarr_unchecked(
        pattern: &[bool],
    ) -> BitArray<[u8; N.div_ceil(u8::BITS as usize)]> {
        let mut arr: BitArray<[u8; N.div_ceil(u8::BITS as usize)]> = BitArray::ZERO;
        let mut i = 0;
        while i < pattern.len() {
            if pattern[i] {
                const BITS: usize = u8::BITS as usize;
                let (byte_pos, rem) = (i / BITS, i % BITS);
                let one: u8 = 1;
                arr.data[byte_pos] |= one << rem;
            }
            i += 1;
        }
        arr
    }
}

#[cfg(feature = "simd")]
pub mod simd;
pub(crate) mod utils;
