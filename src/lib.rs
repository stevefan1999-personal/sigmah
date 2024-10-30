#![cfg_attr(not(feature = "std"), no_std)]
// Enabling SIMD feature means including portable_simd and AVX512(BW), this enables the mask register access for smaller sizes
#![cfg_attr(feature = "simd", feature(portable_simd, avx512_target_feature))]
// Unfortunately, we only need a subset of generic_const_exprs which the minimal implementation should suffice
#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use bitvec::prelude::*;

#[cfg(feature = "simd")]
use {
    crate::{
        simd::simd_match,
        utils::{pad_zeroes_slice_unchecked, Bits},
    },
    core::simd::{LaneCount, SupportedLaneCount},
    num_traits::PrimInt,
};

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
        #[cfg(feature = "simd")]
        {
            self.scan_simd::<usize>(haystack)
        }

        #[cfg(not(feature = "simd"))]
        {
            self.scan_naive(haystack)
        }
    }

    #[inline(always)]
    pub fn scan_naive<'a>(&self, mut haystack: &'a [u8]) -> &'a [u8] {
        while !haystack.is_empty() {
            if unsafe { pad_zeroes_slice_unchecked::<N>(haystack) }
                .into_iter()
                .zip(self.pattern)
                .zip(self.mask)
                .all(|((haystack, pattern), mask)| !mask || haystack == pattern)
            {
                break;
            }
            haystack = &haystack[1..];
        }
        haystack
    }

    #[cfg(feature = "simd")]
    #[inline(always)]
    pub fn scan_simd<'a, T>(&self, mut haystack: &'a [u8]) -> &'a [u8]
    where
        T: Bits + PrimInt,
        [(); T::BITS as usize]:,
        LaneCount<{ T::BITS as usize }>: SupportedLaneCount,
        usize: From<T>,
    {
        let bits: usize = T::BITS as usize;

        while !haystack.is_empty() {
            if haystack
                .chunks(bits)
                .map(|x| unsafe { pad_zeroes_slice_unchecked::<{ T::BITS as usize }>(x) })
                .zip(
                    self.pattern
                        .chunks(bits)
                        .map(|x| unsafe { pad_zeroes_slice_unchecked::<{ T::BITS as usize }>(x) }),
                )
                .zip(self.mask.chunks(bits))
                .all(|((haystack, pattern), mask)| {
                    simd_match::<T, { T::BITS as usize }>(
                        pattern,
                        mask.iter_ones()
                            .fold(T::zero(), |acc, x| acc | (T::one() << x)),
                        haystack,
                    )
                })
            {
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
