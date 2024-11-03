use crate::{
    concise_bitvec::ConciseBitArray,
    mask::Mask,
    utils::{const_get_unchecked, const_set_unchecked},
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

#[cfg(feature = "simd")]
pub mod simd;

#[cfg(feature = "rayon")]
pub mod rayon;

#[cfg(all(feature = "rayon", feature = "simd"))]
pub mod rayon_simd;

pub mod naive;
