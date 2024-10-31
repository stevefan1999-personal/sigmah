use core::simd::{cmp::SimdPartialEq, LaneCount, Mask, Simd, SupportedLaneCount};

use crate::utils::{pad_zeroes_slice_unchecked, simd::Bits};
use multiversion::multiversion;

/// Computes the result with the given the formula:
/// ```
/// pat:    [1, 2, 3, 4]
/// data:   [1, 1, 3, 3]
/// mask:   [1, 0, 1, 1] # 1 indicates equal, 0 indicates optional/don't care
/// result: [1, 1, 1, 0]
/// ```
///
/// This is an alternative algorithm for result:
/// ```
/// for i in range(len(pat)):
///   result[i] = mask[i] ? pat[i] == data[i] : 1
/// all(matches(x, true) for x in result)
/// ```
#[inline(always)]
#[multiversion(targets(
    "x86_64+avx512vl",
    "x86_64+avx2",
    "x86_64+avx",
    "x86_64+sse2",
    "x86_64+sse",
    "x86+avx512bw",
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
pub fn match_simd_select_core<M, const N: usize>(
    pattern: [u8; N],
    data: [u8; N],
    mask: M,
) -> Mask<i8, N>
where
    M: Into<u64>,
    LaneCount<N>: SupportedLaneCount,
{
    Mask::from_bitmask(mask.into()).select_mask(
        Simd::from_array(data).simd_eq(Simd::from_array(pattern)),
        Mask::from_bitmask(u64::MAX),
    )
}

#[inline(always)]
#[multiversion(targets(
    "x86_64+avx512bw",
    "x86_64+avx2",
    "x86_64+avx",
    "x86_64+sse2",
    "x86_64+sse",
    "x86+avx512bw",
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
pub fn match_simd_core<M, const N: usize>(pattern: [u8; N], data: [u8; N], mask: M) -> Mask<i8, N>
where
    M: Into<u64>,
    LaneCount<N>: SupportedLaneCount,
{
    !Mask::from_bitmask(mask.into()) | Simd::from_array(data).simd_eq(Simd::from_array(pattern))
}

#[inline(always)]
#[multiversion(targets(
    "x86_64+avx512vl",
    "x86_64+avx2",
    "x86_64+avx",
    "x86_64+sse2",
    "x86_64+sse",
    "x86+avx512bw",
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
pub fn equal_then_find_first_position_simd_core<M, const N: usize>(
    first: u8,
    window: [u8; N],
) -> Option<usize>
where
    M: Into<u64>,
    LaneCount<N>: SupportedLaneCount,
{
    Simd::from_array(window)
        .simd_eq(Simd::splat(first))
        .first_set()
}

#[inline(always)]
pub fn equal_then_find_first_position_simd<T>(first: u8, window: &[u8]) -> Option<usize>
where
    T: Into<u64> + Bits,
    LaneCount<{ T::BITS }>: SupportedLaneCount,
    [(); T::BITS]:,
{
    window
        .chunks(T::BITS)
        .map(|x| unsafe { pad_zeroes_slice_unchecked::<{ T::BITS }>(x) })
        .enumerate()
        .filter_map(|(stride, window)| {
            equal_then_find_first_position_simd_core::<T, { T::BITS }>(first, window)
                .map(|position| T::BITS * stride + position)
        })
        .next()
}
