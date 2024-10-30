use core::simd::{cmp::SimdPartialEq, LaneCount, Mask, Simd, SupportedLaneCount};
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
pub fn simd_match_select<M, const N: usize>(pattern: [u8; N], mask: M, data: [u8; N]) -> bool
where
    M: Into<u64>,
    LaneCount<N>: SupportedLaneCount,
{
    Mask::from_bitmask(mask.into())
        .select_mask(
            Simd::from_array(data).simd_eq(Simd::from_array(pattern)),
            Mask::from_bitmask(u64::MAX),
        )
        .all()
}

/// Computes the result with the given the formula:
/// ```
/// pat:    [1, 2, 3, 4]
/// data:   [1, 1, 3, 3]
/// mask:   [1, 0, 1, 1] # 1 indicates equal, 0 indicates optional/don't care
/// result: [1, 1, 1, 0]
/// ```
///
/// This is the main algorithm of choice for result, which is optimized from the select version:
/// ```
/// for i in range(len(pat)):
///   result[i] = !mask[i] || pat[i] == data[i]
/// all(matches(x, true) for x in result)
/// ```
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
pub fn simd_match<M, const N: usize>(pattern: [u8; N], mask: M, data: [u8; N]) -> bool
where
    M: Into<u64>,
    LaneCount<N>: SupportedLaneCount,
{
    (!Mask::from_bitmask(mask.into()) | Simd::from_array(data).simd_eq(Simd::from_array(pattern)))
        .all()
}
