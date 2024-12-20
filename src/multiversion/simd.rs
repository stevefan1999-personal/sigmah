use core::simd::{cmp::SimdPartialEq, LaneCount, Mask, Simd, SupportedLaneCount};

use crate::utils::{pad_zeroes_slice_unchecked, simd::SimdBits};

use multiversion::multiversion;

#[cfg(feature = "rayon")]
use {
    arrayvec::ArrayVec,
    rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator},
};

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
pub fn match_simd_select_core<const N: usize>(
    data: &[u8; N],
    pattern: &[u8; N],
    mask: impl Into<u64>,
) -> Result<(), usize>
where
    LaneCount<N>: SupportedLaneCount,
{
    let mask = Mask::from_bitmask(mask.into()).select_mask(
        Simd::from_array(*data).simd_eq(Simd::from_array(*pattern)),
        Mask::from_bitmask(u64::MAX),
    );
    if mask.all() {
        Ok(())
    } else {
        Err((!mask).first_set().unwrap_or(0))
    }
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
pub fn match_simd_core<const N: usize>(
    data: &[u8; N],
    pattern: &[u8; N],
    mask: impl Into<u64>,
) -> Result<(), usize>
where
    LaneCount<N>: SupportedLaneCount,
{
    let mask = !Mask::from_bitmask(mask.into())
        | Simd::from_array(*data).simd_eq(Simd::from_array(*pattern));
    if mask.all() {
        Ok(())
    } else {
        Err((!mask).first_set().unwrap_or(0))
    }
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
pub fn equal_then_find_second_position_simd_core<const N: usize>(
    first: Simd<u8, N>,
    window: &[u8; N],
    is_first_chunk: bool,
) -> Option<usize>
where
    LaneCount<N>: SupportedLaneCount,
{
    let ignore_first = Mask::from_bitmask(if is_first_chunk {
        (i64::MAX - 1) as u64
    } else {
        u64::MAX
    });
    (Simd::from_array(*window).simd_eq(first) & ignore_first).first_set()
}

#[inline(always)]
pub fn equal_then_find_second_position_simd<T: SimdBits, const N: usize>(
    first: u8,
    window: &[u8; N],
) -> Option<usize>
where
    LaneCount<{ T::LANES }>: SupportedLaneCount,
    [(); N.div_ceil(T::LANES)]:,
{
    let first_splat = Simd::splat(first);

    let f = |stride: usize, window: &[u8]| {
        let window = unsafe {
            if window.len() < T::LANES {
                &pad_zeroes_slice_unchecked(window)
            } else {
                core::mem::transmute_copy(&window)
            }
        };
        equal_then_find_second_position_simd_core(first_splat, window, stride == 0)
            .map(|position| T::LANES * stride + position)
    };

    let window_chunks = window.chunks(T::LANES);

    #[cfg(feature = "rayon")]
    {
        window_chunks
            .collect::<ArrayVec<&[u8], { N.div_ceil(T::LANES) }>>()
            .into_par_iter()
            .enumerate()
            .find_map_first(|(stride, window)| f(stride, window))
    }

    #[cfg(not(feature = "rayon"))]
    {
        window_chunks
            .enumerate()
            .find_map(|(stride, window)| f(stride, window))
    }
}
