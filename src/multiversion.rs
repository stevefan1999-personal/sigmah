use crate::{concise_bitvec::ConciseBitArray, utils::const_get_unchecked};

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
/// This is the naive algorithm of choice for result, that is directly collected and resulted as a boolean:
/// ```
/// for i in range(len(pat)):
///   result[i] = !mask[i] || pat[i] == data[i]
/// all(matches(x, true) for x in result)
/// ```
#[inline(always)]
#[cfg(feature = "rayon")]
pub fn match_naive_rayon<const N: usize>(
    chunk: &[u8; N],
    pattern: &[u8; N],
    mask: &ConciseBitArray<N>,
) -> bool
where
    [(); N.div_ceil(u8::BITS as usize)]:,
{
    chunk
        .into_par_iter()
        .zip(pattern.into_par_iter())
        .zip(
            mask.0
                .to_bitvec()
                .into_iter()
                .take(N)
                .collect::<ArrayVec<bool, N>>()
                .into_par_iter(),
        )
        .all(|((chunk, pattern), mask)| !mask || chunk == pattern)
}

#[inline(always)]
pub const fn match_naive_const<const N: usize>(
    chunk: &[u8; N],
    pattern: &[u8; N],
    mask: &ConciseBitArray<N>,
) -> Result<(), usize>
where
    [(); N.div_ceil(u8::BITS as usize)]:,
{
    let mut i = 0;
    while i < N {
        const BITS: usize = u8::BITS as usize;
        let (idx, bit_pos) = (i / BITS, i % BITS);
        if !unsafe {
            let mask = (const_get_unchecked(&mask.0.data, idx) & (1 << bit_pos)) != 0;
            !mask || (const_get_unchecked(chunk, i) == const_get_unchecked(pattern, i))
        } {
            return Err(i);
        }
        i += 1;
    }
    Ok(())
}

#[inline(always)]
#[cfg(feature = "rayon")]
pub fn equal_then_find_second_position_naive_rayon<const N: usize>(
    first: u8,
    window: &[u8; N],
) -> Option<usize> {
    window
        .into_par_iter()
        .skip(1)
        .position_first(|&x| x == first)
        .map(|x| 1 + x)
}

#[inline(always)]
pub const fn equal_then_find_second_position_naive_const<const N: usize>(
    first: u8,
    window: &[u8; N],
) -> Option<usize> {
    let mut i = 1;
    while i < N {
        if unsafe { const_get_unchecked(window, i) } == first {
            return Some(i);
        }
        i += 1;
    }
    None
}

#[cfg(feature = "simd")]
pub(crate) mod simd;

#[cfg(test)]
mod tests;
