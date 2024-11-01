use bitvec::slice::BitSlice;
use multiversion::multiversion;

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
pub fn match_naive_directly(chunk: &[u8], pattern: &[u8], mask: &BitSlice<u8>) -> bool {
    chunk
        .iter()
        .zip(pattern)
        .zip(mask)
        .all(|((chunk, pattern), mask)| !mask || chunk == pattern)
}

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
pub fn equal_then_find_second_position_naive(first: u8, window: &[u8]) -> Option<usize> {
    window
        .iter()
        .skip(1)
        .position(|&x| x == first)
        .map(|x| 1 + x)
}

#[cfg(feature = "simd")]
pub(crate) mod simd;

#[cfg(test)]
mod tests;
