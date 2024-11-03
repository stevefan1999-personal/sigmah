use super::Signature;
use crate::{
    multiversion::{equal_then_find_second_position_naive_rayon, match_naive_rayon},
    utils::pad_zeroes_slice_unchecked,
};
use core::mem::transmute_copy;
use derive_more::derive::{Deref, From, Into};

#[derive(Debug, Copy, Clone, Deref, From, Into)]
#[repr(C, align(1))]
pub struct SignatureWithRayonNaive<const N: usize>(pub Signature<N>)
where
    [(); N.div_ceil(u8::BITS as usize)]:;

impl<const N: usize> SignatureWithRayonNaive<N>
where
    [(); N.div_ceil(u8::BITS as usize)]:,
{
    #[inline(always)]
    pub fn match_chunk(&self, chunk: &[u8; N]) -> bool {
        match_naive_rayon(chunk, &self.pattern, &self.mask)
    }

    #[inline(always)]
    pub fn scan<'a>(&self, mut haystack: &'a [u8]) -> Option<&'a [u8]> {
        let exact_match = self.mask.is_exact();
        while !haystack.is_empty() {
            let haystack_smaller_than_n = haystack.len() < N;

            let window: &[u8; N] = unsafe {
                if haystack_smaller_than_n {
                    &pad_zeroes_slice_unchecked::<N>(haystack)
                } else {
                    transmute_copy(&haystack)
                }
            };

            if self.match_chunk(window) {
                return Some(haystack);
            } else if exact_match && haystack_smaller_than_n {
                // If we are having the mask to match for all, and the chunk is actually smaller than N, we are cooked anyway
                return None;
            }

            let move_position = if unsafe { self.mask.get_unchecked(0) } && !haystack_smaller_than_n
            {
                equal_then_find_second_position_naive_rayon(
                    unsafe { self.get_unchecked(0) },
                    window,
                )
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
