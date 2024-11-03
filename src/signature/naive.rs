use super::Signature;
use crate::{
    multiversion::{equal_then_find_second_position_naive_const, match_naive_const},
    utils::pad_zeroes_slice_unchecked,
};
use core::mem::transmute_copy;
use derive_more::derive::{Deref, From, Into};

#[derive(Debug, Copy, Clone, From, Into, Deref)]
#[repr(C, align(1))]
pub struct SignatureWithNaive<const N: usize>(pub Signature<N>)
where
    [(); N.div_ceil(u8::BITS as usize)]:;

impl<const N: usize> SignatureWithNaive<N>
where
    [(); N.div_ceil(u8::BITS as usize)]:,
{
    // scan_inner is the main function (due to impl Fn being a trait and not const friendly) and please synchronize in time
    pub const fn scan<'a>(&self, mut haystack: &'a [u8]) -> Option<&'a [u8]> {
        let exact_match = self.0.mask.0.is_exact();
        while !haystack.is_empty() {
            let haystack_smaller_than_n = haystack.len() < N;

            let window: &[u8; N] = unsafe {
                if haystack_smaller_than_n {
                    &pad_zeroes_slice_unchecked::<N>(haystack)
                } else {
                    transmute_copy(&haystack)
                }
            };

            if match_naive_const(window, &self.0.pattern, &self.0.mask.0).is_ok() {
                return Some(haystack);
            } else if exact_match && haystack_smaller_than_n {
                // If we are having the mask to match for all, and the chunk is actually smaller than N, we are cooked anyway
                return None;
            }

            let move_position =
                if unsafe { self.0.mask.0.get_unchecked(0) } && !haystack_smaller_than_n {
                    if let Some(pos) = equal_then_find_second_position_naive_const(
                        unsafe { self.0.get_unchecked(0) },
                        window,
                    ) {
                        pos
                    } else {
                        N
                    }
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
