#![cfg(feature = "simd")]
#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use sigmah::Signature;

const LANES: usize = u64::BITS as usize;
const FILLER: u8 = 0x01;
const PATTERN_BASE: u8 = 0x80;

// The scanner skips ahead to the next occurrence of the pattern's first byte inside the current
// window. When that occurrence sits in the last of 64 lanes, the skip must still land on it.
#[test]
fn match_starting_at_the_last_lane_of_the_first_window_is_found() {
    let pattern: [u8; LANES] = core::array::from_fn(|index| PATTERN_BASE + index as u8);
    let match_offset = LANES - 1;
    let haystack = [vec![FILLER; match_offset], pattern.to_vec()].concat();

    let found = Signature::from_array_with_exact_match_mask(pattern)
        .simd()
        .scan::<u64>(&haystack);

    assert_eq!(found, Some(&haystack[match_offset..]));
}
