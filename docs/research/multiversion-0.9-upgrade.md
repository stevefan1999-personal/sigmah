# Upgrading `multiversion` 0.7.4 to 0.9.0

Date: 2026-09-29. Tracks [issue #1](https://github.com/stevefan1999-personal/sigmah/issues/1).

## Problem

`sigmah` is `no_std` when its `std` feature is off, but builds for targets without `std` failed:

```
$ cargo build --no-default-features --target x86_64-unknown-none
error[E0463]: can't find crate for `std`
error: could not compile `multiversion` (lib) due to 2 previous errors
```

## Root cause

`multiversion` 0.7.4 has a `std` cargo feature but no `#![no_std]` attribute in its crate root, so it
links `std` whatever the feature says. Host builds hide this because the host always has `std`.

0.8.0 has the same defect. 0.9.0 is the first release with
`#![cfg_attr(not(feature = "std"), no_std)]`, so there is no lower version to pick.

## Upstream changes that reach `sigmah`

| Change in 0.9.0 | Effect here |
|---|---|
| `no_std` attribute added | The fix |
| `rust-version = "1.86"` | Toolchain floor rises, see below |
| Clones are safe `#[target_feature]` functions instead of `unsafe fn` wrappers | Needs rustc 1.86 |
| Target strings are passed through, not expanded to implied features | One `#[target_feature]` per clone; rustc adds the implied set, so codegen is the same |
| Runtime detection checks only the named feature | No observable difference, see gaps |
| `target-features` dependency became optional | Dropped from the graph; `sigmah` uses none of its macros |

The attribute syntax `#[multiversion(targets("...", ...))]` is unchanged. The new `nightly` and
`target-features` features of `multiversion` are not needed: enabling either one produced a
byte-identical expansion of the three multiversioned functions.

## Verification

Bare-metal builds, `--no-default-features`:

| Target | 0.7.4 | 0.9.0 |
|---|---|---|
| `x86_64-unknown-none` | fails | passes |
| `riscv32i-unknown-none-elf` | fails | passes |
| `riscv32imac-unknown-none-elf` | fails | passes |
| `aarch64-unknown-none` | fails | passes |
| `armv7a-none-eabi` | fails | passes |
| `thumbv7em-none-eabihf` | fails | passes |

A freestanding `#![no_std] #![no_main]` binary calling the public API links against 0.9.0 with no
undefined symbols on `x86_64-unknown-none` and `riscv32imac-unknown-none-elf`.

No feature and target combination that built with 0.7.4 fails with 0.9.0 on rustc 1.86 or newer:

- 11 host feature combinations pass `cargo check` identically on `nightly-2025-10-01`.
- A differential test of 8,688 cases (95,568 assertions) comparing every SIMD scanner with the naive
  one gives byte-identical results for both versions, with runtime and with static dispatch.
- On x86_64, release machine code of the three multiversioned functions is identical per function.
  Only the one-time feature detection routine differs.

## Consequence: toolchain floor

Cargo refuses 0.9.0 on toolchains older than rustc 1.86, in every feature configuration, because
`multiversion` is a non-optional dependency. 0.7.4 built on 1.85. `--ignore-rust-version` is not
enough for the `simd` feature, which then fails with E0658.

Since 0.6.2 `sigmah` declares `rust-version = "1.86"` itself, so cargo names `sigmah` in the refusal.

## Defects found that predate the upgrade

Each of these behaves the same with 0.7.4 and 0.9.0.

1. **64-lane scanners miss matches. Fixed in 0.6.2.** `equal_then_find_second_position_simd_core`
   built its ignore-first mask as `(i64::MAX - 1) as u64`, which is `0x7FFF_FFFF_FFFF_FFFE` and
   clears bit 63 as well as bit 0. `scan::<u64>` and `scan::<usize>` then returned `None` where the
   naive scanner finds a match. Replacing the expression with `u64::MAX - 1` took the differential
   test from 296 mismatches to 0. `tests/simd_last_lane.rs` guards against a regression.
2. **`simd` does not compile on nightlies from 2026-03 onward.** `core::simd::LaneCount` and
   `SupportedLaneCount` no longer exist, and `Mask::select_mask` is gone.
3. **`simd` does not compile on 32-bit ARM.** The `arm+neon`, `arm+vfp4`, `arm+vfp3` and `arm+vfp2`
   targets need the unstable `arm_target_feature` gate, and the `vfp*` features cannot be detected
   at run time. On bare metal this was hidden behind the `std` error.
4. **The unit tests do not compile.** `src/multiversion/tests.rs` calls
   `equal_then_find_second_position_naive`, which does not exist.
5. **Stale feature gate.** `avx512_target_feature` has been stable since 1.89.

## Not verified

- Nothing ran on non-x86 hardware or an emulator. ARM, AArch64 and i686 results are compile-only.
- AVX-512 clones were never executed; the test host has AVX2 only.
- The toolchain floor was reproduced with stable 1.85.0 plus `RUSTC_BOOTSTRAP=1`, not a real nightly
  of that age.
- Narrowed runtime detection could differ only on a CPU or hypervisor reporting a feature without
  its prerequisites, such as `avx512vl` without `avx512f`. That could not be constructed.
- On nightly 1.100 the deny-by-default lint `x86_softfloat_sse` rejects SSE clones on soft-float
  targets such as `x86_64-unknown-none`. It cannot trigger today because `simd` does not compile
  there, but it will once defect 2 is fixed.
- Windows and macOS targets were not checked.
