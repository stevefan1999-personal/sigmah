# Sigmah

A blatantly simple Rust crate that creates binary signature efficiently, which can be stored on stack or as const item (So can store the pattern without making it on heap!).

It also generate scanner function using either best-effort algorithm chosen from naive linear scanning, or using runtime SIMD selection as backend if feature is available, which also includes a scalar fallback (Thanks you portable SIMD <3).

The generated function is internalized, meaning the signature itself would embed directly as part of the opcode without much preprocessing.

This should directly kill many Find Pattern benchmark while maintaining a decent performance.

# How it works?

By stuffing the mask into a bit array with **precise** byte storage, with each mask encoded directly at the bit level.
This can be considered a bit hack, given that we hard coded the condition for:

```
0 => Optional/don't care => Result = 1
1 => Match => Pattern data at current position should equal current data at current position
```

Thus, for that we want to compute the following:

```
pat:    [1, 2, 3, 4]
data:   [1, 1, 3, 3]
mask:   [1, 0, 1, 1] # 1 indicates equal, 0 indicates optional/don't care
result: [1, 1, 1, 0]
```

For example, given a pattern of "\x01\x02\x03x\04\x05\x06" and mask of "xx??x?", this means we transform the mask to 0b110010 (which can be done in Rust constant evaluation with compiler fuckery), aligned to the closest byte-boundary, which is (6+7)//8 = 1 byte (or use euclidean division towards positive infinity), then we just append it back to the pattern as-is, which makes it insanely more compact than typical byte string based pattern. Notice now endianess do matter here, so you need to either declare either LSB-0 (little endian) or MSB-0 (big endian) if you want to deliver it over the network boundary. 

If you checked the pattern in IDA, you notice the bytes and comparison operations are perfectly reversed -- this is totally normal because you are most likely running in a little endian system known as x86, dumbass.

By appending the binary mask back to the pattern, we achieve a more compact representation. Instead of storing both the pattern and mask in full, this method reduces the storage requirement to just the pattern length plus the mask overhead. For a 16-byte pattern, the mask overhead would be only 2 bytes (16/8), significantly less than the typical 2N bytes where each byte of the pattern and mask is stored separately. This is much more efficient, although quite harder to make since there is a lot of brain damage required here to make this happen with const at the same time.

Given the transitional matrix above, so this is the first algorithm that comes to mind (in pseudocode):

```
for i in range(len(pat)):
  result[i] = mask[i] ? pat[i] == data[i] : 1
all(matches(x, true) for x in result)
```

And with a little bit of wit, we can turn this ternary into a binary logical operation:

```
for i in range(len(pat)):
  result[i] = !mask[i] || pat[i] == data[i]
all(matches(x, true) for x in result)
```

If you want to argue further:

```
for i in range(len(pat)):
  result[i] = mask[i] "IMPLIES" (pat[i] == data[i])
all(matches(x, true) for x in result)
```

Here, "IMPLIES" represents [logical implication](https://en.wikipedia.org/wiki/Material_conditional) (A implies B is true unless A is true and B is false). This logical operation simplifies the check but introduces complexity in implementation, but there is likely no way to encode IMPLIES in SIMD at the moment. Not that I'm aware enough to know if one exists in the wild.

[`VPTERNLOGD`](https://www.felixcloutier.com/x86/vpternlogd:vpternlogq) could be handy here, so I left the OG function available as well if you like to experiment with it. At the moment, it is generating the same code to the binary logical one, i.e. also automatically optimized. Until LLVM team fully juiced out their brain power and figured out how to encode VPTERNLOGD, take that as something you shouldn't use in general. (though why don't you just fetch it from the AVX512 mask register at this point?).

# Nightly status

Currently, the Rust programming language feature known as [`generic_const_exprs`](https://github.com/rust-lang/rust/issues/76560) plays a critical role, especially for scenarios involving **precise byte storage**. This feature allows for the use of generic const expressions, which are essential when you need to define types where the size in bytes must be exactly specified. However, this capability is still incomplete, leading to several limitations and challenges:

1. **Incompleteness of `generic_const_exprs`:**

   - **Current State**: As of now, `generic_const_exprs` is an experimental feature in the Rust nightly builds. It enables developers to use expressions in generic parameters, which is vital for precise byte alignment in data structures.
   - **Usage**: Libraries that require strict byte alignment (for example, in low-level systems programming or when interfacing with hardware) depend on this feature. Without it, ensuring byte-level precision becomes problematic.

2. **Workarounds and Their Drawbacks:**

   - **Macro Usage**: One workaround to achieve precise byte storage without `generic_const_exprs` involves using macros to generate the necessary code. Macros can create structures with predefined byte sizes, but this approach:
     - Adds complexity to the codebase.
     - Can make maintenance more difficult due to the expanded code.
   - **Forcing Byte Alignment**: Another approach is to pad or truncate data to fit into larger, aligned types like `u32` or `u64`. For instance:
     - If you have a pattern that's 20 bytes long, it might need to be expanded to 32 bytes to align with `u32`.
     - A 48-byte pattern would typically be extended to 64 bytes to align with `u64`.
     - This method leads to increased memory usage and can affect performance due to unnecessary data padding.

3. **Current Compatibility**:

   - Despite its incomplete status, libraries can still compile and run with the feature using `cargo 1.84.0-nightly` from October 25, 2024. This indicates a temporary robustness in the feature's implementation but does not address its long-term stability.

4. **Future Outlook**:
   - **Stagnation and Potential Bit Rot**: The `generic_const_exprs` feature has seen little progress over the years, raising concerns about its maintenance and future compatibility. Without active development, there's a risk of "bit rot", where the code might become outdated or fail to integrate with new language features or standards.
   - **Potential Alternatives**: There is hope in the Rust community for solutions like [`min_generic_const_exprs`](https://hackmd.io/@rust-const-generics/S15xxREKF) or similar proposals. These could offer a more stable and safer approach to const generics:
     - **Advantages**: These alternatives might simplify the language syntax, improve compile times, and provide better safety guarantees by limiting the complexity of what can be done at compile-time.
     - **Transition Plan**: Should these features mature, there would likely be a shift towards them, phasing out `generic_const_exprs` in favor of more robust solutions.

I'm keenly watching for advancements that could either enhance this feature or replace it with something more fitting for long-term use.

SIMD backend requires nightly for `portable_simd` (obviously), as well as [`avx512_target_feature`](https://github.com/rust-lang/rust/issues/44839) too for a nifty AVX512BW hack:

```c
bool __fastcall sub_140001C86(__int64 a1, __int64 a2, __int64 _R8)
{
  bool result; // al

  __asm
  {
    vmovdqu64 zmm0, zmmword ptr [r8]
    vpcmpb  k0, zmm0, zmmword ptr [rcx], 4
    kmovq   rax, k0
  }
  result = (a2 & _RAX) == 0;
  __asm { vzeroupper }
  return result;
}
```

Notice the vmovdqu64, vpcmpb and kmovq. All of which requires more than AVX512F, since this directly reads from the mask register, and hence should be killing on the competition at an insane speed due to its compact and direct design. But this is gated behind `avx512_target_feature` at the moment, and it is expected to be stabilized soon in the next half of 2025. AVX2 code generated has variable length, which is due with different signature size, so it is not shown as snippet here, but I strongly suggest you to check out the result in IDA as well.

# SIMD explanation

TODO

# Examples

TODO