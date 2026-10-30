# iridia-d3d9

Reads raw Direct3D 9 `ps_3_0` and `vs_3_0` shader token streams and their CTAB constant table, and lifts them into [`iridia::Program`](../iridia), a [naga](https://github.com/gfx-rs/wgpu/tree/trunk/naga) module with its interface.

Part of [iridia](https://github.com/Kalmarv/iridia), which turns compiled Direct3D shaders into naga modules. The input is the token stream a Direct3D 9 compiler writes (`D3DXCompileShader`, `fxc /T ps_3_0`), version token to `END`, with no container. This crate does no file I/O.

## What it reads

- `read(bytes)` decodes a complete `ps_3_0` or `vs_3_0` stream into declarations, instructions, immediate definitions, every comment, and the CTAB (creator, target, flags, named register allocations, types and raw defaults; the reserved constant-info word is kept, since Microsoft compilers write nonzero values in it).
- `lift(bytes)` returns named varyings, resource layouts, D3D register provenance, the source hash and version, and a validated naga module. No source text is generated on the way.
- The subset is straight-line float arithmetic (`mov`, `add`, `sub`, `mad`, `mul`, `rcp`, `rsq`, `dp3`, `dp4`, `min`, `max`, `exp`, `log`, `lrp`, `frc`, `pow`, `abs`, `nrm`, `sincos`), `nop`, `dcl` and `def`. Vertex programs add `slt`, `sge`, `mova` (the address register `a0`, rounded to the nearest integer, a half away from zero) and float constants addressed relative to `a0` (`c[a0.x + N]`; a read outside the 256 constant registers is zero, and a `def` constant is what its register reads as). Pixel programs add `cmp`, `dp2add`, `texld` (plain, projected or biased) and `texkill`, with 2D, cube or volume textures. Source swizzles, negation, absolute value, destination masks and saturation are kept; `nrm` includes its fourth-lane result and the documented zero-vector `FLT_MAX` scale; `texkill` tests all four lanes (unlike the older xyz-only description on Microsoft's instruction page; see the [D9VK 0.40 release note](https://github.com/misyltoad/d9vk/releases/tag/0.40)). Partial precision is a hardware request whose f32 execution a note on the program reports.
- Vertex programs keep their declared output semantics and masks, including distinct semantics packed into disjoint lanes of one register; `POSITION0` becomes the clip position and other outputs stay named varyings. The banks are bounded to 16 inputs, 32 temporaries, 12 outputs and 256 float constants; CTAB's version must agree with the stage, and every declared output lane must be written.
- The float constant bank is `cb0`, an array of raw `vec4<u32>` rows: register `cN` is byte offset `16 * N`, and `def` values override it locally. Sampler `sN` becomes a texture at `tN` and a sampler at `sN`, bindings keep their original slots. CTAB float scalars, vectors, matrices and arrays get explicit offsets and strides; an optimized array whose allocation is shorter than its declared type is described by the allocated range, with a note, and `read` keeps the full type.
- `frame(bytes)` finds the first `vs_3_0` or `ps_3_0` program in a buffer whose later bytes are something else (a container's remaining bytes, say) and returns its stage and exact bytes through `END`, without searching for `END` inside operands or comments. `sampler_states` maps `D3DSAMP` state/value pairs to the interface vocabulary.
- Derived programs, each with a distinct lifter identity and a note naming what it is, whose source hash still identifies the complete original: `project` (a pixel program cut after an instruction, a temporary's chosen lanes returned as RGB), `reencode` (the whole program with those lanes replaced by a new input, so the original downstream operations run) and `project_vertex_output` (the backwards slice of a vertex program that reaches chosen lanes of a declared output, as a pixel-stage invocation program). They assign no meaning to a semantic; recognizing a useful cut is the caller's.

## Example

A `ps_3_0` program written as tokens: `def c0, 1, 2, 3, 4` and `mov oC0, c0`.

```rust
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let words: [u32; 11] = [
        0xffff_0300,                                    // ps_3_0
        0x0500_0051, 0xa00f_0000, 0x3f80_0000, 0x4000_0000, 0x4040_0000, 0x4080_0000, // def c0, 1, 2, 3, 4
        0x0200_0001, 0x800f_0800, 0xa0e4_0000,          // mov oC0, c0
        0x0000_ffff,                                    // end
    ];
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();

    let program = iridia_d3d9::lift(&bytes)?;
    assert_eq!(program.stage, iridia::Stage::Pixel);
    program.validate().expect("a lifted program validates");
    assert!(iridia::emit::wgsl(&program)?.contains("@fragment"));

    // anything outside the subset is refused, with its word offset
    let mut other_model = bytes.clone();
    other_model[..4].copy_from_slice(&0xffff_0200u32.to_le_bytes()); // ps_2_0
    assert!(iridia_d3d9::lift(&other_model).is_err());
    Ok(())
}
```

## Guarantees and limits

- Unknown models, operations, register banks, relative addressing of anything but a vertex float constant by `a0`, loop counters, predication, control flow, structured, integer and boolean constant layouts and sampler arrays are refused with the word offset and the reason; so are uninitialized temporary lanes and undeclared input lanes. Vertex texture fetch and point size are unsupported. This does not promise general Shader Model 1 and 2 or every legal Shader Model 3 program.
- Arithmetic follows Microsoft's [ps_3_0 instruction reference](https://learn.microsoft.com/windows/win32/direct3dhlsl/dx9-graphics-reference-asm-ps-instructions-ps-3-0).
- It checks tokens and numbers, not rendered pixels: nothing here is compared against a Direct3D 9 device.

## Tests

`cargo test -p iridia-d3d9` needs no game files. `tests/tokens.rs` runs hand-worked token and arithmetic fixtures through the executor (expected numbers come from the instruction definitions, not from a second decoder); `tests/vertex.rs`, `vertex_output.rs`, `projection.rs`, `frame.rs` and `sampler.rs` do the same for their parts, and the `*versions.tsv` tables gate the lifter version. With `GAME_DATA` naming a folder of your own extracted programs (see [`tests/inputs/README.md`](tests/inputs/README.md)), `tests/disassembly.rs` compares every executable opcode and operand, the input, output and sampler declarations, `def` values and CTAB allocations against Microsoft's `D3DDisassemble` (the thin harness `tests/oracles/disassemble.py` loads `d3dcompiler_47.dll` and has no shader-format logic), then validates and writes the lift as WGSL.

## Check the examples

From the repository root, `cargo test -p iridia-d3d9 --doc` checks the crate documentation, and
`cargo build -p iridia-d3d9 --examples` builds its example programs. Examples that read paths need the caller's files; the repository ships no game assets.

## Licence

GPL-3.0-or-later.
