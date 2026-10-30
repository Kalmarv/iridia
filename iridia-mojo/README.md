# iridia-mojo

Lifts the GLSL dialect that [MojoShader](https://icculus.org/mojoshader/) writes when it translates Shader Model 2 and 3 programs for OpenGL into [`iridia::Program`](../iridia), a [naga](https://github.com/gfx-rs/wgpu/tree/trunk/naga) module with its interface.

Part of [iridia](https://github.com/Kalmarv/iridia), which turns compiled Direct3D shaders into naga modules. Some engines ship effects compiled to this GLSL (MonoGame's, for one); this crate reads that text back. It is not a GLSL front end: it reads exactly what the translation writes and refuses anything else by name (`Error`, an `iridia::Refusal` naming the line). It was written against the effects of an XNA/MonoGame game.

## The dialect

- `#ifdef GL_ES` precision blocks (skipped), `#extension GL_OES_standard_derivatives`, and `#define` aliases of registers (expanded);
- `uniform vec4 <stage>_uniforms_vec4[N]` (the constants: a uniform buffer of `vec4<u32>` rows read as floats; an index past the array reads zero), `uniform sampler2D`, `attribute vec4` (vertex), `varying vec4` (`vFrontColor`, `vBackColor`, `vFogFactor`, `vTexCoord<n>`), `const vec4` and `const ivec4` constants, and `vec4` and `ivec4` registers;
- `main`: assignments to swizzled registers, `if`/`else`, `rep` loops of constant count, `discard`, `?:`, relative addressing through an `ivec4` register, `texture2D`, `dFdx`/`dFdy`, `any`, `lessThan`, constructors and the math functions the translation emits.

A vertex program reads its attributes and writes `gl_Position` (`SystemValue::ClipPosition`) and its varyings; a pixel program reads varyings and writes `gl_FragColor` or `gl_FragData[n]` (targets 0 to 3). Constants are the constant buffer `cb0`; a sampler is a texture `t<slot>` and a sampler `s<slot>`.

## What the caller states

The text does not say what its constants are called, which texture a sampler is bound to or which semantic an attribute carries. `Hints` names them: the target (`opengl effect v12`, kept in the program's provenance), the constants' layout, each sampler's slot and the name of what is bound to it, each attribute's semantic and index. Constants that lie past the uniform rows the program declares are left out and the ones that reach past them are cut; the program's notes count both.

## What is dropped, and what is refused

MojoShader ends every vertex program with three statements that fix the clip space for OpenGL (`posFixup`: flip y, half-pixel offset, depth 0..1 to -1..1). They are the API's, not the program's: they are dropped when they are exactly those three, the program keeps Direct3D's clip space (`Conventions::D3D`), and a note says so. `posFixup` anywhere else is refused.

Refused by name, with the line: a name declared twice, a number that is not finite, an operation on vectors of different widths (`vec2 + vec3`, the branches of `?:`, the arguments of a function), and anything else the dialect does not have.

## Example

```rust
use iridia::Stage;
use iridia_mojo::{Constant, Hints, Sampler, lift};

const SOURCE: &str = "uniform vec4 ps_uniforms_vec4[1];
vec4 ps_r0;
#define ps_c0 ps_uniforms_vec4[0]
uniform sampler2D ps_s0;
varying vec4 vTexCoord0;
#define ps_v0 vTexCoord0
#define ps_oC0 gl_FragColor

void main()
{
	ps_r0 = texture2D(ps_s0, ps_v0.xy);
	ps_oC0 = ps_r0 * ps_c0.xxxx;
}
";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let hints = Hints {
        target: "opengl effect".into(),
        constants: vec![Constant { name: "Brightness".into(), offset: 0, bytes: 16, width: 1 }],
        samplers: vec![Sampler { glsl_name: "ps_s0".into(), slot: 0, name: "Albedo".into() }],
        attributes: Vec::new(),
    };

    let program = lift(SOURCE, Stage::Pixel, &hints)?;
    program.validate().expect("a lifted program validates");
    let wgsl = iridia::emit::wgsl(&program)?;
    assert!(wgsl.contains("textureSample"));

    // anything outside the dialect is refused, with its line
    let refusal = lift("void main() { gl_FragColor = vec4(1.0) + vec3(1.0); }", Stage::Pixel, &hints).unwrap_err();
    assert_eq!(refusal.bytecode, "MojoShader GLSL");
    Ok(())
}
```

## Tests

`cargo test -p iridia-mojo` needs no game files. `tests/lift.rs` lifts programs written as MojoShader writes them (a gamma pixel program, a vertex program with the `posFixup` tail, one with relative addressing, a `rep` loop, `if`/`else` and `discard`), validates and writes them as WGSL, checks that each damaged program is refused by name, and compares a table of the lifted fixtures against the committed `tests/lifted.hashes` (the version gate). This crate declares no real-data inputs beyond the scratch folder ([`tests/inputs/README.md`](tests/inputs/README.md)).

## Check the examples

From the repository root, `cargo test -p iridia-mojo --doc` checks the crate documentation, and
`cargo build -p iridia-mojo --examples` builds its example programs. Examples that read paths need the caller's files; the repository ships no game assets.

## Licence

GPL-3.0-or-later.
