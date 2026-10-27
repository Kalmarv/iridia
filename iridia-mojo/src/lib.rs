//! Lift the GLSL dialect MojoShader writes from Shader Model 2 and 3 programs to an [`iridia::Program`].
//!
//! [MojoShader](https://icculus.org/mojoshader/) translates Direct3D 9 shader bytecode to GLSL for OpenGL, and engines
//! (MonoGame's compiled effects, for one) carry their shaders in that form. This crate reads that translation back: it
//! lifts the text into a `naga::Module` with its interface, which can then be validated and written as WGSL
//! ([`iridia::emit::wgsl`]) or handed to any naga backend. It is not a GLSL front end. It reads exactly what the
//! translation writes and refuses anything else by name ([`Error`], an [`iridia::Refusal`] naming the line).
//!
//! The dialect: no `#version` (GLSL 1.10); `attribute` and `varying` declarations; registers as global `vec4` variables
//! (`ps_r0`, and `vs_a0`, an `ivec4`); constants as `const` declarations and `#define` aliases of a
//! `uniform vec4 ps_uniforms_vec4[N]` array; one statement per instruction on swizzled registers; `if`/`else`; `rep`
//! loops with constant counts; relative addressing through `vs_a0`; `texture2D`; and the handful of built-in functions
//! the translation emits.
//!
//! What the program is:
//! - a vertex program reads its attributes (each a `vec4` argument named after the semantic the caller states in
//!   [`Hints`]) and writes `gl_Position` (the system value [`iridia::SystemValue::ClipPosition`]) and its varyings;
//! - a pixel program reads varyings and writes `gl_FragColor` or `gl_FragData[n]` (targets 0 to 3);
//! - constants are the uniform array, a constant buffer `cb0` of `vec4<u32>` rows read as floats (an index past the
//!   array reads zero, as D3D9's relative addressing does); samplers are a texture `t<slot>` and a sampler `s<slot>`;
//! - MojoShader ends every vertex program with three statements that fix the clip space for OpenGL (`posFixup`: flip
//!   y, half-pixel offset, depth 0..1 to -1..1). They are the API's, not the program's: they are dropped when they are
//!   exactly those three, the program keeps D3D's clip space ([`iridia::Conventions::D3D`]), and a note says it did.
//!
//! The text does not say what the constants are called, which texture a sampler is bound to or which semantic an
//! attribute carries; the caller states that in [`Hints`].
//!
//! # Example
//!
//! ```
//! use iridia::Stage;
//! use iridia_mojo::{Constant, Hints, Sampler, lift};
//!
//! let source = "uniform vec4 ps_uniforms_vec4[1];
//! vec4 ps_r0;
//! #define ps_c0 ps_uniforms_vec4[0]
//! uniform sampler2D ps_s0;
//! varying vec4 vTexCoord0;
//! #define ps_v0 vTexCoord0
//! #define ps_oC0 gl_FragColor
//!
//! void main()
//! {
//!     ps_r0 = texture2D(ps_s0, ps_v0.xy);
//!     ps_oC0 = ps_r0 * ps_c0.xxxx;
//! }
//! ";
//! let hints = Hints {
//!     target: "opengl effect".into(),
//!     constants: vec![Constant { name: "Brightness".into(), offset: 0, bytes: 16, width: 1 }],
//!     samplers: vec![Sampler { glsl_name: "ps_s0".into(), slot: 0, name: "Albedo".into() }],
//!     attributes: Vec::new(),
//! };
//!
//! let program = lift(source, Stage::Pixel, &hints)?;
//! program.validate().expect("a lifted program validates");
//! let wgsl = iridia::emit::wgsl(&program)?;
//! assert!(wgsl.contains("textureSample"));
//!
//! // anything outside the dialect is refused, with its line
//! let refusal = lift("void main() { gl_FragColor = vec4(1.0) + vec3(1.0); }", Stage::Pixel, &hints).unwrap_err();
//! assert_eq!(refusal.bytecode, "MojoShader GLSL");
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
#![deny(clippy::disallowed_methods, clippy::disallowed_types)]
#![cfg_attr(
    not(test),
    deny(
        clippy::indexing_slicing,
        clippy::arithmetic_side_effects,
        clippy::cast_possible_truncation
    )
)]

// Run the examples in the README as doc tests.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeExamples;

mod lex;
mod lower;

use iridia::{Program, Stage};

/// Why a program is refused: the line of the source and what in it is outside the dialect
/// ([`iridia::Refusal`]).
pub type Error = iridia::Refusal;

pub type Result<T> = std::result::Result<T, Error>;

pub(crate) fn error(line: usize, detail: impl Into<String>) -> Error {
    Error::new("MojoShader GLSL", iridia::Unit::Line, line, detail)
}

/// What the caller knows about the program that its text does not say.
#[derive(Clone, Debug, Default)]
pub struct Hints {
    /// The source's name for the target (`opengl effect v12`), kept in the program's provenance.
    pub target: String,
    /// The program's constants: the layout of the uniform array.
    pub constants: Vec<Constant>,
    /// The samplers the program declares.
    pub samplers: Vec<Sampler>,
    /// The vertex attributes the program declares.
    pub attributes: Vec<Attribute>,
}

/// A named constant at a byte offset of the uniform array, `bytes` long (a whole number of registers), `width` lanes
/// wide when it is a scalar or a vector. A program declares the rows of the array it reads: a constant past them is
/// left out, one that reaches past them is cut.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Constant {
    pub name: String,
    pub offset: u32,
    pub bytes: u32,
    pub width: u8,
}

/// A sampler: the name the program gives it (`ps_s0`), the slot the texture is bound at and the name of what is bound.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Sampler {
    pub glsl_name: String,
    pub slot: u32,
    pub name: String,
}

/// A vertex attribute: the name the program gives it (`vs_v0`) and the semantic and index it is bound by (`TEXCOORD`,
/// 1).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Attribute {
    pub glsl_name: String,
    pub semantic: String,
    pub index: u32,
}

/// The version of the lifter: bumped when what it produces for the same text changes, so a cache keyed by the text's
/// hash knows what is stale; gated by `tests/lift.rs` (the programs there lifted and hashed under it, `iridia::gate`).
pub const LIFTER_VERSION: u32 = 3;

/// Lift the vertex or pixel program `source`. Only [`Stage::Vertex`] and [`Stage::Pixel`] are programs of this
/// dialect.
pub fn lift(source: &str, stage: Stage, hints: &Hints) -> Result<Program> {
    let mut program = lower::lift(source, stage, hints)?;
    program.provenance.source_hash = Some(iridia::source_hash(source.as_bytes()));
    Ok(program)
}
