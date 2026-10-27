//! Read raw Direct3D 9 `ps_3_0` and `vs_3_0` shader tokens and their CTAB constant table, and lift them to an
//! [`iridia::Program`].
//!
//! The input is the token stream a Direct3D 9 compiler writes, version token to `END` (no container). [`read`] decodes
//! it into declarations, instructions, immediate definitions, every comment and the CTAB (creator, target, flags, named
//! register allocations, types and raw defaults). [`lift`] turns it into a `naga::Module` with an interface, which can
//! then be validated and written as WGSL ([`iridia::emit::wgsl`]). No translated text and no native compiler is needed.
//!
//! The reader is bounded: straight-line float arithmetic (`mov`, `add`, `sub`, `mad`, `mul`, `rcp`, `rsq`, `dp3`, `dp4`,
//! `min`, `max`, `exp`, `log`, `lrp`, `frc`, `pow`, `abs`, `nrm`), `nop`, `dcl` and `def`; pixel programs also have
//! `cmp`, `dp2add`, `texld` (plain, projected or biased) and `texkill`, with 2D, cube or volume textures; vertex programs
//! keep their declared inputs and outputs and the clip position. Source swizzles, negation, absolute value and
//! destination masks and saturation are kept. Vertex texture fetch, point size, other models, control flow,
//! relative addressing, predication and modifiers outside this subset are refused by name and word offset
//! ([`Error`]).
//!
//! The float constant bank is `cb0`, an array of raw `vec4<u32>` rows: register `cN` is byte offset `16 * N`, and a
//! `def` overrides it locally. Sampler `sN` becomes a texture at `tN` and a sampler at `sN`. Vertex register banks are
//! bounded to 16 inputs, 32 temporaries, 12 outputs and 256 float constants.
//!
//! Beyond [`lift`]: [`frame`] finds the program in a buffer that holds more than one, [`sampler_states`] maps a sampler
//! state (`D3DSAMP` state/value pairs) to the interface vocabulary, and [`project`], [`reencode`] and
//! [`project_vertex_output`] derive new programs from the original (a register's value at a chosen instruction, a
//! vertex output as a pixel program).
//!
//! # Example
//!
//! A `ps_3_0` program written as tokens: `def c0, 1, 2, 3, 4` and `mov oC0, c0`.
//!
//! ```
//! let words: [u32; 11] = [
//!     0xffff_0300,                                    // ps_3_0
//!     0x0500_0051, 0xa00f_0000, 0x3f80_0000, 0x4000_0000, 0x4040_0000, 0x4080_0000, // def c0, 1, 2, 3, 4
//!     0x0200_0001, 0x800f_0800, 0xa0e4_0000,          // mov oC0, c0
//!     0x0000_ffff,                                    // end
//! ];
//! let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
//!
//! let program = iridia_d3d9::lift(&bytes)?;
//! assert_eq!(program.stage, iridia::Stage::Pixel);
//! program.validate().expect("a lifted program validates");
//! assert!(iridia::emit::wgsl(&program)?.contains("@fragment"));
//!
//! // anything outside the subset is refused, with its word offset
//! let mut other_model = bytes.clone();
//! other_model[..4].copy_from_slice(&0xffff_0200u32.to_le_bytes()); // ps_2_0
//! assert!(iridia_d3d9::lift(&other_model).is_err());
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! Token layout: Microsoft's [instruction](https://learn.microsoft.com/windows-hardware/drivers/display/instruction-token),
//! [source](https://learn.microsoft.com/windows-hardware/drivers/display/source-parameter-token) and
//! [destination](https://learn.microsoft.com/windows-hardware/drivers/display/destination-parameter-token) tokens;
//! [the VS register reference](https://learn.microsoft.com/windows/win32/direct3dhlsl/dx9-graphics-reference-asm-vs-registers-vs-3-0)
//! and [output declarations](https://learn.microsoft.com/windows/win32/direct3dhlsl/dcl-usage-outut-register---vs) define
//! their layout. Arithmetic follows the [ps_3_0 instruction reference](https://learn.microsoft.com/windows/win32/direct3dhlsl/dx9-graphics-reference-asm-ps-instructions-ps-3-0).
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

mod ctab;
mod lower;
mod sampler;
pub use sampler::sampler_states;
mod tokens;
mod vertex_output;

pub use ctab::{Constant, ConstantTable, Type, TypeMember};
pub use tokens::{Comment, Destination, Input, Instruction, Opcode, Output, Register, Shader, Source, read};

/// A refusal states the word offset in the raw token stream and the unsupported or malformed feature.
pub type Error = iridia::Refusal;
pub type Result<T> = std::result::Result<T, Error>;
pub(crate) fn error(at: usize, detail: impl Into<String>) -> Error {
    Error::new("D3D9", iridia::Unit::Word, at, detail)
}

/// Increment when the same tokens lift differently. Hand-worked fixtures gate this version.
pub const LIFTER_VERSION: u32 = 2;

/// Lift one complete `ps_3_0` or `vs_3_0` program; trailing bytes and unknown features are errors.
pub fn lift(bytes: &[u8]) -> Result<iridia::Program> {
    lower::lift(&read(bytes)?, bytes)
}

/// A register value immediately after one original instruction. `lanes` order is the projected RGB order.
/// The word offset must identify an instruction, and all three distinct temporary lanes must be initialized.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Projection {
    pub after_word: usize,
    pub temporary: u32,
    pub lanes: [u8; 3],
}

/// Pixel-only: lift the original instruction prefix through `cut`, returning its selected lanes as RGB and alpha one.
/// Prior discards remain; subsequent operations/discards are excluded. This is a derived program, whose notes
/// name the cut, whose lifter identity includes the projection, and whose source hash still identifies the complete original bytecode. No meaning is assigned
/// to the chosen value: recognizing a useful cut is the caller's responsibility.
pub fn project(bytes: &[u8], cut: Projection) -> Result<iridia::Program> {
    lower::project(&read(bytes)?, bytes, cut, false)
}

/// Pixel-only: lift the complete original program, replacing just the cut's three register lanes with the new float3
/// input `DERIVED_RGB0` (signature register 31). This runs the original downstream operations, including
/// normalization and discards, rather than a rewritten equation. Other register lanes and the prefix remain.
/// An original input v31 is refused. The caller supplies projected RGB in that new input's xyz lanes.
pub fn reencode(bytes: &[u8], cut: Projection) -> Result<iridia::Program> {
    lower::project(&read(bytes)?, bytes, cut, true)
}

/// Frame the first raw SM3 vertex or pixel program in a container's remaining bytes.
/// Returns its stage and exact bytes through END; subsequent container metadata is not consumed.
/// This checks token/comment spans, not instruction semantics: an unsupported operation is retained as
/// bytecode for a later lifter to refuse. It never searches for END inside operands or comments.
pub fn frame(bytes: &[u8]) -> Result<(iridia::Stage, &[u8])> {
    use iridia::Stage;
    let word = |at: usize| -> Result<u32> {
        let b = bytes
            .get(at..at.saturating_add(4))
            .ok_or_else(|| error(at / 4, "truncated raw program token"))?;
        let a: [u8; 4] = b
            .try_into()
            .map_err(|_| error(at / 4, "truncated raw program word"))?;
        Ok(u32::from_le_bytes(a))
    };
    let stage = match word(0)? {
        0xfffe_0300 => Stage::Vertex,
        0xffff_0300 => Stage::Pixel,
        version => {
            return Err(error(
                0,
                format!("raw program version {version:#x}, expected vs_3_0 or ps_3_0"),
            ));
        }
    };
    let mut at = 4usize;
    while at / 4 < 1_048_576 {
        let token = word(at)?;
        let token_word = at / 4;
        at = at.saturating_add(4);
        if token == 0xffff {
            let program = bytes
                .get(..at)
                .ok_or_else(|| error(token_word, "raw program END is past the container"))?;
            return Ok((stage, program));
        }
        let count = if token & 65535 == 65534 {
            (token >> 16) & 32767
        } else {
            (token >> 24) & 15
        };
        let end = at.saturating_add((count as usize).saturating_mul(4));
        if end > bytes.len() {
            return Err(error(
                token_word,
                format!(
                    "raw program operand/comment span of {count} words exceeds the container's {} bytes",
                    bytes.len()
                ),
            ));
        }
        at = end;
    }
    Err(error(at / 4, "raw program exceeds 1,048,576 words or lacks END"))
}

/// Project selected lanes of a declared `vs_3_0` output into a derived invocation program.
///
/// The complete original program must lift successfully. A backwards register/lane slice retains only the
/// writes that reach the selected output. It preserves original arithmetic, swizzles, masks and local DEF values;
/// it does not interpret the output semantic. The result is pixel-stage IR for an independent invocation,
/// with original vertex inputs and constant-register bindings. Target0 keeps selected lanes in their original
/// RGBA positions; other lanes are explicitly placeholder (0,0,0,1). Other original outputs are excluded.
/// Full original source hash, a distinct lifter identity and a derivation note keep this separate from the VS.
/// Its original declarations remain in the interface; `used` and graph dependencies describe actual reads.
pub fn project_vertex_output(bytes: &[u8], register: u32, mask: u8) -> Result<iridia::Program> {
    vertex_output::project(bytes, register, mask)
}
