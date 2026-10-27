//! Lift Direct3D Shader Model 6 programs (DXIL) to an [`iridia::Program`], every stage a DXIL container holds but
//! amplification, and read their reflection, signatures and a library's subobjects into the `iridia_interface`
//! vocabulary.
//!
//! The container, the LLVM bitcode of its `DXIL` part and the reflection and runtime-data records are read by
//! [`dxil_reader`]; this crate maps them. A DXIL function is LLVM's: values in single static assignment form over basic
//! blocks. The lifter turns them into structured control flow (an irreducible graph is refused by name), reads the
//! `dx.op` intrinsics (inputs and outputs, math, bit operations, constant buffers, textures, derivatives, wave, mesh,
//! ray query and ray tracing operations) and writes Direct3D's results where they differ from naga's out as
//! operations ([`iridia::D3dModes`]). The conventions are the same as `iridia-dxbc`'s, so a program lifted from either
//! has the same shape.
//!
//! [`interface`] reads a container's interface alone, without lifting its program, and [`subobjects`] maps the
//! subobjects of a ray tracing library.
//!
//! # Example
//!
//! ```
//! let container = include_bytes!("../tests/fixtures/flow.dxil"); // `dxc -T ps_6_0` output
//! let program = iridia_dxil::lift(container)?;
//!
//! assert_eq!(program.stage, iridia::Stage::Pixel);
//! assert_eq!(program.provenance.target, "ps_6_0");
//! // two colour targets, a float4 and a uint4, as the HLSL declares
//! let targets: Vec<_> = program.interface.outputs.iter().map(|v| (v.system, v.value.width)).collect();
//! assert_eq!(
//!     targets,
//!     [(Some(iridia::SystemValue::Target(0)), 4), (Some(iridia::SystemValue::Target(1)), 4)]
//! );
//!
//! let wgsl = iridia::emit::wgsl(&program)?;
//! assert!(wgsl.contains("@fragment"));
//!
//! // a refusal names the bit it is at and why
//! let refusal = iridia_dxil::lift(&container[..64]).unwrap_err();
//! assert_eq!(refusal.bytecode, "DXIL");
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
#![deny(missing_docs)]
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
mod lift;
mod reflection;
pub use lift::{LIFTER_VERSION, lift, lift_with};
pub use reflection::{interface, subobjects};

/// A DXIL program or container refused or unreadable: the bit it is at (in the bitcode, or eight times a byte of the
/// container) and why ([`iridia::Refusal`]).
pub type Error = iridia::Refusal;
/// The result of every function in this crate: `Ok` or a refusal that names the bit of the DXIL module.
pub type Result<T> = std::result::Result<T, Error>;
pub(crate) fn error(bit: usize, detail: impl Into<String>) -> Error {
    Error::new("DXIL", iridia::Unit::Bit, bit, detail)
}

/// What the reader could not read, as a refusal of the program at the same bit.
pub(crate) fn read(e: dxil_reader::Error) -> Error {
    error(e.bit, e.detail)
}
