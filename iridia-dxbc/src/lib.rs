//! Lift Direct3D Shader Model 4 and 5 programs (DXBC) to an [`iridia::Program`], any stage, and read their reflection
//! and signatures into the `iridia_interface` vocabulary.
//!
//! The container and token stream are read by [`dxbc_reader`] and [`dxcontainer`]; this crate maps what they read.
//! [`lift`] turns a compiled vertex, pixel, hull, domain, geometry or compute program into a `naga::Module` with its
//! interface, which can then be validated and written as WGSL ([`iridia::emit::wgsl`]) or handed to any naga backend.
//! A register is four raw 32-bit lanes and each instruction reinterprets them as the type it implies; Direct3D's
//! results where they differ from naga's (division by zero, float to integer conversion, `min`/`max` and `saturate`
//! with NaN) are written out as operations and stated in [`iridia::D3dModes`]. What the lifter does not carry is
//! refused, naming the instruction word and the reason; a malformed program is an `Err`, never a panic.
//!
//! The reflection functions read a container's RDEF and signature chunks without lifting its program:
//! [`reflection`] (everything), [`signatures`] (for a container whose RDEF was stripped, as some engines ship them) and
//! [`resource_reflection`] (the resources alone).
//!
//! # Example
//!
//! ```
//! let container = include_bytes!("../tests/fixtures/modes.dxbc"); // `fxc /T ps_5_0` output
//! let program = iridia_dxbc::lift(container)?;
//!
//! assert_eq!(program.stage, iridia::Stage::Pixel);
//! assert_eq!(program.provenance.target, "ps_5_0");
//! // the constant buffer `C : register(b0)` of the HLSL, with the names the container's reflection gives its fields
//! let iridia::ResourceKind::ConstantBuffer { layout } = &program.interface.resources[0].kind else {
//!     unreachable!("the only resource is a constant buffer");
//! };
//! let fields: Vec<_> = layout.constants.iter().map(|c| (c.name.as_str(), c.offset)).collect();
//! assert_eq!(fields, [("f", 0), ("u", 16), ("i", 32)]);
//!
//! let wgsl = iridia::emit::wgsl(&program)?;
//! assert!(wgsl.contains("@fragment"));
//!
//! // a refusal names where and why
//! let refusal = iridia_dxbc::lift(&container[..40]).unwrap_err();
//! assert_eq!(refusal.bytecode, "DXBC");
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! The container's digest is not verified: this reader does not authenticate shaders.
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
pub use reflection::{reflection, resource_reflection, signatures, varying};

/// A DXBC program or container refused or unreadable: the word it is at and why ([`iridia::Refusal`]).
pub type Error = iridia::Refusal;
/// The result of every function in this crate: `Ok` or a refusal that names the word of the shader code it is at.
pub type Result<T> = std::result::Result<T, Error>;
pub(crate) fn error(word: usize, detail: impl Into<String>) -> Error {
    Error::new("DXBC", iridia::Unit::Word, word, detail)
}

/// What the reader could not read, as a refusal of the program at the same word.
pub(crate) fn read(e: dxbc_reader::Error) -> Error {
    error(e.word, e.detail)
}

/// The parts of a DXBC container, an error of the container as one of the program (the word it is at).
pub(crate) fn parts(bytes: &[u8]) -> Result<Vec<dxcontainer::Chunk<'_>>> {
    dxbc_reader::parts(bytes).map_err(read)
}
