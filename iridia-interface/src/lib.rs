//! What a compiled shader program exposes, as plain data: its inputs and outputs, its resources (constant buffers with
//! explicit layouts, textures, samplers, buffers, storage) and the stage's own facts, the fixed-function state it is drawn
//! with, and what the analyses of `iridia` observe of its code, independent of any graphics API.
//!
//! A reader maps its API's reflection into this once and keeps the API's own names and addresses in provenance, so a
//! consumer that only needs meaning (a binding step, a validator, a writer) never sees an API's enum numbers. The
//! rule for differences between APIs: conventions are normalized (D3D register packing is unpacked into one varying
//! per label, D3D, std140 and DX9 constant packings all become explicit byte offsets and strides) and the original is
//! kept in provenance; behaviour that differs is not hidden here at all (that is the program body's business). A value
//! a reader cannot map is refused by name, never mapped to a guess.
//!
//! Everything is plain data: owned strings, integers and enums, no handles and no GPU types, and every type
//! implements `serde::Serialize`, so the vocabulary can be shown, stored and compared. What a program's inputs,
//! constants and outputs stand for (a view-projection matrix, an albedo) is not here: that is decided by whoever binds
//! the program to a renderer or a baker, in its own terms.
//!
//! # Contents
//!
//! - [`Interface`]: the [`Stage`], the [`Varying`]s a program takes and gives (by [`Label`], with a [`SystemValue`]
//!   where the pipeline supplies it, an [`Interpolation`] and the D3D register and lanes in [`VaryingProvenance`]), its
//!   [`Resource`]s and the per-stage [`StageFacts`] (tessellation, geometry, mesh and ray tracing facts).
//! - [`Resource`] and [`ResourceKind`]: constant buffers with a [`BufferLayout`] of [`Constant`]s whose [`ConstantType`]
//!   states every stride, textures, samplers, buffers and storage.
//! - [`ShaderFormat`]: how a program's bytes are encoded; [`InstructionCount`]: how large a program is.
//! - [`RenderState`] (blend, depth, stencil, cull, fill, alpha test) and [`SamplerState`] (filters, addressing,
//!   comparison): Direct3D's fixed-function state, as a source states it.
//! - What a program's code shows, as `iridia`'s analyses read it: [`MatrixLines`] (whether a constant's lines are a
//!   matrix's rows or its columns), [`AlphaComparison`] (what a discard compares a constant with) and [`OutputUse`]
//!   (what tests or blends an output's lanes); and [`ConstantBufferBinding`], the size a source binds a constant buffer
//!   with.
//!
//! # Example
//!
//! An interface is built by a reader (`iridia-dxbc`, `iridia-dxil`), but it is only data; this one is written by hand and
//! serialized:
//!
//! ```
//! use iridia_interface::{
//!     Interface, Label, ScalarType, Stage, StageFacts, SystemValue, Value, Varying, VaryingProvenance,
//! };
//!
//! let colour = Varying {
//!     label: Label { name: "SV_Target".into(), index: 0 },
//!     system: Some(SystemValue::Target(0)),
//!     value: Value { scalar: ScalarType::Float, width: 4 },
//!     interp: None,
//!     used: true,
//!     provenance: VaryingProvenance::default(),
//! };
//! let interface = Interface {
//!     stage: Stage::Pixel,
//!     inputs: Vec::new(),
//!     outputs: vec![colour],
//!     resources: Vec::new(),
//!     facts: StageFacts::None,
//! };
//! let json = serde_json::to_string(&interface)?;
//! assert!(json.contains(r#""stage":"pixel""#));
//! # Ok::<(), serde_json::Error>(())
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

mod binding;
mod facts;
mod interface;
mod resource;
mod stage;
mod state;

pub use binding::{AlphaComparison, ConstantBufferBinding, MatrixLines, OutputUse, Unshown};
pub use facts::{InstructionCount, InstructionUnit};
pub use interface::{
    D3dSignature, DomainFacts, GeometryFacts, GsInput, GsOutput, HitGroup, HitGroupKind, HullFacts,
    Interface, InterpMode, InterpSampling, Interpolation, Label, MeshFacts, MeshTopology, Partitioning,
    RayTracingFacts, ScalarType, StageFacts, Subobject, SystemValue, TessDomain, TessTopology, Value,
    Varying, VaryingProvenance,
};
pub use resource::{
    Access, BufferKind, BufferLayout, Constant, ConstantType, D3dBinding, D3dClass, Majorness, Member,
    Resource, ResourceId, ResourceKind, ResourceProvenance, SampleType, StorageTarget, TextureDim,
};
pub use stage::{ShaderFormat, Stage};
pub use state::{
    AddressMode, AlphaTest, Blend, BlendEquation, BlendFactor, BlendOp, Compare, Cull, DepthBias, DepthState,
    Fill, Filter, Filters, MipFilter, RenderState, SamplerState, Stencil, StencilFace, StencilOp,
    TargetState,
};
