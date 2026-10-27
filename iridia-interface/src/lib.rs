//! What a compiled shader program exposes, as plain data: its inputs and outputs, its resources (constant buffers with
//! explicit layouts, textures, samplers, buffers, storage) and the roles a renderer binds them to, independent of any
//! graphics API.
//!
//! A reader maps its API's reflection into this once and keeps the API's own names and addresses in provenance, so a
//! consumer that only needs meaning (a binding step, a validator, a writer) never sees an API's enum numbers. The
//! rule for differences between APIs: conventions are normalized (D3D register packing is unpacked into one varying
//! per label, D3D, std140 and DX9 constant packings all become explicit byte offsets and strides) and the original is
//! kept in provenance; behaviour that differs is not hidden here at all (that is the program body's business). A value
//! a reader cannot map is refused by name, never mapped to a guess.
//!
//! Everything is plain data: owned strings, integers and enums, no handles and no GPU types, and every type
//! implements `serde::Serialize`, so the vocabulary can be shown, stored and compared.
//!
//! # Contents
//!
//! - [`Interface`]: the [`Stage`], the [`Varying`]s a program takes and gives (by [`Label`], with a [`SystemValue`]
//!   where the pipeline supplies it, an [`Interpolation`] and the D3D register and lanes in [`VaryingProvenance`]), its
//!   [`Resource`]s and the per-stage [`StageFacts`] (tessellation, geometry, mesh and ray tracing facts).
//! - [`Resource`] and [`ResourceKind`]: constant buffers with a [`BufferLayout`] of [`Constant`]s whose [`ConstantType`]
//!   states every stride, textures, samplers, buffers and storage.
//! - [`ShaderFormat`]: how a program's bytes are encoded.
//! - [`InputBinding`], [`ConstantBinding`], [`OutputBinding`], [`RenderState`] and their role enums: what a consumer
//!   decides a program's inputs, constants, textures and outputs mean (a view-projection matrix, an albedo), and the
//!   fixed-function state to draw it with. The readers do not produce these; whoever drives a renderer or baker does.
//! - [`surface`]: one reading of a pixel program's outputs, by their roles, as a lit surface.
//! - [`ProgramFacts`]: a small summary of a program a consumer keeps.
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
#![deny(clippy::disallowed_methods, clippy::disallowed_types)]

// Run the examples in the README as doc tests.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeExamples;

mod binding;
mod facts;
mod interface;
mod resource;
mod stage;
pub mod surface;

pub use binding::{
    AddressMode, AlphaComparison, AlphaTest, Blend, BlendEquation, BlendFactor, BlendOp, BufferBinding,
    BufferSource, BufferTable, Compare, ConstantBinding, ConstantBufferBinding, ConstantRole, Cull,
    DepthBias, DepthState, Fill, FrameParameter, FrameTexture, InputBinding, InputRole, KeyPart, LaneAffine,
    LaneStream, MatrixLines, NormalEncoding, NormalSpace, ObjectMatrix, ObjectParameter, Occluded,
    OutputBinding, OutputRole, RenderState, SamplerState, ScalarEncoding, SkinPrefix, Stencil, StencilFace,
    StencilOp, TargetState, Unshown, VariantKey, ViewParameter,
};
pub use binding::{Attribute, Filter, Filters, MipFilter};
pub use facts::{InstructionCount, InstructionUnit, ProgramFacts, ProgramResourceFacts};
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
pub use surface::Surface;
