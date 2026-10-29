//! Compiled Direct3D shaders as [`naga`] modules, so they can be validated, inspected and written as WGSL, SPIR-V or
//! whatever else a naga backend writes.
//!
//! This is the crate every bytecode reader of the iridia workspace lifts into. A [`Program`] is a `naga::Module` with
//! what a module cannot say: an [`Interface`] (inputs and outputs by semantic name with their D3D register and lanes
//! kept in provenance, constant buffers with explicit layouts, textures, samplers and buffers), the clip-space and
//! texture-origin [`Conventions`] it was written for, which of Direct3D's numeric behaviours its body spells out
//! ([`D3dModes`]) and where it came from ([`Provenance`]). The readers are separate crates:
//! `iridia-dxbc` (Shader Model 4 and 5), `iridia-dxil` (Shader Model 6), `iridia-d3d9` (raw Shader Model 2 and 3
//! tokens) and `iridia-mojo` (the GLSL MojoShader writes from Shader Model 2 and 3).
//!
//! # Example
//!
//! Lift a compiled pixel shader, check it and write WGSL (`iridia-dxbc` is a development dependency of this crate; a
//! program of your own depends on it directly):
//!
//! ```
//! let container = include_bytes!("../../iridia-dxbc/tests/fixtures/modes.dxbc");
//! let program = iridia_dxbc::lift(container)?;
//!
//! assert_eq!(program.stage, iridia::Stage::Pixel);
//! assert_eq!(program.provenance.target, "ps_5_0");
//! // the constant buffer `C` of the HLSL source, named by the container's reflection
//! assert_eq!(program.interface.resources[0].provenance.name, "C");
//!
//! program.validate().expect("a lifted program validates");
//! let wgsl = iridia::emit::wgsl(&program)?;
//! assert!(wgsl.contains("@fragment"));
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # What is here
//!
//! - [`Program`], its [`validate`](Program::validate) (naga's validation, then the interface against the entry point and
//!   the resources against the module's globals) and [`text`](Program::text) (a stable text form, for diffs);
//! - [`emit`]: WGSL writers, and [`emit::wgsl_for`], WGSL as one pipeline binds it;
//! - [`build::Builder`], the typed builder over naga's IR the readers lift with, and [`build::structure`], which turns a
//!   control flow graph into structured control flow;
//! - [`d3d`]: Direct3D's enum numbers and packing rules mapped into the [`iridia_interface`] vocabulary, shared by the
//!   DXBC and DXIL readers;
//! - [`Refusal`]: what every reader returns for bytecode it cannot lift, naming where and why;
//! - [`gate`]: the version check a reader's tests use so that a changed output comes with a bumped version number;
//! - analyses of a lifted program, none of which runs it: [`flow`] (what each output lane is made of), [`matrix`] (how
//!   a constant buffer holds a matrix), [`output_stores`] and [`output_uses`] (what the entry point stores into its
//!   outputs and what a discard or blend shows a lane is used for), and [`Program::constant_reads`] (which constant
//!   words it reads);
//! - [`rt`]: the calls a ray tracing program makes in place of the pipeline's operations.
//!
//! Running a program on the CPU is the `iridia-exec` crate.
//!
//! The vocabulary types are those of [`iridia_interface`], re-exported here.
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
// ... and those of the repository's README.
#[cfg(doctest)]
#[doc = include_str!("../../README.md")]
struct RepositoryReadmeExamples;

pub mod bound;
pub mod build;
pub mod d3d;
pub mod emit;
pub mod flow;
pub mod gate;
mod limits;
pub mod matrix;
pub mod output_stores;
pub mod output_uses;
mod program;
mod reads;
pub mod rt;
pub use iridia_interface::{
    Access, BufferKind, BufferLayout, Constant, ConstantType, D3dBinding, D3dClass, D3dSignature,
    DomainFacts, GeometryFacts, GsInput, GsOutput, HitGroup, HitGroupKind, HullFacts, InstructionCount,
    InstructionUnit, Interface, InterpMode, InterpSampling, Interpolation, Label, Majorness, Member,
    MeshFacts, MeshTopology, Partitioning, RayTracingFacts, Resource, ResourceId, ResourceKind,
    ResourceProvenance, SampleType, ScalarType, ShaderFormat, Stage, StageFacts, StorageTarget, Subobject,
    SystemValue, TessDomain, TessTopology, TextureDim, Value, Varying, VaryingProvenance,
};
mod refusal;
pub use bound::BoundSizes;
pub use limits::{WEBGPU_BASELINE, compute_limit_notes, interface_notes};
pub use program::{
    ClipDepth, Conventions, D3dModes, EntryIo, Error as ProgramError, FnItem, FunctionSig, IoItem, IoType,
    ItemList, Origin, Problem, Program, Provenance, VERSION as PROGRAM_VERSION, entry_io, mesh_vertex_io,
    stage_functions,
};
pub use reads::{ProgramGraph, WordsRead};
pub use refusal::{Refusal, Unit, source_hash};

pub mod specialise;
