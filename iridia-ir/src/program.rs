//! A program: the entry function with the interface and what a function cannot say.

use alloc::{string::String, vec::Vec};

use crate::{
    Function,
    interface::{InstructionCount, Interface, ShaderFormat, Stage},
};

/// Which way clip space's depth runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum ClipDepth {
    /// 0 at the near plane, 1 at the far plane (D3D, Vulkan, Metal, WebGPU).
    ZeroToOne,
    /// -1 at the near plane, 1 at the far plane (OpenGL).
    MinusOneToOne,
}

/// Which way a vertical axis runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum Corner {
    /// Row 0 is at the top (Direct3D, Vulkan, Metal, WebGPU).
    TopLeft,
    /// Row 0 is at the bottom (OpenGL).
    BottomLeft,
}

/// The pipeline rules a program was written for, applied at its boundary by whoever runs it on other rules.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub struct Conventions {
    /// Which way clip-space depth runs.
    pub clip_depth: ClipDepth,
    /// Where window space puts its origin.
    pub fragment_origin: Corner,
    /// Where a texture's first row is.
    pub texture_origin: Corner,
}

impl Conventions {
    /// Direct3D: depth 0 to 1, window and texture origin at the top left.
    pub const D3D: Self = Self {
        clip_depth: ClipDepth::ZeroToOne,
        fragment_origin: Corner::TopLeft,
        texture_origin: Corner::TopLeft,
    };
}

/// What a float type's denormals do.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize)]
pub enum Denormals {
    /// The source does not say; either is correct.
    #[default]
    Any,
    /// Kept (IEEE 754).
    Preserve,
    /// Flushed to zero, sign kept, on input and output.
    Flush,
}

/// The program's floating-point controls, per float type (module defaults; an operation may override them where a
/// source has per-operation controls).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize)]
pub struct FloatEnv {
    /// `f16` denormals.
    pub f16: Denormals,
    /// `f32` denormals.
    pub f32: Denormals,
    /// `f64` denormals.
    pub f64: Denormals,
}

/// Where a program came from.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct Provenance {
    /// The encoding of the bytecode.
    pub format: ShaderFormat,
    /// The source's own name for its target (`ps_3_0`).
    pub target: String,
    /// The lifter that built it.
    pub lifter: String,
    /// The lifter's version, bumped when its output for the same bytes changes.
    pub lifter_version: u32,
    /// `sha256:` and the digits of the bytes it was lifted from, where the lifter computed it.
    pub source_hash: Option<String>,
    /// Source instruction records counted while decoding.
    pub instruction_count: Option<InstructionCount>,
    /// The compatibility policies the lifter chose between implementation-defined behaviours, as `name=value`,
    /// sorted: part of what a cache keys on.
    pub policies: Vec<String>,
}

/// A program: its entry function, interface, conventions, float controls, provenance and notes.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct Program {
    /// The stage it runs in; the same as `interface.stage`.
    pub stage: Stage,
    /// The entry function.
    pub entry: Function,
    /// A hull program's patch constant function: run once per patch after the entry has run for every output control
    /// point, it reads the input and output control points and stores the patch constants.
    pub patch: Option<Function>,
    /// The thread group a compute or mesh program runs in (`numthreads`), invocations along x, y and z; `None` for
    /// every other stage.
    pub workgroup: Option<[u32; 3]>,
    /// The varyings, resources and stage facts.
    pub interface: Interface,
    /// The pipeline rules it was written for.
    pub conventions: Conventions,
    /// Its floating-point controls.
    pub float: FloatEnv,
    /// Where it came from.
    pub provenance: Provenance,
    /// What the lifter could not state exactly, each once and sorted: shown beside the program.
    pub notes: Vec<String>,
}
