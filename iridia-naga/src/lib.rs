//! Lowers iridia's shader IR ([`iridia_ir::Program`]) to a stock naga module, validated, for naga's writers.
//!
//! A program reaches here after its passes: promoted, simplified, structured. [`module`] builds the module, with each
//! resource where a [`BindingPlan`] binds it and the interface carried by the entry point `main` (an argument per
//! input, a member of the result structure `Out` per output); [`validate`] runs naga's validator on it with a target
//! [`Profile`]'s capabilities; [`wgsl`] writes it as WGSL, and [`wgsl_for`] writes it for one pipeline (its bindings,
//! and a vertex program's outputs linked to the pixel program they feed). [`preflight`] refuses, before lowering, a
//! program that needs a capability the profile does not allow, so naga rejecting a module the preflight accepted is a
//! lowering bug.
#![no_std]
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

extern crate alloc;

mod io;
mod lower;
#[cfg(feature = "wgsl")]
mod pipeline;
mod profile;

use alloc::{format, string::String, vec::Vec};
use core::fmt;

use iridia_ir::interface::{D3dClass, Interface, ResourceId, ResourceKind};

pub use io::{EntryIo, Item, entry_io};
pub use lower::module;
#[cfg(feature = "wgsl")]
pub use pipeline::{Pipeline, Written, wgsl_for};
pub use profile::{Profile, Requirement, preflight, requirements};

/// Why a program could not be lowered, validated or written.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error(pub String);

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl core::error::Error for Error {}

/// Where one resource is bound.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Bind {
    /// The resource.
    pub resource: ResourceId,
    /// Its bind group.
    pub group: u32,
    /// Its binding in the group.
    pub binding: u32,
    /// The global's name.
    pub name: String,
}

/// Where an output binds each resource of a program.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct BindingPlan {
    /// One per resource.
    pub bindings: Vec<Bind>,
}

impl BindingPlan {
    /// Direct3D's register classes as bind groups, each register its binding: constant buffers in group 0 (`cbN`),
    /// shader resources in group 1 (`tN`), samplers in group 2 (`sN`), unordered-access views in group 3 (`uN`).
    /// Pipeline state is not bound. A resource with no D3D register is refused.
    pub fn d3d(interface: &Interface) -> Result<BindingPlan, Error> {
        let mut bindings = Vec::new();
        // pipeline state is set with the draw, not bound: specialisation folds it (a state block would be planned here)
        for r in interface
            .resources
            .iter()
            .filter(|r| !matches!(r.kind, ResourceKind::PipelineState(_)))
        {
            let d = r.provenance.d3d.as_ref().ok_or_else(|| {
                Error(format!(
                    "resource {} ({}) has no D3D register",
                    r.id.0, r.provenance.name
                ))
            })?;
            let (group, prefix) = match d.class {
                D3dClass::ConstantBuffer => (0, "cb"),
                D3dClass::ShaderResource => (1, "t"),
                D3dClass::Sampler => (2, "s"),
                D3dClass::UnorderedAccess => (3, "u"),
            };
            if d.space != 0 {
                return Err(Error(format!(
                    "resource {} is in register space {}",
                    r.provenance.name, d.space
                )));
            }
            bindings.push(Bind {
                resource: r.id,
                group,
                binding: d.register,
                name: format!("{prefix}{}", d.register),
            });
        }
        Ok(BindingPlan { bindings })
    }
}

/// Run naga's validator on a module, with the capabilities `profile` allows and no others.
pub fn validate(module: &naga::Module, profile: &Profile) -> Result<naga::valid::ModuleInfo, Error> {
    naga::valid::Validator::new(naga::valid::ValidationFlags::all(), profile.capabilities)
        .validate(module)
        .map_err(|e| {
            // the validator's error is a chain (entry point, function, expression): every link names part of the fault
            let mut message = format!("naga validation: {}", e.as_inner());
            let mut cause = core::error::Error::source(e.as_inner());
            while let Some(c) = cause {
                message.push_str(&format!(": {c}"));
                cause = c.source();
            }
            Error(message)
        })
}

/// The WGSL of a program for `profile`, with the D3D binding plan: preflighted, lowered, validated and written. A
/// module naga rejects after the preflight accepted the program is reported as a lowering bug.
#[cfg(feature = "wgsl")]
pub fn wgsl(program: &iridia_ir::Program, profile: &Profile) -> Result<String, Error> {
    preflight(program, profile)?;
    let plan = BindingPlan::d3d(&program.interface)?;
    let module = module(program, &plan)?;
    let info = validate(&module, profile).map_err(|e| {
        Error(format!(
            "lowering bug: the preflight accepted the program and {}",
            e.0
        ))
    })?;
    naga::back::wgsl::write_string(&module, &info, naga::back::wgsl::WriterFlags::empty())
        .map_err(|e| Error(format!("naga's WGSL writer: {e}")))
}

// Run the examples in the README as doc tests.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeExamples;
