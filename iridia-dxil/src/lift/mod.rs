//! Lifting a Shader Model 6 program of any stage but amplification (DXIL) to an [`iridia::Program`], with the same
//! conventions as the DXBC lift (`iridia-dxbc`): varyings unpacked from
//! D3D registers, resources as module globals (constant buffer `cb<n>` a uniform `array<vec4<u32>, rows>` in group 0,
//! textures and buffer views group 1, samplers group 2, typed texture views group 3, the hidden counters of unordered-access
//! views group 4, binding = register), D3D's
//! results written out as operations ([`iridia::D3dModes`]).
//!
//! What differs is the source. A DXIL function is LLVM's: values in static single assignment form over basic blocks.
//! Basic blocks are structured by [`iridia::build::structure`] (a reducible graph; irreducible flow is refused
//! by name); a phi is a local the edges assign; a value used outside the block that defines it is a local too, the
//! rest stay expressions. `dx.op` calls are the intrinsics (inputs and outputs, math, bit operations, constant
//! buffers, samples, loads, derivatives, discard), and `alloca`, `getelementptr`, `load` and `store` the local arrays
//! and constant tables.

mod compute;
mod dxop;
mod flow;
mod memory;
mod mesh;
mod ops;
mod rayquery;
mod raytracing;
mod resources;
mod setup;
mod stages;
mod values;
mod wide;

use std::collections::{BTreeMap, BTreeSet};

use iridia::build::{Builder, Kind, Val, Width};
use iridia::{Conventions, D3dModes, Program, Provenance, ShaderFormat, Stage, StageFacts};
use naga::{Handle, LocalVariable};

use crate::{Result, error, read};
use dxil_reader::module::{Function, Module, Op, Type, TypeId, ValueId};
use dxil_reader::reflection;

/// The low 32 bits of a constant, as an LLVM integer constant of 32 bits or fewer is stored (sign-extended to 64).
pub(crate) fn low32(value: u64) -> u32 {
    u32::try_from(value & 0xffff_ffff).unwrap_or(u32::MAX)
}

/// A lane or component number as a `u8`: a lane is one of at most four.
pub(crate) fn lane_u8(n: impl TryInto<u8>) -> u8 {
    n.try_into().unwrap_or(u8::MAX)
}

/// The basic blocks' instructions, by index into the function's instructions.
fn blocks_of(f: &Function) -> Result<Vec<Vec<usize>>> {
    let mut blocks = vec![Vec::new(); f.blocks];
    for (i, inst) in f.instructions.iter().enumerate() {
        blocks
            .get_mut(inst.block)
            .ok_or_else(|| {
                error(
                    inst.at,
                    format!(
                        "an instruction in basic block {}, where the function has {}",
                        inst.block, f.blocks
                    ),
                )
            })?
            .push(i);
    }
    Ok(blocks)
}

/// Argument `n` of a call at bit `at`, an error naming the call when it has fewer.
pub(crate) fn argument(args: &[ValueId], n: usize, at: usize) -> Result<ValueId> {
    args.get(n).copied().ok_or_else(|| {
        error(
            at,
            format!(
                "an operation reads argument {n}, where the call has {}",
                args.len()
            ),
        )
    })
}

/// The version of the lifter: bumped when what it produces for the same bytecode changes, so a cache keyed by the
/// bytecode's hash knows what is stale; gated by `tests/versions.rs` (the lifted fixtures hashed under it, `iridia::gate`).
pub const LIFTER_VERSION: u32 = 8;

/// Lift a DXIL container's entry function: Shader Model 6, any stage but amplification (vertex, pixel, hull, domain,
/// geometry, compute, mesh and the ray tracing stages). Every construct the lifter does not carry (a double, a 64-bit
/// integer division, an irreducible loop, an operation it has no meaning for) is an `Err` naming the bit offset and the
/// reason.
pub fn lift(container: &[u8]) -> Result<Program> {
    lift_with(container, &iridia::BoundSizes::new())
}

/// [`lift`] with the constant buffers `bound` names bound at the size it states: their uniform arrays have the rows of
/// that size, so a read past the declaration and inside the bound is the buffer's, and the interface's layout is the
/// bound size.
pub fn lift_with(container: &[u8], bound: &iridia::BoundSizes) -> Result<Program> {
    let mut program = lift_program(container, bound)?;
    program.provenance.source_hash = Some(iridia::source_hash(container));
    Ok(program)
}

fn lift_program(container: &[u8], bound: &iridia::BoundSizes) -> Result<Program> {
    let chunks = dxil_reader::parts(container).map_err(read)?;
    let module = dxil_reader::module(&chunks, b"DXIL")
        .map_err(read)?
        .ok_or_else(|| error(0, "no DXIL part"))?;
    let part = dxil_reader::program(
        chunks
            .iter()
            .find(|c| &c.kind == b"DXIL")
            .ok_or_else(|| error(0, "no DXIL part"))?
            .data,
    )
    .map_err(read)?;
    let entry_value = reflection::entry(&module).map_err(read)?;
    let entry = module
        .functions
        .iter()
        .find(|f| f.value == entry_value)
        .ok_or_else(|| error(0, "the entry point has no body"))?;
    // a hull program is two functions: the control-point phase (the entry) and the patch-constant function
    let patch_constant = if part.program_version >> 16 == 3 {
        let state = reflection::hull_state(&module).map_err(read)?;
        Some(
            module
                .functions
                .iter()
                .find(|f| f.value == state.patch_constant)
                .ok_or_else(|| error(0, "the patch-constant function has no body"))?,
        )
    } else {
        None
    };
    let mut functions = vec![entry];
    functions.extend(patch_constant);
    let mut setup = setup::Setup::new(&chunks, &module, &functions)?;
    bound
        .apply(&mut setup.interface)
        .map_err(|why| error(0, format!("bound sizes: {why}")))?;
    let target = format!(
        "{}_{}_{}",
        match part.program_version >> 16 {
            0 => "ps",
            1 => "vs",
            2 => "gs",
            13 => "ms",
            3 => "hs",
            4 => "ds",
            6..=12 => "lib",
            _ => "cs",
        },
        (part.program_version >> 4) & 15,
        part.program_version & 15
    );
    let mut lift = Lift::new(&module, entry, functions, setup)?;
    if matches!(lift.setup.interface.stage, Stage::Compute | Stage::Mesh) {
        lift.b.workgroup_size = reflection::num_threads(&module)
            .map_err(read)?
            .ok_or_else(|| error(0, "a compute program that states no thread group size"))?;
    }
    match lift.setup.sigs.clone() {
        None => {
            lift.run()?;
            lift.finish(target)
        }
        Some(sigs) => {
            for sig in &sigs {
                let f = if sig.name == "patch_constant" {
                    patch_constant
                        .ok_or_else(|| error(0, "a patch-constant function in a program that has none"))?
                } else {
                    entry
                };
                lift.run_function(f, sig)?;
            }
            lift.finish_functions(target)
        }
    }
}

/// A translated LLVM value.
#[derive(Clone, Debug)]
pub(crate) enum Lv {
    Scalar(Val),
    /// A struct result (`ResRet`, a `dx.op` pair): its elements.
    Agg(Vec<Lv>),
    /// A resource handle: its class (0 shader resource, 1 unordered access, 2 constant buffer, 3 sampler), the index
    /// of the interface resource it names and, for a resource array, the element (from the array's first) the
    /// program picked.
    Handle {
        class: u8,
        resource: usize,
        element: Option<Val>,
    },
    Pointer(memory::Ptr),
    /// An inline ray tracing query (see [`rayquery`]): the query, the flags it is declared with, and the local that
    /// keeps the ray it is traced with (naga's query does not give the ray back).
    RayQuery(Handle<LocalVariable>, u32, Handle<LocalVariable>),
}

/// Where a value lives once defined: as the expressions of its block, or in locals when another block reads it.
#[derive(Clone, Debug)]
enum Slot {
    Value(Lv),
    Stored(Stored),
}

#[derive(Clone, Debug)]
enum Stored {
    Scalar {
        local: Handle<LocalVariable>,
        kind: Kind,
        width: Width,
    },
    Agg(Vec<Stored>),
    /// Not stored: handles and the like, which are the same wherever they are read.
    Plain(Lv),
    /// A handle to an element of a resource array: the element index is a value, kept in a local.
    Handle {
        class: u8,
        resource: usize,
        element: Box<Stored>,
    },
}

/// The state of one lift.
pub(crate) struct Lift<'m> {
    pub(crate) m: &'m Module,
    pub(crate) f: &'m Function,
    /// Every function whose instructions the lift reads: the entry's, and a hull program's patch-constant function.
    functions: Vec<&'m Function>,
    /// The function of a program of a stage with no entry point being lifted.
    sig: Option<iridia::FunctionSig>,
    pub(crate) b: Builder,
    pub(super) setup: setup::Setup,
    /// The slot of each local value (arguments and instruction results), by `v - f.first`.
    slots: Vec<Option<Slot>>,
    /// Whether a value is read outside the block that defines it.
    cross: Vec<bool>,
    /// The instructions of each basic block, by index into `f.instructions`.
    blocks: Vec<Vec<usize>>,
    /// Output registers, four raw lanes each.
    outputs: BTreeMap<u32, Handle<LocalVariable>>,
    depth: Option<Handle<LocalVariable>>,
    coverage: Option<Handle<LocalVariable>>,
    /// The lane number of a thread group with no lane-number built-in, made at the start of the program.
    lane: Option<Handle<LocalVariable>>,
    /// A mesh program's output variable and facts.
    mesh: Option<(iridia::build::MeshOutput, iridia::MeshFacts)>,
    /// A geometry program's emitted vertices so far.
    gs: Option<stages::GsState>,
    /// The ray tracing pipeline's operations the program calls.
    operations: raytracing::Operations,
    resources: resources::Resources,
    memory: memory::Memory,
    arguments: Vec<naga::FunctionArgument>,
    /// The expression of each entry argument, made once.
    argument_exprs: BTreeMap<usize, naga::Handle<naga::Expression>>,
    result_type: Option<Handle<naga::Type>>,
    notes: BTreeSet<String>,
    /// The bit offset of the instruction being lifted, for refusals.
    at: usize,
}

impl<'m> Lift<'m> {
    pub(crate) fn refuse<T>(&self, detail: impl Into<String>) -> Result<T> {
        Err(error(self.at, detail))
    }

    /// The width of an interface value, refused unless it is one to four lanes.
    pub(crate) fn width_of(&self, v: &iridia::Value) -> Result<Width> {
        Width::try_from(v.width).or_else(|e| self.refuse(format!("a value of {e}")))
    }

    /// Argument `n` of a call. An operation's argument count is checked against its table (`dxop::arguments`) before it is
    /// lifted, and the pre-scan reads only the arguments of operations it has the count of, so a miss is the guard of a
    /// lookup, not a case of the input.
    pub(crate) fn arg(&self, args: &[ValueId], n: usize) -> Result<ValueId> {
        argument(args, n, self.at)
    }

    /// The type of the value a call or an instruction defines; an error where it defines none.
    pub(crate) fn result_type(&self, result: Option<(ValueId, TypeId)>) -> Result<TypeId> {
        result
            .map(|(_, ty)| ty)
            .ok_or_else(|| error(self.at, "an instruction that yields a value but defines none"))
    }

    /// Value `v`: a module value, or one of the function's own.
    pub(crate) fn value_ref(&self, v: ValueId) -> Option<&'m dxil_reader::module::Value> {
        match v.checked_sub(self.f.first) {
            None => self.m.values.get(v),
            Some(local) => self.f.locals.get(local),
        }
    }

    /// The type of value `v`: a module value's, or one of the function's own.
    pub(crate) fn value_type_id(&self, v: ValueId) -> Option<TypeId> {
        match v.checked_sub(self.f.first) {
            None => self.m.value_types.get(v).copied(),
            Some(local) => self.f.local_types.get(local).copied(),
        }
    }

    /// The position of value `v` among the function's own values.
    pub(crate) fn local_index(&self, v: ValueId) -> Result<usize> {
        v.checked_sub(self.f.first)
            .filter(|local| *local < self.f.locals.len())
            .ok_or_else(|| {
                error(
                    self.at,
                    format!(
                        "value {v}, where the function's own values are {} to {}",
                        self.f.first,
                        self.f.first.saturating_add(self.f.locals.len())
                    ),
                )
            })
    }

    /// The register and lane of cell (`row`, `column`) of an element that starts at (`first_row`, `first_column`): an
    /// element's cells are in registers of four lanes.
    pub(crate) fn cell_of(
        &self,
        first_row: u32,
        first_column: u8,
        row: i64,
        column: i64,
    ) -> Result<(u32, u8)> {
        let register = u32::try_from(row).ok().and_then(|row| first_row.checked_add(row));
        let lane = u8::try_from(column)
            .ok()
            .and_then(|column| first_column.checked_add(column))
            .filter(|lane| *lane < 4);
        match (register, lane) {
            (Some(register), Some(lane)) => Ok((register, lane)),
            _ => self.refuse(format!(
                "row {row}, column {column} of an element that starts at row {first_row}, column {first_column}: a register has columns 0 to 3"
            )),
        }
    }

    /// The local that holds output register `register`.
    pub(crate) fn output_local(&self, register: u32) -> Result<Handle<LocalVariable>> {
        self.outputs
            .get(&register)
            .copied()
            .ok_or_else(|| error(self.at, format!("output register o{register} has no local")))
    }

    /// Type `ty` of the module, which the reader checked the module's ids against.
    pub(crate) fn type_at(&self, ty: TypeId) -> Result<&'m Type> {
        self.m.types.get(ty).ok_or_else(|| {
            error(
                self.at,
                format!("type {ty}, where the module has {}", self.m.types.len()),
            )
        })
    }

    fn new(
        m: &'m Module,
        f: &'m Function,
        functions: Vec<&'m Function>,
        setup: setup::Setup,
    ) -> Result<Self> {
        let blocks = blocks_of(f)?;
        let mut lift = Lift {
            m,
            f,
            functions,
            sig: None,
            b: Builder::new(),
            notes: setup.notes.clone(),
            setup,
            slots: vec![None; f.locals.len()],
            cross: vec![false; f.locals.len()],
            blocks,
            outputs: BTreeMap::new(),
            depth: None,
            coverage: None,
            lane: None,
            mesh: None,
            gs: None,
            operations: raytracing::Operations::default(),
            resources: resources::Resources::default(),
            memory: memory::Memory::default(),
            arguments: Vec::new(),
            argument_exprs: BTreeMap::new(),
            result_type: None,
            at: 0,
        };
        lift.analyse_uses();
        Ok(lift)
    }

    /// Mark the values read outside the block that defines them. A phi's value is always a local; a phi's incoming
    /// value is read at the end of its predecessor.
    fn analyse_uses(&mut self) {
        let f = self.f;
        let mut def_block: BTreeMap<ValueId, usize> = BTreeMap::new();
        for inst in &f.instructions {
            if let Some((v, _)) = inst.result {
                def_block.insert(v, inst.block);
            }
        }
        // a value of the function's own, by its position among them
        let local = |v: ValueId| v.checked_sub(f.first);
        let mark = |v: ValueId, at: usize, cross: &mut Vec<bool>| {
            if let Some(&d) = def_block.get(&v)
                && d != at
                && let Some(slot) = local(v).and_then(|i| cross.get_mut(i))
            {
                *slot = true;
            }
        };
        let mut cross = vec![false; f.locals.len()];
        for inst in &f.instructions {
            let at = inst.block;
            if let Op::Phi { incoming } = &inst.op {
                // a phi's value is always a local; its incoming value is read at the end of its predecessor
                if let Some(slot) = inst
                    .result
                    .and_then(|(v, _)| local(v))
                    .and_then(|i| cross.get_mut(i))
                {
                    *slot = true;
                }
                for (v, pred) in incoming {
                    mark(*v, *pred, &mut cross);
                }
                continue;
            }
            for v in inst.op.operands() {
                mark(v, at, &mut cross);
            }
        }
        // a pointer is symbolic and is read wherever it is used: its index values are read there too
        let mut changed = true;
        while changed {
            changed = false;
            for inst in &f.instructions {
                if let (Op::Gep { indices, .. }, Some((p, _))) = (&inst.op, inst.result)
                    && local(p).is_some_and(|i| cross.get(i) == Some(&true))
                {
                    for &i in indices {
                        if let Some(slot) = local(i).and_then(|i| cross.get_mut(i))
                            && !*slot
                        {
                            *slot = true;
                            changed = true;
                        }
                    }
                }
            }
        }
        self.cross = cross;
    }

    pub(crate) fn kind_of_type(&self, ty: TypeId) -> Result<Kind> {
        match self.m.types.get(ty) {
            // a half is computed as a float and rounded to a half where it is made (see `Lift::quantized`)
            Some(Type::Float | Type::Half) => Ok(Kind::F),
            Some(Type::Integer(1)) => Ok(Kind::B),
            // an integer narrower than 32 bits (16 for a half's sibling, 4 for a bit-field) is carried as the 32-bit
            // unsigned integer of its bits, zero-extended
            Some(Type::Integer(2..=32)) => Ok(Kind::U),
            Some(Type::Integer(n)) => self.refuse(format!("{n}-bit integers are not lifted")),
            Some(Type::Double) => self.refuse("double-precision floats are not lifted"),
            other => self.refuse(format!("a value of type {other:?}")),
        }
    }

    fn run(&mut self) -> Result<()> {
        let io = self.setup.io.clone();
        let declared = self.b.declare_entry(&io).map_err(|detail| error(0, detail))?;
        self.arguments = declared.arguments;
        self.result_type = declared.result;
        let u4 = self.b.ty_of(Kind::U, Width::Four);
        for v in &self.setup.interface.outputs {
            if let Some(d3d) = v.provenance.d3d
                && d3d.register != u32::MAX
                && !self.outputs.contains_key(&d3d.register)
            {
                let local = self.b.local(&format!("o{}", d3d.register), u4);
                self.outputs.insert(d3d.register, local);
            }
            match v.system {
                Some(
                    iridia::SystemValue::Depth
                    | iridia::SystemValue::DepthGreaterEqual
                    | iridia::SystemValue::DepthLessEqual,
                ) if self.depth.is_none() => self.depth = Some(self.b.local("oDepth", u4)),
                Some(iridia::SystemValue::Coverage) if self.coverage.is_none() => {
                    self.coverage = Some(self.b.local("oMask", u4));
                }
                _ => {}
            }
        }
        if self.setup.interface.stage == Stage::Mesh {
            self.start_mesh()?;
        }
        self.declare_resources()?;
        self.declare_memory()?;
        self.lane_at_start();
        self.define_phis()?;
        self.structure_and_lower()
    }

    /// Lift one function of a program of a stage with no entry point as the module's function `sig`.
    fn run_function(&mut self, f: &'m Function, sig: &iridia::FunctionSig) -> Result<()> {
        // the per-function state: values, blocks, output registers, the argument expressions
        self.f = f;
        self.blocks = blocks_of(f)?;
        self.slots = vec![None; f.locals.len()];
        self.cross = vec![false; f.locals.len()];
        self.outputs.clear();
        self.depth = None;
        self.coverage = None;
        self.argument_exprs.clear();
        self.analyse_uses();
        self.sig = Some(sig.clone());
        // the pointer arguments (a ray tracing program's payload and attributes) are the function's own parameters
        let mut structures = Vec::new();
        if sig.arguments.iter().any(|a| a.ty == iridia::IoType::Memory) {
            let function_type = self.m.value_types.get(f.value).and_then(|&t| self.m.types.get(t));
            let params = match function_type {
                Some(Type::Pointer(t)) => match self.m.types.get(*t) {
                    Some(Type::Function { params, .. }) => params.clone(),
                    _ => Vec::new(),
                },
                _ => Vec::new(),
            };
            let memory: Vec<usize> = sig
                .arguments
                .iter()
                .enumerate()
                .filter(|(_, a)| a.ty == iridia::IoType::Memory)
                .map(|(k, _)| k)
                .collect();
            if params.len() != memory.len() {
                return self.refuse(format!(
                    "a {:?} program whose function takes {} parameters where {} are expected",
                    self.setup.interface.stage,
                    params.len(),
                    memory.len()
                ));
            }
            for (k, (&param, &argument)) in params.iter().zip(&memory).enumerate() {
                let Some(Type::Pointer(pointee)) = self.m.types.get(param).cloned() else {
                    return self.refuse("a parameter that is not a pointer");
                };
                let (structure, _) = self.naga_type(pointee)?;
                structures.push(structure);
                let argument =
                    u32::try_from(argument).map_err(|_| error(self.at, "more than 4294967295 arguments"))?;
                if let Some(slot) = self.slots.get_mut(k) {
                    *slot = Some(Slot::Value(Lv::Pointer(memory::Ptr {
                        root: memory::Root::Argument(argument),
                        steps: Vec::new(),
                    })));
                }
            }
        }
        let decl = self
            .b
            .declare_function_with(sig, &structures)
            .map_err(|detail| error(0, detail))?;
        // the registers its results are made of
        let u4 = self.b.ty_of(Kind::U, Width::Four);
        for v in self.result_varyings() {
            if let Some(d3d) = v.provenance.d3d
                && d3d.register != u32::MAX
                && !self.outputs.contains_key(&d3d.register)
            {
                let local = self.b.local(&format!("o{}", d3d.register), u4);
                self.outputs.insert(d3d.register, local);
            }
        }
        self.result_type = decl.result;
        if self.resources.globals_made {
            // the globals are the module's: made with the first function
        } else {
            self.declare_resources()?;
            self.declare_memory()?;
            self.resources.globals_made = true;
        }
        if let StageFacts::Geometry(g) = &self.setup.interface.facts {
            let max = g.max_vertices;
            self.start_geometry(max)?;
        }
        self.define_phis()?;
        self.structure_and_lower()?;
        self.b
            .end_function(sig.name, decl)
            .map_err(|detail| error(self.at, detail))?;
        Ok(())
    }

    /// The varyings the current function's result is made of: the outputs, or a hull program's patch constants.
    pub(super) fn result_varyings(&self) -> Vec<iridia::Varying> {
        match (&self.sig, &self.setup.interface.facts) {
            (Some(sig), StageFacts::Hull(h)) if sig.name == "patch_constant" => h.patch_constants.clone(),
            _ => self.setup.interface.outputs.clone(),
        }
    }

    fn finish_functions(mut self, target: String) -> Result<Program> {
        self.b.check().map_err(|detail| error(self.at, detail))?;
        let stage = self.setup.interface.stage;
        let at = self.at;
        let module = self.b.finish_functions().map_err(|detail| error(at, detail))?;
        let mut limit_notes = iridia::compute_limit_notes(&module);
        let interface = self.setup.interface.clone();
        limit_notes.extend(iridia::interface_notes(&interface));
        Ok(Program {
            module,
            stage,
            entry: "main".into(),
            interface: self.setup.interface,
            bindings: self.resources.bound,
            conventions: Conventions::D3D,
            modes: D3dModes::ALL,
            provenance: Provenance {
                format: ShaderFormat::Dxil,
                target,
                lifter: "iridia-dxil".into(),
                lifter_version: LIFTER_VERSION,
                // the hash of the bytes, which `lift` sets
                source_hash: None,
                instruction_count: Some(iridia::InstructionCount {
                    count: self
                        .functions
                        .iter()
                        .map(|function| function.instructions.len() as u64)
                        .sum(),
                    unit: iridia::InstructionUnit::DxilInstructions,
                }),
            },
            notes: self.notes.into_iter().chain(limit_notes).collect(),
        })
    }

    fn finish(mut self, target: String) -> Result<Program> {
        self.b.check().map_err(|detail| error(self.at, detail))?;
        let stage = self.setup.interface.stage;
        let early = false;
        let at = self.at;
        let module = if let Some((output, facts)) = &self.mesh {
            self.b.finish_mesh(self.arguments, facts, output)
        } else {
            self.b.finish(
                "main",
                match stage {
                    Stage::Vertex => naga::ShaderStage::Vertex,
                    Stage::Compute => naga::ShaderStage::Compute,
                    _ => naga::ShaderStage::Fragment,
                },
                self.arguments,
                self.result_type,
                early,
            )
        }
        .map_err(|detail| error(at, detail))?;
        let mut limit_notes = iridia::compute_limit_notes(&module);
        let interface = self.setup.interface.clone();
        limit_notes.extend(iridia::interface_notes(&interface));
        Ok(Program {
            module,
            stage,
            entry: "main".into(),
            interface: self.setup.interface,
            bindings: self.resources.bound,
            conventions: Conventions::D3D,
            modes: D3dModes::ALL,
            provenance: Provenance {
                format: ShaderFormat::Dxil,
                target,
                lifter: "iridia-dxil".into(),
                lifter_version: LIFTER_VERSION,
                // the hash of the bytes, which `lift` sets
                source_hash: None,
                instruction_count: Some(iridia::InstructionCount {
                    count: self
                        .functions
                        .iter()
                        .map(|function| function.instructions.len() as u64)
                        .sum(),
                    unit: iridia::InstructionUnit::DxilInstructions,
                }),
            },
            notes: self.notes.into_iter().chain(limit_notes).collect(),
        })
    }
}
