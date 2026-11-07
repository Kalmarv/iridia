//! Functions: a value arena, variables, and a body in one of the two control phases.

use alloc::{string::String, vec::Vec};

use crate::{Address, Barrier, LabelId, Op, OutOfBounds, Res, Type, Unit, ValueId, VarId};

/// Where in the source bytes an operation comes from, for refusals, notes and mapping back.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize)]
pub struct Origin {
    /// What `offset` counts.
    pub unit: Unit,
    /// The offset of the source instruction.
    pub offset: u32,
}

/// The precision an operation is allowed to compute at: a permission, never a statement of what it computes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, serde::Serialize)]
pub enum Precision {
    /// The type's own precision.
    #[default]
    Full,
    /// Any precision of at least 16 bits (D3D9 `_pp`, DXBC `min16float`, GLSL `mediump`).
    AtLeast16,
}

/// A value: its type, the operation that defines it, its source origin and the precision it may be computed at.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct ValueDef {
    /// The value's type.
    pub ty: Type,
    /// The operation.
    pub op: Op,
    /// Where it comes from in the source.
    pub origin: Option<Origin>,
    /// The precision it may be computed at.
    pub precision: Precision,
}

/// Who shares a variable.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, serde::Serialize)]
pub enum Space {
    /// Each invocation has its own.
    #[default]
    Private,
    /// The invocations of a compute workgroup share it (D3D's group shared memory); it starts as zero.
    Workgroup,
    /// A mesh program's task payload: the words the amplification program's `DispatchMesh` handed the thread group,
    /// laid out as Direct3D lays out the payload's structure (as C does: each field at its natural alignment, a bool
    /// four bytes). `u32` elements, read alone.
    Payload,
    /// The payload of the ray a hit or miss program runs for, or a callable program's parameter: the words the caller
    /// (`TraceRay`, `CallShader`) hands over, laid out as Direct3D lays out the structure (as [`Space::Payload`] is).
    /// `u32` elements, read and written; what they hold when the program returns is the caller's.
    RayPayload,
    /// The attributes of the hit an any-hit or closest-hit program runs for (a triangle's barycentrics, what an
    /// intersection program reported), laid out as [`Space::RayPayload`] is. `u32` elements, read alone.
    HitAttributes,
}

/// A variable: a source register until `promote` turns its loads and stores into values, or an array (a source's
/// indexable registers, an immediate constant table, workgroup memory) read and written by element.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct Var {
    /// Its name, for the text form and target names (`r0`).
    pub name: String,
    /// Its type, or its elements' type.
    pub ty: Type,
    /// The elements of an array; `None` for a variable that is not one.
    pub len: Option<u32>,
    /// Who shares it.
    pub space: Space,
    /// Its value at the start, each element's lanes as a literal's (an immediate constant table); empty for none, where
    /// a private variable starts undefined and a workgroup one zero.
    pub init: Vec<[u64; 4]>,
}

impl Var {
    /// A private variable of `ty` that is not an array.
    pub fn new(name: &str, ty: Type) -> Var {
        Var {
            name: name.into(),
            ty,
            len: None,
            space: Space::Private,
            init: Vec::new(),
        }
    }
}

/// A step of a region or a block.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub enum Node {
    /// The value is computed here (its operation's state reads happen here).
    Value(ValueId),
    /// The variable takes the value.
    StoreVar(VarId, ValueId),
    /// Lanes of the interface output `slot` take the value: the lanes of mask `.1` (bit n is lane n, at least one,
    /// within the varying's width), in ascending order, from the value's lanes in order. The value has the varying's
    /// scalar type and one lane per lane written. A lane never written is zero when the program returns.
    StoreOutput(u32, u8, ValueId),
    /// Discards the invocation of a pixel program (D3D's `texkill` and `discard`, WGSL's `discard`) without ending
    /// it: it goes on as a helper invocation, which its quad's derivatives still read, and its later output stores have
    /// no effect. It ends as killed, with the outputs it stored before.
    Demote,
    /// Element `.1` (a `u32` or `i32`) of an array variable takes the value, with the stated rule for an index past
    /// its end.
    StoreElement(VarId, ValueId, ValueId, OutOfBounds),
    /// Consecutive 32-bit words of a storage buffer from the byte address (its low two bits ignored) take the value's
    /// lanes; a word past the buffer's size is not written.
    StoreBuffer {
        /// The buffer.
        resource: Res,
        /// The byte address of the first word.
        address: Address,
        /// The value: 32-bit lanes.
        value: ValueId,
    },
    /// The texel of a storage image or typed storage buffer at an integer coordinate (an array layer as its last
    /// lane; the element for a buffer) takes the value: `vec4` of the image's lanes' type. A texel outside it is not
    /// written.
    StoreTexel {
        /// The image or buffer.
        image: Res,
        /// The coordinate (`i32` or `u32`).
        coord: ValueId,
        /// The texel.
        value: ValueId,
    },
    /// Lanes of patch constant slot `.0` take the value (a hull program's patch constant function), as `StoreOutput`.
    StorePatch(u32, u8, ValueId),
    /// A barrier of the workgroup.
    Barrier(Barrier),
    /// A geometry program emits the outputs stored so far as a vertex of stream `.0`; the outputs are undefined after it
    /// until stored again.
    Emit(u8),
    /// A geometry program ends the strip it is emitting on stream `.0`.
    Cut(u8),
    /// A mesh program sets how many vertices and primitives its meshlet has (`u32` each), at most the most its facts
    /// state. Every invocation of the thread group that sets them sets the same (Direct3D's `SetMeshOutputCounts`,
    /// which requires it).
    SetMeshCounts {
        /// The vertices.
        vertices: ValueId,
        /// The primitives.
        primitives: ValueId,
    },
    /// Lanes of output `slot` of a vertex of a mesh program's meshlet take the value, as `StoreOutput`'s lanes; a
    /// vertex past the most the facts state is not written.
    StoreVertex {
        /// The vertex (`u32`).
        vertex: ValueId,
        /// The output.
        slot: u32,
        /// The lanes written.
        mask: u8,
        /// The value.
        value: ValueId,
    },
    /// The vertex indices of a primitive of a mesh program's meshlet take the value: `vecN<u32>`, N the topology's
    /// indices. A primitive past the most the facts state is not written.
    StoreIndices {
        /// The primitive (`u32`).
        primitive: ValueId,
        /// Its vertices.
        indices: ValueId,
    },
    /// An amplification program launches thread groups of the mesh program, handing each the words of a variable as
    /// its task payload (Direct3D's `DispatchMesh`). It is a barrier of the workgroup that every invocation reaches
    /// once; the values that count are the thread group's first invocation's, and an invocation does nothing after it.
    DispatchMesh {
        /// The thread groups along x, y and z (`vec3<u32>`).
        groups: ValueId,
        /// The task payload: `u32` elements, as many as the facts' payload bytes fill.
        payload: VarId,
    },
    /// A ray tracing program traces a ray into an acceleration structure (Direct3D's `TraceRay`): the pipeline runs the
    /// programs the ray's hits and misses select, which read and write the payload; the variable holds what they left
    /// when it returns.
    TraceRay {
        /// The acceleration structure.
        scene: Res,
        /// The ray.
        ray: Ray,
        /// The payload: `u32` elements laid out as [`Space::RayPayload`] is.
        payload: VarId,
    },
    /// A ray tracing program runs the callable program the shader table's index selects (Direct3D's `CallShader`),
    /// which reads and writes the parameter.
    CallShader {
        /// The shader table index (`u32`).
        index: ValueId,
        /// The parameter: `u32` elements laid out as [`Space::RayPayload`] is.
        parameter: VarId,
    },
    /// A multi-way choice on an integer (`u32` or `i32`, by its bits): the case whose values hold it, else the default.
    /// Each region ends in a yield of the results' values or an exit; none falls into the next.
    Switch {
        /// The label a `Br` names to leave it.
        label: LabelId,
        /// The value chosen on.
        selector: ValueId,
        /// The cases: the values that choose each, and its region.
        cases: Vec<Case>,
        /// The region run for every other value.
        default: Region,
        /// The values the construct defines (`Op::Param`), given by each region's yield and each `Br` to it.
        results: Vec<ValueId>,
    },
    /// Two-way choice. Each arm ends in a yield of the results' values, or an exit that leaves it.
    If {
        /// A scalar bool.
        cond: ValueId,
        /// The region run where `cond` holds.
        accept: Region,
        /// The region run where it does not.
        reject: Region,
        /// The values defined by the construct (`Op::Param`), given by each arm's yield.
        results: Vec<ValueId>,
    },
    /// A region that `End::Br(label)` leaves early, with values for its results.
    Block {
        /// The label exits name.
        label: LabelId,
        /// The region.
        body: Region,
        /// The values the construct defines (`Op::Param`), given by the body's yield and by each `Br` to it.
        results: Vec<ValueId>,
    },
    /// A loop: `End::Continue(label)` runs the body again with new parameter values; a yield or `Br(label)` leaves it.
    Loop {
        /// The label exits name.
        label: LabelId,
        /// The loop-carried values (`Op::Param`), visible in the body.
        params: Vec<ValueId>,
        /// The parameters' values on entry.
        inits: Vec<ValueId>,
        /// The body.
        body: Region,
        /// The values the construct defines (`Op::Param`), given by the body's yield and each `Br` to it.
        results: Vec<ValueId>,
    },
}

/// The ray a `TraceRay` traces: what Direct3D's call takes besides the acceleration structure and the payload.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize)]
pub struct Ray {
    /// The ray flags (`u32`, `RAY_FLAG_*`).
    pub flags: ValueId,
    /// The instance inclusion mask (`u32`, its low 8 bits).
    pub mask: ValueId,
    /// The offset of the ray's hit groups in the shader table (`u32`).
    pub hit_group_offset: ValueId,
    /// The stride between the hit groups of a bottom level's geometries (`u32`).
    pub hit_group_stride: ValueId,
    /// The miss program's index (`u32`).
    pub miss_index: ValueId,
    /// The origin (`vec3<f32>`).
    pub origin: ValueId,
    /// The least distance along the ray a hit counts at (`f32`).
    pub t_min: ValueId,
    /// The direction (`vec3<f32>`).
    pub direction: ValueId,
    /// The greatest distance (`f32`).
    pub t_max: ValueId,
}

impl Ray {
    /// The values, in Direct3D's order.
    pub fn values(&self) -> [ValueId; 9] {
        [
            self.flags,
            self.mask,
            self.hit_group_offset,
            self.hit_group_stride,
            self.miss_index,
            self.origin,
            self.t_min,
            self.direction,
            self.t_max,
        ]
    }

    /// The values, in Direct3D's order, to change.
    pub fn values_mut(&mut self) -> [&mut ValueId; 9] {
        [
            &mut self.flags,
            &mut self.mask,
            &mut self.hit_group_offset,
            &mut self.hit_group_stride,
            &mut self.miss_index,
            &mut self.origin,
            &mut self.t_min,
            &mut self.direction,
            &mut self.t_max,
        ]
    }
}

impl Node {
    /// The values the node itself reads (a construct's condition, selector or initial values; not its regions').
    pub fn values(&self) -> Vec<ValueId> {
        match self {
            Node::Value(_)
            | Node::Demote
            | Node::Barrier(_)
            | Node::Emit(_)
            | Node::Cut(_)
            | Node::Block { .. } => Vec::new(),
            Node::StoreVar(_, v) | Node::StoreOutput(_, _, v) | Node::StorePatch(_, _, v) => alloc::vec![*v],
            Node::StoreElement(_, i, v, _) => alloc::vec![*i, *v],
            Node::SetMeshCounts { vertices, primitives } => alloc::vec![*vertices, *primitives],
            Node::StoreVertex { vertex, value, .. } => alloc::vec![*vertex, *value],
            Node::StoreIndices { primitive, indices } => alloc::vec![*primitive, *indices],
            Node::DispatchMesh { groups, .. } => alloc::vec![*groups],
            Node::TraceRay { scene, ray, .. } => scene.index.into_iter().chain(ray.values()).collect(),
            Node::CallShader { index, .. } => alloc::vec![*index],
            Node::StoreBuffer {
                resource,
                address,
                value,
            } => resource
                .index
                .into_iter()
                .chain(address.index)
                .chain(core::iter::once(*value))
                .collect(),
            Node::StoreTexel { image, coord, value } => {
                image.index.into_iter().chain([*coord, *value]).collect()
            }
            Node::If { cond, .. } => alloc::vec![*cond],
            Node::Switch { selector, .. } => alloc::vec![*selector],
            Node::Loop { inits, .. } => inits.clone(),
        }
    }

    /// The values the node itself reads, to change.
    pub fn values_mut(&mut self) -> Vec<&mut ValueId> {
        match self {
            Node::Value(_)
            | Node::Demote
            | Node::Barrier(_)
            | Node::Emit(_)
            | Node::Cut(_)
            | Node::Block { .. } => Vec::new(),
            Node::StoreVar(_, v) | Node::StoreOutput(_, _, v) | Node::StorePatch(_, _, v) => alloc::vec![v],
            Node::StoreElement(_, i, v, _) => alloc::vec![i, v],
            Node::SetMeshCounts { vertices, primitives } => alloc::vec![vertices, primitives],
            Node::StoreVertex { vertex, value, .. } => alloc::vec![vertex, value],
            Node::StoreIndices { primitive, indices } => alloc::vec![primitive, indices],
            Node::DispatchMesh { groups, .. } => alloc::vec![groups],
            Node::TraceRay { scene, ray, .. } => {
                scene.index.as_mut().into_iter().chain(ray.values_mut()).collect()
            }
            Node::CallShader { index, .. } => alloc::vec![index],
            Node::StoreBuffer {
                resource,
                address,
                value,
            } => resource
                .index
                .as_mut()
                .into_iter()
                .chain(address.index.as_mut())
                .chain(core::iter::once(value))
                .collect(),
            Node::StoreTexel { image, coord, value } => {
                image.index.as_mut().into_iter().chain([coord, value]).collect()
            }
            Node::If { cond, .. } => alloc::vec![cond],
            Node::Switch { selector, .. } => alloc::vec![selector],
            Node::Loop { inits, .. } => inits.iter_mut().collect(),
        }
    }

    /// The variables the node itself names (a store's, a mesh launch's payload; not its regions' nor its values'), to
    /// change.
    pub fn vars_mut(&mut self) -> Vec<&mut VarId> {
        match self {
            Node::StoreVar(var, _)
            | Node::StoreElement(var, ..)
            | Node::DispatchMesh { payload: var, .. }
            | Node::TraceRay { payload: var, .. }
            | Node::CallShader { parameter: var, .. } => alloc::vec![var],
            _ => Vec::new(),
        }
    }

    /// Whether the node changes state other than a variable or an output (memory, the pipeline): it stays where it is.
    pub fn has_effect(&self) -> bool {
        matches!(
            self,
            Node::StoreBuffer { .. }
                | Node::StoreTexel { .. }
                | Node::Barrier(_)
                | Node::Emit(_)
                | Node::Cut(_)
                | Node::SetMeshCounts { .. }
                | Node::StoreVertex { .. }
                | Node::StoreIndices { .. }
                | Node::DispatchMesh { .. }
                | Node::TraceRay { .. }
                | Node::CallShader { .. }
        )
    }

    /// The regions of a construct, in order.
    pub fn regions(&self) -> Vec<&Region> {
        match self {
            Node::If { accept, reject, .. } => alloc::vec![accept, reject],
            Node::Block { body, .. } | Node::Loop { body, .. } => alloc::vec![body],
            Node::Switch { cases, default, .. } => cases
                .iter()
                .map(|c| &c.body)
                .chain(core::iter::once(default))
                .collect(),
            _ => Vec::new(),
        }
    }

    /// The regions of a construct, in order, to change.
    pub fn regions_mut(&mut self) -> Vec<&mut Region> {
        match self {
            Node::If { accept, reject, .. } => alloc::vec![accept, reject],
            Node::Block { body, .. } | Node::Loop { body, .. } => alloc::vec![body],
            Node::Switch { cases, default, .. } => cases
                .iter_mut()
                .map(|c| &mut c.body)
                .chain(core::iter::once(default))
                .collect(),
            _ => Vec::new(),
        }
    }
}

/// A case of a `Switch`.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct Case {
    /// The selector's values (as 32-bit words) that choose it; at least one, none chosen by two cases.
    pub values: Vec<u32>,
    /// The region run.
    pub body: Region,
}

/// How a region ends.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub enum End {
    /// Falls off the end, giving its construct's results these values.
    Yield(Vec<ValueId>),
    /// Leaves the enclosing `Block` or `Loop` named by the label, with values for its results.
    Br(LabelId, Vec<ValueId>),
    /// Restarts the enclosing `Loop` named by the label, with values for its parameters.
    Continue(LabelId, Vec<ValueId>),
    /// Leaves the function.
    Return,
    /// Discards the invocation and ends it: [`Node::Demote`] then `Return`.
    Kill,
    /// An any-hit program ignores the candidate hit and ends (Direct3D's `IgnoreHit`): the ray goes on as though the
    /// hit had not been met.
    IgnoreHit,
    /// An any-hit program accepts the candidate hit, ends, and ends the ray's search (Direct3D's
    /// `AcceptHitAndEndSearch`): the closest hit so far is the ray's.
    AcceptHitAndEndSearch,
    /// Cannot be reached.
    Unreachable,
}

/// A sequence of nodes and how it ends.
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize)]
pub struct Region {
    /// The nodes, in order.
    pub nodes: Vec<Node>,
    /// How it ends; `None` only while a builder is filling it.
    pub end: Option<End>,
}

/// A branch target of a `Cfg` block: the block and the values of its parameters.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct Target {
    /// The block, by index.
    pub block: u32,
    /// The values its parameters take.
    pub args: Vec<ValueId>,
}

/// How a `Cfg` block ends.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub enum Terminator {
    /// Continues to the target.
    Jump(Target),
    /// Two-way branch on a scalar bool.
    Branch {
        /// The condition.
        cond: ValueId,
        /// Where it holds.
        accept: Target,
        /// Where it does not.
        reject: Target,
    },
    /// Multi-way branch on an integer (`u32` or `i32`, by its bits).
    Switch {
        /// The value chosen on.
        selector: ValueId,
        /// Each value and where it goes; none twice.
        cases: Vec<(u32, Target)>,
        /// Where every other value goes.
        default: Target,
    },
    /// Leaves the function.
    Return,
    /// Discards the invocation.
    Kill,
    /// Ignores the candidate hit, as [`End::IgnoreHit`].
    IgnoreHit,
    /// Accepts the candidate hit and ends the search, as [`End::AcceptHitAndEndSearch`].
    AcceptHitAndEndSearch,
    /// Cannot be reached.
    Unreachable,
}

impl Terminator {
    /// The targets, in order.
    pub fn targets(&self) -> Vec<&Target> {
        match self {
            Terminator::Jump(t) => alloc::vec![t],
            Terminator::Branch { accept, reject, .. } => alloc::vec![accept, reject],
            Terminator::Switch { cases, default, .. } => cases
                .iter()
                .map(|(_, t)| t)
                .chain(core::iter::once(default))
                .collect(),
            Terminator::Return
            | Terminator::Kill
            | Terminator::IgnoreHit
            | Terminator::AcceptHitAndEndSearch
            | Terminator::Unreachable => Vec::new(),
        }
    }
    /// The targets, in order, to change.
    pub fn targets_mut(&mut self) -> Vec<&mut Target> {
        match self {
            Terminator::Jump(t) => alloc::vec![t],
            Terminator::Branch { accept, reject, .. } => alloc::vec![accept, reject],
            Terminator::Switch { cases, default, .. } => cases
                .iter_mut()
                .map(|(_, t)| t)
                .chain(core::iter::once(default))
                .collect(),
            Terminator::Return
            | Terminator::Kill
            | Terminator::IgnoreHit
            | Terminator::AcceptHitAndEndSearch
            | Terminator::Unreachable => Vec::new(),
        }
    }
}

/// A basic block of the `Cfg` phase: parameters, plain nodes (no constructs) and a terminator.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct Block {
    /// Its parameters (`Op::Param`).
    pub params: Vec<ValueId>,
    /// Its nodes: values, stores; never a construct.
    pub nodes: Vec<Node>,
    /// How it ends.
    pub term: Terminator,
}

/// A function body, in one of the two control phases.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub enum Body {
    /// Structured regions: what structured sources build and structured targets take.
    Regions(Region),
    /// A control flow graph; block 0 is the entry.
    Cfg(Vec<Block>),
}

/// A function: its values, variables and body.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
pub struct Function {
    /// The name (`main` for the entry point).
    pub name: String,
    /// Every value, by `ValueId`.
    pub values: Vec<ValueDef>,
    /// Every variable, by `VarId`.
    pub vars: Vec<Var>,
    /// The body.
    pub body: Body,
}

impl Function {
    /// The definition of a value, if the id is in range.
    pub fn value(&self, v: ValueId) -> Option<&ValueDef> {
        self.values.get(v.0 as usize)
    }
    /// The type of a value, if the id is in range.
    pub fn ty(&self, v: ValueId) -> Option<Type> {
        self.value(v).map(|d| d.ty)
    }
}
