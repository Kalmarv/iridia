//! An [`iridia_ir::Program`] compiled into the flat form: each value of the entry function is the register its id
//! numbers, its operation run by `iridia_exec_ir::eval` (the IR's one definition of what it computes) where it is an
//! operation on values, and by the flat form's own operations where it reads inputs, variables, buffers or textures.
//! Registers past the values' own hold what the compile adds: the entry arguments, copies into a construct's results,
//! and the integer layer of an array sample.
//!
//! The body must be structured ([`iridia_ir::Body::Regions`]) with its exits as `iridia_passes::break_flags` leaves
//! them: a `Br` leaves the innermost loop or switch, a `Continue` restarts the innermost loop.

use std::collections::BTreeMap;

use iridia::{FunctionSig, IoType, ItemList};
use iridia_ir::{
    self as ir, Body, End, FloatToInt, Function, Node, Region, Scalar, Type, ValueId, VarId,
    interface::{
        BufferKind, Interface, ResourceKind, SampleType, Stage, StageFacts, StorageTarget, TextureDim,
    },
};

use crate::flat::{
    AtomOp, Executable, GlobalDesc, GlobalKind, LevelOp, Loc, Member, Op, Pick, Reg, Res, St, WaveGather,
    WaveKind, WaveOp, WaveScan, WordAt,
};
use crate::request::{Dim, ImageAtomicOp, QueryKind};
use crate::run::Decision;

/// Where a variable's words are.
#[derive(Clone, Copy)]
struct Place {
    loc: Loc,
    /// Words per element (the variable's lanes).
    stride: u32,
    /// Elements: 1 for a variable that is not an array.
    len: u32,
}

/// A construct a `Br` or `Continue` names.
struct Frame {
    label: ir::LabelId,
    /// A loop's parameters; `None` for a switch or a block.
    params: Option<Vec<ValueId>>,
    results: Vec<ValueId>,
    /// A `Block` a `Br` leaves, run as a loop that ends after one pass: a `Continue` inside it cannot reach an outer
    /// loop.
    block: bool,
}

/// A geometry program's emitted vertices: the words of how many it emitted and whether the next starts a strip, the
/// most it may emit, each output's array (its slot, first word and lanes) and the strip starts' array.
#[derive(Clone)]
struct Emitted {
    count: u32,
    pending: u32,
    max: u32,
    arrays: Vec<(usize, u32, u8)>,
    strips: u32,
}

/// An amplification program's launch: a workgroup global in the words [`crate::amplification::AmplificationShader`]
/// reads, how many invocations reached `DispatchMesh`, the thread groups along x, y and z, then the payload's words.
#[derive(Clone)]
struct Launch {
    global: u32,
    /// The payload's words.
    payload: u32,
}

/// A mesh program's meshlet: a workgroup global in the words [`crate::mesh::MeshShader`] reads, the two counts, then
/// each vertex's outputs packed, then each primitive's indices.
#[derive(Clone)]
struct Meshlet {
    global: u32,
    /// Per interface output: its first word in a vertex.
    output_at: Vec<u32>,
    /// Words of a vertex, and the most vertices.
    vertex: u32,
    max_vertices: u32,
    /// The first word of the primitives, a primitive's words (its indices), and the most primitives.
    primitives_at: u32,
    index: u32,
    max_primitives: u32,
}

impl Meshlet {
    /// The words of a mesh program of `interface`, in global `global`.
    fn of(interface: &Interface, global: u32) -> Result<Self, String> {
        let StageFacts::Mesh(facts) = &interface.facts else {
            return Err("a mesh program without its facts".into());
        };
        let mut output_at = Vec::with_capacity(interface.outputs.len());
        let mut vertex = 0u32;
        for o in &interface.outputs {
            output_at.push(vertex);
            vertex = vertex.saturating_add(u32::from(o.value.width));
        }
        let primitives_at = vertex
            .checked_mul(facts.max_vertices)
            .and_then(|w| w.checked_add(2))
            .ok_or("a meshlet past the words of memory")?;
        Ok(Self {
            global,
            output_at,
            vertex,
            max_vertices: facts.max_vertices,
            primitives_at,
            index: facts.topology.indices(),
            max_primitives: facts.max_primitives,
        })
    }

    /// All its words.
    fn words(&self) -> Result<u32, String> {
        self.index
            .checked_mul(self.max_primitives)
            .and_then(|w| w.checked_add(self.primitives_at))
            .ok_or_else(|| "a meshlet past the words of memory".to_owned())
    }
}

/// Whether region `r` (or one inside it) ends by leaving the construct `label` names.
fn breaks_to(r: &Region, label: ir::LabelId) -> bool {
    matches!(&r.end, Some(End::Br(l, _)) if *l == label)
        || r.nodes.iter().any(|n| match n {
            Node::If { accept, reject, .. } => breaks_to(accept, label) || breaks_to(reject, label),
            Node::Block { body, .. } | Node::Loop { body, .. } => breaks_to(body, label),
            Node::Switch { cases, default, .. } => {
                cases.iter().any(|c| breaks_to(&c.body, label)) || breaks_to(default, label)
            }
            _ => false,
        })
}

/// What the end of a region does with the values it yields.
#[derive(Clone, Copy)]
enum Yield<'r> {
    /// They become these results (an `If`'s or a `Block`'s, a switch case's).
    Into(&'r [ValueId]),
    /// They become the results of the loop being left (a loop body's yield leaves it).
    Leave(&'r [ValueId]),
    /// The function's own region: it returns.
    Return,
}

struct Cx<'p> {
    f: &'p Function,
    interface: &'p Interface,
    /// Each register's type: the values' own, then those the compile adds.
    types: Vec<Type>,
    ops: Vec<(u32, Op)>,
    /// The first operation not yet in a statement's run.
    run_from: u32,
    inits: Vec<(u32, Reg)>,
    local_words: u32,
    globals: Vec<GlobalDesc>,
    /// The global of each buffer resource.
    buffer_of: BTreeMap<ir::interface::ResourceId, u32>,
    vars: Vec<Place>,
    /// Per interface input: its entry argument (index, register).
    inputs: Vec<Option<(u32, u32)>>,
    /// Per interface input and output of a stage that reads arrays of points: the array argument of its points.
    point_inputs: Vec<Option<Place>>,
    point_outputs: Vec<Option<Place>>,
    /// Per patch constant: the register a domain program takes it in.
    patch_in: Vec<Option<u32>>,
    /// Per patch constant a hull program's patch constant function stores: its first local word.
    patch_out: Vec<Option<u32>>,
    /// The registers of the values the pipeline gives.
    builtins: Vec<(ir::Builtin, u32)>,
    /// A geometry program's vertices.
    emitted: Option<Emitted>,
    /// A mesh program's meshlet.
    meshlet: Option<Meshlet>,
    /// An amplification program's launch.
    launch: Option<Launch>,
    /// Per interface output: its first local word and its words.
    outputs: Vec<(u32, u32)>,
    /// The members of the result, in the order of the entry point's results.
    returned: Vec<(Option<(u32, u8)>, Member)>,
    frames: Vec<Frame>,
}

fn number(n: usize) -> Result<u32, String> {
    u32::try_from(n).map_err(|_| "a program over 4294967295 registers".to_owned())
}

/// The lanes a mask marks, in order.
fn lanes_of(mask: u8) -> Vec<u8> {
    (0..4).filter(|l| mask >> l & 1 != 0).collect()
}

fn dim_of(d: TextureDim) -> Dim {
    match d {
        TextureDim::D1 => Dim::D1,
        TextureDim::D2 => Dim::D2,
        TextureDim::D3 => Dim::D3,
        TextureDim::Cube => Dim::Cube,
    }
}

/// The coordinate lanes of a texture of this dimension, without a layer.
fn coord_lanes(d: TextureDim) -> u8 {
    match d {
        TextureDim::D1 => 1,
        TextureDim::D2 => 2,
        TextureDim::D3 | TextureDim::Cube => 3,
    }
}

impl Cx<'_> {
    fn value(&self, v: ValueId) -> Result<&ir::ValueDef, String> {
        self.f
            .value(v)
            .ok_or_else(|| format!("value {} is not in the function", v.0))
    }

    /// A register the compile adds, of type `t`.
    fn fresh(&mut self, t: Type) -> Result<u32, String> {
        let r = number(self.types.len())?;
        self.types.push(t);
        Ok(r)
    }

    fn grow(&mut self, words: u32) -> Result<u32, String> {
        let at = self.local_words;
        self.local_words = at
            .checked_add(words)
            .filter(|&w| w <= crate::compile::MAX_LOCAL_WORDS)
            .ok_or("local memory past what a function may have")?;
        Ok(at)
    }

    /// A register holding a literal from the start.
    fn literal(&mut self, t: Type, bits: Reg) -> Result<u32, String> {
        let r = self.fresh(t)?;
        self.inits.push((r, bits));
        Ok(r)
    }

    /// An operation on values the compile adds, of type `t`, on registers it names as values.
    fn compute(&mut self, t: Type, op: ir::Op) -> Result<u32, String> {
        let r = self.fresh(t)?;
        self.ops.push((r, Op::Ir(Box::new(op))));
        Ok(r)
    }

    fn copy(&mut self, to: u32, from: u32) {
        self.ops.push((
            to,
            Op::Swizzle {
                src: from,
                pat: [0, 1, 2, 3],
                n: 4,
            },
        ));
    }

    /// The operations since the last statement, as a run of their own.
    fn flush(&mut self, out: &mut Vec<St>) -> Result<(), String> {
        let end = number(self.ops.len())?;
        if end > self.run_from {
            out.push(St::Run(self.run_from, end));
        }
        self.run_from = end;
        Ok(())
    }

    fn statement(&mut self, out: &mut Vec<St>, s: St) -> Result<(), String> {
        self.flush(out)?;
        out.push(s);
        Ok(())
    }

    fn resource(&self, id: ir::interface::ResourceId) -> Result<&ir::interface::Resource, String> {
        self.interface
            .resources
            .iter()
            .find(|r| r.id == id)
            .ok_or_else(|| format!("resource {} is not in the interface", id.0))
    }

    fn res(&self, r: &ir::Res) -> Result<Res, String> {
        let count = self.resource(r.id)?.count;
        Ok(Res {
            id: r.id,
            pick: match r.index {
                Some(v) => Pick::Slot(v.0),
                None if count == 1 => Pick::One,
                None => Pick::Fixed(0),
            },
            count,
        })
    }

    /// The register holding the first word an address names: `(offset + stride * index) / 4`, wrapping.
    fn word_of(&mut self, a: &ir::Address) -> Result<u32, String> {
        let offset = self.literal(Type::U32, [a.offset, 0, 0, 0])?;
        let bytes = match a.index {
            None => offset,
            Some(index) => {
                let stride = self.literal(Type::U32, [a.stride, 0, 0, 0])?;
                let scaled = self.compute(
                    Type::U32,
                    ir::Op::Binary(ir::Binary::IMul, ValueId(index.0), ValueId(stride)),
                )?;
                self.compute(
                    Type::U32,
                    ir::Op::Binary(ir::Binary::IAdd, ValueId(scaled), ValueId(offset)),
                )?
            }
        };
        let two = self.literal(Type::U32, [2, 0, 0, 0])?;
        self.compute(
            Type::U32,
            ir::Op::Binary(ir::Binary::ShrU, ValueId(bytes), ValueId(two)),
        )
    }

    /// The buffer global of a resource and the element of an array of them it names.
    fn buffer(&self, r: &ir::Res) -> Result<(u32, Pick), String> {
        let g = *self
            .buffer_of
            .get(&r.id)
            .ok_or_else(|| format!("resource {} is not a buffer", r.id.0))?;
        Ok((g, self.res(r)?.pick))
    }

    /// The position of sample `index` of a multisampled texture in Direct3D's standard pattern for its sample count
    /// (`iridia_ir::standard_sample_position`), (0, 0) past its samples: each position of the table chosen where the
    /// count and the index are its own.
    fn sample_position(&mut self, texture: &ir::Res, index: ValueId) -> Result<Op, String> {
        let (dim, _, _, _) = self.texture(texture)?;
        let count = self.fresh(Type::U32)?;
        let query = Op::Query {
            tex: self.res(texture)?,
            dim: dim_of(dim),
            kind: QueryKind::Samples,
            level: None,
        };
        self.ops.push((count, query));
        let v2 = Type::vector(Scalar::F32, 2);
        let mut position = self.literal(v2, [0; 4])?;
        for n in [1u32, 2, 4, 8, 16] {
            let is_count = {
                let n = self.literal(Type::U32, [n, 0, 0, 0])?;
                self.compute(
                    Type::BOOL,
                    ir::Op::Binary(ir::Binary::UCmp(ir::Cmp::Eq), ValueId(count), ValueId(n)),
                )?
            };
            for s in 0..n {
                let Some([x, y]) = ir::standard_sample_position(n, s) else {
                    continue;
                };
                let at = self.literal(v2, [x.to_bits(), y.to_bits(), 0, 0])?;
                let s = self.literal(Type::U32, [s, 0, 0, 0])?;
                let is_index = self.compute(
                    Type::BOOL,
                    ir::Op::Binary(ir::Binary::UCmp(ir::Cmp::Eq), index, ValueId(s)),
                )?;
                let both = self.compute(
                    Type::BOOL,
                    ir::Op::Binary(ir::Binary::And, ValueId(is_count), ValueId(is_index)),
                )?;
                position = self.compute(
                    v2,
                    ir::Op::Select {
                        cond: ValueId(both),
                        accept: ValueId(at),
                        reject: ValueId(position),
                    },
                )?;
            }
        }
        Ok(Op::Ir(Box::new(ir::Op::Swizzle(ValueId(position), [0, 1, 0, 0]))))
    }

    /// The integer layer of an array sample: the coordinate's lane `at` rounded to the nearest even and clamped to 0 or
    /// more, as `u32`.
    fn layer_of(&mut self, coord: ValueId, at: u8) -> Result<u32, String> {
        let lane = self.compute(Type::F32, ir::Op::Swizzle(coord, [at, 0, 0, 0]))?;
        let rounded = self.compute(
            Type::F32,
            ir::Op::Unary(ir::Unary::FRound(ir::Rounding::NearestEven), ValueId(lane)),
        )?;
        let zero = self.literal(Type::F32, [0; 4])?;
        let clamped = self.compute(
            Type::F32,
            ir::Op::Binary(
                ir::Binary::FMax(ir::MinMaxNan::MinNum),
                ValueId(rounded),
                ValueId(zero),
            ),
        )?;
        self.compute(Type::U32, ir::Op::Convert(ValueId(clamped), FloatToInt::Saturate))
    }

    /// The texture a sample, gather, load or query names: its shape, whether it is arrayed, and whether it is a depth
    /// texture or a storage image.
    fn texture(&self, r: &ir::Res) -> Result<(TextureDim, bool, bool, bool), String> {
        match &self.resource(r.id)?.kind {
            ResourceKind::Texture {
                dim, arrayed, sample, ..
            } => Ok((*dim, *arrayed, *sample == SampleType::Depth, false)),
            ResourceKind::Storage {
                target: StorageTarget::Image { dim, arrayed, .. },
                ..
            } => Ok((*dim, *arrayed, false, true)),
            k => Err(format!("resource {} ({k:?}) is not a texture", r.id.0)),
        }
    }

    /// Where the lanes of input `slot` reach a derivative unchanged: (argument, lane) per lane of the value.
    fn provenance(&self, v: ValueId) -> Option<[(u32, u8); 4]> {
        let d = self.f.value(v)?;
        match &d.op {
            ir::Op::LoadInput(slot, mask) => {
                let (arg, _) = (*self.inputs.get(*slot as usize)?)?;
                let mut p = [(arg, 0u8); 4];
                for (to, lane) in p.iter_mut().zip(lanes_of(*mask)) {
                    *to = (arg, lane);
                }
                Some(p)
            }
            ir::Op::Swizzle(x, pat) => {
                let from = self.provenance(*x)?;
                let mut p = from;
                for (to, &lane) in p.iter_mut().zip(pat).take(usize::from(d.ty.lanes)) {
                    *to = *from.get(usize::from(lane))?;
                }
                Some(p)
            }
            _ => None,
        }
    }

    /// The operation that computes value `v`, where it is placed.
    fn place(&mut self, v: ValueId) -> Result<(), String> {
        let d = self.value(v)?.clone();
        let t = d.ty;
        let n = t.lanes;
        let op = match &d.op {
            ir::Op::Literal(_) | ir::Op::Undef | ir::Op::Param => return Ok(()),
            ir::Op::Unary(ir::Unary::Derivative(axis, fineness), a) => Op::Deriv {
                src: a.0,
                axis: match axis {
                    ir::Axis::X => 0,
                    ir::Axis::Y => 1,
                },
                fine: *fineness == ir::Fineness::Fine,
                n,
                prov: self.provenance(*a),
            },
            ir::Op::Unary(..)
            | ir::Op::Binary(..)
            | ir::Op::Ternary(..)
            | ir::Op::BitInsert { .. }
            | ir::Op::Select { .. }
            | ir::Op::Compose(_)
            | ir::Op::Swizzle(..)
            | ir::Op::Splat(_)
            | ir::Op::Dot(..)
            | ir::Op::Convert(..)
            | ir::Op::Bitcast(_) => Op::Ir(Box::new(d.op.clone())),
            ir::Op::LoadInput(slot, mask) => {
                let (_, arg) = self
                    .inputs
                    .get(*slot as usize)
                    .copied()
                    .flatten()
                    .ok_or_else(|| format!("input {slot} is not an entry argument"))?;
                let mut pat = [0u8; 4];
                for (p, l) in pat.iter_mut().zip(lanes_of(*mask)) {
                    *p = l;
                }
                Op::Swizzle { src: arg, pat, n }
            }
            ir::Op::LoadInputAt(slot, mask, index) | ir::Op::LoadOutputAt(slot, mask, index) => {
                let points = if matches!(d.op, ir::Op::LoadInputAt(..)) {
                    &self.point_inputs
                } else {
                    &self.point_outputs
                };
                let p =
                    points.get(*slot as usize).copied().flatten().ok_or_else(|| {
                        format!("varying {slot} is not an array of points the function takes")
                    })?;
                let whole = u8::try_from(p.stride).unwrap_or(4);
                let element = self.fresh(Type::vector(Scalar::U32, whole))?;
                self.ops.push((
                    element,
                    Op::Element {
                        loc: p.loc,
                        index: index.0,
                        stride: p.stride,
                        len: p.len,
                        n: whole,
                    },
                ));
                let mut pat = [0u8; 4];
                for (to, l) in pat.iter_mut().zip(lanes_of(*mask)) {
                    *to = l;
                }
                Op::Swizzle { src: element, pat, n }
            }
            ir::Op::LoadPatch(slot, mask) => {
                let mut pat = [0u8; 4];
                for (to, l) in pat.iter_mut().zip(lanes_of(*mask)) {
                    *to = l;
                }
                let given = self.patch_in.get(*slot as usize).copied().flatten();
                let stored = self.patch_out.get(*slot as usize).copied().flatten();
                let src = match (given, stored) {
                    (Some(r), _) => r,
                    (_, Some(word)) => {
                        let four = self.fresh(Type::U32X4)?;
                        self.ops.push((
                            four,
                            Op::Load {
                                loc: Loc::Local(word),
                                n: 4,
                            },
                        ));
                        four
                    }
                    _ => return Err(format!("patch constant {slot} is not one the function has")),
                };
                Op::Swizzle { src, pat, n }
            }
            ir::Op::Builtin(b) => {
                let src = self
                    .builtins
                    .iter()
                    .find(|(x, _)| x == b)
                    .map(|(_, r)| *r)
                    .ok_or_else(|| format!("the pipeline gives the function no {b:?}"))?;
                Op::Swizzle {
                    src,
                    pat: [0, 1, 2, 3],
                    n,
                }
            }
            ir::Op::LoadVar(var) => Op::Load {
                loc: self.var(*var)?.loc,
                n,
            },
            ir::Op::LoadElement(var, index, _) => {
                let p = self.var(*var)?;
                Op::Element {
                    loc: p.loc,
                    index: index.0,
                    stride: p.stride,
                    len: p.len,
                    n,
                }
            }
            ir::Op::LoadBuffer {
                resource, address, ..
            } => {
                let (g, pick) = self.buffer(resource)?;
                let word = self.word_of(address)?;
                Op::Words { g, pick, word, n }
            }
            ir::Op::Sample {
                texture,
                sampler,
                coord,
                level,
                compare,
                offset,
                clamp,
            } => {
                if clamp.is_some() {
                    return Err("a sample with a level-of-detail clamp is not compiled yet".into());
                }
                let (dim, arrayed, depth, _) = self.texture(texture)?;
                let cn = coord_lanes(dim);
                Op::Sample {
                    tex: self.res(texture)?,
                    smp: self.res(sampler)?,
                    coord: coord.0,
                    cn,
                    layer: if arrayed {
                        Some(self.layer_of(*coord, cn)?)
                    } else {
                        None
                    },
                    level: match level {
                        ir::Level::Auto => LevelOp::Auto,
                        ir::Level::Zero => LevelOp::Zero,
                        ir::Level::Exact(x) => LevelOp::Exact(x.0),
                        ir::Level::Bias(x) => LevelOp::Bias(x.0),
                        ir::Level::Grad(x, y) => LevelOp::Gradient(x.0, y.0),
                    },
                    dim: dim_of(dim),
                    offset: offset.map(i32::from),
                    gather: None,
                    depth_ref: compare.map(|c| c.0),
                    depth,
                }
            }
            ir::Op::Gather {
                texture,
                sampler,
                coord,
                component,
                compare,
                offset,
            } => {
                let ir::GatherOffset::Fixed(offset) = offset else {
                    return Err("a gather with an offset the program computes is not compiled yet".into());
                };
                let (dim, arrayed, depth, _) = self.texture(texture)?;
                let cn = coord_lanes(dim);
                Op::Sample {
                    tex: self.res(texture)?,
                    smp: self.res(sampler)?,
                    coord: coord.0,
                    cn,
                    layer: if arrayed {
                        Some(self.layer_of(*coord, cn)?)
                    } else {
                        None
                    },
                    level: LevelOp::Zero,
                    dim: dim_of(dim),
                    offset: offset.map(i32::from),
                    gather: Some(*component),
                    depth_ref: compare.map(|c| c.0),
                    depth,
                }
            }
            ir::Op::LoadTexel {
                texture,
                coord,
                level,
                sample,
                offset,
            } => {
                if typed_buffer(&self.resource(texture.id)?.kind) {
                    // a typed buffer's element is four words (the binding converts from the view's format)
                    let (g, pick) = self.buffer(texture)?;
                    let word = self.word_of(&ir::Address {
                        index: Some(*coord),
                        stride: 16,
                        offset: 0,
                    })?;
                    self.ops.push((v.0, Op::Words { g, pick, word, n: 4 }));
                    return Ok(());
                }
                let (dim, arrayed, _, storage) = self.texture(texture)?;
                let cn = coord_lanes(dim);
                // a texel offset moves the coordinate, at the level loaded
                let coord = if *offset == [0; 3] {
                    *coord
                } else {
                    let t = self.f.ty(*coord).ok_or("a texel coordinate has no type")?;
                    let mut lanes = [0u32; 4];
                    for (to, &o) in lanes.iter_mut().zip(offset).take(usize::from(cn)) {
                        *to = i32::from(o).cast_unsigned();
                    }
                    let moved = self.literal(t, lanes)?;
                    ValueId(self.compute(t, ir::Op::Binary(ir::Binary::IAdd, *coord, ValueId(moved)))?)
                };
                let layer = if arrayed {
                    let r = self.fresh(Type::I32)?;
                    self.ops.push((r, Op::Lane { src: coord.0, i: cn }));
                    Some(r)
                } else {
                    None
                };
                Op::Texel {
                    tex: self.res(texture)?,
                    coord: coord.0,
                    cn,
                    dim: dim_of(dim),
                    storage,
                    layer,
                    sample: sample.map(|s| s.0),
                    mip: level.map(|l| l.0),
                }
            }
            ir::Op::Query(texture, query) => match &self.resource(texture.id)?.kind {
                ResourceKind::Buffer(kind)
                | ResourceKind::Storage {
                    target: StorageTarget::Buffer(kind),
                    ..
                } if *query == ir::Query::Elements => {
                    let (g, pick) = self.buffer(texture)?;
                    let elem_words = match kind {
                        BufferKind::Structured { stride } => stride / 4,
                        BufferKind::Raw => 1,
                        // four 32-bit lanes an element
                        BufferKind::Typed(_) => 4,
                    };
                    let count = Op::ArrLen {
                        gid: g,
                        elem_words,
                        pick: match pick {
                            Pick::One => None,
                            p => Some(p),
                        },
                    };
                    if *kind == BufferKind::Raw {
                        // the bytes of a raw buffer: its words times four
                        let words = self.fresh(Type::U32)?;
                        self.ops.push((words, count));
                        let four = self.literal(Type::U32, [4, 0, 0, 0])?;
                        Op::Ir(Box::new(ir::Op::Binary(
                            ir::Binary::IMul,
                            ValueId(words),
                            ValueId(four),
                        )))
                    } else {
                        count
                    }
                }
                _ if matches!(query, ir::Query::SamplePosition(_)) => {
                    let ir::Query::SamplePosition(index) = *query else {
                        return Err("a sample position without its index".into());
                    };
                    self.sample_position(texture, index)?
                }
                _ => {
                    let (dim, _, _, _) = self.texture(texture)?;
                    let (kind, level) = match query {
                        ir::Query::Size(level) => (QueryKind::Size, level.map(|l| l.0)),
                        ir::Query::Layers => (QueryKind::Layers, None),
                        ir::Query::Levels => (QueryKind::Levels, None),
                        ir::Query::Samples => (QueryKind::Samples, None),
                        q => return Err(format!("the query {q:?} is not compiled yet")),
                    };
                    Op::Query {
                        tex: self.res(texture)?,
                        dim: dim_of(dim),
                        kind,
                        level,
                    }
                }
            },
            op => return Err(format!("{} is not compiled yet", op_name(op))),
        };
        self.ops.push((v.0, op));
        Ok(())
    }

    fn var(&self, v: VarId) -> Result<Place, String> {
        self.vars
            .get(v.0 as usize)
            .copied()
            .ok_or_else(|| format!("variable {} is not in the function", v.0))
    }

    /// `n` lanes of register `src` stored to output `slot`'s lanes of `mask`.
    fn store_output(&mut self, out: &mut Vec<St>, slot: u32, mask: u8, src: u32) -> Result<(), String> {
        let (base, _) = *self
            .outputs
            .get(slot as usize)
            .ok_or_else(|| format!("output {slot} is not in the interface"))?;
        self.store_lanes(out, base, mask, src)
    }

    /// Lanes `mask` of the four words at `base` take the lanes of register `src`, in order.
    fn store_lanes(&mut self, out: &mut Vec<St>, base: u32, mask: u8, src: u32) -> Result<(), String> {
        self.store_masked(out, mask, src, |lane, src, n| St::Store {
            loc: Loc::Local(base.saturating_add(u32::from(lane))),
            src,
            n,
        })
    }

    /// Lanes `mask` of four words take the lanes of register `src`, in order: `put(lane, register, n)` stores `n`
    /// lanes of a register from lane `lane` on, once for the whole mask where its lanes are consecutive, else once
    /// for each lane.
    fn store_masked(
        &mut self,
        out: &mut Vec<St>,
        mask: u8,
        src: u32,
        put: impl Fn(u8, u32, u8) -> St,
    ) -> Result<(), String> {
        let lanes = lanes_of(mask);
        let first = lanes.first().copied().unwrap_or(0);
        let contiguous = lanes.iter().zip(first..).all(|(&l, k)| l == k);
        if contiguous {
            return self.statement(out, put(first, src, u8::try_from(lanes.len()).unwrap_or(4)));
        }
        for (k, lane) in lanes.into_iter().enumerate() {
            let one = self.fresh(Type::U32)?;
            self.ops.push((
                one,
                Op::Lane {
                    src,
                    i: u8::try_from(k).unwrap_or(0),
                },
            ));
            self.statement(out, put(lane, one, 1))?;
        }
        Ok(())
    }

    /// A mesh program's meshlet store: the counts, lanes of a vertex's output, or a primitive's indices. A vertex or
    /// primitive past the most the program states is not written.
    fn meshlet(&mut self, out: &mut Vec<St>, node: &Node) -> Result<(), String> {
        let m = self
            .meshlet
            .clone()
            .ok_or("a meshlet store outside a mesh program")?;
        match *node {
            Node::SetMeshCounts { vertices, primitives } => {
                for (word, v) in [(0, vertices), (1, primitives)] {
                    self.statement(
                        out,
                        St::Store {
                            loc: Loc::Global(m.global, word),
                            src: v.0,
                            n: 1,
                        },
                    )?;
                }
                Ok(())
            }
            Node::StoreVertex {
                vertex,
                slot,
                mask,
                value,
            } => {
                let at = m
                    .output_at
                    .get(slot as usize)
                    .copied()
                    .ok_or_else(|| format!("vertex output {slot} is not in the interface"))?
                    .saturating_add(2);
                self.store_masked(out, mask, value.0, |lane, src, n| St::StoreElement {
                    loc: Loc::Global(m.global, at.saturating_add(u32::from(lane))),
                    index: vertex.0,
                    stride: m.vertex,
                    len: m.max_vertices,
                    src,
                    n,
                })
            }
            Node::StoreIndices { primitive, indices } => self.statement(
                out,
                St::StoreElement {
                    loc: Loc::Global(m.global, m.primitives_at),
                    index: primitive.0,
                    stride: m.index,
                    len: m.max_primitives,
                    src: indices.0,
                    n: u8::try_from(m.index).unwrap_or(3),
                },
            ),
            _ => Err("not a meshlet store".into()),
        }
    }

    /// `Node::DispatchMesh`: at a barrier each invocation counts itself into the launch's first word, and the first
    /// invocation (the one that finds it zero: the atomic operations run in lane order) writes the thread groups and
    /// the payload's words after it; then the invocation ends.
    fn dispatch_mesh(
        &mut self,
        out: &mut Vec<St>,
        groups: ValueId,
        payload: ir::VarId,
    ) -> Result<(), String> {
        let launch = self
            .launch
            .clone()
            .ok_or("a mesh launch outside an amplification program")?;
        self.statement(out, St::Barrier)?;
        let zero = self.literal(Type::U32, [0; 4])?;
        let one = self.literal(Type::U32, [1, 0, 0, 0])?;
        let reached = self.fresh(Type::U32)?;
        self.statement(
            out,
            St::AtomicWord {
                at: WordAt::Element {
                    loc: Loc::Global(launch.global, 0),
                    index: zero,
                    len: 1,
                },
                fun: AtomOp::Add,
                signed: false,
                value: one,
                compare: None,
                result: reached,
            },
        )?;
        let p = self.var(payload)?;
        let words = p.stride.saturating_mul(p.len);
        if words != launch.payload {
            return Err(format!(
                "a mesh launch of {words} payload words, the facts state {}",
                launch.payload
            ));
        }
        let mut first = Vec::new();
        self.statement(
            &mut first,
            St::Store {
                loc: Loc::Global(launch.global, 1),
                src: groups.0,
                n: 3,
            },
        )?;
        for k in 0..words {
            let w = self.fresh(Type::U32)?;
            let loc = match p.loc {
                Loc::Local(at) => Loc::Local(at.saturating_add(k)),
                Loc::Global(g, at) => Loc::Global(g, at.saturating_add(k)),
                Loc::Elem(..) => return Err("a task payload in an array of buffers".into()),
            };
            self.ops.push((w, Op::Load { loc, n: 1 }));
            self.statement(
                &mut first,
                St::Store {
                    loc: Loc::Global(launch.global, k.saturating_add(4)),
                    src: w,
                    n: 1,
                },
            )?;
        }
        out.push(St::If {
            c: reached,
            a: Vec::new(),
            r: first,
        });
        self.ret(out)
    }

    /// The local words of the variable a ray tracing operation hands over: where they start and how many there are.
    fn handed(&self, var: VarId) -> Result<(u32, u32), String> {
        let p = self.var(var)?;
        match p.loc {
            Loc::Local(at) => Ok((at, p.stride.saturating_mul(p.len))),
            _ => Err("ray tracing memory outside the invocation's own".into()),
        }
    }

    /// Leave the function with the outputs stored so far.
    fn ret(&mut self, out: &mut Vec<St>) -> Result<(), String> {
        let mut members = Vec::with_capacity(self.returned.len());
        for (load, member) in self.returned.clone() {
            match (load, member) {
                (Some((word, width)), Member::Reg(..)) => {
                    let r = self.fresh(Type::U32X4)?;
                    self.ops.push((
                        r,
                        Op::Load {
                            loc: Loc::Local(word),
                            n: width,
                        },
                    ));
                    members.push(Member::Reg(r, width));
                }
                (_, m) => members.push(m),
            }
        }
        self.statement(out, St::Return(members))
    }

    /// An atomic operation or a counter change: a statement of its own, its value the word or count it gives.
    /// `Op::Wave`: the executor's wave statement, as the naga-path compile makes it of naga's subgroup operation (a
    /// lane read at a number is its shuffle). Each value is 32-bit lanes.
    fn wave(&mut self, v: ValueId, w: ir::Wave, out: &mut Vec<St>) -> Result<(), String> {
        let lanes = |c: &Self, x: ValueId| -> Result<u8, String> {
            let t = c.value(x)?.ty;
            if t.scalar != Scalar::Bool && t.scalar.bits() != 32 {
                return Err(format!("a wave operation of {t} is not run"));
            }
            Ok(t.lanes)
        };
        let (mode, value) = match w {
            ir::Wave::Ballot(c) => {
                return self.statement(
                    out,
                    St::WaveBallot {
                        predicate: Some(c.0),
                        result: v.0,
                    },
                );
            }
            ir::Wave::Collective { op, scan, value } => {
                let n = lanes(self, value)?;
                let t = self.value(value)?.ty;
                let kind = if t.scalar.is_float() {
                    WaveKind::F
                } else if t.scalar.is_signed() {
                    WaveKind::I
                } else {
                    WaveKind::U
                };
                let op = match op {
                    ir::WaveOp::Add => WaveOp::Add,
                    ir::WaveOp::Mul => WaveOp::Mul,
                    ir::WaveOp::Min => WaveOp::Min,
                    ir::WaveOp::Max => WaveOp::Max,
                    ir::WaveOp::And => WaveOp::And,
                    ir::WaveOp::Or => WaveOp::Or,
                    ir::WaveOp::Xor => WaveOp::Xor,
                    ir::WaveOp::All => WaveOp::All,
                    ir::WaveOp::Any => WaveOp::Any,
                };
                let scan = match scan {
                    ir::WaveScan::Reduce => WaveScan::Reduce,
                    ir::WaveScan::Inclusive => WaveScan::Inclusive,
                    ir::WaveScan::Exclusive => WaveScan::Exclusive,
                };
                return self.statement(
                    out,
                    St::WaveOp {
                        op,
                        scan,
                        value: value.0,
                        n,
                        kind,
                        result: v.0,
                    },
                );
            }
            ir::Wave::First(a) => (WaveGather::First, a),
            ir::Wave::Lane(a, l) => (WaveGather::Shuffle(l.0), a),
            ir::Wave::QuadLane(a, l) => (WaveGather::QuadLane(l.0), a),
            ir::Wave::QuadSwap(a, s) => (
                WaveGather::QuadSwap(match s {
                    ir::QuadSwap::X => 1,
                    ir::QuadSwap::Y => 2,
                    ir::QuadSwap::Diagonal => 3,
                }),
                a,
            ),
        };
        let n = lanes(self, value)?;
        self.statement(
            out,
            St::WaveGather {
                mode,
                value: value.0,
                n,
                result: v.0,
            },
        )
    }

    fn effect(&mut self, v: ValueId, out: &mut Vec<St>) -> Result<(), String> {
        let d = self.value(v)?.clone();
        match &d.op {
            ir::Op::Wave(w) => self.wave(v, *w, out),
            ir::Op::ReportHit { t, kind, attributes } => {
                let attributes = self.handed(*attributes)?;
                self.statement(
                    out,
                    St::RtReport {
                        t: t.0,
                        kind: kind.0,
                        attributes,
                        result: Some(v.0),
                    },
                )
            }
            ir::Op::Counter(r, op) => {
                let tex = self.res(r)?;
                self.statement(
                    out,
                    St::Counter {
                        tex,
                        decrement: *op == ir::CounterOp::Decrement,
                        result: v.0,
                    },
                )
            }
            ir::Op::Atomic {
                memory,
                op,
                value,
                compare,
            } => {
                let (fun, signed) = match op {
                    ir::AtomicOp::Add => (AtomOp::Add, false),
                    ir::AtomicOp::Sub => (AtomOp::Sub, false),
                    ir::AtomicOp::And => (AtomOp::And, false),
                    ir::AtomicOp::Or => (AtomOp::Or, false),
                    ir::AtomicOp::Xor => (AtomOp::Xor, false),
                    ir::AtomicOp::SMin => (AtomOp::Min, true),
                    ir::AtomicOp::SMax => (AtomOp::Max, true),
                    ir::AtomicOp::UMin => (AtomOp::Min, false),
                    ir::AtomicOp::UMax => (AtomOp::Max, false),
                    ir::AtomicOp::Exchange => (AtomOp::Exchange, false),
                    ir::AtomicOp::CompareExchange => (AtomOp::CompareExchange, false),
                };
                let compare = compare.map(|c| c.0);
                let at = match memory {
                    ir::Memory::Buffer(r, address) => {
                        let (g, pick) = self.buffer(r)?;
                        let word = self.word_of(address)?;
                        WordAt::Buffer { g, pick, word }
                    }
                    // a typed buffer's element is four words: its first lane is the one an atomic operation reaches
                    ir::Memory::Image(r, coord) if typed_buffer(&self.resource(r.id)?.kind) => {
                        let (g, pick) = self.buffer(r)?;
                        let word = self.word_of(&ir::Address {
                            index: Some(*coord),
                            stride: 16,
                            offset: 0,
                        })?;
                        WordAt::Buffer { g, pick, word }
                    }
                    ir::Memory::Image(r, coord) => {
                        let (dim, arrayed, _, _) = self.texture(r)?;
                        let cn = coord_lanes(dim);
                        let layer = if arrayed {
                            let l = self.fresh(Type::U32)?;
                            self.ops.push((l, Op::Lane { src: coord.0, i: cn }));
                            Some(l)
                        } else {
                            None
                        };
                        let image_op = match fun {
                            AtomOp::Add => ImageAtomicOp::Add,
                            AtomOp::Sub => ImageAtomicOp::Sub,
                            AtomOp::And => ImageAtomicOp::And,
                            AtomOp::Or => ImageAtomicOp::Or,
                            AtomOp::Xor => ImageAtomicOp::Xor,
                            AtomOp::Min => ImageAtomicOp::Min,
                            AtomOp::Max => ImageAtomicOp::Max,
                            AtomOp::Exchange => ImageAtomicOp::Exchange,
                            AtomOp::CompareExchange => ImageAtomicOp::CompareExchange,
                        };
                        let tex = self.res(r)?;
                        return self.statement(
                            out,
                            St::ImageAtomic {
                                tex,
                                dim: dim_of(dim),
                                coord: coord.0,
                                cn,
                                layer,
                                op: image_op,
                                signed,
                                value: value.0,
                                compare,
                                result: Some(v.0),
                            },
                        );
                    }
                    ir::Memory::Var(var, index) => {
                        let p = self.var(*var)?;
                        if p.stride != 1 {
                            return Err("an atomic operation on a variable of more than one word".into());
                        }
                        let index = match index {
                            Some(i) => i.0,
                            None => self.literal(Type::U32, [0; 4])?,
                        };
                        WordAt::Element {
                            loc: p.loc,
                            index,
                            len: p.len,
                        }
                    }
                };
                self.statement(
                    out,
                    St::AtomicWord {
                        at,
                        fun,
                        signed,
                        value: value.0,
                        compare,
                        result: v.0,
                    },
                )
            }
            _ => self.place(v),
        }
    }

    fn node(&mut self, n: &Node, out: &mut Vec<St>) -> Result<(), String> {
        match n {
            Node::Value(v) => self.effect(*v, out),
            Node::StoreVar(var, v) => {
                let p = self.var(*var)?;
                let n = self.value(*v)?.ty.lanes;
                self.statement(
                    out,
                    St::Store {
                        loc: p.loc,
                        src: v.0,
                        n,
                    },
                )
            }
            Node::StoreElement(var, index, v, _) => {
                let p = self.var(*var)?;
                let n = self.value(*v)?.ty.lanes;
                self.statement(
                    out,
                    St::StoreElement {
                        loc: p.loc,
                        index: index.0,
                        stride: p.stride,
                        len: p.len,
                        src: v.0,
                        n,
                    },
                )
            }
            Node::StoreOutput(slot, mask, v) => self.store_output(out, *slot, *mask, v.0),
            Node::DispatchMesh { groups, payload } => self.dispatch_mesh(out, *groups, *payload),
            Node::TraceRay { scene, ray, payload } => {
                if scene.index.is_some() {
                    return Err("a ray traced into an element of an array of acceleration structures".into());
                }
                let payload = self.handed(*payload)?;
                self.statement(
                    out,
                    St::RtTrace {
                        scene: scene.id,
                        operands: ray.values().map(|v| v.0),
                        payload,
                    },
                )
            }
            Node::CallShader { index, parameter } => {
                let parameter = self.handed(*parameter)?;
                self.statement(
                    out,
                    St::RtCall {
                        index: index.0,
                        parameter,
                    },
                )
            }
            Node::SetMeshCounts { .. } | Node::StoreVertex { .. } | Node::StoreIndices { .. } => {
                self.meshlet(out, n)
            }
            Node::StorePatch(slot, mask, v) => {
                let base = self
                    .patch_out
                    .get(*slot as usize)
                    .copied()
                    .flatten()
                    .ok_or_else(|| format!("patch constant {slot} is not one the function stores"))?;
                self.store_lanes(out, base, *mask, v.0)
            }
            Node::Emit(stream) => {
                if *stream != 0 {
                    return Err(format!("a vertex of stream {stream}: only stream 0 is carried"));
                }
                self.emit(out)
            }
            Node::Cut(stream) => {
                if *stream != 0 {
                    return Err(format!(
                        "a strip cut on stream {stream}: only stream 0 is carried"
                    ));
                }
                let e = self
                    .emitted
                    .clone()
                    .ok_or("a strip cut outside a geometry program")?;
                let one = self.literal(Type::U32, [1, 0, 0, 0])?;
                self.statement(
                    out,
                    St::Store {
                        loc: Loc::Local(e.pending),
                        src: one,
                        n: 1,
                    },
                )
            }
            Node::Demote => self.statement(out, St::Kill),
            Node::StoreBuffer {
                resource,
                address,
                value,
            } => {
                let (g, pick) = self.buffer(resource)?;
                let word = self.word_of(address)?;
                let n = self.value(*value)?.ty.lanes;
                self.statement(
                    out,
                    St::StoreWords {
                        g,
                        pick,
                        word,
                        src: value.0,
                        n,
                    },
                )
            }
            Node::StoreTexel { image, coord, value } if typed_buffer(&self.resource(image.id)?.kind) => {
                // a typed buffer's element is four words
                let (g, pick) = self.buffer(image)?;
                let word = self.word_of(&ir::Address {
                    index: Some(*coord),
                    stride: 16,
                    offset: 0,
                })?;
                self.statement(
                    out,
                    St::StoreWords {
                        g,
                        pick,
                        word,
                        src: value.0,
                        n: 4,
                    },
                )
            }
            Node::StoreTexel { image, coord, value } => {
                let (dim, arrayed, _, _) = self.texture(image)?;
                let cn = coord_lanes(dim);
                let layer = if arrayed {
                    let r = self.fresh(Type::U32)?;
                    self.ops.push((r, Op::Lane { src: coord.0, i: cn }));
                    Some(r)
                } else {
                    None
                };
                let vn = self.value(*value)?.ty.lanes;
                self.statement(
                    out,
                    St::ImageStore {
                        tex: self.res(image)?,
                        dim: dim_of(dim),
                        coord: coord.0,
                        cn,
                        layer,
                        value: value.0,
                        vn,
                    },
                )
            }
            Node::Barrier(_) => self.statement(out, St::Barrier),
            Node::If {
                cond,
                accept,
                reject,
                results,
            } => {
                let (mut a, mut r) = (Vec::new(), Vec::new());
                self.flush(out)?;
                self.region(accept, Yield::Into(results), &mut a)?;
                self.region(reject, Yield::Into(results), &mut r)?;
                out.push(St::If { c: cond.0, a, r });
                Ok(())
            }
            Node::Block { label, body, results } if breaks_to(body, *label) => {
                self.flush(out)?;
                self.frames.push(Frame {
                    label: *label,
                    params: None,
                    results: results.clone(),
                    block: true,
                });
                let mut b = Vec::new();
                let done = self.region(body, Yield::Into(results), &mut b);
                self.frames.pop();
                done?;
                // falling off its end leaves it
                b.push(St::Break);
                out.push(St::Loop {
                    body: b,
                    cont: Vec::new(),
                    break_if: None,
                });
                Ok(())
            }
            Node::Block { body, results, .. } => self.region(body, Yield::Into(results), out),
            Node::Loop {
                label,
                params,
                inits,
                body,
                results,
            } => {
                for (p, i) in params.iter().zip(inits) {
                    self.copy(p.0, i.0);
                }
                self.flush(out)?;
                self.frames.push(Frame {
                    label: *label,
                    params: Some(params.clone()),
                    results: results.clone(),
                    block: false,
                });
                let mut b = Vec::new();
                let done = self.region(body, Yield::Leave(results), &mut b);
                self.frames.pop();
                done?;
                out.push(St::Loop {
                    body: b,
                    cont: Vec::new(),
                    break_if: None,
                });
                Ok(())
            }
            Node::Switch {
                label,
                selector,
                cases,
                default,
                results,
            } => {
                self.flush(out)?;
                self.frames.push(Frame {
                    label: *label,
                    params: None,
                    results: results.clone(),
                    block: false,
                });
                let mut arms = Vec::new();
                let mut done = Ok(());
                for case in cases {
                    let mut b = Vec::new();
                    done = self.region(&case.body, Yield::Into(results), &mut b);
                    if done.is_err() {
                        break;
                    }
                    let (last, rest) = case.values.split_last().ok_or("a switch case without a value")?;
                    // the other values fall into the last one's body
                    arms.extend(rest.iter().map(|v| (Some(*v), true, Vec::new())));
                    arms.push((Some(*last), false, b));
                }
                if done.is_ok() {
                    let mut b = Vec::new();
                    done = self.region(default, Yield::Into(results), &mut b);
                    arms.push((None, false, b));
                }
                self.frames.pop();
                done?;
                out.push(St::Switch {
                    sel: selector.0,
                    cases: arms,
                });
                Ok(())
            }
        }
    }

    /// The frame `label` names, which must be the innermost loop (`params`) or loop or switch.
    fn frame(&self, label: ir::LabelId, loop_only: bool) -> Result<&Frame, String> {
        let f = self
            .frames
            .iter()
            .rev()
            .find(|f| !loop_only || f.params.is_some() || f.block)
            .ok_or("an exit outside a loop or switch")?;
        if f.label != label {
            return Err(
                "an exit past the innermost loop or switch: run iridia_passes::break_flags first".into(),
            );
        }
        Ok(f)
    }

    fn region(&mut self, r: &Region, y: Yield, out: &mut Vec<St>) -> Result<(), String> {
        for n in &r.nodes {
            self.node(n, out)?;
        }
        match r.end.as_ref().ok_or("a region without an end")? {
            End::Yield(vs) => match y {
                Yield::Into(results) => {
                    for (to, from) in results.iter().zip(vs) {
                        self.copy(to.0, from.0);
                    }
                    self.flush(out)
                }
                Yield::Leave(results) => {
                    for (to, from) in results.iter().zip(vs) {
                        self.copy(to.0, from.0);
                    }
                    self.statement(out, St::Break)
                }
                Yield::Return => self.ret(out),
            },
            End::Br(label, vs) => {
                let results = self.frame(*label, false)?.results.clone();
                for (to, from) in results.iter().zip(vs) {
                    self.copy(to.0, from.0);
                }
                self.statement(out, St::Break)
            }
            End::Continue(label, vs) => {
                let params = self
                    .frame(*label, true)?
                    .params
                    .clone()
                    .ok_or("a continue of a switch")?;
                // through temporaries: a new value may read the parameters it replaces
                let mut temps = Vec::with_capacity(vs.len());
                for v in vs {
                    let t = self.value(*v)?.ty;
                    let temp = self.fresh(t)?;
                    self.copy(temp, v.0);
                    temps.push(temp);
                }
                for (p, t) in params.iter().zip(temps) {
                    self.copy(p.0, t);
                }
                self.statement(out, St::Continue)
            }
            End::Return => self.ret(out),
            End::Kill => {
                self.statement(out, St::Kill)?;
                self.ret(out)
            }
            End::IgnoreHit => {
                self.statement(out, St::RtDecide(Decision::Ignore))?;
                self.ret(out)
            }
            End::AcceptHitAndEndSearch => {
                self.statement(out, St::RtDecide(Decision::AcceptAndEnd))?;
                self.ret(out)
            }
            End::Unreachable => self.flush(out),
        }
    }
}

/// Whether a resource is a typed buffer (read-only or storage): four words an element, the binding converting from
/// the view's format.
fn typed_buffer(kind: &ResourceKind) -> bool {
    matches!(
        kind,
        ResourceKind::Buffer(BufferKind::Typed(_))
            | ResourceKind::Storage {
                target: StorageTarget::Buffer(BufferKind::Typed(_)),
                ..
            }
    )
}

/// The name of an operation the compile does not take, for its refusal.
fn op_name(op: &ir::Op) -> &'static str {
    match op {
        ir::Op::LoadState(_) => "a pipeline state read (fold it with iridia_passes::specialise)",
        _ => "this operation",
    }
}

impl Cx<'_> {
    /// A geometry program's `emit`: past the most vertices nothing, else each output as stored becomes the next
    /// vertex's, the strip start its flag, and the count one more.
    fn emit(&mut self, out: &mut Vec<St>) -> Result<(), String> {
        let e = self.emitted.clone().ok_or("an emit outside a geometry program")?;
        let count = self.fresh(Type::U32)?;
        self.ops.push((
            count,
            Op::Load {
                loc: Loc::Local(e.count),
                n: 1,
            },
        ));
        let max = self.literal(Type::U32, [e.max, 0, 0, 0])?;
        let inside = self.compute(
            Type::vector(Scalar::Bool, 1),
            ir::Op::Binary(ir::Binary::UCmp(ir::Cmp::Lt), ValueId(count), ValueId(max)),
        )?;
        self.flush(out)?;
        let mut a = Vec::new();
        for &(slot, base, width) in &e.arrays {
            let (word, _) = *self
                .outputs
                .get(slot)
                .ok_or_else(|| format!("output {slot} is not in the interface"))?;
            let v = self.fresh(Type::vector(Scalar::U32, width))?;
            self.ops.push((
                v,
                Op::Load {
                    loc: Loc::Local(word),
                    n: width,
                },
            ));
            self.statement(
                &mut a,
                St::StoreElement {
                    loc: Loc::Local(base),
                    index: count,
                    stride: u32::from(width),
                    len: e.max,
                    src: v,
                    n: width,
                },
            )?;
        }
        let pending = self.fresh(Type::U32)?;
        self.ops.push((
            pending,
            Op::Load {
                loc: Loc::Local(e.pending),
                n: 1,
            },
        ));
        self.statement(
            &mut a,
            St::StoreElement {
                loc: Loc::Local(e.strips),
                index: count,
                stride: 1,
                len: e.max,
                src: pending,
                n: 1,
            },
        )?;
        let zero = self.literal(Type::U32, [0; 4])?;
        self.statement(
            &mut a,
            St::Store {
                loc: Loc::Local(e.pending),
                src: zero,
                n: 1,
            },
        )?;
        let one = self.literal(Type::U32, [1, 0, 0, 0])?;
        let next = self.compute(
            Type::U32,
            ir::Op::Binary(ir::Binary::IAdd, ValueId(count), ValueId(one)),
        )?;
        self.statement(
            &mut a,
            St::Store {
                loc: Loc::Local(e.count),
                src: next,
                n: 1,
            },
        )?;
        self.flush(&mut a)?;
        out.push(St::If {
            c: inside,
            a,
            r: Vec::new(),
        });
        Ok(())
    }
}

/// How a compiled function takes its arguments: per argument its register and lanes (none for an array), where its
/// registers start among those passed and how many there are, its array arguments, and the built-in each carries.
struct Arguments {
    args: Vec<(u32, u8)>,
    arg_at: Vec<usize>,
    arg_len: usize,
    array_args: Vec<(usize, u32, u32, u8)>,
    /// The pointer arguments (a ray tracing program's payload and attributes): where their words start among the
    /// passed registers, the space of the variable they fill and its words.
    pointers: Vec<(usize, ir::Space, u32)>,
    builtins: Vec<Option<naga::BuiltIn>>,
}

/// The state of a compile of function `f` of `p` before its arguments and results are placed: its literals.
fn start<'p>(p: &'p ir::Program, f: &'p Function) -> Result<Cx<'p>, String> {
    let interface = &p.interface;
    let patch_constants = match &interface.facts {
        StageFacts::Hull(h) => h.patch_constants.len(),
        StageFacts::Domain(d) => d.patch_constants.len(),
        _ => 0,
    };
    let mut cx = Cx {
        f,
        interface,
        types: f.values.iter().map(|d| d.ty).collect(),
        ops: Vec::new(),
        run_from: 0,
        inits: Vec::new(),
        local_words: 0,
        globals: Vec::new(),
        buffer_of: BTreeMap::new(),
        vars: Vec::new(),
        inputs: vec![None; interface.inputs.len()],
        point_inputs: vec![None; interface.inputs.len()],
        point_outputs: vec![None; interface.outputs.len()],
        patch_in: vec![None; patch_constants],
        patch_out: vec![None; patch_constants],
        builtins: Vec::new(),
        emitted: None,
        meshlet: None,
        launch: None,
        outputs: Vec::new(),
        returned: Vec::new(),
        frames: Vec::new(),
    };
    for d in &f.values {
        if !iridia_exec_ir::runs(d.ty) {
            return Err(format!("{} values are not compiled yet", d.ty));
        }
    }
    // the literals, and an undefined value as zero
    for (i, d) in f.values.iter().enumerate() {
        match &d.op {
            ir::Op::Literal(bits) => cx
                .inits
                .push((number(i)?, iridia_exec_ir::literal_words(bits, d.ty))),
            ir::Op::Undef => cx.inits.push((number(i)?, [0; 4])),
            _ => {}
        }
    }
    Ok(cx)
}

/// Compile the entry function of `p`, a program of a stage that is an entry point (vertex, pixel, compute, mesh,
/// amplification).
pub(crate) fn compile(p: &ir::Program) -> Result<Executable, String> {
    if !matches!(
        p.stage,
        Stage::Vertex | Stage::Pixel | Stage::Compute | Stage::Mesh | Stage::Amplification
    ) || p.interface.facts == StageFacts::VertexWithoutPosition
    {
        return Err(format!(
            "a {:?} program is made of functions: compile it with Executable::function_from_ir",
            p.stage
        ));
    }
    let io = iridia::entry_io(&p.interface)?;
    let mut cx = start(p, &p.entry)?;
    // the entry arguments: one register each, the varying's lanes first
    let mut args = Vec::with_capacity(io.arguments.len());
    let mut builtins = Vec::with_capacity(io.arguments.len());
    for (k, item) in io.arguments.iter().enumerate() {
        let width = match item.ty {
            iridia::IoType::Value(v) => v.width,
            ref t => return Err(format!("an entry argument of {t:?}")),
        };
        let r = cx.fresh(Type::U32X4)?;
        for &i in &item.varyings {
            if let Some(slot) = cx.inputs.get_mut(i) {
                *slot = Some((number(k)?, r));
            }
        }
        args.push((r, width));
        builtins.push(match item.binding {
            naga::Binding::BuiltIn(b) => Some(b),
            naga::Binding::Location { .. } => None,
        });
    }
    // the outputs: four words each in local memory, the clip distances one after another
    let mut outputs = vec![None; p.interface.outputs.len()];
    for item in &io.results {
        match item.ty {
            iridia::IoType::Value(v) => {
                let word = cx.grow(4)?;
                for &i in &item.varyings {
                    if let Some(o) = outputs.get_mut(i) {
                        *o = Some((word, 4));
                    }
                }
                cx.returned.push((Some((word, v.width)), Member::Reg(0, v.width)));
            }
            iridia::IoType::ClipDistances(n) => {
                let first = cx.local_words;
                for &i in &item.varyings {
                    let width = p.interface.outputs.get(i).map_or(1, |v| u32::from(v.value.width));
                    let word = cx.grow(width)?;
                    if let Some(o) = outputs.get_mut(i) {
                        *o = Some((word, width));
                    }
                }
                cx.returned.push((
                    None,
                    Member::Mem {
                        word: first,
                        count: n,
                        width: 1,
                        keep: None,
                    },
                ));
            }
            ref t => return Err(format!("an entry result of {t:?}")),
        }
    }
    let n_args = args.len();
    let arguments = Arguments {
        args,
        arg_at: (0..n_args).collect(),
        arg_len: n_args,
        array_args: Vec::new(),
        pointers: Vec::new(),
        builtins,
    };
    finish(p, cx, outputs, arguments)
}

/// The value the pipeline gives that a stage function's argument of this name carries.
fn builtin_named(name: &str) -> Option<ir::Builtin> {
    Some(match name {
        "SV_OutputControlPointID" => ir::Builtin::OutputControlPoint,
        "SV_PrimitiveID" => ir::Builtin::PrimitiveId,
        "SV_DomainLocation" => ir::Builtin::DomainLocation,
        "SV_GSInstanceID" => ir::Builtin::GsInstance,
        _ => return None,
    })
}

/// Compile function `f` of `p` (its entry, or a hull program's patch constant function) as `sig`, its stage function
/// of [`iridia::stage_functions`], takes and gives it: the same registers in and members out as
/// [`Executable::function`] compiles a naga module's function with.
pub(crate) fn compile_function(
    p: &ir::Program,
    f: &Function,
    sig: &FunctionSig,
) -> Result<Executable, String> {
    let mut cx = start(p, f)?;
    let mut args = Vec::with_capacity(sig.arguments.len());
    let mut arg_at = Vec::with_capacity(sig.arguments.len());
    let mut array_args = Vec::new();
    let mut pointers = Vec::new();
    let mut at = 0usize;
    for (k, item) in sig.arguments.iter().enumerate() {
        arg_at.push(at);
        match item.ty {
            IoType::Array(carried, n) => {
                let stride = u32::from(carried.width);
                let base = cx.grow(
                    stride
                        .checked_mul(n)
                        .ok_or("an array argument past the words of memory")?,
                )?;
                array_args.push((at, base, n, carried.width));
                args.push((0, 0));
                let place = Place {
                    loc: Loc::Local(base),
                    stride,
                    len: n,
                };
                let points = match item.list {
                    ItemList::Inputs => &mut cx.point_inputs,
                    ItemList::Outputs => &mut cx.point_outputs,
                    _ => return Err(format!("the array argument {} carries no points", item.name)),
                };
                let slot = item
                    .varying
                    .and_then(|v| points.get_mut(v))
                    .ok_or_else(|| format!("the array argument {} carries no varying", item.name))?;
                *slot = Some(place);
                at = at.saturating_add(n as usize);
            }
            IoType::Value(v) => {
                let r = cx.fresh(Type::U32X4)?;
                args.push((r, v.width));
                match (item.list, item.varying) {
                    (ItemList::Inputs, Some(i)) => {
                        if let Some(slot) = cx.inputs.get_mut(i) {
                            *slot = Some((number(k)?, r));
                        }
                    }
                    (ItemList::PatchConstants, Some(i)) => {
                        if let Some(slot) = cx.patch_in.get_mut(i) {
                            *slot = Some(r);
                        }
                    }
                    (ItemList::Pipeline, None) => {
                        let b = builtin_named(&item.name).ok_or_else(|| {
                            format!("the argument {} is no value the pipeline gives", item.name)
                        })?;
                        cx.builtins.push((b, r));
                    }
                    _ => {
                        return Err(format!(
                            "the argument {} carries nothing the function reads",
                            item.name
                        ));
                    }
                }
                at = at.saturating_add(1);
            }
            IoType::Memory => {
                // a ray tracing program's payload (or callable parameter) and the hit's attributes: the words of the
                // program's variable of that space, four to a register (none where the program has no such variable)
                let space = match item.name.as_str() {
                    "payload" => ir::Space::RayPayload,
                    "attributes" => ir::Space::HitAttributes,
                    other => return Err(format!("the pointer argument {other} is no ray tracing memory")),
                };
                let words = match f.vars.iter().find(|v| v.space == space) {
                    Some(v) => u32::from(v.ty.lanes)
                        .checked_mul(v.len.unwrap_or(1))
                        .ok_or("a variable past the words of memory")?,
                    None => 0,
                };
                args.push((0, 0));
                pointers.push((at, space, words));
                at = at.saturating_add(words.div_ceil(4) as usize);
            }
            ref t => return Err(format!("an argument of {t:?}")),
        }
    }
    // the results: an output or patch constant in four words of local memory each; a geometry program's vertices as
    // arrays, how many there are and where strips start
    let mut outputs = vec![None; p.interface.outputs.len()];
    let mut emitted = Emitted {
        count: 0,
        pending: 0,
        max: 0,
        arrays: Vec::new(),
        strips: 0,
    };
    let mut geometry = false;
    for item in &sig.result {
        match (item.ty, item.list, item.varying) {
            (IoType::Value(v), ItemList::Outputs, Some(i)) => {
                let word = cx.grow(4)?;
                if let Some(o) = outputs.get_mut(i) {
                    *o = Some((word, 4));
                }
                cx.returned.push((Some((word, v.width)), Member::Reg(0, v.width)));
            }
            (IoType::Value(v), ItemList::PatchConstants, Some(i)) => {
                let word = cx.grow(4)?;
                if let Some(o) = cx.patch_out.get_mut(i) {
                    *o = Some(word);
                }
                cx.returned.push((Some((word, v.width)), Member::Reg(0, v.width)));
            }
            (IoType::Array(carried, max), ItemList::Outputs, Some(i)) => {
                geometry = true;
                let base = cx.grow(
                    u32::from(carried.width)
                        .checked_mul(max)
                        .ok_or("an emitted vertex array past the words of memory")?,
                )?;
                emitted.max = max;
                emitted.arrays.push((i, base, carried.width));
                cx.returned.push((
                    None,
                    Member::Mem {
                        word: base,
                        count: max,
                        width: carried.width,
                        keep: None,
                    },
                ));
            }
            (IoType::Value(_), ItemList::Pipeline, None) if item.name == "SV_VertexCount" => {
                emitted.count = cx.grow(1)?;
                cx.returned.push((Some((emitted.count, 1)), Member::Reg(0, 1)));
            }
            (IoType::Array(_, max), ItemList::Pipeline, None) if item.name == "SV_StripStart" => {
                emitted.strips = cx.grow(max)?;
                cx.returned.push((
                    None,
                    Member::Mem {
                        word: emitted.strips,
                        count: max,
                        width: 1,
                        keep: None,
                    },
                ));
            }
            _ => {
                return Err(format!(
                    "the result member {} is not one a stage function gives",
                    item.name
                ));
            }
        }
    }
    if geometry {
        emitted.pending = cx.grow(1)?;
        cx.emitted = Some(emitted);
    }
    let arguments = Arguments {
        args,
        arg_at,
        arg_len: at,
        array_args,
        pointers,
        builtins: vec![None; sig.arguments.len()],
    };
    finish(p, cx, outputs, arguments)
}

/// The rest of a compile once its arguments and results are placed: the buffers, the variables, the outputs' words
/// (an output no result carries gets words of its own, never read), and the body.
fn finish(
    p: &ir::Program,
    mut cx: Cx<'_>,
    mut outputs: Vec<Option<(u32, u32)>>,
    arguments: Arguments,
) -> Result<Executable, String> {
    let f = cx.f;
    let Body::Regions(body) = &f.body else {
        return Err(
            "the body is a control flow graph: structure it first (iridia_passes::structurize)".into(),
        );
    };
    let interface = &p.interface;
    for o in &mut outputs {
        if o.is_none() {
            *o = Some((cx.grow(4)?, 4));
        }
    }
    cx.outputs = outputs.into_iter().flatten().collect();
    // the buffers
    for r in &interface.resources {
        let kind = match r.kind {
            ResourceKind::ConstantBuffer { .. } if r.count == 1 => GlobalKind::Constants(r.id),
            ResourceKind::ConstantBuffer { .. } => GlobalKind::ConstantsArray(r.id, r.count),
            ResourceKind::Buffer(_)
            | ResourceKind::Storage {
                target: StorageTarget::Buffer(_),
                ..
            } if r.count == 1 => GlobalKind::Storage(r.id),
            ResourceKind::Buffer(_)
            | ResourceKind::Storage {
                target: StorageTarget::Buffer(_),
                ..
            } => GlobalKind::StorageArray(r.id, r.count),
            _ => continue,
        };
        cx.buffer_of.insert(r.id, number(cx.globals.len())?);
        cx.globals.push(GlobalDesc { kind, elem_words: 1 });
    }
    // the variables: private ones in local memory, workgroup ones in globals
    let mut prologue = Vec::new();
    for var in &f.vars {
        let stride = u32::from(var.ty.lanes);
        let len = var.len.unwrap_or(1);
        let words = stride
            .checked_mul(len)
            .ok_or("a variable past the words of memory")?;
        let loc = match var.space {
            // the memory the caller hands over is filled from the passed registers, not started here
            ir::Space::Private | ir::Space::RayPayload | ir::Space::HitAttributes => {
                Loc::Local(cx.grow(words)?)
            }
            ir::Space::Workgroup | ir::Space::Payload => {
                let g = number(cx.globals.len())?;
                cx.globals.push(GlobalDesc {
                    kind: if var.space == ir::Space::Payload {
                        GlobalKind::Payload(words)
                    } else {
                        GlobalKind::Workgroup(words)
                    },
                    elem_words: stride,
                });
                Loc::Global(g, 0)
            }
        };
        cx.vars.push(Place { loc, stride, len });
        if var.space == ir::Space::Private {
            let Loc::Local(base) = loc else { continue };
            for e in 0..len {
                let bits = var
                    .init
                    .get(e as usize)
                    .map_or([0; 4], |b| b.map(|w| (w & 0xffff_ffff) as u32));
                let r = cx.literal(var.ty, bits)?;
                prologue.push(St::Store {
                    loc: Loc::Local(base.saturating_add(e.saturating_mul(stride))),
                    src: r,
                    n: var.ty.lanes,
                });
            }
        }
    }
    // every output lane and patch constant starts as zero, a lane never written stays so; a geometry program's first
    // vertex starts a strip
    let zero = cx.literal(Type::U32X4, [0; 4])?;
    let patch_words: Vec<u32> = cx.patch_out.iter().flatten().copied().collect();
    for (word, words) in cx
        .outputs
        .iter()
        .copied()
        .chain(patch_words.into_iter().map(|w| (w, 4)))
    {
        prologue.push(St::Store {
            loc: Loc::Local(word),
            src: zero,
            n: u8::try_from(words).unwrap_or(4),
        });
    }
    if let Some(pending) = cx.emitted.as_ref().map(|e| e.pending) {
        let one = cx.literal(Type::U32, [1, 0, 0, 0])?;
        prologue.push(St::Store {
            loc: Loc::Local(pending),
            src: one,
            n: 1,
        });
    }
    // a mesh program's meshlet, an amplification program's launch: thread-group memory, read when the group has
    // finished
    let group_output = match (p.stage, &p.interface.facts) {
        (Stage::Mesh, _) => {
            let m = Meshlet::of(cx.interface, number(cx.globals.len())?)?;
            let words = m.words()?;
            cx.globals.push(GlobalDesc {
                kind: GlobalKind::Workgroup(words),
                elem_words: 1,
            });
            let global = m.global as usize;
            cx.meshlet = Some(m);
            Some((global, words))
        }
        (Stage::Amplification, StageFacts::Amplification(a)) => {
            let payload = a.payload.div_ceil(4);
            let words = payload.saturating_add(4);
            let global = number(cx.globals.len())?;
            cx.globals.push(GlobalDesc {
                kind: GlobalKind::Workgroup(words),
                elem_words: 1,
            });
            cx.launch = Some(Launch { global, payload });
            Some((global as usize, words))
        }
        (Stage::Amplification, _) => return Err("an amplification program without its facts".into()),
        _ => None,
    };
    // the pointer arguments' words, in the variables of their spaces
    let mut pointer_args = Vec::with_capacity(arguments.pointers.len());
    for &(at, space, words) in &arguments.pointers {
        let base = match f.vars.iter().zip(&cx.vars).find(|(v, _)| v.space == space) {
            Some((
                _,
                Place {
                    loc: Loc::Local(base),
                    ..
                },
            )) => *base,
            Some(_) => return Err("ray tracing memory outside local memory".into()),
            None => 0,
        };
        pointer_args.push((at, base, words));
    }
    let mut out = prologue;
    cx.region(body, Yield::Return, &mut out)?;
    cx.flush(&mut out)?;
    Ok(Executable {
        ops: cx.ops,
        body: out,
        slots: cx.types.len(),
        inits: cx.inits,
        args: arguments.args,
        arg_at: arguments.arg_at,
        arg_len: arguments.arg_len,
        pointer_args,
        array_args: arguments.array_args,
        local_words: cx.local_words as usize,
        globals: cx.globals,
        flush_denormals: p.float.f32 == ir::Denormals::Flush,
        workgroup: p.workgroup,
        group_output,
        builtins: arguments.builtins,
        ir_types: cx.types,
    })
}
