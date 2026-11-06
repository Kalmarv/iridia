//! An [`iridia_ir::Program`] compiled into the flat form: each value of the entry function is the register its id
//! numbers, its operation run by `iridia_exec_ir::eval` (the IR's one definition of what it computes) where it is an
//! operation on values, and by the flat form's own operations where it reads inputs, variables, buffers or textures.
//! Registers past the values' own hold what the compile adds: the entry arguments, copies into a construct's results,
//! and the integer layer of an array sample.
//!
//! The body must be structured ([`iridia_ir::Body::Regions`]) with its exits as `iridia_passes::break_flags` leaves
//! them: a `Br` leaves the innermost loop or switch, a `Continue` restarts the innermost loop.

use std::collections::BTreeMap;

use iridia_ir::{
    self as ir, Body, End, FloatToInt, Function, Node, Region, Scalar, Type, ValueId, VarId,
    interface::{BufferKind, Interface, ResourceKind, SampleType, Stage, StorageTarget, TextureDim},
};

use crate::flat::{
    Executable, GlobalDesc, GlobalKind, LevelOp, Loc, Member, Op, Pick, Reg, Res, St,
};
use crate::request::{Dim, QueryKind};

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
    /// A loop's parameters; `None` for a switch.
    params: Option<Vec<ValueId>>,
    results: Vec<ValueId>,
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

    /// The integer layer of an array sample: the coordinate's lane `at` rounded to the nearest even and clamped to 0 or
    /// more, as `u32`.
    fn layer_of(&mut self, coord: ValueId, at: u8) -> Result<u32, String> {
        let lane = self.compute(Type::F32, ir::Op::Swizzle(coord, [at, 0, 0, 0]))?;
        let rounded = self.compute(
            Type::F32,
            ir::Op::Unary(
                ir::Unary::FRound(ir::Rounding::NearestEven),
                ValueId(lane),
            ),
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
        self.compute(
            Type::U32,
            ir::Op::Convert(ValueId(clamped), FloatToInt::Saturate),
        )
    }

    /// The texture a sample, gather, load or query names: its shape, whether it is arrayed, and whether it is a depth
    /// texture or a storage image.
    fn texture(&self, r: &ir::Res) -> Result<(TextureDim, bool, bool, bool), String> {
        match &self.resource(r.id)?.kind {
            ResourceKind::Texture {
                dim,
                arrayed,
                sample,
                ..
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
                if let ResourceKind::Buffer(BufferKind::Typed(_)) = &self.resource(texture.id)?.kind {
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
                        BufferKind::Typed(_) => {
                            return Err("the elements of a typed buffer are not compiled yet".into());
                        }
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
        let lanes = lanes_of(mask);
        let first = lanes.first().copied().unwrap_or(0);
        let contiguous = lanes.iter().zip(first..).all(|(&l, k)| l == k);
        if contiguous {
            return self.statement(
                out,
                St::Store {
                    loc: Loc::Local(base.saturating_add(u32::from(first))),
                    src,
                    n: u8::try_from(lanes.len()).unwrap_or(4),
                },
            );
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
            self.statement(
                out,
                St::Store {
                    loc: Loc::Local(base.saturating_add(u32::from(lane))),
                    src: one,
                    n: 1,
                },
            )?;
        }
        Ok(())
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

    fn node(&mut self, n: &Node, out: &mut Vec<St>) -> Result<(), String> {
        match n {
            Node::Value(v) => self.place(*v),
            Node::StoreVar(var, v) => {
                let p = self.var(*var)?;
                let n = self.value(*v)?.ty.lanes;
                self.statement(out, St::Store { loc: p.loc, src: v.0, n })
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
            Node::StorePatch(..) | Node::Emit(_) | Node::Cut(_) => Err(format!(
                "{} is not compiled yet",
                match n {
                    Node::StorePatch(..) => "a patch constant store",
                    Node::Emit(_) => "an emit",
                    _ => "a cut",
                }
            )),
        }
    }

    /// The frame `label` names, which must be the innermost loop (`params`) or loop or switch.
    fn frame(&self, label: ir::LabelId, loop_only: bool) -> Result<&Frame, String> {
        let f = self
            .frames
            .iter()
            .rev()
            .find(|f| !loop_only || f.params.is_some())
            .ok_or("an exit outside a loop or switch")?;
        if f.label != label {
            return Err("an exit past the innermost loop or switch: run iridia_passes::break_flags first".into());
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
            End::Unreachable => self.flush(out),
        }
    }
}

/// The name of an operation the compile does not take, for its refusal.
fn op_name(op: &ir::Op) -> &'static str {
    match op {
        ir::Op::LoadInputAt(..) => "an input of one control point or vertex",
        ir::Op::LoadOutputAt(..) => "an output control point read",
        ir::Op::LoadPatch(..) => "a patch constant read",
        ir::Op::Builtin(_) => "a pipeline value of a stage made of functions",
        ir::Op::LoadState(_) => "a pipeline state read (fold it with iridia_passes::specialise)",
        ir::Op::Atomic { .. } => "an atomic operation",
        ir::Op::Counter(..) => "a buffer counter",
        _ => "this operation",
    }
}

/// Compile the entry function of `p`.
pub(crate) fn compile(p: &ir::Program) -> Result<Executable, String> {
    if !matches!(p.stage, Stage::Vertex | Stage::Pixel | Stage::Compute) {
        return Err(format!("a {:?} program is not compiled from the IR yet", p.stage));
    }
    let f = &p.entry;
    let Body::Regions(body) = &f.body else {
        return Err("the body is a control flow graph: structure it first (iridia_passes::structurize)".into());
    };
    let interface = &p.interface;
    let io = iridia::entry_io(interface)?;
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
        outputs: Vec::new(),
        returned: Vec::new(),
        frames: Vec::new(),
    };
    for d in &f.values {
        if d.ty.scalar.bits() > 32 || (d.ty.scalar.bits() < 32 && d.ty.scalar != Scalar::Bool) {
            return Err(format!("{} values are not compiled yet", d.ty));
        }
    }
    // the literals, and an undefined value as zero
    for (i, d) in f.values.iter().enumerate() {
        match d.op {
            ir::Op::Literal(bits) => {
                // a lane is at most 32 bits here
                let words = bits.map(|b| (b & 0xffff_ffff) as u32);
                cx.inits.push((number(i)?, words));
            }
            ir::Op::Undef => cx.inits.push((number(i)?, [0; 4])),
            _ => {}
        }
    }
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
    let mut outputs = vec![None; interface.outputs.len()];
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
                    let width = interface
                        .outputs
                        .get(i)
                        .map_or(1, |v| u32::from(v.value.width));
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
    // an output the entry point does not carry is stored to words of its own, never read
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
        cx.globals.push(GlobalDesc {
            kind,
            elem_words: 1,
        });
    }
    // the variables: private ones in local memory, workgroup ones in globals
    let mut start = Vec::new();
    for var in &f.vars {
        let stride = u32::from(var.ty.lanes);
        let len = var.len.unwrap_or(1);
        let words = stride.checked_mul(len).ok_or("a variable past the words of memory")?;
        let loc = match var.space {
            ir::Space::Private => Loc::Local(cx.grow(words)?),
            ir::Space::Workgroup => {
                let g = number(cx.globals.len())?;
                cx.globals.push(GlobalDesc {
                    kind: GlobalKind::Workgroup(words),
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
                start.push(St::Store {
                    loc: Loc::Local(base.saturating_add(e.saturating_mul(stride))),
                    src: r,
                    n: var.ty.lanes,
                });
            }
        }
    }
    // every output lane starts as zero; a lane never written stays so
    let zero = cx.literal(Type::U32X4, [0; 4])?;
    for &(word, words) in &cx.outputs {
        start.push(St::Store {
            loc: Loc::Local(word),
            src: zero,
            n: u8::try_from(words).unwrap_or(4),
        });
    }
    let mut out = start;
    cx.region(body, Yield::Return, &mut out)?;
    cx.flush(&mut out)?;
    let n_args = args.len();
    Ok(Executable {
        ops: cx.ops,
        body: out,
        slots: cx.types.len(),
        inits: cx.inits,
        args,
        arg_at: (0..n_args).collect(),
        arg_len: n_args,
        pointer_args: Vec::new(),
        array_args: Vec::new(),
        local_words: cx.local_words as usize,
        globals: cx.globals,
        flush_denormals: p.float.f32 == ir::Denormals::Flush,
        workgroup: p.workgroup,
        mesh_output: None,
        builtins,
        ir_types: cx.types,
    })
}
