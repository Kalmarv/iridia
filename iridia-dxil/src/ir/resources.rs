//! Resource handles and the operations on them: constant buffer rows, samples and gathers, texel loads and stores,
//! buffer words, sizes, atomic operations and hidden counters. Each takes Direct3D's result as the naga-path lift
//! ([`crate::lift`]) does, and makes the same changes to the interface: a texture compared against a reference is a
//! depth texture, a typed buffer is carried as four 32-bit lanes an element, and an unordered-access view's access is
//! what the program does with it.

use std::collections::{BTreeMap, BTreeSet};

use iridia_ir::{
    Address, AtomicOp, Binary, CounterOp, GatherOffset, Level, Memory, Node, Op, OutOfBounds, Query, Res,
    Scalar, Type, Unary, ValueId as IrValue,
    interface::{
        Access, BufferKind, ResourceId, ResourceKind, SampleType, StorageTarget, TextureDim, Value as Carried,
    },
};

use super::{Lift, Lv};
use crate::Result;
use crate::lift::setup::{binding_of, const_int, dx_opcode};
use dxil_reader::module::{Constant, Instruction, Op as LlvmOp, Type as LlvmType, TypeId, Value, ValueId};

/// DXIL's `AtomicBinOp` operation codes, in order: add, and, or, xor, signed min and max, unsigned min and max,
/// exchange.
const ATOMIC_OPS: [AtomicOp; 9] = [
    AtomicOp::Add,
    AtomicOp::And,
    AtomicOp::Or,
    AtomicOp::Xor,
    AtomicOp::SMin,
    AtomicOp::SMax,
    AtomicOp::UMin,
    AtomicOp::UMax,
    AtomicOp::Exchange,
];

/// The shape of a texture or storage image.
#[derive(Clone, Copy, Debug)]
struct Shape {
    dim: TextureDim,
    arrayed: bool,
    multisampled: bool,
    /// The scalar its texels' lanes have (`f32` for a depth texture).
    scalar: Scalar,
    depth: bool,
    /// A storage image (an unordered-access view) rather than a texture.
    storage: bool,
}

/// The coordinate lanes of a dimension, without a layer: 1D 1, 2D 2, 3D and cube 3.
fn coordinate_lanes(dim: TextureDim) -> u8 {
    match dim {
        TextureDim::D1 => 1,
        TextureDim::D2 => 2,
        TextureDim::D3 | TextureDim::Cube => 3,
    }
}

fn scalar_of_sample(s: SampleType) -> Scalar {
    match s {
        SampleType::Sint => Scalar::I32,
        SampleType::Uint => Scalar::U32,
        _ => Scalar::F32,
    }
}

impl Lift<'_> {
    /// The interface resource a handle value names, found without lifting it: a `CreateHandle` call of constant
    /// arguments, one from a constant binding or from the heap, or an annotation of one of those.
    fn handle_of(&self, mut v: ValueId) -> Option<usize> {
        for _ in 0..=self.f.instructions.len() {
            let inst = self
                .f
                .instructions
                .iter()
                .find(|i| i.result.is_some_and(|(r, _)| r == v))?;
            let LlvmOp::Call { function, args, .. } = &inst.op else {
                return None;
            };
            return match dx_opcode(self.m, self.f, *function, args)? {
                57 => {
                    let class = const_int(self.m, self.f, *args.get(1)?)?;
                    let id = const_int(self.m, self.f, *args.get(2)?)?;
                    self.setup
                        .resource(u8::try_from(class).ok()?, u32::try_from(id).ok()?)
                }
                217 => {
                    let (class, lower, space) = binding_of(self.m, self.f, *args.get(1)?)?;
                    self.setup.resource_by_binding(class, lower, space)
                }
                216 => {
                    v = *args.get(1)?;
                    continue;
                }
                // a library's handle is made from the value of the resource's global
                160 => self
                    .setup
                    .resource_of_variable(crate::lift::setup::variable_loaded(
                        self.m,
                        self.f,
                        *args.get(1)?,
                    )?),
                218 => self.setup.heap.get(&v).copied(),
                _ => None,
            };
        }
        None
    }

    /// The interface's resources as the naga-path lift makes them, from how the program's functions use each: a texture
    /// compared against a reference is a depth texture, a typed buffer has four lanes an element, and a view's access
    /// is the loads and stores the program makes of it (an atomic operation does both).
    pub(crate) fn declare_resources(&mut self) -> Result<()> {
        let mut depth = BTreeSet::new();
        let mut views: BTreeMap<usize, (bool, bool)> = BTreeMap::new();
        let lifting = self.f;
        for f in self.functions.clone() {
            // a handle is found among the instructions of the function that uses it
            self.f = f;
            for inst in &f.instructions {
                let LlvmOp::Call { function, args, .. } = &inst.op else {
                    continue;
                };
                let Some(code) = dx_opcode(self.m, f, *function, args) else {
                    continue;
                };
                let Some(r) = args.get(1).and_then(|&a| self.handle_of(a)) else {
                    continue;
                };
                match code {
                    64 | 65 | 74 => {
                        depth.insert(r);
                    }
                    66 | 68 | 139 => {
                        if matches!(self.setup.kind(r), ResourceKind::Storage { .. }) {
                            views.entry(r).or_default().0 = true;
                        }
                    }
                    67 | 69 | 140 => views.entry(r).or_default().1 = true,
                    78 | 79 => {
                        views.insert(r, (true, true));
                    }
                    _ => {}
                }
            }
        }
        self.f = lifting;
        for i in 0..self.setup.interface.resources.len() {
            let kind = self.setup.kind(i).clone();
            let changed = match kind {
                ResourceKind::Texture {
                    dim,
                    arrayed,
                    multisampled,
                    ..
                } if depth.contains(&i) => Some(ResourceKind::Texture {
                    dim,
                    arrayed,
                    multisampled,
                    sample: SampleType::Depth,
                }),
                ResourceKind::Buffer(BufferKind::Typed(v)) => {
                    Some(ResourceKind::Buffer(BufferKind::Typed(Carried { width: 4, ..v })))
                }
                ResourceKind::Storage { target, .. } => {
                    let (load, store) = views.get(&i).copied().unwrap_or((true, false));
                    let target = match target {
                        StorageTarget::Buffer(BufferKind::Typed(v)) => {
                            StorageTarget::Buffer(BufferKind::Typed(Carried { width: 4, ..v }))
                        }
                        other => other,
                    };
                    Some(ResourceKind::Storage {
                        target,
                        access: match (load, store) {
                            (true, true) => Access::ReadWrite,
                            (false, true) => Access::Write,
                            _ => Access::Read,
                        },
                    })
                }
                ResourceKind::PipelineState(_) => {
                    return self.refuse("a DXIL resource is pipeline state");
                }
                _ => None,
            };
            if let (Some(kind), Some(r)) = (changed, self.setup.interface.resources.get_mut(i)) {
                r.kind = kind;
            }
        }
        Ok(())
    }

    /// The id of interface resource `i`.
    pub(crate) fn resource_id(&self, i: usize) -> Result<ResourceId> {
        match self.setup.interface.resources.get(i) {
            Some(r) => Ok(r.id),
            None => self.refuse(format!(
                "resource {i}, where the program has {}",
                self.setup.interface.resources.len()
            )),
        }
    }

    /// The interface position of the resource a handle names.
    fn position(&self, r: Res) -> Result<usize> {
        self.setup
            .interface
            .resources
            .iter()
            .position(|x| x.id == r.id)
            .map_or_else(
                || self.refuse(format!("resource {:?}, which the interface does not have", r.id)),
                Ok,
            )
    }

    /// The handle value `v` is.
    pub(crate) fn handle(&mut self, v: ValueId) -> Result<(Res, usize)> {
        match self.get(v)? {
            Lv::Handle(r) => {
                let i = self.position(r)?;
                Ok((r, i))
            }
            other => self.refuse(format!("a handle was expected, found {other:?}")),
        }
    }

    /// Whether a call argument is the constant `true`.
    fn const_true(&self, v: ValueId) -> bool {
        const_int(self.m, self.f, v).is_some_and(|k| k & 1 != 0)
    }

    /// The handle of element `index` (a register of the range) of resource `i`: the element counted from the range's
    /// first register, for an array; a resource that is not an array must be named at its register.
    fn element_handle(&mut self, i: usize, index: ValueId, nonuniform: bool) -> Result<Lv> {
        let id = self.resource_id(i)?;
        let (Some(range), Some(count)) = (
            self.setup.ranges.get(i).copied(),
            self.setup.interface.resources.get(i).map(|r| r.count),
        ) else {
            return self.refuse(format!("resource {i} has no range"));
        };
        let constant = const_int(self.m, self.f, index);
        if count == 1 {
            return match constant {
                Some(k) if k == i64::from(range.lower) => Ok(Lv::Handle(Res::of(id))),
                Some(_) => self.refuse("a handle at an index other than its range's register"),
                None => self.refuse("a handle of a computed index into a range of one"),
            };
        }
        let element = match constant {
            Some(k) => match u32::try_from(k).ok().and_then(|k| k.checked_sub(range.lower)) {
                Some(e) => self.u32_lit(e),
                None => return self.refuse("a handle at an index below its range"),
            },
            None => {
                let at = self.uint(index)?;
                let lower = self.u32_lit(range.lower);
                self.binary(Type::U32, Binary::ISub, at, lower)
            }
        };
        Ok(Lv::Handle(Res {
            id,
            index: Some(element),
            nonuniform,
        }))
    }

    /// `CreateHandle` (57), `AnnotateHandle` (216), `CreateHandleFromBinding` (217) and `CreateHandleFromHeap` (218).
    pub(crate) fn create_handle(&mut self, inst: &Instruction, code: u64, args: &[ValueId]) -> Result<()> {
        let lv = match code {
            57 => {
                let (Some(class), Some(id)) = (self.const_arg(args, 1), self.const_arg(args, 2)) else {
                    return self.refuse("a handle of a computed range");
                };
                let (Ok(class), Ok(id)) = (u8::try_from(class), u32::try_from(id)) else {
                    return self.refuse(format!("a handle of range {id} of class {class}"));
                };
                let Some(i) = self.setup.resource(class, id) else {
                    return self.refuse("a handle of an undeclared range");
                };
                let nonuniform = self.const_true(self.arg(args, 4)?);
                self.element_handle(i, self.arg(args, 3)?, nonuniform)?
            }
            216 => {
                let handle = self.get(self.arg(args, 1)?)?;
                if !matches!(handle, Lv::Handle(_)) {
                    return self.refuse("an annotation of something that is not a handle");
                }
                handle
            }
            // CreateHandleForLib(the value of a library's resource global): the resource's handle
            160 => {
                let handle = self.get(self.arg(args, 1)?)?;
                if !matches!(handle, Lv::Handle(_)) {
                    return self.refuse("a handle made from something that is not a resource of the library");
                }
                handle
            }
            217 => {
                let Some((class, lower, space)) = binding_of(self.m, self.f, self.arg(args, 1)?) else {
                    return self.refuse("a handle of a binding that is not constant");
                };
                let Some(i) = self.setup.resource_by_binding(class, lower, space) else {
                    return self.refuse("a handle of an undeclared binding");
                };
                let nonuniform = self.const_true(self.arg(args, 3)?);
                self.element_handle(i, self.arg(args, 2)?, nonuniform)?
            }
            _ => {
                let Some((handle, _)) = inst.result else {
                    return self.refuse("a heap handle without a result");
                };
                let Some(&i) = self.setup.heap.get(&handle) else {
                    return self.refuse("a heap handle the setup did not find");
                };
                self.notes.insert(
                    "a descriptor heap's resources are carried as one unbounded binding array for each kind of view \
                     the program annotates them as, in a register space of its own: the renderer binds a heap's \
                     descriptors of that kind"
                        .into(),
                );
                let id = self.resource_id(i)?;
                let index = self.uint(self.arg(args, 1)?)?;
                let nonuniform = self.const_true(self.arg(args, 3)?);
                Lv::Handle(Res {
                    id,
                    index: Some(index),
                    nonuniform,
                })
            }
        };
        self.define(
            inst.result
                .map(|r| r.0)
                .ok_or_else(|| crate::error(self.at, "a handle without a result"))?,
            lv,
        )
    }

    /// The scalars of a call's result structure's first `n` fields, as carried.
    fn result_fields(&self, inst: &Instruction) -> Result<Vec<Type>> {
        let Some((_, ty)) = inst.result else {
            return self.refuse("an operation that defines no result");
        };
        let LlvmType::Struct { elements, .. } = self.type_at(ty)?.clone() else {
            return self.refuse("an operation whose result is not a structure");
        };
        let mut out = Vec::new();
        for e in elements {
            out.push(
                self.carried_type(e)?
                    .ok_or_else(|| crate::error(self.at, "a result field of no value type"))?,
            );
        }
        Ok(out)
    }

    /// The lane `k` of a value as the carried type `want`: the same bits, and a bool where the word is not zero.
    fn lane_as(&mut self, x: IrValue, k: u8, want: Type) -> Result<IrValue> {
        let t = self
            .g
            .ty(x)
            .ok_or_else(|| crate::error(self.at, "a value of no type"))?;
        let lane = if t.lanes == 1 {
            x
        } else {
            self.g.value(Type::scalar(t.scalar), Op::Swizzle(x, [k, 0, 0, 0]))
        };
        if want.scalar == Scalar::Bool {
            let bits = self.retype(lane, Type::scalar(t.scalar).with(Scalar::U32))?;
            let zero = self.u32_lit(0);
            return Ok(self.binary(Type::BOOL, Binary::UCmp(iridia_ir::Cmp::Ne), bits, zero));
        }
        self.retype(lane, want)
    }

    /// A four-lane result as a `ResRet` structure of the call's field types: the four lanes and the status (0, every
    /// access mapped).
    fn res_ret(&mut self, inst: &Instruction, x: IrValue) -> Result<()> {
        let fields = self.result_fields(inst)?;
        let mut items = Vec::new();
        for (k, want) in (0u8..4).zip(&fields) {
            items.push(Lv::Value(self.lane_as(x, k, *want)?));
        }
        if fields.len() > 4 {
            items.push(Lv::Value(self.u32_lit(0)));
        }
        self.define_result(inst, Lv::Agg(items))
    }

    /// `CBufferLoadLegacy(handle, row)`: a row of four 32-bit lanes; a row outside the buffer is zero.
    pub(crate) fn cbuffer_load(&mut self, inst: &Instruction, args: &[ValueId]) -> Result<()> {
        let (resource, i) = self.handle(self.arg(args, 1)?)?;
        if !matches!(self.setup.kind(i), ResourceKind::ConstantBuffer { .. }) {
            return self.refuse("a constant load from a non-buffer");
        }
        let fields = self.result_fields(inst)?;
        if fields.len() != 4 {
            return self.refuse(
                "a constant load of other than four 32-bit lanes (16-bit and 64-bit rows are not lifted)",
            );
        }
        let row = self.arg(args, 2)?;
        let address = match self.const_arg(args, 2).and_then(|r| u32::try_from(r).ok()) {
            Some(r) => Address::fixed(r.saturating_mul(16)),
            None => Address {
                index: Some(self.uint(row)?),
                stride: 16,
                offset: 0,
            },
        };
        let lanes = self.g.value(
            Type::U32X4,
            Op::LoadBuffer {
                resource,
                address,
                oob: OutOfBounds::Zero,
            },
        );
        let Some((_, ty)) = inst.result else {
            return self.refuse("an operation that defines no result");
        };
        let LlvmType::Struct { elements, .. } = self.type_at(ty)?.clone() else {
            return self.refuse("an operation whose result is not a structure");
        };
        let mut items = Vec::new();
        for ((k, want), e) in (0u8..4).zip(&fields).zip(elements) {
            // a row of four 16-bit lanes (minimum precision: `min16float`, `min16int`) keeps each lane in a word of its
            // own, its value the word's low 16 bits
            let lane = match self.type_at(e)? {
                LlvmType::Half => {
                    let word = self.g.value(Type::U32, Op::Swizzle(lanes, [k, 0, 0, 0]));
                    self.unary(Type::F32, Unary::F16ToF32, word)
                }
                LlvmType::Integer(16) => {
                    let word = self.g.value(Type::U32, Op::Swizzle(lanes, [k, 0, 0, 0]));
                    self.mask_narrow(word, 16)
                }
                _ => self.lane_as(lanes, k, *want)?,
            };
            items.push(Lv::Value(lane));
        }
        self.define_result(inst, Lv::Agg(items))
    }

    /// The shape of the texture or storage image resource `i` is.
    fn shape(&self, i: usize) -> Result<Shape> {
        match self.setup.kind(i) {
            ResourceKind::Texture {
                dim,
                arrayed,
                multisampled,
                sample,
            } => Ok(Shape {
                dim: *dim,
                arrayed: *arrayed,
                multisampled: *multisampled,
                scalar: scalar_of_sample(*sample),
                depth: *sample == SampleType::Depth,
                storage: false,
            }),
            ResourceKind::Storage {
                target: StorageTarget::Image { dim, arrayed, sample },
                ..
            } => Ok(Shape {
                dim: *dim,
                arrayed: *arrayed,
                multisampled: false,
                scalar: scalar_of_sample(*sample),
                depth: false,
                storage: true,
            }),
            _ => self.refuse("a handle that is not a texture"),
        }
    }

    /// The coordinate of a texture operation from argument `first`: the dimension's lanes, then the layer of an
    /// arrayed texture, each as `scalar`.
    fn coordinate(
        &mut self,
        args: &[ValueId],
        first: usize,
        shape: Shape,
        scalar: Scalar,
    ) -> Result<IrValue> {
        let n = usize::from(coordinate_lanes(shape.dim)).saturating_add(usize::from(shape.arrayed));
        let mut parts = Vec::new();
        for k in 0..n {
            parts.push(self.value_as(self.arg(args, first.saturating_add(k))?, Type::scalar(scalar))?);
        }
        Ok(match parts.as_slice() {
            [one] => *one,
            _ => {
                let lanes = u8::try_from(parts.len()).unwrap_or(4);
                self.g.value(Type::vector(scalar, lanes), Op::Compose(parts))
            }
        })
    }

    fn is_undef(&self, v: ValueId) -> bool {
        matches!(self.value_ref(v), Some(Value::Constant(Constant::Undef)))
    }

    /// A texel offset of constant arguments from `first` (an undefined one is 0), each -8 to 7; none for a cube.
    fn texel_offset(&self, args: &[ValueId], first: usize, dim: TextureDim) -> Result<[i8; 3]> {
        let mut offset = [0i8; 3];
        if dim == TextureDim::Cube {
            return Ok(offset);
        }
        for (k, o) in offset
            .iter_mut()
            .enumerate()
            .take(usize::from(coordinate_lanes(dim)))
        {
            let a = self.arg(args, first.saturating_add(k))?;
            *o = match const_int(self.m, self.f, a) {
                Some(v) => match i8::try_from(v) {
                    Ok(v) if (-8..=7).contains(&v) => v,
                    _ => return self.refuse(format!("a texel offset of {v}, outside -8 to 7")),
                },
                None if self.is_undef(a) => 0,
                None => return self.refuse("a sample offset that is not constant"),
            };
        }
        Ok(offset)
    }

    /// Whether a float argument is the constant 0.0.
    fn is_zero_float(&self, v: ValueId) -> bool {
        matches!(
            self.value_ref(v),
            Some(Value::Constant(Constant::Null | Constant::Float(0)))
        )
    }

    /// A depth texture's one lane as D3D's float view returns it: the depth, then 0, 0, 1.
    fn depth_view(&mut self, d: IrValue) -> IrValue {
        let zero = self.f32_lit(0.0);
        let one = self.f32_lit(1.0);
        self.g.value(Type::F32X4, Op::Compose(vec![d, zero, zero, one]))
    }

    /// `Sample`, `SampleBias`, `SampleLevel`, `SampleGrad`, `SampleCmp`, `SampleCmpLevelZero`, `TextureGather` and
    /// `TextureGatherCmp`.
    pub(crate) fn sample(&mut self, inst: &Instruction, code: u64, args: &[ValueId]) -> Result<()> {
        let (texture, i) = self.handle(self.arg(args, 1)?)?;
        let (sampler, _) = self.handle(self.arg(args, 2)?)?;
        let shape = self.shape(i)?;
        if shape.multisampled {
            return self.refuse("a sample of a multisampled texture");
        }
        let coord = self.coordinate(args, 3, shape, Scalar::F32)?;
        let gather = matches!(code, 73 | 74);
        if gather && shape.dim == TextureDim::D3 {
            return self.refuse("a gather of a 3D texture");
        }
        let offset = self.texel_offset(args, 7, shape.dim)?;
        let no_clamp = |l: &Self, v: ValueId| {
            if l.is_undef(v) || const_int(l.m, l.f, v) == Some(0) {
                Ok(())
            } else {
                l.refuse("a LOD clamp")
            }
        };
        let op = if gather {
            let Some(component) = self
                .const_arg(args, 9)
                .and_then(|c| u8::try_from(c).ok())
                .filter(|c| *c < 4)
            else {
                return self.refuse("a gather channel that is not a constant lane");
            };
            let compare = if code == 74 {
                Some(self.float(self.arg(args, 10)?)?)
            } else {
                None
            };
            Op::Gather {
                texture,
                sampler,
                coord,
                component,
                compare,
                offset: GatherOffset::Fixed(offset),
            }
        } else {
            let (level, compare) = match code {
                60 => {
                    no_clamp(self, self.arg(args, 10)?)?;
                    (Level::Auto, None)
                }
                61 if shape.depth => {
                    return self
                        .refuse("a depth texture sampled with a bias (WGSL's depth sampling has none)");
                }
                63 if shape.depth => {
                    return self
                        .refuse("a depth texture sampled with gradients (WGSL's depth sampling has none)");
                }
                61 => {
                    no_clamp(self, self.arg(args, 11)?)?;
                    (Level::Bias(self.float(self.arg(args, 10)?)?), None)
                }
                62 => {
                    let lod = self.arg(args, 10)?;
                    if self.is_zero_float(lod) {
                        (Level::Zero, None)
                    } else if shape.depth {
                        return self.refuse(
                            "a depth texture sampled at a level the program computes (WGSL takes an integer level \
                             there)",
                        );
                    } else {
                        (Level::Exact(self.float(lod)?), None)
                    }
                }
                63 => {
                    no_clamp(self, self.arg(args, 16)?)?;
                    let n = coordinate_lanes(shape.dim);
                    let mut x = Vec::new();
                    let mut y = Vec::new();
                    for k in 0..usize::from(n) {
                        x.push(self.float(self.arg(args, 10usize.saturating_add(k))?)?);
                        y.push(self.float(self.arg(args, 13usize.saturating_add(k))?)?);
                    }
                    let t = Type::vector(Scalar::F32, n);
                    let (x, y) = if n == 1 {
                        (x.first().copied(), y.first().copied())
                    } else {
                        (
                            Some(self.g.value(t, Op::Compose(x))),
                            Some(self.g.value(t, Op::Compose(y))),
                        )
                    };
                    let (Some(x), Some(y)) = (x, y) else {
                        return self.refuse("a gradient of no lanes");
                    };
                    (Level::Grad(x, y), None)
                }
                64 => {
                    no_clamp(self, self.arg(args, 11)?)?;
                    (Level::Auto, Some(self.float(self.arg(args, 10)?)?))
                }
                65 => (Level::Zero, Some(self.float(self.arg(args, 10)?)?)),
                other => return self.refuse(format!("sample operation {other}")),
            };
            Op::Sample {
                texture,
                sampler,
                coord,
                level,
                compare,
                offset,
                clamp: None,
            }
        };
        let compare = matches!(code, 64 | 65 | 74);
        let one = shape.depth && !gather;
        // a sample or gather gives the texture's sample type; a depth texture's, and a comparison's, floats
        let scalar = if shape.depth || compare {
            Scalar::F32
        } else {
            shape.scalar
        };
        let mut result = self
            .g
            .value(if one { Type::F32 } else { Type::vector(scalar, 4) }, op);
        if one {
            // D3D's float view of a depth texture: the depth, then 0, 0, 1; a comparison's result in every lane
            result = if compare {
                self.g.value(Type::F32X4, Op::Splat(result))
            } else {
                self.depth_view(result)
            };
        }
        self.res_ret(inst, result)
    }

    /// `TextureLoad(handle, level or sample, x, y, z, offsets)`: a texel of a texture or a storage image, zero outside
    /// it.
    pub(crate) fn texture_load(&mut self, inst: &Instruction, args: &[ValueId]) -> Result<()> {
        let (texture, i) = self.handle(self.arg(args, 1)?)?;
        let shape = self.shape(i)?;
        let coord = self.coordinate(
            args,
            3,
            shape,
            if shape.storage { Scalar::U32 } else { Scalar::I32 },
        )?;
        let offset = self.texel_offset(args, 6, shape.dim)?;
        let (level, sample) = if shape.storage {
            (None, None)
        } else if shape.multisampled {
            (None, Some(self.int(self.arg(args, 2)?)?))
        } else {
            (Some(self.int(self.arg(args, 2)?)?), None)
        };
        let op = Op::LoadTexel {
            texture,
            coord,
            level,
            sample,
            offset,
        };
        let result = if shape.depth {
            let d = self.g.value(Type::F32, op);
            self.depth_view(d)
        } else {
            self.g.value(Type::vector(shape.scalar, 4), op)
        };
        self.res_ret(inst, result)
    }

    /// `TextureStore(view, x, y, z, v0..v3, mask)`: a texel of a storage image; nothing outside it.
    pub(crate) fn texture_store(&mut self, args: &[ValueId]) -> Result<()> {
        let (image, i) = self.handle(self.arg(args, 1)?)?;
        let shape = self.shape(i)?;
        if self.const_arg(args, 9) != Some(15) {
            return self.refuse("a store through a view with a partial write mask");
        }
        let coord = self.coordinate(args, 2, shape, Scalar::U32)?;
        let mut lanes = Vec::new();
        for k in 0..4usize {
            lanes.push(self.value_as(
                self.arg(args, 5usize.saturating_add(k))?,
                Type::scalar(shape.scalar),
            )?);
        }
        let value = self.g.value(Type::vector(shape.scalar, 4), Op::Compose(lanes));
        self.g.node(Node::StoreTexel { image, coord, value });
        Ok(())
    }

    /// The kind of buffer resource `i` is: a shader resource's or an unordered-access view's.
    fn buffer_kind(&self, i: usize) -> Result<BufferKind> {
        match self.setup.kind(i) {
            ResourceKind::Buffer(k)
            | ResourceKind::Storage {
                target: StorageTarget::Buffer(k),
                ..
            } => Ok(*k),
            _ => self.refuse("a buffer operation on something that is not a buffer"),
        }
    }

    /// The scalar of a typed buffer's lanes.
    fn typed_scalar(v: Carried) -> Scalar {
        match v.scalar {
            iridia_ir::interface::ScalarType::Sint => Scalar::I32,
            iridia_ir::interface::ScalarType::Uint => Scalar::U32,
            _ => Scalar::F32,
        }
    }

    /// The byte address of a raw buffer's byte index, or a structured buffer's element and byte offset.
    fn byte_address(&mut self, kind: BufferKind, index: ValueId, offset: ValueId) -> Result<Address> {
        let i = self.uint(index)?;
        Ok(match kind {
            BufferKind::Structured { stride } => match const_int(self.m, self.f, offset)
                .and_then(|o| u32::try_from(o).ok())
                .or_else(|| self.is_undef(offset).then_some(0))
            {
                Some(o) => Address {
                    index: Some(i),
                    stride,
                    offset: o,
                },
                None => {
                    let o = self.uint(offset)?;
                    let s = self.u32_lit(stride);
                    let base = self.binary(Type::U32, Binary::IMul, i, s);
                    Address::bytes(self.binary(Type::U32, Binary::IAdd, base, o))
                }
            },
            _ => Address::bytes(i),
        })
    }

    /// `BufferLoad` and `RawBufferLoad(handle, index, offset[, mask, alignment])`: four lanes of a typed, raw or
    /// structured buffer, each zero outside it.
    pub(crate) fn buffer_load(&mut self, inst: &Instruction, args: &[ValueId]) -> Result<()> {
        let (resource, i) = self.handle(self.arg(args, 1)?)?;
        let kind = self.buffer_kind(i)?;
        let row = match kind {
            BufferKind::Typed(v) => {
                let scalar = Self::typed_scalar(v);
                let coord = self.uint(self.arg(args, 2)?)?;
                self.g.value(
                    Type::vector(scalar, 4),
                    Op::LoadTexel {
                        texture: resource,
                        coord,
                        level: None,
                        sample: None,
                        offset: [0; 3],
                    },
                )
            }
            _ => {
                let address = self.byte_address(kind, self.arg(args, 2)?, self.arg(args, 3)?)?;
                let bytes = self.result_lane_bytes(inst)?;
                if bytes != 4 {
                    return self.buffer_load_lanes(inst, resource, address, bytes);
                }
                self.g.value(
                    Type::U32X4,
                    Op::LoadBuffer {
                        resource,
                        address,
                        oob: OutOfBounds::Zero,
                    },
                )
            }
        };
        self.res_ret(inst, row)
    }

    /// The bytes a lane of LLVM type `ty` takes in a raw or structured buffer: 2 for a half or an `i16`, 8 for an
    /// `i64` or a double, 4 for the rest.
    fn lane_bytes(&self, ty: TypeId) -> Result<u32> {
        Ok(match self.type_at(ty)? {
            LlvmType::Half | LlvmType::Integer(16) => 2,
            LlvmType::Integer(64) | LlvmType::Double => 8,
            _ => 4,
        })
    }

    /// The bytes a lane of a buffer read's `ResRet` result takes (its first field's type).
    fn result_lane_bytes(&self, inst: &Instruction) -> Result<u32> {
        let Some((_, ty)) = inst.result else {
            return self.refuse("an operation that defines no result");
        };
        match self.type_at(ty)?.clone() {
            LlvmType::Struct { elements, .. } => match elements.first() {
                Some(&e) => self.lane_bytes(e),
                None => self.refuse("a buffer read of an empty structure"),
            },
            _ => self.refuse("a buffer read whose result is not a structure"),
        }
    }

    /// The byte an address names plus `extra`, as a `u32`.
    fn address_byte(&mut self, address: Address, extra: u32) -> IrValue {
        let offset = address.offset.wrapping_add(extra);
        let Some(index) = address.index else {
            return self.u32_lit(offset);
        };
        let scaled = if address.stride == 1 {
            index
        } else {
            let stride = self.u32_lit(address.stride);
            self.binary(Type::U32, Binary::IMul, index, stride)
        };
        if offset == 0 {
            return scaled;
        }
        let offset = self.u32_lit(offset);
        self.binary(Type::U32, Binary::IAdd, scaled, offset)
    }

    /// The word holding the 16-bit lane at byte `byte` (a multiple of 2) as an address of the word, and the shift
    /// of the lane within it (0 or 16).
    fn half_word(&mut self, byte: IrValue) -> (Address, IrValue) {
        let mask = self.u32_lit(!3);
        let word = self.binary(Type::U32, Binary::And, byte, mask);
        let two = self.u32_lit(2);
        let high = self.binary(Type::U32, Binary::And, byte, two);
        let three = self.u32_lit(3);
        let shift = self.binary(Type::U32, Binary::Shl, high, three);
        (Address::bytes(word), shift)
    }

    /// A read of 16- or 64-bit lanes of a raw or structured buffer: lane `k` is at `bytes * k` past the address, each
    /// zero outside the buffer. A 16-bit lane is half of a word (a half widened to `f32`, an `i16` as the low bits of
    /// a `u32`); a 64-bit one is two words, the low first.
    fn buffer_load_lanes(
        &mut self,
        inst: &Instruction,
        resource: Res,
        address: Address,
        bytes: u32,
    ) -> Result<()> {
        let fields = self.result_fields(inst)?;
        let Some((_, ty)) = inst.result else {
            return self.refuse("an operation that defines no result");
        };
        let LlvmType::Struct { elements, .. } = self.type_at(ty)?.clone() else {
            return self.refuse("a buffer read whose result is not a structure");
        };
        let half = matches!(
            elements.first().map(|&e| self.type_at(e)).transpose()?,
            Some(LlvmType::Half)
        );
        let mut items = Vec::new();
        for (k, want) in (0u32..4).zip(&fields) {
            let at = bytes.saturating_mul(k);
            let lane = if bytes == 8 {
                let address = Address {
                    offset: address.offset.wrapping_add(at),
                    ..address
                };
                let words = self.g.value(
                    Type::vector(Scalar::U32, 2),
                    Op::LoadBuffer {
                        resource,
                        address,
                        oob: OutOfBounds::Zero,
                    },
                );
                self.bitcast(*want, words)
            } else {
                let byte = self.address_byte(address, at);
                let (word, shift) = self.half_word(byte);
                let w = self.g.value(
                    Type::U32,
                    Op::LoadBuffer {
                        resource,
                        address: word,
                        oob: OutOfBounds::Zero,
                    },
                );
                let shifted = self.binary(Type::U32, Binary::ShrU, w, shift);
                let low = self.u32_lit(0xffff);
                let bits = self.binary(Type::U32, Binary::And, shifted, low);
                if half {
                    self.unary(Type::F32, Unary::F16ToF32, bits)
                } else {
                    bits
                }
            };
            items.push(Lv::Value(lane));
        }
        if fields.len() > 4 {
            items.push(Lv::Value(self.u32_lit(0)));
        }
        self.define_result(inst, Lv::Agg(items))
    }

    /// `BufferStore` and `RawBufferStore(handle, index, offset, v0..v3, mask)`: the masked lanes to the buffer's words
    /// (a typed buffer's element); a word past the end of the buffer is not written.
    pub(crate) fn buffer_store(&mut self, args: &[ValueId]) -> Result<()> {
        let (resource, i) = self.handle(self.arg(args, 1)?)?;
        let kind = self.buffer_kind(i)?;
        let Some(mask) = self.const_arg(args, 8) else {
            return self.refuse("a buffer write with a computed mask");
        };
        let mask = u8::try_from(mask & 15).unwrap_or(0);
        if let BufferKind::Typed(v) = kind {
            let scalar = Self::typed_scalar(v);
            let coord = self.uint(self.arg(args, 2)?)?;
            let t = Type::scalar(scalar);
            // the lanes the mask leaves keep the element's
            let before = if mask == 15 {
                None
            } else {
                Some(self.g.value(
                    Type::vector(scalar, 4),
                    Op::LoadTexel {
                        texture: resource,
                        coord,
                        level: None,
                        sample: None,
                        offset: [0; 3],
                    },
                ))
            };
            let mut lanes = Vec::new();
            for k in 0..4u8 {
                lanes.push(match before {
                    Some(old) if mask >> k & 1 == 0 => self.g.value(t, Op::Swizzle(old, [k, 0, 0, 0])),
                    _ => self.value_as(self.arg(args, 4usize.saturating_add(usize::from(k)))?, t)?,
                });
            }
            let value = self.g.value(Type::vector(scalar, 4), Op::Compose(lanes));
            self.g.node(Node::StoreTexel {
                image: resource,
                coord,
                value,
            });
            return Ok(());
        }
        let address = self.byte_address(kind, self.arg(args, 2)?, self.arg(args, 3)?)?;
        let ty = self
            .type_of(self.arg(args, 4)?)
            .ok_or_else(|| crate::error(self.at, "a buffer write of a value of no type"))?;
        let bytes = self.lane_bytes(ty)?;
        if bytes != 4 {
            return self.buffer_store_lanes(args, resource, address, mask, ty, bytes);
        }
        // each run of consecutive lanes of the mask is one store of that many words
        let mut k = 0u8;
        while k < 4 {
            if mask >> k & 1 == 0 {
                k = k.saturating_add(1);
                continue;
            }
            let mut words = Vec::new();
            let first = k;
            while k < 4 && mask >> k & 1 != 0 {
                words.push(self.uint(self.arg(args, 4usize.saturating_add(usize::from(k)))?)?);
                k = k.saturating_add(1);
            }
            let value = match words.as_slice() {
                [one] => *one,
                _ => {
                    let n = u8::try_from(words.len()).unwrap_or(4);
                    self.g.value(Type::vector(Scalar::U32, n), Op::Compose(words))
                }
            };
            let address = Address {
                offset: address.offset.wrapping_add(u32::from(first).saturating_mul(4)),
                ..address
            };
            self.g.node(Node::StoreBuffer {
                resource,
                address,
                value,
            });
        }
        Ok(())
    }

    /// A write of the masked 16- or 64-bit lanes (of LLVM type `ty`) of a raw or structured buffer, lane `k` at
    /// `bytes * k` past the address. A 64-bit lane is two words, the low first. A 16-bit lane is half of a word, the
    /// other half another invocation's to write: the half is cleared and set with an atomic and and or, so the word is
    /// never written whole.
    fn buffer_store_lanes(
        &mut self,
        args: &[ValueId],
        resource: Res,
        address: Address,
        mask: u8,
        ty: TypeId,
        bytes: u32,
    ) -> Result<()> {
        let t = self
            .carried_type(ty)?
            .ok_or_else(|| crate::error(self.at, "a buffer write of a value of no type"))?;
        let half = matches!(self.type_at(ty)?, LlvmType::Half);
        for k in (0u8..4).filter(|k| mask >> k & 1 != 0) {
            let v = self.value_as(self.arg(args, 4usize.saturating_add(usize::from(k)))?, t)?;
            let at = bytes.saturating_mul(u32::from(k));
            if bytes == 8 {
                let value = self.bitcast(Type::vector(Scalar::U32, 2), v);
                let address = Address {
                    offset: address.offset.wrapping_add(at),
                    ..address
                };
                self.g.node(Node::StoreBuffer {
                    resource,
                    address,
                    value,
                });
                continue;
            }
            let wide = if half {
                self.unary(Type::U32, Unary::F32ToF16, v)
            } else {
                v
            };
            let low = self.u32_lit(0xffff);
            let bits = self.binary(Type::U32, Binary::And, wide, low);
            let byte = self.address_byte(address, at);
            let (word, shift) = self.half_word(byte);
            let lane = self.binary(Type::U32, Binary::Shl, low, shift);
            let keep = self.unary(Type::U32, Unary::Not, lane);
            let set = self.binary(Type::U32, Binary::Shl, bits, shift);
            for (op, value) in [(AtomicOp::And, keep), (AtomicOp::Or, set)] {
                self.g.value(
                    Type::U32,
                    Op::Atomic {
                        memory: Memory::Buffer(resource, word),
                        op,
                        value,
                        compare: None,
                    },
                );
            }
        }
        Ok(())
    }

    /// `BufferUpdateCounter(view, step)`: add 1 or subtract 1 from the view's hidden counter; an increment gives the
    /// count before, a decrement the count after.
    pub(crate) fn buffer_counter(&mut self, inst: &Instruction, args: &[ValueId]) -> Result<()> {
        let (view, _) = self.handle(self.arg(args, 1)?)?;
        let op = match self.const_arg(args, 2) {
            Some(1) => CounterOp::Increment,
            Some(-1) => CounterOp::Decrement,
            Some(step) => return self.refuse(format!("a counter updated by {step}")),
            None => return self.refuse("a counter updated by a step that is not constant"),
        };
        let count = self.g.value(Type::U32, Op::Counter(view, op));
        self.define_result(inst, Lv::Value(count))
    }

    /// `GetDimensions(handle, level)`: a texture's size at the level, its layers and its levels; a buffer's elements
    /// (bytes for a raw buffer) and a structured buffer's stride.
    pub(crate) fn dimensions(&mut self, inst: &Instruction, args: &[ValueId]) -> Result<()> {
        let (resource, i) = self.handle(self.arg(args, 1)?)?;
        let zero = self.u32_lit(0);
        let parts: Vec<IrValue> = if let ResourceKind::Buffer(kind)
        | ResourceKind::Storage {
            target: StorageTarget::Buffer(kind),
            ..
        } = self.setup.kind(i).clone()
        {
            let elements = self.g.value(Type::U32, Op::Query(resource, Query::Elements));
            let stride = match kind {
                BufferKind::Structured { stride } => self.u32_lit(stride),
                _ => zero,
            };
            vec![elements, stride, zero, zero]
        } else {
            let shape = self.shape(i)?;
            let mipped = !shape.multisampled && !shape.storage;
            let level = self.uint(self.arg(args, 2)?)?;
            let n = match shape.dim {
                TextureDim::D1 => 1,
                TextureDim::D2 | TextureDim::Cube => 2,
                TextureDim::D3 => 3,
            };
            let size = self.g.value(
                Type::vector(Scalar::U32, n),
                Op::Query(resource, Query::Size(mipped.then_some(level))),
            );
            // the fourth lane is the levels, or a multisampled texture's samples (where dxc reads
            // `Texture2DMS::GetDimensions`'s sample count)
            let levels = if shape.multisampled {
                self.g.value(Type::U32, Op::Query(resource, Query::Samples))
            } else if mipped {
                self.g.value(Type::U32, Op::Query(resource, Query::Levels))
            } else {
                self.u32_lit(1)
            };
            let mut parts = Vec::new();
            for k in 0..n {
                parts.push(if n == 1 {
                    size
                } else {
                    self.g.value(Type::U32, Op::Swizzle(size, [k, 0, 0, 0]))
                });
            }
            if shape.arrayed {
                parts.push(self.g.value(Type::U32, Op::Query(resource, Query::Layers)));
            }
            parts.resize(3, zero);
            parts.push(levels);
            parts
        };
        let fields = self.result_fields(inst)?;
        let mut items = Vec::new();
        for (x, want) in parts.into_iter().zip(fields) {
            items.push(Lv::Value(self.retype(x, want)?));
        }
        self.define_result(inst, Lv::Agg(items))
    }

    /// `Texture2DMSGetSamplePosition(handle, sample)`: the position of sample `sample` modulo the texture's count, in
    /// D3D's standard pattern for that count.
    pub(crate) fn sample_position(&mut self, inst: &Instruction, args: &[ValueId]) -> Result<()> {
        let (texture, i) = self.handle(self.arg(args, 1)?)?;
        if !self.shape(i)?.multisampled {
            return self.refuse("a sample position of a texture that is not multisampled");
        }
        self.notes.insert(
            "a sample position is D3D's standard pattern for the texture's sample count (1, 2, 4, 8 or 16)"
                .into(),
        );
        let count = self.g.value(Type::U32, Op::Query(texture, Query::Samples));
        let one = self.u32_lit(1);
        let below = self.binary(Type::U32, Binary::ISub, count, one);
        let sample = self.uint(self.arg(args, 2)?)?;
        let within = self.binary(Type::U32, Binary::And, sample, below);
        let position = self.g.value(
            Type::vector(Scalar::F32, 2),
            Op::Query(texture, Query::SamplePosition(within)),
        );
        let x = self.g.value(Type::F32, Op::Swizzle(position, [0, 0, 0, 0]));
        let y = self.g.value(Type::F32, Op::Swizzle(position, [1, 0, 0, 0]));
        self.define_result(inst, Lv::Agg(vec![Lv::Value(x), Lv::Value(y)]))
    }

    /// The memory an atomic operation on view `resource` reaches, and the scalar its operands have: a storage image's
    /// texel or a typed buffer's element (of the view's scalar), or a raw or structured buffer's word.
    fn atomic_target(
        &mut self,
        resource: Res,
        i: usize,
        args: &[ValueId],
        first: usize,
    ) -> Result<(Memory, Scalar)> {
        match self.setup.kind(i).clone() {
            ResourceKind::Storage {
                target: StorageTarget::Image { sample, .. },
                ..
            } => {
                let shape = self.shape(i)?;
                let scalar = scalar_of_sample(sample);
                if scalar == Scalar::F32 {
                    return self
                        .refuse("an atomic operation on a float view: Direct3D takes R32_UINT and R32_SINT");
                }
                let coord = self.coordinate(args, first, shape, Scalar::U32)?;
                Ok((Memory::Image(resource, coord), scalar))
            }
            ResourceKind::Storage {
                target: StorageTarget::Buffer(kind),
                ..
            } => match kind {
                BufferKind::Typed(v) => {
                    let scalar = Self::typed_scalar(v);
                    if scalar == Scalar::F32 {
                        return self.refuse(
                            "an atomic operation on a float view: Direct3D takes R32_UINT and R32_SINT",
                        );
                    }
                    let coord = self.uint(self.arg(args, first)?)?;
                    Ok((Memory::Image(resource, coord), scalar))
                }
                _ => {
                    let address = self.byte_address(
                        kind,
                        self.arg(args, first)?,
                        self.arg(args, first.saturating_add(1))?,
                    )?;
                    Ok((Memory::Buffer(resource, address), Scalar::U32))
                }
            },
            _ => self.refuse("an atomic operation on something that is not an unordered-access view"),
        }
    }

    /// Refuse an atomic operation on a 64-bit integer by name: the lift takes 32-bit words.
    fn narrow_atomic(&self, inst: &Instruction) -> Result<()> {
        let wide = match inst.result {
            Some((_, ty)) => self.carried_type(ty)?.is_some_and(|t| t.scalar.bits() == 64),
            None => false,
        };
        if wide {
            return self.refuse("a 64-bit atomic operation is not lifted onto the IR yet");
        }
        Ok(())
    }

    /// `AtomicBinOp(view, op, a0, a1, a2, value)`: the word as it was; outside the view nothing changes and it is 0.
    pub(crate) fn buffer_atomic(&mut self, inst: &Instruction, args: &[ValueId]) -> Result<()> {
        self.narrow_atomic(inst)?;
        let (resource, i) = self.handle(self.arg(args, 1)?)?;
        let Some(op) = self
            .const_arg(args, 2)
            .and_then(|k| usize::try_from(k).ok())
            .and_then(|k| ATOMIC_OPS.get(k).copied())
        else {
            return self.refuse("an atomic operation of a computed or unknown kind");
        };
        let (memory, scalar) = self.atomic_target(resource, i, args, 3)?;
        let t = Type::scalar(scalar);
        let value = self.value_as(self.arg(args, 6)?, t)?;
        let old = self.g.value(
            t,
            Op::Atomic {
                memory,
                op,
                value,
                compare: None,
            },
        );
        let old = self.retype(old, Type::U32)?;
        self.define_result(inst, Lv::Value(old))
    }

    /// `AtomicCompareExchange(view, a0, a1, a2, compare, value)`.
    pub(crate) fn buffer_compare_exchange(&mut self, inst: &Instruction, args: &[ValueId]) -> Result<()> {
        self.narrow_atomic(inst)?;
        let (resource, i) = self.handle(self.arg(args, 1)?)?;
        let (memory, scalar) = self.atomic_target(resource, i, args, 2)?;
        let t = Type::scalar(scalar);
        let compare = self.value_as(self.arg(args, 5)?, t)?;
        let value = self.value_as(self.arg(args, 6)?, t)?;
        let old = self.g.value(
            t,
            Op::Atomic {
                memory,
                op: AtomicOp::CompareExchange,
                value,
                compare: Some(compare),
            },
        );
        let old = self.retype(old, Type::U32)?;
        self.define_result(inst, Lv::Value(old))
    }
}
