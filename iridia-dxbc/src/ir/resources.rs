//! The declared resources, and the instructions that read them: samples, gathers, loads, sizes and buffer words.

use std::collections::BTreeMap;

use iridia_ir::{
    Address, Binary, FloatToInt, GatherOffset, Level, Op, OutOfBounds, Query, Res, Scalar, Space, Type, ValueId,
    VarId,
    interface::{BufferKind, ResourceId, ResourceKind, ScalarType},
};

use super::operands::{OperandAt, of_four};
use super::{Lift, lanes_of};
use crate::Result;
use crate::lift::setup::{Declaration, Texture};
use dxbc_reader::decode::{Instruction, Operand, Register, Selection};

/// How a shader-resource buffer `t#` is addressed.
#[derive(Clone, Copy, Debug)]
pub(super) enum Buffer {
    /// Four lanes per element, of this scalar.
    Typed(Scalar),
    /// Bytes; a structured element takes `stride` of them.
    Words { stride: u32 },
}

#[derive(Default)]
pub(super) struct Resources {
    pub constant_buffers: BTreeMap<u32, ResourceId>,
    pub textures: BTreeMap<u32, (ResourceId, Texture)>,
    pub samplers: BTreeMap<u32, ResourceId>,
    pub buffers: BTreeMap<u32, (ResourceId, Buffer)>,
    /// The immediate constant buffer, as an array variable, and its rows.
    pub immediate: Option<(VarId, u32)>,
}

/// The coordinate lanes of a texture: 1 (1D), 2 (2D, cube faces' size), 3 (3D, cube direction).
fn dims(t: &Texture) -> u8 {
    match t.dim {
        naga::ImageDimension::D1 => 1,
        naga::ImageDimension::D2 => 2,
        naga::ImageDimension::D3 | naga::ImageDimension::Cube => 3,
    }
}

/// The scalar a texture's loads return.
fn texel_scalar(t: &Texture) -> Scalar {
    match t.kind {
        iridia::build::Kind::I => Scalar::I32,
        iridia::build::Kind::U => Scalar::U32,
        _ => Scalar::F32,
    }
}

impl Lift<'_> {
    pub(super) fn declare_resources(&mut self) -> Result<()> {
        for (i, d) in self.setup.declarations.clone().iter().enumerate() {
            let Ok(id) = u32::try_from(i).map(ResourceId) else {
                return self.refuse("more than 4294967295 resources");
            };
            match d {
                Declaration::ConstantBuffer { slot, .. } => {
                    self.resources.constant_buffers.insert(*slot, id);
                }
                Declaration::Texture(t) => {
                    self.resources.textures.insert(t.slot, (id, *t));
                }
                Declaration::Sampler { slot, .. } => {
                    self.resources.samplers.insert(*slot, id);
                }
                Declaration::TypedBuffer { slot } => {
                    let scalar = match self.setup.interface.resources.iter().find(|r| r.id == id).map(|r| &r.kind) {
                        Some(ResourceKind::Buffer(BufferKind::Typed(v))) => match v.scalar {
                            ScalarType::Sint => Scalar::I32,
                            ScalarType::Uint => Scalar::U32,
                            _ => Scalar::F32,
                        },
                        _ => return self.refuse(format!("t{slot}: a typed buffer the interface does not state")),
                    };
                    self.resources.buffers.insert(*slot, (id, Buffer::Typed(scalar)));
                }
                Declaration::RawBuffer { slot } => {
                    self.resources.buffers.insert(*slot, (id, Buffer::Words { stride: 4 }));
                }
                Declaration::StructuredBuffer { slot, stride } => {
                    self.resources.buffers.insert(
                        *slot,
                        (
                            id,
                            Buffer::Words {
                                stride: (*stride).max(4),
                            },
                        ),
                    );
                }
                Declaration::View(_) | Declaration::UavBuffer(_) => {
                    return self.refuse("unordered-access views are not lifted to the IR yet");
                }
            }
        }
        if !self.setup.shared.is_empty() {
            return self.refuse("thread-group shared memory is not lifted to the IR yet");
        }
        if !self.setup.immediates.is_empty() {
            let Ok(rows) = u32::try_from(self.setup.immediates.len()) else {
                return self.refuse("more than 4294967295 immediate rows");
            };
            let init = self
                .setup
                .immediates
                .iter()
                .map(|row| row.map(u64::from))
                .collect();
            let var = self.b.array("icb", Type::U32X4, rows, Space::Private, init);
            self.resources.immediate = Some((var, rows));
        }
        Ok(())
    }

    fn texture(&self, slot: u32) -> Result<(ResourceId, Texture)> {
        match self.resources.textures.get(&slot) {
            Some(t) => Ok(*t),
            None if self.resources.buffers.contains_key(&slot) => {
                self.refuse(format!("t{slot} is a buffer, used as a texture"))
            }
            None => self.refuse(format!("t{slot} is used but not declared")),
        }
    }

    /// The slot operand `n` of `ins` names, which must be a register of one of `kinds`.
    fn slot_of(&self, ins: &Instruction, n: usize, kinds: &[Register]) -> Result<u32> {
        let operand = ins.operand(n)?;
        match operand.indices.first() {
            Some(index) if kinds.contains(&operand.register) => Ok(index.base),
            _ => self.refuse(format!(
                "operand {n} is {:?}, where {kinds:?} is taken",
                operand.register
            )),
        }
    }

    /// The result lanes of `mask` of a resource operand's swizzle applied to a four-lane result.
    fn pick(&mut self, result: ValueId, op: &Operand, mask: u8) -> Result<ValueId> {
        let lanes: Vec<u8> = lanes_of(mask)
            .iter()
            .map(|l| match op.selection {
                Selection::Swizzle(s) => of_four(s, *l),
                Selection::Scalar(c) => c,
                _ => 0,
            })
            .collect();
        self.swizzle(result, &lanes)
    }

    /// A depth texture's one lane as D3D's float view returns it: the depth in x, then 0, 0, 1.
    fn depth_view(&mut self, d: ValueId) -> Result<ValueId> {
        let zero = self.b.f32(0.0);
        let one = self.b.f32(1.0);
        self.op(Op::Compose(vec![d, zero, zero, one]))
    }

    pub(super) fn sample(&mut self, ins: &Instruction) -> Result<()> {
        let name = ins.name();
        let slot = self.slot_of(ins, 2, &[Register::Texture])?;
        let sampler = self.slot_of(ins, 3, &[Register::Sampler])?;
        let (texture, t) = self.texture(slot)?;
        let Some(&sampler) = self.resources.samplers.get(&sampler) else {
            return self.refuse(format!("s{sampler} is used but not declared"));
        };
        if t.multi {
            return self.refuse("a sample of a multisampled texture");
        }
        let n = dims(&t);
        let mask = self.dest_mask(ins.operand(0)?);
        let lanes: Vec<u8> = (0..n.saturating_add(u8::from(t.arrayed))).collect();
        let coord = self.read_lanes(ins.operand(1)?, &lanes, Scalar::F32)?;
        let mut offset = [0i8; 3];
        if let Some(o) = ins.sample_offset() {
            for (to, from) in offset.iter_mut().zip(o).take(usize::from(n)) {
                *to = i8::try_from(from).unwrap_or(0);
            }
        }
        let component = match ins.operand(3)?.selection {
            Selection::Scalar(c) => c,
            Selection::Swizzle(s) => s[0],
            _ => 0,
        };
        let reference = |l: &mut Self| l.read(ins.operand(4)?, 1, Scalar::F32);
        let gradient = |l: &mut Self, k: usize| {
            let g: Vec<u8> = (0..n).collect();
            l.read_lanes(ins.operand(k)?, &g, Scalar::F32)
        };
        let (texture, sampler) = (Res::of(texture), Res::of(sampler));
        let (op, gather, compare) = match name {
            "gather4" | "gather4_c" => {
                let compare = if name == "gather4_c" {
                    Some(reference(self)?)
                } else {
                    None
                };
                (
                    Op::Gather {
                        texture,
                        sampler,
                        coord,
                        component,
                        compare,
                        offset: GatherOffset::Fixed(offset),
                    },
                    true,
                    compare.is_some(),
                )
            }
            _ => {
                let (level, compare) = match name {
                    "sample" => (Level::Auto, None),
                    "sample_l" => (Level::Exact(self.read(ins.operand(4)?, 1, Scalar::F32)?), None),
                    "sample_b" => (Level::Bias(self.read(ins.operand(4)?, 1, Scalar::F32)?), None),
                    "sample_d" => (Level::Grad(gradient(self, 4)?, gradient(self, 5)?), None),
                    "sample_c" => (Level::Auto, Some(reference(self)?)),
                    "sample_c_lz" => (Level::Zero, Some(reference(self)?)),
                    other => return self.refuse(format!("{other} is not lifted to the IR")),
                };
                (
                    Op::Sample {
                        texture,
                        sampler,
                        coord,
                        level,
                        compare,
                        offset,
                        clamp: None,
                    },
                    false,
                    compare.is_some(),
                )
            }
        };
        let one = t.depth && !gather;
        // a sample or gather gives the texture's sample type; a depth texture's, and a comparison's, floats
        let scalar = if t.depth || compare { Scalar::F32 } else { texel_scalar(&t) };
        let mut result = self
            .b
            .value(if one { Type::F32 } else { Type::vector(scalar, 4) }, op);
        if one {
            // D3D's float view of a depth texture: the depth, then 0, 0, 1; a comparison's result in every lane
            result = if compare {
                self.b.value(Type::F32X4, Op::Splat(result))
            } else {
                self.depth_view(result)?
            };
        }
        let bits = self.bitcast(result, Scalar::U32)?;
        let picked = self.pick(bits, ins.operand(2)?, mask)?;
        self.write(ins.operand(0)?, picked, ins.saturate)
    }

    pub(super) fn load(&mut self, ins: &Instruction) -> Result<()> {
        let slot = self.slot_of(ins, 2, &[Register::Texture])?;
        let mask = self.dest_mask(ins.operand(0)?);
        if let Some((id, buffer)) = self.resources.buffers.get(&slot).copied() {
            let Buffer::Typed(scalar) = buffer else {
                return self.refuse("ld on a raw or structured buffer");
            };
            let element = self.read(ins.operand(1)?, 1, Scalar::I32)?;
            let texel = self.b.value(
                Type::vector(scalar, 4),
                Op::LoadTexel {
                    texture: Res::of(id),
                    coord: element,
                    level: None,
                    sample: None,
                    offset: [0; 3],
                },
            );
            let row = self.bitcast(texel, Scalar::U32)?;
            let picked = self.pick(row, ins.operand(2)?, mask)?;
            return self.write(ins.operand(0)?, picked, false);
        }
        let (id, t) = self.texture(slot)?;
        let n = dims(&t);
        let all = self.read_lanes(ins.operand(1)?, &[0, 1, 2, 3], Scalar::I32)?;
        let lanes: Vec<u8> = (0..n.saturating_add(u8::from(t.arrayed))).collect();
        let coord = self.swizzle(all, &lanes)?;
        let level = if t.multi { None } else { Some(self.lane(all, 3)?) };
        let sample = if ins.name() == "ld_ms" {
            Some(self.read(ins.operand(3)?, 1, Scalar::I32)?)
        } else {
            None
        };
        let mut offset = [0i8; 3];
        if let Some(o) = ins.sample_offset() {
            for (to, from) in offset.iter_mut().zip(o).take(usize::from(n)) {
                *to = i8::try_from(from).unwrap_or(0);
            }
        }
        let op = Op::LoadTexel {
            texture: Res::of(id),
            coord,
            level,
            sample,
            offset,
        };
        let result = if t.depth {
            let d = self.b.value(Type::F32, op);
            self.depth_view(d)?
        } else {
            self.b.value(Type::vector(texel_scalar(&t), 4), op)
        };
        let bits = self.bitcast(result, Scalar::U32)?;
        let picked = self.pick(bits, ins.operand(2)?, mask)?;
        self.write(ins.operand(0)?, picked, false)
    }

    /// `sample_info dest, t#`: the sample count of a multisampled texture in x (0 for one that is not), and 0 in the
    /// other lanes, as a uint or a float by the instruction's control.
    pub(super) fn sample_info(&mut self, ins: &Instruction) -> Result<()> {
        if ins.operand(1)?.register != Register::Texture {
            return self.refuse("the sample count of the rasterizer is the pipeline's, not the program's");
        }
        let slot = self.index_of(ins.operand(1)?, 0)?.base;
        let (id, t) = self.texture(slot)?;
        let mask = self.dest_mask(ins.operand(0)?);
        let count = if t.multi {
            self.b.value(Type::U32, Op::Query(Res::of(id), Query::Samples))
        } else {
            self.b.u32(0)
        };
        let zero = self.b.u32(0);
        let all = self.op(Op::Compose(vec![count, zero, zero, zero]))?;
        let result = if ins.control & 1 != 0 {
            all
        } else {
            self.b.value(Type::F32X4, Op::Convert(all, FloatToInt::Saturate))
        };
        let bits = self.bitcast(result, Scalar::U32)?;
        let picked = self.pick(bits, ins.operand(1)?, mask)?;
        self.write(ins.operand(0)?, picked, false)
    }

    /// `resinfo dest, level, t#`: the size at the level, the layers of an array, then the level count; as floats,
    /// their reciprocals or uints by the instruction's control.
    pub(super) fn resource_info(&mut self, ins: &Instruction) -> Result<()> {
        let slot = self.slot_of(ins, 2, &[Register::Texture])?;
        let (id, t) = self.texture(slot)?;
        let mask = self.dest_mask(ins.operand(0)?);
        let level = self.read(ins.operand(1)?, 1, Scalar::U32)?;
        let size_lanes = match t.dim {
            naga::ImageDimension::D1 => 1,
            naga::ImageDimension::D2 | naga::ImageDimension::Cube => 2,
            naga::ImageDimension::D3 => 3,
        };
        let texture = Res::of(id);
        let size = self.b.value(
            Type::vector(Scalar::U32, size_lanes),
            Op::Query(texture, Query::Size((!t.multi).then_some(level))),
        );
        let levels = if t.multi {
            self.b.u32(1)
        } else {
            self.b.value(Type::U32, Op::Query(texture, Query::Levels))
        };
        let zero = self.b.u32(0);
        let mut parts = Vec::new();
        for k in 0..size_lanes {
            parts.push(self.lane(size, k)?);
        }
        if t.arrayed {
            parts.push(self.b.value(Type::U32, Op::Query(texture, Query::Layers)));
        }
        while parts.len() < 3 {
            parts.push(zero);
        }
        parts.truncate(3);
        parts.push(levels);
        let all = self.op(Op::Compose(parts))?;
        let result = match ins.control & 3 {
            2 => all,
            0 => self.b.value(Type::F32X4, Op::Convert(all, FloatToInt::Saturate)),
            _ => {
                let f = self.b.value(Type::F32X4, Op::Convert(all, FloatToInt::Saturate));
                let one = self.b.f32x4([1.0; 4]);
                self.binary(Binary::FDiv, one, f)?
            }
        };
        let bits = self.bitcast(result, Scalar::U32)?;
        let picked = self.pick(bits, ins.operand(2)?, mask)?;
        self.write(ins.operand(0)?, picked, false)
    }

    /// The lanes of a raw or structured read from byte `byte`: for each result lane, the word of its source
    /// component; a word outside the buffer reads as zero.
    fn load_words(&mut self, id: ResourceId, byte: ValueId, resource: &Operand, mask: u8) -> Result<ValueId> {
        let mut lanes = Vec::new();
        for l in lanes_of(mask) {
            let c = match resource.selection {
                Selection::Swizzle(s) => of_four(s, l),
                Selection::Scalar(c) => c,
                _ => 0,
            };
            lanes.push(self.b.value(
                Type::U32,
                Op::LoadBuffer {
                    resource: Res::of(id),
                    address: Address {
                        index: Some(byte),
                        stride: 1,
                        offset: u32::from(c).saturating_mul(4),
                    },
                    oob: OutOfBounds::Zero,
                },
            ));
        }
        if let [one] = lanes[..] {
            return Ok(one);
        }
        self.op(Op::Compose(lanes))
    }

    fn words_buffer(&self, op: &Operand) -> Result<(ResourceId, u32)> {
        if op.register != Register::Texture {
            return self.refuse("a read of an unordered-access buffer is not lifted to the IR yet");
        }
        let slot = self.index_of(op, 0)?.base;
        match self.resources.buffers.get(&slot) {
            Some(&(id, Buffer::Words { stride })) => Ok((id, stride)),
            _ => self.refuse(format!("t{slot} is not a raw or structured buffer")),
        }
    }

    pub(super) fn load_raw(&mut self, ins: &Instruction) -> Result<()> {
        let (id, _) = self.words_buffer(ins.operand(2)?)?;
        let mask = self.dest_mask(ins.operand(0)?);
        let byte = self.read(ins.operand(1)?, 1, Scalar::U32)?;
        let result = self.load_words(id, byte, ins.operand(2)?, mask)?;
        self.write(ins.operand(0)?, result, false)
    }

    pub(super) fn load_structured(&mut self, ins: &Instruction) -> Result<()> {
        let (id, stride) = self.words_buffer(ins.operand(3)?)?;
        let mask = self.dest_mask(ins.operand(0)?);
        let element = self.read(ins.operand(1)?, 1, Scalar::U32)?;
        let offset = self.read(ins.operand(2)?, 1, Scalar::U32)?;
        let stride = self.b.u32(stride);
        let base = self.binary(Binary::IMul, element, stride)?;
        let byte = self.binary(Binary::IAdd, base, offset)?;
        let result = self.load_words(id, byte, ins.operand(3)?, mask)?;
        self.write(ins.operand(0)?, result, false)
    }
}
