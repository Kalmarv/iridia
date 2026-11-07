//! LLVM values and the instructions that are not `dx.op` calls: constants, arithmetic, casts, comparisons, selects,
//! aggregates, vectors, and the memory of `alloca`, `getelementptr`, `load`, `store` and the atomics on it.

use half::f16;
use iridia_ir::{
    AtomicOp, Binary, Cmp, FloatToInt, Memory, MinMaxNan, MulZero, Node, OnZero, Op, OutOfBounds, Scalar,
    Space, Type, Unary, ValueId as IrValue, Var,
};

use super::{Lift, Lv, Pointer};
use crate::Result;
use crate::lift::setup::const_int;
use dxil_reader::module::{
    BinaryOp, CastOp, Constant, Instruction, Op as LlvmOp, Type as LlvmType, TypeId, Value, ValueId,
};

/// The most 32-bit words of memory (a local or thread-group array, a constant table) the lifter makes a variable of:
/// more than D3D's 64 KiB of thread-group and scratch memory.
const MAX_WORDS: u32 = 1 << 18;

/// The 32-bit words a value of `t` takes: its lanes times its width, over 32.
fn word_count(t: Type) -> u32 {
    u32::from(t.lanes).saturating_mul(t.scalar.bits()) / 32
}

impl Lift<'_> {
    // ---- types

    /// The IR type of a value of LLVM type `ty`: `bool` for `i1`, `u32` for a 2- to 32-bit integer (a narrow one
    /// carried as its low bits), `u64` for a 64-bit one, `f32` for a half (carried at 32 bits) or a float, `f64` for a
    /// double, a vector of those; `None` for a type no value of the IR has (a pointer, a structure, a handle).
    pub(crate) fn carried_type(&self, ty: TypeId) -> Result<Option<Type>> {
        Ok(match self.type_at(ty)? {
            LlvmType::Integer(1) => Some(Type::BOOL),
            LlvmType::Integer(2..=32) => Some(Type::U32),
            LlvmType::Integer(64) => Some(Type::scalar(Scalar::U64)),
            LlvmType::Half | LlvmType::Float => Some(Type::F32),
            LlvmType::Double => Some(Type::scalar(Scalar::F64)),
            LlvmType::Vector(n, e) => {
                let lanes = u8::try_from(*n).ok().filter(|n| (2..=4).contains(n));
                match (lanes, self.carried_type(*e)?) {
                    (Some(lanes), Some(t)) if t.lanes == 1 => Some(t.with(t.scalar).with_lanes(lanes)),
                    _ => return self.refuse(format!("a vector of {n} lanes")),
                }
            }
            _ => None,
        })
    }

    /// The width of a narrow integer type (2 to 31 bits), carried as the low bits of a `u32`.
    pub(crate) fn narrow_of_type(&self, ty: TypeId) -> Option<u32> {
        match self.m.types.get(ty) {
            Some(LlvmType::Integer(w @ 2..=31)) => Some(*w),
            _ => None,
        }
    }

    /// The LLVM type of a value.
    pub(crate) fn type_of(&self, v: ValueId) -> Option<TypeId> {
        match v.checked_sub(self.f.first) {
            None => self.m.value_types.get(v).copied(),
            Some(local) => self.f.local_types.get(local).copied(),
        }
    }

    /// The LLVM value at `v`: the module's, or the function's own.
    pub(crate) fn value_ref(&self, v: ValueId) -> Option<&Value> {
        match v.checked_sub(self.f.first) {
            None => self.m.values.get(v),
            Some(local) => self.f.locals.get(local),
        }
    }

    // ---- reading values

    /// A module value or a function value, translated where it is read.
    pub(crate) fn get(&mut self, v: ValueId) -> Result<Lv> {
        if let Some(i) = v.checked_sub(self.f.first)
            && let Some(Some(lv)) = self.values.get(i)
        {
            return Ok(lv.clone());
        }
        let (Some(value), Some(ty)) = (self.value_ref(v).cloned(), self.type_of(v)) else {
            return self.refuse(format!(
                "value {v}, which neither the module nor the function has"
            ));
        };
        match value {
            Value::Constant(c) => self.constant(&c, ty),
            Value::Global { .. } => self.global_pointer(v),
            _ if v >= self.f.first => self.refuse(format!("value {v} read before it is defined")),
            other => self.refuse(format!("a module value that is {other:?}")),
        }
    }

    /// An IR value.
    pub(crate) fn value(&mut self, v: ValueId) -> Result<IrValue> {
        match self.get(v)? {
            Lv::Value(x) => Ok(x),
            other => self.refuse(format!("a value was expected, found {other:?}")),
        }
    }

    /// An IR value as `want`: the same bits as another scalar of the same width and lanes.
    pub(crate) fn value_as(&mut self, v: ValueId, want: Type) -> Result<IrValue> {
        let x = self.value(v)?;
        self.retype(x, want)
    }

    /// `x` as `want`, by its bits.
    pub(crate) fn retype(&mut self, x: IrValue, want: Type) -> Result<IrValue> {
        let have = self
            .g
            .ty(x)
            .ok_or_else(|| crate::error(self.at, "a value of no type"))?;
        if have == want {
            return Ok(x);
        }
        if have.lanes == want.lanes
            && have.scalar.bits() == want.scalar.bits()
            && have.scalar != Scalar::Bool
            && want.scalar != Scalar::Bool
        {
            return Ok(self.g.value(want, Op::Bitcast(x)));
        }
        self.refuse(format!("a {have} where a {want} is read"))
    }

    pub(crate) fn float(&mut self, v: ValueId) -> Result<IrValue> {
        self.value_as(v, Type::F32)
    }
    pub(crate) fn uint(&mut self, v: ValueId) -> Result<IrValue> {
        self.value_as(v, Type::U32)
    }
    pub(crate) fn int(&mut self, v: ValueId) -> Result<IrValue> {
        self.value_as(v, Type::I32)
    }

    pub(crate) fn u32_lit(&mut self, x: u32) -> IrValue {
        self.g.value(Type::U32, Op::Literal([u64::from(x), 0, 0, 0]))
    }
    pub(crate) fn f32_lit(&mut self, x: f32) -> IrValue {
        self.g
            .value(Type::F32, Op::Literal([u64::from(x.to_bits()), 0, 0, 0]))
    }
    pub(crate) fn bool_lit(&mut self, x: bool) -> IrValue {
        self.g.value(Type::BOOL, Op::Literal([u64::from(x), 0, 0, 0]))
    }
    pub(crate) fn binary(&mut self, ty: Type, op: Binary, a: IrValue, b: IrValue) -> IrValue {
        self.g.value(ty, Op::Binary(op, a, b))
    }
    pub(crate) fn unary(&mut self, ty: Type, op: Unary, a: IrValue) -> IrValue {
        self.g.value(ty, Op::Unary(op, a))
    }
    pub(crate) fn select(&mut self, ty: Type, cond: IrValue, accept: IrValue, reject: IrValue) -> IrValue {
        self.g.value(ty, Op::Select { cond, accept, reject })
    }
    pub(crate) fn bitcast(&mut self, ty: Type, x: IrValue) -> IrValue {
        self.g.value(ty, Op::Bitcast(x))
    }

    /// The value of a constant of LLVM type `ty`.
    pub(crate) fn constant(&mut self, c: &Constant, ty: TypeId) -> Result<Lv> {
        if let LlvmType::Vector(n, element) = self.type_at(ty)?.clone() {
            let t = self
                .carried_type(ty)?
                .ok_or_else(|| crate::error(self.at, "a vector constant of no type"))?;
            let lanes: Vec<u64> = match c {
                Constant::Null | Constant::Undef => vec![0; usize::try_from(n).unwrap_or(0)],
                Constant::Aggregate(items) => {
                    let mut lanes = Vec::new();
                    for &item in items {
                        let Some(Value::Constant(lane)) = self.value_ref(item).cloned() else {
                            return self.refuse("a vector constant whose lane is not a constant");
                        };
                        lanes.push(self.constant_bits(&lane, element)?);
                    }
                    lanes
                }
                Constant::Data(words) => {
                    let mut lanes = Vec::new();
                    for &w in words {
                        lanes.push(self.constant_bits(&Constant::Integer(w as i64), element)?);
                    }
                    lanes
                }
                other => return self.refuse(format!("a vector constant of this form ({other:?})")),
            };
            let mut bits = [0u64; 4];
            for (b, l) in bits.iter_mut().zip(lanes) {
                *b = l;
            }
            return Ok(Lv::Value(self.g.value(t, Op::Literal(bits))));
        }
        match c {
            Constant::Gep { base, indices, .. } => {
                let (base, indices) = (*base, indices.clone());
                return self.gep(base, &indices);
            }
            Constant::Cast { op, value } => {
                if *op == CastOp::BitCast
                    && let p @ (Lv::Pointer(_) | Lv::Nothing) = self.get(*value)?
                {
                    return Ok(p);
                }
                let x = self.value(*value)?;
                let from = self.type_of(*value);
                return Ok(Lv::Value(self.cast(*op, x, from, ty)?));
            }
            Constant::Binary { op, lhs, rhs, .. } => {
                let r = self.binary_op(*op, *lhs, *rhs, ty)?;
                return Ok(Lv::Value(r));
            }
            Constant::Null | Constant::Undef
                if matches!(self.type_at(ty)?, LlvmType::Struct { .. } | LlvmType::Array(..)) =>
            {
                let elements: Vec<TypeId> = match self.type_at(ty)?.clone() {
                    LlvmType::Struct { elements, .. } => elements,
                    LlvmType::Array(n, e) => vec![e; usize::try_from(n).unwrap_or(0).min(4096)],
                    _ => Vec::new(),
                };
                let mut items = Vec::new();
                for e in elements {
                    items.push(self.constant(c, e)?);
                }
                return Ok(Lv::Agg(items));
            }
            _ => {}
        }
        let t = self
            .carried_type(ty)?
            .ok_or_else(|| crate::error(self.at, format!("a constant of a type no value has ({c:?})")))?;
        let bits = self.constant_bits(c, ty)?;
        Ok(Lv::Value(self.g.value(t, Op::Literal([bits, 0, 0, 0]))))
    }

    /// The bits of a scalar constant as its carried type holds them.
    fn constant_bits(&self, c: &Constant, ty: TypeId) -> Result<u64> {
        Ok(match (c, self.type_at(ty)?) {
            (Constant::Null | Constant::Undef, _) => 0,
            (Constant::Integer(i), LlvmType::Integer(1)) => (*i & 1) as u64,
            (Constant::Integer(i), LlvmType::Integer(64)) => *i as u64,
            (Constant::Integer(i), LlvmType::Integer(w @ 2..=32)) => {
                let mask = u64::MAX.checked_shl(*w).map_or(u64::MAX, |high| !high);
                (*i as u64) & mask
            }
            (Constant::Float(bits), LlvmType::Float) => u64::from(crate::lift::low32(*bits)),
            (Constant::Float(bits), LlvmType::Half) => u64::from(
                f16::from_bits(u16::try_from(bits & 0xffff).unwrap_or(0))
                    .to_f32()
                    .to_bits(),
            ),
            (Constant::Float(bits), LlvmType::Double) => *bits,
            // a constant array's data are its elements' bits, whatever their type
            (Constant::Integer(i), LlvmType::Float) => u64::from(crate::lift::low32(*i as u64)),
            (Constant::Integer(i), LlvmType::Half) => u64::from(
                f16::from_bits(u16::try_from(i & 0xffff).unwrap_or(0))
                    .to_f32()
                    .to_bits(),
            ),
            (Constant::Integer(i), LlvmType::Double) => *i as u64,
            (other, t) => return self.refuse(format!("a constant {other:?} of type {t:?}")),
        })
    }

    // ---- instructions

    /// An instruction that is no call, phi or terminator.
    pub(crate) fn plain(&mut self, inst: &Instruction) -> Result<()> {
        let result = inst.result;
        let define = |l: &mut Self, lv: Lv| -> Result<()> {
            match result {
                Some((v, ty)) => {
                    let lv = l.quantized(lv, ty)?;
                    l.define(v, lv)
                }
                None => Ok(()),
            }
        };
        match &inst.op {
            LlvmOp::Binary { op, lhs, rhs } => {
                let ty = result
                    .map(|r| r.1)
                    .ok_or_else(|| crate::error(self.at, "an operation of no result"))?;
                let r = self.binary_op(*op, *lhs, *rhs, ty)?;
                define(self, Lv::Value(r))
            }
            LlvmOp::Cast { op, value } => {
                let ty = result
                    .map(|r| r.1)
                    .ok_or_else(|| crate::error(self.at, "a cast of no result"))?;
                // a pointer cast is the same pointer
                if matches!(op, CastOp::BitCast | CastOp::AddrSpaceCast)
                    && let p @ (Lv::Pointer(_) | Lv::Nothing) = self.get(*value)?
                {
                    return define(self, p);
                }
                let x = self.value(*value)?;
                let from = self.type_of(*value);
                let r = self.cast(*op, x, from, ty)?;
                define(self, Lv::Value(r))
            }
            LlvmOp::Compare { predicate, lhs, rhs } => {
                let r = self.compare(*predicate, *lhs, *rhs)?;
                define(self, Lv::Value(r))
            }
            LlvmOp::Select {
                condition,
                then,
                otherwise,
            } => {
                let c = self.get(*condition)?;
                let (a, b) = (self.get(*then)?, self.get(*otherwise)?);
                let r = self.select_lv(c, a, b)?;
                define(self, r)
            }
            LlvmOp::ExtractValue { aggregate, indices } => {
                let mut lv = self.get(*aggregate)?;
                for &k in indices {
                    lv = match lv {
                        Lv::Agg(items) => items
                            .get(usize::try_from(k).unwrap_or(usize::MAX))
                            .cloned()
                            .ok_or_else(|| crate::error(self.at, format!("field {k} of a structure")))?,
                        other => return self.refuse(format!("a field of {other:?}")),
                    };
                }
                define(self, lv)
            }
            LlvmOp::InsertValue {
                aggregate,
                value,
                indices,
            } => {
                let base = self.get(*aggregate)?;
                let new = self.get(*value)?;
                let r = Self::insert(base, new, indices)
                    .map_or_else(|| self.refuse("an insertion into a non-structure"), Ok)?;
                define(self, r)
            }
            LlvmOp::ExtractElement { vector, index } => {
                let x = self.value(*vector)?;
                let Some(k) = const_int(self.m, self.f, *index) else {
                    return self.refuse("a vector lane read at a computed index");
                };
                let t = self
                    .g
                    .ty(x)
                    .ok_or_else(|| crate::error(self.at, "a vector of no type"))?;
                let lane = u8::try_from(k)
                    .ok()
                    .filter(|&k| k < t.lanes)
                    .ok_or_else(|| crate::error(self.at, "a lane past the vector"))?;
                let r = self
                    .g
                    .value(Type::scalar(t.scalar), Op::Swizzle(x, [lane, 0, 0, 0]));
                define(self, Lv::Value(r))
            }
            LlvmOp::InsertElement { vector, value, index } => {
                let x = self.value(*vector)?;
                let t = self
                    .g
                    .ty(x)
                    .ok_or_else(|| crate::error(self.at, "a vector of no type"))?;
                let y = self.value_as(*value, Type::scalar(t.scalar))?;
                let Some(k) = const_int(self.m, self.f, *index) else {
                    return self.refuse("a vector lane written at a computed index");
                };
                let mut parts = Vec::new();
                for lane in 0..t.lanes {
                    parts.push(if i64::from(lane) == k {
                        y
                    } else {
                        self.g
                            .value(Type::scalar(t.scalar), Op::Swizzle(x, [lane, 0, 0, 0]))
                    });
                }
                let r = self.g.value(t, Op::Compose(parts));
                define(self, Lv::Value(r))
            }
            LlvmOp::ShuffleVector { first, second, mask } => {
                let (a, b) = (self.value(*first)?, self.value(*second)?);
                let ta = self
                    .g
                    .ty(a)
                    .ok_or_else(|| crate::error(self.at, "a vector of no type"))?;
                let picks = self.vector_mask(*mask)?;
                let mut parts = Vec::new();
                for pick in &picks {
                    let lane = pick.unwrap_or(0);
                    let (src, k) = if lane < usize::from(ta.lanes) {
                        (a, lane)
                    } else {
                        (b, lane.saturating_sub(usize::from(ta.lanes)))
                    };
                    let k = u8::try_from(k).map_err(|_| crate::error(self.at, "a shuffle lane past 4"))?;
                    parts.push(
                        self.g
                            .value(Type::scalar(ta.scalar), Op::Swizzle(src, [k, 0, 0, 0])),
                    );
                }
                let n = u8::try_from(parts.len())
                    .map_err(|_| crate::error(self.at, "a shuffle of many lanes"))?;
                let r = match parts.as_slice() {
                    [one] => *one,
                    _ => self.g.value(Type::vector(ta.scalar, n), Op::Compose(parts)),
                };
                define(self, Lv::Value(r))
            }
            LlvmOp::Alloca { ty, .. } => {
                let ty = *ty;
                let v = result
                    .map(|r| r.0)
                    .ok_or_else(|| crate::error(self.at, "an alloca of no result"))?;
                let bytes = self.payload_roots.contains(&v);
                let var = self.memory_var("local", ty, Space::Private, Vec::new(), bytes)?;
                self.define(
                    v,
                    Lv::Pointer(Pointer {
                        var,
                        at: 0,
                        index: None,
                    }),
                )
            }
            LlvmOp::Gep { base, indices, .. } => {
                let r = self.gep(*base, indices)?;
                define(self, r)
            }
            LlvmOp::Load { pointer, .. } => {
                // the value of a library's resource global is the resource's handle
                if let Some(i) = self.setup.resource_of_variable(*pointer) {
                    let id = self.resource_id(i)?;
                    return define(self, Lv::Handle(iridia_ir::Res::of(id)));
                }
                if self
                    .root_global(*pointer)
                    .is_some_and(|g| self.setup.resource_of_variable(g).is_some())
                {
                    return self.refuse(
                        "an element of a library's array of resources is not lifted onto the IR yet",
                    );
                }
                let ty = result
                    .map(|r| r.1)
                    .ok_or_else(|| crate::error(self.at, "a load of no result"))?;
                let r = self.load(*pointer, ty)?;
                define(self, r)
            }
            LlvmOp::Store { pointer, value, .. } => self.store(*pointer, *value),
            LlvmOp::AtomicRmw {
                pointer, value, op, ..
            } => {
                let aop = match op {
                    0 => AtomicOp::Exchange,
                    1 => AtomicOp::Add,
                    2 => AtomicOp::Sub,
                    3 => AtomicOp::And,
                    5 => AtomicOp::Or,
                    6 => AtomicOp::Xor,
                    7 => AtomicOp::SMax,
                    8 => AtomicOp::SMin,
                    9 => AtomicOp::UMax,
                    10 => AtomicOp::UMin,
                    other => return self.refuse(format!("atomicrmw operation {other}")),
                };
                let r = self.atomic_memory(*pointer, aop, *value, None)?;
                define(self, Lv::Value(r))
            }
            LlvmOp::CmpXchg {
                pointer,
                compare,
                value,
                ..
            } => {
                let old = self.atomic_memory(*pointer, AtomicOp::CompareExchange, *value, Some(*compare))?;
                let c = self.uint(*compare)?;
                let swapped = self.binary(Type::BOOL, Binary::UCmp(Cmp::Eq), old, c);
                define(self, Lv::Agg(vec![Lv::Value(old), Lv::Value(swapped)]))
            }
            // ordering between invocations, which no single invocation's reading has
            LlvmOp::Fence(_) => Ok(()),
            other => self.refuse(format!("the instruction {other:?}")),
        }
    }

    /// `base` with field `indices` replaced by `new`.
    fn insert(base: Lv, new: Lv, indices: &[u64]) -> Option<Lv> {
        let Some((&first, rest)) = indices.split_first() else {
            return Some(new);
        };
        let Lv::Agg(mut items) = base else { return None };
        let at = usize::try_from(first).ok()?;
        let inner = items.get(at)?.clone();
        *items.get_mut(at)? = Self::insert(inner, new, rest)?;
        Some(Lv::Agg(items))
    }

    /// A select of values, of structures field by field, of handles and pointers that are the same either way.
    fn select_lv(&mut self, c: Lv, a: Lv, b: Lv) -> Result<Lv> {
        let Lv::Value(c) = c else {
            return self.refuse("a select on something that is not a value");
        };
        Ok(match (a, b) {
            (Lv::Value(x), Lv::Value(y)) => {
                let t = self
                    .g
                    .ty(x)
                    .ok_or_else(|| crate::error(self.at, "a value of no type"))?;
                let y = self.retype(y, t)?;
                let ct = self
                    .g
                    .ty(c)
                    .ok_or_else(|| crate::error(self.at, "a value of no type"))?;
                // a select of vectors on one condition takes the whole vector
                let c = if ct.lanes == 1 && t.lanes > 1 {
                    self.g.value(Type::vector(Scalar::Bool, t.lanes), Op::Splat(c))
                } else {
                    c
                };
                Lv::Value(self.select(t, c, x, y))
            }
            (Lv::Agg(xs), Lv::Agg(ys)) if xs.len() == ys.len() => {
                let mut out = Vec::new();
                for (x, y) in xs.into_iter().zip(ys) {
                    out.push(self.select_lv(Lv::Value(c), x, y)?);
                }
                Lv::Agg(out)
            }
            (other, _) => return self.refuse(format!("a select of {other:?}")),
        })
    }

    /// The lanes a shuffle's constant mask picks (`None` for an undefined lane).
    fn vector_mask(&self, mask: ValueId) -> Result<Vec<Option<usize>>> {
        let lane = |item: Option<&Value>| match item {
            Some(Value::Constant(Constant::Integer(k))) => usize::try_from(*k).ok().map(Some),
            Some(Value::Constant(Constant::Null)) => Some(Some(0)),
            Some(Value::Constant(Constant::Undef)) => Some(None),
            _ => None,
        };
        let lanes = match self.type_of(mask).and_then(|t| self.m.types.get(t)) {
            Some(LlvmType::Vector(n, _)) => usize::try_from(*n).unwrap_or(usize::MAX).min(4),
            _ => return self.refuse("a shuffle mask that is not a vector"),
        };
        let picks: Vec<Option<Option<usize>>> = match self.value_ref(mask) {
            Some(Value::Constant(Constant::Null)) => vec![Some(Some(0)); lanes],
            Some(Value::Constant(Constant::Undef)) => vec![Some(None); lanes],
            Some(Value::Constant(Constant::Aggregate(items))) => {
                items.iter().map(|&i| lane(self.value_ref(i))).collect()
            }
            Some(Value::Constant(Constant::Data(words))) => {
                words.iter().map(|&w| usize::try_from(w).ok().map(Some)).collect()
            }
            _ => return self.refuse("a shuffle mask that is not constant"),
        };
        picks
            .into_iter()
            .collect::<Option<Vec<_>>>()
            .map_or_else(|| self.refuse("a shuffle mask lane that is not a number"), Ok)
    }

    /// A value of type `ty` as the half it is: a half is carried as an `f32`, so each half made is rounded to the
    /// nearest half where it is defined (a structure's halves one by one), as the naga-path lift rounds it.
    pub(crate) fn quantized(&mut self, lv: Lv, ty: TypeId) -> Result<Lv> {
        Ok(match (lv, self.m.types.get(ty).cloned()) {
            (Lv::Value(v), Some(LlvmType::Half)) => {
                let bits = self.unary(Type::U32, Unary::F32ToF16, v);
                Lv::Value(self.unary(Type::F32, Unary::F16ToF32, bits))
            }
            (Lv::Agg(items), Some(LlvmType::Struct { elements, .. })) if items.len() == elements.len() => {
                let mut out = Vec::new();
                for (item, e) in items.into_iter().zip(elements) {
                    out.push(self.quantized(item, e)?);
                }
                Lv::Agg(out)
            }
            (lv, _) => lv,
        })
    }

    /// An LLVM binary operation of type `ty` on `lhs` and `rhs`.
    pub(crate) fn binary_op(
        &mut self,
        op: BinaryOp,
        lhs: ValueId,
        rhs: ValueId,
        ty: TypeId,
    ) -> Result<IrValue> {
        let t = self
            .carried_type(ty)?
            .ok_or_else(|| crate::error(self.at, "an operation on something that is not a value"))?;
        match t.scalar {
            Scalar::F32 | Scalar::F64 => {
                let (a, b) = (self.value_as(lhs, t)?, self.value_as(rhs, t)?);
                let bop = match op {
                    BinaryOp::Add => Binary::FAdd,
                    BinaryOp::Sub => Binary::FSub,
                    BinaryOp::Mul => Binary::FMul(MulZero::Ieee),
                    BinaryOp::UDiv | BinaryOp::SDiv => Binary::FDiv,
                    BinaryOp::URem | BinaryOp::SRem => Binary::FRem,
                    other => return self.refuse(format!("float {other:?}")),
                };
                Ok(self.binary(t, bop, a, b))
            }
            Scalar::Bool => {
                let (a, b) = (self.value_as(lhs, t)?, self.value_as(rhs, t)?);
                Ok(match op {
                    BinaryOp::And => self.binary(t, Binary::And, a, b),
                    BinaryOp::Or => self.binary(t, Binary::Or, a, b),
                    BinaryOp::Xor | BinaryOp::Add | BinaryOp::Sub => {
                        let not_b = self.unary(t, Unary::Not, b);
                        self.select(t, a, not_b, b)
                    }
                    other => return self.refuse(format!("boolean {other:?}")),
                })
            }
            _ => {
                let (a, b) = (self.value_as(lhs, t)?, self.value_as(rhs, t)?);
                match self.narrow_of_type(ty) {
                    Some(w) => self.integer_narrow(op, a, b, w),
                    None => self.integer(op, a, b, t),
                }
            }
        }
    }

    /// An integer operation on carried unsigned values of type `t` (`u32` or `u64`).
    fn integer(&mut self, op: BinaryOp, a: IrValue, b: IrValue, t: Type) -> Result<IrValue> {
        let signed = t.with(if t.scalar == Scalar::U64 {
            Scalar::I64
        } else {
            Scalar::I32
        });
        Ok(match op {
            BinaryOp::Add => self.binary(t, Binary::IAdd, a, b),
            BinaryOp::Sub => self.binary(t, Binary::ISub, a, b),
            BinaryOp::Mul => self.binary(t, Binary::IMul, a, b),
            BinaryOp::And => self.binary(t, Binary::And, a, b),
            BinaryOp::Or => self.binary(t, Binary::Or, a, b),
            BinaryOp::Xor => self.binary(t, Binary::Xor, a, b),
            // the IR takes a shift's count modulo the width, as Direct3D does
            BinaryOp::Shl => {
                let c = self.shift_count(b, t)?;
                self.binary(t, Binary::Shl, a, c)
            }
            BinaryOp::LShr => {
                let c = self.shift_count(b, t)?;
                self.binary(t, Binary::ShrU, a, c)
            }
            BinaryOp::AShr => {
                let c = self.shift_count(b, t)?;
                let s = self.bitcast(signed, a);
                let r = self.binary(signed, Binary::ShrS, s, c);
                self.bitcast(t, r)
            }
            // a zero divisor gives all ones, quotient and remainder, as Direct3D's udiv does
            BinaryOp::UDiv => self.binary(t, Binary::UDiv(OnZero::AllOnes), a, b),
            BinaryOp::URem => self.binary(t, Binary::URem(OnZero::AllOnes), a, b),
            BinaryOp::SDiv | BinaryOp::SRem => {
                let (sa, sb) = (self.bitcast(signed, a), self.bitcast(signed, b));
                let r = self.binary(
                    signed,
                    if op == BinaryOp::SDiv {
                        Binary::SDiv(OnZero::AllOnes)
                    } else {
                        Binary::SRem(OnZero::AllOnes)
                    },
                    sa,
                    sb,
                );
                self.bitcast(t, r)
            }
        })
    }

    /// A shift's count as the IR takes it: a `u32` of the shifted value's lanes.
    fn shift_count(&mut self, b: IrValue, t: Type) -> Result<IrValue> {
        if t.scalar == Scalar::U64 {
            Ok(self.g.value(
                Type::vector(Scalar::U32, t.lanes),
                Op::Convert(b, FloatToInt::Undef),
            ))
        } else {
            Ok(b)
        }
    }

    /// The low `width` bits of a `u32`: a narrow integer's carried form.
    pub(crate) fn mask_narrow(&mut self, v: IrValue, width: u32) -> IrValue {
        let mask = self.u32_lit(1u32.checked_shl(width).unwrap_or(0).wrapping_sub(1));
        self.binary(Type::U32, Binary::And, v, mask)
    }

    /// The sign extension of a carried narrow integer to 32 bits, as the same bits.
    pub(crate) fn sign_extend_narrow(&mut self, v: IrValue, width: u32) -> IrValue {
        let shift = self.u32_lit(32u32.saturating_sub(width));
        let high = self.binary(Type::U32, Binary::Shl, v, shift);
        let signed = self.bitcast(Type::I32, high);
        let back = self.binary(Type::I32, Binary::ShrS, signed, shift);
        self.bitcast(Type::U32, back)
    }

    /// An operation on narrow integers: the 32-bit operation, signed ones on sign extensions, the result cut back.
    fn integer_narrow(&mut self, op: BinaryOp, a: IrValue, b: IrValue, width: u32) -> Result<IrValue> {
        let a = if matches!(op, BinaryOp::AShr | BinaryOp::SDiv | BinaryOp::SRem) {
            self.sign_extend_narrow(a, width)
        } else {
            a
        };
        let b = if matches!(op, BinaryOp::SDiv | BinaryOp::SRem) {
            self.sign_extend_narrow(b, width)
        } else {
            b
        };
        let r = self.integer(op, a, b, Type::U32)?;
        Ok(self.mask_narrow(r, width))
    }

    /// A conversion of `x` (of LLVM type `from`) to LLVM type `to`.
    pub(crate) fn cast(
        &mut self,
        op: CastOp,
        x: IrValue,
        from: Option<TypeId>,
        to: TypeId,
    ) -> Result<IrValue> {
        let target = self.carried_type(to)?.ok_or_else(|| {
            crate::error(
                self.at,
                format!("the cast {op:?} to something that is not a value"),
            )
        })?;
        let have = self
            .g
            .ty(x)
            .ok_or_else(|| crate::error(self.at, "a value of no type"))?;
        let narrow_from = from.and_then(|t| self.narrow_of_type(t));
        let narrow_to = self.narrow_of_type(to);
        Ok(match op {
            CastOp::BitCast => self.retype(x, target)?,
            CastOp::ZExt | CastOp::SExt if have.scalar == Scalar::Bool => {
                let on = match (op, target.scalar) {
                    (CastOp::SExt, Scalar::U64) => u64::MAX,
                    (CastOp::SExt, _) => u64::from(
                        narrow_to.map_or(u32::MAX, |w| 1u32.checked_shl(w).unwrap_or(0).wrapping_sub(1)),
                    ),
                    _ => 1,
                };
                let on = self.g.value(target, Op::Literal([on, 0, 0, 0]));
                let off = self.g.value(target, Op::Literal([0; 4]));
                self.select(target, x, on, off)
            }
            CastOp::ZExt => {
                if target.scalar == Scalar::U64 {
                    self.g.value(target, Op::Convert(x, FloatToInt::Undef))
                } else {
                    x
                }
            }
            CastOp::SExt => {
                let extended = match narrow_from {
                    Some(w) => self.sign_extend_narrow(x, w),
                    None => x,
                };
                if target.scalar == Scalar::U64 {
                    let s = self.bitcast(Type::I32, extended);
                    let wide = self
                        .g
                        .value(Type::scalar(Scalar::I64), Op::Convert(s, FloatToInt::Undef));
                    self.bitcast(target, wide)
                } else {
                    match narrow_to {
                        Some(t) => self.mask_narrow(extended, t),
                        None => extended,
                    }
                }
            }
            CastOp::Trunc if target.scalar == Scalar::Bool => {
                let low = if have.scalar == Scalar::U64 {
                    self.g.value(Type::U32, Op::Convert(x, FloatToInt::Undef))
                } else {
                    x
                };
                let one = self.u32_lit(1);
                let bit = self.binary(Type::U32, Binary::And, low, one);
                let zero = self.u32_lit(0);
                self.binary(Type::BOOL, Binary::UCmp(Cmp::Ne), bit, zero)
            }
            CastOp::Trunc => {
                let low = if have.scalar == Scalar::U64 {
                    self.g.value(Type::U32, Op::Convert(x, FloatToInt::Undef))
                } else {
                    x
                };
                match narrow_to {
                    Some(t) => self.mask_narrow(low, t),
                    None => low,
                }
            }
            // a half is carried at 32 bits: widening changes nothing, narrowing is the rounding where it is defined
            CastOp::FpExt | CastOp::FpTrunc if have.scalar == target.scalar => x,
            CastOp::FpExt | CastOp::FpTrunc => self.g.value(target, Op::Convert(x, FloatToInt::Undef)),
            CastOp::FpToUi | CastOp::FpToSi => {
                let signed = op == CastOp::FpToSi;
                let int = if signed { Scalar::I32 } else { Scalar::U32 };
                if target.scalar == Scalar::U64 {
                    return self.refuse("a float converted to a 64-bit integer");
                }
                let r = self
                    .g
                    .value(Type::scalar(int), Op::Convert(x, FloatToInt::Saturate));
                let r = if signed { self.bitcast(Type::U32, r) } else { r };
                match narrow_to {
                    Some(t) => self.mask_narrow(r, t),
                    None => r,
                }
            }
            // a bool as an unsigned integer is 1, as a signed one -1
            CastOp::UiToFp | CastOp::SiToFp if have.scalar == Scalar::Bool => {
                let on = if op == CastOp::UiToFp { 1.0f64 } else { -1.0 };
                let bits = |v: f64| match target.scalar {
                    Scalar::F64 => v.to_bits(),
                    // 1, -1 and 0 are each a float exactly
                    #[allow(clippy::cast_possible_truncation)]
                    _ => u64::from((v as f32).to_bits()),
                };
                let on = self.g.value(target, Op::Literal([bits(on), 0, 0, 0]));
                let off = self.g.value(target, Op::Literal([bits(0.0), 0, 0, 0]));
                self.select(target, x, on, off)
            }
            CastOp::UiToFp => self.g.value(target, Op::Convert(x, FloatToInt::Undef)),
            CastOp::SiToFp => {
                let extended = match narrow_from {
                    Some(w) => self.sign_extend_narrow(x, w),
                    None => x,
                };
                let s = if have.scalar == Scalar::U64 {
                    self.bitcast(Type::scalar(Scalar::I64), extended)
                } else {
                    self.bitcast(Type::I32, extended)
                };
                self.g.value(target, Op::Convert(s, FloatToInt::Undef))
            }
            other => return self.refuse(format!("the cast {other:?}")),
        })
    }

    /// An LLVM comparison: float predicates 0 to 15, integer ones 32 to 41.
    fn compare(&mut self, predicate: u64, lhs: ValueId, rhs: ValueId) -> Result<IrValue> {
        let a = self.value(lhs)?;
        let t = self
            .g
            .ty(a)
            .ok_or_else(|| crate::error(self.at, "a value of no type"))?;
        let b = self.value_as(rhs, t)?;
        let bt = Type::vector(Scalar::Bool, t.lanes);
        if predicate < 32 {
            let f = |l: &mut Self, c: Cmp| l.binary(bt, Binary::FCmp(c), a, b);
            let not = |l: &mut Self, v: IrValue| l.unary(bt, Unary::Not, v);
            return Ok(match predicate {
                0 => self.bool_lit(false),
                1 => f(self, Cmp::Eq),
                2 => f(self, Cmp::Gt),
                3 => f(self, Cmp::Ge),
                4 => f(self, Cmp::Lt),
                5 => f(self, Cmp::Le),
                6 => {
                    let (l, g) = (f(self, Cmp::Lt), f(self, Cmp::Gt));
                    self.binary(bt, Binary::Or, l, g)
                }
                7 | 8 => {
                    let na = self.unary(bt, Unary::IsNan, a);
                    let nb = self.unary(bt, Unary::IsNan, b);
                    let unordered = self.binary(bt, Binary::Or, na, nb);
                    if predicate == 8 {
                        unordered
                    } else {
                        not(self, unordered)
                    }
                }
                9 => {
                    let (l, g) = (f(self, Cmp::Lt), f(self, Cmp::Gt));
                    let one = self.binary(bt, Binary::Or, l, g);
                    not(self, one)
                }
                10 => {
                    let c = f(self, Cmp::Le);
                    not(self, c)
                }
                11 => {
                    let c = f(self, Cmp::Lt);
                    not(self, c)
                }
                12 => {
                    let c = f(self, Cmp::Ge);
                    not(self, c)
                }
                13 => {
                    let c = f(self, Cmp::Gt);
                    not(self, c)
                }
                14 => f(self, Cmp::Ne),
                15 => self.bool_lit(true),
                other => return self.refuse(format!("the float predicate {other}")),
            });
        }
        if t.scalar == Scalar::Bool {
            return Ok(match predicate {
                32 => {
                    let ne = {
                        let not_b = self.unary(t, Unary::Not, b);
                        self.select(t, a, not_b, b)
                    };
                    self.unary(t, Unary::Not, ne)
                }
                33 => {
                    let not_b = self.unary(t, Unary::Not, b);
                    self.select(t, a, not_b, b)
                }
                other => return self.refuse(format!("the predicate {other} on booleans")),
            });
        }
        let signed = t.with(if t.scalar == Scalar::U64 {
            Scalar::I64
        } else {
            Scalar::I32
        });
        let s = |l: &mut Self, c: Cmp| {
            let (sa, sb) = (l.bitcast(signed, a), l.bitcast(signed, b));
            l.binary(bt, Binary::SCmp(c), sa, sb)
        };
        let u = |l: &mut Self, c: Cmp| l.binary(bt, Binary::UCmp(c), a, b);
        Ok(match predicate {
            32 => u(self, Cmp::Eq),
            33 => u(self, Cmp::Ne),
            34 => u(self, Cmp::Gt),
            35 => u(self, Cmp::Ge),
            36 => u(self, Cmp::Lt),
            37 => u(self, Cmp::Le),
            38 => s(self, Cmp::Gt),
            39 => s(self, Cmp::Ge),
            40 => s(self, Cmp::Lt),
            41 => s(self, Cmp::Le),
            other => return self.refuse(format!("the integer predicate {other}")),
        })
    }

    /// `min` or `max` of two floats with Direct3D's NaN rule (the number over a NaN).
    pub(crate) fn d3d_min_max(&mut self, max: bool, a: IrValue, b: IrValue) -> IrValue {
        let op = if max {
            Binary::FMax(MinMaxNan::MinNum)
        } else {
            Binary::FMin(MinMaxNan::MinNum)
        };
        let t = self.g.ty(a).unwrap_or(Type::F32);
        self.binary(t, op, a, b)
    }

    // ---- memory

    /// The 32-bit words a value of LLVM type `ty` takes in memory.
    fn words_of(&self, ty: TypeId) -> Result<u32> {
        Ok(match self.type_at(ty)? {
            LlvmType::Integer(1..=32) | LlvmType::Float | LlvmType::Half => 1,
            LlvmType::Integer(64) | LlvmType::Double => 2,
            LlvmType::Vector(n, e) | LlvmType::Array(n, e) => {
                let each = self.words_of(*e)?;
                u32::try_from(*n)
                    .ok()
                    .and_then(|n| n.checked_mul(each))
                    .filter(|&w| w <= MAX_WORDS)
                    .map_or_else(|| self.refuse("memory past the most the lifter makes"), Ok)?
            }
            LlvmType::Struct { elements, .. } => {
                let mut total = 0u32;
                for &e in elements {
                    total = total.saturating_add(self.words_of(e)?);
                }
                total
            }
            other => return self.refuse(format!("memory of type {other:?}")),
        })
    }

    /// A variable of the 32-bit words of a value of type `ty`, in `space`, starting as `init` (words): each value of
    /// 32 bits or fewer in a word of its own, or with `bytes` laid out in bytes as the module's datalayout lays the
    /// type out (a task payload).
    pub(crate) fn memory_var(
        &mut self,
        name: &str,
        ty: TypeId,
        space: Space,
        init: Vec<u32>,
        bytes: bool,
    ) -> Result<iridia_ir::VarId> {
        let words = if bytes {
            if !init.is_empty() {
                return self.refuse("a task payload with a start");
            }
            let (_, size) = self.byte_layout(ty)?;
            size.div_ceil(4)
        } else {
            self.words_of(ty)?
        };
        let id = iridia_ir::VarId(u32::try_from(self.g.vars.len()).unwrap_or(u32::MAX));
        if bytes {
            self.bytes.insert(id);
        }
        self.g.vars.push(Var {
            name: format!("{name}{}", id.0),
            ty: Type::U32,
            len: Some(words.max(1)),
            space,
            init: init.into_iter().map(|w| [u64::from(w), 0, 0, 0]).collect(),
        });
        Ok(id)
    }

    /// The module's memory the program reaches: thread-group variables (address space 3) and constant tables, each a
    /// variable of words; made where the function is first entered.
    pub(crate) fn declare_memory(&mut self) -> Result<()> {
        self.payload_roots = self.payload_roots();
        // a payload global DispatchMesh alone reaches (the program never writes it) is memory too
        let mut globals: std::collections::BTreeSet<ValueId> = self
            .payload_roots
            .iter()
            .copied()
            .filter(|&v| v < self.f.first)
            .collect();
        for inst in &self.f.instructions {
            let pointers: Vec<ValueId> = match &inst.op {
                LlvmOp::Load { pointer, .. } | LlvmOp::Store { pointer, .. } => vec![*pointer],
                LlvmOp::AtomicRmw { pointer, .. } | LlvmOp::CmpXchg { pointer, .. } => vec![*pointer],
                LlvmOp::Gep { base, .. } => vec![*base],
                _ => Vec::new(),
            };
            for p in pointers {
                if let Some(root) = self.root_global(p) {
                    globals.insert(root);
                }
            }
        }
        for g in globals {
            let Some(Value::Global {
                value_ty,
                address_space,
                init: initializer,
                constant,
                ..
            }) = self.m.values.get(g).cloned()
            else {
                continue;
            };
            let name = self.m.name(g).unwrap_or("global").to_owned();
            // dxc's no-op markers, and a library's resources, whose loads are their handles
            if name.starts_with("dx.nothing") || self.setup.resource_of_variable(g).is_some() {
                continue;
            }
            let (space, init) = match (address_space, initializer) {
                (3, _) => (Space::Workgroup, Vec::new()),
                (_, Some(init)) if constant => {
                    let mut words = Vec::new();
                    self.constant_words(init, value_ty, &mut words)?;
                    (Space::Private, words)
                }
                (space, init) => {
                    return self.refuse(format!(
                        "the module variable {name} of address space {space}{}",
                        if init.is_some() { " with a start" } else { "" }
                    ));
                }
            };
            let bytes = self.payload_roots.contains(&g);
            let var = self.memory_var("g", value_ty, space, init, bytes)?;
            self.memory.insert(g, var);
        }
        Ok(())
    }

    /// The words of a constant initializer, in memory order.
    fn constant_words(&mut self, c: ValueId, ty: TypeId, out: &mut Vec<u32>) -> Result<()> {
        let Some(Value::Constant(k)) = self.value_ref(c).cloned() else {
            return self.refuse("a module variable whose start is not a constant");
        };
        match (k, self.type_at(ty)?.clone()) {
            (Constant::Null | Constant::Undef, _) => {
                let n = self.words_of(ty)?;
                out.extend(std::iter::repeat_n(0, n as usize));
            }
            (Constant::Aggregate(items), LlvmType::Array(_, e) | LlvmType::Vector(_, e)) => {
                for item in items {
                    self.constant_words(item, e, out)?;
                }
            }
            (Constant::Aggregate(items), LlvmType::Struct { elements, .. }) => {
                for (item, e) in items.into_iter().zip(elements) {
                    self.constant_words(item, e, out)?;
                }
            }
            (Constant::Data(words), LlvmType::Array(_, e) | LlvmType::Vector(_, e)) => {
                for w in words {
                    let bits = self.constant_bits(&Constant::Integer(w as i64), e)?;
                    self.push_bits(bits, e, out)?;
                }
            }
            (k, _) => {
                let bits = self.constant_bits(&k, ty)?;
                self.push_bits(bits, ty, out)?;
            }
        }
        Ok(())
    }

    fn push_bits(&self, bits: u64, ty: TypeId, out: &mut Vec<u32>) -> Result<()> {
        match self.words_of(ty)? {
            1 => out.push(u32::try_from(bits & 0xffff_ffff).unwrap_or(0)),
            _ => {
                out.push(u32::try_from(bits & 0xffff_ffff).unwrap_or(0));
                out.push(u32::try_from(bits >> 32).unwrap_or(0));
            }
        }
        Ok(())
    }

    /// The module global a pointer value is made from, through getelementptrs and casts.
    fn root_global(&self, v: ValueId) -> Option<ValueId> {
        let mut v = v;
        for _ in 0..=self.f.instructions.len() {
            match self.value_ref(v) {
                Some(Value::Global { .. }) if v < self.f.first => return Some(v),
                Some(Value::Constant(Constant::Gep { base, .. })) => {
                    v = *base;
                    continue;
                }
                Some(Value::Constant(Constant::Cast { value, .. })) => {
                    v = *value;
                    continue;
                }
                Some(Value::Constant(_)) => return None,
                _ if v < self.f.first => return None,
                _ => {}
            }
            let inst = self
                .f
                .instructions
                .iter()
                .find(|i| i.result.is_some_and(|(r, _)| r == v))?;
            v = match &inst.op {
                LlvmOp::Gep { base, .. } => *base,
                LlvmOp::Cast { value, .. } => *value,
                _ => return None,
            };
        }
        None
    }

    /// The pointer a module global is: word 0 of its variable.
    fn global_pointer(&mut self, g: ValueId) -> Result<Lv> {
        if self.m.name(g).is_some_and(|n| n.starts_with("dx.nothing")) {
            return Ok(Lv::Nothing);
        }
        match self.memory.get(&g) {
            Some(&var) => Ok(Lv::Pointer(Pointer {
                var,
                at: 0,
                index: None,
            })),
            None => self.refuse(format!("the module variable {}", self.m.name(g).unwrap_or("?"))),
        }
    }

    /// The size of a value of LLVM type `ty` in variable `var`'s unit: words, or bytes in a byte-laid one.
    fn size_in(&self, var: iridia_ir::VarId, ty: TypeId) -> Result<u32> {
        if self.bytes.contains(&var) {
            Ok(self.byte_layout(ty)?.1)
        } else {
            self.words_of(ty)
        }
    }

    /// `getelementptr base, indices`: where it reaches, from the base's.
    fn gep(&mut self, base: ValueId, indices: &[ValueId]) -> Result<Lv> {
        let p = match self.get(base)? {
            Lv::Pointer(p) => p,
            Lv::Nothing => return Ok(Lv::Nothing),
            _ => return self.refuse("a getelementptr of something that is not a pointer"),
        };
        let Some(mut ty) = self.type_of(base).and_then(|t| match self.m.types.get(t) {
            Some(LlvmType::Pointer(inner, _)) => Some(*inner),
            _ => None,
        }) else {
            return self.refuse("a getelementptr of a pointer of no type");
        };
        let mut out = p;
        for (k, &index) in indices.iter().enumerate() {
            // the first index steps over the pointee itself; the rest step into it
            let (stride, next) = if k == 0 {
                (self.size_in(p.var, ty)?, ty)
            } else {
                match self.type_at(ty)?.clone() {
                    LlvmType::Array(_, e) | LlvmType::Vector(_, e) => (self.size_in(p.var, e)?, e),
                    LlvmType::Struct { elements, packed, .. } => {
                        let Some(field) = const_int(self.m, self.f, index) else {
                            return self.refuse("a structure field picked at a computed index");
                        };
                        let field =
                            usize::try_from(field).map_err(|_| crate::error(self.at, "a negative field"))?;
                        let before = if self.bytes.contains(&p.var) {
                            self.field_offset(&elements, packed, field)?
                        } else {
                            let mut before = 0u32;
                            for &e in elements.iter().take(field) {
                                before = before.saturating_add(self.words_of(e)?);
                            }
                            before
                        };
                        out.at = out.at.saturating_add(before);
                        ty = *elements
                            .get(field)
                            .ok_or_else(|| crate::error(self.at, "a field past the structure"))?;
                        continue;
                    }
                    other => return self.refuse(format!("a getelementptr into {other:?}")),
                }
            };
            match const_int(self.m, self.f, index) {
                Some(c) => {
                    let c = u32::try_from(c).map_err(|_| crate::error(self.at, "a negative fixed index"))?;
                    out.at = out.at.saturating_add(c.saturating_mul(stride));
                }
                None => {
                    let i = self.uint(index)?;
                    let step = if stride == 1 {
                        i
                    } else {
                        let s = self.u32_lit(stride);
                        self.binary(Type::U32, Binary::IMul, i, s)
                    };
                    out.index = Some(match out.index {
                        None => step,
                        Some(prior) => self.binary(Type::U32, Binary::IAdd, prior, step),
                    });
                }
            }
            if k > 0 {
                ty = next;
            }
        }
        Ok(Lv::Pointer(out))
    }

    /// The word index (`u32`) of a word-laid pointer's word plus `k`.
    fn word_index(&mut self, p: Pointer, k: u32) -> IrValue {
        let fixed = self.u32_lit(p.at.saturating_add(k));
        match p.index {
            None => fixed,
            Some(i) => self.binary(Type::U32, Binary::IAdd, i, fixed),
        }
    }

    /// A load of a value of LLVM type `ty` through `pointer`.
    fn load(&mut self, pointer: ValueId, ty: TypeId) -> Result<Lv> {
        match self.get(pointer)? {
            Lv::Pointer(p) => self.load_at(p, ty),
            Lv::Nothing => {
                let t = self
                    .carried_type(ty)?
                    .ok_or_else(|| crate::error(self.at, "a load of nothing of no value type"))?;
                Ok(Lv::Value(self.g.value(t, Op::Literal([0; 4]))))
            }
            _ => self.refuse("a load through something that is not a pointer"),
        }
    }

    fn load_at(&mut self, p: Pointer, ty: TypeId) -> Result<Lv> {
        if self.bytes.contains(&p.var) {
            return self.load_bytes(p, ty);
        }
        if let LlvmType::Struct { .. } | LlvmType::Array(_, _) = self.type_at(ty)? {
            let elements: Vec<TypeId> = match self.type_at(ty)?.clone() {
                LlvmType::Struct { elements, .. } => elements,
                LlvmType::Array(n, e) => vec![e; usize::try_from(n).unwrap_or(0).min(4096)],
                _ => Vec::new(),
            };
            let mut items = Vec::new();
            let mut at = p;
            for e in elements {
                items.push(self.load_at(at, e)?);
                at.at = at.at.saturating_add(self.words_of(e)?);
            }
            return Ok(Lv::Agg(items));
        }
        let t = self
            .carried_type(ty)?
            .ok_or_else(|| crate::error(self.at, "a load of something that is no value"))?;
        let words = self.words_of(ty)?;
        let mut lanes = Vec::new();
        for k in 0..words {
            let index = self.word_index(p, k);
            lanes.push(
                self.g
                    .value(Type::U32, Op::LoadElement(p.var, index, OutOfBounds::Zero)),
            );
        }
        Ok(Lv::Value(self.compose_words(&lanes, t)?))
    }

    /// The value of type `t` whose bits are `words` in order.
    pub(super) fn compose_words(&mut self, words: &[IrValue], t: Type) -> Result<IrValue> {
        let bits = match words {
            [one] => *one,
            _ => {
                let n =
                    u8::try_from(words.len()).map_err(|_| crate::error(self.at, "a value of many words"))?;
                if n > 4 {
                    return self.refuse("a value of more than four words");
                }
                self.g
                    .value(Type::vector(Scalar::U32, n), Op::Compose(words.to_vec()))
            }
        };
        if t.scalar == Scalar::Bool {
            let zero = self.u32_lit(0);
            return Ok(self.binary(Type::BOOL, Binary::UCmp(Cmp::Ne), bits, zero));
        }
        let have = self.g.ty(bits).unwrap_or(Type::U32);
        if have == t {
            Ok(bits)
        } else {
            Ok(self.g.value(t, Op::Bitcast(bits)))
        }
    }

    /// The 32-bit words of a value, in order.
    pub(super) fn split_words(&mut self, x: IrValue) -> Result<Vec<IrValue>> {
        let t = self
            .g
            .ty(x)
            .ok_or_else(|| crate::error(self.at, "a value of no type"))?;
        if t.scalar == Scalar::Bool {
            let (one, zero) = (self.u32_lit(1), self.u32_lit(0));
            return Ok(vec![self.select(Type::U32, x, one, zero)]);
        }
        let words = u8::try_from(word_count(t)).unwrap_or(0);
        let bits = if words == 1 {
            if t == Type::U32 {
                x
            } else {
                self.bitcast(Type::U32, x)
            }
        } else {
            self.bitcast(Type::vector(Scalar::U32, words), x)
        };
        if words == 1 {
            return Ok(vec![bits]);
        }
        Ok((0..words)
            .map(|k| self.g.value(Type::U32, Op::Swizzle(bits, [k, 0, 0, 0])))
            .collect())
    }

    /// A store of `value` through `pointer`.
    fn store(&mut self, pointer: ValueId, value: ValueId) -> Result<()> {
        let p = match self.get(pointer)? {
            Lv::Pointer(p) => p,
            Lv::Nothing => return Ok(()),
            _ => return self.refuse("a store through something that is not a pointer"),
        };
        let lv = self.get(value)?;
        if self.bytes.contains(&p.var) {
            let ty = self
                .type_of(value)
                .ok_or_else(|| crate::error(self.at, "a store of a value of no type"))?;
            return self.store_bytes(p, lv, ty);
        }
        self.store_at(p, lv)
    }

    fn store_at(&mut self, p: Pointer, lv: Lv) -> Result<()> {
        match lv {
            Lv::Value(x) => {
                let words = self.split_words(x)?;
                for (k, w) in (0..).zip(words) {
                    let index = self.word_index(p, k);
                    self.g
                        .node(Node::StoreElement(p.var, index, w, OutOfBounds::Zero));
                }
                Ok(())
            }
            Lv::Agg(items) => {
                let mut at = p;
                for item in items {
                    let width = self.lv_words(&item)?;
                    self.store_at(at, item)?;
                    at.at = at.at.saturating_add(width);
                }
                Ok(())
            }
            other => self.refuse(format!("a store of {other:?}")),
        }
    }

    fn lv_words(&self, lv: &Lv) -> Result<u32> {
        Ok(match lv {
            Lv::Value(x) => {
                let t = self
                    .g
                    .ty(*x)
                    .ok_or_else(|| crate::error(self.at, "a value of no type"))?;
                if t.scalar == Scalar::Bool {
                    u32::from(t.lanes)
                } else {
                    word_count(t)
                }
            }
            Lv::Agg(items) => {
                let mut total = 0u32;
                for item in items {
                    total = total.saturating_add(self.lv_words(item)?);
                }
                total
            }
            other => return self.refuse(format!("the size of {other:?}")),
        })
    }

    /// An atomic operation on the word a pointer reaches (thread-group memory): the word as it was before.
    fn atomic_memory(
        &mut self,
        pointer: ValueId,
        op: AtomicOp,
        value: ValueId,
        compare: Option<ValueId>,
    ) -> Result<IrValue> {
        let Lv::Pointer(p) = self.get(pointer)? else {
            return self.refuse("an atomic through something that is not a pointer");
        };
        let v = self.uint(value)?;
        let c = match compare {
            Some(c) => Some(self.uint(c)?),
            None => None,
        };
        let index = if self.bytes.contains(&p.var) {
            self.byte_word(p, 0)
        } else {
            self.word_index(p, 0)
        };
        Ok(self.g.value(
            Type::U32,
            Op::Atomic {
                memory: Memory::Var(p.var, Some(index)),
                op,
                value: v,
                compare: c,
            },
        ))
    }
}

/// `with_lanes`, which `Type` lacks: the same scalar in `lanes` lanes.
trait WithLanes {
    fn with_lanes(self, lanes: u8) -> Self;
}

impl WithLanes for Type {
    fn with_lanes(self, lanes: u8) -> Type {
        Type::vector(self.scalar, lanes)
    }
}
