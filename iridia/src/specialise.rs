//! Conservative material specialisation. Only explicitly supplied uniform words are constants.
//! The original program remains unchanged; unsupported control flow is refused rather than guessed.
use crate::{Program, ResourceId, ResourceKind};
use naga::{Expression as E, Handle, Literal as L, ScalarKind as K, Statement as S, TypeInner as T};
use std::collections::{BTreeMap, BTreeSet};
/// A known raw word in a constant buffer, at its byte offset.
pub type Words = BTreeMap<(ResourceId, u32), u32>;
/// A specialised copy and the evidence for its eliminated resource reads.
#[derive(Clone, Debug)]
pub struct Specialised {
    /// The new program, with material constants folded and dead code removed.
    pub program: Program,
    /// The supplied source words selected as fully stated feature decisions.
    pub words: Words,
    /// Resources whose declarations no longer survive dead code elimination.
    pub unread: Vec<ResourceId>,
    /// Number of expression results replaced with known values.
    pub folded: usize,
}
type V = Vec<Option<L>>;
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Root {
    Local(Handle<naga::LocalVariable>),
    Global(Handle<naga::GlobalVariable>),
}
fn lanes(t: Handle<naga::Type>, m: &naga::Module) -> Result<usize, String> {
    Ok(match &m.types[t].inner {
        T::Scalar(_) => 1,
        T::Vector { size, .. } => *size as usize,
        T::Matrix { columns, rows, .. } => (*columns as usize)
            .checked_mul(*rows as usize)
            .ok_or("matrix lane count overflow")?,
        T::Array {
            base,
            size: naga::ArraySize::Constant(n),
            ..
        } => lanes(*base, m)?
            .checked_mul(n.get() as usize)
            .ok_or("array lane count overflow")?,
        T::Struct { members, .. } => members.iter().try_fold(0usize, |count, s| {
            count
                .checked_add(lanes(s.ty, m)?)
                .ok_or_else(|| "structure lane count overflow".to_owned())
        })?,
        T::Pointer { base, .. } => lanes(*base, m)?,
        T::ValuePointer { size, .. } => size.map_or(1, |s| s as usize),
        _ => 1,
    })
}
fn literal(l: L) -> V {
    vec![Some(l)]
}
fn float(v: Option<L>) -> Option<f32> {
    match v? {
        L::F32(f) => Some(f),
        _ => None,
    }
}
fn truth(v: Option<L>) -> Option<bool> {
    match v? {
        L::Bool(b) => Some(b),
        _ => None,
    }
}
fn same(a: Option<L>, b: Option<L>) -> bool {
    match (a, b) {
        (Some(L::F32(a)), Some(L::F32(b))) => a.to_bits() == b.to_bits(),
        (Some(L::F64(a)), Some(L::F64(b))) | (Some(L::AbstractFloat(a)), Some(L::AbstractFloat(b))) => {
            a.to_bits() == b.to_bits()
        }
        (Some(L::F16(a)), Some(L::F16(b))) => a.to_bits() == b.to_bits(),
        (a, b) => a == b,
    }
}
fn binary(op: naga::BinaryOperator, a: L, b: L) -> Option<L> {
    use naga::BinaryOperator as B;
    match (a, b) {
        (L::F32(a), L::F32(b)) => Some(match op {
            B::Add => L::F32(a + b),
            B::Subtract => L::F32(a - b),
            B::Multiply => L::F32(a * b),
            B::Divide => L::F32(a / b),
            B::Equal => L::Bool(a == b),
            B::NotEqual => L::Bool(a != b),
            B::Less => L::Bool(a < b),
            B::LessEqual => L::Bool(a <= b),
            B::Greater => L::Bool(a > b),
            B::GreaterEqual => L::Bool(a >= b),
            _ => return None,
        }),
        (L::Bool(a), L::Bool(b)) => Some(match op {
            B::LogicalAnd => L::Bool(a && b),
            B::LogicalOr => L::Bool(a || b),
            B::Equal => L::Bool(a == b),
            B::NotEqual => L::Bool(a != b),
            _ => return None,
        }),
        (L::U32(a), L::U32(b)) => Some(match op {
            B::Add => L::U32(a.wrapping_add(b)),
            B::Subtract => L::U32(a.wrapping_sub(b)),
            B::Multiply => L::U32(a.wrapping_mul(b)),
            B::Equal => L::Bool(a == b),
            B::NotEqual => L::Bool(a != b),
            B::And => L::U32(a & b),
            B::InclusiveOr => L::U32(a | b),
            _ => return None,
        }),
        _ => None,
    }
}
struct Fold<'a> {
    m: &'a mut naga::Module,
    f: naga::Function,
    types: Vec<Handle<naga::Type>>,
    values: Vec<V>,
    state: BTreeMap<Root, V>,
    folded: usize,
    pre: BTreeSet<Handle<E>>,
}
impl Fold<'_> {
    fn ty(&self, h: Handle<E>) -> Result<Handle<naga::Type>, String> {
        self.types
            .get(h.index())
            .copied()
            .ok_or_else(|| "expression has no type in specialisation".into())
    }
    fn width(&self, h: Handle<E>) -> Result<usize, String> {
        lanes(self.ty(h)?, self.m)
    }
    fn val(&self, h: Handle<E>) -> Result<V, String> {
        self.values
            .get(h.index())
            .cloned()
            .ok_or_else(|| "expression has no value slot in specialisation".into())
    }
    fn ptr(&self, h: Handle<E>) -> Option<(Root, usize, usize)> {
        match self.f.expressions[h] {
            E::LocalVariable(l) => Some((Root::Local(l), 0, self.width(h).ok()?)),
            E::GlobalVariable(g) => Some((Root::Global(g), 0, self.width(h).ok()?)),
            E::AccessIndex { base, index } => {
                let (root, offset, _) = self.ptr(base)?;
                let base_type = self.ty(base).ok()?;
                let t = match self.m.types[base_type].inner {
                    T::Pointer { base, .. } => base,
                    _ => base_type,
                };
                let start = match &self.m.types[t].inner {
                    T::Struct { members, .. } => members
                        .iter()
                        .take(index as usize)
                        .try_fold(0usize, |count, s| count.checked_add(lanes(s.ty, self.m).ok()?))?,
                    T::Array { base, .. } => (index as usize).checked_mul(lanes(*base, self.m).ok()?)?,
                    T::Matrix { rows, .. } => (index as usize).checked_mul(*rows as usize)?,
                    _ => index as usize,
                };
                Some((root, offset.checked_add(start)?, self.width(h).ok()?))
            }
            _ => None,
        }
    }
    fn root(&self, h: Handle<E>) -> Option<Root> {
        match self.f.expressions[h] {
            E::LocalVariable(l) => Some(Root::Local(l)),
            E::GlobalVariable(g) => Some(Root::Global(g)),
            E::AccessIndex { base, .. } | E::Access { base, .. } => self.root(base),
            _ => None,
        }
    }
    fn eval(&self, h: Handle<E>) -> Result<V, String> {
        let width = self.width(h)?;
        let unknown = || vec![None; width];
        let a = |h| self.val(h);
        Ok(match &self.f.expressions[h] {
            E::Literal(l) => literal(*l),
            E::Constant(c) => self.global(self.m.constants[*c].init),
            E::ZeroValue(_) => unknown(),
            E::Compose { components, .. } => components
                .iter()
                .map(|h| a(*h))
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .flatten()
                .collect(),
            E::Splat { size, value } => vec![a(*value)?.first().copied().flatten(); *size as usize],
            E::Swizzle {
                size,
                vector,
                pattern,
            } => {
                let vector = a(*vector)?;
                pattern
                    .iter()
                    .take(*size as usize)
                    .map(|p| vector.get(*p as usize).copied().flatten())
                    .collect()
            }
            E::AccessIndex { base, index } if self.ptr(h).is_none() => {
                let start = match &self.m.types[self.ty(*base)?].inner {
                    T::Struct { members, .. } => {
                        members
                            .iter()
                            .take(*index as usize)
                            .try_fold(0usize, |count, s| {
                                count
                                    .checked_add(lanes(s.ty, self.m)?)
                                    .ok_or_else(|| "member lane offset overflow".to_owned())
                            })?
                    }
                    _ => (*index as usize)
                        .checked_mul(width)
                        .ok_or("element lane offset overflow")?,
                };
                a(*base)?.into_iter().skip(start).take(width).collect()
            }
            E::Load { pointer } => self
                .ptr(*pointer)
                .and_then(|(r, o, n)| {
                    self.state
                        .get(&r)
                        .map(|v| v.iter().skip(o).take(n).copied().collect())
                })
                .unwrap_or_else(unknown),
            E::As { expr, kind, convert } => a(*expr)?
                .into_iter()
                .map(|v| match (v, *kind, *convert) {
                    (Some(L::U32(v)), K::Float, None) => Some(L::F32(f32::from_bits(v))),
                    (Some(L::F32(v)), K::Uint, None) => Some(L::U32(v.to_bits())),
                    (Some(L::I32(v)), K::Float, None) => Some(L::F32(f32::from_bits(v as u32))),
                    (Some(L::F32(v)), K::Float, _) => Some(L::F32(v)),
                    (Some(L::U32(v)), K::Float, Some(_)) => Some(L::F32(v as f32)),
                    (Some(L::I32(v)), K::Float, Some(_)) => Some(L::F32(v as f32)),
                    (Some(L::Bool(v)), K::Float, Some(_)) => Some(L::F32(if v { 1. } else { 0. })),
                    (Some(L::Bool(v)), K::Sint, Some(_)) => Some(L::I32(i32::from(v))),
                    _ => None,
                })
                .collect(),
            E::Unary { op, expr } => a(*expr)?
                .into_iter()
                .map(|v| match (op, v) {
                    (naga::UnaryOperator::Negate, Some(L::F32(v))) => Some(L::F32(-v)),
                    (naga::UnaryOperator::LogicalNot, Some(L::Bool(v))) => Some(L::Bool(!v)),
                    _ => None,
                })
                .collect(),
            E::Binary { op, left, right } => {
                if matches!(self.m.types[self.ty(*left)?].inner, T::Matrix { .. })
                    || matches!(self.m.types[self.ty(*right)?].inner, T::Matrix { .. })
                {
                    return Ok(unknown());
                }
                let values: V = a(*left)?
                    .into_iter()
                    .zip(a(*right)?)
                    .map(|(a, b)| binary(*op, a?, b?))
                    .collect();
                if values.len() == width { values } else { unknown() }
            }
            E::Select {
                condition,
                accept,
                reject,
            } => {
                let c = a(*condition)?;
                let yes = a(*accept)?;
                let no = a(*reject)?;
                yes.into_iter()
                    .zip(no)
                    .enumerate()
                    .map(|(i, (yes, no))| {
                        match truth(c.get(if c.len() == 1 { 0 } else { i }).copied().flatten()) {
                            Some(true) => yes,
                            Some(false) => no,
                            None => {
                                if same(yes, no) {
                                    yes
                                } else {
                                    None
                                }
                            }
                        }
                    })
                    .collect()
            }
            E::Math { fun, arg, arg1, .. } => {
                let x = a(*arg)?;
                let y = arg1.map(a).transpose()?;
                if x.len() != width {
                    return Ok(unknown());
                }
                x.into_iter()
                    .enumerate()
                    .map(|(i, x)| {
                        let x = float(x)?;
                        Some(L::F32(match fun {
                            naga::MathFunction::Abs => x.abs(),
                            naga::MathFunction::Floor => x.floor(),
                            naga::MathFunction::Ceil => x.ceil(),
                            naga::MathFunction::Min => x.min(float(y.as_ref()?.get(i).copied().flatten())?),
                            naga::MathFunction::Max => x.max(float(y.as_ref()?.get(i).copied().flatten())?),
                            _ => return None,
                        }))
                    })
                    .collect()
            }
            _ => unknown(),
        })
    }
    fn global(&self, h: Handle<E>) -> V {
        match &self.m.global_expressions[h] {
            E::Literal(l) => literal(*l),
            E::Compose { components, .. } => components.iter().flat_map(|h| self.global(*h)).collect(),
            _ => vec![],
        }
    }
    fn replace(&mut self, h: Handle<E>, v: &V) -> Result<(), String> {
        if let E::Select {
            condition,
            accept,
            reject,
        } = self.f.expressions[h]
        {
            let c = self.val(condition)?;
            let first = c.first().copied().flatten().and_then(|v| truth(Some(v)));
            if let Some(choice) = first.filter(|b| c.iter().all(|v| truth(*v) == Some(*b))) {
                let chosen = if choice { accept } else { reject };
                self.f.expressions[h] = E::Select {
                    condition,
                    accept: chosen,
                    reject: chosen,
                };
            }
        }
        let Some(v): Option<Vec<L>> = v.iter().copied().collect() else {
            return Ok(());
        };
        if v.is_empty() {
            return Ok(());
        }
        if matches!(self.f.expressions[h], E::Literal(_) | E::Constant(_)) {
            self.pre.insert(h);
            return Ok(());
        }
        let ty = self.ty(h)?;
        // Aggregates keep their constructor shape: a flattened value has no member boundaries.
        // Their scalar/vector children can still fold, and their values still propagate.
        if !matches!(self.m.types[ty].inner, T::Scalar(_) | T::Vector { .. }) {
            return Ok(());
        }
        let result = if let [value] = v.as_slice() {
            E::Literal(*value)
        } else {
            let components = v
                .iter()
                .map(|v| {
                    self.m
                        .global_expressions
                        .append(E::Literal(*v), naga::Span::UNDEFINED)
                })
                .collect();
            let init = self
                .m
                .global_expressions
                .append(E::Compose { ty, components }, naga::Span::UNDEFINED);
            let c = self
                .m
                .constants
                .append(naga::Constant { name: None, ty, init }, naga::Span::UNDEFINED);
            E::Constant(c)
        };
        self.f.expressions[h] = result;
        self.pre.insert(h);
        self.folded = self.folded.checked_add(1).ok_or("fold count overflow")?;
        Ok(())
    }
    fn block(&mut self, block: naga::Block) -> Result<naga::Block, String> {
        let mut out = naga::Block::new();
        for (stmt, span) in block.span_iter() {
            match stmt {
                S::Emit(range) => {
                    for h in range.clone() {
                        let v = self.eval(h)?;
                        self.replace(h, &v)?;
                        *self
                            .values
                            .get_mut(h.index())
                            .ok_or("expression has no value slot")? = v;
                    }
                    let mut first = None;
                    let mut last = None;
                    for h in range.clone() {
                        if self.pre.contains(&h) {
                            if let (Some(a), Some(b)) = (first.take(), last.take()) {
                                out.push(S::Emit(naga::Range::new_from_bounds(a, b)), *span);
                            }
                        } else {
                            first.get_or_insert(h);
                            last = Some(h);
                        }
                    }
                    if let (Some(a), Some(b)) = (first, last) {
                        out.push(S::Emit(naga::Range::new_from_bounds(a, b)), *span);
                    }
                }
                S::Store { pointer, value } => {
                    if let Some((r, o, n)) = self.ptr(*pointer)
                        && matches!(r, Root::Local(_))
                    {
                        let v = self.val(*value)?;
                        if let Some(state) = self.state.get_mut(&r) {
                            for (i, v) in v.into_iter().take(n).enumerate() {
                                let at = o.checked_add(i).ok_or("local store lane offset overflow")?;
                                *state.get_mut(at).ok_or("local store outside value state")? = v;
                            }
                        }
                    }
                    out.push(stmt.clone(), *span);
                }
                S::If {
                    condition,
                    accept,
                    reject,
                } => match truth(self.val(*condition)?.first().copied().flatten()) {
                    Some(b) => {
                        out.extend_block(self.block(if b { accept.clone() } else { reject.clone() })?)
                    }
                    None => {
                        let start = self.state.clone();
                        let a = self.block(accept.clone())?;
                        let yes = self.state.clone();
                        self.state = start;
                        let b = self.block(reject.clone())?;
                        for (r, v) in &mut self.state {
                            if let Some(y) = yes.get(r) {
                                for (v, y) in v.iter_mut().zip(y) {
                                    if !same(*v, *y) {
                                        *v = None;
                                    }
                                }
                            }
                        }
                        out.push(
                            S::If {
                                condition: *condition,
                                accept: a,
                                reject: b,
                            },
                            *span,
                        );
                    }
                },
                S::Block(b) => {
                    let b = self.block(b.clone())?;
                    out.push(S::Block(b), *span);
                }
                S::Loop { .. }
                | S::Switch { .. }
                | S::Call { .. }
                | S::Atomic { .. }
                | S::ImageStore { .. } => {
                    return Err(
                        "specialisation does not yet transform loops, switches, calls or memory writes"
                            .into(),
                    );
                }
                _ => out.push(stmt.clone(), *span),
            }
        }
        Ok(out)
    }
}
/// The subset of stated words that decides a branch or a select without any runtime input.
/// Values mixed with vertex inputs, texture samples, unknown words or loop state are not frozen.
/// Ordinary colours and scales remain uniforms unless the program itself uses them as a fully
/// stated feature decision. This keeps specialisations tied to feature sets rather than data values.
pub fn feature_words(source: &Program, known: &Words) -> Result<Words, String> {
    use crate::flow::{At, Node, NodeId, Op, Values};
    fn dependencies(
        values: &Values,
        id: NodeId,
        known: &Words,
        cache: &mut BTreeMap<NodeId, Option<BTreeSet<(ResourceId, u32)>>>,
    ) -> Option<BTreeSet<(ResourceId, u32)>> {
        if let Some(v) = cache.get(&id) {
            return v.clone();
        }
        let result = match values.node(id) {
            Node::Literal(_) => Some(BTreeSet::new()),
            Node::Global {
                resource: Some(resource),
                at: At::Component(word),
            } => {
                let key = (*resource, word.checked_mul(4)?);
                known.contains_key(&key).then(|| BTreeSet::from([key]))
            }
            Node::Op { operands, .. } => {
                let mut out = BTreeSet::new();
                for operand in operands {
                    out.extend(dependencies(values, *operand, known, cache)?);
                }
                Some(out)
            }
            _ => None,
        };
        cache.insert(id, result.clone());
        result
    }
    let values = Values::of_program(source)?;
    let mut keys = BTreeSet::new();
    let mut cache = BTreeMap::new();
    for id in values.reached_from(values.roots()) {
        match values.node(id) {
            Node::Op {
                op: Op::Select,
                operands,
            } => {
                if let Some(condition) = operands.first()
                    && let Some(found) = dependencies(&values, *condition, known, &mut cache)
                {
                    keys.extend(found);
                }
            }
            Node::Merge { condition, .. } => {
                for condition in condition {
                    if let Some(found) = dependencies(&values, *condition, known, &mut cache) {
                        keys.extend(found);
                    }
                }
            }
            _ => {}
        }
    }
    Ok(known
        .iter()
        .filter(|(key, _)| keys.contains(key))
        .map(|(k, v)| (*k, *v))
        .collect())
}

/// Fold only `words`, retaining all unknown runtime values. Unsupported constructs return a refusal.
/// A result is validated before it is returned. The original bytes, interface and lift are not changed.
pub fn specialise(source: &Program, words: &Words) -> Result<Specialised, String> {
    let info = source
        .validate()
        .map_err(|p| format!("invalid source program: {p:?}"))?;
    if source.module.entry_points.len() != 1 || !source.module.functions.is_empty() {
        return Err("specialisation needs one entry point and no helper functions".into());
    }
    for &(id, offset) in words.keys() {
        let Some(resource) = source.interface.resource(id) else {
            return Err(format!("known word names absent resource {}", id.0));
        };
        let ResourceKind::ConstantBuffer { layout } = &resource.kind else {
            return Err("known word is not a uniform word".into());
        };
        if offset % 4 != 0 || offset.checked_add(4).is_none_or(|end| end > layout.size) {
            return Err(format!("known word {}+{} outside uniform buffer", id.0, offset));
        }
    }
    let selected = feature_words(source, words)?;
    let words = &selected;
    let mut program = source.clone();
    let f = std::mem::take(
        &mut program
            .module
            .entry_points
            .first_mut()
            .ok_or("missing entry point")?
            .function,
    );
    let types = f
        .expressions
        .iter()
        .map(|(h, _)| match &info.get_entry_point(0)[h].ty {
            naga::proc::TypeResolution::Handle(t) => *t,
            naga::proc::TypeResolution::Value(t) => program.module.types.insert(
                naga::Type {
                    name: None,
                    inner: t.clone(),
                },
                naga::Span::UNDEFINED,
            ),
        })
        .collect::<Vec<_>>();
    let mut state = BTreeMap::new();
    for (h, l) in f.local_variables.iter() {
        state.insert(Root::Local(h), vec![None; lanes(l.ty, &program.module)?]);
    }
    for (resource, global) in &program.bindings {
        if !program
            .interface
            .resource(*resource)
            .is_some_and(|r| matches!(r.kind, ResourceKind::ConstantBuffer { .. }))
        {
            continue;
        }
        if words.keys().any(|(id, _)| id == resource) {
            let ty = program.module.global_variables[*global].ty;
            let raw_rows = match program.module.types[ty].inner {
                T::Array { base, stride: 16, .. } => matches!(
                    program.module.types[base].inner,
                    T::Vector {
                        size: naga::VectorSize::Quad,
                        scalar: naga::Scalar {
                            kind: K::Uint,
                            width: 4
                        }
                    }
                ),
                _ => false,
            };
            if !raw_rows {
                return Err("feature uniform layout is not an array of raw four-word rows".into());
            }
        }
        let mut v = vec![None; lanes(program.module.global_variables[*global].ty, &program.module)?];
        for (&(id, offset), &word) in words {
            if id == *resource
                && offset % 4 == 0
                && let Some(at) = v.get_mut((offset / 4) as usize)
            {
                *at = Some(L::U32(word));
            }
        }
        state.insert(Root::Global(*global), v);
    }
    let values = types
        .iter()
        .map(|t| Ok(vec![None; lanes(*t, &program.module)?]))
        .collect::<Result<Vec<_>, String>>()?;
    let mut fold = Fold {
        m: &mut program.module,
        f,
        types,
        values,
        state,
        folded: 0,
        pre: BTreeSet::new(),
    };
    for (h, expr) in fold.f.expressions.iter() {
        if matches!(expr, E::Literal(_) | E::Constant(_)) {
            let value = fold.eval(h)?;
            *fold
                .values
                .get_mut(h.index())
                .ok_or("expression has no value slot")? = value;
            fold.pre.insert(h);
        }
    }
    let body = std::mem::take(&mut fold.f.body);
    fold.f.body = fold.block(body)?;
    let mut live = Live {
        fold: &fold,
        expressions: BTreeMap::new(),
        locals: BTreeMap::new(),
    };
    let body = live.block(fold.f.body.clone())?;
    let folded = fold.folded;
    fold.f.body = body;
    fold.f.named_expressions.clear();
    program
        .module
        .entry_points
        .first_mut()
        .ok_or("missing entry point")?
        .function = fold.f;
    let names = program
        .bindings
        .iter()
        .map(|(id, h)| {
            let name = format!("specialise_resource_{}", id.0);
            program.module.global_variables[*h].name = Some(name.clone());
            (*id, name)
        })
        .collect::<Vec<_>>();
    naga::compact::compact(&mut program.module, naga::compact::KeepUnused::No);
    program.bindings = names
        .iter()
        .filter_map(|(id, name)| {
            program
                .module
                .global_variables
                .iter()
                .find(|(_, g)| g.name.as_ref() == Some(name))
                .map(|(h, _)| (*id, h))
        })
        .collect();
    let retained = program
        .bindings
        .iter()
        .map(|(id, _)| *id)
        .collect::<BTreeSet<_>>();
    let unread = program
        .interface
        .resources
        .iter()
        .filter(|r| !retained.contains(&r.id))
        .map(|r| r.id)
        .collect();
    program.interface.resources.retain(|r| retained.contains(&r.id));
    program
        .validate()
        .map_err(|p| format!("specialised program invalid: {p:?}"))?;
    Ok(Specialised {
        program,
        words: words.clone(),
        unread,
        folded,
    })
}

struct Live<'a, 'b> {
    fold: &'a Fold<'b>,
    expressions: BTreeMap<Handle<E>, BTreeSet<usize>>,
    locals: BTreeMap<Root, BTreeSet<usize>>,
}
impl Live<'_, '_> {
    fn need(&mut self, h: Handle<E>, lanes: impl IntoIterator<Item = usize>) {
        self.expressions.entry(h).or_default().extend(lanes);
    }
    fn all(&mut self, h: Handle<E>) -> Result<(), String> {
        self.need(h, 0..self.fold.width(h)?);
        Ok(())
    }
    fn expression(&mut self, h: Handle<E>, wanted: &BTreeSet<usize>) -> Result<(), String> {
        match self.fold.f.expressions[h].clone() {
            E::Literal(_)
            | E::Constant(_)
            | E::ZeroValue(_)
            | E::FunctionArgument(_)
            | E::GlobalVariable(_)
            | E::LocalVariable(_) => {}
            E::Load { pointer } => {
                if self.fold.ptr(pointer).is_none()
                    && !matches!(self.fold.root(pointer), Some(Root::Global(_)))
                {
                    return Err("dynamic or unknown local load address in specialisation".into());
                }
                if let Some((root, start, _)) = self.fold.ptr(pointer)
                    && matches!(root, Root::Local(_))
                {
                    self.locals.entry(root).or_default().extend(
                        wanted
                            .iter()
                            .map(|i| start.checked_add(*i).ok_or("local load lane offset overflow"))
                            .collect::<Result<Vec<_>, _>>()?,
                    );
                }
                self.all(pointer)?;
            }
            E::AccessIndex { base, index } => {
                if self.fold.ptr(h).is_none() {
                    let t = self.fold.ty(base)?;
                    let start = match &self.fold.m.types[t].inner {
                        T::Struct { members, .. } => {
                            members.iter().take(index as usize).try_fold(0usize, |count, s| {
                                count
                                    .checked_add(lanes(s.ty, self.fold.m)?)
                                    .ok_or_else(|| "member lane offset overflow".to_owned())
                            })?
                        }
                        _ => (index as usize)
                            .checked_mul(self.fold.width(h)?)
                            .ok_or("element lane offset overflow")?,
                    };
                    self.need(
                        base,
                        wanted
                            .iter()
                            .map(|i| start.checked_add(*i).ok_or("element lane offset overflow"))
                            .collect::<Result<Vec<_>, _>>()?,
                    );
                } else {
                    self.all(base)?;
                }
            }
            E::Compose { components, .. } => {
                let mut offset = 0;
                for c in components {
                    let n = self.fold.width(c)?;
                    self.need(
                        c,
                        wanted
                            .iter()
                            .filter_map(|i| i.checked_sub(offset).filter(|j| *j < n)),
                    );
                    offset = offset.checked_add(n).ok_or("composite lane offset overflow")?;
                }
            }
            E::Splat { value, .. } => self.need(value, [0]),
            E::Swizzle { vector, pattern, .. } => self.need(
                vector,
                wanted.iter().filter_map(|i| pattern.get(*i).map(|p| *p as usize)),
            ),
            E::Unary { expr, .. } | E::As { expr, .. } | E::Derivative { expr, .. } => {
                self.need(expr, wanted.iter().copied())
            }
            E::Binary {
                op: naga::BinaryOperator::Multiply,
                left,
                right,
            } => {
                self.all(left)?;
                self.all(right)?;
            }
            E::Binary { left, right, .. } => {
                self.need(left, wanted.iter().copied());
                self.need(right, wanted.iter().copied());
            }
            E::Select {
                condition,
                accept,
                reject,
            } => {
                self.all(condition)?;
                self.need(accept, wanted.iter().copied());
                self.need(reject, wanted.iter().copied());
            }
            E::Math {
                arg,
                arg1,
                arg2,
                arg3,
                ..
            } => {
                for a in [Some(arg), arg1, arg2, arg3].into_iter().flatten() {
                    self.all(a)?;
                }
            }
            E::Relational { argument, .. } => self.all(argument)?,
            E::ImageSample {
                image,
                sampler,
                coordinate,
                array_index,
                offset,
                level,
                depth_ref,
                gather: _,
                clamp_to_edge: _,
            } => {
                for a in [
                    Some(image),
                    Some(sampler),
                    Some(coordinate),
                    array_index,
                    offset,
                    depth_ref,
                ]
                .into_iter()
                .flatten()
                {
                    self.all(a)?;
                }
                match level {
                    naga::SampleLevel::Exact(h) | naga::SampleLevel::Bias(h) => self.all(h)?,
                    naga::SampleLevel::Gradient { x, y } => {
                        self.all(x)?;
                        self.all(y)?;
                    }
                    _ => {}
                }
            }
            E::Access { base, index } => {
                self.all(base)?;
                self.all(index)?;
            }
            _ => {
                return Err("specialisation dead-code analysis encountered an unsupported expression".into());
            }
        }
        Ok(())
    }
    fn block(&mut self, block: naga::Block) -> Result<naga::Block, String> {
        let mut result = Vec::new();
        for (stmt, span) in block.span_iter().collect::<Vec<_>>().into_iter().rev() {
            match stmt {
                S::Emit(range) => {
                    let mut used = false;
                    for h in range.clone().collect::<Vec<_>>().into_iter().rev() {
                        if let Some(wanted) = self.expressions.remove(&h) {
                            used = true;
                            self.expression(h, &wanted)?;
                        }
                    }
                    if used {
                        result.push((stmt.clone(), *span));
                    }
                }
                S::Return { value } => {
                    if let Some(h) = value {
                        self.all(*h)?;
                    }
                    result.push((stmt.clone(), *span));
                }
                S::Store { pointer, value } => {
                    let Some((root, start, count)) = self.fold.ptr(*pointer) else {
                        return Err("unknown store address in specialisation".into());
                    };
                    if matches!(root, Root::Global(_)) {
                        return Err("global store in specialisation".into());
                    }
                    let end = start.checked_add(count).ok_or("store lane range overflow")?;
                    let wanted = self
                        .locals
                        .get(&root)
                        .map(|l| l.range(start..end).copied().collect::<Vec<_>>())
                        .unwrap_or_default();
                    if !wanted.is_empty() {
                        if let Some(l) = self.locals.get_mut(&root) {
                            for i in &wanted {
                                l.remove(i);
                            }
                        }
                        self.need(
                            *value,
                            wanted
                                .iter()
                                .map(|i| i.checked_sub(start).ok_or("store lane precedes its value"))
                                .collect::<Result<Vec<_>, _>>()?,
                        );
                        result.push((stmt.clone(), *span));
                    }
                }
                S::If {
                    condition,
                    accept,
                    reject,
                } => {
                    let locals = self.locals.clone();
                    let expressions = self.expressions.clone();
                    let a = self.block(accept.clone())?;
                    let al = self.locals.clone();
                    let ae = self.expressions.clone();
                    self.locals = locals;
                    self.expressions = expressions;
                    let b = self.block(reject.clone())?;
                    for (r, live) in al {
                        self.locals.entry(r).or_default().extend(live);
                    }
                    for (h, live) in ae {
                        self.expressions.entry(h).or_default().extend(live);
                    }
                    if !a.is_empty() || !b.is_empty() {
                        self.all(*condition)?;
                        result.push((
                            S::If {
                                condition: *condition,
                                accept: a,
                                reject: b,
                            },
                            *span,
                        ));
                    }
                }
                S::Block(b) => {
                    let b = self.block(b.clone())?;
                    result.push((S::Block(b), *span));
                }
                S::Kill => result.push((stmt.clone(), *span)),
                _ => return Err("unsupported statement in specialisation dead-code analysis".into()),
            }
        }
        let mut out = naga::Block::new();
        for (s, p) in result.into_iter().rev() {
            out.push(s, p);
        }
        Ok(out)
    }
}
