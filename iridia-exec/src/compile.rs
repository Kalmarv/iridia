//! Compiling a program into its flat form. The module is walked once: expressions become operations on registers
//! (pointers into locals and globals are folded to word offsets, bitcasts are aliases), statements keep their
//! structure, and an operation nothing reads is dropped.

use std::collections::BTreeMap;

use naga::{
    BinaryOperator as Bo, Expression as E, Handle, ImageClass, ImageQuery, Literal, MathFunction as Mf,
    Module, SampleLevel, ScalarKind, Statement, SwitchValue, TypeInner, UnaryOperator as Uo,
};

use super::flat::{
    AtomOp, BinOp, CvtOp, Executable, GlobalDesc, GlobalKind, LevelOp, Loc, MathOp, Member, Op, Pick, Ptr,
    Reg, Res, St, UnOp, WaveGather, WaveKind, WaveOp, WaveScan,
};
use super::request::{Dim, ImageAtomicOp, QueryKind};
use super::run::Decision;
use iridia::rt as iridia_rt;
use iridia::{Program, ResourceId};

/// The scalar kind of a typed value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    U,
    I,
    F,
    B,
}

// ---------------------------------------------------------------- compile

/// A lane number as a `u8`: a lane is one of at most four.
fn lane_u8(lane: usize) -> u8 {
    u8::try_from(lane).unwrap_or(u8::MAX)
}

/// The most words of local memory a function may need: far more than D3D's 64 KiB of scratch and thread group memory, and
/// what bounds the registers a run allocates (`Scratch`).
const MAX_LOCAL_WORDS: u32 = 1 << 20;

/// `a + b` words of layout, an error naming the layout when it does not fit.
fn add(what: &str, a: u32, b: u32) -> Result<u32, String> {
    a.checked_add(b)
        .ok_or_else(|| format!("{what}: {a} + {b} words overflow 32 bits"))
}

/// `a * b` words of layout, an error naming the layout when it does not fit.
fn mul(what: &str, a: u32, b: u32) -> Result<u32, String> {
    a.checked_mul(b)
        .ok_or_else(|| format!("{what}: {a} * {b} words overflow 32 bits"))
}

/// `local_words` grown by `by`, within [`MAX_LOCAL_WORDS`].
fn grow(local_words: u32, by: u32) -> Result<u32, String> {
    add("local memory", local_words, by)
        .ok()
        .filter(|total| *total <= MAX_LOCAL_WORDS)
        .ok_or_else(|| {
            format!(
                "local memory of {local_words} + {by} words, where a function has at most {MAX_LOCAL_WORDS}"
            )
        })
}

/// The number of global `g` among the globals the flat form keeps.
fn gid(map: &BTreeMap<usize, u32>, g: Handle<naga::GlobalVariable>) -> Result<u32, String> {
    map.get(&g.index())
        .copied()
        .ok_or_else(|| format!("global {} is not memory", g.index()))
}

/// A count of expressions, registers or globals as the `u32` the flat form numbers them by.
fn number(n: usize) -> Result<u32, String> {
    u32::try_from(n).map_err(|_| format!("{n} entries, where the flat form numbers them in 32 bits"))
}

/// A position in one of the tables the compile pass fills (an entry per expression, register, argument or global). The
/// positions come from a validated module's own handles, so a miss is a bug of the pass; it is reported as an error naming
/// the position, never a panic.
trait At<T> {
    fn at(&self, i: usize) -> Result<&T, String>;
    fn at_mut(&mut self, i: usize) -> Result<&mut T, String>;
}
impl<T> At<T> for [T] {
    fn at(&self, i: usize) -> Result<&T, String> {
        self.get(i)
            .ok_or_else(|| format!("compile: position {i} of a table of {}", self.len()))
    }
    fn at_mut(&mut self, i: usize) -> Result<&mut T, String> {
        let len = self.len();
        self.get_mut(i)
            .ok_or_else(|| format!("compile: position {i} of a table of {len}"))
    }
}

/// The counts of reads of each register, which dead-operation removal works from.
trait Uses {
    fn bump(&mut self, i: usize) -> Result<(), String>;
    fn drop_one(&mut self, i: usize) -> Result<(), String>;
}
impl Uses for [u32] {
    fn bump(&mut self, i: usize) -> Result<(), String> {
        let count = self.at_mut(i)?;
        *count = count.saturating_add(1);
        Ok(())
    }
    fn drop_one(&mut self, i: usize) -> Result<(), String> {
        let count = self.at_mut(i)?;
        *count = count.saturating_sub(1);
        Ok(())
    }
}

fn words(m: &Module, ty: Handle<naga::Type>) -> Result<u32, String> {
    Ok(match &m.types[ty].inner {
        TypeInner::Scalar(_) | TypeInner::Atomic(_) => 1,
        TypeInner::Vector { size, .. } => *size as u32,
        TypeInner::Array {
            base,
            size: naga::ArraySize::Constant(n),
            ..
        } => mul("an array", n.get(), words(m, *base)?)?,
        TypeInner::Array { base, .. } => words(m, *base)?,
        TypeInner::Struct { members, .. } => {
            let mut total = 0u32;
            for x in members {
                total = add("a structure", total, words(m, x.ty)?)?;
            }
            total
        }
        t => return Err(format!("memory layout of {t:?}")),
    })
}

fn kind_of(s: ScalarKind) -> Result<Kind, String> {
    Ok(match s {
        ScalarKind::Float => Kind::F,
        ScalarKind::Uint => Kind::U,
        ScalarKind::Sint => Kind::I,
        ScalarKind::Bool => Kind::B,
        k => return Err(format!("scalar kind {k:?}")),
    })
}
#[derive(Clone, Copy, Debug)]
enum Shape {
    V(Kind, u8),
    Pointer,
    Other,
}
fn shape_of(inner: &TypeInner) -> Result<Shape, String> {
    Ok(match inner {
        TypeInner::Scalar(s) => Shape::V(kind_of(s.kind)?, 1),
        TypeInner::Vector { size, scalar } => Shape::V(kind_of(scalar.kind)?, *size as u8),
        TypeInner::Pointer { .. } | TypeInner::ValuePointer { .. } => Shape::Pointer,
        _ => Shape::Other,
    })
}

/// Compile the entry point of `p`, or when `function` names one, that function of a program made of functions.
pub(super) fn compile(
    p: &Program,
    info: &naga::valid::ModuleInfo,
    function: Option<&str>,
) -> Result<Executable, String> {
    let m = &p.module;
    let ep = match function {
        None => Some(m.entry_points.first().ok_or("no entry point")?),
        Some(_) => None,
    };
    // each global that is a resource: its id and, for an array, how many elements the interface says it has
    let resource_of: BTreeMap<usize, (ResourceId, u32)> = p
        .bindings
        .iter()
        .map(|(id, h)| {
            let count = p
                .interface
                .resources
                .iter()
                .find(|r| r.id == *id)
                .map_or(1, |r| r.count);
            (h.index(), (*id, count))
        })
        .collect();
    let (f, finfo) = match (ep, function) {
        (Some(ep), _) => (&ep.function, info.get_entry_point(0)),
        (None, name) => {
            let name = name.unwrap_or_default();
            let (h, func) = m
                .functions
                .iter()
                .find(|(_, func)| func.name.as_deref() == Some(name))
                .ok_or_else(|| format!("no function {name}"))?;
            (func, &info[h])
        }
    };
    let n_expr = f.expressions.len();
    let inner = |h: Handle<E>| -> &TypeInner { finfo[h].ty.inner_with(&m.types) };
    let shape = |h: Handle<E>| shape_of(inner(h));

    // locals
    let mut local_off: Vec<u32> = Vec::new();
    let mut local_words = 0u32;
    // a ray query is state the run keeps apart from local memory: its number
    let mut rq_local: BTreeMap<usize, u32> = BTreeMap::new();
    for (h, l) in f.local_variables.iter() {
        local_off.push(local_words);
        if matches!(m.types[l.ty].inner, TypeInner::RayQuery { .. }) {
            rq_local.insert(h.index(), number(rq_local.len())?);
            continue;
        }
        local_words = grow(local_words, words(m, l.ty)?)?;
    }
    // arguments that are arrays (a stage function's control points) live in local memory, one element after another;
    // the caller passes each element as a register of its own
    let mut array_arg: Vec<Option<(u32, u32, u8)>> = Vec::new();
    // pointers to structures of function memory (a ray tracing program's payload and attributes): local words, count
    let mut ptr_arg: Vec<Option<(u32, u32)>> = Vec::new();
    let mut arg_at: Vec<usize> = Vec::new();
    let mut arg_len = 0usize;
    for a in f.arguments.iter() {
        arg_at.push(arg_len);
        match &m.types[a.ty].inner {
            TypeInner::Array {
                base,
                size: naga::ArraySize::Constant(n),
                ..
            } => {
                let width = words(m, *base)?;
                if width > 4
                    || !matches!(
                        m.types[*base].inner,
                        TypeInner::Scalar(_) | TypeInner::Vector { .. }
                    )
                {
                    return Err("an array argument of anything but scalars and vectors".into());
                }
                ptr_arg.push(None);
                array_arg.push(Some((local_words, n.get(), lane_u8(width as usize))));
                local_words = grow(local_words, mul("an array argument", n.get(), width)?)?;
                arg_len = arg_len.saturating_add(n.get() as usize);
            }
            TypeInner::Pointer {
                base,
                space: naga::AddressSpace::Function,
            } => {
                let w = words(m, *base)?;
                ptr_arg.push(Some((local_words, w)));
                array_arg.push(None);
                local_words = grow(local_words, w)?;
                arg_len = arg_len.saturating_add(w.div_ceil(4) as usize);
            }
            _ => {
                ptr_arg.push(None);
                array_arg.push(None);
                arg_len = arg_len.saturating_add(1);
            }
        }
    }
    // globals
    let mut globals: Vec<GlobalDesc> = Vec::new();
    let mut gid_of: BTreeMap<usize, u32> = BTreeMap::new();
    for (h, g) in m.global_variables.iter() {
        let kind = match g.space {
            naga::AddressSpace::Uniform => Some(GlobalKind::Constants(
                resource_of.get(&h.index()).ok_or("unbound uniform")?.0,
            )),
            naga::AddressSpace::Private => {
                let init = g.init.ok_or("private global without initialiser")?;
                Some(GlobalKind::Private(const_words(m, init)?))
            }
            naga::AddressSpace::Storage { .. } => {
                let &(id, count) = resource_of.get(&h.index()).ok_or("unbound storage")?;
                Some(match &m.types[g.ty].inner {
                    TypeInner::BindingArray { .. } => GlobalKind::StorageArray(id, count),
                    _ => GlobalKind::Storage(id),
                })
            }
            naga::AddressSpace::WorkGroup => Some(GlobalKind::Workgroup(words(m, g.ty)?)),
            naga::AddressSpace::Handle => None,
            s => return Err(format!("address space {s:?}")),
        };
        if let Some(kind) = kind {
            let elem_words = match &m.types[g.ty].inner {
                TypeInner::Array { base, .. } => words(m, *base)?,
                // an array of buffers: the words of one element of the array each holds
                TypeInner::BindingArray { base, .. } => match &m.types[*base].inner {
                    TypeInner::Struct { members, .. } => match members.last().map(|x| &m.types[x.ty].inner) {
                        Some(TypeInner::Array { base, .. }) => words(m, *base)?,
                        _ => return Err("an array of buffers whose elements are not a runtime array".into()),
                    },
                    TypeInner::Array { base, .. } => words(m, *base)?,
                    _ => return Err("an array of buffers of anything but structures".into()),
                },
                _ => 1,
            };
            gid_of.insert(h.index(), number(globals.len())?);
            globals.push(GlobalDesc { kind, elem_words });
        }
    }

    let mut slot: Vec<u32> = (0..number(n_expr)?).collect();
    let mut n_slots = n_expr;
    let mut op_of: Vec<Option<Op>> = vec![None; n_expr];
    let mut ptr_of: Vec<Option<Ptr>> = vec![None; n_expr];
    let mut agg_of: Vec<Option<Vec<(u32, u8)>>> = vec![None; n_expr];
    // the ray query a local-variable expression names
    let mut rq_of: Vec<Option<u32>> = vec![None; n_expr];
    // the columns of a matrix member, by the slot that stands for the matrix
    let mut nested_of: BTreeMap<u32, Vec<(u32, u8)>> = BTreeMap::new();
    // a whole array of a local, read as a value (words, element count, element width)
    let mut mem_agg: Vec<Option<(u32, u32, u8)>> = vec![None; n_expr];
    // the register of the copy that holds a loaded array, which keeps the copy alive
    let mut mem_keep: Vec<Option<u32>> = vec![None; n_expr];
    // the members of a structure that has array members kept in memory
    let mut members_of: Vec<Option<Vec<Member>>> = vec![None; n_expr];
    let mut inits: Vec<(u32, Reg)> = Vec::new();
    let mut args: Vec<(u32, u8)> = Vec::new();
    // static load per expression (for fusion): (loc, n)
    let mut static_load: Vec<Option<(Loc, u8)>> = vec![None; n_expr];

    let lit = |l: &Literal| -> Result<u32, String> {
        Ok(match l {
            Literal::F32(x) => x.to_bits(),
            Literal::U32(x) => *x,
            Literal::I32(x) => *x as u32,
            Literal::Bool(x) => *x as u32,
            l => return Err(format!("literal {l:?}")),
        })
    };
    let add_loc = |loc: Loc, d: u32| -> Result<Loc, String> {
        Ok(match loc {
            Loc::Local(o) => Loc::Local(add("a pointer", o, d)?),
            Loc::Global(g, o) => Loc::Global(g, add("a pointer", o, d)?),
            Loc::Elem(g, o, p) => Loc::Elem(g, add("a pointer", o, d)?, p),
        })
    };
    // provenance: each value lane's source (argument index, lane)
    type Prov = [Option<(u32, u8)>; 4];
    let compute_prov = |local_src: &BTreeMap<u32, (u32, u8)>,
                        slot: &Vec<u32>,
                        static_load: &Vec<Option<(Loc, u8)>>|
     -> Vec<Option<Prov>> {
        let mut prov: Vec<Option<Prov>> = vec![None; n_expr];
        for (h, e) in f.expressions.iter() {
            let i = h.index();
            // a register past the expressions' own (an aggregate's member) carries no provenance
            let get = |x: Handle<E>| -> Option<Prov> {
                slot.get(x.index())
                    .and_then(|&s| prov.get(s as usize))
                    .copied()
                    .flatten()
            };
            let derived = match e {
                E::FunctionArgument(a) => {
                    let n = match shape(h) {
                        Ok(Shape::V(_, n)) => n,
                        _ => 0,
                    };
                    let mut p: Prov = [None; 4];
                    for (l, slot) in p.iter_mut().enumerate().take(n as usize) {
                        *slot = Some((*a, lane_u8(l)));
                    }
                    Some(p)
                }
                E::As {
                    expr, convert: None, ..
                } => get(*expr),
                E::Swizzle {
                    size,
                    vector,
                    pattern,
                } => get(*vector).map(|s| {
                    let mut p: Prov = [None; 4];
                    for (dst, &lane) in p.iter_mut().zip(pattern.iter()).take(*size as usize) {
                        *dst = s.get(lane as usize).copied().flatten();
                    }
                    p
                }),
                E::AccessIndex { base, index } => get(*base)
                    .and_then(|s| s.get(*index as usize).copied().flatten())
                    .map(|x| [Some(x), None, None, None]),
                E::Load { .. } => static_load.get(i).copied().flatten().and_then(|(loc, n)| {
                    let Loc::Local(o) = loc else { return None };
                    let mut p: Prov = [None; 4];
                    for (l, slot) in p.iter_mut().enumerate().take(n as usize) {
                        *slot = local_src.get(&o.saturating_add(u32::from(lane_u8(l)))).copied();
                    }
                    Some(p)
                }),
                _ => None,
            };
            if let Some(entry) = prov.get_mut(i) {
                *entry = derived;
            }
        }
        prov
    };

    // ---- pass 1: static shapes, pointers, aliases and operations
    for (h, e) in f.expressions.iter() {
        let i = h.index();
        let sh = shape(h)?;
        let nlanes = |x: Handle<E>| -> u8 {
            match shape(x) {
                Ok(Shape::V(_, n)) => n,
                _ => 1,
            }
        };
        let kind = |x: Handle<E>| -> Kind {
            match shape(x) {
                Ok(Shape::V(k, _)) => k,
                _ => Kind::U,
            }
        };
        match e {
            E::Literal(l) => {
                inits.push((number(i)?, [lit(l)?; 4]));
            }
            E::ZeroValue(_) => {
                if let Shape::V(..) = sh {
                    inits.push((number(i)?, [0; 4]));
                } else {
                    return Err("aggregate zero value".into());
                }
            }
            E::Compose { components, .. } => match sh {
                Shape::V(_, n) => {
                    let mut parts = [(0u32, 0u8); 4];
                    if components.len() > parts.len() {
                        return Err(format!("a vector composed of {} parts", components.len()));
                    }
                    for (part, c) in parts.iter_mut().zip(components) {
                        *part = (*slot.at(c.index())?, nlanes(*c));
                    }
                    let _ = n;
                    *op_of.at_mut(i)? = Some(Op::Compose {
                        parts,
                        np: lane_u8(components.len()),
                    });
                }
                _ => {
                    // a member that is itself an aggregate (an array, a structure) contributes its members in
                    // order, so a returned structure is the flat list of its vectors and scalars
                    let mut members = Vec::new();
                    for c in components {
                        match agg_of.at(c.index())? {
                            Some(inner) => members.extend(inner.iter().copied()),
                            None => members.push((*slot.at(c.index())?, nlanes(*c))),
                        }
                    }
                    *agg_of.at_mut(i)? = Some(members);
                    if components
                        .iter()
                        .any(|c| mem_agg.get(c.index()).is_some_and(Option::is_some))
                    {
                        let mut list = Vec::new();
                        for c in components {
                            match (mem_agg.at(c.index())?, agg_of.at(c.index())?) {
                                (&Some((word, count, width)), _) => list.push(Member::Mem {
                                    word,
                                    count,
                                    width,
                                    keep: *mem_keep.at(c.index())?,
                                }),
                                (None, Some(inner)) => {
                                    list.extend(inner.iter().map(|&(s, n)| Member::Reg(s, n)))
                                }
                                (None, None) => list.push(Member::Reg(*slot.at(c.index())?, nlanes(*c))),
                            }
                        }
                        *members_of.at_mut(i)? = Some(list);
                    }
                }
            },
            E::Splat { size, value } => {
                *op_of.at_mut(i)? = Some(Op::Splat {
                    src: *slot.at(value.index())?,
                    n: *size as u8,
                })
            }
            E::Swizzle {
                size,
                vector,
                pattern,
            } => {
                let mut pat = [0u8; 4];
                for (to, &from) in pat.iter_mut().zip(pattern.iter()).take(*size as usize) {
                    *to = from as u8;
                }
                if let Some((loc, _)) = *static_load.at(vector.index())? {
                    *op_of.at_mut(i)? = Some(Op::LoadLanes {
                        loc,
                        offs: pat,
                        n: *size as u8,
                    });
                    *static_load.at_mut(i)? = None;
                } else {
                    *op_of.at_mut(i)? = Some(Op::Swizzle {
                        src: *slot.at(vector.index())?,
                        pat,
                        n: *size as u8,
                    });
                }
            }
            // an element of an array of buffers: where its words are
            E::Access { base, index } if is_buffer_array(f, &gid_of, &globals, *base) => {
                let E::GlobalVariable(g) = f.expressions[*base] else {
                    return Err("an array of buffers that is not a global".into());
                };
                *ptr_of.at_mut(i)? = Some(Ptr {
                    loc: Loc::Elem(gid(&gid_of, g)?, 0, Pick::Slot(*slot.at(index.index())?)),
                    dynamic: None,
                });
            }
            E::AccessIndex { base, index } if is_buffer_array(f, &gid_of, &globals, *base) => {
                let E::GlobalVariable(g) = f.expressions[*base] else {
                    return Err("an array of buffers that is not a global".into());
                };
                *ptr_of.at_mut(i)? = Some(Ptr {
                    loc: Loc::Elem(gid(&gid_of, g)?, 0, Pick::Fixed(*index)),
                    dynamic: None,
                });
            }
            // an element of a resource array is named by the operation that reads it
            E::Access { base, .. } | E::AccessIndex { base, .. } if is_binding_array(m, f, *base) => {}
            // an element of an array argument: a read of its words
            E::AccessIndex { base, index } if array_of(f, &array_arg, *base).is_some() => {
                let (off, count, width) = array_of(f, &array_arg, *base).expect("checked");
                if *index >= count {
                    return Err("an array argument indexed past its end".into());
                }
                let loc = Loc::Local(add(
                    "an array argument",
                    off,
                    mul("an array argument", *index, u32::from(width))?,
                )?);
                *op_of.at_mut(i)? = Some(Op::Load { loc, n: width });
                *static_load.at_mut(i)? = Some((loc, width));
            }
            E::Access { base, index } if array_of(f, &array_arg, *base).is_some() => {
                let (off, count, width) = array_of(f, &array_arg, *base).expect("checked");
                *op_of.at_mut(i)? = Some(Op::LoadDyn {
                    ptr: Ptr {
                        loc: Loc::Local(off),
                        dynamic: Some((*slot.at(index.index())?, u32::from(width), count)),
                    },
                    n: width,
                });
            }
            E::AccessIndex { base, index } => {
                if let Some(p) = *ptr_of.at(base.index())? {
                    // pointer arithmetic: the pointee decides the step
                    let step = match inner(*base) {
                        TypeInner::Pointer { base: t, .. } => match &m.types[*t].inner {
                            TypeInner::Array { base, .. } => {
                                mul("a pointer into an array", words(m, *base)?, *index)?
                            }
                            TypeInner::Vector { .. } => *index,
                            TypeInner::Struct { members, .. } => members
                                .iter()
                                .take(*index as usize)
                                .map(|x| words(m, x.ty))
                                .sum::<Result<u32, _>>()?,
                            t => return Err(format!("access into {t:?}")),
                        },
                        TypeInner::ValuePointer { size: Some(_), .. } => *index,
                        t => return Err(format!("access through {t:?}")),
                    };
                    *ptr_of.at_mut(i)? = Some(Ptr {
                        loc: add_loc(p.loc, step)?,
                        dynamic: p.dynamic,
                    });
                } else if let Some(a) = agg_of.at(base.index())? {
                    let (s, _) = *a.get(*index as usize).ok_or("aggregate index")?;
                    *slot.at_mut(i)? = s;
                    if let Some(columns) = nested_of.get(&s) {
                        *agg_of.at_mut(i)? = Some(columns.clone());
                    }
                } else if let Some((loc, _)) = *static_load.at(base.index())? {
                    *op_of.at_mut(i)? = Some(Op::LoadLanes {
                        loc,
                        offs: [lane_u8(*index as usize), 0, 0, 0],
                        n: 1,
                    });
                } else {
                    *op_of.at_mut(i)? = Some(Op::Lane {
                        src: *slot.at(base.index())?,
                        i: lane_u8(*index as usize),
                    });
                }
            }
            E::Access { base, index } => {
                let p = ptr_of.at(base.index())?.ok_or("dynamic index of a value")?;
                if p.dynamic.is_some() {
                    return Err("two dynamic indices".into());
                }
                let (stride, len) = match inner(*base) {
                    TypeInner::Pointer { base: t, .. } => match &m.types[*t].inner {
                        TypeInner::Array { base, size, .. } => (
                            words(m, *base)?,
                            match size {
                                naga::ArraySize::Constant(n) => n.get(),
                                _ => u32::MAX,
                            },
                        ),
                        TypeInner::Vector { size, .. } => (1, *size as u32),
                        t => return Err(format!("dynamic access into {t:?}")),
                    },
                    t => return Err(format!("dynamic access through {t:?}")),
                };
                *ptr_of.at_mut(i)? = Some(Ptr {
                    loc: p.loc,
                    dynamic: Some((*slot.at(index.index())?, stride, len)),
                });
            }
            // read by index, or returned whole
            E::FunctionArgument(a) if ptr_arg.at(*a as usize)?.is_some() => {
                let (off, _) = ptr_arg.at(*a as usize)?.expect("checked");
                *ptr_of.at_mut(i)? = Some(Ptr {
                    loc: Loc::Local(off),
                    dynamic: None,
                });
            }
            E::FunctionArgument(a) if array_arg.at(*a as usize)?.is_some() => {
                *mem_agg.at_mut(i)? = *array_arg.at(*a as usize)?
            }
            // written by the call that makes it
            E::CallResult(_) => {}
            E::FunctionArgument(a) => {
                let Shape::V(_, n) = sh else {
                    return Err("aggregate argument".into());
                };
                if args.len() <= *a as usize {
                    args.resize(*a as usize + 1, (0, 0));
                }
                // an argument may be named by several expressions: they share the first one's register
                if args.at(*a as usize)?.1 == 0 {
                    *args.at_mut(*a as usize)? = (number(i)?, n);
                } else {
                    *slot.at_mut(i)? = args.at(*a as usize)?.0;
                }
            }
            E::GlobalVariable(g) => {
                if let Some(gid) = gid_of.get(&g.index()) {
                    *ptr_of.at_mut(i)? = Some(Ptr {
                        loc: Loc::Global(*gid, 0),
                        dynamic: None,
                    });
                }
            }
            E::LocalVariable(l) if rq_local.contains_key(&l.index()) => {
                *rq_of.at_mut(i)? = rq_local.get(&l.index()).copied();
            }
            E::RayQueryGetIntersection { query, committed } => {
                let q = rq_of
                    .at(query.index())?
                    .ok_or("an intersection of something that is not a ray query")?;
                let first = number(n_slots)?;
                n_slots = n_slots.saturating_add(19);
                let r = |k: u32| first.saturating_add(k);
                let (o2w, w2o) = (
                    (0..4u32).map(|c| (r(c.saturating_add(9)), 3)).collect::<Vec<_>>(),
                    (0..4u32)
                        .map(|c| (r(c.saturating_add(13)), 3))
                        .collect::<Vec<_>>(),
                );
                // the matrices are stood for by registers of their own (17 and 18), not by their first columns'
                nested_of.insert(r(17), o2w);
                nested_of.insert(r(18), w2o);
                *agg_of.at_mut(i)? = Some(vec![
                    (r(0), 1),
                    (r(1), 1),
                    (r(2), 1),
                    (r(3), 1),
                    (r(4), 1),
                    (r(5), 1),
                    (r(6), 1),
                    (r(7), 2),
                    (r(8), 1),
                    (r(17), 0),
                    (r(18), 0),
                ]);
                *op_of.at_mut(i)? = Some(Op::RqHit {
                    q,
                    committed: *committed,
                    first,
                });
            }
            // written by the statement that proceeds the query
            E::RayQueryProceedResult => {}
            E::RayQueryVertexPositions { .. } => {
                return Err("the vertex positions of a ray query's triangle are not carried".into());
            }
            E::LocalVariable(l) => {
                *ptr_of.at_mut(i)? = Some(Ptr {
                    loc: Loc::Local(*local_off.at(l.index())?),
                    dynamic: None,
                })
            }
            // a whole array of a local, for a result that returns it
            E::Load { pointer } if matches!(sh, Shape::Other) => {
                let p = ptr_of.at(pointer.index())?.ok_or("load through a non-pointer")?;
                let (Loc::Local(word), None) = (p.loc, p.dynamic) else {
                    return Err("load of an aggregate that is not a whole local".into());
                };
                let TypeInner::Pointer { base: t, .. } = inner(*pointer) else {
                    return Err("load of an aggregate through a non-pointer".into());
                };
                let TypeInner::Array {
                    base,
                    size: naga::ArraySize::Constant(n),
                    ..
                } = &m.types[*t].inner
                else {
                    return Err("load of an aggregate that is not an array".into());
                };
                let width = words(m, *base)?;
                if width > 4
                    || !matches!(
                        m.types[*base].inner,
                        TypeInner::Scalar(_) | TypeInner::Vector { .. }
                    )
                {
                    return Err("load of an array of anything but scalars and vectors".into());
                }
                // the array as it is now: a store between this load and the return must not change the value
                let snapshot = local_words;
                let copied = mul("an array load", n.get(), width)?;
                local_words = grow(local_words, copied)?;
                *op_of.at_mut(i)? = Some(Op::CopyWords {
                    src: Loc::Local(word),
                    dst: snapshot,
                    words: copied,
                });
                *mem_agg.at_mut(i)? = Some((snapshot, n.get(), lane_u8(width as usize)));
                *mem_keep.at_mut(i)? = Some(*slot.at(i)?);
            }
            E::Load { pointer } => {
                let Shape::V(_, n) = sh else {
                    return Err("load of an aggregate".into());
                };
                let p = ptr_of.at(pointer.index())?.ok_or("load through a non-pointer")?;
                if p.dynamic.is_some() {
                    *op_of.at_mut(i)? = Some(Op::LoadDyn { ptr: p, n });
                } else {
                    *op_of.at_mut(i)? = Some(Op::Load { loc: p.loc, n });
                    *static_load.at_mut(i)? = Some((p.loc, n));
                }
            }
            E::Unary { op, expr } => {
                let n = nlanes(*expr);
                let k = kind(*expr);
                let uo = match (op, k) {
                    (Uo::Negate, Kind::F) => UnOp::NegF,
                    (Uo::Negate, _) => UnOp::NegI,
                    (Uo::LogicalNot, _) => UnOp::Not,
                    (Uo::BitwiseNot, _) => UnOp::BitNot,
                };
                *op_of.at_mut(i)? = Some(Op::Un {
                    op: uo,
                    src: *slot.at(expr.index())?,
                    n,
                });
            }
            E::Binary { op, left, right } => {
                let k = kind(*left);
                let (a1, b1) = (nlanes(*left) == 1, nlanes(*right) == 1);
                let n = nlanes(*left).max(nlanes(*right));
                use BinOp::*;
                let bo = match (op, k) {
                    (Bo::Add, Kind::F) => AddF,
                    (Bo::Subtract, Kind::F) => SubF,
                    (Bo::Multiply, Kind::F) => MulF,
                    (Bo::Divide, Kind::F) => DivF,
                    (Bo::Modulo, Kind::F) => ModF,
                    (Bo::Add, _) => AddU,
                    (Bo::Subtract, _) => SubU,
                    (Bo::Multiply, _) => MulU,
                    (Bo::Divide, Kind::U) => DivU,
                    (Bo::Modulo, Kind::U) => ModU,
                    (Bo::And, _) => And,
                    (Bo::InclusiveOr, _) => Or,
                    (Bo::ExclusiveOr, _) => Xor,
                    (Bo::ShiftLeft, _) => Shl,
                    (Bo::ShiftRight, Kind::I) => ShrI,
                    (Bo::ShiftRight, _) => ShrU,
                    (Bo::Equal, Kind::F) => EqF,
                    (Bo::NotEqual, Kind::F) => NeF,
                    (Bo::Equal, _) => EqW,
                    (Bo::NotEqual, _) => NeW,
                    (Bo::Less, Kind::F) => LtF,
                    (Bo::LessEqual, Kind::F) => LeF,
                    (Bo::Greater, Kind::F) => GtF,
                    (Bo::GreaterEqual, Kind::F) => GeF,
                    (Bo::Less, Kind::U) => LtU,
                    (Bo::LessEqual, Kind::U) => LeU,
                    (Bo::Greater, Kind::U) => GtU,
                    (Bo::GreaterEqual, Kind::U) => GeU,
                    (Bo::Less, Kind::I) => LtI,
                    (Bo::LessEqual, Kind::I) => LeI,
                    (Bo::Greater, Kind::I) => GtI,
                    (Bo::GreaterEqual, Kind::I) => GeI,
                    (Bo::LogicalAnd, _) => LAnd,
                    (Bo::LogicalOr, _) => LOr,
                    (o, k) => return Err(format!("binary {o:?} on {k:?}")),
                };
                *op_of.at_mut(i)? = Some(Op::Bin {
                    op: bo,
                    a: *slot.at(left.index())?,
                    b: *slot.at(right.index())?,
                    n,
                    a1,
                    b1,
                });
            }
            E::Select {
                condition,
                accept,
                reject,
            } => {
                let n = nlanes(*accept);
                *op_of.at_mut(i)? = Some(Op::Select {
                    c: *slot.at(condition.index())?,
                    a: *slot.at(accept.index())?,
                    b: *slot.at(reject.index())?,
                    n,
                    c1: nlanes(*condition) == 1,
                });
            }
            E::As {
                expr,
                kind: tk,
                convert,
            } => match convert {
                None => *slot.at_mut(i)? = *slot.at(expr.index())?,
                Some(_) => {
                    let n = nlanes(*expr);
                    let from = kind(*expr);
                    let co = match (from, tk) {
                        (Kind::F, ScalarKind::Sint) => CvtOp::F2I,
                        (Kind::F, ScalarKind::Uint) => CvtOp::F2U,
                        (Kind::I, ScalarKind::Float) => CvtOp::I2F,
                        (Kind::U, ScalarKind::Float) => CvtOp::U2F,
                        (Kind::B, ScalarKind::Float) => CvtOp::B2F,
                        (Kind::B, ScalarKind::Uint | ScalarKind::Sint) => CvtOp::B2U,
                        (Kind::I, ScalarKind::Uint) | (Kind::U, ScalarKind::Sint) => {
                            *slot.at_mut(i)? = *slot.at(expr.index())?;
                            continue;
                        }
                        (a, b) => return Err(format!("conversion {a:?} to {b:?}")),
                    };
                    *op_of.at_mut(i)? = Some(Op::Cvt {
                        op: co,
                        src: *slot.at(expr.index())?,
                        n,
                    });
                }
            },
            E::Derivative { axis, ctrl, expr } => {
                let n = nlanes(*expr);
                let ax = match axis {
                    naga::DerivativeAxis::X => 0,
                    naga::DerivativeAxis::Y => 1,
                    _ => return Err("width derivative".into()),
                };
                *op_of.at_mut(i)? = Some(Op::Deriv {
                    src: *slot.at(expr.index())?,
                    axis: ax,
                    fine: matches!(ctrl, naga::DerivativeControl::Fine),
                    n,
                    prov: None,
                });
            }
            E::Math {
                fun, arg, arg1, arg2, ..
            } => {
                let k = kind(*arg);
                let n = nlanes(*arg);
                let a = *slot.at(arg.index())?;
                let b = arg1.map_or(Ok(0), |x| slot.at(x.index()).copied())?;
                let c = arg2.map_or(Ok(0), |x| slot.at(x.index()).copied())?;
                let f = match (fun, k) {
                    (Mf::Abs, Kind::F) => MathOp::AbsF,
                    (Mf::Abs, _) => MathOp::AbsI,
                    (Mf::Min, Kind::F) => MathOp::MinF,
                    (Mf::Max, Kind::F) => MathOp::MaxF,
                    (Mf::Min, Kind::U) => MathOp::MinU,
                    (Mf::Max, Kind::U) => MathOp::MaxU,
                    (Mf::Min, Kind::I) => MathOp::MinI,
                    (Mf::Max, Kind::I) => MathOp::MaxI,
                    (Mf::Clamp, Kind::F) => MathOp::ClampF,
                    (Mf::Saturate, _) => MathOp::SatF,
                    (Mf::Floor, _) => MathOp::Floor,
                    (Mf::Ceil, _) => MathOp::Ceil,
                    (Mf::Trunc, _) => MathOp::Trunc,
                    (Mf::Round, _) => MathOp::Round,
                    (Mf::Fract, _) => MathOp::Fract,
                    (Mf::Exp2, _) => MathOp::Exp2,
                    (Mf::Log2, _) => MathOp::Log2,
                    (Mf::Sin, _) => MathOp::Sin,
                    (Mf::Cos, _) => MathOp::Cos,
                    (Mf::QuantizeToF16, _) => MathOp::QuantizeF16,
                    (Mf::Tan, _) => MathOp::Tan,
                    (Mf::Asin, _) => MathOp::Asin,
                    (Mf::Acos, _) => MathOp::Acos,
                    (Mf::Atan, _) => MathOp::Atan,
                    (Mf::Sinh, _) => MathOp::Sinh,
                    (Mf::Cosh, _) => MathOp::Cosh,
                    (Mf::Tanh, _) => MathOp::Tanh,
                    (Mf::Fma, _) => MathOp::Fma,
                    (Mf::Sqrt, _) => MathOp::Sqrt,
                    (Mf::InverseSqrt, _) => MathOp::Rsqrt,
                    (Mf::Dot, _) => MathOp::Dot,
                    (Mf::CountOneBits, _) => MathOp::Popcnt,
                    (Mf::ReverseBits, _) => MathOp::RevBits,
                    (Mf::CountLeadingZeros, _) => MathOp::Clz,
                    (Mf::FirstTrailingBit, _) => MathOp::Ctz,
                    (Mf::Pack2x16float, _) => MathOp::PackHalf,
                    (Mf::Unpack2x16float, _) => MathOp::UnpackHalf,
                    (Mf::Mix, Kind::F) => MathOp::Mix,
                    (Mf::Pow, Kind::F) => MathOp::Pow,
                    (Mf::Normalize, Kind::F) => MathOp::Normalize,
                    (Mf::Sign, Kind::F) => MathOp::SignF,
                    (Mf::Sign, Kind::I) => MathOp::SignI,
                    (f, k) => return Err(format!("math function {f:?} on {k:?}")),
                };
                let c1 = arg2.is_some_and(|x| nlanes(x) == 1) && n > 1;
                *op_of.at_mut(i)? = Some(Op::Math { f, a, b, c, n, c1 });
            }
            E::Relational { fun, argument } => {
                let f = match fun {
                    naga::RelationalFunction::All => MathOp::All,
                    naga::RelationalFunction::Any => MathOp::Any,
                    naga::RelationalFunction::IsNan => MathOp::IsNan,
                    naga::RelationalFunction::IsInf => MathOp::IsInf,
                };
                *op_of.at_mut(i)? = Some(Op::Math {
                    f,
                    a: *slot.at(argument.index())?,
                    b: 0,
                    c: 0,
                    n: nlanes(*argument),
                    c1: false,
                });
            }
            E::ArrayLength(p) => {
                let pp = ptr_of.at(p.index())?.ok_or("arrayLength of a non-pointer")?;
                let (g, pick) = match pp.loc {
                    Loc::Global(g, _) => (g, None),
                    Loc::Elem(g, _, pick) => (g, Some(pick)),
                    Loc::Local(_) => return Err("arrayLength of a local".into()),
                };
                *op_of.at_mut(i)? = Some(Op::ArrLen {
                    gid: g,
                    elem_words: globals.at(g as usize)?.elem_words,
                    pick,
                });
            }
            E::ImageSample {
                image,
                sampler,
                gather,
                coordinate,
                array_index,
                offset,
                level,
                depth_ref,
                clamp_to_edge: _,
            } => {
                let (tex, class, dim, arrayed) = image_of(m, f, &resource_of, &slot, *image)?;
                if matches!(
                    class,
                    ImageClass::Sampled { multi: true, .. }
                        | ImageClass::Depth { multi: true }
                        | ImageClass::Storage { .. }
                ) {
                    return Err("sample of a multisampled or storage image".into());
                }
                let smp = binding_of(m, f, &resource_of, &slot, *sampler)?;
                let lv = match level {
                    SampleLevel::Auto => LevelOp::Auto,
                    SampleLevel::Zero => LevelOp::Zero,
                    SampleLevel::Exact(x) => LevelOp::Exact(*slot.at(x.index())?),
                    SampleLevel::Bias(x) => LevelOp::Bias(*slot.at(x.index())?),
                    SampleLevel::Gradient { x, y } => {
                        LevelOp::Gradient(*slot.at(x.index())?, *slot.at(y.index())?)
                    }
                };
                let layer = if arrayed {
                    array_index.map(|x| slot.at(x.index()).copied()).transpose()?
                } else {
                    None
                };
                let off = match offset {
                    Some(h) => const_ints(f, *h)?,
                    None => [0; 3],
                };
                let g = gather.map(|c| match c {
                    naga::SwizzleComponent::X => 0,
                    naga::SwizzleComponent::Y => 1,
                    naga::SwizzleComponent::Z => 2,
                    naga::SwizzleComponent::W => 3,
                });
                *op_of.at_mut(i)? = Some(Op::Sample {
                    tex,
                    smp,
                    coord: *slot.at(coordinate.index())?,
                    cn: nlanes(*coordinate),
                    layer,
                    level: lv,
                    dim: dim_of(dim),
                    offset: off,
                    gather: g,
                    depth_ref: depth_ref.map(|x| slot.at(x.index()).copied()).transpose()?,
                    depth: matches!(class, ImageClass::Depth { .. }),
                });
            }
            E::ImageLoad {
                image,
                coordinate,
                array_index,
                sample,
                level,
            } => {
                let (tex, class, dim, arrayed) = image_of(m, f, &resource_of, &slot, *image)?;
                *op_of.at_mut(i)? = Some(Op::Texel {
                    tex,
                    coord: *slot.at(coordinate.index())?,
                    cn: nlanes(*coordinate),
                    dim: dim_of(dim),
                    storage: matches!(class, ImageClass::Storage { .. }),
                    layer: if arrayed {
                        array_index.map(|x| slot.at(x.index()).copied()).transpose()?
                    } else {
                        None
                    },
                    sample: sample.map(|x| slot.at(x.index()).copied()).transpose()?,
                    mip: level.map(|x| slot.at(x.index()).copied()).transpose()?,
                });
            }
            E::ImageQuery { image, query } => {
                let (tex, _, dim, _) = image_of(m, f, &resource_of, &slot, *image)?;
                let (kind, level) = match query {
                    ImageQuery::Size { level } => (
                        QueryKind::Size,
                        level.map(|x| slot.at(x.index()).copied()).transpose()?,
                    ),
                    ImageQuery::NumLevels => (QueryKind::Levels, None),
                    ImageQuery::NumLayers => (QueryKind::Layers, None),
                    ImageQuery::NumSamples => (QueryKind::Samples, None),
                };
                *op_of.at_mut(i)? = Some(Op::Query {
                    tex,
                    dim: dim_of(dim),
                    kind,
                    level,
                });
            }
            // written by the atomic statement that names it; a compare-exchange's is the old value and whether it
            // was exchanged
            E::SubgroupBallotResult | E::SubgroupOperationResult { .. } => {}
            E::AtomicResult { comparison, .. } => {
                if *comparison {
                    let extra = n_slots;
                    n_slots = n_slots.saturating_add(1);
                    *agg_of.at_mut(i)? = Some(vec![(*slot.at(i)?, 1), (number(extra)?, 1)]);
                }
            }
            e => return Err(format!("expression {:?}", std::mem::discriminant(e))),
        }
    }

    // ---- statements
    // used-by counts for dead operation removal, and the statement tree
    fn operands(op: &Op) -> Vec<u32> {
        match op {
            Op::Swizzle { src, .. }
            | Op::Splat { src, .. }
            | Op::Lane { src, .. }
            | Op::Un { src, .. }
            | Op::Cvt { src, .. }
            | Op::Deriv { src, .. } => vec![*src],
            Op::Compose { parts, np } => parts.iter().take(*np as usize).map(|p| p.0).collect(),
            Op::LoadDyn { ptr, .. } => ptr
                .dynamic
                .map(|d| d.0)
                .into_iter()
                .chain(ptr.loc.slot())
                .collect(),
            Op::Load { loc, .. } | Op::LoadLanes { loc, .. } => loc.slot().into_iter().collect(),
            Op::ArrLen {
                pick: Some(Pick::Slot(s)),
                ..
            } => vec![*s],
            Op::Bin { a, b, .. } => vec![*a, *b],
            Op::Select { c, a, b, .. } => vec![*c, *a, *b],
            Op::Math { a, b, c, .. } => vec![*a, *b, *c],
            Op::Sample {
                tex,
                smp,
                coord,
                layer,
                level,
                depth_ref,
                ..
            } => {
                let mut v = vec![*coord];
                v.extend(tex.slot());
                v.extend(smp.slot());
                v.extend(*layer);
                v.extend(*depth_ref);
                match level {
                    LevelOp::Exact(x) | LevelOp::Bias(x) => v.push(*x),
                    LevelOp::Gradient(x, y) => v.extend([*x, *y]),
                    _ => {}
                }
                v
            }
            Op::Texel {
                tex,
                coord,
                mip,
                layer,
                sample,
                ..
            } => {
                let mut v = vec![*coord];
                v.extend(tex.slot());
                v.extend(*mip);
                v.extend(*layer);
                v.extend(*sample);
                v
            }
            Op::Query { tex, level, .. } => tex.slot().into_iter().chain(level.iter().copied()).collect(),
            _ => vec![],
        }
    }
    struct Ctx<'a> {
        slot: &'a Vec<u32>,
        op_of: &'a Vec<Option<Op>>,
        ptr_of: &'a Vec<Option<Ptr>>,
        agg_of: &'a Vec<Option<Vec<(u32, u8)>>>,
        members_of: &'a Vec<Option<Vec<Member>>>,
        rq_of: &'a Vec<Option<u32>>,
        functions: &'a naga::Arena<naga::Function>,
        rt_pointer: &'a dyn Fn(Handle<E>) -> Result<(u32, u32), String>,
        exprs: &'a naga::Arena<E>,
        resource_of: &'a BTreeMap<usize, (ResourceId, u32)>,
        shape: &'a dyn Fn(Handle<E>) -> Result<Shape, String>,
        image: &'a dyn Fn(Handle<E>) -> Result<ImageRef, String>,
        ops: Vec<(u32, Op)>,
        uses: Vec<u32>,
    }
    fn block(cx: &mut Ctx, b: &naga::Block) -> Result<Vec<St>, String> {
        let mut out: Vec<St> = Vec::new();
        for s in b.iter() {
            match s {
                Statement::Emit(r) => {
                    let lo = number(cx.ops.len())?;
                    for h in r.clone() {
                        if let Some(op) = cx.op_of.at(h.index())? {
                            let dst = *cx.slot.at(h.index())?;
                            for o in operands(op) {
                                cx.uses.bump(o as usize)?;
                            }
                            // reads the query and writes registers nothing names: never dropped
                            if matches!(op, Op::RqHit { .. }) {
                                cx.uses.bump(dst as usize)?;
                            }
                            cx.ops.push((dst, op.clone()));
                        }
                    }
                    let hi = number(cx.ops.len())?;
                    if hi > lo {
                        out.push(St::Run(lo, hi));
                    }
                }
                Statement::Block(inner) => {
                    let v = block(cx, inner)?;
                    out.extend(v);
                }
                Statement::If {
                    condition,
                    accept,
                    reject,
                } => {
                    let c = *cx.slot.at(condition.index())?;
                    cx.uses.bump(c as usize)?;
                    let a = block(cx, accept)?;
                    let r = block(cx, reject)?;
                    out.push(St::If { c, a, r });
                }
                Statement::Loop {
                    body,
                    continuing,
                    break_if,
                } => {
                    let body = block(cx, body)?;
                    let cont = block(cx, continuing)?;
                    let bi = break_if.map(|x| cx.slot.at(x.index()).copied()).transpose()?;
                    if let Some(x) = bi {
                        cx.uses.bump(x as usize)?;
                    }
                    out.push(St::Loop {
                        body,
                        cont,
                        break_if: bi,
                    });
                }
                Statement::Switch { selector, cases } => {
                    let sel = *cx.slot.at(selector.index())?;
                    cx.uses.bump(sel as usize)?;
                    let mut cs = Vec::new();
                    for c in cases {
                        let v = match c.value {
                            SwitchValue::U32(x) => Some(x),
                            SwitchValue::I32(x) => Some(x as u32),
                            SwitchValue::Default => None,
                        };
                        cs.push((v, c.fall_through, block(cx, &c.body)?));
                    }
                    out.push(St::Switch { sel, cases: cs });
                }
                Statement::Break => out.push(St::Break),
                Statement::Continue => out.push(St::Continue),
                Statement::Kill => out.push(St::Kill),
                Statement::Return { value } => {
                    let mut members: Vec<Member> = Vec::new();
                    if let Some(v) = value {
                        // a structure of members, or one vector or scalar
                        members = match (cx.members_of.at(v.index())?, cx.agg_of.at(v.index())?) {
                            (Some(list), _) => list.clone(),
                            (None, Some(a)) => a.iter().map(|&(s, n)| Member::Reg(s, n)).collect(),
                            (None, None) => match (cx.shape)(*v)? {
                                Shape::V(_, n) => vec![Member::Reg(*cx.slot.at(v.index())?, n)],
                                _ => return Err("return of a non-aggregate".into()),
                            },
                        };
                        for m in &members {
                            match m {
                                Member::Reg(s, _) => cx.uses.bump(*s as usize)?,
                                Member::Mem { keep: Some(s), .. } => cx.uses.bump(*s as usize)?,
                                Member::Mem { .. } => {}
                            }
                        }
                    }
                    out.push(St::Return(members));
                }
                Statement::Call {
                    function,
                    arguments,
                    result,
                } => {
                    let name = cx.functions[*function].name.clone().unwrap_or_default();
                    fn take(cx: &mut Ctx, h: &Handle<E>) -> Result<u32, String> {
                        let s = *cx.slot.at(h.index())?;
                        cx.uses.bump(s as usize)?;
                        Ok(s)
                    }
                    match name.as_str() {
                        iridia_rt::TRACE_RAY => {
                            let &[acceleration, o0, o1, o2, o3, o4, o5, o6, o7, o8, pointer] =
                                arguments.as_slice()
                            else {
                                return Err(format!(
                                    "TraceRay with {} arguments, where it takes 11",
                                    arguments.len()
                                ));
                            };
                            let naga::Expression::GlobalVariable(g) = cx.exprs[acceleration] else {
                                return Err(
                                    "a ray traced into an acceleration structure that is not a global".into(),
                                );
                            };
                            let scene = cx
                                .resource_of
                                .get(&g.index())
                                .ok_or("an unbound acceleration structure")?
                                .0;
                            let mut operands = [0u32; 9];
                            for (to, a) in operands.iter_mut().zip([o0, o1, o2, o3, o4, o5, o6, o7, o8]) {
                                *to = take(cx, &a)?;
                            }
                            let payload = (cx.rt_pointer)(pointer)?;
                            out.push(St::RtTrace {
                                scene,
                                operands,
                                payload,
                            });
                        }
                        iridia_rt::REPORT_HIT => {
                            let &[t_arg, kind_arg, attributes_arg] = arguments.as_slice() else {
                                return Err(format!(
                                    "ReportHit with {} arguments, where it takes 3",
                                    arguments.len()
                                ));
                            };
                            let (t, kind) = (take(cx, &t_arg)?, take(cx, &kind_arg)?);
                            let attributes = (cx.rt_pointer)(attributes_arg)?;
                            out.push(St::RtReport {
                                t,
                                kind,
                                attributes,
                                result: result.map(|r| cx.slot.at(r.index()).copied()).transpose()?,
                            });
                        }
                        iridia_rt::CALL_SHADER => {
                            let &[index_arg, parameter_arg] = arguments.as_slice() else {
                                return Err(format!(
                                    "CallShader with {} arguments, where it takes 2",
                                    arguments.len()
                                ));
                            };
                            let index = take(cx, &index_arg)?;
                            let parameter = (cx.rt_pointer)(parameter_arg)?;
                            out.push(St::RtCall { index, parameter });
                        }
                        iridia_rt::IGNORE_HIT => out.push(St::RtDecide(Decision::Ignore)),
                        iridia_rt::ACCEPT_HIT_AND_END_SEARCH => {
                            out.push(St::RtDecide(Decision::AcceptAndEnd))
                        }
                        other => return Err(format!("a call of the function {other}")),
                    }
                }
                Statement::Store { pointer, value } if cx.agg_of.at(value.index())?.is_some() => {
                    let p = cx
                        .ptr_of
                        .at(pointer.index())?
                        .ok_or("store through a non-pointer")?;
                    if p.dynamic.is_some() || !matches!(p.loc, Loc::Local(_)) {
                        return Err("a store of an aggregate to anything but a whole local".into());
                    }
                    let Some(members) = cx.agg_of.at(value.index())?.clone() else {
                        return Err("a store of an aggregate that has no members".into());
                    };
                    for (s, _) in &members {
                        cx.uses.bump(*s as usize)?;
                    }
                    out.push(St::StoreAgg { loc: p.loc, members });
                }
                Statement::RayQuery { query, fun } => {
                    let q = (*cx.rq_of.at(query.index())?)
                        .ok_or("a ray query statement on something that is not a query")?;
                    match fun {
                        naga::RayQueryFunction::Initialize {
                            acceleration_structure,
                            descriptor,
                        } => {
                            let naga::Expression::GlobalVariable(g) = cx.exprs[*acceleration_structure]
                            else {
                                return Err(
                                    "a ray query on an acceleration structure that is not a global".into()
                                );
                            };
                            let scene = cx
                                .resource_of
                                .get(&g.index())
                                .ok_or("an unbound acceleration structure")?
                                .0;
                            let members = cx
                                .agg_of
                                .at(descriptor.index())?
                                .clone()
                                .ok_or("a ray descriptor that is not a composite")?;
                            let [
                                (flags, _),
                                (mask, _),
                                (tmin, _),
                                (tmax, _),
                                (origin, _),
                                (direction, _),
                            ] = members[..]
                            else {
                                return Err("a ray descriptor without its six fields".into());
                            };
                            for s in [flags, mask, tmin, tmax, origin, direction] {
                                cx.uses.bump(s as usize)?;
                            }
                            out.push(St::RqInit {
                                q,
                                scene,
                                flags,
                                mask,
                                tmin,
                                tmax,
                                origin,
                                direction,
                            });
                        }
                        naga::RayQueryFunction::Proceed { result } => out.push(St::RqProceed {
                            q,
                            result: *cx.slot.at(result.index())?,
                        }),
                        naga::RayQueryFunction::ConfirmIntersection => out.push(St::RqConfirm { q }),
                        naga::RayQueryFunction::Terminate => out.push(St::RqTerminate { q }),
                        naga::RayQueryFunction::GenerateIntersection { hit_t } => {
                            let t = *cx.slot.at(hit_t.index())?;
                            cx.uses.bump(t as usize)?;
                            out.push(St::RqGenerate { q, t });
                        }
                    }
                }
                Statement::Store { pointer, value } => {
                    let p = cx
                        .ptr_of
                        .at(pointer.index())?
                        .ok_or("store through a non-pointer")?;
                    if let Some(s) = p.loc.slot() {
                        cx.uses.bump(s as usize)?;
                    }
                    let Shape::V(_, n) = (cx.shape)(*value)? else {
                        return Err("store of an aggregate".into());
                    };
                    let src = *cx.slot.at(value.index())?;
                    cx.uses.bump(src as usize)?;
                    match p.dynamic {
                        None => out.push(St::Store { loc: p.loc, src, n }),
                        Some(d) => {
                            cx.uses.bump(d.0 as usize)?;
                            out.push(St::StoreDyn { ptr: p, src, n })
                        }
                    }
                }
                Statement::ImageStore {
                    image,
                    coordinate,
                    array_index,
                    value,
                } => {
                    let (tex, _, dim, arrayed) = (cx.image)(*image)?;
                    if let Some(s) = tex.slot() {
                        cx.uses.bump(s as usize)?;
                    }
                    let Shape::V(_, cn) = (cx.shape)(*coordinate)? else {
                        return Err("store coordinate".into());
                    };
                    let Shape::V(_, vn) = (cx.shape)(*value)? else {
                        return Err("store of an aggregate".into());
                    };
                    let (c, v) = (*cx.slot.at(coordinate.index())?, *cx.slot.at(value.index())?);
                    let layer = if arrayed {
                        array_index.map(|x| cx.slot.at(x.index()).copied()).transpose()?
                    } else {
                        None
                    };
                    cx.uses.bump(c as usize)?;
                    cx.uses.bump(v as usize)?;
                    if let Some(l) = layer {
                        cx.uses.bump(l as usize)?;
                    }
                    out.push(St::ImageStore {
                        tex,
                        dim: dim_of(dim),
                        coord: c,
                        cn,
                        layer,
                        value: v,
                        vn,
                    });
                }
                Statement::ImageAtomic {
                    image,
                    coordinate,
                    array_index,
                    fun,
                    value,
                    result,
                } => {
                    let (tex, _, dim, arrayed) = (cx.image)(*image)?;
                    if let Some(s) = tex.slot() {
                        cx.uses.bump(s as usize)?;
                    }
                    let op = match fun {
                        naga::AtomicFunction::Add => ImageAtomicOp::Add,
                        naga::AtomicFunction::Subtract => ImageAtomicOp::Sub,
                        naga::AtomicFunction::And => ImageAtomicOp::And,
                        naga::AtomicFunction::InclusiveOr => ImageAtomicOp::Or,
                        naga::AtomicFunction::ExclusiveOr => ImageAtomicOp::Xor,
                        naga::AtomicFunction::Min => ImageAtomicOp::Min,
                        naga::AtomicFunction::Max => ImageAtomicOp::Max,
                        naga::AtomicFunction::Exchange { compare: None } => ImageAtomicOp::Exchange,
                        naga::AtomicFunction::Exchange { compare: Some(_) } => {
                            return Err("a compare-exchange on an image".into());
                        }
                    };
                    let Shape::V(_, cn) = (cx.shape)(*coordinate)? else {
                        return Err("atomic coordinate".into());
                    };
                    let signed = matches!((cx.shape)(*value)?, Shape::V(Kind::I, _));
                    let (c, v) = (*cx.slot.at(coordinate.index())?, *cx.slot.at(value.index())?);
                    let layer = if arrayed {
                        array_index.map(|x| cx.slot.at(x.index()).copied()).transpose()?
                    } else {
                        None
                    };
                    cx.uses.bump(c as usize)?;
                    cx.uses.bump(v as usize)?;
                    if let Some(l) = layer {
                        cx.uses.bump(l as usize)?;
                    }
                    out.push(St::ImageAtomic {
                        tex,
                        dim: dim_of(dim),
                        coord: c,
                        cn,
                        layer,
                        op,
                        signed,
                        value: v,
                        result: result.map(|r| cx.slot.at(r.index()).copied()).transpose()?,
                    });
                }
                Statement::Atomic {
                    pointer,
                    fun,
                    value,
                    result,
                } => {
                    let (op, compare) = match fun {
                        naga::AtomicFunction::Add => (AtomOp::Add, None),
                        naga::AtomicFunction::Subtract => (AtomOp::Sub, None),
                        naga::AtomicFunction::And => (AtomOp::And, None),
                        naga::AtomicFunction::InclusiveOr => (AtomOp::Or, None),
                        naga::AtomicFunction::ExclusiveOr => (AtomOp::Xor, None),
                        naga::AtomicFunction::Min => (AtomOp::Min, None),
                        naga::AtomicFunction::Max => (AtomOp::Max, None),
                        naga::AtomicFunction::Exchange { compare: None } => (AtomOp::Exchange, None),
                        naga::AtomicFunction::Exchange { compare: Some(c) } => {
                            (AtomOp::CompareExchange, Some(*cx.slot.at(c.index())?))
                        }
                    };
                    let p = cx
                        .ptr_of
                        .at(pointer.index())?
                        .ok_or("atomic through a non-pointer")?;
                    if let Some(s) = p.loc.slot() {
                        cx.uses.bump(s as usize)?;
                    }
                    let v = *cx.slot.at(value.index())?;
                    let signed = matches!((cx.shape)(*value)?, Shape::V(Kind::I, _));
                    cx.uses.bump(v as usize)?;
                    if let Some(c) = compare {
                        cx.uses.bump(c as usize)?;
                    }
                    if let Some(d) = p.dynamic {
                        cx.uses.bump(d.0 as usize)?;
                    }
                    let (result, exchanged) = match result {
                        Some(r) => match cx.agg_of.at(r.index())? {
                            Some(members) => match members[..] {
                                [(old, _), (exchanged, _), ..] => (Some(old), Some(exchanged)),
                                _ => {
                                    return Err(
                                        "an atomic compare-exchange result without its two members".into()
                                    );
                                }
                            },
                            None => (Some(*cx.slot.at(r.index())?), None),
                        },
                        None => (None, None),
                    };
                    out.push(St::Atomic {
                        ptr: p,
                        fun: op,
                        signed,
                        value: v,
                        compare,
                        result,
                        exchanged,
                    });
                }
                Statement::ControlBarrier(_) => out.push(St::Barrier),
                Statement::SubgroupBallot { result, predicate } => {
                    let predicate = predicate.map(|p| cx.slot.at(p.index()).copied()).transpose()?;
                    if let Some(p) = predicate {
                        cx.uses.bump(p as usize)?;
                    }
                    out.push(St::WaveBallot {
                        predicate,
                        result: *cx.slot.at(result.index())?,
                    });
                }
                Statement::SubgroupGather {
                    mode,
                    argument,
                    result,
                } => {
                    use naga::GatherMode as G;
                    let mut slot_of = |h: &Handle<E>| -> Result<u32, String> {
                        let s = *cx.slot.at(h.index())?;
                        cx.uses.bump(s as usize)?;
                        Ok(s)
                    };
                    let mode = match mode {
                        G::BroadcastFirst => WaveGather::First,
                        G::Broadcast(i) => WaveGather::Lane(slot_of(i)?),
                        G::Shuffle(i) => WaveGather::Shuffle(slot_of(i)?),
                        G::ShuffleDown(i) => WaveGather::Down(slot_of(i)?),
                        G::ShuffleUp(i) => WaveGather::Up(slot_of(i)?),
                        G::ShuffleXor(i) => WaveGather::Xor(slot_of(i)?),
                        G::QuadBroadcast(i) => WaveGather::QuadLane(slot_of(i)?),
                        G::QuadSwap(naga::Direction::X) => WaveGather::QuadSwap(1),
                        G::QuadSwap(naga::Direction::Y) => WaveGather::QuadSwap(2),
                        G::QuadSwap(naga::Direction::Diagonal) => WaveGather::QuadSwap(3),
                    };
                    let Shape::V(_, n) = (cx.shape)(*argument)? else {
                        return Err("wave gather of an aggregate".into());
                    };
                    let value = *cx.slot.at(argument.index())?;
                    cx.uses.bump(value as usize)?;
                    out.push(St::WaveGather {
                        mode,
                        value,
                        n,
                        result: *cx.slot.at(result.index())?,
                    });
                }
                Statement::SubgroupCollectiveOperation {
                    op,
                    collective_op,
                    argument,
                    result,
                } => {
                    use naga::SubgroupOperation as S;
                    let op = match op {
                        S::All => WaveOp::All,
                        S::Any => WaveOp::Any,
                        S::Add => WaveOp::Add,
                        S::Mul => WaveOp::Mul,
                        S::Min => WaveOp::Min,
                        S::Max => WaveOp::Max,
                        S::And => WaveOp::And,
                        S::Or => WaveOp::Or,
                        S::Xor => WaveOp::Xor,
                    };
                    let scan = match collective_op {
                        naga::CollectiveOperation::Reduce => WaveScan::Reduce,
                        naga::CollectiveOperation::InclusiveScan => WaveScan::Inclusive,
                        naga::CollectiveOperation::ExclusiveScan => WaveScan::Exclusive,
                    };
                    let Shape::V(k, n) = (cx.shape)(*argument)? else {
                        return Err("wave operation on an aggregate".into());
                    };
                    let kind = match k {
                        Kind::F => WaveKind::F,
                        Kind::I => WaveKind::I,
                        _ => WaveKind::U,
                    };
                    let value = *cx.slot.at(argument.index())?;
                    cx.uses.bump(value as usize)?;
                    out.push(St::WaveOp {
                        op,
                        scan,
                        value,
                        n,
                        kind,
                        result: *cx.slot.at(result.index())?,
                    });
                }
                // one sequential schedule: every write is visible at once
                Statement::MemoryBarrier(_) => {}
                s => return Err(format!("statement {:?}", std::mem::discriminant(s))),
            }
        }
        Ok(out)
    }
    // derivative provenance (scalar mode): arguments' lanes that reach a derivative unchanged
    let prov0 = compute_prov(&BTreeMap::new(), &slot, &static_load);
    let mut local_src: BTreeMap<u32, (u32, u8)> = BTreeMap::new();
    let mut stored: BTreeMap<u32, u32> = BTreeMap::new();
    fn walk_stores(b: &naga::Block, f: &mut dyn FnMut(&Statement)) {
        for s in b.iter() {
            f(s);
            match s {
                Statement::Block(x) => walk_stores(x, f),
                Statement::If { accept, reject, .. } => {
                    walk_stores(accept, f);
                    walk_stores(reject, f);
                }
                Statement::Loop { body, continuing, .. } => {
                    walk_stores(body, f);
                    walk_stores(continuing, f);
                }
                Statement::Switch { cases, .. } => cases.iter().for_each(|c| walk_stores(&c.body, f)),
                _ => {}
            }
        }
    }
    walk_stores(&f.body, &mut |s| {
        let Statement::Store { pointer, value } = s else {
            return;
        };
        let Some(Ptr {
            loc: Loc::Local(o),
            dynamic,
        }) = ptr_of.get(pointer.index()).copied().flatten()
        else {
            return;
        };
        let n = match shape(*value) {
            Ok(Shape::V(_, n)) => n,
            _ => 0,
        };
        if dynamic.is_some() {
            let count = stored.entry(u32::MAX).or_default();
            *count = count.saturating_add(1);
        }
        for l in 0..n as u32 {
            let count = stored.entry(o.saturating_add(l)).or_default();
            *count = count.saturating_add(1);
            // a register past the expressions' own (an aggregate's member) carries no provenance
            if let Some(src) = prov0
                .get(slot.get(value.index()).copied().unwrap_or(u32::MAX) as usize)
                .copied()
                .flatten()
                .and_then(|pr| pr.get(l as usize).copied().flatten())
            {
                local_src.insert(o.saturating_add(l), src);
            }
        }
    });
    local_src.retain(|w, _| stored.get(w) == Some(&1));
    let prov = compute_prov(&local_src, &slot, &static_load);

    // a pointer to a structure the pipeline's operations read and write: its words in local memory, and how many
    let rt_pointer = |h: Handle<E>| -> Result<(u32, u32), String> {
        let p = ptr_of
            .at(h.index())?
            .ok_or("a pipeline operation on something that is not a pointer")?;
        let (Loc::Local(off), None) = (p.loc, p.dynamic) else {
            return Err("a pipeline operation on a pointer that is not a whole local structure".into());
        };
        let TypeInner::Pointer { base, .. } = inner(h) else {
            return Err("a pipeline operation on a pointer of an unexpected type".into());
        };
        Ok((off, words(m, *base)?))
    };
    let image = |h: Handle<E>| image_of(m, f, &resource_of, &slot, h);
    let mut cx = Ctx {
        slot: &slot,
        op_of: &op_of,
        ptr_of: &ptr_of,
        agg_of: &agg_of,
        members_of: &members_of,
        rq_of: &rq_of,
        functions: &m.functions,
        rt_pointer: &rt_pointer,
        exprs: &f.expressions,
        resource_of: &resource_of,
        shape: &shape,
        image: &image,
        ops: Vec::new(),
        uses: vec![0; n_slots],
    };
    let mut body = block(&mut cx, &f.body)?;
    let mut ops = std::mem::take(&mut cx.ops);
    let mut uses = std::mem::take(&mut cx.uses);

    // derivative provenance onto the ops
    for (dst, op) in ops.iter_mut() {
        let _ = dst;
        let Op::Deriv { src, prov: p, n, .. } = op else {
            continue;
        };
        let Some(pr) = prov.get(*src as usize).copied().flatten() else {
            continue;
        };
        let mut a = [(0u32, 0u8); 4];
        let mut whole = true;
        for (l, slot) in a.iter_mut().enumerate().take(*n as usize) {
            match pr.get(l).copied().flatten() {
                Some(x) => *slot = x,
                None => whole = false,
            }
        }
        if whole {
            *p = Some(a);
        }
    }
    // dead operations: nothing reads their result
    let mut keep = vec![true; ops.len()];
    for (k, (dst, op)) in ops.iter().enumerate().rev() {
        if *uses.at(*dst as usize)? == 0 {
            *keep.at_mut(k)? = false;
            for o in operands(op) {
                if *uses.at(o as usize)? > 0 {
                    uses.drop_one(o as usize)?;
                }
            }
        }
    }
    // rebuild: compact the ops and remap the Run ranges
    let mut remap = vec![0u32; ops.len().saturating_add(1)];
    let mut kept: Vec<(u32, Op)> = Vec::new();
    for (k, o) in ops.into_iter().enumerate() {
        *remap.at_mut(k)? = number(kept.len())?;
        if *keep.at(k)? {
            kept.push(o);
        }
    }
    *remap.at_mut(keep.len())? = number(kept.len())?;
    fn fix(b: &mut Vec<St>, remap: &Vec<u32>) -> Result<(), String> {
        for s in b.iter_mut() {
            match s {
                St::Run(lo, hi) => {
                    *lo = *remap.at(*lo as usize)?;
                    *hi = *remap.at(*hi as usize)?;
                }
                St::If { a, r, .. } => {
                    fix(a, remap)?;
                    fix(r, remap)?;
                }
                St::Loop { body, cont, .. } => {
                    fix(body, remap)?;
                    fix(cont, remap)?;
                }
                St::Switch { cases, .. } => {
                    for case in cases.iter_mut() {
                        fix(&mut case.2, remap)?;
                    }
                }
                _ => {}
            }
        }
        b.retain(|s| !matches!(s, St::Run(lo, hi) if lo == hi));
        Ok(())
    }
    fix(&mut body, &remap)?;
    args.resize(f.arguments.len(), (0, 0));
    let pointer_args: Vec<(usize, u32, u32)> = ptr_arg
        .iter()
        .zip(&arg_at)
        .filter_map(|(a, at)| a.map(|(off, words)| (*at, off, words)))
        .collect();
    let array_args: Vec<(usize, u32, u32, u8)> = array_arg
        .iter()
        .zip(&arg_at)
        .filter_map(|(a, at)| a.map(|(off, count, width)| (*at, off, count, width)))
        .collect();
    let mesh_output = match ep.and_then(|e| e.mesh_info.as_ref()) {
        Some(info) => {
            let global = *gid_of
                .get(&info.output_variable.index())
                .ok_or("a mesh program whose output variable is not memory")?
                as usize;
            match globals.at(global)?.kind {
                GlobalKind::Workgroup(words) => Some((global, words)),
                _ => return Err("a mesh program whose output variable is not thread-group memory".into()),
            }
        }
        None => None,
    };
    Ok(Executable {
        ops: kept,
        body,
        slots: n_slots,
        inits,
        args,
        arg_at,
        arg_len,
        array_args,
        pointer_args,
        local_words: local_words as usize,
        globals,
        flush_denormals: p.modes.flush_denormals,
        workgroup: ep
            .filter(|_| matches!(p.stage, iridia::Stage::Compute | iridia::Stage::Mesh))
            .map(|e| e.workgroup_size),
        mesh_output,
        builtins: f
            .arguments
            .iter()
            .map(|a| match a.binding {
                Some(naga::Binding::BuiltIn(b)) => Some(b),
                _ => None,
            })
            .collect(),
    })
}

/// The array argument an expression is, as its local words' offset, element count and element width.
fn array_of(
    f: &naga::Function,
    array_arg: &[Option<(u32, u32, u8)>],
    h: Handle<E>,
) -> Option<(u32, u32, u8)> {
    match f.expressions[h] {
        E::FunctionArgument(a) => array_arg.get(a as usize).copied().flatten(),
        _ => None,
    }
}

/// Whether the expression is a global that is an array of storage buffers.
fn is_buffer_array(
    f: &naga::Function,
    gid_of: &BTreeMap<usize, u32>,
    globals: &[GlobalDesc],
    h: Handle<E>,
) -> bool {
    match f.expressions[h] {
        E::GlobalVariable(g) => gid_of.get(&g.index()).is_some_and(|&gid| {
            globals
                .get(gid as usize)
                .is_some_and(|g| matches!(g.kind, GlobalKind::StorageArray(..)))
        }),
        _ => false,
    }
}

/// An image global as an expression names it: the resource, its class, its dimension and whether it is arrayed.
type ImageRef = (Res, ImageClass, naga::ImageDimension, bool);

/// Whether a global of the expression is a binding array.
fn is_binding_array(m: &Module, f: &naga::Function, h: Handle<E>) -> bool {
    matches!(f.expressions[h], E::GlobalVariable(g)
        if matches!(m.types[m.global_variables[g].ty].inner, TypeInner::BindingArray { .. }))
}

/// The resource a handle expression names, and its type: a global, or one element of a binding array picked by a
/// literal or a computed index.
fn resource_ref<'a>(
    m: &'a Module,
    f: &naga::Function,
    resource_of: &BTreeMap<usize, (ResourceId, u32)>,
    slot: &[u32],
    h: Handle<E>,
    what: &str,
) -> Result<(Res, &'a TypeInner), String> {
    let (g, pick) = match f.expressions[h] {
        E::GlobalVariable(g) => (g, Pick::One),
        E::Access { base, index } => match f.expressions[base] {
            E::GlobalVariable(g) => (g, Pick::Slot(*slot.at(index.index())?)),
            _ => return Err(format!("{what} expression")),
        },
        E::AccessIndex { base, index } => match f.expressions[base] {
            E::GlobalVariable(g) => (g, Pick::Fixed(index)),
            _ => return Err(format!("{what} expression")),
        },
        _ => return Err(format!("{what} expression")),
    };
    let &(id, count) = resource_of
        .get(&g.index())
        .ok_or_else(|| format!("unbound {what}"))?;
    let inner = match (&m.types[m.global_variables[g].ty].inner, pick) {
        (TypeInner::BindingArray { base, .. }, Pick::Fixed(_) | Pick::Slot(_)) => &m.types[*base].inner,
        (TypeInner::BindingArray { .. }, Pick::One) => return Err(format!("a whole array of {what}s")),
        (t, Pick::One) => t,
        (_, _) => return Err(format!("{what} indexed but not an array")),
    };
    Ok((Res { id, pick, count }, inner))
}
fn image_of(
    m: &Module,
    f: &naga::Function,
    resource_of: &BTreeMap<usize, (ResourceId, u32)>,
    slot: &[u32],
    h: Handle<E>,
) -> Result<ImageRef, String> {
    match resource_ref(m, f, resource_of, slot, h, "image")? {
        (res, TypeInner::Image { dim, arrayed, class }) => Ok((res, *class, *dim, *arrayed)),
        _ => Err("not an image".into()),
    }
}
fn dim_of(d: naga::ImageDimension) -> Dim {
    match d {
        naga::ImageDimension::D1 => Dim::D1,
        naga::ImageDimension::D2 => Dim::D2,
        naga::ImageDimension::D3 => Dim::D3,
        naga::ImageDimension::Cube => Dim::Cube,
    }
}
/// A constant integer vector or scalar of the function's expressions (a texel offset).
fn const_ints(f: &naga::Function, h: Handle<E>) -> Result<[i32; 3], String> {
    let one = |h: Handle<E>| -> Result<i32, String> {
        match f.expressions[h] {
            E::Literal(Literal::I32(x)) => Ok(x),
            E::Literal(Literal::U32(x)) => Ok(x as i32),
            _ => Err("texel offset is not a constant integer".into()),
        }
    };
    let mut r = [0; 3];
    match &f.expressions[h] {
        E::Compose { components, .. } => {
            for (to, c) in r.iter_mut().zip(components.iter().take(3)) {
                *to = one(*c)?;
            }
        }
        _ => r[0] = one(h)?,
    }
    Ok(r)
}
fn binding_of(
    m: &Module,
    f: &naga::Function,
    resource_of: &BTreeMap<usize, (ResourceId, u32)>,
    slot: &[u32],
    h: Handle<E>,
) -> Result<Res, String> {
    match resource_ref(m, f, resource_of, slot, h, "sampler")? {
        (res, TypeInner::Sampler { .. }) => Ok(res),
        _ => Err("not a sampler".into()),
    }
}
/// The words of a constant expression: literals, composites, splats, zero values and named constants. Anything
/// else (an operation naga did not fold, an override) is refused naming what it is.
fn const_words(m: &Module, h: Handle<E>) -> Result<Vec<u32>, String> {
    Ok(match &m.global_expressions[h] {
        E::Literal(Literal::U32(x)) => vec![*x],
        E::Literal(Literal::F32(x)) => vec![x.to_bits()],
        E::Literal(Literal::I32(x)) => vec![*x as u32],
        E::Literal(Literal::Bool(x)) => vec![u32::from(*x)],
        E::Compose { components, .. } => components
            .iter()
            .map(|c| const_words(m, *c))
            .collect::<Result<Vec<_>, _>>()?
            .concat(),
        E::Splat { size, value } => {
            let one = const_words(m, *value)?;
            (0..*size as usize).flat_map(|_| one.iter().copied()).collect()
        }
        E::ZeroValue(ty) => vec![0; words(m, *ty)? as usize],
        E::Constant(c) => const_words(m, m.constants[*c].init)?,
        e => {
            return Err(format!(
                "constant expression {:?} (the executor reads literals, composites, splats, zero values and named constants)",
                std::mem::discriminant(e)
            ));
        }
    })
}
