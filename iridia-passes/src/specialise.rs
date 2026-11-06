//! Known pipeline state folded into the program: each read of a known [`StateParam`] becomes a literal, what is
//! computed from literals is folded ([`crate::fold`]: a comparison of a texture's dimension, a select on it), each
//! branch on a literal condition takes its one way, and code no longer reached is removed. The state resources folded
//! leave the interface, and so does every resource that only the code removed read (the textures of a pixel 1.x stage's
//! other dimensions): the program no longer depends on them. A resource it never read stays.
//!
//! [`StateParam`]: iridia_ir::interface::StateParam

use alloc::{format, string::String, vec, vec::Vec};

use iridia_ir::{
    Block, Body, End, Function, Node, Op, Program, Region, Terminator, Type, ValueId,
    interface::{ResourceId, ResourceKind},
    value_type,
};

/// The condition's value, where it is a literal bool.
fn known(f: &Function, cond: ValueId) -> Option<bool> {
    match f.value(cond) {
        Some(d) if d.ty == Type::BOOL => match d.op {
            Op::Literal(bits) => Some(bits[0] != 0),
            _ => None,
        },
        _ => None,
    }
}

/// The selector's value, where it is a literal 32-bit integer.
fn selected(f: &Function, selector: ValueId) -> Option<u32> {
    match f.value(selector) {
        Some(d) if d.ty.lanes == 1 && d.ty.scalar.is_integer() && d.ty.scalar.bits() == 32 => match d.op {
            Op::Literal(bits) => u32::try_from(bits[0]).ok(),
            _ => None,
        },
        _ => None,
    }
}

/// Branches on a literal condition taken; the blocks then unreached removed and the rest renumbered in order.
fn fold_graph(f: &Function, blocks: Vec<Block>) -> Vec<Block> {
    let mut blocks = blocks;
    for b in &mut blocks {
        if let Terminator::Branch { cond, accept, reject } = &b.term
            && let Some(way) = known(f, *cond)
        {
            b.term = Terminator::Jump(if way { accept.clone() } else { reject.clone() });
        }
        if let Terminator::Switch {
            selector,
            cases,
            default,
        } = &b.term
            && let Some(v) = selected(f, *selector)
        {
            let to = cases.iter().find(|(c, _)| *c == v).map_or(default, |(_, t)| t);
            b.term = Terminator::Jump(to.clone());
        }
    }
    let mut reached = vec![false; blocks.len()];
    let mut stack = vec![0usize];
    while let Some(at) = stack.pop() {
        match reached.get_mut(at) {
            Some(r) if !*r => *r = true,
            _ => continue,
        }
        if let Some(b) = blocks.get(at) {
            stack.extend(b.term.targets().into_iter().map(|t| t.block as usize));
        }
    }
    let mut number = vec![u32::MAX; blocks.len()];
    let mut next = 0u32;
    for (n, r) in number.iter_mut().zip(&reached) {
        if *r {
            *n = next;
            next = next.saturating_add(1);
        }
    }
    let renumber = |b: &mut u32| *b = number.get(*b as usize).copied().unwrap_or(u32::MAX);
    blocks
        .into_iter()
        .zip(reached)
        .filter(|(_, r)| *r)
        .map(|(mut b, _)| {
            b.term.targets_mut().into_iter().for_each(|t| renumber(&mut t.block));
            b
        })
        .collect()
}

/// An `If` on a literal condition replaced by its one arm, in place: the arm's nodes join the region, and its yield
/// gives the `If`'s results (`subst`); an arm that leaves ends the region, and what followed is not reached. A `Switch`
/// on a literal selector becomes a `Block` of the case it takes (the default where no case names the value), under the
/// switch's label, so a break out of the case still leaves to the same place with the same results.
fn fold_region(f: &Function, r: Region, subst: &mut Vec<Option<ValueId>>) -> Region {
    let mut out = Region {
        nodes: Vec::new(),
        end: None,
    };
    for n in r.nodes {
        match n {
            Node::If {
                cond,
                accept,
                reject,
                results,
            } => match known(f, cond) {
                Some(way) => {
                    let arm = fold_region(f, if way { accept } else { reject }, subst);
                    out.nodes.extend(arm.nodes);
                    match arm.end {
                        Some(End::Yield(vs)) => {
                            for (r, v) in results.iter().zip(vs) {
                                if let Some(s) = subst.get_mut(r.0 as usize) {
                                    *s = Some(v);
                                }
                            }
                        }
                        end => {
                            out.end = end;
                            return out;
                        }
                    }
                }
                None => out.nodes.push(Node::If {
                    cond,
                    accept: fold_region(f, accept, subst),
                    reject: fold_region(f, reject, subst),
                    results,
                }),
            },
            Node::Switch {
                label,
                selector,
                cases,
                default,
                results,
            } => match selected(f, selector) {
                Some(v) => {
                    let taken = cases
                        .into_iter()
                        .find(|c| c.values.contains(&v))
                        .map_or(default, |c| c.body);
                    out.nodes.push(Node::Block {
                        label,
                        body: fold_region(f, taken, subst),
                        results,
                    });
                }
                None => out.nodes.push(Node::Switch {
                    label,
                    selector,
                    cases: cases
                        .into_iter()
                        .map(|c| iridia_ir::Case {
                            values: c.values,
                            body: fold_region(f, c.body, subst),
                        })
                        .collect(),
                    default: fold_region(f, default, subst),
                    results,
                }),
            },
            Node::Block { label, body, results } => {
                out.nodes.push(Node::Block {
                    label,
                    body: fold_region(f, body, subst),
                    results,
                });
            }
            Node::Loop {
                label,
                params,
                inits,
                body,
                results,
            } => {
                out.nodes.push(Node::Loop {
                    label,
                    params,
                    inits,
                    body: fold_region(f, body, subst),
                    results,
                });
            }
            other => out.nodes.push(other),
        }
    }
    out.end = r.end;
    out
}

/// The value `v` stands for after the folds: the end of its chain of substitutions.
fn resolve(subst: &[Option<ValueId>], mut v: ValueId) -> ValueId {
    // each substitution names a value defined before the one it replaces, so the chain ends
    for _ in 0..subst.len() {
        match subst.get(v.0 as usize).copied().flatten() {
            Some(w) => v = w,
            None => break,
        }
    }
    v
}

fn rename_region(r: &mut Region, s: &impl Fn(ValueId) -> ValueId) {
    for n in &mut r.nodes {
        n.values_mut().into_iter().for_each(|v| *v = s(*v));
        n.regions_mut().into_iter().for_each(|r| rename_region(r, s));
    }
    if let Some(End::Yield(vs) | End::Br(_, vs) | End::Continue(_, vs)) = &mut r.end {
        vs.iter_mut().for_each(|v| *v = s(*v));
    }
}

/// Fold the known pipeline state `known_state` into `p`: each value is the state's lanes as an [`Op::Literal`] holds
/// them (a bool is 0 or 1, lanes past the state's width are zero). A state the program declares but `known_state`
/// does not name stays as it is. Refused: a value for a resource that is not pipeline state, or that its type cannot
/// hold.
pub fn specialise(p: &mut Program, known_state: &[(ResourceId, [u64; 4])]) -> Result<(), String> {
    for (id, bits) in known_state {
        let param = match p.interface.resources.iter().find(|r| r.id == *id) {
            Some(r) => match r.kind {
                ResourceKind::PipelineState(param) => param,
                _ => {
                    return Err(format!(
                        "resource {} ({}) is not pipeline state",
                        id.0, r.provenance.name
                    ));
                }
            },
            None => return Err(format!("resource {} is not in the interface", id.0)),
        };
        let t = value_type(param.value());
        let limit = if t == Type::BOOL {
            1
        } else {
            (1u64 << t.scalar.bits()).saturating_sub(1)
        };
        let fits = bits.iter().enumerate().all(|(n, &b)| {
            if n < usize::from(t.lanes) {
                b <= limit
            } else {
                b == 0
            }
        });
        if !fits {
            return Err(format!("state {} ({param:?}, {t}) cannot hold {bits:?}", id.0));
        }
    }
    let read_before = read(&p.entry);
    let f = &mut p.entry;
    for d in &mut f.values {
        if let Op::LoadState(r) = d.op
            && let Some((_, bits)) = known_state.iter().find(|(id, _)| *id == r)
        {
            d.op = Op::Literal(*bits);
        }
    }
    crate::fold(f);
    f.body = match core::mem::replace(&mut f.body, Body::Cfg(Vec::new())) {
        Body::Cfg(blocks) => Body::Cfg(fold_graph(f, blocks)),
        Body::Regions(r) => {
            let mut subst = vec![None; f.values.len()];
            let mut r = fold_region(f, r, &mut subst);
            let s = |v| resolve(&subst, v);
            rename_region(&mut r, &s);
            for d in &mut f.values {
                d.op.map_operands(s);
            }
            Body::Regions(r)
        }
    };
    let read_after = read(&p.entry);
    p.interface.resources.retain(|r| {
        !known_state.iter().any(|(id, _)| *id == r.id)
            && (read_after.contains(&r.id) || !read_before.contains(&r.id))
    });
    Ok(())
}

/// The resources a function's body uses: those its placed values read, and the buffers and images it stores to.
fn read(f: &Function) -> Vec<ResourceId> {
    fn region(f: &Function, r: &Region, out: &mut Vec<ResourceId>) {
        for n in &r.nodes {
            nodes(f, n, out);
        }
    }
    fn nodes(f: &Function, n: &Node, out: &mut Vec<ResourceId>) {
        match n {
            Node::Value(v) => out.extend(f.value(*v).map(|d| d.op.resources()).unwrap_or_default()),
            Node::StoreBuffer { resource, .. } => out.push(resource.id),
            Node::StoreTexel { image, .. } => out.push(image.id),
            _ => n.regions().into_iter().for_each(|r| region(f, r, out)),
        }
    }
    let mut out = Vec::new();
    match &f.body {
        Body::Regions(r) => region(f, r, &mut out),
        Body::Cfg(blocks) => blocks
            .iter()
            .flat_map(|b| &b.nodes)
            .for_each(|n| nodes(f, n, &mut out)),
    }
    out.sort_unstable();
    out.dedup();
    out
}
