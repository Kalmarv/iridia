//! A `Cfg` body made `Regions`, by [`structure`](crate::structure::structure) (Ramsey's translation by the dominator
//! tree). An irreducible graph is refused by name.
//!
//! A value of one block that another uses, and every block parameter, goes through a variable first: the defining
//! block stores it, a block that uses it loads it at its start, and an edge stores its arguments into its target's
//! parameters. The structured code then needs no value to cross a construct, and [`promote`](crate::promote) turns
//! the variables back into values carried as construct results and loop parameters.

use alloc::{
    collections::{BTreeMap, btree_map},
    format,
    string::String,
    vec,
    vec::Vec,
};

use iridia_ir::{
    Block, Body, End, Function, LabelId, Node, Op, Precision, Region, Terminator, ValueDef, ValueId, Var,
    VarId,
};

use crate::structure::{self, Cfg, Code, Label, Term};

/// The values a block reads, in order: its nodes' operands and stored values, then its terminator's.
fn reads(f: &Function, b: &Block) -> Vec<ValueId> {
    let mut out = Vec::new();
    for n in &b.nodes {
        match n {
            Node::Value(v) => out.extend(f.value(*v).map(|d| d.op.operands()).unwrap_or_default()),
            other => out.extend(other.values()),
        }
    }
    match &b.term {
        Terminator::Branch { cond, .. } => out.push(*cond),
        Terminator::Switch { selector, .. } => out.push(*selector),
        _ => {}
    }
    for t in b.term.targets() {
        out.extend(t.args.iter().copied());
    }
    out
}

fn add_value(f: &mut Function, def: ValueDef) -> Result<ValueId, String> {
    let id = ValueId(u32::try_from(f.values.len()).map_err(|_| String::from("more than 2^32 values"))?);
    f.values.push(def);
    Ok(id)
}

fn add_var(f: &mut Function, v: ValueId) -> Result<VarId, String> {
    let ty = f.ty(v).ok_or_else(|| format!("%{} is not a value", v.0))?;
    let id = VarId(u32::try_from(f.vars.len()).map_err(|_| String::from("more than 2^32 variables"))?);
    f.vars.push(Var::new(&format!("v{}", v.0), ty));
    Ok(id)
}

/// The blocks rewritten so that no value crosses from one block to another (see the module's account).
fn demote(f: &mut Function, blocks: &mut [Block]) -> Result<Vec<Vec<VarId>>, String> {
    // the block that defines each value
    let mut home: Vec<Option<usize>> = vec![None; f.values.len()];
    for (i, b) in blocks.iter().enumerate() {
        for v in b.params.iter().chain(b.nodes.iter().filter_map(|n| match n {
            Node::Value(v) => Some(v),
            _ => None,
        })) {
            if let Some(h) = home.get_mut(v.0 as usize) {
                *h = Some(i);
            }
        }
    }
    let mut var: BTreeMap<ValueId, VarId> = BTreeMap::new();
    let mut params = Vec::new();
    for b in blocks.iter() {
        let mut vs = Vec::new();
        for &p in &b.params {
            let v = add_var(f, p)?;
            var.insert(p, v);
            vs.push(v);
        }
        params.push(vs);
    }
    let crossing: Vec<Vec<ValueId>> = blocks
        .iter()
        .enumerate()
        .map(|(i, b)| {
            let mut seen = Vec::new();
            for v in reads(f, b) {
                let from = home.get(v.0 as usize).copied().flatten();
                if from.is_some_and(|h| h != i) && !seen.contains(&v) {
                    seen.push(v);
                }
            }
            seen
        })
        .collect();
    // the values another block reads that are not parameters: stored where they are defined
    let mut stored: BTreeMap<ValueId, VarId> = BTreeMap::new();
    for &v in crossing.iter().flatten() {
        if let btree_map::Entry::Vacant(e) = var.entry(v) {
            let id = add_var(f, v)?;
            e.insert(id);
            stored.insert(v, id);
        }
    }
    for (i, b) in blocks.iter_mut().enumerate() {
        let mut start = Vec::new();
        // a parameter is its variable's value on entry
        for (&p, &v) in b
            .params
            .iter()
            .zip(params.get(i).map(Vec::as_slice).unwrap_or_default())
        {
            if let Some(d) = f.values.get_mut(p.0 as usize) {
                d.op = Op::LoadVar(v);
            }
            start.push(Node::Value(p));
        }
        b.params.clear();
        // a value of another block is loaded once, at the start
        let mut local: BTreeMap<ValueId, ValueId> = BTreeMap::new();
        for &v in crossing.get(i).map(Vec::as_slice).unwrap_or_default() {
            let (Some(&x), Some(ty)) = (var.get(&v), f.ty(v)) else {
                continue;
            };
            let def = ValueDef {
                ty,
                op: Op::LoadVar(x),
                origin: None,
                precision: Precision::Full,
            };
            let loaded = add_value(f, def)?;
            local.insert(v, loaded);
            start.push(Node::Value(loaded));
        }
        let sub = |v: ValueId| local.get(&v).copied().unwrap_or(v);
        let mut nodes = start;
        for mut n in core::mem::take(&mut b.nodes) {
            match &mut n {
                Node::Value(v) => {
                    if let Some(d) = f.values.get_mut(v.0 as usize) {
                        d.op.map_operands(sub);
                    }
                }
                Node::If { .. } | Node::Block { .. } | Node::Loop { .. } | Node::Switch { .. } => {
                    return Err(format!("block {i} holds a construct"));
                }
                other => other.values_mut().into_iter().for_each(|x| *x = sub(*x)),
            }
            let store = match n {
                Node::Value(v) => stored.get(&v).map(|&x| Node::StoreVar(x, v)),
                _ => None,
            };
            nodes.push(n);
            nodes.extend(store);
        }
        b.nodes = nodes;
        match &mut b.term {
            Terminator::Branch { cond, .. } => *cond = sub(*cond),
            Terminator::Switch { selector, .. } => *selector = sub(*selector),
            _ => {}
        }
        for t in b.term.targets_mut() {
            t.args.iter_mut().for_each(|a| *a = sub(*a));
        }
    }
    Ok(params)
}

struct Build {
    nodes: Vec<Option<Vec<Node>>>,
    terms: Vec<Terminator>,
    params: Vec<Vec<VarId>>,
    blocks: BTreeMap<Label, LabelId>,
    loops: BTreeMap<Label, LabelId>,
    labels: u32,
    /// The `If` or `Switch` arm being built: its node and which of its distinct targets (an `If`'s accept 0, reject
    /// 1), for the edge it starts with.
    arm: Option<(usize, usize)>,
}

/// The distinct target blocks of a terminator, in order of first appearance, as the structure's graph has them: an
/// `If`'s two (even where they are one block), a switch's each once.
fn distinct(term: &Terminator) -> Vec<usize> {
    match term {
        Terminator::Branch { accept, reject, .. } => {
            alloc::vec![accept.block as usize, reject.block as usize]
        }
        other => {
            let mut out: Vec<usize> = Vec::new();
            for t in other.targets() {
                if !out.contains(&(t.block as usize)) {
                    out.push(t.block as usize);
                }
            }
            out
        }
    }
}

impl Build {
    fn label(&mut self) -> LabelId {
        let l = LabelId(self.labels);
        self.labels = self.labels.saturating_add(1);
        l
    }

    /// The edge `from -> to`: its arguments stored into the target's parameters. Each argument is a value computed
    /// before the first store, so the copies are taken together.
    fn edge(&self, from: usize, to: usize) -> Result<Vec<Node>, String> {
        let t = match self.terms.get(from) {
            Some(Terminator::Jump(t)) => t,
            Some(Terminator::Branch { accept, reject, .. }) => match self.arm {
                Some((n, way)) if n == from => {
                    if way == 0 {
                        accept
                    } else {
                        reject
                    }
                }
                _ if accept.block as usize == to => accept,
                _ => reject,
            },
            Some(term @ Terminator::Switch { .. }) => {
                // every edge from a switch to one block carries the same arguments (a source's phi has one entry
                // per predecessor), checked where the graph is structured
                term.targets()
                    .into_iter()
                    .find(|t| t.block as usize == to)
                    .ok_or_else(|| {
                        format!("an edge from block {from} to {to}, which it does not branch to")
                    })?
            }
            _ => return Err(format!("an edge from block {from}, which does not branch")),
        };
        let vars = self
            .params
            .get(to)
            .ok_or_else(|| format!("an edge to block {to}, which does not exist"))?;
        Ok(vars
            .iter()
            .zip(&t.args)
            .map(|(&v, &a)| Node::StoreVar(v, a))
            .collect())
    }

    /// The region `code` is, ending as `fall` where it falls off its end.
    fn region(&mut self, code: &Code, fall: End) -> Result<Region, String> {
        let mut r = Region::default();
        self.code(code, &mut r)?;
        if r.end.is_none() {
            r.end = Some(fall);
        }
        Ok(r)
    }

    fn code(&mut self, code: &Code, r: &mut Region) -> Result<(), String> {
        if r.end.is_some() {
            // what follows an exit is not reached
            return Ok(());
        }
        match code {
            Code::Node(n) => {
                let nodes = self.nodes.get_mut(*n).and_then(Option::take);
                r.nodes
                    .extend(nodes.ok_or_else(|| format!("block {n} is placed twice"))?);
            }
            Code::Seq(items) => {
                for c in items {
                    self.code(c, r)?;
                }
            }
            Code::Edge(from, to) => {
                let copies = self.edge(*from, *to)?;
                r.nodes.extend(copies);
            }
            Code::Block(label, body) => {
                let l = self.label();
                self.blocks.insert(*label, l);
                let body = self.region(body, End::Yield(Vec::new()))?;
                self.blocks.remove(label);
                r.nodes.push(Node::Block {
                    label: l,
                    body,
                    results: Vec::new(),
                });
            }
            Code::Loop(label, body) => {
                let l = self.label();
                self.loops.insert(*label, l);
                // falling off a loop's body leaves it
                let body = self.region(body, End::Yield(Vec::new()))?;
                self.loops.remove(label);
                r.nodes.push(Node::Loop {
                    label: l,
                    params: Vec::new(),
                    inits: Vec::new(),
                    body,
                    results: Vec::new(),
                });
            }
            Code::If(n, accept, reject) => {
                let Some(Terminator::Branch { cond, .. }) = self.terms.get(*n) else {
                    return Err(format!("block {n} is an if, and does not branch"));
                };
                let cond = *cond;
                let outer = self.arm;
                self.arm = Some((*n, 0));
                let accept = self.region(accept, End::Yield(Vec::new()))?;
                self.arm = Some((*n, 1));
                let reject = self.region(reject, End::Yield(Vec::new()))?;
                self.arm = outer;
                r.nodes.push(Node::If {
                    cond,
                    accept,
                    reject,
                    results: Vec::new(),
                });
            }
            Code::Switch(n, arms) => {
                let Some(term) = self.terms.get(*n).cloned() else {
                    return Err(format!("block {n} is a switch, and does not switch"));
                };
                let targets = distinct(&term);
                let Terminator::Switch {
                    selector,
                    cases,
                    default,
                } = term
                else {
                    return Err(format!("block {n} is a switch, and does not switch"));
                };
                if arms.len() != targets.len() {
                    return Err(format!(
                        "block {n}'s switch has {} arms for {} targets",
                        arms.len(),
                        targets.len()
                    ));
                }
                let label = self.label();
                let outer = self.arm;
                let mut built = Vec::new();
                let mut default_region = None;
                for (i, (arm, &to)) in arms.iter().zip(&targets).enumerate() {
                    self.arm = Some((*n, i));
                    let body = self.region(arm, End::Yield(Vec::new()))?;
                    if to == default.block as usize {
                        default_region = Some(body);
                    } else {
                        let values = cases
                            .iter()
                            .filter(|(_, t)| t.block as usize == to)
                            .map(|(v, _)| *v)
                            .collect();
                        built.push(iridia_ir::Case { values, body });
                    }
                }
                self.arm = outer;
                r.nodes.push(Node::Switch {
                    label,
                    selector,
                    cases: built,
                    default: default_region
                        .ok_or_else(|| format!("block {n}'s switch has no default arm"))?,
                    results: Vec::new(),
                });
            }
            Code::Br(label) => {
                let l = self
                    .blocks
                    .get(label)
                    .ok_or_else(|| format!("a branch to block {label}, outside its block"))?;
                r.end = Some(End::Br(*l, Vec::new()));
            }
            Code::Continue(label) => {
                let l = self
                    .loops
                    .get(label)
                    .ok_or_else(|| format!("a continue to block {label}, outside its loop"))?;
                r.end = Some(End::Continue(*l, Vec::new()));
            }
            Code::Exit(n) => {
                r.end = Some(match self.terms.get(*n) {
                    Some(Terminator::Return) => End::Return,
                    Some(Terminator::Kill) => End::Kill,
                    Some(Terminator::IgnoreHit) => End::IgnoreHit,
                    Some(Terminator::AcceptHitAndEndSearch) => End::AcceptHitAndEndSearch,
                    Some(Terminator::Unreachable) => End::Unreachable,
                    _ => return Err(format!("block {n} exits, and does not end the program")),
                });
            }
        }
        Ok(())
    }
}

/// Make `f`'s body `Regions` if it is a `Cfg` (a `Regions` body is left as it is), then promote the variables the
/// translation introduced. Refused: an irreducible graph, by the loop and the second entry into it.
pub fn structurize(f: &mut Function) -> Result<(), String> {
    let mut blocks = match core::mem::replace(&mut f.body, Body::Cfg(Vec::new())) {
        Body::Cfg(blocks) => blocks,
        regions => {
            f.body = regions;
            return Ok(());
        }
    };
    let cfg = Cfg {
        terms: blocks
            .iter()
            .map(|b| match &b.term {
                Terminator::Jump(t) => Term::Goto(t.block as usize),
                Terminator::Branch { accept, reject, .. } => {
                    Term::If(accept.block as usize, reject.block as usize)
                }
                t @ Terminator::Switch { .. } => Term::Switch(distinct(t)),
                Terminator::Return
                | Terminator::Kill
                | Terminator::IgnoreHit
                | Terminator::AcceptHitAndEndSearch
                | Terminator::Unreachable => Term::Exit,
            })
            .collect(),
    };
    for (i, b) in blocks.iter().enumerate() {
        if let Terminator::Switch { .. } = &b.term {
            let targets = b.term.targets();
            for t in &targets {
                if targets.iter().any(|u| u.block == t.block && u.args != t.args) {
                    return Err(format!(
                        "block {i}'s switch gives block {} two sets of arguments",
                        t.block
                    ));
                }
            }
        }
    }
    let code = structure::structure(&cfg).map_err(|e| format!("{e}"))?;
    let params = demote(f, &mut blocks)?;
    let mut build = Build {
        terms: blocks.iter().map(|b| b.term.clone()).collect(),
        nodes: blocks.into_iter().map(|b| Some(b.nodes)).collect(),
        params,
        blocks: BTreeMap::new(),
        loops: BTreeMap::new(),
        labels: 0,
        arm: None,
    };
    // every way through the structured code ends in an exit: falling off the body is not reached
    let body = build.region(&code, End::Unreachable)?;
    f.body = Body::Regions(body);
    crate::promote(f)
}
