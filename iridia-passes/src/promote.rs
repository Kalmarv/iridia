//! Variables to values: each load of a variable becomes the value last stored to it on that path, and a construct
//! that stores to a variable carries it out (an `If`'s or a `Block`'s results, a `Loop`'s parameters and results).
//! A load before any store reads an undefined value. The promoted variables are gone afterwards; those that are not
//! values (arrays, workgroup memory, a variable an atomic operation reaches) stay, renumbered.

use alloc::{collections::BTreeSet, format, string::String, vec::Vec};

use iridia_ir::{
    Body, End, Function, LabelId, Memory, Node, Op, Precision, Region, Space, ValueDef, ValueId, VarId,
};

struct Label {
    id: LabelId,
    /// The variables an exit to it carries, in order.
    carried: Vec<VarId>,
}

struct Promote<'a> {
    f: &'a mut Function,
    /// Each variable's current value on the path being walked.
    env: Vec<Option<ValueId>>,
    /// The value each removed load stands for.
    subst: Vec<Option<ValueId>>,
    labels: Vec<Label>,
    /// Each variable's number after the pass, where it stays; `None` for one promoted.
    keep: Vec<Option<VarId>>,
}

/// The variables stored to anywhere in a region, in order.
fn stored(r: &Region, out: &mut BTreeSet<VarId>) {
    for n in &r.nodes {
        match n {
            Node::StoreVar(v, _) => {
                out.insert(*v);
            }
            other => {
                for r in other.regions() {
                    stored(r, out);
                }
            }
        }
    }
}

/// The variables an atomic operation reaches anywhere in a region.
fn atomic_vars(f: &Function, r: &Region, out: &mut BTreeSet<VarId>) {
    for n in &r.nodes {
        if let Node::Value(v) = n
            && let Some(Op::Atomic {
                memory: Memory::Var(var, _),
                ..
            }) = f.value(*v).map(|d| &d.op)
        {
            out.insert(*var);
        }
        for inner in n.regions() {
            atomic_vars(f, inner, out);
        }
    }
}

impl Promote<'_> {
    fn promoted(&self, var: VarId) -> bool {
        self.keep.get(var.0 as usize).is_some_and(|k| k.is_none())
    }
    fn id(&self) -> Result<ValueId, String> {
        u32::try_from(self.f.values.len())
            .map(ValueId)
            .map_err(|_| "more than 2^32 values".into())
    }
    fn new_value(&mut self, var: VarId, op: Op) -> Result<ValueId, String> {
        let ty = self
            .f
            .vars
            .get(var.0 as usize)
            .ok_or_else(|| format!("variable {} does not exist", var.0))?
            .ty;
        let id = self.id()?;
        self.f.values.push(ValueDef {
            ty,
            op,
            origin: None,
            precision: Precision::Full,
        });
        Ok(id)
    }
    fn sub(&self, v: ValueId) -> ValueId {
        self.subst.get(v.0 as usize).copied().flatten().unwrap_or(v)
    }
    fn subs(&self, vs: &mut [ValueId]) {
        for v in vs {
            *v = self.sub(*v);
        }
    }
    /// The variable's current value, an undefined value placed in `out` where it has none.
    fn current(&mut self, var: VarId, out: &mut Vec<Node>) -> Result<ValueId, String> {
        if let Some(v) = self.env.get(var.0 as usize).copied().flatten() {
            return Ok(v);
        }
        let v = self.new_value(var, Op::Undef)?;
        out.push(Node::Value(v));
        Ok(v)
    }
    fn set(&mut self, var: VarId, v: ValueId) -> Result<(), String> {
        let slot = self
            .env
            .get_mut(var.0 as usize)
            .ok_or_else(|| format!("variable {} does not exist", var.0))?;
        *slot = Some(v);
        Ok(())
    }
    fn carry(&mut self, vars: &[VarId], vs: &mut Vec<ValueId>, out: &mut Vec<Node>) -> Result<(), String> {
        for &var in vars {
            let v = self.current(var, out)?;
            vs.push(v);
        }
        Ok(())
    }
    fn label(&self, l: LabelId) -> Result<Vec<VarId>, String> {
        self.labels
            .iter()
            .rev()
            .find(|x| x.id == l)
            .map(|x| x.carried.clone())
            .ok_or_else(|| format!("an exit to label {}, which no enclosing construct has", l.0))
    }
    /// New results (or parameters) for the carried variables, which then hold them.
    fn results(&mut self, carried: &[VarId], results: &mut Vec<ValueId>) -> Result<Vec<ValueId>, String> {
        let mut new = Vec::new();
        for &var in carried {
            let p = self.new_value(var, Op::Param)?;
            results.push(p);
            new.push(p);
        }
        Ok(new)
    }
    /// Rewrite a region; a yield at its end carries `yields`.
    fn region(&mut self, r: Region, yields: &[VarId]) -> Result<Region, String> {
        let mut out = Vec::new();
        for n in r.nodes {
            match n {
                Node::Value(v) => {
                    let Some(def) = self.f.values.get(v.0 as usize) else {
                        return Err(format!("%{} is not a value of the function", v.0));
                    };
                    let mut op = def.op.clone();
                    op.map_operands(|o| self.sub(o));
                    if let Op::LoadVar(var) = op
                        && self.promoted(var)
                    {
                        let value = self.current(var, &mut out)?;
                        if let Some(s) = self.subst.get_mut(v.0 as usize) {
                            *s = Some(value);
                        }
                        continue;
                    }
                    if let Some(def) = self.f.values.get_mut(v.0 as usize) {
                        def.op = op;
                    }
                    out.push(Node::Value(v));
                }
                Node::StoreVar(var, x) if self.promoted(var) => {
                    let x = self.sub(x);
                    self.set(var, x)?;
                }
                Node::Switch {
                    label,
                    selector,
                    mut cases,
                    default,
                    mut results,
                } => {
                    let mut carried = BTreeSet::new();
                    for c in &cases {
                        stored(&c.body, &mut carried);
                    }
                    stored(&default, &mut carried);
                    let carried: Vec<VarId> = carried.into_iter().collect();
                    let before = self.env.clone();
                    self.labels.push(Label {
                        id: label,
                        carried: carried.clone(),
                    });
                    for c in &mut cases {
                        self.env.clone_from(&before);
                        let body = core::mem::take(&mut c.body);
                        c.body = self.region(body, &carried)?;
                    }
                    self.env.clone_from(&before);
                    let default = self.region(default, &carried)?;
                    self.labels.pop();
                    self.env = before;
                    let new = self.results(&carried, &mut results)?;
                    for (&var, p) in carried.iter().zip(new) {
                        self.set(var, p)?;
                    }
                    out.push(Node::Switch {
                        label,
                        selector: self.sub(selector),
                        cases,
                        default,
                        results,
                    });
                }
                Node::If {
                    cond,
                    accept,
                    reject,
                    mut results,
                } => {
                    let mut carried = BTreeSet::new();
                    stored(&accept, &mut carried);
                    stored(&reject, &mut carried);
                    let carried: Vec<VarId> = carried.into_iter().collect();
                    let before = self.env.clone();
                    let accept = self.region(accept, &carried)?;
                    self.env.clone_from(&before);
                    let reject = self.region(reject, &carried)?;
                    self.env = before;
                    let new = self.results(&carried, &mut results)?;
                    for (&var, p) in carried.iter().zip(new) {
                        self.set(var, p)?;
                    }
                    out.push(Node::If {
                        cond: self.sub(cond),
                        accept,
                        reject,
                        results,
                    });
                }
                Node::Block {
                    label,
                    body,
                    mut results,
                } => {
                    let mut carried = BTreeSet::new();
                    stored(&body, &mut carried);
                    let carried: Vec<VarId> = carried.into_iter().collect();
                    let before = self.env.clone();
                    self.labels.push(Label {
                        id: label,
                        carried: carried.clone(),
                    });
                    let body = self.region(body, &carried)?;
                    self.labels.pop();
                    self.env = before;
                    let new = self.results(&carried, &mut results)?;
                    for (&var, p) in carried.iter().zip(new) {
                        self.set(var, p)?;
                    }
                    out.push(Node::Block { label, body, results });
                }
                Node::Loop {
                    label,
                    mut params,
                    mut inits,
                    body,
                    mut results,
                } => {
                    let mut carried = BTreeSet::new();
                    stored(&body, &mut carried);
                    let carried: Vec<VarId> = carried.into_iter().collect();
                    self.subs(&mut inits);
                    self.carry(&carried, &mut inits, &mut out)?;
                    let before = self.env.clone();
                    let new = self.results(&carried, &mut params)?;
                    for (&var, p) in carried.iter().zip(new) {
                        self.set(var, p)?;
                    }
                    self.labels.push(Label {
                        id: label,
                        carried: carried.clone(),
                    });
                    let body = self.region(body, &carried)?;
                    self.labels.pop();
                    self.env = before;
                    let new = self.results(&carried, &mut results)?;
                    for (&var, p) in carried.iter().zip(new) {
                        self.set(var, p)?;
                    }
                    out.push(Node::Loop {
                        label,
                        params,
                        inits,
                        body,
                        results,
                    });
                }
                mut other => {
                    for v in other.values_mut() {
                        *v = self.sub(*v);
                    }
                    out.push(other);
                }
            }
        }
        let end = match r.end {
            Some(End::Yield(mut vs)) => {
                self.subs(&mut vs);
                self.carry(yields, &mut vs, &mut out)?;
                Some(End::Yield(vs))
            }
            Some(End::Br(l, mut vs)) => {
                self.subs(&mut vs);
                let carried = self.label(l)?;
                self.carry(&carried, &mut vs, &mut out)?;
                Some(End::Br(l, vs))
            }
            Some(End::Continue(l, mut vs)) => {
                self.subs(&mut vs);
                let carried = self.label(l)?;
                self.carry(&carried, &mut vs, &mut out)?;
                Some(End::Continue(l, vs))
            }
            other => other,
        };
        Ok(Region { nodes: out, end })
    }
}

/// Renumber the variables a region names by `keep`.
fn renumber(f: &mut Function, r: &mut Region, keep: &[Option<VarId>]) {
    let new = |v: &mut VarId| {
        if let Some(Some(n)) = keep.get(v.0 as usize) {
            *v = *n;
        }
    };
    for n in &mut r.nodes {
        match n {
            Node::Value(v) => match f.values.get_mut(v.0 as usize).map(|d| &mut d.op) {
                Some(Op::LoadVar(var) | Op::LoadElement(var, ..))
                | Some(Op::Atomic {
                    memory: Memory::Var(var, _),
                    ..
                }) => new(var),
                _ => {}
            },
            Node::StoreVar(var, _) | Node::StoreElement(var, ..) => new(var),
            _ => {}
        }
        for inner in n.regions_mut() {
            renumber(f, inner, keep);
        }
    }
}

/// Promote every private variable of a `Regions` function that is not an array, and that no atomic operation
/// reaches, to values; the variables that stay are renumbered in order. A graph body is refused: its variables are
/// promoted by the graph's own construction (block parameters), not here.
pub fn promote(f: &mut Function) -> Result<(), String> {
    let Body::Regions(body) = core::mem::replace(&mut f.body, Body::Regions(Region::default())) else {
        return Err("promote takes a Regions body".into());
    };
    let mut atomic = BTreeSet::new();
    atomic_vars(f, &body, &mut atomic);
    let mut keep = Vec::new();
    let mut next = 0u32;
    for (v, i) in f.vars.iter().zip(0u32..) {
        let stays = v.len.is_some() || v.space != Space::Private || atomic.contains(&VarId(i));
        keep.push(stays.then(|| {
            let n = VarId(next);
            next = next.saturating_add(1);
            n
        }));
    }
    let mut p = Promote {
        env: alloc::vec![None; f.vars.len()],
        subst: alloc::vec![None; f.values.len()],
        labels: Vec::new(),
        keep: keep.clone(),
        f,
    };
    let mut body = p.region(body, &[])?;
    renumber(f, &mut body, &keep);
    f.body = Body::Regions(body);
    let vars = core::mem::take(&mut f.vars);
    f.vars = vars
        .into_iter()
        .zip(&keep)
        .filter(|(_, k)| k.is_some())
        .map(|(v, _)| v)
        .collect();
    Ok(())
}
