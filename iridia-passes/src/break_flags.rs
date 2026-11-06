//! Exits a target with innermost-only `break` and `continue` cannot take, routed through a jump variable: the form
//! naga and WGSL need, where `break` leaves the innermost loop or switch and `continue` the innermost loop. An exit is
//! left as it is where it can be taken directly: a `Br` to a loop or a switch with no other loop or switch between, a
//! `Continue` to a loop with no other loop between, and a `Br` to a block from the end of the block's own body (which
//! becomes its fall-off). Any other exit (a `Br` to a block from inside a construct within it, an exit across a loop
//! or a switch) is routed:
//!
//! - it sets the jump variable to its own number and leaves its innermost construct the ordinary way: off the end of
//!   an `If` arm or a block's body, or by a `Br` to its innermost loop;
//! - after each construct a route leaves, a guard tests the variable: a route that arrives clears it and takes the
//!   exit (falls off its block, or breaks or continues its loop, all from inside it); a route that does not leaves
//!   this construct the same way; with no route taken, the rest of the region runs.
//!
//! Values are carried through variables meanwhile (construct results and loop parameters are demoted first, and
//! promoted again after), so no route has to give a value to a construct it leaves. Control is changed only on the
//! routed paths; what each invocation computes is not.

use alloc::{collections::BTreeMap, format, string::String, vec, vec::Vec};

use iridia_ir::{
    Binary, Body, Cmp, End, Function, LabelId, Node, Op, Precision, Region, Type, ValueDef, ValueId, Var,
    VarId,
};

fn new_value(f: &mut Function, ty: Type, op: Op) -> Result<ValueId, String> {
    let id = ValueId(u32::try_from(f.values.len()).map_err(|_| String::from("more than 2^32 values"))?);
    f.values.push(ValueDef {
        ty,
        op,
        origin: None,
        precision: Precision::Full,
    });
    Ok(id)
}

fn new_var(f: &mut Function, name: String, ty: Type) -> Result<VarId, String> {
    let id = VarId(u32::try_from(f.vars.len()).map_err(|_| String::from("more than 2^32 variables"))?);
    f.vars.push(Var::new(&name, ty));
    Ok(id)
}

/// The variables each label's exits store into: its results, and for a loop its parameters.
#[derive(Clone)]
struct Carried {
    results: Vec<VarId>,
    params: Vec<VarId>,
}

/// Construct results and loop parameters made variables: each exit stores its values, the construct's results are
/// loaded after it and a loop's parameters at the start of its body. Exits then carry no values.
struct Demote<'a> {
    f: &'a mut Function,
    labels: BTreeMap<LabelId, Carried>,
}

impl Demote<'_> {
    fn vars_for(&mut self, vs: &[ValueId], what: &str) -> Result<Vec<VarId>, String> {
        vs.iter()
            .map(|v| {
                let ty = self.f.ty(*v).ok_or_else(|| format!("%{} is not a value", v.0))?;
                new_var(self.f, format!("{what}{}", v.0), ty)
            })
            .collect()
    }
    /// The construct's results defined after it, from their variables.
    fn loads(&mut self, results: &[ValueId], vars: &[VarId], out: &mut Vec<Node>) {
        for (&r, &v) in results.iter().zip(vars) {
            if let Some(d) = self.f.values.get_mut(r.0 as usize) {
                d.op = Op::LoadVar(v);
            }
            out.push(Node::Value(r));
        }
    }
    fn stores(vars: &[VarId], vs: &[ValueId], out: &mut Vec<Node>) {
        out.extend(vars.iter().zip(vs).map(|(&var, &v)| Node::StoreVar(var, v)));
    }
    /// A region whose yield stores into `yields`.
    fn region(&mut self, r: Region, yields: &[VarId]) -> Result<Region, String> {
        let mut out = Vec::new();
        for n in r.nodes {
            match n {
                Node::If {
                    cond,
                    accept,
                    reject,
                    results,
                } => {
                    let vars = self.vars_for(&results, "if")?;
                    let accept = self.region(accept, &vars)?;
                    let reject = self.region(reject, &vars)?;
                    out.push(Node::If {
                        cond,
                        accept,
                        reject,
                        results: Vec::new(),
                    });
                    self.loads(&results, &vars, &mut out);
                }
                Node::Block { label, body, results } => {
                    let vars = self.vars_for(&results, "block")?;
                    self.labels.insert(
                        label,
                        Carried {
                            results: vars.clone(),
                            params: Vec::new(),
                        },
                    );
                    let body = self.region(body, &vars)?;
                    out.push(Node::Block {
                        label,
                        body,
                        results: Vec::new(),
                    });
                    self.loads(&results, &vars, &mut out);
                }
                Node::Loop {
                    label,
                    params,
                    inits,
                    body,
                    results,
                } => {
                    let pvars = self.vars_for(&params, "param")?;
                    let rvars = self.vars_for(&results, "loop")?;
                    Self::stores(&pvars, &inits, &mut out);
                    self.labels.insert(
                        label,
                        Carried {
                            results: rvars.clone(),
                            params: pvars.clone(),
                        },
                    );
                    let mut body = self.region(body, &rvars)?;
                    let mut start = Vec::new();
                    self.loads(&params, &pvars, &mut start);
                    start.append(&mut body.nodes);
                    body.nodes = start;
                    out.push(Node::Loop {
                        label,
                        params: Vec::new(),
                        inits: Vec::new(),
                        body,
                        results: Vec::new(),
                    });
                    self.loads(&results, &rvars, &mut out);
                }
                Node::Switch {
                    label,
                    selector,
                    mut cases,
                    default,
                    results,
                } => {
                    let vars = self.vars_for(&results, "switch")?;
                    self.labels.insert(
                        label,
                        Carried {
                            results: vars.clone(),
                            params: Vec::new(),
                        },
                    );
                    for c in &mut cases {
                        let body = core::mem::take(&mut c.body);
                        c.body = self.region(body, &vars)?;
                    }
                    let default = self.region(default, &vars)?;
                    out.push(Node::Switch {
                        label,
                        selector,
                        cases,
                        default,
                        results: Vec::new(),
                    });
                    self.loads(&results, &vars, &mut out);
                }
                other => out.push(other),
            }
        }
        let end = match r.end {
            Some(End::Yield(vs)) => {
                Self::stores(yields, &vs, &mut out);
                Some(End::Yield(Vec::new()))
            }
            Some(End::Br(l, vs)) => {
                let c = self
                    .labels
                    .get(&l)
                    .cloned()
                    .ok_or_else(|| format!("an exit to label {}, unknown", l.0))?;
                Self::stores(&c.results, &vs, &mut out);
                Some(End::Br(l, Vec::new()))
            }
            Some(End::Continue(l, vs)) => {
                let c = self
                    .labels
                    .get(&l)
                    .cloned()
                    .ok_or_else(|| format!("an exit to label {}, unknown", l.0))?;
                Self::stores(&c.params, &vs, &mut out);
                Some(End::Continue(l, Vec::new()))
            }
            other => other,
        };
        Ok(Region { nodes: out, end })
    }
}

/// A construct being walked: what a region's own exits leave, and what a route may arrive at.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Frame {
    If,
    Block(LabelId),
    Loop(LabelId),
    Switch(LabelId),
}

impl Frame {
    fn label(self) -> Option<LabelId> {
        match self {
            Frame::If => None,
            Frame::Block(l) | Frame::Loop(l) | Frame::Switch(l) => Some(l),
        }
    }
    /// Whether a target's `break` leaves it.
    fn breakable(self) -> bool {
        matches!(self, Frame::Loop(_) | Frame::Switch(_))
    }
    fn of(n: &Node) -> Option<Frame> {
        match n {
            Node::If { .. } => Some(Frame::If),
            Node::Block { label, .. } => Some(Frame::Block(*label)),
            Node::Loop { label, .. } => Some(Frame::Loop(*label)),
            Node::Switch { label, .. } => Some(Frame::Switch(*label)),
            _ => None,
        }
    }
}

/// Whether an exit to the frame at `at` of `stack` is taken directly: a break of the innermost loop or switch, a
/// continue of the innermost loop.
fn direct(stack: &[Frame], at: usize, resume: bool) -> bool {
    let inner = stack.get(at.saturating_add(1)..).unwrap_or_default();
    match (stack.get(at), resume) {
        (Some(Frame::Loop(_)), true) => !inner.iter().any(|f| matches!(f, Frame::Loop(_))),
        (Some(f), false) if f.breakable() => !inner.iter().any(|f| f.breakable()),
        _ => false,
    }
}

/// A routed exit: its number in the jump variable, its target, and whether it continues the target.
#[derive(Clone, Copy)]
struct Route {
    id: u32,
    target: LabelId,
    resume: bool,
}

struct Flags<'a> {
    f: &'a mut Function,
    jump: VarId,
    routes: u32,
}

impl Flags<'_> {
    fn literal(&mut self, x: u32, out: &mut Vec<Node>) -> Result<ValueId, String> {
        let v = new_value(self.f, Type::U32, Op::Literal([u64::from(x), 0, 0, 0]))?;
        out.push(Node::Value(v));
        Ok(v)
    }
    /// A bool of whether the jump variable `cmp` the number.
    fn test(&mut self, cmp: Cmp, x: u32, out: &mut Vec<Node>) -> Result<ValueId, String> {
        let jump = new_value(self.f, Type::U32, Op::LoadVar(self.jump))?;
        out.push(Node::Value(jump));
        let x = self.literal(x, out)?;
        let c = new_value(self.f, Type::BOOL, Op::Binary(Binary::UCmp(cmp), jump, x))?;
        out.push(Node::Value(c));
        Ok(c)
    }
    fn set(&mut self, x: u32, out: &mut Vec<Node>) -> Result<(), String> {
        let v = self.literal(x, out)?;
        out.push(Node::StoreVar(self.jump, v));
        Ok(())
    }

    /// How a region of construct `k` leaves it the ordinary way.
    fn leave(k: Option<Frame>) -> Result<End, String> {
        match k {
            Some(Frame::Loop(l)) => Ok(End::Br(l, Vec::new())),
            Some(Frame::If | Frame::Block(_) | Frame::Switch(_)) => Ok(End::Yield(Vec::new())),
            None => Err("a route past the function's body".into()),
        }
    }

    /// The guard's routing arm in a region of construct `k`, for the routes `s` that left a construct in it: those
    /// that arrive at `k` take their exit, the rest leave `k`. Also the routes that leave.
    fn dispatch(&mut self, s: &[Route], k: Option<Frame>) -> Result<(Region, Vec<Route>), String> {
        let here = k.and_then(Frame::label);
        let (arriving, leaving): (Vec<Route>, Vec<Route>) = s.iter().partition(|r| Some(r.target) == here);
        let mut arm = Region {
            nodes: Vec::new(),
            end: Some(if leaving.is_empty() {
                End::Unreachable
            } else {
                Self::leave(k)?
            }),
        };
        for r in arriving.iter().rev() {
            let mut nodes = Vec::new();
            let cond = self.test(Cmp::Eq, r.id, &mut nodes)?;
            let mut take = Vec::new();
            self.set(0, &mut take)?;
            let end = match (k, r.resume) {
                (Some(Frame::Loop(l)), true) => End::Continue(l, Vec::new()),
                (Some(Frame::Loop(l)), false) => End::Br(l, Vec::new()),
                (Some(Frame::Block(_) | Frame::Switch(_)), false) => End::Yield(Vec::new()),
                _ => return Err("a route that arrives at a construct its exit cannot take".into()),
            };
            nodes.push(Node::If {
                cond,
                accept: Region {
                    nodes: take,
                    end: Some(end),
                },
                reject: arm,
                results: Vec::new(),
            });
            // an arrival at a block falls past the inner `If`, to the guard's end and off the block's body
            arm = Region {
                nodes,
                end: Some(End::Yield(Vec::new())),
            };
        }
        Ok((arm, leaving))
    }

    /// A region of the innermost frame on `stack` rewritten, and the routes that leave that frame's construct.
    fn region(&mut self, r: Region, stack: &mut Vec<Frame>) -> Result<(Region, Vec<Route>), String> {
        let k = stack.last().copied();
        let mut out = Vec::new();
        let mut nodes = r.nodes.into_iter();
        while let Some(n) = nodes.next() {
            let (node, left) = match n {
                Node::If {
                    cond,
                    accept,
                    reject,
                    results,
                } => {
                    stack.push(Frame::If);
                    let (accept, mut a) = self.region(accept, stack)?;
                    let (reject, b) = self.region(reject, stack)?;
                    stack.pop();
                    a.extend(b);
                    (
                        Node::If {
                            cond,
                            accept,
                            reject,
                            results,
                        },
                        a,
                    )
                }
                Node::Block { label, body, results } => {
                    stack.push(Frame::Block(label));
                    let (body, s) = self.region(body, stack)?;
                    stack.pop();
                    (Node::Block { label, body, results }, s)
                }
                Node::Loop {
                    label,
                    params,
                    inits,
                    body,
                    results,
                } => {
                    stack.push(Frame::Loop(label));
                    let (body, s) = self.region(body, stack)?;
                    stack.pop();
                    (
                        Node::Loop {
                            label,
                            params,
                            inits,
                            body,
                            results,
                        },
                        s,
                    )
                }
                Node::Switch {
                    label,
                    selector,
                    mut cases,
                    default,
                    results,
                } => {
                    stack.push(Frame::Switch(label));
                    let mut left = Vec::new();
                    for c in &mut cases {
                        let (body, s) = self.region(core::mem::take(&mut c.body), stack)?;
                        c.body = body;
                        left.extend(s);
                    }
                    let (default, s) = self.region(default, stack)?;
                    left.extend(s);
                    stack.pop();
                    (
                        Node::Switch {
                            label,
                            selector,
                            cases,
                            default,
                            results,
                        },
                        left,
                    )
                }
                other => (other, Vec::new()),
            };
            out.push(node);
            if left.is_empty() {
                continue;
            }
            // routes left the construct: the rest of the region runs only where none is taken
            let rest = Region {
                nodes: nodes.collect(),
                end: r.end,
            };
            let (rest, mut pending) = self.region(rest, stack)?;
            let (arm, leaving) = self.dispatch(&left, k)?;
            let cond = self.test(Cmp::Ne, 0, &mut out)?;
            out.push(Node::If {
                cond,
                accept: arm,
                reject: rest,
                results: Vec::new(),
            });
            pending.extend(leaving);
            return Ok((
                Region {
                    nodes: out,
                    end: Some(End::Yield(Vec::new())),
                },
                pending,
            ));
        }
        let mut pending = Vec::new();
        let end = match r.end {
            Some(End::Br(t, vs)) | Some(End::Continue(t, vs)) if !vs.is_empty() => {
                return Err(format!("an exit to label {} with values after demotion", t.0));
            }
            Some(e @ (End::Br(..) | End::Continue(..))) => {
                let (t, resume) = match &e {
                    End::Br(t, _) => (*t, false),
                    End::Continue(t, _) => (*t, true),
                    _ => return Err("an exit that is neither a break nor a continue".into()),
                };
                let at = stack
                    .iter()
                    .rposition(|f| f.label() == Some(t))
                    .ok_or_else(|| format!("an exit to label {}, which no enclosing construct has", t.0))?;
                let inner = stack.get(at.saturating_add(1)..).unwrap_or_default();
                if direct(stack, at, resume) {
                    Some(e)
                } else if inner.is_empty() && !resume {
                    // the end of the block's own body: falling off it is the exit
                    Some(End::Yield(Vec::new()))
                } else {
                    self.routes = self.routes.checked_add(1).ok_or("more than 2^32 routed exits")?;
                    let id = self.routes;
                    self.set(id, &mut out)?;
                    pending.push(Route {
                        id,
                        target: t,
                        resume,
                    });
                    Some(Self::leave(k)?)
                }
            }
            other => other,
        };
        Ok((Region { nodes: out, end }, pending))
    }
}

/// Whether any exit in `r` (within `stack`) needs routing.
fn needs(r: &Region, stack: &mut Vec<Frame>) -> bool {
    let inner = r.nodes.iter().any(|n| match Frame::of(n) {
        Some(frame) => {
            stack.push(frame);
            let x = n.regions().into_iter().any(|inner| needs(inner, stack));
            stack.pop();
            x
        }
        None => false,
    });
    inner
        || match &r.end {
            Some(e @ (End::Br(t, _) | End::Continue(t, _))) => {
                let resume = matches!(e, End::Continue(..));
                match stack.iter().rposition(|f| f.label() == Some(*t)) {
                    Some(at) => {
                        let between = stack.get(at.saturating_add(1)..).unwrap_or_default();
                        // a block's exit from the end of its own body is its fall-off
                        let tail = matches!(stack.get(at), Some(Frame::Block(_))) && between.is_empty() && !resume;
                        !direct(stack, at, resume) && !tail
                    }
                    None => false,
                }
            }
            _ => false,
        }
}

/// Each `Br` to a block from the end of the block's own body made its fall-off.
fn tails(r: &mut Region, stack: &mut Vec<Frame>) {
    for n in &mut r.nodes {
        if let Some(frame) = Frame::of(n) {
            stack.push(frame);
            for inner in n.regions_mut() {
                tails(inner, stack);
            }
            stack.pop();
        }
    }
    if let Some(End::Br(t, vs)) = &mut r.end
        && stack.last() == Some(&Frame::Block(*t))
    {
        r.end = Some(End::Yield(core::mem::take(vs)));
    }
}

/// Route the exits of a `Regions` function that innermost-only `break` and `continue` cannot take (see the module's
/// account); the number of routed exits. With none to route, only the block exits at the end of their own bodies
/// change (to fall-offs). A graph body is refused.
pub fn break_flags(f: &mut Function) -> Result<usize, String> {
    let Body::Regions(mut body) = core::mem::replace(&mut f.body, Body::Regions(Region::default())) else {
        return Err("break_flags takes a Regions body".into());
    };
    if !needs(&body, &mut Vec::new()) {
        tails(&mut body, &mut Vec::new());
        f.body = Body::Regions(body);
        return Ok(0);
    }
    let body = Demote {
        f,
        labels: BTreeMap::new(),
    }
    .region(body, &[])?;
    let jump = new_var(f, "jump".into(), Type::U32)?;
    let mut flags = Flags { f, jump, routes: 0 };
    let mut start = Vec::new();
    flags.set(0, &mut start)?;
    let (mut body, pending) = flags.region(body, &mut vec![])?;
    if !pending.is_empty() {
        return Err("a route past the function's body".into());
    }
    let routes = flags.routes;
    start.append(&mut body.nodes);
    body.nodes = start;
    f.body = Body::Regions(body);
    crate::promote(f)?;
    crate::simplify(f);
    usize::try_from(routes).map_err(|_| String::from("more routes than the address space"))
}
