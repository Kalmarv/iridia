//! A builder of `Regions` functions: values are placed in the region being filled, and constructs are opened and
//! closed in source order, the way a structured source reads.

use alloc::{format, string::String, vec, vec::Vec};

use crate::{
    Address, Barrier, Body, Case, End, Function, LabelId, Node, Op, Origin, OutOfBounds, Precision, Region, Res,
    Space, Type, ValueDef, ValueId, Var, VarId, derived_type,
};

/// An open construct.
#[derive(Debug)]
enum Frame {
    If {
        cond: ValueId,
        accept: Option<Region>,
    },
    Block {
        label: LabelId,
    },
    Loop {
        label: LabelId,
        params: Vec<ValueId>,
        inits: Vec<ValueId>,
    },
    Switch {
        label: LabelId,
        selector: ValueId,
        cases: Vec<Case>,
        default: Option<Region>,
        /// The values of the case being filled; `None` for the default; nothing before the first case.
        current: Option<Option<Vec<u32>>>,
    },
}

/// Builds a function in the `Regions` phase.
#[derive(Debug)]
pub struct Builder {
    f: Function,
    /// Each open construct with the region it is placed in.
    stack: Vec<(Frame, Region)>,
    /// The region being filled.
    current: Region,
    labels: u32,
    origin: Option<Origin>,
    precision: Precision,
}

impl Builder {
    /// A builder of a function named `name`.
    pub fn new(name: &str) -> Builder {
        Builder {
            f: Function {
                name: name.into(),
                values: Vec::new(),
                vars: Vec::new(),
                body: Body::Regions(Region::default()),
            },
            stack: Vec::new(),
            current: Region::default(),
            labels: 0,
            origin: None,
            precision: Precision::Full,
        }
    }

    /// The function so far (its values and variables).
    pub fn function(&self) -> &Function {
        &self.f
    }

    /// The type of a value made so far.
    pub fn ty(&self, v: ValueId) -> Option<Type> {
        self.f.ty(v)
    }

    /// Set the source origin of the values made from here on.
    pub fn at(&mut self, origin: Option<Origin>) {
        self.origin = origin;
    }

    /// Set the precision permission of the values made from here on.
    pub fn precision(&mut self, precision: Precision) {
        self.precision = precision;
    }

    /// A new variable.
    pub fn var(&mut self, name: &str, ty: Type) -> VarId {
        self.declare(Var::new(name, ty))
    }

    /// A new variable as stated (an array, workgroup memory, a constant table).
    pub fn declare(&mut self, var: Var) -> VarId {
        let id = VarId(u32::try_from(self.f.vars.len()).unwrap_or(u32::MAX));
        self.f.vars.push(var);
        id
    }

    /// A new array variable of `len` elements of `ty` in `space`, starting as `init` (empty for none).
    pub fn array(&mut self, name: &str, ty: Type, len: u32, space: Space, init: Vec<[u64; 4]>) -> VarId {
        self.declare(Var {
            name: name.into(),
            ty,
            len: Some(len),
            space,
            init,
        })
    }

    fn def(&mut self, ty: Type, op: Op) -> ValueId {
        let id = ValueId(u32::try_from(self.f.values.len()).unwrap_or(u32::MAX));
        self.f.values.push(ValueDef {
            ty,
            op,
            origin: self.origin,
            precision: self.precision,
        });
        id
    }

    /// A value of a stated type, placed in the current region.
    pub fn value(&mut self, ty: Type, op: Op) -> ValueId {
        let id = self.def(ty, op);
        self.current.nodes.push(Node::Value(id));
        id
    }

    /// A value whose type its operation gives, placed in the current region; an error names the rule the operands
    /// break, or says the operation's type must be stated.
    pub fn op(&mut self, op: Op) -> Result<ValueId, String> {
        match derived_type(&op, |v| self.f.ty(v))? {
            Some(ty) => Ok(self.value(ty, op)),
            None => Err(format!("{op:?} needs its type stated")),
        }
    }

    /// A literal of `ty`, each lane's bits in `lanes` (a scalar's in the first).
    pub fn literal(&mut self, ty: Type, lanes: [u64; 4]) -> ValueId {
        self.value(ty, Op::Literal(lanes))
    }

    /// An `f32` literal.
    pub fn f32(&mut self, x: f32) -> ValueId {
        self.literal(Type::F32, [u64::from(x.to_bits()), 0, 0, 0])
    }

    /// A `vec4<f32>` literal.
    pub fn f32x4(&mut self, x: [f32; 4]) -> ValueId {
        self.literal(Type::F32X4, x.map(|l| u64::from(l.to_bits())))
    }

    /// A `u32` literal.
    pub fn u32(&mut self, x: u32) -> ValueId {
        self.literal(Type::U32, [u64::from(x), 0, 0, 0])
    }

    /// An `i32` literal.
    pub fn i32(&mut self, x: i32) -> ValueId {
        self.literal(Type::I32, [u64::from(x.cast_unsigned()), 0, 0, 0])
    }

    /// A `bool` literal.
    pub fn bool(&mut self, x: bool) -> ValueId {
        self.literal(Type::BOOL, [u64::from(x), 0, 0, 0])
    }

    /// The variable takes the value.
    pub fn store_var(&mut self, var: VarId, value: ValueId) {
        self.current.nodes.push(Node::StoreVar(var, value));
    }

    /// Element `index` of an array variable takes the value.
    pub fn store_element(&mut self, var: VarId, index: ValueId, value: ValueId, oob: OutOfBounds) {
        self.current.nodes.push(Node::StoreElement(var, index, value, oob));
    }

    /// Words of a storage buffer take the value's lanes.
    pub fn store_buffer(&mut self, resource: Res, address: Address, value: ValueId) {
        self.current.nodes.push(Node::StoreBuffer {
            resource,
            address,
            value,
        });
    }

    /// A texel of a storage image or typed buffer takes the value.
    pub fn store_texel(&mut self, image: Res, coord: ValueId, value: ValueId) {
        self.current.nodes.push(Node::StoreTexel { image, coord, value });
    }

    /// Lanes `mask` of patch constant `slot` take the value.
    pub fn store_patch(&mut self, slot: u32, mask: u8, value: ValueId) {
        self.current.nodes.push(Node::StorePatch(slot, mask, value));
    }

    /// A workgroup barrier.
    pub fn barrier(&mut self, barrier: Barrier) {
        self.current.nodes.push(Node::Barrier(barrier));
    }

    /// Emit a vertex to stream `stream`.
    pub fn emit(&mut self, stream: u8) {
        self.current.nodes.push(Node::Emit(stream));
    }

    /// End the strip of stream `stream`.
    pub fn cut(&mut self, stream: u8) {
        self.current.nodes.push(Node::Cut(stream));
    }

    /// Place any node in the current region.
    pub fn node(&mut self, node: Node) {
        self.current.nodes.push(node);
    }

    /// Demote the invocation: a helper from here, its later output stores without effect ([`Node::Demote`]).
    pub fn demote(&mut self) {
        self.current.nodes.push(Node::Demote);
    }

    /// Lanes `mask` of the interface output `slot` take the value.
    pub fn store_output(&mut self, slot: u32, mask: u8, value: ValueId) {
        self.current.nodes.push(Node::StoreOutput(slot, mask, value));
    }

    fn label(&mut self) -> LabelId {
        let l = LabelId(self.labels);
        self.labels = self.labels.saturating_add(1);
        l
    }

    fn open(&mut self, frame: Frame) {
        let outer = core::mem::take(&mut self.current);
        self.stack.push((frame, outer));
    }

    /// Open an `If` on a scalar bool: what follows is its accept arm.
    pub fn open_if(&mut self, cond: ValueId) {
        self.open(Frame::If { cond, accept: None });
    }

    /// End the accept arm of the innermost open `If` with `end`: what follows is its reject arm.
    pub fn else_(&mut self, end: End) -> Result<(), String> {
        self.current.end = Some(end);
        let arm = core::mem::take(&mut self.current);
        match self.stack.last_mut() {
            Some((
                Frame::If {
                    accept: accept @ None,
                    ..
                },
                _,
            )) => {
                *accept = Some(arm);
                Ok(())
            }
            _ => Err("an else with no open if".into()),
        }
    }

    /// Open a `Block`: its label is what a `Br` names to leave it.
    pub fn open_block(&mut self) -> LabelId {
        let label = self.label();
        self.open(Frame::Block { label });
        label
    }

    /// Open a `Loop` whose parameters start as `inits`: its label and the parameters, visible in its body.
    pub fn open_loop(&mut self, inits: &[ValueId]) -> Result<(LabelId, Vec<ValueId>), String> {
        let label = self.label();
        let mut params = Vec::new();
        for i in inits {
            let ty = self
                .f
                .ty(*i)
                .ok_or_else(|| format!("loop initial value %{} does not exist", i.0))?;
            params.push(self.def(ty, Op::Param));
        }
        self.open(Frame::Loop {
            label,
            params: params.clone(),
            inits: inits.to_vec(),
        });
        Ok((label, params))
    }

    /// Open a `Switch` on `selector`: its label is what a `Br` names to leave it. Its regions follow, each started by
    /// [`Builder::case`].
    pub fn open_switch(&mut self, selector: ValueId) -> LabelId {
        let label = self.label();
        self.open(Frame::Switch {
            label,
            selector,
            cases: Vec::new(),
            default: None,
            current: None,
        });
        label
    }

    /// In the innermost open `Switch`, end the region being filled with `end` (ignored before the first case) and start
    /// the region of the case of `values`, or of the default for `None`.
    pub fn case(&mut self, end: End, values: Option<Vec<u32>>) -> Result<(), String> {
        let region = core::mem::take(&mut self.current);
        let Some((
            Frame::Switch {
                cases,
                default,
                current,
                ..
            },
            _,
        )) = self.stack.last_mut()
        else {
            return Err("a case with no open switch".into());
        };
        match current.take() {
            None if !region.nodes.is_empty() => return Err("a switch with code before its first case".into()),
            None => {}
            Some(prev) => {
                let mut body = region;
                body.end = Some(end);
                match prev {
                    Some(values) => cases.push(Case { values, body }),
                    None => *default = Some(body),
                }
            }
        }
        if values.is_none() && default.is_some() {
            return Err("a switch with two defaults".into());
        }
        *current = Some(values);
        Ok(())
    }

    /// End the region being filled with `end` and close the innermost open construct, which defines values of the
    /// `results` types (given by its yields and breaks). An `If` with no else arm has a reject arm that yields
    /// nothing, so it can have no results.
    pub fn close(&mut self, end: End, results: &[Type]) -> Result<Vec<ValueId>, String> {
        self.current.end = Some(end);
        let body = core::mem::take(&mut self.current);
        let (mut frame, outer) = self.stack.pop().ok_or("a close with no open construct")?;
        let mut body = body;
        if let Frame::Switch {
            cases,
            default,
            current,
            ..
        } = &mut frame
        {
            // the region being filled is the last case's
            match current.take() {
                Some(Some(values)) => cases.push(Case { values, body }),
                Some(None) => *default = Some(body),
                None if body.nodes.is_empty() => {}
                None => return Err("a switch with code and no case".into()),
            }
            body = Region::default();
        }
        self.current = outer;
        let results: Vec<ValueId> = results.iter().map(|t| self.def(*t, Op::Param)).collect();
        let node = match frame {
            Frame::If {
                cond,
                accept: Some(accept),
            } => Node::If {
                cond,
                accept,
                reject: body,
                results: results.clone(),
            },
            Frame::If { cond, accept: None } => {
                if !results.is_empty() {
                    return Err("an if with results and no else arm".into());
                }
                let reject = Region {
                    nodes: Vec::new(),
                    end: Some(End::Yield(vec![])),
                };
                Node::If {
                    cond,
                    accept: body,
                    reject,
                    results: results.clone(),
                }
            }
            Frame::Block { label } => Node::Block {
                label,
                body,
                results: results.clone(),
            },
            Frame::Loop { label, params, inits } => Node::Loop {
                label,
                params,
                inits,
                body,
                results: results.clone(),
            },
            Frame::Switch {
                label,
                selector,
                cases,
                default,
                ..
            } => {
                let default = match default {
                    Some(d) => d,
                    None if results.is_empty() => Region {
                        nodes: Vec::new(),
                        end: Some(End::Yield(vec![])),
                    },
                    None => return Err("a switch with results and no default".into()),
                };
                let _ = body;
                Node::Switch {
                    label,
                    selector,
                    cases,
                    default,
                    results: results.clone(),
                }
            }
        };
        self.current.nodes.push(node);
        Ok(results)
    }

    /// Whether a construct is open.
    pub fn is_open(&self) -> bool {
        !self.stack.is_empty()
    }

    /// End the body with `end` and give the function.
    pub fn finish(mut self, end: End) -> Result<Function, String> {
        if self.is_open() {
            return Err(format!("{} constructs left open", self.stack.len()));
        }
        self.current.end = Some(end);
        self.f.body = Body::Regions(self.current);
        Ok(self.f)
    }
}
