//! A node of a program's value graph ([`iridia::flow::Values`]) evaluated from the words of the leaves it reads, with the
//! executor's own operations: what a run of the program computes for that lane where it is given the same leaves, bit
//! for bit. The caller answers the leaves (an input lane, a supplied value, a global's word, a texel, a query) or leaves
//! them unknown; a node that needs an unknown leaf is unknown, naming it. A select whose condition is unknown is known
//! where both its values are and are the same bits.
//!
//! Refused with an error, never guessed: an operation the graph states no kind for, a derivative, a subgroup operation,
//! a loop's value, a merge whose arms differ (the graph does not state which condition picks which arm), an element
//! picked by a run-time index ([`Op::Index`]: the graph does not state which candidate an index picks), a math function
//! that mixes lanes other than a dot product, and an operation the executor refuses.
use std::collections::HashMap;

use iridia::flow::{At, MathFunction, Node, NodeId, Op, ScalarKind, Values};

use crate::compile::{Kind, binary_op, convert_op, kind_of, math_op, relational_op, unary_op};
use crate::flat::MathOp;
use crate::run::{bin, convert, math, unary};

/// What a node evaluates to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Evaluated {
    /// The lane's word.
    Word(u32),
    /// The leaf whose word is not known (the caller's answer was `None`, or a value the leaf is read at is unknown).
    Unknown(NodeId),
}

/// Evaluates nodes of one value graph, keeping what it has evaluated: the leaves are asked once each.
pub struct Evaluator<'v, L> {
    values: &'v Values,
    flush_denormals: bool,
    leaf: L,
    known: HashMap<NodeId, Evaluated>,
}

impl<L> std::fmt::Debug for Evaluator<'_, L> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Evaluator")
            .field("flush_denormals", &self.flush_denormals)
            .field("known", &self.known.len())
            .finish_non_exhaustive()
    }
}

/// One step of the walk: the node's result, or the operands it needs first.
enum Step {
    Done(Evaluated),
    Need(Vec<NodeId>),
}

impl<'v, L: FnMut(NodeId, &Node, &[u32]) -> Option<u32>> Evaluator<'v, L> {
    /// An evaluator of `values`, whose program flushes subnormal floats to zero where `flush_denormals` (the lifted
    /// program's modes), asking `leaf(id, node, at)` for each leaf's word: `at` the words it is read at, a run-time
    /// indexed global's index values (outermost first) or a texel's coordinate then its controls, empty for the other
    /// leaves.
    pub fn new(values: &'v Values, flush_denormals: bool, leaf: L) -> Self {
        Evaluator {
            values,
            flush_denormals,
            leaf,
            known: HashMap::new(),
        }
    }

    /// The word node `id` evaluates to, or the leaf it needs that is unknown.
    pub fn evaluate(&mut self, id: NodeId) -> Result<Evaluated, String> {
        let mut stack = vec![id];
        while let Some(&top) = stack.last() {
            if self.known.contains_key(&top) {
                stack.pop();
                continue;
            }
            match self.step(top)? {
                Step::Done(v) => {
                    self.known.insert(top, v);
                    stack.pop();
                }
                // a node may be pushed again while an earlier push of it waits; the later one is evaluated first and
                // the earlier then found known. The graph's only cycles are loops' values, refused before their
                // operands are asked for, so the walk ends
                Step::Need(more) => stack.extend(more),
            }
        }
        self.known
            .get(&id)
            .copied()
            .ok_or_else(|| "the walk ended without the node's value".into())
    }

    /// The words of `ids`, or the first unknown leaf among them; `None` while some are not evaluated yet.
    fn words(&self, ids: &[NodeId]) -> Option<Result<Vec<u32>, NodeId>> {
        let mut out = Vec::with_capacity(ids.len());
        for id in ids {
            match self.known.get(id)? {
                Evaluated::Word(w) => out.push(*w),
                Evaluated::Unknown(leaf) => return Some(Err(*leaf)),
            }
        }
        Some(Ok(out))
    }

    /// The operands of `ids` not yet evaluated.
    fn pending(&self, ids: &[NodeId]) -> Vec<NodeId> {
        ids.iter()
            .filter(|id| !self.known.contains_key(id))
            .copied()
            .collect()
    }

    /// The kind operation `id` reads its operands in.
    fn reads(&self, id: NodeId) -> Result<Kind, String> {
        kind_of(
            self.values
                .reads(id)
                .ok_or_else(|| format!("an operation (node {id:?}) the graph states no kind for"))?,
        )
    }

    /// A leaf read at the words of `at`: unknown where one of them is (the leaf itself is what is not known).
    fn leaf(&mut self, id: NodeId, node: &Node, at: &[NodeId]) -> Step {
        match self.words(at) {
            None => Step::Need(self.pending(at)),
            Some(Err(_)) => Step::Done(Evaluated::Unknown(id)),
            Some(Ok(words)) => Step::Done(match (self.leaf)(id, node, &words) {
                Some(w) => Evaluated::Word(w),
                None => Evaluated::Unknown(id),
            }),
        }
    }

    fn step(&mut self, id: NodeId) -> Result<Step, String> {
        let node = self
            .values
            .get(id)
            .ok_or_else(|| format!("node {id:?} is not of this graph"))?
            .clone();
        let ftz = self.flush_denormals;
        Ok(match &node {
            Node::Input { .. } | Node::Supplied { .. } | Node::Query { .. } => self.leaf(id, &node, &[]),
            Node::Global { at, .. } => match at {
                At::Component(_) => self.leaf(id, &node, &[]),
                At::Indexed { index, .. } => self.leaf(id, &node, index),
            },
            Node::Texel {
                coordinate, controls, ..
            } => {
                let at: Vec<NodeId> = coordinate.iter().chain(controls).copied().collect();
                self.leaf(id, &node, &at)
            }
            Node::Literal(l) => Step::Done(Evaluated::Word(literal(l.0, self.values.kind(id))?)),
            Node::Loop { .. } => return Err("a loop's value is not evaluated".into()),
            Node::Merge { arms, .. } => match self.words(arms) {
                None => Step::Need(self.pending(arms)),
                Some(Err(leaf)) => Step::Done(Evaluated::Unknown(leaf)),
                Some(Ok(words))
                    if let Some(&first) = words.first()
                        && words.iter().all(|w| *w == first) =>
                {
                    Step::Done(Evaluated::Word(first))
                }
                Some(Ok(_)) => {
                    return Err(
                        "a merge whose arms differ: the graph does not state which condition picks which"
                            .into(),
                    );
                }
            },
            Node::Op { op, operands } => match op {
                Op::Select => {
                    let [condition, accept, reject] = operands[..] else {
                        return Err(format!("a select of {} operands", operands.len()));
                    };
                    match self.known.get(&condition) {
                        None => Step::Need(vec![condition]),
                        Some(Evaluated::Word(c)) => {
                            let chosen = if *c != 0 { accept } else { reject };
                            match self.known.get(&chosen) {
                                None => Step::Need(vec![chosen]),
                                Some(v) => Step::Done(*v),
                            }
                        }
                        // unknown which: known where both values are and are the same bits
                        Some(Evaluated::Unknown(leaf)) => {
                            let leaf = *leaf;
                            match self.words(&[accept, reject]) {
                                None => Step::Need(self.pending(&[accept, reject])),
                                Some(Ok(w))
                                    if let &[a, b] = w.as_slice()
                                        && a == b =>
                                {
                                    Step::Done(Evaluated::Word(a))
                                }
                                Some(_) => Step::Done(Evaluated::Unknown(leaf)),
                            }
                        }
                    }
                }
                _ => match self.words(operands) {
                    None => Step::Need(self.pending(operands)),
                    Some(Err(leaf)) => Step::Done(Evaluated::Unknown(leaf)),
                    Some(Ok(w)) => Step::Done(Evaluated::Word(self.operation(id, *op, &w, ftz)?)),
                },
            },
        })
    }

    /// Operation `op` of node `id` on the operand words `w`.
    fn operation(&self, id: NodeId, op: Op, w: &[u32], ftz: bool) -> Result<u32, String> {
        let lane = |i: usize| -> Result<u32, String> {
            w.get(i)
                .copied()
                .ok_or_else(|| format!("{op:?} with {} operands", w.len()))
        };
        Ok(match op {
            Op::Unary(u) => unary(unary_op(u, self.reads(id)?), lane(0)?),
            Op::Binary(b) => {
                if w.len() != 2 {
                    return Err("a matrix product is not evaluated".into());
                }
                bin(binary_op(b, self.reads(id)?)?, lane(0)?, lane(1)?, ftz)?
            }
            Op::Convert(to) => match convert_op(self.reads(id)?, to)? {
                Some(c) => convert(c, lane(0)?),
                None => lane(0)?,
            },
            Op::Math(fun) => {
                let m = math_op(fun, self.reads(id)?)?;
                match m {
                    MathOp::Dot => {
                        let n = w.len() / 2;
                        let dot = || format!("a dot product of {} operands", w.len());
                        if !w.len().is_multiple_of(2) || !(1..=4).contains(&n) {
                            return Err(dot());
                        }
                        let (a, b) = w.split_at_checked(n).ok_or_else(dot)?;
                        math(m, [reg(a), reg(b), [0; 4], [0; 4]], u8::try_from(n).map_err(|_| dot())?, ftz)[0]
                    }
                    MathOp::Normalize | MathOp::PackHalf | MathOp::UnpackHalf => {
                        return Err(format!("{fun:?} mixes lanes: not evaluated"));
                    }
                    _ if lanes_mixed(fun) => return Err(format!("{fun:?} mixes lanes: not evaluated")),
                    _ => {
                        let at = |i: usize| [w.get(i).copied().unwrap_or(0), 0, 0, 0];
                        math(m, [at(0), at(1), at(2), at(3)], 1, ftz)[0]
                    }
                }
            }
            Op::Relational(fun) => {
                let m = relational_op(fun);
                let lanes = u8::try_from(w.len())
                    .ok()
                    .filter(|n| (1..=4).contains(n))
                    .ok_or_else(|| format!("{fun:?} of {} lanes", w.len()))?;
                math(m, [reg(w), [0; 4], [0; 4], [0; 4]], lanes, ftz)[0]
            }
            Op::Select => {
                if lane(0)? != 0 {
                    lane(1)?
                } else {
                    lane(2)?
                }
            }
            Op::Index => {
                return Err("an element picked by a run-time index: the graph does not state which candidate an index picks".into());
            }
            Op::Derivative(..) => return Err("a derivative is not evaluated".into()),
            Op::Subgroup => return Err("a subgroup operation is not evaluated".into()),
        })
    }
}

/// Up to four words as a register.
fn reg(w: &[u32]) -> [u32; 4] {
    let mut r = [0; 4];
    for (out, v) in r.iter_mut().zip(w) {
        *out = *v;
    }
    r
}

/// Whether math function `fun` makes each lane of its result from every lane of its arguments (the value graph keeps one
/// node for all of them): the ones the executor runs that are not a dot product.
fn lanes_mixed(fun: MathFunction) -> bool {
    matches!(
        fun,
        MathFunction::Cross
            | MathFunction::Length
            | MathFunction::Distance
            | MathFunction::Outer
            | MathFunction::Determinant
            | MathFunction::Transpose
            | MathFunction::Inverse
            | MathFunction::Reflect
            | MathFunction::Refract
            | MathFunction::FaceForward
    )
}

/// A literal's word: its value as the scalar kind the graph states it of.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // each is checked to hold the value exactly
fn literal(v: f64, kind: Option<ScalarKind>) -> Result<u32, String> {
    let exact = |w: u32, back: f64| {
        if back.to_bits() == v.to_bits() || (v.is_nan() && back.is_nan()) {
            Ok(w)
        } else {
            Err(format!("literal {v} is no 32-bit {kind:?}"))
        }
    };
    match kind {
        Some(ScalarKind::Float) => {
            let f = v as f32;
            exact(f.to_bits(), f64::from(f))
        }
        Some(ScalarKind::Uint) => exact(v as u32, f64::from(v as u32)),
        Some(ScalarKind::Sint) => exact((v as i32) as u32, f64::from(v as i32)),
        Some(ScalarKind::Bool) => Ok(u32::from(v != 0.)),
        other => Err(format!("a literal of kind {other:?}")),
    }
}
