//! The basic blocks as structured control flow: the control flow graph of the function, the structurizer
//! ([`iridia::build::structure`]) and the callbacks that write each block, edge and branch.

use iridia::build::structure::{self, Cfg, Emit, Term};
use iridia::build::{Builder, Kind};
use naga::{Expression, Handle, SwitchValue};

use super::Lift;
use super::setup::const_int;
use crate::{Error, Result, error};
use dxil_reader::module::{Instruction, Op};

/// The lowering's view of a lift: the builder the lowering writes into is swapped in for each callback.
struct Blocks<'a, 'm> {
    lift: &'a mut Lift<'m>,
}

impl Blocks<'_, '_> {
    fn with<R>(&mut self, b: &mut Builder, f: impl FnOnce(&mut Lift<'_>) -> R) -> R {
        std::mem::swap(&mut self.lift.b, b);
        let r = f(self.lift);
        std::mem::swap(&mut self.lift.b, b);
        r
    }
}

impl Emit for Blocks<'_, '_> {
    type Error = Error;

    fn node(&mut self, b: &mut Builder, node: usize) -> Result<()> {
        self.with(b, |lift| {
            for i in lift.block(node)?.to_vec() {
                let inst = lift.instruction_at(i)?;
                if matches!(inst.op, Op::Phi { .. }) {
                    continue;
                }
                lift.instruction(inst)?;
            }
            Ok(())
        })
    }

    fn edge(&mut self, b: &mut Builder, from: usize, to: usize) -> Result<()> {
        self.with(b, |lift| lift.phi_copies(from, to))
    }

    fn condition(&mut self, b: &mut Builder, node: usize) -> Result<Handle<Expression>> {
        self.with(b, |lift| {
            let inst = lift.terminator(node)?;
            lift.at = inst.at;
            let Op::CondBranch { condition, .. } = &inst.op else {
                return lift.refuse("a conditional branch was expected");
            };
            let c = lift.scalar(*condition)?;
            Ok(lift.b.bitcast_to(c, Kind::B).expr)
        })
    }

    fn selector(&mut self, b: &mut Builder, node: usize) -> Result<(Handle<Expression>, Vec<SwitchValue>)> {
        self.with(b, |lift| {
            let inst = lift.terminator(node)?;
            lift.at = inst.at;
            let Op::Switch { value, cases, .. } = &inst.op else {
                return lift.refuse("a switch was expected");
            };
            let selector = lift.uint(*value)?;
            let mut values = Vec::new();
            for (case, _) in cases {
                let Some(k) = const_int(lift.m, lift.f, *case) else {
                    return lift.refuse("a switch case that is not a constant");
                };
                values.push(SwitchValue::U32(super::low32(k as u64)));
            }
            Ok((selector.expr, values))
        })
    }

    fn exit(&mut self, b: &mut Builder, node: usize) -> Result<()> {
        self.with(b, |lift| {
            lift.at = lift.terminator(node)?.at;
            lift.epilogue()
        })
    }
}

impl<'m> Lift<'m> {
    /// The instructions of basic block `node`, by index into the function's.
    pub(super) fn block(&self, node: usize) -> Result<&[usize]> {
        self.blocks.get(node).map(Vec::as_slice).ok_or_else(|| {
            error(
                self.at,
                format!("basic block {node}, where the function has {}", self.blocks.len()),
            )
        })
    }

    /// Instruction `i` of the function.
    pub(super) fn instruction_at(&self, i: usize) -> Result<&'m Instruction> {
        self.f.instructions.get(i).ok_or_else(|| {
            error(
                self.at,
                format!(
                    "instruction {i}, where the function has {}",
                    self.f.instructions.len()
                ),
            )
        })
    }

    /// The last instruction of basic block `node`, which ends it.
    fn terminator(&self, node: usize) -> Result<&'m Instruction> {
        match self.block(node)?.last() {
            Some(&last) => self.instruction_at(last),
            None => Err(error(self.at, format!("basic block {node} is empty"))),
        }
    }

    /// The graph of the function's basic blocks.
    fn cfg(&self) -> Result<Cfg> {
        let mut terms = Vec::new();
        for (n, insts) in self.blocks.iter().enumerate() {
            let Some(&last) = insts.last() else {
                return Err(error(0, format!("basic block {n} is empty")));
            };
            let terminator = self.instruction_at(last)?;
            terms.push(match &terminator.op {
                Op::Return(_) | Op::Unreachable => Term::Exit,
                Op::Branch { target } => Term::Goto(*target),
                Op::CondBranch { then, otherwise, .. } => Term::If(*then, *otherwise),
                Op::Switch { default, cases, .. } => {
                    Term::Switch(cases.iter().map(|(_, t)| *t).chain([*default]).collect())
                }
                _ => {
                    return Err(error(terminator.at, "a block that does not end in a terminator"));
                }
            });
        }
        Ok(Cfg { terms })
    }

    pub(super) fn structure_and_lower(&mut self) -> Result<()> {
        let cfg = self.cfg()?;
        let code = structure::structure(&cfg).map_err(|e| error(0, e.to_string()))?;
        let u1 = self.b.ty_of(Kind::U, 1);
        let jump = self.b.local("jump", u1);
        // the lowering writes into a builder of its own; each callback swaps the lift's in
        let mut writing = Builder::new();
        std::mem::swap(&mut self.b, &mut writing);
        let result = structure::lower(&mut writing, &code, &mut Blocks { lift: self }, jump);
        std::mem::swap(&mut self.b, &mut writing);
        result
    }
}
