//! A `Regions` body made a `Cfg`: the form a graph-shaped source lifts to, for a structured program to take the same
//! way through [`structurize`](crate::structurize) as one.
//!
//! Each construct becomes blocks. An `If` branches to an arm block each and joins at a block whose parameters are its
//! results; a `Block` continues in the block it starts in and leaves to an exit block, whose parameters are its
//! results; a `Loop` jumps from the block it starts in to a header block, whose parameters are the loop's, and leaves
//! to an exit block. A yield jumps to its construct's join or exit, a `Br` to its label's exit and a `Continue` to its
//! label's header, each with its values as the target's arguments. A `Switch` branches to a block per case and its
//! default, which join (and its `Br`s leave) to a block whose parameters are its results. A value defined in a region is visible only in it
//! and the regions it holds, whose blocks its block dominates, so no value needs moving.

use alloc::{collections::BTreeMap, format, string::String, vec::Vec};

use iridia_ir::{Block, Body, End, Function, LabelId, Node, Region, Target, Terminator, ValueId};

/// Where an exit to a label goes: its exit block, and its header block for a loop.
#[derive(Clone, Copy)]
struct Exits {
    exit: u32,
    header: Option<u32>,
}

struct Flat {
    blocks: Vec<Block>,
    labels: BTreeMap<LabelId, Exits>,
    /// The block being filled.
    at: usize,
}

impl Flat {
    /// A new block with these parameters, ending (until it is given its end) as unreachable.
    fn block(&mut self, params: Vec<ValueId>) -> Result<u32, String> {
        let id = u32::try_from(self.blocks.len()).map_err(|_| String::from("more than 4294967295 blocks"))?;
        self.blocks.push(Block {
            params,
            nodes: Vec::new(),
            term: Terminator::Unreachable,
        });
        Ok(id)
    }

    fn current(&mut self) -> Result<&mut Block, String> {
        self.blocks
            .get_mut(self.at)
            .ok_or_else(|| String::from("the block being filled is missing"))
    }

    /// End the block being filled with `term`, and continue in `next`.
    fn end(&mut self, term: Terminator, next: u32) -> Result<(), String> {
        self.current()?.term = term;
        self.at = next as usize;
        Ok(())
    }

    fn exits(&self, label: LabelId) -> Result<Exits, String> {
        self.labels
            .get(&label)
            .copied()
            .ok_or_else(|| format!("an exit to label {}, which no construct around it has", label.0))
    }

    /// Fill blocks from the block being filled with `r`, a yield going to `join` (none: the function's body, where
    /// falling off the end returns). The block being filled afterwards is one nothing reaches, or a construct's join.
    fn region(&mut self, r: Region, join: Option<u32>) -> Result<(), String> {
        for node in r.nodes {
            match node {
                Node::Switch {
                    label,
                    selector,
                    cases,
                    default,
                    results,
                } => {
                    let after = self.block(results)?;
                    self.labels.insert(
                        label,
                        Exits {
                            exit: after,
                            header: None,
                        },
                    );
                    let to = |block| Target {
                        block,
                        args: Vec::new(),
                    };
                    let mut arms = Vec::new();
                    let mut targets = Vec::new();
                    for c in cases {
                        let b = self.block(Vec::new())?;
                        targets.extend(c.values.iter().map(|&v| (v, to(b))));
                        arms.push((b, c.body));
                    }
                    let d = self.block(Vec::new())?;
                    arms.push((d, default));
                    self.end(
                        Terminator::Switch {
                            selector,
                            cases: targets,
                            default: to(d),
                        },
                        d,
                    )?;
                    for (b, body) in arms {
                        self.at = b as usize;
                        self.region(body, Some(after))?;
                    }
                    self.at = after as usize;
                }
                Node::If {
                    cond,
                    accept,
                    reject,
                    results,
                } => {
                    let after = self.block(results)?;
                    let (a, b) = (self.block(Vec::new())?, self.block(Vec::new())?);
                    let to = |block| Target {
                        block,
                        args: Vec::new(),
                    };
                    self.end(
                        Terminator::Branch {
                            cond,
                            accept: to(a),
                            reject: to(b),
                        },
                        a,
                    )?;
                    self.region(accept, Some(after))?;
                    self.at = b as usize;
                    self.region(reject, Some(after))?;
                    self.at = after as usize;
                }
                Node::Block { label, body, results } => {
                    let exit = self.block(results)?;
                    self.labels.insert(label, Exits { exit, header: None });
                    self.region(body, Some(exit))?;
                    self.at = exit as usize;
                }
                Node::Loop {
                    label,
                    params,
                    inits,
                    body,
                    results,
                } => {
                    let exit = self.block(results)?;
                    let header = self.block(params)?;
                    self.labels.insert(
                        label,
                        Exits {
                            exit,
                            header: Some(header),
                        },
                    );
                    self.end(
                        Terminator::Jump(Target {
                            block: header,
                            args: inits,
                        }),
                        header,
                    )?;
                    self.region(body, Some(exit))?;
                    self.at = exit as usize;
                }
                other => self.current()?.nodes.push(other),
            }
        }
        let term = match r.end {
            Some(End::Yield(args)) => match join {
                Some(block) => Terminator::Jump(Target { block, args }),
                None => Terminator::Return,
            },
            Some(End::Br(label, args)) => Terminator::Jump(Target {
                block: self.exits(label)?.exit,
                args,
            }),
            Some(End::Continue(label, args)) => Terminator::Jump(Target {
                block: self
                    .exits(label)?
                    .header
                    .ok_or_else(|| format!("a continue of label {}, which is not a loop's", label.0))?,
                args,
            }),
            Some(End::Return) => Terminator::Return,
            Some(End::Kill) => Terminator::Kill,
            Some(End::IgnoreHit) => Terminator::IgnoreHit,
            Some(End::AcceptHitAndEndSearch) => Terminator::AcceptHitAndEndSearch,
            Some(End::Unreachable) => Terminator::Unreachable,
            None => return Err("a region with no end".into()),
        };
        // the rest of an enclosing region, until its own end, is in a block nothing reaches
        let rest = self.block(Vec::new())?;
        self.end(term, rest)
    }
}

/// Make `f`'s body a `Cfg` if it is `Regions` (a `Cfg` body is left as it is). Refused: an exit to a label no
/// construct around it has, and a region with no end, which a valid function does not have.
pub fn flatten(f: &mut Function) -> Result<(), String> {
    let r = match core::mem::replace(&mut f.body, Body::Cfg(Vec::new())) {
        Body::Regions(r) => r,
        graph => {
            f.body = graph;
            return Ok(());
        }
    };
    let mut flat = Flat {
        blocks: Vec::new(),
        labels: BTreeMap::new(),
        at: 0,
    };
    flat.block(Vec::new())?;
    flat.region(r, None)?;
    f.body = Body::Cfg(flat.blocks);
    Ok(())
}
