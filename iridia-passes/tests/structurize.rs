//! `specialise`, `structurize` and `break_flags` on graph bodies, checked by running each program before and after
//! in the IR executor, and by naga's validation of the module each lowers to.

#![allow(clippy::unwrap_used, reason = "hand-built graphs")]

mod support;

use iridia_ir::{
    Binary, Block, Body, Cmp, End, Function, Node, Op, Precision, Program, Scalar, Target, Terminator, Type,
    ValueDef, ValueId,
    interface::{Resource, ResourceId, ResourceKind, ResourceProvenance, ScalarType, StateParam},
};

/// A graph function being written by hand.
struct Graph {
    f: Function,
}

impl Graph {
    fn new() -> Graph {
        Graph {
            f: Function {
                name: "main".into(),
                values: Vec::new(),
                vars: Vec::new(),
                body: Body::Cfg(Vec::new()),
            },
        }
    }
    fn value(&mut self, ty: Type, op: Op) -> ValueId {
        let id = ValueId(u32::try_from(self.f.values.len()).unwrap());
        self.f.values.push(ValueDef {
            ty,
            op,
            origin: None,
            precision: Precision::Full,
        });
        id
    }
    /// A value placed in `nodes`.
    fn placed(&mut self, nodes: &mut Vec<Node>, ty: Type, op: Op) -> ValueId {
        let v = self.value(ty, op);
        nodes.push(Node::Value(v));
        v
    }
    fn param(&mut self, ty: Type) -> ValueId {
        self.value(ty, Op::Param)
    }
}

fn to(block: u32, args: Vec<ValueId>) -> Target {
    Target { block, args }
}

fn state(id: u32, param: StateParam) -> Resource {
    Resource {
        id: ResourceId(id),
        kind: ResourceKind::PipelineState(param),
        count: 1,
        provenance: ResourceProvenance {
            name: format!("{param:?}"),
            ..ResourceProvenance::default()
        },
    }
}

/// The one output's first lane after a run, or the run's refusal.
fn run(p: &Program, input: u32, state: &[(ResourceId, [u32; 4])]) -> Result<(u32, bool), String> {
    let (outputs, killed) = support::run(p, &[[input, 0, 0, 0]], state)?;
    Ok((outputs[0].map_or(u32::MAX, |l| l[0]), killed))
}

/// A pixel program of one input and one target of `scalar`.
fn program(entry: Function, scalar: ScalarType, resources: Vec<Resource>) -> Program {
    support::program(entry, &[(scalar, 1)], &[(scalar, 1)], resources)
}

/// Design §1.4's graph, which only specialisation makes reducible:
///
/// ```text
/// entry: if state -> A(0) else B(0)    A(i): j = i + 1; if j < t -> B(j) else exit(j)
/// B(k): m = k + 2; if m < t -> A(m) else exit(m)                exit(r): output r
/// ```
fn counterexample() -> Program {
    let mut g = Graph::new();
    let (f32t, boolt) = (Type::scalar(Scalar::F32), Type::BOOL);
    let mut entry = Vec::new();
    let s = g.placed(&mut entry, boolt, Op::LoadState(ResourceId(0)));
    let t = g.placed(&mut entry, f32t, Op::LoadInput(0, 0b1));
    let zero = g.placed(&mut entry, f32t, Op::Literal([0f32.to_bits().into(), 0, 0, 0]));
    let (one, two) = (
        g.placed(&mut entry, f32t, Op::Literal([1f32.to_bits().into(), 0, 0, 0])),
        g.placed(&mut entry, f32t, Op::Literal([2f32.to_bits().into(), 0, 0, 0])),
    );
    let i = g.param(f32t);
    let mut a = Vec::new();
    let j = g.placed(&mut a, f32t, Op::Binary(Binary::FAdd, i, one));
    let c = g.placed(&mut a, boolt, Op::Binary(Binary::FCmp(Cmp::Lt), j, t));
    let k = g.param(f32t);
    let mut b = Vec::new();
    let m = g.placed(&mut b, f32t, Op::Binary(Binary::FAdd, k, two));
    let d = g.placed(&mut b, boolt, Op::Binary(Binary::FCmp(Cmp::Lt), m, t));
    let r = g.param(f32t);
    g.f.body = Body::Cfg(vec![
        Block {
            params: vec![],
            nodes: entry,
            term: Terminator::Branch {
                cond: s,
                accept: to(1, vec![zero]),
                reject: to(2, vec![zero]),
            },
        },
        Block {
            params: vec![i],
            nodes: a,
            term: Terminator::Branch {
                cond: c,
                accept: to(2, vec![j]),
                reject: to(3, vec![j]),
            },
        },
        Block {
            params: vec![k],
            nodes: b,
            term: Terminator::Branch {
                cond: d,
                accept: to(1, vec![m]),
                reject: to(3, vec![m]),
            },
        },
        Block {
            params: vec![r],
            nodes: vec![Node::StoreOutput(0, 0b1, r)],
            term: Terminator::Return,
        },
    ]);
    program(g.f, ScalarType::Float, vec![state(0, StateParam::Projected(0))])
}

#[test]
fn the_counterexample_structures_once_specialised_and_keeps_its_values() {
    let p = counterexample();
    iridia_ir::validate(&p).unwrap();
    // the graph runs as it is, with the state either way: (t, state) -> the hand-worked result
    let worked = [
        (0.5f32, true, 1.0f32),
        (3.5, true, 4.0),
        (7.2, true, 9.0),
        (0.5, false, 2.0),
        (3.5, false, 5.0),
        (7.2, false, 8.0),
    ];
    for (t, s, want) in worked {
        let got = run(&p, t.to_bits(), &[(ResourceId(0), [u32::from(s), 0, 0, 0])]).unwrap();
        assert_eq!(got, (want.to_bits(), false), "t {t} state {s}");
    }
    // unknown state: the cycle {A, B} has two entries
    let mut unknown = p.clone();
    let e = iridia_passes::structurize(&mut unknown.entry).unwrap_err();
    assert!(e.contains("irreducible"), "{e}");
    // state = true: the entry goes to A only, and the graph is reducible
    let mut known = p.clone();
    iridia_passes::specialise(&mut known, &[(ResourceId(0), [1, 0, 0, 0])]).unwrap();
    assert!(known.interface.resources.is_empty());
    iridia_ir::validate(&known).unwrap();
    let mut structured = known.clone();
    iridia_passes::structurize(&mut structured.entry).unwrap();
    iridia_ir::validate(&structured)
        .unwrap_or_else(|e| panic!("{e:?}\n{}", iridia_ir::text::print(&structured.entry)));
    assert!(matches!(structured.entry.body, Body::Regions(_)));
    for (t, s, want) in worked.into_iter().filter(|w| w.1) {
        for q in [&known, &structured] {
            assert_eq!(
                run(q, t.to_bits(), &[]).unwrap(),
                (want.to_bits(), false),
                "t {t} state {s}"
            );
        }
    }
    // the exit block is dominated by the loop header A, so it is placed inside the loop: every exit is the loop's
    // own or the block's fall-off, and nothing is routed
    let mut flagged = structured.clone();
    assert_eq!(iridia_passes::break_flags(&mut flagged.entry).unwrap(), 0);
    assert_eq!(flagged, structured);
    let wgsl = iridia_naga::wgsl(&structured, &iridia_naga::Profile::NAGA)
        .unwrap_or_else(|e| panic!("{e}\n{}", iridia_ir::text::print(&structured.entry)));
    assert!(wgsl.contains("loop {"), "{wgsl}");
}

/// Nested loops whose inner loop leaves the outer one with a value:
///
/// ```text
/// r = loop L0 (i = 0) { loop L1 (j = 0) { s = 10 i + j; if s >= t { br L0 s }; if j >= 3 { br L1 } else
///     { continue L1 j + 1 } }; continue L0 i + 1 }
/// output r
/// ```
fn nested_loops() -> Program {
    let u = Type::U32;
    let mut b = iridia_ir::Builder::new("main");
    let t = b.value(u, Op::LoadInput(0, 0b1));
    let zero = b.u32(0);
    let (l0, p0) = b.open_loop(&[zero]).unwrap();
    let i = p0[0];
    let (l1, p1) = b.open_loop(&[zero]).unwrap();
    let j = p1[0];
    let ten = b.u32(10);
    let i10 = b.op(Op::Binary(Binary::IMul, i, ten)).unwrap();
    let s = b.op(Op::Binary(Binary::IAdd, i10, j)).unwrap();
    let c1 = b.op(Op::Binary(Binary::UCmp(Cmp::Ge), s, t)).unwrap();
    b.open_if(c1);
    b.else_(End::Br(l0, vec![s])).unwrap();
    b.close(End::Yield(vec![]), &[]).unwrap();
    let three = b.u32(3);
    let c2 = b.op(Op::Binary(Binary::UCmp(Cmp::Ge), j, three)).unwrap();
    b.open_if(c2);
    b.else_(End::Br(l1, vec![])).unwrap();
    let one = b.u32(1);
    let j1 = b.op(Op::Binary(Binary::IAdd, j, one)).unwrap();
    b.close(End::Continue(l1, vec![j1]), &[]).unwrap();
    b.close(End::Unreachable, &[]).unwrap();
    let one = b.u32(1);
    let i1 = b.op(Op::Binary(Binary::IAdd, i, one)).unwrap();
    let r = b.close(End::Continue(l0, vec![i1]), &[u]).unwrap()[0];
    b.store_output(0, 0b1, r);
    program(b.finish(End::Return).unwrap(), ScalarType::Uint, Vec::new())
}

#[test]
fn a_break_across_a_loop_is_routed_and_keeps_its_value() {
    let p = nested_loops();
    iridia_ir::validate(&p).unwrap_or_else(|e| panic!("{e:?}\n{}", iridia_ir::text::print(&p.entry)));
    let e = iridia_naga::wgsl(&p, &iridia_naga::Profile::NAGA).unwrap_err();
    assert!(e.0.contains("break_flags"), "{e}");
    let mut flagged = p.clone();
    assert_eq!(iridia_passes::break_flags(&mut flagged.entry).unwrap(), 1);
    iridia_ir::validate(&flagged)
        .unwrap_or_else(|e| panic!("{e:?}\n{}", iridia_ir::text::print(&flagged.entry)));
    // t -> the first 10 i + j at or past it, with j at most 3
    for (t, want) in [(0, 0), (12, 12), (21, 21), (35, 40), (4, 10)] {
        for q in [&p, &flagged] {
            assert_eq!(run(q, t, &[]).unwrap(), (want, false), "t {t}");
        }
        support::agrees_flat(&flagged, &[[t, 0, 0, 0]], &[]);
    }
    iridia_naga::wgsl(&flagged, &iridia_naga::Profile::NAGA)
        .unwrap_or_else(|e| panic!("{e}\n{}", iridia_ir::text::print(&flagged.entry)));
}

#[test]
fn specialising_a_resource_that_is_not_state_or_a_value_it_cannot_hold_is_refused() {
    let p = counterexample();
    let e = iridia_passes::specialise(&mut p.clone(), &[(ResourceId(7), [1, 0, 0, 0])]).unwrap_err();
    assert!(e.contains("not in the interface"), "{e}");
    let e = iridia_passes::specialise(&mut p.clone(), &[(ResourceId(0), [2, 0, 0, 0])]).unwrap_err();
    assert!(e.contains("cannot hold"), "{e}");
}

/// A deterministic generator for the random graphs.
struct Rng(u64);
impl Rng {
    fn below(&mut self, n: usize) -> usize {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 % n as u64) as usize
    }
}

/// A random graph of `n` nodes carrying values: each node `i` is a guard block `2i` (leave for the exit once the step
/// count reaches 40) and a body block `2i + 1` that mixes the hash with a value of the entry and its own number, then
/// branches on the hash's low bit, or switches on its low two bits (a stream of choices of its own, so the graphs
/// without switches are the same as before). The exit (block `2n`) outputs the hash.
fn random_graph(rng: &mut Rng) -> Program {
    let u = Type::scalar(Scalar::U32);
    let n = 2 + rng.below(7);
    let mut g = Graph::new();
    let lit = |g: &mut Graph, nodes: &mut Vec<Node>, x: u32| {
        g.placed(nodes, u, Op::Literal([u64::from(x), 0, 0, 0]))
    };
    let mut blocks: Vec<Block> = Vec::new();
    let exit = u32::try_from(2 * n).unwrap();
    // the entry's own values, read by every block (they cross from it)
    let mut first = Vec::new();
    let seed = g.placed(&mut first, u, Op::LoadInput(0, 0b1));
    let k = {
        let seven = lit(&mut g, &mut first, 7);
        g.placed(&mut first, u, Op::Binary(Binary::IMul, seed, seven))
    };
    let mut first_nodes = Some(first);
    let mut switches = Rng(rng.0 ^ 0x5bd1_e995_d3a2_646c);
    for i in 0..n {
        let terms: (usize, usize, bool) = (rng.below(n), rng.below(n), rng.below(4) == 0);
        // guard
        let (h, steps) = (g.param(u), g.param(u));
        let mut gn = first_nodes.take().unwrap_or_default();
        let one = lit(&mut g, &mut gn, 1);
        let s2 = g.placed(&mut gn, u, Op::Binary(Binary::IAdd, steps, one));
        let limit = lit(&mut g, &mut gn, 40);
        let done = g.placed(&mut gn, Type::BOOL, Op::Binary(Binary::UCmp(Cmp::Ge), s2, limit));
        let body = u32::try_from(2 * i + 1).unwrap();
        blocks.push(Block {
            params: vec![h, steps],
            nodes: gn,
            term: Terminator::Branch {
                cond: done,
                accept: to(exit, vec![h]),
                reject: to(body, vec![h, s2]),
            },
        });
        // body
        let (bh, bs) = (g.param(u), g.param(u));
        let mut bn = Vec::new();
        let m = lit(&mut g, &mut bn, 31);
        let x = g.placed(&mut bn, u, Op::Binary(Binary::IMul, bh, m));
        let x = g.placed(&mut bn, u, Op::Binary(Binary::IAdd, x, k));
        let id = lit(&mut g, &mut bn, u32::try_from(i).unwrap() + 1);
        let h2 = g.placed(&mut bn, u, Op::Binary(Binary::Xor, x, id));
        let target = |t: usize| u32::try_from(2 * t).unwrap();
        let term = if terms.2 {
            Terminator::Jump(to(target(terms.0), vec![h2, bs]))
        } else if switches.below(3) == 0 {
            // two cases and the default, one way to the exit; a case value shares a target with another now and then
            let three = lit(&mut g, &mut bn, 3);
            let low = g.placed(&mut bn, u, Op::Binary(Binary::And, h2, three));
            let other = target(switches.below(n));
            Terminator::Switch {
                selector: low,
                cases: vec![
                    (0, to(target(terms.0), vec![h2, bs])),
                    (2, to(exit, vec![h2])),
                    (3, if switches.below(2) == 0 { to(target(terms.0), vec![h2, bs]) } else { to(other, vec![h2, bs]) }),
                ],
                default: to(target(terms.1), vec![h2, bs]),
            }
        } else {
            let one = lit(&mut g, &mut bn, 1);
            let zero = lit(&mut g, &mut bn, 0);
            let low = g.placed(&mut bn, u, Op::Binary(Binary::And, h2, one));
            let c = g.placed(&mut bn, Type::BOOL, Op::Binary(Binary::UCmp(Cmp::Eq), low, zero));
            // the second way leaves for the exit now and then, so most runs end before the guard
            if rng.below(3) == 0 {
                Terminator::Branch {
                    cond: c,
                    accept: to(target(terms.0), vec![h2, bs]),
                    reject: to(exit, vec![h2]),
                }
            } else {
                Terminator::Branch {
                    cond: c,
                    accept: to(target(terms.0), vec![h2, bs]),
                    reject: to(target(terms.1), vec![h2, bs]),
                }
            }
        };
        blocks.push(Block {
            params: vec![bh, bs],
            nodes: bn,
            term,
        });
    }
    // the entry block is node 0's guard: it is entered with the seed and no steps, through a block of its own
    let r = g.param(u);
    blocks.push(Block {
        params: vec![r],
        nodes: vec![Node::StoreOutput(0, 0b1, r)],
        term: Terminator::Return,
    });
    // block 0 has parameters, which an entry cannot: put a start block in front and shift every target by one
    for b in &mut blocks {
        b.term.targets_mut().into_iter().for_each(|t| t.block += 1);
    }
    let mut start_nodes = std::mem::take(&mut blocks[0].nodes);
    // the entry's values move to the start block; the guard keeps its own
    let split = start_nodes
        .iter()
        .position(|n| matches!(n, Node::Value(v) if *v == k))
        .unwrap()
        + 1;
    let guard_nodes = start_nodes.split_off(split);
    blocks[0].nodes = guard_nodes;
    let zero = g.placed(&mut start_nodes, u, Op::Literal([0, 0, 0, 0]));
    blocks.insert(
        0,
        Block {
            params: vec![],
            nodes: start_nodes,
            term: Terminator::Jump(to(1, vec![seed, zero])),
        },
    );
    g.f.body = Body::Cfg(blocks);
    program(g.f, ScalarType::Uint, Vec::new())
}

#[test]
fn random_graphs_keep_their_values_when_structured() {
    let mut rng = Rng(0x0dd_ba11_cafe_f00d);
    let (mut structured, mut refused, mut routed, mut switched) = (0, 0, 0, 0);
    for _ in 0..1500 {
        let p = random_graph(&mut rng);
        iridia_ir::validate(&p).unwrap_or_else(|e| panic!("{e:?}\n{}", iridia_ir::text::print(&p.entry)));
        let mut s = p.clone();
        match iridia_passes::structurize(&mut s.entry) {
            Ok(()) => structured += 1,
            Err(e) => {
                assert!(e.contains("irreducible"), "{e}");
                refused += 1;
                continue;
            }
        }
        iridia_ir::validate(&s).unwrap_or_else(|e| panic!("{e:?}\n{}", iridia_ir::text::print(&s.entry)));
        support::round_trip(&s.entry);
        if iridia_ir::text::print(&s.entry).contains("switch '") {
            switched += 1;
        }
        let mut b = s.clone();
        routed += iridia_passes::break_flags(&mut b.entry).unwrap();
        iridia_ir::validate(&b).unwrap_or_else(|e| panic!("{e:?}\n{}", iridia_ir::text::print(&b.entry)));
        for seed in [0u32, 1, 2, 3, 0x9e37_79b9, 12345] {
            let want = run(&p, seed, &[]);
            assert!(want.is_ok(), "{want:?}");
            assert_eq!(
                want,
                run(&s, seed, &[]),
                "seed {seed}\n{}",
                iridia_ir::text::print(&s.entry)
            );
            assert_eq!(
                want,
                run(&b, seed, &[]),
                "seed {seed}\n{}",
                iridia_ir::text::print(&b.entry)
            );
            support::agrees_flat(&b, &[[seed, 0, 0, 0]], &[]);
        }
        iridia_naga::wgsl(&b, &iridia_naga::Profile::NAGA)
            .unwrap_or_else(|e| panic!("{e}\n{}", iridia_ir::text::print(&b.entry)));
    }
    assert!(
        structured > 500 && refused > 0 && routed > 500 && switched > 200,
        "{structured} structured, {refused} refused, {routed} routed, {switched} with a switch"
    );
}
