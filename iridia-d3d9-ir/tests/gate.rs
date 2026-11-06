#![allow(
    clippy::unwrap_used,
    reason = "a real-data gate: a broken input should stop it"
)]
//! The migration gate of shader model 3 on the IR, over every distinct vertex and pixel program of Shaun White
//! Skateboarding's three SDB2 bundles (the oracle set `d3d-disassembler/shaun-white/sdb2`), against `iridia-d3d9`, the
//! naga-based lifter it replaces:
//!
//! - each program lifts, IR-validates, prints, parses back to the same text, lowers and writes WGSL for browsers;
//! - its interface equals the older lifter's, and so do its WGSL declarations (structures, bindings, globals, the
//!   entry point's signature);
//! - it runs four ways on the same seeded inputs, constants and texels, and the output bits agree: A, the older lift
//!   in `iridia-exec`; B, the IR in `iridia-exec-ir`; C, the new WGSL read back by naga's front end in `iridia-exec`;
//!   D, the IR compiled into `iridia-exec` itself (`ProgramShader::from_ir`);
//! - the harness sees a change: with each program's first float addition made a subtraction in B, A and B disagree on
//!   most programs.
//!
//! Every failure is counted by class and the test fails on any. Without the oracle set it prints SKIP and passes.
mod harness;
mod inputs;
mod tally;

use std::path::PathBuf;

use harness::{Rng, Texels, View, ir_texels, lanes_of, round_trip};
use inputs::{Game, OracleSet, oracle};
use iridia_exec::ProgramShader;
use iridia_ir::interface::{Interface, ResourceId, ResourceKind};
use tally::{Tally, class};

/// Seeded runs per program.
const RUNS: u64 = 16;

fn programs() -> Option<Vec<(String, Vec<u8>)>> {
    let folder = oracle(OracleSet::D3dDisassembler, Game::ShaunWhite, "sdb2")?;
    let mut paths: Vec<PathBuf> = std::fs::read_dir(&folder)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "vs" || e == "ps"))
        .collect();
    paths.sort();
    Some(
        paths
            .into_iter()
            .map(|p| {
                (
                    p.file_name().unwrap().to_string_lossy().into_owned(),
                    std::fs::read(&p).unwrap(),
                )
            })
            .collect(),
    )
}

/// The IR pipeline: lifted, promoted and simplified. The lift takes the older lifter's multiply rule, IEEE 754, so the
/// two mean the same; the default, zero wins, is held to Direct3D 9 on Windows by the vkd3d cases.
fn lifted(bytes: &[u8]) -> Result<iridia_ir::Program, String> {
    let options = iridia_d3d9_ir::Options {
        multiply: iridia_ir::MulZero::Ieee,
        ..Default::default()
    };
    let mut p = iridia_d3d9_ir::lift_with(bytes, &options).map_err(|e| format!("lift: {}", e.group()))?;
    iridia_passes::promote(&mut p.entry).map_err(|e| format!("promote: {e}"))?;
    iridia_passes::simplify(&mut p.entry);
    Ok(p)
}

/// Everything outside function bodies: the structures, the globals and the entry point's signature line.
fn declarations(wgsl: &str) -> String {
    let mut out = String::new();
    let mut depth = 0i64;
    for line in wgsl.lines() {
        let in_fn = depth > 0;
        if line.starts_with("fn ") || !in_fn {
            out.push_str(line);
            out.push('\n');
        }
        if line.starts_with("fn ") || in_fn {
            depth += line.matches('{').count() as i64 - line.matches('}').count() as i64;
        }
    }
    out
}

/// A run of the older lift (A) or of the WGSL read back (C) in `iridia-exec`.
fn run_old(
    shader: &ProgramShader,
    interface: &Interface,
    inputs: &[[u32; 4]],
    buffers: &[(u32, &[[u32; 4]])],
) -> Result<View, String> {
    harness::run_naga(shader, interface, inputs, buffers, &mut Texels)
}

/// A run of the IR (B) in `iridia-exec-ir`.
fn run_ir(
    p: &iridia_ir::Program,
    inputs: &[[u32; 4]],
    buffers: &[(ResourceId, &[[u32; 4]])],
) -> Result<View, String> {
    let mut sampler = ir_texels(&p.interface);
    harness::run_ir(
        p,
        inputs,
        &mut harness::Bound {
            buffers,
            state: &[],
            sampler: &mut sampler,
            denormals: iridia_ir::Denormals::Any,
        },
    )
}

/// What WGSL text cannot say, put back from the older module by position: a name ending in a digit (naga's namer
/// writes `cb0_` for `cb0`), and the entry point's argument and result member names and bindings (WGSL cannot say a
/// varying has no interpolation, so naga's front end supplies the default). Anything else that differs is left for
/// the executor to refuse or to compute differently.
fn normalise(m: &mut naga::Module, old: &naga::Module) -> Result<(), String> {
    for (_, g) in m.global_variables.iter_mut() {
        if let Some(n) = &mut g.name
            && n.len() > 1
            && n.ends_with('_')
            && n[..n.len() - 1].ends_with(|c: char| c.is_ascii_digit())
        {
            n.pop();
        }
    }
    let (Some(ep), Some(oe)) = (m.entry_points.first_mut(), old.entry_points.first()) else {
        return Err("no entry point".into());
    };
    if ep.function.arguments.len() != oe.function.arguments.len() {
        return Err("argument count".into());
    }
    for (a, o) in ep.function.arguments.iter_mut().zip(&oe.function.arguments) {
        a.name.clone_from(&o.name);
        a.binding.clone_from(&o.binding);
    }
    let (Some(r), Some(or)) = (&ep.function.result, &oe.function.result) else {
        return Err("no result".into());
    };
    let (rt, ort) = (r.ty, or.ty);
    let naga::TypeInner::Struct { members: om, .. } = &old.types[ort].inner else {
        return Err("the older result is not a structure".into());
    };
    let mut t = m.types[rt].clone();
    let naga::TypeInner::Struct { members, .. } = &mut t.inner else {
        return Err("the result is not a structure".into());
    };
    if members.len() != om.len() {
        return Err("member count".into());
    }
    for (a, o) in members.iter_mut().zip(om) {
        a.name.clone_from(&o.name);
        a.binding.clone_from(&o.binding);
    }
    t.name.clone_from(&old.types[ort].name);
    m.types.replace(rt, t);
    Ok(())
}

/// The program the new WGSL is, for `iridia-exec`: the older lift's, with its module read from the WGSL and each
/// resource bound to the global of its name.
fn from_wgsl(old: &iridia::Program, wgsl: &str) -> Result<iridia::Program, String> {
    let mut module = naga::front::wgsl::parse_str(wgsl).map_err(|e| format!("naga's WGSL front end: {e}"))?;
    normalise(&mut module, &old.module)?;
    let mut bindings = Vec::new();
    for r in &old.interface.resources {
        let d = r.provenance.d3d.unwrap();
        let name = match r.kind {
            ResourceKind::ConstantBuffer { .. } => format!("cb{}", d.register),
            ResourceKind::Texture { .. } => format!("t{}", d.register),
            ResourceKind::Sampler { .. } => format!("s{}", d.register),
            _ => return Err("an unexpected resource".into()),
        };
        let g = module
            .global_variables
            .iter()
            .find(|(_, g)| g.name.as_deref() == Some(&name))
            .map(|(h, _)| h)
            .ok_or_else(|| format!("no global {name}"))?;
        bindings.push((r.id, g));
    }
    let mut p = old.clone();
    p.module = module;
    p.bindings = bindings;
    Ok(p)
}

/// One seeded run's inputs, in the register view (A, C) and by slot (B), and its constant banks.
struct Inputs {
    registers: Vec<[u32; 4]>,
    slots: Vec<[u32; 4]>,
    banks: Vec<(ResourceId, u32, Vec<[u32; 4]>)>,
}

fn seeded(interface: &Interface, rng: &mut Rng) -> Inputs {
    let mut registers = vec![[0u32; 4]; 16];
    let mut slots = Vec::new();
    for v in &interface.inputs {
        let d = v.provenance.d3d.unwrap();
        let mut packed = [0u32; 4];
        for (k, l) in lanes_of(d.mask).into_iter().enumerate() {
            let x = rng.float().to_bits();
            registers[d.register as usize][l] = x;
            packed[k] = x;
        }
        slots.push(packed);
    }
    let mut banks = Vec::new();
    for r in &interface.resources {
        if let ResourceKind::ConstantBuffer { layout } = &r.kind {
            let rows = layout.size.div_ceil(16);
            let data = (0..rows)
                .map(|_| [(); 4].map(|_| rng.float().to_bits()))
                .collect();
            banks.push((r.id, r.provenance.d3d.unwrap().register, data));
        }
    }
    Inputs {
        registers,
        slots,
        banks,
    }
}

impl Inputs {
    fn by_register(&self) -> Vec<(u32, &[[u32; 4]])> {
        self.banks.iter().map(|(_, r, d)| (*r, d.as_slice())).collect()
    }
    fn by_id(&self) -> Vec<(ResourceId, &[[u32; 4]])> {
        self.banks.iter().map(|(i, _, d)| (*i, d.as_slice())).collect()
    }
}

const AGREE: &str = "A = B = C = D on every run";
const KILLED: &str = "A = B = C = D, killed in all four on every run (outputs never compared)";

/// One program through every check; the class it lands in and the example to show for it.
fn gate(name: &str, bytes: &[u8], runs: &mut (usize, usize)) -> (String, String) {
    let fail = |what: String| (class(&what), format!("{name}: {what}"));
    let new = match lifted(bytes) {
        Ok(p) => p,
        Err(e) => return fail(e),
    };
    if let Err(e) = iridia_ir::validate(&new) {
        return fail(format!("IR validation: {}", e[0]));
    }
    if let Err(e) = round_trip(&new.entry) {
        return fail(e);
    }
    let wgsl = match iridia_naga::wgsl(&new, &iridia_naga::Profile::BROWSER) {
        Ok(w) => w,
        Err(e) => return fail(format!("WGSL: {e}")),
    };
    let old = iridia_d3d9::lift(bytes).unwrap();
    if old.interface != new.interface {
        return fail("interface differs from iridia-d3d9's".into());
    }
    if declarations(&iridia::emit::wgsl(&old).unwrap()) != declarations(&wgsl) {
        return fail("WGSL declarations differ from iridia-d3d9's".into());
    }
    let c = match from_wgsl(&old, &wgsl) {
        Ok(p) => p,
        Err(e) => return fail(format!("C not built: {e}")),
    };
    let (sa, sc) = match (ProgramShader::new(old.clone()), ProgramShader::new(c)) {
        (Ok(a), Ok(c)) => (a, c),
        (a, c) => {
            return fail(format!(
                "not compiled for iridia-exec: A {:?}, C {:?}",
                a.err().map(|e| e.to_string()),
                c.err().map(|e| e.to_string())
            ));
        }
    };
    let sd = match ProgramShader::from_ir(new.clone()) {
        Ok(d) => d,
        Err(e) => return fail(format!("D not compiled: {e}")),
    };
    let interface = &old.interface;
    let mut rng = Rng::of(name);
    let mut compared = 0;
    for run in 0..RUNS {
        let i = seeded(interface, &mut rng);
        let ra = run_old(&sa, interface, &i.registers, &i.by_register());
        let rb = run_ir(&new, &i.slots, &i.by_id());
        let rc = run_old(&sc, interface, &i.registers, &i.by_register());
        let rd = run_old(&sd, interface, &i.registers, &i.by_register());
        let (ra, rb, rc, rd) = match (ra, rb, rc, rd) {
            (Ok(a), Ok(b), Ok(c), Ok(d)) => (a, b, c, d),
            (a, b, c, d) => {
                return fail(format!(
                    "not run: A {:?}, B {:?}, C {:?}, D {:?}",
                    a.err(),
                    b.err(),
                    c.err(),
                    d.err()
                ));
            }
        };
        if ra.killed && rb.killed && rc.killed && rd.killed {
            runs.1 += 1;
            continue;
        }
        runs.0 += 1;
        compared += 1;
        if !(ra.equal(&rb) && rb.equal(&rc) && ra.equal(&rd)) {
            return (
                format!(
                    "differ: A=B {} B=C {} A=C {} A=D {}",
                    ra.equal(&rb),
                    rb.equal(&rc),
                    ra.equal(&rc),
                    ra.equal(&rd)
                ),
                format!("{name} run {run}\n   A {ra:?}\n   B {rb:?}\n   C {rc:?}\n   D {rd:?}"),
            );
        }
    }
    let outcome = if compared == 0 { KILLED } else { AGREE };
    (outcome.into(), name.into())
}

#[test]
fn every_program_lifts_round_trips_and_runs_as_iridia_d3d9s() {
    let Some(programs) = programs() else { return };
    let mut tally = Tally::default();
    let mut runs = (0, 0);
    for (name, bytes) in &programs {
        let (class, example) = gate(name, bytes, &mut runs);
        tally.note(class, example);
    }
    println!(
        "{} programs; runs compared {}, runs killed in all three (outputs not compared) {}",
        programs.len(),
        runs.0,
        runs.1
    );
    tally.print("outcomes");
    assert_eq!(tally.failures(&[AGREE, KILLED]), 0, "programs failed the gate");
    assert_eq!(programs.len(), 7_306);
}

/// The values a function's body places, in order.
fn placed(f: &iridia_ir::Function) -> Vec<iridia_ir::ValueId> {
    fn walk(r: &iridia_ir::Region, out: &mut Vec<iridia_ir::ValueId>) {
        for n in &r.nodes {
            match n {
                iridia_ir::Node::Value(v) => out.push(*v),
                iridia_ir::Node::If { accept, reject, .. } => {
                    walk(accept, out);
                    walk(reject, out);
                }
                iridia_ir::Node::Block { body, .. } | iridia_ir::Node::Loop { body, .. } => walk(body, out),
                _ => {}
            }
        }
    }
    let mut out = Vec::new();
    if let iridia_ir::Body::Regions(r) = &f.body {
        walk(r, &mut out);
    }
    out
}

#[test]
fn the_three_way_run_sees_an_addition_made_a_subtraction() {
    let Some(programs) = programs() else { return };
    let (mut with_addition, mut seen, mut seen_d) = (0usize, 0usize, 0usize);
    for (name, bytes) in &programs {
        let mut b = lifted(bytes).unwrap();
        let first = placed(&b.entry).into_iter().find(|v| {
            b.entry
                .value(*v)
                .is_some_and(|d| matches!(d.op, iridia_ir::Op::Binary(iridia_ir::Binary::FAdd, ..)))
        });
        let Some(v) = first else { continue };
        if let iridia_ir::Op::Binary(op, ..) = &mut b.entry.values[v.0 as usize].op {
            *op = iridia_ir::Binary::FSub;
        }
        with_addition += 1;
        let old = iridia_d3d9::lift(bytes).unwrap();
        let sa = ProgramShader::new(old.clone()).unwrap();
        let sd = ProgramShader::from_ir(b.clone()).unwrap();
        let mut rng = Rng::of(name);
        let (mut in_b, mut in_d) = (false, false);
        for _ in 0..RUNS {
            let i = seeded(&old.interface, &mut rng);
            let ra = run_old(&sa, &old.interface, &i.registers, &i.by_register()).unwrap();
            let rb = run_ir(&b, &i.slots, &i.by_id()).unwrap();
            let rd = run_old(&sd, &old.interface, &i.registers, &i.by_register()).unwrap();
            in_b |= !(ra.killed && rb.killed) && !ra.equal(&rb);
            in_d |= !(ra.killed && rd.killed) && !ra.equal(&rd);
        }
        seen += usize::from(in_b);
        seen_d += usize::from(in_d);
    }
    println!(
        "{seen} (B) and {seen_d} (D) of {with_addition} programs with a float addition differ once it is a subtraction"
    );
    assert!(seen_d * 4 >= with_addition * 3, "D: {seen_d} of {with_addition}");
    // the rest kill on every run or take the changed value through flow that does not see it on these inputs (an
    // absolute value, a maximum, a comparison into a select)
    assert!(seen * 4 >= with_addition * 3, "{seen} of {with_addition}");
}
