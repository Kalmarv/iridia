#![allow(clippy::unwrap_used)]
//! The IR lifter against the lifter it replaces: every DXBC container of a corpus is lifted both ways, and each
//! vertex or pixel program both lift is run in the executor on the same deterministic inputs, constant buffers and
//! textures. Column A is the old lift, B the IR lift as lifted, P the IR lift after `promote`, `simplify` and
//! `break_flags`. Every written output lane, the masks and the discard must agree bit for bit (NaNs alike); a
//! program the IR lifter refuses is counted by reason, and a program the old lifter lifts but the IR lifter does not
//! is a lifted-count regression.
//! Needs the corpus oracle set's `sekiro/shaders`, `riders/shaders` or `sledding/shaders`.
mod inputs;

use std::collections::BTreeMap;

use inputs::Game;
use iridia_exec::{
    Dim, ImageLoad, ImageLod, ImageQuery, ImageSample, Load, PixelShader, ProgramShader, QueryKind, Sample, Sampler,
};

mod common;

/// The vectors each program is run on.
const CASES: u32 = 4;

/// Answers every request from a hash of everything it states, so two programs that ask the same agree and two that
/// ask differently (another coordinate, offset, level, component) almost surely do not.
#[derive(Default)]
struct Hashed {
    asked: Vec<String>,
}

fn hash_words(words: &[u32]) -> u64 {
    words
        .iter()
        .fold(0x1234_5678u64, |h, &w| common::mix64(h ^ u64::from(w)))
}

fn dim_word(d: Dim) -> u32 {
    match d {
        Dim::D1 => 1,
        Dim::D2 => 2,
        Dim::D3 => 3,
        Dim::Cube => 4,
    }
}

fn texel_floats(h: u64) -> [f32; 4] {
    std::array::from_fn(|k| common::unit(common::mix(h ^ k as u64)) * 2.0 - 0.5)
}

impl Sampler for Hashed {
    fn sample(&mut self, r: Sample) -> Result<[f32; 4], String> {
        self.sample_image(ImageSample {
            texture: r.texture,
            sampler: r.sampler,
            dim: Dim::D2,
            coord: [r.uv[0], r.uv[1], 0.0],
            layer: r.layer,
            lod: ImageLod::from(r.lod),
            offset: [0; 3],
            gather: None,
            depth_ref: None,
            depth: false,
        })
    }
    fn sample_image(&mut self, r: ImageSample<u32>) -> Result<[f32; 4], String> {
        let mut w = vec![1, r.texture, r.sampler, dim_word(r.dim)];
        w.extend(r.coord.map(f32::to_bits));
        w.push(r.layer.map_or(u32::MAX, f32::to_bits));
        match r.lod {
            ImageLod::Implicit => w.push(10),
            ImageLod::Bias(b) => w.extend([11, b.to_bits()]),
            ImageLod::Level(l) => w.extend([12, l.to_bits()]),
            ImageLod::Gradients(g) => {
                w.push(13);
                w.extend(g.iter().flatten().map(|f| f.to_bits()));
            }
            ImageLod::GradientsBias(g, b) => {
                w.extend([14, b.to_bits()]);
                w.extend(g.iter().flatten().map(|f| f.to_bits()));
            }
        }
        w.extend(r.offset.map(i32::cast_unsigned));
        w.push(r.gather.map_or(u32::MAX, u32::from));
        w.push(r.depth_ref.map_or(u32::MAX, f32::to_bits));
        w.push(u32::from(r.depth));
        self.asked.push(format!("{r:?}"));
        let mut t = texel_floats(hash_words(&w));
        if r.depth || r.depth_ref.is_some() {
            // a depth texture's one lane; the rest as a depth view reads them
            t = [t[0].clamp(0.0, 1.0), 0.0, 0.0, 1.0];
        }
        Ok(t)
    }
    fn load(&mut self, r: Load) -> Result<[u32; 4], String> {
        self.load_image(ImageLoad {
            texture: r.texture,
            dim: Dim::D2,
            storage: false,
            texel: r.texel,
            layer: None,
            sample: None,
            mip: Some(r.mip),
        })
    }
    fn load_image(&mut self, r: ImageLoad<u32>) -> Result<[u32; 4], String> {
        let mut w = vec![2, r.texture, dim_word(r.dim), u32::from(r.storage)];
        w.extend(r.texel.map(i32::cast_unsigned));
        w.push(r.layer.map_or(u32::MAX, i32::cast_unsigned));
        w.push(r.sample.map_or(u32::MAX, i32::cast_unsigned));
        w.push(r.mip.map_or(u32::MAX, i32::cast_unsigned));
        self.asked.push(format!("{r:?}"));
        Ok(texel_floats(hash_words(&w)).map(f32::to_bits))
    }
    fn query_image(&mut self, r: ImageQuery<u32>) -> Result<[u32; 4], String> {
        let w = [
            3,
            r.texture,
            dim_word(r.dim),
            r.kind as u32,
            r.level.unwrap_or(u32::MAX),
        ];
        self.asked.push(format!("{r:?}"));
        let h = common::mix(hash_words(&w));
        Ok(match r.kind {
            QueryKind::Size => {
                let shift = r.level.unwrap_or(0).min(8);
                [(256 >> shift).max(1), (128 >> shift).max(1), 4, 0]
            }
            QueryKind::Levels => [9, 0, 0, 0],
            QueryKind::Layers => [h % 6 + 1, 0, 0, 0],
            QueryKind::Samples => [4, 0, 0, 0],
        })
    }
    fn structured(&mut self, slot: u32, element: u32, offset: u32) -> Result<u32, String> {
        self.asked.push(format!("structured t{slot}[{element}]+{offset}"));
        Ok(common::mix(hash_words(&[4, slot, element, offset])) % 1024)
    }
    fn buffer_elements(&mut self, slot: u32) -> Option<u32> {
        let _ = slot;
        Some(64)
    }
}

/// What a run gives: each output register's written lanes (unwritten lanes zero), the masks and the discard; or
/// the error.
type Run = Result<(Vec<[u32; 4]>, Vec<u8>, bool), String>;

fn run(shader: &ProgramShader, c: &common::Case) -> Run {
    run_asking(shader, c, &mut Hashed::default())
}

fn run_asking(shader: &ProgramShader, c: &common::Case, sampler: &mut Hashed) -> Run {
    let buffers = common::refs(c, shader);
    let e = shader
        .execute(&c.inputs, &buffers, sampler)
        .map_err(|e| e.to_string())?;
    let outputs = e
        .outputs
        .iter()
        .zip(&e.output_masks)
        .map(|(o, &m)| std::array::from_fn(|k| if m >> k & 1 != 0 { canonical(o[k]) } else { 0 }))
        .collect();
    Ok((outputs, e.output_masks, e.discarded))
}

/// A NaN as one bit pattern: two lifts may produce different NaNs.
fn canonical(w: u32) -> u32 {
    if f32::from_bits(w).is_nan() { 0x7fc0_0000 } else { w }
}

/// The IR lift's program after the passes the cut-over runs.
fn passed(mut p: iridia_ir::Program) -> Result<iridia_ir::Program, String> {
    iridia_passes::promote(&mut p.entry)?;
    iridia_passes::simplify(&mut p.entry);
    iridia_passes::break_flags(&mut p.entry)?;
    iridia_ir::validate(&p).map_err(|e| format!("{e:?}"))?;
    Ok(p)
}

/// One program's comparison over `cases` vectors.
enum Verdict {
    /// Every case ran in A and gave what B gave.
    Agree,
    /// A case A itself cannot evaluate (its sampler request, a step limit): nothing to compare.
    AFails(String),
    /// The first case where B gives something else.
    Differs(String),
}

fn compare(a: &ProgramShader, b: &ProgramShader, cases: u32) -> Verdict {
    for c in 0..cases {
        let case = common::case(a, c);
        let (x, y) = (run(a, &case), run(b, &case));
        if let Err(e) = &x {
            return Verdict::AFails(e.clone());
        }
        if x != y {
            return Verdict::Differs(format!("case {c}: A {x:x?}\n  but {y:x?}"));
        }
    }
    Verdict::Agree
}

#[derive(Default)]
struct Gate {
    programs: usize,
    /// Old-lift vertex and pixel programs the executor runs.
    a: usize,
    agree_b: usize,
    agree_p: usize,
    /// Programs the IR lifter refuses that A lifted, by reason, with the first.
    refused: BTreeMap<String, (usize, String)>,
    /// Programs B lifted the executor (or a pass) refuses, by reason.
    unrun: BTreeMap<String, (usize, String)>,
    /// Programs A cannot evaluate, by reason.
    a_fails: BTreeMap<String, (usize, String)>,
    /// Disagreements: program and the first difference.
    differ: Vec<String>,
    /// Programs the IR lifter lifts and A does not.
    only_ir: usize,
}

fn tally(map: &mut BTreeMap<String, (usize, String)>, why: &str, at: &str) {
    map.entry(common::reason(why)).or_insert((0, at.to_string())).0 += 1;
}

fn gate(game: Game) {
    let Some((name, paths)) = common::corpus(game) else {
        return;
    };
    let limit = common::diff_limit();
    let mut g = Gate::default();
    for path in paths.iter().take(limit) {
        let at = path.display().to_string();
        let bytes = std::fs::read(path).unwrap();
        g.programs += 1;
        if g.programs.is_multiple_of(500) {
            eprintln!("  {name}: {} programs", g.programs);
        }
        let old = iridia_dxbc::lift(&bytes);
        let new = iridia_dxbc::ir::lift(&bytes);
        let a = match old.ok().map(ProgramShader::new) {
            Some(Ok(a)) => a,
            _ => {
                if new.is_ok() {
                    g.only_ir += 1;
                }
                continue;
            }
        };
        g.a += 1;
        let p = match new {
            Ok(p) => p,
            Err(e) => {
                tally(&mut g.refused, &e.to_string(), &at);
                continue;
            }
        };
        let b = match ProgramShader::from_ir(p.clone()) {
            Ok(b) => b,
            Err(e) => {
                tally(&mut g.unrun, &format!("B: {e}"), &at);
                continue;
            }
        };
        match compare(&a, &b, CASES) {
            Verdict::Agree => g.agree_b += 1,
            Verdict::AFails(e) => {
                tally(&mut g.a_fails, &e, &at);
                continue;
            }
            Verdict::Differs(d) => g.differ.push(format!("{at} (B)\n  {d}")),
        }
        let pp = match passed(p).and_then(|p| ProgramShader::from_ir(p).map_err(|e| e.to_string())) {
            Ok(pp) => pp,
            Err(e) => {
                tally(&mut g.unrun, &format!("P: {e}"), &at);
                continue;
            }
        };
        match compare(&a, &pp, CASES) {
            Verdict::Agree => g.agree_p += 1,
            Verdict::AFails(e) => {
                tally(&mut g.a_fails, &e, &at);
                continue;
            }
            Verdict::Differs(d) => g.differ.push(format!("{at} (P)\n  {d}")),
        }
    }
    println!(
        "{name}: {} programs, A runs {}, B agrees {}, P agrees {}, {} differ, IR refuses {}, only the IR lifts {}",
        g.programs,
        g.a,
        g.agree_b,
        g.agree_p,
        g.differ.len(),
        g.refused.values().map(|(n, _)| n).sum::<usize>(),
        g.only_ir,
    );
    for (why, (n, e)) in &g.refused {
        println!("  refused {n:5}  {why}  (e.g. {e})");
    }
    for (why, (n, e)) in &g.unrun {
        println!("  not run {n:5}  {why}  (e.g. {e})");
    }
    for (why, (n, e)) in &g.a_fails {
        println!("  A fails {n:5}  {why}  (e.g. {e})");
    }
    for d in g.differ.iter().take(20) {
        println!("  DIFFER {d}");
    }
    assert!(g.differ.is_empty(), "{} programs differ", g.differ.len());
    assert!(g.unrun.is_empty(), "{} IR-lifted programs do not run", g.unrun.values().map(|(n, _)| n).sum::<usize>());
    assert!(g.refused.is_empty(), "the IR lifter refuses programs the old one lifts");
}

#[test]
fn sekiro_programs() {
    gate(Game::Sekiro);
}
#[test]
fn riders_republic_programs() {
    gate(Game::Riders);
}
#[test]
fn unity_programs() {
    gate(Game::Sledding);
}

/// The gate catches a wrong lift: the IR program's first float add made a subtract no longer agrees.
#[test]
fn a_planted_bug_is_caught() {
    let Some((_, paths)) = common::corpus(Game::Sekiro) else {
        return;
    };
    for path in &paths {
        let bytes = std::fs::read(path).unwrap();
        let (Ok(old), Ok(mut p)) = (iridia_dxbc::lift(&bytes), iridia_dxbc::ir::lift(&bytes)) else {
            continue;
        };
        let Some(add) = p
            .entry
            .values
            .iter_mut()
            .find(|v| matches!(v.op, iridia_ir::Op::Binary(iridia_ir::Binary::FAdd, ..)))
        else {
            continue;
        };
        let iridia_ir::Op::Binary(op, ..) = &mut add.op else { unreachable!() };
        *op = iridia_ir::Binary::FSub;
        let (Ok(a), Ok(b)) = (ProgramShader::new(old), ProgramShader::from_ir(p)) else {
            continue;
        };
        // an add whose result no output reads is no plant: the first program the plant reaches is caught
        if matches!(compare(&a, &b, CASES), Verdict::Differs(_)) {
            return;
        }
    }
    panic!("no planted subtract was caught");
}

/// One program's two lifts and runs, for a disagreement the gate prints: `IR_GATE_ONE=<container> cargo test -p
/// iridia-dxbc --test ir_gate one -- --ignored --nocapture`.
#[test]
#[ignore = "a diagnostic: names its program in IR_GATE_ONE"]
fn one() {
    let path = std::env::var("IR_GATE_ONE").expect("IR_GATE_ONE names a container");
    let bytes = std::fs::read(&path).unwrap();
    let old = iridia_dxbc::lift(&bytes).unwrap();
    println!("A (old lift) as WGSL:\n{}", iridia::emit::wgsl(&old).unwrap_or_else(|e| e.to_string()));
    let p = iridia_dxbc::ir::lift(&bytes).unwrap();
    println!("B (IR lift):\n{}", iridia_ir::text::print(&p.entry));
    let a = ProgramShader::new(old).unwrap();
    let b = ProgramShader::from_ir(p).unwrap();
    for c in 0..CASES {
        let case = common::case(&a, c);
        println!("case {c}: inputs {:x?}", case.inputs);
        let (mut sa, mut sb) = (Hashed::default(), Hashed::default());
        println!("  A {:x?}\n  B {:x?}", run_asking(&a, &case, &mut sa), run_asking(&b, &case, &mut sb));
        for (k, (x, y)) in sa.asked.iter().zip(&sb.asked).enumerate() {
            if x != y {
                println!("  request {k} differs:\n    A {x}\n    B {y}");
            }
        }
        if c == 0 {
            println!("  A asked {:#?}\n  B asked {:#?}", sa.asked, sb.asked);
        }
        if sa.asked.len() != sb.asked.len() {
            println!("  A made {} requests, B {}", sa.asked.len(), sb.asked.len());
        }
    }
}
