#![allow(clippy::unwrap_used)]
//! Every distinct vertex and pixel program of Shaun White Skateboarding's three SDB2 bundles against Microsoft's
//! D3DDisassemble: the executable instructions the reader decodes (opcode, modifiers, destination, sources with their
//! swizzles, negation, absolute value and relative addressing) equal the disassembly's, line for line; every program the
//! lifter takes validates and writes as WGSL; every program it refuses is counted by reason. Needs the oracle set
//! `d3d-disassembler/shaun-white/sdb2` (a folder of `<hash>.vs|.ps` raw token streams with `.asm` listings).
mod inputs;

use std::collections::BTreeMap;

use inputs::{Game, OracleSet, oracle};
use iridia_d3d9::{Opcode, Register, Source, lift, read};

fn register(r: Register) -> String {
    match r {
        Register::Temporary(n) => format!("r{n}"),
        Register::Input(n) => format!("v{n}"),
        Register::Constant(n) => format!("c{n}"),
        Register::Sampler(n) => format!("s{n}"),
        Register::Target(n) => format!("oC{n}"),
        Register::Output(n) => format!("o{n}"),
        Register::Address => "a0".to_owned(),
    }
}
fn lanes(mask: u8) -> String {
    if mask == 15 {
        return String::new();
    }
    format!(
        ".{}",
        (0..4)
            .filter(|n| mask & (1 << n) != 0)
            .map(|n| ['x', 'y', 'z', 'w'][n])
            .collect::<String>()
    )
}
fn source(s: Source) -> String {
    // Microsoft's spelling: `-c3_abs[a0.x].xyz`, the modifier after the register, then the relative index, then the swizzle
    let mut text = register(s.register);
    if s.absolute {
        text.push_str("_abs");
    }
    if let Some(lane) = s.relative {
        text.push_str(&format!("[a0.{}]", ['x', 'y', 'z', 'w'][usize::from(lane)]));
    }
    if s.swizzle != [0, 1, 2, 3] {
        text.push('.');
        let used = if s.swizzle.iter().all(|&l| l == s.swizzle[0]) {
            &s.swizzle[..1]
        } else {
            &s.swizzle[..]
        };
        text.extend(used.iter().map(|n| ['x', 'y', 'z', 'w'][usize::from(*n)]));
    }
    if s.negate {
        text.insert(0, '-');
    }
    text
}
fn normalize(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// The executable instructions as Microsoft's disassembler spells them.
fn listing(shader: &iridia_d3d9::Shader) -> Vec<String> {
    shader
        .instructions
        .iter()
        .map(|i| {
            let mut op = match i.opcode {
                Opcode::Tex => match i.control {
                    0 => "texld",
                    1 => "texldp",
                    _ => "texldb",
                }
                .into(),
                _ => format!("{:?}", i.opcode).to_ascii_lowercase(),
            };
            let mut args = Vec::new();
            if let Some(d) = i.destination {
                if d.saturate {
                    op.push_str("_sat");
                }
                if d.partial_precision {
                    op.push_str("_pp");
                }
                args.push(format!("{}{}", register(d.register), lanes(d.mask)));
            }
            args.extend(i.sources.iter().copied().map(source));
            normalize(&format!("{op} {}", args.join(", ")))
        })
        .collect()
}

/// The instruction lines of a listing: no comment, declaration, definition or version line.
fn instructions(asm: &str) -> Vec<String> {
    asm.lines()
        .map(str::trim)
        .filter(|s| {
            !s.is_empty()
                && !s.starts_with("//")
                && !s.starts_with("dcl_")
                && !s.starts_with("def ")
                && *s != "ps_3_0"
                && *s != "vs_3_0"
        })
        .map(normalize)
        .collect()
}

#[derive(Default)]
struct Census {
    programs: usize,
    instructions: usize,
    equal: usize,
    differ: Vec<String>,
    lifted: usize,
    /// Refusals by reason: how many, and the first program.
    refused: BTreeMap<String, (usize, String)>,
}

fn census(extension: &str) -> Option<Census> {
    let folder = oracle(OracleSet::D3dDisassembler, Game::ShaunWhite, "sdb2")?;
    let mut paths: Vec<_> = std::fs::read_dir(&folder)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == extension))
        .collect();
    paths.sort();
    let mut c = Census::default();
    for path in &paths {
        let bytes = std::fs::read(path).unwrap();
        let asm = std::fs::read_to_string(path.with_extension(format!("{extension}.asm"))).unwrap();
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        c.programs += 1;
        let shader = read(&bytes).unwrap_or_else(|e| panic!("{name}: {e}"));
        let (actual, expected) = (listing(&shader), instructions(&asm));
        c.instructions += actual.len();
        if actual == expected {
            c.equal += 1;
        } else {
            let at = actual
                .iter()
                .zip(&expected)
                .position(|(a, b)| a != b)
                .unwrap_or(0);
            c.differ.push(format!(
                "{name}: {:?} against {:?}",
                actual.get(at),
                expected.get(at)
            ));
        }
        match lift(&bytes) {
            Ok(program) => {
                program.validate().unwrap_or_else(|e| panic!("{name}: {e:?}"));
                iridia::emit::wgsl(&program).unwrap_or_else(|e| panic!("{name}: {e}"));
                c.lifted += 1;
            }
            Err(e) => {
                c.refused.entry(e.group()).or_insert((0, name)).0 += 1;
            }
        }
    }
    Some(c)
}

fn report(what: &str, c: &Census) {
    println!(
        "{what}: {} programs, {} instructions, {} listings equal, {} lifted",
        c.programs, c.instructions, c.equal, c.lifted
    );
    for d in c.differ.iter().take(5) {
        println!("  DIFFERS {d}");
    }
    for (why, (n, first)) in &c.refused {
        println!("  REFUSED {n:5} {why} (first {first})");
    }
}

#[test]
fn vertex_programs_equal_microsofts_listing() {
    let Some(c) = census("vs") else { return };
    report("vertex", &c);
    assert!(c.differ.is_empty(), "{} listings differ", c.differ.len());
    assert_eq!((c.programs, c.instructions, c.lifted), (1_099, 63_679, 1_099));
}

#[test]
fn pixel_programs_equal_microsofts_listing() {
    let Some(c) = census("ps") else { return };
    report("pixel", &c);
    assert!(c.differ.is_empty(), "{} listings differ", c.differ.len());
    assert_eq!((c.programs, c.instructions, c.lifted), (6_207, 601_220, 6_207));
}
