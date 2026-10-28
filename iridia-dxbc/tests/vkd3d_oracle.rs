#![allow(clippy::unwrap_used)]
//! A second implementation, run as a tool: vkd3d's `vkd3d-dxbc -t` (its container reader and digest check) and
//! `vkd3d-compiler -x dxbc-tpf -b d3d-asm` (its Shader Model 4 and 5 token reader and disassembler), over the games' own
//! programs, against `dxcontainer`, the signature readers and `dxbc_reader::decode`. Compared per container:
//!
//! - the part directory: kinds, sizes and the offsets of the parts' bytes equal;
//! - the digest: vkd3d accepts the stored one (it validates by default), and refuses a container with one byte of the
//!   digest changed that `dxcontainer::checksum` also finds different;
//! - the input, output and patch-constant signatures: each element's name and index, register and component mask equal
//!   vkd3d's `.param` lines;
//! - the instructions: vkd3d's mnemonic sequence, with its modifier suffixes removed, equals `decode`'s opcode names.
//!
//! Needs the corpus oracle set's `sekiro/shaders`, `riders/shaders` or `sledding/shaders` and the two tools, each a
//! program with vkd3d's command line (`tools.vkd3d_dxbc`, `tools.vkd3d_compiler`; built from vkd3d's source, run here
//! through a wrapper that starts the Linux program). A tool that is not configured skips the test. The debug key
//! `vkd3d.limit` sets how many containers of a corpus are compared (200 spread evenly by default; `all` for every one).
mod inputs;

use std::collections::BTreeMap;
use std::io::Write as _;
use std::path::Path;
use std::process::{Command, Stdio};

use dxbc_reader::{decode, opcode_name, signature};
use inputs::{Game, Tool, debug, tool};

mod common;

/// Runs `program` with `args`, the container on its standard input; its exit status and standard output.
fn run(program: &Path, args: &[&str], stdin: &[u8]) -> (bool, String) {
    let mut child = Command::new(program)
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut input = child.stdin.take().unwrap();
    let data = stdin.to_vec();
    let writer = std::thread::spawn(move || {
        // the tool may exit before it reads everything (it refuses the container): that is not an error here
        let _ = input.write_all(&data);
    });
    let out = child.wait_with_output().unwrap();
    writer.join().unwrap();
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    (out.status.success(), text)
}

/// `vkd3d-dxbc -t`'s part table as `(kind, size, offset)`.
fn parts_of(listing: &str) -> Vec<(String, usize, usize)> {
    let number = |s: &str| s.trim_matches(|c| c == '(' || c == ')').parse::<usize>().unwrap();
    listing
        .lines()
        .filter_map(|line| {
            let f: Vec<&str> = line.split_whitespace().collect();
            // `<index> <tag hex> (<TAG>) 0x<size> (<size>) 0x<offset> (<offset>)`
            if f.len() == 7 && f.first()?.parse::<usize>().is_ok() && f.get(2)?.starts_with('(') {
                let kind = f[2].trim_matches(|c| c == '(' || c == ')').to_owned();
                Some((kind, number(f[4]), number(f[6])))
            } else {
                None
            }
        })
        .collect()
}

/// One `.param` line of the disassembly: `NAME[index].mask, register.mask, type[, system value]`.
#[derive(Debug, PartialEq, Eq)]
struct Param {
    name: String,
    index: u32,
    mask: u8,
    register: u32,
}

fn mask_of(lanes: &str) -> u8 {
    lanes
        .chars()
        .map(|c| match c {
            'x' => 1,
            'y' => 2,
            'z' => 4,
            'w' => 8,
            _ => 0,
        })
        .fold(0, |m, b| m | b)
}

/// The input, output and patch-constant `.param` lines of a disassembly.
fn params_of(asm: &str) -> (Vec<Param>, Vec<Param>, Vec<Param>) {
    let (mut inputs, mut outputs, mut patch) = (Vec::new(), Vec::new(), Vec::new());
    let mut section = 0;
    for line in asm.lines() {
        match line.trim() {
            ".input" => section = 1,
            ".output" => section = 2,
            ".patch_constant" => section = 3,
            ".text" => break,
            l if l.starts_with(".param ") => {
                let fields: Vec<&str> = l.trim_start_matches(".param ").split(", ").collect();
                let Some(&semantic) = fields.first() else { continue };
                let Some(&place) = fields.get(1) else { continue };
                let (name_index, lanes) = semantic.split_once('.').unwrap_or((semantic, ""));
                let digits = name_index.trim_end_matches(|c: char| c.is_ascii_digit());
                let index = name_index[digits.len()..].parse().unwrap_or(0);
                let (register, _) = place.split_once('.').unwrap_or((place, ""));
                let p = Param {
                    name: digits.to_owned(),
                    index,
                    mask: mask_of(lanes),
                    register: register
                        .trim_start_matches(|c: char| c.is_ascii_alphabetic())
                        .parse()
                        .unwrap_or(u32::MAX),
                };
                match section {
                    1 => inputs.push(p),
                    2 => outputs.push(p),
                    3 => patch.push(p),
                    _ => {}
                }
            }
            _ => {}
        }
    }
    (inputs, outputs, patch)
}

/// The mnemonics of the disassembly's instructions (its `.text` section without declarations and immediate data), each
/// with the modifier suffixes vkd3d appends removed.
fn mnemonics_of(asm: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut in_text = false;
    let mut data = false;
    for line in asm.lines() {
        let l = line.trim();
        if l == ".text" {
            in_text = true;
            continue;
        }
        if !in_text || l.is_empty() {
            continue;
        }
        if data {
            data = !l.starts_with('}');
            continue;
        }
        let head = l.split([' ', '(']).next().unwrap();
        if head.starts_with("dcl_") || head.starts_with("hs_") || head == "{" {
            data = l.starts_with("dcl_immediateConstantBuffer") && l.ends_with('{');
            continue;
        }
        let mut name = head.to_owned();
        for suffix in [
            "_sat",
            "_indexable",
            "_uint",
            "_rcpfloat",
            "_precise",
            "_pp",
            "_aoffimmi",
        ] {
            while let Some(stripped) = name.strip_suffix(suffix) {
                name = stripped.to_owned();
            }
        }
        // a conditional's test (`if_nz`, `discard_z`, `breakp_nz`) is a suffix; `round_z` is an opcode
        if matches!(
            name.split('_').next(),
            Some("if" | "discard" | "breakp" | "continuep" | "retp" | "breakc" | "continuec" | "retc")
        ) {
            for suffix in ["_nz", "_z"] {
                if let Some(stripped) = name.strip_suffix(suffix) {
                    name = stripped.to_owned();
                    break;
                }
            }
        }
        // vkd3d spells these differently: Direct3D's disassembler names the conditional break, continue and return after
        // the predicate, the derivatives `dsx` and `dsy`, the multisampled load `ld2dms` and the barrier by its flags
        let name = match name.as_str() {
            "breakp" => "breakc".to_owned(),
            "continuep" => "continuec".to_owned(),
            "retp" => "retc".to_owned(),
            "dsx" => "deriv_rtx".to_owned(),
            "dsy" => "deriv_rty".to_owned(),
            "ld2dms" => "ld_ms".to_owned(),
            n if n.starts_with("sync") => "sync".to_owned(),
            _ => name,
        };
        out.push(name);
    }
    out
}

fn sample<T: Clone>(items: &[T], limit: usize) -> Vec<T> {
    if limit >= items.len() {
        return items.to_vec();
    }
    (0..limit)
        .map(|i| items[i * items.len() / limit].clone())
        .collect()
}

fn compare(game: Game) {
    let (Some(dxbc), Some(compiler), Some((name, paths))) = (
        tool(Tool::Vkd3dDxbc),
        tool(Tool::Vkd3dCompiler),
        common::corpus(game),
    ) else {
        return;
    };
    let limit = match debug("vkd3d.limit").as_deref() {
        Some("all") => usize::MAX,
        Some(n) => n.parse().unwrap(),
        None => 200,
    };
    let mut seen = 0;
    let mut sections = 0;
    let (mut signatures, mut instructions, mut refused_digests) = (0, 0, 0);
    let mut differences: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for path in sample(&paths, limit) {
        let bytes = std::fs::read(&path).unwrap();
        let here = path.display().to_string();
        let mut differ = |what: &str, detail: String| {
            differences
                .entry(what.to_owned())
                .or_default()
                .push(format!("{here}: {detail}"));
        };
        seen += 1;

        // the part directory and the digest
        let ours = dxcontainer::chunks(&bytes).unwrap();
        let (accepted, listing) = run(&dxbc, &["-t"], &bytes);
        if !accepted {
            differ(
                "vkd3d refuses a stock container",
                listing.lines().next().unwrap_or("").to_owned(),
            );
            continue;
        }
        let theirs = parts_of(&listing);
        let base = bytes.as_ptr().addr();
        let ours_listing: Vec<_> = ours
            .iter()
            .map(|p| {
                (
                    String::from_utf8_lossy(&p.kind).into_owned(),
                    p.data.len(),
                    p.data.as_ptr().addr() - base,
                )
            })
            .collect();
        if theirs != ours_listing {
            differ("part directory", format!("{theirs:?} against {ours_listing:?}"));
        }
        sections += ours.len();
        let mut stale = bytes.clone();
        stale[4] ^= 0x55;
        let stored: [u8; 16] = stale[4..20].try_into().unwrap();
        let (accepted_stale, _) = run(&dxbc, &["-t"], &stale);
        if accepted_stale || dxcontainer::checksum(&stale) == Some(stored) {
            differ("stale digest", format!("vkd3d accepts: {accepted_stale}"));
        } else {
            refused_digests += 1;
        }

        // the signatures and the instructions
        let (ok, asm) = run(
            &compiler,
            &["-x", "dxbc-tpf", "-b", "d3d-asm", "--formatting=signatures"],
            &bytes,
        );
        if !ok {
            differ(
                "vkd3d-compiler refuses a stock program",
                asm.lines().next().unwrap_or("").to_owned(),
            );
            continue;
        }
        let (their_in, their_out, their_patch) = params_of(&asm);
        for (kind, theirs) in [(b"ISGN", their_in), (b"OSGN", their_out), (b"PCSG", their_patch)] {
            let Some(part) = ours.iter().find(|p| {
                &p.kind == kind
                    || (kind == b"ISGN" && &p.kind == b"ISG1")
                    || (kind == b"OSGN" && (&p.kind == b"OSG1" || &p.kind == b"OSG5"))
                    || (kind == b"PCSG" && &p.kind == b"PSG1")
            }) else {
                if !theirs.is_empty() {
                    differ(
                        "signature",
                        format!("{} elements in a container with none", theirs.len()),
                    );
                }
                continue;
            };
            let elements = signature(part.data, &part.kind).unwrap();
            let mine: Vec<Param> = elements
                .iter()
                .map(|e| Param {
                    name: e.name.clone(),
                    index: e.index,
                    // vkd3d lists no lanes for an output that has no register (`SV_Depth`, `SV_Coverage`)
                    mask: if e.register == u32::MAX { 0 } else { e.mask },
                    register: e.register,
                })
                .collect();
            signatures += mine.len();
            if mine != theirs {
                differ(
                    "signature",
                    format!(
                        "{} {mine:?} against {theirs:?}",
                        String::from_utf8_lossy(&part.kind)
                    ),
                );
            }
        }
        let words = dxbc_reader::code(&bytes).unwrap();
        let program = decode(&words).unwrap();
        let mine: Vec<String> = program
            .instructions
            .iter()
            .map(|i| opcode_name(i.opcode).to_owned())
            .collect();
        let theirs = mnemonics_of(&asm);
        instructions += mine.len();
        if mine != theirs {
            let at = mine
                .iter()
                .zip(&theirs)
                .position(|(a, b)| a != b)
                .unwrap_or(mine.len().min(theirs.len()));
            differ(
                "instruction sequence",
                format!(
                    "{} against {} instructions, first difference at {at}: {:?} against {:?}",
                    mine.len(),
                    theirs.len(),
                    mine.get(at),
                    theirs.get(at)
                ),
            );
        }
    }
    println!(
        "{name}: {seen} containers, {sections} parts, {refused_digests} stale digests refused by both, {signatures} signature elements, {instructions} instructions"
    );
    for (what, list) in &differences {
        println!("  DIFFERENCE {what}: {} (first {})", list.len(), list[0]);
    }
    assert!(
        differences.is_empty(),
        "{name}: {} kinds of difference",
        differences.len()
    );
}

#[test]
fn sekiro_programs() {
    compare(Game::Sekiro);
}
#[test]
fn riders_republic_programs() {
    compare(Game::Riders);
}
#[test]
fn unity_programs() {
    compare(Game::Sledding);
}
