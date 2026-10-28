#![allow(clippy::unwrap_used)]
//! The reader's per-opcode operand formats against the games' own programs: every instruction of every Shader Model 4
//! and 5 program of a corpus has the operand count its opcode's format gives (`dxbc_reader::decode` refuses any other),
//! the operands of a `Resource` slot are resource, sampler, view, shared-memory or stream registers and no other slot's
//! are, and the float modifiers (negate, absolute value) sit on float slots only. Needs the corpus oracle set's
//! `sekiro/shaders`, `riders/shaders` or `sledding/shaders` (each a folder, searched recursively, of containers as the game
//! ships them).
mod inputs;

use std::collections::BTreeMap;

use dxbc_reader::{OperandClass, Register, Selection, decode, opcode_name, operand_format};
use inputs::Game;

mod common;

fn is_resource(register: Register) -> bool {
    matches!(
        register,
        Register::Texture
            | Register::Sampler
            | Register::Uav
            | Register::GroupSharedMemory
            | Register::Stream
    )
}

#[derive(Default)]
struct Census {
    programs: usize,
    instructions: usize,
    /// Instructions of each opcode that has a format.
    formatted: BTreeMap<u16, usize>,
    /// Instructions of each opcode that has none (`call`, `label`, ...).
    unformatted: BTreeMap<u16, usize>,
    /// Programs the reader refuses, by reason.
    refused: BTreeMap<String, usize>,
    /// Destinations selected by one lane, by register.
    scalar_destinations: BTreeMap<String, usize>,
}

fn census(game: Game) -> Option<(String, Census)> {
    let (name, paths) = common::corpus(game)?;
    let mut c = Census::default();
    for path in &paths {
        let bytes = std::fs::read(path).unwrap();
        let words = match dxbc_reader::code(&bytes) {
            Ok(words) => words,
            Err(e) => {
                *c.refused.entry(format!("container: {}", e.detail)).or_default() += 1;
                continue;
            }
        };
        let program = match decode(&words) {
            Ok(program) => program,
            Err(e) => {
                *c.refused.entry(e.detail).or_default() += 1;
                continue;
            }
        };
        c.programs += 1;
        for ins in &program.instructions {
            c.instructions += 1;
            let Some(format) = operand_format(ins.opcode) else {
                *c.unformatted.entry(ins.opcode).or_default() += 1;
                continue;
            };
            *c.formatted.entry(ins.opcode).or_default() += 1;
            let (destinations, sources) = ins.destinations_and_sources().unwrap();
            assert_eq!(destinations.len(), format.destinations.len());
            assert_eq!(sources.len(), format.sources.len());
            let slots = destinations
                .iter()
                .zip(format.destinations)
                .map(|(o, class)| (o, class, true))
                .chain(
                    sources
                        .iter()
                        .zip(format.sources)
                        .map(|(o, class)| (o, class, false)),
                );
            for (operand, class, destination) in slots {
                let here = format!(
                    "{}: {} at word {}",
                    path.display(),
                    opcode_name(ins.opcode),
                    ins.word
                );
                assert_eq!(
                    is_resource(operand.register),
                    *class == OperandClass::Resource,
                    "{here}: a {class:?} slot holds {:?}",
                    operand.register
                );
                // a negate or absolute value is a modifier of data: a resource register has none
                if *class == OperandClass::Resource {
                    assert_eq!(operand.modifier, 0, "{here}: a modifier on a resource slot");
                }
                if destination {
                    assert_ne!(
                        operand.register,
                        Register::Immediate,
                        "{here}: an immediate destination"
                    );
                    // a destination selects lanes by a write mask; a scalar register (`oDepth`) by its one lane
                    assert!(
                        !matches!(operand.selection, Selection::Swizzle(_)),
                        "{here}: a destination selected by a swizzle"
                    );
                    if matches!(operand.selection, Selection::Scalar(_)) {
                        *c.scalar_destinations
                            .entry(format!("{:?}", operand.register))
                            .or_default() += 1;
                    }
                }
            }
        }
    }
    Some((name, c))
}

fn report(game: Game, programs: usize, instructions: usize, opcodes: usize) {
    let Some((name, c)) = census(game) else {
        return;
    };
    assert!(c.refused.is_empty(), "{name}: refused {:?}", c.refused);
    let unformatted: Vec<_> = c
        .unformatted
        .iter()
        .map(|(op, n)| format!("{} {n}", opcode_name(*op)))
        .collect();
    println!(
        "{name}: {} programs, {} instructions, {} opcodes with a format seen, without one: [{}];          destinations selected by one lane: {:?}",
        c.programs,
        c.instructions,
        c.formatted.len(),
        unformatted.join(", "),
        c.scalar_destinations
    );
    let seen: Vec<_> = c.formatted.keys().map(|op| opcode_name(*op)).collect();
    println!("  opcodes seen: {}", seen.join(" "));
    assert_eq!(
        (c.programs, c.instructions, c.formatted.len()),
        (programs, instructions, opcodes),
        "{name}"
    );
}

#[test]
fn sekiro_programs() {
    report(Game::Sekiro, 432, 54_945, 53);
}
#[test]
fn riders_republic_programs() {
    report(Game::Riders, 2_443, 563_505, 105);
}
#[test]
fn unity_programs() {
    report(Game::Sledding, 4_854, 2_796_869, 82);
}
