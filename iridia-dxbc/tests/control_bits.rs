#![allow(clippy::unwrap_used)]
//! The typed views of the opcode token's control bits (`dxbc_reader::Instruction::precise_mask`, `sync_flags`,
//! `info_return`, `Declaration::global_flags` and the accessors of the geometry and tessellation declarations) over the
//! games' Shader Model 4 and 5 programs. The expected counts are those of a second reader of the same containers (a
//! Python walker of the token streams written for the census, sharing no code with `dxbc-reader`); the meaning of
//! each value is the name Microsoft's disassembler prints for it (`sync_g_t`, `resinfo_..._uint`, `sampleinfo_uint`,
//! `dcl_globalFlags refactoringAllowed | forceEarlyDepthStencil`, ...). Needs the corpus oracle set's `sekiro/shaders`,
//! `riders/shaders` or `sledding/shaders`.
mod inputs;

use std::collections::BTreeMap;

use dxbc_reader::{GlobalFlags, SyncFlags, decode};
use inputs::Game;

mod common;

#[derive(Default)]
struct Census {
    programs: usize,
    /// Instructions by their `precise` mask.
    precise: BTreeMap<u8, usize>,
    /// `sync` instructions by their flags, as bits 0 to 3 of the control field.
    sync: BTreeMap<u8, usize>,
    /// `resinfo` and `sample_info` instructions by name and return type.
    info: BTreeMap<String, usize>,
    /// `dcl_globalFlags` declarations by their bits 0 to 5 (and 6 and above, shifted down, after a dot).
    flags: BTreeMap<String, usize>,
    /// Stage declarations by name and value.
    stage: BTreeMap<String, usize>,
}

fn bits(f: GlobalFlags) -> String {
    let low = u32::from(f.refactoring_allowed)
        | u32::from(f.double_precision) << 1
        | u32::from(f.force_early_depth_stencil) << 2
        | u32::from(f.raw_and_structured_buffers) << 3
        | u32::from(f.skip_optimization) << 4
        | u32::from(f.minimum_precision) << 5;
    format!("{low}.{}", f.other)
}

fn sync_bits(s: SyncFlags) -> u8 {
    u8::from(s.threads_in_group)
        | u8::from(s.group_shared_memory) << 1
        | u8::from(s.uav_group) << 2
        | u8::from(s.uav_global) << 3
}

fn census(game: Game) -> Option<(String, Census)> {
    let (name, paths) = common::corpus(game)?;
    let mut c = Census::default();
    for path in &paths {
        let bytes = std::fs::read(path).unwrap();
        let program = decode(&dxbc_reader::code(&bytes).unwrap()).unwrap();
        c.programs += 1;
        for ins in &program.instructions {
            if ins.precise_mask() != 0 {
                *c.precise.entry(ins.precise_mask()).or_default() += 1;
            }
            if let Some(s) = ins.sync_flags() {
                *c.sync.entry(sync_bits(s)).or_default() += 1;
            }
            if let Some(r) = ins.info_return() {
                *c.info.entry(format!("{} {r:?}", ins.name())).or_default() += 1;
            }
        }
        let mut bump = |key: String| *c.stage.entry(key).or_default() += 1;
        for d in &program.declarations {
            if let Some(f) = d.global_flags() {
                *c.flags.entry(bits(f)).or_default() += 1;
            }
            let values = [
                ("input primitive", d.geometry_input_primitive()),
                ("output topology", d.geometry_output_topology()),
                ("max output vertices", d.max_output_vertex_count()),
                ("input control points", d.input_control_point_count()),
                ("output control points", d.output_control_point_count()),
                ("domain", d.tessellator_domain()),
                ("partitioning", d.tessellator_partitioning()),
                ("output primitive", d.tessellator_output_primitive()),
                ("fork instances", d.fork_phase_instance_count()),
                ("join instances", d.join_phase_instance_count()),
            ];
            for (label, value) in values {
                if let Some(v) = value {
                    bump(format!("{label} {v}"));
                }
            }
            if let Some(f) = d.hull_max_tess_factor() {
                bump(format!("max tess factor {f}"));
            }
        }
    }
    Some((name, c))
}

fn counts<K: Ord + ToString>(pairs: &[(K, usize)]) -> BTreeMap<String, usize> {
    pairs.iter().map(|(k, n)| (k.to_string(), *n)).collect()
}

fn text(m: &BTreeMap<String, usize>) -> BTreeMap<String, usize> {
    m.clone()
}

#[test]
fn sekiro_programs() {
    let Some((name, c)) = census(Game::Sekiro) else {
        return;
    };
    assert_eq!(c.programs, 432, "{name}");
    // every program states `refactoringAllowed` and nothing else
    assert_eq!(text(&c.flags), counts(&[("1.0", 432)]), "{name}");
    assert!(c.precise.is_empty() && c.sync.is_empty() && c.info.is_empty() && c.stage.is_empty());
}

#[test]
fn riders_republic_programs() {
    let Some((name, c)) = census(Game::Riders) else {
        return;
    };
    assert_eq!(c.programs, 2_443, "{name}");
    // 4,960 instructions of the pixel and compute programs are `precise` for some lanes: masks 1 to 15
    let precise: Vec<usize> = (1..=15)
        .map(|m| c.precise.get(&m).copied().unwrap_or(0))
        .collect();
    assert_eq!(
        precise,
        [
            824, 484, 548, 648, 84, 100, 384, 584, 28, 100, 4, 240, 32, 152, 748
        ],
        "{name}"
    );
    assert_eq!(precise.iter().sum::<usize>(), 4_960);
    // `sync_g` (2) 27 times and `sync_g_t` (3) 218 times, all in compute programs
    assert_eq!(c.sync, BTreeMap::from([(2, 27), (3, 218)]), "{name}");
    assert_eq!(text(&c.info), counts(&[("resinfo Uint", 14)]), "{name}");
    // `refactoringAllowed` in all 2,443 programs, `| forceEarlyDepthStencil` in 24 of them
    assert_eq!(text(&c.flags), counts(&[("1.0", 2_419), ("5.0", 24)]), "{name}");
    // 7 geometry programs: a triangle (3) in, a triangle strip (5) out, 3 vertices (6 programs) or 12 (1); 9 hull
    // programs and their 9 domain programs: 4 control points (18 input counts, 9 output counts), domain 3 (quad),
    // partitioning 3, output primitive 3, largest factor 15 (8) or 64 (1), fork phase instance counts 2 and 4
    assert_eq!(
        text(&c.stage),
        counts(&[
            ("domain 3", 18),
            ("fork instances 2", 1),
            ("fork instances 4", 1),
            ("input control points 4", 18),
            ("input primitive 3", 7),
            ("max output vertices 12", 1),
            ("max output vertices 3", 6),
            ("max tess factor 15", 8),
            ("max tess factor 64", 1),
            ("output control points 4", 9),
            ("output primitive 3", 9),
            ("output topology 5", 7),
            ("partitioning 3", 9),
        ]),
        "{name}"
    );
}

#[test]
fn unity_programs() {
    let Some((name, c)) = census(Game::Sledding) else {
        return;
    };
    assert_eq!(c.programs, 4_854, "{name}");
    // one `resinfo_..._uint` and one `sampleinfo_uint`
    assert_eq!(
        text(&c.info),
        counts(&[("resinfo Uint", 1), ("sample_info Uint", 1)]),
        "{name}"
    );
    // 1,559 programs state global flags (the Shader Model 4.0 pixel programs state none): 1,538 only
    // `refactoringAllowed`, one `| enableRawAndStructuredBuffers`, 20 `| enableMinimumPrecision`
    assert_eq!(
        text(&c.flags),
        counts(&[("1.0", 1_538), ("9.0", 1), ("33.0", 20)]),
        "{name}"
    );
    assert!(c.precise.is_empty() && c.sync.is_empty() && c.stage.is_empty());
}
