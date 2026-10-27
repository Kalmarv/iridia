#![allow(clippy::unwrap_used)]
//! The geometry fixture (`geometry.hlsl`: a triangle in, two strips out) in each build: its facts are the
//! ones the entry point states, its function is the one `stage_functions` describes, and it validates and is written as
//! WGSL functions. Every prefix of it, and of the optimised build, is refused by name, never a panic.
use iridia::{GsInput, GsOutput, Stage, StageFacts, stage_functions};
use iridia_dxbc::lift;

fn fixture(name: &str) -> Vec<u8> {
    std::fs::read(format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"))).unwrap()
}

#[test]
fn a_geometry_program_is_the_function_its_facts_describe() {
    for name in ["geometry.dxbc", "geometry.od.dxbc"] {
        let p = lift(&fixture(name)).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(p.stage, Stage::Geometry, "{name}");
        let StageFacts::Geometry(g) = &p.interface.facts else {
            panic!("{name}: {:?}", p.interface.facts)
        };
        assert_eq!(
            (g.input, g.output, g.max_vertices, g.instances),
            (GsInput::Triangle, GsOutput::TriangleStrip, 6, 1),
            "{name}"
        );
        let sigs = stage_functions(&p.interface).unwrap().unwrap();
        let [main] = sigs.as_slice() else { panic!("{name}") };
        let arguments: Vec<_> = main.arguments.iter().map(|a| a.name.as_str()).collect();
        assert_eq!(
            arguments,
            ["SV_Position0", "TEXCOORD0", "SV_PrimitiveID", "SV_GSInstanceID"],
            "{name}"
        );
        let results: Vec<_> = main.result.iter().map(|a| a.name.as_str()).collect();
        assert_eq!(
            results,
            ["SV_Position0", "TEXCOORD0", "SV_VertexCount", "SV_StripStart"],
            "{name}"
        );
        p.validate().unwrap_or_else(|e| panic!("{name}: {e:?}"));
        let wgsl = iridia::emit::wgsl_functions(&p).unwrap();
        assert!(wgsl.contains("fn main("), "{name}: {wgsl}");
        assert!(wgsl.contains("SV_VertexCount"), "{name}");
    }
}

#[test]
fn every_truncation_of_a_geometry_program_is_refused() {
    for name in ["geometry.dxbc", "geometry.od.dxbc"] {
        let bytes = fixture(name);
        for n in 0..bytes.len() {
            assert!(lift(&bytes[..n]).is_err(), "{name}: prefix {n}");
        }
    }
}
