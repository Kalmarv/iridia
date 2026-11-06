//! What the pass tests share: a pixel program around a function, and a run of it in the IR executor.

#![allow(dead_code, reason = "each test file uses part of it")]

use iridia_ir::{
    FloatEnv, Function, Program, Provenance,
    interface::{
        Interface, Label, Resource, ResourceId, ScalarType, ShaderFormat, Stage, StageFacts, SystemValue,
        Value, Varying, VaryingProvenance,
    },
};

fn varying(name: &str, index: u32, value: (ScalarType, u8), system: Option<SystemValue>) -> Varying {
    Varying {
        label: Label {
            name: name.into(),
            index,
        },
        system,
        value: Value {
            scalar: value.0,
            width: value.1,
        },
        interp: None,
        used: true,
        provenance: VaryingProvenance::default(),
    }
}

/// A pixel program of `entry`, with `TEXCOORD` inputs and `SV_Target` outputs of the shapes given.
pub fn program(
    entry: Function,
    inputs: &[(ScalarType, u8)],
    outputs: &[(ScalarType, u8)],
    resources: Vec<Resource>,
) -> Program {
    let n = |i: usize| u32::try_from(i).unwrap_or(u32::MAX);
    Program {
        stage: Stage::Pixel,
        entry,
        patch: None,
        workgroup: None,
        interface: Interface {
            stage: Stage::Pixel,
            inputs: inputs
                .iter()
                .enumerate()
                .map(|(i, v)| varying("TEXCOORD", n(i), *v, None))
                .collect(),
            outputs: outputs
                .iter()
                .enumerate()
                .map(|(i, v)| varying("SV_Target", n(i), *v, Some(SystemValue::Target(n(i)))))
                .collect(),
            resources,
            facts: StageFacts::default(),
        },
        conventions: iridia_ir::Conventions::D3D,
        float: FloatEnv::default(),
        provenance: Provenance {
            format: ShaderFormat::D3d9,
            target: "test".into(),
            lifter: "test".into(),
            lifter_version: 1,
            source_hash: None,
            instruction_count: None,
            policies: Vec::new(),
        },
        notes: Vec::new(),
    }
}

/// A program of the text's function; the text must be in the printer's own form.
pub fn from_text(
    text: &str,
    inputs: &[(ScalarType, u8)],
    outputs: &[(ScalarType, u8)],
    resources: Vec<Resource>,
) -> Program {
    let entry = iridia_ir::text::parse(text).unwrap_or_else(|e| panic!("{e}\n{text}"));
    assert_eq!(
        iridia_ir::text::print(&entry),
        text,
        "the fixture is in the printer's own form"
    );
    let p = program(entry, inputs, outputs, resources);
    iridia_ir::validate(&p).unwrap_or_else(|e| panic!("{e:?}\n{text}"));
    p
}

/// Each output slot's lanes (`None` for one never stored) and whether the run killed.
pub type Outcome = (Vec<Option<[u32; 4]>>, bool);

/// The outcome of a run, or the run's refusal.
pub fn run(p: &Program, inputs: &[[u32; 4]], state: &[(ResourceId, [u32; 4])]) -> Result<Outcome, String> {
    let mut sampler = |_| Err::<[f32; 4], String>("no textures".into());
    let mut bound = iridia_exec_ir::Bound {
        inputs,
        buffers: &[],
        state,
        sampler: &mut sampler,
        denormals: iridia_ir::Denormals::Any,
    };
    let r = iridia_exec_ir::run(p, &mut bound)?;
    Ok((r.outputs, r.killed))
}

/// No textures: a run that samples one is refused.
struct NoTextures;
impl iridia_exec::Backend for NoTextures {
    fn sample(&mut self, _: iridia_exec::ImageSample<iridia_exec::Element>) -> Result<[f32; 4], String> {
        Err("no textures".into())
    }
}

/// The outcome of a run of `p` compiled into `iridia-exec`'s flat form (`Executable::from_ir`; its body must be as
/// `break_flags` leaves it), as [`run`] gives it: an output slot that is never stored reads as zero there, so each
/// slot is `Some`.
pub fn run_flat(p: &Program, inputs: &[[u32; 4]]) -> Result<Outcome, String> {
    let exe = iridia_exec::Executable::from_ir(p).map_err(|e| e.to_string())?;
    let bound = exe.bind(|_| None, |_| None)?;
    let mut backend = NoTextures;
    let [out] = iridia_exec::Runner::<1>::new(&exe, &bound, &mut backend, None).run([inputs])?;
    Ok((out.values.into_iter().map(Some).collect(), out.discarded))
}

/// That a run of `p` in `iridia-exec-ir` and of its flat form agree (an output never stored is zero in both).
pub fn agrees_flat(p: &Program, inputs: &[[u32; 4]], state: &[(ResourceId, [u32; 4])]) {
    let (ir, ir_killed) = run(p, inputs, state).unwrap_or_else(|e| panic!("{e}\n{}", iridia_ir::text::print(&p.entry)));
    let (flat, flat_killed) =
        run_flat(p, inputs).unwrap_or_else(|e| panic!("flat: {e}\n{}", iridia_ir::text::print(&p.entry)));
    // a killed invocation's outputs are not compared
    let ir: Vec<[u32; 4]> = ir.into_iter().map(Option::unwrap_or_default).filter(|_| !ir_killed).collect();
    let flat: Vec<[u32; 4]> = flat.into_iter().map(Option::unwrap_or_default).filter(|_| !flat_killed).collect();
    assert_eq!(
        (ir, ir_killed),
        (flat, flat_killed),
        "inputs {inputs:?}\n{}",
        iridia_ir::text::print(&p.entry)
    );
}

/// That `f` prints as text that parses back to what prints the same.
pub fn round_trip(f: &iridia_ir::Function) {
    let text = iridia_ir::text::print(f);
    let back = iridia_ir::text::parse(&text).unwrap_or_else(|e| panic!("{e}\n{text}"));
    assert_eq!(iridia_ir::text::print(&back), text, "printed differently after parsing");
}
