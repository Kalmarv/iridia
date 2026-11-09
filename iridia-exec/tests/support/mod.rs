//! What the executor's tests share: programs of the IR written as text, in the form `iridia-exec`'s `::from_ir`
//! constructors take, and the reference interpreter `iridia-exec-ir`'s run of a program beside `iridia-exec`'s.

#![allow(dead_code, reason = "each test file uses part of it")]

use iridia_exec::Execution;
use iridia_ir::{
    Conventions, FloatEnv, Function, Program, Provenance,
    interface::{
        D3dBinding, D3dClass, D3dSignature, Interface, InterpMode, InterpSampling, Interpolation, Label,
        Resource, ResourceId, ResourceKind, ResourceProvenance, ScalarType, ShaderFormat, Stage, StageFacts,
        SystemValue, Value, Varying, VaryingProvenance,
    },
};

/// A varying of `width` lanes of `scalar`, at D3D register `register` with lanes `mask` where it has one.
pub fn varying(
    name: &str,
    index: u32,
    scalar: ScalarType,
    width: u8,
    d3d: Option<(u32, u8)>,
    system: Option<SystemValue>,
) -> Varying {
    Varying {
        label: Label {
            name: name.into(),
            index,
        },
        system,
        value: Value { scalar, width },
        interp: None,
        used: true,
        provenance: VaryingProvenance {
            d3d: d3d.map(|(register, mask)| D3dSignature {
                register,
                mask,
                read_write_mask: 0,
            }),
            ..Default::default()
        },
    }
}

/// A float input of a pixel program, `name` index 0 of `width` lanes in register 0's lanes `mask`, interpolated with
/// perspective.
pub fn input(name: &str, width: u8, mask: u8) -> Varying {
    let mut v = varying(name, 0, ScalarType::Float, width, Some((0, mask)), None);
    v.interp = Some(Interpolation {
        mode: InterpMode::Perspective,
        sampling: InterpSampling::Center,
    });
    v
}

/// `SV_Target0`, four float lanes in register 0.
pub fn target() -> Varying {
    varying(
        "SV_Target",
        0,
        ScalarType::Float,
        4,
        Some((0, 0xf)),
        Some(SystemValue::Target(0)),
    )
}

/// A resource bound at a D3D register of `class`.
pub fn resource(
    id: u32,
    name: &str,
    kind: ResourceKind,
    class: D3dClass,
    register: u32,
    count: u32,
) -> Resource {
    Resource {
        id: ResourceId(id),
        kind,
        count,
        provenance: ResourceProvenance {
            name: name.into(),
            d3d: Some(D3dBinding {
                class,
                register,
                space: 0,
            }),
            ..ResourceProvenance::default()
        },
    }
}

/// The function of `text`, which must be in the printer's own form.
pub fn function(text: &str) -> Function {
    let f = iridia_ir::text::parse(text).unwrap_or_else(|e| panic!("{e}\n{text}"));
    assert_eq!(
        iridia_ir::text::print(&f),
        text,
        "the fixture is in the printer's own form"
    );
    f
}

/// The provenance of a program written here, for `target`.
pub fn provenance(format: ShaderFormat, target: &str) -> Provenance {
    Provenance {
        format,
        target: target.into(),
        lifter: "test".into(),
        lifter_version: 0,
        source_hash: None,
        instruction_count: None,
        policies: Vec::new(),
    }
}

/// A program of `entry` with `interface`, not validated: a hull program's patch constant function, or a thread group,
/// is the caller's to add.
pub fn unchecked(entry: Function, interface: Interface, format: ShaderFormat, target: &str) -> Program {
    Program {
        stage: interface.stage,
        entry,
        patch: None,
        workgroup: None,
        interface,
        conventions: Conventions::D3D,
        float: FloatEnv::default(),
        provenance: provenance(format, target),
        notes: Vec::new(),
    }
}

/// A program of `entry` with `interface`, validated.
pub fn program(entry: Function, interface: Interface, format: ShaderFormat, target: &str) -> Program {
    let p = unchecked(entry, interface, format, target);
    iridia_ir::validate(&p).unwrap_or_else(|e| panic!("{e:?}\n{}", iridia_ir::text::print(&p.entry)));
    p
}

/// A `ps_5_0` pixel program of `text` with these varyings and resources.
pub fn pixel(text: &str, inputs: Vec<Varying>, outputs: Vec<Varying>, resources: Vec<Resource>) -> Program {
    let interface = Interface {
        stage: Stage::Pixel,
        inputs,
        outputs,
        resources,
        facts: StageFacts::default(),
    };
    program(function(text), interface, ShaderFormat::Dxbc, "ps_5_0")
}

/// The lanes of register `register`'s `mask`, in ascending order, as a varying's lanes.
fn packed(register: [u32; 4], mask: u8) -> [u32; 4] {
    let mut out = [0; 4];
    for (k, lane) in (0..4).filter(|l| mask >> l & 1 != 0).enumerate() {
        out[k] = register[lane];
    }
    out
}

/// Each interface input's lanes as `registers` (by D3D register) hold them.
pub fn slots(interface: &Interface, registers: &[[u32; 4]]) -> Vec<[u32; 4]> {
    interface
        .inputs
        .iter()
        .map(|v| {
            let d = v.provenance.d3d.expect("an input with a D3D register");
            packed(registers[d.register as usize], d.mask)
        })
        .collect()
}

/// Each constant buffer's rows as the register view (`(register, rows)`) gives them, by resource.
pub fn buffers<'a>(
    interface: &Interface,
    by_register: &[(u32, &'a [[u32; 4]])],
) -> Vec<(ResourceId, &'a [[u32; 4]])> {
    interface
        .resources
        .iter()
        .filter(|r| matches!(r.kind, ResourceKind::ConstantBuffer { .. }))
        .filter_map(|r| {
            let register = r.provenance.d3d?.register;
            by_register
                .iter()
                .find(|(n, _)| *n == register)
                .map(|(_, rows)| (r.id, *rows))
        })
        .collect()
}

/// A sampler for `iridia-exec-ir` that answers every sample with `answer`.
pub fn answering(answer: [f32; 4]) -> impl FnMut(iridia_exec_ir::Sample) -> Result<[f32; 4], String> {
    move |_| Ok(answer)
}

/// `iridia-exec-ir`'s run of `p` on the inputs `registers` (by D3D register) and constant buffers (by register).
pub fn reference(
    p: &Program,
    registers: &[[u32; 4]],
    constants: &[(u32, &[[u32; 4]])],
    sampler: &mut dyn FnMut(iridia_exec_ir::Sample) -> Result<[f32; 4], String>,
) -> iridia_exec_ir::Run {
    let inputs = slots(&p.interface, registers);
    let buffers = buffers(&p.interface, constants);
    iridia_exec_ir::run(
        p,
        &mut iridia_exec_ir::Bound {
            inputs: &inputs,
            buffers: &buffers,
            state: &[],
            sampler,
            denormals: iridia_ir::Denormals::Any,
        },
    )
    .unwrap_or_else(|e| panic!("iridia-exec-ir: {e}\n{}", iridia_ir::text::print(&p.entry)))
}

/// `iridia-exec-ir`'s run of `p`'s quad, each lane on its own registers.
pub fn reference_quad(
    p: &Program,
    registers: [&[[u32; 4]]; 4],
    constants: &[(u32, &[[u32; 4]])],
    sampler: &dyn Fn(iridia_exec_ir::Sample) -> Result<[f32; 4], String>,
) -> [iridia_exec_ir::Run; 4] {
    let inputs = registers.map(|r| slots(&p.interface, r));
    let buffers = buffers(&p.interface, constants);
    let mut samplers = [(); 4].map(|()| move |s| sampler(s));
    let mut bounds: Vec<iridia_exec_ir::Bound<'_>> = inputs
        .iter()
        .zip(&mut samplers)
        .map(|(inputs, sampler)| iridia_exec_ir::Bound {
            inputs,
            buffers: &buffers,
            state: &[],
            sampler,
            denormals: iridia_ir::Denormals::Any,
        })
        .collect();
    let [b0, b1, b2, b3] = bounds.as_mut_slice() else {
        unreachable!("four lanes")
    };
    iridia_exec_ir::run_quad(p, [b0, b1, b2, b3])
        .unwrap_or_else(|e| panic!("iridia-exec-ir: {e}\n{}", iridia_ir::text::print(&p.entry)))
}

/// That `iridia-exec`'s execution of `p` and `iridia-exec-ir`'s run of it agree: every output lane the program wrote,
/// by the output's D3D register and lanes, and whether the invocation was discarded.
pub fn agree(p: &Program, execution: &Execution, run: &iridia_exec_ir::Run) {
    assert_eq!(
        execution.discarded,
        run.killed,
        "discarded\n{}",
        iridia_ir::text::print(&p.entry)
    );
    for (slot, v) in p.interface.outputs.iter().enumerate() {
        let d = v.provenance.d3d.expect("an output with a D3D register");
        let register = d.register as usize;
        let reference = run.outputs[slot].unwrap_or_default();
        let written = execution.output_masks.get(register).copied().unwrap_or(0);
        for (k, lane) in (0..4).filter(|l| d.mask >> l & 1 != 0).enumerate() {
            if written >> lane & 1 != 0 {
                assert_eq!(
                    execution.outputs[register][lane],
                    reference[k],
                    "output {}{} lane {lane}\n{}",
                    v.label.name,
                    v.label.index,
                    iridia_ir::text::print(&p.entry)
                );
            }
        }
    }
}
