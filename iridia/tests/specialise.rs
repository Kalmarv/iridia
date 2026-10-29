//! The executor is an independent oracle for specialising a shader, including an unknown branch.
#![allow(clippy::unwrap_used)]
use iridia::specialise::{Words, specialise};
use iridia::{ResourceKind, Stage};
use iridia_exec::probe::AnsweredReads;
use iridia_exec::{ImageSample, PixelShader, ProgramShader};
// GLSL has no D3D signature. Assign the executor's register rows in interface order on
// test-only copies, identically for both programs; the shader module remains untouched.
fn execution(mut program: iridia::Program) -> ProgramShader {
    for (output, varyings) in [
        (false, &mut program.interface.inputs),
        (true, &mut program.interface.outputs),
    ] {
        for (index, varying) in varyings.iter_mut().enumerate() {
            let mask = (1u8 << varying.value.width.min(4)) - 1;
            varying.provenance.d3d = Some(iridia_interface::D3dSignature {
                register: index as u32,
                mask,
                read_write_mask: if output { 0 } else { mask },
            });
        }
    }
    ProgramShader::new(program).unwrap()
}
fn program() -> iridia::Program {
    iridia_mojo::lift("uniform vec4 ps_uniforms_vec4[1];\nuniform sampler2D ps_s0;\nvarying vec4 vTexCoord0;\nvec4 ps_r0;\nvoid main() {\nps_r0 = texture2D(ps_s0,vTexCoord0.xy);\nif (ps_uniforms_vec4[0].x != 0.0) {\nps_r0 = ps_r0 * ps_uniforms_vec4[0].yyyy;\n} else {\nps_r0 = vTexCoord0;\n}\ngl_FragColor = ps_r0;\n}",Stage::Pixel,&iridia_mojo::Hints {target:"fixture".into(),constants:vec![iridia_mojo::Constant{name:"SwitchAndScale".into(),offset:0,bytes:16,width:4}],samplers:vec![iridia_mojo::Sampler{glsl_name:"ps_s0".into(),name:"image".into(),slot:0}],attributes:vec![]}).unwrap()
}
#[test]
fn a_known_output_structure_retains_its_member_shape() {
    let source = iridia_mojo::lift(
        "uniform vec4 ps_uniforms_vec4[1];\nuniform sampler2D ps_s0;\nvoid main() {\nif (ps_uniforms_vec4[0].x != 0.0) {\ngl_FragColor = texture2D(ps_s0, vec2(0.5));\n} else {\ngl_FragColor = vec4(0.0, 1.0, 0.0, 1.0);\n}\n}",
        Stage::Pixel,
        &iridia_mojo::Hints {
            target: "output structure fixture".into(),
            constants: vec![iridia_mojo::Constant { name: "Feature".into(), offset: 0, bytes: 16, width: 4 }],
            samplers: vec![iridia_mojo::Sampler { glsl_name: "ps_s0".into(), name: "image".into(), slot: 0 }],
            attributes: vec![],
        },
    ).unwrap();
    let buffer = source
        .interface
        .resources
        .iter()
        .find(|r| matches!(r.kind, ResourceKind::ConstantBuffer { .. }))
        .unwrap()
        .id;
    for enabled in [false, true] {
        let flag = f32::from(u8::from(enabled)).to_bits();
        let result = specialise(&source, &Words::from([((buffer, 0), flag)])).unwrap();
        assert_eq!(
            result
                .program
                .interface
                .resources
                .iter()
                .filter(|r| matches!(r.kind, ResourceKind::Texture { .. }))
                .count(),
            usize::from(enabled)
        );
        let original = execution(source.clone());
        let specialised = execution(result.program);
        for i in 0..256 {
            let cb = [[flag, 0, 0, 0]];
            let sample = |_: &ImageSample<u32>| Ok([i as f32 / 256., 0.25, 0.5, 1.]);
            let a = original
                .execute(&[], &[(0, &cb)], &mut AnsweredReads::new(sample))
                .unwrap();
            let b = specialised
                .execute(&[], &[(0, &cb)], &mut AnsweredReads::new(sample))
                .unwrap();
            assert_eq!(a.outputs, b.outputs, "enabled={enabled}, input={i}");
            assert_eq!(a.discarded, b.discarded);
        }
    }
}
#[test]
fn stated_switch_removes_a_sampler_and_unknown_switch_does_not() {
    let source = program();
    let buffer = source
        .interface
        .resources
        .iter()
        .find(|r| matches!(r.kind, ResourceKind::ConstantBuffer { .. }))
        .unwrap()
        .id;
    for known in [false, true] {
        let words = if known {
            Words::from([((buffer, 0), 0f32.to_bits())])
        } else {
            Words::new()
        };
        let result = specialise(&source, &words).unwrap();
        let textures = |p: &iridia::Program| {
            p.interface
                .resources
                .iter()
                .filter(|r| matches!(r.kind, ResourceKind::Texture { .. }))
                .count()
        };
        assert_eq!(textures(&source), 1);
        assert_eq!(textures(&result.program), usize::from(!known));
        assert_eq!(result.words, words);
        let original = execution(source.clone());
        let specialised = execution(result.program);
        for i in 0..256 {
            let input = [[
                (i as f32 / 257.).to_bits(),
                ((i * 37 % 257) as f32 / 257.).to_bits(),
                0,
                1f32.to_bits(),
            ]];
            let cb = [[
                if known || i % 2 == 0 { 0f32 } else { 1f32 }.to_bits(),
                (i as f32 * 0.03125).to_bits(),
                0,
                0,
            ]];
            let sample = |_: &ImageSample<u32>| Ok([i as f32 * 0.0078125, 0.2, 0.3, 1.]);
            let a = original
                .execute(&input, &[(0, &cb)], &mut AnsweredReads::new(sample))
                .unwrap();
            let b = specialised
                .execute(&input, &[(0, &cb)], &mut AnsweredReads::new(sample))
                .unwrap();
            assert_eq!(a.outputs, b.outputs, "known={known}, input={i}");
            assert_eq!(a.discarded, b.discarded);
        }
    }
}
#[test]
fn a_plain_scale_stays_uniform_for_the_same_feature_set() {
    let source = program();
    let buffer = source
        .interface
        .resources
        .iter()
        .find(|r| matches!(r.kind, ResourceKind::ConstantBuffer { .. }))
        .unwrap()
        .id;
    let known = Words::from([((buffer, 0), 1f32.to_bits()), ((buffer, 4), 99f32.to_bits())]);
    let result = specialise(&source, &known).unwrap();
    assert_eq!(result.words, Words::from([((buffer, 0), 1f32.to_bits())]));
    let a = execution(source);
    let b = execution(result.program);
    for i in 0..256 {
        let input = [[0; 4]];
        let cb = [[1f32.to_bits(), (i as f32 / 16.).to_bits(), 0, 0]];
        let sample = |_: &ImageSample<u32>| Ok([0.125, 0.25, 0.5, 1.]);
        let original = a
            .execute(&input, &[(0, &cb)], &mut AnsweredReads::new(sample))
            .unwrap();
        let specialised = b
            .execute(&input, &[(0, &cb)], &mut AnsweredReads::new(sample))
            .unwrap();
        assert_eq!(original.outputs, specialised.outputs);
    }
}
#[test]
fn an_unknown_branch_preserves_the_sign_of_zero() {
    let source=iridia_mojo::lift("varying vec4 vTexCoord0;\nvec4 ps_r0;\nvoid main() {\nif (vTexCoord0.x > 0.0) {\nps_r0 = vec4(0.0);\n} else {\nps_r0 = vec4(-0.0);\n}\ngl_FragColor = ps_r0;\n}",Stage::Pixel,&Default::default()).unwrap();
    let specialised = specialise(&source, &Words::new()).unwrap();
    let original = execution(source);
    let specialised = execution(specialised.program);
    for x in [-1f32, 0., 1.] {
        let input = [[x.to_bits(), 0, 0, 0]];
        let sample = |_: &ImageSample<u32>| Err("no sample expected".to_owned());
        assert_eq!(
            original
                .execute(&input, &[], &mut AnsweredReads::new(sample))
                .unwrap()
                .outputs,
            specialised
                .execute(&input, &[], &mut AnsweredReads::new(sample))
                .unwrap()
                .outputs
        );
    }
}
