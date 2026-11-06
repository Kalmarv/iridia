//! `flatten`, `promote`, `simplify` and `specialise` on programs in IR text, each checked against the form it should
//! give and by running it before and after on hand-worked inputs.

#![allow(clippy::unwrap_used, reason = "hand-worked fixtures")]

mod support;

use iridia_ir::interface::{
    Resource, ResourceId, ResourceKind, ResourceProvenance, SampleType, ScalarType, StateParam, TextureDim,
};
use support::{from_text, run};

const F1: (ScalarType, u8) = (ScalarType::Float, 1);

fn f(x: f32) -> [u32; 4] {
    [x.to_bits(), 0, 0, 0]
}

#[test]
fn promote_carries_variables_out_of_an_if_and_through_a_loop() {
    // r = x; if x < 0 { r = -x }; loop: while r < 10 { r = r * 2 }; output r
    let text = "iridia-ir 1
fn main {
  var r: f32
  %0: f32 = input 0.x
  r = %0
  %1: f32 = f32(0.0)
  %2: bool = fcmp.lt %0, %1
  if %2 {
    %3: f32 = fneg %0
    r = %3
    yield
  } else {
    yield
  }
  %4: f32 = f32(10.0)
  %5: f32 = f32(2.0)
  loop '0 () = () {
    %6: f32 = load r
    %7: bool = fcmp.ge %6, %4
    if %7 {
      br '0
    } else {
      yield
    }
    %8: f32 = fmul.ieee %6, %5
    r = %8
    continue '0
  }
  %9: f32 = load r
  output 0.x = %9
  return
}
";
    let p = from_text(text, &[F1], &[F1], Vec::new());
    let mut q = p.clone();
    iridia_passes::promote(&mut q.entry).unwrap();
    iridia_ir::validate(&q).unwrap_or_else(|e| panic!("{e:?}\n{}", iridia_ir::text::print(&q.entry)));
    assert!(q.entry.vars.is_empty());
    let printed = iridia_ir::text::print(&q.entry);
    assert!(!printed.contains("load "), "{printed}");
    // r through the if and the loop: -3 -> 3 -> 6 -> 12; 0.5 -> 16; 1 -> 16; 20 stays
    for (x, want) in [(-3.0f32, 12.0f32), (0.5, 16.0), (1.0, 16.0), (20.0, 20.0)] {
        for prog in [&p, &q] {
            let (out, _) = run(prog, &[f(x)], &[]).unwrap();
            assert_eq!(out[0].unwrap()[0], want.to_bits(), "x {x}");
            support::agrees_flat(prog, &[f(x)], &[]);
        }
    }
}

#[test]
fn simplify_folds_lane_bookkeeping_and_removes_what_is_unused() {
    let text = "iridia-ir 1
fn main {
  %0: vec4<f32> = input 0.xyzw
  %1: vec4<f32> = %0.wzyx
  %2: vec2<f32> = %1.yx
  %3: f32 = %0.x
  %4: f32 = %0.y
  %5: f32 = %0.z
  %6: f32 = %0.w
  %7: vec4<f32> = compose %3, %4, %5, %6
  %8: vec4<f32> = %7.xyzw
  %9: f32 = fneg %3
  %10: vec2<f32> = compose %5, %4
  output 0.xyzw = %8
  output 1.xy = %2
  output 2.xy = %10
  return
}
";
    let p = from_text(
        text,
        &[(ScalarType::Float, 4)],
        &[(ScalarType::Float, 4); 3],
        Vec::new(),
    );
    let mut q = p.clone();
    iridia_passes::simplify(&mut q.entry);
    iridia_ir::validate(&q).unwrap();
    // the swizzle of a swizzle is one swizzle; the compose of every lane in order is the input; the compose of
    // two lanes of it is a swizzle; the negation nothing uses is gone
    let want = "iridia-ir 1
fn main {
  %0: vec4<f32> = input 0.xyzw
  %2: vec2<f32> = %0.zw
  %10: vec2<f32> = %0.zy
  output 0.xyzw = %0
  output 1.xy = %2
  output 2.xy = %10
  return
}
";
    assert_eq!(iridia_ir::text::print(&q.entry), want);
    let input = [1.0f32, 2.0, 3.0, 4.0].map(f32::to_bits);
    assert_eq!(run(&p, &[input], &[]), run(&q, &[input], &[]));
}

#[test]
fn specialise_folds_a_known_state_and_the_branch_it_decides() {
    let text = "iridia-ir 1
fn main {
  %0: bool = state r0
  %1: f32 = input 0.x
  if %0 {
    %2: f32 = fneg %1
    yield %2
  } else {
    yield %1
  } -> %3: f32
  output 0.x = %3
  return
}
";
    let state = Resource {
        id: ResourceId(0),
        kind: ResourceKind::PipelineState(StateParam::Projected(0)),
        count: 1,
        provenance: ResourceProvenance {
            name: "projected0".into(),
            ..ResourceProvenance::default()
        },
    };
    let p = from_text(text, &[F1], &[F1], vec![state]);
    for (known, want) in [(true, -5.0f32), (false, 5.0)] {
        let mut q = p.clone();
        iridia_passes::specialise(&mut q, &[(ResourceId(0), [u64::from(known), 0, 0, 0])]).unwrap();
        iridia_passes::simplify(&mut q.entry);
        iridia_ir::validate(&q).unwrap();
        assert!(q.interface.resources.is_empty());
        let printed = iridia_ir::text::print(&q.entry);
        assert!(
            !printed.contains("if ") && !printed.contains("state"),
            "{printed}"
        );
        let (out, _) = run(&q, &[f(5.0)], &[]).unwrap();
        assert_eq!(out[0].unwrap()[0], want.to_bits());
        // and the unspecialised program agrees, given the state
        let (out, _) = run(&p, &[f(5.0)], &[(ResourceId(0), [u32::from(known), 0, 0, 0])]).unwrap();
        assert_eq!(out[0].unwrap()[0], want.to_bits());
        iridia_naga::wgsl(&q, &iridia_naga::Profile::NAGA).unwrap();
    }
}

#[test]
fn fold_computes_integer_and_bool_operations_on_literals() {
    let text = "iridia-ir 1
fn main {
  %0: u32 = u32(4294967295)
  %1: u32 = u32(2)
  %2: u32 = iadd %0, %1
  %3: i32 = i32(-3)
  %4: i32 = i32(2)
  %5: bool = scmp.lt %3, %4
  %6: bool = ucmp.lt %3, %4
  %7: u32 = u32(0)
  %8: u32 = udiv.allones %1, %7
  %9: u32 = urem.byone %1, %7
  %10: u32 = udiv.undef %1, %7
  %11: i32 = smin %3, %4
  %12: bool = and %5, %6
  %13: bool = not %12
  %14: u32 = select %13, %2, %8
  %15: vec2<u32> = compose %14, %9
  %16: vec2<u32> = %15.yx
  %17: f32 = f32(1.5)
  %18: f32 = fneg %17
  %19: f32 = fadd %17, %17
  %20: f32 = input 0.x
  return
}
";
    let mut entry = iridia_ir::text::parse(text).unwrap();
    let folded = iridia_passes::fold(&mut entry);
    let lit = |v: usize| match entry.values[v].op {
        iridia_ir::Op::Literal(bits) => Some(bits),
        _ => None,
    };
    // wrapping in the type's width; a signed comparison of the bits as signed, an unsigned one as unsigned
    assert_eq!(lit(2), Some([1, 0, 0, 0]));
    assert_eq!(lit(5), Some([1, 0, 0, 0]));
    assert_eq!(lit(6), Some([0, 0, 0, 0]));
    // a division by zero by its rule; one whose rule is undefined is the target's
    assert_eq!(lit(8), Some([u64::from(u32::MAX), 0, 0, 0]));
    assert_eq!(lit(9), Some([0, 0, 0, 0]));
    assert_eq!(lit(10), None);
    assert_eq!(lit(11), Some([u64::from((-3i32).cast_unsigned()), 0, 0, 0]));
    assert_eq!(lit(13), Some([1, 0, 0, 0]));
    assert_eq!(lit(14), Some([1, 0, 0, 0]));
    assert_eq!(lit(16), Some([2, 1, 0, 0]));
    // a sign flip is exact everywhere; float arithmetic depends on the target's denormals and precision
    assert_eq!(lit(18), Some([u64::from((-1.5f32).to_bits()), 0, 0, 0]));
    assert_eq!(lit(19), None);
    assert_eq!(lit(20), None);
    assert_eq!(folded, 12);
    assert_eq!(iridia_passes::fold(&mut entry), 0);
}

#[test]
fn specialise_folds_a_texture_dimension_behind_a_comparison() {
    // the ps_1_x shape: a sampler of unknown dimension reads its dimension as state and picks the sample by it
    let text = "iridia-ir 1
fn main {
  %0: u32 = state r0
  %1: u32 = u32(4)
  %2: bool = ucmp.eq %0, %1
  %3: f32 = input 0.x
  if %2 {
    %4: f32 = fneg %3
    yield %4
  } else {
    yield %3
  } -> %5: f32
  output 0.x = %5
  return
}
";
    let state = Resource {
        id: ResourceId(0),
        kind: ResourceKind::PipelineState(StateParam::TextureDim(0)),
        count: 1,
        provenance: ResourceProvenance {
            name: "dim0".into(),
            ..ResourceProvenance::default()
        },
    };
    let p = from_text(text, &[F1], &[F1], vec![state]);
    for (dim, want) in [(4u32, -5.0f32), (2, 5.0)] {
        let mut q = p.clone();
        iridia_passes::specialise(&mut q, &[(ResourceId(0), [u64::from(dim), 0, 0, 0])]).unwrap();
        iridia_passes::simplify(&mut q.entry);
        iridia_ir::validate(&q).unwrap();
        let printed = iridia_ir::text::print(&q.entry);
        assert!(
            !printed.contains("if ") && !printed.contains("state"),
            "{printed}"
        );
        let (out, _) = run(&q, &[f(5.0)], &[]).unwrap();
        assert_eq!(out[0].unwrap()[0], want.to_bits());
        let (out, _) = run(&p, &[f(5.0)], &[(ResourceId(0), [dim, 0, 0, 0])]).unwrap();
        assert_eq!(out[0].unwrap()[0], want.to_bits());
        iridia_naga::wgsl(&q, &iridia_naga::Profile::BROWSER).unwrap();
    }
}

#[test]
fn flatten_makes_blocks_of_every_construct_and_structurize_takes_them_back() {
    // b = x < 0 ? -x (leaving the block early) : x; then double b until it reaches 10, carried as a loop parameter
    let text = "iridia-ir 1
fn main {
  %0: f32 = input 0.x
  %1: f32 = f32(0.0)
  %2: f32 = f32(10.0)
  %3: f32 = f32(2.0)
  block '0 {
    %4: bool = fcmp.lt %0, %1
    if %4 {
      %5: f32 = fneg %0
      br '0 %5
    } else {
      yield
    }
    yield %0
  } -> %6: f32
  loop '1 (%7: f32) = (%6) {
    %8: bool = fcmp.ge %7, %2
    if %8 {
      br '1 %7
    } else {
      yield
    }
    %9: f32 = fmul.ieee %7, %3
    continue '1 %9
  } -> %10: f32
  output 0.x = %10
  return
}
";
    let p = from_text(text, &[F1], &[F1], Vec::new());
    let mut graph = p.clone();
    iridia_passes::flatten(&mut graph.entry).unwrap();
    iridia_ir::validate(&graph).unwrap_or_else(|e| panic!("{e:?}\n{}", iridia_ir::text::print(&graph.entry)));
    let iridia_ir::Body::Cfg(blocks) = &graph.entry.body else {
        panic!("not a graph")
    };
    // the loop's header carries its parameter, and the block's and the loop's exits their results
    assert!(blocks.iter().any(|b| b.params == [iridia_ir::ValueId(7)]));
    assert!(blocks.iter().any(|b| b.params == [iridia_ir::ValueId(6)]));
    assert!(blocks.iter().any(|b| b.params == [iridia_ir::ValueId(10)]));
    let mut again = graph.clone();
    iridia_passes::structurize(&mut again.entry).unwrap();
    iridia_ir::validate(&again).unwrap_or_else(|e| panic!("{e:?}\n{}", iridia_ir::text::print(&again.entry)));
    // -3 -> 3 -> 6 -> 12; 0.5 -> 16; 20 stays; -40 -> 40
    for (x, want) in [(-3.0f32, 12.0f32), (0.5, 16.0), (20.0, 20.0), (-40.0, 40.0)] {
        for prog in [&p, &graph, &again] {
            let (out, _) = run(prog, &[f(x)], &[]).unwrap();
            assert_eq!(out[0].unwrap()[0], want.to_bits(), "x {x}");
        }
        let mut flagged = again.clone();
        iridia_passes::break_flags(&mut flagged.entry).unwrap();
        support::agrees_flat(&flagged, &[f(x)], &[]);
    }
}

#[test]
fn flatten_and_structurize_leave_the_other_form_alone_and_flatten_refuses_an_exit_to_no_construct() {
    let ok = "iridia-ir 1
fn main {
  %0: f32 = input 0.x
  output 0.x = %0
  return
}
";
    let mut p = from_text(ok, &[F1], &[F1], Vec::new());
    let mut bad = p.clone();
    let iridia_ir::Body::Regions(r) = &mut bad.entry.body else {
        panic!("not regions")
    };
    r.end = Some(iridia_ir::End::Br(iridia_ir::LabelId(3), Vec::new()));
    let e = iridia_passes::flatten(&mut bad.entry).unwrap_err();
    assert!(e.contains("label 3"), "{e}");
    // structurize leaves regions as they are, flatten a graph
    let regions = p.entry.body.clone();
    iridia_passes::structurize(&mut p.entry).unwrap();
    assert_eq!(p.entry.body, regions);
    iridia_passes::flatten(&mut p.entry).unwrap();
    let graph = p.entry.body.clone();
    iridia_passes::flatten(&mut p.entry).unwrap();
    assert_eq!(p.entry.body, graph);
}

#[test]
fn specialise_takes_away_the_textures_only_the_folded_branch_sampled() {
    // a ps_1_x stage of unknown dimension: a cube texture and a 2D one at the same stage, picked by the state; a third
    // texture the program never samples stays
    let text = "iridia-ir 1
fn main {
  %0: u32 = state r0
  %1: u32 = u32(4)
  %2: bool = ucmp.eq %0, %1
  %3: vec4<f32> = input 0.xyzw
  if %2 {
    %4: vec3<f32> = %3.xyz
    %5: vec4<f32> = sample r2, r3, %4
    yield %5
  } else {
    %6: vec2<f32> = %3.xy
    %7: vec4<f32> = sample r1, r3, %6
    yield %7
  } -> %8: vec4<f32>
  output 0.xyzw = %8
  return
}
";
    let resource = |id, kind, name: &str| Resource {
        id: ResourceId(id),
        kind,
        count: 1,
        provenance: ResourceProvenance {
            name: name.into(),
            ..ResourceProvenance::default()
        },
    };
    let texture = |dim| ResourceKind::Texture {
        dim,
        arrayed: false,
        multisampled: false,
        sample: SampleType::Float,
    };
    let resources = vec![
        resource(0, ResourceKind::PipelineState(StateParam::TextureDim(0)), "dim0"),
        resource(1, texture(TextureDim::D2), "s0_D2"),
        resource(2, texture(TextureDim::Cube), "s0_Cube"),
        resource(3, ResourceKind::Sampler { comparison: false }, "s0"),
        resource(4, texture(TextureDim::D2), "s1"),
    ];
    let p = from_text(
        text,
        &[(ScalarType::Float, 4)],
        &[(ScalarType::Float, 4)],
        resources,
    );
    for (dim, kept) in [(TextureDim::Cube, 2), (TextureDim::D2, 1)] {
        let mut q = p.clone();
        iridia_passes::specialise(&mut q, &[(ResourceId(0), [u64::from(dim.state_code()), 0, 0, 0])])
            .unwrap();
        iridia_ir::validate(&q).unwrap();
        let ids: Vec<u32> = q.interface.resources.iter().map(|r| r.id.0).collect();
        assert_eq!(ids, [kept, 3, 4], "{dim:?}");
    }
}

/// A switch with a case of two values, a break and a default, around reads and writes of an array with an immediate
/// start, one of them past its end: the value of `x & 3` and `x` it outputs.
const SWITCH: &str = "iridia-ir 1
fn main {
  var t: [u32; 4] = {u32(10), u32(20), u32(30), u32(40)}
  %0: u32 = input 0.x
  %1: u32 = u32(3)
  %2: u32 = and %0, %1
  switch '0 %2 {
  case 0:
    %3: u32 = load t[%0] oob.zero
    br '0 %3
  case 1, 3:
    %4: u32 = u32(7)
    t[%2] = %4 oob.zero
    %5: u32 = load t[%2] oob.zero
    yield %5
  default:
    %6: u32 = u32(99)
    yield %6
  } -> %7: u32
  output 0.x = %7
  return
}
";

/// What [`SWITCH`] outputs for `x`: element `x` of the table for `x & 3 = 0` (zero past its end), 7 for 1 and 3, else 99.
fn switched(x: u32) -> u32 {
    match x & 3 {
        0 => [10, 20, 30, 40].get(x as usize).copied().unwrap_or(0),
        1 | 3 => 7,
        _ => 99,
    }
}

#[test]
fn a_switch_and_an_array_keep_their_values_through_every_pass_and_lower() {
    const U1: (ScalarType, u8) = (ScalarType::Uint, 1);
    let p = from_text(SWITCH, &[U1], &[U1], Vec::new());
    let mut promoted = p.clone();
    iridia_passes::promote(&mut promoted.entry).unwrap();
    iridia_passes::simplify(&mut promoted.entry);
    // an array is memory: promotion leaves it a variable
    assert!(iridia_ir::text::print(&promoted.entry).contains("var t: [u32; 4]"));
    let mut graph = promoted.clone();
    iridia_passes::flatten(&mut graph.entry).unwrap();
    let mut again = graph.clone();
    iridia_passes::structurize(&mut again.entry).unwrap();
    iridia_passes::simplify(&mut again.entry);
    let mut flagged = again.clone();
    iridia_passes::break_flags(&mut flagged.entry).unwrap();
    for (form, q) in [
        ("promoted", &promoted),
        ("graph", &graph),
        ("restructured", &again),
        ("flagged", &flagged),
    ] {
        iridia_ir::validate(q).unwrap_or_else(|e| panic!("{form}: {e:?}\n{}", iridia_ir::text::print(&q.entry)));
        support::round_trip(&q.entry);
        for x in [0u32, 1, 2, 3, 4, 5, 6, 7, 8, 12, 0xffff_fffc] {
            let (out, _) = run(q, &[[x, 0, 0, 0]], &[]).unwrap();
            assert_eq!(out[0].unwrap()[0], switched(x), "{form}, x {x}");
            if form == "flagged" {
                support::agrees_flat(q, &[[x, 0, 0, 0]], &[]);
            }
        }
    }
    assert!(iridia_ir::text::print(&again.entry).contains("switch '"), "the graph's switch is structured as one");
    for q in [&promoted, &flagged] {
        let wgsl = iridia_naga::wgsl(q, &iridia_naga::Profile::NAGA).unwrap();
        assert!(wgsl.contains("switch") && wgsl.contains("case 1u, 3u"), "{wgsl}");
    }
}

#[test]
fn specialise_takes_the_case_a_known_selector_names() {
    let text = SWITCH.replace("%0: u32 = input 0.x\n  %1: u32 = u32(3)\n  %2: u32 = and %0, %1", "%0: u32 = input 0.x\n  %1: u32 = u32(3)\n  %2: u32 = state r0");
    let state = Resource {
        id: ResourceId(0),
        kind: ResourceKind::PipelineState(StateParam::TextureDim(0)),
        count: 1,
        provenance: ResourceProvenance {
            name: "dim0".into(),
            ..ResourceProvenance::default()
        },
    };
    const U1: (ScalarType, u8) = (ScalarType::Uint, 1);
    let p = from_text(&text, &[U1], &[U1], vec![state]);
    for (known, x, want) in [(0u32, 2u32, 30u32), (0, 9, 0), (1, 0, 7), (3, 0, 7), (2, 0, 99), (4, 0, 99)] {
        let mut q = p.clone();
        iridia_passes::specialise(&mut q, &[(ResourceId(0), [u64::from(known), 0, 0, 0])]).unwrap();
        iridia_passes::simplify(&mut q.entry);
        iridia_ir::validate(&q).unwrap_or_else(|e| panic!("{e:?}\n{}", iridia_ir::text::print(&q.entry)));
        let printed = iridia_ir::text::print(&q.entry);
        assert!(!printed.contains("switch") && !printed.contains("state"), "{printed}");
        let (out, _) = run(&q, &[[x, 0, 0, 0]], &[]).unwrap();
        assert_eq!(out[0].unwrap()[0], want, "state {known}, x {x}");
        let (out, _) = run(&p, &[[x, 0, 0, 0]], &[(ResourceId(0), [known, 0, 0, 0])]).unwrap();
        assert_eq!(out[0].unwrap()[0], want, "unspecialised, state {known}, x {x}");
        iridia_naga::wgsl(&q, &iridia_naga::Profile::NAGA).unwrap_or_else(|e| panic!("{e}\n{printed}"));
    }
}
