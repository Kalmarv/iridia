//! What a pixel program's own discards and its targets' stated blend show its output lanes are used for
//! ([`output_uses`]), and what its discards compare a constant word with ([`discard_compares`], [`alpha_tests`]). These
//! are observations of the program and its state; what they make a lane or a word stand for is the caller's to decide.
//! Nothing is read from a program's or a target's name.
use iridia_interface::{BlendFactor, OutputUse, RenderState};

use crate::flow::Node;
use crate::{Interface, ProgramGraph, SystemValue};

/// What tests or blends each render target output of a pixel program, in the order of `interface`'s outputs (the
/// stage's program's interface): an output is a target by its system value `SV_Target<n>`, or by its label's index
/// where it has none, and an output that is not a target is left out.
///
/// - [`OutputUse::blend_weight`] is lane 3 where `state`'s blend for the target weighs the colour by the source alpha
///   (the colour equation's source or destination factor `SrcAlpha` or `OneMinusSrcAlpha`);
/// - [`OutputUse::discard_tested`] are the lanes whose written value is among what a discard's conditions are made of
///   (`graph`, the program's value graph; a literal is no evidence, as equal literals are one node). No graph, no lanes.
pub fn output_uses(
    interface: &Interface,
    graph: Option<&ProgramGraph>,
    state: &RenderState,
) -> Vec<OutputUse> {
    let tested = graph.map(|g| {
        g.values
            .reached_from(g.values.discards().iter().flatten().copied())
    });
    interface
        .outputs
        .iter()
        .filter_map(|v| {
            let target = match v.system {
                None => v.label.index,
                Some(SystemValue::Target(n)) => n,
                Some(_) => return None,
            };
            let blend_weight = if weighs_by_source_alpha(state, target) {
                1 << 3
            } else {
                0
            };
            let mut discard_tested = 0u8;
            if let (Some(g), Some(tested)) = (graph, &tested) {
                for lane in 0..4u8 {
                    if let Some(id) = g.values.output_labelled(&g.interface, &v.label, lane)
                        && !matches!(g.values.node(id), Node::Literal(_))
                        && tested.contains(&id)
                    {
                        discard_tested |= 1 << lane;
                    }
                }
            }
            Some(OutputUse {
                label: v.label.clone(),
                target,
                blend_weight,
                discard_tested,
            })
        })
        .collect()
}

mod alpha;
pub use alpha::{AlphaDither, AlphaExpr, AlphaTest, Known, Linear, alpha_tests, discard_compares};

/// Whether target `target`'s stated blend weighs the colour by the source alpha: the state's blend for that target
/// (render target 0's is also the state's own `blend`).
fn weighs_by_source_alpha(state: &RenderState, target: u32) -> bool {
    let blend = usize::try_from(target)
        .ok()
        .and_then(|t| state.targets.get(t).copied().flatten())
        .and_then(|t| t.blend)
        .or(if target == 0 { state.blend } else { None });
    blend.is_some_and(|b| {
        [b.colour.source, b.colour.destination]
            .iter()
            .any(|f| matches!(f, BlendFactor::SrcAlpha | BlendFactor::OneMinusSrcAlpha))
    })
}
