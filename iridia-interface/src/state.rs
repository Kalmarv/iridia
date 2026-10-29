//! Fixed-function state as Direct3D states it: the sampler state a texture is read through and the blend, depth,
//! stencil, cull, fill and alpha test a program is drawn with.

use serde::Serialize;

/// How texels are filtered when a texture is magnified or minified (glTF's sampler filters).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub enum Filter {
    /// The nearest texel: crisp texels.
    Nearest,
    /// Bilinear between the four nearest texels.
    Linear,
}
/// How mip levels are used when a texture is minified.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub enum MipFilter {
    /// No mip levels: the base level only (glTF's `NEAREST` or `LINEAR` minification filter).
    None,
    /// The nearest mip level.
    Nearest,
    /// Between the two nearest mip levels (trilinear with a linear `min`).
    Linear,
}
/// A sampler's filters as the source states them: each `None` where it states none (the renderer's default).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub struct Filters {
    /// Magnification.
    pub mag: Option<Filter>,
    /// Minification within a level.
    pub min: Option<Filter>,
    /// Minification across mip levels.
    pub mip: Option<MipFilter>,
}

/// What a texture coordinate outside 0 to 1 reads, along one axis.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AddressMode {
    /// The texture repeats.
    Repeat,
    /// The texture repeats, every other copy mirrored.
    Mirror,
    /// The edge texel.
    Clamp,
    /// The sampler's border colour.
    Border,
    /// Mirrored once about the origin, then the edge texel.
    MirrorOnce,
}
/// A sampler's state as the source stores it, for a texture slot or a program's sampler: each part `None` where the
/// source states none.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct SamplerState {
    /// Addressing along u, v and w.
    pub wrap: [Option<AddressMode>; 3],
    /// The sampler's filters.
    pub filters: Filters,
    /// The most texels along the axis of anisotropy an anisotropic filter takes (1: none beyond the filters'), where
    /// the source states an anisotropic filter.
    pub max_anisotropy: Option<u8>,
    /// The comparison a depth-comparison sampler makes of the reference against the texel, where the sampler is one.
    pub compare: Option<Compare>,
    /// Added to the mip level the hardware computes, where the source states a non-zero bias.
    pub mip_lod_bias: Option<f32>,
    /// The colour (red, green, blue, alpha) a coordinate outside the texture reads along an axis addressed
    /// [`AddressMode::Border`], where the source states it.
    pub border: Option<[f32; 4]>,
    /// The least and the greatest mip level the sampler reads (the level of detail is clamped to them), where the
    /// source states a range; `None` is every level the texture has.
    pub lod: Option<[f32; 2]>,
}

impl SamplerState {
    /// The state agrees with itself: a stated border colour and level range are finite (the range's top may be
    /// `f32::MAX`, every level), the range is not reversed, and a maximum anisotropy is at least 1.
    pub fn verify(&self) -> Result<(), String> {
        if let Some(b) = self.border
            && b.iter().any(|c| !c.is_finite())
        {
            return Err(format!("border colour {b:?} is not finite"));
        }
        if let Some([low, high]) = self.lod
            && (!low.is_finite() || !high.is_finite() || low > high)
        {
            return Err(format!(
                "level range {low}..{high} is not a finite, ordered range"
            ));
        }
        if let Some(b) = self.mip_lod_bias
            && !b.is_finite()
        {
            return Err(format!("mip level bias {b} is not finite"));
        }
        if self.max_anisotropy == Some(0) {
            return Err("a maximum anisotropy of 0".into());
        }
        Ok(())
    }
}
/// A blend factor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BlendFactor {
    /// Zero.
    Zero,
    /// One.
    One,
    /// The source colour.
    SrcColour,
    /// One minus the source colour.
    OneMinusSrcColour,
    /// The source alpha.
    SrcAlpha,
    /// One minus the source alpha.
    OneMinusSrcAlpha,
    /// The colour already in the target.
    DstColour,
    /// One minus the target colour.
    OneMinusDstColour,
    /// The alpha already in the target.
    DstAlpha,
    /// One minus the target alpha.
    OneMinusDstAlpha,
}
/// How the two weighted terms of a blend are combined.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BlendOp {
    /// `source + destination`.
    Add,
    /// `source - destination`.
    Subtract,
    /// `destination - source`.
    ReverseSubtract,
    /// The smaller of the two unweighted values.
    Min,
    /// The larger of the two unweighted values.
    Max,
}
/// `op(src * source, dst * destination)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
pub struct BlendEquation {
    /// The factor the incoming (source) value is multiplied by.
    pub source: BlendFactor,
    /// The factor the value already in the target is multiplied by.
    pub destination: BlendFactor,
    /// How the two products are combined.
    pub op: BlendOp,
}
/// Blending of the colour and the alpha channel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
pub struct Blend {
    /// The equation for the red, green and blue channels.
    pub colour: BlendEquation,
    /// The equation for the alpha channel.
    pub alpha: BlendEquation,
}
/// How two values are compared; the left one is the fragment's own (its depth or alpha, or the stencil reference), the
/// right one what it is tested against.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Compare {
    /// Never passes.
    Never,
    /// Passes where less.
    Less,
    /// Passes where equal.
    Equal,
    /// Passes where less than or equal.
    LessEqual,
    /// Passes where greater.
    Greater,
    /// Passes where not equal.
    NotEqual,
    /// Passes where greater than or equal.
    GreaterEqual,
    /// Always passes.
    Always,
}
/// Which facing of triangle is not drawn.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Cull {
    /// Neither: both facings are drawn.
    None,
    /// Front-facing triangles are culled.
    Front,
    /// Back-facing triangles are culled.
    Back,
}
/// Depth test and write.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
pub struct DepthState {
    /// The depth comparison a fragment must pass.
    pub test: Compare,
    /// Whether a passing fragment writes its depth.
    pub write: bool,
}
/// Alpha test: a pixel is kept where `alpha compare reference`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct AlphaTest {
    /// The comparison of the fragment's alpha with `reference`.
    pub compare: Compare,
    /// The reference alpha, 0 to 1.
    pub reference: f32,
}
/// A stencil operation (D3D12 and Vulkan have the same eight).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StencilOp {
    /// Keep the stored value.
    Keep,
    /// Set it to zero.
    Zero,
    /// Set it to the reference value.
    Replace,
    /// Increment, clamped at the maximum.
    IncrementClamp,
    /// Decrement, clamped at zero.
    DecrementClamp,
    /// Invert its bits.
    Invert,
    /// Increment, wrapping to zero.
    IncrementWrap,
    /// Decrement, wrapping to the maximum.
    DecrementWrap,
}
/// What the stencil test does for one facing.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
pub struct StencilFace {
    /// On a failed stencil test.
    pub fail: StencilOp,
    /// On a passed stencil test and a failed depth test.
    pub depth_fail: StencilOp,
    /// On both passed.
    pub pass: StencilOp,
    /// The comparison of the reference with the stored value.
    pub compare: Compare,
}
/// An enabled stencil test (a disabled one is `None` in [`RenderState::stencil`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
pub struct Stencil {
    /// Mask applied to the reference and the stored value before they are compared.
    pub read_mask: u8,
    /// Mask of the stencil bits a write changes.
    pub write_mask: u8,
    /// The reference value the test compares against and `Replace` writes, where the source states one (D3D12 sets it
    /// per draw: a source that stores it per draw states none).
    pub reference: Option<u8>,
    /// The operations for front-facing triangles.
    pub front: StencilFace,
    /// The operations for back-facing triangles.
    pub back: StencilFace,
}
/// What one render target does with a fragment: its blend and which channels it writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
pub struct TargetState {
    /// `None` where the target is not blended.
    pub blend: Option<Blend>,
    /// The channels written: bit 0 red, 1 green, 2 blue, 3 alpha (`0xf` all).
    pub write_mask: u8,
}
/// A depth bias, D3D12's three terms: `bias = constant * r + slope_scaled * max_slope`, limited to `clamp` when that
/// is not zero (r the smallest resolvable depth difference).
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct DepthBias {
    /// The constant term, in units of the smallest resolvable depth difference.
    pub constant: f32,
    /// The factor of the polygon's maximum depth slope.
    pub slope_scaled: f32,
    /// The most the bias may reach in magnitude; 0 for no limit.
    pub clamp: f32,
}
/// How a triangle is filled.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Fill {
    /// Filled triangles.
    Solid,
    /// Triangle outlines.
    Wireframe,
}
/// Fixed-function state as the source stores it; each `None` where it states none. D3D12-shaped: `blend` is render
/// target 0's blend (what a consumer with one render target reads), and `targets` has every render target's blend
/// and write mask where the source states them, with `targets[0]`'s blend equal to `blend` (a consumer's
/// verification checks it).
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct RenderState {
    /// Blending of render target 0; `None` where it is not blended.
    pub blend: Option<Blend>,
    /// The depth test and write.
    pub depth: Option<DepthState>,
    /// The face culling.
    pub cull: Option<Cull>,
    /// The alpha test.
    pub alpha_test: Option<AlphaTest>,
    /// Per render target, in order.
    pub targets: [Option<TargetState>; 8],
    /// The stencil test, where one is enabled.
    pub stencil: Option<Stencil>,
    /// The depth bias.
    pub depth_bias: Option<DepthBias>,
    /// How triangles are filled.
    pub fill: Option<Fill>,
    /// Whether counter-clockwise triangles are front faces.
    pub front_counter_clockwise: Option<bool>,
}
impl RenderState {
    /// The state is consistent: the write masks fit four channels, the bias is finite, and `blend` is the first
    /// target's blend (a state with a blend has a first target, and the other way round).
    pub fn verify(&self) -> Result<(), String> {
        if let Some(a) = self.alpha_test
            && !a.reference.is_finite()
        {
            return Err("the alpha test's reference is not finite".into());
        }
        for (i, t) in self.targets.iter().enumerate() {
            if let Some(t) = t
                && t.write_mask > 0xf
            {
                return Err(format!(
                    "render target {i}'s write mask {:#x} is wider than four channels",
                    t.write_mask
                ));
            }
        }
        match self.targets[0] {
            Some(first) if first.blend != self.blend => {
                return Err("render target 0's blend is not the state's blend".into());
            }
            None if self.blend.is_some() => {
                return Err("the state has a blend and render target 0 has no state to hold it".into());
            }
            _ => {}
        }
        if let Some(b) = self.depth_bias
            && !(b.constant.is_finite() && b.slope_scaled.is_finite() && b.clamp.is_finite())
        {
            return Err("the depth bias is not finite".into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_render_state_is_consistent() {
        let eq = BlendEquation {
            source: BlendFactor::SrcAlpha,
            destination: BlendFactor::OneMinusSrcAlpha,
            op: BlendOp::Add,
        };
        let blend = Blend {
            colour: eq,
            alpha: eq,
        };
        let mut state = RenderState::default();
        assert_eq!(state.verify(), Ok(()));
        // the first target's blend is the state's
        state.targets[0] = Some(TargetState {
            blend: Some(blend),
            write_mask: 0xf,
        });
        assert!(state.verify().unwrap_err().contains("target 0's blend"));
        state.blend = Some(blend);
        assert_eq!(state.verify(), Ok(()));
        // a blend with no state for the first target to hold it is not the same as no blend
        let mut lone = RenderState {
            blend: Some(blend),
            ..RenderState::default()
        };
        assert!(lone.verify().unwrap_err().contains("no state to hold it"));
        lone.targets[0] = Some(TargetState {
            blend: Some(blend),
            write_mask: 0xf,
        });
        assert_eq!(lone.verify(), Ok(()));
        // other targets are free; a mask is four channels
        state.targets[3] = Some(TargetState {
            blend: None,
            write_mask: 0x1f,
        });
        assert!(state.verify().unwrap_err().contains("target 3's write mask"));
        state.targets[3] = Some(TargetState {
            blend: None,
            write_mask: 0x8,
        });
        state.depth_bias = Some(DepthBias {
            constant: 0.0,
            slope_scaled: f32::NAN,
            clamp: 0.0,
        });
        assert!(state.verify().unwrap_err().contains("depth bias"));
    }

    #[test]
    fn a_sampler_state_is_consistent() {
        let mut state = SamplerState {
            wrap: [Some(AddressMode::Border); 3],
            border: Some([0.0, 0.0, 0.0, 0.0]),
            lod: Some([0.0, 0.0]),
            ..SamplerState::default()
        };
        assert_eq!(state.verify(), Ok(()));
        assert_eq!(SamplerState::default().verify(), Ok(()));
        // Direct3D's greatest level, f32::MAX, is every level and finite
        state.lod = Some([0.0, f32::MAX]);
        assert_eq!(state.verify(), Ok(()));
        state.lod = Some([2.0, 1.0]);
        assert!(state.verify().unwrap_err().contains("level range"));
        state.lod = Some([0.0, f32::INFINITY]);
        assert!(state.verify().unwrap_err().contains("level range"));
        state.lod = None;
        state.border = Some([0.0, f32::NAN, 0.0, 1.0]);
        assert!(state.verify().unwrap_err().contains("border colour"));
        state.border = Some([1.0; 4]);
        state.mip_lod_bias = Some(f32::NAN);
        assert!(state.verify().unwrap_err().contains("bias"));
        state.mip_lod_bias = Some(-0.75);
        state.max_anisotropy = Some(0);
        assert!(state.verify().unwrap_err().contains("anisotropy"));
        state.max_anisotropy = Some(16);
        assert_eq!(state.verify(), Ok(()));
    }
}
