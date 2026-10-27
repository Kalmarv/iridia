//! D3D9 sampler-state pairs, mapped without supplying unstated defaults.
use crate::{Result, error};
use iridia_interface::{AddressMode, SamplerState};
use iridia_interface::{Filter, MipFilter};

/// Map native D3DSAMP pairs. Unknown states and values are refused at the pair's word offset.
pub fn sampler_states(states: impl IntoIterator<Item = (u32, u32)>) -> Result<SamplerState> {
    let mut out = SamplerState::default();
    for (index, (kind, value)) in states.into_iter().enumerate() {
        // a pair is two words
        let at = index.saturating_mul(2);
        let address = |value: u32| match value {
            1 => Ok(Some(AddressMode::Repeat)),
            2 => Ok(Some(AddressMode::Mirror)),
            3 => Ok(Some(AddressMode::Clamp)),
            4 => Ok(Some(AddressMode::Border)),
            5 => Ok(Some(AddressMode::MirrorOnce)),
            x => Err(error(
                at,
                format!("sampler address {x}, where D3DTADDRESS is 1 to 5"),
            )),
        };
        let filter = |value: u32| match value {
            1 => Ok(Some(Filter::Nearest)),
            2 => Ok(Some(Filter::Linear)),
            x => Err(error(
                at,
                format!("sampler filter {x}, where D3DTEXF is 1 (point) or 2 (linear) here"),
            )),
        };
        match kind {
            1 => out.wrap[0] = address(value)?,
            2 => out.wrap[1] = address(value)?,
            3 => out.wrap[2] = address(value)?,
            5 => out.filters.mag = filter(value)?,
            6 => out.filters.min = filter(value)?,
            7 => {
                out.filters.mip = Some(match value {
                    0 => MipFilter::None,
                    1 => MipFilter::Nearest,
                    2 => MipFilter::Linear,
                    x => {
                        return Err(error(
                            at,
                            format!("mip filter {x}, where D3DTEXF is 0 (none), 1 (point) or 2 (linear)"),
                        ));
                    }
                })
            }
            8 => {
                let bias = f32::from_bits(value);
                if !bias.is_finite() {
                    return Err(error(
                        at,
                        format!("sampler LOD bias {value:#010x} is not a finite float"),
                    ));
                }
                out.mip_lod_bias = Some(bias);
            }
            // An anisotropic filter's enable is independent of its limit in D3D9. Until both
            // per-filter enables can be represented, only the non-anisotropic limit is accepted.
            10 if value == 1 => out.max_anisotropy = Some(1),
            x => {
                return Err(error(
                    at,
                    format!(
                        "sampler state {x} not resolved; the readable states are 1 to 3 (address), 5 to 7 (filters), 8 (LOD bias) and 10 set to 1 (anisotropy)"
                    ),
                ));
            }
        }
    }
    Ok(out)
}
