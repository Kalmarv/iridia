//! Texture operations. From 2.0, `texld` (plain, projected or biased), `texldl` and `texldd` sample a declared
//! sampler. In pixel 1.x a texture stage is named by the destination register, its dimension and projection are
//! pipeline state, and the 1.1 to 1.3 operations read the texture coordinate of the register they name: `tex`,
//! `texcoord`, `texbem`, `texbeml`, `texreg2ar`, `texreg2gb`, `texreg2rgb`, the `texm3x2` and `texm3x3` families,
//! `texdp3`, `texdp3tex`; in 1.4 `texld`, `texcrd` and `texdepth`.

use iridia_ir::{
    Binary, Cmp as Compare, End, Level, MinMaxNan, Op, Res, SaturateNan, Type, Unary, ValueId,
    interface::{StateParam, TextureDim},
};

use super::Lift;
use crate::{Destination, Instruction, Opcode, Register, Result, error};

/// The coordinate lanes a texture of `dim` samples at.
fn coordinate_lanes(dim: TextureDim) -> &'static [u8] {
    if dim == TextureDim::D2 {
        &[0, 1]
    } else {
        &[0, 1, 2]
    }
}

impl Lift<'_> {
    /// Lift `i` if it is a texture operation; false for every other instruction.
    pub(super) fn texture_instruction(&mut self, i: &Instruction, d: Destination) -> Result<bool> {
        use Opcode::*;
        let pixel_1x = self.source.pixel_1x();
        let early = pixel_1x && self.source.version.minor < 4;
        let v = match i.opcode {
            Tex | TexLdl | TexLdd if !pixel_1x => self.sample(i)?,
            Tex if !early => {
                // pixel 1.4: the destination register names the stage; no projection by state
                let Register::Temporary(stage) = d.register else {
                    return Err(error(self.at, "texld writes a register that is not a temporary"));
                };
                // the coordinate is three lanes at most: `.xyz`, or `.xyw` divided by w for `_dw`
                let coordinate = self.source(i.source(0)?, 7)?;
                self.sample_stage(stage, coordinate, None)?
            }
            TexCoord if !early => self.source(i.source(0)?, d.mask)?,
            TexDepth => {
                // depth = r5.r / r5.g, r5.g at most 1 (wined3d's hardware tests: -0.25 / -0.5 gives 0.5, and a g past
                // 1 is taken as 1); the depth is then within 0..1 as every depth output is, by the pipeline
                let r = self.register(d.register, 3)?;
                let x = self.lane(r, 0)?;
                let y = self.lane(r, 1)?;
                let one = self.b.f32(1.0);
                let y = self.binary(Binary::FMin(MinMaxNan::MinNum), y, one)?;
                let depth = self.binary(Binary::FDiv, x, y)?;
                return self.depth(i, depth).map(|()| true);
            }
            Tex | TexCoord | TexBem | TexBemL | TexReg2Ar | TexReg2Gb | TexReg2Rgb | TexM3x2Pad
            | TexM3x2Tex | TexM3x2Depth | TexM3x3Pad | TexM3x3Tex | TexM3x3 | TexM3x3Spec | TexM3x3VSpec
            | TexDp3 | TexDp3Tex => {
                let Register::Texture(stage) = d.register else {
                    return Err(error(
                        self.at,
                        format!(
                            "{} writes a register that is not a texture register",
                            i.opcode.name()
                        ),
                    ));
                };
                match self.texture_1x(i, stage)? {
                    Some(v) => v,
                    None => return Ok(true),
                }
            }
            _ => return Ok(false),
        };
        self.write(i, d, v)?;
        Ok(true)
    }

    /// Write `depth` (a scalar) to the depth output.
    fn depth(&mut self, i: &Instruction, depth: ValueId) -> Result<()> {
        let v = self.splat(depth, 4)?;
        let d = Destination {
            register: Register::Depth,
            mask: 1,
            saturate: false,
            partial_precision: false,
            shift: 0,
            relative: None,
        };
        self.write(i, d, v)
    }

    /// The texture coordinate input of `tN`, four lanes.
    fn coordinate(&mut self, n: u32) -> Result<ValueId> {
        self.input(Register::Texture(n), 15)
    }

    /// The coordinate divided by its last lane where stage `stage` projects (`D3DTTFF_PROJECTED`).
    fn project(&mut self, stage: u32, coordinate: ValueId) -> Result<ValueId> {
        let state = self.signature.state(StateParam::Projected(stage))?;
        let projected = self.b.value(Type::BOOL, Op::LoadState(state));
        self.divide_by_w(coordinate, projected)
    }

    /// The coordinate divided by its last lane where `projected` holds.
    fn divide_by_w(&mut self, coordinate: ValueId, projected: ValueId) -> Result<ValueId> {
        let w = self.lane(coordinate, 3)?;
        let w = self.splat(w, 4)?;
        let divided = self.binary(Binary::FDiv, coordinate, w)?;
        self.select(projected, divided, coordinate)
    }

    /// The x, y and z dot product of `tN`'s texture coordinate with `with`.
    fn row(&mut self, n: u32, with: ValueId) -> Result<ValueId> {
        let t = self.coordinate(n)?;
        self.dot(t, with, &[0, 1, 2])
    }

    /// One of the pixel 1.1 to 1.3 texture operations on stage (and register) `stage`; `None` for the pads, which
    /// write nothing but a row for the operation that follows.
    fn texture_1x(&mut self, i: &Instruction, stage: u32) -> Result<Option<ValueId>> {
        use Opcode::*;
        let zero = self.b.f32(0.0);
        let one = self.b.f32(1.0);
        Ok(Some(match i.opcode {
            Tex => {
                let t = self.coordinate(stage)?;
                let state = self.signature.state(StateParam::Projected(stage))?;
                let projected = self.b.value(Type::BOOL, Op::LoadState(state));
                self.sample_stage(stage, t, Some(projected))?
            }
            TexCoord => {
                // the coordinate within 0 to 1, and w 1
                let t = self.coordinate(stage)?;
                let t = self.unary(Unary::Saturate(SaturateNan::Zero), t)?;
                let x = self.lane(t, 0)?;
                let y = self.lane(t, 1)?;
                let z = self.lane(t, 2)?;
                self.op(Op::Compose(vec![x, y, z, one]))?
            }
            TexBem | TexBemL => {
                let t = self.coordinate(stage)?;
                let c = self.project(stage, t)?;
                let width = if i.opcode == TexBemL { 7 } else { 3 };
                let offset = self.source(i.source(0)?, width)?;
                let xy = self.bump(stage, c, offset)?;
                let x = self.lane(xy, 0)?;
                let y = self.lane(xy, 1)?;
                let z = self.lane(c, 2)?;
                let w = self.lane(c, 3)?;
                let coordinate = self.op(Op::Compose(vec![x, y, z, w]))?;
                let mut v = self.sample_stage(stage, coordinate, None)?;
                if i.opcode == TexBemL {
                    // times the offset's z scaled and offset by the stage's luminance state, within 0 to 1
                    let scale = self.signature.state(StateParam::BumpLScale(stage))?;
                    let scale = self.b.value(Type::F32, Op::LoadState(scale));
                    let offset_state = self.signature.state(StateParam::BumpLOffset(stage))?;
                    let bias = self.b.value(Type::F32, Op::LoadState(offset_state));
                    let z = self.lane(offset, 2)?;
                    let scaled = self.mul(z, scale)?;
                    let luminance = self.binary(Binary::FAdd, scaled, bias)?;
                    let luminance = self.unary(Unary::Saturate(SaturateNan::Zero), luminance)?;
                    let luminance = self.splat(luminance, 4)?;
                    v = self.mul(v, luminance)?;
                }
                v
            }
            TexReg2Ar | TexReg2Gb | TexReg2Rgb => {
                let s = self.source(i.source(0)?, 15)?;
                let lanes: [u8; 4] = match i.opcode {
                    TexReg2Ar => [3, 0, 0, 0],
                    TexReg2Gb => [1, 2, 2, 2],
                    _ => [0, 1, 2, 2],
                };
                let coordinate = self.swizzle(s, &lanes)?;
                self.sample_stage(stage, coordinate, None)?
            }
            TexM3x2Pad | TexM3x3Pad => {
                let n = self.source(i.source(0)?, 7)?;
                let r = self.row(stage, n)?;
                self.rows.push(r);
                return Ok(None);
            }
            TexM3x2Tex | TexM3x2Depth | TexM3x3Tex | TexM3x3 | TexM3x3Spec | TexM3x3VSpec => {
                let pads = if matches!(i.opcode, TexM3x2Tex | TexM3x2Depth) {
                    1
                } else {
                    2
                };
                if self.rows.len() != pads {
                    return Err(error(
                        self.at,
                        format!(
                            "{} follows {} pad rows, where it takes {pads}",
                            i.opcode.name(),
                            self.rows.len()
                        ),
                    ));
                }
                let n = self.source(i.source(0)?, 7)?;
                let last = self.row(stage, n)?;
                let mut rows = std::mem::take(&mut self.rows);
                rows.push(last);
                match i.opcode {
                    TexM3x2Depth => {
                        // z / w, and 1 where w is 0
                        let (z, w) = (rows.first().copied().unwrap_or(zero), last);
                        let none = self.binary(Binary::FCmp(Compare::Eq), w, zero)?;
                        let ratio = self.binary(Binary::FDiv, z, w)?;
                        let depth = self.select(none, one, ratio)?;
                        self.depth(i, depth)?;
                        return Ok(None);
                    }
                    TexM3x3 => {
                        rows.push(one);
                        self.op(Op::Compose(rows))?
                    }
                    TexM3x2Tex | TexM3x3Tex => {
                        rows.resize(4, zero);
                        let coordinate = self.op(Op::Compose(rows))?;
                        self.sample_stage(stage, coordinate, None)?
                    }
                    _ => {
                        // the reflection of the eye vector about the normal: 2 * (N.E / N.N) * N - E
                        let normal = self.op(Op::Compose(rows))?;
                        let eye = if i.opcode == TexM3x3Spec {
                            let e = self.source(i.source(1)?, 7)?;
                            self.swizzle(e, &[0, 1, 2])?
                        } else {
                            // the w lanes of the three coordinates
                            let mut lanes = Vec::new();
                            for k in 0..3u32 {
                                let t = self.coordinate(stage.saturating_add(k).saturating_sub(2))?;
                                lanes.push(self.lane(t, 3)?);
                            }
                            self.op(Op::Compose(lanes))?
                        };
                        let ne = self.op(Op::Dot(normal, eye))?;
                        let nn = self.op(Op::Dot(normal, normal))?;
                        let ratio = self.binary(Binary::FDiv, ne, nn)?;
                        let two = self.b.f32(2.0);
                        let twice = self.mul(two, ratio)?;
                        let twice = self.splat(twice, 3)?;
                        let scaled = self.mul(normal, twice)?;
                        let reflected = self.binary(Binary::FSub, scaled, eye)?;
                        let x = self.lane(reflected, 0)?;
                        let y = self.lane(reflected, 1)?;
                        let z = self.lane(reflected, 2)?;
                        let coordinate = self.op(Op::Compose(vec![x, y, z, zero]))?;
                        self.sample_stage(stage, coordinate, None)?
                    }
                }
            }
            TexDp3 | TexDp3Tex => {
                let n = self.source(i.source(0)?, 7)?;
                let d = self.row(stage, n)?;
                if i.opcode == TexDp3 {
                    self.splat(d, 4)?
                } else {
                    let coordinate = self.op(Op::Compose(vec![d, zero, zero, zero]))?;
                    self.sample_stage(stage, coordinate, None)?
                }
            }
            _ => {
                return Err(error(
                    self.at,
                    format!("{} is not a texture operation", i.opcode.name()),
                ));
            }
        }))
    }

    /// Sample texture stage `stage` at the lanes of a four-lane coordinate its dimension takes: of the one texture its
    /// dimension is known for, or of the one `StateParam::TextureDim` picks (cube 4, volume 3, else 2D). Where
    /// `projected` is given and holds, a 2D or volume coordinate is divided by its w first; a cube one is not, as
    /// wined3d (the direction is the same unless w is negative).
    fn sample_stage(
        &mut self,
        stage: u32,
        coordinate: ValueId,
        projected: Option<ValueId>,
    ) -> Result<ValueId> {
        let sampler = self
            .signature
            .samplers
            .get(&stage)
            .cloned()
            .ok_or_else(|| error(self.at, format!("texture stage {stage} is not sampled")))?;
        let sample = |l: &mut Self, dim: TextureDim, texture| -> Result<ValueId> {
            let coordinate = match projected {
                Some(p) if dim != TextureDim::Cube => l.divide_by_w(coordinate, p)?,
                _ => coordinate,
            };
            let coord = l.swizzle(coordinate, coordinate_lanes(dim))?;
            Ok(l.b.value(
                Type::F32X4,
                Op::Sample {
                    texture: Res::of(texture),
                    sampler: Res::of(sampler.sampler),
                    coord,
                    level: Level::Auto,
                    compare: None,
                    offset: [0; 3],
                    clamp: None,
                },
            ))
        };
        let (Some(state), [(d2, t2), (cube, tc), (d3, t3)]) =
            (sampler.dim_state, sampler.textures.as_slice())
        else {
            let &(dim, texture) = sampler
                .textures
                .first()
                .ok_or_else(|| error(self.at, format!("texture stage {stage} has no texture")))?;
            return sample(self, dim, texture);
        };
        let code = self.b.value(Type::U32, Op::LoadState(state));
        let is_cube = self.b.u32(cube.state_code());
        let is_cube = self.binary(Binary::UCmp(Compare::Eq), code, is_cube)?;
        self.b.open_if(is_cube);
        let v = sample(self, *cube, *tc)?;
        self.b.else_(End::Yield(vec![v])).map_err(|e| error(self.at, e))?;
        let is_volume = self.b.u32(d3.state_code());
        let is_volume = self.binary(Binary::UCmp(Compare::Eq), code, is_volume)?;
        self.b.open_if(is_volume);
        let v = sample(self, *d3, *t3)?;
        self.b.else_(End::Yield(vec![v])).map_err(|e| error(self.at, e))?;
        let v = sample(self, *d2, *t2)?;
        let inner = self
            .b
            .close(End::Yield(vec![v]), &[Type::F32X4])
            .map_err(|e| error(self.at, e))?;
        let outer = self
            .b
            .close(End::Yield(inner), &[Type::F32X4])
            .map_err(|e| error(self.at, e))?;
        outer
            .first()
            .copied()
            .ok_or_else(|| error(self.at, "a texture dimension pick with no result"))
    }

    /// `texld`, `texldp`, `texldb`, `texldl` and `texldd` from 2.0: the declared sampler at the coordinate's lanes its
    /// dimension takes, the result swizzled by the sampler's swizzle in 3.0.
    fn sample(&mut self, i: &Instruction) -> Result<ValueId> {
        let s = i.source(1)?;
        let Register::Sampler(slot) = s.register else {
            return Err(error(self.at, "TEX source 1 is not a sampler"));
        };
        if s.absolute() || s.negate() || (self.source.version.major >= 3 && s.swizzle != [0, 1, 2, 3]) {
            return Err(error(self.at, "modified/swizzled TEX sampler"));
        }
        let sampler = self
            .signature
            .samplers
            .get(&slot)
            .cloned()
            .ok_or_else(|| error(self.at, format!("undeclared sampler s{slot}")))?;
        let &(dim, texture) = sampler
            .textures
            .first()
            .ok_or_else(|| error(self.at, format!("sampler s{slot} has no texture")))?;
        let lanes = coordinate_lanes(dim);
        // `texldp` divides by w but for a cube texture, as wined3d
        let projects = i.opcode == Opcode::Tex && i.control == 1 && dim != TextureDim::Cube;
        let mut required = if dim == TextureDim::D2 { 3 } else { 7 };
        if projects || (i.opcode == Opcode::Tex && i.control == 2) || i.opcode == Opcode::TexLdl {
            required |= 8;
        }
        let coordinates = self.source(i.source(0)?, required)?;
        let mut coord = self.swizzle(coordinates, lanes)?;
        let width = u8::try_from(lanes.len()).unwrap_or(3);
        let level = match (i.opcode, i.control) {
            (Opcode::TexLdl, _) => Level::Exact(self.lane(coordinates, 3)?),
            (Opcode::TexLdd, _) => {
                let dx = self.source(i.source(2)?, required)?;
                let dx = self.swizzle(dx, lanes)?;
                let dy = self.source(i.source(3)?, required)?;
                let dy = self.swizzle(dy, lanes)?;
                Level::Grad(dx, dy)
            }
            (_, 0) => Level::Auto,
            (_, 1) if !projects => Level::Auto,
            (_, 1) => {
                let w = self.lane(coordinates, 3)?;
                let w = self.splat(w, width)?;
                coord = self.binary(Binary::FDiv, coord, w)?;
                Level::Auto
            }
            (_, 2) => Level::Bias(self.lane(coordinates, 3)?),
            _ => return Err(error(self.at, "TEX control 3 (projection and bias together)")),
        };
        let op = Op::Sample {
            texture: Res::of(texture),
            sampler: Res::of(sampler.sampler),
            coord,
            level,
            compare: None,
            offset: [0; 3],
            clamp: None,
        };
        // a 2.x sampler's swizzle is not applied, as DXVK; a 3.0 one is checked above to be the identity
        Ok(self.b.value(Type::F32X4, op))
    }
}
