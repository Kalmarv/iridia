use std::collections::BTreeMap;

use iridia::{Stage, TextureDim};

use crate::{ConstantTable, Result, error};

/// A float register in the supported stage-specific banks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Register {
    /// Temporary register `rN`, N below 32.
    Temporary(u32),
    /// Input register `vN`: below 16 in a vertex program, below 10 in a pixel program.
    Input(u32),
    /// Float constant register `cN`: below 256 in a vertex program, below 224 in a pixel program.
    Constant(u32),
    /// Sampler `sN`: below 4 in a vertex program, below 16 in a pixel program.
    Sampler(u32),
    /// Colour output `oCN` of a pixel program, N below 4.
    Target(u32),
    /// A declared vertex output register, distinct from pixel render targets.
    Output(u32),
    /// The vertex address register `a0`, written by `mova` and read as the index of a relative constant.
    Address,
}

/// A source token after decoding its swizzle and modifier.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Source {
    /// The register read.
    pub register: Register,
    /// The four source lanes (0 x, 1 y, 2 z, 3 w) that supply the operand's x, y, z and w, from the two-bit fields of the
    /// token at bits 16, 18, 20 and 22.
    pub swizzle: [u8; 4],
    /// True for the `abs` modifier (the token's source modifier 11, or 12 with negation).
    pub absolute: bool,
    /// True for the negate modifier (source modifier 1, or 12 with the absolute value taken first).
    pub negate: bool,
    /// For a float constant `c[a0.l + N]` of a vertex program, the lane `l` (0 x to 3 w) of the address register `a0`
    /// that is added to the register number; `None` for a plain register.
    pub relative: Option<u8>,
}

impl Source {
    /// Source register lanes needed for selected result lanes after the operand's swizzle.
    pub(crate) fn read_mask(self, needed: u8) -> u8 {
        self.swizzle
            .iter()
            .enumerate()
            .filter(|(n, _)| needed & (1 << n) != 0)
            .fold(0, |mask, (_, lane)| mask | (1 << lane))
    }
}

/// A destination token; masks keep their original lane positions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Destination {
    /// The register written.
    pub register: Register,
    /// The lanes written, bit 0 x to bit 3 w (bits 16 to 19 of the token); never empty.
    pub mask: u8,
    /// True for the `_sat` modifier (clamp the result to 0..1).
    pub saturate: bool,
    /// A request that hardware may execute at partial precision. The lifter uses f32 and reports it.
    pub partial_precision: bool,
}

/// The operations this lifter reads. Declarations and immediate definitions are kept separately.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Opcode {
    /// `nop` (opcode 0).
    Nop,
    /// `mov` (1): copy.
    Mov,
    /// `add` (2).
    Add,
    /// `sub` (3).
    Sub,
    /// `mad` (4): multiply and add.
    Mad,
    /// `mul` (5).
    Mul,
    /// `rcp` (6): reciprocal of the x lane.
    Rcp,
    /// `rsq` (7): reciprocal square root of the x lane.
    Rsq,
    /// `dp3` (8): three-lane dot product.
    Dp3,
    /// `dp4` (9): four-lane dot product.
    Dp4,
    /// `min` (10).
    Min,
    /// `max` (11).
    Max,
    /// `slt` (12): 1 where the first operand is less than the second, else 0.
    Slt,
    /// `sge` (13): 1 where the first operand is at least the second, else 0.
    Sge,
    /// `exp` (14): 2 raised to the x lane.
    Exp,
    /// `log` (15): base 2 logarithm of the x lane.
    Log,
    /// `lrp` (18): linear interpolation.
    Lrp,
    /// `frc` (19): fractional part.
    Frc,
    /// `pow` (32): first operand's x lane raised to the second's.
    Pow,
    /// `abs` (35).
    Abs,
    /// `nrm` (36): normalise the three-lane vector.
    Nrm,
    /// `sincos` (37): the cosine of the x lane in x and its sine in y (the Shader Model 3 form, one source).
    SinCos,
    /// `mova` (46): round the operand to the nearest integer into the vertex address register.
    Mova,
    /// `texkill` (65): discard the pixel when any lane of the register is negative.
    TexKill,
    /// `texld` (66): sample a texture; `Instruction::control` says plain, projected or biased.
    Tex,
    /// `cmp` (88): pick the second operand where the first is at least zero, the third elsewhere.
    Cmp,
    /// `dp2add` (90): two-lane dot product plus the x lane of the third operand.
    Dp2Add,
}

impl Opcode {
    fn decode(op: u32, at: usize) -> Result<(Self, usize)> {
        use Opcode::*;
        Ok(match op {
            0 => (Nop, 0),
            1 => (Mov, 1),
            2 => (Add, 2),
            3 => (Sub, 2),
            4 => (Mad, 3),
            5 => (Mul, 2),
            6 => (Rcp, 1),
            7 => (Rsq, 1),
            8 => (Dp3, 2),
            9 => (Dp4, 2),
            10 => (Min, 2),
            11 => (Max, 2),
            12 => (Slt, 2),
            13 => (Sge, 2),
            14 => (Exp, 1),
            15 => (Log, 1),
            18 => (Lrp, 3),
            19 => (Frc, 1),
            32 => (Pow, 2),
            35 => (Abs, 1),
            36 => (Nrm, 1),
            37 => (SinCos, 1),
            46 => (Mova, 1),
            65 => (TexKill, 0),
            66 => (Tex, 2),
            88 => (Cmp, 3),
            90 => (Dp2Add, 3),
            _ => {
                let name = match op {
                    25 => "CALL",
                    26 => "CALLNZ",
                    27 => "LOOP",
                    28 => "RET",
                    29 => "ENDLOOP",
                    30 => "LABEL",
                    33 => "CRS",
                    34 => "SGN",
                    38 => "REP",
                    39 => "ENDREP",
                    40 => "IF",
                    41 => "IFC",
                    42 => "ELSE",
                    43 => "ENDIF",
                    44 => "BREAK",
                    45 => "BREAKC",
                    47 => "DEFB",
                    48 => "DEFI",
                    91 => "DSX",
                    92 => "DSY",
                    93 => "TEXLDD",
                    94 => "SETP",
                    95 => "TEXLDL",
                    96 => "BREAKP",
                    _ => "unrecognized",
                };
                return Err(error(
                    at,
                    format!("opcode {op} {name} outside the supported SM3 subset"),
                ));
            }
        })
    }
}

/// An executable operation. `control` is TEX's projection/bias mode (0, 1, 2); zero for arithmetic.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Instruction {
    /// Offset of the instruction token in 32-bit words from the version token, the position a refusal and
    /// `Projection::after_word` name.
    pub word: usize,
    /// The operation.
    pub opcode: Opcode,
    /// The register written (for `texkill`, the register tested, which may be an input); `None` for `nop`.
    pub destination: Option<Destination>,
    /// The source operands in the order the token stream writes them.
    pub sources: Vec<Source>,
    /// For `texld`, the bits 16 to 17 of the instruction token: 0 plain, 1 projected (`texldp`), 2 biased (`texldb`); 0
    /// for every other operation.
    pub control: u8,
}

impl Instruction {
    /// Source operand `n`, an error naming the instruction when it has fewer.
    pub(crate) fn source(&self, n: usize) -> Result<Source> {
        self.sources.get(n).copied().ok_or_else(|| {
            error(
                self.word,
                format!(
                    "{:?} reads source {n}, where it has {}",
                    self.opcode,
                    self.sources.len()
                ),
            )
        })
    }

    /// Arithmetic operand lanes before swizzle; shared by execution and backwards output slicing.
    pub(crate) fn source_mask(&self, operand: usize) -> u8 {
        let mask = self.destination.map_or(0, |d| d.mask);
        match self.opcode {
            Opcode::Rcp | Opcode::Rsq | Opcode::Exp | Opcode::Log | Opcode::Pow | Opcode::SinCos => 1,
            Opcode::Dp3 => 7,
            Opcode::Dp4 => 15,
            Opcode::Nrm => 7 | mask,
            Opcode::Dp2Add if operand == 2 => 1,
            Opcode::Dp2Add => 3,
            _ => mask,
        }
    }
}

/// A declared pixel or vertex input (a `dcl` of an input register).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Input {
    /// The input register number N of `vN`.
    pub register: u32,
    /// The semantic usage (D3DDECLUSAGE: 0 position, 1 blend weight, 2 blend indices, 3 normal, 5 texcoord, 10 color and
    /// so on), bits 0 to 3 of the declaration word; 0 to 13.
    pub usage: u8,
    /// The semantic index (the 1 of `TEXCOORD1`), bits 16 to 19 of the declaration word.
    pub index: u8,
    /// The lanes declared, bit 0 x to bit 3 w.
    pub mask: u8,
    /// True when the declaration carries the centroid modifier.
    pub centroid: bool,
    /// True when the declaration carries the partial-precision modifier.
    pub partial_precision: bool,
}

/// A COMMENT's words, excluding its token. Unknown FourCCs are retained, not interpreted as instructions.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Comment {
    /// Offset of the comment token in 32-bit words from the version token.
    pub word: usize,
    /// The comment's payload words, including its leading FourCC (`CTAB` for the constant table).
    pub data: Vec<u32>,
}

/// A declared vertex output; disjoint masks can pack distinct semantics into one register.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Output {
    /// The output register number N of `oN`, below 12.
    pub register: u32,
    /// The semantic usage (D3DDECLUSAGE, 0 to 13 except 4, point size), as for `Input::usage`.
    pub usage: u8,
    /// The semantic index.
    pub index: u8,
    /// The lanes of the register this semantic occupies, bit 0 x to bit 3 w.
    pub mask: u8,
}

/// Reflection and executable tokens of one complete `ps_3_0` or `vs_3_0` stream.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Shader {
    /// Vertex or pixel, from the version token (`0xfffe0300` or `0xffff0300`).
    pub stage: Stage,
    /// The declared vertex outputs in declaration order; empty for a pixel program.
    pub outputs: Vec<Output>,
    /// The declared inputs in declaration order.
    pub inputs: Vec<Input>,
    /// The declared samplers by register number, with the texture dimension each declares (2D, cube or volume).
    pub samplers: BTreeMap<u32, TextureDim>,
    /// `def` constants, as exact f32 bit patterns.
    pub definitions: BTreeMap<u32, [u32; 4]>,
    /// The executable instructions in program order.
    pub instructions: Vec<Instruction>,
    /// Every comment, in order, including the one that holds the constant table.
    pub comments: Vec<Comment>,
    /// The CTAB comment decoded; `None` when the program has none.
    pub constants: Option<ConstantTable>,
}

/// The bits of `word` from `shift` that `mask` keeps, which is at most 8 bits wide.
fn bits(word: u32, shift: u32, mask: u32) -> u8 {
    u8::try_from(word.checked_shr(shift).unwrap_or(0) & mask & 0xff).unwrap_or(u8::MAX)
}

fn register(t: u32, at: usize, stage: Stage) -> Result<Register> {
    // bit 13 is the relative-addressing flag of a source token (`source` reads it); bits 14 and 15 are reserved
    if t & 0x8000_0000 == 0 || t & 0xc000 != 0 {
        return Err(error(at, "parameter marker/reserved bits"));
    }
    let kind = ((t >> 28) & 7) | ((t >> 8) & 24);
    let n = t & 2047;
    Ok(match (kind, n) {
        (0, 0..32) => Register::Temporary(n),
        (1, _) if n < if stage == Stage::Vertex { 16 } else { 10 } => Register::Input(n),
        (2, _) if n < if stage == Stage::Vertex { 256 } else { 224 } => Register::Constant(n),
        (3, 0) if stage == Stage::Vertex => Register::Address,
        (6, 0..12) if stage == Stage::Vertex => Register::Output(n),
        (8, 0..4) if stage == Stage::Pixel => Register::Target(n),
        (10, _) if n < if stage == Stage::Vertex { 4 } else { 16 } => Register::Sampler(n),
        _ => {
            return Err(error(
                at,
                format!("register type {kind}, index {n} outside the supported {stage:?} SM3 banks"),
            ));
        }
    })
}

fn destination(t: u32, at: usize, declaration: bool, stage: Stage) -> Result<Destination> {
    if t & 0x2000 != 0 {
        return Err(error(at, "relative addressing of a destination (not supported)"));
    }
    let register = register(t, at, stage)?;
    let mask = bits(t, 16, 15);
    let modifiers = (t >> 20) & 15;
    if mask == 0 || t & 0x0f00_0000 != 0 || modifiers & !if declaration { 6 } else { 3 } != 0 {
        return Err(error(
            at,
            format!(
                "destination mask/shift/modifier {t:#010x}, where the mask is not empty and the shift is 0"
            ),
        ));
    }
    Ok(Destination {
        register,
        mask,
        saturate: modifiers & 1 != 0,
        partial_precision: modifiers & 2 != 0,
    })
}

/// The lane of `a0` that a relative-address token (`a0.x`, `a0.y`, ...) names. The token is a source token whose
/// register is the address register with one lane replicated.
fn relative_lane(t: u32, at: usize) -> Result<u8> {
    let [x, y, z, w] = [16, 18, 20, 22].map(|shift| bits(t, shift, 3));
    // register type 3 (the address register), number 0, no further relative flag
    if t & 0x8000_0000 == 0 || (t >> 28) & 7 != 3 || (t >> 11) & 3 != 0 || t & 2047 != 0 || t & 0x2000 != 0 {
        return Err(error(
            at,
            "relative address token other than the vertex address register a0 (a loop counter aL is not supported)",
        ));
    }
    if (t >> 24) & 15 != 0 || !(x == y && y == z && z == w) {
        return Err(error(
            at,
            "relative address token with a modifier or a swizzle that does not replicate one lane",
        ));
    }
    Ok(x)
}

fn source(t: u32, relative: Option<u32>, at: usize, stage: Stage) -> Result<Source> {
    // the relative flag is for a float constant of a vertex program only
    let flagged = t & 0x2000 != 0;
    let register = register(t & !0x2000, at, stage)?;
    if flagged && !(stage == Stage::Vertex && matches!(register, Register::Constant(_))) {
        return Err(error(
            at,
            "parameter marker/reserved bits or relative addressing (not supported)",
        ));
    }
    let relative = match (flagged, relative) {
        (true, Some(r)) => Some(relative_lane(r, at.saturating_add(1))?),
        (true, None) => return Err(error(at, "relative addressing without its address token")),
        (false, _) => None,
    };
    let modifier = (t >> 24) & 15;
    if !matches!(modifier, 0 | 1 | 11 | 12) {
        return Err(error(
            at,
            format!("source modifier {modifier} outside NONE/NEG/ABS/ABSNEG"),
        ));
    }
    Ok(Source {
        register,
        swizzle: [16, 18, 20, 22].map(|shift| bits(t, shift, 3)),
        absolute: matches!(modifier, 11 | 12),
        negate: matches!(modifier, 1 | 12),
        relative,
    })
}

/// Read a complete raw pixel or vertex shader, including comment/CTAB payloads and END. Lengths, bank limits,
/// declaration order, duplicate definitions, reserved bits and every operand count are checked.
pub fn read(bytes: &[u8]) -> Result<Shader> {
    if !bytes.len().is_multiple_of(4) || bytes.len() > 4 * 1_048_576 {
        return Err(error(
            0,
            format!(
                "token stream of {} bytes is not whole words or exceeds 1,048,576 words",
                bytes.len()
            ),
        ));
    }
    let words: Vec<_> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&word| u32::from_le_bytes(word))
        .collect();
    let version = words.first().copied();
    let stage = match version {
        Some(0xffff_0300) => Stage::Pixel,
        Some(0xfffe_0300) => Stage::Vertex,
        _ => {
            return Err(error(
                0,
                "version: only raw ps_3_0 (0xffff0300) or vs_3_0 (0xfffe0300) is supported",
            ));
        }
    };
    let version = version.unwrap_or_default();
    let mut shader = Shader {
        stage,
        outputs: Vec::new(),
        inputs: Vec::new(),
        samplers: BTreeMap::new(),
        definitions: BTreeMap::new(),
        instructions: Vec::new(),
        comments: Vec::new(),
        constants: None,
    };
    let mut at = 1usize;
    let mut executable = false;
    while let Some(&token) = words.get(at) {
        let word = at;
        // the word of the operand `n` operands after the token
        let operand = |n: usize| word.saturating_add(n);
        at = at.saturating_add(1);
        if token == 0xffff {
            if at != words.len() {
                return Err(error(
                    at,
                    format!("{} words after END", words.len().saturating_sub(at)),
                ));
            }
            return Ok(shader);
        }
        let op = token & 65535;
        let count = if op == 65534 {
            (token >> 16) & 32767
        } else {
            (token >> 24) & 15
        } as usize;
        let args = at
            .checked_add(count)
            .and_then(|end| words.get(at..end))
            .ok_or_else(|| {
                error(
                    word,
                    format!(
                        "instruction/comment of {count} operands extends past the stream's {} words",
                        words.len()
                    ),
                )
            })?;
        at = at.saturating_add(count);
        if op == 65534 {
            if token & 0x8000_0000 != 0 {
                return Err(error(word, "COMMENT reserved bit"));
            }
            if let Some((&tag, rest)) = args.split_first()
                && tag == u32::from_le_bytes(*b"CTAB")
            {
                if shader.constants.is_some() {
                    return Err(error(word, "duplicate CTAB"));
                }
                let data: Vec<_> = rest.iter().flat_map(|w| w.to_le_bytes()).collect();
                shader.constants = Some(crate::ctab::read(&data, operand(2), version)?);
            }
            shader.comments.push(Comment {
                word,
                data: args.to_vec(),
            });
            continue;
        }
        if token & 0xf0ff_0000 != 0 && !(op == 66 && token & 0xf0fc_0000 == 0) {
            return Err(error(
                word,
                format!("instruction controls/predication/coissue {token:#010x}"),
            ));
        }
        if op == 31 {
            if executable {
                return Err(error(word, "DCL after executable code"));
            }
            let &[declaration, destination_token] = args else {
                return Err(error(word, format!("DCL has {count} operands, where it has 2")));
            };
            let d = destination(destination_token, operand(2), true, stage)?;
            if stage == Stage::Vertex && (d.partial_precision || destination_token & 0x0040_0000 != 0) {
                return Err(error(
                    operand(2),
                    "vertex DCL partial precision or centroid modifier",
                ));
            }
            match d.register {
                Register::Output(register) => {
                    if declaration & !0x800f_000f != 0 || declaration & 0x8000_0000 == 0 {
                        return Err(error(operand(1), "DCL semantic reserved bits"));
                    }
                    let usage = bits(declaration, 0, 15);
                    let index = bits(declaration, 16, 15);
                    if usage > 13 || usage == 4 {
                        return Err(error(
                            operand(1),
                            "vertex output usage outside the subset (0 to 13, PSIZE not supported)",
                        ));
                    }
                    if usage == 0 && index == 0 && d.mask != 15 {
                        return Err(error(operand(2), "vertex POSITION0 must declare all four lanes"));
                    }
                    if shader.outputs.iter().any(|o| {
                        (o.register == register && o.mask & d.mask != 0)
                            || (o.usage == usage && o.index == index)
                    }) {
                        return Err(error(word, "overlapping output masks or duplicate semantic"));
                    }
                    shader.outputs.push(Output {
                        register,
                        usage,
                        index,
                        mask: d.mask,
                    });
                }
                Register::Input(register) => {
                    if declaration & !0x800f_000f != 0 || declaration & 0x8000_0000 == 0 {
                        return Err(error(operand(1), "DCL semantic reserved bits"));
                    }
                    let usage = bits(declaration, 0, 15);
                    if usage > 13 {
                        return Err(error(
                            operand(1),
                            format!("DCL usage {usage}, where it is 0 to 13"),
                        ));
                    }
                    let index = bits(declaration, 16, 15);
                    if shader
                        .inputs
                        .iter()
                        .any(|i| i.register == register || (i.usage == usage && i.index == index))
                    {
                        return Err(error(word, "duplicate input register or semantic"));
                    }
                    shader.inputs.push(Input {
                        register,
                        usage,
                        index,
                        mask: d.mask,
                        centroid: destination_token & 0x0040_0000 != 0,
                        partial_precision: d.partial_precision,
                    });
                }
                Register::Sampler(n) => {
                    let dim = match declaration {
                        0x9000_0000 => TextureDim::D2,
                        0x9800_0000 => TextureDim::Cube,
                        0xa000_0000 => TextureDim::D3,
                        _ => {
                            return Err(error(
                                operand(1),
                                format!(
                                    "DCL sampler texture type {declaration:#x}, where it is 0x90000000 (2d), 0x98000000 (cube) or 0xa0000000 (volume)"
                                ),
                            ));
                        }
                    };
                    if d.mask != 15
                        || d.partial_precision
                        || destination_token & 0x0040_0000 != 0
                        || shader.samplers.insert(n, dim).is_some()
                    {
                        return Err(error(word, "duplicate or modified sampler declaration"));
                    }
                }
                _ => {
                    return Err(error(
                        operand(2),
                        "DCL destination is neither input, vertex output nor sampler",
                    ));
                }
            }
            continue;
        }
        if op == 81 {
            if executable {
                return Err(error(word, "DEF after executable code"));
            }
            let &[destination_token, x, y, z, w] = args else {
                return Err(error(word, format!("DEF has {count} operands, where it has 5")));
            };
            let d = destination(destination_token, operand(1), false, stage)?;
            let Register::Constant(n) = d.register else {
                return Err(error(operand(1), "DEF destination is not cN"));
            };
            if d.mask != 15
                || d.saturate
                || d.partial_precision
                || shader.definitions.insert(n, [x, y, z, w]).is_some()
            {
                return Err(error(word, "duplicate or modified DEF"));
            }
            continue;
        }
        let (opcode, sources) = Opcode::decode(op, word)?;
        if stage == Stage::Vertex && matches!(opcode, Opcode::Cmp | Opcode::Dp2Add) {
            return Err(error(word, format!("{opcode:?} is a pixel-only SM3 opcode")));
        }
        if stage == Stage::Pixel && matches!(opcode, Opcode::Slt | Opcode::Sge | Opcode::Mova) {
            return Err(error(word, format!("{opcode:?} is a vertex-only SM3 opcode")));
        }
        if stage == Stage::Vertex && matches!(opcode, Opcode::Tex | Opcode::TexKill) {
            return Err(error(
                word,
                "pixel sampling/discard opcode in vertex stage (vertex texture fetch not supported)",
            ));
        }
        let expected = if opcode == Opcode::Nop {
            0
        } else {
            sources.saturating_add(1)
        };
        // a relatively addressed source is followed by its address token, so the count is the operands' plus those
        let relative_tokens = args
            .iter()
            .skip(1)
            .filter(|&&t| t & 0x8000_0000 != 0 && t & 0x2000 != 0)
            .count();
        if count != expected && !(count > expected && relative_tokens > 0) {
            return Err(error(
                word,
                format!("{opcode:?} has {count} operands, expected {expected}"),
            ));
        }
        let (destination, source_tokens) = match args.split_first() {
            Some((&first, rest)) => (Some(destination(first, operand(1), false, stage)?), rest),
            None => (None, args),
        };
        if let Some(d) = destination {
            let allowed = if opcode == Opcode::Mova {
                d.register == Register::Address
            } else {
                matches!(
                    d.register,
                    Register::Temporary(_) | Register::Target(_) | Register::Output(_)
                ) || (opcode == Opcode::TexKill && matches!(d.register, Register::Input(_)))
            };
            if !allowed {
                return Err(error(
                    operand(1),
                    format!("{opcode:?} destination {:?}", d.register),
                ));
            }
            if let Register::Output(n) = d.register {
                let declared = shader
                    .outputs
                    .iter()
                    .filter(|o| o.register == n)
                    .fold(0, |m, o| m | o.mask);
                if d.mask & !declared != 0 {
                    return Err(error(
                        operand(1),
                        format!(
                            "write to undeclared vertex output lanes {:#x} of o{n}, which declares {declared:#x}",
                            d.mask & !declared
                        ),
                    ));
                }
            }
            if stage == Stage::Vertex && d.partial_precision {
                return Err(error(
                    operand(1),
                    "vertex partial precision modifier not supported",
                ));
            }
            if opcode == Opcode::TexKill && (d.saturate || d.partial_precision) {
                return Err(error(operand(1), "modified TEXKILL destination"));
            }
        }
        let mut sources = Vec::new();
        let mut tokens = source_tokens.iter().enumerate();
        while let Some((i, &t)) = tokens.next() {
            let at = operand(i.saturating_add(2));
            let relative = if t & 0x2000 != 0 {
                tokens.next().map(|(_, &r)| r)
            } else {
                None
            };
            sources.push(source(t, relative, at, stage)?);
        }
        if sources.len() != expected.saturating_sub(1) {
            return Err(error(
                word,
                format!("{opcode:?} has {count} operands, expected {expected} and one per relative address"),
            ));
        }
        shader.instructions.push(Instruction {
            word,
            opcode,
            destination,
            sources,
            control: bits(token, 16, 3),
        });
        executable = true;
    }
    Err(error(at, "missing END"))
}
