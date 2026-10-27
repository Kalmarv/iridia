use std::collections::BTreeMap;

use iridia::{Stage, TextureDim};

use crate::{ConstantTable, Result, error};

/// A float register in the supported stage-specific banks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Register {
    Temporary(u32),
    Input(u32),
    Constant(u32),
    Sampler(u32),
    Target(u32),
    /// A declared vertex output register, distinct from pixel render targets.
    Output(u32),
}

/// A source token after decoding its swizzle and modifier.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Source {
    pub register: Register,
    pub swizzle: [u8; 4],
    pub absolute: bool,
    pub negate: bool,
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
    pub register: Register,
    pub mask: u8,
    pub saturate: bool,
    /// A request that hardware may execute at partial precision. The lifter uses f32 and reports it.
    pub partial_precision: bool,
}

/// The operations this lifter reads. Declarations and immediate definitions are kept separately.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Opcode {
    Nop,
    Mov,
    Add,
    Sub,
    Mad,
    Mul,
    Rcp,
    Rsq,
    Dp3,
    Dp4,
    Min,
    Max,
    Exp,
    Log,
    Lrp,
    Frc,
    Pow,
    Abs,
    Nrm,
    TexKill,
    Tex,
    Cmp,
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
            14 => (Exp, 1),
            15 => (Log, 1),
            18 => (Lrp, 3),
            19 => (Frc, 1),
            32 => (Pow, 2),
            35 => (Abs, 1),
            36 => (Nrm, 1),
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
                    37 => "SINCOS",
                    38 => "REP",
                    39 => "ENDREP",
                    40 => "IF",
                    41 => "IFC",
                    42 => "ELSE",
                    43 => "ENDIF",
                    44 => "BREAK",
                    45 => "BREAKC",
                    46 => "MOVA",
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
    pub word: usize,
    pub opcode: Opcode,
    pub destination: Option<Destination>,
    pub sources: Vec<Source>,
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
            Opcode::Rcp | Opcode::Rsq | Opcode::Exp | Opcode::Log | Opcode::Pow => 1,
            Opcode::Dp3 => 7,
            Opcode::Dp4 => 15,
            Opcode::Nrm => 7 | mask,
            Opcode::Dp2Add if operand == 2 => 1,
            Opcode::Dp2Add => 3,
            _ => mask,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Input {
    pub register: u32,
    pub usage: u8,
    pub index: u8,
    pub mask: u8,
    pub centroid: bool,
    pub partial_precision: bool,
}

/// A COMMENT's words, excluding its token. Unknown FourCCs are retained, not interpreted as instructions.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Comment {
    pub word: usize,
    pub data: Vec<u32>,
}

/// A declared vertex output; disjoint masks can pack distinct semantics into one register.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Output {
    pub register: u32,
    pub usage: u8,
    pub index: u8,
    pub mask: u8,
}

/// Reflection and executable tokens of one complete `ps_3_0` or `vs_3_0` stream.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Shader {
    pub stage: Stage,
    pub outputs: Vec<Output>,
    pub inputs: Vec<Input>,
    pub samplers: BTreeMap<u32, TextureDim>,
    /// `def` constants, as exact f32 bit patterns.
    pub definitions: BTreeMap<u32, [u32; 4]>,
    pub instructions: Vec<Instruction>,
    pub comments: Vec<Comment>,
    pub constants: Option<ConstantTable>,
}

/// The bits of `word` from `shift` that `mask` keeps, which is at most 8 bits wide.
fn bits(word: u32, shift: u32, mask: u32) -> u8 {
    u8::try_from(word.checked_shr(shift).unwrap_or(0) & mask & 0xff).unwrap_or(u8::MAX)
}

fn register(t: u32, at: usize, stage: Stage) -> Result<Register> {
    if t & 0x8000_0000 == 0 || t & 0xe000 != 0 {
        return Err(error(
            at,
            "parameter marker/reserved bits or relative addressing (not supported)",
        ));
    }
    let kind = ((t >> 28) & 7) | ((t >> 8) & 24);
    let n = t & 2047;
    Ok(match (kind, n) {
        (0, 0..32) => Register::Temporary(n),
        (1, _) if n < if stage == Stage::Vertex { 16 } else { 10 } => Register::Input(n),
        (2, _) if n < if stage == Stage::Vertex { 256 } else { 224 } => Register::Constant(n),
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

fn source(t: u32, at: usize, stage: Stage) -> Result<Source> {
    let register = register(t, at, stage)?;
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
        if count != expected {
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
            let allowed = matches!(
                d.register,
                Register::Temporary(_) | Register::Target(_) | Register::Output(_)
            ) || (opcode == Opcode::TexKill && matches!(d.register, Register::Input(_)));
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
        let sources = source_tokens
            .iter()
            .enumerate()
            .map(|(i, &t)| source(t, operand(i.saturating_add(2)), stage))
            .collect::<Result<_>>()?;
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
