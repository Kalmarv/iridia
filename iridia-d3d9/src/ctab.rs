//! D3DX's CTAB comment: offsets are relative to the header following the FourCC.
use crate::{Result, error};

/// A CTAB type. Numeric class/type values are D3DXPARAMETER_CLASS/TYPE, kept alongside their dimensions.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Type {
    pub class: u16,
    pub kind: u16,
    pub rows: u16,
    pub columns: u16,
    pub elements: u16,
    pub members: Vec<TypeMember>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TypeMember {
    pub name: String,
    pub ty: Type,
}

/// A named allocation, including the original type even when optimization shortened its register range.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Constant {
    pub name: String,
    /// D3DXREGISTER_SET: BOOL=0, INT4=1, FLOAT4=2, SAMPLER=3.
    pub register_set: u16,
    pub register: u16,
    pub count: u16,
    /// Compiler-reserved word of D3DXSHADER_CONSTANTINFO. Compilers write nonzero values;
    /// it is retained verbatim and does not change the explicit register allocation.
    pub reserved: u16,
    pub ty: Type,
    /// Raw register defaults (BOOL rows are 4 bytes, other numeric rows 16); no defaults on samplers.
    pub default: Option<Vec<u8>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConstantTable {
    pub creator: String,
    pub target: String,
    pub version: u32,
    pub flags: u32,
    pub constants: Vec<Constant>,
}

struct Reader<'a> {
    data: &'a [u8],
    word: usize,
}
impl Reader<'_> {
    /// The word of the program that byte `at` of the table lies in, for an error.
    fn word_of(&self, at: usize) -> usize {
        self.word.saturating_add(at / 4)
    }
    /// `at + by`, an error naming `at` when it does not fit.
    fn plus(&self, at: usize, by: usize) -> Result<usize> {
        at.checked_add(by).ok_or_else(|| {
            error(
                self.word_of(at),
                format!("CTAB offset {at}+{by} is past the largest offset"),
            )
        })
    }
    /// `count * size` bytes, an error naming `at` when it does not fit.
    fn bytes(&self, at: usize, count: usize, size: usize) -> Result<usize> {
        count.checked_mul(size).ok_or_else(|| {
            error(
                self.word_of(at),
                format!("CTAB {count} entries of {size} bytes do not fit"),
            )
        })
    }
    fn range(&self, at: usize, len: usize) -> Result<&[u8]> {
        at.checked_add(len)
            .and_then(|end| self.data.get(at..end))
            .ok_or_else(|| {
                error(
                    self.word_of(at),
                    format!(
                        "CTAB range {at}+{len} outside the comment's {} bytes",
                        self.data.len()
                    ),
                )
            })
    }
    fn u16(&self, at: usize) -> Result<u16> {
        match self.range(at, 2)? {
            &[a, b] => Ok(u16::from_le_bytes([a, b])),
            _ => Err(error(self.word_of(at), "CTAB u16 outside comment")),
        }
    }
    fn u32(&self, at: usize) -> Result<u32> {
        match self.range(at, 4)? {
            &[a, b, c, d] => Ok(u32::from_le_bytes([a, b, c, d])),
            _ => Err(error(self.word_of(at), "CTAB u32 outside comment")),
        }
    }
    fn offset(&self, at: usize) -> Result<usize> {
        let offset = self.u32(at)?;
        usize::try_from(offset).map_err(|_| {
            error(
                self.word_of(at),
                format!("CTAB offset {offset} does not fit an address"),
            )
        })
    }
    fn string(&self, at: usize) -> Result<String> {
        let bytes = self.range(at, self.data.len().saturating_sub(at))?;
        let end = bytes
            .iter()
            .position(|&b| b == 0)
            .ok_or_else(|| error(self.word_of(at), "CTAB string has no NUL"))?;
        let text = bytes.get(..end).unwrap_or_default();
        std::str::from_utf8(text)
            .map(str::to_owned)
            .map_err(|_| error(self.word_of(at), "CTAB string is not UTF-8"))
    }
    fn ty(&self, at: usize, depth: usize, budget: &mut usize) -> Result<Type> {
        if depth > 32 || *budget == 0 {
            return Err(error(
                self.word_of(at),
                "CTAB type nesting/cycle or 4096-node limit",
            ));
        }
        *budget = budget.saturating_sub(1);
        self.range(at, 16)?;
        let count = usize::from(self.u16(self.plus(at, 10)?)?);
        let members_at = self.offset(self.plus(at, 12)?)?;
        self.range(members_at, self.bytes(members_at, count, 8)?)?;
        let mut members = Vec::new();
        for n in 0..count {
            let member = self.plus(members_at, self.bytes(members_at, n, 8)?)?;
            members.push(TypeMember {
                name: self.string(self.offset(member)?)?,
                ty: self.ty(
                    self.offset(self.plus(member, 4)?)?,
                    depth.saturating_add(1),
                    budget,
                )?,
            });
        }
        Ok(Type {
            class: self.u16(at)?,
            kind: self.u16(self.plus(at, 2)?)?,
            rows: self.u16(self.plus(at, 4)?)?,
            columns: self.u16(self.plus(at, 6)?)?,
            elements: self.u16(self.plus(at, 8)?)?,
            members,
        })
    }
}

pub(crate) fn read(data: &[u8], word: usize, expected_version: u32) -> Result<ConstantTable> {
    let r = Reader { data, word };
    if r.u32(0)? != 28 {
        return Err(error(word, "CTAB header size is not 28"));
    }
    let version = r.u32(8)?;
    if version != expected_version {
        return Err(error(
            r.word_of(8),
            format!(
                "CTAB version disagrees with the program stage: {version:#010x}, where the stage's is {expected_version:#010x}"
            ),
        ));
    }
    let count = r.offset(12)?;
    if count > 1024 {
        return Err(error(
            r.word_of(12),
            format!("CTAB has {count} constants, where it has at most 1024"),
        ));
    }
    let base = r.offset(16)?;
    r.range(base, r.bytes(base, count, 20)?)?;
    let mut constants: Vec<Constant> = Vec::new();
    let mut budget = 4096;
    for n in 0..count {
        let at = r.plus(base, r.bytes(base, n, 20)?)?;
        let name = r.string(r.offset(at)?)?;
        let register_set = r.u16(r.plus(at, 4)?)?;
        let register = r.u16(r.plus(at, 6)?)?;
        let count = r.u16(r.plus(at, 8)?)?;
        let limit = match register_set {
            0 | 1 => 16,
            3 => {
                if version == 0xfffe_0300 {
                    4
                } else {
                    16
                }
            }
            2 => {
                if version == 0xfffe_0300 {
                    256
                } else {
                    224
                }
            }
            _ => {
                return Err(error(
                    r.word_of(at),
                    format!(
                        "CTAB register set {register_set}, where it is 0 (bool), 1 (int4), 2 (float4) or 3 (sampler)"
                    ),
                ));
            }
        };
        if count == 0 || u32::from(register).saturating_add(u32::from(count)) > limit {
            return Err(error(
                r.word_of(at),
                format!(
                    "CTAB register range c{register}+{count} is empty or out-of-bank (the bank has {limit} registers)"
                ),
            ));
        }
        if constants.iter().any(|c| {
            c.name == name
                || (c.register_set == register_set
                    && u32::from(c.register) < u32::from(register).saturating_add(u32::from(count))
                    && register < c.register.saturating_add(c.count))
        }) {
            return Err(error(
                r.word_of(at),
                format!("CTAB constant {name} overlaps an allocation or repeats a name"),
            ));
        }
        let default_at = r.offset(r.plus(at, 16)?)?;
        let default = if default_at == 0 {
            None
        } else {
            if register_set == 3 {
                return Err(error(r.word_of(at), "CTAB sampler default"));
            }
            let row = if register_set == 0 { 4 } else { 16 };
            Some(
                r.range(default_at, r.bytes(default_at, usize::from(count), row)?)?
                    .to_vec(),
            )
        };
        constants.push(Constant {
            name,
            register_set,
            register,
            count,
            reserved: r.u16(r.plus(at, 10)?)?,
            ty: r.ty(r.offset(r.plus(at, 12)?)?, 0, &mut budget)?,
            default,
        });
    }
    Ok(ConstantTable {
        creator: r.string(r.offset(4)?)?,
        version,
        flags: r.u32(20)?,
        target: r.string(r.offset(24)?)?,
        constants,
    })
}
