//! What a program's own code shows of how it uses what it is bound: how a constant buffer's lines hold a matrix, what a
//! discard compares a constant with, and what an output's lanes are tested or blended by. These are observations the
//! analyses of `iridia` make (`iridia::matrix`, `iridia::output_uses`), held as plain data so that a consumer without
//! naga can keep them; what an observation makes a value stand for is the consumer's to decide.

use serde::Serialize;

use crate::{Label, ResourceId};

/// What a discard compares against a constant word (an alpha reference), proved from the program's data flow
/// (`iridia::output_uses::discard_compares`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AlphaComparison {
    /// The output alpha itself (including unchanged branch arms).
    Output,
    /// A scalar that reaches output alpha through arithmetic. Cutting the output is an approximation.
    Transformed,
    /// A pre-transform alpha plus `(dither.a - 0.5) * scale`. `scale_word` is a word offset in the reference's own
    /// buffer. A hard cutoff on output alpha approximates the program's stochastic, pre-transform test.
    Dithered {
        /// Word offset of the dither scale in the reference's own buffer.
        scale_word: u32,
    },
    /// The discard compares a value made of texture samples and constants that reaches no target's alpha (a G-buffer
    /// program writes none: `t1.w * t0.w + scale * (noise - 0.5) < reference`). The test is the program's own
    /// expression, read exactly from its graph (`iridia::output_uses::alpha_tests`); a cutoff on any output alpha is not
    /// it, and a consumer that cuts evaluates that expression, or does not cut.
    Unwritten,
}
/// How a matrix constant's 16-byte lines hold the matrix the program applies to column vectors (a view-projection's
/// clip position is `M * p`). The layout's packing (`row_major`, `column_major`) does not decide it, since a program may
/// multiply on either side (`mul(M, v)`, `mul(v, M)`): `iridia::matrix::lines_at` reads it from what the program does.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MatrixLines {
    /// Each line is one of the matrix's rows.
    Rows,
    /// Each line is one of its columns.
    Columns,
    /// The program does not show which, and why ([`Unshown`]). A consumer that writes a matrix unequal to its
    /// transpose (a camera's) cannot bind it; one that writes the identity can.
    Unshown(Unshown),
}
/// Why a program does not show a matrix's lines ([`MatrixLines::Unshown`]): the kind a consumer branches on, its text
/// ([`std::fmt::Display`]) the reason.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Unshown {
    /// Its results are made of none of the matrix's words.
    NotRead,
    /// It reads them but sums no product of them with other values: it does not apply the matrix.
    NotApplied,
    /// It applies the matrix both ways: `along` sums of one line's words, `across` of one lane's.
    BothWays {
        /// How many sums of the words of one line there are.
        along: usize,
        /// How many sums of the words of one lane across lines there are.
        across: usize,
    },
    /// The program could not be examined for it, and `why`: its value graph could not be built, or the matrix's buffer
    /// could not be found in it.
    Unexamined {
        /// Why the program could not be examined.
        why: String,
    },
}
impl std::fmt::Display for Unshown {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Unshown::NotRead => f.write_str("the program's results are made of none of the matrix's words"),
            Unshown::NotApplied => f.write_str(
                "the program sums no product of the matrix's words with other values: it does not show how it \
                 applies the matrix",
            ),
            Unshown::BothWays { along, across } => write!(
                f,
                "the program applies the matrix both ways: {along} sums of one line's words and {across} of one lane's"
            ),
            Unshown::Unexamined { why } => f.write_str(why),
        }
    }
}
/// What the source states of how it binds a constant buffer of the program, beyond what its constants are.
///
/// A program declares a constant buffer at the size it indexes it, and a source can bind more: D3D12 bounds a constant
/// buffer read by the size of the view the draw binds (the CBV's `SizeInBytes`), not by the shader's declaration, so a
/// skinning palette a program declares as two matrices is bound as the hundreds the draw uploads, and a joint index past
/// the declaration reads the palette, not zero. A lifter takes the rows of a buffer's uniform array from the bound size
/// when it has one (`iridia::BoundSizes`).
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ConstantBufferBinding {
    /// A constant buffer resource of the shader.
    pub resource: ResourceId,
    /// The bytes the source binds the buffer with, more than the program declares and a whole number of 16-byte rows;
    /// `None` is the declared size. Where the engine binds different sizes for different draws, the largest it binds:
    /// a consumer that fills fewer rows leaves the rest zero.
    pub bound_size: Option<u32>,
}
/// What a pixel program's code and its target's stated blend show of the lanes of one output that is a render target
/// (`iridia::output_uses::output_uses`): observations, not roles. Each is a lane mask, bit k lane k.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct OutputUse {
    /// The output.
    pub label: Label,
    /// The render target it is written to: its system value `SV_Target<n>`, or its label's index where it has none.
    pub target: u32,
    /// Lane 3 where the target's stated blend weighs the colour by the source alpha (the colour equation's source or
    /// destination factor is `SrcAlpha` or `OneMinusSrcAlpha`), else none.
    pub blend_weight: u8,
    /// The lanes whose written value is among what a discard's conditions are made of. A literal is no evidence (equal
    /// literals are one node of the value graph); none where the program's value graph was not given.
    pub discard_tested: u8,
}
