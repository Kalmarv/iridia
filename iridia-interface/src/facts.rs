//! How large a program is, in the unit its reader counts.
use serde::Serialize;

/// The source's counting unit. DXBC counts decoded instructions including control flow, excluding declarations;
/// DXIL counts LLVM instructions
/// in the functions the lifter reads. Neither is an executed instruction count or an IR expression count.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum InstructionUnit {
    /// Instructions of a DXBC shader (Shader Model 4 or 5) as decoded.
    DxbcInstructions,
    /// Instructions of a DXIL shader (Shader Model 6) in the functions the lifter reads.
    DxilInstructions,
}
/// How many instructions a program has, in the unit that says how they were counted.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct InstructionCount {
    /// The number of instructions counted.
    pub count: u64,
    /// What `count` counts; counts of different units are not comparable.
    pub unit: InstructionUnit,
}
