//! Translate a compiled Direct3D shader to SPIR-V, through naga's SPIR-V writer.
//!
//! ```text
//! cargo run -p iridia --example to_spirv -- path/to/shader.dxbc out.spv
//! ```
//!
//! A [`iridia::Program`] holds a `naga::Module`, so any writer naga has can take it; this one enables naga's `spv-out`
//! feature. The writer is naga's: it refuses what SPIR-V output cannot express, and the module is written as the
//! program was lifted, in Direct3D's clip-space and texture conventions
//! ([`iridia::Program::conventions`]), which a pipeline that wants Vulkan's applies where it meets the program.

use std::process::ExitCode;

fn main() -> ExitCode {
    let mut args = std::env::args_os().skip(1);
    let (Some(path), Some(out)) = (args.next(), args.next()) else {
        eprintln!("usage: to_spirv <shader.dxbc|shader.dxil> <out.spv>");
        return ExitCode::from(2);
    };
    let bytes = match std::fs::read(&path) {
        Ok(bytes) => bytes,
        Err(e) => {
            eprintln!("{}: {e}", path.to_string_lossy());
            return ExitCode::from(2);
        }
    };
    let lifted = if bytes.windows(4).any(|w| w == b"DXIL") {
        iridia_dxil::lift(&bytes)
    } else {
        iridia_dxbc::lift(&bytes)
    };
    let program = match lifted {
        Ok(program) => program,
        Err(refusal) => {
            eprintln!("refused: {refusal}");
            return ExitCode::FAILURE;
        }
    };
    let info = match program.validate() {
        Ok(info) => info,
        Err(problems) => {
            for problem in problems {
                eprintln!("invalid: {problem}");
            }
            return ExitCode::FAILURE;
        }
    };
    let words = match naga::back::spv::write_vec(
        &program.module,
        &info,
        &naga::back::spv::Options::default(),
        None,
    ) {
        Ok(words) => words,
        Err(e) => {
            eprintln!("spirv: {e}");
            return ExitCode::FAILURE;
        }
    };
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    if let Err(e) = std::fs::write(&out, bytes) {
        eprintln!("{}: {e}", out.to_string_lossy());
        return ExitCode::from(2);
    }
    ExitCode::SUCCESS
}
