//! Translate a compiled Direct3D shader to WGSL.
//!
//! ```text
//! cargo run -p iridia --example to_wgsl -- path/to/shader.dxbc
//! ```
//!
//! The file is a DXBC container: Shader Model 4 or 5 (`DXBC` bytecode) or Shader Model 6 (a `DXIL` part). The WGSL goes
//! to standard output, with the program's notes (what the lifter approximated) on standard error.

use std::process::ExitCode;

fn main() -> ExitCode {
    let Some(path) = std::env::args_os().nth(1) else {
        eprintln!("usage: to_wgsl <shader.dxbc|shader.dxil>");
        return ExitCode::from(2);
    };
    let bytes = match std::fs::read(&path) {
        Ok(bytes) => bytes,
        Err(e) => {
            eprintln!("{}: {e}", path.to_string_lossy());
            return ExitCode::from(2);
        }
    };
    // a Shader Model 6 container carries its program in a part called DXIL
    let program = if bytes.windows(4).any(|w| w == b"DXIL") {
        iridia_dxil::lift(&bytes)
    } else {
        iridia_dxbc::lift(&bytes)
    };
    let program = match program {
        Ok(program) => program,
        Err(refusal) => {
            eprintln!("refused: {refusal}");
            return ExitCode::FAILURE;
        }
    };
    for note in &program.notes {
        eprintln!("note: {note}");
    }
    match iridia::emit::wgsl(&program) {
        Ok(text) => {
            print!("{text}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("{e}");
            ExitCode::FAILURE
        }
    }
}
