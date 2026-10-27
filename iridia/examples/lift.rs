use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let path = std::env::args().nth(1).ok_or("usage: lift <shader.dxbc>")?;
    let container = std::fs::read(path)?;

    // Shader Model 4 or 5; `iridia_dxil::lift` takes Shader Model 6 the same way
    let program = iridia_dxbc::lift(&container)?;
    println!(
        "{:?} program, target {}",
        program.stage, program.provenance.target
    );
    for input in program.interface.inputs.iter().filter(|input| input.used) {
        println!("reads {}{}", input.label.name, input.label.index);
    }

    // the same program as WGSL; `program.module` is a `naga::Module` for any other naga backend
    print!("{}", iridia::emit::wgsl(&program)?);
    Ok(())
}
