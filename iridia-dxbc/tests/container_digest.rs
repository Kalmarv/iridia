#![allow(clippy::unwrap_used)]
//! The container digest over the games' own Shader Model 4 and 5 containers: for every container of a corpus,
//! `dxcontainer::checksum` equals the digest the compiler stored, the stored digest is not all zero, and the part
//! directory is read to the exact end. Needs the corpus oracle set's `sekiro/shaders`, `riders/shaders` or
//! `sledding/shaders` (each a folder, searched recursively, of containers as the game ships them).
mod inputs;

use inputs::Game;

mod common;

fn digests(game: Game) {
    let Some((name, paths)) = common::corpus(game) else {
        return;
    };
    for path in &paths {
        let bytes = std::fs::read(path).unwrap();
        let stored: [u8; 16] = bytes[4..20].try_into().unwrap();
        assert_ne!(stored, [0; 16], "{}: an unsigned container", path.display());
        assert_eq!(
            dxcontainer::checksum(&bytes),
            Some(stored),
            "{}: the digest differs from the stored one",
            path.display()
        );
        dxcontainer::chunks(&bytes).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    }
    println!(
        "{name}: {} containers, every digest equal to the stored one",
        paths.len()
    );
}

#[test]
fn sekiro_containers() {
    digests(Game::Sekiro);
}
#[test]
fn riders_republic_containers() {
    digests(Game::Riders);
}
#[test]
fn unity_containers() {
    digests(Game::Sledding);
}
