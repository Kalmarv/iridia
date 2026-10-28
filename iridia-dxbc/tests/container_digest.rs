#![allow(clippy::unwrap_used)]
//! The container digest over the games' own Shader Model 4 and 5 containers: for every container of a corpus,
//! `dxcontainer::checksum` equals the digest the compiler stored, the stored digest is not all zero, the part
//! directory is read to the exact end and `dxcontainer::serialize` writes the container again byte for byte. Needs the corpus oracle set's `sekiro/shaders`, `riders/shaders` or
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
        let parts = dxcontainer::chunks(&bytes).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        // the writer gives the container back byte for byte: the parts in their order, unpadded as stored, the digest
        // it computes equal to the stored one; and a zeroed digest is repaired to the stored one
        assert_eq!(
            dxcontainer::serialize(&parts).unwrap(),
            bytes,
            "{}: re-serialized differently",
            path.display()
        );
        let mut unsigned = bytes.clone();
        unsigned[4..20].fill(0);
        dxcontainer::repair_digest(&mut unsigned).unwrap();
        assert_eq!(unsigned, bytes, "{}: repaired differently", path.display());
    }
    println!(
        "{name}: {} containers, every digest equal to the stored one, every container re-serialized byte for byte",
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
