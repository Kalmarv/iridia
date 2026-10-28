#![allow(clippy::unwrap_used)]
//! The container digest over the games' own Shader Model 6 containers: for every container of a corpus,
//! `dxcontainer::checksum` equals the digest the compiler stored, the stored digest is not all zero (so no stock
//! container is unsigned), and the part directory is read to the exact end (parts of Elden Ring's are not padded to
//! four bytes). Needs the corpus oracle set's `skate/shaders` or `elden/shaders` (each a folder, searched
//! recursively, of containers as the game ships them).
mod inputs;

use std::path::{Path, PathBuf};

use inputs::{Game, OracleSet, Query, expect_found, resolve};

fn containers(root: &Path, out: &mut Vec<PathBuf>) {
    let mut entries: Vec<_> = std::fs::read_dir(root)
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect();
    entries.sort();
    for path in entries {
        if path.is_dir() {
            containers(&path, out);
        } else if std::fs::read(&path).is_ok_and(|b| b.starts_with(b"DXBC")) {
            out.push(path);
        }
    }
}

fn digests(game: Game) {
    let query = Query::Oracle(OracleSet::Corpus, game, "shaders");
    let Some(root) = resolve(query) else {
        return;
    };
    let mut paths = Vec::new();
    containers(&root, &mut paths);
    expect_found(query, "DXBC containers", paths.len());
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
        "{}: {} containers, every digest equal to the stored one",
        query.key(),
        paths.len()
    );
}

#[test]
fn skate_containers() {
    digests(Game::Skate);
}
#[test]
fn elden_ring_containers() {
    digests(Game::Elden);
}
