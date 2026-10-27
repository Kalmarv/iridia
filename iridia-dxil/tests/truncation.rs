//! Every strict prefix of every DXIL fixture is refused: no length of a truncated container is read as a program, and
//! none makes the reader panic. (`fixtures.rs` does this for one fixture; the corpus covers every stage and shader model.)
use iridia_dxil::lift;

#[test]
fn every_truncation_of_every_fixture_is_refused() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let mut files = 0;
    let mut prefixes = 0;
    for entry in std::fs::read_dir(&dir).expect("the fixtures folder") {
        let path = entry.expect("a directory entry").path();
        if path.extension().is_none_or(|e| e != "dxil") {
            continue;
        }
        let bytes = std::fs::read(&path).expect("a fixture");
        // a fixture that is refused whole (a library the lifter has no model for) has no prefixes worth trying
        if lift(&bytes).is_err() {
            continue;
        }
        for n in 0..bytes.len() {
            let prefix = bytes.get(..n).expect("a prefix");
            assert!(
                lift(prefix).is_err(),
                "{}: prefix {n} of {}",
                path.display(),
                bytes.len()
            );
            prefixes += 1;
        }
        files += 1;
    }
    assert!(files > 50, "{files} fixtures found");
    assert!(prefixes > 100_000, "{prefixes} prefixes tried");
}
