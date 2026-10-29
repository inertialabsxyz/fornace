use std::path::PathBuf;

use fornace_engine::{Config, DeviceKind, Dtype, Engine};

/// Greedy decoding of a known prompt gives the known answer.
///
/// Set `FORNACE_MODEL_DIR` to a Qwen2.5-0.5B directory to run this test.
/// While the engine is stubs, the test fails with `NotImplemented`.
#[test]
fn greedy_decode_names_the_capital_of_spain() {
    let Ok(model_dir) = std::env::var("FORNACE_MODEL_DIR") else {
        eprintln!("skipped: FORNACE_MODEL_DIR is not set");
        return;
    };
    let config = Config {
        model_dir: PathBuf::from(model_dir),
        dtype: Dtype::F16,
        device: DeviceKind::Cpu,
        max_new_tokens: 8,
        batch: 1,
    };
    let mut engine = Engine::load(&config).expect("load the engine");
    let report = engine
        .generate("The capital of Spain is")
        .expect("generate text");
    assert!(
        report.text.trim_start().starts_with("Madrid"),
        "unexpected text: {:?}",
        report.text
    );
}
