use std::path::PathBuf;

use fornace_engine::{DeviceKind, Dtype, Engine, LoadConfig};

/// Greedy decoding of a known prompt gives the known answer.
///
/// Set `FORNACE_MODEL_DIR` to the absolute path of a Qwen2.5-0.5B directory
/// to run this test.
#[test]
fn greedy_decode_names_the_capital_of_spain() {
    let Ok(model_dir) = std::env::var("FORNACE_MODEL_DIR") else {
        eprintln!("skipped: FORNACE_MODEL_DIR is not set");
        return;
    };
    let config = LoadConfig {
        model_dir: PathBuf::from(model_dir),
        dtype: Dtype::F16,
        device: DeviceKind::Cpu,
        max_new_tokens: 8,
        batch: 1,
        kv_cache: Default::default(),
        attention: Default::default(),
        sync_spans: false,
        ignore_eos: false,
    };
    let mut engine = Engine::load(&config).expect("load the engine");
    let report = engine
        .generate("The capital of Spain is")
        .expect("generate text");
    assert!(
        report.texts[0].trim_start().starts_with("Madrid"),
        "unexpected text: {:?}",
        report.texts[0]
    );
}
