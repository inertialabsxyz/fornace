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

/// The engine gives the same text as a reference loop that puts new token i
/// at position T + i, where T is the prompt token count. The reference uses
/// the model directly, with the greedy sampler and the repeat penalty of
/// the engine. It runs on the CPU in F32. The prompt is `alice-512.txt`:
/// with a short prompt, a position one too high did not change the text.
#[test]
fn generate_puts_each_new_token_at_its_position() {
    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarBuilder;
    use fornace_engine::model::{Config, ModelForCausalLM};
    use fornace_engine::{AttentionKind, KvCacheKind};
    use tokenizers::Tokenizer;

    const NEW_TOKENS: usize = 64;
    let Ok(model_dir) = std::env::var("FORNACE_MODEL_DIR") else {
        eprintln!("skipped: FORNACE_MODEL_DIR is not set");
        return;
    };
    let dir = PathBuf::from(model_dir);
    let prompt_text = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../bench/prompts/alice-512.txt"),
    )
    .unwrap();
    let prompt = prompt_text.trim_end_matches('\n');

    let mut engine = Engine::load(&LoadConfig {
        model_dir: dir.clone(),
        dtype: Dtype::F16,
        device: DeviceKind::Cpu,
        max_new_tokens: NEW_TOKENS,
        batch: 1,
        kv_cache: KvCacheKind::Prealloc,
        attention: AttentionKind::Naive,
        sync_spans: false,
        ignore_eos: true,
    })
    .expect("load the engine");
    let text = engine
        .generate(prompt)
        .expect("generate text")
        .texts
        .remove(0);

    let device = Device::Cpu;
    let config: Config =
        serde_json::from_slice(&std::fs::read(dir.join("config.json")).unwrap()).unwrap();
    let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json")).unwrap();
    // SAFETY: the test does not change the weights file while it is mapped.
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[dir.join("model.safetensors")], DType::F32, &device)
    }
    .unwrap();
    let mut model = ModelForCausalLM::new(&config, vb, false).unwrap();
    model.reset_kv_cache(KvCacheKind::Prealloc, 1024);
    let ids = tokenizer.encode(prompt, true).unwrap().get_ids().to_vec();
    let argmax = |logits: &Tensor| -> u32 {
        logits
            .flatten_all()
            .unwrap()
            .argmax(0)
            .unwrap()
            .to_scalar()
            .unwrap()
    };
    let input = Tensor::new(ids.as_slice(), &device)
        .unwrap()
        .unsqueeze(0)
        .unwrap();
    let mut tokens = vec![argmax(&model.forward(&input, 0).unwrap())];
    for i in 1..NEW_TOKENS {
        let input = Tensor::new(&[tokens[i - 1]], &device)
            .unwrap()
            .unsqueeze(0)
            .unwrap();
        let logits = model
            .forward(&input, ids.len() + i - 1)
            .unwrap()
            .flatten_all()
            .unwrap();
        let start = tokens.len().saturating_sub(64);
        let logits =
            candle_transformers::utils::apply_repeat_penalty(&logits, 1.1, &tokens[start..])
                .unwrap();
        tokens.push(argmax(&logits));
    }
    let expected = tokenizer.decode(&tokens, false).unwrap();
    assert_eq!(text, expected);
}
