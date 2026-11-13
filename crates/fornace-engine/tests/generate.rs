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
        block_size: fornace_engine::DEFAULT_BLOCK_SIZE,
        kv_capacity: None,
        share_prefix: false,
        attention: Default::default(),
        sync_spans: false,
        ignore_eos: false,
        repeat_penalty: fornace_engine::DEFAULT_REPEAT_PENALTY,
        draft: None,
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
        block_size: fornace_engine::DEFAULT_BLOCK_SIZE,
        kv_capacity: None,
        share_prefix: false,
        attention: AttentionKind::Naive,
        sync_spans: false,
        ignore_eos: true,
        repeat_penalty: fornace_engine::DEFAULT_REPEAT_PENALTY,
        draft: None,
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

/// An engine on the CPU for the stream tests, with `max_new_tokens` and
/// EOS ignored. The tests give their own options to `generate_stream`.
fn stream_engine(max_new_tokens: usize) -> Option<Engine> {
    let Ok(model_dir) = std::env::var("FORNACE_MODEL_DIR") else {
        eprintln!("skipped: FORNACE_MODEL_DIR is not set");
        return None;
    };
    Some(
        Engine::load(&LoadConfig {
            model_dir: PathBuf::from(model_dir),
            dtype: Dtype::F16,
            device: DeviceKind::Cpu,
            max_new_tokens,
            batch: 1,
            kv_cache: fornace_engine::KvCacheKind::Prealloc,
            block_size: fornace_engine::DEFAULT_BLOCK_SIZE,
            kv_capacity: None,
            share_prefix: false,
            attention: Default::default(),
            sync_spans: false,
            ignore_eos: true,
            repeat_penalty: fornace_engine::DEFAULT_REPEAT_PENALTY,
            draft: None,
        })
        .expect("load the engine"),
    )
}

const STREAM_PROMPT: &str = "Write a short list of words with accents and emoji: café, 🦀,";

/// The events of a stream give the tokens and the text of the report, and
/// the same tokens as `generate` with the same settings.
#[test]
fn stream_events_equal_the_report_and_generate() {
    use std::ops::ControlFlow;

    let Some(mut engine) = stream_engine(32) else {
        return;
    };
    let options = fornace_engine::GenerateOptions {
        max_new_tokens: 32,
        ignore_eos: true,
        repeat_penalty: fornace_engine::DEFAULT_REPEAT_PENALTY,
    };
    let mut ids = Vec::new();
    let mut text = String::new();
    let report = engine
        .generate_stream(STREAM_PROMPT, &options, &mut |event| {
            ids.push(event.id);
            text.push_str(&event.text);
            ControlFlow::Continue(())
        })
        .expect("stream");
    assert_eq!(ids, report.tokens[0]);
    // The last token of this prompt ends inside a character. The report
    // holds that part as U+FFFD, and no event gives it.
    let rest = report.texts[0]
        .strip_prefix(&text)
        .expect("the report text starts with the text of the events");
    assert!(
        rest.chars().all(|c| c == char::REPLACEMENT_CHARACTER),
        "the events miss more than an incomplete last character: {rest:?}"
    );
    assert_eq!(report.generated_tokens, 32);

    let plain = engine.generate(STREAM_PROMPT).expect("generate");
    assert_eq!(plain.tokens[0], report.tokens[0]);
}

/// `ControlFlow::Break` after the fifth event stops the generation with
/// five tokens.
#[test]
fn stream_break_after_five_tokens_gives_five_tokens() {
    use std::ops::ControlFlow;

    let Some(mut engine) = stream_engine(32) else {
        return;
    };
    let options = fornace_engine::GenerateOptions {
        max_new_tokens: 32,
        ignore_eos: true,
        repeat_penalty: 1.0,
    };
    let mut events = 0;
    let report = engine
        .generate_stream(STREAM_PROMPT, &options, &mut |_| {
            events += 1;
            if events == 5 {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        })
        .expect("stream");
    assert_eq!(events, 5);
    assert_eq!(report.generated_tokens, 5);
}

/// The options of the call replace the `max_new_tokens` of the load config.
#[test]
fn stream_options_set_the_token_count() {
    use std::ops::ControlFlow;

    let Some(mut engine) = stream_engine(32) else {
        return;
    };
    let options = fornace_engine::GenerateOptions {
        max_new_tokens: 7,
        ignore_eos: true,
        repeat_penalty: 1.0,
    };
    let report = engine
        .generate_stream(STREAM_PROMPT, &options, &mut |_| ControlFlow::Continue(()))
        .expect("stream");
    assert_eq!(report.generated_tokens, 7);
}

/// A character that the tokenizer splits over two or more tokens comes out
/// whole, in the event of its last token. The joined text equals the
/// decoded text.
#[test]
fn text_stream_gives_a_split_character_whole() {
    use fornace_engine::stream::TextStream;
    use tokenizers::Tokenizer;

    let Ok(model_dir) = std::env::var("FORNACE_MODEL_DIR") else {
        eprintln!("skipped: FORNACE_MODEL_DIR is not set");
        return;
    };
    let tokenizer = Tokenizer::from_file(PathBuf::from(model_dir).join("tokenizer.json")).unwrap();
    let text = "🦀 Rust, 中文, ünïcødé, Ελληνικά";
    let ids = tokenizer.encode(text, false).unwrap().get_ids().to_vec();
    let mut stream = TextStream::default();
    let parts: Vec<String> = (1..=ids.len())
        .map(|n| stream.push(&tokenizer, &ids[..n]).unwrap())
        .collect();
    let split = (1..ids.len()).any(|n| {
        tokenizer
            .decode(&ids[..n], false)
            .unwrap()
            .ends_with(char::REPLACEMENT_CHARACTER)
    });
    assert!(
        split,
        "no token of {text:?} ends inside a character, so the test checks nothing"
    );
    assert!(
        parts
            .iter()
            .all(|p| !p.contains(char::REPLACEMENT_CHARACTER)),
        "an event has an incomplete character: {parts:?}"
    );
    assert_eq!(parts.concat(), text);
}

/// The chunked perplexity uses whole chunks only, and scores fewer tokens
/// with more context than the plain perplexity, so the two differ.
#[test]
fn chunked_perplexity_needs_one_whole_chunk() {
    let Some(mut engine) = stream_engine(1) else {
        return;
    };
    let text = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../bench/prompts/alice-512.txt"),
    )
    .unwrap();
    assert!(engine.perplexity_chunked(&text, 1024).is_err());
    let chunked = engine.perplexity_chunked(&text, 256).expect("two chunks");
    let plain = engine.perplexity(&text).expect("plain");
    assert!(chunked.is_finite() && chunked > 1.0, "{chunked}");
    assert!((chunked - plain).abs() > 1e-3, "{chunked} {plain}");
}
