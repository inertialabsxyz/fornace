//! A hook that changes nothing keeps the tokens of `generate_stream`, and a
//! hook that changes the logits changes the tokens.
//!
//! Set `FORNACE_MODEL_DIR` to the absolute path of a Qwen2.5-0.5B directory
//! to run these tests. With the `metal` feature, they run on Metal.

use std::ops::ControlFlow;
use std::path::PathBuf;

use fornace_engine::{
    DEFAULT_SEED, DeviceKind, Dtype, Engine, GenerateOptions, LoadConfig, LogitsHook,
};

const DEVICE: DeviceKind = if cfg!(feature = "metal") {
    DeviceKind::Metal
} else {
    DeviceKind::Cpu
};

const PROMPTS: [&str; 3] = [
    "<|im_start|>user\nWrite one sentence about the sea.<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>user\nName three colours.<|im_end|>\n<|im_start|>assistant\n",
    "The capital of Spain is",
];

const NEW_TOKENS: usize = 16;

fn engine() -> Option<(Engine, LoadConfig)> {
    let Ok(model_dir) = std::env::var("FORNACE_MODEL_DIR") else {
        eprintln!("skipped: FORNACE_MODEL_DIR is not set");
        return None;
    };
    let config = LoadConfig {
        model_dir: PathBuf::from(model_dir),
        dtype: Dtype::F16,
        device: DEVICE,
        max_new_tokens: NEW_TOKENS,
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
    let engine = Engine::load(&config).expect("load the engine");
    Some((engine, config))
}

fn options(config: &LoadConfig, temperature: Option<f64>, seed: u64) -> GenerateOptions {
    GenerateOptions {
        temperature,
        seed,
        ..GenerateOptions::from(config)
    }
}

/// The token IDs of the events of one call.
fn stream(
    engine: &mut Engine,
    prompt: &str,
    options: &GenerateOptions,
    hook: Option<&mut dyn LogitsHook>,
) -> Vec<u32> {
    let mut ids = Vec::new();
    let mut on_token = |event: fornace_engine::TokenEvent| {
        ids.push(event.id);
        ControlFlow::Continue(())
    };
    let report = match hook {
        None => engine.generate_stream(prompt, options, &mut on_token),
        Some(hook) => engine.generate_stream_with_hook(prompt, options, hook, &mut on_token),
    }
    .expect("generate");
    assert_eq!(report.tokens[0], ids);
    ids
}

/// One call of the hook.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Call {
    Logits(usize),
    Token(usize, u32),
}

/// Records each call. If `only` has a list, step `i` permits only the
/// token `only[i]` (the last one for the steps after the list).
#[derive(Default)]
struct Recorder {
    calls: Vec<Call>,
    only: Vec<u32>,
}

impl LogitsHook for Recorder {
    fn on_logits(&mut self, step: usize, logits: &mut [f32]) {
        self.calls.push(Call::Logits(step));
        if let Some(&only) = self.only.get(step).or(self.only.last()) {
            for (id, logit) in logits.iter_mut().enumerate() {
                if id != only as usize {
                    *logit = f32::NEG_INFINITY;
                }
            }
        }
    }

    fn on_token(&mut self, step: usize, token: u32) {
        self.calls.push(Call::Token(step, token));
    }
}

#[test]
fn a_hook_that_changes_nothing_keeps_the_tokens() {
    let Some((mut engine, config)) = engine() else {
        return;
    };
    for penalty in [1.0, fornace_engine::DEFAULT_REPEAT_PENALTY] {
        for temperature in [None, Some(0.7)] {
            for prompt in PROMPTS {
                for seed in 1..=3 {
                    let options = GenerateOptions {
                        repeat_penalty: penalty,
                        ..options(&config, temperature, seed)
                    };
                    let plain = stream(&mut engine, prompt, &options, None);
                    let mut hook = Recorder::default();
                    let hooked = stream(&mut engine, prompt, &options, Some(&mut hook));
                    assert_eq!(
                        plain, hooked,
                        "penalty {penalty}, temperature {temperature:?}, seed {seed}, {prompt:?}"
                    );
                }
            }
        }
    }
}

#[test]
fn the_same_seed_gives_the_same_tokens() {
    let Some((mut engine, config)) = engine() else {
        return;
    };
    let mut differ = 0;
    for prompt in PROMPTS {
        let one = stream(&mut engine, prompt, &options(&config, Some(0.7), 1), None);
        let again = stream(&mut engine, prompt, &options(&config, Some(0.7), 1), None);
        assert_eq!(one, again, "{prompt:?}");
        let two = stream(&mut engine, prompt, &options(&config, Some(0.7), 2), None);
        if one != two {
            differ += 1;
        }
    }
    assert!(
        differ > 0,
        "seeds 1 and 2 give the same tokens for all prompts"
    );
}

#[test]
fn a_hook_that_permits_one_token_gives_only_that_token() {
    let Some((mut engine, config)) = engine() else {
        return;
    };
    let only = engine.tokenize("2").expect("tokenize")[0];
    let mut hook = Recorder {
        only: vec![only],
        ..Default::default()
    };
    let options = options(&config, Some(0.7), DEFAULT_SEED);
    let ids = stream(&mut engine, PROMPTS[0], &options, Some(&mut hook));
    assert_eq!(ids, vec![only; NEW_TOKENS]);
    let tokens: Vec<u32> = hook
        .calls
        .iter()
        .filter_map(|call| match call {
            Call::Token(_, token) => Some(*token),
            Call::Logits(_) => None,
        })
        .collect();
    assert_eq!(tokens, ids);
}

#[test]
fn on_token_sees_the_first_token_and_eos() {
    let Some((mut engine, config)) = engine() else {
        return;
    };
    let word = engine.tokenize("Hi").expect("tokenize")[0];
    let eos = engine.tokenize("<|im_end|>").expect("tokenize")[0];
    let mut hook = Recorder {
        only: vec![word, word, eos],
        ..Default::default()
    };
    let options = options(&config, None, DEFAULT_SEED);
    let ids = stream(&mut engine, PROMPTS[0], &options, Some(&mut hook));
    // The generation stops at EOS, and EOS is not in the tokens.
    assert_eq!(ids, vec![word, word]);
    assert_eq!(
        hook.calls,
        vec![
            Call::Logits(0),
            Call::Token(0, word),
            Call::Logits(1),
            Call::Token(1, word),
            Call::Logits(2),
            Call::Token(2, eos),
        ]
    );
}

#[test]
fn on_logits_and_on_token_alternate_from_step_0() {
    let Some((mut engine, config)) = engine() else {
        return;
    };
    let mut hook = Recorder::default();
    let options = GenerateOptions {
        ignore_eos: true,
        ..options(&config, Some(0.7), DEFAULT_SEED)
    };
    let ids = stream(&mut engine, PROMPTS[1], &options, Some(&mut hook));
    assert_eq!(ids.len(), NEW_TOKENS);
    assert_eq!(hook.calls.len(), 2 * NEW_TOKENS);
    for (step, pair) in hook.calls.chunks(2).enumerate() {
        assert_eq!(pair, [Call::Logits(step), Call::Token(step, ids[step])]);
    }
}
