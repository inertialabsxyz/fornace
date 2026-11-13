//! Quality numbers for a model: perplexity from log-probabilities.
//!
//! [`crate::Engine::perplexity`] and [`crate::Engine::perplexity_chunked`]
//! call these functions on chunks of positions. They are pure, so the tests need no model.

use candle_core::{D, Result, Tensor};

/// The sum of −log p(target) over the rows of `log_probs`. `log_probs` is
/// `[rows, vocab]` (F32), and `targets` is `[rows]` (u32): the next token of
/// each row.
pub fn nll_sum(log_probs: &Tensor, targets: &Tensor) -> Result<f64> {
    let picked = log_probs.gather(&targets.unsqueeze(D::Minus1)?.contiguous()?, D::Minus1)?;
    let sum: f32 = picked.sum_all()?.to_scalar()?;
    Ok(-f64::from(sum))
}

/// exp of the mean negative log-probability over `count` predicted tokens.
pub fn perplexity(nll_sum: f64, count: usize) -> f64 {
    (nll_sum / count as f64).exp()
}

/// The chunks of `llama-perplexity` for `n_tokens` tokens and a context of
/// `ctx`: `n_tokens / ctx` chunks of `ctx` tokens, from the start. A rest
/// shorter than `ctx` is not used. Below 3 tokens, a chunk scores no
/// token, so there are no chunks.
pub fn llama_chunks(n_tokens: usize, ctx: usize) -> Vec<std::ops::Range<usize>> {
    if ctx < 3 {
        return Vec::new();
    }
    (0..n_tokens / ctx)
        .map(|i| i * ctx..(i + 1) * ctx)
        .collect()
}

/// The positions of a chunk of `ctx` tokens whose logits `llama-perplexity`
/// scores: from `ctx / 2` to `ctx − 2`. The logits at position p predict
/// token p + 1, so each chunk scores `ctx − ctx / 2 − 1` tokens, and each
/// has at least `ctx / 2` tokens before it.
pub fn llama_scored(ctx: usize) -> std::ops::Range<usize> {
    ctx / 2..ctx.saturating_sub(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn llama_chunks_drop_the_rest_and_score_the_second_half() {
        assert_eq!(llama_chunks(2049, 1024), [0..1024, 1024..2048]);
        assert!(llama_chunks(1023, 1024).is_empty());
        assert!(llama_chunks(10, 2).is_empty());
        assert_eq!(llama_chunks(10, 3).len(), 3);
        // 1024 tokens: the logits at 512 to 1022 predict tokens 513 to 1023.
        assert_eq!(llama_scored(1024), 512..1023);
        assert_eq!(llama_scored(1024).len(), 511);
        assert_eq!(llama_scored(5), 2..4);
    }

    fn log_softmax(logits: Vec<f32>, rows: usize, vocab: usize) -> Tensor {
        let logits = Tensor::from_vec(logits, (rows, vocab), &Device::Cpu).unwrap();
        candle_nn::ops::log_softmax(&logits, D::Minus1).unwrap()
    }

    #[test]
    fn uniform_logits_give_a_perplexity_equal_to_the_vocabulary() {
        let (rows, vocab) = (3, 8);
        let log_probs = log_softmax(vec![0.0; rows * vocab], rows, vocab);
        let targets = Tensor::new(&[1u32, 5, 7], &Device::Cpu).unwrap();
        let nll = nll_sum(&log_probs, &targets).unwrap();
        let ppl = perplexity(nll, rows);
        assert!((ppl - vocab as f64).abs() < 1e-4, "{ppl}");
    }

    #[test]
    fn a_confident_correct_model_has_a_perplexity_near_one() {
        // Row 0 predicts token 2 and row 1 predicts token 0, each with a
        // logit 20 above the others.
        let mut logits = vec![0.0; 2 * 4];
        logits[2] = 20.0;
        logits[4] = 20.0;
        let log_probs = log_softmax(logits, 2, 4);
        let right = Tensor::new(&[2u32, 0], &Device::Cpu).unwrap();
        let wrong = Tensor::new(&[1u32, 3], &Device::Cpu).unwrap();
        let ppl_right = perplexity(nll_sum(&log_probs, &right).unwrap(), 2);
        let ppl_wrong = perplexity(nll_sum(&log_probs, &wrong).unwrap(), 2);
        assert!(ppl_right < 1.001, "{ppl_right}");
        assert!(ppl_wrong > 1e8, "{ppl_wrong}");
    }

    #[test]
    fn chunks_add_up_to_the_whole() {
        let (rows, vocab) = (4, 5);
        let logits: Vec<f32> = (0..rows * vocab).map(|i| (i as f32 * 0.7).sin()).collect();
        let log_probs = log_softmax(logits, rows, vocab);
        let targets = Tensor::new(&[0u32, 4, 2, 3], &Device::Cpu).unwrap();
        let whole = nll_sum(&log_probs, &targets).unwrap();
        let parts = nll_sum(
            &log_probs.narrow(0, 0, 3).unwrap(),
            &targets.narrow(0, 0, 3).unwrap(),
        )
        .unwrap()
            + nll_sum(
                &log_probs.narrow(0, 3, 1).unwrap(),
                &targets.narrow(0, 3, 1).unwrap(),
            )
            .unwrap();
        assert!((whole - parts).abs() < 1e-5);
    }
}
