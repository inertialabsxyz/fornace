//! The Qwen2 model: the layers and the forward pass.
//!
//! This file is a copy of `src/models/qwen2.rs` from candle-transformers
//! 0.11.0 (https://github.com/huggingface/candle), licensed MIT OR
//! Apache-2.0. Fornace changes the model here. To see the changes, compare
//! this file with the original. Changes so far:
//!
//! - this header and the import paths;
//! - DEBUG trace spans in the forward pass, and a `sync_spans` argument to
//!   `ModelForCausalLM::new` and the constructors below it;
//! - a choice of KV cache (`KvCacheKind`), set with `reset_kv_cache`. A new
//!   model starts with `Cat`, which is the original code. `Engine` resets
//!   the cache to `LoadConfig::kv_cache` (default `Prealloc`) before each
//!   generation.
//! - a choice of attention (`AttentionKind`), set with `set_attention`. A
//!   new model starts with `Naive`, which is the original code.
//! - quantized weights from a GGUF file (`ModelForCausalLM::new_quantized`).
//!   Each projection, the embedding, and `lm_head` are then Q8_0 or Q4_0,
//!   and the model computes in F32, because Candle's quantized matmuls take
//!   F32 input. The norms use `candle_nn::RmsNorm`, which the
//!   `with_tracing` version wraps, so the dense model does not change.
//! - `ModelForCausalLM::hidden_states` and `ModelForCausalLM::lm_head`, to
//!   get the logits at every position (for perplexity).
//!
//! The original documentation follows.
//!
//! Qwen2 model implementation with quantization support.
//!
//! Qwen2 is a large language model from Alibaba optimized for efficiency.
//! This implementation provides quantization for reduced memory and compute.
//!
//! Key characteristics:
//! - Streaming decode support
//! - Grouped query attention (GQA)
//! - RMSNorm for layer normalization
//! - Rotary positional embeddings (RoPE)
//! - Support for 8-bit quantization
//!
//! References:
//! - 🤗 [Qwen2 Model](https://huggingface.co/Qwen/Qwen2-7B)
//!

use candle_core::quantized::QMatMul;
use candle_core::{D, DType, Device, IndexOp, Module, Result, Tensor};
use candle_nn::{Activation, RmsNorm, VarBuilder};
use candle_transformers::models::with_tracing::{Linear, linear, linear_no_bias};
use candle_transformers::quantized_nn;
use candle_transformers::quantized_var_builder::VarBuilder as QVarBuilder;
use std::sync::Arc;

use std::fmt;
use std::str::FromStr;

use crate::cache::{KvCacheKind, LayerCache};
use crate::trace::device_debug_span;

/// How each attention layer computes softmax(QK^T * scale)V.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AttentionKind {
    /// Repeat K and V for each query head, then use matmul and softmax.
    /// This builds the [heads, q_len, kv_len] score matrix in each layer.
    /// This is what Candle's Qwen2 does.
    #[default]
    Naive,
    /// Candle's fused Metal kernel (`candle_nn::ops::sdpa`). It uses the
    /// grouped KV heads directly and does not build the score matrix. Metal
    /// only.
    Fused,
}

impl FromStr for AttentionKind {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "naive" => Ok(AttentionKind::Naive),
            "fused" => Ok(AttentionKind::Fused),
            _ => Err(format!("unknown attention `{s}`: use naive or fused")),
        }
    }
}

impl fmt::Display for AttentionKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            AttentionKind::Naive => "naive",
            AttentionKind::Fused => "fused",
        })
    }
}

/// Where the weights come from: a dense safetensors file, or a GGUF file
/// with quantized weights.
#[derive(Clone)]
enum Weights<'a> {
    Dense(VarBuilder<'a>),
    Quantized(QVarBuilder),
}

impl Weights<'_> {
    fn pp<S: ToString>(&self, s: S) -> Self {
        match self {
            Weights::Dense(vb) => Weights::Dense(vb.pp(s)),
            Weights::Quantized(vb) => Weights::Quantized(vb.pp(s)),
        }
    }

    fn device(&self) -> &Device {
        match self {
            Weights::Dense(vb) => vb.device(),
            Weights::Quantized(vb) => vb.device(),
        }
    }

    /// The dtype of the activations. Quantized matmuls take F32 input.
    fn compute_dtype(&self) -> DType {
        match self {
            Weights::Dense(vb) => vb.dtype(),
            Weights::Quantized(_) => DType::F32,
        }
    }

    fn proj(&self, in_dim: usize, out_dim: usize, bias: bool) -> Result<Proj> {
        Ok(match (self, bias) {
            (Weights::Dense(vb), true) => Proj::Dense(linear(in_dim, out_dim, vb.clone())?),
            (Weights::Dense(vb), false) => {
                Proj::Dense(linear_no_bias(in_dim, out_dim, vb.clone())?)
            }
            (Weights::Quantized(vb), true) => {
                Proj::Quantized(quantized_nn::linear(in_dim, out_dim, vb.clone())?)
            }
            (Weights::Quantized(vb), false) => {
                Proj::Quantized(quantized_nn::linear_no_bias(in_dim, out_dim, vb.clone())?)
            }
        })
    }

    fn rms_norm(&self, size: usize, eps: f64) -> Result<RmsNorm> {
        match self {
            Weights::Dense(vb) => candle_nn::rms_norm(size, eps, vb.clone()),
            Weights::Quantized(vb) => {
                let weight = vb.get(size, "weight")?.dequantize(vb.device())?;
                Ok(RmsNorm::new(weight, eps))
            }
        }
    }
}

/// A projection with dense or quantized weights.
#[derive(Debug, Clone)]
enum Proj {
    Dense(Linear),
    Quantized(quantized_nn::Linear),
}

impl Module for Proj {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Proj::Dense(p) => p.forward(xs),
            Proj::Quantized(p) => p.forward(xs),
        }
    }
}

/// The token embedding. The quantized version reads only the rows of the
/// ids from the quantized table, and returns F32.
#[derive(Debug, Clone)]
enum Embedding {
    Dense(candle_nn::Embedding),
    Quantized(QMatMul),
}

impl Module for Embedding {
    fn forward(&self, ids: &Tensor) -> Result<Tensor> {
        match self {
            Embedding::Dense(e) => e.forward(ids),
            Embedding::Quantized(e) => e.embedding(ids),
        }
    }
}

/// Fused attention with `candle_nn::ops::sdpa`. `k` and `v` have the KV
/// heads, not repeated. With `causal`, query i sees the keys up to
/// i + kv_len - q_len, as the causal mask of the model does. The model's
/// sliding window is not applied: for Qwen2.5 it equals
/// `max_position_embeddings`, so it never masks a position.
fn fused_attention(q: &Tensor, k: &Tensor, v: &Tensor, scale: f64, causal: bool) -> Result<Tensor> {
    candle_nn::ops::sdpa(
        q,
        &dense_rows(k)?,
        &dense_rows(v)?,
        None,
        causal,
        scale as f32,
        1.0,
    )
}

/// Returns `x` if each row of the last dimension is dense and the rows are
/// next to each other, else a contiguous copy. The one-token kernel of
/// `sdpa` reads K and V with only the head stride, and assumes this layout.
/// The `Cat` cache gives V as a transposed view, which does not have it.
fn dense_rows(x: &Tensor) -> Result<Tensor> {
    let stride = x.stride();
    if stride[3] == 1 && stride[2] == x.dim(3)? {
        Ok(x.clone())
    } else {
        x.contiguous()
    }
}

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    pub sliding_window: usize,
    pub max_window_layers: usize,
    pub tie_word_embeddings: bool,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    pub use_sliding_window: bool,
    pub hidden_act: Activation,
}

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(dtype: DType, cfg: &Config, dev: &Device) -> Result<Self> {
        let dim = cfg.hidden_size / cfg.num_attention_heads;
        let max_seq_len = cfg.max_position_embeddings;
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(dtype)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(dtype)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?,
            cos: freqs.cos()?,
        })
    }

    fn apply_rotary_emb_qkv(
        &self,
        q: &Tensor,
        k: &Tensor,
        seqlen_offset: usize,
    ) -> Result<(Tensor, Tensor)> {
        let (_b_sz, _h, seq_len, _n_embd) = q.dims4()?;
        let cos = self.cos.narrow(0, seqlen_offset, seq_len)?;
        let sin = self.sin.narrow(0, seqlen_offset, seq_len)?;
        let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

#[derive(Debug, Clone)]
#[allow(clippy::upper_case_acronyms)]
struct MLP {
    gate_proj: Proj,
    up_proj: Proj,
    down_proj: Proj,
    act_fn: Activation,
}

impl MLP {
    fn new(cfg: &Config, vb: Weights) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let intermediate_sz = cfg.intermediate_size;
        let gate_proj = vb.pp("gate_proj").proj(hidden_sz, intermediate_sz, false)?;
        let up_proj = vb.pp("up_proj").proj(hidden_sz, intermediate_sz, false)?;
        let down_proj = vb.pp("down_proj").proj(intermediate_sz, hidden_sz, false)?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act_fn: cfg.hidden_act,
        })
    }
}

impl Module for MLP {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let lhs = xs.apply(&self.gate_proj)?.apply(&self.act_fn)?;
        let rhs = xs.apply(&self.up_proj)?;
        (lhs * rhs)?.apply(&self.down_proj)
    }
}

#[derive(Debug, Clone)]
struct Attention {
    q_proj: Proj,
    k_proj: Proj,
    v_proj: Proj,
    o_proj: Proj,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    hidden_size: usize,
    rotary_emb: Arc<RotaryEmbedding>,
    kv_cache: LayerCache,
    attention: AttentionKind,
    sync_spans: bool,
}

impl Attention {
    fn new(
        rotary_emb: Arc<RotaryEmbedding>,
        cfg: &Config,
        vb: Weights,
        sync_spans: bool,
    ) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let num_kv_groups = num_heads / num_kv_heads;
        let head_dim = hidden_sz / num_heads;
        let q_proj = vb
            .pp("q_proj")
            .proj(hidden_sz, num_heads * head_dim, true)?;
        let k_proj = vb
            .pp("k_proj")
            .proj(hidden_sz, num_kv_heads * head_dim, true)?;
        let v_proj = vb
            .pp("v_proj")
            .proj(hidden_sz, num_kv_heads * head_dim, true)?;
        let o_proj = vb
            .pp("o_proj")
            .proj(num_heads * head_dim, hidden_sz, false)?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            num_kv_groups,
            head_dim,
            hidden_size: hidden_sz,
            rotary_emb,
            kv_cache: LayerCache::new(KvCacheKind::Cat, 0),
            attention: AttentionKind::Naive,
            sync_spans,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;
        let (device, sync) = (xs.device(), self.sync_spans);

        let (query_states, key_states, value_states) = {
            let _span = device_debug_span!(device, sync, "attn.qkv")?;
            let query_states = self.q_proj.forward(xs)?;
            let key_states = self.k_proj.forward(xs)?;
            let value_states = self.v_proj.forward(xs)?;

            let query_states = query_states
                .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
                .transpose(1, 2)?;
            let key_states = key_states
                .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
                .transpose(1, 2)?;
            let value_states = value_states
                .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
                .transpose(1, 2)?;
            (query_states, key_states, value_states)
        };

        let (query_states, key_states) = {
            let _span = device_debug_span!(device, sync, "attn.rope")?;
            self.rotary_emb
                .apply_rotary_emb_qkv(&query_states, &key_states, seqlen_offset)?
        };

        let (key_states, value_states) = {
            let _span = device_debug_span!(device, sync, "attn.kv_append")?;
            self.kv_cache.append(key_states, value_states)?
        };

        let scale = 1f64 / f64::sqrt(self.head_dim as f64);
        let attn_output = match self.attention {
            AttentionKind::Naive => {
                let (key_states, value_states) = {
                    let _span = device_debug_span!(device, sync, "attn.repeat_kv")?;
                    let key_states =
                        candle_transformers::utils::repeat_kv(key_states, self.num_kv_groups)?
                            .contiguous()?;
                    let value_states =
                        candle_transformers::utils::repeat_kv(value_states, self.num_kv_groups)?
                            .contiguous()?;
                    (key_states, value_states)
                };

                let _span = device_debug_span!(device, sync, "attn.sdpa")?;
                let attn_weights = (query_states.matmul(&key_states.transpose(2, 3)?)? * scale)?;

                let attn_weights = match attention_mask {
                    None => attn_weights,
                    Some(mask) => attn_weights.broadcast_add(mask)?,
                };
                let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
                attn_weights.matmul(&value_states)?
            }
            AttentionKind::Fused => {
                let _span = device_debug_span!(device, sync, "attn.sdpa")?;
                if attention_mask.is_some() {
                    candle_core::bail!("fused attention does not take an attention mask");
                }
                fused_attention(&query_states, &key_states, &value_states, scale, q_len > 1)?
            }
        };
        let _span = device_debug_span!(device, sync, "attn.o_proj")?;
        attn_output
            .transpose(1, 2)?
            .reshape((b_sz, q_len, self.hidden_size))?
            .apply(&self.o_proj)
    }

    fn reset_kv_cache(&mut self, kind: KvCacheKind, capacity: usize) {
        self.kv_cache = LayerCache::new(kind, capacity)
    }

    fn set_attention(&mut self, kind: AttentionKind) {
        self.attention = kind
    }
}

#[derive(Debug, Clone)]
struct DecoderLayer {
    self_attn: Attention,
    mlp: MLP,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    sync_spans: bool,
}

impl DecoderLayer {
    fn new(
        rotary_emb: Arc<RotaryEmbedding>,
        cfg: &Config,
        vb: Weights,
        sync_spans: bool,
    ) -> Result<Self> {
        let self_attn = Attention::new(rotary_emb, cfg, vb.pp("self_attn"), sync_spans)?;
        let mlp = MLP::new(cfg, vb.pp("mlp"))?;
        let input_layernorm = vb
            .pp("input_layernorm")
            .rms_norm(cfg.hidden_size, cfg.rms_norm_eps)?;
        let post_attention_layernorm = vb
            .pp("post_attention_layernorm")
            .rms_norm(cfg.hidden_size, cfg.rms_norm_eps)?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            sync_spans,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let (device, sync) = (xs.device(), self.sync_spans);
        let xs = {
            let _span = device_debug_span!(device, sync, "attn")?;
            let residual = xs;
            let xs = self.input_layernorm.forward(xs)?;
            let xs = self.self_attn.forward(&xs, attention_mask, seqlen_offset)?;
            (xs + residual)?
        };
        let _span = device_debug_span!(device, sync, "mlp")?;
        let residual = &xs;
        let xs = xs.apply(&self.post_attention_layernorm)?.apply(&self.mlp)?;
        residual + xs
    }

    fn reset_kv_cache(&mut self, kind: KvCacheKind, capacity: usize) {
        self.self_attn.reset_kv_cache(kind, capacity)
    }

    fn set_attention(&mut self, kind: AttentionKind) {
        self.self_attn.set_attention(kind)
    }
}

#[derive(Debug, Clone)]
pub struct Model {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    sliding_window: usize,
    device: Device,
    dtype: DType,
    attention: AttentionKind,
    sync_spans: bool,
}

impl Model {
    /// Builds the model. `embed_tokens` is the token embedding, built by
    /// the caller, because `lm_head` can share its weights.
    fn new(cfg: &Config, vb: Weights, embed_tokens: Embedding, sync_spans: bool) -> Result<Self> {
        let vb_m = vb.pp("model");
        let dtype = vb.compute_dtype();
        let rotary_emb = Arc::new(RotaryEmbedding::new(dtype, cfg, vb_m.device())?);
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for layer_idx in 0..cfg.num_hidden_layers {
            let layer = DecoderLayer::new(rotary_emb.clone(), cfg, vb_l.pp(layer_idx), sync_spans)?;
            layers.push(layer)
        }
        let norm = vb_m
            .pp("norm")
            .rms_norm(cfg.hidden_size, cfg.rms_norm_eps)?;
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            sliding_window: cfg.sliding_window,
            device: vb.device().clone(),
            dtype,
            attention: AttentionKind::Naive,
            sync_spans,
        })
    }

    fn prepare_causal_attention_mask(
        &self,
        b_size: usize,
        tgt_len: usize,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        // Sliding window mask?
        let mask: Vec<_> = (0..tgt_len)
            .flat_map(|i| {
                (0..tgt_len).map(move |j| {
                    if i < j || j + self.sliding_window < i {
                        f32::NEG_INFINITY
                    } else {
                        0.
                    }
                })
            })
            .collect();
        let mask = Tensor::from_slice(&mask, (tgt_len, tgt_len), &self.device)?;
        let mask = if seqlen_offset > 0 {
            let mask0 = Tensor::zeros((tgt_len, seqlen_offset), self.dtype, &self.device)?;
            Tensor::cat(&[&mask0, &mask], D::Minus1)?
        } else {
            mask
        };
        mask.expand((b_size, 1, tgt_len, tgt_len + seqlen_offset))?
            .to_dtype(self.dtype)
    }

    fn prepare_attention_mask(&self, attn_mask: &Tensor) -> Result<Tensor> {
        let (b_sz, sql_len) = attn_mask.dims2()?;
        let mut mask: Vec<Tensor> = vec![];
        for b in 0..b_sz {
            mask.push(attn_mask.i((b, ..))?.expand((1, 1, sql_len, sql_len))?);
        }
        let mask = Tensor::cat(&mask, 0)?;
        let on_true = mask.zeros_like()?.to_dtype(self.dtype)?;
        let on_false = Tensor::new(f32::NEG_INFINITY, &self.device)?
            .broadcast_as(mask.shape())?
            .to_dtype(self.dtype)?;
        mask.where_cond(&on_true, &on_false)
    }

    pub fn forward(
        &mut self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        attn_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b_size, seq_len) = input_ids.dims2()?;
        let (device, sync) = (&self.device, self.sync_spans);
        let attention_mask: Option<Tensor> = {
            let _span = device_debug_span!(device, sync, "mask")?;
            match attn_mask {
                Some(mask) => Some(self.prepare_attention_mask(mask)?),
                None => {
                    // The fused kernel applies the causal mask itself.
                    if seq_len <= 1 || self.attention == AttentionKind::Fused {
                        None
                    } else {
                        Some(self.prepare_causal_attention_mask(b_size, seq_len, seqlen_offset)?)
                    }
                }
            }
        };
        let mut xs = {
            let _span = device_debug_span!(device, sync, "embed")?;
            self.embed_tokens.forward(input_ids)?
        };
        for (i, layer) in self.layers.iter_mut().enumerate() {
            let _span = device_debug_span!(device, sync, "layer", i)?;
            xs = layer.forward(&xs, attention_mask.as_ref(), seqlen_offset)?
        }
        let _span = device_debug_span!(device, sync, "norm")?;
        xs.apply(&self.norm)
    }

    /// Empties the KV cache of each layer and sets its kind. For
    /// `KvCacheKind::Prealloc`, `capacity` is the number of positions to
    /// allocate.
    pub fn reset_kv_cache(&mut self, kind: KvCacheKind, capacity: usize) {
        for layer in self.layers.iter_mut() {
            layer.reset_kv_cache(kind, capacity)
        }
    }

    /// Sets how each attention layer computes attention.
    pub fn set_attention(&mut self, kind: AttentionKind) {
        self.attention = kind;
        for layer in self.layers.iter_mut() {
            layer.set_attention(kind)
        }
    }
}

#[derive(Debug, Clone)]
pub struct ModelForCausalLM {
    base_model: Model,
    lm_head: Proj,
}

impl ModelForCausalLM {
    /// Builds the model from dense weights. When `sync_spans` is true, each
    /// trace span inside the model waits for the device when it starts and
    /// ends. See `LoadConfig::sync_spans`.
    pub fn new(cfg: &Config, vb: VarBuilder, sync_spans: bool) -> Result<Self> {
        let embeddings = vb
            .pp("model.embed_tokens")
            .get((cfg.vocab_size, cfg.hidden_size), "weight")?;
        let embed_tokens = Embedding::Dense(candle_nn::Embedding::new(
            embeddings.clone(),
            cfg.hidden_size,
        ));
        let lm_head = if vb.contains_tensor("lm_head.weight") {
            Proj::Dense(linear_no_bias(
                cfg.hidden_size,
                cfg.vocab_size,
                vb.pp("lm_head"),
            )?)
        } else {
            Proj::Dense(Linear::from_weights(embeddings, None))
        };
        let base_model = Model::new(cfg, Weights::Dense(vb), embed_tokens, sync_spans)?;
        Ok(Self {
            base_model,
            lm_head,
        })
    }

    /// Builds the model from quantized weights in a GGUF file, as written
    /// by `fornace-quantize`. The tied embedding and `lm_head` share one
    /// quantized tensor. The model computes in F32.
    pub fn new_quantized(cfg: &Config, vb: QVarBuilder, sync_spans: bool) -> Result<Self> {
        let embeddings = vb
            .pp("model.embed_tokens")
            .get((cfg.vocab_size, cfg.hidden_size), "weight")?;
        let embed_tokens = Embedding::Quantized(QMatMul::from_arc(embeddings.clone())?);
        let lm_head = if vb.contains_key("lm_head.weight") {
            Proj::Quantized(quantized_nn::linear_no_bias(
                cfg.hidden_size,
                cfg.vocab_size,
                vb.pp("lm_head"),
            )?)
        } else {
            Proj::Quantized(quantized_nn::Linear::from_arc(embeddings, None)?)
        };
        let base_model = Model::new(cfg, Weights::Quantized(vb), embed_tokens, sync_spans)?;
        Ok(Self {
            base_model,
            lm_head,
        })
    }

    /// The logits of the last position of each row: `[batch, 1, vocab]`.
    pub fn forward(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        let (_b_size, seq_len) = input_ids.dims2()?;
        let xs = self
            .hidden_states(input_ids, seqlen_offset)?
            .narrow(1, seq_len - 1, 1)?;
        self.lm_head(&xs)
    }

    /// The hidden states after the last norm, at every position:
    /// `[batch, seq, hidden]`. This adds the positions to the KV cache, as
    /// `forward` does.
    pub fn hidden_states(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        self.base_model.forward(input_ids, seqlen_offset, None)
    }

    /// Applies `lm_head` to hidden states: `[.., hidden]` to `[.., vocab]`.
    pub fn lm_head(&self, xs: &Tensor) -> Result<Tensor> {
        let _span = device_debug_span!(
            &self.base_model.device,
            self.base_model.sync_spans,
            "lm_head"
        )?;
        xs.apply(&self.lm_head)
    }

    /// See [`Model::reset_kv_cache`].
    pub fn reset_kv_cache(&mut self, kind: KvCacheKind, capacity: usize) {
        self.base_model.reset_kv_cache(kind, capacity)
    }

    /// See [`Model::set_attention`].
    pub fn set_attention(&mut self, kind: AttentionKind) {
        self.base_model.set_attention(kind)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attention_kind_round_trips_through_display_and_from_str() {
        for kind in [AttentionKind::Naive, AttentionKind::Fused] {
            assert_eq!(kind.to_string().parse::<AttentionKind>(), Ok(kind));
        }
        assert!("flash".parse::<AttentionKind>().is_err());
    }

    /// The layouts of K and V from each KV cache kind: contiguous, a narrow
    /// view of a larger buffer (`Prealloc`), and a transposed view (`Cat`).
    /// Only the transposed view needs a copy.
    #[test]
    fn dense_rows_copies_only_the_transposed_view() -> Result<()> {
        let device = Device::Cpu;
        let contiguous = Tensor::zeros((1, 2, 5, 8), DType::F32, &device)?;
        let narrow = Tensor::zeros((1, 2, 9, 8), DType::F32, &device)?.narrow(2, 0, 5)?;
        let transposed = Tensor::zeros((1, 5, 2, 8), DType::F32, &device)?.transpose(1, 2)?;
        assert!(!narrow.is_contiguous());
        assert_eq!(dense_rows(&contiguous)?.stride(), contiguous.stride());
        assert_eq!(dense_rows(&narrow)?.stride(), narrow.stride());
        assert_eq!(dense_rows(&transposed)?.stride(), [80, 40, 8, 1]);
        Ok(())
    }

    /// The fused kernel gives the same result as matmul and softmax with a
    /// causal mask, in F32, for a prefill, a chunk at an offset, and one
    /// decode token, with the KV layouts of each cache kind.
    #[cfg(feature = "metal")]
    #[test]
    fn fused_attention_matches_the_reference_on_metal() -> Result<()> {
        let device = Device::new_metal(0)?;
        let (heads, kv_heads, head_dim, kv_len) = (14, 2, 64, 300);
        let scale = 1f64 / (head_dim as f64).sqrt();
        let reference = |q: &Tensor, k: &Tensor, v: &Tensor| -> Result<Tensor> {
            let q_len = q.dim(2)?;
            let k = candle_transformers::utils::repeat_kv(k.contiguous()?, heads / kv_heads)?;
            let v = candle_transformers::utils::repeat_kv(v.contiguous()?, heads / kv_heads)?;
            let offset = kv_len - q_len;
            let mask: Vec<f32> = (0..q_len)
                .flat_map(|i| {
                    (0..kv_len).map(move |j| {
                        if j > i + offset {
                            f32::NEG_INFINITY
                        } else {
                            0.
                        }
                    })
                })
                .collect();
            let mask = Tensor::from_slice(&mask, (1, 1, q_len, kv_len), q.device())?;
            let w = (q.matmul(&k.t()?)? * scale)?.broadcast_add(&mask)?;
            candle_nn::ops::softmax_last_dim(&w)?.matmul(&v)
        };
        for q_len in [kv_len, 100, 1] {
            let q = Tensor::randn(0f32, 1., (1, heads, q_len, head_dim), &device)?;
            let k = Tensor::randn(0f32, 1., (1, kv_heads, kv_len, head_dim), &device)?;
            let v = Tensor::randn(0f32, 1., (1, kv_heads, kv_len, head_dim), &device)?;
            let want = reference(&q, &k, &v)?;
            let narrow = |x: &Tensor| -> Result<Tensor> {
                Tensor::cat(&[x, &x.zeros_like()?], 2)?.narrow(2, 0, kv_len)
            };
            let transposed =
                |x: &Tensor| -> Result<Tensor> { x.transpose(1, 2)?.contiguous()?.transpose(1, 2) };
            let layouts = [
                ("contiguous", k.clone(), v.clone()),
                ("narrow", narrow(&k)?, narrow(&v)?),
                ("transposed", transposed(&k)?, transposed(&v)?),
            ];
            for (layout, k, v) in layouts {
                let got = fused_attention(&q, &k, &v, scale, q_len > 1)?;
                let diff: f32 = (got - &want)?.abs()?.max_all()?.to_scalar()?;
                assert!(diff < 1e-4, "q_len {q_len}, {layout}: max diff {diff}");
            }
        }
        Ok(())
    }
}
