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
//! - the RoPE tables are computed in F32 and then converted to the compute
//!   dtype. The original converts the positions and the frequencies to BF16
//!   and computes the angles in BF16, so each angle is rounded
//!   (`RotaryEmbedding::new`).
//! - a paged KV cache (`KvCacheKind::Paged`, `reset_paged_kv_cache`). The
//!   model owns the block tables. It gives each layer the write and read
//!   slots of the step, and each layer gathers K and V before attention.
//!   `fork_kv_cache` makes B rows from one prefill: with shared blocks
//!   (paged), or with one copy (prealloc).
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

use crate::cache::{
    BlockUsage, DEFAULT_BLOCK_SIZE, KvCacheKind, LayerCache, PagedCache, PagedStep, paged_step,
    scatter_ids,
};
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
            (Weights::Quantized(vb), true) => Proj::Quantized(
                quantized_nn::linear(in_dim, out_dim, vb.clone())?,
                usize::MAX,
            ),
            (Weights::Quantized(vb), false) => Proj::Quantized(
                quantized_nn::linear_no_bias(in_dim, out_dim, vb.clone())?,
                usize::MAX,
            ),
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
    /// The usize: the fewest rows of a decode batch that go on the M axis
    /// (see [`ModelForCausalLM::set_quantized_m_axis_rows`]). `usize::MAX`
    /// is never.
    Quantized(quantized_nn::Linear, usize),
}

impl Proj {
    fn set_m_axis_rows(&mut self, min_rows: usize) {
        if let Proj::Quantized(_, rows) = self {
            *rows = min_rows;
        }
    }
}

impl Module for Proj {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Proj::Dense(p) => p.forward(xs),
            // Candle's quantized matmul takes only contiguous input, and on
            // Metal it ignores the start offset of an input with more than
            // one row. A batch gives both: `lm_head` gets the last position
            // of each row, `[batch, 1, hidden]`, as a strided view. So copy
            // such an input to a new buffer at offset 0.
            Proj::Quantized(p, m_axis_rows) => {
                let xs = if xs.is_contiguous() && xs.layout().start_offset() == 0 {
                    xs.clone()
                } else {
                    xs.force_contiguous()?
                };
                match xs.dims() {
                    // Candle sends an input with one row on the M axis to
                    // its matrix-vector kernel, once for each row of the
                    // batch axis. As `[B, K]`, the B rows go to the matmul
                    // kernel, which reads each weight block once.
                    &[b, 1, k] if b > 1 && b >= *m_axis_rows => {
                        p.forward(&xs.reshape((b, k))?)?.unsqueeze(1)
                    }
                    _ => p.forward(&xs),
                }
            }
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

/// Returns `x` if each row of the last dimension is dense, the rows are
/// next to each other, and the batch stride is the head stride × the head
/// count, else a contiguous copy. The one-token kernel of `sdpa` reads K
/// and V with only the head stride, and assumes this layout. The `Cat`
/// cache gives V as a transposed view, which does not have it.
fn dense_rows(x: &Tensor) -> Result<Tensor> {
    let stride = x.stride();
    let batch_ok = x.dim(0)? == 1 || stride[0] == x.dim(1)? * stride[1];
    if stride[3] == 1 && stride[2] == x.dim(3)? && batch_ok {
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
        // Compute the angles in F32, and convert only sin and cos to
        // `dtype`. BF16 has an 8-bit mantissa. If the positions, the
        // frequencies, and their products were BF16, each angle would be
        // rounded, and the error would grow with the position. Positions
        // above 256 would also not be exact. On alice-2048, that gave a
        // BF16 perplexity of 39.7 against 12.5 in F32.
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?.to_dtype(dtype)?,
            cos: freqs.cos()?.to_dtype(dtype)?,
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

    /// The cos and sin rows of one position for each row, as
    /// `[B, 1, dim / 2]`, for [`Self::apply_rows`]. One gather for each step,
    /// shared by all layers.
    fn rows_cos_sin(&self, positions: &[usize]) -> Result<(Tensor, Tensor)> {
        let ids: Vec<u32> = positions.iter().map(|&p| p as u32).collect();
        let ids = Tensor::new(ids.as_slice(), self.cos.device())?;
        let cos = self.cos.index_select(&ids, 0)?.unsqueeze(1)?;
        let sin = self.sin.index_select(&ids, 0)?.unsqueeze(1)?;
        Ok((cos, sin))
    }

    /// RoPE for one token in each row, each row at its own position.
    fn apply_rows(
        &self,
        q: &Tensor,
        k: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, cos, sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, cos, sin)?;
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
        paged: Option<&PagedStep>,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;
        let (device, sync) = (xs.device(), self.sync_spans);
        let (query_states, key_states, value_states) = self.qkv(xs)?;

        let (query_states, key_states) = {
            let _span = device_debug_span!(device, sync, "attn.rope")?;
            self.rotary_emb
                .apply_rotary_emb_qkv(&query_states, &key_states, seqlen_offset)?
        };

        let (key_states, value_states) = match paged {
            None => {
                let _span = device_debug_span!(device, sync, "attn.kv_append")?;
                self.kv_cache.append(key_states, value_states)?
            }
            // The write and the gather are one span: together they are
            // what the paged cache adds to a step.
            Some(step) => {
                let _span = device_debug_span!(device, sync, "attn.gather")?;
                self.kv_cache.append_paged(key_states, value_states, step)?
            }
        };

        let attn_output = self.attend(&query_states, &key_states, &value_states, attention_mask)?;
        self.output(&attn_output, b_sz, q_len)
    }

    /// The Q, K, and V projections of all rows, as `[B, heads, T, dim]`.
    fn qkv(&self, xs: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let (b_sz, q_len, _) = xs.dims3()?;
        let (device, sync) = (xs.device(), self.sync_spans);
        let states = {
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
        Ok(states)
    }

    /// Attention of the queries over all keys and values so far.
    fn attend(
        &self,
        query_states: &Tensor,
        key_states: &Tensor,
        value_states: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let q_len = query_states.dim(2)?;
        let (device, sync) = (query_states.device(), self.sync_spans);
        let scale = 1f64 / f64::sqrt(self.head_dim as f64);
        let attn_output = match self.attention {
            AttentionKind::Naive => {
                let (key_states, value_states) = {
                    let _span = device_debug_span!(device, sync, "attn.repeat_kv")?;
                    let key_states = candle_transformers::utils::repeat_kv(
                        key_states.clone(),
                        self.num_kv_groups,
                    )?
                    .contiguous()?;
                    let value_states = candle_transformers::utils::repeat_kv(
                        value_states.clone(),
                        self.num_kv_groups,
                    )?
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
                fused_attention(query_states, key_states, value_states, scale, q_len > 1)?
            }
        };
        Ok(attn_output)
    }

    /// `[B, heads, T, dim]` back to `[B, T, hidden]`, and the O projection.
    fn output(&self, attn_output: &Tensor, b_sz: usize, q_len: usize) -> Result<Tensor> {
        let (device, sync) = (attn_output.device(), self.sync_spans);
        let _span = device_debug_span!(device, sync, "attn.o_proj")?;
        attn_output
            .transpose(1, 2)?
            .reshape((b_sz, q_len, self.hidden_size))?
            .apply(&self.o_proj)
    }

    /// As `forward`, for rows that each have their own cache and position.
    /// The projections run on all rows at once. RoPE, the cache append,
    /// and the attention run once for each row.
    fn forward_rows(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        caches: &mut [&mut LayerCache],
        rope: &RowsRope,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;
        let (device, sync) = (xs.device(), self.sync_spans);
        let (query_states, key_states, value_states) = self.qkv(xs)?;
        // One RoPE launch for all rows when each row has one token.
        let roped = match rope {
            RowsRope::Tables(cos, sin) => {
                let _span = device_debug_span!(device, sync, "attn.rope")?;
                Some(
                    self.rotary_emb
                        .apply_rows(&query_states, &key_states, cos, sin)?,
                )
            }
            RowsRope::Offset(_) => None,
        };
        let mut outputs = Vec::with_capacity(b_sz);
        for (b, cache) in caches.iter_mut().enumerate() {
            let (q, k) = match (&roped, rope) {
                (Some((q, k)), _) => (q.narrow(0, b, 1)?, k.narrow(0, b, 1)?),
                (None, RowsRope::Offset(offset)) => {
                    let _span = device_debug_span!(device, sync, "attn.rope")?;
                    self.rotary_emb.apply_rotary_emb_qkv(
                        &query_states.narrow(0, b, 1)?,
                        &key_states.narrow(0, b, 1)?,
                        *offset,
                    )?
                }
                (None, RowsRope::Tables(..)) => unreachable!("tables give roped states"),
            };
            let (k, v) = {
                let _span = device_debug_span!(device, sync, "attn.kv_append")?;
                cache.append(k, value_states.narrow(0, b, 1)?)?
            };
            outputs.push(self.attend(&q, &k, &v, attention_mask)?);
        }
        let attn_output = if outputs.len() == 1 {
            outputs.pop().expect("one row")
        } else {
            Tensor::cat(&outputs, 0)?
        };
        self.output(&attn_output, b_sz, q_len)
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
        paged: Option<&PagedStep>,
    ) -> Result<Tensor> {
        let (device, sync) = (xs.device(), self.sync_spans);
        let xs = {
            let _span = device_debug_span!(device, sync, "attn")?;
            let residual = xs;
            let xs = self.input_layernorm.forward(xs)?;
            let xs = self
                .self_attn
                .forward(&xs, attention_mask, seqlen_offset, paged)?;
            (xs + residual)?
        };
        let _span = device_debug_span!(device, sync, "mlp")?;
        let residual = &xs;
        let xs = xs.apply(&self.post_attention_layernorm)?.apply(&self.mlp)?;
        residual + xs
    }

    fn forward_rows(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        caches: &mut [&mut LayerCache],
        rope: &RowsRope,
    ) -> Result<Tensor> {
        let (device, sync) = (xs.device(), self.sync_spans);
        let xs = {
            let _span = device_debug_span!(device, sync, "attn")?;
            let h = self.input_layernorm.forward(xs)?;
            (self
                .self_attn
                .forward_rows(&h, attention_mask, caches, rope)?
                + xs)?
        };
        let _span = device_debug_span!(device, sync, "mlp")?;
        let h = xs.apply(&self.post_attention_layernorm)?.apply(&self.mlp)?;
        xs + h
    }

    fn reset_kv_cache(&mut self, kind: KvCacheKind, capacity: usize) {
        self.self_attn.reset_kv_cache(kind, capacity)
    }

    fn set_attention(&mut self, kind: AttentionKind) {
        self.self_attn.set_attention(kind)
    }
}

/// How [`Model::forward_rows`] gives the positions to RoPE.
enum RowsRope {
    /// One row of T tokens, from this position.
    Offset(usize),
    /// One token in each row: the cos and sin rows of each position.
    Tables(Tensor, Tensor),
}

/// The KV cache of one sequence for [`ModelForCausalLM::forward_rows`]: a
/// `prealloc` cache in each layer, and the positions stored. A batch of
/// rows has one `RowCache` for each row, so the rows can have different
/// lengths, and a row can join or leave the batch.
#[derive(Debug, Clone)]
pub struct RowCache {
    layers: Vec<LayerCache>,
    len: usize,
}

impl RowCache {
    /// The positions stored.
    pub fn len(&self) -> usize {
        self.len
    }

    /// True if no position is stored.
    pub fn is_empty(&self) -> bool {
        self.len == 0
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
    /// The block tables, if the KV cache is `Paged`.
    paged: Option<PagedCache>,
    num_kv_heads: usize,
    head_dim: usize,
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
            paged: None,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.hidden_size / cfg.num_attention_heads,
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
            // F32, as `mask`. The cast to the compute dtype comes at the end.
            let mask0 = Tensor::zeros((tgt_len, seqlen_offset), DType::F32, &self.device)?;
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
        // One set of slots for all layers, made on the CPU and uploaded
        // once.
        let paged = match &mut self.paged {
            None => None,
            Some(cache) => {
                let _span = device_debug_span!(device, sync, "paged.slots")?;
                let (write, read) = cache.plan(b_size, seqlen_offset, seq_len)?;
                Some(paged_step(
                    write,
                    read,
                    b_size,
                    self.num_kv_heads,
                    self.head_dim,
                    device,
                )?)
            }
        };
        let mut xs = {
            let _span = device_debug_span!(device, sync, "embed")?;
            self.embed_tokens.forward(input_ids)?
        };
        for (i, layer) in self.layers.iter_mut().enumerate() {
            let _span = device_debug_span!(device, sync, "layer", i)?;
            xs = layer.forward(&xs, attention_mask.as_ref(), seqlen_offset, paged.as_ref())?
        }
        let _span = device_debug_span!(device, sync, "norm")?;
        xs.apply(&self.norm)
    }

    /// The hidden states of rows that each have their own cache. `input_ids`
    /// is `[B, T]`, with one cache in `rows` for each row. Each row starts
    /// at the position after its cache. T above 1 (a prefill) needs one row.
    pub fn forward_rows(
        &mut self,
        input_ids: &Tensor,
        rows: &mut [&mut RowCache],
    ) -> Result<Tensor> {
        let (b_size, seq_len) = input_ids.dims2()?;
        if b_size != rows.len() {
            candle_core::bail!("{b_size} rows of ids and {} caches", rows.len());
        }
        if seq_len > 1 && b_size > 1 {
            candle_core::bail!("a forward of {seq_len} tokens needs one row, not {b_size}");
        }
        let (device, sync) = (&self.device, self.sync_spans);
        let offsets: Vec<usize> = rows.iter().map(|r| r.len).collect();
        let attention_mask = if seq_len <= 1 || self.attention == AttentionKind::Fused {
            None
        } else {
            let _span = device_debug_span!(device, sync, "mask")?;
            Some(self.prepare_causal_attention_mask(1, seq_len, offsets[0])?)
        };
        let rope = if seq_len == 1 {
            // All layers share one table (an `Arc`).
            let (cos, sin) = self.layers[0].self_attn.rotary_emb.rows_cos_sin(&offsets)?;
            RowsRope::Tables(cos, sin)
        } else {
            RowsRope::Offset(offsets[0])
        };
        let mut xs = {
            let _span = device_debug_span!(device, sync, "embed")?;
            self.embed_tokens.forward(input_ids)?
        };
        for (i, layer) in self.layers.iter().enumerate() {
            let _span = device_debug_span!(device, sync, "layer", i)?;
            let mut caches: Vec<&mut LayerCache> =
                rows.iter_mut().map(|r| &mut r.layers[i]).collect();
            xs = layer.forward_rows(&xs, attention_mask.as_ref(), &mut caches, &rope)?;
        }
        for row in rows.iter_mut() {
            row.len += seq_len;
        }
        let _span = device_debug_span!(device, sync, "norm")?;
        xs.apply(&self.norm)
    }

    /// Empties the KV cache of each layer and sets its kind. For
    /// `KvCacheKind::Prealloc`, `capacity` is the number of positions to
    /// allocate. For `KvCacheKind::Paged`, the pool holds `capacity`
    /// positions for one row, in blocks of `DEFAULT_BLOCK_SIZE`. Use
    /// `reset_paged_kv_cache` for other sizes.
    pub fn reset_kv_cache(&mut self, kind: KvCacheKind, capacity: usize) {
        if kind == KvCacheKind::Paged {
            let blocks = capacity.div_ceil(DEFAULT_BLOCK_SIZE);
            return self.reset_paged_kv_cache(DEFAULT_BLOCK_SIZE, blocks);
        }
        self.paged = None;
        for layer in self.layers.iter_mut() {
            layer.reset_kv_cache(kind, capacity)
        }
    }

    /// Empties the KV cache and makes it a paged cache with a pool of
    /// `num_blocks` blocks of `block_size` positions, for all rows.
    pub fn reset_paged_kv_cache(&mut self, block_size: usize, num_blocks: usize) {
        let cache = PagedCache::new(num_blocks, block_size);
        for layer in self.layers.iter_mut() {
            layer.reset_kv_cache(KvCacheKind::Paged, cache.slots())
        }
        self.paged = Some(cache);
    }

    /// Makes `rows` rows from the one row of the cache, for example after
    /// the prefill of a shared prefix. A paged cache shares the full blocks
    /// of the prefix, and each new row gets a copy of the last block if it
    /// is not full. A `Prealloc` cache copies the prefix into `rows` rows.
    pub fn fork_kv_cache(&mut self, rows: usize) -> Result<()> {
        let Some(cache) = &mut self.paged else {
            let _span = device_debug_span!(&self.device, self.sync_spans, "prealloc.fork")?;
            for layer in self.layers.iter_mut() {
                layer.self_attn.kv_cache.fork_prealloc(rows)?;
            }
            return Ok(());
        };
        let (from, to) = cache.fork(rows)?;
        if from.is_empty() {
            return Ok(());
        }
        let _span = device_debug_span!(&self.device, self.sync_spans, "paged.fork")?;
        let from = Tensor::from_vec(from, to.len(), &self.device)?;
        let to = scatter_ids(to, self.num_kv_heads, self.head_dim, &self.device)?;
        for layer in self.layers.iter_mut() {
            layer.self_attn.kv_cache.copy_slots(&from, &to)?;
        }
        Ok(())
    }

    /// The block usage of a paged cache, or `None` for the other kinds.
    pub fn kv_block_usage(&self) -> Option<BlockUsage> {
        self.paged.as_ref().map(PagedCache::usage)
    }

    /// Keeps only the first `len` positions in the KV cache of each layer.
    /// Only the `Prealloc` cache supports it.
    pub fn truncate_kv_cache(&mut self, len: usize) -> Result<()> {
        for layer in self.layers.iter_mut() {
            layer.self_attn.kv_cache.truncate(len)?;
        }
        Ok(())
    }

    /// Sets how each attention layer computes attention.
    pub fn set_attention(&mut self, kind: AttentionKind) {
        self.attention = kind;
        for layer in self.layers.iter_mut() {
            layer.set_attention(kind)
        }
    }

    fn set_quantized_m_axis_rows(&mut self, min_rows: usize) {
        for layer in self.layers.iter_mut() {
            let (a, m) = (&mut layer.self_attn, &mut layer.mlp);
            for p in [
                &mut a.q_proj,
                &mut a.k_proj,
                &mut a.v_proj,
                &mut a.o_proj,
                &mut m.gate_proj,
                &mut m.up_proj,
                &mut m.down_proj,
            ] {
                p.set_m_axis_rows(min_rows);
            }
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
            Proj::Quantized(
                quantized_nn::linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?,
                usize::MAX,
            )
        } else {
            Proj::Quantized(
                quantized_nn::Linear::from_arc(embeddings, None)?,
                usize::MAX,
            )
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

    /// See [`Model::reset_paged_kv_cache`].
    pub fn reset_paged_kv_cache(&mut self, block_size: usize, num_blocks: usize) {
        self.base_model.reset_paged_kv_cache(block_size, num_blocks)
    }

    /// See [`Model::fork_kv_cache`].
    pub fn fork_kv_cache(&mut self, rows: usize) -> Result<()> {
        self.base_model.fork_kv_cache(rows)
    }

    /// See [`Model::kv_block_usage`].
    pub fn kv_block_usage(&self) -> Option<BlockUsage> {
        self.base_model.kv_block_usage()
    }

    /// See [`Model::truncate_kv_cache`].
    pub fn truncate_kv_cache(&mut self, len: usize) -> Result<()> {
        self.base_model.truncate_kv_cache(len)
    }

    /// See [`Model::set_attention`].
    pub fn set_attention(&mut self, kind: AttentionKind) {
        self.base_model.set_attention(kind)
    }

    /// An empty cache for one row of [`Self::forward_rows`], with
    /// `capacity` positions in each layer. It grows if a row needs more.
    pub fn new_row_cache(&self, capacity: usize) -> RowCache {
        RowCache {
            layers: (0..self.base_model.layers.len())
                .map(|_| LayerCache::new(KvCacheKind::Prealloc, capacity))
                .collect(),
            len: 0,
        }
    }

    /// The logits of the last position of each row, `[B, 1, vocab]`, for
    /// rows that each have their own cache and position. See
    /// [`Model::forward_rows`].
    pub fn forward_rows(
        &mut self,
        input_ids: &Tensor,
        rows: &mut [&mut RowCache],
    ) -> Result<Tensor> {
        let (_b_size, seq_len) = input_ids.dims2()?;
        let xs = self
            .base_model
            .forward_rows(input_ids, rows)?
            .narrow(1, seq_len - 1, 1)?;
        self.lm_head(&xs)
    }

    /// With quantized weights: run a decode step of B rows (`[B, 1, K]`)
    /// as one `[B, K]` matmul in each projection and in `lm_head`, not as
    /// one matrix-vector launch for each row. Dense weights already do
    /// this (`candle_nn::Linear` reshapes the input). Off by default.
    pub fn set_quantized_rows_on_m(&mut self, on: bool) {
        self.set_quantized_m_axis_rows(if on { 2 } else { usize::MAX });
    }

    /// As [`Self::set_quantized_rows_on_m`], from `min_rows` rows. Below it,
    /// one matrix-vector launch for each row is faster (Candle's matmul
    /// kernels have a floor of about 12 ms a step at a small M, on an M4
    /// Pro with the 0.5B model). `usize::MAX` is never.
    pub fn set_quantized_m_axis_rows(&mut self, min_rows: usize) {
        self.base_model.set_quantized_m_axis_rows(min_rows);
        self.lm_head.set_m_axis_rows(min_rows);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The RoPE tables of Qwen2.5-0.5B, for the given dtype.
    fn rope(dtype: DType) -> Result<RotaryEmbedding> {
        let cfg = Config {
            vocab_size: 8,
            hidden_size: 896,
            intermediate_size: 8,
            num_hidden_layers: 1,
            num_attention_heads: 14,
            num_key_value_heads: 2,
            max_position_embeddings: 4096,
            sliding_window: 4096,
            max_window_layers: 1,
            tie_word_embeddings: true,
            rope_theta: 1_000_000.0,
            rms_norm_eps: 1e-6,
            use_sliding_window: false,
            hidden_act: Activation::Silu,
        };
        RotaryEmbedding::new(dtype, &cfg, &Device::Cpu)
    }

    /// In BF16, the sin and cos of each position are the F32 values, rounded
    /// once. If the angles were computed in BF16, they would be rounded
    /// before sin and cos, and late tokens would get the largest error.
    #[test]
    fn bf16_rope_tables_are_the_f32_tables_rounded_once() -> Result<()> {
        let f32 = rope(DType::F32)?;
        let bf16 = rope(DType::BF16)?;
        for (name, exact, got) in [("sin", &f32.sin, &bf16.sin), ("cos", &f32.cos, &bf16.cos)] {
            assert_eq!(got.dtype(), DType::BF16);
            let expected = exact.to_dtype(DType::BF16)?.to_dtype(DType::F32)?;
            let diff: f32 = (got.to_dtype(DType::F32)? - expected)?
                .abs()?
                .flatten_all()?
                .max(0)?
                .to_scalar()?;
            assert_eq!(diff, 0.0, "{name}");
        }
        Ok(())
    }

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
        // A batch of narrow views keeps the batch stride = heads × head
        // stride. A batch with the heads outside the rows does not.
        let batch_narrow = Tensor::zeros((3, 2, 9, 8), DType::F32, &device)?.narrow(2, 0, 5)?;
        assert_eq!(dense_rows(&batch_narrow)?.stride(), batch_narrow.stride());
        let heads_outside = Tensor::zeros((2, 3, 5, 8), DType::F32, &device)?.transpose(0, 1)?;
        assert_eq!(heads_outside.stride(), [40, 120, 8, 1]);
        assert_eq!(dense_rows(&heads_outside)?.stride(), [80, 40, 8, 1]);
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
