# Results

This is the list of results, newest first. The title of each result gives
the result with a number. To write a result, use [TEMPLATE.md](TEMPLATE.md).

| Date | Title | Commit | Link |
|---|---|---|---|
| 2026-11-05 | Batch 4 gives 1.6× the total decode speed of batch 1 for Qwen2 on Metal, but each sequence decodes at 0.40× | `dae67a1` | [result](2026-11-05-batching-qwen2-metal.md) |
| 2026-11-04 | Fused attention makes a 4,096-token Qwen2 prefill on Metal 4.5× faster and lowers its peak from 3,403 MiB to 1,470 MiB | `c3693e3` | [result](2026-11-04-fused-attention-prefill-qwen2-metal.md) |
| 2026-11-04 | A 1 MiB KV cache tensor holds a 512 MiB buffer in each layer, and makes a 4,096-token Qwen2 prefill on Metal peak at 14.5 GiB instead of 3.4 GiB | `bd42949`, `243a9a2` | [result](2026-11-04-metal-buffer-reuse-prefill-memory-qwen2.md) |
| 2026-10-29 | A non-contiguous KV cache cuts Qwen2 decode speed by 59 % at 2k context in Candle on Metal | `e76e1ba` | [result](2026-10-29-noncontiguous-kv-cache-qwen2-metal.md) |
