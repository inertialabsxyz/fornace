# Prompts

Fixed prompts for prefill measurements. Each `alice-<N>.txt` file is the start of *Alice's
Adventures in Wonderland* by Lewis Carroll, from Project Gutenberg eBook #11
(https://www.gutenberg.org/ebooks/11). The text is in the public domain. The
Project Gutenberg header and licence are removed.

| File | Tokens |
|---|---|
| `alice-128.txt` | 128 |
| `alice-512.txt` | 512 |
| `alice-2048.txt` | 2048 |
| `alice-4096.txt` | 4096 |

Two more prompts test speculative decoding on other kinds of text:

| File | Tokens | Source |
|---|---|---|
| `code-512.txt` | 512 | The start of `crates/fornace-engine/src/cache.rs` in this repository, at commit `4170ca0` (MIT licence). New code, so no model has it from training |
| `alice-de-512.txt` | 512 | The start of chapter 1 of *Alice's Abenteuer im Wunderland*, the German translation by Antonie Zimmermann, from Project Gutenberg eBook #19778 (https://www.gutenberg.org/ebooks/19778). Public domain. The header, the licence, and the front matter are removed |

**How the language was chosen.** A draft must be weak at the language,
so the brief takes the language with the largest gap in perplexity
between the draft (Qwen2.5-0.5B) and the target (Qwen2.5-1.5B). The
candidates were the first 512 tokens of chapter 1 of five translations
of the same book on Project Gutenberg: German (#19778), French (#55456),
Italian (#28371), Finnish (#46569), and Esperanto (#17482). Perplexity in
BF16 on Metal, from `fornace-bench --quality` on 2026-11-12, working tree
on top of `f2d7519`:

| Text | 0.5B | 1.5B | 0.5B / 1.5B |
|---|---:|---:|---:|
| English, `alice-512.txt` | 9.74 | 1.74 | 5.61 |
| Esperanto | 254.27 | 113.01 | 2.25 |
| German | 22.45 | 11.91 | 1.88 |
| Italian | 29.19 | 16.66 | 1.75 |
| Finnish | 74.08 | 42.19 | 1.76 |
| French | 15.83 | 10.00 | 1.58 |
| Code, `code-512.txt` | 10.97 | 8.64 | 1.27 |

Esperanto has the largest gap, but the Gutenberg text uses the ASCII
"x-system" ("cxapitro" for "ĉapitro"), which is not the normal spelling.
So German, the next largest gap, is the prompt. Note the English row: the
1.5B has a perplexity of 1.74 on the start of *Alice*, so it knows the
text nearly by heart. The 0.5B does not.

Command for each text and each model:

```sh
cargo run --release --features metal -p fornace-bench -- \
    --model ./models/qwen2.5-0.5b-instruct --prompts bench/prompts/alice-128.txt \
    --attention fused --max-new-tokens 2 --warmup 0 --repeats 1 \
    --quality <the 512-token text>
```

The token counts are for the Qwen2.5 tokenizer. They are exact when you pass
the file with `"$(cat FILE)"`, because the shell removes the final newline. If
a tool reads the file with its final newline, the count is one more.

Example:

```sh
cargo run --release --features metal -p fornace-engine --bin fornace -- \
    --model ./models/qwen2.5-0.5b-instruct \
    --prompt "$(cat bench/prompts/alice-2048.txt)" \
    --max-new-tokens 1 --trace summary
```
