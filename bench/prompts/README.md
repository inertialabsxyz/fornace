# Prompts

Fixed prompts for prefill measurements. Each file is the start of *Alice's
Adventures in Wonderland* by Lewis Carroll, from Project Gutenberg eBook #11
(https://www.gutenberg.org/ebooks/11). The text is in the public domain. The
Project Gutenberg header and licence are removed.

| File | Tokens |
|---|---|
| `alice-128.txt` | 128 |
| `alice-512.txt` | 512 |
| `alice-2048.txt` | 2048 |
| `alice-4096.txt` | 4096 |

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
