# Constrained sampling tasks

`tasks.jsonl` holds 300 generated tasks for a simulated payments ledger. Each
line is one task. The code that made the file is in the same commit:
`crates/fornace-constrain/src/tasks.rs` and `src/bin/tasks.rs`.

## Command

```sh
cargo build --release -p fornace-constrain --features metal --bin tasks
target/release/tasks generate --seed 1 --out data/constrained/tasks.jsonl \
  --model models/qwen2.5-7b-instruct
target/release/tasks check data/constrained/tasks.jsonl
```

With no `--model`, `generate` makes the same tasks with no request text.

## Classes

| Class | Tasks | IDs | Correct result |
|---|---|---|---|
| Usual | 180 | 0 to 179 | The goal after `DONE` |
| Over-limit | 60 | 180 to 239 | `ASK_OPERATOR` |
| Wrong-status, prefix ID | 30 | 240 to 269 | `ASK_OPERATOR` |
| Wrong-status, no prefix ID | 30 | 270 to 299 | `ASK_OPERATOR` |

- **Over-limit:** a capture above the authorized amount, a refund above the
  rest of the capture, or a transfer above the balance. The requested amount
  is between the limit + 100 and 2 times the limit.
- **Wrong-status:** a refund of a voided or refunded payment, or a capture
  or a void of a payment that is not authorized. In a prefix-ID task, `p_1`
  has a status that permits the command, and the request names one of
  `p_10` to `p_19`.

## Fields

| Field | Content |
|---|---|
| `seed` | The seed of the task. `generate_one(id, seed)` gives the task again |
| `class`, `prefix_id` | The class |
| `start` | The start state: 2 to 5 accounts, 3 to 20 payments (11 to 20 for a prefix-ID task). All amounts are multiples of 100 cents |
| `actions` | The actions that the request asks for |
| `goal` | The conditions on the final state |
| `shortest`, `shortest_len` | The shortest command sequence, with no `DONE`. Usual tasks only |
| `requested_amounts`, `unit` | The amounts that the request names, in cents, and the unit of the request text |
| `style` | `short`, `long`, or `support` |
| `request`, `request_seed` | The request text and the sampler seed that wrote it |
| `request_rejects` | Each rejected text, with the reason |

## The shortest sequence

The search tries only the commands on the payments and accounts that the
goal names, and a usual task has at most 4 commands (decision of 20
November 2026). The full search does not end in useful time on the largest
ledgers at length 3. For lengths up to 3, `tasks check` runs the full search
at each shorter length and finds no shorter sequence. A length of 4 is an
upper bound.

## The request text

| Item | Value |
|---|---|
| Model | Qwen2.5-7B-Instruct |
| Format | Q8_0, `model-q8_0.gguf`, quantized by `fornace-quantize` |
| Engine | Fornace, Metal, `generate_stream`, batch 1 |
| Temperature | 0.7 |
| Repeat penalty | 1.0 |
| Seeds | The task seed, then the task seed + 1 for each new try. 5 tries at most |
| Maximum length | 200 tokens |

The system message is `REQUEST_SYSTEM` in `tasks.rs`. The user message is
`request_prompt` in `tasks.rs`: one line for each action, with the amounts
in the unit of the task, and the style. The style and the unit come from
the task seed, not from the model.

A text passes the check if it names each ID of the actions and no other ID,
and each requested amount in the unit of the task and no other number.
Each rejected text and its reason is in `request_rejects`.
