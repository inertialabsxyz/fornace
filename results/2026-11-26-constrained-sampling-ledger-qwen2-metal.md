# On Qwen2.5-1.5B, a state-aware mask makes 100 % of ledger commands valid but answers only 11 % of impossible requests correctly; a check of the request before the model runs raises that to 90 %

## Why it matters

An agent that sends commands to a payment system must not send a command
that the system refuses. A grammar fixes the form of the output, but not
the state: it still permits a refund of a voided payment. So this work
puts the live state into the sampler. Before each command, the engine
reads the ledger and permits only the commands, IDs, and amounts that the
ledger accepts. The model then cannot write a command that the ledger
refuses.

That guarantee has a cost. When a request is impossible, a hard mask can
turn it into a valid command that is wrong, and nothing reports it. This
result measures how often that happens. Then it tests two changes that
make the error visible: a stop when the mask blocks most of the
probability, and a check of the request against the ledger before the
model runs.

## What was measured and how

| Item | Value |
|---|---|
| Machine | Apple M4 Pro, 20 GPU cores, 24 GB unified memory, macOS 27.2.1 |
| Engine | Fornace, Metal: `27cf7f3` for the baseline and the workarounds, `ce316ce` (run as `42d231b`) and `61b8dcd` for the mask pressure, `591cf7d` for the request check and the fault and name tests. The sampler has a hook between the repeat penalty and the sampling (`LogitsHook`) |
| Model | Qwen2.5-1.5B-Instruct, `f16`. A pilot of configuration A on Qwen2.5-0.5B-Instruct gave 1 of 30 usual tasks (3.3 %), less than the 10 % limit of the plan, so all runs use the 1.5B (`bench/runs/constrained/2026-11-26-be2d12b-a-qwen2.5-0.5b-instruct-pilot-outcomes.jsonl`) |
| Grammar engine | `llguidance` 1.9.1, Lark grammars, compiled again for each command |
| Ledger | A simulator with accounts and payments. Five money commands, each with a guard on the status and on the amount, plus `ASK_OPERATOR` and `DONE` |
| Tasks | 300 generated tasks (`data/constrained/tasks.jsonl`). 180 usual tasks of 1 to 4 commands. 60 over-limit tasks: the request asks for more than the limit. 60 wrong-status tasks: the request names a payment whose status does not permit the command. In 30 of them, the requested ID (`p_10` to `p_19`) starts with the ID of a payment that the state permits (`p_1`). Qwen2.5-7B-Instruct (Q8_0) on Fornace wrote the request texts. In 161 tasks, the amounts are in dollars. In 139, they are in cents |
| Sampling | Temperature 0.7, repeat penalty 1.0, seeds 1, 2, 3. At most 10 commands for each task and 128 tokens for each command. The prompt has the full state and each command so far, with the answer of the ledger |
| Score | A usual task succeeds if the goal is true after `DONE`, with no refused command, in at most 2 × the shortest length + 2 commands. A trap task succeeds if `ASK_OPERATOR` comes before the ledger executes a money command |
| Runs | One process for each configuration, 300 tasks × 3 seeds (900 runs). Workarounds 1 and 5 ran on the 120 trap tasks only (360 runs). The baseline ran on 2026-11-23, the mask pressure on 2026-11-25, and the request check on 2026-11-26 |
| Data | `bench/runs/constrained/`, two files for each run: the baseline `2026-11-23-27cf7f3-*`, the changes `2026-11-25-42d231b-*`, `2026-11-25-61b8dcd-*`, and `2026-11-26-591cf7d-*`. The tables come from these files, made by `cargo run -p fornace-constrain --bin tables` and `--bin pressure` at `b4fa5c5`: `2026-11-26-b4fa5c5-tables.md` and `2026-11-26-b4fa5c5-pressure.md`. The record run `42d231b` has the tree of `ce316ce` (the same code, a corrected message) |

The configurations:

- **A, no constraint.** The rules are in the prompt only. The ledger
  refuses a bad command, and the model sees the refusal and tries again.
- **B, grammar only.** A grammar of the seven command forms, with any ID
  and any amount of 1 to 7 digits. The same retry rule.
- **C, state mask.** Before each command, the rules read the state and
  give a grammar with one branch for each permitted ID. Each branch has
  the amount range of that ID, as a pattern on digits.
- **D, rejection.** The grammar of B. Then the check of the ledger. If the
  state does not permit the command, the bench discards it and samples
  again with a new seed, at most five times. After five refused samples,
  the bench sends `ASK_OPERATOR`. A discarded sample is not in the
  context of the model.

**The configurations** (each rate with a 95 % Wilson interval):

| Configuration | Parse failures | Valid commands | Usual tasks | Over-limit tasks | Wrong-status tasks | False alarms (usual) | Tokens for each task |
|---|---:|---:|---:|---:|---:|---:|---:|
| A, no constraint | 903 of 3,942 outputs | 64.4 % [62.7, 66.1] | 23.3 % [20.0, 27.1] | 38.3 % [31.5, 45.6] | 35.0 % [28.4, 42.2] | 171 / 540 | 89.6 |
| B, grammar only | 0 | 63.0 % [61.4, 64.6] | 27.2 % [23.6, 31.1] | 46.7 % [39.5, 53.9] | 38.9 % [32.1, 46.2] | 188 / 540 | 67.9 |
| C, state mask | 0 | **100.0 %** [99.9, 100.0] | 31.7 % [27.9, 35.7] | **17.2 %** [12.4, 23.4] | **5.6 %** [3.0, 9.9] | 138 / 540 | 65.9 |
| D, rejection | 0 | 100.0 % [99.8, 100.0] | **35.6 %** [31.6, 39.7] | 40.0 % [33.1, 47.3] | 72.2 % [65.3, 78.2] | 205 / 540 | 79.6 |

For D, the valid commands include the 287 `ASK_OPERATOR` commands that
the bench sent, and the tokens include the 2,020 discarded samples (53 %
of the tokens of D).

**The `ASK_OPERATOR` of D.** Most of the trap successes of D come from the
bench, not from the model. Without the `ASK_OPERATOR` of the bench, D gets
38 of 180 over-limit tasks (21.1 %) and 5 of 180 wrong-status tasks
(2.8 %). That is 43 of 360 trap runs, against 41 for C. On the usual
tasks, 65 of the 205 false alarms of D are an `ASK_OPERATOR` of the bench.

**Paired counts** (the same task and seed; "first only" is the number of
runs where the first configuration succeeds and the second fails):

| First | Second | Tasks | First only | Second only | Both | Neither |
|---|---|---|---:|---:|---:|---:|
| A | B | usual | 6 | 27 | 120 | 387 |
| A | B | trap | 5 | 27 | 127 | 201 |
| B | C | usual | 0 | 24 | 147 | 369 |
| B | C | trap | 119 | 6 | 35 | 200 |
| C | D | usual | 7 | 28 | 164 | 341 |
| C | D | trap | 1 | 162 | 40 | 157 |

**The first command of the trap tasks** (360 runs for each configuration;
over-limit / wrong-status with a prefix ID / wrong-status with no prefix
ID):

| Configuration | `ASK_OPERATOR` | Shortened value | Wrong ID | Requested value or ID (refused) | Other |
|---|---|---|---|---|---|
| A | 27 / 1 / 4 | 15 / — / — | 11 / 14 / 4 | 53 / 62 / 64 | 74 / 13 / 18 |
| B | 31 / 1 / 2 | 15 / — / — | 10 / 14 / 4 | 54 / 63 / 67 | 70 / 12 / 17 |
| C | 31 / 1 / 9 | **65** / — / — | 16 / **77** / **58** | 0 / 0 / 0 | 68 / 12 / 23 |
| D | 72 / 61 / 69 | 15 / — / — | 4 / 10 / 1 | 0 / 0 / 0 | 89 / 19 / 20 |

"Other" is a limit value, another valid value, another command, `DONE`,
or no command. A shortened value is a shorter amount whose digits start
the requested amount: the request is 4,000 cents, the limit is 2,500, and
the command sends 400. The full table is in the tables file.

**The cost.** The mask costs about 0.05 ms a token, against a decode step
of about 17.7 ms: less than 1 % of a step. The grammar compile is about
1.5 ms a command. These times come from the quality runs, on a machine
that was in use. They are not a careful cost measurement.

## Cause and workaround

**The hard mask does not stop an error. It changes the error into another
valid command.** When the request cannot be done, the model starts to
write the command that the request asks for. Each prefix that the state
permits stays open, and the mask closes the rest:

- **The prefix trap.** The limit is 2,500 and the request is 4,000. The
  digits 4, 40, and 400 are all in the range, so the mask permits them.
  It blocks the fourth digit. In C, 65 of 180 over-limit runs (36.1 %)
  sent a shortened value, against 15 in A, B, and D. A shortened value
  can equal the amount ÷ 100, so the unit error also rises: C has 34 unit
  errors on the requests in cents, and A, B, and D have 0.
- **The ID trap.** The request names `p_17`, which is voided. `p_1`
  permits the refund. The model writes `p_1`, and the mask blocks the
  `7`. In C, 77 of 90 runs with a prefix ID (85.6 %) sent a command for
  the wrong payment. With no prefix ID, the mask forces another permitted
  ID: 58 of 90 runs (64.4 %).

In A, B, and D, the same requests give a command that the ledger refuses
(B: 54, 63, and 67 runs). The model then sees the refusal. In A and B, a
refused command also fails the task. In D, the check discards it and the
bench asks the operator. So the hard mask decreases the visible errors
and increases the silent ones.

**The workarounds** (C with one change; trap tasks for all four, usual
tasks for workarounds 3 and 4 also):

| Configuration | Over-limit | Wrong-status | Shortened values (over-limit) | Wrong IDs (wrong-status) | Usual tasks | False alarms (usual) |
|---|---:|---:|---:|---:|---:|---:|
| C | 31 / 180 | 10 / 180 | 65 | 135 | 171 / 540 | 138 |
| 1: the limit of each ID in the prompt | 38 / 180 | 8 / 180 | 39 | 119 | not run | not run |
| 3: an amount that is not in the request becomes `ASK_OPERATOR` | **166 / 180** | 80 / 180 | 0 | 82 | 154 / 540 | **353** |
| 4: a `requested` field with no limit before the amount | 94 / 180 | 68 / 180 | 18 | 78 | 144 / 540 | 265 |
| 5: each existing payment ID in the mask | 42 / 180 | 56 / 180 | 66 | 18 | not run | not run |

- **Workaround 1** changes little. The model gets the limit, but it still
  writes the requested amount, and the mask still shortens it.
- **Workaround 3** reads the amounts from the request text at run time:
  `$40` is dollars, and another number is cents. It does not read the
  goal of the task. It removes the prefix trap and the unit errors (0 of
  900 runs). But in 135 of its 166 over-limit successes, the bench sent
  the `ASK_OPERATOR`, not the model. On the usual tasks it costs a lot.
  Many correct commands have an amount that the request does not write,
  for example a capture of the full authorized amount. The false alarms
  rose from 138 to 353 of 540, and 225 of them came from the bench. Usual
  success fell from 171 to 154 (17 runs lost, 0 gained).
- **Workaround 4** gives the model a field for the requested amount. If
  that amount is above the limit, the grammar ends the object, and the
  bench sends `ASK_OPERATOR`. Over-limit success rises from 31 to 94 of
  180, and 64 of the 94 come from the bench. The false alarms rose to 265,
  and usual success fell to 144 (52 runs lost, 25 gained).
- **Workaround 5** removes the ID trap: 18 wrong IDs, against 135. The
  model writes the requested ID, and the ledger refuses it (130 runs).
  The valid-command rate falls to 62.5 %, so the guarantee of C is gone
  for the ID field. For the wrong-status tasks, its first commands are
  the same as those of B.
- **Workaround 2 is configuration D.**

No workaround makes C better than D on both task kinds. Workaround 3 is
best on the over-limit tasks, but it is worst on the usual tasks.

### Make the silent errors visible: the mask pressure

A first change (E1) adds a
measurement to the mask of C. At each token in an ID or an amount field,
the hook records the **blocked mass**: the part of the probability of the
model (the softmax at the sampler temperature, before the mask) that the
mask removes. When the model wants a value that the state does not
permit, this mass is high. A stop then ends the generation, and the bench
sends `ASK_OPERATOR` in place of the command that the mask would force.

The record run gave outcomes byte-identical to C
(`bench/runs/constrained/2026-11-25-42d231b-c-qwen2.5-1.5b-instruct-mass-*`).
The rule for the threshold was written before the data: the threshold of
the grid 0.1 to 0.99 with the largest value of (trap successes − false
alarms) on seed 1 only. It selected 0.99. A real stop run at 0.99
(`2026-11-25-61b8dcd-c-p0.99-*`) gave the same outputs and scores as the
replay of the record run in 900 of 900 runs.

| Configuration | Usual tasks | False alarms (from the bench) | Trap tasks (from the bench) | Shortened values | Wrong IDs | Unit errors |
|---|---:|---:|---:|---:|---:|---:|
| C | 171 / 540 | 138 (0) | 41 / 360 (0) | 65 | 151 | 223 |
| C + stop at 0.99 | 160 / 540 | 274 (143) | **237 / 360** (196) | **15** | **18** | 101 |
| D | 192 / 540 | 205 (65) | 202 / 360 (159) | 15 | 15 | 106 |

- The stop makes the silent errors of C visible. Shortened values fall
  from 65 to 15, and wrong IDs from 151 to 18. Trap success rises from 41
  to 237 of 360, which is more than the 202 of D.
- The cost is false alarms: 274 of 540, more than the 205 of D. The mask
  also blocks the model when the model is wrong in a usual task, and the
  blocked mass cannot tell the two cases apart. 125 of the 143 false
  alarms from the bench are in runs that C failed in another way. 11 are
  in runs that C completed.
- The seeds that the rule did not see agree with seed 1: trap success
  160 / 240 (seeds 2 and 3) against 77 / 120 (seed 1), false alarms
  189 / 360 against 85 / 180.
- The grid stopped at 0.99, and the false alarms fell up to its end. A
  higher threshold can be better. It was not tried, because it would be a
  choice after the data.

### Check the request before the model runs

A second change (E3) adds a check before the first command. A reader
finds the commands in the request text: a verb (`capture`, `refund`,
`void`, `authorize`, `transfer`), then its IDs and its amounts, in order.
The check executes these commands on a copy of the state. If the state
refuses one, the bench sends `ASK_OPERATOR` and the model does not run. If
the reader cannot link the IDs and the amounts to the commands, there is
no alarm, and the model runs as before. The check reads only the request
text and the state.

The reader could not read 37 of the 300 requests (12.3 %). Most of them
have no verb ("$149 p_5") or a verb after its amount ("$45 refund p_2").
The check refused 125 requests and ended 375 of 900 runs
(`bench/runs/constrained/2026-11-26-591cf7d-{c-r,d-r}-*`).

| Configuration | Usual tasks | Over-limit tasks | Wrong-status tasks | False alarms (usual) | Shortened values + wrong IDs (trap first commands) | Tokens for each task |
|---|---:|---:|---:|---:|---:|---:|
| C | 31.7 % [27.9, 35.7] | 17.2 % [12.4, 23.4] | 5.6 % [3.0, 9.9] | 138 / 540 | 216 | 65.9 |
| C + stop at 0.99 | 29.6 % [25.9, 33.6] | 54.4 % [47.2, 61.6] | 77.2 % [70.6, 82.7] | 274 / 540 | 33 | 39.5 |
| C + request check | 31.7 % [27.9, 35.7] | **88.9 %** [83.5, 92.7] | **90.6 %** [85.4, 94.0] | **178 / 540** | **9** | 37.2 |
| D | 35.6 % [31.6, 39.7] | 40.0 % [33.1, 47.3] | 72.2 % [65.3, 78.2] | 205 / 540 | 30 | 79.6 |
| D + request check | **35.6 %** [31.6, 39.7] | **89.4 %** [84.1, 93.1] | **94.4 %** [90.1, 97.0] | 207 / 540 | **3** | 40.1 |

- **The check does not change usual success.** C and C + check succeed on
  the same 171 usual runs, and D and D + check on the same 192 (paired: 0
  runs lost, 0 gained).
- **The check catches the traps that it can read.** In C, trap success
  rises from 41 to 323 of 360 (paired: 282 gained, 0 lost). The silent
  errors of the first command fall from 216 to 9. In D, the check takes
  the place of the fallback: 7 trap successes come from the fallback,
  against 159 in D, and the tokens fall from 79.6 to 40.1 for each task.
- **The cost is 40 false alarms in C.** The check ended 57 usual runs (19
  tasks × 3 seeds). C failed all 57 of them without the check. In these
  tasks the request names a command that the state does not permit yet,
  for example a refund from an account with a balance of 0. The shortest
  plan first moves the money with a transfer that the request does not
  name.
- **Against workaround 3,** which also reads the request: 17 usual runs
  more and 99 trap runs more succeed with the check, and 22 trap runs
  fewer. False alarms: 178 against 353.
- **C + check is the only method that meets the three aims of the changes:** each command is valid, it has fewer silent errors than C (9
  against 216), and it has fewer false alarms than D (178 against 205).
  D + check has the highest success (usual 192 / 540, traps 331 / 360),
  but the false alarms of D stay (207).

## What it means if you run models on your own hardware

A grammar mask that knows the state of your system can guarantee that a
small local model sends only commands that the system accepts. On this
ledger: 3,559 of 3,559 commands, for less than 1 % of a decode step. But
valid is not correct. Ask a 1.5B model for something that the system
cannot do, and it almost never says so. With a hard mask, the wrong
command goes through as a different valid one: a smaller refund, or a
refund of another payment.

So do not let the mask be the only check. Check the request against the
state with plain code before the model runs, and ask a person when the
state says no. Here, that raised the correct answers on impossible
requests from 11 % to 90 %, with no change on the usual requests. Keep the
mask for the requests that the code cannot read. Keep the check before
execution too. If you cannot read the requests, the next best signal is a
stop on the blocked probability of the mask. It costs many false alarms.

## Conclusion

On Qwen2.5-1.5B-Instruct on an Apple M4 Pro (Metal), a state-aware mask
changed the valid-command rate from 63.0 % to 100 % and task success on
usual tasks from 27.2 % to 31.7 %. On requests that the state cannot do,
it gave the correct answer (ask the operator) in 41 of 360 runs (11.4 %),
and it changed most of the others into a different valid command that
nothing reported. A stop when the mask blocks more than 99 % of the
probability raised that to 237 of 360, with 274 false alarms in 540 usual
runs. A check of the request text against the state, before the model
runs, raised it to 323 of 360 (89.7 %), with 178 false alarms and no
change in usual success. The mask cost less than 1 % of a decode step.
The limit that stays is that the model itself almost never asks: the
correct answers on impossible requests come from the code around it.

The three questions that decide the next step:

- **Is the state mask better than rejection for a small model, or the
  opposite?** Rejection is better on this model. D is above C on usual
  tasks (28 runs against 7, paired) and has fewer silent errors (15
  shortened values against 65, 15 wrong IDs against 151 on all trap
  tasks, 106 unit errors against 223). Its trap successes come mostly from its fallback, so on
  traps the model itself is no better with either method.
- **Is the compile time for each command small compared with one forward
  pass?** Yes, at the scale that the quality runs show: about 1.5 ms a
  command, against about 17.7 ms for one decode step. A careful cost
  measurement is not done.
- **Can a person write the `Rules` for a new system in one day without
  help?** Not measured. The ledger rules (`LedgerRules::params`) are about
  50 lines of Rust.

## Correctness

- **The guard and the grammar agree.** A test sends random logits through
  the grammar of the start state of 30 tasks. Each output parses, and the
  ledger permits each one. A second test checks 17 boundary cases (the
  limit, the limit + 1, an ID with the wrong status, a `from` equal to
  `to`).
- **The range pattern is exact.** A property test on 1,000 random ranges
  checks the boundaries, each power of 10, 1,000 random integers, and
  each integer up to 99,999 for the small ranges.
- **The check before execution stays.** In a separate fault test, the
  state changed after the model wrote the ID field, in 5 % of the
  generations. The grammar of that generation was from the old state.
  165 generations were selected and 107 got a change. The check stopped
  107 of 107. No other command was refused
  (`bench/runs/constrained/2026-11-26-591cf7d-c-qwen2.5-1.5b-instruct-fault-outcomes.jsonl`).
- **Seeded and repeatable.** Each run of B, C, and D in step 10 is
  byte-identical to the earlier run of the same configuration on other
  code. The table script gives the same counts as the summaries of the
  bench.
- **D sends no refused command.** In 900 runs, `execute` refused 0
  commands of D.

## Predictions and measurements

The predictions were written before the first run.

| Prediction | Measurement | Correct |
|---|---|---|
| A has 5 to 20 % parse failures on the 0.5B | The 0.5B failed the pilot. The 1.5B: 903 of 3,942 outputs (22.9 %) | No (other model) |
| B has 0 parse failures and 70 to 90 % valid commands | 0 parse failures. 63.0 % valid commands | Half |
| C has 100 % valid commands | 3,559 of 3,559 | Yes |
| On usual tasks, C is 5 to 15 points above B | 4.5 points (31.7 % against 27.2 %) | No, a little below |
| On over-limit tasks, C gives `ASK_OPERATOR` in less than 20 % of runs, and a shortened value in most of the others | `ASK_OPERATOR` in 17.2 %. A shortened value in 65 of the 149 others (43.6 %) | Half |
| On wrong-status tasks with a prefix ID, C selects the wrong payment in more than 50 % of runs | 85.6 % | Yes |
| D is above C on trap-task success, and uses 1.2 to 2 times the tokens | 56.1 % against 11.4 %, mostly from the fallback. 1.21 times the tokens | Yes |
| The mask costs less than 5 % of a 6.24 ms decode step | About 0.05 ms, less than 1 % of either step (machine in use) | Yes |
| The grammar compile for each command is less than 1 ms | About 1.5 ms (machine in use) | No |

**The changes** (predictions written on 24 November 2026, before the
first run of the changes):

| Prediction | Measurement | Correct |
|---|---|---|
| E1: shortened values fall to 10 to 30 | 15 | Yes |
| E1: wrong IDs fall to 40 to 90 | 18 | No, lower |
| E1: false alarms 140 to 190 | 274 | No |
| E1: over-limit success 90 to 130 / 180 | 98 / 180 | Yes |
| E1: wrong-status success 30 to 80 / 180 | 139 / 180 | No, higher |
| E1: usual success 150 to 171 / 540 | 160 / 540 | Yes |
| E3: the reader cannot read 5 % to 20 % of the requests | 37 / 300 = 12.3 % | Yes |
| E3: over-limit success of C 120 to 170 / 180 | 160 / 180 | Yes |
| E3: wrong-status success of C 100 to 170 / 180 | 163 / 180 | Yes |
| E3: false alarms of C 138 to 160 | 178 | No |
| E3: false alarms of D 205 to 225 | 207 | Yes |
| E3: fewer than 200 false alarms, against 353 for workaround 3 | 178 | Yes |

## What this does not prove

- One model (Qwen2.5-1.5B-Instruct, `f16`), one machine, one domain. A
  larger model can ask the operator more often, and then the cost of the
  hard mask is smaller.
- Synthetic requests, written by a 7B model from a template. Each request
  is clear and asks for one thing. Real requests are less clear.
- A simulated ledger, with integer amounts and short ID lists. A system
  with thousands of IDs makes the ID trap more frequent and the grammar
  larger.
- A state that is always true. The state source of the mask is the
  ledger itself. A stale state makes the mask wrong, and only the check
  before execution stops that. The fault test shows that the check does.
- Temperature 0.7 and three seeds. With 180 runs, an interval is about
  ±7 points, so a difference of less than about 10 points between two
  rows is not certain. Use the paired counts.
- The prompt is new for each command (no cache between commands), so the
  prefill time is the same in each configuration. A cache would change
  the cost of D, which discards samples.
- The cost. The times come from the quality runs, on a machine that was
  in use. They show the scale only. A cost measurement on a quiet machine
  is not done.
- The workarounds 1 and 5 ran on the trap tasks only. Their cost on the
  usual tasks is not measured.
- Workaround 3 compares only the amount, not the ID.
- The request reader of E3 was made on the 300 request texts of this
  test. I changed it one time, from the texts only (one verb with several
  amounts, and the forms `refunding` and `transfers`), before I looked at
  any result by task class. On new texts, it can read fewer requests. The
  requests come from a template, so they are more regular than real
  requests.
- The check of E3 sees only the commands that the request names. A request
  that needs a command it does not name (19 usual tasks here) gets a false
  alarm.
- The threshold of E1 is the top of its grid. A higher threshold was not
  tried.

## Reproduce

```sh
git checkout b4fa5c5
cargo build --release -p fornace-constrain --features metal --bin bench
M=models/qwen2.5-1.5b-instruct
for c in a b c d; do target/release/bench --config $c --model $M; done
target/release/bench --config c --workaround limits --ids 180-299 --model $M
target/release/bench --config c --workaround all-ids --ids 180-299 --model $M
target/release/bench --config c --workaround compare --model $M
target/release/bench --config c --workaround requested --model $M
target/release/bench --config c --fault --tag fault --model $M
cargo build --release -p fornace-constrain --features metal --bin names
for m in off scan bitmask; do target/release/names --mask $m --model $M; done
# The changes (E1 and E3):
target/release/bench --config c --mask-pressure --tag mass --model $M
target/release/bench --config c --pressure-stop 0.99 --mask-pressure --model $M
target/release/bench --config c --request-check --model $M
target/release/bench --config d --request-check --model $M
cargo run -p fornace-constrain --bin pressure -- bench/runs/constrained/<date>-<commit>-c-qwen2.5-1.5b-instruct-mass-outcomes.jsonl
cargo run -p fornace-constrain --bin tables -- <the outcomes files of the runs above, but the mass run and the fault test>
```

Expected: the bench prints "task success 258 of 900" (A), 301 (B), 212
(C), 394 (D), 46 of 360 (workaround 1), 98 of 360 (workaround 5), 400 of
900 (workaround 3), and 306 of 900 (workaround 4). The changes print 212 of 900 (the
record run), 397 of 900 (the stop at 0.99), 494 of 900 (C + check), and
523 of 900 (D + check). The fault test prints
"107 of 107 stopped by the check". The outcomes files are byte-identical
to the files of `27cf7f3`, because the seeds fix the samples.
