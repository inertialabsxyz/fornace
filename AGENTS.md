# Instructions for agents

## Write in Simplified Technical English

Write all text in ASD-STE100 Simplified Technical English. This includes
documents, results, comments, and commit messages.

- Use short sentences: 20 words or fewer for instructions, 25 words or fewer
  for descriptions.
- Use the active voice.
- Use one word for one meaning.
- Give each instruction in a separate sentence.

## Check before you commit

Run `make check`. It must pass.

## Private folder

If `fornace-private/` exists, read `fornace-private/CLAUDE.md` first. It
holds the plan, the tracker, the post queue, and the daily commands. It is a
separate private repository and is ignored here. Never copy its content into
a public file, and never put its URL in a public file.

## Results

A result is a file in `results/` made from `results/TEMPLATE.md`. Its title
is the result in one sentence with a number, never an activity. Every number
has a commit hash and a command behind it. Say what the result does not
prove. Do not use the word "lab" in any public file.

## Commits

Conventional commits (`feat(cache): ...`, `fix(metrics): ...`,
`docs(results): ...`). One idea per commit. Never commit model weights,
`.idea/`, run output outside `bench/runs/`, or anything from
`fornace-private/`.
