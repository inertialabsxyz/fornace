# Contributing to Fornace

## Results

If a benchmark run shows something new, write a result in `results/`. Use
[results/TEMPLATE.md](results/TEMPLATE.md).

## Do this check before you commit

```sh
make check
```

This runs `cargo fmt --check`, clippy, and the tests. It must pass. CI does
the same check.

## Commit messages

Use this format: `type(scope): short description`. The types are `feat`,
`fix`, `refactor`, `test`, `docs`, and `chore`.

## Licence

When you send a contribution, you agree to licence it under the MIT licence.
