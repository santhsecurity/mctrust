# Contributing to mctrust

Thank you for contributing to `mctrust`.

## Development workflow

1. Fork the repository.
2. Create a feature branch from `main`.
3. Make your changes.
4. Add or update tests for any behavior change.
5. Run:
   - `cargo check`
   - `cargo test`
6. Open a pull request with a short description and rationale.

## Running locally

```bash
cargo test
cargo check
cargo fmt
```

## Quality checklist for PRs

- New logic includes tests.
- APIs keep public behavior backwards compatible unless migration is documented.
- Code uses descriptive names and clear comments only where needed.
- Long-running or randomized search defaults are deterministic in tests.
- Performance-sensitive code avoids accidental quadratic behavior.

## Project standards

- Enforce clean, idiomatic Rust.
- Keep `README.md` current with any public behavior change.
- Add changelog notes when behavior changes are user-visible.
- Prefer compile-time guarantees (types, traits) over runtime checks.

## License

All contributions are released under MIT and by submitting code you agree to
this project’s license.
