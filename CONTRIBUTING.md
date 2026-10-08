# Contributing to zkpp

## Contributor License Agreement

Before a first pull request can be merged, you sign the Structured World Contributor License Agreement once, at <https://sw.foundation/cla>. It covers every repository of the organisation and takes a minute: sign in with GitHub, confirm your e-mail address, sign. The `CLA` status on your pull request then turns green by itself.

You keep the copyright in your contribution. If you contribute as part of your job, your employer may also need to sign the corporate agreement; the page above explains when.

## Development

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
# Proving takes seconds per test in optimized builds and minutes in debug.
cargo nextest run --workspace --release
cargo test --doc --workspace --release
```

A change to the circuit, transcript, gadgets, hash-to-curve, key stretching or wire format also changes the TypeScript client [`@structured-id/opaque`](https://github.com/structured-id/opaque): regenerate the conformance vectors with this crate's examples and update both together.

## Pull requests

- One concern per pull request.
- Tests for every behaviour change, including the failure paths.
- Conventional commit titles, for example `fix(verifier): refuse a truncated proof`.

## Security vulnerabilities

Do not report them in public issues; see [SECURITY.md](SECURITY.md).
