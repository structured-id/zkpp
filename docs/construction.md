# ZKPP registration relation: construction for review

This document states the relation `sid-pake-core` proves and verifies when a
password is installed (registration, password change, authorized reset), the
encodings it relies on, and the checks that happen outside the proof. It is
written for an independent cryptographic review: every statement below is
taken from the code in this repository, and the limits of what the relation
establishes are listed explicitly. It is not a claim that the composition has
been reviewed.

Notation: Pallas `E_p: y² = x³ + 5` over the base field `F_p`, scalar field
`F_q` (`p < q`), generator `G`. Poseidon is the P128Pow5T3 permutation
(`halo2_poseidon`, width 3, rate 2); `Poseidon(a, b)` is one hash with
`ConstantLength<2>` and `chain(x_1, ..., x_n) = Poseidon(...Poseidon(Poseidon(x_1, x_2), x_3)..., x_n)`
for `n ≥ 2` (`ConstantLength<1>` for a single element).

## Participants

| Party | Holds | Sees |
|---|---|---|
| Client (prover) | password `P`, OPRF blind `b`, history blind `r` | its own witnesses |
| Credential service | OPAQUE server setup, the pending operation | proof, public inputs, OPAQUE request and record |
| History evaluator | per-owner VOPRF key `k_j` per comparison domain | blinded request `B` |
| History checker | retained entries `s`, epoch KSF parameters and salts | `t_j` (transient), evaluator proofs |

The evaluator never receives `t_j` or retained entries; the checker never holds
`k_j`. A single-process deployment co-locates them; its compromise merges them.

## Encodings

- **Password.** `P` is a byte string of at most 128 bytes with no zero byte.
  It is zero-padded to 128 bytes and packed little-endian, 31 bytes per field
  element, giving `P̂ = (p_1, ..., p_5) ∈ F_p^5`. The policy gadget requires
  every active byte to be nonzero, every padding byte to be zero and the
  active bytes to form a prefix, so `P ↦ P̂` is injective on provable inputs.
- **Domain elements.** `domain_element(purpose, parts)` is `chain` over the
  packed bytes of `len(purpose) ‖ purpose ‖ len(part_1) ‖ part_1 ‖ ...`, every
  length a little-endian `u32`, so no two part lists collide.
  - owner domain `d = domain_element("SID-HISTORY-INPUT-v1", [installation id, owner id])`;
  - comparison domain `c_j = domain_element("SID-HISTORY-TAG-v1", [suite, epoch id, KSF memory ‖ passes ‖ lanes (u32 LE each), epoch salt])`.
- **Points** are compressed 32-byte Pallas encodings; the identity is refused
  wherever a point is decoded.

## Relation proved by the SNARK

Public instances, for `D` comparison domains (`5 + 4D` field elements):
`M.x, M.y, d, c_1..c_D, B.x, B.y, (Z_j.x, Z_j.y, t_j) for j = 1..D`.

Witness: `P`, `b ∈ F_p \ {0}` (used as an integer in `F_q`), `r ∈ F_p \ {0}`,
`H_M`, its offset, `H`, its offset and non-square witnesses, `N_1..N_D`.

1. **Policy (gadget A).** `P` has at least the policy's minimum length and
   class counts. The minimums are fixed columns of the proving and verifying
   keys: each policy is a different key, and no instance carries it.
2. **Breach (gadget D).** `chain(P̂)` is a non-member of the Bloom filter
   compiled into the circuit (`m = 256`, `k = 3` in this version).
3. **OPAQUE element (gadget C).** `u_P = chain(p_1, ..., p_5)`;
   `H_M ∈ E_p`, `H_M ≠ O`, `x(H_M) = u_P + o_M` with `0 ≤ o_M < 2^8`;
   `M = b·H_M`. `M` is the instance the verifier compares with the element of
   the client's OPAQUE registration request.
4. **History tags (gadget H).** `u = Poseidon(d, u_P)`;
   `H ∈ E_p`, `x(H) = u + o`, `o` the **minimal** offset in `[0, 2^8)` with
   `x³ + 5` a square: for every `i < o`, a witness `w_i ≠ 0` with
   `w_i² = g·((u+i)³ + 5)`, `g` the fixed non-square (the multiplicative
   generator of `F_p`); `y(H) = 2h` with `h < 2^253` (the even root);
   `r ≠ 0`; `B = r·H`; for each `j`: `Z_j = r·N_j`,
   `t_j = chain(c_j, u, x(N_j))`.

The gadgets share cells: the policy gadget's bytes are packed by constrained
Horner steps and copy-constrained to the field elements gadget C hashes;
gadget D hashes the same elements; gadget H takes gadget C's `u_P`. One `P`
therefore satisfies all four parts.

## Binding to the operation

The prover and verifier absorb `transcript_context(ctx)` into the Fiat-Shamir
transcript before any proof element, where

```
ctx = "SID_ZKPP_PASSWORD_OPERATION_v1" ‖ operation id (16 bytes) ‖ OPAQUE registration request
transcript_context(ctx) = from_uniform_bytes(SHA-512("SID_ZKPP_TRANSCRIPT_CONTEXT_v1" ‖ len(ctx) u64 LE ‖ ctx))
```

The operation id is fresh per operation; the server keeps the operation's
owner, purpose, expected credential and history revisions, policy and required
comparison domains in its sealed operation state, not in the transcript. Each
operation installs its password under its own OPRF credential identifier
(RFC 9807 §5), evaluated for exactly one request, the one in `ctx`.

## History evaluation (outside the SNARK)

- Client sends `B`; for each required domain the evaluator returns
  `Z_j = k_j·B` with a Chaum-Pedersen proof that `log_G(pk_j) = log_B(Z_j)`
  (RFC 9497 §2.2 for one element), challenge
  `hash_to_scalar(len(op) ‖ op ‖ pk ‖ B ‖ Z ‖ T2 ‖ T3; "SID-HISTORY-VOPRF-DLEQ-v1")`
  over SHA-256, bound to the operation id `op`.
- Before the SNARK the server compares the proof's claimed `d`, `B`, `c_j` and
  `Z_j` with what the operation holds. After it, the checker verifies each DLEQ
  proof against the epoch's stored public key. With `Z_j = k_j·B = k_j·r·H` and
  `Z_j = r·N_j` in the circuit (`r ≠ 0`, prime order), `N_j = k_j·H`, so
  `t_j = chain(c_j, u, x(k_j·H))` is a function of `P`, `d`, `c_j` and `k_j`
  only.
- **Checker.** For each domain, `s_j = Argon2id(t_j, salt_j, m, t, p)` with
  the epoch's immutable parameters, under a memory reservation taken before
  the KSF starts. `s_j` is compared in constant time with every retained entry
  of that domain; a match refuses the password. On acceptance the entry under
  the active epoch is written in the same transaction as the credential, its
  evidence and the operation result, against the history revision read at
  preparation. `t_j` and rejected `s_j` are never stored.

## One password, end to end

1. Client: `P`, fresh `b`, OPAQUE request `M' = b·HashToCurve(P)`.
2. Server: prepares the operation (id, owner, `d`, required `c_j` with `pk_j`).
3. Client: `B = r·H(u)`; evaluator: `Z_j`, DLEQ proofs; client verifies them.
4. Client: proof over `(P, b, r, H_M, H, N_j)` with `ctx`.
5. Server: lengths and encodings, claimed inputs equal to the operation's,
   SNARK under the key for the policy and `D`, `M = M'`.
6. Checker: DLEQ proofs, KSF per domain, comparison, accepted `s_j`.
7. Commit: credential, evidence (policy version and verifying-key identity),
   history entry and operation result, atomically.

## What the relation does not establish

- **The final OPAQUE record.** The proof covers the registration request `M`,
  not `Finalize`, the client KSF, the key schedule or the envelope. A client
  that deviates from the published SDK can upload a record that is not
  derived from `P`. That record is its own; it cannot borrow another
  operation's evaluation, because the credential's OPRF key evaluated only `M`.
- **Canonical `H_M`.** Gadget C does not require the minimal offset or a fixed
  sign of `y` for `H_M`. A non-standard choice changes `M` only, so the record
  is unusable for login; it does not affect the history tags, which use the
  canonical `H`.
- **Breach coverage.** The compiled Bloom filter is a fixed demonstration
  table; it is not evidence against a production breach corpus.
- **Server compromise.** A party that holds `k_j` and obtains `t_j` for some
  operation can test guesses for that password without the KSF. The split
  between evaluator and checker limits this to a compromise of both.
- **Post-quantum security.** Discrete-log assumptions on Pallas and Vesta.

## Questions for the reviewer

1. Knowledge soundness of the combined circuit under the stated shared cells,
   in particular that `u_P` used by gadget H is the hash of the policy-checked
   bytes for every satisfying assignment.
2. That the minimal-offset scan and the even-root constraint make `H` a
   function of `u`, and that `t_j` is therefore unique per `(P, d, c_j, k_j)`.
3. Composition of the IPA proof (with `ctx` absorbed first) and the per-domain
   DLEQ proofs bound to the operation id: replay of a proof, an evaluation or
   a tag across operations, owners or domains.
4. Zero knowledge of the instances with respect to `P` given `M`, `B`, `Z_j`
   and `t_j` to a party without `k_j`.
5. Whether `b` and `r` drawn from `F_p` (embedded in `F_q`) introduce a bias
   that matters for the OPRF or the VOPRF.
