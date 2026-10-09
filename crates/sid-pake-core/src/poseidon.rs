// SPDX-License-Identifier: AGPL-3.0-only
//! Poseidon hash configuration for OPAQUE-ZKPP.
//!
//! Uses the P128Pow5T3 spec (width=3, rate=2) from halo2_gadgets, native to
//! the Pallas base field. The circuit and the native code hash the same way:
//! the password hash of the OPAQUE binder, the history input and tag, and the
//! domain-separation elements.

use halo2_gadgets::poseidon::primitives::{self as poseidon, ConstantLength};
use pasta_curves::pallas;

/// Poseidon spec: 128-bit security, Pow5 S-box, width 3 (t=3).
/// Standard Orchard/Zcash specification.
pub type PoseidonSpec = poseidon::P128Pow5T3;

/// Poseidon absorption rate (field elements per permutation).
pub const POSEIDON_RATE: usize = 2;

/// Poseidon state width.
pub const POSEIDON_WIDTH: usize = 3;

/// Poseidon over two field elements, as the circuit computes it.
pub fn poseidon_hash_2(inputs: &[pallas::Base; 2]) -> pallas::Base {
    poseidon::Hash::<_, PoseidonSpec, ConstantLength<2>, POSEIDON_WIDTH, POSEIDON_RATE>::init()
        .hash(*inputs)
}

/// Compute Poseidon hash of a single field element (for chaining).
pub fn poseidon_hash_1(inputs: &[pallas::Base; 1]) -> pallas::Base {
    poseidon::Hash::<_, PoseidonSpec, ConstantLength<1>, POSEIDON_WIDTH, POSEIDON_RATE>::init()
        .hash(*inputs)
}

/// Encode raw bytes into Pallas base field elements.
///
/// Each field element holds up to 31 bytes (248 bits, safely within
/// the ~254-bit Pallas base field). Bytes are packed little-endian.
/// Padding with zeros for the last chunk.
pub fn bytes_to_field_elements(bytes: &[u8]) -> Vec<pallas::Base> {
    use ff::PrimeField;

    const BYTES_PER_FE: usize = 31;
    let mut elements = Vec::new();

    for chunk in bytes.chunks(BYTES_PER_FE) {
        let mut repr = [0u8; 32];
        repr[..chunk.len()].copy_from_slice(chunk);
        // Little-endian encoding into field element
        elements.push(pallas::Base::from_repr(repr).unwrap());
    }

    elements
}

/// Iterative Poseidon hash over multiple field elements.
///
/// Hashes pairs: H(H(H(fe_0, fe_1), fe_2), fe_3) ...
/// For odd number of elements, the last one is paired with zero.
pub fn poseidon_hash_chain(elements: &[pallas::Base]) -> pallas::Base {
    assert!(!elements.is_empty(), "Cannot hash empty input");

    if elements.len() == 1 {
        return poseidon_hash_1(&[elements[0]]);
    }

    let mut acc = poseidon_hash_2(&[elements[0], elements[1]]);
    for element in &elements[2..] {
        acc = poseidon_hash_2(&[acc, *element]);
    }
    acc
}

/// The legacy history digest `LegacyHash(P, salt) = Poseidon(P || salt)` over
/// the zero-padded password. Only the input of a legacy comparison domain
/// during migration; never a new
/// history entry, since the digest with its salt tests guesses directly.
///
/// # Panics
/// If `password` is longer than `MAX_PASSWORD_LEN`, which the prover refuses.
pub fn legacy_history_digest(password: &[u8], salt: &[u8; 32]) -> pallas::Base {
    let mut padded = [0u8; crate::types::MAX_PASSWORD_LEN];
    padded[..password.len()].copy_from_slice(password);
    let mut password_fes = bytes_to_field_elements(&padded);
    let salt_fe = bytes_to_field_elements(salt);
    password_fes.extend_from_slice(&salt_fe);
    poseidon_hash_chain(&password_fes)
}

#[cfg(test)]
mod tests;
