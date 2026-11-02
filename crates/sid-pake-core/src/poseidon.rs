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

/// The domain tag of the password hash: the ASCII bytes of
/// `SID-PALLAS-PASSWORD-v2` packed as one field element.
pub fn password_hash_tag() -> pallas::Base {
    bytes_to_field_elements(b"SID-PALLAS-PASSWORD-v2")[0]
}

/// The password as the hash absorbs it: zero-padded to `MAX_PASSWORD_LEN`
/// when it fits, so the circuit's fixed buffer packs alike, and as given
/// otherwise.
pub fn password_elements(password: &[u8]) -> Vec<pallas::Base> {
    use crate::types::MAX_PASSWORD_LEN;
    if password.len() <= MAX_PASSWORD_LEN {
        let mut padded = [0u8; MAX_PASSWORD_LEN];
        padded[..password.len()].copy_from_slice(password);
        bytes_to_field_elements(&padded)
    } else {
        bytes_to_field_elements(password)
    }
}

/// The password hash `p`: `Poseidon(TAG, L)` with `L` the password's byte
/// length, then each element of [`password_elements`] absorbed in turn. The
/// number of elements follows from `L`, so two passwords hash alike only by
/// a Poseidon collision: neither trailing zero bytes nor a shortened chain
/// make another password (RFC 9497 §4.6).
///
/// # Panics
/// On a length beyond `u64`, which no input reaches.
pub fn password_hash(password: &[u8]) -> pallas::Base {
    let length = u64::try_from(password.len()).expect("a length fits u64");
    password_elements(password).into_iter().fold(
        poseidon_hash_2(&[password_hash_tag(), pallas::Base::from(length)]),
        |acc, element| poseidon_hash_2(&[acc, element]),
    )
}

#[cfg(test)]
mod tests;
