// SPDX-License-Identifier: AGPL-3.0-only
//! Native side of the private password-history relation: the client derives
//! the tag `t` it proves,
//! the VOPRF evaluator answers the blinded request, the history checker runs
//! the KSF over `t` and compares.
//!
//! ```text
//! u = Poseidon(d, Poseidon(P))            d: history purpose, installation, owner
//! H = canonical_point(u)                  minimal offset, even y
//! B = r·H,  Z = k·B,  N = r⁻¹·Z = k·H
//! t = Poseidon(c, u, x(N))                c: comparison domain (epoch, format)
//! ```
//! The circuit proves both the minimal offset and the even `y`. `t` also uses
//! only the x-coordinate of `N` (`k·(−H) = −N` has the same x), so even a
//! wrong sign could not yield a second tag for one password.

use ff::{Field, PrimeField};
use group::{Curve, Group};
use pasta_curves::{arithmetic::CurveAffine, pallas};

use crate::poseidon::{bytes_to_field_elements, poseidon_hash_chain};
use crate::types::MAX_PASSWORD_LEN;

/// Hash-to-curve tries `2^HTC_TRY_BITS` offsets from `u`.
pub use crate::circuit::gadget_c::HTC_TRY_BITS;

/// A domain-separation element: Poseidon over `purpose` and each part, every
/// field length-prefixed so no two part lists encode to the same bytes.
pub fn domain_element(purpose: &[u8], parts: &[&[u8]]) -> pallas::Base {
    let mut bytes = Vec::new();
    for field in core::iter::once(purpose).chain(parts.iter().copied()) {
        let len = u32::try_from(field.len()).expect("domain field fits u32");
        bytes.extend_from_slice(&len.to_le_bytes());
        bytes.extend_from_slice(field);
    }
    poseidon_hash_chain(&bytes_to_field_elements(&bytes))
}

/// The password packed as the circuit packs it: zero-padded to
/// `MAX_PASSWORD_LEN`, 31 bytes per field element.
///
/// # Panics
/// If `password` is longer than `MAX_PASSWORD_LEN`, which the prover refuses.
pub fn password_elements(password: &[u8]) -> Vec<pallas::Base> {
    let mut padded = [0u8; MAX_PASSWORD_LEN];
    padded[..password.len()].copy_from_slice(password);
    bytes_to_field_elements(&padded)
}

/// `u = Poseidon(d, Poseidon(P))`: the history input for one owner domain.
/// The inner hash is the one the binder computes over the same padded
/// password, so the circuit reuses it.
pub fn history_input(d: pallas::Base, password: &[u8]) -> pallas::Base {
    let p = poseidon_hash_chain(&password_elements(password));
    crate::poseidon::poseidon_hash_2(&[d, p])
}

/// The first curve point at `x = u + offset`, `offset < 2^HTC_TRY_BITS`, with
/// the even `y`, and that offset. All offsets are tried, so the time does not
/// depend on `u`.
///
/// # Panics
/// If no offset gives a point (probability 2^-256).
pub fn canonical_point(u: pallas::Base) -> (pallas::Affine, u64) {
    use subtle::{Choice, ConditionallySelectable};
    let b = pallas::Base::from(5u64);
    let mut found = Choice::from(0u8);
    let mut sel_x = pallas::Base::ZERO;
    let mut sel_y = pallas::Base::ZERO;
    let mut sel_offset = 0u64;
    for i in 0u64..(1 << HTC_TRY_BITS) {
        let x = u + pallas::Base::from(i);
        let y = (x.square() * x + b).sqrt();
        let take = y.is_some() & !found;
        sel_x = pallas::Base::conditional_select(&sel_x, &x, take);
        sel_y = pallas::Base::conditional_select(&sel_y, &y.unwrap_or(pallas::Base::ZERO), take);
        sel_offset = u64::conditional_select(&sel_offset, &i, take);
        found |= y.is_some();
    }
    assert!(bool::from(found), "no curve point within the try range");
    // Prescribed sign: the even root (sgn0 of RFC 9380 §4.1).
    let odd = subtle::Choice::from(sel_y.to_repr()[0] & 1);
    let sel_y = pallas::Base::conditional_select(&sel_y, &-sel_y, odd);
    let point = Option::from(pallas::Affine::from_xy(sel_x, sel_y)).expect("on the curve");
    (point, sel_offset)
}

/// The blind as the scalar it acts as: a nonzero base-field element read as an
/// integer, which the scalar field (larger than the base field) holds unchanged.
pub fn blind_scalar(r: pallas::Base) -> pallas::Scalar {
    Option::from(pallas::Scalar::from_repr(r.to_repr())).expect("base fits the scalar field")
}

/// A fresh nonzero blind drawn from the base field.
pub fn random_blind(mut rng: impl rand_core::CryptoRng) -> pallas::Base {
    loop {
        let r = pallas::Base::random(&mut rng);
        if !bool::from(r.is_zero()) {
            return r;
        }
    }
}

/// Client request: `B = r·H` for the canonical point of `u`.
pub fn blind_request(u: pallas::Base, r: pallas::Base) -> pallas::Affine {
    let (h, _) = canonical_point(u);
    (pallas::Point::from(h) * blind_scalar(r)).to_affine()
}

/// Evaluator: `Z = k·B`. Refuses the identity, which no honest request is.
pub fn evaluate(k: pallas::Scalar, b: pallas::Affine) -> Option<pallas::Affine> {
    let z = pallas::Point::from(b) * k;
    (!bool::from(z.is_identity())).then(|| z.to_affine())
}

/// Client: unblinds `N = r⁻¹·Z` and derives `t = Poseidon(c, u, x(N))`.
pub fn finalize_tag(
    c: pallas::Base,
    u: pallas::Base,
    r: pallas::Base,
    z: pallas::Affine,
) -> pallas::Base {
    let inv = Option::<pallas::Scalar>::from(blind_scalar(r).invert()).expect("nonzero blind");
    let n = (pallas::Point::from(z) * inv).to_affine();
    let coords =
        Option::<pasta_curves::arithmetic::Coordinates<pallas::Affine>>::from(n.coordinates())
            .expect("N is not the identity");
    poseidon_hash_chain(&[c, u, *coords.x()])
}

/// Proof that `Z = k·B` for the `k` behind the public key `pk = k·G`
/// (Chaum-Pedersen, RFC 9497 §2.2 for a single element).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EvaluationProof {
    pub c: pallas::Scalar,
    pub s: pallas::Scalar,
}

/// Domain separator of the evaluation proof challenge.
const DLEQ_DST: &[u8] = b"SID-HISTORY-VOPRF-DLEQ-v1";

fn challenge(
    context: &[u8],
    pk: pallas::Affine,
    b: pallas::Affine,
    z: pallas::Affine,
    t2: pallas::Affine,
    t3: pallas::Affine,
) -> pallas::Scalar {
    use group::GroupEncoding;
    use sid_voprf::Group as _;
    let points = [pk, b, z, t2, t3].map(|p| p.to_bytes());
    let context_len = u32::try_from(context.len())
        .expect("context fits u32")
        .to_le_bytes();
    let input: [&[u8]; 7] = [
        &context_len,
        context,
        &points[0],
        &points[1],
        &points[2],
        &points[3],
        &points[4],
    ];
    crate::pallas_opaque::PallasGroup::hash_to_scalar::<sha2::Sha256>(&input, &[DLEQ_DST])
        .expect("nonempty input")
        .0
}

/// Evaluator: `Z = k·B` with its proof, bound to the operation `context`.
pub fn evaluate_with_proof(
    k: pallas::Scalar,
    b: pallas::Affine,
    context: &[u8],
    mut rng: impl rand_core::CryptoRng,
) -> Option<(pallas::Affine, EvaluationProof)> {
    let z = evaluate(k, b)?;
    let pk = (pallas::Point::generator() * k).to_affine();
    let nonce = pallas::Scalar::random(&mut rng);
    let t2 = (pallas::Point::generator() * nonce).to_affine();
    let t3 = (pallas::Point::from(b) * nonce).to_affine();
    let c = challenge(context, pk, b, z, t2, t3);
    Some((
        z,
        EvaluationProof {
            c,
            s: nonce - c * k,
        },
    ))
}

/// History checker: accepts `Z` only as the key `pk`'s evaluation of `B`.
pub fn verify_evaluation(
    pk: pallas::Affine,
    b: pallas::Affine,
    z: pallas::Affine,
    context: &[u8],
    proof: &EvaluationProof,
) -> bool {
    let t2 = (pallas::Point::generator() * proof.s + pallas::Point::from(pk) * proof.c).to_affine();
    let t3 = (pallas::Point::from(b) * proof.s + pallas::Point::from(z) * proof.c).to_affine();
    bool::from(subtle::ConstantTimeEq::ct_eq(
        &challenge(context, pk, b, z, t2, t3),
        &proof.c,
    ))
}

/// KSF parameters of one comparison domain; part of its immutable manifest.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KsfParams {
    /// Argon2id memory, KiB.
    pub memory_kib: u32,
    /// Argon2id passes.
    pub passes: u32,
    /// Argon2id lanes.
    pub lanes: u32,
}

/// A retained history entry: `s = KSF(t)` of a previously accepted password.
pub type HistoryEntry = [u8; 32];

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum HistoryError {
    #[error("password matches a retained history entry")]
    Reused,
    #[error("KSF parameters rejected")]
    KsfParams,
}

/// History checker: `s = Argon2id(t, epoch_salt)` under `params`.
pub fn ksf(
    t: pallas::Base,
    epoch_salt: &[u8],
    params: KsfParams,
) -> Result<HistoryEntry, HistoryError> {
    use argon2::{Algorithm, Argon2, Params, Version};
    let params = Params::new(params.memory_kib, params.passes, params.lanes, Some(32))
        .map_err(|_| HistoryError::KsfParams)?;
    let mut out = [0u8; 32];
    Argon2::new(Algorithm::Argon2id, Version::V0x13, params)
        .hash_password_into(&t.to_repr(), epoch_salt, &mut out)
        .map_err(|_| HistoryError::KsfParams)?;
    Ok(out)
}

/// History checker: one KSF for the candidate `t`, compared with every entry
/// retained in this comparison domain. Returns the entry to store when the
/// password is new. Every entry is compared, so the time does not reveal which
/// one matched.
pub fn check_candidate(
    t: pallas::Base,
    epoch_salt: &[u8],
    params: KsfParams,
    retained: &[HistoryEntry],
) -> Result<HistoryEntry, HistoryError> {
    use subtle::ConstantTimeEq;
    let s = ksf(t, epoch_salt, params)?;
    let reused = retained
        .iter()
        .fold(subtle::Choice::from(0u8), |acc, e| acc | e.ct_eq(&s));
    if bool::from(reused) {
        Err(HistoryError::Reused)
    } else {
        Ok(s)
    }
}

#[cfg(test)]
mod tests;
