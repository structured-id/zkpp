// SPDX-License-Identifier: AGPL-3.0-only
//! Pallas CipherSuite adapter for sid-opaque-ke.
//!
//! Implements `sid_voprf::Group`, `sid_voprf::CipherSuite` and the key-exchange
//! group for the Pallas curve, enabling the OPAQUE protocol to run on Pallas
//! (native for Halo2 ZK circuits).
//!
//! Critical invariant: `hash_to_curve` here MUST match circuit Gadget C's
//! Poseidon-based hash-to-curve implementation exactly. Both use:
//! 1. Zero-pad the password to `MAX_PASSWORD_LEN`, pack → field elements (31 bytes/FE, little-endian)
//! 2. Poseidon hash chain → field element `u`
//! 3. Trial: x = u + offset (offset ∈ 0..256), find y where y² = x³ + 5
//! 4. Return first valid Pallas affine point

use core::num::NonZeroU16;
use core::ops::{Add, Mul, Sub};

use digest::block_api::BlockSizeUser;
use digest::{FixedOutput, HashMarker};
use ff::{Field, FromUniformBytes, PrimeField};
use group::{CurveAffine as _, Group as GroupTrait, GroupEncoding};
use hash2curve::{ExpandMsg, ExpandMsgXmd, Expander};
use hybrid_array::Array;
use hybrid_array::typenum::{
    IsGreaterOrEqual, IsLess, IsLessOrEqual, Prod, True, U2, U16, U32, U256,
};
use pasta_curves::pallas;
use rand_core::{CryptoRng, TryCryptoRng, TryRng};
use subtle::{Choice, ConstantTimeEq};
use zeroize::Zeroize;

use crate::circuit::gadget_c::hash_to_curve_outside;
use crate::types::MAX_PASSWORD_LEN;

// ─── Newtype Wrappers ───

/// Pallas curve point wrapper satisfying `sid_voprf::Group::Elem` bounds.
#[derive(Clone, Copy, Debug)]
pub struct PallasElem(pub pallas::Point);

/// Pallas scalar wrapper satisfying `sid_voprf::Group::Scalar` bounds.
#[derive(Clone, Copy, Debug)]
pub struct PallasScalar(pub pallas::Scalar);

// ─── ConstantTimeEq ───

impl ConstantTimeEq for PallasElem {
    fn ct_eq(&self, other: &Self) -> Choice {
        self.0.ct_eq(&other.0)
    }
}

impl ConstantTimeEq for PallasScalar {
    fn ct_eq(&self, other: &Self) -> Choice {
        self.0.ct_eq(&other.0)
    }
}

// ─── Zeroize ───

impl Zeroize for PallasElem {
    fn zeroize(&mut self) {
        self.0 = pallas::Point::identity();
    }
}

impl Zeroize for PallasScalar {
    fn zeroize(&mut self) {
        self.0 = pallas::Scalar::ZERO;
    }
}

// ─── Arithmetic: PallasElem ───

impl Add<&PallasElem> for PallasElem {
    type Output = PallasElem;
    fn add(self, rhs: &PallasElem) -> PallasElem {
        PallasElem(self.0 + rhs.0)
    }
}

impl Mul<&PallasScalar> for PallasElem {
    type Output = PallasElem;
    fn mul(self, rhs: &PallasScalar) -> PallasElem {
        PallasElem(self.0 * rhs.0)
    }
}

// ─── Arithmetic: PallasScalar ───

impl Add<&PallasScalar> for PallasScalar {
    type Output = PallasScalar;
    fn add(self, rhs: &PallasScalar) -> PallasScalar {
        PallasScalar(self.0 + rhs.0)
    }
}

impl Mul<&PallasScalar> for PallasScalar {
    type Output = PallasScalar;
    fn mul(self, rhs: &PallasScalar) -> PallasScalar {
        PallasScalar(self.0 * rhs.0)
    }
}

impl Sub<&PallasScalar> for PallasScalar {
    type Output = PallasScalar;
    fn sub(self, rhs: &PallasScalar) -> PallasScalar {
        PallasScalar(self.0 - rhs.0)
    }
}

/// Concatenate multiple byte slices into one.
fn concat_slices(slices: &[&[u8]]) -> Vec<u8> {
    let total: usize = slices.iter().map(|s| s.len()).sum();
    let mut result = Vec::with_capacity(total);
    for s in slices {
        result.extend_from_slice(s);
    }
    result
}

/// The password as the circuit hashes it: zero-padded to `MAX_PASSWORD_LEN`.
/// The OPRF element must be `blind·H_p` of this same buffer, or no proof can
/// bind it. A longer password cannot be
/// proven and is hashed as given.
fn circuit_password(mut bytes: Vec<u8>) -> Vec<u8> {
    if bytes.len() < MAX_PASSWORD_LEN {
        bytes.resize(MAX_PASSWORD_LEN, 0);
    }
    bytes
}

// ─── sid_voprf::Group ───

/// Marker type for Pallas group in VOPRF context.
pub struct PallasGroup;

impl sid_voprf::Group for PallasGroup {
    type Elem = PallasElem;
    type ElemLen = U32;
    type Scalar = PallasScalar;
    type ScalarLen = U32;
    // Pallas targets the 128-bit security level.
    type SecurityLevel = U16;

    fn hash_to_curve<H>(
        input: &[&[u8]],
        _dst: &[&[u8]],
    ) -> sid_voprf::Result<Self::Elem, sid_voprf::InternalError>
    where
        H: BlockSizeUser + Default + FixedOutput + HashMarker,
        H::OutputSize: IsLess<U256, Output = True>
            + IsLessOrEqual<H::BlockSize, Output = True>
            + IsGreaterOrEqual<Prod<Self::SecurityLevel, U2>, Output = True>,
    {
        // Poseidon-based hash-to-curve matching circuit Gadget C.
        // DST intentionally ignored — circuit uses Poseidon inherently,
        // and OPAQUE protocol provides its own domain separation.
        let bytes = concat_slices(input);
        if bytes.is_empty() {
            return Err(sid_voprf::InternalError::Input);
        }
        let (affine_point, _u, _offset) = hash_to_curve_outside(&circuit_password(bytes));
        Ok(PallasElem(affine_point.to_curve()))
    }

    fn hash_to_scalar<H>(
        input: &[&[u8]],
        dst: &[&[u8]],
    ) -> sid_voprf::Result<Self::Scalar, sid_voprf::InternalError>
    where
        H: BlockSizeUser + Default + FixedOutput + HashMarker,
        H::OutputSize: IsLess<U256, Output = True>
            + IsLessOrEqual<H::BlockSize, Output = True>
            + IsGreaterOrEqual<Prod<Self::SecurityLevel, U2>, Output = True>,
    {
        const WIDE: NonZeroU16 = NonZeroU16::new(64).unwrap();
        if input.iter().all(|part| part.is_empty()) {
            return Err(sid_voprf::InternalError::Input);
        }
        // RFC 9380 section 5.3.1 expand_message_xmd to 64 uniform bytes,
        // reduced mod q (RFC 9380 section 5.2 hash_to_field with L = 64).
        let mut wide = [0u8; 64];
        <ExpandMsgXmd<H> as ExpandMsg<Self::SecurityLevel>>::expand_message(input, dst, WIDE)
            .map_err(|_| sid_voprf::InternalError::Input)?
            .fill_bytes(&mut wide)
            .map_err(|_| sid_voprf::InternalError::Input)?;
        Ok(PallasScalar(pallas::Scalar::from_uniform_bytes(&wide)))
    }

    fn base_elem() -> Self::Elem {
        PallasElem(pallas::Point::generator())
    }

    fn identity_elem() -> Self::Elem {
        PallasElem(pallas::Point::identity())
    }

    fn serialize_elem(elem: Self::Elem) -> Array<u8, Self::ElemLen> {
        Array::from(elem.0.to_bytes())
    }

    fn deserialize_elem(bytes: &[u8]) -> sid_voprf::Result<Self::Elem> {
        let repr: [u8; 32] = bytes
            .try_into()
            .map_err(|_| sid_voprf::Error::Deserialization)?;
        let point: pallas::Point = Option::from(pallas::Point::from_bytes(&repr))
            .ok_or(sid_voprf::Error::Deserialization)?;
        // Reject identity element
        if bool::from(point.ct_eq(&pallas::Point::identity())) {
            return Err(sid_voprf::Error::Deserialization);
        }
        Ok(PallasElem(point))
    }

    fn random_scalar<R: TryRng + TryCryptoRng>(rng: &mut R) -> sid_voprf::Result<Self::Scalar> {
        pallas::Scalar::try_random(rng)
            .map(PallasScalar)
            .map_err(|_| sid_voprf::Error::Rng)
    }

    fn invert_scalar(scalar: Self::Scalar) -> Self::Scalar {
        let inv: Option<pallas::Scalar> = scalar.0.invert().into();
        PallasScalar(inv.expect("invert_scalar called on zero"))
    }

    fn is_zero_scalar(scalar: Self::Scalar) -> Choice {
        scalar.0.ct_eq(&pallas::Scalar::ZERO)
    }

    fn serialize_scalar(scalar: Self::Scalar) -> Array<u8, Self::ScalarLen> {
        Array::from(scalar.0.to_repr())
    }

    fn deserialize_scalar(bytes: &[u8]) -> sid_voprf::Result<Self::Scalar> {
        let repr: [u8; 32] = bytes
            .try_into()
            .map_err(|_| sid_voprf::Error::Deserialization)?;
        let scalar: pallas::Scalar = Option::from(pallas::Scalar::from_repr(repr))
            .ok_or(sid_voprf::Error::Deserialization)?;
        // Reject zero scalar
        if bool::from(scalar.ct_eq(&pallas::Scalar::ZERO)) {
            return Err(sid_voprf::Error::Deserialization);
        }
        Ok(PallasScalar(scalar))
    }
}

// ─── sid_voprf::CipherSuite ───

/// VOPRF CipherSuite: Pallas with Poseidon hash-to-curve and SHA-256 framing.
pub struct PallasVoprf;

impl sid_voprf::CipherSuite for PallasVoprf {
    const ID: &'static str = "Pallas-Poseidon-SHA256";
    type Group = PallasGroup;
    type Hash = sha2::Sha256;
}

// ─── sid_opaque_ke::key_exchange::group::Group ───

/// The key-exchange secret key.
///
/// Distinct from [`PallasScalar`] on purpose: a VOPRF scalar has to be `Copy`,
/// and a value that can be copied cannot wipe itself when the last holder
/// drops it. The built-in curves make the same split (`Sk = SecretKey<C>`,
/// not the bare scalar).
#[derive(Clone, Debug)]
pub struct PallasSecretKey(pub PallasScalar);

impl Drop for PallasSecretKey {
    fn drop(&mut self) {
        self.0.zeroize();
    }
}

impl zeroize::ZeroizeOnDrop for PallasSecretKey {}

impl ConstantTimeEq for PallasSecretKey {
    fn ct_eq(&self, other: &Self) -> Choice {
        self.0.ct_eq(&other.0)
    }
}

/// Domain separator for deriving the key-exchange key pair, from the OPAQUE
/// specification (RFC 9807 §6.4.2, `DeriveDiffieHellmanKeyPair`). The built-in
/// groups derive with the same string; a different one would put this curve on
/// a different protocol.
const STR_DERIVE_DIFFIE_HELLMAN_KEY_PAIR: [u8; 33] = *b"OPAQUE-DeriveDiffieHellmanKeyPair";

/// Take the leading 32 bytes, leaving the rest for the next field.
fn take_32(bytes: &mut &[u8]) -> Result<[u8; 32], sid_opaque_ke::errors::ProtocolError> {
    let (head, rest) = bytes
        .split_first_chunk::<32>()
        .ok_or(sid_opaque_ke::errors::ProtocolError::SerializationError)?;
    *bytes = rest;
    Ok(*head)
}

impl sid_opaque_ke::key_exchange::group::Group for PallasGroup {
    type Pk = PallasElem;
    type PkLen = U32;
    type Sk = PallasSecretKey;
    type SkLen = U32;

    fn serialize_pk(pk: &Self::Pk) -> Array<u8, Self::PkLen> {
        Array::from(pk.0.to_bytes())
    }

    fn deserialize_take_pk(
        bytes: &mut &[u8],
    ) -> Result<Self::Pk, sid_opaque_ke::errors::ProtocolError> {
        let repr = take_32(bytes)?;
        let point = Option::from(pallas::Point::from_bytes(&repr))
            .ok_or(sid_opaque_ke::errors::ProtocolError::SerializationError)?;
        Ok(PallasElem(point))
    }

    fn random_sk<R: CryptoRng>(rng: &mut R) -> Self::Sk {
        PallasSecretKey(PallasScalar(pallas::Scalar::random(rng)))
    }

    fn derive_scalar(
        seed: Array<u8, Self::SkLen>,
    ) -> Result<Self::Sk, sid_opaque_ke::errors::InternalError> {
        let scalar = sid_voprf::derive_key::<PallasVoprf>(
            &seed,
            &STR_DERIVE_DIFFIE_HELLMAN_KEY_PAIR,
            sid_voprf::Mode::Oprf,
        )
        .map_err(sid_opaque_ke::errors::InternalError::OprfError)?;
        Ok(PallasSecretKey(scalar))
    }

    fn public_key(sk: &Self::Sk) -> Self::Pk {
        PallasElem(pallas::Point::generator() * (sk.0).0)
    }

    fn serialize_sk(sk: &Self::Sk) -> Array<u8, Self::SkLen> {
        Array::from((sk.0).0.to_repr())
    }

    fn deserialize_take_sk(
        bytes: &mut &[u8],
    ) -> Result<Self::Sk, sid_opaque_ke::errors::ProtocolError> {
        let repr = take_32(bytes)?;
        let scalar: pallas::Scalar = Option::from(pallas::Scalar::from_repr(repr))
            .ok_or(sid_opaque_ke::errors::ProtocolError::SerializationError)?;
        // A zero scalar has the generator as its public key and no secret.
        if bool::from(scalar.ct_eq(&pallas::Scalar::ZERO)) {
            return Err(sid_opaque_ke::errors::ProtocolError::SerializationError);
        }
        Ok(PallasSecretKey(PallasScalar(scalar)))
    }
}

impl sid_opaque_ke::key_exchange::tripledh::DiffieHellman<PallasGroup> for PallasSecretKey {
    fn diffie_hellman(&self, pk: &PallasElem) -> Array<u8, U32> {
        Array::from((pk.0 * (self.0).0).to_bytes())
    }
}

// ─── sid_opaque_ke::CipherSuite ───

/// Complete OPAQUE CipherSuite for Pallas curve with ZKPP support.
pub struct PallasCipherSuite;

impl sid_opaque_ke::CipherSuite for PallasCipherSuite {
    type OprfCs = PallasVoprf;
    type KeyExchange = sid_opaque_ke::key_exchange::tripledh::TripleDh<PallasGroup, sha2::Sha256>;
    type Ksf = argon2::Argon2<'static>;
}

#[cfg(test)]
mod tests;
