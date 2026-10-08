// SPDX-License-Identifier: AGPL-3.0-only
//! Pallas CipherSuite adapter for opaque-ke.
//!
//! Implements voprf::Group, voprf::CipherSuite, and KeGroup for Pallas curve,
//! enabling OPAQUE protocol to run on Pallas (native for Halo2 ZK circuits).
//!
//! Critical invariant: `hash_to_curve` here MUST match circuit Gadget C's
//! Poseidon-based hash-to-curve implementation exactly. Both use:
//! 1. Zero-pad the password to `MAX_PASSWORD_LEN`, pack → field elements (31 bytes/FE, little-endian)
//! 2. Poseidon hash chain → field element `u`
//! 3. Trial: x = u + offset (offset ∈ 0..256), find y where y² = x³ + 5
//! 4. Return first valid Pallas affine point

// The traits implemented here belong to voprf and opaque-ke, whose signatures
// name `generic_array::GenericArray`. generic-array 0.14 deprecated its own
// types in favour of hybrid-array; until those crates move, the deprecated type
// is the one the traits require, so it cannot be avoided at this boundary.
#![allow(deprecated)]

use core::ops::{Add, Mul, Sub};

use digest::core_api::BlockSizeUser;
use digest::{FixedOutput, HashMarker, OutputSizeUser};
use ff::{Field, FromUniformBytes, PrimeField};
use generic_array::GenericArray;
use generic_array::typenum::{IsLess, IsLessOrEqual, U32, U256, Unsigned};
use group::prime::PrimeCurveAffine;
use group::{Group as GroupTrait, GroupEncoding};
use pasta_curves::pallas;
use rand::{CryptoRng, RngCore};
use subtle::{Choice, ConstantTimeEq};
use zeroize::Zeroize;

use crate::circuit::gadget_c::hash_to_curve_outside;
use crate::types::MAX_PASSWORD_LEN;

// ─── Newtype Wrappers ───

/// Pallas curve point wrapper satisfying voprf::Group::Elem bounds.
#[derive(Clone, Copy, Debug)]
pub struct PallasElem(pub pallas::Point);

/// Pallas scalar wrapper satisfying voprf::Group::Scalar bounds.
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

// ─── expand_message_xmd (RFC 9380 §5.3.1) ───

/// Produces `len_in_bytes` of uniformly random bytes from `msg` and `dst`
/// using the hash function `H` in XMD mode.
fn expand_message_xmd<H>(msg: &[u8], dst: &[u8], len_in_bytes: usize) -> Vec<u8>
where
    H: BlockSizeUser + Default + FixedOutput + HashMarker,
{
    use digest::Digest;

    let b_in_bytes = <H as OutputSizeUser>::OutputSize::to_usize();
    let ell = len_in_bytes.div_ceil(b_in_bytes);
    let r_in_bytes = <H as BlockSizeUser>::block_size();
    let z_pad = vec![0u8; r_in_bytes];
    let l_i_b_str = [(len_in_bytes >> 8) as u8, (len_in_bytes & 0xff) as u8];
    let dst_len = [dst.len() as u8];

    // b_0 = H(Z_pad || msg || l_i_b_str || I2OSP(0, 1) || DST_prime)
    let mut h0 = H::default();
    Digest::update(&mut h0, &z_pad);
    Digest::update(&mut h0, msg);
    Digest::update(&mut h0, l_i_b_str);
    Digest::update(&mut h0, [0u8]);
    Digest::update(&mut h0, dst);
    Digest::update(&mut h0, dst_len);
    let b_0 = h0.finalize();

    // b_1 = H(b_0 || I2OSP(1, 1) || DST_prime)
    let mut h1 = H::default();
    Digest::update(&mut h1, b_0.as_ref());
    Digest::update(&mut h1, [1u8]);
    Digest::update(&mut h1, dst);
    Digest::update(&mut h1, dst_len);
    let mut b_vals = vec![h1.finalize()];

    // b_i = H(strxor(b_0, b_{i-1}) || I2OSP(i, 1) || DST_prime)
    for i in 2..=ell {
        let prev = &b_vals[i - 2];
        let xored: Vec<u8> = b_0.iter().zip(prev.iter()).map(|(a, b)| a ^ b).collect();
        let mut hi = H::default();
        Digest::update(&mut hi, &xored);
        Digest::update(&mut hi, [i as u8]);
        Digest::update(&mut hi, dst);
        Digest::update(&mut hi, dst_len);
        b_vals.push(hi.finalize());
    }

    let mut result = Vec::with_capacity(len_in_bytes);
    for b in &b_vals {
        result.extend_from_slice(b.as_ref());
    }
    result.truncate(len_in_bytes);
    result
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

// ─── voprf::Group ───

/// Marker type for Pallas group in VOPRF context.
pub struct PallasGroup;

impl voprf::Group for PallasGroup {
    type Elem = PallasElem;
    type ElemLen = U32;
    type Scalar = PallasScalar;
    type ScalarLen = U32;

    fn hash_to_curve<H>(
        input: &[&[u8]],
        _dst: &[&[u8]],
    ) -> voprf::Result<Self::Elem, voprf::InternalError>
    where
        H: BlockSizeUser + Default + FixedOutput + HashMarker,
        H::OutputSize: IsLess<U256> + IsLessOrEqual<H::BlockSize>,
    {
        // Poseidon-based hash-to-curve matching circuit Gadget C.
        // DST intentionally ignored — circuit uses Poseidon inherently,
        // and OPAQUE protocol provides its own domain separation.
        let bytes = concat_slices(input);
        if bytes.is_empty() {
            return Err(voprf::InternalError::Input);
        }
        let (affine_point, _u, _offset) = hash_to_curve_outside(&circuit_password(bytes));
        Ok(PallasElem(affine_point.to_curve()))
    }

    fn hash_to_scalar<H>(
        input: &[&[u8]],
        dst: &[&[u8]],
    ) -> voprf::Result<Self::Scalar, voprf::InternalError>
    where
        H: BlockSizeUser + Default + FixedOutput + HashMarker,
        H::OutputSize: IsLess<U256> + IsLessOrEqual<H::BlockSize>,
    {
        let msg = concat_slices(input);
        let dst_bytes = concat_slices(dst);
        if msg.is_empty() {
            return Err(voprf::InternalError::Input);
        }
        // RFC 9380 expand_message_xmd → 64 uniform bytes → reduce mod Fq.
        let uniform = expand_message_xmd::<H>(&msg, &dst_bytes, 64);
        let mut wide = [0u8; 64];
        wide.copy_from_slice(&uniform);
        Ok(PallasScalar(pallas::Scalar::from_uniform_bytes(&wide)))
    }

    fn base_elem() -> Self::Elem {
        PallasElem(pallas::Point::generator())
    }

    fn identity_elem() -> Self::Elem {
        PallasElem(pallas::Point::identity())
    }

    fn serialize_elem(elem: Self::Elem) -> GenericArray<u8, Self::ElemLen> {
        GenericArray::from(elem.0.to_bytes())
    }

    fn deserialize_elem(bytes: &[u8]) -> voprf::Result<Self::Elem> {
        if bytes.len() != 32 {
            return Err(voprf::Error::Deserialization);
        }
        let mut repr = [0u8; 32];
        repr.copy_from_slice(bytes);
        let point: pallas::Point =
            Option::from(pallas::Point::from_bytes(&repr)).ok_or(voprf::Error::Deserialization)?;
        // Reject identity element
        if bool::from(point.ct_eq(&pallas::Point::identity())) {
            return Err(voprf::Error::Deserialization);
        }
        Ok(PallasElem(point))
    }

    fn random_scalar<R: RngCore + CryptoRng>(rng: &mut R) -> Self::Scalar {
        PallasScalar(pallas::Scalar::random(rng))
    }

    fn invert_scalar(scalar: Self::Scalar) -> Self::Scalar {
        let inv: Option<pallas::Scalar> = scalar.0.invert().into();
        PallasScalar(inv.expect("invert_scalar called on zero"))
    }

    fn is_zero_scalar(scalar: Self::Scalar) -> Choice {
        scalar.0.ct_eq(&pallas::Scalar::ZERO)
    }

    fn serialize_scalar(scalar: Self::Scalar) -> GenericArray<u8, Self::ScalarLen> {
        GenericArray::from(scalar.0.to_repr())
    }

    fn deserialize_scalar(bytes: &[u8]) -> voprf::Result<Self::Scalar> {
        if bytes.len() != 32 {
            return Err(voprf::Error::Deserialization);
        }
        let mut repr = [0u8; 32];
        repr.copy_from_slice(bytes);
        let scalar: pallas::Scalar =
            Option::from(pallas::Scalar::from_repr(repr)).ok_or(voprf::Error::Deserialization)?;
        // Reject zero scalar
        if bool::from(scalar.ct_eq(&pallas::Scalar::ZERO)) {
            return Err(voprf::Error::Deserialization);
        }
        Ok(PallasScalar(scalar))
    }
}

// ─── voprf::CipherSuite ───

/// VOPRF CipherSuite: Pallas with Poseidon hash-to-curve and SHA-256 framing.
pub struct PallasVoprf;

impl voprf::CipherSuite for PallasVoprf {
    const ID: &'static str = "Pallas-Poseidon-SHA256";
    type Group = PallasGroup;
    type Hash = sha2::Sha256;
}

// ─── opaque_ke::key_exchange::group::Group ───

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
fn take_32(bytes: &mut &[u8]) -> Result<[u8; 32], opaque_ke::errors::ProtocolError> {
    if bytes.len() < 32 {
        return Err(opaque_ke::errors::ProtocolError::SerializationError);
    }
    let (head, rest) = bytes.split_at(32);
    let mut out = [0u8; 32];
    out.copy_from_slice(head);
    *bytes = rest;
    Ok(out)
}

impl opaque_ke::key_exchange::group::Group for PallasGroup {
    type Pk = PallasElem;
    type PkLen = U32;
    type Sk = PallasSecretKey;
    type SkLen = U32;

    fn serialize_pk(pk: &Self::Pk) -> GenericArray<u8, Self::PkLen> {
        GenericArray::from(pk.0.to_bytes())
    }

    fn deserialize_take_pk(
        bytes: &mut &[u8],
    ) -> Result<Self::Pk, opaque_ke::errors::ProtocolError> {
        let repr = take_32(bytes)?;
        let point = Option::from(pallas::Point::from_bytes(&repr))
            .ok_or(opaque_ke::errors::ProtocolError::SerializationError)?;
        Ok(PallasElem(point))
    }

    fn random_sk<R: RngCore + CryptoRng>(rng: &mut R) -> Self::Sk {
        PallasSecretKey(PallasScalar(pallas::Scalar::random(rng)))
    }

    fn derive_scalar(
        seed: GenericArray<u8, Self::SkLen>,
    ) -> Result<Self::Sk, opaque_ke::errors::InternalError> {
        let scalar = voprf::derive_key::<PallasVoprf>(
            &seed,
            &STR_DERIVE_DIFFIE_HELLMAN_KEY_PAIR,
            voprf::Mode::Oprf,
        )
        .map_err(opaque_ke::errors::InternalError::OprfError)?;
        Ok(PallasSecretKey(scalar))
    }

    fn public_key(sk: &Self::Sk) -> Self::Pk {
        PallasElem(pallas::Point::generator() * (sk.0).0)
    }

    fn serialize_sk(sk: &Self::Sk) -> GenericArray<u8, Self::SkLen> {
        GenericArray::from((sk.0).0.to_repr())
    }

    fn deserialize_take_sk(
        bytes: &mut &[u8],
    ) -> Result<Self::Sk, opaque_ke::errors::ProtocolError> {
        let repr = take_32(bytes)?;
        let scalar: pallas::Scalar = Option::from(pallas::Scalar::from_repr(repr))
            .ok_or(opaque_ke::errors::ProtocolError::SerializationError)?;
        // A zero scalar has the generator as its public key and no secret.
        if bool::from(scalar.ct_eq(&pallas::Scalar::ZERO)) {
            return Err(opaque_ke::errors::ProtocolError::SerializationError);
        }
        Ok(PallasSecretKey(PallasScalar(scalar)))
    }
}

impl opaque_ke::key_exchange::tripledh::DiffieHellman<PallasGroup> for PallasSecretKey {
    fn diffie_hellman(&self, pk: &PallasElem) -> GenericArray<u8, U32> {
        GenericArray::from((pk.0 * (self.0).0).to_bytes())
    }
}

// ─── opaque_ke::CipherSuite ───

/// Complete OPAQUE CipherSuite for Pallas curve with ZKPP support.
pub struct PallasCipherSuite;

impl opaque_ke::CipherSuite for PallasCipherSuite {
    type OprfCs = PallasVoprf;
    type KeyExchange = opaque_ke::key_exchange::tripledh::TripleDh<PallasGroup, sha2::Sha256>;
    type Ksf = argon2::Argon2<'static>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit::gadget_c::hash_to_curve_outside;

    #[test]
    fn test_hash_to_curve_consistency() {
        // The OPRF hash must equal the circuit's hash of the padded password
        // buffer, the one the prover commits to.
        let password = b"Str0ngP@ssword!";
        let mut padded = [0u8; MAX_PASSWORD_LEN];
        padded[..password.len()].copy_from_slice(password);
        let (native_point, _u, _offset) = hash_to_curve_outside(&padded);
        let native_projective = native_point.to_curve();

        let voprf_point = <PallasGroup as voprf::Group>::hash_to_curve::<sha2::Sha256>(
            &[password.as_ref()],
            &[b"test-dst"],
        )
        .unwrap();

        assert_eq!(native_projective, voprf_point.0);
    }

    #[test]
    fn test_hash_to_curve_different_inputs() {
        let p1 = <PallasGroup as voprf::Group>::hash_to_curve::<sha2::Sha256>(
            &[b"password1"],
            &[b"dst"],
        )
        .unwrap();
        let p2 = <PallasGroup as voprf::Group>::hash_to_curve::<sha2::Sha256>(
            &[b"password2"],
            &[b"dst"],
        )
        .unwrap();
        assert!(bool::from(!p1.ct_eq(&p2)));
    }

    #[test]
    fn test_hash_to_curve_ignores_dst() {
        // DST is intentionally ignored for circuit compatibility
        let p1 = <PallasGroup as voprf::Group>::hash_to_curve::<sha2::Sha256>(
            &[b"same_input"],
            &[b"dst1"],
        )
        .unwrap();
        let p2 = <PallasGroup as voprf::Group>::hash_to_curve::<sha2::Sha256>(
            &[b"same_input"],
            &[b"dst2"],
        )
        .unwrap();
        assert!(bool::from(p1.ct_eq(&p2)));
    }

    #[test]
    fn test_hash_to_scalar_deterministic() {
        let s1 =
            <PallasGroup as voprf::Group>::hash_to_scalar::<sha2::Sha256>(&[b"input"], &[b"dst"])
                .unwrap();
        let s2 =
            <PallasGroup as voprf::Group>::hash_to_scalar::<sha2::Sha256>(&[b"input"], &[b"dst"])
                .unwrap();
        assert!(bool::from(s1.ct_eq(&s2)));
    }

    #[test]
    fn test_hash_to_scalar_different_dst() {
        let s1 =
            <PallasGroup as voprf::Group>::hash_to_scalar::<sha2::Sha256>(&[b"input"], &[b"dst1"])
                .unwrap();
        let s2 =
            <PallasGroup as voprf::Group>::hash_to_scalar::<sha2::Sha256>(&[b"input"], &[b"dst2"])
                .unwrap();
        assert!(bool::from(!s1.ct_eq(&s2)));
    }

    #[test]
    fn test_elem_roundtrip() {
        let elem = <PallasGroup as voprf::Group>::base_elem();
        let bytes = <PallasGroup as voprf::Group>::serialize_elem(elem);
        let deserialized = <PallasGroup as voprf::Group>::deserialize_elem(&bytes).unwrap();
        assert!(bool::from(elem.ct_eq(&deserialized)));
    }

    #[test]
    fn test_scalar_roundtrip() {
        let mut rng = rand::thread_rng();
        let scalar = <PallasGroup as voprf::Group>::random_scalar(&mut rng);
        let bytes = <PallasGroup as voprf::Group>::serialize_scalar(scalar);
        let deserialized = <PallasGroup as voprf::Group>::deserialize_scalar(&bytes).unwrap();
        assert!(bool::from(scalar.ct_eq(&deserialized)));
    }

    #[test]
    fn test_identity_rejected_on_deserialize() {
        let identity = pallas::Point::identity();
        let bytes = identity.to_bytes();
        let result = <PallasGroup as voprf::Group>::deserialize_elem(&bytes);
        assert!(result.is_err());
    }

    #[test]
    fn test_zero_scalar_rejected_on_deserialize() {
        let zero = pallas::Scalar::ZERO;
        let bytes = zero.to_repr();
        let result = <PallasGroup as voprf::Group>::deserialize_scalar(&bytes);
        assert!(result.is_err());
    }

    #[test]
    fn test_scalar_invert() {
        let mut rng = rand::thread_rng();
        let s = <PallasGroup as voprf::Group>::random_scalar(&mut rng);
        let inv = <PallasGroup as voprf::Group>::invert_scalar(s);
        // s * inv should equal 1
        let product = s * &inv;
        let one = PallasScalar(pallas::Scalar::ONE);
        assert!(bool::from(product.ct_eq(&one)));
    }

    #[test]
    fn test_ke_public_key_deterministic() {
        use opaque_ke::key_exchange::group::Group;
        let sk = PallasSecretKey(PallasScalar(pallas::Scalar::from(42u64)));
        let pk1 = PallasGroup::public_key(&sk);
        let pk2 = PallasGroup::public_key(&sk);
        assert!(bool::from(pk1.ct_eq(&pk2)));
    }

    #[test]
    fn test_ke_diffie_hellman() {
        use opaque_ke::key_exchange::group::Group;
        use opaque_ke::key_exchange::tripledh::DiffieHellman;
        let mut rng = rand::thread_rng();
        let sk_a = PallasGroup::random_sk(&mut rng);
        let sk_b = PallasGroup::random_sk(&mut rng);
        let pk_a = PallasGroup::public_key(&sk_a);
        let pk_b = PallasGroup::public_key(&sk_b);
        // DH(pk_b, sk_a) == DH(pk_a, sk_b)
        assert_eq!(sk_a.diffie_hellman(&pk_b), sk_b.diffie_hellman(&pk_a));
    }

    #[test]
    fn test_ke_sk_roundtrip() {
        use opaque_ke::key_exchange::group::Group;
        let mut rng = rand::thread_rng();
        let sk = PallasGroup::random_sk(&mut rng);
        let bytes = PallasGroup::serialize_sk(&sk);
        let mut reader: &[u8] = &bytes;
        let deserialized = PallasGroup::deserialize_take_sk(&mut reader).unwrap();
        assert!(bool::from(sk.ct_eq(&deserialized)));
        assert!(reader.is_empty(), "the whole key was consumed");
    }

    /// Deserialization reads its own field and leaves the rest, because the
    /// caller reads the next field from where this one stopped.
    #[test]
    fn test_ke_sk_leaves_trailing_bytes() {
        use opaque_ke::key_exchange::group::Group;
        let mut rng = rand::thread_rng();
        let sk = PallasGroup::random_sk(&mut rng);
        let mut buf = PallasGroup::serialize_sk(&sk).to_vec();
        buf.extend_from_slice(b"next field");
        let mut reader: &[u8] = &buf;
        PallasGroup::deserialize_take_sk(&mut reader).unwrap();
        assert_eq!(reader, b"next field");
    }

    #[test]
    fn test_ke_sk_rejects_short_input() {
        use opaque_ke::key_exchange::group::Group;
        let mut reader: &[u8] = &[0u8; 31];
        assert!(PallasGroup::deserialize_take_sk(&mut reader).is_err());
    }

    /// A zero scalar is not a key: its public key is the identity and anyone
    /// can produce it.
    #[test]
    fn test_ke_sk_rejects_zero() {
        use opaque_ke::key_exchange::group::Group;
        let mut reader: &[u8] = &[0u8; 32];
        assert!(PallasGroup::deserialize_take_sk(&mut reader).is_err());
    }

    /// The same seed must always give the same key pair, or a server restart
    /// would invalidate every credential derived from it.
    #[test]
    fn test_ke_derive_scalar_is_deterministic() {
        use opaque_ke::key_exchange::group::Group;
        let seed = GenericArray::from([7u8; 32]);
        let a = PallasGroup::derive_scalar(seed).unwrap();
        let b = PallasGroup::derive_scalar(GenericArray::from([7u8; 32])).unwrap();
        assert!(bool::from(a.ct_eq(&b)));

        let other = PallasGroup::derive_scalar(GenericArray::from([8u8; 32])).unwrap();
        assert!(
            !bool::from(a.ct_eq(&other)),
            "a different seed, a different key"
        );
    }

    #[test]
    fn test_ke_pk_roundtrip() {
        use opaque_ke::key_exchange::group::Group;
        let mut rng = rand::thread_rng();
        let sk = PallasGroup::random_sk(&mut rng);
        let pk = PallasGroup::public_key(&sk);
        let bytes = PallasGroup::serialize_pk(&pk);
        let mut reader: &[u8] = &bytes;
        let deserialized = PallasGroup::deserialize_take_pk(&mut reader).unwrap();
        assert!(bool::from(pk.ct_eq(&deserialized)));
    }
}
