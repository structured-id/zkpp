use ff::{Field, PrimeField};
use group::{Group as GroupTrait, GroupEncoding};
use hybrid_array::Array;
use pasta_curves::pallas;
use sid_opaque_ke::key_exchange::group::Group as KeGroup;
use sid_opaque_ke::key_exchange::tripledh::DiffieHellman;
use sid_voprf::Group as _;
use subtle::ConstantTimeEq;

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

    let voprf_point =
        PallasGroup::hash_to_curve::<sha2::Sha256>(&[password.as_ref()], &[b"test-dst"]).unwrap();

    assert_eq!(native_projective, voprf_point.0);
}

#[test]
fn test_hash_to_curve_different_inputs() {
    let p1 = PallasGroup::hash_to_curve::<sha2::Sha256>(&[b"password1"], &[b"dst"]).unwrap();
    let p2 = PallasGroup::hash_to_curve::<sha2::Sha256>(&[b"password2"], &[b"dst"]).unwrap();
    assert!(bool::from(!p1.ct_eq(&p2)));
}

#[test]
fn test_hash_to_curve_ignores_dst() {
    // DST is intentionally ignored for circuit compatibility
    let p1 = PallasGroup::hash_to_curve::<sha2::Sha256>(&[b"same_input"], &[b"dst1"]).unwrap();
    let p2 = PallasGroup::hash_to_curve::<sha2::Sha256>(&[b"same_input"], &[b"dst2"]).unwrap();
    assert!(bool::from(p1.ct_eq(&p2)));
}

#[test]
fn test_hash_to_scalar_deterministic() {
    let s1 = PallasGroup::hash_to_scalar::<sha2::Sha256>(&[b"input"], &[b"dst"]).unwrap();
    let s2 = PallasGroup::hash_to_scalar::<sha2::Sha256>(&[b"input"], &[b"dst"]).unwrap();
    assert!(bool::from(s1.ct_eq(&s2)));
}

#[test]
fn test_hash_to_scalar_different_dst() {
    let s1 = PallasGroup::hash_to_scalar::<sha2::Sha256>(&[b"input"], &[b"dst1"]).unwrap();
    let s2 = PallasGroup::hash_to_scalar::<sha2::Sha256>(&[b"input"], &[b"dst2"]).unwrap();
    assert!(bool::from(!s1.ct_eq(&s2)));
}

/// An empty message has nothing to hash; RFC 9497 hashes only non-empty
/// transcripts, so an empty one is a caller error rather than a valid scalar.
#[test]
fn test_hash_to_scalar_rejects_empty_input() {
    assert!(PallasGroup::hash_to_scalar::<sha2::Sha256>(&[b""], &[b"dst"]).is_err());
}

/// `hash_to_scalar` is RFC 9380 `expand_message_xmd` to 64 bytes reduced
/// mod q. Pinned against the bytes the previous implementation produced, so
/// the TypeScript prover and stored derivations stay on the same scalar.
#[test]
fn test_hash_to_scalar_matches_expand_message_xmd() {
    use digest::Digest;

    // RFC 9380 section 5.3.1, written out for one 64-byte output.
    let msg = b"input";
    let dst = b"dst";
    let dst_prime = [dst.as_slice(), &[dst.len() as u8]].concat();
    let b0 = sha2::Sha256::new()
        .chain_update([0u8; 64])
        .chain_update(msg)
        .chain_update([0u8, 64])
        .chain_update([0u8])
        .chain_update(&dst_prime)
        .finalize();
    let b1 = sha2::Sha256::new()
        .chain_update(b0)
        .chain_update([1u8])
        .chain_update(&dst_prime)
        .finalize();
    let xored: Vec<u8> = b0.iter().zip(b1.iter()).map(|(a, b)| a ^ b).collect();
    let b2 = sha2::Sha256::new()
        .chain_update(&xored)
        .chain_update([2u8])
        .chain_update(&dst_prime)
        .finalize();
    let mut wide = [0u8; 64];
    wide[..32].copy_from_slice(&b1);
    wide[32..].copy_from_slice(&b2);
    let expected = <pallas::Scalar as ff::FromUniformBytes<64>>::from_uniform_bytes(&wide);

    let scalar = PallasGroup::hash_to_scalar::<sha2::Sha256>(&[msg], &[dst]).unwrap();
    assert_eq!(scalar.0, expected);
}

#[test]
fn test_elem_roundtrip() {
    let elem = PallasGroup::base_elem();
    let bytes = PallasGroup::serialize_elem(elem);
    let deserialized = PallasGroup::deserialize_elem(&bytes).unwrap();
    assert!(bool::from(elem.ct_eq(&deserialized)));
}

#[test]
fn test_scalar_roundtrip() {
    let mut rng = rand::rng();
    let scalar = PallasGroup::random_scalar(&mut rng).unwrap();
    let bytes = PallasGroup::serialize_scalar(scalar);
    let deserialized = PallasGroup::deserialize_scalar(&bytes).unwrap();
    assert!(bool::from(scalar.ct_eq(&deserialized)));
}

#[test]
fn test_identity_rejected_on_deserialize() {
    let identity = pallas::Point::identity();
    let bytes = identity.to_bytes();
    let result = PallasGroup::deserialize_elem(&bytes);
    assert!(result.is_err());
}

/// A point encoding of the wrong length is rejected, not truncated or padded.
#[test]
fn test_elem_rejects_wrong_length() {
    let bytes = PallasGroup::serialize_elem(PallasGroup::base_elem());
    assert!(PallasGroup::deserialize_elem(&bytes[..31]).is_err());
}

#[test]
fn test_zero_scalar_rejected_on_deserialize() {
    let zero = pallas::Scalar::ZERO;
    let bytes = zero.to_repr();
    let result = PallasGroup::deserialize_scalar(&bytes);
    assert!(result.is_err());
}

#[test]
fn test_scalar_invert() {
    let mut rng = rand::rng();
    let s = PallasGroup::random_scalar(&mut rng).unwrap();
    let inv = PallasGroup::invert_scalar(s);
    // s * inv should equal 1
    let product = s * &inv;
    let one = PallasScalar(pallas::Scalar::ONE);
    assert!(bool::from(product.ct_eq(&one)));
}

#[test]
fn test_ke_public_key_deterministic() {
    let sk = PallasSecretKey(PallasScalar(pallas::Scalar::from(42u64)));
    let pk1 = <PallasGroup as KeGroup>::public_key(&sk);
    let pk2 = <PallasGroup as KeGroup>::public_key(&sk);
    assert!(bool::from(pk1.ct_eq(&pk2)));
}

#[test]
fn test_ke_diffie_hellman() {
    let mut rng = rand::rng();
    let sk_a = <PallasGroup as KeGroup>::random_sk(&mut rng);
    let sk_b = <PallasGroup as KeGroup>::random_sk(&mut rng);
    let pk_a = <PallasGroup as KeGroup>::public_key(&sk_a);
    let pk_b = <PallasGroup as KeGroup>::public_key(&sk_b);
    // DH(pk_b, sk_a) == DH(pk_a, sk_b)
    assert_eq!(sk_a.diffie_hellman(&pk_b), sk_b.diffie_hellman(&pk_a));
}

#[test]
fn test_ke_sk_roundtrip() {
    let mut rng = rand::rng();
    let sk = <PallasGroup as KeGroup>::random_sk(&mut rng);
    let bytes = <PallasGroup as KeGroup>::serialize_sk(&sk);
    let mut reader: &[u8] = &bytes;
    let deserialized = <PallasGroup as KeGroup>::deserialize_take_sk(&mut reader).unwrap();
    assert!(bool::from(sk.ct_eq(&deserialized)));
    assert!(reader.is_empty(), "the whole key was consumed");
}

/// Deserialization reads its own field and leaves the rest, because the
/// caller reads the next field from where this one stopped.
#[test]
fn test_ke_sk_leaves_trailing_bytes() {
    let mut rng = rand::rng();
    let sk = <PallasGroup as KeGroup>::random_sk(&mut rng);
    let mut buf = <PallasGroup as KeGroup>::serialize_sk(&sk).to_vec();
    buf.extend_from_slice(b"next field");
    let mut reader: &[u8] = &buf;
    <PallasGroup as KeGroup>::deserialize_take_sk(&mut reader).unwrap();
    assert_eq!(reader, b"next field");
}

#[test]
fn test_ke_sk_rejects_short_input() {
    let mut reader: &[u8] = &[0u8; 31];
    assert!(<PallasGroup as KeGroup>::deserialize_take_sk(&mut reader).is_err());
}

/// A zero scalar is not a key: its public key is the identity and anyone
/// can produce it.
#[test]
fn test_ke_sk_rejects_zero() {
    let mut reader: &[u8] = &[0u8; 32];
    assert!(<PallasGroup as KeGroup>::deserialize_take_sk(&mut reader).is_err());
}

/// The same seed must always give the same key pair, or a server restart
/// would invalidate every credential derived from it.
#[test]
fn test_ke_derive_scalar_is_deterministic() {
    let a = <PallasGroup as KeGroup>::derive_scalar(Array::from([7u8; 32])).unwrap();
    let b = <PallasGroup as KeGroup>::derive_scalar(Array::from([7u8; 32])).unwrap();
    assert!(bool::from(a.ct_eq(&b)));

    let other = <PallasGroup as KeGroup>::derive_scalar(Array::from([8u8; 32])).unwrap();
    assert!(
        !bool::from(a.ct_eq(&other)),
        "a different seed, a different key"
    );
}

#[test]
fn test_ke_pk_rejects_the_identity() {
    // RFC 9807 §6.4.1: DeserializeElement fails on the identity; a peer's
    // key-exchange key is decoded the same way, or TripleDH multiplies by it.
    let bytes = pallas::Point::identity().to_bytes();
    let mut reader: &[u8] = &bytes;
    assert!(<PallasGroup as KeGroup>::deserialize_take_pk(&mut reader).is_err());
}

#[test]
fn test_ke_pk_roundtrip() {
    let mut rng = rand::rng();
    let sk = <PallasGroup as KeGroup>::random_sk(&mut rng);
    let pk = <PallasGroup as KeGroup>::public_key(&sk);
    let bytes = <PallasGroup as KeGroup>::serialize_pk(&pk);
    let mut reader: &[u8] = &bytes;
    let deserialized = <PallasGroup as KeGroup>::deserialize_take_pk(&mut reader).unwrap();
    assert!(bool::from(pk.ct_eq(&deserialized)));
}
