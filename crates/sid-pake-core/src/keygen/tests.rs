use super::*;
use crate::circuit::ZKPP_K;
use crate::prover::ZkppProver;
use crate::test_support::{history_evaluation, oprf_element};
use crate::types::{CE_DEFAULT_POLICY, PolicyParams};
use crate::verifier::ZkppVerifier;
use ff::Field;
use pasta_curves::pallas;

/// A stricter policy than CE's, for custom-policy key tests.
const STRICT: PolicyParams = PolicyParams {
    min_length: 12,
    min_upper: 2,
    min_lower: 2,
    min_digit: 2,
    min_symbol: 1,
};

const CE: CircuitShape = CircuitShape::single_domain(CE_DEFAULT_POLICY);

fn keys(n: usize) -> Vec<pallas::Scalar> {
    (0..n).map(|_| pallas::Scalar::random(&mut rand::rng())).collect()
}

/// A witness-free copy of the circuit keeps its shape: keygen and reload lay
/// out the fixed minimums and the domain count from it.
#[test]
fn without_witnesses_keeps_the_shape() {
    use halo2_proofs::plonk::Circuit;
    let shape = CircuitShape {
        policy: STRICT,
        history_domains: 2,
    };
    assert_eq!(
        ZkppCircuit::for_shape(shape).without_witnesses().shape(),
        shape
    );
}

/// Keys for a custom policy prove only that policy: a proof under them
/// verifies under their verifying key, not under CE's, and a password meeting
/// CE but not the custom policy has no proof under them.
#[test]
fn custom_policy_keys_prove_only_their_policy() {
    let strict = CircuitShape::single_domain(STRICT);
    let params = generate_params(ZKPP_K);
    let strict_pk = generate_pk(&params, strict).unwrap();
    let strict_vk = generate_vk(&params, strict).unwrap();
    let ce_vk = generate_vk(&params, CE).unwrap();
    let prover = ZkppProver::new(params.clone(), strict_pk, strict);
    let ks = keys(1);

    let password = b"AAbb12!xyzqw";
    let blind = pallas::Scalar::random(&mut rand::rng());
    let bound = prover
        .prove(password, blind, b"ctx", &history_evaluation(password, &ks))
        .unwrap();
    let m = oprf_element(password, blind);
    ZkppVerifier::new(params.clone(), strict_vk, strict)
        .verify(&bound, b"ctx", m)
        .expect("the custom keys agree");
    assert!(
        ZkppVerifier::new(params, ce_vk, CE)
            .verify(&bound, b"ctx", m)
            .is_err(),
        "a custom-policy proof is not a CE-policy proof"
    );
    assert!(
        prover
            .prove(
                b"Str0ngPwd",
                blind,
                b"ctx",
                &history_evaluation(b"Str0ngPwd", &ks)
            )
            .is_err(),
        "a CE-compliant password below the custom policy has no proof"
    );
}

/// A proof carries the tags of its key's domain count: a one-domain proof is
/// no proof for a key that requires two, and the prover refuses evaluations
/// for a different count.
#[test]
fn a_proof_for_fewer_domains_is_refused() {
    let two = CircuitShape {
        policy: CE_DEFAULT_POLICY,
        history_domains: 2,
    };
    let params = generate_params(ZKPP_K);
    let one_pk = generate_pk(&params, CE).unwrap();
    let two_vk = generate_vk(&params, two).unwrap();
    let prover = ZkppProver::new(params.clone(), one_pk, CE);

    let password = b"Str0ngP@ssword!";
    let blind = pallas::Scalar::random(&mut rand::rng());
    let bound = prover
        .prove(
            password,
            blind,
            b"ctx",
            &history_evaluation(password, &keys(1)),
        )
        .unwrap();
    assert!(
        ZkppVerifier::new(params, two_vk, two)
            .verify(&bound, b"ctx", oprf_element(password, blind))
            .is_err()
    );
    assert!(matches!(
        prover.prove(
            password,
            blind,
            b"ctx",
            &history_evaluation(password, &keys(2))
        ),
        Err(crate::prover::ZkppProveError::HistoryShape { .. })
    ));
}

#[test]
fn test_keygen() {
    let params = generate_params(ZKPP_K);
    let _pk = generate_pk(&params, CE).unwrap();
}

/// Keygen with a progress callback reports the verifying key, then the
/// proving key, once each, and builds the same key as without it (the
/// transcript representative of the verifying key is equal).
#[test]
fn keygen_reports_its_steps_in_order() {
    let params = generate_params(ZKPP_K);
    let mut steps = Vec::new();
    let pk = generate_pk_with(&params, CE, |s| steps.push(s)).unwrap();
    assert_eq!(steps, [KeygenStep::VerifyingKey, KeygenStep::ProvingKey]);
    let plain = generate_pk(&params, CE).unwrap();
    assert_eq!(
        format!("{:?}", pk.get_vk().pinned()),
        format!("{:?}", plain.get_vk().pinned())
    );
}

/// Measures the keygen cost split: params-gen vs vk-gen vs pk-gen.
#[test]
fn test_keygen_split_timing() {
    use std::time::Instant;
    let t0 = Instant::now();
    let params = generate_params(ZKPP_K);
    let params_ms = t0.elapsed().as_millis();

    let circuit = ZkppCircuit::for_shape(CE);
    let t1 = Instant::now();
    let vk = halo2_proofs::plonk::keygen_vk(&params, &circuit).unwrap();
    let vk_ms = t1.elapsed().as_millis();

    let t2 = Instant::now();
    let _pk = halo2_proofs::plonk::keygen_pk(&params, vk, &circuit).unwrap();
    let pk_ms = t2.elapsed().as_millis();

    println!("KEYGEN SPLIT (k={ZKPP_K}): params={params_ms}ms  vk={vk_ms}ms  pk={pk_ms}ms");
}

#[test]
fn test_params_roundtrip() {
    let params = generate_params(4); // small k for fast test
    let mut buf = Vec::new();
    write_params(&params, &mut buf).unwrap();
    let params2 = read_params(&mut &buf[..]).unwrap();

    let mut buf2 = Vec::new();
    write_params(&params2, &mut buf2).unwrap();
    assert_eq!(buf, buf2);
}

#[test]
fn test_cache_roundtrip() {
    let dir = std::env::temp_dir().join("sid_zkpp_test_cache");
    let _ = std::fs::remove_dir_all(&dir);

    assert!(load_from_cache(&dir, CE).is_none());

    // Must use ZKPP_K — smaller k can't fit the circuit for VK generation
    let params = generate_params(ZKPP_K);
    save_params_cache(&dir, &params).unwrap();

    let (params2, _vk) = load_from_cache(&dir, CE).unwrap();
    let mut buf1 = Vec::new();
    let mut buf2 = Vec::new();
    write_params(&params, &mut buf1).unwrap();
    write_params(&params2, &mut buf2).unwrap();
    assert_eq!(buf1, buf2);

    let _ = std::fs::remove_dir_all(&dir);
}

/// End-to-end commit-and-prove bound registration: prove → verify against the
/// request's own element, with another element and a wrong context rejected.
#[test]
fn test_bound_registration_roundtrip() {
    use group::{Curve, Group};

    let params = generate_params(ZKPP_K);
    let pk = generate_pk(&params, CE).unwrap();
    let vk = pk.get_vk().clone();
    let prover = ZkppProver::new(params.clone(), pk, CE);
    let verifier = ZkppVerifier::new(params, vk, CE);

    let password = b"Str0ngP@ssword!";
    let blind = pallas::Scalar::random(&mut rand::rng());
    let ctx = b"sid-reg-nonce-1";
    let bound = prover
        .prove(
            password,
            blind,
            ctx,
            &history_evaluation(password, &keys(1)),
        )
        .unwrap();
    let m = oprf_element(password, blind);

    verifier
        .verify(&bound, ctx, m)
        .expect("bound proof must verify");

    let other_m = pallas::Point::random(&mut rand::rng()).to_affine();
    assert!(verifier.verify(&bound, ctx, other_m).is_err());

    assert!(verifier.verify(&bound, b"other-nonce", m).is_err());
}
