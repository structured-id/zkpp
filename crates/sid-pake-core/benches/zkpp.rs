// SPDX-License-Identifier: AGPL-3.0-only
//! Benchmarks for ZKPP circuit: keygen, prove, verify.
//!
//! Run: cargo bench -p sid-pake-core

use criterion::{Criterion, criterion_group, criterion_main};
use ff::{Field, PrimeField};
use group::GroupEncoding;
use pasta_curves::pallas;
use sid_opaque_ke::ClientRegistration;
use sid_pake_core::circuit::{CircuitShape, ZKPP_K};
use sid_pake_core::history::{
    blind_request, domain_element, evaluate, history_input, random_blind,
};
use sid_pake_core::keygen::{generate_params, generate_pk, generate_vk};
use sid_pake_core::pallas_opaque::PallasCipherSuite;
use sid_pake_core::prover::{HistoryEvaluation, ZkppProver};
use sid_pake_core::types::CE_DEFAULT_POLICY;
use sid_pake_core::verifier::ZkppVerifier;

const PASSWORD: &[u8] = b"Str0ngP@ssword!";

/// A client's OPAQUE registration request for `password`: its bytes, its OPRF
/// element `M` and the blind behind it (the state starts with the blind).
fn registration_request(password: &[u8]) -> (Vec<u8>, pallas::Affine, pallas::Scalar) {
    let start = ClientRegistration::<PallasCipherSuite>::start(&mut rand::rng(), password).unwrap();
    let bytes = start.message.serialize().to_vec();
    let m = pallas::Affine::from_bytes(bytes.as_slice().try_into().unwrap()).unwrap();
    let mut repr = <pallas::Scalar as PrimeField>::Repr::default();
    repr.copy_from_slice(&start.state.serialize()[..32]);
    let blind = pallas::Scalar::from_repr(repr).unwrap();
    (bytes, m, blind)
}

/// The evaluator's answers for `password` over `domains` keys.
fn history(password: &[u8], domains: usize) -> HistoryEvaluation {
    let d = domain_element(b"SID-HISTORY-INPUT-v1", &[b"bench", b"alice"]);
    let r = random_blind(rand::rng());
    let b = blind_request(history_input(d, password), r);
    HistoryEvaluation {
        d,
        domains: (0..domains)
            .map(|j| domain_element(b"SID-HISTORY-TAG-v1", &[&[j as u8]]))
            .collect(),
        r,
        evaluations: (0..domains)
            .map(|_| evaluate(pallas::Scalar::random(&mut rand::rng()), b).unwrap())
            .collect(),
    }
}

fn shape(domains: usize) -> CircuitShape {
    CircuitShape {
        policy: CE_DEFAULT_POLICY,
        history_domains: domains,
    }
}

fn bench_keygen(c: &mut Criterion) {
    let mut group = c.benchmark_group("zkpp_keygen");
    group.sample_size(10);

    group.bench_function("params_gen", |b| {
        b.iter(|| generate_params(ZKPP_K));
    });

    let params = generate_params(ZKPP_K);
    group.bench_function("pk_gen", |b| {
        b.iter(|| generate_pk(&params, shape(1)).unwrap());
    });

    group.bench_function("vk_gen", |b| {
        b.iter(|| generate_vk(&params, shape(1)).unwrap());
    });

    group.finish();
}

fn bench_prove(c: &mut Criterion) {
    let mut group = c.benchmark_group("zkpp_prove");
    group.sample_size(10);

    let params = generate_params(ZKPP_K);
    for domains in [1, 2] {
        let pk = generate_pk(&params, shape(domains)).unwrap();
        let prover = ZkppProver::new(params.clone(), pk, shape(domains));
        let (request, _, blind) = registration_request(PASSWORD);
        let eval = history(PASSWORD, domains);
        group.bench_function(format!("domains_{domains}"), |b| {
            b.iter(|| prover.prove(PASSWORD, blind, &request, &eval).unwrap());
        });
    }

    group.finish();
}

fn bench_verify(c: &mut Criterion) {
    let mut group = c.benchmark_group("zkpp_verify");
    group.sample_size(10);

    let params = generate_params(ZKPP_K);
    let pk = generate_pk(&params, shape(1)).unwrap();
    let vk = pk.get_vk().clone();

    let prover = ZkppProver::new(params.clone(), pk, shape(1));
    let (request, m, blind) = registration_request(PASSWORD);
    let bound = prover
        .prove(PASSWORD, blind, &request, &history(PASSWORD, 1))
        .unwrap();

    let verifier = ZkppVerifier::new(params, vk, shape(1));

    group.bench_function("single", |b| {
        b.iter(|| verifier.verify(&bound, &request, m).unwrap());
    });

    group.finish();
}

criterion_group!(benches, bench_keygen, bench_prove, bench_verify);
criterion_main!(benches);
