// Interop: the pure-TypeScript halo2 prover (@structured-id/zkpp) must
// reproduce the proof for this exact circuit and history witness byte-for-byte
// (zkpp/tests/zkpp-fullproof.test.ts writes /tmp/sid_zkpp_proof.txt). This test
// feeds those bytes through halo2 verify_proof: a TS-produced SNARK accepted by
// the Rust verifier.
use ff::Field;
use halo2_proofs::plonk::{SingleVerifier, verify_proof};
use halo2_proofs::transcript::{Blake2bRead, Challenge255};
use pasta_curves::{pallas, vesta};
use sid_pake_core::circuit::gadget_d::BloomFilter;
use sid_pake_core::circuit::gadget_h::HistoryTagWitness;
use sid_pake_core::circuit::{BREACH_PARAMS, CircuitShape, HistoryInputs, ZKPP_K, ZkppCircuit};
use sid_pake_core::history::{domain_element, history_input};
use sid_pake_core::keygen::{generate_params, generate_pk};
use sid_pake_core::types::{CE_DEFAULT_POLICY, MAX_PASSWORD_LEN};

#[test]
#[ignore = "needs /tmp/sid_zkpp_proof.txt written by the pure-TS prover test"]
fn interop_pure_ts_proof_passes_rust_verify() {
    let path = "/tmp/sid_zkpp_proof.txt";
    let content = std::fs::read_to_string(path)
        .expect("run the pure-TS prover test first: it writes /tmp/sid_zkpp_proof.txt");

    let shape = CircuitShape::single_domain(CE_DEFAULT_POLICY);
    let params = generate_params(ZKPP_K);
    let pk = generate_pk(&params, shape).unwrap();
    let vk = pk.get_vk().clone();

    let mut pw = [0u8; MAX_PASSWORD_LEN];
    pw[..10].copy_from_slice(b"Str0ngP@ss");
    // Fixed history vector: owner, domain, blind 11, history key 5.
    let d = domain_element(b"SID-HISTORY-INPUT-v1", &[b"interop", b"alice"]);
    let c = domain_element(b"SID-HISTORY-TAG-v1", &[b"epoch-0"]);
    let witness = HistoryTagWitness::honest(
        history_input(d, &pw),
        pallas::Base::from(11u64),
        &[pallas::Scalar::from(5u64)],
    );
    let circuit = ZkppCircuit {
        password: pw,
        password_len: 10,
        blind: pallas::Base::from(7u64),
        policy: CE_DEFAULT_POLICY,
        breach_filter: BloomFilter::new(BREACH_PARAMS),
        history_domains: 1,
        history: Some(HistoryInputs {
            d,
            domains: vec![c],
            witness,
        }),
    };
    assert!(!bool::from(pallas::Base::from(11u64).is_zero()));
    let instances = circuit.instance_values();

    let proof_hex = content.lines().nth(1).unwrap().trim();
    let proof_bytes: Vec<u8> = (0..proof_hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&proof_hex[i..i + 2], 16).unwrap())
        .collect();

    let mut transcript =
        Blake2bRead::<&[u8], vesta::Affine, Challenge255<vesta::Affine>>::init(&proof_bytes[..]);
    verify_proof(
        &params,
        &vk,
        SingleVerifier::new(&params),
        &[&[&instances]],
        &mut transcript,
    )
    .expect("pure-TS proof must pass Rust verify_proof (interop)");
}
