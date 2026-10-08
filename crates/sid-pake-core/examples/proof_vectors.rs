// SPDX-License-Identifier: AGPL-3.0-only
//! Cross-check vectors for the TypeScript prover: one registration proof of
//! the reference circuit (CE policy, one history domain) with every byte the
//! prover drew from its RNG recorded, the inputs it was built from, the
//! public instances and the proof bytes. Replaying the byte stream, the
//! TypeScript prover must write the same proof.
//!
//! `cargo run --release -p sid-pake-core --example proof_vectors -- <dir>`
//! writes `<dir>/proof-vector.json` and `<dir>/proof-rng.bin`.

use ff::{Field, PrimeField};
use group::GroupEncoding;
use halo2_proofs::{
    plonk::{SingleVerifier, create_proof, verify_proof},
    transcript::{Blake2bRead, Blake2bWrite, Challenge255, Transcript},
};
use pasta_curves::{arithmetic::CurveAffine, pallas, vesta};
use rand::{SeedableRng, rngs::StdRng};
use rand_core::{Rng, TryRng};
use std::convert::Infallible;
use sid_pake_core::binding::{operation_context, transcript_context};
use sid_pake_core::circuit::{
    BREACH_PARAMS, CircuitShape, HistoryInputs, ZKPP_K, ZkppCircuit, gadget_d::BloomFilter,
    gadget_h::HistoryTagWitness,
};
use sid_pake_core::history::{blind_request, domain_element, evaluate, history_input};
use sid_pake_core::keygen::{generate_params, generate_pk};
use sid_pake_core::types::{CE_DEFAULT_POLICY, MAX_PASSWORD_LEN};

struct Recording {
    inner: StdRng,
    drawn: Vec<u8>,
}

impl TryRng for Recording {
    type Error = Infallible;

    fn try_next_u32(&mut self) -> Result<u32, Infallible> {
        let mut b = [0u8; 4];
        self.try_fill_bytes(&mut b)?;
        Ok(u32::from_le_bytes(b))
    }
    fn try_next_u64(&mut self) -> Result<u64, Infallible> {
        let mut b = [0u8; 8];
        self.try_fill_bytes(&mut b)?;
        Ok(u64::from_le_bytes(b))
    }
    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), Infallible> {
        self.inner.fill_bytes(dest);
        self.drawn.extend_from_slice(dest);
        Ok(())
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn main() {
    let dir = std::path::PathBuf::from(
        std::env::args()
            .nth(1)
            .expect("usage: proof_vectors <output-dir>"),
    );
    std::fs::create_dir_all(&dir).expect("create the output directory");

    let shape = CircuitShape {
        policy: CE_DEFAULT_POLICY,
        history_domains: 1,
    };
    let params = generate_params(ZKPP_K);
    let pk = generate_pk(&params, shape).expect("keygen");

    let pw = b"Str0ngP@ssword!";
    let mut password = [0u8; MAX_PASSWORD_LEN];
    password[..pw.len()].copy_from_slice(pw);
    let mut seed = StdRng::seed_from_u64(5_000);
    let blind = pallas::Base::random(&mut seed);
    let d = domain_element(b"SID-HISTORY-INPUT-v1", &[b"test-installation", b"alice"]);
    let c = domain_element(b"SID-HISTORY-TAG-v1", &[b"epoch-1", b"format-1"]);
    let r = pallas::Base::random(&mut seed);
    let key = pallas::Scalar::random(&mut seed);
    let u = history_input(d, &password);
    let b = blind_request(u, r);
    let z = evaluate(key, b).expect("nonidentity");
    let circuit = ZkppCircuit {
        password,
        password_len: pw.len() as u32,
        blind,
        policy: shape.policy,
        breach_filter: BloomFilter::new(BREACH_PARAMS),
        history_domains: 1,
        history: Some(HistoryInputs {
            d,
            domains: vec![c],
            witness: HistoryTagWitness::from_evaluations(u, r, &[z]),
        }),
    };
    let instances = circuit.instance_values();
    let m = pallas::Affine::from_xy(instances[0], instances[1]).expect("M on the curve");
    let operation_id = [7u8; 16];
    let context = operation_context(&operation_id, m.to_bytes().as_ref());
    let context_scalar = transcript_context(&context);

    let mut rng = Recording {
        inner: StdRng::seed_from_u64(6_000),
        drawn: Vec::new(),
    };
    let mut transcript =
        Blake2bWrite::<Vec<u8>, vesta::Affine, Challenge255<vesta::Affine>>::init(vec![]);
    transcript
        .common_scalar(context_scalar)
        .expect("absorb context");
    create_proof(
        &params,
        &pk,
        &[circuit],
        &[&[&instances]],
        &mut rng,
        &mut transcript,
    )
    .expect("prove");
    let proof = transcript.finalize();

    let mut read = Blake2bRead::<_, vesta::Affine, Challenge255<vesta::Affine>>::init(&proof[..]);
    read.common_scalar(context_scalar).expect("absorb context");
    verify_proof(
        &params,
        pk.get_vk(),
        SingleVerifier::new(&params),
        &[&[&instances]],
        &mut read,
    )
    .expect("the reference proof verifies");

    let instance_hex: Vec<String> = instances
        .iter()
        .map(|i| format!("\"{}\"", hex(i.to_repr().as_ref())))
        .collect();
    let json = format!(
        "{{\"password\":\"{}\",\"blind\":\"{}\",\"d\":\"{}\",\"c\":\"{}\",\"r\":\"{}\",\
         \"z\":\"{}\",\"operationId\":\"{}\",\"context\":\"{}\",\"instances\":[{}],\"proof\":\"{}\"}}\n",
        hex(pw),
        hex(blind.to_repr().as_ref()),
        hex(d.to_repr().as_ref()),
        hex(c.to_repr().as_ref()),
        hex(r.to_repr().as_ref()),
        hex(z.to_bytes().as_ref()),
        hex(&operation_id),
        hex(context_scalar.to_repr().as_ref()),
        instance_hex.join(","),
        hex(&proof),
    );
    std::fs::write(dir.join("proof-vector.json"), json).expect("write the vector");
    std::fs::write(dir.join("proof-rng.bin"), &rng.drawn).expect("write the RNG stream");
}
