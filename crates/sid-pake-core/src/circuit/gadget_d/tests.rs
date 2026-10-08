// SPDX-License-Identifier: AGPL-3.0-only
//! Correctness tests for Gadget D (breach Bloom non-membership).
//!
//! Focus: SOUNDNESS (a member cannot prove non-membership) and COMPLETENESS
//! (a non-member can), plus the regression for the original "all bits == 0" bug.

use super::*;
use crate::poseidon::poseidon_hash_2;
use halo2_proofs::{circuit::SimpleFloorPlanner, dev::MockProver, plonk::Circuit};

const TEST_PARAMS: BloomParams = BloomParams {
    index_bits: 8, // m = 256
    k: 3,
};
const K_CIRCUIT: u32 = 12;

#[derive(Clone)]
struct TestCircuit {
    filter: BloomFilter,
    inputs: [pallas::Base; 2],
}

impl Circuit<pallas::Base> for TestCircuit {
    type Config = BreachBloomConfig;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        self.clone()
    }

    fn configure(meta: &mut ConstraintSystem<pallas::Base>) -> Self::Config {
        BreachBloomChip::configure(meta, TEST_PARAMS)
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<pallas::Base>,
    ) -> Result<(), Error> {
        let col = config.poseidon_input;
        let in0 = Value::known(self.inputs[0]);
        let in1 = Value::known(self.inputs[1]);
        let cells = layouter.assign_region(
            || "test_inputs",
            |mut region| {
                let c0 = region.assign_advice(|| "i0", col, 0, || in0)?;
                let c1 = region.assign_advice(|| "i1", col, 1, || in1)?;
                Ok(vec![c0, c1])
            },
        )?;
        let chip = BreachBloomChip::construct(config);
        chip.synthesize(&mut layouter, &self.filter, &cells)
    }
}

/// Hash matching the in-circuit Poseidon chain for exactly 2 inputs.
fn hash_of(inputs: &[pallas::Base; 2]) -> pallas::Base {
    poseidon_hash_2(inputs)
}

fn populated_filter(seeds: &[u64]) -> BloomFilter {
    let mut f = BloomFilter::new(TEST_PARAMS);
    for &s in seeds {
        let h = hash_of(&[pallas::Base::from(s), pallas::Base::from(s + 1)]);
        f.insert_hash(h);
    }
    f
}

/// COMPLETENESS: a non-member password produces a satisfying proof.
#[test]
fn nonmember_accepted() {
    let filter = populated_filter(&[7, 11, 13, 42, 99, 123, 250]);
    let inputs = [
        pallas::Base::from(1_000_000u64),
        pallas::Base::from(2_000_000u64),
    ];
    let h = hash_of(&inputs);
    assert!(
        !filter.maybe_contains_hash(h),
        "test input is accidentally a member — choose different inputs"
    );

    let circuit = TestCircuit { filter, inputs };
    MockProver::run(K_CIRCUIT, &circuit, vec![])
        .unwrap()
        .assert_satisfied();
}

/// SOUNDNESS (the key test): a MEMBER cannot prove non-membership.
/// The member's k bits are all 1 → product 1 → the `prod == 0` gate fails.
#[test]
fn member_rejected() {
    let mut filter = populated_filter(&[7, 11, 13]);
    let inputs = [pallas::Base::from(555u64), pallas::Base::from(777u64)];
    let h = hash_of(&inputs);
    filter.insert_hash(h); // make it a member
    assert!(filter.maybe_contains_hash(h));

    let circuit = TestCircuit { filter, inputs };
    let prover = MockProver::run(K_CIRCUIT, &circuit, vec![]).unwrap();
    assert!(
        prover.verify().is_err(),
        "a member MUST NOT be able to prove non-membership"
    );
}

/// REGRESSION for the original "all bits == 0" bug: a non-member whose k
/// positions include SET bits (the common case at realistic fill) must still be
/// ACCEPTED. The old predicate (∀ bit == 0) would wrongly REJECT this.
#[test]
fn nonmember_with_set_bits_accepted() {
    let mut filter = BloomFilter::new(TEST_PARAMS);
    let inputs = [pallas::Base::from(123u64), pallas::Base::from(456u64)];
    let h = hash_of(&inputs);

    let idxs = BloomFilter::indices(TEST_PARAMS, h);
    let mut distinct: Vec<u64> = idxs.clone();
    distinct.sort_unstable();
    distinct.dedup();
    assert!(
        distinct.len() >= 2,
        "need >= 2 distinct indices for this regression; choose other inputs"
    );
    // Set all-but-one distinct position to 1 → non-member (one position stays 0),
    // but NOT all-zero, so the old all-zero predicate would have rejected it.
    for &d in &distinct[..distinct.len() - 1] {
        filter.bits[d as usize] = 1;
    }
    assert!(
        !filter.maybe_contains_hash(h),
        "setup error: should be a non-member"
    );
    let set_count = idxs
        .iter()
        .filter(|&&i| filter.bits[i as usize] == 1)
        .count();
    assert!(
        set_count >= 1,
        "regression precondition: at least one queried bit must be set"
    );

    let circuit = TestCircuit { filter, inputs };
    MockProver::run(K_CIRCUIT, &circuit, vec![])
        .unwrap()
        .assert_satisfied();
}

/// Off-circuit/in-circuit index derivation must agree (consistency).
#[test]
fn index_derivation_in_range() {
    let inputs = [pallas::Base::from(31337u64), pallas::Base::from(42u64)];
    let h = hash_of(&inputs);
    let idxs = BloomFilter::indices(TEST_PARAMS, h);
    assert_eq!(idxs.len(), TEST_PARAMS.k);
    for idx in idxs {
        assert!(idx < TEST_PARAMS.m(), "index out of range");
    }
}

/// SOUNDNESS (safe direction): a non-member whose k positions all happen to be
/// set (a Bloom false positive) is REJECTED. The circuit never accepts a
/// "maybe-member" — it over-rejects, which is the safe failure mode.
#[test]
fn false_positive_rejected() {
    let mut filter = BloomFilter::new(TEST_PARAMS);
    let inputs = [pallas::Base::from(888u64), pallas::Base::from(999u64)];
    let h = hash_of(&inputs);
    // Set all k positions to 1 without actually inserting h (a false positive).
    for idx in BloomFilter::indices(TEST_PARAMS, h) {
        filter.bits[idx as usize] = 1;
    }
    assert!(
        filter.maybe_contains_hash(h),
        "setup: should look like a member"
    );

    let circuit = TestCircuit { filter, inputs };
    let prover = MockProver::run(K_CIRCUIT, &circuit, vec![]).unwrap();
    assert!(
        prover.verify().is_err(),
        "all-bits-set (member / false positive) must be rejected"
    );
}

/// COMPLETENESS edge: an empty filter (all bits 0) accepts every password.
#[test]
fn empty_filter_accepts() {
    let filter = BloomFilter::new(TEST_PARAMS);
    let inputs = [pallas::Base::from(5u64), pallas::Base::from(6u64)];
    let circuit = TestCircuit { filter, inputs };
    MockProver::run(K_CIRCUIT, &circuit, vec![])
        .unwrap()
        .assert_satisfied();
}

/// Build pipeline: a filter built from plaintext passwords detects each of them.
#[test]
fn build_pipeline_detects_members() {
    let pad = 128;
    let breached: &[&[u8]] = &[b"password", b"123456", b"qwerty", b"letmein"];
    let f = BloomFilter::from_passwords(TEST_PARAMS, breached, pad);
    for &b in breached {
        assert!(
            f.maybe_contains_password(b, pad),
            "breached password must be detected"
        );
    }
}

/// Serialization roundtrip: bit-packed bytes reconstruct the same filter.
#[test]
fn packed_serialization_roundtrip() {
    let pad = 128;
    let f = BloomFilter::from_passwords(TEST_PARAMS, &[b"password", b"abc123"], pad);
    let packed = f.to_packed_bytes();
    assert_eq!(packed.len(), (TEST_PARAMS.m() as usize).div_ceil(8));
    let f2 = BloomFilter::from_packed_bytes(TEST_PARAMS, &packed);
    assert_eq!(f.bits, f2.bits);
    // member detection survives the roundtrip
    assert!(f2.maybe_contains_password(b"password", pad));
}
