// SPDX-License-Identifier: AGPL-3.0-only
//! OPAQUE-ZKPP Halo2 circuits.
//!
//! Combined circuit:
//! - Gadget A: Policy Engine (character classification via lookup tables)
//! - Gadget C: Opaque-Binder (HashToCurve + the OPAQUE element M = blind·H_p)
//! - Gadget D: breach Bloom non-membership
//! - Gadget H: history tags for the history checker

pub mod gadget_a;
pub mod gadget_c;
pub mod gadget_d;
pub mod gadget_h;

use halo2_proofs::{
    circuit::{Layouter, SimpleFloorPlanner, Value},
    plonk::{Advice, Circuit, Column, ConstraintSystem, Error, Expression, Instance, Selector},
    poly::Rotation,
};
use pasta_curves::pallas;

use crate::types::{CE_DEFAULT_POLICY, MAX_PASSWORD_LEN, PolicyParams};
use gadget_a::{PolicyEngineChip, PolicyEngineConfig};
use gadget_c::{OpaqueBinderChip, OpaqueBinderConfig, hash_to_curve_outside};
use gadget_d::{BloomFilter, BloomParams, BreachBloomChip, BreachBloomConfig};
use gadget_h::{HistoryTagChip, HistoryTagConfig, HistoryTagWitness};

/// The combined circuit's domain: 2^11 = 2048 rows.
pub const ZKPP_K: u32 = 11;

/// Breach Bloom filter parameters embedded in the combined circuit.
/// Demo size (m = 256); production m/k tuning + table packing tracked separately.
pub const BREACH_PARAMS: BloomParams = BloomParams {
    index_bits: 8,
    k: 3,
};

/// Bytes per field element for packing (31 bytes, safely within ~254-bit Pallas field).
const BYTES_PER_FE: usize = 31;

/// Configuration for the combined ZKPP circuit.
#[derive(Debug, Clone)]
pub struct ZkppCircuitConfig {
    policy_engine: PolicyEngineConfig,
    opaque_binder: OpaqueBinderConfig,
    history: HistoryTagConfig,
    breach: BreachBloomConfig,
    instance: Column<Instance>,
    /// Cross-gadget packing: Horner's method byte→FE with copy constraints.
    pack_byte: Column<Advice>,
    pack_acc: Column<Advice>,
    q_pack_init: Selector,
    q_pack_step: Selector,
}

/// The history inputs of one proof: owner domain `d`, the comparison domains
/// `c_j` the server requires, and the tag witness built from the evaluator's
/// answers.
#[derive(Clone, Debug)]
pub struct HistoryInputs {
    pub d: pallas::Base,
    pub domains: Vec<pallas::Base>,
    pub witness: HistoryTagWitness,
}

/// What a proving/verifying key is built for: the policy minimums (fixed
/// columns) and the number of history comparison domains (layout). A proof
/// verifies only under the key of its own shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CircuitShape {
    pub policy: PolicyParams,
    pub history_domains: usize,
}

impl CircuitShape {
    /// `policy` with one comparison domain.
    pub const fn single_domain(policy: PolicyParams) -> Self {
        Self {
            policy,
            history_domains: 1,
        }
    }
}

/// Number of public instances for `domains` comparison domains:
/// `M (2) + d (1) + c_j (D) + B (2) + (Z_j (2) + t_j (1))·D`.
pub const fn instance_count(domains: usize) -> usize {
    5 + 4 * domains
}

/// Combined ZKPP circuit: Policy Engine + Opaque-Binder + breach + history tags.
///
/// Public instance layout, `D` = comparison domains:
/// - \[0, 1\] M = blind·H(P), the OPAQUE registration element
/// - \[2\] d (owner domain)
/// - \[3 .. 3+D\] c_j (comparison domains)
/// - \[3+D, 4+D\] B (blinded history request)
/// - then per domain: Z_j.x, Z_j.y, t_j
///
/// Policy compliance is not an instance: the policy minimums are fixed columns
/// of the keys, and a password below them has no satisfying witness. The
/// history comparison is not in the circuit: the checker runs the KSF over
/// each `t_j` and compares.
///
/// Cross-gadget password binding: Gadget A's bytes are packed and
/// copy-constrained to Gadget C's field elements; Gadget D hashes the same
/// elements, Gadget H hashes Gadget C's Poseidon output.
#[derive(Clone)]
pub struct ZkppCircuit {
    pub password: [u8; MAX_PASSWORD_LEN],
    pub password_len: u32,
    /// OPRF blind of the OPAQUE request, as a base-field element: the circuit
    /// computes `M = blind·H_p` with it.
    pub blind: pallas::Base,
    /// Policy minimums, fixed at key generation: the keys built from this
    /// circuit verify only proofs made against this policy.
    pub policy: PolicyParams,
    /// Embedded breach Bloom filter (public). Empty = no breach check.
    pub breach_filter: BloomFilter,
    /// Comparison domains per proof, fixed at key generation like the policy.
    pub history_domains: usize,
    /// History witness; `None` for key generation.
    pub history: Option<HistoryInputs>,
}

impl Default for ZkppCircuit {
    fn default() -> Self {
        Self {
            password: [0u8; MAX_PASSWORD_LEN],
            password_len: 0,
            blind: pallas::Base::one(),
            policy: CE_DEFAULT_POLICY,
            breach_filter: BloomFilter::new(BREACH_PARAMS),
            history_domains: 1,
            history: None,
        }
    }
}

impl ZkppCircuit {
    /// The circuit for keys of `shape`: keygen takes only its structure and
    /// fixed columns, never a witness.
    pub fn for_shape(shape: CircuitShape) -> Self {
        Self {
            policy: shape.policy,
            history_domains: shape.history_domains,
            ..Self::default()
        }
    }

    pub fn shape(&self) -> CircuitShape {
        CircuitShape {
            policy: self.policy,
            history_domains: self.history_domains,
        }
    }

    /// Compute the expected public instance values for this circuit.
    ///
    /// # Panics
    /// Without history inputs, or with a domain count other than the shape's.
    pub fn instance_values(&self) -> Vec<pallas::Base> {
        use group::Curve;
        use pasta_curves::arithmetic::CurveAffine;

        // M = blind·H_p (public instances [0], [1]).
        let (h_p, _, _) = hash_to_curve_outside(&self.password);
        let m = (pallas::Point::from(h_p) * crate::history::blind_scalar(self.blind)).to_affine();
        let coords = m.coordinates().unwrap();

        let history = self.history.as_ref().expect("history inputs");
        assert_eq!(history.domains.len(), self.history_domains);
        let u = crate::history::history_input(history.d, &self.password);
        let (b, tags) = gadget_h::tag_instances(&history.domains, u, &history.witness);
        let b = b.coordinates().unwrap();

        let mut values = vec![*coords.x(), *coords.y(), history.d];
        values.extend(&history.domains);
        values.extend([*b.x(), *b.y()]);
        for (z, t) in tags {
            let z = z.coordinates().unwrap();
            values.extend([*z.x(), *z.y(), t]);
        }
        values
    }
}

impl Circuit<pallas::Base> for ZkppCircuit {
    type Config = ZkppCircuitConfig;
    type FloorPlanner = SimpleFloorPlanner;

    /// Keeps the policy and the domain count: they set fixed columns and the
    /// layout, which a witness-free copy must reproduce exactly.
    fn without_witnesses(&self) -> Self {
        Self::for_shape(self.shape())
    }

    fn configure(meta: &mut ConstraintSystem<pallas::Base>) -> Self::Config {
        let policy_engine = PolicyEngineChip::configure(meta);
        let opaque_binder = OpaqueBinderChip::configure(meta);
        let history = HistoryTagChip::configure(
            meta,
            opaque_binder.ecc_config.clone(),
            opaque_binder.poseidon_config.clone(),
        );
        let breach = BreachBloomChip::configure(meta, BREACH_PARAMS);

        let instance = meta.instance_column();
        meta.enable_equality(instance);

        let pack_byte = meta.advice_column();
        let pack_acc = meta.advice_column();
        meta.enable_equality(pack_byte);
        meta.enable_equality(pack_acc);

        // ── Cross-gadget packing gates (Horner's method) ──
        // Pack init: acc = byte (first byte of reversed chunk)
        let q_pack_init = meta.selector();
        meta.create_gate("pack_init", |meta| {
            let q = meta.query_selector(q_pack_init);
            let acc = meta.query_advice(pack_acc, Rotation::cur());
            let byte = meta.query_advice(pack_byte, Rotation::cur());
            vec![q * (acc - byte)]
        });

        // Pack step: acc_cur = acc_prev * 256 + byte_cur
        let q_pack_step = meta.selector();
        meta.create_gate("pack_step", |meta| {
            let q = meta.query_selector(q_pack_step);
            let acc_cur = meta.query_advice(pack_acc, Rotation::cur());
            let acc_prev = meta.query_advice(pack_acc, Rotation::prev());
            let byte = meta.query_advice(pack_byte, Rotation::cur());
            let c256 = Expression::Constant(pallas::Base::from(256u64));
            vec![q * (acc_cur - acc_prev * c256 - byte)]
        });

        ZkppCircuitConfig {
            policy_engine,
            opaque_binder,
            history,
            breach,
            instance,
            pack_byte,
            pack_acc,
            q_pack_init,
            q_pack_step,
        }
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<pallas::Base>,
    ) -> Result<(), Error> {
        // Prepare witness values
        let pw_vals: [Value<pallas::Base>; MAX_PASSWORD_LEN] =
            core::array::from_fn(|i| Value::known(pallas::Base::from(self.password[i] as u64)));
        let active: [Value<pallas::Base>; MAX_PASSWORD_LEN] = core::array::from_fn(|i| {
            Value::known(if (i as u64) < u64::from(self.password_len) {
                pallas::Base::one()
            } else {
                pallas::Base::zero()
            })
        });

        // HashToCurve witness
        let (h_p, _, offset) = hash_to_curve_outside(&self.password);

        // ── Gadget A: Policy Engine ──
        let policy_chip = PolicyEngineChip::construct(config.policy_engine.clone(), self.policy);
        let pw_byte_cells = policy_chip.synthesize(&mut layouter, &pw_vals, &active)?;

        // ── Gadget C: Opaque-Binder ──
        let binder_chip = OpaqueBinderChip::construct(config.opaque_binder.clone());
        let binder_out = binder_chip.synthesize(
            &mut layouter,
            &pw_vals,
            Value::known(self.blind),
            Value::known(h_p),
            Value::known(offset),
        )?;

        // ── Cross-gadget binding: A↔C (sound packing constraint) ──
        // Horner's method: copy Gadget A byte cells, pack into FEs with
        // constrained gates, then constrain_equal result to Gadget C FE cells.
        // This proves in-circuit that C's Poseidon input = packing(A's bytes).
        layouter.assign_region(
            || "pack_and_bind_a_to_c",
            |mut region| {
                let mut row = 0;
                for (chunk_idx, chunk) in pw_byte_cells.chunks(BYTES_PER_FE).enumerate() {
                    let chunk_len = chunk.len();
                    let mut acc_val = Value::known(pallas::Base::zero());

                    // Process bytes in reverse order (Horner's method):
                    // acc = b[n-1]; acc = acc*256 + b[n-2]; ...; acc = acc*256 + b[0] = FE
                    for (i, byte_idx) in (0..chunk_len).rev().enumerate() {
                        // Copy byte cell from Gadget A
                        let byte_cell = chunk[byte_idx].copy_advice(
                            || format!("pk_byte_{chunk_idx}_{byte_idx}"),
                            &mut region,
                            config.pack_byte,
                            row,
                        )?;
                        let byte_val = byte_cell.value().copied();

                        if i == 0 {
                            // Init: acc = byte
                            config.q_pack_init.enable(&mut region, row)?;
                            acc_val = byte_val;
                        } else {
                            // Step: acc = prev_acc * 256 + byte
                            config.q_pack_step.enable(&mut region, row)?;
                            acc_val = acc_val
                                .zip(byte_val)
                                .map(|(a, b)| a * pallas::Base::from(256u64) + b);
                        }

                        let acc_cell = region.assign_advice(
                            || format!("pk_acc_{chunk_idx}_{byte_idx}"),
                            config.pack_acc,
                            row,
                            || acc_val,
                        )?;

                        // Last row of chunk (byte_idx == 0): bind to Gadget C FE
                        if byte_idx == 0 {
                            region.constrain_equal(
                                acc_cell.cell(),
                                binder_out.pw_fe_cells[chunk_idx].cell(),
                            )?;
                        }

                        row += 1;
                    }
                }
                Ok(())
            },
        )?;

        // ── Gadget D: Breach Bloom non-membership ──
        // Hashes the SAME packed-password FEs that Gadget C binds to Gadget A,
        // so the breach check is over the real registered password.
        let breach_chip = BreachBloomChip::construct(config.breach.clone());
        breach_chip.synthesize(&mut layouter, &self.breach_filter, &binder_out.pw_fe_cells)?;

        // ── Gadget H: history tags over Gadget C's password hash ──
        // d and every c_j come from the instance, so the verifier fixes the
        // owner and the domains the tags are for.
        let domains = self.history_domains;
        let (d, cs) = layouter.assign_region(
            || "history domains",
            |mut region| {
                let col = config.pack_byte;
                let d = region.assign_advice_from_instance(|| "d", config.instance, 2, col, 0)?;
                let cs = (0..domains)
                    .map(|j| {
                        region.assign_advice_from_instance(
                            || format!("c {j}"),
                            config.instance,
                            3 + j,
                            col,
                            1 + j,
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok((d, cs))
            },
        )?;
        let history_witness = match &self.history {
            Some(h) => Value::known(&h.witness),
            None => Value::unknown(),
        };
        let tags = HistoryTagChip::construct(config.history.clone()).synthesize(
            &mut layouter,
            d,
            binder_out.u_hash.clone(),
            &cs,
            history_witness,
        )?;

        // ── Instance constraints ──
        layouter.constrain_instance(binder_out.m_x.cell(), config.instance, 0)?;
        layouter.constrain_instance(binder_out.m_y.cell(), config.instance, 1)?;
        let exposed = [&tags.b_x, &tags.b_y].into_iter().chain(
            tags.tags
                .iter()
                .flat_map(|tag| [&tag.z_x, &tag.z_y, &tag.t]),
        );
        for (i, cell) in exposed.enumerate() {
            layouter.constrain_instance(cell.cell(), config.instance, 3 + domains + i)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests;
