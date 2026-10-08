// SPDX-License-Identifier: AGPL-3.0-only
//! Gadget C: Opaque-Binder
//!
//! Binds the proof to the OPAQUE registration message.
//! 1. HashToCurve: Poseidon(password) → field element → Pallas point H_p
//! 2. M = b·H_p, variable-base scalar multiplication with the OPRF blind `b`
//!    as a base-field element (every blind below the base-field modulus, which
//!    is all but a 2^-126 fraction of the scalar field, has this form)
//! 3. Expose M: the verifier compares it with the element of the OPAQUE
//!    registration request, so the request is the blinded hash of the proved
//!    password.

use ff::{Field, PrimeField};
use halo2_gadgets::{
    ecc::{
        FixedPoints, NonIdentityPoint, ScalarVar,
        chip::{
            BaseFieldElem, CircuitVersion, EccChip, EccConfig, FixedPoint, FullScalar, H,
            ShortScalar,
        },
    },
    poseidon::{Hash as PoseidonHash, Pow5Chip, Pow5Config, primitives::ConstantLength},
    utilities::lookup_range_check::{LookupRangeCheck, LookupRangeCheckConfig},
};

use halo2_proofs::{
    circuit::{AssignedCell, Layouter, Value},
    plonk::{Advice, Column, ConstraintSystem, Error, Fixed, Selector, TableColumn},
    poly::Rotation,
};
use pasta_curves::pallas;

use crate::poseidon::PoseidonSpec;
use crate::types::MAX_PASSWORD_LEN;

/// Bytes per field element for packing.
const BYTES_PER_FE: usize = 31;

// ─── Dummy FixedPoints for EccChip (only variable-base mul is used) ───

/// Dummy FixedPoints — we only use variable-base multiplication.
/// These types satisfy trait bounds but their methods are never called.
#[derive(Debug, Eq, PartialEq, Clone)]
pub struct ZkppFixedBases;

#[derive(Debug, Eq, PartialEq, Clone)]
pub struct ZkppFullWidth;

#[derive(Debug, Eq, PartialEq, Clone)]
pub struct ZkppShortScalar;

#[derive(Debug, Eq, PartialEq, Clone)]
pub struct ZkppBaseField;

impl FixedPoint<pallas::Affine> for ZkppFullWidth {
    type FixedScalarKind = FullScalar;
    fn generator(&self) -> pallas::Affine {
        unimplemented!("ZKPP only uses variable-base multiplication")
    }
    fn u(&self) -> Vec<[<pallas::Base as PrimeField>::Repr; H]> {
        unimplemented!("ZKPP only uses variable-base multiplication")
    }
    fn z(&self) -> Vec<u64> {
        unimplemented!("ZKPP only uses variable-base multiplication")
    }
}

impl FixedPoint<pallas::Affine> for ZkppShortScalar {
    type FixedScalarKind = ShortScalar;
    fn generator(&self) -> pallas::Affine {
        unimplemented!("ZKPP only uses variable-base multiplication")
    }
    fn u(&self) -> Vec<[<pallas::Base as PrimeField>::Repr; H]> {
        unimplemented!("ZKPP only uses variable-base multiplication")
    }
    fn z(&self) -> Vec<u64> {
        unimplemented!("ZKPP only uses variable-base multiplication")
    }
}

impl FixedPoint<pallas::Affine> for ZkppBaseField {
    type FixedScalarKind = BaseFieldElem;
    fn generator(&self) -> pallas::Affine {
        unimplemented!("ZKPP only uses variable-base multiplication")
    }
    fn u(&self) -> Vec<[<pallas::Base as PrimeField>::Repr; H]> {
        unimplemented!("ZKPP only uses variable-base multiplication")
    }
    fn z(&self) -> Vec<u64> {
        unimplemented!("ZKPP only uses variable-base multiplication")
    }
}

impl FixedPoints<pallas::Affine> for ZkppFixedBases {
    type FullScalar = ZkppFullWidth;
    type ShortScalar = ZkppShortScalar;
    type Base = ZkppBaseField;
}

// ─── Opaque-Binder Gadget ───

/// Output cells from the Opaque-Binder gadget.
pub struct OpaqueBinderOutput {
    /// M_x coordinate of the OPAQUE registration element.
    pub m_x: AssignedCell<pallas::Base, pallas::Base>,
    /// M_y coordinate of the OPAQUE registration element.
    pub m_y: AssignedCell<pallas::Base, pallas::Base>,
    /// Poseidon hash of password (u), used for cross-gadget binding.
    pub u_hash: AssignedCell<pallas::Base, pallas::Base>,
    /// Password field-element cells assigned during Poseidon hashing.
    /// Used for cross-gadget binding (copy-constrain to Gadget A packing).
    pub pw_fe_cells: Vec<AssignedCell<pallas::Base, pallas::Base>>,
}

/// Configuration for the Opaque-Binder gadget.
#[derive(Debug, Clone)]
pub struct OpaqueBinderConfig {
    /// ECC chip config (10 advice + 8 fixed + range check).
    pub ecc_config: EccConfig<ZkppFixedBases, LookupRangeCheckConfig<pallas::Base, 10>>,
    /// Poseidon chip config for HashToCurve.
    pub poseidon_config: Pow5Config<pallas::Base, 3, 2>,
    /// Advice column for general inputs.
    pub input_col: Column<Advice>,
    /// Selector for hash-to-curve binding: x = u + offset.
    pub q_htc_bind: Selector,
    /// Range check lookup table column (for manual table loading).
    pub range_check_table: TableColumn,
}

/// Chip implementing the Opaque-Binder gadget.
pub struct OpaqueBinderChip {
    config: OpaqueBinderConfig,
}

impl OpaqueBinderChip {
    pub fn construct(config: OpaqueBinderConfig) -> Self {
        Self { config }
    }

    pub fn configure(meta: &mut ConstraintSystem<pallas::Base>) -> OpaqueBinderConfig {
        // 10 advice columns for ECC chip
        let advices: [Column<Advice>; 10] = core::array::from_fn(|_| meta.advice_column());
        for col in &advices {
            meta.enable_equality(*col);
        }

        // Extra advice columns
        let input_col = meta.advice_column();
        meta.enable_equality(input_col);

        // Lookup table for range check
        let lookup_table = meta.lookup_table_column();

        // 8 fixed columns for ECC Lagrange coefficients
        let lagrange_coeffs: [Column<Fixed>; 8] = core::array::from_fn(|_| meta.fixed_column());

        // Constants column
        let constants = meta.fixed_column();
        meta.enable_constant(constants);

        // Range check (uses advices[9])
        let range_check = LookupRangeCheckConfig::configure(meta, advices[9], lookup_table);

        // ECC chip
        let ecc_config =
            EccChip::<ZkppFixedBases, _>::configure(meta, advices, lagrange_coeffs, range_check);

        // Poseidon chip (separate columns to avoid conflicts)
        let poseidon_state: [Column<Advice>; 3] = core::array::from_fn(|_| meta.advice_column());
        let poseidon_partial_sbox = meta.advice_column();
        let poseidon_rc_a: [Column<Fixed>; 3] = core::array::from_fn(|_| meta.fixed_column());
        let poseidon_rc_b: [Column<Fixed>; 3] = core::array::from_fn(|_| meta.fixed_column());

        for col in poseidon_state
            .iter()
            .chain(std::iter::once(&poseidon_partial_sbox))
        {
            meta.enable_equality(*col);
        }
        meta.enable_constant(poseidon_rc_b[0]);

        let poseidon_config = Pow5Chip::configure::<PoseidonSpec>(
            meta,
            poseidon_state,
            poseidon_partial_sbox,
            poseidon_rc_a,
            poseidon_rc_b,
        );

        // Gate: hash-to-curve binding x = u + offset. The curve equation needs
        // no gate of its own: the ECC chip constrains every witnessed
        // NonIdentityPoint to y^2 = x^3 + 5, and `x` here is a copy of that
        // point's x-coordinate.
        let q_htc_bind = meta.selector();
        meta.create_gate("htc_bind", |meta| {
            let q = meta.query_selector(q_htc_bind);
            // input_col rows: [0]=u, [1]=offset, [2]=x
            let u = meta.query_advice(input_col, Rotation::cur());
            let offset = meta.query_advice(input_col, Rotation(1));
            let x = meta.query_advice(input_col, Rotation(2));
            vec![q * (x - u - offset)]
        });

        OpaqueBinderConfig {
            ecc_config,
            poseidon_config,
            input_col,
            q_htc_bind,
            range_check_table: lookup_table,
        }
    }

    /// Load the range check table (0..1023 for K=10), once per circuit.
    pub(crate) fn load_range_check_table(
        &self,
        layouter: &mut impl Layouter<pallas::Base>,
    ) -> Result<(), Error> {
        layouter.assign_table(
            || "range_check_table",
            |mut table| {
                for index in 0..(1u64 << 10) {
                    table.assign_cell(
                        || "table_idx",
                        self.config.range_check_table,
                        index as usize,
                        || Value::known(pallas::Base::from(index)),
                    )?;
                }
                Ok(())
            },
        )
    }

    /// Pack raw bytes into a single field element (31 bytes per FE, little-endian).
    fn pack_chunk_to_fe(chunk: &[Value<pallas::Base>]) -> Value<pallas::Base> {
        chunk
            .iter()
            .enumerate()
            .fold(Value::known(pallas::Base::zero()), |acc, (j, byte_val)| {
                acc.zip(*byte_val).map(|(a, b)| {
                    let mut shift = pallas::Base::one();
                    for _ in 0..j {
                        shift *= pallas::Base::from(256u64);
                    }
                    a + b * shift
                })
            })
    }

    fn pack_bytes_to_fes(bytes: &[Value<pallas::Base>]) -> Vec<Value<pallas::Base>> {
        bytes
            .chunks(BYTES_PER_FE)
            .map(Self::pack_chunk_to_fe)
            .collect()
    }

    fn assign_fe(
        layouter: &mut impl Layouter<pallas::Base>,
        col: Column<Advice>,
        name: &str,
        val: Value<pallas::Base>,
    ) -> Result<AssignedCell<pallas::Base, pallas::Base>, Error> {
        layouter.assign_region(
            || name.to_string(),
            |mut region| region.assign_advice(|| name, col, 0, || val),
        )
    }

    /// Synthesize: HashToCurve(password) → H_p, then M = blind·H_p.
    ///
    /// Returns `OpaqueBinderOutput` including `pw_fe_cells` — the assigned
    /// password field-element cells used as Poseidon input. The combined
    /// circuit copy-constrains these to packed Gadget A bytes for binding.
    pub fn synthesize(
        &self,
        layouter: &mut impl Layouter<pallas::Base>,
        password: &[Value<pallas::Base>; MAX_PASSWORD_LEN],
        blind: Value<pallas::Base>,
        h_p: Value<pallas::Affine>,
        htc_offset: Value<pallas::Base>,
    ) -> Result<OpaqueBinderOutput, Error> {
        let config = &self.config;

        // Load range check table (0..1023 for K=10)
        self.load_range_check_table(layouter)?;

        // Step 1: Poseidon hash of password → u
        let password_fes = Self::pack_bytes_to_fes(password.as_slice());

        // Assign each FE into input_col (returned for cross-gadget binding)
        let mut all_fe_cells = Vec::with_capacity(password_fes.len());
        for (idx, fe_val) in password_fes.iter().enumerate() {
            let cell =
                Self::assign_fe(layouter, config.input_col, &format!("pw_fe_{idx}"), *fe_val)?;
            all_fe_cells.push(cell);
        }

        let first = all_fe_cells[0].clone();
        let second = all_fe_cells[1].clone();

        let poseidon_chip = Pow5Chip::construct(config.poseidon_config.clone());
        let hasher = PoseidonHash::<_, _, PoseidonSpec, ConstantLength<2>, 3, 2>::init(
            poseidon_chip,
            layouter.namespace(|| "poseidon_init_0"),
        )?;
        let mut u_hash = hasher.hash(layouter.namespace(|| "poseidon_hash_0"), [first, second])?;

        for (idx, next) in all_fe_cells.iter().skip(2).enumerate() {
            let idx = idx + 2;
            let next = next.clone();
            let chip = Pow5Chip::construct(config.poseidon_config.clone());
            let hasher = PoseidonHash::<_, _, PoseidonSpec, ConstantLength<2>, 3, 2>::init(
                chip,
                layouter.namespace(|| format!("poseidon_init_{idx}")),
            )?;
            u_hash = hasher.hash(
                layouter.namespace(|| format!("poseidon_hash_{idx}")),
                [u_hash, next],
            )?;
        }

        // Step 2: H_p is the point hashed from these bytes. The ECC chip keeps
        // it on the curve; its x-coordinate is copied into the binding row, so
        // x = u + offset ties the point to Poseidon(password). The sign of y is
        // left free: (x, -y) is the hash of the same bytes, so it cannot carry
        // a different password past the policy proof.
        // AnchoredBase is the only version that may prove: the alternative exists
        // solely to rebuild the verifying key of circuits built before the
        // incomplete-addition base was anchored to the real base, so that proofs
        // made by them can still be verified.
        let ecc_chip = EccChip::construct(config.ecc_config.clone(), CircuitVersion::AnchoredBase);

        let h_p_point =
            NonIdentityPoint::new(ecc_chip.clone(), layouter.namespace(|| "witness_H_p"), h_p)?;

        let offset_cell = layouter.assign_region(
            || "hash_to_curve_verify",
            |mut region| {
                config.q_htc_bind.enable(&mut region, 0)?;
                u_hash.copy_advice(|| "u", &mut region, config.input_col, 0)?;
                let offset_cell =
                    region.assign_advice(|| "offset", config.input_col, 1, || htc_offset)?;
                h_p_point
                    .inner()
                    .x()
                    .copy_advice(|| "x", &mut region, config.input_col, 2)?;
                Ok(offset_cell)
            },
        )?;

        // The offset is the try counter of `hash_to_curve_outside`, below
        // HTC_TRIES = 2^8. Unbounded, it would move x onto any curve point
        // and detach H_p from the password.
        config.ecc_config.lookup_config.copy_short_check(
            layouter.namespace(|| "offset < 2^8"),
            offset_cell,
            HTC_TRY_BITS,
        )?;

        // Step 3: the OPAQUE element M = blind·H_p, computed here from the same
        // H_p, so the exposed M is the blinded hash of this password and of no
        // other value.
        let blind_cell = layouter.assign_region(
            || "blind",
            |mut region| region.assign_advice(|| "blind", config.input_col, 0, || blind),
        )?;
        let blind = ScalarVar::from_base(
            ecc_chip.clone(),
            layouter.namespace(|| "blind scalar"),
            &blind_cell,
        )?;
        let (m, _) = h_p_point.mul(layouter.namespace(|| "M = blind·H_p"), blind)?;

        let m_x = m.inner().x();
        let m_y = m.inner().y();

        Ok(OpaqueBinderOutput {
            m_x,
            m_y,
            u_hash,
            pw_fe_cells: all_fe_cells,
        })
    }
}

/// Hash-to-curve tries `2^HTC_TRY_BITS` offsets; the circuit range-checks the
/// offset to this width.
pub const HTC_TRY_BITS: usize = 8;

/// Compute hash-to-curve outside the circuit (for witness generation).
/// Poseidon → field element → constant-time scan of x=u+offset for y^2 = x^3 + 5
/// (fixed 256 trials, no early exit — no timing side-channel; RFC 9380 §10.1).
pub fn hash_to_curve_outside(password: &[u8]) -> (pallas::Affine, pallas::Base, pallas::Base) {
    use crate::poseidon::{bytes_to_field_elements, poseidon_hash_chain};
    use pasta_curves::arithmetic::CurveAffine;

    use subtle::{Choice, ConditionallySelectable};

    let fes = bytes_to_field_elements(password);
    let u = poseidon_hash_chain(&fes);

    // Constant-time hash-to-curve: ALWAYS scan a fixed 256 offsets (no early exit)
    // and select the first offset whose x yields a square y via constant-time
    // conditional selection. This removes the try-and-increment timing side-channel
    // (RFC 9380 §10.1 — a variable iteration count leaks information about
    // Poseidon(pw)). The selected point is identical to the original first-valid
    // result, so it stays byte-compatible with the in-circuit fixed-trial binding.
    let b = pallas::Base::from(5u64);
    let mut found = Choice::from(0u8);
    let mut sel_x = pallas::Base::ZERO;
    let mut sel_y = pallas::Base::ZERO;
    let mut sel_offset = pallas::Base::ZERO;
    for i in 0u64..(1 << HTC_TRY_BITS) {
        let offset = pallas::Base::from(i);
        let x = u + offset;
        let x3b = x * x * x + b;
        let y = x3b.sqrt(); // CtOption — constant-time Tonelli-Shanks
        let is_sq = y.is_some();
        let take = is_sq & !found;
        sel_x = pallas::Base::conditional_select(&sel_x, &x, take);
        sel_y = pallas::Base::conditional_select(&sel_y, &y.unwrap_or(pallas::Base::ZERO), take);
        sel_offset = pallas::Base::conditional_select(&sel_offset, &offset, take);
        found |= is_sq;
    }
    assert!(
        bool::from(found),
        "hash_to_curve: no valid point found in 256 tries"
    );
    let point = pallas::Affine::from_xy(sel_x, sel_y).unwrap();
    (point, u, sel_offset)
}

#[cfg(test)]
mod tests;
