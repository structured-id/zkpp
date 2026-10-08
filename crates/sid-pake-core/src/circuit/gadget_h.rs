// SPDX-License-Identifier: AGPL-3.0-only
//! Gadget H: private password-history tag.
//!
//! Proves that the tag `t` handed to the history checker derives from the
//! same password bytes the other gadgets check:
//! ```text
//! u = Poseidon(d, Poseidon(P))   inner hash shared with the binder
//! H = (u + offset, y)            offset minimal, y even
//! B = r·H                        r nonzero; one request for every domain
//! Z_j = r·N_j                    per comparison domain j (its own key)
//! t_j = Poseidon(Poseidon(c_j, u), x(N_j))
//! ```
//! Both scalar multiplications stay in the circuit: a native link over a
//! Pedersen commitment of `N_j` would leave its `G2` component free, letting a
//! prover shift `N_j` and so choose another tag, while exposing `N_j` would
//! give the key holder the same cheap verifier as `t`.
//! Minimality: every offset below the chosen one gets a witness that
//! `x³ + 5` is a nonzero non-square (`w² = g·(x³ + 5)`, `w ≠ 0`, `g` a fixed
//! non-square). Even `y`: `y = 2h` with `h < 2^253 < p/2`, so `2h` does not wrap.
//! The KSF and the comparison with retained entries are the checker's; they
//! are not in this relation.

use ff::{Field, PrimeField};
use halo2_gadgets::{
    ecc::{
        NonIdentityPoint, ScalarVar,
        chip::{CircuitVersion, EccChip, EccConfig},
    },
    poseidon::{Hash as PoseidonHash, Pow5Chip, Pow5Config, primitives::ConstantLength},
    utilities::lookup_range_check::{LookupRangeCheck, LookupRangeCheckConfig},
};
use halo2_proofs::{
    circuit::{AssignedCell, Layouter, Value},
    plonk::{Advice, Column, ConstraintSystem, Error, Expression, Fixed, Selector},
    poly::Rotation,
};
use pasta_curves::{arithmetic::CurveAffine, pallas};

use super::gadget_c::{HTC_TRY_BITS, ZkppFixedBases};
use crate::poseidon::PoseidonSpec;

/// Offsets scanned for the minimal one.
const TRIES: usize = 1 << HTC_TRY_BITS;

/// `h < 2^(10·HALF_WORDS + HALF_TOP_BITS) = 2^253`.
const HALF_WORDS: usize = 25;
const HALF_TOP_BITS: usize = 3;

type Cell = AssignedCell<pallas::Base, pallas::Base>;

/// The fixed non-square: the multiplicative generator of the base field.
pub fn non_square() -> pallas::Base {
    pallas::Base::MULTIPLICATIVE_GENERATOR
}

#[derive(Debug, Clone)]
pub struct HistoryTagConfig {
    ecc: EccConfig<ZkppFixedBases, LookupRangeCheckConfig<pallas::Base, 10>>,
    poseidon: Pow5Config<pallas::Base, 3, 2>,
    /// Scan columns: u, lt (offset below the chosen one), w, w⁻¹, running count.
    u: Column<Advice>,
    lt: Column<Advice>,
    w: Column<Advice>,
    w_inv: Column<Advice>,
    acc: Column<Advice>,
    /// Offset of the scan row.
    idx: Column<Fixed>,
    q_scan: Selector,
    q_scan_step: Selector,
    q_scan_first: Selector,
    /// Bind row: x = u + offset, y = 2h, r·r⁻¹ = 1 (columns u, lt, w, w_inv, acc).
    q_bind: Selector,
}

/// Witnesses of one history tag, computed by [`HistoryTagWitness::honest`]
/// or chosen freely by tests.
#[derive(Clone, Debug)]
pub struct HistoryTagWitness {
    /// Canonical point of `u`.
    pub h: pallas::Affine,
    pub offset: u64,
    /// Non-square witnesses for offsets below `offset`.
    pub w: Vec<pallas::Base>,
    /// Blind, a nonzero base-field element.
    pub r: pallas::Base,
    /// `N_j = k_j·H`, the unblinded evaluation of each comparison domain.
    pub n: Vec<pallas::Affine>,
}

impl HistoryTagWitness {
    /// The honest witness for input `u`, blind `r` and the evaluations `Z_j`
    /// the evaluator returned for `B = r·H`.
    pub fn from_evaluations(u: pallas::Base, r: pallas::Base, z: &[pallas::Affine]) -> Self {
        use group::Curve;
        let inv = Option::<pallas::Scalar>::from(crate::history::blind_scalar(r).invert())
            .expect("nonzero blind");
        let n = z
            .iter()
            .map(|z| (pallas::Point::from(*z) * inv).to_affine())
            .collect();
        Self {
            n,
            ..Self::honest(u, r, &[])
        }
    }

    /// The honest witness for input `u`, blind `r` and history keys `k_j`.
    pub fn honest(u: pallas::Base, r: pallas::Base, keys: &[pallas::Scalar]) -> Self {
        use group::Curve;
        let (h, offset) = crate::history::canonical_point(u);
        let b5 = pallas::Base::from(5u64);
        let w = (0..offset)
            .map(|i| {
                let x = u + pallas::Base::from(i);
                (non_square() * (x.square() * x + b5))
                    .sqrt()
                    .expect("a non-square times g is a square")
            })
            .collect();
        let n = keys
            .iter()
            .map(|k| (pallas::Point::from(h) * k).to_affine())
            .collect();
        Self { h, offset, w, r, n }
    }
}

/// One comparison domain's evaluation and tag cells.
pub struct DomainTag {
    pub z_x: Cell,
    pub z_y: Cell,
    pub t: Cell,
}

/// Cells the combined circuit exposes as instances.
pub struct HistoryTagOutput {
    pub b_x: Cell,
    pub b_y: Cell,
    pub tags: Vec<DomainTag>,
}

pub struct HistoryTagChip {
    config: HistoryTagConfig,
}

impl HistoryTagChip {
    pub fn construct(config: HistoryTagConfig) -> Self {
        Self { config }
    }

    /// Shares the ECC and Poseidon configuration (and their columns) of the
    /// Opaque-Binder; the scan and bind gates use five of its ECC advice columns.
    pub fn configure(
        meta: &mut ConstraintSystem<pallas::Base>,
        ecc: EccConfig<ZkppFixedBases, LookupRangeCheckConfig<pallas::Base, 10>>,
        poseidon: Pow5Config<pallas::Base, 3, 2>,
    ) -> HistoryTagConfig {
        let [u, lt, w, w_inv, acc] = [0, 1, 2, 3, 4].map(|i| ecc.advices[i]);
        let idx = meta.fixed_column();
        let q_scan = meta.selector();
        let q_scan_step = meta.selector();
        let q_scan_first = meta.selector();
        let q_bind = meta.selector();
        let one = || Expression::Constant(pallas::Base::ONE);

        meta.create_gate("history scan", |meta| {
            let q = meta.query_selector(q_scan);
            let u = meta.query_advice(u, Rotation::cur());
            let lt = meta.query_advice(lt, Rotation::cur());
            let w = meta.query_advice(w, Rotation::cur());
            let w_inv = meta.query_advice(w_inv, Rotation::cur());
            let x = u + meta.query_fixed(idx);
            let rhs = Expression::Constant(non_square())
                * (x.clone() * x.clone() * x + Expression::Constant(pallas::Base::from(5u64)));
            vec![
                q.clone() * lt.clone() * (one() - lt.clone()),
                q.clone() * lt.clone() * (w.clone() * w.clone() - rhs),
                q * lt * (w * w_inv - one()),
            ]
        });
        meta.create_gate("history scan step", |meta| {
            let q = meta.query_selector(q_scan_step);
            let u_cur = meta.query_advice(u, Rotation::cur());
            let u_prev = meta.query_advice(u, Rotation::prev());
            let lt_cur = meta.query_advice(lt, Rotation::cur());
            let lt_prev = meta.query_advice(lt, Rotation::prev());
            let acc_cur = meta.query_advice(acc, Rotation::cur());
            let acc_prev = meta.query_advice(acc, Rotation::prev());
            vec![
                q.clone() * (u_cur - u_prev),
                // Offsets below the chosen one form a prefix.
                q.clone() * lt_cur.clone() * (one() - lt_prev),
                q * (acc_cur - acc_prev - lt_cur),
            ]
        });
        meta.create_gate("history scan first", |meta| {
            let q = meta.query_selector(q_scan_first);
            let lt = meta.query_advice(lt, Rotation::cur());
            let acc = meta.query_advice(acc, Rotation::cur());
            vec![q * (acc - lt)]
        });
        meta.create_gate("history bind", |meta| {
            let q = meta.query_selector(q_bind);
            // Columns on this row: u, offset, x, y, h; next row: r, r⁻¹.
            let u = meta.query_advice(u, Rotation::cur());
            let offset = meta.query_advice(lt, Rotation::cur());
            let x = meta.query_advice(w, Rotation::cur());
            let y = meta.query_advice(w_inv, Rotation::cur());
            let h = meta.query_advice(acc, Rotation::cur());
            let r = meta.query_advice(lt, Rotation::next());
            let r_inv = meta.query_advice(w, Rotation::next());
            vec![
                q.clone() * (x - u - offset),
                q.clone() * (y - Expression::Constant(pallas::Base::from(2u64)) * h),
                q * (r * r_inv - one()),
            ]
        });

        HistoryTagConfig {
            ecc,
            poseidon,
            u,
            lt,
            w,
            w_inv,
            acc,
            idx,
            q_scan,
            q_scan_step,
            q_scan_first,
            q_bind,
        }
    }

    fn poseidon2(
        &self,
        layouter: &mut impl Layouter<pallas::Base>,
        name: &str,
        inputs: [Cell; 2],
    ) -> Result<Cell, Error> {
        let hasher = PoseidonHash::<_, _, PoseidonSpec, ConstantLength<2>, 3, 2>::init(
            Pow5Chip::construct(self.config.poseidon.clone()),
            layouter.namespace(|| format!("{name} init")),
        )?;
        hasher.hash(layouter.namespace(|| name.to_string()), inputs)
    }

    /// `d` is the owner domain element, `u_c` the binder's Poseidon hash of the
    /// packed password (bound to the policy and breach gadgets), `domains` one
    /// comparison-domain element per required domain, in the order of
    /// `witness.n`.
    pub fn synthesize(
        &self,
        layouter: &mut impl Layouter<pallas::Base>,
        d: Cell,
        u_c: Cell,
        domains: &[Cell],
        witness: Value<&HistoryTagWitness>,
    ) -> Result<HistoryTagOutput, Error> {
        let config = &self.config;

        // u = Poseidon(d, Poseidon(P)): `crate::history::history_input`.
        let u = self.poseidon2(layouter, "u", [d, u_c])?;

        let ecc = EccChip::construct(config.ecc.clone(), CircuitVersion::AnchoredBase);
        let h_point = NonIdentityPoint::new(
            ecc.clone(),
            layouter.namespace(|| "H"),
            witness.map(|w| w.h),
        )?;

        // Minimal offset: lt = 1 exactly on the offsets below it, each with a
        // non-square witness; the running count ends at the offset.
        let count = layouter.assign_region(
            || "history scan",
            |mut region| {
                let mut acc_cell = None;
                for i in 0..TRIES {
                    config.q_scan.enable(&mut region, i)?;
                    if i == 0 {
                        config.q_scan_first.enable(&mut region, i)?;
                        u.copy_advice(|| "u", &mut region, config.u, i)?;
                    } else {
                        config.q_scan_step.enable(&mut region, i)?;
                        region.assign_advice(|| "u", config.u, i, || u.value().copied())?;
                    }
                    region.assign_fixed(
                        || "idx",
                        config.idx,
                        i,
                        || Value::known(pallas::Base::from(i as u64)),
                    )?;
                    let below = witness.map(|w| (i as u64) < w.offset);
                    let wi = witness.map(|w| w.w.get(i).copied().unwrap_or(pallas::Base::ZERO));
                    region.assign_advice(
                        || "lt",
                        config.lt,
                        i,
                        || below.map(|b| pallas::Base::from(u64::from(b))),
                    )?;
                    region.assign_advice(|| "w", config.w, i, || wi)?;
                    region.assign_advice(
                        || "w_inv",
                        config.w_inv,
                        i,
                        || wi.map(|w| w.invert().unwrap_or(pallas::Base::ZERO)),
                    )?;
                    let count = witness.map(|w| pallas::Base::from(w.offset.min(i as u64 + 1)));
                    acc_cell = Some(region.assign_advice(|| "acc", config.acc, i, || count)?);
                }
                Ok(acc_cell.expect("TRIES > 0"))
            },
        )?;

        // Bind: x = u + offset, y = 2h, then r·r⁻¹ = 1 on the next row.
        let (h_cell, r_cell) = layouter.assign_region(
            || "history bind",
            |mut region| {
                config.q_bind.enable(&mut region, 0)?;
                u.copy_advice(|| "u", &mut region, config.u, 0)?;
                count.copy_advice(|| "offset", &mut region, config.lt, 0)?;
                h_point
                    .inner()
                    .x()
                    .copy_advice(|| "x", &mut region, config.w, 0)?;
                h_point
                    .inner()
                    .y()
                    .copy_advice(|| "y", &mut region, config.w_inv, 0)?;
                let half = h_point
                    .inner()
                    .y()
                    .value()
                    .map(|y| *y * pallas::Base::from(2u64).invert().unwrap());
                let h_cell = region.assign_advice(|| "h", config.acc, 0, || half)?;
                let r = witness.map(|w| w.r);
                let r_cell = region.assign_advice(|| "r", config.lt, 1, || r)?;
                region.assign_advice(
                    || "r_inv",
                    config.w,
                    1,
                    || r.map(|r| r.invert().unwrap_or(pallas::Base::ZERO)),
                )?;
                Ok((h_cell, r_cell))
            },
        )?;

        // h < 2^253: 25 ten-bit words, then the top 3 bits.
        let running = config.ecc.lookup_config.copy_check(
            layouter.namespace(|| "h words"),
            h_cell,
            HALF_WORDS,
            false,
        )?;
        config.ecc.lookup_config.copy_short_check(
            layouter.namespace(|| "h top bits"),
            running[HALF_WORDS].clone(),
            HALF_TOP_BITS,
        )?;

        // One B = r·H for every domain; each domain j has Z_j = r·N_j with the
        // same r cell, and t_j = Poseidon(Poseidon(c_j, u), x(N_j)).
        let r_b = ScalarVar::from_base(ecc.clone(), layouter.namespace(|| "r for B"), &r_cell)?;
        let (b, _) = h_point.mul(layouter.namespace(|| "B = r·H"), r_b)?;
        let mut tags = Vec::with_capacity(domains.len());
        for (j, c) in domains.iter().enumerate() {
            let n_point = NonIdentityPoint::new(
                ecc.clone(),
                layouter.namespace(|| format!("N {j}")),
                witness.map(|w| w.n[j]),
            )?;
            let r_z = ScalarVar::from_base(
                ecc.clone(),
                layouter.namespace(|| format!("r for Z {j}")),
                &r_cell,
            )?;
            let (z, _) = n_point.mul(layouter.namespace(|| format!("Z {j} = r·N {j}")), r_z)?;
            let cu = self.poseidon2(layouter, &format!("t {j} inner"), [c.clone(), u.clone()])?;
            let t = self.poseidon2(layouter, &format!("t {j}"), [cu, n_point.inner().x()])?;
            tags.push(DomainTag {
                z_x: z.inner().x(),
                z_y: z.inner().y(),
                t,
            });
        }

        Ok(HistoryTagOutput {
            b_x: b.inner().x(),
            b_y: b.inner().y(),
            tags,
        })
    }
}

/// Instance values of an honest witness: `B` and, per domain `c_j`, `(Z_j, t_j)`.
pub fn tag_instances(
    domains: &[pallas::Base],
    u: pallas::Base,
    witness: &HistoryTagWitness,
) -> (pallas::Affine, Vec<(pallas::Affine, pallas::Base)>) {
    use group::Curve;
    let r = crate::history::blind_scalar(witness.r);
    let b = (pallas::Point::from(witness.h) * r).to_affine();
    let tags = domains
        .iter()
        .zip(&witness.n)
        .map(|(c, n)| {
            let z = (pallas::Point::from(*n) * r).to_affine();
            let nx = *n.coordinates().unwrap().x();
            (z, crate::poseidon::poseidon_hash_chain(&[*c, u, nx]))
        })
        .collect();
    (b, tags)
}

#[cfg(test)]
mod tests;
