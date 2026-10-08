// SPDX-License-Identifier: AGPL-3.0-only
//! Gadget D: Breach Bloom-Filter Non-Membership.
//!
//! Proves in zero-knowledge that a password is NOT in an embedded public Bloom
//! filter of breached passwords, without revealing the password.
//!
//! ## Soundness model (why each part exists)
//! 1. **Index binding** — the k bit-indices are derived from `Poseidon(password)`
//!    *in-circuit*. If indices were free, a breached (member) password could aim
//!    them at `0` bits and forge non-membership.
//! 2. **Sound reduction** — `hash` is bit-decomposed (255 bits) with a
//!    recomposition gate forcing the bits to recompose *exactly* to `hash`. The k
//!    indices are disjoint B-bit slices of the low bits.
//! 3. **Trusted lookup** — each `(index_i, bit_i)` is looked up in the public
//!    Bloom table, forcing `bit_i` to the true filter bit.
//! 4. **Non-membership = ∏ bit_i == 0** — a member has all k bits = 1 (product 1,
//!    rejected); a non-member has ≥1 zero bit (product 0, accepted). This is the
//!    corrected predicate (the original "all bits == 0" would reject ~99.9% of
//!    clean passwords).

use ff::PrimeField;
use halo2_gadgets::poseidon::{
    Hash as PoseidonHash, Pow5Chip, Pow5Config, primitives::ConstantLength,
};
use halo2_proofs::{
    circuit::{AssignedCell, Layouter, Value},
    plonk::{Advice, Column, ConstraintSystem, Error, Expression, Fixed, Selector, TableColumn},
    poly::Rotation,
};
use pasta_curves::pallas;

use crate::poseidon::{PoseidonSpec, bytes_to_field_elements, poseidon_hash_chain};

/// Bits decomposed from the password hash (Pallas base field is ~255 bits).
pub const HASH_BITS: usize = 255;

type Cell = AssignedCell<pallas::Base, pallas::Base>;

/// Compile-time Bloom filter parameters.
#[derive(Debug, Clone, Copy)]
pub struct BloomParams {
    /// log2 of the filter size. `m = 1 << index_bits`.
    pub index_bits: usize,
    /// Number of hash slices. Requires `k * index_bits <= HASH_BITS`.
    pub k: usize,
}

impl BloomParams {
    pub const fn m(&self) -> u64 {
        1u64 << self.index_bits
    }
}

/// Reference Bloom filter (off-circuit): builds the table and witnesses bits.
/// Index derivation MUST match the circuit: disjoint B-bit slices of the low
/// bits of `Poseidon(password)`.
#[derive(Clone)]
pub struct BloomFilter {
    pub params: BloomParams,
    pub bits: Vec<u8>,
}

impl BloomFilter {
    pub fn new(params: BloomParams) -> Self {
        Self {
            params,
            bits: vec![0u8; params.m() as usize],
        }
    }

    /// Derive the k indices from a hash (low B-bit slices, little-endian bits).
    pub fn indices(params: BloomParams, hash: pallas::Base) -> Vec<u64> {
        let repr = hash.to_repr(); // 32 bytes, little-endian
        let bit = |j: usize| -> u64 { ((repr[j / 8] >> (j % 8)) & 1) as u64 };
        (0..params.k)
            .map(|i| {
                let mut idx = 0u64;
                for l in 0..params.index_bits {
                    idx |= bit(i * params.index_bits + l) << l;
                }
                idx
            })
            .collect()
    }

    pub fn insert_hash(&mut self, hash: pallas::Base) {
        for idx in Self::indices(self.params, hash) {
            self.bits[idx as usize] = 1;
        }
    }

    /// Off-circuit membership probe (true = "maybe present", subject to the Bloom FP rate).
    pub fn maybe_contains_hash(&self, hash: pallas::Base) -> bool {
        Self::indices(self.params, hash)
            .into_iter()
            .all(|idx| self.bits[idx as usize] == 1)
    }

    /// Breach hash of a password, matching the in-circuit derivation in the
    /// combined `ZkppCircuit`: pad to `pad_len`, pack into 31-byte field
    /// elements, then Poseidon-chain them.
    pub fn breach_hash(password: &[u8], pad_len: usize) -> pallas::Base {
        let mut buf = vec![0u8; pad_len];
        let n = password.len().min(pad_len);
        buf[..n].copy_from_slice(&password[..n]);
        poseidon_hash_chain(&bytes_to_field_elements(&buf))
    }

    /// Build a filter from plaintext breached passwords (offline build pipeline).
    pub fn from_passwords(params: BloomParams, passwords: &[&[u8]], pad_len: usize) -> Self {
        let mut f = Self::new(params);
        for &pw in passwords {
            f.insert_hash(Self::breach_hash(pw, pad_len));
        }
        f
    }

    /// Membership probe for a plaintext password (matches the circuit's hashing).
    pub fn maybe_contains_password(&self, password: &[u8], pad_len: usize) -> bool {
        self.maybe_contains_hash(Self::breach_hash(password, pad_len))
    }

    /// Serialize the filter as a bit-packed bitfield (m bits → ceil(m/8) bytes)
    /// for embedding in the WASM binary.
    pub fn to_packed_bytes(&self) -> Vec<u8> {
        let m = self.bits.len();
        let mut out = vec![0u8; m.div_ceil(8)];
        for (i, &b) in self.bits.iter().enumerate() {
            if b != 0 {
                out[i / 8] |= 1u8 << (i % 8);
            }
        }
        out
    }

    /// Reconstruct a filter from a bit-packed bitfield.
    pub fn from_packed_bytes(params: BloomParams, data: &[u8]) -> Self {
        let m = params.m() as usize;
        assert!(data.len() >= m.div_ceil(8), "packed data too short for m");
        let mut bits = vec![0u8; m];
        for (i, slot) in bits.iter_mut().enumerate() {
            if (data[i / 8] >> (i % 8)) & 1 == 1 {
                *slot = 1;
            }
        }
        Self { params, bits }
    }
}

/// Configuration for Gadget D.
#[derive(Clone, Debug)]
pub struct BreachBloomConfig {
    params: BloomParams,
    poseidon_config: Pow5Config<pallas::Base, 3, 2>,
    poseidon_input: Column<Advice>,

    bit: Column<Advice>,
    pow: Column<Fixed>,
    acc: Column<Advice>,
    q_bit: Selector,
    q_acc_init: Selector,
    q_acc_step: Selector,

    ipow: Column<Fixed>,
    iacc: Column<Advice>,
    q_iacc_init: Selector,
    q_iacc_step: Selector,

    idx: Column<Advice>,
    lbit: Column<Advice>,
    active: Column<Advice>,
    table_idx: TableColumn,
    table_bit: TableColumn,
    prod: Column<Advice>,
    q_prod_init: Selector,
    q_prod_step: Selector,
    q_nonmember: Selector,
}

pub struct BreachBloomChip {
    config: BreachBloomConfig,
}

impl BreachBloomChip {
    pub fn construct(config: BreachBloomConfig) -> Self {
        Self { config }
    }

    pub fn configure(
        meta: &mut ConstraintSystem<pallas::Base>,
        params: BloomParams,
    ) -> BreachBloomConfig {
        assert!(
            params.k * params.index_bits <= HASH_BITS,
            "k * index_bits must fit in HASH_BITS"
        );

        let poseidon_input = meta.advice_column();
        let bit = meta.advice_column();
        let pow = meta.fixed_column();
        let acc = meta.advice_column();
        let ipow = meta.fixed_column();
        let iacc = meta.advice_column();
        let idx = meta.advice_column();
        let lbit = meta.advice_column();
        let active = meta.advice_column();
        let prod = meta.advice_column();

        meta.enable_equality(poseidon_input);
        meta.enable_equality(bit);
        meta.enable_equality(acc);
        meta.enable_equality(iacc);
        meta.enable_equality(idx);
        meta.enable_equality(lbit);
        meta.enable_equality(prod);

        let table_idx = meta.lookup_table_column();
        let table_bit = meta.lookup_table_column();

        // Poseidon chip.
        let state = (0..3).map(|_| meta.advice_column()).collect::<Vec<_>>();
        let partial_sbox = meta.advice_column();
        let rc_a = (0..3).map(|_| meta.fixed_column()).collect::<Vec<_>>();
        let rc_b = (0..3).map(|_| meta.fixed_column()).collect::<Vec<_>>();
        for col in state.iter().chain(std::iter::once(&partial_sbox)) {
            meta.enable_equality(*col);
        }
        meta.enable_constant(rc_b[0]);
        let poseidon_config = Pow5Chip::configure::<PoseidonSpec>(
            meta,
            state.try_into().unwrap(),
            partial_sbox,
            rc_a.try_into().unwrap(),
            rc_b.try_into().unwrap(),
        );

        let q_bit = meta.selector();
        let q_acc_init = meta.selector();
        let q_acc_step = meta.selector();
        let q_iacc_init = meta.selector();
        let q_iacc_step = meta.selector();
        let q_prod_init = meta.selector();
        let q_prod_step = meta.selector();
        let q_nonmember = meta.selector();

        let one = Expression::Constant(pallas::Base::one());

        meta.create_gate("bit_boolean", |meta| {
            let q = meta.query_selector(q_bit);
            let b = meta.query_advice(bit, Rotation::cur());
            vec![q * b.clone() * (one.clone() - b)]
        });
        meta.create_gate("acc_init", |meta| {
            let q = meta.query_selector(q_acc_init);
            let b = meta.query_advice(bit, Rotation::cur());
            let p = meta.query_fixed(pow);
            let a = meta.query_advice(acc, Rotation::cur());
            vec![q * (a - b * p)]
        });
        meta.create_gate("acc_step", |meta| {
            let q = meta.query_selector(q_acc_step);
            let b = meta.query_advice(bit, Rotation::cur());
            let p = meta.query_fixed(pow);
            let a_prev = meta.query_advice(acc, Rotation::prev());
            let a_cur = meta.query_advice(acc, Rotation::cur());
            vec![q * (a_cur - a_prev - b * p)]
        });
        meta.create_gate("iacc_init", |meta| {
            let q = meta.query_selector(q_iacc_init);
            let b = meta.query_advice(bit, Rotation::cur());
            let p = meta.query_fixed(ipow);
            let a = meta.query_advice(iacc, Rotation::cur());
            vec![q * (a - b * p)]
        });
        meta.create_gate("iacc_step", |meta| {
            let q = meta.query_selector(q_iacc_step);
            let b = meta.query_advice(bit, Rotation::cur());
            let p = meta.query_fixed(ipow);
            let a_prev = meta.query_advice(iacc, Rotation::prev());
            let a_cur = meta.query_advice(iacc, Rotation::cur());
            vec![q * (a_cur - a_prev - b * p)]
        });

        // Bloom lookup with conditional default (sentinel row (m, 0)).
        let m_const = Expression::Constant(pallas::Base::from(params.m()));
        meta.lookup(|meta| {
            let a = meta.query_advice(active, Rotation::cur());
            let i = meta.query_advice(idx, Rotation::cur());
            let b = meta.query_advice(lbit, Rotation::cur());
            let idx_val = a.clone() * i + (one.clone() - a.clone()) * m_const.clone();
            let bit_val = a * b;
            vec![(idx_val, table_idx), (bit_val, table_bit)]
        });

        meta.create_gate("prod_init", |meta| {
            let q = meta.query_selector(q_prod_init);
            let b = meta.query_advice(lbit, Rotation::cur());
            let p = meta.query_advice(prod, Rotation::cur());
            vec![q * (p - b)]
        });
        meta.create_gate("prod_step", |meta| {
            let q = meta.query_selector(q_prod_step);
            let b = meta.query_advice(lbit, Rotation::cur());
            let p_prev = meta.query_advice(prod, Rotation::prev());
            let p_cur = meta.query_advice(prod, Rotation::cur());
            vec![q * (p_cur - p_prev * b)]
        });
        meta.create_gate("nonmember", |meta| {
            let q = meta.query_selector(q_nonmember);
            let p = meta.query_advice(prod, Rotation::cur());
            vec![q * p]
        });

        BreachBloomConfig {
            params,
            poseidon_config,
            poseidon_input,
            bit,
            pow,
            acc,
            q_bit,
            q_acc_init,
            q_acc_step,
            ipow,
            iacc,
            q_iacc_init,
            q_iacc_step,
            idx,
            lbit,
            active,
            table_idx,
            table_bit,
            prod,
            q_prod_init,
            q_prod_step,
            q_nonmember,
        }
    }

    fn load_table(
        &self,
        layouter: &mut impl Layouter<pallas::Base>,
        filter: &BloomFilter,
    ) -> Result<(), Error> {
        let config = &self.config;
        let m = config.params.m() as usize;
        layouter.assign_table(
            || "bloom_table",
            |mut table| {
                for i in 0..m {
                    table.assign_cell(
                        || "t_idx",
                        config.table_idx,
                        i,
                        || Value::known(pallas::Base::from(i as u64)),
                    )?;
                    table.assign_cell(
                        || "t_bit",
                        config.table_bit,
                        i,
                        || Value::known(pallas::Base::from(filter.bits[i] as u64)),
                    )?;
                }
                // sentinel (m, 0)
                table.assign_cell(
                    || "t_idx_s",
                    config.table_idx,
                    m,
                    || Value::known(pallas::Base::from(m as u64)),
                )?;
                table.assign_cell(
                    || "t_bit_s",
                    config.table_bit,
                    m,
                    || Value::known(pallas::Base::zero()),
                )?;
                Ok(())
            },
        )
    }

    /// Prove `Poseidon(hash_inputs)` is NOT a Bloom member (non-membership).
    /// `hash_inputs.len()` must be >= 2.
    pub fn synthesize(
        &self,
        layouter: &mut impl Layouter<pallas::Base>,
        filter: &BloomFilter,
        input_cells: &[Cell],
    ) -> Result<(), Error> {
        let config = &self.config;
        let params = config.params;
        assert!(input_cells.len() >= 2, "need >= 2 hash inputs");

        self.load_table(layouter, filter)?;

        // ── 1. hash = Poseidon chain over the (copy-bound) inputs ──
        // Copy-constraining the inputs binds the breach hash to the SAME password
        // witnessed by the other gadgets (e.g. Gadget A's packed bytes).
        let mut bound: Vec<Cell> = Vec::with_capacity(input_cells.len());
        for (i, c) in input_cells.iter().enumerate() {
            let copied = layouter.assign_region(
                || format!("hin_{i}"),
                |mut region| c.copy_advice(|| "in", &mut region, config.poseidon_input, 0),
            )?;
            bound.push(copied);
        }
        let chip = Pow5Chip::construct(config.poseidon_config.clone());
        let hasher = PoseidonHash::<_, _, PoseidonSpec, ConstantLength<2>, 3, 2>::init(
            chip,
            layouter.namespace(|| "ph_init_0"),
        )?;
        let mut hash = hasher.hash(
            layouter.namespace(|| "ph_0"),
            [bound[0].clone(), bound[1].clone()],
        )?;
        for (i, c) in bound.iter().enumerate().skip(2) {
            let chip = Pow5Chip::construct(config.poseidon_config.clone());
            let hasher = PoseidonHash::<_, _, PoseidonSpec, ConstantLength<2>, 3, 2>::init(
                chip,
                layouter.namespace(|| format!("ph_init_{i}")),
            )?;
            hash = hasher.hash(layouter.namespace(|| format!("ph_{i}")), [hash, c.clone()])?;
        }
        let hash_val = hash.value().copied();

        // ── 2. bit-decompose hash; recomposition bound to hash ──
        let bit_cells = layouter.assign_region(
            || "hash_bits",
            |mut region| {
                let hash_in = hash.clone();
                let mut bit_cells: Vec<Cell> = Vec::with_capacity(HASH_BITS);
                let mut acc_prev = Value::known(pallas::Base::zero());
                let mut pow_j = pallas::Base::one();
                let mut last_acc: Option<Cell> = None;
                for j in 0..HASH_BITS {
                    config.q_bit.enable(&mut region, j)?;
                    region.assign_fixed(|| "pow", config.pow, j, || Value::known(pow_j))?;
                    let bit_v = hash_val.map(|h| {
                        let r = h.to_repr();
                        pallas::Base::from(((r[j / 8] >> (j % 8)) & 1) as u64)
                    });
                    let bc = region.assign_advice(|| "bit", config.bit, j, || bit_v)?;
                    bit_cells.push(bc);

                    let term = bit_v.map(|b| b * pow_j);
                    let acc_v = if j == 0 {
                        config.q_acc_init.enable(&mut region, j)?;
                        term
                    } else {
                        config.q_acc_step.enable(&mut region, j)?;
                        acc_prev.zip(term).map(|(a, t)| a + t)
                    };
                    let ac = region.assign_advice(|| "acc", config.acc, j, || acc_v)?;
                    acc_prev = acc_v;
                    pow_j = pow_j.double();
                    last_acc = Some(ac);
                }
                // recomposition == hash
                region.constrain_equal(last_acc.unwrap().cell(), hash_in.cell())?;
                Ok(bit_cells)
            },
        )?;

        // ── 3. assemble k indices (B-bit slices) ──
        let b = params.index_bits;
        let idx_cells = layouter.assign_region(
            || "index_assembly",
            |mut region| {
                let mut idx_cells: Vec<Cell> = Vec::with_capacity(params.k);
                let mut row = 0usize;
                for i in 0..params.k {
                    let mut iacc_prev = Value::known(pallas::Base::zero());
                    let mut ipow = pallas::Base::one();
                    let mut last_iacc: Option<Cell> = None;
                    for l in 0..b {
                        let src = &bit_cells[i * b + l];
                        let bc = src.copy_advice(|| "slice_bit", &mut region, config.bit, row)?;
                        region.assign_fixed(|| "ipow", config.ipow, row, || Value::known(ipow))?;
                        let term = bc.value().copied().map(|bv| bv * ipow);
                        let iacc_v = if l == 0 {
                            config.q_iacc_init.enable(&mut region, row)?;
                            term
                        } else {
                            config.q_iacc_step.enable(&mut region, row)?;
                            iacc_prev.zip(term).map(|(a, t)| a + t)
                        };
                        let ic = region.assign_advice(|| "iacc", config.iacc, row, || iacc_v)?;
                        iacc_prev = iacc_v;
                        ipow = ipow.double();
                        last_iacc = Some(ic);
                        row += 1;
                    }
                    idx_cells.push(last_iacc.unwrap());
                }
                Ok(idx_cells)
            },
        )?;

        // ── 4. lookup + product + non-membership ──
        layouter.assign_region(
            || "lookup_product",
            |mut region| {
                let mut prod_prev = Value::known(pallas::Base::zero());
                for i in 0..params.k {
                    region.assign_advice(
                        || "active",
                        config.active,
                        i,
                        || Value::known(pallas::Base::one()),
                    )?;
                    idx_cells[i].copy_advice(|| "idx", &mut region, config.idx, i)?;

                    let lbit_v = hash_val.map(|h| {
                        let indices = BloomFilter::indices(params, h);
                        pallas::Base::from(filter.bits[indices[i] as usize] as u64)
                    });
                    region.assign_advice(|| "lbit", config.lbit, i, || lbit_v)?;

                    let prod_v = if i == 0 {
                        config.q_prod_init.enable(&mut region, i)?;
                        lbit_v
                    } else {
                        config.q_prod_step.enable(&mut region, i)?;
                        prod_prev.zip(lbit_v).map(|(p, b)| p * b)
                    };
                    region.assign_advice(|| "prod", config.prod, i, || prod_v)?;
                    prod_prev = prod_v;
                }
                // non-membership: final product == 0
                config.q_nonmember.enable(&mut region, params.k - 1)?;
                Ok(())
            },
        )?;

        Ok(())
    }
}

#[cfg(test)]
mod tests;
