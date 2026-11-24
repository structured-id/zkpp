// SPDX-License-Identifier: AGPL-3.0-only
//! Key generation for OPAQUE-ZKPP proofs.
//!
//! Generates structured reference string (SRS) parameters and
//! proving/verifying keys for the combined ZKPP circuit.
//!
//! Uses Vesta curve for IPA commitments because the circuit operates
//! over pallas::Base (= Fp), which equals vesta::Scalar.
//!
//! Supports params caching: SRS params are deterministic for a given k,
//! so they can be generated once and loaded from disk on subsequent starts.
//! VK is regenerated from cached params (fast: ~100ms vs ~5s for params).

use halo2_proofs::{
    plonk::{Error, ProvingKey, VerifyingKey},
    poly::commitment::Params,
};
use pasta_curves::vesta;
use std::io;
use std::path::Path;

use crate::circuit::{CircuitShape, ZkppCircuit};
use crate::types::PolicyError;

/// Generate SRS parameters for a given circuit size.
///
/// `k` determines the maximum number of rows: 2^k.
/// For the combined ZKPP circuit, use `ZKPP_K`.
pub fn generate_params(k: u32) -> Params<vesta::Affine> {
    Params::new(k)
}

/// Generate the proving key for `shape`.
///
/// The policy minimums are fixed columns and the domain count sets the layout,
/// so the key (and the verifying key it embeds, `pk.get_vk()`) proves and
/// verifies only for this shape.
pub fn generate_pk(
    params: &Params<vesta::Affine>,
    shape: CircuitShape,
) -> Result<ProvingKey<vesta::Affine>, KeygenError> {
    generate_pk_with(params, shape, |_| {})
}

/// A boundary of proving-key generation, for a progress display.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeygenStep {
    /// The verifying key is built; the proving key follows.
    VerifyingKey,
    /// The proving key is built.
    ProvingKey,
}

/// Why no key was generated for a shape.
#[derive(Debug, thiserror::Error)]
pub enum KeygenError {
    /// The shape's policy is one the circuit cannot hold; a key for it would
    /// enforce something else than the policy says.
    #[error(transparent)]
    Policy(#[from] PolicyError),
    /// The proving system refused the circuit.
    #[error("key generation failed: {0:?}")]
    Circuit(Error),
}

/// The circuit of `shape`, once its policy is one the circuit can hold.
fn circuit_of(shape: CircuitShape) -> Result<ZkppCircuit, KeygenError> {
    shape.policy.check()?;
    Ok(ZkppCircuit::for_shape(shape))
}

/// [`generate_pk`], calling `on_step` as each key is finished.
pub fn generate_pk_with(
    params: &Params<vesta::Affine>,
    shape: CircuitShape,
    mut on_step: impl FnMut(KeygenStep),
) -> Result<ProvingKey<vesta::Affine>, KeygenError> {
    let vk = generate_vk(params, shape)?;
    on_step(KeygenStep::VerifyingKey);
    let pk = ZkppCircuit::for_shape(shape)
        .keygen_pk(params, vk)
        .map_err(KeygenError::Circuit)?;
    on_step(KeygenStep::ProvingKey);
    Ok(pk)
}

/// Generate the verifying key for `shape` only (faster than full PK).
pub fn generate_vk(
    params: &Params<vesta::Affine>,
    shape: CircuitShape,
) -> Result<VerifyingKey<vesta::Affine>, KeygenError> {
    circuit_of(shape)?
        .keygen_vk(params)
        .map_err(KeygenError::Circuit)
}

/// Write SRS params to a writer (binary format).
pub fn write_params<W: io::Write>(
    params: &Params<vesta::Affine>,
    writer: &mut W,
) -> io::Result<()> {
    params.write(writer)
}

/// Read SRS params from a reader (binary format).
pub fn read_params<R: io::Read>(reader: &mut R) -> io::Result<Params<vesta::Affine>> {
    Params::read(reader)
}

/// Save SRS params to cache directory.
///
/// VK is not cached (halo2 v0.3 has no VK serialization);
/// it's regenerated from params on load (~100ms).
pub fn save_params_cache(dir: &Path, params: &Params<vesta::Affine>) -> io::Result<()> {
    std::fs::create_dir_all(dir)?;
    let mut f = std::fs::File::create(dir.join("zkpp_params.bin"))?;
    write_params(params, &mut f)
}

/// Load cached params and regenerate the VK for `shape`.
///
/// Returns None if cache file doesn't exist or is corrupted.
pub fn load_from_cache(
    dir: &Path,
    shape: CircuitShape,
) -> Option<(Params<vesta::Affine>, VerifyingKey<vesta::Affine>)> {
    let params_path = dir.join("zkpp_params.bin");
    if !params_path.exists() {
        return None;
    }

    let mut f = std::fs::File::open(&params_path).ok()?;
    let params = read_params(&mut f).ok()?;
    let vk = generate_vk(&params, shape).ok()?;

    Some((params, vk))
}

#[cfg(test)]
mod tests;
