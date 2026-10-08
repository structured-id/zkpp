// SPDX-License-Identifier: AGPL-3.0-only
//! Cross-check vectors for the TypeScript OPAQUE-on-Pallas client: full
//! registration and login runs of the reference client against a reference
//! server, with every byte the client drew from its RNG recorded in draw
//! order. The TypeScript client replays that byte stream and must produce the
//! same messages, state and keys.
//!
//! `cargo run -p sid-pake-core --example opaque_vectors -- opaque-vectors.json`

use std::convert::Infallible;

use rand::{SeedableRng, rngs::StdRng};
use rand_core::{Rng, TryCryptoRng, TryRng};
use sid_opaque_ke::{
    ClientLogin, ClientLoginFinishParameters, ClientRegistration,
    ClientRegistrationFinishParameters, ServerLogin, ServerLoginParameters, ServerRegistration,
    ServerSetup,
};
use sid_pake_core::pallas_opaque::PallasCipherSuite;

/// An RNG that records every byte it hands out, in order.
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

impl TryCryptoRng for Recording {}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn recording(seed: u64) -> Recording {
    Recording {
        inner: StdRng::seed_from_u64(seed),
        drawn: Vec::new(),
    }
}

/// One registration and one login of `password`, as a JSON object.
fn vector(case: u64, password: &[u8], login_password: &[u8]) -> String {
    let mut server_rng = StdRng::seed_from_u64(1_000 + case);
    let setup = ServerSetup::<PallasCipherSuite>::new(&mut server_rng);

    let mut reg_rng = recording(2_000 + case);
    let start = ClientRegistration::<PallasCipherSuite>::start(&mut reg_rng, password)
        .expect("registration start");
    let start_drawn = std::mem::take(&mut reg_rng.drawn);
    let response = ServerRegistration::<PallasCipherSuite>::start(
        &setup,
        start.message.clone(),
        b"credential",
    )
    .expect("server registration start");
    let finished = start
        .state
        .clone()
        .finish(
            &mut reg_rng,
            password,
            response.message.clone(),
            ClientRegistrationFinishParameters::default(),
        )
        .expect("registration finish");
    let finish_drawn = std::mem::take(&mut reg_rng.drawn);
    let file = ServerRegistration::<PallasCipherSuite>::finish(finished.message.clone());

    let mut login_rng = recording(3_000 + case);
    let login = ClientLogin::<PallasCipherSuite>::start(&mut login_rng, login_password)
        .expect("login start");
    let login_drawn = std::mem::take(&mut login_rng.drawn);
    let server_login = ServerLogin::<PallasCipherSuite>::start(
        &mut server_rng,
        &setup,
        Some(file),
        login.message.clone(),
        b"credential",
        ServerLoginParameters::default(),
    )
    .expect("server login start");
    let login_finished = login.state.clone().finish(
        &mut login_rng,
        login_password,
        server_login.message.clone(),
        ClientLoginFinishParameters::default(),
    );
    let outcome = match login_finished {
        Ok(f) => format!(
            "\"finalization\":\"{}\",\"sessionKey\":\"{}\",\"exportKey\":\"{}\",\"registrationExportKey\":\"{}\"",
            hex(&f.message.serialize()),
            hex(&f.session_key),
            hex(&f.export_key),
            hex(&finished.export_key),
        ),
        Err(e) => format!("\"loginError\":\"{e}\""),
    };
    format!(
        "{{\"password\":\"{}\",\"loginPassword\":\"{}\",\
         \"registrationStartDrawn\":\"{}\",\"registrationRequest\":\"{}\",\"registrationState\":\"{}\",\
         \"registrationResponse\":\"{}\",\"registrationFinishDrawn\":\"{}\",\"registrationRecord\":\"{}\",\
         \"loginStartDrawn\":\"{}\",\"credentialRequest\":\"{}\",\"loginState\":\"{}\",\
         \"credentialResponse\":\"{}\",{outcome}}}",
        hex(password),
        hex(login_password),
        hex(&start_drawn),
        hex(&start.message.serialize()),
        hex(&start.state.serialize()),
        hex(&response.message.serialize()),
        hex(&finish_drawn),
        hex(&finished.message.serialize()),
        hex(&login_drawn),
        hex(&login.message.serialize()),
        hex(&login.state.serialize()),
        hex(&server_login.message.serialize()),
    )
}

fn main() {
    let long = vec![b'a'; 130];
    let cases: Vec<String> = vec![
        vector(0, b"Str0ngP@ssword!", b"Str0ngP@ssword!"),
        vector(
            1,
            "пароль-Ünïcode-1".as_bytes(),
            "пароль-Ünïcode-1".as_bytes(),
        ),
        vector(2, &long, &long),
        vector(3, b"Correct-Horse-9", b"Wrong-Horse-9"),
    ];
    let out = std::env::args()
        .nth(1)
        .expect("usage: opaque_vectors <output.json>");
    std::fs::write(&out, format!("[{}]\n", cases.join(",\n"))).expect("write the vectors");
}
