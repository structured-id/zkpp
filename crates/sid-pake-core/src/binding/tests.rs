// SPDX-License-Identifier: AGPL-3.0-only
//! The operation context separates operations and requests, and its
//! transcript scalar moves with every byte of it.

use super::*;

/// Different operations or different requests give different contexts and
/// different transcript scalars; the same inputs give the same ones.
#[test]
fn the_context_names_one_operation_and_one_request() {
    let request = b"registration-request-bytes";
    let a = operation_context(&[0x11; 16], request);
    assert_eq!(a, operation_context(&[0x11; 16], request));
    assert_ne!(a, operation_context(&[0x22; 16], request));
    assert_ne!(a, operation_context(&[0x11; 16], b"another-request"));

    let scalar = transcript_context(&a);
    assert_eq!(scalar, transcript_context(&a));
    assert_ne!(
        scalar,
        transcript_context(&operation_context(&[0x22; 16], request))
    );
}

/// The operation id has a fixed length, so no split of the bytes between id
/// and request yields another operation's context.
#[test]
fn the_encoding_is_unambiguous() {
    let a = operation_context(&[0x01; 16], &[0x02, 0x03]);
    let mut shifted_id = [0x01; 16];
    shifted_id[15] = 0x02;
    let b = operation_context(&shifted_id, &[0x03]);
    assert_ne!(a, b);
}
