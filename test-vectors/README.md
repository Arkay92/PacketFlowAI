# PFCASE 1.0 Verification Vectors

Each directory describes a portable conformance mutation and the control that must detect it. `tests/test_v5.py` constructs equivalent bundles and executes these vectors against the independent verifier.

| Vector | Required result |
| --- | --- |
| `valid-case` | All supplied cryptographic material verifies. |
| `modified-event` | File digest and Merkle root fail. |
| `missing-event` | Missing file or sequence/count commitment fails. |
| `broken-sequence` | `PF-SEQUENCE-1` is partial. |
| `invalid-signature` | Manifest signature fails. |
| `wrong-model` | Model artifact digest fails. |
| `missing-source` | `PF-COVERAGE-1` is partial. |
| `redacted-case` | Redaction remains declared and linked. |
| `bad-anchor` | External checkpoint fails. |
| `split-view-case` | Witness reconciliation reports split view. |

