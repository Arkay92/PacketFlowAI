# PacketFlow Assurance Claims

`VERIFIED` means the stated cryptographic or deterministic claim was reproduced. `COVERED` means a source declared by the applicable Evidence Contract contributed. `CONTINUOUS` means no gap exists within the stated producer/epoch sequence range. `UNKNOWN` means the available material cannot evaluate the property. `COMPLETE` is not a general evidence state.

| Claim | Meaning |
| --- | --- |
| `PF-INTEGRITY-1` | All supplied records match committed hashes. |
| `PF-INCLUSION-1` | Supplied records resolve through valid Merkle commitments. |
| `PF-SEQUENCE-1` | Continuity was evaluated for stated producer epoch ranges. |
| `PF-COVERAGE-1` | Contract-declared source contribution was measured. |
| `PF-ANCHOR-1` | The epoch root matches externally observed checkpoints. |
| `PF-REPRODUCE-1` | Deterministic local claims were independently re-derived. |

Assurance levels are cumulative shorthand, not completeness: A0 unverified, A1 integrity verified, A2 continuity verified, A3 source coverage evaluated, A4 independently witnessed, and A5 externally re-derivable. A5 is not proof that no unobserved event existed.

