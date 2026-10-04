# ST-SRI BSPC reproducibility artifact

Target journal: Biomedical Signal Processing and Control (BSPC).
Public artifact repository for the ST-SRI revision audit.
Current release: **v2026.09.18** (frozen evidence package).

## Where the revision material lives

| Material | Location |
| --- | --- |
| Code, experiment runners, split manifests, configuration records, SHA-256 manifests | Archived under DOI [10.5281/zenodo.22907764](https://doi.org/10.5281/zenodo.22907764) (release tag `v2026.09.18`) |
| Code and manifests bundle (223,494 bytes) | Release asset `ST_SRI_BSPC_code_and_manifests_v2026.09.18.zip` |
| Result bundle (269,688,179 bytes): detector reports, audit curves, stratified tables, comparison-method outputs | Release asset `ST_SRI_BSPC_results_v2026.09.18.zip` |
| Checkpoint bundle (1,372,748,916 bytes): all 269 trained checkpoints | Release asset `ST_SRI_BSPC_checkpoints_v2026.09.18.zip` |

## Repository contents at `main`

- `experiments/bspc_revision/` - training, windowing, detector, audit, and the test suite used by the revision.
- `common.py`, `environment.yml`, `requirements.txt` - shared definitions and the pinned environment specification.
- `metadata/` - release receipts for the 2026-09-04 code-only candidate: scope statement, build receipt, file SHA-256 manifest, and release checklist.
- `tools/verify_public_candidate.py` - verifies a downloaded candidate against the published SHA-256 manifest.
- `LICENSE` (MIT, original code only), `CITATION.cff`.

## Releases

### `v2026.09.18` - frozen evidence package (current)

| Asset | Size (bytes) | SHA-256 |
| --- | --- | --- |
| `ST_SRI_BSPC_code_and_manifests_v2026.09.18.zip` | 223,494 | `7B9B18C9DBA46751A71007BAF68BF0B7269CDC9247723F46432864ACE31CE08C` |
| `ST_SRI_BSPC_results_v2026.09.18.zip` | 269,688,179 | `7375D9FA190A612E2980BCBB80E559FD62A906B7F13AF7ABBA21B265C01BEEBF` |
| `ST_SRI_BSPC_checkpoints_v2026.09.18.zip` | 1,372,748,916 | `4B3FDE542FE9DB9376B3EB9B42A9B2D77E06465C6E5D7D56E8CB6EC5C082A290` |

The code and manifests bundle carries the source code, experiment runners, environment specifications, split manifests, and the SHA-256 manifests. The result bundle carries the detector reports, audit curves, stratified tables, and comparison-method outputs. The checkpoint bundle carries all 269 trained models: R005b (40), label-shuffle (40), R012 pilot (6), R015b (3 seeds x 40 = 120), and the remaining protocol checkpoints. Checkpoint integrity was verified against the source SHA-256 manifest on 2026-09-18 with zero failures.

The tag `v2026.09.18-frozen-evidence` points to the same commit and carries the same three assets.

### `v2026.09.04-code-only-final` - code-only candidate

One asset, `ST_SRI_BSPC_code_only_v2026.09.04-final.zip` (156,765 bytes): original code, tests, launchers, environment specifications, MIT licence, citation metadata, and the SHA-256 file manifest. This candidate contains no NinaPro-derived payload; it remains the licence-clean option for readers who need the code only. `v2026.09.04-code-only` is the first code-only archive.

## Excluded from every release

NinaPro DB2 raw recordings and the derived E1/E2 arrays (9.0 GiB). Obtain them through the official route:

https://ninapro.hevs.ch/instructions/DB2.html

## Licence boundary

The MIT licence in `LICENSE` covers the original project code only. It does not extend to NinaPro data or to any restricted data-derived payload. The frozen evidence release does contain data-derived outputs (results, curves, stratified tables, and trained checkpoints), so their use remains subject to the current NinaPro access terms. The code-only candidate asserts no rights over NinaPro material because it contains none.

## Verification

1. Download an asset together with the SHA-256 values listed above.
2. Recompute the digest, for example `certutil -hashfile <asset>.zip SHA256` on Windows or `sha256sum <asset>.zip` on Linux and macOS.
3. Optionally run `python tools/verify_public_candidate.py` against the code-only candidate.

Cite the fixed release tag `v2026.09.18` and DOI 10.5281/zenodo.22907764 when reusing this material; citation metadata is in `CITATION.cff`.

## Related repository

https://github.com/Caiheng-Yu/ST-SRI - project source repository.
