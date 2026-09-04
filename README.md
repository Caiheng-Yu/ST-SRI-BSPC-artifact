# ST-SRI BSPC code-only artifact

Status: prepared for deposit in a dedicated public code-only repository.
Target journal: Biomedical Signal Processing and Control (BSPC).
Release scope: original project code only.

This package contains only the original project code, tests, launchers,
environment specifications, the MIT licence, and release metadata. It does
not contain NinaPro raw `.mat` recordings, processed E1/E2 signal arrays,
model checkpoints, result reports, derived tables, `.npz` curve or attribution
payloads, per-subject result directories, or per-window manifests.

The original code in this package is provided under the MIT License in
`LICENSE`. That licence does not apply to NinaPro data or any restricted
data-derived payload.

Users who need to run the code must obtain NinaPro DB2 through the official
route and satisfy its current access terms. The package does not provide the
dataset or any data-derived result payload.

Official DB2 instructions:
https://ninapro.hevs.ch/instructions/DB2.html

The related project repository is:
https://github.com/Caiheng-Yu/ST-SRI
It is the project source repository. This package is a separate code-only
artifact and must be published from a dedicated repository or release.

The public candidate contains no NinaPro-derived payload. No NinaPro data
licence is asserted here. See `metadata/` for the code-only release status,
provenance notes, citation metadata, and generated file manifest.

Before citing a deposited copy, use the repository tag and SHA-256 manifest
for that copy. The code-only package does not reproduce historical checkpoint
inference because checkpoints and signal-derived payloads are excluded.
