# BSPC code-only release status

Candidate version: `2026-09-04`
Release scope: original project code only
Target journal: Biomedical Signal Processing and Control (BSPC)

## Included

- Original project source code, tests, and launchers.
- Environment specifications and dependency instructions.
- The MIT licence for original code only.
- Release metadata and file hashes.

## Excluded

- NinaPro raw `.mat` recordings.
- Processed E1/E2 arrays.
- Model checkpoints.
- Result reports, derived tables, subject-level outputs, and all signal-derived
  binary payloads.

Users must obtain NinaPro DB2 through the official route:

https://ninapro.hevs.ch/instructions/DB2.html

The MIT licence does not authorize access to, reuse of, or redistribution of
NinaPro data. For this code-only release, contacting the NinaPro custodians is
not required before publication because no NinaPro data or data-derived result
payload is included. Any future release that adds such payloads must be
reviewed against the applicable NinaPro terms first.

Project source repository:

https://github.com/Caiheng-Yu/ST-SRI

The source repository commit containing the MIT file is recorded in
`metadata/PUBLIC_CANDIDATE_BUILD.json`.

Public code-only artifact repository:

https://github.com/Caiheng-Yu/ST-SRI-BSPC-artifact

Fixed release tag: `v2026.09.04-code-only-final`.
No DOI has been assigned. The ZIP SHA-256 is recorded alongside the ZIP in
the project workspace.
