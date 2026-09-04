# Code-only release checklist

Scope: publish only original code under the MIT License.

Completed locally:

- NinaPro raw recordings, processed arrays, checkpoints, result files, and derived signal payloads excluded.
- Official NinaPro DB2 access route recorded in `README.md`.
- `LICENSE` and `CITATION.cff` included.
- SHA-256 manifest generated for every included file.
- Python and JSON integrity checks passed for the local candidate.
- The project source repository contains the MIT license at commit `3f8e3d021ed041bee144e6c9206110cf1f5e7bcb`.

External release actions still required:

1. Create a separate public repository for this code-only directory.
2. Push the directory and create an immutable tag such as `v2026.09.04`.
3. Record the final repository URL, tag, and ZIP SHA-256 in the manuscript and response letter.
4. Optionally mirror the tagged ZIP to Zenodo to obtain a DOI; do not describe a DOI as available until it is issued.

The code-only release does not authorize access to, reuse of, or redistribution
of NinaPro data. Users must obtain DB2 through the official NinaPro route and
follow the provider's current terms.
