# ADR-0005: SBOM, signed releases, and blocking supply-chain audit

## Status
Proposed (target: Wave 6).

## Context
Current supply-chain posture:
- `requirements.lock` is committed (99 pinned deps) — good.
- `weekly-quality-scan.yml` runs `pip-audit` as *warn-only*.
- No SBOM is produced per release.
- No release artefact is signed; no provenance attestation.
- No PyInstaller dist is integrity-verified beyond filesystem hash
  by `model_loader.TrustStore` (model bundles only, not binaries).

A desktop tool that ingests customer data and ships PyInstaller
binaries needs a demonstrable chain of custody: which dependencies,
at which versions, built on which runner, signed by which identity.

## Decision
1. **SBOM (CycloneDX JSON)** generated at every tagged release by
   `cyclonedx-py` inside `quality-gates.yml` — attached to the
   GitHub release and stored alongside the wheel/dist.
2. **Signed releases via cosign keyless** (sigstore OIDC, no
   long-lived secrets) — produces a `.sig` and a transparency-log
   entry. Consumers verify with
   `cosign verify-blob --certificate-identity=<repo-tag-workflow>`.
3. **SLSA provenance attestation** at level 3 via
   `actions/attest-build-provenance` — ties the published binary to
   the exact commit and workflow run.
4. **`pip-audit` becomes blocking** on PRs (severity ≥ high fails
   the gate); findings below are reported but not blocking.
5. **PyInstaller build reproducibility**: pin Python minor
   (currently 3.11) and the PyInstaller version in
   `requirements.lock`; emit a build-info JSON next to the binary
   listing `python`, `pyinstaller`, `commit`, `timestamp`.

## Consequences
+ Customers can verify "this binary came from this commit", which
  is a precondition for any procurement conversation.
+ Upstream CVEs stop silently entering main; the PR that adds or
  bumps a vulnerable dep fails with a concrete SARIF hint.
+ SBOM diffs make dependency review routine instead of ad-hoc.
− Blocking `pip-audit` will occasionally block merges until an
  upstream fix or a documented ignore-with-expiry lands. Allow-list
  with expiry date in `.pip-audit-ignore.toml`.
− `cosign` needs a keyless identity provider; GitHub OIDC is free
  but requires `id-token: write` permission on the workflow.

## References
- `.github/workflows/weekly-quality-scan.yml` (current warn-only
  audit — target for the blocking gate)
- `.github/workflows/quality-gates.yml` (where SBOM step lands)
- `requirements.lock` (the input that SBOM describes)
- Roadmap: Wave 6 in `/root/.claude/plans/graceful-cooking-taco.md`
