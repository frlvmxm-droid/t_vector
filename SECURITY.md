# Security Policy

## Reporting a Vulnerability

**Do not open a public GitHub issue for security vulnerabilities.**

If you discover a vulnerability in BankReasonTrainer, please report it privately
via GitHub's private vulnerability reporting feature:

1. Navigate to the repository's **Security** tab.
2. Click **Report a vulnerability**.
3. Include:
   - affected component (module path, function, line numbers if known);
   - reproduction steps or proof-of-concept;
   - assessed impact (confidentiality / integrity / availability);
   - environment details (OS, Python version, commit SHA).

If you cannot use GitHub's reporting flow, contact the maintainers listed in
`CODEOWNERS` via their GitHub-registered email addresses.

## Response SLA

| Severity | Acknowledgement | Mitigation target |
|---|---|---|
| Critical (RCE, secret disclosure, arbitrary file write) | 2 business days | 7 days |
| High (privilege escalation, SSRF bypass, auth bypass) | 3 business days | 14 days |
| Medium (DoS, info leak, weak crypto) | 5 business days | 30 days |
| Low (hardening, defence-in-depth) | 10 business days | Next release |

Severities map to CVSS v3.1 bands (Critical ≥ 9.0, High 7.0–8.9, Medium 4.0–6.9,
Low < 4.0).

## Supported Versions

Only the latest commit on `main` is supported for security fixes. Tagged
releases receive patches on a best-effort basis for six months after release.

## Security Features Currently Enforced

The following mitigations are wired in and verified by tests — report any way to
circumvent them as a vulnerability:

- **Model trust-store** (`model_loader.py`): `.joblib` files are only loaded
  after streaming SHA-256 verification. An architecture test
  (`tests/test_architecture_boundaries.py`) ensures `joblib.load` is not called
  from any module other than `model_loader.py`.
- **Secret encryption** (`llm_key_store.py`): LLM API keys are encrypted with
  Fernet (AES-128-CBC + HMAC-SHA-256). Key rotation is supported via the
  `LLM_SNAPSHOT_KEY` environment variable. Default mode is `fail_closed`.
- **SSRF guard** (`llm_client.py`): LLM endpoints are constrained by an
  allowlist. DNS is re-resolved after URL validation (TOCTOU hardening);
  private, loopback, link-local, and multicast IPs are rejected.
- **Circuit breaker** (`llm_client.py`): 3 consecutive failures open the
  breaker for 90 seconds to prevent error amplification.
- **Size limits**: `excel_utils.py` caps XLSX input at `MAX_XLSX_BYTES` to
  prevent ZIP-bomb OOM; `entity_normalizer.py` caps per-string length before
  regex normalisation to mitigate catastrophic backtracking.
- **Path safety** (`model_loader.py:is_safe_path`): resolves symlinks and
  rejects paths escaping an allowed root; callers re-check after resolution
  to mitigate remaining TOCTOU windows.

## Out of Scope

- Vulnerabilities requiring a malicious local shell or Python interpreter
  already executing on the user's machine (this is a desktop application —
  local code execution is the security boundary).
- Denial-of-service by submitting genuinely large legitimate datasets; use
  OS-level resource limits for multi-tenant scenarios.
- Issues in third-party Python packages — please report those upstream.

## Safe Harbour

We will not pursue legal action against researchers who:

- act in good faith and avoid privacy violations, data destruction, or service
  disruption;
- report findings promptly and do not publicly disclose before a fix is
  shipped (coordinated disclosure);
- test only against their own installations.
