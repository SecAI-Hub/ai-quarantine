# Threat model

## Scope and assets

ai-quarantine decides whether hostile model artifacts may enter a trusted local
registry. Protected assets include registry integrity, registry credentials,
host confidentiality/availability, audit evidence, scanner integrity, and the
correct association between an artifact digest and its verdict.

The project does not claim to establish that a model is free of backdoors or
that its outputs are generally safe.

## Trust boundaries

- Artifact bytes, names, directory trees, source sidecars, model metadata,
  tokenizer/configuration files, and model-generated output are untrusted.
- Policy, hash locks, source allowlists, scanner binaries/images, registry
  service, audit HMAC key, and signing keys are administrative trust inputs.
- Third-party scanners and model runtimes are trusted for their verdict
  contract but treated as hostile-input parsers that may be exploitable.
- The quarantine drop location is untrusted. The registry storage and registry
  database are trusted and must not be writable by artifact submitters.
- The watcher is a single trusted coordinator, not a sandbox boundary.

## Threats and controls

| Threat | Primary controls |
|---|---|
| Source allowlist confusion/SSRF metadata | Canonical HTTPS origin/path matching; credentials, query, fragment, traversal, unsafe encoding, controls, and nonstandard ports rejected. |
| Format confusion or parser bombs | Strict safetensors layout checks; one bounded fail-closed GGUF metadata parser for templates/weights; descriptor-relative bounded directory traversal; strict JSON; only GGUF/safetensors accepted. |
| Malicious diffusion component loading | Exact allowlisted `[library, class]` tuples, allowlisted pipeline classes, required component configuration, required safetensors for weight-bearing components, and directory-marker-bound hashes. |
| Symlink, hard-link, FIFO/device, or replacement race | Descriptor-relative/no-follow single-link regular-file opens, special-file rejection, bounded link-free traversal, metadata revalidation, and pre/post-move hashes. |
| Unpinned remote artifact | Remote artifacts require both an allowlisted source and exact SHA-256 pin. |
| Untrusted local TOFU | Production policy disables local imports and requires pins; development may opt in. |
| Scanner missing/crash/timeout/partial report | Static schema-v2 verdicts distinguish pass/fail/not_applicable/unavailable/error; only validated formats can exempt an inapplicable parser, with zero coverage and passed=false. Applicable scanners require positive complete evidence; errors cannot become exemptions. Every behavioral prompt and a nonzero bounded Garak report remain required. |
| Scanner/runtime escape | Deployment must use a separate disposable worker with no egress, minimal read-only mounts, seccomp/namespaces or microVM isolation, and no secrets. |
| Unauthorized registry promotion | Mandatory bearer token by default; secure credential files; non-loopback plaintext HTTP rejected. |
| Partial promotion or trusted-name overwrite | Existing names are never overwritten; a durable pre-move marker enables rollback of incomplete staging; artifact, provenance, signatures, guard evidence, payload, and journal are hashed/fsynced/reverified before an idempotent API commit; fully staged failures replay from the pending journal. |
| Queue flooding, double processing, or permanent retry loops | Bounded streaming `scandir`, fixed-slot nonblocking claims, persistent exponential backoff, a three-attempt ceiling, private terminal hold, and bounded pending-commit replay. |
| Audit alteration/deletion | Verified append-only chain, restrictive no-follow I/O, fsync, rotation, optional HMAC entries and authenticated tail checkpoint, remote/WORM replication required to detect deletion of all co-located evidence. |
| Provenance overwrite/partial write | Atomic no-replace manifest creation and optional detached cosign signature. |
| CI/dependency compromise | Reviewed digest-pinned base image with no mutable OS-package build step, hash-locked Python dependencies, full-SHA actions, pip-audit/Bandit/Ruff/test/Trivy gates, root-owned read-only source, SBOM/provenance, and keyless image signing in release workflow. |
| Secret leakage or weak secret material | Credentials are not logged; token/HMAC files are no-follow single-link regular files owned by root/current UID with no group/other bits; values require 32–4096 printable non-whitespace bytes; source logs contain only a hostname. |

## Required deployment assumptions

- Quarantine and registry paths are distinct and non-overlapping.
- Artifact submitters cannot write registry storage, policy, locks, allowlists,
  scanner binaries, audit keys, or service credentials.
- One watcher process writes each audit chain.
- A fail-closed production profile is mounted and validated at startup.
- Required scanners are installed from a reviewed, immutable build.
- Behavioral tests run only in a genuinely isolated worker; setting
  `SECAI_BEHAVIOR_SANDBOXED=1` does not create isolation.
- Registry and audit storage failures are monitored and fail the workflow.
- Registry consumers use only committed API/database records and never infer
  trust from a filename present in registry storage. The registry honors the
  watcher-provided idempotency key.
- Signed checkpoints/archives are replicated outside the host.

## Residual risk

- Scanner applicability is not a complete admission compatibility proof.
  ModelAudit 0.2.52 and Fickling 0.1.12 identify the invoked tool version
  separately from their report schemas. Only the verified safe classification
  and complete applicable evidence can pass; no cache/whitelist downgrade or
  weaker report fallback is used. Unknown versions and incomplete/stacked
  unsupported reports remain nonpassing. Safe pickle tests do not authorize
  pickle admission. GGUF header coverage is not whole-tensor inspection;
  independent weight and gguf-guard gates remain required.
- Helper registry wire/lifecycle/idempotency compatibility remains unresolved;
  the deployment assumptions above are requirements, not implemented guarantees
  provided by the sibling registry.
- Novel scanner evasions, poisoned but correctly signed upstream models, and
  subtle weight-level backdoors remain possible.
- Native parser and model-runtime zero-days remain a high-impact risk.
- The durable watcher-side staging journal makes crashes recoverable, but the
  filesystem bundle and registry database are still not a server-owned atomic
  transaction; correctness depends on the committed-record and idempotency
  contract above.
- fs-verity, mTLS/workload identity, remote transparency anchoring, and
  mandatory signed verdict policy require additional integration.
- A compromised host or trusted administrator can defeat local controls.
- The local HMAC checkpoint does not replace independent remote retention.
- If both all logs and their co-located checkpoint are removed, the local node
  cannot prove that deletion occurred.

See [SECURITY_AUDIT.md](SECURITY_AUDIT.md) for priorities and the SecAI_OS
integration roadmap.
