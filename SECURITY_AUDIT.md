# Security and production-readiness audit

Audit date: 2026-08-02
Scope: repository source, tests, policy examples, dependency locks, container,
and GitHub Actions workflows on the audited commit and this working tree.

This is a source-assisted engineering audit, not a formal certification or a
claim that arbitrary models are safe.

## Executive result

The original implementation had several production-blocking trust-boundary
gaps: string-prefix source allowlisting, incomplete safetensors validation,
fail-open GGUF parsing, link/TOCTOU exposure, optional registry
authentication, partial promotions left in the trusted directory, a
fail-open audit writer, and mutable best-effort scanner installation in the
container.

The high-confidence remediations below are implemented and covered by the
regression suite. The standalone project is materially safer as an admission
controller, but native scanners and behavioral inference still require the
separate, resource-limited scanner worker model already used by SecAI_OS.

## Remediated findings

| Severity | Finding | Resolution |
|---|---|---|
| High | URL `startswith` checks allowed hostname and path-prefix confusion. | Canonical HTTPS origin and path-segment matching; credentials, query, fragment, traversal, controls, backslashes, and nonstandard ports are rejected. |
| High | Safetensors validation checked only a length and opening brace. | Strict UTF-8/JSON decoding, duplicate-key and complexity rejection, descriptor/schema validation, dtype/shape/offset checks, and exact non-overlapping buffer coverage. |
| High | Malformed or unreadable GGUF metadata passed the chat-template gate, while weight analysis duplicated that parser and could fail open. | Template and weight analysis now share bounded exact-read metadata helpers with count/key/string/array/rank/name/alignment limits, duplicate rejection, and fail-closed parsing. |
| High | Files, sidecars, and directory walks could follow links or accept special/hard-linked files. | Descriptor-relative/no-follow single-link regular-file opens, bounded link-free tree enumeration, directory-marker-bound hashes, and mutation checks around scanning/move. |
| High | External scanner success ignored ModelScan errors/skips/zero coverage, treated missing Fickling safety as safe, and treated missing ModelAudit issues as clean. | Versioned explicit integration verdicts; ModelScan requires a consistent versioned report, zero errors/skips, positive requested-artifact coverage, and no issues; Fickling/ModelAudit require their explicit fields. |
| High | Behavioral request failures became empty strings, undrained server pipes could block, and a zero-probe Garak run passed. | Every configured prompt requires a bounded non-empty schema-valid response with distinct failure codes; server logs use a sink; Garak must emit exactly one bounded JSONL report with nonzero validated evaluations. |
| High | Diffusion validation used mutable path walks, raw-text regexes, permissive component tuples, and did not require component evidence. | Descriptor-relative bounded traversal, strict duplicate-free JSON, parsed-string checks, allowlisted pipeline/component tuples, component configs, required weight files, and empty-directory hash markers. |
| High | Registry authentication was optional and an unreadable configured token silently degraded to no auth. | Secure bounded token reads; authentication is mandatory. Only an explicit loopback development override exists. |
| High | Promotion called the registry before provenance/evidence was durable, so failures and crashes could leave an unbound or unrecoverable filesystem/API state. | A staging marker is durable before the move; artifact and all evidence are hashed, fsynced, journaled, and reverified before any network call; the API commit is last and carries a stable idempotency key; startup rolls incomplete staging back and replays fully pending commits. |
| High | Container builds installed unpinned scanners, ignored installation failures, and left application source writable by the runtime account. | Mutable scanner and OS-package installs were removed; the reviewed base is digest-pinned; dependencies are hash locked; application directories/files are root-owned 0555/0444; runtime uses UID/GID 65532; CI/release apply a fixable high/critical image gate. Production derived images must add reviewed scanners and fail closed. |
| High | Audit writes swallowed I/O failures and resumed from an unverified tail. | Existing chains are verified before append; secure append uses no-follow/O_APPEND, restrictive modes, flush/fsync, and propagates failures. |
| Medium | A plain hash chain could not detect tail or full-log deletion. | Optional HMAC entries plus an authenticated checkpoint detect retained-entry modification and tail/archive/full-log deletion while the checkpoint survives. |
| Medium | Directory trees and JSON/source/policy sidecars were unbounded. | File/entry/depth/path/per-file/total-size limits and bounded sidecar/JSON reads were added. |
| Medium | Production policy keys were advertised but not enforced. | Examples now contain only evaluated keys; local imports, local pins, allowed formats, scanner fail mode, diffusion directories, and behavioral thresholds are enforced. |
| Medium | Required behavioral testing silently passed without llama-server. | Required mode fails when isolation is not acknowledged or llama-server is unavailable; once Garak is installed, execution/report failures fail rather than becoming optional skips. |
| Medium | Provenance writes could follow/overwrite paths or leave partial JSON. | New provenance files are created atomically without replacement at mode 0600. |
| Medium | Queue enumeration materialized/sorted attacker-controlled directories, work could be processed twice, and permanently failing junk retried forever. | Streaming bounded inspection/batches, a fixed 256-slot nonblocking claim set, durable exponential backoff, three-attempt ceiling, terminal hold, orphan/special-entry handling, and bounded promotion-journal replay. |
| Medium | Registry and HMAC secrets accepted weak/control-bearing data or files with unsafe ownership/permissions. | Both require 32–4096 printable ASCII non-whitespace bytes and no-follow, single-link, owner-only files owned by root or the current service UID. |
| Medium | CI installed mutable dependencies and performed syntax-only linting, while tag publication could bypass the complete gate set, publish a revision outside `main`, package a mismatched version, or collapse distinct SemVer build metadata into one OCI tag. | Hash-locked CI/release installation, Gitleaks, tests, Ruff, pip-audit, Bandit medium/high, immutable action pins, package build, container build/invariant checks, Trivy, and package/container attestations now gate publication; the workflow requires an annotated strict-SemVer tag on `main`, exact tag/project/package-version equality, and a collision-free immutable OCI tag. |
| Low | The unused Watchdog runtime dependency increased attack surface. | Removed from the runtime dependency set. |
| Low | CI did not scan repository history/worktrees for accidentally committed credentials. | Added an immutable-pinned Gitleaks gate with a narrow allowlist for the scanner's inert behavioral canary strings. |

## Validation performed

- Python 3.12 project-local environment.
- 152 tests passing after hardening, including transaction restart/replay,
  evidence-before-commit, bounded queue/claim/backoff/terminal-hold, and strict
  secret regressions.
- Ruff correctness/import gate passing.
- Bandit: no medium/high findings.
- `pip-audit`: no known vulnerabilities in runtime or development locks.
- Python bytecode compilation passing.
- Workflow action references checked for full 40-character commit pins.
- Gitleaks history/worktree scans pass with only the exact documented inert
  behavioral canary line allowlisted.
- Container image built successfully from the current digest-pinned Python
  base without mutable OS-package resolution, with hash verification, non-root execution,
  root-owned read-only application source, and bytecode compilation.
- Trivy reports zero fixable high/critical findings in the final core image.

## Residual risks

| Severity | Residual risk | Required production control |
|---|---|---|
| High | Native and third-party model parsers may have memory-safety, parser-complexity, or zero-day flaws. | Run each scanner in a disposable, no-network, read-only, resource-limited worker; never in the registry service process. |
| High | Behavioral tests invoke a complex model runtime. An environment flag is an assertion, not isolation. | Use SecAI_OS scanner broker/worker isolation (namespaces/seccomp/cgroups or a microVM) and set the flag only inside that deployment. |
| High | Static/behavioral analysis cannot prove absence of backdoors or unsafe behavior. | Curated publishers, immutable revisions, independent signatures, reproducible provenance, human approval for high-impact models, and ongoing monitoring. |
| Medium | Watcher-side staging is durable and replayable, but registry storage and its database are not one server-owned transaction. Storage faults can still prevent recovery. | Consumers must trust only committed records, the API must enforce idempotency, operators must alert on pending/invalid journals, and a registry-owned staging/commit API can further reduce coupling. |
| Medium | Core GGUF parsing validates the bounded tensor table and supported sampled tensor spans, but full quantization-specific block semantics still depend on gguf-guard. | Require a pinned gguf-guard in production and keep both parser fuzzing corpora current. |
| Medium | fs-verity and provenance signing remain optional capabilities. | Add explicit policy requirements and startup self-tests for both where the filesystem supports them. |
| Medium | HMAC checkpoints remain local state and the logger supports one writer only; deletion of logs plus their co-located checkpoint cannot be proven locally. | Forward signed checkpoints/archives to WORM or remote append-only storage and enforce a single writer. |
| Medium | Optional scanner extras are not part of the minimal runtime lock. | Produce a separate reviewed scanner image/lock/SBOM and run pip-audit plus container CVE scanning for that exact image. |
| Low | Registry API authentication is bearer-token based. | Add workload identity or mTLS, scoped short-lived credentials, rotation, and explicit audience binding. |

## Comparison with SecAI_OS

SecAI_OS contains a substantially more complete quarantine subsystem with a
scanner broker/worker boundary, staging/claim semantics, broader parser
limits, and appliance-level sandbox controls. The standalone repository
should remain a small reusable core and consume shared hardened components,
not evolve a second incompatible security implementation.

Recommended consolidation:

1. Extract canonical source, secure-file, bounded-tree, safetensors, and audit
   primitives into a versioned shared package used by both projects.
2. Make the SecAI_OS scanner broker protocol the standalone production mode.
3. Preserve the shared committed-record/idempotency contract and consider
   moving staging ownership into the registry for tighter atomicity.
4. Share policy schemas and compatibility tests between repositories.

## Prioritized production roadmap

### P0 — before production

- Ship a hash-locked scanner worker image with SBOM, signatures, CVE gates,
  memory/CPU/PID/time limits, no egress, and read-only mounts.
- Require HMAC audit checkpoints, remote archive replication, service-token
  rotation, and startup self-tests.
- Add fuzz targets/corpora for URL canonicalization, safetensors, GGUF metadata,
  policy YAML, and directory-tree edge cases.
- Add integration tests using the real registry API and fault injection at
  every promotion transition.
- Version and enforce the registry idempotency/committed-record contract, and
  alert on pending/invalid journals and terminal holds.

### P1 — production operations

- Publish a JSON Schema for policy and reject unknown keys.
- Add structured state/status APIs: queued, claimed, scanning, rejected,
  staged, committed, rollback-required.
- Add privacy-safe metrics and health/readiness endpoints without filenames,
  model contents, source tokens, or raw scanner output.
- Add signed DSSE/in-toto verdicts, immutable source revision evidence,
  scanner image digests, policy digest, and transparency-log anchoring.
- Add explicit policies for required fs-verity, signed provenance, scanner
  versions, minimum signature identities, and rejection retention.

### P2 — feature improvements

- Encrypted, access-controlled rejection retention for forensic workflows.
- Admin review/override with dual authorization and an immutable audit event.
- Registry deduplication by digest and resumable chunked ingestion.
- Model inventory, expiration/re-scan scheduling, CVE/advisory-triggered
  quarantine, and revocation propagation into SecAI_OS runtimes.
- A local operator UI showing bounded evidence and remediation actions while
  never rendering attacker-controlled HTML or raw terminal escape sequences.
