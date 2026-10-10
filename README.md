# ai-quarantine

AI model artifact admission control for GGUF, safetensors, and bounded
Diffusers-style directories. The watcher scans an artifact in quarantine,
durably stages the artifact and its evidence, verifies the staged bundle, and
then performs an authenticated idempotent registry commit. Registry consumers
must treat only committed API/database records as trusted; filesystem presence
alone is never an admission verdict.

For the standalone helper admission v2 client, provision its registry-admission
credential with the registry's `promote` scope. `AdmissionClient` uses that one
credential for capability preflight, prepare, commit and exact single-model
receipt lookup; it does not require the administrator credential. Read and
verify credentials cannot admit artifacts. Legacy unscoped deployments retain
their shared service key; that mode provides no scoped authority separation.

The watcher selects its path with `QUARANTINE_ADMISSION_MODE`: `legacy` is the
default, and `helper-v2-required` selects the v2 controller. Unknown values are
refused. The v2 path is currently unavailable: its public entry points refuse
before parsing, creating journals, copying artifacts or preparing registry
transactions because the required sandbox probes are not yet implemented and
qualified. No flag or receipt file can bypass that boundary. Source tests cover
the controller's exact request replay and retained transaction state; they do
not establish operational scanner isolation or enable deployment.

## Security properties

The core pipeline performs seven ordered gates:

1. Canonical HTTPS source allowlisting.
2. Strict format validation.
3. SHA-256 pin enforcement.
4. Available provenance checks.
5. Static scanners and deterministic heuristics.
6. Optional behavioral probing in an externally isolated worker.
7. Diffusion configuration and tree validation.

The parser rejects symlinks, hard links, special files, duplicate JSON keys,
malformed GGUF metadata, ambiguous URL prefixes, safetensors buffer holes and
overlaps, and unsafe or overlarge directory trees. GGUF template and weight
analysis share the same bounded metadata parser. Descriptor-relative diffusion
validation requires allowlisted `[library, class]` tuples, strict JSON, a
configuration per component, and safetensors for weight-bearing components.
Directory digests bind file content, paths, sizes, and empty-directory markers.
Files and directories are hashed again around the registry move. A destination
collision is rejected. A durable staging marker exists before the move;
provenance and other evidence are written, hashed, and fsynced before a pending
record permits any registry call. A crash during incomplete staging rolls the
bundle back to quarantine. Once fully pending, API failure leaves a verified
uncommitted bundle for bounded-batch, backoff-controlled replay using a stable
idempotency key.

This is admission control, not proof that a model is safe. Static and
behavioral scanners can produce false positives and false negatives, and
native parsers must still run in a strongly isolated process.

## Profiles

- [appliance-profile.yaml](examples/appliance-profile.yaml) is the production
  baseline. It disables local imports, requires hash pins and behavioral
  testing, and fails closed if a scanner is unavailable.
- [standalone-profile.yaml](examples/standalone-profile.yaml) is for tests and
  development. It permits local TOFU and optional-scanner skips.

Only documented keys in those profiles are currently evaluated. The pipeline
always runs every applicable gate; there are no per-stage bypass toggles.

## Development

Use Python 3.12 or newer in a project-local environment:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --require-hashes -r requirements-dev.lock
.venv/bin/python -m pip install --no-deps --no-build-isolation -e .
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check .
.venv/bin/python -m pip_audit -r requirements.lock --disable-pip
.venv/bin/python -m pip_audit -r requirements-dev.lock --disable-pip
.venv/bin/python -m bandit -q -r quarantine -ll
```

`requirements.lock` and `requirements-dev.lock` are generated with hashes.
Regenerate and review both whenever dependency constraints change.

### Library use

```python
from pathlib import Path

from quarantine.pipeline import run_pipeline

result = run_pipeline(
    Path("model.gguf"),
    file_hash="expected-sha256",
    policy={
        "models": {
            "require_scan": True,
            "require_behavior_tests": False,
            "allow_local_imports": True,
        },
        "quarantine": {"scanner_missing": "fail-closed"},
    },
)
print(result["passed"], result["details"])
```

Local imports and disabled behavioral testing above are for development only.
Applicable static gates still require complete evidence.

## Production prerequisites

Before starting the watcher:

- Mount a non-world-writable policy file at `POLICY_PATH`.
- Mount a registry bearer token as an owner-only regular single-link file and
  set `SERVICE_TOKEN_PATH`. It must be owned by root or the service UID and
  contain 32–4096 printable ASCII bytes with no whitespace or control bytes.
- Use distinct, non-overlapping quarantine and registry directories.
- Install every required scanner at a reviewed, pinned version.
- Put scanner binaries on a controlled `PATH`, or configure their absolute
  paths where supported.
- Provision an audit HMAC key with the same owner, link, permission, length,
  and printable-byte requirements and set `AUDIT_HMAC_KEY_PATH`.
- Run the service with a private temporary directory, read-only system paths,
  resource limits, and outbound-network restrictions.
- Put behavioral inference in a separate sandboxed worker. Set
  `SECAI_BEHAVIOR_SANDBOXED=1` only after that isolation is actually
  enforced.

The core container intentionally does not perform mutable, best-effort scanner
installation. A production image should derive from it and add hash-locked,
reviewed scanner binaries. With the appliance profile, missing scanners cause
rejection.

### Watcher configuration

```bash
export QUARANTINE_DIR=/data/quarantine
export REGISTRY_DIR=/data/registry
export REGISTRY_URL=https://registry.internal.example
export POLICY_PATH=/etc/secure-ai/policy/policy.yaml
export MODELS_LOCK_PATH=/etc/secure-ai/policy/models.lock.yaml
export SOURCES_ALLOWLIST_PATH=/etc/secure-ai/policy/sources.allowlist.yaml
export SERVICE_TOKEN_PATH=/run/secrets/registry-token
export AUDIT_HMAC_KEY_PATH=/run/secrets/audit-hmac-key
export AUDIT_LOG_PATH=/var/lib/secure-ai/logs/quarantine-audit.jsonl
ai-quarantine
```

`REGISTRY_URL` must be an HTTP(S) origin with no credentials, path, query, or
fragment. Plain HTTP is accepted only for a loopback address. Authentication
is mandatory by default. For an explicitly disposable loopback-only
development setup, `SECAI_ALLOW_INSECURE_REGISTRY=1` permits unauthenticated
promotion; do not use it in production.

Prefer file-backed credentials. `SERVICE_TOKEN` is supported for constrained
environments but exposes the token to the process environment. Environment
tokens are subject to the same 32–4096 printable non-whitespace byte rule.

The watcher processes at most 128 queue objects per pass and inspects at most
eight times that number without materializing or sorting the whole directory.
A fixed 256-slot nonblocking `flock` set prevents concurrent double processing
without attacker-controlled lock-file growth. Transient failures persist an
exponential-backoff state; after three attempts the artifact and its source
metadata move to a private terminal hold with a bounded rejection record.
Hidden or special queue entries are also removed from the active namespace.
Each pass replays at most 32 durable pending registry commits.

### Container

Build the deterministic core image:

```bash
docker build --pull=false -f Containerfile -t ai-quarantine .
```

Run it in the same tightly controlled pod/network namespace as a loopback
registry, or use an authenticated HTTPS registry. Mount policy, lock, allowlist,
token, and audit key files read-only; mount quarantine, registry, and audit
storage separately. Do not use host networking merely for convenience.

Recommended runtime controls include a read-only root filesystem,
`no-new-privileges`, dropped Linux capabilities, a restrictive seccomp
profile, memory/CPU/PID limits, and an explicit egress policy. The container
uses a reviewed digest-pinned base without mutable OS-package resolution during
the build and is gated against fixable high/critical image vulnerabilities.
Application directories/files are root-owned mode 0555/0444, while only the
explicit data mounts are writable. It runs as numeric UID/GID 65532 and
contains only the core Python dependency.

## Policy keys

| Key | Meaning |
|---|---|
| `models.allowed_formats` | Subset of `gguf` and `safetensors` |
| `models.require_scan` | Static scan requirement; false cannot waive applicable evidence |
| `models.require_behavior_tests` | Require llama-server behavioral checks |
| `models.allow_local_imports` | Permit artifacts without source metadata |
| `models.require_hash_pin_for_local` | Reject unpinned local artifacts |
| `models.allow_diffusion_directories` | Permit directory artifacts |
| `quarantine.scanner_missing` | `fail-closed` or legacy `warn-and-skip`; neither waives applicable static evidence |
| `quarantine.smoke_test_max_score` | Maximum behavioral flag ratio |
| `quarantine.smoke_test_max_critical` | Maximum critical-category flags |
| `gguf_guard.required` | Require gguf-guard for GGUF artifacts |

Remote artifacts always require both a canonical allowlisted HTTPS source and a
matching entry in `models.lock.yaml`. Source URLs containing credentials,
queries, fragments, nonstandard ports, traversal, or control characters are
rejected.

## External scanners

The integration supports ModelScan, Fickling, ModelAudit, gguf-guard, Garak,
llama-server, cosign, and fs-verity. Scanner processes handle hostile data and
must be treated as an untrusted parsing tier. The production profile fails
closed on missing static and behavioral scanners.

Static scanner outcomes use schema version 2 with `verdict`, `passed`,
`available`, `scanner`, `scanner_version`, `reason`, and `scanned_count`.
The verdict is `pass`, `fail`, `not_applicable`, `unavailable`, or `error`.
Only `pass` sets `passed=true`. Nonapplicable outcomes have zero coverage,
no observed scanner version, and unknown availability; they never claim the
scanner ran. Aggregation separately accepts an expected nonapplicable gate
only after validating the artifact format. Unknown/corrupt formats, symlinks,
and unexpected directory files cannot obtain that exemption.

| Scanner | Validated GGUF | Validated safetensors |
|---|---|---|
| ModelScan | Not applicable | Not applicable |
| Fickling | Not applicable | Not applicable |
| ModelAudit | Required | Required |
| gguf-guard | Required | Not applicable |

Header/format, pickle-polyglot, entropy and weight checks remain required.
Weight analysis must cover at least one tensor; unsupported numerical types
or files too small for meaningful statistics cannot silently pass this gate.
Directory scanning validates the inventory first and retains each safetensors
file's detailed outcomes within a 512 KiB static-evidence budget; overflow is
nonpassing. Its file count is not ModelScan coverage. Existing source, pinning,
provenance and behavioral gates remain separate.

External results are not accepted merely because a process exits zero.
For applicable library inputs, ModelScan requires a version, zero errors/skips,
positive expected coverage and no issues. An installed broken API cannot fall
back to CLI. Fickling 0.1.12 must report `LIKELY_SAFE`, no contradictory
findings, and the matching successful exit; its JSON report is written to a
private temporary file, separate from the input. ModelAudit 0.2.52 must report
successful completion, the exact requested file and applicable format scanner,
positive file/byte/check coverage, and no unsafe or incomplete evidence.
The invoked ModelAudit and Fickling executables identify their versions through
`--version` with a five-second timeout; adapter schema numbers are separate.
ModelAudit scans disable result-cache reuse and whitelist severity downgrading.
ModelAudit's GGUF byte coverage can describe parsed headers rather than all
tensor bytes; `bytes_scanned` and `artifact_size_bytes` remain distinct, with
independent format, weight and guard checks still required. gguf-guard must
return its completed PASS/WARN/FAIL output consistently with its exit status
and identify its version. Existing guard warning thresholds are preserved.
Missing, contradictory, malformed, empty or partial applicable reports fail,
even with legacy `require_scan=false` or `scanner_missing=warn-and-skip`.
These settings remain accepted for configuration compatibility but cannot
create static scan evidence. Other development policy behavior is unchanged.

Qualified contracts currently cover ModelAudit 0.2.52, Fickling 0.1.12 and
gguf-guard 0.2.0; unknown versions fail closed until their report contracts are
qualified. ModelScan 0.8.8 has API and CLI fixture coverage. Real synthetic GGUF,
safetensors and directory static-stage checks exercise the applicable scanners,
while safe/unsafe/invalid pickle controls exercise library adapters only. The
production allowlist still rejects pickle. Fickling reports containing multiple
stacked JSON documents and ModelAudit reports with incomplete or inconsistent
check inventories remain nonpassing. These checks do not qualify behavioral
inference, all model families, production deployment or registry promotion.

Garak must produce one bounded JSONL report with at least one evaluated probe.

Every configured behavioral prompt must return one bounded, non-empty,
schema-valid response. Timeouts, transport errors, malformed JSON, oversized
bodies, and empty content are distinct failures. llama-server output is sent
to a bounded sink rather than retained in undrained pipes or audit evidence.

The behavioral score is a regression signal, not a general safety benchmark.
The current library starts llama-server locally; network and filesystem
isolation are deployment responsibilities. SecAI_OS should use its dedicated
scanner worker/sandbox rather than invoking native model parsers in a broadly
privileged watcher process.

## Audit and provenance

Audit entries are JSONL records chained with SHA-256 or, when configured,
HMAC-SHA-256. The watcher refuses to append to a corrupt existing chain,
securely appends and fsyncs mode-0600 records, and verifies rotated archives.

A plain SHA-256 chain detects modification within retained logs but cannot
prove that an attacker did not delete the entire tail. Configure
`AUDIT_HMAC_KEY_PATH` to enable an authenticated checkpoint that detects tail,
archive, and complete-log deletion while that checkpoint survives. If an
attacker deletes both all co-located logs and the checkpoint, the local node no
longer has evidence with which to prove the deletion. Keep the key outside the
log storage and replicate checkpoints/archives to append-only or remote
storage. The implementation is single-process; do not run multiple watcher
writers against one log.

Before a registry call, the watcher creates and fsyncs provenance JSON and any
configured detached signature/guard evidence, records their hashes and sizes
in a durable pending journal, re-verifies the artifact/evidence/payload
binding, and sends the journal-derived promotion ID as the API idempotency key.
The registry commit is the last admission step. If
`/etc/secure-ai/keys/cosign.key` is present, signing is mandatory for that
bundle; unsigned local provenance remains metadata, not non-repudiation.

## Known limits

- Scanner and model-runtime zero-days remain possible.
- The standalone process does not itself create a kernel-enforced sandbox.
- Source allowlisting plus a hash pin proves identity, not publisher intent.
- Local TOFU is intentionally available only for development profiles.
- Recovery and rollback cannot succeed while the underlying filesystem is
  unavailable; operators must alert on pending/invalid journals and terminal
  holds.
- Registry storage and its database are not one filesystem transaction. The
  integration therefore requires registry consumers to ignore uncommitted
  files and the API to honor idempotency keys; a server-owned staging/commit
  API would reduce this residual coupling.
- Hash-chain HMAC checkpoints do not replace remote/WORM log replication.
- Applicable ModelScan integrations reject errors, skips and zero coverage.
  Unsupported scanners remain explicit nonapplicable outcomes, not evidence
  of a clean artifact. Helper promotion protocol compatibility remains separate
  from scanner applicability; the sibling registry does not yet implement the
  watcher's stated idempotent commit contract.
- Multi-process audit writers are unsupported.

See [SECURITY_AUDIT.md](SECURITY_AUDIT.md) for the audit record, completed
remediations, residual risks, and production roadmap.

## Security reports

Do not open public issues for suspected bypasses. Follow
[SECURITY.md](SECURITY.md).

## License

Apache-2.0
