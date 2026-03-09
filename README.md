# ai-quarantine

Seven-stage AI artifact admission-control pipeline. Verifies, scans, and
smoke-tests model files before promoting them to a trusted registry.

## What it does

Every model artifact (GGUF, safetensors, or diffusion directory) passes
through a deterministic pipeline before it can be used:

| Stage | Gate | Purpose |
|-------|------|---------|
| 1 | Source policy | Verify origin URL against an allowlist |
| 2 | Format gate | Validate headers, reject unsafe formats (pickle, pt, bin) |
| 3 | Integrity | SHA-256 hash pinning (TOFU for local, strict for remote) |
| 4 | Provenance | Cosign signature verification where available |
| 5 | Static scan | modelscan + fickling + entropy + weight distribution + gguf-guard |
| 6 | Behavioral | Adversarial prompt suite (41 prompts, 50+ danger patterns) in CPU-only sandbox |
| 7 | Diffusion deep | Config integrity, component validation, symlink detection |

All stages run automatically. If any stage fails, the artifact is rejected.

## Deployment profiles

Choose the profile that matches your environment. Profiles are complete policy
files you can copy directly to `/etc/secure-ai/policy/policy.yaml`.

### Appliance profile (recommended for production)

Use [`examples/appliance-profile.yaml`](examples/appliance-profile.yaml) when
deploying as part of an immutable-OS appliance (e.g. SecAI_OS) or any
production environment:

- All seven stages enabled
- `require_scan: true` and `require_behavior_tests: true`
- `gguf_guard.required: true`
- `scanner_missing: fail-closed` — missing scanners cause hard rejection
- Registry URL restricted to localhost only
- Offline by default — no outbound network access
- No retained raw artifacts after scan

### Standalone profile (development and CI)

Use [`examples/standalone-profile.yaml`](examples/standalone-profile.yaml) for
developer workstations, CI pipelines, or environments without llama-server:

- All seven stages enabled
- `require_scan: true`, `require_behavior_tests: false` (no llama-server assumed)
- `gguf_guard.required: false`
- `scanner_missing: warn-and-skip` for optional scanners

## Artifact states

```
PENDING -> SCANNING -> PASSED  (promoted to registry)
                    -> FAILED  (stage failure, rejected)
REJECTED                       (immediate format/policy violation)
```

## Quick start

### As a library

```python
from quarantine.pipeline import run_pipeline
from pathlib import Path

# Secure default: all scanners required, fail-closed
result = run_pipeline(
    artifact_path=Path("model.gguf"),
    file_hash="sha256...",
    policy={"models": {"require_scan": True}},
)
print(result["passed"])  # True/False
print(result["details"]) # per-stage evidence
```

For development or testing where scanners are not installed, use a minimal
policy that skips scanner requirements:

```python
# Development only — NOT for production use
result = run_pipeline(
    artifact_path=Path("model.gguf"),
    file_hash="sha256...",
    policy={
        "models": {"require_scan": True, "require_behavior_tests": False},
        "quarantine": {"scanner_missing": "warn-and-skip"},
    },
)
```

### As a watcher (filesystem monitor)

```bash
export QUARANTINE_DIR=/data/quarantine
export REGISTRY_DIR=/data/registry
export REGISTRY_URL=http://127.0.0.1:8470
export SERVICE_TOKEN_PATH=/run/secure-ai/service-token
ai-quarantine
```

### Container

```bash
docker build -f Containerfile -t ai-quarantine .
docker run --network=host \
           -v /data/quarantine:/quarantine \
           -v /data/registry:/registry \
           -e REGISTRY_URL=http://127.0.0.1:8470 \
           -e SERVICE_TOKEN_PATH=/run/secrets/registry-token \
           -v /path/to/token:/run/secrets/registry-token:ro \
           ai-quarantine
```

> **Note:** The quarantine watcher communicates with the registry over HTTP. Use `127.0.0.1` (localhost) to ensure traffic stays on the loopback interface. Never expose the registry on a remote address without TLS.

## Configuration

The pipeline reads policy from YAML files:

- **policy.yaml** — stage toggles, thresholds, scanner requirements
- **sources.allowlist.yaml** — trusted model sources (URL prefixes)
- **models.lock.yaml** — pinned SHA-256 hashes for known-good artifacts

See [examples/](examples/) for reference configurations, including the
[appliance](examples/appliance-profile.yaml) and
[standalone](examples/standalone-profile.yaml) deployment profiles.

### Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `QUARANTINE_DIR` | `/quarantine` | Drop directory for incoming artifacts |
| `REGISTRY_DIR` | `/registry` | Destination for promoted artifacts |
| `REGISTRY_URL` | `http://127.0.0.1:8470` | Registry API endpoint |
| `POLICY_PATH` | `/etc/secure-ai/policy/policy.yaml` | Policy config path |
| `MODELS_LOCK_PATH` | `/etc/secure-ai/policy/models.lock.yaml` | Hash pins |
| `SOURCES_ALLOWLIST_PATH` | `/etc/secure-ai/policy/sources.allowlist.yaml` | Source allowlist |
| `LLAMA_SERVER_BIN` | `/usr/bin/llama-server` | Path to llama-server for smoke tests |
| `GGUF_GUARD_BIN` | `/usr/local/bin/gguf-guard` | Path to gguf-guard scanner |
| `SMOKE_TEST_TIMEOUT` | `120` | Seconds to wait for llama-server startup |
| `AUDIT_LOG_PATH` | `/var/lib/secure-ai/logs/quarantine-audit.jsonl` | Hash-chained audit log |
| `SERVICE_TOKEN` | *(empty)* | Bearer token for authenticated registry promotion |
| `SERVICE_TOKEN_PATH` | *(empty)* | Path to file containing the service token |

## Registry integration

The watcher promotes passing artifacts to the registry via its HTTP API
(`POST /v1/model/promote`).

### Authenticated promotion

When `SERVICE_TOKEN` is set (or `SERVICE_TOKEN_PATH` points to a token file),
the watcher includes an `Authorization: Bearer <token>` header in all
promotion requests. This prevents unauthorized artifact promotion when the
registry enforces authentication.

```bash
# Option 1: token in environment variable
export SERVICE_TOKEN="your-registry-token"

# Option 2: token in a file (useful with Kubernetes secrets or systemd credentials)
export SERVICE_TOKEN_PATH="/run/secrets/registry-token"
```

The token is read once per promotion request. If both `SERVICE_TOKEN` and
`SERVICE_TOKEN_PATH` are set, the environment variable takes precedence.

## External scanners

The pipeline integrates with optional external tools. In the default secure
policy (`scanner_missing: fail-closed`), a missing scanner causes the artifact
to be rejected. Use the standalone profile for environments where not all
scanners are installed.

| Scanner | Stage | Purpose |
|---------|-------|---------|
| [modelscan](https://github.com/protectai/modelscan) | 5 | Detect malicious serialization |
| [fickling](https://github.com/trailofbits/fickling) | 5 | Pickle safety analysis |
| [modelaudit](https://pypi.org/project/modelaudit/) | 5 | Second-opinion static scan |
| [gguf-guard](https://github.com/SecAI-Hub/gguf-guard) | 5 | GGUF weight-level anomaly detection |
| [garak](https://github.com/NVIDIA/garak) | 6 | LLM vulnerability probing |
| [llama-server](https://github.com/ggml-org/llama.cpp) | 6 | CPU-only inference for smoke tests |
| [cosign](https://github.com/sigstore/cosign) | 4 | Signature verification |
| [fsverity](https://www.kernel.org/doc/html/latest/filesystems/fsverity.html) | post | Kernel-level file integrity |

## Sandbox isolation

Stage 6 (behavioral smoke test) runs adversarial prompts against the model
inside a constrained sandbox. The sandbox provides the following isolation
guarantees:

- **CPU-only inference**: llama-server is invoked with `--n-gpu-layers 0`,
  ensuring no GPU driver interaction and reducing the attack surface.
- **Localhost-only binding**: The server binds to `127.0.0.1` only. It is
  not reachable from any external network interface.
- **No network access from the model**: The model runs inside llama-server
  which has no outbound networking capability. The model cannot fetch remote
  resources or exfiltrate data.
- **Ephemeral**: The server is started, tested against the adversarial prompt
  suite, and immediately terminated. No persistent state is retained.
- **Resource limits (systemd deployment)**: When deployed via systemd, the
  service unit enforces hard limits:
  - `MemoryMax=4G` — 4 GB memory ceiling
  - `TasksMax=128` — maximum 128 concurrent tasks
  - `SystemCallFilter=@system-service` — seccomp syscall allowlist
- **No ambient secrets**: The systemd unit uses `PrivateTmp=yes`,
  `ProtectHome=yes`, and `ReadOnlyPaths=/` to prevent the sandbox from
  accessing user data, temporary files, or modifying the filesystem.

## Signed verdicts

Each promoted artifact receives a JSON provenance manifest
(`<filename>.provenance.json`) stored alongside the artifact in the registry
directory. The manifest records:

- Artifact SHA-256 hash and size
- Source URL and revision
- Scanner versions used during scanning
- Policy version (hash of policy file at scan time)
- Promotion timestamp and hostname
- Scan summary (pass/fail per stage)
- fs-verity digest (if enabled)

If a cosign signing key is available at `/etc/secure-ai/keys/cosign.key`, the
manifest is automatically signed. Verify the signature:

```bash
cosign verify-blob --key cosign.pub --signature file.provenance.json.sig file.provenance.json
```

Unsigned manifests still provide a tamper-evident record when combined with
the hash-chained audit log. Signing adds non-repudiation.

## Audit logging

All scan decisions are recorded in a hash-chained JSONL audit log. Each entry
contains a SHA-256 link to the previous entry, forming a tamper-evident chain.
Verify chain integrity:

```python
from quarantine.audit_chain import AuditChain
result = AuditChain.verify("/path/to/quarantine-audit.jsonl")
print(result)  # {"valid": True, "entries": 42, ...}
```

## Privacy: metadata and retention

### What is stored in the audit log

Each audit entry records: event type, filename, SHA-256 hash, timestamp,
scan pass/fail result, and scanner versions. Entries are hash-chained
(each entry includes the SHA-256 of the previous entry) for tamper evidence.

### What is NOT stored

The audit log does not contain: raw model weights, user identity, IP
addresses, or any personally identifiable information.

### Artifact retention

- **Rejected artifacts** are deleted immediately after the pipeline verdict.
  No copy of a rejected artifact is retained on disk.
- **Promoted artifacts** are the only artifacts retained, stored in the
  registry directory alongside their provenance manifests.

### Provenance manifests

Each promoted artifact gets a provenance manifest containing a scan summary
(pass/fail per stage, scanner versions, policy hash). The manifest does not
include raw scan output or detailed scanner logs.

### Log rotation

The audit log rotates at 50 MB by default (configurable via the `max_size_mb`
parameter). Archived logs are set to read-only mode (0444 permissions) to
prevent accidental modification.

## License

Apache-2.0
