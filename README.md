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

result = run_pipeline(
    artifact_path=Path("model.gguf"),
    file_hash="sha256...",
    policy={"models": {"require_scan": False}},
)
print(result["passed"])  # True/False
print(result["details"]) # per-stage evidence
```

### As a watcher (filesystem monitor)

```bash
export QUARANTINE_DIR=/data/quarantine
export REGISTRY_DIR=/data/registry
export REGISTRY_URL=http://127.0.0.1:8470
ai-quarantine
```

### Container

```bash
docker build -f Containerfile -t ai-quarantine .
docker run -v /data/quarantine:/quarantine \
           -v /data/registry:/registry \
           -e REGISTRY_URL=http://registry:8470 \
           ai-quarantine
```

## Configuration

The pipeline reads policy from YAML files:

- **policy.yaml** — stage toggles, thresholds, scanner requirements
- **sources.allowlist.yaml** — trusted model sources (URL prefixes)
- **models.lock.yaml** — pinned SHA-256 hashes for known-good artifacts

See [examples/](examples/) for reference configurations.

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

## External scanners

The pipeline integrates with optional external tools. Each degrades gracefully
if not installed (unless marked required in policy):

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

## Audit logging

All scan decisions are recorded in a hash-chained JSONL audit log. Each entry
contains a SHA-256 link to the previous entry, forming a tamper-evident chain.
Verify chain integrity:

```python
from quarantine.audit_chain import AuditChain
result = AuditChain.verify("/path/to/quarantine-audit.jsonl")
print(result)  # {"valid": True, "entries": 42, ...}
```

## License

Apache-2.0
