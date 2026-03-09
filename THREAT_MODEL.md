# Threat Model

This document describes the trust boundaries, threat actors, mitigations,
and residual risks for the ai-quarantine pipeline.

## Trust boundaries

```
  Untrusted                        Trusted
  ┌──────────┐   ┌───────────────────────────────────────┐   ┌──────────┐
  │ Artifact │──▶│  Quarantine Pipeline (7 stages)       │──▶│ Registry │
  │ Source   │   │  ┌────────┐ ┌──────────┐ ┌─────────┐ │   │ (local)  │
  └──────────┘   │  │External│ │llama-srvr│ │ Audit   │ │   └──────────┘
                 │  │Scanners│ │ Sandbox  │ │ Chain   │ │
                 │  └────────┘ └──────────┘ └─────────┘ │
                 └───────────────────────────────────────┘
```

| Boundary | Description |
|----------|-------------|
| **Artifact source** | Untrusted. Models arrive from the internet, local imports, or peer transfers. Source URL is verified against an allowlist but the artifact content is untrusted until all stages pass. |
| **Quarantine pipeline** | Trusted execution environment. Runs as a system service with controlled privileges. All scan decisions are recorded in the hash-chained audit log. |
| **External scanners** | Semi-trusted. modelscan, fickling, garak, and gguf-guard are invoked as subprocesses. They process untrusted input and may themselves be compromised. |
| **llama-server sandbox** | Constrained trust. CPU-only, localhost-bound, no outbound network, ephemeral process. Resource-limited via systemd when deployed (4 GB memory, 128 tasks, seccomp). |
| **Registry** | Trusted store. Only artifacts that pass all seven stages are promoted. Protected by fs-verity and signed provenance manifests. |

## Threats

### T1 — Malicious model bypass

**Attacker goal:** Get a trojaned or backdoored model promoted to the trusted
registry.

**Attack vectors:**
- Craft a model that passes static scans but contains hidden behaviors
- Embed malicious payloads in non-weight metadata (chat templates, tokenizer configs)
- Use subtle weight modifications that evade statistical anomaly detection

### T2 — Scanner evasion

**Attacker goal:** Craft artifacts that exploit blind spots in the scanning
tools.

**Attack vectors:**
- Polyglot files (valid GGUF header but embedded pickle/executable payload)
- Format confusion (e.g. .safetensors extension with pickle content)
- Adversarial inputs that crash or confuse scanners
- Entropy manipulation to avoid anomaly triggers

### T3 — Sandbox escape

**Attacker goal:** Break out of the behavioral smoke test sandbox to access
the host system.

**Attack vectors:**
- Exploit llama-server vulnerabilities during inference
- Generate outputs that trigger shell injection in post-processing
- Attempt network access from the sandboxed process

### T4 — Promotion without scan

**Attacker goal:** Get an artifact into the registry without completing all
pipeline stages.

**Attack vectors:**
- Race condition: drop artifact directly into registry directory
- Bypass watcher by manipulating file metadata or symlinks
- Exploit pipeline logic errors to skip stages

### T5 — Audit log tampering

**Attacker goal:** Modify or delete audit entries to hide evidence of a
rejected or suspicious artifact.

**Attack vectors:**
- Direct file modification (truncation, entry deletion)
- Insert forged entries to obscure timeline
- Corrupt the hash chain to invalidate the log

### T6 — Scanner supply chain compromise

**Attacker goal:** Compromise an external scanner (modelscan, fickling, garak,
gguf-guard) to produce false negatives.

**Attack vectors:**
- Publish a malicious update to a scanner package
- Dependency confusion attack on scanner dependencies
- Compromise the scanner's release signing key

## Mitigations

| Threat | Mitigation |
|--------|------------|
| T1 — Malicious bypass | Seven-stage pipeline with defense in depth: format validation, polyglot detection, entropy analysis, weight anomaly detection, adversarial prompt suite (41 prompts, 50+ danger patterns), and Jinja template scanning. |
| T2 — Scanner evasion | Multiple independent scanners (modelscan, fickling, gguf-guard, modelaudit) reduce single-point-of-failure risk. Fail-closed defaults reject artifacts when scanners are unavailable. Format allowlist blocks unsafe serialization formats (pickle, pt, bin). |
| T3 — Sandbox escape | CPU-only inference (--n-gpu-layers 0), localhost-only binding (127.0.0.1), no outbound network from model process, ephemeral server lifecycle, systemd resource limits (MemoryMax=4G, TasksMax=128, seccomp, PrivateTmp, ProtectHome). |
| T4 — Promotion without scan | Pipeline enforces sequential stage completion. Watcher owns the quarantine-to-registry file move. Registry directory is not writable by unprivileged users. Provenance manifest records which stages ran. |
| T5 — Audit log tampering | Hash-chained JSONL log: each entry includes SHA-256 of previous entry. Chain verification detects any modification, deletion, or insertion. Rotated logs are set read-only (0444). |
| T6 — Scanner supply chain | Pin scanner versions in requirements.lock with hash verification (--require-hashes). Record scanner versions in provenance manifests. Container image uses pinned base image digest. |

## Residual risks

These risks are acknowledged but not fully mitigated by the current pipeline:

| Risk | Description | Severity |
|------|-------------|----------|
| **Novel evasion techniques** | New attack methods may bypass current scanners before detection signatures are updated. | High |
| **Scanner false negatives** | No scanner is perfect. A model may contain subtle backdoors that pass all current checks. | High |
| **Sophisticated weight-level attacks** | Weight anomaly detection uses statistical heuristics (kurtosis, zero fraction, entropy). Carefully crafted attacks that stay within normal statistical bounds may evade detection. | Medium |
| **Zero-day in llama-server** | A vulnerability in llama-server could allow sandbox escape during behavioral testing. Mitigated by CPU-only mode and systemd hardening but not eliminated. | Medium |
| **Compromised host** | If the host OS is compromised, all pipeline guarantees are void. The immutable-OS deployment (uBlue/Silverblue) mitigates but does not eliminate this. | High |
| **Time-of-check/time-of-use** | A model file could theoretically be modified between scan completion and promotion. Mitigated by atomic file move and fs-verity. | Low |
