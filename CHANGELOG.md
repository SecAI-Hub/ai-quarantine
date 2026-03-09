# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-03-09

### Added

- Seven-stage deterministic admission-control pipeline for AI model artifacts
  - Stage 1: Source policy gate (URL allowlist verification)
  - Stage 2: Format gate (magic byte validation, unsafe format rejection)
  - Stage 3: Integrity gate (SHA-256 hash pinning, TOFU for local imports)
  - Stage 4: Provenance gate (cosign signature verification)
  - Stage 5: Static scan (modelscan, fickling, entropy, weight analysis, gguf-guard)
  - Stage 6: Behavioral smoke test (41 adversarial prompts, 50+ danger patterns)
  - Stage 7: Diffusion deep scan (config integrity, component validation)
- Filesystem watcher for automatic artifact processing
- Hash-chained append-only audit log with tamper detection
- Provenance manifest generation with optional cosign signing
- fs-verity integration for kernel-level file integrity
- Deployment profiles: appliance (fail-closed) and standalone (warn-and-skip)
- SERVICE_TOKEN support for authenticated registry promotion
- Container image with Containerfile
- CI pipeline with tests and container build verification
- Signed release workflow with cosign keyless signing
- Threat model documentation
- Policy configuration via YAML (stages, scanners, thresholds)
- Source allowlist and model hash pinning
- Support for GGUF, safetensors, and diffusion model directories

### Security

- Fail-closed default policy: missing scanners cause hard failure
- CPU-only sandbox for behavioral tests (no GPU, localhost-only, ephemeral)
- Pickle, .pt, and .bin formats denied by default
- Polyglot detection (pickle signatures embedded in valid formats)
- Jinja template injection scanning in GGUF chat templates
- Weight anomaly detection (kurtosis, zero fraction, entropy thresholds)
