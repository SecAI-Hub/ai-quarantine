# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in ai-quarantine, please report it
responsibly:

1. **Do NOT open a public issue.**
2. Email: security@secai-hub.dev (or open a private advisory on GitHub)
3. Include: description, reproduction steps, and impact assessment.

## Disclosure Timeline

| Step | Target |
|------|--------|
| Acknowledge receipt | 48 hours |
| Confirm and assign severity | 5 business days |
| Publish fix (critical/high) | 7 calendar days |
| Publish fix (medium/low) | 30 calendar days |
| Public advisory | After fix is released |

We will coordinate disclosure with the reporter. If a fix requires more time,
we will communicate the revised timeline and may issue a temporary mitigation
advisory.

## Scope

This project scans AI model artifacts for safety issues. Vulnerabilities in
the scanning pipeline itself (bypass, false negative, sandbox escape) are
considered critical.

### In scope

- Pipeline bypass (artifact promoted without completing all stages)
- Scanner evasion (crafted artifact evades detection)
- Sandbox escape (behavioral test sandbox compromised)
- Audit log tampering (chain integrity circumvented)
- Promotion without authentication (SERVICE_TOKEN bypass)
- Denial of service against the scanning pipeline

### Out of scope

- Vulnerabilities in upstream scanners (modelscan, fickling, garak) — report
  these to the respective projects
- Social engineering attacks
- Physical access attacks

## Supported Versions

| Version | Supported | Notes |
|---------|-----------|-------|
| 0.1.x   | Yes       | Current stable release |
| < 0.1   | No        | Pre-release, unsupported |

Users should always run the latest patch release within their supported
major.minor version.
