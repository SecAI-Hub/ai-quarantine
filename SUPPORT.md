# Support

## Getting help

- **Issues**: https://github.com/SecAI-Hub/ai-quarantine/issues
- **Security**: See [SECURITY.md](SECURITY.md)

## FAQ

**Q: Do I need all external scanners installed?**
A: No. Each scanner degrades gracefully if not installed, unless your policy
sets `require_scan: true` (which is the default for modelscan).

**Q: Can I use this without a registry?**
A: Yes. Use `run_pipeline()` directly as a library to get pass/fail verdicts
without needing a running registry service.

**Q: What model formats are supported?**
A: GGUF and safetensors (single-file), plus diffusion model directories
(containing model_index.json with safetensors components).
