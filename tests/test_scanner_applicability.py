"""Format applicability and positive evidence at the static admission boundary."""

import builtins
import json
import os
import struct
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from quarantine import pipeline as p


def artifact(tmp_path, suffix):
    path = tmp_path / f"model{suffix}"
    values = [float(i - 16) / 32 for i in range(32)]
    data = struct.pack("<32f", *values)
    if suffix == ".gguf":
        header = b"GGUF" + struct.pack("<IQQ", 3, 1, 0)
        header += struct.pack("<Q", 1) + b"a" + struct.pack("<IQIQ", 1, 32, 0, 0)
        path.write_bytes(header + b"\0" * (-len(header) % 32) + data)
    else:
        header = json.dumps({"weight": {"dtype": "F32", "shape": [32], "data_offsets": [0, len(data)]}}).encode()
        header += b" " * (-len(header) % 8)
        path.write_bytes(struct.pack("<Q", len(header)) + header + data)
    return path


def report(path):
    return {"issues": [], "errors": [], "summary": {
        "modelscan_version": "0.8.8", "total_issues": 0,
        "scanned": {"total_scanned": 1, "scanned_files": [str(path)]},
        "skipped": {"total_skipped": 0, "skipped_files": []},
    }}


def no_process(*_args, **_kwargs):
    raise AssertionError("a nonapplicable scanner must not start a process")


def passing_scanners(monkeypatch):
    monkeypatch.setattr(p, "_run_modelaudit", lambda *_a, **_k: p._static_scanner_result(
        "pass", "modelaudit", reason="fixture complete report", scanner_version="0.2.52", scanned_count=1))
    monkeypatch.setattr(p, "_run_gguf_guard_scan", lambda path, **_k:
                        p._static_applicability(path, "gguf-guard") or p._static_scanner_result(
                            "pass", "gguf-guard", reason="fixture completed scan", scanner_version="0.2.0", scanned_count=1))


@pytest.mark.parametrize("suffix", [".gguf", ".safetensors"])
@pytest.mark.parametrize("scanner", [p._run_modelscan, p._run_fickling_scan])
def test_valid_safe_format_is_not_a_scanner_pass(tmp_path, monkeypatch, suffix, scanner):
    path = artifact(tmp_path, suffix)
    monkeypatch.setattr(p.subprocess, "run", no_process)
    result = scanner(path)
    assert result["verdict"] == "not_applicable"
    assert result["passed"] is False
    assert result["scanned_count"] == 0
    assert result["scanner_version"] is None
    assert json.loads(json.dumps(result))["verdict"] != "pass"


@pytest.mark.parametrize("suffix", [".gguf", ".safetensors", ".unknown"])
@pytest.mark.parametrize("scanner", [p._run_modelscan, p._run_fickling_scan, p._run_gguf_guard_scan])
def test_corrupt_or_unknown_is_never_not_applicable(tmp_path, monkeypatch, suffix, scanner):
    path = tmp_path / f"model{suffix}"
    path.write_bytes(b"broken")
    monkeypatch.setattr(p.subprocess, "run", no_process)
    result = scanner(path)
    assert result["verdict"] in {"fail", "error"}
    assert not result["passed"]


@pytest.mark.parametrize("suffix", [".gguf", ".safetensors"])
def test_applicability_requires_regular_unlinked_input(tmp_path, suffix):
    path = artifact(tmp_path, suffix)
    link = tmp_path / f"link{suffix}"
    link.symlink_to(path)
    assert p._run_modelscan(link)["verdict"] != "not_applicable"
    link.unlink()
    link.hardlink_to(path)
    assert p._run_modelscan(link)["verdict"] != "not_applicable"


@pytest.mark.parametrize("suffix", [".gguf", ".safetensors"])
def test_aggregate_accepts_na_but_keeps_coverage_honest(tmp_path, monkeypatch, suffix):
    passing_scanners(monkeypatch)
    result = p.check_static_scan(artifact(tmp_path, suffix))
    assert result["passed"], result
    assert result["details"]["modelscan"]["scanned_count"] == 0
    assert result["details"]["modelscan"]["passed"] is False
    assert result["details"]["weight_stats"]["tensors_analyzed"] == 1


@pytest.mark.parametrize("name", ["modelaudit", "gguf_guard"])
@pytest.mark.parametrize("verdict", ["not_applicable", "unavailable", "error", "fail"])
def test_required_gate_cannot_be_replaced_by_nonpass(tmp_path, monkeypatch, name, verdict):
    passing_scanners(monkeypatch)
    fn = {"modelaudit": "_run_modelaudit", "gguf_guard": "_run_gguf_guard_scan"}[name]
    monkeypatch.setattr(p, fn, lambda *_a, **_k: p._static_scanner_result(verdict, name, reason="fault"))
    result = p.check_static_scan(artifact(tmp_path, ".gguf"), {"models": {"require_scan": False}, "quarantine": {"scanner_missing": "warn-and-skip"}})
    assert not result["passed"]


@pytest.mark.parametrize("fault", ["zero", "bool", "no_version", "na_as_pass", "missing", "unknown"])
def test_malformed_outcome_cannot_pass(tmp_path, monkeypatch, fault):
    passing_scanners(monkeypatch)
    outcome = p._static_scanner_result("pass", "modelaudit", reason="fixture", scanner_version="1.0", scanned_count=1)
    if fault == "zero":
        outcome["scanned_count"] = 0
    elif fault == "bool":
        outcome["scanned_count"] = True
    elif fault == "no_version":
        outcome["scanner_version"] = "unknown"
    elif fault == "na_as_pass":
        outcome["verdict"] = "not_applicable"
    elif fault == "missing":
        outcome.pop("passed")
    else:
        outcome["verdict"] = "skipped"
    monkeypatch.setattr(p, "_run_modelaudit", lambda *_a, **_k: outcome)
    assert not p.check_static_scan(artifact(tmp_path, ".safetensors"))["passed"]


@pytest.mark.parametrize("gate", ["_check_pickle_polyglot", "_check_file_entropy", "_analyze_weight_distribution"])
def test_internal_gate_rejection_is_preserved(tmp_path, monkeypatch, gate):
    passing_scanners(monkeypatch)
    monkeypatch.setattr(p, gate, lambda *_a, **_k: {"passed": False, "reason": "fixture rejection"})
    assert not p.check_static_scan(artifact(tmp_path, ".safetensors"))["passed"]


def test_operational_weight_failure_is_not_optional(tmp_path, monkeypatch):
    passing_scanners(monkeypatch)
    monkeypatch.setattr(p, "_analyze_weight_distribution", lambda *_a: {"passed": True, "operational_error": True})
    assert not p.check_static_scan(artifact(tmp_path, ".safetensors"), {"quarantine": {"scanner_missing": "warn-and-skip"}})["passed"]


def test_directory_preserves_per_file_na_and_rejects_extra_pickle(tmp_path, monkeypatch):
    passing_scanners(monkeypatch)
    directory = tmp_path / "diffusion"
    directory.mkdir()
    (directory / "model_index.json").write_text('{"_class_name":"StableDiffusionPipeline"}')
    artifact(directory, ".safetensors")
    result = p.check_static_scan_directory(directory)
    assert result["passed"], result
    nested = result["files"]["model.safetensors"]["details"]["modelscan"]
    assert nested["verdict"] == "not_applicable" and nested["scanned_count"] == 0
    (directory / "unexpected.pkl").write_bytes(b"N.")
    assert not p.check_static_scan_directory(directory)["passed"]


def fake_modelscan(monkeypatch, callback):
    package = ModuleType("modelscan")
    api = ModuleType("modelscan.modelscan")
    api.ModelScan = lambda: SimpleNamespace(scan=callback)
    monkeypatch.setitem(sys.modules, "modelscan", package)
    monkeypatch.setitem(sys.modules, "modelscan.modelscan", api)


@pytest.mark.parametrize("fault", ["error", "zero", "skip", "raise", "wrong_file"])
def test_applicable_modelscan_errors_never_fall_back(tmp_path, monkeypatch, fault):
    path = tmp_path / "safe.pkl"
    path.write_bytes(b"N.")
    data = report(path)
    if fault == "error":
        data["errors"] = [{"category": "scanner_error"}]
    elif fault == "zero":
        data["summary"]["scanned"] = {"total_scanned": 0, "scanned_files": []}
    elif fault == "skip":
        data["summary"]["skipped"] = {"total_skipped": 1, "skipped_files": [{"source": str(path)}]}
    elif fault == "wrong_file":
        data["summary"]["scanned"]["scanned_files"] = ["different.pkl"]

    def callback(_path):
        if fault == "raise":
            raise ValueError("broken installed API")
        return data

    fake_modelscan(monkeypatch, callback)
    monkeypatch.setattr(p.subprocess, "run", no_process)
    assert p._run_modelscan(path)["verdict"] == "error"


@pytest.mark.parametrize("exit_code", [0, 1, 2, 3])
def test_cli_positive_report_must_match_exit(tmp_path, monkeypatch, exit_code):
    path = tmp_path / "safe.pkl"
    path.write_bytes(b"N.")
    original_import = builtins.__import__

    def absent(name, *args, **kwargs):
        if name == "modelscan.modelscan":
            raise ModuleNotFoundError("absent", name="modelscan")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", absent)

    def cli(args, **_kwargs):
        Path(args[args.index("--output-file") + 1]).write_text(json.dumps(report(path)))
        return SimpleNamespace(returncode=exit_code)

    monkeypatch.setattr(p.subprocess, "run", cli)
    outcome = p._run_modelscan(path)
    assert outcome["passed"] == (exit_code == 0)
    assert outcome["verdict"] == ("pass" if exit_code == 0 else "error")


@pytest.mark.parametrize("scanner,attribute", [(p._run_fickling_scan, "FICKLING_BIN"), (p._run_modelaudit, "MODELAUDIT_BIN"), (p._run_gguf_guard_scan, "GGUF_GUARD_BIN")])
def test_missing_applicable_scanner_is_unavailable(tmp_path, monkeypatch, scanner, attribute):
    path = artifact(tmp_path, ".gguf")
    if attribute == "FICKLING_BIN":
        path = tmp_path / "safe.pkl"
        path.write_bytes(b"N.")
    monkeypatch.setattr(p, attribute, str(tmp_path / "does-not-exist"))
    result = scanner(path)
    assert result["verdict"] == "unavailable" and not result["passed"]


@pytest.mark.parametrize("stdout,code", [("", 0), ("FAIL score=0.70 detected", 0), ("PASS score=0.00 clean", 2), ("PASS score=0.90 clean", 0), ("PASS score=0.00 clean", 9)])
def test_guard_incomplete_or_contradictory_result(tmp_path, monkeypatch, stdout, code):
    monkeypatch.setattr(p.subprocess, "run", lambda *_a, **_k: SimpleNamespace(stdout=stdout, stderr="", returncode=code))
    result = p._run_gguf_guard_scan(artifact(tmp_path, ".gguf"))
    assert result["verdict"] == "error"


def modelaudit_report(path):
    return {"issues": [], "success": True, "has_errors": False,
            "files_scanned": 1, "bytes_scanned": path.stat().st_size,
            "total_checks": 1, "passed_checks": 1, "failed_checks": 0,
            "checks": [{"name": "header", "status": "passed"}],
            "scanner_names": [path.suffix.lstrip(".")],
            "file_metadata": {str(path): {"file_size": path.stat().st_size}}}


@pytest.mark.parametrize("field,value", [
    ("success", False), ("has_errors", True), ("files_scanned", 0), ("files_scanned", True),
    ("bytes_scanned", 0), ("bytes_scanned", 10**9), ("total_checks", 0), ("dry_run", True),
    ("metadata", {"analysis_incomplete": True}), ("version", "0.2.51"),
    ("file_metadata", {}), ("passed_checks", 0), ("unknown_field", True),
    ("scanner_names", []), ("scanner_names", ["text"]),
    ("checks", [{"status": "skipped"}]), ("issues", [{"severity": "unknown"}]),
])
def test_modelaudit_requires_positive_complete_report(tmp_path, monkeypatch, field, value):
    path = artifact(tmp_path, ".safetensors")
    data = modelaudit_report(path)
    data[field] = value
    monkeypatch.setattr(p, "_scanner_cli_version", lambda *_a: "0.2.52")
    monkeypatch.setattr(p.subprocess, "run", lambda *_a, **_k: SimpleNamespace(stdout=json.dumps(data), returncode=0))
    assert p._run_modelaudit(path)["verdict"] == "error"


@pytest.mark.parametrize("code,expected", [(0, "pass"), (1, "error"), (2, "error")])
def test_modelaudit_identified_cli_version_and_exit_binding(tmp_path, monkeypatch, code, expected):
    path = artifact(tmp_path, ".safetensors")
    data = modelaudit_report(path)
    calls = []

    def cli(args, **_kwargs):
        calls.append(args)
        if args[-1] == "--version":
            return SimpleNamespace(stdout="modelaudit, version 0.2.52\n", returncode=0)
        return SimpleNamespace(stdout=json.dumps(data), returncode=code)

    monkeypatch.setattr(p.subprocess, "run", cli)
    result = p._run_modelaudit(path)
    assert result["verdict"] == expected
    assert result["scanner_version"] == "0.2.52"
    assert calls[0][0] == calls[1][0] == p.MODELAUDIT_BIN


@pytest.mark.parametrize("code,expected", [(0, "error"), (1, "fail"), (2, "error")])
def test_modelaudit_unsafe_evidence_and_exit_binding(tmp_path, code, expected):
    path = artifact(tmp_path, ".safetensors")
    data = modelaudit_report(path)
    data["issues"] = [{"severity": "warning"}]
    data["has_errors"] = True
    if expected == "error":
        with pytest.raises(ValueError):
            p._validated_modelaudit_result(data, path, "0.2.52", code)
    else:
        assert p._validated_modelaudit_result(data, path, "0.2.52", code)["verdict"] == expected


@pytest.mark.parametrize("output,code", [("", 0), ("0.2.52", 0), ("modelaudit, version 0.2.99", 0), ("modelaudit, version 0.2.52", 1)])
def test_unknown_or_failed_tool_version_cannot_scan(tmp_path, monkeypatch, output, code):
    calls = []

    def cli(args, **_kwargs):
        calls.append(args)
        return SimpleNamespace(stdout=output, returncode=code)

    monkeypatch.setattr(p.subprocess, "run", cli)
    assert p._run_modelaudit(artifact(tmp_path, ".safetensors"))["verdict"] == "error"
    assert len(calls) == 1


@pytest.mark.parametrize("severity,code,details,expected", [
    ("LIKELY_SAFE", 0, {}, "pass"), ("SUSPICIOUS", 1, {}, "fail"),
    ("OVERTLY_MALICIOUS", 1, {"AnalysisResult": {"bad": "global"}}, "fail"),
    ("LIKELY_SAFE", 1, {}, "error"), ("LIKELY_SAFE", 0, {"unexpected": "finding"}, "error"),
    ("UNKNOWN", 0, {}, "error"), ("LIKELY_SAFE", 2, {}, "error"),
])
def test_fickling_classification_and_separate_output(tmp_path, monkeypatch, severity, code, details, expected):
    path = tmp_path / "safe.pkl"
    path.write_bytes(b"N.")
    monkeypatch.setattr(p, "_scanner_cli_version", lambda *_a: "0.1.12")

    def cli(args, **kwargs):
        output = Path(args[args.index("--json-output") + 1])
        assert output != path and args[-1] == str(path)
        assert kwargs["stdin"] == p.subprocess.DEVNULL
        output.write_text(json.dumps({"severity": severity, "analysis": "fixture analysis", "detailed_results": details}))
        return SimpleNamespace(returncode=code)

    monkeypatch.setattr(p.subprocess, "run", cli)
    assert p._run_fickling_scan(path)["verdict"] == expected
    assert path.read_bytes() == b"N."


def test_directory_report_budget_is_nonpassing(tmp_path, monkeypatch):
    passing_scanners(monkeypatch)
    (tmp_path / "model_index.json").write_text('{"_class_name":"StableDiffusionPipeline"}')
    artifact(tmp_path, ".safetensors")
    monkeypatch.setattr(p, "STATIC_DIRECTORY_REPORT_MAX_BYTES", 1)
    result = p.check_static_scan_directory(tmp_path)
    assert not result["passed"] and "safety limit" in result["reason"]


def test_real_installed_modelscan(tmp_path):
    if os.getenv("SECAI_TEST_REAL_SCANNERS") != "1":
        pytest.skip("explicit installed-scanner qualification run required")
    from modelscan.modelscan import ModelScan

    for suffix in (".gguf", ".safetensors"):
        path = artifact(tmp_path, suffix)
        actual = ModelScan().scan(str(path))
        assert actual["summary"]["scanned"]["total_scanned"] == 0
        assert p._run_modelscan(path)["verdict"] == "not_applicable"
    path = tmp_path / "safe.pkl"
    path.write_bytes(b"N.")
    assert p._run_modelscan(path)["verdict"] == "pass"


def test_real_guard(tmp_path, monkeypatch):
    binary = os.getenv("SECAI_TEST_GGUF_GUARD")
    if not binary:
        pytest.skip("explicit actual gguf-guard binary required")
    monkeypatch.setattr(p, "GGUF_GUARD_BIN", binary)
    result = p._run_gguf_guard_scan(artifact(tmp_path, ".gguf"))
    assert result["verdict"] == "pass", result
    assert result["scanner_version"] == "0.2.0"


def test_real_modelaudit_and_fickling(tmp_path, monkeypatch):
    binaries = os.getenv("SECAI_TEST_SCANNER_BIN_DIR")
    if not binaries:
        pytest.skip("explicit installed ModelAudit/Fickling binaries required")
    monkeypatch.setattr(p, "MODELAUDIT_BIN", str(Path(binaries) / "modelaudit"))
    monkeypatch.setattr(p, "FICKLING_BIN", str(Path(binaries) / "fickling"))
    for suffix in (".gguf", ".safetensors"):
        assert p._run_modelaudit(artifact(tmp_path, suffix))["verdict"] == "pass"
    path = tmp_path / "safe.pkl"
    path.write_bytes(b"N.")
    assert p._run_fickling_scan(path)["verdict"] == "pass"
    path.write_bytes(b"cos\nsystem\n.")  # Static global reference only; never executed.
    assert p._run_fickling_scan(path)["verdict"] == "fail"
    path.write_bytes(b"not pickle")
    assert p._run_fickling_scan(path)["verdict"] == "error"


def test_real_modelscan_cli(tmp_path, monkeypatch):
    binaries = os.getenv("SECAI_TEST_SCANNER_BIN_DIR")
    if not binaries:
        pytest.skip("explicit installed ModelScan binary required")
    original_import = builtins.__import__

    def absent(name, *args, **kwargs):
        if name == "modelscan.modelscan":
            raise ModuleNotFoundError("API unavailable in coordinator", name="modelscan")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", absent)
    monkeypatch.setattr(p, "MODELSCAN_BIN", str(Path(binaries) / "modelscan"))
    path = tmp_path / "safe.pkl"
    path.write_bytes(b"N.")
    result = p._run_modelscan(path)
    assert result["verdict"] == "pass", result
    assert result["scanner"] == "modelscan-cli" and result["scanner_version"] == "0.8.8"


def test_modelaudit_header_coverage_is_not_whole_file_coverage(tmp_path):
    path = artifact(tmp_path, ".gguf")
    data = modelaudit_report(path)
    data["bytes_scanned"] = 57
    result = p._validated_modelaudit_result(data, path, "0.2.52", 0)
    assert result["passed"] and result["bytes_scanned"] == 57
    assert result["artifact_size_bytes"] == path.stat().st_size > 57


@pytest.mark.parametrize("count", [0, False, None])
def test_missing_numerical_weight_coverage_is_nonpassing(tmp_path, monkeypatch, count):
    passing_scanners(monkeypatch)
    monkeypatch.setattr(p, "_analyze_weight_distribution", lambda *_a: {"passed": True, "tensors_analyzed": count})
    result = p.check_static_scan(artifact(tmp_path, ".safetensors"))
    assert not result["passed"] and "positive tensor coverage" in result["reason"]
