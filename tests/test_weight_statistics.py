"""Regression coverage for sampled finite weights and precise threshold decisions."""

import json
import struct
from pathlib import Path

import pytest

from quarantine import pipeline


@pytest.mark.parametrize("tensor_type", [0, 1, 8])
@pytest.mark.parametrize("invalid", [None, float("nan"), float("inf"), -float("inf")])
def test_gguf_weight_reader_records_nonfinite_samples(tmp_path, tensor_type, invalid):
    count = 64 if tensor_type == 8 else 32
    if tensor_type == 8:
        valid = struct.pack("<e32b", 1.0, *([1, -1] * 16))
        scale = 1.0 if invalid is None else invalid
        data = valid + struct.pack("<e32b", scale, *([1, -1] * 16))
    else:
        values = [1.0, -1.0] * 16
        if invalid is not None:
            values[-1] = invalid
        data = struct.pack("<32" + ("f" if tensor_type == 0 else "e"), *values)
    header = struct.pack("<4sIQQ", b"GGUF", 3, 1, 0)
    header += struct.pack("<Q", 6) + b"weight"
    header += struct.pack("<IQIQ", 1, count, tensor_type, 0)
    header += b"\0" * (-len(header) % 32)
    payload = tmp_path / "sample.gguf"
    payload.write_bytes(header + data)

    result = pipeline._analyze_weight_distribution(payload)

    if invalid is None:
        assert result["passed"] is True
        assert result["tensors_analyzed"] == 1
    else:
        assert result["operational_error"] is True


@pytest.mark.parametrize("count", [0, 1, 2, 15])
def test_weight_stats_preserve_small_finite_tensor_handling(count):
    data = struct.pack(f"<{count}f", *([1.0] * count))

    assert pipeline._compute_tensor_stats(data, count, "f32") is None


def test_weight_stats_preserve_empty_buffer_handling():
    assert pipeline._compute_tensor_stats(b"", 32, "f32") is None


def test_weight_stats_keep_extreme_kurtosis_rejection():
    stats = pipeline._stats_from_values([0.0] * 255 + [1.0])

    assert stats["kurtosis"] > pipeline.WEIGHT_STATS_MAX_KURTOSIS
    assert any("extreme kurtosis" in issue for issue in pipeline._check_weight_anomalies("weight", stats))


@pytest.mark.parametrize("dtype,format_code", [("f32", "f"), ("f16", "e")])
@pytest.mark.parametrize("count", [2, 32])
@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_weight_stats_reject_sampled_nonfinite_values(dtype, format_code, count, invalid):
    values = [1.0, -1.0] * (count // 2)
    values[-1] = invalid
    data = struct.pack(f"<{count}{format_code}", *values)

    with pytest.raises(ValueError, match="non-finite sampled tensor value"):
        pipeline._compute_tensor_stats(data, count, dtype)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_weight_stats_reject_nonfinite_quantization_scale(invalid):
    valid_block = struct.pack("<e32b", 1.0, *([1, -1] * 16))
    invalid_block = struct.pack("<e32b", invalid, *([1, -1] * 16))

    with pytest.raises(ValueError, match="non-finite sampled Q8_0 scale"):
        pipeline._compute_tensor_stats(valid_block + invalid_block, 64, "q8_0")


@pytest.mark.parametrize("count", [2, 32])
@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_safetensors_weight_analysis_fails_closed_for_nonfinite_values(
    tmp_path, monkeypatch, count, invalid
):
    values = [1.0, -1.0] * (count // 2)
    values[-1] = invalid
    payload = tmp_path / "nonfinite.safetensors"
    _write_safetensors(
        payload,
        {"weight": {"dtype": "F32", "shape": [count], "data_offsets": [0, count * 4]}},
        struct.pack(f"<{count}f", *values),
    )

    analysis = pipeline._analyze_weight_distribution(payload)
    assert analysis["operational_error"] is True
    for function, scanner in (
        ("_run_modelscan", "modelscan"),
        ("_run_fickling_scan", "fickling"),
        ("_run_gguf_guard_scan", "gguf-guard"),
        ("_run_modelaudit", "modelaudit"),
    ):
        outcome = pipeline._static_scanner_result(
            "pass" if scanner == "modelaudit" else "not_applicable",
            scanner,
            reason="synthetic scanner fixture",
            scanner_version="1.0" if scanner == "modelaudit" else None,
            scanned_count=1 if scanner == "modelaudit" else 0,
        )
        monkeypatch.setattr(
            pipeline, function, lambda *_args, _result=outcome, **_kwargs: _result
        )
    result = pipeline.check_static_scan(payload)
    assert result["passed"] is False
    assert result["scanner"] == "weight_stats"
    assert result["details"]["weight_stats"]["operational_error"] is True


@pytest.mark.parametrize("amplitude,passed", [(1.25e-6, True), (5e-7, False)])
def test_safetensors_weight_variance_uses_unrounded_threshold(tmp_path, amplitude, passed):
    values = [amplitude, -amplitude] * 16
    payload = tmp_path / "small-variance.safetensors"
    _write_safetensors(
        payload,
        {"weight": {"dtype": "F32", "shape": [32], "data_offsets": [0, 128]}},
        struct.pack("<32f", *values),
    )

    result = pipeline._analyze_weight_distribution(payload)

    assert result["passed"] is passed
    if passed:
        assert result["tensor_stats"][0]["variance"] > pipeline.WEIGHT_STATS_MIN_VARIANCE
    else:
        assert "near-zero variance" in result["reason"]


@pytest.mark.parametrize("zeros,passed", [(9900, True), (9901, False)])
def test_safetensors_weight_zero_fraction_uses_unrounded_threshold(tmp_path, zeros, passed):
    count = 10001
    values = [0.0] * zeros + [1.0 if i % 2 else -1.0 for i in range(count - zeros)]
    payload = tmp_path / "zero-fraction.safetensors"
    _write_safetensors(
        payload,
        {"weight": {"dtype": "F32", "shape": [count], "data_offsets": [0, count * 4]}},
        struct.pack(f"<{count}f", *values),
    )

    result = pipeline._analyze_weight_distribution(payload)

    assert result["passed"] is passed
    if not passed:
        assert "zeros" in result["reason"]


def _write_safetensors(
    path: Path,
    header: dict,
    data: bytes,
) -> None:
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(len(encoded).to_bytes(8, "little") + encoded + data)
