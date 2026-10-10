"""The v2 client requires capability identity before any admission request."""
import copy

import pytest
from test_registry_protocol import capabilities_fixture, fixture_values

from quarantine.admission_client import AdmissionClient, RegistryUnavailable
from quarantine.registry_protocol import ProtocolError, encode_request


@pytest.mark.parametrize("origin", ["http://remote.example", "https://user:password@example.com", "https://example.com/path",
                                      "https://example.com?token=secret", "http://127.0.0.1:invalid", "file:///etc/passwd"])
def test_registry_origin_refuses_unsafe_transport(origin):
    with pytest.raises(ProtocolError):
        AdmissionClient(origin, "t" * 32)


def test_required_behavior_refused_during_capability_preflight(monkeypatch, tmp_path):
    client = AdmissionClient("http://127.0.0.1:8470", "t" * 32)
    capabilities = capabilities_fixture()
    calls = []

    def request(path, **kwargs):
        calls.append((path, kwargs))
        return 200, capabilities

    monkeypatch.setattr(client, "_request", request)
    prepare, context, _receipt = fixture_values()
    with pytest.raises(ProtocolError, match="behavior runtime unavailable"):
        client.preflight(prepare["policy_bundle"], context["pipeline_sha256"], artifact_format="gguf")
    assert calls == [("/v2/admission/capabilities", {})]
    assert client.capabilities is None and list(tmp_path.iterdir()) == []
    assert client.preflight(prepare["policy_bundle"], context["pipeline_sha256"], artifact_format="safetensors") == capabilities


def test_directory_and_unready_backend_never_send_prepare(monkeypatch):
    client = AdmissionClient("http://127.0.0.1:8470", "t" * 32)
    prepare, context, _receipt = fixture_values()
    monkeypatch.setattr(client, "_request", lambda *_args, **_kwargs: pytest.fail("unsupported directory touched backend"))
    with pytest.raises(ProtocolError):
        client.preflight(prepare["policy_bundle"], context["pipeline_sha256"], artifact_format="directory")
    with pytest.raises(ProtocolError):
        client.prepare(encode_request(prepare))
    wrong = copy.deepcopy(capabilities_fixture())
    wrong["service"] = "main-registry"
    monkeypatch.setattr(client, "_request", lambda *_args, **_kwargs: (200, wrong))
    with pytest.raises(ProtocolError):
        client.preflight(prepare["policy_bundle"], context["pipeline_sha256"], artifact_format="safetensors")
    assert client.capabilities is None


def test_lookup_distinguishes_absence_from_unknown_durability(monkeypatch):
    client = AdmissionClient("http://127.0.0.1:8470", "t" * 32)
    client.capabilities = capabilities_fixture()
    prepare, _context, _receipt = fixture_values()

    def missing(*_args, **_kwargs):
        raise RegistryUnavailable(404)

    monkeypatch.setattr(client, "_request", missing)
    assert client.lookup(encode_request(prepare)) is None

    def uncertain(*_args, **_kwargs):
        raise RegistryUnavailable(503)

    monkeypatch.setattr(client, "_request", uncertain)
    with pytest.raises(RegistryUnavailable) as error:
        client.lookup(encode_request(prepare))
    assert error.value.status == 503
