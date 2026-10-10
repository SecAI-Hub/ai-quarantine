"""Bounded authenticated helper-v2 transport with exact-byte retries."""
from __future__ import annotations

import ipaddress
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from .registry_protocol import (
    MAX_BODY_BYTES,
    NAME,
    ProtocolError,
    strict_json,
    validate_artifact,
    validate_capabilities,
    validate_commit,
    validate_preparation,
)


class RegistryUnavailable(ProtocolError):
    def __init__(self, status: int | None = None):
        self.status = status
        super().__init__("helper registry response unavailable" + (f" (HTTP {status})" if status else ""))


class _NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class AdmissionClient:
    def __init__(self, base_url: str, token: str):
        if (not isinstance(base_url, str) or not base_url or base_url != base_url.strip() or
                len(base_url) > 2048 or any(ord(c) <= 0x20 or ord(c) == 0x7f for c in base_url)):
            raise ProtocolError("helper registry origin invalid")
        parsed = urlparse(base_url)
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username is not None or
                parsed.password is not None or parsed.path not in {"", "/"} or parsed.query or parsed.fragment or parsed.params):
            raise ProtocolError("helper registry requires a credential-free HTTP(S) origin")
        try:
            _ = parsed.port
            loopback = parsed.hostname.lower() == "localhost" or ipaddress.ip_address(parsed.hostname).is_loopback
        except ValueError:
            loopback = parsed.hostname.lower() == "localhost"
            try:
                _ = parsed.port
            except ValueError as error:
                raise ProtocolError("helper registry port invalid") from error
        if parsed.scheme == "http" and not loopback:
            raise ProtocolError("remote helper registry requires HTTPS")
        if not isinstance(token, str) or not 32 <= len(token) <= 4096 or any(not 0x21 <= ord(c) <= 0x7e for c in token):
            raise ProtocolError("helper registry requires a bounded service token")
        self.base_url, self._token = base_url.rstrip("/"), token
        self._opener = build_opener(ProxyHandler({}), _NoRedirects())
        self.capabilities = None

    def _request(self, path: str, *, raw: bytes | None = None, admission_id: str | None = None):
        headers = {"Authorization": "Bearer " + self._token, "Accept": "application/json"}
        if raw is not None:
            strict_json(raw)
            headers.update({"Content-Type": "application/json", "Idempotency-Key": admission_id})
        request = Request(self.base_url + path, data=raw, headers=headers, method="GET" if raw is None else "POST")
        try:
            # Fixed validated origin with redirects and proxies disabled above.
            with self._opener.open(request, timeout=30) as response:  # nosec B310
                status = response.status
                content = response.read(MAX_BODY_BYTES + 1)
        except HTTPError as error:
            error.close()
            raise RegistryUnavailable(error.code) from None
        except (OSError, URLError) as error:
            raise RegistryUnavailable() from error
        if status not in {200, 201}:
            raise RegistryUnavailable(status)
        return status, strict_json(content)

    def preflight(self, bundle: dict, pipeline_sha256: str, *, artifact_format: str):
        # Call before artifact hashing, new journal/copy state or recovery writes.
        self.capabilities = None
        if artifact_format not in {"gguf", "safetensors"}:
            raise ProtocolError("helper-v2 admits only file GGUF/safetensors")
        status, value = self._request("/v2/admission/capabilities")
        if status != 200:
            raise RegistryUnavailable(status)
        value = validate_capabilities(value, bundle, pipeline_sha256)
        if artifact_format == "gguf" and value["scan_context"]["require_behavior"]:
            raise ProtocolError("bounded GGUF behavior runtime unavailable; preparation is forbidden")
        self.capabilities = value
        return value

    def _context(self):
        if self.capabilities is None:
            raise ProtocolError("helper capability preflight is required before admission/readback")
        return self.capabilities["scan_context"]

    def prepare(self, raw: bytes, *, commit_raw: bytes | None = None):
        context = self._context()
        value = strict_json(raw)
        validate_preparation(value)
        status, artifact = self._request("/v2/admission/prepare", raw=raw, admission_id=value["admission_id"])
        return status, validate_artifact(artifact, raw, context, commit_raw)

    def commit(self, prepare_raw: bytes, commit_raw: bytes):
        context = self._context()
        value = strict_json(commit_raw)
        validate_commit(value)
        prepare = strict_json(prepare_raw)
        validate_preparation(prepare)
        if value["admission_id"] != prepare["admission_id"]:
            raise ProtocolError("commit does not match exact preparation")
        status, artifact = self._request("/v2/admission/commit", raw=commit_raw, admission_id=value["admission_id"])
        artifact = validate_artifact(artifact, prepare_raw, context, commit_raw)
        if "commit_receipt" not in artifact["admission"]:
            raise ProtocolError("registry commit response has no complete historical receipt")
        return status, artifact

    def lookup(self, prepare_raw: bytes, *, commit_raw: bytes | None = None):
        context = self._context()
        prepare = strict_json(prepare_raw)
        validate_preparation(prepare)
        name = prepare["name"]
        if NAME.fullmatch(name) is None:
            raise ProtocolError("receipt lookup name invalid")
        try:
            status, artifact = self._request("/v1/model?name=" + quote(name, safe=""))
        except RegistryUnavailable as error:
            if error.status == 404:
                return None
            raise
        if status != 200:
            raise RegistryUnavailable(status)
        return validate_artifact(artifact, prepare_raw, context, commit_raw)
