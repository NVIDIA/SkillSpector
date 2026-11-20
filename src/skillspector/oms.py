# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded, untrusted DSSE content projections; never signature verification."""

from __future__ import annotations

import base64
import binascii
import json
import re
from array import array
from dataclasses import dataclass

from skillspector.artifacts import SecurityTextView

_MAX_CHARS = 1_000_000
_MAX_NODES = 10_000
_PRINTABLE = re.compile(r"[^\x00-\x1f\x7f-\x9f\ufffd]{4,}")


@dataclass(frozen=True)
class OMSProjection:
    view: SecurityTextView
    complete: bool


def _object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _load(text: str) -> object:
    value = json.loads(text, object_pairs_hook=_object)
    pending = [(value, 0)]
    count = 0
    while pending:
        item, depth = pending.pop()
        count += 1
        if count > _MAX_NODES or depth > 64:
            raise ValueError("JSON projection limit")
        if isinstance(item, dict):
            pending.extend((child, depth + 1) for child in item.values())
        elif isinstance(item, list):
            pending.extend((child, depth + 1) for child in item)
    return value


def _decode(value: object) -> bytes:
    if not isinstance(value, str):
        raise ValueError("expected base64 string")
    return base64.b64decode(value, validate=True)


def _printable(data: bytes) -> list[str]:
    # Preserve readable Unicode and format characters for semantic/obfuscation
    # analysis. Invalid bytes and control bytes separate runs rather than
    # silently joining attacker-controlled fragments.
    return _PRINTABLE.findall(data.decode("utf-8", errors="replace"))


def _raw_projection(content: str) -> OMSProjection | None:
    recognized = "dsseEnvelope" in content
    if not recognized:
        # Recognition only: escaped field names also identify an oversized or
        # duplicate-key bundle. Never use this permissive parse as analysis data.
        try:
            untrusted = json.loads(content)
            recognized = isinstance(untrusted, dict) and "dsseEnvelope" in untrusted
        except (ValueError, RecursionError):
            pass
    return OMSProjection(SecurityTextView("raw", content), False) if recognized else None


def _certificate(value: object, *, pem: bool = False) -> dict[str, object]:
    from cryptography import x509
    from cryptography.hazmat.primitives.serialization import Encoding

    encoded = _decode(value)
    cert = (x509.load_pem_x509_certificate if pem else x509.load_der_x509_certificate)(encoded)
    if cert.public_bytes(Encoding.PEM if pem else Encoding.DER) != encoded:
        raise ValueError("noncanonical certificate")
    der = cert.public_bytes(Encoding.DER)
    return {
        "subject": cert.subject.rfc4514_string(),
        "issuer": cert.issuer.rfc4514_string(),
        "extensions": [str(extension.value) for extension in cert.extensions],
        "printable_der": _printable(der),
    }


def project_oms_content(content: str) -> OMSProjection | None:
    """Decode untrusted DSSE content without allowing parser failures to escape.

    Readable content and unknown fields remain in the view. Unsupported or
    over-limit content remains raw and partial; decoding never verifies a signer.
    """
    if not content.lstrip().startswith("{"):
        return None
    if len(content) > _MAX_CHARS:
        return _raw_projection(content)
    try:
        bundle = _load(content)
    except (ValueError, RecursionError):
        return _raw_projection(content)
    if not isinstance(bundle, dict) or "dsseEnvelope" not in bundle:
        return None
    try:
        return _project_bundle(bundle, content)
    except Exception:
        # All callers retain raw evidence and an explicit gap if a parser fails.
        return OMSProjection(SecurityTextView("raw", content), False)


def _project_bundle(bundle: dict[str, object], content: str) -> OMSProjection:
    envelope = bundle["dsseEnvelope"]
    if not isinstance(envelope, dict):
        return OMSProjection(SecurityTextView("raw", content), False)
    complete = True
    try:
        payload = _load(_decode(envelope.get("payload")).decode("utf-8"))
        if not isinstance(payload, (dict, list)):
            raise ValueError("expected JSON payload")
        envelope["payload"] = payload
    except (ValueError, UnicodeError, binascii.Error, RecursionError):
        complete = False

    def project_certificate(entry: object, key: str = "rawBytes", *, pem: bool = False) -> None:
        nonlocal complete
        try:
            if not isinstance(entry, dict):
                raise ValueError("expected certificate object")
            entry[key] = _certificate(entry.get(key), pem=pem)
        except Exception:
            # cryptography exposes malformed-DER exceptions outside ValueError.
            complete = False

    material = bundle.get("verificationMaterial")
    if not isinstance(material, dict):
        complete = False
    else:
        recognized_material = False
        if "x509CertificateChain" in material:
            recognized_material = True
            chain = material["x509CertificateChain"]
            certificates = chain.get("certificates") if isinstance(chain, dict) else None
            if not isinstance(certificates, list) or not 0 < len(certificates) <= 64:
                complete = False
            else:
                for entry in certificates:
                    project_certificate(entry)
        if "certificate" in material:
            recognized_material = True
            project_certificate(material["certificate"])
        if "publicKey" in material:
            recognized_material = True
            key = material["publicKey"]
            if not isinstance(key, dict) or not isinstance(key.get("hint"), str):
                complete = False
        if not recognized_material:
            complete = False

        entries = material.get("tlogEntries", [])
        if not isinstance(entries, list) or len(entries) > 64:
            complete = False
        else:
            for entry in entries:
                try:
                    if not isinstance(entry, dict):
                        raise ValueError("expected transparency log entry")
                    body = _load(_decode(entry.get("canonicalizedBody")).decode("utf-8"))
                    if not isinstance(body, dict):
                        raise ValueError("expected transparency log JSON object")
                    entry["canonicalizedBody"] = body
                    spec = body.get("spec")
                    if isinstance(spec, dict):
                        signatures = spec.get("signatures", [])
                        if not isinstance(signatures, list) or len(signatures) > 64:
                            complete = False
                        else:
                            for signature in signatures:
                                project_certificate(signature, "verifier", pem=True)
                        record = spec.get("hashedRekordV002")
                        signature = record.get("signature") if isinstance(record, dict) else None
                        verifier = (
                            signature.get("verifier") if isinstance(signature, dict) else None
                        )
                        if isinstance(verifier, dict) and "x509Certificate" in verifier:
                            project_certificate(verifier["x509Certificate"])
                except Exception:
                    complete = False

    try:
        from cryptography.hazmat.primitives.asymmetric.utils import (
            decode_dss_signature,
            encode_dss_signature,
        )

        signatures = envelope.get("signatures")
        if not isinstance(signatures, list) or not 0 < len(signatures) <= 64:
            complete = False
        else:
            for entry in signatures:
                try:
                    if not isinstance(entry, dict):
                        raise ValueError("expected signature object")
                    der = _decode(entry.get("sig"))
                    r, s = decode_dss_signature(der)
                    if (
                        not (0 < r.bit_length() <= 521 and 0 < s.bit_length() <= 521)
                        or encode_dss_signature(r, s) != der
                    ):
                        raise ValueError("noncanonical ECDSA signature")
                    entry["sig"] = {"printable_der": _printable(der)}
                except Exception:
                    complete = False
    except ImportError:
        complete = False

    projected = json.dumps(bundle, ensure_ascii=False, indent=2)
    # Keep real Unicode (including bidi controls) visible while escaping lone
    # surrogates that cannot be passed to UTF-8 analyzers or provider clients.
    projected = projected.encode("utf-8", "backslashreplace").decode("utf-8")
    if len(projected) > _MAX_CHARS:
        return OMSProjection(SecurityTextView("raw", content), False)
    # Findings refer to the original carrier, not synthetic files or lines in
    # decoded data. Their evidence explicitly labels this derived view.
    return OMSProjection(
        SecurityTextView("oms-decoded", projected, array("I", [0]) * len(projected)),
        complete,
    )
