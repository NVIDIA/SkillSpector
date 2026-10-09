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


def project_oms_content(content: str) -> OMSProjection | None:
    """Decode DSSE payloads independently of file names or media-type spelling.

    Only strictly parsed DER certificate/signature fields lose their base64
    representation. Their readable content and every unknown wrapper field stay
    in the view. Unsupported fields stay raw and make coverage partial. Neither
    successful decoding nor a complete projection establishes signer trust.
    """
    if not content.lstrip().startswith("{") or len(content) > _MAX_CHARS:
        return None
    try:
        bundle = _load(content)
    except (ValueError, RecursionError):
        recognized = "dsseEnvelope" in content
        if not recognized:
            # Recognition only: a duplicate-key bundle may escape its field
            # names. Never use this permissive parse as analysis content.
            try:
                untrusted = json.loads(content)
                recognized = isinstance(untrusted, dict) and "dsseEnvelope" in untrusted
            except (ValueError, RecursionError):
                pass
        return OMSProjection(SecurityTextView("raw", content), False) if recognized else None
    if not isinstance(bundle, dict) or "dsseEnvelope" not in bundle:
        return None
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

    # cryptography is optional: environments without it inspect the raw binary
    # fields and explicitly report the interpretation gap.
    try:
        from cryptography import x509
        from cryptography.exceptions import UnsupportedAlgorithm
        from cryptography.hazmat.primitives.asymmetric.utils import (
            decode_dss_signature,
            encode_dss_signature,
        )
        from cryptography.hazmat.primitives.serialization import Encoding

        material = bundle.get("verificationMaterial")
        chain = material.get("x509CertificateChain") if isinstance(material, dict) else None
        certificates = chain.get("certificates") if isinstance(chain, dict) else None
        if not isinstance(certificates, list) or not 0 < len(certificates) <= 64:
            complete = False
        else:
            for entry in certificates:
                try:
                    if not isinstance(entry, dict):
                        raise ValueError("expected certificate object")
                    der = _decode(entry.get("rawBytes"))
                    cert = x509.load_der_x509_certificate(der)
                    if cert.public_bytes(Encoding.DER) != der:
                        raise ValueError("noncanonical certificate")
                    readable = {
                        "subject": cert.subject.rfc4514_string(),
                        "issuer": cert.issuer.rfc4514_string(),
                        "extensions": [str(extension.value) for extension in cert.extensions],
                        "printable_der": _printable(der),
                    }
                    entry["rawBytes"] = readable
                except (
                    ValueError,
                    TypeError,
                    binascii.Error,
                    x509.DuplicateExtension,
                    UnsupportedAlgorithm,
                ):
                    complete = False
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
                except (ValueError, TypeError, binascii.Error):
                    complete = False
    except ImportError:
        complete = False

    projected = json.dumps(bundle, ensure_ascii=False, indent=2)
    if len(projected) > _MAX_CHARS:
        return OMSProjection(SecurityTextView("raw", content), False)
    # Findings refer to the original carrier, not synthetic files or lines in
    # decoded data. Their evidence explicitly labels this derived view.
    return OMSProjection(
        SecurityTextView("oms-decoded", projected, array("I", [0]) * len(projected)), complete
    )
