"""The canonical encoding: RFC 8785 bytes, the one hash function, the per-turn key and digests.

Every hash in the core is SHA-256 over the bytes ``encode`` gives. The module is pure: it
holds no key and no hash, and reads no record.
"""

import hashlib
import hmac
import json
import math
import re
from collections.abc import Sequence
from typing import Any, NoReturn, cast

from jurat._values import (
    MAX_DOCUMENT_DEPTH,
    MAX_INTEGER,
    clipped,
    domain_error,
    string_data,
    type_name,
)


class _CanonicalError(ValueError):
    """A value outside the value domain, or a malformed hash string; nothing is encoded."""


class _DecodeError(ValueError):
    """Bytes that do not decode: see ``decode``."""


class _DuplicateKey(_DecodeError):
    """Bytes holding an object with a key twice; a reader may name this cause alone."""


_SHA256_REF = re.compile(r"sha256:[0-9a-f]{64}")

# RFC 8785 section 3.2.2.2: the two-character escapes, then \u00xx for the other controls.
_ESCAPES = {
    "\b": "\\b",
    "\t": "\\t",
    "\n": "\\n",
    "\f": "\\f",
    "\r": "\\r",
    '"': '\\"',
    "\\": "\\\\",
}
_ESCAPES.update({chr(c): f"\\u{c:04x}" for c in range(0x20) if chr(c) not in _ESCAPES})
_NEEDS_ESCAPE = re.compile('[\x00-\x1f"\\\\]')


def encode(value: object, max_depth: int | None = None) -> bytes:
    """The canonical bytes of a value in the value domain, or ``_CanonicalError``.

    With no ``max_depth`` the value is a value: the domain bounds its depth and its size
    (``domain_error``). A caller encoding a record item, a document or another holder of
    values, each bounded where it stands, passes the holder's own ``max_depth``, never above
    ``MAX_DOCUMENT_DEPTH``, so a document nesting more than that is refused with the rest; a
    holder is held to no size bound (``domain_error``).
    """
    if max_depth is not None and max_depth > MAX_DOCUMENT_DEPTH:
        raise ValueError(f"max_depth {max_depth} is above {MAX_DOCUMENT_DEPTH}")
    err = domain_error(value, max_depth)
    if err is not None:
        raise _CanonicalError(err)
    parts: list[str] = []
    _encode_into(value, parts)
    return "".join(parts).encode("utf-8")


def _encode_into(value: object, out: list[str]) -> None:
    # The value is in the domain: domain_error has checked its types, cycles and depth, so
    # this recursion is at most max_depth deep.
    if value is None:
        out.append("null")
    elif value is True:
        out.append("true")
    elif value is False:
        out.append("false")
    elif isinstance(value, str):
        out.append(_string(value))
    elif isinstance(value, int):
        out.append(int.__repr__(value))
    elif isinstance(value, float):
        out.append(_number(value))
    elif isinstance(value, dict):
        # Keys by their string data: domain_error has checked no two share it.
        members = [(string_data(k), v) for k, v in cast("dict[str, object]", value).items()]
        out.append("{")
        for i, (key, member) in enumerate(sorted(members, key=lambda kv: _utf16_key(kv[0]))):
            if i:
                out.append(",")
            out.append(_string(key))
            out.append(":")
            _encode_into(member, out)
        out.append("}")
    else:
        out.append("[")
        for i, member in enumerate(cast("list[object] | tuple[object, ...]", value)):
            if i:
                out.append(",")
            _encode_into(member, out)
        out.append("]")


def _utf16_key(key: str) -> bytes:
    # Big-endian UTF-16 bytes compare as the code-unit sequence RFC 8785 sorts by.
    return key.encode("utf-16-be")


def _string(value: str) -> str:
    text = string_data(value)
    return '"' + _NEEDS_ESCAPE.sub(lambda m: _ESCAPES[m.group()], text) + '"'


def _number(value: float) -> str:
    """ECMAScript Number::toString (ECMA-262, 7.1.12.1) of a finite double."""
    if value == 0:
        return "0"
    sign = "-" if value < 0 else ""
    # repr gives the shortest digits that round-trip, nearest the value: ES6's s and k.
    mantissa, _, exp = float.__repr__(abs(value)).partition("e")
    whole, _, frac = mantissa.partition(".")
    digits = whole + frac
    n = len(whole) + (int(exp) if exp else 0)
    stripped = digits.lstrip("0")
    n -= len(digits) - len(stripped)
    digits = stripped.rstrip("0")
    k = len(digits)
    if k <= n <= 21:
        body = digits + "0" * (n - k)
    elif 0 < n <= 21:
        body = digits[:n] + "." + digits[n:]
    elif -6 < n <= 0:
        body = "0." + "0" * -n + digits
    else:
        e = n - 1
        exponent = ("+" if e >= 0 else "-") + str(abs(e))
        body = digits[0] + ("." + digits[1:] if k > 1 else "") + "e" + exponent
    return sign + body


def decode(data: bytes) -> Any:
    """Parse UTF-8 JSON bytes, refusing what has no one reading as domain values.

    Refuses, raising ``_DecodeError`` with the parse error chained where there is one: a
    document nesting more than ``MAX_DOCUMENT_DEPTH`` deep, counted on the bytes before they
    are parsed so the refusal is alike on every Python; bytes that are
    not UTF-8 JSON; a duplicate key at any depth; the constants ``NaN``, ``Infinity`` and
    ``-Infinity``; a number whose double is not finite, such as ``1e400``; and an integer
    literal beyond 2^53 - 1 in magnitude that names no double exactly (see
    ``integer_literal``). An integer literal within that magnitude is an ``int``, and one
    beyond it is the ``float`` it names. The result is not otherwise checked against the
    value domain; the caller does that.
    """
    depth = document_depth(data)
    if depth > MAX_DOCUMENT_DEPTH:
        raise _DecodeError(f"the document nests more than {MAX_DOCUMENT_DEPTH} deep")
    try:
        text = data.decode("utf-8")
        return json.loads(
            text,
            parse_int=_parse_int,
            parse_float=_parse_float,
            parse_constant=_refuse_constant,
            object_pairs_hook=_no_duplicates,
        )
    except _DecodeError:
        raise
    except ValueError as exc:  # Unicode and JSON decode errors included
        raise _DecodeError(str(exc)) from exc


_JSON_STRING = re.compile(rb'"[^"\\]*(?:\\.[^"\\]*)*"', re.DOTALL)
_STRUCTURE = re.compile(rb'["\[\]{}]')


def document_depth(data: bytes) -> int:
    """How deep JSON bytes nest, from the brackets outside strings, without parsing them.

    UTF-8 never puts a quote, a backslash or a bracket inside a multi-byte sequence, so the
    count is read on the bytes. One pass reads them left to right, skipping each string
    whole. For bytes that are not JSON the count is some number, and the parse refuses them;
    the count stops at a string with no closing quote, past which nothing parses.
    """
    depth = deepest = pos = 0
    while (m := _STRUCTURE.search(data, pos)) is not None:
        c = m.group()
        if c == b'"':
            s = _JSON_STRING.match(data, m.start())
            if s is None:
                break
            pos = s.end()
            continue
        if c in b"[{":
            depth += 1
            deepest = max(deepest, depth)
        else:
            depth -= 1
        pos = m.end()
    return deepest


def integer_literal(text: str) -> int | float | None:
    """The domain number a JSON integer literal (no fraction, no exponent) names, or ``None``.

    Within 2^53 - 1 in magnitude it is that ``int``. Beyond, it is the ``float`` it names when
    the literal is that double exactly, or is the double as RFC 8785 writes it (so canonical
    bytes read back as written); any other literal, such as ``9007199254740993``, names no
    double exactly and is ``None``. Raises ``ValueError`` for a literal over Python's digit
    limit for ``int``.
    """
    n = int(text)
    if -MAX_INTEGER <= n <= MAX_INTEGER:
        return n
    f = float(text)
    if math.isfinite(f) and (int(f) == n or _number(f) == text):
        return f
    return None


def _parse_int(text: str) -> int | float:
    try:
        number = integer_literal(text)
    except ValueError as exc:
        raise _DecodeError(f"integer literal of {len(text)} characters is too long") from exc
    if number is None:
        raise _DecodeError(f"integer {text} names no double exactly")
    return number


def _parse_float(text: str) -> float:
    number = float(text)
    if not math.isfinite(number):
        raise _DecodeError(f"number {text} is not a finite double")
    return number


def _refuse_constant(name: str) -> NoReturn:
    raise _DecodeError(f"{name} is not a JSON number")


def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise _DuplicateKey(f"duplicate key {key!r}")
        out[key] = value
    return out


def sha256_ref(data: bytes) -> str:
    """SHA-256 of ``data`` serialized as ``sha256:`` and 64 lowercase hex characters."""
    return "sha256:" + hashlib.sha256(data).hexdigest()


def hash_value(value: object, max_depth: int | None = None) -> str:
    """The hash of a value, or with ``max_depth`` of a holder: SHA-256 over its canonical
    bytes, the same bytes either way (``encode``)."""
    return sha256_ref(encode(value, max_depth))


def record_hash(items: Sequence[bytes]) -> str:
    """The record hash of a log prefix, from each item's canonical bytes in log order.

    Equal to ``hash_value`` of the list of items, without re-encoding the items.
    """
    return sha256_ref(b"[" + b",".join(items) + b"]")


def _expand32(prk: bytes, info: bytes) -> bytes:
    """HKDF-Expand (RFC 5869, section 2.3) with length 32: one block, T(1)."""
    return hmac.digest(prk, info + b"\x01", "sha256")


def turn_key(root_key: bytes, record_hash: str) -> bytes:
    """The per-turn key: HKDF-Expand under the root key, the record hash as the info.

    Raises ``ValueError`` unless ``root_key`` is an instance of ``bytes`` of length 32,
    before anything else is read.
    """
    # By its type and bytes' own length, so no override of a subclass runs.
    if not issubclass(type(root_key), bytes) or bytes.__len__(root_key) != 32:  # pyright: ignore[reportUnnecessaryIsInstance]
        raise ValueError("the root key is not an instance of bytes of length 32")
    key = bytes.__getitem__(root_key, slice(None))  # its data, as exactly bytes
    if not issubclass(type(record_hash), str):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise _CanonicalError(f"record hash of type {type_name(record_hash)} is not a string")
    text = string_data(record_hash)
    if not _SHA256_REF.fullmatch(text):
        raise _CanonicalError(f"record hash {clipped(text)} is not sha256: and 64 hex")
    return _expand32(key, text.encode("ascii"))


def keyed_digest(key: bytes, data: bytes) -> str:
    """HMAC-SHA-256 of canonical bytes under a per-turn key, as ``hmac-sha256:`` and hex."""
    return "hmac-sha256:" + hmac.digest(key, data, "sha256").hex()
