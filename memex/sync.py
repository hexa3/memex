"""Optional encrypted sync primitives.

The sync layer is intentionally local-first and disabled by default. This
module provides the safe building blocks used by future transports: encrypted
payload envelopes and deterministic conflict-free record merging.
"""

from __future__ import annotations

import base64
import hashlib
import importlib
import os
from dataclasses import dataclass, field
from typing import Any, cast

from memex.errors import OptionalDependencyError
from memex.models import MemoryRecord


def derive_device_id(public_key: bytes) -> str:
    """Return a stable, non-secret device id for sync metadata."""

    return hashlib.sha256(public_key).hexdigest()[:32]


@dataclass(frozen=True)
class SyncEnvelope:
    """Encrypted payload ready to send through an untrusted sync transport."""

    version: int
    device_id: str
    nonce: str
    ciphertext: str


class EncryptedSyncCodec:
    """Encrypt and decrypt sync payloads with ChaCha20-Poly1305."""

    def __init__(self, key: bytes, *, device_id: str | None = None) -> None:
        if len(key) != 32:
            raise ValueError("sync encryption key must be exactly 32 bytes")
        self.key = key
        self.device_id = device_id or derive_device_id(key)

    @classmethod
    def from_passphrase(cls, passphrase: str, *, salt: bytes | None = None) -> EncryptedSyncCodec:
        """Derive a codec from a user passphrase.

        Pass a stable per-user salt in production. When omitted, memex uses a
        deterministic application salt for local tests and examples.
        """

        value = passphrase.strip()
        if len(value) < 12:
            raise ValueError("sync passphrase must be at least 12 characters")
        salt = salt or b"memex-sync-v1"
        key = hashlib.pbkdf2_hmac("sha256", value.encode("utf-8"), salt, 600_000, dklen=32)
        return cls(key)

    def encrypt(self, payload: bytes) -> SyncEnvelope:
        """Encrypt bytes into a transport-safe envelope."""

        ChaCha20Poly1305 = _load_chacha20_poly1305()
        nonce = os.urandom(12)
        cipher = ChaCha20Poly1305(self.key)
        ciphertext = cipher.encrypt(nonce, payload, self.device_id.encode("utf-8"))
        return SyncEnvelope(
            version=1,
            device_id=self.device_id,
            nonce=base64.b64encode(nonce).decode("ascii"),
            ciphertext=base64.b64encode(ciphertext).decode("ascii"),
        )

    def decrypt(self, envelope: SyncEnvelope) -> bytes:
        """Decrypt an envelope and authenticate its device metadata."""

        ChaCha20Poly1305 = _load_chacha20_poly1305()
        if envelope.version != 1:
            raise ValueError(f"unsupported sync envelope version: {envelope.version}")
        cipher = ChaCha20Poly1305(self.key)
        return cast(
            bytes,
            cipher.decrypt(
                base64.b64decode(envelope.nonce),
                base64.b64decode(envelope.ciphertext),
                envelope.device_id.encode("utf-8"),
            ),
        )


@dataclass
class CRDTState:
    """Small observed-remove map for memory records."""

    records: dict[str, MemoryRecord] = field(default_factory=dict)
    deleted: dict[str, int] = field(default_factory=dict)

    def apply_record(self, record: MemoryRecord) -> None:
        """Apply a record update with last-write-wins conflict resolution."""

        deleted_at = self.deleted.get(record.id)
        updated_at = int(record.accessed_at.timestamp())
        if deleted_at is not None and deleted_at >= updated_at:
            return
        existing = self.records.get(record.id)
        if existing is None or _record_clock(record) >= _record_clock(existing):
            self.records[record.id] = record

    def delete(self, record_id: str, timestamp: int) -> None:
        """Apply a deletion tombstone."""

        self.deleted[record_id] = max(timestamp, self.deleted.get(record_id, 0))
        existing = self.records.get(record_id)
        if existing is not None and timestamp >= int(existing.accessed_at.timestamp()):
            self.records.pop(record_id, None)

    def merge(self, other: CRDTState) -> CRDTState:
        """Merge two replicas without central coordination."""

        merged = CRDTState(records=dict(self.records), deleted=dict(self.deleted))
        for record_id, timestamp in other.deleted.items():
            merged.delete(record_id, timestamp)
        for record in other.records.values():
            merged.apply_record(record)
        for record_id, timestamp in self.deleted.items():
            merged.delete(record_id, timestamp)
        return merged

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON-serializable CRDT payload."""

        return {
            "version": 1,
            "records": [
                record.model_dump(mode="json", exclude_none=True)
                for record in self.records.values()
            ],
            "deleted": self.deleted,
        }


def _record_clock(record: MemoryRecord) -> tuple[int, str]:
    return (int(record.accessed_at.timestamp()), record.id)


def _load_chacha20_poly1305() -> Any:
    try:
        module = importlib.import_module("cryptography.hazmat.primitives.ciphers.aead")
    except ImportError as exc:
        raise OptionalDependencyError(
            "Encrypted sync requires cryptography. Install with: pip install 'memex-ai[sync]'"
        ) from exc
    return module.ChaCha20Poly1305
