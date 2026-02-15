"""
Life OS — Connector Credential Encryption

Field-level encryption for sensitive connector configuration values.
Non-sensitive values stay readable in the DB for debugging.
Sensitive values are stored with an ENC: prefix.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from cryptography.fernet import Fernet


class ConfigEncryptor:
    """Encrypts and decrypts sensitive config fields using Fernet."""

    PREFIX = "ENC:"

    def __init__(self, data_dir: str = "./data"):
        self._key_path = Path(data_dir) / ".connector_key"
        self._fernet = Fernet(self._load_or_create_key())

    def _load_or_create_key(self) -> bytes:
        """Load existing key or generate a new one with restricted permissions."""
        if self._key_path.exists():
            return self._key_path.read_bytes().strip()

        key = Fernet.generate_key()
        self._key_path.parent.mkdir(parents=True, exist_ok=True)
        self._key_path.write_bytes(key)
        os.chmod(self._key_path, 0o600)
        return key

    def encrypt_config(self, config: dict[str, Any],
                       sensitive_fields: set[str]) -> dict[str, Any]:
        """Encrypt sensitive field values, returning a new dict."""
        result = {}
        for key, value in config.items():
            if key in sensitive_fields and value and not str(value).startswith(self.PREFIX):
                encrypted = self._fernet.encrypt(str(value).encode()).decode()
                result[key] = f"{self.PREFIX}{encrypted}"
            else:
                result[key] = value
        return result

    def decrypt_config(self, config: dict[str, Any],
                       sensitive_fields: set[str]) -> dict[str, Any]:
        """Decrypt ENC:-prefixed values, returning a new dict."""
        result = {}
        for key, value in config.items():
            if key in sensitive_fields and isinstance(value, str) and value.startswith(self.PREFIX):
                token = value[len(self.PREFIX):]
                result[key] = self._fernet.decrypt(token.encode()).decode()
            else:
                result[key] = value
        return result

    def mask_config(self, config: dict[str, Any],
                    sensitive_fields: set[str]) -> dict[str, Any]:
        """Replace sensitive field values with ******** for API responses."""
        result = {}
        for key, value in config.items():
            if key in sensitive_fields and value:
                result[key] = "********"
            else:
                result[key] = value
        return result
