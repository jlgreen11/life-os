"""
Tests for connectors/crypto.py — ConfigEncryptor.

Validates Fernet-based field-level encryption of sensitive connector config
values, including key creation, roundtrip encrypt/decrypt, masking, and
edge cases (special characters, empty values, already-encrypted values).
"""

import os
import stat

import pytest

from connectors.crypto import ConfigEncryptor


@pytest.fixture()
def encryptor(tmp_path):
    """A ConfigEncryptor backed by a temporary data directory."""
    return ConfigEncryptor(data_dir=str(tmp_path))


# --- Key management ---


def test_creates_key_file_on_init(tmp_path):
    """ConfigEncryptor creates .connector_key in data_dir on first use."""
    ConfigEncryptor(data_dir=str(tmp_path))
    assert (tmp_path / ".connector_key").exists()


def test_key_file_has_restricted_permissions(tmp_path):
    """Key file is created with 0o600 (owner read/write only)."""
    ConfigEncryptor(data_dir=str(tmp_path))
    key_path = tmp_path / ".connector_key"
    mode = stat.S_IMODE(os.stat(key_path).st_mode)
    assert mode == 0o600


def test_reuses_existing_key(tmp_path):
    """A second ConfigEncryptor instance loads the same key and can decrypt data from the first."""
    enc1 = ConfigEncryptor(data_dir=str(tmp_path))
    config = {"password": "s3cret"}
    encrypted = enc1.encrypt_config(config, {"password"})

    enc2 = ConfigEncryptor(data_dir=str(tmp_path))
    decrypted = enc2.decrypt_config(encrypted, {"password"})
    assert decrypted["password"] == "s3cret"


# --- encrypt_config / decrypt_config roundtrip ---


def test_encrypt_roundtrip(encryptor):
    """Encrypting then decrypting returns the original values."""
    config = {"username": "alice", "password": "hunter2", "token": "abc123"}
    sensitive = {"password", "token"}

    encrypted = encryptor.encrypt_config(config, sensitive)
    decrypted = encryptor.decrypt_config(encrypted, sensitive)

    assert decrypted == config


def test_encrypt_only_sensitive_fields(encryptor):
    """Non-sensitive fields stay as plaintext after encryption."""
    config = {"host": "localhost", "password": "s3cret"}
    encrypted = encryptor.encrypt_config(config, {"password"})

    assert encrypted["host"] == "localhost"
    assert encrypted["password"].startswith("ENC:")


def test_encrypt_skips_already_encrypted(encryptor):
    """Values already prefixed with ENC: are not double-encrypted."""
    config = {"password": "ENC:already_encrypted_token"}
    encrypted = encryptor.encrypt_config(config, {"password"})

    assert encrypted["password"] == "ENC:already_encrypted_token"


def test_encrypt_skips_empty_values(encryptor):
    """Empty string and None values in sensitive fields are not encrypted."""
    config = {"password": "", "token": None, "key": "real_value"}
    sensitive = {"password", "token", "key"}
    encrypted = encryptor.encrypt_config(config, sensitive)

    assert encrypted["password"] == ""
    assert encrypted["token"] is None
    assert encrypted["key"].startswith("ENC:")


def test_decrypt_skips_non_encrypted(encryptor):
    """Values without ENC: prefix in sensitive fields are returned as-is."""
    config = {"password": "plaintext_value", "host": "localhost"}
    decrypted = encryptor.decrypt_config(config, {"password"})

    assert decrypted["password"] == "plaintext_value"
    assert decrypted["host"] == "localhost"


# --- mask_config ---


def test_mask_replaces_sensitive_values(encryptor):
    """Sensitive field values are replaced with '********'."""
    config = {"password": "s3cret", "token": "ENC:something"}
    masked = encryptor.mask_config(config, {"password", "token"})

    assert masked["password"] == "********"
    assert masked["token"] == "********"


def test_mask_preserves_non_sensitive_values(encryptor):
    """Non-sensitive fields are not changed by masking."""
    config = {"host": "localhost", "port": 5432, "password": "s3cret"}
    masked = encryptor.mask_config(config, {"password"})

    assert masked["host"] == "localhost"
    assert masked["port"] == 5432


def test_mask_skips_empty_values(encryptor):
    """Empty/None sensitive values are not masked (left as-is)."""
    config = {"password": "", "token": None}
    masked = encryptor.mask_config(config, {"password", "token"})

    assert masked["password"] == ""
    assert masked["token"] is None


# --- Edge cases ---


def test_encrypt_decrypt_special_characters(encryptor):
    """Handles passwords with special chars (!, @, #, unicode, emoji)."""
    config = {
        "password": "p@ss!w0rd#$%^&*()",
        "token": "tök€n_wïth_ünîcödé_🔑",
    }
    sensitive = {"password", "token"}

    encrypted = encryptor.encrypt_config(config, sensitive)
    decrypted = encryptor.decrypt_config(encrypted, sensitive)

    assert decrypted == config
