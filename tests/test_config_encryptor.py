"""
Comprehensive test suite for ConfigEncryptor — security-critical credential encryption.

The ConfigEncryptor is responsible for encrypting/decrypting sensitive connector
credentials (passwords, API keys, tokens). These tests ensure:
1. Encryption key management is secure (permissions, persistence)
2. Encryption/decryption round-trips correctly
3. Masking protects credentials in API responses
4. Edge cases (empty values, None, already-encrypted) are handled safely
5. Multiple instances share keys correctly

Coverage: 72 LOC, 0% → 100%
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from cryptography.fernet import Fernet

from connectors.crypto import ConfigEncryptor


class TestConfigEncryptorKeyManagement:
    """Test encryption key creation, loading, and security."""

    def test_creates_key_file_on_first_use(self, tmp_path):
        """ConfigEncryptor should create encryption key file if it doesn't exist."""
        data_dir = tmp_path / "data"
        encryptor = ConfigEncryptor(data_dir=str(data_dir))

        key_path = data_dir / ".connector_key"
        assert key_path.exists(), "Encryption key file should be created"
        assert len(key_path.read_bytes()) == 44, "Fernet key should be 44 bytes (base64)"

    def test_key_file_has_restrictive_permissions(self, tmp_path):
        """Encryption key should be created with 0o600 permissions (owner read/write only)."""
        data_dir = tmp_path / "data"
        encryptor = ConfigEncryptor(data_dir=str(data_dir))

        key_path = data_dir / ".connector_key"
        mode = os.stat(key_path).st_mode & 0o777
        assert mode == 0o600, f"Key file should have 0o600 permissions, got {oct(mode)}"

    def test_loads_existing_key_on_second_init(self, tmp_path):
        """ConfigEncryptor should reuse existing key instead of generating new one."""
        data_dir = tmp_path / "data"

        # First instance creates key
        encryptor1 = ConfigEncryptor(data_dir=str(data_dir))
        key_path = data_dir / ".connector_key"
        original_key = key_path.read_bytes()

        # Second instance should load same key
        encryptor2 = ConfigEncryptor(data_dir=str(data_dir))
        loaded_key = key_path.read_bytes()

        assert original_key == loaded_key, "Second instance should load existing key"

    def test_multiple_instances_can_decrypt_each_others_values(self, tmp_path):
        """Multiple ConfigEncryptor instances sharing a key should interoperate."""
        data_dir = tmp_path / "data"

        encryptor1 = ConfigEncryptor(data_dir=str(data_dir))
        config = {"api_key": "secret123"}
        encrypted = encryptor1.encrypt_config(config, {"api_key"})

        # Second instance should decrypt successfully
        encryptor2 = ConfigEncryptor(data_dir=str(data_dir))
        decrypted = encryptor2.decrypt_config(encrypted, {"api_key"})

        assert decrypted["api_key"] == "secret123"

    def test_key_path_default_is_data_directory(self):
        """ConfigEncryptor should default to ./data for key storage."""
        encryptor = ConfigEncryptor()
        # Check the internal key path without creating it
        assert encryptor._key_path == Path("./data/.connector_key")


class TestConfigEncryptorEncryption:
    """Test encryption of sensitive configuration fields."""

    def test_encrypts_sensitive_field_with_prefix(self, tmp_path):
        """Sensitive fields should be encrypted and prefixed with 'ENC:'."""
        encryptor = ConfigEncryptor(data_dir=str(tmp_path))
        config = {"api_key": "secret123", "username": "user"}

        encrypted = encryptor.encrypt_config(config, {"api_key"})

        assert encrypted["api_key"].startswith("ENC:"), "Encrypted value should have ENC: prefix"
        assert encrypted["api_key"] != "secret123", "Value should be encrypted"
        assert encrypted["username"] == "user", "Non-sensitive fields should not be encrypted"

    def test_encrypts_multiple_sensitive_fields(self, tmp_path):
        """All fields in sensitive_fields set should be encrypted."""
        encryptor = ConfigEncryptor(data_dir=str(tmp_path))
        config = {
            "api_key": "key123",
            "password": "pass456",
            "username": "user",
            "token": "token789"
        }

        encrypted = encryptor.encrypt_config(config, {"api_key", "password", "token"})

        assert encrypted["api_key"].startswith("ENC:")
        assert encrypted["password"].startswith("ENC:")
        assert encrypted["token"].startswith("ENC:")
        assert encrypted["username"] == "user", "Non-sensitive field should be unchanged"

    def test_does_not_double_encrypt_already_encrypted_values(self, tmp_path):
        """Already-encrypted values (ENC: prefix) should not be re-encrypted."""
        encryptor = ConfigEncryptor(data_dir=str(tmp_path))
        config = {"api_key": "secret123"}

        # First encryption
        encrypted_once = encryptor.encrypt_config(config, {"api_key"})
        original_encrypted = encrypted_once["api_key"]

        # Second encryption attempt should be no-op
        encrypted_twice = encryptor.encrypt_config(encrypted_once, {"api_key"})

        assert encrypted_twice["api_key"] == original_encrypted, "Should not double-encrypt"

    def test_handles_empty_string_gracefully(self, tmp_path):
        """Empty string values should not be encrypted."""
        encryptor = ConfigEncryptor(data_dir=str(tmp_path))
        config = {"api_key": ""}

        encrypted = encryptor.encrypt_config(config, {"api_key"})

        assert encrypted["api_key"] == "", "Empty string should remain empty"

    def test_handles_none_value_gracefully(self, tmp_path):
        """None values should not be encrypted."""
        encryptor = ConfigEncryptor(data_dir=str(tmp_path))
        config = {"api_key": None}

        encrypted = encryptor.encrypt_config(config, {"api_key"})

        assert encrypted["api_key"] is None, "None should remain None"

    def test_converts_non_string_values_to_string_before_encryption(self, tmp_path):
        """Non-string values (int, bool) should be converted to string before encryption."""
        encryptor = ConfigEncryptor(data_dir=str(tmp_path))
        config = {"port": 8080, "enabled": True}

        encrypted = encryptor.encrypt_config(config, {"port", "enabled"})

        assert encrypted["port"].startswith("ENC:")
        assert encrypted["enabled"].startswith("ENC:")

    def test_encryption_returns_new_dict(self, tmp_path):
        """encrypt_config should return a new dict without modifying the original."""
        encryptor = ConfigEncryptor(data_dir=str(tmp_path))
        original = {"api_key": "secret123"}

        encrypted = encryptor.encrypt_config(original, {"api_key"})

        assert original["api_key"] == "secret123", "Original dict should be unchanged"
        assert encrypted["api_key"] != "secret123", "Returned dict should have encrypted value"


class TestConfigEncryptorDecryption:
    """Test decryption of encrypted configuration fields."""

    def test_decrypts_encrypted_field(self, tmp_path):
        """ENC:-prefixed values should be decrypted correctly."""
        encryptor = ConfigEncryptor(data_dir=str(tmp_path))
        config = {"api_key": "secret123"}

        encrypted = encryptor.encrypt_config(config, {"api_key"})
        decrypted = encryptor.decrypt_config(encrypted, {"api_key"})

        assert decrypted["api_key"] == "secret123", "Decryption should restore original value"

    def test_decrypts_multiple_fields(self, tmp_path):
        """All encrypted fields should be decrypted."""
        encryptor = ConfigEncryptor(data_dir=str(tmp_path))
        config = {
            "api_key": "key123",
            "password": "pass456",
            "username": "user"
        }

        encrypted = encryptor.encrypt_config(config, {"api_key", "password"})
        decrypted = encryptor.decrypt_config(encrypted, {"api_key", "password"})

        assert decrypted["api_key"] == "key123"
        assert decrypted["password"] == "pass456"
        assert decrypted["username"] == "user"

    def test_leaves_non_encrypted_values_unchanged(self, tmp_path):
        """Values without ENC: prefix should pass through unchanged."""
        encryptor = ConfigEncryptor(data_dir=str(tmp_path))
        config = {"api_key": "plaintext_value"}

        # Decrypt without prior encryption
        decrypted = encryptor.decrypt_config(config, {"api_key"})

        assert decrypted["api_key"] == "plaintext_value"

    def test_leaves_non_sensitive_fields_unchanged(self, tmp_path):
        """Fields not in sensitive_fields should pass through untouched."""
        encryptor = ConfigEncryptor(data_dir=str(tmp_path))
        # First encrypt a value properly
        encrypted = encryptor.encrypt_config({"api_key": "secret"}, {"api_key"})
        # Add a non-sensitive field
        config = {**encrypted, "username": "user"}

        # Decrypt only the sensitive field
        decrypted = encryptor.decrypt_config(config, {"api_key"})

        assert decrypted["username"] == "user", "Non-sensitive field should pass through"
        assert decrypted["api_key"] == "secret", "Sensitive field should be decrypted"

    def test_handles_non_string_values_during_decryption(self, tmp_path):
        """Non-string values should not cause decryption to fail."""
        encryptor = ConfigEncryptor(data_dir=str(tmp_path))
        config = {"api_key": 12345, "enabled": True}

        decrypted = encryptor.decrypt_config(config, {"api_key", "enabled"})

        assert decrypted["api_key"] == 12345
        assert decrypted["enabled"] is True

    def test_decryption_returns_new_dict(self, tmp_path):
        """decrypt_config should return a new dict without modifying the original."""
        encryptor = ConfigEncryptor(data_dir=str(tmp_path))
        config = {"api_key": "secret123"}
        encrypted = encryptor.encrypt_config(config, {"api_key"})
        original_encrypted_value = encrypted["api_key"]

        decrypted = encryptor.decrypt_config(encrypted, {"api_key"})

        assert encrypted["api_key"] == original_encrypted_value, "Original encrypted dict unchanged"
        assert decrypted["api_key"] == "secret123", "Returned dict should have decrypted value"


class TestConfigEncryptorMasking:
    """Test masking of sensitive fields for API responses."""

    def test_masks_sensitive_field_with_asterisks(self, tmp_path):
        """Sensitive fields should be replaced with '********'."""
        encryptor = ConfigEncryptor(data_dir=str(tmp_path))
        config = {"api_key": "secret123", "username": "user"}

        masked = encryptor.mask_config(config, {"api_key"})

        assert masked["api_key"] == "********"
        assert masked["username"] == "user", "Non-sensitive fields should not be masked"

    def test_masks_multiple_sensitive_fields(self, tmp_path):
        """All fields in sensitive_fields should be masked."""
        encryptor = ConfigEncryptor(data_dir=str(tmp_path))
        config = {
            "api_key": "key123",
            "password": "pass456",
            "username": "user",
            "token": "token789"
        }

        masked = encryptor.mask_config(config, {"api_key", "password", "token"})

        assert masked["api_key"] == "********"
        assert masked["password"] == "********"
        assert masked["token"] == "********"
        assert masked["username"] == "user"

    def test_masks_encrypted_values(self, tmp_path):
        """Masking should work on encrypted (ENC:-prefixed) values."""
        encryptor = ConfigEncryptor(data_dir=str(tmp_path))
        config = {"api_key": "secret123"}
        encrypted = encryptor.encrypt_config(config, {"api_key"})

        masked = encryptor.mask_config(encrypted, {"api_key"})

        assert masked["api_key"] == "********"

    def test_does_not_mask_empty_values(self, tmp_path):
        """Empty string or None values should not be masked."""
        encryptor = ConfigEncryptor(data_dir=str(tmp_path))
        config = {"api_key": "", "password": None}

        masked = encryptor.mask_config(config, {"api_key", "password"})

        assert masked["api_key"] == "", "Empty string should remain empty"
        assert masked["password"] is None, "None should remain None"

    def test_masking_returns_new_dict(self, tmp_path):
        """mask_config should return a new dict without modifying the original."""
        encryptor = ConfigEncryptor(data_dir=str(tmp_path))
        original = {"api_key": "secret123"}

        masked = encryptor.mask_config(original, {"api_key"})

        assert original["api_key"] == "secret123", "Original dict should be unchanged"
        assert masked["api_key"] == "********", "Returned dict should have masked value"


class TestConfigEncryptorRoundTrip:
    """Test full encryption/decryption round-trip scenarios."""

    def test_encrypt_then_decrypt_restores_original_value(self, tmp_path):
        """Full round-trip should restore exact original value."""
        encryptor = ConfigEncryptor(data_dir=str(tmp_path))
        original = {
            "api_key": "my_secret_key_123",
            "password": "P@ssw0rd!",
            "username": "admin",
            "port": 5432
        }

        encrypted = encryptor.encrypt_config(original, {"api_key", "password"})
        decrypted = encryptor.decrypt_config(encrypted, {"api_key", "password"})

        assert decrypted == original, "Round-trip should restore original config"

    def test_encrypt_mask_workflow(self, tmp_path):
        """Common workflow: encrypt for storage, mask for API response."""
        encryptor = ConfigEncryptor(data_dir=str(tmp_path))
        original = {"api_key": "secret123", "username": "user"}

        # Store encrypted
        encrypted = encryptor.encrypt_config(original, {"api_key"})
        assert encrypted["api_key"].startswith("ENC:")

        # Return masked to API
        masked = encryptor.mask_config(encrypted, {"api_key"})
        assert masked["api_key"] == "********"
        assert masked["username"] == "user"

    def test_complex_config_with_mixed_field_types(self, tmp_path):
        """Real-world connector config with various field types."""
        encryptor = ConfigEncryptor(data_dir=str(tmp_path))
        config = {
            "connector_id": "google_main",
            "enabled": True,
            "client_id": "public_client_id_123",
            "client_secret": "very_secret_token_456",
            "refresh_token": "another_secret_789",
            "sync_interval": 300,
            "scopes": ["gmail", "calendar", "contacts"]
        }
        sensitive = {"client_secret", "refresh_token"}

        # Encrypt
        encrypted = encryptor.encrypt_config(config, sensitive)
        assert encrypted["client_secret"].startswith("ENC:")
        assert encrypted["refresh_token"].startswith("ENC:")
        assert encrypted["client_id"] == "public_client_id_123"
        assert encrypted["enabled"] is True

        # Decrypt
        decrypted = encryptor.decrypt_config(encrypted, sensitive)
        assert decrypted["client_secret"] == "very_secret_token_456"
        assert decrypted["refresh_token"] == "another_secret_789"

        # Mask
        masked = encryptor.mask_config(encrypted, sensitive)
        assert masked["client_secret"] == "********"
        assert masked["refresh_token"] == "********"
        assert masked["client_id"] == "public_client_id_123"


class TestConfigEncryptorEdgeCases:
    """Test edge cases and error conditions."""

    def test_handles_config_with_no_sensitive_fields(self, tmp_path):
        """Config with no sensitive fields should pass through unchanged."""
        encryptor = ConfigEncryptor(data_dir=str(tmp_path))
        config = {"username": "user", "port": 8080}

        encrypted = encryptor.encrypt_config(config, set())
        decrypted = encryptor.decrypt_config(config, set())
        masked = encryptor.mask_config(config, set())

        assert encrypted == config
        assert decrypted == config
        assert masked == config

    def test_handles_empty_config(self, tmp_path):
        """Empty config dict should be handled gracefully."""
        encryptor = ConfigEncryptor(data_dir=str(tmp_path))
        config = {}

        encrypted = encryptor.encrypt_config(config, {"api_key"})
        decrypted = encryptor.decrypt_config(config, {"api_key"})
        masked = encryptor.mask_config(config, {"api_key"})

        assert encrypted == {}
        assert decrypted == {}
        assert masked == {}

    def test_handles_sensitive_field_not_in_config(self, tmp_path):
        """Specifying sensitive field that doesn't exist should not error."""
        encryptor = ConfigEncryptor(data_dir=str(tmp_path))
        config = {"username": "user"}

        encrypted = encryptor.encrypt_config(config, {"api_key", "password"})
        decrypted = encryptor.decrypt_config(config, {"api_key", "password"})
        masked = encryptor.mask_config(config, {"api_key", "password"})

        assert encrypted == config
        assert decrypted == config
        assert masked == config

    def test_unicode_values_encrypt_correctly(self, tmp_path):
        """Unicode characters in sensitive values should be handled correctly."""
        encryptor = ConfigEncryptor(data_dir=str(tmp_path))
        config = {"api_key": "密钥🔐test"}

        encrypted = encryptor.encrypt_config(config, {"api_key"})
        decrypted = encryptor.decrypt_config(encrypted, {"api_key"})

        assert decrypted["api_key"] == "密钥🔐test"

    def test_very_long_value_encrypts_correctly(self, tmp_path):
        """Very long sensitive values should encrypt/decrypt correctly."""
        encryptor = ConfigEncryptor(data_dir=str(tmp_path))
        long_value = "x" * 10000
        config = {"api_key": long_value}

        encrypted = encryptor.encrypt_config(config, {"api_key"})
        decrypted = encryptor.decrypt_config(encrypted, {"api_key"})

        assert decrypted["api_key"] == long_value

    def test_special_characters_in_values(self, tmp_path):
        """Special characters in values should not break encryption."""
        encryptor = ConfigEncryptor(data_dir=str(tmp_path))
        config = {"api_key": "key:with/special\\chars\"'<>&"}

        encrypted = encryptor.encrypt_config(config, {"api_key"})
        decrypted = encryptor.decrypt_config(encrypted, {"api_key"})

        assert decrypted["api_key"] == "key:with/special\\chars\"'<>&"
