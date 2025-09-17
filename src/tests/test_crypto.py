"""
Unit tests for the encryption module.

Tests AES-256 encryption, decryption, and key derivation functionality.
"""

import pytest
import os
from neuralsteg.crypto import PayloadCrypto, encrypt_payload, decrypt_payload
from neuralsteg.exceptions import EncryptionError


class TestPayloadCrypto:
    """Test cases for PayloadCrypto class."""
    
    def test_encrypt_decrypt_roundtrip(self):
        """Test that encryption and decryption work correctly."""
        payload = b"Hello, this is a test payload for encryption!"
        password = "test_password_123"
        
        # Encrypt
        encrypted = PayloadCrypto.encrypt_payload(payload, password)
        
        # Verify encrypted data is different and longer
        assert encrypted != payload
        assert len(encrypted) > len(payload)
        
        # Decrypt
        decrypted = PayloadCrypto.decrypt_payload(encrypted, password)
        
        # Verify decryption worked
        assert decrypted == payload
    
    def test_different_passwords_fail(self):
        """Test that wrong password fails decryption."""
        payload = b"Secret data"
        password1 = "correct_password"
        password2 = "wrong_password"
        
        encrypted = PayloadCrypto.encrypt_payload(payload, password1)
        
        with pytest.raises(EncryptionError):
            PayloadCrypto.decrypt_payload(encrypted, password2)
    
    def test_empty_payload_fails(self):
        """Test that empty payload raises error."""
        with pytest.raises(EncryptionError):
            PayloadCrypto.encrypt_payload(b"", "password")
    
    def test_empty_password_fails(self):
        """Test that empty password raises error."""
        with pytest.raises(EncryptionError):
            PayloadCrypto.encrypt_payload(b"data", "")
    
    def test_corrupted_data_fails(self):
        """Test that corrupted encrypted data fails decryption."""
        payload = b"Test data"
        password = "test_password"
        
        encrypted = PayloadCrypto.encrypt_payload(payload, password)
        
        # Corrupt the encrypted data
        corrupted = encrypted[:-1] + b'\x00'
        
        with pytest.raises(EncryptionError):
            PayloadCrypto.decrypt_payload(corrupted, password)
    
    def test_verify_password(self):
        """Test password verification functionality."""
        payload = b"Test data for verification"
        correct_password = "correct123"
        wrong_password = "wrong456"
        
        encrypted = PayloadCrypto.encrypt_payload(payload, correct_password)
        
        assert PayloadCrypto.verify_password(encrypted, correct_password) == True
        assert PayloadCrypto.verify_password(encrypted, wrong_password) == False
    
    def test_get_encrypted_size(self):
        """Test encrypted size calculation."""
        original_size = 1000
        encrypted_size = PayloadCrypto.get_encrypted_size(original_size)
        
        # Should be original size plus overhead
        expected_overhead = PayloadCrypto.SALT_LENGTH + PayloadCrypto.IV_LENGTH + PayloadCrypto.TAG_LENGTH
        assert encrypted_size == original_size + expected_overhead
    
    def test_large_payload(self):
        """Test encryption/decryption with large payload."""
        # Create 1MB of random data
        payload = os.urandom(1024 * 1024)
        password = "large_payload_test"
        
        encrypted = PayloadCrypto.encrypt_payload(payload, password)
        decrypted = PayloadCrypto.decrypt_payload(encrypted, password)
        
        assert decrypted == payload
    
    def test_unicode_password(self):
        """Test encryption with unicode password."""
        payload = b"Unicode password test"
        password = "пароль_тест_🔐"
        
        encrypted = PayloadCrypto.encrypt_payload(payload, password)
        decrypted = PayloadCrypto.decrypt_payload(encrypted, password)
        
        assert decrypted == payload


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_encrypt_payload_function(self):
        """Test encrypt_payload convenience function."""
        payload = b"Test data"
        password = "test123"
        
        encrypted = encrypt_payload(payload, password)
        decrypted = decrypt_payload(encrypted, password)
        
        assert decrypted == payload
    
    def test_functions_match_class_methods(self):
        """Test that convenience functions match class methods."""
        payload = b"Consistency test"
        password = "consistent"
        
        class_encrypted = PayloadCrypto.encrypt_payload(payload, password)
        func_encrypted = encrypt_payload(payload, password)
        
        # Both should decrypt successfully with same password
        class_decrypted = PayloadCrypto.decrypt_payload(class_encrypted, password)
        func_decrypted = decrypt_payload(func_encrypted, password)
        
        assert class_decrypted == payload
        assert func_decrypted == payload