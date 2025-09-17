"""
Encryption module for payload security.

Provides AES-256 encryption with password-based key derivation and integrity
verification using HMAC. Designed for secure payload embedding in neural networks.
"""

import os
import hmac
import hashlib
from typing import Tuple
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

from .exceptions import EncryptionError


class PayloadCrypto:
    """
    Handles encryption and decryption of payloads using AES-256-GCM.
    
    Uses PBKDF2 for key derivation from passwords and includes integrity
    verification to detect tampering.
    """
    
    # Cryptographic parameters
    KEY_LENGTH = 32  # 256 bits for AES-256
    IV_LENGTH = 16   # 128 bits for AES
    SALT_LENGTH = 16 # 128 bits for PBKDF2 salt
    TAG_LENGTH = 16  # 128 bits for GCM authentication tag
    PBKDF2_ITERATIONS = 100000  # OWASP recommended minimum
    
    @classmethod
    def encrypt_payload(cls, payload: bytes, password: str) -> bytes:
        """
        Encrypt payload with AES-256-GCM using password-based key derivation.
        
        Args:
            payload: Raw payload bytes to encrypt
            password: Password for key derivation
            
        Returns:
            Encrypted payload with salt, IV, tag, and ciphertext
            
        Format: [salt:16][iv:16][tag:16][ciphertext:variable]
        """
        if not password:
            raise EncryptionError("Password cannot be empty")
        
        if not payload:
            raise EncryptionError("Payload cannot be empty")
        
        try:
            # Generate random salt and IV
            salt = os.urandom(cls.SALT_LENGTH)
            iv = os.urandom(cls.IV_LENGTH)
            
            # Derive key from password using PBKDF2
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=cls.KEY_LENGTH,
                salt=salt,
                iterations=cls.PBKDF2_ITERATIONS,
                backend=default_backend()
            )
            key = kdf.derive(password.encode('utf-8'))
            
            # Encrypt with AES-256-GCM
            cipher = Cipher(
                algorithms.AES(key),
                modes.GCM(iv),
                backend=default_backend()
            )
            encryptor = cipher.encryptor()
            ciphertext = encryptor.update(payload) + encryptor.finalize()
            
            # Get authentication tag
            tag = encryptor.tag
            
            # Combine all components: salt + iv + tag + ciphertext
            encrypted_payload = salt + iv + tag + ciphertext
            
            return encrypted_payload
            
        except Exception as e:
            raise EncryptionError(f"Encryption failed: {e}")
    
    @classmethod
    def decrypt_payload(cls, encrypted_payload: bytes, password: str) -> bytes:
        """
        Decrypt payload encrypted with encrypt_payload.
        
        Args:
            encrypted_payload: Encrypted payload from encrypt_payload
            password: Password used for encryption
            
        Returns:
            Decrypted payload bytes
        """
        if not password:
            raise EncryptionError("Password cannot be empty")
        
        if not encrypted_payload:
            raise EncryptionError("Encrypted payload cannot be empty")
        
        # Check minimum length
        min_length = cls.SALT_LENGTH + cls.IV_LENGTH + cls.TAG_LENGTH
        if len(encrypted_payload) < min_length:
            raise EncryptionError(
                f"Encrypted payload too short: {len(encrypted_payload)} < {min_length}"
            )
        
        try:
            # Extract components
            salt = encrypted_payload[:cls.SALT_LENGTH]
            iv = encrypted_payload[cls.SALT_LENGTH:cls.SALT_LENGTH + cls.IV_LENGTH]
            tag = encrypted_payload[
                cls.SALT_LENGTH + cls.IV_LENGTH:
                cls.SALT_LENGTH + cls.IV_LENGTH + cls.TAG_LENGTH
            ]
            ciphertext = encrypted_payload[cls.SALT_LENGTH + cls.IV_LENGTH + cls.TAG_LENGTH:]
            
            # Derive key from password using same parameters
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=cls.KEY_LENGTH,
                salt=salt,
                iterations=cls.PBKDF2_ITERATIONS,
                backend=default_backend()
            )
            key = kdf.derive(password.encode('utf-8'))
            
            # Decrypt with AES-256-GCM
            cipher = Cipher(
                algorithms.AES(key),
                modes.GCM(iv, tag),
                backend=default_backend()
            )
            decryptor = cipher.decryptor()
            payload = decryptor.update(ciphertext) + decryptor.finalize()
            
            return payload
            
        except Exception as e:
            raise EncryptionError(f"Decryption failed: {e}")
    
    @classmethod
    def verify_password(cls, encrypted_payload: bytes, password: str) -> bool:
        """
        Verify if a password can decrypt the payload without full decryption.
        
        Args:
            encrypted_payload: Encrypted payload to test
            password: Password to verify
            
        Returns:
            True if password is correct, False otherwise
        """
        try:
            cls.decrypt_payload(encrypted_payload, password)
            return True
        except EncryptionError:
            return False
    
    @classmethod
    def get_encrypted_size(cls, payload_size: int) -> int:
        """
        Calculate the size of payload after encryption.
        
        Args:
            payload_size: Size of original payload in bytes
            
        Returns:
            Size after encryption including salt, IV, tag
        """
        return payload_size + cls.SALT_LENGTH + cls.IV_LENGTH + cls.TAG_LENGTH


def encrypt_payload(payload: bytes, password: str) -> bytes:
    """
    Convenience function for payload encryption.
    
    Args:
        payload: Raw payload bytes
        password: Encryption password
        
    Returns:
        Encrypted payload bytes
    """
    return PayloadCrypto.encrypt_payload(payload, password)


def decrypt_payload(encrypted_payload: bytes, password: str) -> bytes:
    """
    Convenience function for payload decryption.
    
    Args:
        encrypted_payload: Encrypted payload bytes
        password: Decryption password
        
    Returns:
        Decrypted payload bytes
    """
    return PayloadCrypto.decrypt_payload(encrypted_payload, password)


def get_encrypted_size(payload_size: int) -> int:
    """
    Convenience function to calculate encrypted payload size.
    
    Args:
        payload_size: Original payload size in bytes
        
    Returns:
        Size after encryption
    """
    return PayloadCrypto.get_encrypted_size(payload_size)