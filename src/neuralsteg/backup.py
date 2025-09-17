"""
Model backup and integrity management.

Provides automatic backup creation, integrity verification, and rollback
capabilities to protect valuable models during steganographic operations.
"""

import json
import shutil
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime

from .exceptions import CorruptionDetectedError, ModelLoadError

logger = logging.getLogger(__name__)


class ModelBackupManager:
    """
    Manages model backups and integrity verification.
    
    Creates automatic backups before modifications and provides rollback
    capabilities in case of corruption or errors.
    """
    
    def __init__(self, model_path: Union[str, Path]):
        """
        Initialize backup manager for a model.
        
        Args:
            model_path: Path to the neural network model directory
        """
        self.model_path = Path(model_path)
        self.backup_dir = self.model_path.parent / f".{self.model_path.name}_backups"
        
        if not self.model_path.exists():
            raise ModelLoadError(f"Model path does not exist: {self.model_path}")
    
    def create_backup(self, backup_name: Optional[str] = None) -> Path:
        """
        Create a backup of the current model.
        
        Args:
            backup_name: Optional custom backup name, defaults to timestamp
            
        Returns:
            Path to the created backup directory
        """
        if backup_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"backup_{timestamp}"
        
        backup_path = self.backup_dir / backup_name
        
        logger.info(f"Creating model backup: {backup_path}")
        
        try:
            # Create backup directory
            backup_path.mkdir(parents=True, exist_ok=True)
            
            # Copy all model files
            for item in self.model_path.iterdir():
                if item.is_file():
                    shutil.copy2(item, backup_path / item.name)
                elif item.is_dir():
                    shutil.copytree(item, backup_path / item.name, dirs_exist_ok=True)
            
            # Create backup metadata
            metadata = {
                "created_at": datetime.now().isoformat(),
                "original_path": str(self.model_path),
                "backup_name": backup_name,
                "file_checksums": self._calculate_checksums(backup_path)
            }
            
            with open(backup_path / ".backup_metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Backup created successfully: {backup_path}")
            return backup_path
            
        except Exception as e:
            # Clean up partial backup on failure
            if backup_path.exists():
                shutil.rmtree(backup_path, ignore_errors=True)
            raise CorruptionDetectedError(f"Failed to create backup: {e}")
    
    def restore_backup(self, backup_name: str) -> None:
        """
        Restore model from a backup.
        
        Args:
            backup_name: Name of the backup to restore
        """
        backup_path = self.backup_dir / backup_name
        
        if not backup_path.exists():
            raise ModelLoadError(f"Backup not found: {backup_path}")
        
        logger.info(f"Restoring model from backup: {backup_path}")
        
        try:
            # Verify backup integrity first
            if not self.verify_backup_integrity(backup_name):
                raise CorruptionDetectedError(f"Backup integrity check failed: {backup_name}")
            
            # Create a temporary backup of current state
            temp_backup = self.create_backup("temp_before_restore")
            
            try:
                # Remove current model files (except backup directory)
                for item in self.model_path.iterdir():
                    if item.name.startswith('.') and 'backup' in item.name:
                        continue  # Skip backup directories
                    
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item)
                
                # Restore files from backup
                for item in backup_path.iterdir():
                    if item.name == ".backup_metadata.json":
                        continue  # Skip metadata file
                    
                    if item.is_file():
                        shutil.copy2(item, self.model_path / item.name)
                    elif item.is_dir():
                        shutil.copytree(item, self.model_path / item.name)
                
                # Verify restored model
                if not self.verify_model_integrity():
                    raise CorruptionDetectedError("Restored model failed integrity check")
                
                # Remove temporary backup on success
                shutil.rmtree(temp_backup, ignore_errors=True)
                
                logger.info(f"Model restored successfully from backup: {backup_name}")
                
            except Exception as e:
                # Restore from temporary backup if restoration failed
                logger.error(f"Restoration failed, reverting: {e}")
                self._restore_from_path(temp_backup)
                raise CorruptionDetectedError(f"Failed to restore backup: {e}")
                
        except Exception as e:
            raise CorruptionDetectedError(f"Backup restoration failed: {e}")
    
    def list_backups(self) -> List[Dict[str, str]]:
        """
        List all available backups.
        
        Returns:
            List of backup information dictionaries
        """
        if not self.backup_dir.exists():
            return []
        
        backups = []
        
        for backup_path in self.backup_dir.iterdir():
            if not backup_path.is_dir():
                continue
            
            metadata_path = backup_path / ".backup_metadata.json"
            
            if metadata_path.exists():
                try:
                    with open(metadata_path) as f:
                        metadata = json.load(f)
                    
                    backups.append({
                        "name": backup_path.name,
                        "created_at": metadata.get("created_at", "unknown"),
                        "path": str(backup_path),
                        "size_mb": self._get_directory_size(backup_path) / (1024 * 1024)
                    })
                except Exception as e:
                    logger.warning(f"Failed to read backup metadata for {backup_path}: {e}")
                    backups.append({
                        "name": backup_path.name,
                        "created_at": "unknown",
                        "path": str(backup_path),
                        "size_mb": self._get_directory_size(backup_path) / (1024 * 1024)
                    })
        
        # Sort by creation time (newest first)
        backups.sort(key=lambda x: x["created_at"], reverse=True)
        return backups
    
    def cleanup_old_backups(self, keep_count: int = 5) -> None:
        """
        Remove old backups, keeping only the most recent ones.
        
        Args:
            keep_count: Number of backups to keep
        """
        backups = self.list_backups()
        
        if len(backups) <= keep_count:
            return
        
        backups_to_remove = backups[keep_count:]
        
        logger.info(f"Cleaning up {len(backups_to_remove)} old backups")
        
        for backup in backups_to_remove:
            try:
                backup_path = Path(backup["path"])
                if backup_path.exists():
                    shutil.rmtree(backup_path)
                    logger.info(f"Removed old backup: {backup['name']}")
            except Exception as e:
                logger.warning(f"Failed to remove backup {backup['name']}: {e}")
    
    def verify_model_integrity(self) -> bool:
        """
        Verify the integrity of the current model.
        
        Returns:
            True if model structure is intact, False otherwise
        """
        try:
            # Check required files exist
            required_files = ["config.json", "model.safetensors.index.json"]
            
            for file in required_files:
                file_path = self.model_path / file
                if not file_path.exists():
                    logger.error(f"Required file missing: {file}")
                    return False
            
            # Load and validate index
            index_path = self.model_path / "model.safetensors.index.json"
            with open(index_path) as f:
                index = json.load(f)
            
            # Check that all shard files exist
            for shard_file in set(index["weight_map"].values()):
                shard_path = self.model_path / shard_file
                if not shard_path.exists():
                    logger.error(f"Shard file missing: {shard_file}")
                    return False
            
            logger.info("Model integrity verification passed")
            return True
            
        except Exception as e:
            logger.error(f"Model integrity verification failed: {e}")
            return False
    
    def verify_backup_integrity(self, backup_name: str) -> bool:
        """
        Verify the integrity of a backup.
        
        Args:
            backup_name: Name of the backup to verify
            
        Returns:
            True if backup is intact, False otherwise
        """
        backup_path = self.backup_dir / backup_name
        metadata_path = backup_path / ".backup_metadata.json"
        
        if not metadata_path.exists():
            logger.error(f"Backup metadata missing: {backup_name}")
            return False
        
        try:
            with open(metadata_path) as f:
                metadata = json.load(f)
            
            # Verify checksums
            stored_checksums = metadata.get("file_checksums", {})
            current_checksums = self._calculate_checksums(backup_path)
            
            for file_path, stored_checksum in stored_checksums.items():
                if file_path not in current_checksums:
                    logger.error(f"Backup file missing: {file_path}")
                    return False
                
                if current_checksums[file_path] != stored_checksum:
                    logger.error(f"Backup file corrupted: {file_path}")
                    return False
            
            logger.info(f"Backup integrity verification passed: {backup_name}")
            return True
            
        except Exception as e:
            logger.error(f"Backup integrity verification failed: {e}")
            return False
    
    def _calculate_checksums(self, directory: Path) -> Dict[str, str]:
        """Calculate MD5 checksums for all files in a directory."""
        checksums = {}
        
        for file_path in directory.rglob("*"):
            if file_path.is_file() and not file_path.name.startswith(".backup_"):
                try:
                    with open(file_path, "rb") as f:
                        checksum = hashlib.md5(f.read()).hexdigest()
                    
                    # Store relative path as key
                    rel_path = file_path.relative_to(directory)
                    checksums[str(rel_path)] = checksum
                    
                except Exception as e:
                    logger.warning(f"Failed to calculate checksum for {file_path}: {e}")
        
        return checksums
    
    def _get_directory_size(self, directory: Path) -> int:
        """Calculate total size of a directory in bytes."""
        total_size = 0
        
        for file_path in directory.rglob("*"):
            if file_path.is_file():
                try:
                    total_size += file_path.stat().st_size
                except Exception:
                    pass  # Skip files we can't access
        
        return total_size
    
    def _restore_from_path(self, backup_path: Path) -> None:
        """Internal method to restore from a specific backup path."""
        # Remove current model files
        for item in self.model_path.iterdir():
            if item.name.startswith('.') and 'backup' in item.name:
                continue
            
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
        
        # Restore files
        for item in backup_path.iterdir():
            if item.name.startswith(".backup_"):
                continue
            
            if item.is_file():
                shutil.copy2(item, self.model_path / item.name)
            elif item.is_dir():
                shutil.copytree(item, self.model_path / item.name)