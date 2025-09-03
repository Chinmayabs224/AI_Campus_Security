"""
Local Storage Buffering Service

This module handles local storage of events and clips during network outages
with automatic synchronization when connectivity is restored.
"""

import os
import json
import time
import sqlite3
import threading
from typing import Dict, List, Optional, Callable, Any
from pathlib import Path
import logging
from queue import Queue, Empty
import pickle
import gzip
from datetime import datetime

from .models import SecurityEvent, IncidentClip, EventBuffer


class BufferConfig:
    """Configuration for local buffering"""
    
    def __init__(self):
        # Storage settings
        self.buffer_directory = "local_buffer"
        self.max_buffer_size_mb = 1000  # 1GB
        self.max_events_per_buffer = 1000
        self.max_buffer_age_hours = 48
        
        # Database settings
        self.db_file = "events_buffer.db"
        self.enable_compression = True
        
        # Sync settings
        self.sync_batch_size = 50
        self.sync_retry_interval = 30  # seconds
        self.max_sync_retries = 10
        
        # Network monitoring
        self.network_check_interval = 10  # seconds
        self.network_timeout = 5  # seconds


class LocalEventDatabase:
    """
    SQLite database for storing events and clips locally
    """
    
    def __init__(self, db_path: Path, config: BufferConfig):
        self.db_path = db_path
        self.config = config
        self.logger = logging.getLogger("local_db")
        
        # Connection management
        self._local = threading.local()
        self.lock = threading.Lock()
        
        # Initialize database
        self._init_database()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection"""
        if not hasattr(self._local, 'connection'):
            self._local.connection = sqlite3.connect(
                str(self.db_path),
                timeout=30.0,
                check_same_thread=False
            )
            self._local.connection.row_factory = sqlite3.Row
        return self._local.connection
    
    def _init_database(self):
        """Initialize database tables"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Events table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    camera_id TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    event_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    confidence_score REAL NOT NULL,
                    event_data BLOB NOT NULL,
                    created_at REAL NOT NULL,
                    synced BOOLEAN DEFAULT FALSE,
                    sync_attempts INTEGER DEFAULT 0,
                    last_sync_attempt REAL
                )
            ''')
            
            # Clips table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS clips (
                    clip_id TEXT PRIMARY KEY,
                    event_id TEXT NOT NULL,
                    camera_id TEXT NOT NULL,
                    start_timestamp REAL NOT NULL,
                    end_timestamp REAL NOT NULL,
                    file_path TEXT,
                    file_size INTEGER,
                    clip_data BLOB NOT NULL,
                    created_at REAL NOT NULL,
                    synced BOOLEAN DEFAULT FALSE,
                    sync_attempts INTEGER DEFAULT 0,
                    last_sync_attempt REAL,
                    FOREIGN KEY (event_id) REFERENCES events (event_id)
                )
            ''')
            
            # Sync status table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sync_status (
                    id INTEGER PRIMARY KEY,
                    last_sync_time REAL,
                    network_available BOOLEAN DEFAULT FALSE,
                    pending_events INTEGER DEFAULT 0,
                    pending_clips INTEGER DEFAULT 0,
                    total_synced_events INTEGER DEFAULT 0,
                    total_synced_clips INTEGER DEFAULT 0
                )
            ''')
            
            # Create indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_events_camera_timestamp ON events (camera_id, timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_events_synced ON events (synced)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_clips_synced ON clips (synced)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_clips_event ON clips (event_id)')
            
            # Initialize sync status if not exists
            cursor.execute('SELECT COUNT(*) FROM sync_status')
            if cursor.fetchone()[0] == 0:
                cursor.execute('''
                    INSERT INTO sync_status (last_sync_time, network_available, pending_events, pending_clips)
                    VALUES (?, FALSE, 0, 0)
                ''', (time.time(),))
            
            conn.commit()
            self.logger.info("Database initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing database: {e}")
            raise
    
    def store_event(self, event: SecurityEvent) -> bool:
        """Store event in local database"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Serialize event data
            event_data = self._serialize_data(event.to_dict())
            
            cursor.execute('''
                INSERT OR REPLACE INTO events 
                (event_id, camera_id, timestamp, event_type, severity, confidence_score, 
                 event_data, created_at, synced, sync_attempts)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, FALSE, 0)
            ''', (
                event.event_id,
                event.camera_id,
                event.timestamp,
                event.event_type.value,
                event.severity.value,
                event.confidence_score,
                event_data,
                event.created_at
            ))
            
            conn.commit()
            self._update_pending_counts()
            return True
            
        except Exception as e:
            self.logger.error(f"Error storing event {event.event_id}: {e}")
            return False
    
    def store_clip(self, clip: IncidentClip) -> bool:
        """Store clip metadata in local database"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Serialize clip data
            clip_data = self._serialize_data(clip.to_dict())
            
            cursor.execute('''
                INSERT OR REPLACE INTO clips 
                (clip_id, event_id, camera_id, start_timestamp, end_timestamp, 
                 file_path, file_size, clip_data, created_at, synced, sync_attempts)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, FALSE, 0)
            ''', (
                clip.clip_id,
                clip.event_id,
                clip.camera_id,
                clip.start_timestamp,
                clip.end_timestamp,
                clip.file_path,
                clip.file_size,
                clip_data,
                clip.created_at
            ))
            
            conn.commit()
            self._update_pending_counts()
            return True
            
        except Exception as e:
            self.logger.error(f"Error storing clip {clip.clip_id}: {e}")
            return False
    
    def get_pending_events(self, limit: int = 50) -> List[SecurityEvent]:
        """Get events that need to be synced"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT event_data FROM events 
                WHERE synced = FALSE 
                ORDER BY timestamp ASC 
                LIMIT ?
            ''', (limit,))
            
            events = []
            for row in cursor.fetchall():
                try:
                    event_dict = self._deserialize_data(row['event_data'])
                    event = SecurityEvent.from_dict(event_dict)
                    events.append(event)
                except Exception as e:
                    self.logger.warning(f"Error deserializing event: {e}")
            
            return events
            
        except Exception as e:
            self.logger.error(f"Error getting pending events: {e}")
            return []
    
    def get_pending_clips(self, limit: int = 50) -> List[IncidentClip]:
        """Get clips that need to be synced"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT clip_data FROM clips 
                WHERE synced = FALSE 
                ORDER BY start_timestamp ASC 
                LIMIT ?
            ''', (limit,))
            
            clips = []
            for row in cursor.fetchall():
                try:
                    clip_dict = self._deserialize_data(row['clip_data'])
                    clip = IncidentClip(**clip_dict)
                    clips.append(clip)
                except Exception as e:
                    self.logger.warning(f"Error deserializing clip: {e}")
            
            return clips
            
        except Exception as e:
            self.logger.error(f"Error getting pending clips: {e}")
            return []
    
    def mark_events_synced(self, event_ids: List[str]) -> bool:
        """Mark events as successfully synced"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            placeholders = ','.join(['?' for _ in event_ids])
            cursor.execute(f'''
                UPDATE events 
                SET synced = TRUE, last_sync_attempt = ? 
                WHERE event_id IN ({placeholders})
            ''', [time.time()] + event_ids)
            
            conn.commit()
            self._update_pending_counts()
            
            # Update sync statistics
            cursor.execute('''
                UPDATE sync_status 
                SET total_synced_events = total_synced_events + ?, 
                    last_sync_time = ?
            ''', (len(event_ids), time.time()))
            
            conn.commit()
            return True
            
        except Exception as e:
            self.logger.error(f"Error marking events as synced: {e}")
            return False
    
    def mark_clips_synced(self, clip_ids: List[str]) -> bool:
        """Mark clips as successfully synced"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            placeholders = ','.join(['?' for _ in clip_ids])
            cursor.execute(f'''
                UPDATE clips 
                SET synced = TRUE, last_sync_attempt = ? 
                WHERE clip_id IN ({placeholders})
            ''', [time.time()] + clip_ids)
            
            conn.commit()
            self._update_pending_counts()
            
            # Update sync statistics
            cursor.execute('''
                UPDATE sync_status 
                SET total_synced_clips = total_synced_clips + ?, 
                    last_sync_time = ?
            ''', (len(clip_ids), time.time()))
            
            conn.commit()
            return True
            
        except Exception as e:
            self.logger.error(f"Error marking clips as synced: {e}")
            return False
    
    def increment_sync_attempts(self, event_ids: List[str] = None, clip_ids: List[str] = None):
        """Increment sync attempt counters for failed syncs"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            current_time = time.time()
            
            if event_ids:
                placeholders = ','.join(['?' for _ in event_ids])
                cursor.execute(f'''
                    UPDATE events 
                    SET sync_attempts = sync_attempts + 1, last_sync_attempt = ? 
                    WHERE event_id IN ({placeholders})
                ''', [current_time] + event_ids)
            
            if clip_ids:
                placeholders = ','.join(['?' for _ in clip_ids])
                cursor.execute(f'''
                    UPDATE clips 
                    SET sync_attempts = sync_attempts + 1, last_sync_attempt = ? 
                    WHERE clip_id IN ({placeholders})
                ''', [current_time] + clip_ids)
            
            conn.commit()
            
        except Exception as e:
            self.logger.error(f"Error incrementing sync attempts: {e}")
    
    def cleanup_old_data(self, max_age_hours: float = 48.0) -> Dict[str, int]:
        """Clean up old synced data"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cutoff_time = time.time() - (max_age_hours * 3600)
            
            # Count items to be removed
            cursor.execute('SELECT COUNT(*) FROM events WHERE synced = TRUE AND created_at < ?', (cutoff_time,))
            events_to_remove = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM clips WHERE synced = TRUE AND created_at < ?', (cutoff_time,))
            clips_to_remove = cursor.fetchone()[0]
            
            # Remove old synced events
            cursor.execute('DELETE FROM events WHERE synced = TRUE AND created_at < ?', (cutoff_time,))
            
            # Remove old synced clips
            cursor.execute('DELETE FROM clips WHERE synced = TRUE AND created_at < ?', (cutoff_time,))
            
            conn.commit()
            self._update_pending_counts()
            
            self.logger.info(f"Cleaned up {events_to_remove} events and {clips_to_remove} clips")
            
            return {
                'events_removed': events_to_remove,
                'clips_removed': clips_to_remove
            }
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
            return {'events_removed': 0, 'clips_removed': 0}
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Get counts
            cursor.execute('SELECT COUNT(*) FROM events')
            total_events = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM events WHERE synced = FALSE')
            pending_events = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM clips')
            total_clips = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM clips WHERE synced = FALSE')
            pending_clips = cursor.fetchone()[0]
            
            # Get sync status
            cursor.execute('SELECT * FROM sync_status ORDER BY id DESC LIMIT 1')
            sync_row = cursor.fetchone()
            
            # Get database size
            db_size = self.db_path.stat().st_size if self.db_path.exists() else 0
            
            return {
                'total_events': total_events,
                'pending_events': pending_events,
                'total_clips': total_clips,
                'pending_clips': pending_clips,
                'database_size_mb': db_size / (1024 * 1024),
                'last_sync_time': sync_row['last_sync_time'] if sync_row else None,
                'total_synced_events': sync_row['total_synced_events'] if sync_row else 0,
                'total_synced_clips': sync_row['total_synced_clips'] if sync_row else 0
            }
            
        except Exception as e:
            self.logger.error(f"Error getting statistics: {e}")
            return {}
    
    def _serialize_data(self, data: Any) -> bytes:
        """Serialize data for storage"""
        try:
            json_data = json.dumps(data).encode('utf-8')
            if self.config.enable_compression:
                return gzip.compress(json_data)
            return json_data
        except Exception as e:
            self.logger.error(f"Error serializing data: {e}")
            raise
    
    def _deserialize_data(self, data: bytes) -> Any:
        """Deserialize data from storage"""
        try:
            if self.config.enable_compression:
                json_data = gzip.decompress(data)
            else:
                json_data = data
            return json.loads(json_data.decode('utf-8'))
        except Exception as e:
            self.logger.error(f"Error deserializing data: {e}")
            raise
    
    def _update_pending_counts(self):
        """Update pending counts in sync status"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute('SELECT COUNT(*) FROM events WHERE synced = FALSE')
            pending_events = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM clips WHERE synced = FALSE')
            pending_clips = cursor.fetchone()[0]
            
            cursor.execute('''
                UPDATE sync_status 
                SET pending_events = ?, pending_clips = ?
            ''', (pending_events, pending_clips))
            
            conn.commit()
            
        except Exception as e:
            self.logger.warning(f"Error updating pending counts: {e}")


class LocalBufferService:
    """
    Main service for local buffering and synchronization
    """
    
    def __init__(self, config: Optional[BufferConfig] = None):
        self.config = config or BufferConfig()
        self.logger = logging.getLogger("local_buffer")
        
        # Setup storage directory
        self.buffer_dir = Path(self.config.buffer_directory)
        self.buffer_dir.mkdir(exist_ok=True)
        
        # Initialize database
        db_path = self.buffer_dir / self.config.db_file
        self.database = LocalEventDatabase(db_path, self.config)
        
        # Sync management
        self.sync_callbacks: List[Callable[[List[SecurityEvent], List[IncidentClip]], bool]] = []
        self.network_check_callback: Optional[Callable[[], bool]] = None
        
        # Service state
        self.running = False
        self.sync_thread: Optional[threading.Thread] = None
        self.cleanup_thread: Optional[threading.Thread] = None
        
        # Network state
        self.network_available = False
        self.last_network_check = 0
        
        # Statistics
        self.stats = {
            'events_buffered': 0,
            'clips_buffered': 0,
            'sync_attempts': 0,
            'successful_syncs': 0,
            'failed_syncs': 0,
            'start_time': time.time()
        }
    
    def start(self) -> bool:
        """Start the local buffer service"""
        if self.running:
            return True
        
        self.logger.info("Starting Local Buffer Service")
        self.running = True
        
        # Start sync thread
        self.sync_thread = threading.Thread(target=self._sync_loop, daemon=True)
        self.sync_thread.start()
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()
        
        return True
    
    def stop(self):
        """Stop the local buffer service"""
        if not self.running:
            return
        
        self.logger.info("Stopping Local Buffer Service")
        self.running = False
        
        # Wait for threads
        if self.sync_thread and self.sync_thread.is_alive():
            self.sync_thread.join(timeout=10)
        
        if self.cleanup_thread and self.cleanup_thread.is_alive():
            self.cleanup_thread.join(timeout=5)
    
    def buffer_event(self, event: SecurityEvent) -> bool:
        """
        Buffer an event locally
        
        Args:
            event: Security event to buffer
            
        Returns:
            True if event was buffered successfully
        """
        try:
            success = self.database.store_event(event)
            if success:
                self.stats['events_buffered'] += 1
            return success
        except Exception as e:
            self.logger.error(f"Error buffering event: {e}")
            return False
    
    def buffer_clip(self, clip: IncidentClip) -> bool:
        """
        Buffer a clip locally
        
        Args:
            clip: Incident clip to buffer
            
        Returns:
            True if clip was buffered successfully
        """
        try:
            success = self.database.store_clip(clip)
            if success:
                self.stats['clips_buffered'] += 1
            return success
        except Exception as e:
            self.logger.error(f"Error buffering clip: {e}")
            return False
    
    def add_sync_callback(self, callback: Callable[[List[SecurityEvent], List[IncidentClip]], bool]):
        """
        Add callback for synchronizing data
        
        Args:
            callback: Function that takes events and clips, returns True if sync successful
        """
        self.sync_callbacks.append(callback)
    
    def set_network_check_callback(self, callback: Callable[[], bool]):
        """
        Set callback for checking network connectivity
        
        Args:
            callback: Function that returns True if network is available
        """
        self.network_check_callback = callback
    
    def _sync_loop(self):
        """Main synchronization loop"""
        while self.running:
            try:
                # Check network connectivity
                self._check_network_connectivity()
                
                if self.network_available:
                    # Attempt to sync pending data
                    self._attempt_sync()
                
                # Wait before next sync attempt
                time.sleep(self.config.sync_retry_interval)
                
            except Exception as e:
                self.logger.error(f"Error in sync loop: {e}")
                time.sleep(5)
    
    def _cleanup_loop(self):
        """Periodic cleanup of old data"""
        while self.running:
            try:
                # Clean up old data every hour
                time.sleep(3600)
                
                if not self.running:
                    break
                
                self.database.cleanup_old_data(self.config.max_buffer_age_hours)
                
            except Exception as e:
                self.logger.error(f"Error in cleanup loop: {e}")
    
    def _check_network_connectivity(self):
        """Check if network is available for syncing"""
        current_time = time.time()
        
        # Rate limit network checks
        if current_time - self.last_network_check < self.config.network_check_interval:
            return
        
        self.last_network_check = current_time
        
        try:
            if self.network_check_callback:
                self.network_available = self.network_check_callback()
            else:
                # Default network check (ping a reliable server)
                import subprocess
                result = subprocess.run(
                    ['ping', '-c', '1', '-W', str(self.config.network_timeout), '8.8.8.8'],
                    capture_output=True,
                    timeout=self.config.network_timeout + 2
                )
                self.network_available = result.returncode == 0
                
        except Exception as e:
            self.logger.debug(f"Network check failed: {e}")
            self.network_available = False
    
    def _attempt_sync(self):
        """Attempt to synchronize pending data"""
        if not self.sync_callbacks:
            return
        
        try:
            # Get pending data
            pending_events = self.database.get_pending_events(self.config.sync_batch_size)
            pending_clips = self.database.get_pending_clips(self.config.sync_batch_size)
            
            if not pending_events and not pending_clips:
                return
            
            self.logger.info(f"Attempting to sync {len(pending_events)} events and {len(pending_clips)} clips")
            self.stats['sync_attempts'] += 1
            
            # Try each sync callback until one succeeds
            sync_successful = False
            for callback in self.sync_callbacks:
                try:
                    if callback(pending_events, pending_clips):
                        sync_successful = True
                        break
                except Exception as e:
                    self.logger.warning(f"Sync callback failed: {e}")
            
            if sync_successful:
                # Mark as synced
                event_ids = [event.event_id for event in pending_events]
                clip_ids = [clip.clip_id for clip in pending_clips]
                
                if event_ids:
                    self.database.mark_events_synced(event_ids)
                if clip_ids:
                    self.database.mark_clips_synced(clip_ids)
                
                self.stats['successful_syncs'] += 1
                self.logger.info(f"Successfully synced {len(event_ids)} events and {len(clip_ids)} clips")
            else:
                # Increment attempt counters
                event_ids = [event.event_id for event in pending_events]
                clip_ids = [clip.clip_id for clip in pending_clips]
                
                self.database.increment_sync_attempts(event_ids, clip_ids)
                self.stats['failed_syncs'] += 1
                self.logger.warning("All sync callbacks failed")
                
        except Exception as e:
            self.logger.error(f"Error during sync attempt: {e}")
            self.stats['failed_syncs'] += 1
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive buffer status"""
        db_stats = self.database.get_statistics()
        
        return {
            'running': self.running,
            'network_available': self.network_available,
            'last_network_check': self.last_network_check,
            **db_stats,
            **self.stats,
            'sync_success_rate': (
                self.stats['successful_syncs'] / self.stats['sync_attempts']
                if self.stats['sync_attempts'] > 0 else 0
            )
        }
    
    def force_sync(self) -> bool:
        """Force an immediate sync attempt"""
        if not self.network_available:
            self._check_network_connectivity()
        
        if self.network_available:
            self._attempt_sync()
            return True
        else:
            self.logger.warning("Cannot force sync: network not available")
            return False