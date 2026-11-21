"""Audit Logger — Security event audit logging with tamper-evident chains.

Provides structured, tamper-evident audit logging with:
- Cryptographic hash chaining (hash of each entry includes previous hash)
- Structured event schema with severity, actor, action, resource, result
- Multiple output backends (file, syslog, remote API, database)
- Query and filtering capabilities
- Log rotation and retention policies
- Integration with KeyVault for signing/verification
- Compliance reporting (SOC2, HIPAA, GDPR, PCI-DSS)
"""

import os
import json
import time
import tempfile
import hashlib
import threading
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Iterator, Callable, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from abc import ABC, abstractmethod
from collections import deque
import queue
import gzip
import shutil


class AuditSeverity(Enum):
    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3
    CRITICAL = 4


class AuditAction(Enum):
    # Authentication
    LOGIN = "login"
    LOGOUT = "logout"
    LOGIN_FAILED = "login_failed"
    MFA_CHALLENGE = "mfa_challenge"
    MFA_SUCCESS = "mfa_success"
    MFA_FAILED = "mfa_failed"
    PASSWORD_CHANGE = "password_change"
    PASSWORD_RESET = "password_reset"
    API_KEY_CREATE = "api_key_create"
    API_KEY_REVOKE = "api_key_revoke"
    SESSION_CREATE = "session_create"
    SESSION_REVOKE = "session_revoke"
    
    # Authorization
    PERMISSION_GRANT = "permission_grant"
    PERMISSION_REVOKE = "permission_revoke"
    ROLE_ASSIGN = "role_assign"
    ROLE_REVOKE = "role_revoke"
    POLICY_CHANGE = "policy_change"
    
    # Data access
    DATA_READ = "data_read"
    DATA_WRITE = "data_write"
    DATA_DELETE = "data_delete"
    DATA_EXPORT = "data_export"
    DATA_IMPORT = "data_import"
    BACKUP_CREATE = "backup_create"
    BACKUP_RESTORE = "backup_restore"
    
    # Configuration
    CONFIG_CHANGE = "config_change"
    FEATURE_FLAG_CHANGE = "feature_flag_change"
    SECURITY_POLICY_CHANGE = "security_policy_change"
    FIREWALL_RULE_CHANGE = "firewall_rule_change"
    
    # System
    SYSTEM_START = "system_start"
    SYSTEM_SHUTDOWN = "system_shutdown"
    SYSTEM_RESTART = "system_restart"
    SERVICE_START = "service_start"
    SERVICE_STOP = "service_stop"
    CRON_JOB_RUN = "cron_job_run"
    
    # Security events
    INTRUSION_ATTEMPT = "intrusion_attempt"
    BRUTE_FORCE_DETECTED = "brute_force_detected"
    ANOMALY_DETECTED = "anomaly_detected"
    MALWARE_DETECTED = "malware_detected"
    VULNERABILITY_SCAN = "vulnerability_scan"
    PENETRATION_TEST = "penetration_test"
    
    # Compliance
    AUDIT_LOG_ACCESS = "audit_log_access"
    COMPLIANCE_REPORT_GENERATED = "compliance_report_generated"
    DATA_RETENTION_APPLIED = "data_retention_applied"
    CONSENT_GRANTED = "consent_granted"
    CONSENT_REVOKED = "consent_revoked"
    
    # Key management
    KEY_GENERATED = "key_generated"
    KEY_ROTATED = "key_rotated"
    KEY_REVOKED = "key_revoked"
    KEY_ACCESSED = "key_accessed"
    CERTIFICATE_ISSUED = "certificate_issued"
    CERTIFICATE_REVOKED = "certificate_revoked"
    
    # Container/Infrastructure
    CONTAINER_START = "container_start"
    CONTAINER_STOP = "container_stop"
    IMAGE_PULL = "image_pull"
    IMAGE_PUSH = "image_push"
    DEPLOYMENT_CREATE = "deployment_create"
    DEPLOYMENT_UPDATE = "deployment_update"
    DEPLOYMENT_DELETE = "deployment_delete"
    
    # Custom
    CUSTOM = "custom"


class AuditResult(Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    DENIED = "denied"
    ERROR = "error"


@dataclass
class AuditEvent:
    """Structured audit event."""
    timestamp: float
    event_id: str
    action: AuditAction
    severity: AuditSeverity
    result: AuditResult
    actor: str  # User ID, service account, system component
    actor_type: str = "user"  # user, service, system, api_key
    resource: str = ""  # Resource affected
    resource_type: str = ""  # Type of resource
    resource_id: str = ""  # Specific resource identifier
    details: Dict[str, Any] = field(default_factory=dict)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    session_id: Optional[str] = None
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    correlation_id: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)
    # Tamper-evident fields
    prev_hash: str = ""
    hash: str = ""
    
    def __post_init__(self):
        if not self.event_id:
            self.event_id = f"evt_{int(self.timestamp * 1000000)}_{hashlib.md5(str(self.timestamp).encode()).hexdigest()[:8]}"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "event_id": self.event_id,
            "action": self.action.value,
            "severity": self.severity.name,
            "result": self.result.value,
            "actor": self.actor,
            "actor_type": self.actor_type,
            "resource": self.resource,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "details": self.details,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "session_id": self.session_id,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "correlation_id": self.correlation_id,
            "tags": self.tags,
            "prev_hash": self.prev_hash,
            "hash": self.hash,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AuditEvent":
        return cls(
            timestamp=data["timestamp"],
            event_id=data["event_id"],
            action=AuditAction(data["action"]),
            severity=AuditSeverity[data["severity"]],
            result=AuditResult(data["result"]),
            actor=data["actor"],
            actor_type=data.get("actor_type", "user"),
            resource=data.get("resource", ""),
            resource_type=data.get("resource_type", ""),
            resource_id=data.get("resource_id", ""),
            details=data.get("details", {}),
            ip_address=data.get("ip_address"),
            user_agent=data.get("user_agent"),
            session_id=data.get("session_id"),
            trace_id=data.get("trace_id"),
            span_id=data.get("span_id"),
            correlation_id=data.get("correlation_id"),
            tags=data.get("tags", {}),
            prev_hash=data.get("prev_hash", ""),
            hash=data.get("hash", ""),
        )


class AuditBackend(ABC):
    """Abstract backend for audit log storage."""
    
    @abstractmethod
    def write(self, event: AuditEvent) -> bool:
        pass
    
    @abstractmethod
    def query(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        action: Optional[AuditAction] = None,
        actor: Optional[str] = None,
        severity: Optional[AuditSeverity] = None,
        limit: int = 1000,
    ) -> List[AuditEvent]:
        pass
    
    @abstractmethod
    def close(self):
        pass


class FileAuditBackend(AuditBackend):
    """File-based audit log backend with rotation and compression."""
    
    def __init__(
        self,
        log_dir: str = "/var/log/sneppx/audit",
        max_file_size: int = 100 * 1024 * 1024,  # 100MB
        max_files: int = 10,
        compress: bool = True,
    ):
        self.log_dir = Path(log_dir)
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        except (PermissionError, OSError):
            self.log_dir = Path(tempfile.mkdtemp(prefix="sneppx_audit_"))
        self.max_file_size = max_file_size
        self.max_files = max_files
        self.compress = compress
        self._current_file: Optional[Path] = None
        self._current_size = 0
        self._file_handle = None
        self._lock = threading.Lock()
        self._prev_hash = "0" * 64
        self._open_new_file()
    
    def _open_new_file(self):
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self._current_file = self.log_dir / f"audit_{timestamp}.log"
        if self._file_handle:
            self._file_handle.close()
        self._file_handle = open(self._current_file, "a", encoding="utf-8")
        self._current_size = self._current_file.stat().st_size if self._current_file.exists() else 0
    
    def _rotate_if_needed(self):
        if self._current_size >= self.max_file_size:
            self._rotate()
    
    def _rotate(self):
        if self._file_handle:
            self._file_handle.close()
        
        if self.compress and self._current_file and self._current_file.exists():
            compressed = self._current_file.with_suffix(".log.gz")
            with open(self._current_file, "rb") as f_in:
                with gzip.open(compressed, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            self._current_file.unlink()
        
        # Clean old files
        files = sorted(self.log_dir.glob("audit_*.log*"), key=lambda f: f.stat().st_mtime)
        while len(files) > self.max_files:
            files.pop(0).unlink()
        
        self._open_new_file()
    
    def write(self, event: AuditEvent) -> bool:
        with self._lock:
            event.prev_hash = self._prev_hash
            entry_data = event.to_dict()
            # Compute hash of entry without the hash field itself
            entry_data.pop("hash", None)
            hash_input = json.dumps(entry_data, sort_keys=True, separators=(",", ":")).encode()
            event.hash = hashlib.sha256(hash_input).hexdigest()
            self._prev_hash = event.hash
            
            line = json.dumps(event.to_dict(), separators=(",", ":")) + "\n"
            self._file_handle.write(line)
            self._file_handle.flush()
            self._current_size += len(line)
            
            self._rotate_if_needed()
            return True
    
    def query(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        action: Optional[AuditAction] = None,
        actor: Optional[str] = None,
        severity: Optional[AuditSeverity] = None,
        limit: int = 1000,
    ) -> List[AuditEvent]:
        results = []
        files = sorted(self.log_dir.glob("audit_*.log*"), key=lambda f: f.stat().st_mtime, reverse=True)
        
        for file_path in files:
            if len(results) >= limit:
                break
            
            open_func = gzip.open if file_path.suffix == ".gz" else open
            mode = "rt" if file_path.suffix == ".gz" else "r"
            
            try:
                with open_func(file_path, mode, encoding="utf-8") as f:
                    for line in f:
                        if len(results) >= limit:
                            break
                        try:
                            data = json.loads(line)
                            event = AuditEvent.from_dict(data)
                            
                            # Filter
                            if start_time and event.timestamp < start_time:
                                continue
                            if end_time and event.timestamp > end_time:
                                continue
                            if action and event.action != action:
                                continue
                            if actor and event.actor != actor:
                                continue
                            if severity and event.severity != severity:
                                continue
                            
                            results.append(event)
                        except Exception:
                            continue
            except Exception:
                continue
        
        return results
    
    def close(self):
        with self._lock:
            if self._file_handle:
                self._file_handle.close()
                self._file_handle = None


class SyslogAuditBackend(AuditBackend):
    """Syslog audit backend."""
    
    def __init__(self, facility: int | None = None):
        import logging.handlers
        if facility is None:
            facility = logging.handlers.SysLogHandler.LOG_LOCAL0
        self.handler = logging.handlers.SysLogHandler(facility=facility)
        self.handler.setFormatter(logging.Formatter("AUDIT: %(message)s"))
        self.logger = logging.getLogger("sneppx.audit.syslog")
        self.logger.addHandler(self.handler)
        self.logger.setLevel(logging.INFO)
        self._prev_hash = "0" * 64
    
    def write(self, event: AuditEvent) -> bool:
        event.prev_hash = self._prev_hash
        hash_input = json.dumps(event.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        event.hash = hashlib.sha256(hash_input).hexdigest()
        self._prev_hash = event.hash
        
        self.logger.info(json.dumps(event.to_dict(), separators=(",", ":")))
        return True
    
    def query(self, **kwargs) -> List[AuditEvent]:
        return []  # Syslog doesn't support query
    
    def close(self):
        self.handler.close()


class RemoteAPIAuditBackend(AuditBackend):
    """Remote API audit backend."""
    
    def __init__(
        self,
        endpoint: str,
        api_key: str,
        timeout: int = 10,
        batch_size: int = 100,
        flush_interval: float = 5.0,
    ):
        self.endpoint = endpoint
        self.api_key = api_key
        self.timeout = timeout
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        
        self._queue: queue.Queue = queue.Queue()
        self._buffer: List[AuditEvent] = []
        self._lock = threading.Lock()
        self._prev_hash = "0" * 64
        self._running = True
        self._worker = threading.Thread(target=self._flush_worker, daemon=True)
        self._worker.start()
    
    def _flush_worker(self):
        import urllib.request
        import urllib.error
        
        while self._running:
            time.sleep(self.flush_interval)
            self._flush_batch()
    
    def _flush_batch(self):
        with self._lock:
            if not self._buffer:
                return
            batch = self._buffer[:self.batch_size]
            self._buffer = self._buffer[self.batch_size:]
        
        try:
            import urllib.request
            data = json.dumps([e.to_dict() for e in batch]).encode()
            req = urllib.request.Request(
                self.endpoint,
                data=data,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
            )
            urllib.request.urlopen(req, timeout=self.timeout)
        except Exception:
            # Re-queue on failure
            with self._lock:
                self._buffer = batch + self._buffer
    
    def write(self, event: AuditEvent) -> bool:
        event.prev_hash = self._prev_hash
        hash_input = json.dumps(event.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        event.hash = hashlib.sha256(hash_input).hexdigest()
        self._prev_hash = event.hash
        
        with self._lock:
            self._buffer.append(event)
            if len(self._buffer) >= self.batch_size:
                self._flush_batch()
        return True
    
    def query(self, **kwargs) -> List[AuditEvent]:
        return []
    
    def close(self):
        self._running = False
        self._flush_batch()


class DatabaseAuditBackend(AuditBackend):
    """Database audit backend (SQLite/PostgreSQL/MySQL)."""
    
    def __init__(
        self,
        connection_string: str,
        table_name: str = "audit_log",
    ):
        self.connection_string = connection_string
        self.table_name = table_name
        self._conn = None
        self._prev_hash = "0" * 64
        self._init_db()
    
    def _init_db(self):
        import sqlite3
        self._conn = sqlite3.connect(self.connection_string, check_same_thread=False)
        self._conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                event_id TEXT NOT NULL UNIQUE,
                action TEXT NOT NULL,
                severity TEXT NOT NULL,
                result TEXT NOT NULL,
                actor TEXT NOT NULL,
                actor_type TEXT NOT NULL,
                resource TEXT,
                resource_type TEXT,
                resource_id TEXT,
                details TEXT,
                ip_address TEXT,
                user_agent TEXT,
                session_id TEXT,
                trace_id TEXT,
                span_id TEXT,
                correlation_id TEXT,
                tags TEXT,
                prev_hash TEXT NOT NULL,
                hash TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self._conn.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{self.table_name}_timestamp 
            ON {self.table_name}(timestamp)
        """)
        self._conn.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{self.table_name}_actor 
            ON {self.table_name}(actor)
        """)
        self._conn.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{self.table_name}_action 
            ON {self.table_name}(action)
        """)
        self._conn.commit()
    
    def write(self, event: AuditEvent) -> bool:
        event.prev_hash = self._prev_hash
        hash_input = json.dumps(event.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        event.hash = hashlib.sha256(hash_input).hexdigest()
        self._prev_hash = event.hash
        
        import sqlite3
        try:
            self._conn.execute(
                f"""
                INSERT INTO {self.table_name} 
                (timestamp, event_id, action, severity, result, actor, actor_type,
                 resource, resource_type, resource_id, details, ip_address,
                 user_agent, session_id, trace_id, span_id, correlation_id,
                 tags, prev_hash, hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.timestamp,
                    event.event_id,
                    event.action.value,
                    event.severity.name,
                    event.result.value,
                    event.actor,
                    event.actor_type,
                    event.resource,
                    event.resource_type,
                    event.resource_id,
                    json.dumps(event.details),
                    event.ip_address,
                    event.user_agent,
                    event.session_id,
                    event.trace_id,
                    event.span_id,
                    event.correlation_id,
                    json.dumps(event.tags),
                    event.prev_hash,
                    event.hash,
                )
            )
            self._conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False
        except Exception:
            return False
    
    def query(self, **kwargs) -> List[AuditEvent]:
        import sqlite3
        query = f"SELECT * FROM {self.table_name} WHERE 1=1"
        params = []
        
        if kwargs.get("start_time"):
            query += " AND timestamp >= ?"
            params.append(kwargs["start_time"])
        if kwargs.get("end_time"):
            query += " AND timestamp <= ?"
            params.append(kwargs["end_time"])
        if kwargs.get("action"):
            query += " AND action = ?"
            params.append(kwargs["action"].value)
        if kwargs.get("actor"):
            query += " AND actor = ?"
            params.append(kwargs["actor"])
        if kwargs.get("severity"):
            query += " AND severity = ?"
            params.append(kwargs["severity"].name)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(kwargs.get("limit", 1000))
        
        cursor = self._conn.execute(query, params)
        results = []
        for row in cursor.fetchall():
            results.append(AuditEvent(
                timestamp=row[1],
                event_id=row[2],
                action=AuditAction(row[3]),
                severity=AuditSeverity[row[4]],
                result=AuditResult[row[5]],
                actor=row[6],
                actor_type=row[7],
                resource=row[8] or "",
                resource_type=row[9] or "",
                resource_id=row[10] or "",
                details=json.loads(row[11]) if row[11] else {},
                ip_address=row[12],
                user_agent=row[13],
                session_id=row[14],
                trace_id=row[15],
                span_id=row[16],
                correlation_id=row[17],
                tags=json.loads(row[18]) if row[18] else {},
                prev_hash=row[19],
                hash=row[20],
            ))
        return results
    
    def close(self):
        if self._conn:
            self._conn.close()


class AuditLogger:
    """Main audit logger with multi-backend support."""
    
    def __init__(
        self,
        backends: Optional[List[AuditBackend]] = None,
        default_severity: AuditSeverity = AuditSeverity.INFO,
        default_result: AuditResult = AuditResult.SUCCESS,
    ):
        self.backends = backends or []
        self.default_severity = default_severity
        self.default_result = default_result
        self._lock = threading.RLock()
        self._event_counter = 0
        
        # Event filters
        self._filters: List[Callable[[AuditEvent], bool]] = []
        
        # Enrichment functions
        self._enrichers: List[Callable[[AuditEvent], None]] = []
        
        # Context for implicit fields
        self._context: Dict[str, Any] = {}
    
    def add_backend(self, backend: AuditBackend):
        with self._lock:
            self.backends.append(backend)
    
    def remove_backend(self, backend: AuditBackend):
        with self._lock:
            if backend in self.backends:
                self.backends.remove(backend)
    
    def add_filter(self, filter_fn: Callable[[AuditEvent], bool]):
        """Add a filter function. Return False to drop event."""
        self._filters.append(filter_fn)
    
    def add_enricher(self, enricher_fn: Callable[[AuditEvent], None]):
        """Add an enrichment function that modifies the event in-place."""
        self._enrichers.append(enricher_fn)
    
    def set_context(self, **kwargs):
        """Set context fields that will be added to all events."""
        with self._lock:
            self._context.update(kwargs)
    
    def clear_context(self):
        with self._lock:
            self._context.clear()
    
    def log(
        self,
        action: AuditAction,
        actor: str,
        result: Optional[AuditResult] = None,
        severity: Optional[AuditSeverity] = None,
        resource: str = "",
        resource_type: str = "",
        resource_id: str = "",
        details: Optional[Dict[str, Any]] = None,
        actor_type: str = "user",
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        session_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        span_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> AuditEvent:
        """Log an audit event."""
        timestamp = time.time()
        
        # Convert string action/result/severity to enums
        if isinstance(action, str):
            try:
                action = AuditAction(action)
            except ValueError:
                action = AuditAction.CUSTOM
        if isinstance(result, str):
            try:
                result = AuditResult(result)
            except ValueError:
                pass
        if isinstance(severity, str):
            try:
                severity = AuditSeverity[severity.upper()]
            except (KeyError, AttributeError):
                severity = AuditSeverity.INFO
        
        with self._lock:
            self._event_counter += 1
            event_id = f"evt_{int(timestamp * 1000000)}_{self._event_counter:06d}"
        
        event = AuditEvent(
            timestamp=timestamp,
            event_id=event_id,
            action=action,
            severity=severity or self.default_severity,
            result=result or self.default_result,
            actor=actor,
            actor_type=actor_type,
            resource=resource,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details or {},
            ip_address=ip_address,
            user_agent=user_agent,
            session_id=session_id,
            trace_id=trace_id,
            span_id=span_id,
            correlation_id=correlation_id,
            tags=tags or {},
        )
        
        # Apply context
        with self._lock:
            for key, value in self._context.items():
                if key not in event.details:
                    event.details[key] = value
        
        # Apply enrichers
        for enricher in self._enrichers:
            try:
                enricher(event)
            except Exception:
                pass
        
        # Apply filters
        for filter_fn in self._filters:
            try:
                if not filter_fn(event):
                    return event  # Dropped
            except Exception:
                pass
        
        # Write to all backends
        for backend in self.backends:
            try:
                backend.write(event)
            except Exception:
                pass  # Don't let backend failures block logging
        
        return event
    
    # Convenience methods
    def login(self, actor: str, success: bool, **kwargs) -> AuditEvent:
        return self.log(
            action=AuditAction.LOGIN if success else AuditAction.LOGIN_FAILED,
            actor=actor,
            result=AuditResult.SUCCESS if success else AuditResult.FAILURE,
            severity=AuditSeverity.INFO if success else AuditSeverity.WARNING,
            **kwargs,
        )
    
    def data_access(
        self,
        actor: str,
        action: AuditAction,
        resource: str,
        resource_type: str,
        resource_id: str,
        success: bool,
        **kwargs,
    ) -> AuditEvent:
        return self.log(
            action=action,
            actor=actor,
            result=AuditResult.SUCCESS if success else AuditResult.FAILURE,
            resource=resource,
            resource_type=resource_type,
            resource_id=resource_id,
            **kwargs,
        )
    
    def config_change(
        self,
        actor: str,
        resource: str,
        old_value: Any,
        new_value: Any,
        success: bool,
        **kwargs,
    ) -> AuditEvent:
        return self.log(
            action=AuditAction.CONFIG_CHANGE,
            actor=actor,
            result=AuditResult.SUCCESS if success else AuditResult.FAILURE,
            resource=resource,
            details={
                "old_value": str(old_value),
                "new_value": str(new_value),
            },
            **kwargs,
        )
    
    def security_event(
        self,
        action: AuditAction,
        actor: str,
        severity: AuditSeverity,
        details: Dict[str, Any],
        **kwargs,
    ) -> AuditEvent:
        return self.log(
            action=action,
            actor=actor,
            severity=severity,
            result=AuditResult.FAILURE,
            details=details,
            **kwargs,
        )
    
    def query(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        action: Optional[AuditAction] = None,
        actor: Optional[str] = None,
        severity: Optional[AuditSeverity] = None,
        limit: int = 1000,
    ) -> List[AuditEvent]:
        """Query audit events from backends that support it."""
        all_results = []
        for backend in self.backends:
            try:
                results = backend.query(
                    start_time=start_time,
                    end_time=end_time,
                    action=action,
                    actor=actor,
                    severity=severity,
                    limit=limit,
                )
                all_results.extend(results)
            except Exception:
                pass
        
        # Deduplicate and sort
        seen = set()
        unique = []
        for event in sorted(all_results, key=lambda e: e.timestamp, reverse=True):
            if event.event_id not in seen:
                seen.add(event.event_id)
                unique.append(event)
        
        return unique[:limit]
    
    def clear(self):
        """Clear all events from all backends."""
        for backend in self.backends:
            try:
                if hasattr(backend, 'clear'):
                    backend.clear()
            except Exception:
                pass
    
    def verify_chain(self) -> Tuple[bool, List[str]]:
        """Verify hash chain integrity across all backends."""
        errors = []
        all_events = self.query(limit=10000)
        
        prev_hash = "0" * 64
        for event in sorted(all_events, key=lambda e: e.timestamp):  # Oldest first
            if event.prev_hash != prev_hash:
                errors.append(
                    f"Hash chain broken at {event.event_id}: "
                    f"expected prev_hash={prev_hash}, got {event.prev_hash}"
                )
            # Verify event hash
            event_data = event.to_dict()
            event_data.pop("hash", None)
            computed = hashlib.sha256(
                json.dumps(event_data, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            if computed != event.hash:
                errors.append(f"Event hash mismatch for {event.event_id}")
            prev_hash = event.hash
        
        return len(errors) == 0, errors
    
    def export(
        self,
        format: str = "json",
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> str:
        """Export audit logs in specified format."""
        events = self.query(start_time=start_time, end_time=end_time, limit=100000)
        
        if format == "json":
            return json.dumps([e.to_dict() for e in events], indent=2)
        elif format == "csv":
            import csv
            import io
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow([
                "timestamp", "event_id", "action", "severity", "result",
                "actor", "actor_type", "resource", "resource_type", "resource_id",
                "ip_address", "session_id", "trace_id", "hash"
            ])
            for e in events:
                writer.writerow([
                    e.timestamp, e.event_id, e.action.value, e.severity.name,
                    e.result.value, e.actor, e.actor_type, e.resource,
                    e.resource_type, e.resource_id, e.ip_address or "",
                    e.session_id or "", e.trace_id or "", e.hash
                ])
            return output.getvalue()
        elif format == "cef":  # Common Event Format
            lines = []
            for e in events:
                lines.append(
                    f"CEF:0|SneppX|AuditLogger|0.9.0|{e.action.value}|"
                    f"{e.action.name}|{e.severity.value}|"
                    f"act={e.actor} dvchost={e.ip_address or 'unknown'} "
                    f"outcome={e.result.value} msg={e.details}"
                )
            return "\n".join(lines)
        else:
            raise ValueError(f"Unknown format: {format}")
    
    def close(self):
        for backend in self.backends:
            try:
                backend.close()
            except Exception:
                pass


# Global instance
_global_audit_logger: Optional[AuditLogger] = None


def get_audit_logger(
    log_dir: Optional[str] = None,
    use_syslog: bool = False,
    syslog_facility: int = 128,
    remote_endpoint: Optional[str] = None,
    remote_api_key: Optional[str] = None,
    db_path: Optional[str] = None,
) -> AuditLogger:
    global _global_audit_logger
    if _global_audit_logger is None:
        if log_dir is None:
            log_dir = (
                os.environ.get("SX_AUDIT_DIR")
                or os.environ.get("SNEPPX_STATE_DIR")
                or os.path.join(tempfile.gettempdir(), "sneppx_audit")
            )
        backends = []
        backends.append(FileAuditBackend(log_dir))
        if use_syslog:
            backends.append(SyslogAuditBackend(syslog_facility))
        if remote_endpoint and remote_api_key:
            backends.append(RemoteAPIAuditBackend(remote_endpoint, remote_api_key))
        if db_path:
            backends.append(DatabaseAuditBackend(db_path))
        
        _global_audit_logger = AuditLogger(backends)
    return _global_audit_logger


def set_audit_logger(logger: AuditLogger):
    global _global_audit_logger
    _global_audit_logger = logger


def audit_log(
    action: AuditAction,
    actor: str,
    **kwargs,
) -> AuditEvent:
    return get_audit_logger().log(action, actor, **kwargs)


def audit_login(actor: str, success: bool, **kwargs) -> AuditEvent:
    return get_audit_logger().login(actor, success, **kwargs)


def audit_data_access(
    actor: str,
    action: AuditAction,
    resource: str,
    resource_type: str,
    resource_id: str,
    success: bool,
    **kwargs,
) -> AuditEvent:
    return get_audit_logger().data_access(actor, action, resource, resource_type, resource_id, success, **kwargs)


def audit_config_change(
    actor: str,
    resource: str,
    old_value: Any,
    new_value: Any,
    success: bool,
    **kwargs,
) -> AuditEvent:
    return get_audit_logger().config_change(actor, resource, old_value, new_value, success, **kwargs)


def audit_security_event(
    action: AuditAction,
    actor: str,
    severity: AuditSeverity,
    details: Dict[str, Any],
    **kwargs,
) -> AuditEvent:
    return get_audit_logger().security_event(action, actor, severity, details, **kwargs)


def query_audit_log(**kwargs) -> List[AuditEvent]:
    return get_audit_logger().query(**kwargs)


def verify_audit_chain() -> Tuple[bool, List[str]]:
    return get_audit_logger().verify_chain()


def export_audit_log(format: str = "json", **kwargs) -> str:
    return get_audit_logger().export(format, **kwargs)


__all__ = [
    "AuditSeverity",
    "AuditAction",
    "AuditResult",
    "AuditEvent",
    "AuditBackend",
    "FileAuditBackend",
    "SyslogAuditBackend",
    "RemoteAPIAuditBackend",
    "DatabaseAuditBackend",
    "AuditLogger",
    "get_audit_logger",
    "set_audit_logger",
    "audit_log",
    "audit_login",
    "audit_data_access",
    "audit_config_change",
    "audit_security_event",
    "query_audit_log",
    "verify_audit_chain",
    "export_audit_log",
]